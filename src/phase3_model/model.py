"""MiniLLM: a decoder-only Transformer language model (GPT-like)."""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.common.config import MiniLLMConfig


class TiedLinear(nn.Module):
    """Linear projection that shares its weight tensor with an embedding layer.

    Used for weight tying between the input token embedding and the output
    LM head, reducing total parameter count and improving generalization.

    Args:
        weight: The weight tensor from an nn.Embedding (shape: vocab_size x d_model).
    """

    def __init__(self, weight: torch.Tensor) -> None:
        super().__init__()
        # Store as a plain tensor reference; not a Parameter to avoid duplication
        self.weight = weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute F.linear using the tied embedding weight.

        Args:
            x: Input tensor of shape (..., d_model).

        Returns:
            Output tensor of shape (..., vocab_size).
        """
        return F.linear(x, self.weight)


class MiniLLM(nn.Module):
    """Decoder-only Transformer language model.

    Architecture:
        Token Embedding
            -> Dropout
            -> TransformerBlock x n_layers
            -> Final LayerNorm
            -> LM Head (optionally tied with embedding)

    Features:
      - Pre-LayerNorm transformer blocks for training stability.
      - Optional Rotary Position Embedding (RoPE) for length generalization.
      - Optional weight tying between embedding and output projection.
      - Autoregressive generation with temperature, top-k, and top-p sampling.

    Args:
        config: MiniLLMConfig instance with model hyperparameters.
    """

    def __init__(self, config: MiniLLMConfig) -> None:
        super().__init__()
        self.config = config

        # ---- Token embedding ----
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)

        # ---- Transformer blocks ----
        from src.phase3_model.transformer_block import TransformerBlock

        self.blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layers)]
        )

        # ---- Final LayerNorm ----
        self.final_norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)

        # ---- LM Head (output projection to vocabulary) ----
        if config.use_tied_weights:
            self.lm_head = TiedLinear(self.token_embedding.weight)
        else:
            self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self.dropout = nn.Dropout(config.dropout)

        # ---- Weight initialization ----
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize weights following GPT-2 style initialization.

        - Linear / Embedding: normal(mean=0, std=0.02)
        - Bias terms: zero
        - LayerNorm: weight=1, bias=0
        """
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
            if module.weight is not None:
                torch.nn.init.ones_(module.weight)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass producing logits for next-token prediction.

        Args:
            input_ids: Token indices of shape (batch_size, seq_len).

        Returns:
            Logits of shape (batch_size, seq_len, vocab_size).
        """
        # Embedding lookup
        x = self.token_embedding(input_ids)  # (B, S, D_model)
        x = self.dropout(x)

        # Pass through all transformer blocks
        for block in self.blocks:
            x = block(x)

        # Final normalization
        x = self.final_norm(x)

        # Project to vocabulary
        logits = self.lm_head(x)  # (B, S, vocab_size)

        return logits

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 128,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        """Autoregressive text generation with sampling strategies.

        Supports temperature scaling, top-k filtering, and top-p (nucleus)
        filtering. Generation stops early when the EOS token is produced.

        Args:
            input_ids: Starting token sequence of shape (1, seq_len).
            max_new_tokens: Maximum number of tokens to generate.
            temperature: Softmax temperature (>0). Lower = more deterministic.
            top_k: Retain only the top-k highest-probability tokens. 0 disables.
            top_p: Nucleus sampling threshold. 1.0 disables.
            eos_token_id: Token ID that signals end of generation.
                           Defaults to config.eos_token_id.

        Returns:
            Full token sequence including the prompt and generated tokens,
            shape (1, prompt_len + num_generated).
        """
        if eos_token_id is None:
            eos_token_id = self.config.eos_token_id

        self.eval()
        greedy = (temperature <= 0)

        for _ in range(max_new_tokens):
            # Truncate to max_seq_len if the sequence exceeds it
            if input_ids.size(1) > self.config.max_seq_len:
                input_ids = input_ids[:, -self.config.max_seq_len:]

            # Forward pass to get logits for the last position
            logits = self(input_ids)                      # (1, S, vocab_size)
            next_logits = logits[:, -1, :]                # (1, vocab_size)

            # Greedy decoding
            if greedy:
                next_token = torch.argmax(next_logits, dim=-1, keepdim=True)
                input_ids = torch.cat([input_ids, next_token], dim=-1)
                if next_token.item() == eos_token_id:
                    break
                continue

            # Temperature scaling
            next_logits = next_logits / temperature

            # ---- Top-K filtering ----
            if top_k > 0:
                k = min(top_k, next_logits.size(-1))
                topk_vals, _ = torch.topk(next_logits, k, dim=-1)
                threshold = topk_vals[:, -1:]             # k-th largest value per batch
                next_logits[next_logits < threshold] = float("-inf")

            # ---- Top-P (nucleus) filtering ----
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_logits, descending=True, dim=-1)
                cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                # Mask tokens with cumulative probability > top_p
                sorted_mask = cum_probs > top_p
                # Shift mask right: always keep the first (highest probability) token
                sorted_mask[:, 1:] = sorted_mask[:, :-1].clone()
                sorted_mask[:, 0] = False

                # Scatter mask back to original index order
                indices_to_remove = sorted_mask.scatter(
                    dim=1, index=sorted_indices, src=sorted_mask
                )
                next_logits[indices_to_remove] = float("-inf")

            # ---- Sample next token ----
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (1, 1)

            # Stop if EOS token is generated
            if next_token.item() == eos_token_id:
                break

            # Append to sequence
            input_ids = torch.cat([input_ids, next_token], dim=-1)

        return input_ids

    def get_num_params(self, non_embedding: bool = False) -> int:
        """Return the total number of trainable parameters.

        Args:
            non_embedding: If True, exclude the token embedding parameters
                           (useful for comparing model sizes independent of vocabulary).

        Returns:
            Total parameter count.
        """
        total = sum(p.numel() for p in self.parameters())
        if non_embedding:
            total -= self.token_embedding.weight.numel()
        return total

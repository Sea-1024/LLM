"""Causal multi-head self-attention with optional RoPE support."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.common.config import MiniLLMConfig


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with optional Rotary Position Embedding.

    Follows the Pre-LayerNorm convention: this module expects its input to be
    normalized externally and does NOT apply LayerNorm internally.

    Optimizations:
      - Combined QKV projection for a single matrix multiply.
      - Causal mask registered as a persistent buffer to avoid re-allocation.

    Args:
        config: MiniLLMConfig instance with model hyperparameters.
    """

    def __init__(self, config: MiniLLMConfig) -> None:
        super().__init__()

        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        self.d_model = config.d_model
        self.dropout_rate = config.dropout

        # Combined QKV projection: (d_model) -> (3 * d_model)
        self.qkv_proj = nn.Linear(config.d_model, 3 * config.d_model, bias=False)

        # Output projection: (d_model) -> (d_model)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)

        # Rotary Position Embedding (applied to Q and K only)
        if config.use_rotary:
            from src.phase3_model.rotary import RotaryEmbedding

            self.rotary = RotaryEmbedding(
                dim=config.head_dim,
                max_seq_len=config.max_seq_len,
                theta=config.rope_theta,
            )
        else:
            self.rotary = None

        # Causal mask: upper-triangular True means "mask out" (set to -inf)
        causal_mask = torch.triu(
            torch.ones(config.max_seq_len, config.max_seq_len, dtype=torch.bool),
            diagonal=1,
        )
        self.register_buffer("causal_mask", causal_mask)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.out_dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for causal self-attention.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model).

        Returns:
            Output tensor of shape (batch_size, seq_len, d_model).
        """
        B, S, D = x.shape

        # ---- QKV projection & split ----
        qkv = self.qkv_proj(x)                     # (B, S, 3*D)
        q, k, v = qkv.chunk(3, dim=-1)             # each (B, S, D)

        # ---- Reshape to multi-head ----
        q = q.view(B, S, self.n_heads, self.head_dim).transpose(1, 2)  # (B, H, S, hd)
        k = k.view(B, S, self.n_heads, self.head_dim).transpose(1, 2)  # (B, H, S, hd)
        v = v.view(B, S, self.n_heads, self.head_dim).transpose(1, 2)  # (B, H, S, hd)

        # ---- Apply RoPE to Q and K ----
        if self.rotary is not None:
            q = self.rotary(q)
            k = self.rotary(k)

        # ---- Scaled dot-product attention ----
        scale = self.head_dim ** -0.5
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale  # (B, H, S, S)

        # Apply causal mask: tokens can only attend to positions <= their own
        attn_weights = attn_weights.masked_fill(
            self.causal_mask[:S, :S], float("-inf")
        )

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # ---- Weighted sum of values ----
        attn_output = torch.matmul(attn_weights, v)  # (B, H, S, hd)

        # ---- Merge heads back ----
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, S, D)

        # ---- Output projection ----
        output = self.out_dropout(self.out_proj(attn_output))

        return output

"""Transformer block with Pre-LayerNorm architecture."""

import torch
import torch.nn as nn

from src.common.config import MiniLLMConfig


class TransformerBlock(nn.Module):
    """A single Transformer block using Pre-LayerNorm architecture.

    Structure (modern standard used in GPT-2/3, LLaMA, etc.):
        x = x + Attention(LayerNorm(x))
        x = x + FFN(LayerNorm(x))

    LayerNorm is applied BEFORE each sub-layer, which improves training
    stability compared to the original Post-LN design.

    Args:
        config: MiniLLMConfig instance with model hyperparameters.
    """

    def __init__(self, config: MiniLLMConfig) -> None:
        super().__init__()

        from src.phase3_model.attention import CausalSelfAttention
        from src.phase3_model.feedforward import FeedForward

        self.attn_norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.attn = CausalSelfAttention(config)

        self.ffn_norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.ffn = FeedForward(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model).

        Returns:
            Output tensor of shape (batch_size, seq_len, d_model).
        """
        # Self-attention sub-layer (Pre-LN)
        x = x + self.attn(self.attn_norm(x))

        # Feed-forward sub-layer (Pre-LN)
        x = x + self.ffn(self.ffn_norm(x))

        return x

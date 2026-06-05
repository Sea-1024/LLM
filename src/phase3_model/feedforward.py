"""Feed-Forward Network with GELU, SiLU, and SwiGLU activation support."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.common.config import MiniLLMConfig


class FeedForward(nn.Module):
    """Position-wise Feed-Forward Network.

    Supports three activation variants:
      - "gelu":   Standard GELU (Gaussian Error Linear Unit).
      - "silu":   SiLU (Sigmoid Linear Unit, also known as Swish).
      - "swiglu": SwiGLU gated activation (used in LLaMA, PaLM).

    For SwiGLU, the hidden dimension is adjusted so the total parameter count
    remains roughly equivalent to the standard (non-gated) FFN.

    Args:
        config: MiniLLMConfig instance with model hyperparameters.
    """

    def __init__(self, config: MiniLLMConfig) -> None:
        super().__init__()

        self.activation = config.activation
        valid_activations = ("gelu", "silu", "swiglu")
        if self.activation not in valid_activations:
            raise ValueError(
                f"Unsupported activation '{self.activation}'. "
                f"Must be one of {valid_activations}."
            )

        if self.activation == "swiglu":
            # SwiGLU: FFN(x) = (SiLU(x @ W_gate) * (x @ W_up)) @ W_down
            # Two parallel projections -> gate + up, then element-wise multiply
            # Hidden dim is adjusted: 2 * (2/3 * d_ff) keeps param count ~ 2 * d_model * d_ff
            hidden_dim = int(2 * config.d_ff * 2 / 3)
            self.gate_proj = nn.Linear(config.d_model, hidden_dim, bias=False)
            self.up_proj = nn.Linear(config.d_model, hidden_dim, bias=False)
            self.down_proj = nn.Linear(hidden_dim, config.d_model, bias=False)
        else:
            # Standard FFN: up -> activation -> down
            self.up_proj = nn.Linear(config.d_model, config.d_ff, bias=False)
            self.down_proj = nn.Linear(config.d_ff, config.d_model, bias=False)
            self.gate_proj = None

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model).

        Returns:
            Output tensor of shape (batch_size, seq_len, d_model).
        """
        if self.activation == "swiglu":
            # Gated activation: gate * up
            gate = F.silu(self.gate_proj(x))   # (B, S, hidden_dim)
            up = self.up_proj(x)               # (B, S, hidden_dim)
            hidden = gate * up
        else:
            hidden = self.up_proj(x)           # (B, S, d_ff)
            if self.activation == "gelu":
                hidden = F.gelu(hidden)
            elif self.activation == "silu":
                hidden = F.silu(hidden)
            else:
                # Fallback (should not be reached due to __init__ validation)
                hidden = F.relu(hidden)

        hidden = self.dropout(hidden)
        output = self.down_proj(hidden)        # (B, S, d_model)
        return output

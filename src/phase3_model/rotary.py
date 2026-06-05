"""Rotary Position Embedding (RoPE) implementation.

Implements the GPT-NeoX style RoPE, which applies 2D rotation to pairs
of adjacent dimensions rather than splitting the tensor into halves.
"""

import torch
import torch.nn as nn


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) as used in LLaMA, GPT-NeoX, etc.

    Precomputes cosine and sine frequency tables and applies them
    to query/key tensors during the forward pass via pair-wise rotation.

    Args:
        dim: Head dimension (must be even).
        max_seq_len: Maximum sequence length for which to precompute frequencies.
        theta: Base frequency, defaults to 10000.0 (standard RoPE).
    """

    def __init__(self, dim: int, max_seq_len: int = 512, theta: float = 10000.0) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"dim ({dim}) must be even for RoPE")

        self.dim = dim
        self.max_seq_len = max_seq_len
        self.theta = theta

        # Precompute frequency bands: theta^(-2i/dim) for i in [0, dim/2)
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        # Positions: [0, 1, ..., max_seq_len-1]
        t = torch.arange(max_seq_len, dtype=torch.float32)
        # Outer product: (max_seq_len, dim/2)
        freqs = torch.outer(t, freqs)

        # Register precomputed cos/sin as persistent buffers
        self.register_buffer("freqs_cos", freqs.cos())  # (max_seq_len, dim/2)
        self.register_buffer("freqs_sin", freqs.sin())  # (max_seq_len, dim/2)

    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        """Apply RoPE to the input tensor.

        Uses GPT-NeoX style pair-wise rotation:
          1. Reshape last dim from (head_dim,) to (head_dim/2, 2) representing adjacent pairs.
          2. Apply 2D rotation to each pair: (x1*cos - x2*sin, x1*sin + x2*cos).
          3. Reshape back to original shape.

        Args:
            x: Tensor of shape (batch_size, n_heads, seq_len, head_dim).
            offset: Starting position offset for cached/incremental inference.

        Returns:
            Rotated tensor of the same shape as input.
        """
        seq_len = x.shape[2]

        if offset + seq_len > self.max_seq_len:
            raise ValueError(
                f"Requested positions [{offset}, {offset + seq_len}) exceed "
                f"max_seq_len ({self.max_seq_len})"
            )

        # Slice precomputed frequencies for the current sequence range
        cos = self.freqs_cos[offset:offset + seq_len]  # (seq_len, dim/2)
        sin = self.freqs_sin[offset:offset + seq_len]  # (seq_len, dim/2)

        # Reshape for broadcasting: (1, 1, seq_len, dim/2)
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)

        # Reshape x into pairs: (..., dim/2, 2)
        # x_reshaped[..., 0] = x1 (even-indexed), x_reshaped[..., 1] = x2 (odd-indexed)
        x_reshaped = x.reshape(*x.shape[:-1], -1, 2)
        x1 = x_reshaped[..., 0]  # (..., dim/2)
        x2 = x_reshaped[..., 1]  # (..., dim/2)

        # Apply rotation: [x1*cos - x2*sin, x1*sin + x2*cos]
        rotated_x1 = x1 * cos - x2 * sin
        rotated_x2 = x1 * sin + x2 * cos

        # Stack pairs back together and reshape to original shape
        rotated = torch.stack([rotated_x1, rotated_x2], dim=-1)  # (..., dim/2, 2)
        return rotated.reshape(*x.shape)

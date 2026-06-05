"""
Unit tests for the MiniLLM model architecture.

Covers model components: configuration, attention, RoPE,
feedforward, transformer block, full model forward, generation,
weight tying, and parameter counting.
"""
import torch
import sys
sys.path.insert(0, ".")

from src.common.config import MiniLLMConfig
from src.phase3_model.model import MiniLLM
from src.phase3_model.attention import CausalSelfAttention
from src.phase3_model.rotary import RotaryEmbedding
from src.phase3_model.feedforward import FeedForward
from src.phase3_model.transformer_block import TransformerBlock


def test_config() -> None:
    """Test config creation and validation."""
    config = MiniLLMConfig(d_model=512, n_heads=8)
    assert config.head_dim == 64, f"Expected head_dim=64, got {config.head_dim}"
    assert config.d_model % config.n_heads == 0, "d_model must be divisible by n_heads"
    print("[PASS] test_config")


def test_rotary() -> None:
    """Test RoPE forward pass preserves shape and approximately preserves norm."""
    dim = 32
    max_seq_len = 128
    rotary = RotaryEmbedding(dim=dim, max_seq_len=max_seq_len)
    x = torch.randn(1, 2, 16, dim)  # (B, n_heads, S, head_dim)
    out = rotary(x)
    assert out.shape == x.shape, f"Shape mismatch: {out.shape} vs {x.shape}"
    # Rotation should preserve norm approximately
    assert torch.allclose(x.norm(dim=-1), out.norm(dim=-1), atol=1e-4), \
        "RoPE did not preserve per-token norms"
    print("[PASS] test_rotary")


def test_attention() -> None:
    """Test causal self-attention preserves input shape."""
    config = MiniLLMConfig(d_model=256, n_heads=4, max_seq_len=64)
    attn = CausalSelfAttention(config)
    x = torch.randn(2, 32, 256)  # (B, S, D)
    out = attn(x)
    assert out.shape == x.shape, f"Attention output shape {out.shape} != {x.shape}"
    print("[PASS] test_attention")


def test_feedforward() -> None:
    """Test feedforward network preserves input shape."""
    config = MiniLLMConfig(d_model=256, d_ff=1024)
    ffn = FeedForward(config)
    x = torch.randn(2, 32, 256)  # (B, S, D)
    out = ffn(x)
    assert out.shape == x.shape, f"FFN output shape {out.shape} != {x.shape}"
    print("[PASS] test_feedforward")


def test_transformer_block() -> None:
    """Test full transformer block preserves input shape."""
    config = MiniLLMConfig(d_model=256, n_heads=4, d_ff=1024, max_seq_len=64)
    block = TransformerBlock(config)
    x = torch.randn(2, 32, 256)  # (B, S, D)
    out = block(x)
    assert out.shape == x.shape, f"Block output shape {out.shape} != {x.shape}"
    print("[PASS] test_transformer_block")


def test_model_forward() -> None:
    """Test full model forward pass produces correct output shape."""
    config = MiniLLMConfig(
        vocab_size=1024, d_model=256, n_heads=4, n_layers=2,
        d_ff=1024, max_seq_len=64,
    )
    model = MiniLLM(config)
    input_ids = torch.randint(0, 1024, (2, 32))
    logits = model(input_ids)
    assert logits.shape == (2, 32, 1024), \
        f"Logits shape {logits.shape} != (2, 32, 1024)"
    print("[PASS] test_model_forward")


def test_model_generate() -> None:
    """Test generation produces output at least as long as input."""
    config = MiniLLMConfig(
        vocab_size=1024, d_model=256, n_heads=4, n_layers=2,
        d_ff=1024, max_seq_len=64, eos_token_id=2,
    )
    model = MiniLLM(config)
    input_ids = torch.randint(3, 1024, (1, 8))  # Non-special tokens
    output = model.generate(input_ids, max_new_tokens=16)
    assert output.size(1) >= 8, \
        f"Generated sequence length {output.size(1)} < input length 8"
    print("[PASS] test_model_generate")


def test_weight_tying() -> None:
    """Test that LM head shares weights with token embedding when tied."""
    config = MiniLLMConfig(
        vocab_size=1024, d_model=256, n_heads=4, n_layers=2,
        d_ff=1024, max_seq_len=64, use_tied_weights=True,
    )
    model = MiniLLM(config)
    # Check that lm_head.weight is the same tensor as token_embedding.weight
    assert model.lm_head.weight.data_ptr() == model.token_embedding.weight.data_ptr(), \
        "LM head weight should share storage with token embedding"
    print("[PASS] test_weight_tying")


def test_parameter_count() -> None:
    """Test parameter counting using the project's count_parameters utility."""
    from src.common.utils import count_parameters
    config = MiniLLMConfig(
        vocab_size=8192, d_model=512, n_heads=8, n_layers=6,
        d_ff=2048, use_tied_weights=True,
    )
    model = MiniLLM(config)
    n_params = count_parameters(model)

    # Estimated formula (weights tied, no separate LM head):
    #   Embedding: vocab_size * d_model
    #   Per block: 4 * d_model^2 + 2 * d_model * d_ff + 4 * d_model (LayerNorm)
    expected = 8192 * 512 + 6 * (4 * 512 ** 2 + 2 * 512 * 2048 + 4 * 512)
    # Plus final LayerNorm: 2 * d_model
    expected += 2 * 512
    assert abs(n_params - expected) < 20000, \
        f"Expected ~{expected} parameters, got {n_params}"
    print(f"[PASS] test_parameter_count: {n_params:,} parameters")


if __name__ == "__main__":
    test_config()
    test_rotary()
    test_attention()
    test_feedforward()
    test_transformer_block()
    test_model_forward()
    test_model_generate()
    test_weight_tying()
    test_parameter_count()
    print("\nAll tests passed!")

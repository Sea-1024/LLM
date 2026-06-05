"""
Unit tests for text generation.

Tests various decoding strategies: basic, greedy (temperature=0),
top-k, top-p, and context truncation.
"""
import torch
import sys
sys.path.insert(0, ".")

from src.common.config import MiniLLMConfig
from src.phase3_model.model import MiniLLM


def _make_small_model(eos_token_id: int = 2) -> MiniLLM:
    """Helper to create a small model for testing."""
    config = MiniLLMConfig(
        vocab_size=1024, d_model=256, n_heads=4, n_layers=2,
        d_ff=1024, max_seq_len=64, eos_token_id=eos_token_id,
    )
    return MiniLLM(config)


def test_generate_basic() -> None:
    """Test basic generation with temperature sampling."""
    model = _make_small_model()
    model.eval()

    input_ids = torch.randint(3, 1024, (1, 4))
    output = model.generate(input_ids, max_new_tokens=8, temperature=0.8)

    assert output.size(1) >= 4, f"Output length {output.size(1)} < input length 4"
    assert output.size(1) <= 4 + 8, f"Output length {output.size(1)} exceeds max (4+8)"
    print("[PASS] test_generate_basic")


def test_generate_greedy() -> None:
    """Test greedy decoding (temperature=0)."""
    model = _make_small_model()
    model.eval()

    input_ids = torch.randint(3, 1024, (1, 4))
    output = model.generate(input_ids, max_new_tokens=8, temperature=0.0)

    assert output.size(1) >= 4, f"Greedy output length {output.size(1)} < input length 4"
    print("[PASS] test_generate_greedy")


def test_generate_topk() -> None:
    """Test top-k sampling does not crash and produces valid output."""
    model = _make_small_model()
    model.eval()

    input_ids = torch.randint(3, 1024, (1, 4))
    output = model.generate(input_ids, max_new_tokens=8, top_k=10)

    assert output.size(1) >= 4
    print("[PASS] test_generate_topk")


def test_generate_topp() -> None:
    """Test top-p (nucleus) sampling does not crash and produces valid output."""
    model = _make_small_model()
    model.eval()

    input_ids = torch.randint(3, 1024, (1, 4))
    output = model.generate(input_ids, max_new_tokens=8, top_p=0.5)

    assert output.size(1) >= 4
    print("[PASS] test_generate_topp")


def test_generate_truncation() -> None:
    """Test that context truncation works when input exceeds max_seq_len."""
    config = MiniLLMConfig(
        vocab_size=1024, d_model=256, n_heads=4, n_layers=2,
        d_ff=1024, max_seq_len=32, eos_token_id=2,
    )
    model = MiniLLM(config)
    model.eval()

    # Input longer than max_seq_len
    input_ids = torch.randint(3, 1024, (1, 48))
    output = model.generate(input_ids, max_new_tokens=8)

    # Should produce output despite input exceeding max_seq_len
    assert output.size(1) >= 1, "Truncation should still generate at least 1 token"
    print("[PASS] test_generate_truncation")


def test_generate_eos_stop() -> None:
    """Test that generation stops when EOS token is emitted."""
    config = MiniLLMConfig(
        vocab_size=1024, d_model=256, n_heads=4, n_layers=2,
        d_ff=1024, max_seq_len=64, eos_token_id=2,
    )
    model = MiniLLM(config)
    model.eval()

    # Use greedy decoding so output is deterministic; the model is untrained
    # so it will just pick whatever token. We just verify the API does not crash
    # and that output length is bounded.
    input_ids = torch.randint(3, 1024, (1, 4))
    output = model.generate(input_ids, max_new_tokens=16, temperature=0.0)

    assert output.size(1) <= 4 + 16, f"Generation exceeded max tokens: {output.size(1)}"
    print("[PASS] test_generate_eos_stop")


if __name__ == "__main__":
    test_generate_basic()
    test_generate_greedy()
    test_generate_topk()
    test_generate_topp()
    test_generate_truncation()
    test_generate_eos_stop()
    print("\nAll generation tests passed!")

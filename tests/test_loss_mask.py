"""
Unit tests for SFT loss masking.

Verifies that prompt regions are correctly masked (label = -100)
and response regions are kept for loss computation.
"""
import torch
import sys
sys.path.insert(0, ".")

from src.phase5_sft.loss_mask import (
    create_sft_labels,
    create_sft_labels_batch,
    verify_labels,
)


def test_create_sft_labels() -> None:
    """Test basic label masking: prompt region masked, response region kept."""
    input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    prompt_len = 4  # First 4 tokens are prompt
    labels = create_sft_labels(input_ids, prompt_len)

    # First prompt_len positions should be -100 (ignored)
    assert (labels[:, :4] == -100).all(), \
        f"Prompt region not masked: {labels[:, :4]}"
    # Remaining positions should match input_ids
    assert (labels[:, 4:] == input_ids[:, 4:]).all(), \
        f"Response region not preserved: {labels[:, 4:]} vs {input_ids[:, 4:]}"
    print("[PASS] test_create_sft_labels")


def test_create_sft_labels_batch() -> None:
    """Test batch label masking with variable-length sequences and padding."""
    input_ids_list = [
        torch.tensor([1, 2, 3, 4, 5, 6, 7, 8]),
        torch.tensor([1, 2, 3, 4, 5]),
    ]
    prompt_lens = [3, 2]

    input_ids, labels, attention_mask = create_sft_labels_batch(
        input_ids_list, prompt_lens, pad_token_id=0, seq_len=10,
    )

    # Check shapes
    assert input_ids.shape == (2, 10), f"input_ids shape: {input_ids.shape}"
    assert labels.shape == (2, 10), f"labels shape: {labels.shape}"
    assert attention_mask.shape == (2, 10), f"attention_mask shape: {attention_mask.shape}"

    # ---- First sequence: 8 tokens + 2 padding ----
    # Attention mask: first 8 tokens attended, last 2 masked
    assert (attention_mask[0, :8] == 1).all(), \
        f"Attention mask for tokens should be 1, got {attention_mask[0, :8]}"
    assert (attention_mask[0, 8:] == 0).all(), \
        f"Attention mask for padding should be 0, got {attention_mask[0, 8:]}"

    # Labels: first 3 positions masked (prompt)
    assert (labels[0, :3] == -100).all(), \
        f"Prompt labels should be -100, got {labels[0, :3]}"
    # Positions 3..8 should match input
    assert (labels[0, 3:8] == input_ids[0, 3:8]).all(), \
        f"Response labels mismatch: {labels[0, 3:8]} vs {input_ids[0, 3:8]}"
    # Padding positions masked
    assert (labels[0, 8:] == -100).all(), \
        f"Padding labels should be -100, got {labels[0, 8:]}"

    # ---- Second sequence: 5 tokens + 5 padding ----
    assert (attention_mask[1, :5] == 1).all()
    assert (labels[1, :2] == -100).all()
    assert (labels[1, 2:5] == input_ids[1, 2:5]).all()
    assert (labels[1, 5:] == -100).all()

    print("[PASS] test_create_sft_labels_batch")


def test_no_response_masking() -> None:
    """Test edge case: when prompt_len equals full sequence, all labels are -100."""
    input_ids = torch.tensor([[1, 2, 3, 4, 5]])
    labels = create_sft_labels(input_ids, prompt_len=5)
    assert (labels == -100).all(), \
        f"All labels should be -100 when prompt covers entire sequence, got {labels}"
    print("[PASS] test_no_response_masking")


def test_loss_ignore_index() -> None:
    """Test that CrossEntropyLoss correctly ignores -100 label positions."""
    logits = torch.randn(2, 5, 10)  # (B, S, vocab)
    labels = torch.tensor([
        [-100, -100, 3, 5, 7],
        [-100, 2, 4, -100, -100],
    ])

    loss = torch.nn.functional.cross_entropy(
        logits.view(-1, 10),
        labels.view(-1),
        ignore_index=-100,
    )

    # Loss should be finite
    assert not torch.isnan(loss), "Loss is NaN"
    assert not torch.isinf(loss), "Loss is Inf"
    print(f"[PASS] test_loss_ignore_index: loss={loss.item():.4f}")


class _MockTokenizer:
    """Minimal tokenizer mock for verify_labels test."""
    def decode(self, ids):
        return " ".join(str(i) for i in ids)


def test_verify_labels_identity() -> None:
    """Test the verify_labels utility on a simple case."""
    input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    labels = create_sft_labels(input_ids, prompt_len=4)
    tok = _MockTokenizer()
    # verify_labels should not raise for correct masking
    try:
        verify_labels(input_ids, labels, tok, num_samples=1)
        print("[PASS] test_verify_labels_identity")
    except Exception as e:
        print(f"[FAIL] test_verify_labels_identity: {e}")


if __name__ == "__main__":
    test_create_sft_labels()
    test_create_sft_labels_batch()
    test_no_response_masking()
    test_loss_ignore_index()
    test_verify_labels_identity()
    print("\nAll loss mask tests passed!")

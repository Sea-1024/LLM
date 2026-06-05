"""
Loss masking logic for SFT training.

During SFT we only want to compute the loss on the assistant response tokens.
All prompt tokens (including instruction, system message, etc.) should be
masked with ignore_index (-100) so they do not contribute to the loss.
"""

from typing import Any, Optional

import torch


def create_sft_labels(
    input_ids: torch.Tensor,
    prompt_len: int,
    ignore_index: int = -100,
) -> torch.Tensor:
    """
    Create labels for SFT training: mask everything before the response.

    Args:
        input_ids: (batch_size, seq_len) full tokenized sequence.
        prompt_len: Number of tokens in the prompt (before the response).
                    This many tokens from the start of each sequence will be
                    set to *ignore_index*.
        ignore_index: Value for masked positions (default -100 for
                      torch.nn.CrossEntropyLoss).

    Returns:
        labels: (batch_size, seq_len) with prompt positions set to ignore_index.
    """
    labels = input_ids.clone()
    labels[:, :prompt_len] = ignore_index
    return labels


def create_sft_labels_batch(
    input_ids_list: list[torch.Tensor],
    prompt_lens: list[int],
    pad_token_id: int = 0,
    ignore_index: int = -100,
    seq_len: int = 512,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Batch version: pad sequences to a fixed length and create labels.

    Args:
        input_ids_list: List of 1D LongTensors of varying lengths.
        prompt_lens: Prompt length for each sequence.
        pad_token_id: Padding token ID.
        ignore_index: Label value for masked positions.
        seq_len: Target sequence length.

    Returns:
        input_ids:   (batch_size, seq_len) padded input tensor.
        labels:      (batch_size, seq_len) masked label tensor.
        attention_mask: (batch_size, seq_len) 1 for real tokens, 0 for padding.
    """
    batch_size = len(input_ids_list)
    assert batch_size == len(prompt_lens), (
        f"input_ids_list and prompt_lens must have the same length "
        f"({batch_size} vs {len(prompt_lens)})"
    )

    input_ids = torch.full(
        (batch_size, seq_len), pad_token_id, dtype=torch.long
    )
    labels = torch.full(
        (batch_size, seq_len), ignore_index, dtype=torch.long
    )
    attention_mask = torch.zeros(
        (batch_size, seq_len), dtype=torch.long
    )

    for i, (ids, p_len) in enumerate(zip(input_ids_list, prompt_lens)):
        # Truncate if too long
        ids = ids[:seq_len]
        actual_len = len(ids)

        if actual_len == 0:
            # All padding -- already filled with ignore_index
            continue

        # Fill input_ids and attention_mask
        input_ids[i, :actual_len] = ids
        attention_mask[i, :actual_len] = 1

        # Copy labels from input_ids, then mask prompt
        labels[i, :actual_len] = ids

        # Mask prompt part (first p_len tokens)
        mask_end = min(p_len, actual_len)
        labels[i, :mask_end] = ignore_index

        # Mask padding positions (already set, but be explicit)
        labels[i, actual_len:] = ignore_index

    return input_ids, labels, attention_mask


def verify_labels(
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    tokenizer: Any,
    num_samples: int = 3,
) -> None:
    """
    Debug utility: print decoded text for verification.

    Shows which tokens are being learned (label != -100) vs masked (label == -100).

    Args:
        input_ids: (batch_size, seq_len) token ids.
        labels: (batch_size, seq_len) label ids.
        tokenizer: Tokenizer with a `decode` method that accepts a list of ints.
        num_samples: Number of samples to print.
    """
    for i in range(min(num_samples, input_ids.size(0))):
        print(f"\n{'='*60}")
        print(f" Sample {i} ")
        print(f"{'='*60}")

        learned_parts: list[str] = []
        masked_parts: list[str] = []

        for j in range(input_ids.size(1)):
            token_id = input_ids[i, j].item()
            label_id = labels[i, j].item()
            is_learned = label_id != -100
            marker = " [LEARN]" if is_learned else " [MASK]"

            if token_id > 0:  # Skip padding
                try:
                    decoded = tokenizer.decode([token_id])
                except Exception:
                    decoded = f"<{token_id}>"
                print(
                    f"  pos={j:4d}: {decoded!r:20s} -> label={label_id}{marker}"
                )

                if is_learned:
                    learned_parts.append(decoded)
                else:
                    masked_parts.append(decoded)

        print(f"\n  Learned tokens: {len(learned_parts)}")
        print(f"  Masked  tokens: {len(masked_parts)}")


# ---------------------------------------------------------------------------
# Utility: compute prompt length from formatted text
# ---------------------------------------------------------------------------

def compute_prompt_len(
    tokenizer: Any,
    prompt_text: str,
) -> int:
    """
    Compute the number of tokens in a prompt string.

    Args:
        tokenizer: Tokenizer with an `encode` method.
        prompt_text: The prompt portion of the formatted instruction.

    Returns:
        Number of tokens.
    """
    if hasattr(tokenizer, "encode"):
        result = tokenizer.encode(prompt_text)
    else:
        result = tokenizer(prompt_text).input_ids

    if hasattr(result, "__len__"):
        return len(result)
    if hasattr(result, "shape"):
        return result.shape[-1] if result.shape else 0
    return 0

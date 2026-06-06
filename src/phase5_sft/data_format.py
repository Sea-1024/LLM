"""
Convert JSON SFT data to tokenized binary format for efficient training.

Memory-mapped numpy arrays are used so the full dataset does not need to fit
in RAM during training.
"""

import json
import os
from pathlib import Path
from typing import Any, Optional

import numpy as np

from src.phase5_sft.templates import PromptTemplate


def format_and_tokenize(
    data_path: str,
    tokenizer: Any,
    template: PromptTemplate,
    output_dir: str,
    max_seq_len: int = 512,
    pad_token_id: Optional[int] = None,
) -> int:
    """
    Load JSON SFT data, format with the given template, tokenize, and save as
    memory-mapped binary arrays.

    Produces two files in *output_dir*:
        data.bin   -- (N, max_seq_len) uint16  token ids
        prompt_lens.npy -- (N,) int32  number of prompt tokens per sample

    Args:
        data_path: Path to a JSON file containing a list of
            {"instruction": ..., "input": ..., "output": ...} dicts.
        tokenizer: A tokenizer with an `encode` method (e.g. HuggingFace style)
            that returns a list (or ndarray) of int token ids.
        template: PromptTemplate instance.
        output_dir: Directory to write the binary files.
        max_seq_len: Maximum sequence length. Longer sequences are truncated.
        pad_token_id: Padding token id. If None, defaults to tokenizer.pad_token_id
            or 0.

    Returns:
        Number of samples successfully processed.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    with open(data_path, "r", encoding="utf-8") as f:
        samples: list[dict[str, str]] = json.load(f)

    if not samples:
        raise ValueError(f"No samples found in {data_path}")

    # Determine pad token id
    if pad_token_id is None:
        pad_token_id = getattr(tokenizer, "pad_token_id", 0)

    num_samples = len(samples)

    # Allocate arrays
    data = np.full((num_samples, max_seq_len), pad_token_id, dtype=np.uint16)
    prompt_lens = np.zeros(num_samples, dtype=np.int32)

    skipped: int = 0
    idx: int = 0

    for sample in samples:
        instruction = (sample.get("instruction") or "").strip()
        input_text = (sample.get("input") or "").strip()
        output_text = (sample.get("output") or "").strip()

        if not instruction or not output_text:
            skipped += 1
            continue

        # 1. Build the prompt portion (everything before the response)
        prompt_str, _resp_prefix = template.format_instruction(
            instruction, input_text
        )

        # 2. Build the full text
        full_str = template.format_full(instruction, input_text, output_text)

        # 3. Tokenize
        prompt_ids = _encode(tokenizer, prompt_str)
        full_ids = _encode(tokenizer, full_str)

        # Truncate
        full_ids = full_ids[:max_seq_len]

        if len(full_ids) == 0:
            skipped += 1
            continue

        prompt_len = min(len(prompt_ids), max_seq_len)
        actual_len = len(full_ids)

        # Store
        data[idx, :actual_len] = full_ids
        prompt_lens[idx] = prompt_len
        idx += 1

    if skipped:
        print(
            f"[data_format] Skipped {skipped} samples "
            f"(empty instruction/output or too short)"
        )

    # Trim to actual count
    data = data[:idx]
    prompt_lens = prompt_lens[:idx]

    # Save as memory-mapped arrays
    data_path_out = os.path.join(output_dir, "data.bin")
    prompt_lens_path_out = os.path.join(output_dir, "prompt_lens.npy")

    # Create memory-mapped file and copy data
    mmap = np.memmap(
        data_path_out, dtype=np.uint16, mode="w+", shape=data.shape
    )
    mmap[:] = data[:]
    mmap.flush()
    del mmap

    np.save(prompt_lens_path_out, prompt_lens)

    print(
        f"[data_format] Saved {idx} samples: "
        f"data={data_path_out} (shape={data.shape}), "
        f"prompt_lens={prompt_lens_path_out}"
    )

    return idx


def _encode(tokenizer: Any, text: str) -> list[int]:
    """Encode text to token ids, handling different tokenizer interfaces."""
    if text is None or not text:
        return []

    if hasattr(tokenizer, "encode"):
        result = tokenizer.encode(text)
        # Handle both list and tensor returns
        if hasattr(result, "tolist"):
            result = result.tolist()
        if isinstance(result, list):
            return [int(t) for t in result]
        return [int(result)]

    # Fallback: call as callable
    result = tokenizer(text)
    if hasattr(result, "input_ids"):
        result = result.input_ids
    if hasattr(result, "tolist"):
        result = result.tolist()
    if isinstance(result, list):
        return [int(t) for t in result]
    return [int(result)]


# ---------------------------------------------------------------------------
# Main entry-point
# ---------------------------------------------------------------------------

def main() -> None:
    """Tokenize and save binary SFT data from processed JSON files."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Tokenize SFT JSON data to binary format."
    )
    parser.add_argument(
        "--data_path", type=str, required=True,
        help="Path to JSON data file (e.g. data/sft_data/processed/train.json)."
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Directory for output binary files."
    )
    parser.add_argument(
        "--template_type", type=str, default="alpaca",
        choices=["alpaca", "chatml", "llama"],
        help="Prompt template type."
    )
    parser.add_argument(
        "--max_seq_len", type=int, default=512,
        help="Maximum sequence length."
    )
    parser.add_argument(
        "--tokenizer_path", type=str, default=None,
        help="Path to a saved tokenizer. If None, uses a default placeholder."
    )
    args = parser.parse_args()

    # Load tokenizer
    if args.tokenizer_path and os.path.exists(args.tokenizer_path):
        import pickle
        with open(args.tokenizer_path, "rb") as f:
            tokenizer = pickle.load(f)
    else:
        # Placeholder: a minimal tokenizer for standalone testing.
        # In production the real tokenizer is loaded from a checkpoint.
        class _DummyTokenizer:
            pad_token_id = 0
            def encode(self, text: str) -> list[int]:
                # Simple char-level encoding for testing
                return [ord(c) % 30000 for c in text]
        tokenizer = _DummyTokenizer()
        print("[data_format] WARNING: using dummy tokenizer (real one not found)")

    template = PromptTemplate(args.template_type)
    format_and_tokenize(
        data_path=args.data_path,
        tokenizer=tokenizer,
        template=template,
        output_dir=args.output_dir,
        max_seq_len=args.max_seq_len,
    )


if __name__ == "__main__":
    main()

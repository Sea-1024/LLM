"""
SFT data preparation: download, filter, and split the Alpaca dataset.

Usage:
    python -m src.phase5_sft.data_prepare [--max_samples 10000]
"""

import json
import os
import random
from collections import OrderedDict
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def download_alpaca(
    output_path: str,
    max_samples: int = 10000,
) -> list[dict[str, str]]:
    """
    Download the Alpaca dataset from HuggingFace (tatsu-lab/alpaca),
    filter it, and save as JSON.

    Args:
        output_path: Path to save the downloaded JSON file.
        max_samples: Maximum number of samples to keep.

    Returns:
        List of samples (dicts with instruction / input / output).
    """
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError:
        raise ImportError(
            "Please install the `datasets` library: pip install datasets"
        )

    print(f"[data_prepare] Downloading tatsu-lab/alpaca (max_samples={max_samples}) ...")
    dataset = load_dataset("tatsu-lab/alpaca", split="train")

    # Convert to list of dicts
    samples: list[dict[str, str]] = []
    for item in dataset:  # type: ignore[attr-defined]
        samples.append({
            "instruction": item.get("instruction", "") or "",
            "input": item.get("input", "") or "",
            "output": item.get("output", "") or "",
        })
        if len(samples) >= max_samples:
            break

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)

    print(f"[data_prepare] Saved {len(samples)} raw samples to {output_path}")
    return samples


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

def filter_samples(
    samples: list[dict[str, str]],
    min_output_len: int = 10,
    max_output_len: int = 500,
) -> list[dict[str, str]]:
    """
    Filter samples by output quality criteria.

    - Remove samples with empty instruction or output.
    - Remove samples where output is shorter than *min_output_len*.
    - Remove samples where output is longer than *max_output_len*.
    - De-duplicate by (instruction, input) key.

    Args:
        samples: List of sample dicts.
        min_output_len: Minimum character length of output.
        max_output_len: Maximum character length of output.

    Returns:
        Filtered list of samples.
    """
    seen: set[tuple[str, str]] = set()
    filtered: list[dict[str, str]] = []

    for s in samples:
        instruction = (s.get("instruction") or "").strip()
        input_text = (s.get("input") or "").strip()
        output_text = (s.get("output") or "").strip()

        # Must have instruction and output
        if not instruction or not output_text:
            continue

        # Output length bounds
        out_len = len(output_text)
        if out_len < min_output_len or out_len > max_output_len:
            continue

        # De-duplicate
        key = (instruction, input_text)
        if key in seen:
            continue
        seen.add(key)

        filtered.append({
            "instruction": instruction,
            "input": input_text,
            "output": output_text,
        })

    removed = len(samples) - len(filtered)
    print(
        f"[data_prepare] Filtered: {len(samples)} -> {len(filtered)} "
        f"(removed {removed})"
    )
    return filtered


# ---------------------------------------------------------------------------
# Split
# ---------------------------------------------------------------------------

def split_dataset(
    samples: list[dict[str, str]],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    """
    Split samples into train / val / test sets.

    Args:
        samples: Full list of samples.
        train_ratio: Fraction for training.
        val_ratio: Fraction for validation (test gets the remainder).
        seed: Random seed for reproducibility.

    Returns:
        (train_samples, val_samples, test_samples)
    """
    assert 0.0 < train_ratio < 1.0, "train_ratio must be in (0, 1)"
    assert 0.0 < val_ratio < 1.0, "val_ratio must be in (0, 1)"
    assert train_ratio + val_ratio < 1.0, "train_ratio + val_ratio must be < 1"

    rng = random.Random(seed)
    shuffled = list(samples)
    rng.shuffle(shuffled)

    n = len(shuffled)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    train = shuffled[:train_end]
    val = shuffled[train_end:val_end]
    test = shuffled[val_end:]

    print(
        f"[data_prepare] Split: train={len(train)}, val={len(val)}, "
        f"test={len(test)}"
    )
    return train, val, test


# ---------------------------------------------------------------------------
# Main entry-point
# ---------------------------------------------------------------------------

def load_from_parquet(parquet_path: str) -> list[dict[str, str]]:
    """
    Load Alpaca-format data from a local parquet file.

    Args:
        parquet_path: Path to the .parquet file.

    Returns:
        List of samples (dicts with instruction / input / output).
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError(
            "Please install pandas: pip install pandas"
        )

    print(f"[data_prepare] Loading from parquet: {parquet_path}")
    df = pd.read_parquet(parquet_path)
    samples: list[dict[str, str]] = []
    for _, row in df.iterrows():
        samples.append({
            "instruction": str(row.get("instruction", "") or ""),
            "input": str(row.get("input", "") or ""),
            "output": str(row.get("output", "") or ""),
        })
    print(f"[data_prepare] Loaded {len(samples)} samples from parquet")
    return samples


def main() -> None:
    """Download, filter, split, and save the Alpaca dataset."""
    import argparse

    parser = argparse.ArgumentParser(description="Prepare SFT data from Alpaca.")
    parser.add_argument(
        "--max_samples", type=int, default=10000,
        help="Maximum samples to download."
    )
    parser.add_argument(
        "--output_dir", type=str, default="data/sft_data/processed",
        help="Directory to save train/val/test JSON files."
    )
    parser.add_argument(
        "--raw_path", type=str, default="data/sft_data/raw/alpaca.json",
        help="Path to save the raw downloaded dataset (JSON)."
    )
    parser.add_argument(
        "--parquet_path", type=str, default=None,
        help="Optional: load data from a local parquet file instead of downloading."
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for splitting."
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_dir = Path(args.raw_path).parent
    raw_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load data
    if args.parquet_path:
        # Load from local parquet file
        samples = load_from_parquet(args.parquet_path)
        if args.max_samples and len(samples) > args.max_samples:
            samples = samples[:args.max_samples]
            print(f"[data_prepare] Truncated to {args.max_samples} samples")
    elif Path(args.raw_path).exists():
        print(f"[data_prepare] Loading cached raw data from {args.raw_path}")
        with open(args.raw_path, "r", encoding="utf-8") as f:
            samples = json.load(f)
    else:
        samples = download_alpaca(args.raw_path, max_samples=args.max_samples)

    # 2. Filter
    samples = filter_samples(samples)

    # 3. Split
    train, val, test = split_dataset(samples, seed=args.seed)

    # 4. Save
    for name, subset in [("train", train), ("val", val), ("test", test)]:
        path = output_dir / f"{name}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(subset, f, ensure_ascii=False, indent=2)
        print(f"[data_prepare] Saved {len(subset)} samples to {path}")

    print("[data_prepare] Done.")


if __name__ == "__main__":
    main()

"""
Batch tokenization of the corpus into memory-mapped binary files.

Tokenizes the corpus line-by-line in streaming fashion, packs tokens into
fixed-length sequences, and saves as numpy uint16 memmap files.

Usage:
    python -m src.phase2_tokenizer.tokenize_data
"""

import logging
import random
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from tqdm import tqdm

from src.common.utils import ensure_dir, setup_logging, set_seed

logger = logging.getLogger(__name__)

# <EOS> token ID (index 2 in special tokens list: <PAD>=0, <UNK>=1, <BOS>=2, <EOS>=3)
# Actually, the order is: <PAD>=0, <UNK>=1, <BOS>=2, <EOS>=3
_EOS_TOKEN_ID = 3
_BOS_TOKEN_ID = 2


def _resolve_eos_id(tokenizer: object) -> int:
    """Resolve the EOS token ID from a tokenizer.

    Args:
        tokenizer: Tokenizer object with get_vocab() or token_to_id() method.

    Returns:
        EOS token ID (defaults to 3 if not found).
    """
    if hasattr(tokenizer, "get_vocab"):
        vocab = tokenizer.get_vocab()
        eos = vocab.get("<EOS>") or vocab.get("<eos>") or vocab.get("</s>")
        if eos is not None:
            return eos
    if hasattr(tokenizer, "token_to_id"):
        eos = tokenizer.token_to_id("<EOS>") or tokenizer.token_to_id("<eos>")
        if eos is not None:
            return eos
    logger.warning("Could not resolve EOS token ID, defaulting to 3")
    return 3


def tokenize_corpus(
    tokenizer: object,
    corpus_path: str = "data/processed/corpus.txt",
    output_dir: str = "data/tokenized",
    seq_len: int = 512,
    val_split: float = 0.05,
    seed: int = 42,
) -> Tuple[int, int]:
    """Tokenize the corpus into fixed-length sequences.

    Reads the corpus in streaming fashion (line by line), tokenizes each line,
    and packs token IDs into sequences of length seq_len. Documents are
    separated by <EOS> tokens.

    Saves the result as:
    - {output_dir}/train.bin: training sequences (numpy uint16 memmap)
    - {output_dir}/val.bin: validation sequences (numpy uint16 memmap)

    Args:
        tokenizer: Tokenizer object with encode() method. The encode method
                   should return an object with .ids attribute.
        corpus_path: Path to the corpus text file (one document per line).
        output_dir: Directory to save .bin files.
        seq_len: Sequence length (number of tokens per sequence).
        val_split: Fraction of sequences to use for validation.
        seed: Random seed for reproducible train/val split.

    Returns:
        Tuple of (num_train_sequences, num_val_sequences).

    Raises:
        FileNotFoundError: If corpus_path does not exist.
        ValueError: If corpus is empty or parameters are invalid.
    """
    set_seed(seed)
    random.seed(seed)

    corpus_file = Path(corpus_path)
    if not corpus_file.exists():
        raise FileNotFoundError(f"Corpus file not found: {corpus_path}")

    file_size = corpus_file.stat().st_size
    if file_size == 0:
        raise ValueError(f"Corpus file is empty: {corpus_path}")
    if val_split < 0 or val_split >= 1:
        raise ValueError(f"val_split must be in [0, 1), got {val_split}")

    eos_id = _resolve_eos_id(tokenizer)
    logger.info("EOS token ID: %d", eos_id)

    output_path = ensure_dir(output_dir)

    logger.info("=" * 60)
    logger.info("Tokenizing Corpus")
    logger.info("=" * 60)
    logger.info("Corpus:   %s (%.2f MB)", corpus_path, file_size / (1024 * 1024))
    logger.info("Seq len:  %d", seq_len)
    logger.info("Val split: %.1f%%", val_split * 100)

    # First pass: collect all sequences into a list of lists
    # Each sequence is a list of seq_len token IDs
    all_sequences: List[np.ndarray] = []
    buffer: List[int] = []

    # Count total lines for progress bar
    logger.info("Counting lines...")
    with open(corpus_file, "r", encoding="utf-8") as f:
        total_lines = sum(1 for _ in f)

    logger.info("Tokenizing %d lines...", total_lines)
    with open(corpus_file, "r", encoding="utf-8") as f:
        pbar = tqdm(total=total_lines, desc="  Tokenizing", unit="lines")
        for line in f:
            line = line.strip()
            if not line:
                pbar.update(1)
                continue

            # Encode the line
            try:
                encoding = tokenizer.encode(line)
                token_ids = encoding.ids
            except Exception as e:
                logger.warning("Encoding error for line: %s... Error: %s", line[:50], e)
                pbar.update(1)
                continue

            if not token_ids:
                pbar.update(1)
                continue

            # Add token IDs to buffer, with EOS at end of document
            for tid in token_ids:
                buffer.append(tid)
                if len(buffer) >= seq_len:
                    seq = np.array(buffer[:seq_len], dtype=np.uint16)
                    all_sequences.append(seq)
                    buffer = buffer[seq_len:]

            # Add EOS separator between documents
            buffer.append(eos_id)
            if len(buffer) >= seq_len:
                seq = np.array(buffer[:seq_len], dtype=np.uint16)
                all_sequences.append(seq)
                buffer = buffer[seq_len:]

            pbar.update(1)
        pbar.close()

    logger.info("Total sequences generated: %d", len(all_sequences))
    logger.info("Remaining tokens in buffer: %d (discarded)", len(buffer))

    if len(all_sequences) == 0:
        raise ValueError(
            f"No sequences generated. Check corpus content and seq_len ({seq_len})."
        )

    # Shuffle and split into train/val
    logger.info("Shuffling and splitting...")
    random.shuffle(all_sequences)

    num_val = max(1, int(len(all_sequences) * val_split))
    num_train = len(all_sequences) - num_val

    train_seqs = all_sequences[:num_train]
    val_seqs = all_sequences[num_train:]

    logger.info("Train sequences: %d", num_train)
    logger.info("Val sequences:   %d", num_val)

    # Write train.bin as memmap
    train_path = output_path / "train.bin"
    logger.info("Writing train data to %s ...", train_path)
    _write_memmap(train_path, train_seqs, seq_len, num_train)

    # Write val.bin as memmap
    val_path = output_path / "val.bin"
    logger.info("Writing validation data to %s ...", val_path)
    _write_memmap(val_path, val_seqs, seq_len, num_val)

    # Print statistics
    total_tokens = len(all_sequences) * seq_len
    logger.info("--- Tokenization Statistics ---")
    logger.info("Total sequences:   %d", len(all_sequences))
    logger.info("Total tokens:      %d", total_tokens)
    logger.info("Train sequences:   %d", num_train)
    logger.info("Train tokens:      %d", num_train * seq_len)
    logger.info("Val sequences:     %d", num_val)
    logger.info("Val tokens:        %d", num_val * seq_len)
    logger.info("Train file size:   %.2f MB", train_path.stat().st_size / (1024 * 1024))
    logger.info("Val file size:     %.2f MB", val_path.stat().st_size / (1024 * 1024))

    return num_train, num_val


def _write_memmap(
    output_path: Path,
    sequences: List[np.ndarray],
    seq_len: int,
    num_sequences: int,
) -> None:
    """Write sequences to a numpy memory-mapped file.

    Args:
        output_path: Path for the output .bin file.
        sequences: List of uint16 numpy arrays, each of length seq_len.
        seq_len: Length of each sequence.
        num_sequences: Number of sequences to write.
    """
    mmap = np.memmap(
        str(output_path),
        dtype=np.uint16,
        mode="w+",
        shape=(num_sequences, seq_len),
    )
    for i in tqdm(range(num_sequences), desc=f"  Writing {output_path.name}"):
        mmap[i] = sequences[i]
    mmap.flush()
    # Close the memmap explicitly to ensure all data is written
    del mmap


def load_tokenizer(tokenizer_path: str = "models/tokenizer/tokenizer.json") -> object:
    """Load a trained tokenizer from JSON.

    Args:
        tokenizer_path: Path to tokenizer.json.

    Returns:
        Tokenizer object.

    Raises:
        FileNotFoundError: If tokenizer file not found.
    """
    from tokenizers import Tokenizer

    path = Path(tokenizer_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Tokenizer not found at {tokenizer_path}. "
            f"Run 'python -m src.phase2_tokenizer.train_tokenizer' first."
        )

    tokenizer = Tokenizer.from_file(str(path))
    logger.info("Loaded tokenizer from %s (vocab_size=%d)", tokenizer_path, tokenizer.get_vocab_size())
    return tokenizer


def main() -> None:
    """Run batch tokenization with default paths."""
    logger = setup_logging("tokenize_data", "logs/tokenize_data")

    try:
        # Load tokenizer
        tokenizer = load_tokenizer("models/tokenizer/tokenizer.json")

        # Tokenize corpus
        num_train, num_val = tokenize_corpus(
            tokenizer=tokenizer,
            corpus_path="data/processed/corpus.txt",
            output_dir="data/tokenized",
            seq_len=512,
            val_split=0.05,
        )

        logger.info(
            "Tokenization complete: %d train + %d val sequences",
            num_train,
            num_val,
        )
    except FileNotFoundError as e:
        logger.error("Required file not found: %s", e)
        raise
    except Exception as e:
        logger.error("Tokenization failed: %s", e)
        raise


if __name__ == "__main__":
    main()

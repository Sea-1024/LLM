"""
BPE tokenizer training using HuggingFace Tokenizers library.

Trains a Byte-Level BPE tokenizer on the merged corpus.

Usage:
    python -m src.phase2_tokenizer.train_tokenizer
"""

import os
import logging
import time
from pathlib import Path
from typing import List, Optional

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.trainers import BpeTrainer
from tokenizers.processors import TemplateProcessing

from src.common.utils import ensure_dir, setup_logging

logger = logging.getLogger(__name__)


def train_tokenizer(
    corpus_path: str = "data/processed/corpus.txt",
    vocab_size: int = 8192,
    output_path: str = "models/tokenizer/tokenizer.json",
    min_frequency: int = 2,
    special_tokens: Optional[List[str]] = None,
    show_progress: bool = True,
) -> Tokenizer:
    """Train a Byte-Level BPE tokenizer on the provided corpus.

    Args:
        corpus_path: Path to the training corpus (one document per line).
        vocab_size: Target vocabulary size.
        output_path: Path to save the trained tokenizer JSON.
        min_frequency: Minimum frequency for tokens to be merged.
        special_tokens: List of special tokens. Defaults to
                        ["<PAD>", "<UNK>", "<BOS>", "<EOS>"].
        show_progress: Whether to show training progress.

    Returns:
        Trained tokenizers.Tokenizer instance.

    Raises:
        FileNotFoundError: If corpus_path does not exist.
        ValueError: If corpus is empty or vocab_size is invalid.
    """
    if special_tokens is None:
        special_tokens = ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]

    corpus_file = Path(corpus_path)
    if not corpus_file.exists():
        raise FileNotFoundError(f"Corpus file not found: {corpus_path}")
    if corpus_file.stat().st_size == 0:
        raise ValueError(f"Corpus file is empty: {corpus_path}")
    if vocab_size < len(special_tokens) + 1:
        raise ValueError(
            f"vocab_size ({vocab_size}) must be at least "
            f"{len(special_tokens) + 1} (special tokens + 1)"
        )

    logger.info("=" * 60)
    logger.info("Training Byte-Level BPE Tokenizer")
    logger.info("=" * 60)
    logger.info("Corpus:        %s", corpus_path)
    logger.info("Corpus size:   %.2f MB", corpus_file.stat().st_size / (1024 * 1024))
    logger.info("Vocab size:    %d", vocab_size)
    logger.info("Min frequency: %d", min_frequency)
    logger.info("Special tokens: %s", special_tokens)

    # Initialize tokenizer with BPE model
    tokenizer = Tokenizer(BPE(unk_token="<UNK>"))

    # Pre-tokenizer: ByteLevel splits text into byte-level tokens
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)

    # Decoder: ByteLevel reconstructs text from byte tokens
    tokenizer.decoder = ByteLevelDecoder()

    # Trainer configuration
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        min_frequency=min_frequency,
        show_progress=show_progress,
    )

    # Train the tokenizer
    start_time = time.time()
    logger.info("Training tokenizer (this may take a few minutes)...")
    tokenizer.train([str(corpus_file)], trainer)
    elapsed = time.time() - start_time
    logger.info("Training completed in %.1f seconds", elapsed)

    # Get EOS token ID for post-processing
    vocab = tokenizer.get_vocab()
    eos_id = vocab.get("<EOS>")
    if eos_id is None:
        logger.warning("<EOS> token not found in vocabulary, post-processing disabled")
    else:
        # Post-processing: append <EOS> to each sequence
        tokenizer.post_processor = TemplateProcessing(
            single="$A <EOS>",
            pair="$A $B <EOS>",
            special_tokens=[
                ("<EOS>", eos_id),
            ],
        )
        logger.info("Post-processor configured (appends <EOS> to sequences)")

    # Save tokenizer
    output_file = Path(output_path)
    ensure_dir(str(output_file.parent))
    tokenizer.save(str(output_file))
    logger.info("Tokenizer saved to: %s", output_file)

    # Print training statistics
    logger.info("--- Training Statistics ---")
    logger.info("Vocabulary size: %d", tokenizer.get_vocab_size())
    logger.info("Training time:   %.1f seconds", elapsed)
    logger.info("Special tokens:")
    for token in special_tokens:
        token_id = vocab.get(token, "N/A")
        logger.info("  %-8s -> ID %s", token, token_id)

    # Verify the tokenizer works
    test_text = "Hello, world! This is a test sentence."
    encoded = tokenizer.encode(test_text)
    logger.info("--- Encoding Test ---")
    logger.info("Input:    %s", test_text)
    logger.info("Token IDs: %s", encoded.ids[:20])
    logger.info("Tokens:   %s", encoded.tokens[:20])
    logger.info("Decoded:  %s", tokenizer.decode(encoded.ids))

    return tokenizer


def main() -> None:
    """Train the BPE tokenizer with default settings."""
    logger = setup_logging("train_tokenizer", "logs/train_tokenizer")

    try:
        tokenizer = train_tokenizer(
            corpus_path="data/processed/corpus.txt",
            vocab_size=8192,
            output_path="models/tokenizer/tokenizer.json",
            min_frequency=2,
        )
        logger.info("Tokenizer training completed successfully.")
    except FileNotFoundError as e:
        logger.error("Corpus file not found. Run preprocessing first: %s", e)
        logger.info(
            "Run: python -m src.phase1_data.download && python -m src.phase1_data.preprocess"
        )
        raise
    except Exception as e:
        logger.error("Tokenizer training failed: %s", e)
        raise


if __name__ == "__main__":
    main()

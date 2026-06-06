"""
Text cleaning and corpus generation for pretraining data.

Usage:
    python -m src.phase1_data.preprocess
"""

import re
import os
import logging
from pathlib import Path
from typing import Iterator, Optional

from tqdm import tqdm

from src.common.utils import ensure_dir, setup_logging

logger = logging.getLogger(__name__)

# HTML tag pattern
_HTML_TAG_RE = re.compile(r"<[^>]*>")

# Non-printable characters (excluding newline and tab)
_NON_PRINTABLE_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")

# Multiple whitespace
_MULTI_SPACE_RE = re.compile(r"[ \t]+")

# Multiple newlines (3 or more -> 2)
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")

# Begins with = (WikiText heading markup)
_WIKI_HEADING_RE = re.compile(r"^=+\s.*\s=+$")

# WikiText section markers like "= Section ="
_WIKI_SECTION_RE = re.compile(r"(?:^|\n)\s*=+\s*[^=]+\s*=+\s*\n")


def clean_text(text: str) -> str:
    """Clean raw text by removing HTML tags and normalizing whitespace.

    Args:
        text: Raw input text.

    Returns:
        Cleaned text string.
    """
    if not isinstance(text, str):
        return ""

    # Remove HTML tags
    text = _HTML_TAG_RE.sub("", text)

    # Remove non-printable characters (keep \n and \t)
    text = _NON_PRINTABLE_RE.sub("", text)

    # Replace tabs with spaces
    text = text.replace("\t", " ")

    # Normalize multiple spaces/tabs to a single space
    text = _MULTI_SPACE_RE.sub(" ", text)

    # Normalize newlines: 3+ newlines -> 2 newlines (single blank line separator)
    text = _MULTI_NEWLINE_RE.sub("\n\n", text)

    # Strip leading/trailing whitespace per line, then overall
    lines = [line.strip() for line in text.splitlines()]
    text = "\n".join(lines)

    # Remove leading/trailing whitespace from entire text
    text = text.strip()

    return text


def filter_text(text: str, min_len: int = 10) -> bool:
    """Check if a text line should be kept.

    Filters out:
    - Lines shorter than min_len
    - Lines that are only whitespace
    - Wiki heading markers (lines like "= Heading =")

    Args:
        text: Text line to check.
        min_len: Minimum character length.

    Returns:
        True if the text should be kept.
    """
    if not text or not text.strip():
        return False

    stripped = text.strip()

    if len(stripped) < min_len:
        return False

    # Filter out Wiki heading markers
    if _WIKI_HEADING_RE.match(stripped):
        return False

    # Filter out lines that are purely punctuation/numeric gibberish
    # (keep lines with at least some alphabetic characters or meaningful content)
    alpha_count = sum(1 for c in stripped if c.isalpha())
    if alpha_count == 0 and len(stripped) < 20:
        return False

    return True


def process_wikitext(
    raw_dir: str = "data/raw/wikitext",
    output_file: str = "data/processed/wikitext_clean.txt",
) -> Optional[Path]:
    """Process WikiText raw files: read, clean, filter, and write to output.

    Reads files matching 'wikitext-103-*.txt' from raw_dir.

    Args:
        raw_dir: Directory containing raw WikiText files.
        output_file: Path for the cleaned output file.

    Returns:
        Path to the output file, or None if no input files found.
    """
    raw_path = Path(raw_dir)
    if not raw_path.exists():
        logger.warning("WikiText raw directory not found: %s", raw_dir)
        return None

    txt_files = sorted(raw_path.glob("wikitext-103-*.txt"))
    if not txt_files:
        logger.warning("No WikiText files found in %s", raw_dir)
        return None

    logger.info("Processing WikiText from %d file(s)...", len(txt_files))
    output_path = ensure_dir(str(Path(output_file).parent))

    total_lines = 0
    kept_lines = 0
    total_chars = 0

    # Count lines for progress bar
    total_file_lines = 0
    for fpath in txt_files:
        with open(fpath, "r", encoding="utf-8") as f:
            total_file_lines += sum(1 for _ in f)

    with open(output_path / Path(output_file).name, "w", encoding="utf-8") as out:
        for fpath in txt_files:
            logger.info("  Reading %s ...", fpath.name)
            with open(fpath, "r", encoding="utf-8") as f:
                for line in tqdm(f, total=total_file_lines, desc="  Cleaning WikiText"):
                    total_lines += 1
                    cleaned = clean_text(line)
                    if filter_text(cleaned):
                        out.write(cleaned + "\n")
                        kept_lines += 1
                        total_chars += len(cleaned)
                    elif cleaned.strip():
                        # Keep as paragraph break if it has meaningful content
                        pass

    out_file = output_path / Path(output_file).name
    logger.info(
        "WikiText processing complete: %d/%d lines kept (%.1f%%), %d chars -> %s",
        kept_lines,
        total_lines,
        100.0 * kept_lines / max(total_lines, 1),
        total_chars,
        out_file,
    )
    return out_file


def process_tinystories(
    raw_dir: str = "data/raw/tinystories",
    output_file: str = "data/processed/tinystories_clean.txt",
) -> Optional[Path]:
    """Process TinyStories raw files: extract story texts, clean, and write.

    Reads files matching 'tinystories-*.txt' from raw_dir.

    Args:
        raw_dir: Directory containing raw TinyStories files.
        output_file: Path for the cleaned output file.

    Returns:
        Path to the output file, or None if no input files found.
    """
    raw_path = Path(raw_dir)
    if not raw_path.exists():
        logger.warning("TinyStories raw directory not found: %s", raw_dir)
        return None

    txt_files = sorted(raw_path.glob("tinystories-*.txt"))
    if not txt_files:
        logger.warning("No TinyStories files found in %s", raw_dir)
        return None

    logger.info("Processing TinyStories from %d file(s)...", len(txt_files))
    output_path = ensure_dir(str(Path(output_file).parent))

    total_stories = 0
    kept_stories = 0
    total_chars = 0

    # Count stories for progress bar
    total_file_stories = 0
    for fpath in txt_files:
        with open(fpath, "r", encoding="utf-8") as f:
            total_file_stories += sum(1 for _ in f)

    with open(output_path / Path(output_file).name, "w", encoding="utf-8") as out:
        for fpath in txt_files:
            logger.info("  Reading %s ...", fpath.name)
            with open(fpath, "r", encoding="utf-8") as f:
                for line in tqdm(f, total=total_file_stories, desc="  Cleaning TinyStories"):
                    total_stories += 1
                    cleaned = clean_text(line.strip())
                    if filter_text(cleaned, min_len=20):
                        out.write(cleaned + "\n\n")
                        kept_stories += 1
                        total_chars += len(cleaned)

    out_file = output_path / Path(output_file).name
    logger.info(
        "TinyStories processing complete: %d/%d stories kept (%.1f%%), %d chars -> %s",
        kept_stories,
        total_stories,
        100.0 * kept_stories / max(total_stories, 1),
        total_chars,
        out_file,
    )
    return out_file


def build_corpus(
    wikitext_dir: str = "data/processed/wikitext_clean.txt",
    tinystories_dir: str = "data/processed/tinystories_clean.txt",
    output_path: str = "data/processed/corpus.txt",
) -> Path:
    """Merge cleaned WikiText and TinyStories into a single corpus file.

    Args:
        wikitext_dir: Path to cleaned WikiText file or directory.
        tinystories_dir: Path to cleaned TinyStories file or directory.
        output_path: Path for the merged corpus file.

    Returns:
        Path to the corpus file.
    """
    ensure_dir(str(Path(output_path).parent))
    logger.info("Building merged corpus...")

    total_chars = 0
    total_lines = 0

    with open(output_path, "w", encoding="utf-8") as out:
        # Process WikiText
        wt_path = Path(wikitext_dir)
        if wt_path.is_file():
            logger.info("  Adding WikiText from %s ...", wt_path)
            with open(wt_path, "r", encoding="utf-8") as f:
                for line in f:
                    stripped = line.strip()
                    if stripped:
                        out.write(stripped + "\n")
                        total_lines += 1
                        total_chars += len(stripped)
        elif wt_path.is_dir():
            for fpath in sorted(wt_path.glob("*.txt")):
                logger.info("  Adding WikiText from %s ...", fpath)
                with open(fpath, "r", encoding="utf-8") as f:
                    for line in f:
                        stripped = line.strip()
                        if stripped:
                            out.write(stripped + "\n")
                            total_lines += 1
                            total_chars += len(stripped)

        # Separator between data sources
        out.write("\n")

        # Process TinyStories
        ts_path = Path(tinystories_dir)
        if ts_path.is_file():
            logger.info("  Adding TinyStories from %s ...", ts_path)
            with open(ts_path, "r", encoding="utf-8") as f:
                for line in f:
                    stripped = line.strip()
                    if stripped:
                        out.write(stripped + "\n")
                        total_lines += 1
                        total_chars += len(stripped)
        elif ts_path.is_dir():
            for fpath in sorted(ts_path.glob("*.txt")):
                logger.info("  Adding TinyStories from %s ...", fpath)
                with open(fpath, "r", encoding="utf-8") as f:
                    for line in f:
                        stripped = line.strip()
                        if stripped:
                            out.write(stripped + "\n")
                            total_lines += 1
                            total_chars += len(stripped)

    corpus_path = Path(output_path)
    corpus_size = corpus_path.stat().st_size

    logger.info("Corpus built successfully:")
    logger.info("  Lines:      %d", total_lines)
    logger.info("  Characters: %d", total_chars)
    logger.info("  File size:  %.2f MB", corpus_size / (1024 * 1024))
    logger.info("  Output:     %s", corpus_path)

    return corpus_path


def main() -> None:
    """Run the full preprocessing pipeline."""
    logger = setup_logging("preprocess", "logs/preprocess")

    logger.info("=" * 60)
    logger.info("Phase 1: Preprocessing Pipeline")
    logger.info("=" * 60)

    # Step 1: Process WikiText
    logger.info("Step 1/3: Processing WikiText...")
    wt_output = process_wikitext()
    if wt_output:
        logger.info("WikiText cleaned: %s", wt_output)
    else:
        logger.warning("WikiText processing skipped (no input files)")

    # Step 2: Process TinyStories
    logger.info("Step 2/3: Processing TinyStories...")
    ts_output = process_tinystories()
    if ts_output:
        logger.info("TinyStories cleaned: %s", ts_output)
    else:
        logger.warning("TinyStories processing skipped (no input files)")

    # Step 3: Build merged corpus
    logger.info("Step 3/3: Building merged corpus...")
    corpus_path = build_corpus()
    logger.info("Final corpus: %s", corpus_path)

    logger.info("Preprocessing pipeline completed.")


if __name__ == "__main__":
    main()

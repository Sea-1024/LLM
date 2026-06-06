"""
Download pretraining datasets: WikiText-103 and TinyStories.

Usage:
    python -m src.phase1_data.download
"""

import os
import logging
from pathlib import Path
from typing import Optional

from tqdm import tqdm
from datasets import load_dataset, DownloadConfig

from src.common.utils import ensure_dir, setup_logging
from src.common.config import DataConfig

logger = logging.getLogger("download")

# Load Hugging Face token from config; fallback to environment variable
_DATA_CONFIG: Optional[DataConfig] = None


def _get_hf_token() -> str:
    """Get Hugging Face token from config file, or fallback to env."""
    global _DATA_CONFIG
    token = os.environ.get("HF_TOKEN", "")
    if token:
        return token
    if _DATA_CONFIG is None:
        config_path = Path("config/data.yaml")
        if config_path.exists():
            _DATA_CONFIG = DataConfig.from_yaml(str(config_path))
        else:
            _DATA_CONFIG = DataConfig()
    return _DATA_CONFIG.hf_token


def download_wikitext(
    output_dir: str = "data/raw/wikitext",
    split: str = "train",
    max_retries: int = 3,
) -> Path:
    """Download WikiText-103 dataset and save as raw text files.

    Args:
        output_dir: Directory to save raw text files.
        split: Dataset split to download ('train', 'validation', 'test').
        max_retries: Maximum number of download retries.

    Returns:
        Path to the output directory.

    Raises:
        RuntimeError: If download fails after all retries.
    """
    output_path = ensure_dir(output_dir)
    logger.info("Downloading WikiText-103 (split=%s)...", split)

    last_error: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            logger.info("  Attempt %d/%d", attempt, max_retries)
            dataset = load_dataset(
                "Salesforce/wikitext",
                "wikitext-103-raw-v1",
                split=split,
                download_config=DownloadConfig(token=_get_hf_token()),
            )

            # Save each article as a separate text file
            texts = []
            for item in tqdm(dataset, desc="  Saving WikiText-103"):
                text = item.get("text", "")
                if text and text.strip():
                    texts.append(text)

            # Write to a single file for this split
            output_file = output_path / f"wikitext-103-{split}.txt"
            with open(output_file, "w", encoding="utf-8") as f:
                for text in tqdm(texts, desc="  Writing texts"):
                    f.write(text + "\n")

            logger.info(
                "WikiText-103 (%s) downloaded successfully: %d documents -> %s",
                split,
                len(texts),
                output_file,
            )
            return output_path

        except Exception as e:
            last_error = e
            logger.warning("  Attempt %d failed: %s", attempt, e)
            if attempt < max_retries:
                import time
                time.sleep(2 ** attempt)

    raise RuntimeError(
        f"Failed to download WikiText-103 after {max_retries} attempts"
    ) from last_error


def download_tinystories(
    output_dir: str = "data/raw/tinystories",
    max_retries: int = 3,
) -> Path:
    """Download TinyStories dataset and save as raw text files.

    Downloads both 'train' and 'validation' splits.

    Args:
        output_dir: Directory to save raw text files.
        max_retries: Maximum number of download retries.

    Returns:
        Path to the output directory.

    Raises:
        RuntimeError: If download fails after all retries.
    """
    output_path = ensure_dir(output_dir)
    logger.info("Downloading TinyStories...")

    for split_name in ["train", "validation"]:
        last_error: Optional[Exception] = None
        for attempt in range(1, max_retries + 1):
            try:
                logger.info("  Split=%s, Attempt %d/%d", split_name, attempt, max_retries)
                dataset = load_dataset(
                    "roneneldan/TinyStories",
                    split=split_name,
                    download_config=DownloadConfig(token=_get_hf_token()),
                )

                texts = []
                for item in tqdm(dataset, desc=f"  Loading TinyStories ({split_name})"):
                    story = item.get("story", "")
                    if story and story.strip():
                        texts.append(story)

                output_file = output_path / f"tinystories-{split_name}.txt"
                with open(output_file, "w", encoding="utf-8") as f:
                    for text in tqdm(texts, desc="  Writing stories"):
                        f.write(text + "\n")

                logger.info(
                    "TinyStories (%s) downloaded: %d stories -> %s",
                    split_name,
                    len(texts),
                    output_file,
                )
                break  # success, exit retry loop

            except Exception as e:
                last_error = e
                logger.warning("  Attempt %d failed: %s", attempt, e)
                if attempt < max_retries:
                    import time
                    time.sleep(2 ** attempt)
        else:
            raise RuntimeError(
                f"Failed to download TinyStories ({split_name}) "
                f"after {max_retries} attempts"
            ) from last_error

    return output_path


def main() -> None:
    """Download both pretraining datasets."""
    logger = setup_logging("download", "logs/download")

    logger.info("=" * 60)
    logger.info("Phase 1: Downloading Pretraining Datasets")
    logger.info("=" * 60)

    try:
        wikitext_dir = download_wikitext()
        logger.info("WikiText-103 saved to: %s", wikitext_dir)
    except Exception as e:
        logger.error("Failed to download WikiText-103: %s", e)
        logger.warning("Continuing with TinyStories download...")

    try:
        tinystories_dir = download_tinystories()
        logger.info("TinyStories saved to: %s", tinystories_dir)
    except Exception as e:
        logger.error("Failed to download TinyStories: %s", e)

    logger.info("Download phase completed.")


if __name__ == "__main__":
    main()

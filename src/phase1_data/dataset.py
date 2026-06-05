"""
PyTorch Dataset for pretraining with memory-mapped token files.

Usage:
    from src.phase1_data.dataset import PretrainDataset, create_dataloader
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)


class PretrainDataset(Dataset):
    """PyTorch Dataset for autoregressive language model pretraining.

    Uses memory-mapped numpy arrays for efficient I/O. Each sample returns
    (input_ids, labels) where labels are input_ids shifted by one position
    for next-token prediction.
    """

    def __init__(
        self,
        data_path: str,
        seq_len: int,
        tokenizer: object,
        pad_token_id: Optional[int] = None,
    ) -> None:
        """Initialize the pretraining dataset.

        Args:
            data_path: Path to the .bin file (numpy memmap file) containing
                       tokenized sequences.
            seq_len: Sequence length for training.
            tokenizer: Tokenizer object (must have pad_token_id attribute or
                       special token IDs accessible via token_to_id).
            pad_token_id: Override pad_token_id. If None, inferred from tokenizer.

        Raises:
            FileNotFoundError: If data_path does not exist.
            ValueError: If the memmap file is empty or shape incompatible.
        """
        self.seq_len = seq_len
        self.tokenizer = tokenizer
        self.data_path = data_path

        data_file = Path(data_path)
        if not data_file.exists():
            raise FileNotFoundError(f"Tokenized data file not found: {data_path}")

        file_size = data_file.stat().st_size
        if file_size == 0:
            raise ValueError(f"Tokenized data file is empty: {data_path}")

        # Try to infer the number of sequences
        # The file should be a flat memmap of shape (num_sequences, seq_len)
        total_elements = file_size // np.dtype(np.uint16).itemsize
        num_sequences = total_elements // seq_len

        if num_sequences == 0:
            raise ValueError(
                f"Not enough data in {data_path}: {total_elements} tokens, "
                f"need at least {seq_len} tokens per sequence"
            )

        # Load as memory-mapped array
        try:
            self.data = np.memmap(
                data_path,
                dtype=np.uint16,
                mode="r",
                shape=(num_sequences, seq_len),
            )
        except ValueError as e:
            # Try reshaping if the file wasn't saved with a fixed shape
            logger.warning(
                "Could not reshape memmap directly, trying flat layout: %s", e
            )
            self.data = np.memmap(
                data_path,
                dtype=np.uint16,
                mode="r",
                shape=(num_sequences, seq_len),
                offset=0,
            )

        self.num_sequences = num_sequences
        logger.info(
            "PretrainDataset loaded: %d sequences of length %d from %s",
            num_sequences,
            seq_len,
            data_path,
        )

        # Resolve pad_token_id
        if pad_token_id is not None:
            self.pad_token_id = pad_token_id
        elif hasattr(self.tokenizer, "pad_token_id"):
            self.pad_token_id = self.tokenizer.pad_token_id
        elif hasattr(self.tokenizer, "token_to_id"):
            self.pad_token_id = self.tokenizer.token_to_id("<PAD>")
        else:
            self.pad_token_id = 0
            logger.warning("Could not determine pad_token_id, defaulting to 0")

    def __len__(self) -> int:
        """Return the number of sequences in the dataset."""
        return self.num_sequences

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a single training sample.

        Args:
            idx: Sequence index.

        Returns:
            Tuple of (input_ids, labels), each of shape (seq_len,).
            input_ids: tokens [0..seq_len-1]
            labels: tokens [1..seq_len], with last position being pad_token_id
                    (for next-token prediction).
        """
        if idx < 0 or idx >= self.num_sequences:
            raise IndexError(
                f"Index {idx} out of range [0, {self.num_sequences})"
            )

        tokens = self.data[idx].astype(np.int64)

        input_ids = torch.from_numpy(tokens.copy())
        labels = torch.full((self.seq_len,), self.pad_token_id, dtype=torch.long)
        labels[:-1] = torch.from_numpy(tokens[1:].copy())

        return input_ids, labels


def create_dataloader(
    dataset: PretrainDataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
    drop_last: bool = True,
) -> DataLoader:
    """Create a DataLoader for the pretraining dataset.

    Args:
        dataset: PretrainDataset instance.
        batch_size: Batch size.
        shuffle: Whether to shuffle the data.
        num_workers: Number of data loading worker processes (0 for main process).
        pin_memory: Whether to pin memory for faster GPU transfer.
        drop_last: Whether to drop the last incomplete batch.

    Returns:
        Configured DataLoader.
    """
    if len(dataset) == 0:
        raise ValueError("Dataset is empty, cannot create DataLoader")

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        # Use a simple collate since we already return tensors
        collate_fn=None,
    )

    logger.info(
        "DataLoader created: batch_size=%d, shuffle=%s, num_batches=%d",
        batch_size,
        shuffle,
        len(dataloader),
    )
    return dataloader

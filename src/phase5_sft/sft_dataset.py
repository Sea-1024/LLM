"""
PyTorch Dataset for SFT training.

Loads pre-tokenized binary data stored as memory-mapped numpy arrays.
The binary file stores token IDs of shape (N, seq_len); prompt lengths
are stored in a separate .npy file.

During __getitem__, the prompt portion of each sequence is masked so the
model only learns from response tokens.
"""

import os
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class SFTDataset(Dataset):
    """
    Dataset for SFT training. Loads pre-tokenized binary data.

    The data file is structured as sequences of (seq_len) tokens.
    Prompt lengths are loaded from a companion .npy file (or inferred).

    Labels are created on-the-fly by masking prompt positions with -100.
    """

    def __init__(
        self,
        data_path: str,
        prompt_len_path: Optional[str] = None,
        seq_len: int = 512,
        pad_token_id: int = 0,
    ) -> None:
        """
        Args:
            data_path: Path to .bin file with shape (N, seq_len) uint16.
            prompt_len_path: Path to .npy file with prompt lengths (N,) int32.
                             If None, assumes 60% of seq_len as prompt.
            seq_len: Sequence length (must match the saved data).
            pad_token_id: Padding token ID used in the binary data.
        """
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")

        self.seq_len = seq_len
        self.pad_token_id = pad_token_id

        # Memory-map the data
        raw = np.memmap(data_path, dtype=np.uint16, mode="r")
        total_elements = len(raw)

        if total_elements % seq_len != 0:
            print(
                f"[SFTDataset] WARNING: data size ({total_elements}) "
                f"is not divisible by seq_len ({seq_len}). "
                f"Truncating {total_elements % seq_len} elements."
            )

        self.n_samples = total_elements // seq_len
        if self.n_samples == 0:
            raise ValueError(
                f"Data file {data_path} is too small for seq_len={seq_len}"
            )

        self.data = raw[: self.n_samples * seq_len].reshape(
            self.n_samples, seq_len
        )

        # Load or infer prompt lengths
        if prompt_len_path and os.path.exists(prompt_len_path):
            self.prompt_lens = np.load(prompt_len_path).astype(np.int32)
            if len(self.prompt_lens) != self.n_samples:
                print(
                    f"[SFTDataset] WARNING: prompt_lens size "
                    f"({len(self.prompt_lens)}) != n_samples "
                    f"({self.n_samples}). Truncating/expanding."
                )
                self.prompt_lens = self._resize_prompt_lens(
                    self.prompt_lens, self.n_samples
                )
        else:
            # Fallback: assume first 60% is prompt (approximate)
            default_ratio = 0.6
            self.prompt_lens = np.full(
                self.n_samples,
                int(seq_len * default_ratio),
                dtype=np.int32,
            )
            print(
                f"[SFTDataset] No prompt_len_path provided. "
                f"Using default prompt_len={int(seq_len * default_ratio)} "
                f"({default_ratio:.0%} of seq_len) for all samples."
            )

    @staticmethod
    def _resize_prompt_lens(arr: np.ndarray, target_size: int) -> np.ndarray:
        """Resize prompt_lens array to match target_size."""
        if len(arr) >= target_size:
            return arr[:target_size]
        # Pad with last value
        pad_val = arr[-1] if len(arr) > 0 else 0
        return np.pad(
            arr, (0, target_size - len(arr)),
            mode="constant", constant_values=pad_val,
        )

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            input_ids: (seq_len,) LongTensor.
            labels:    (seq_len,) LongTensor with prompt positions set to -100.
        """
        input_ids = torch.tensor(self.data[idx], dtype=torch.long)
        prompt_len = int(self.prompt_lens[idx])

        # Create labels: mask prompt part
        labels = input_ids.clone()
        labels[:prompt_len] = -100

        # Also mask padding positions (token id == pad_token_id)
        pad_mask = input_ids == self.pad_token_id
        labels[pad_mask] = -100

        return input_ids, labels


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def create_sft_dataloader(
    data_path: str,
    batch_size: int,
    shuffle: bool = True,
    prompt_len_path: Optional[str] = None,
    num_workers: int = 0,
    seq_len: int = 512,
    drop_last: bool = True,
) -> DataLoader:
    """
    Create a DataLoader for SFT training.

    Args:
        data_path: Path to .bin data file.
        batch_size: Batch size.
        shuffle: Whether to shuffle.
        prompt_len_path: Path to prompt lengths .npy file.
        num_workers: Number of data loading workers.
        seq_len: Sequence length.
        drop_last: Drop the last incomplete batch.

    Returns:
        A torch DataLoader yielding (input_ids, labels) tuples.
    """
    dataset = SFTDataset(
        data_path=data_path,
        prompt_len_path=prompt_len_path,
        seq_len=seq_len,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=drop_last,
    )

"""Packed-token datasets for next-token prediction."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class PackedTokenDataset(Dataset[torch.Tensor]):
    """Read fixed-length training examples from a memory-mapped uint32 stream."""

    def __init__(
        self, path: str | Path, sequence_length: int, random_windows: bool = False
    ) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(
                f"Missing token file: {self.path}. Run src/prepare_data.py first."
            )
        self.tokens = np.memmap(self.path, dtype=np.uint32, mode="r")
        self.sequence_length = sequence_length
        self.random_windows = random_windows
        self.num_sequences = max(0, (len(self.tokens) - 1) // sequence_length)
        if self.num_sequences == 0:
            raise ValueError(f"{self.path} does not contain one complete sequence")

    def __len__(self) -> int:
        return self.num_sequences

    def __getitem__(self, index: int) -> torch.Tensor:
        if self.random_windows:
            # Changing starts across epochs prevents every book from being cut at
            # the same permanent context boundaries. Validation remains fixed.
            max_start = len(self.tokens) - self.sequence_length - 1
            start = int(torch.randint(0, max_start + 1, ()).item())
        else:
            start = index * self.sequence_length
        stop = start + self.sequence_length + 1
        # Copy detaches the tensor from the read-only memory map.
        return torch.from_numpy(np.asarray(self.tokens[start:stop], dtype=np.int64).copy())


def get_dataloaders(
    train_path: str | Path,
    val_path: str | Path,
    sequence_length: int,
    batch_size: int,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> tuple[DataLoader[torch.Tensor], DataLoader[torch.Tensor]]:
    train_dataset = PackedTokenDataset(train_path, sequence_length, random_windows=True)
    val_dataset = PackedTokenDataset(val_path, sequence_length)
    common = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": num_workers > 0,
    }
    if num_workers > 0:
        # Keep a small host-side queue ready without allowing workers to
        # consume unbounded RAM when a large book is being streamed.
        common["prefetch_factor"] = 2
    train_loader = DataLoader(train_dataset, shuffle=True, drop_last=True, **common)
    val_loader = DataLoader(val_dataset, shuffle=False, drop_last=False, **common)
    return train_loader, val_loader

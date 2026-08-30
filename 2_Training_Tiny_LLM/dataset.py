from __future__ import annotations
import torch
from pathlib import Path

class ByteDataset:
    """
        Holds the raw bytes of the text in the text file and yields (x, y) blocks 
        for LM.
        - block_size: sequence length (context_window)
        - split: fraction for training (test is val)
    """

    def __init__(self, path: str, block_size: int = 256, split: float = 0.9):
        
        # Faster & zero-copy loading
        data = torch.frombuffer(bytearray(Path(path).read_bytes()), dtype=torch.uint8).long()

        n = int(len(data) * split)
        self.train = data[:n]
        self.val = data[n:]
        self.block_size = block_size

    def get_batch(self, which: str, batch_size: int, device: torch.device):

        # which -> 'train' or 'validation'

        buf = self.train if which == 'train' else self.val
        assert len(buf) > self.block_size + 1, "File too small for given block_size"
        ix = torch.randint(0, len(buf) - self.block_size - 1, (batch_size,))
        x = torch.stack([buf[i : i + self.block_size] for i in ix])
        y = torch.stack([buf[i + 1: i + self.block_size + 1] for i in ix])

        return x.to(device), y.to(device)
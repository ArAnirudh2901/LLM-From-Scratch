from __future__ import annotations
import torch


class ByteTokenizer:
    """
        The most basic tokenizer. Converts every token 
        to UTF-8 encoding. Example: H -> 72
        But this is very inefficient as there would a 
        lot of tokens to process.
    """

    def encode(self, s: str) -> torch.Tensor:
        return torch.tensor(list(s.encode('utf-8')), dtype=torch.long)

    def decode(self, ids) -> str:
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return bytes(ids).decode('utf-8', errors='ignore')

    @property
    def vocab_size(self) -> int:
        return 256
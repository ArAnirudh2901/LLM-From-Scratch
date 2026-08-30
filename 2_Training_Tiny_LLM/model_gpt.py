import math
import torch
import torch.nn as nn

class GPT:
    def __init__(self, vocab_size: int, block_size: int, n_layer: int = 6, n_head: int = 6, n_embd: int = 384, dropout: float = 0.0):
        super().__init__()
        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.drop - nn.Dropout(dropout)
        self.blocks = nn.ModuleList([Block(n_embd, n_head, dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)

        self.apply(self._init_weights)

    
"""Grammar encoder: a small pretrained English encoder (Ettin, ModernBERT architecture; MIT licence, open training
data) that reads the question alongside the reader. Each reader token maps to fixed encoder tokens through a table
built once from the encoder's tokenizer (no tokenizer at test time); the encoder reads the question both ways and
each question position gets the mean of its tokens' states, added to the reader's input embedding through a
projection that starts at zero, so step 0 is exactly the reader alone. Only the embedding rows the table uses are
kept (exact: no other row can ever be looked up). People ('@k') get their own rows, started near common first names.

    spec, state = build('jhu-clsp/ettin-encoder-68m', vocab.itos, layers=0)   # train_reader --genc
"""
from __future__ import annotations

import re

import torch
import torch.nn as nn

W = 8          # encoder tokens kept per reader token (a longer piece is cut)
N_ENT = 32     # '@0'..'@31' (reader.N_ENT)
NAMES = ['John', 'Mary', 'David', 'Sarah', 'Michael', 'Anna', 'James', 'Emma', 'Ravi', 'Priya', 'Rahul', 'Neha',
         'Amit', 'Sita', 'Arjun', 'Maria', 'Peter', 'Linda', 'Ahmed', 'Fatima', 'Tom', 'Lucy', 'Raj', 'Asha']


def _config(hf: dict):
    from transformers import ModernBertConfig
    cfg = ModernBertConfig(**hf)
    cfg._attn_implementation = 'sdpa'   # no flash-attn on a T4 or in the browser
    return cfg


class GrammarEncoder(nn.Module):
    def __init__(self, spec: dict, d_model: int):
        super().__init__()
        from transformers import ModernBertModel
        from model_reader import RMSNorm
        self.enc = ModernBertModel(_config(spec['hf']))
        h = spec['hf']['hidden_size']
        self.norm = RMSNorm(h)
        self.proj = nn.Linear(h, d_model)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        self.register_buffer('tab', torch.zeros(spec['n_vocab'], W, dtype=torch.long))
        self.register_buffer('tab_n', torch.zeros(spec['n_vocab'], dtype=torch.long))
        self.pad = spec['hf']['pad_token_id']

    def forward(self, idx, n_pre):
        """(B, T, d_model) additions for each row's first n_pre positions (the question); zero elsewhere."""
        B, T = idx.shape
        dev = idx.device
        n = self.tab_n[idx] * (torch.arange(T, device=dev)[None] < n_pre[:, None])
        flat = n.flatten()
        src = torch.repeat_interleave(torch.arange(B * T, device=dev), flat)   # reader position of each encoder token
        out = torch.zeros(B * T, self.proj.in_features, device=dev)
        if src.numel():
            sub = torch.arange(src.numel(), device=dev) - (flat.cumsum(0) - flat)[src]
            row, col = src // T, (n.cumsum(1) - n).flatten()[src] + sub
            L = int(n.sum(1).max())
            ids = torch.full((B, L), self.pad, dtype=torch.long, device=dev)
            att = torch.zeros(B, L, dtype=torch.long, device=dev)
            ids[row, col] = self.tab[idx.flatten()[src], sub]
            att[row, col] = 1
            with torch.set_grad_enabled(torch.is_grad_enabled() and self.enc.embeddings.tok_embeddings.weight.requires_grad):
                h = self.enc(input_ids=ids, attention_mask=att).last_hidden_state
            out = out.index_add(0, src, h[row, col].float()) / flat.clamp(min=1)[:, None]
        return self.proj(self.norm(out.view(B, T, -1))) * (n > 0)[..., None]


def _surface(t: str):
    """The text a reader token stands for, or None when it is a person."""
    if re.fullmatch(r'@\d+', t):
        return None
    if t.startswith('▁'):   # first piece of a split word
        return ' ' + t[1:]
    if t.startswith('##'):   # continuation piece
        return t[2:]
    return ' ' + t if t[0].isalnum() else t


def build(name: str, itos: list[str], layers: int = 0, revision: str | None = None):
    """(spec, state) of a new grammar encoder for the reader vocabulary `itos`: its first `layers` layers (0: all),
    the token table and only the embedding rows the table uses."""
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name, revision=revision)
    src = AutoModel.from_pretrained(name, revision=revision).float()
    cls, sep, mask, pad = tok.cls_token_id, tok.sep_token_id, tok.mask_token_id, tok.pad_token_id
    fixed = {'<bos>': [cls], '<c>': [sep], '<focus>': [sep], '<unk>': [mask]}   # unknown word: the MLM's own blank
    rows = []
    for t in itos:
        if t in fixed:
            rows.append(fixed[t])
        elif t.startswith('<'):   # markers never inside a question (<pad>, <eos>, ...)
            rows.append([])
        else:
            s = _surface(t)
            rows.append(None if s is None else tok(s, add_special_tokens=False).input_ids[:W])
    keep = sorted({i for r in rows if r for i in r} | {cls, sep, mask, pad})
    cmap = {o: i for i, o in enumerate(keep)}
    emb = src.embeddings.tok_embeddings.weight.detach()
    one = [x[0] for x in (tok(' ' + n, add_special_tokens=False).input_ids for n in NAMES) if len(x) == 1]
    gen = torch.Generator().manual_seed(0)
    people = emb[one].mean(0) + 0.5 * emb[one].std(0) * torch.randn(N_ENT, emb.size(1), generator=gen)   # name-like, distinct
    tab = torch.zeros(len(itos), W, dtype=torch.long)
    tab_n = torch.zeros(len(itos), dtype=torch.long)
    for i, r in enumerate(rows):
        r = [len(keep) + min(int(itos[i][1:]), N_ENT - 1)] if r is None else [cmap[x] for x in r]
        tab[i, :len(r)], tab_n[i] = torch.tensor(r, dtype=torch.long), len(r)
    hf = src.config.to_dict()
    k = layers or hf['num_hidden_layers']
    hf.update({'vocab_size': len(keep) + N_ENT, 'num_hidden_layers': k, 'reference_compile': False,
               'pad_token_id': cmap[pad], 'cls_token_id': cmap[cls], 'sep_token_id': cmap[sep],
               'bos_token_id': cmap[cls], 'eos_token_id': cmap[sep]})
    for drop in ('_name_or_path', 'architectures', 'torch_dtype', 'dtype'):
        hf.pop(drop, None)
    state = {'enc.' + n: v for n, v in src.state_dict().items()
             if not (m := re.match(r'layers\.(\d+)\.', n)) or int(m.group(1)) < k}
    state['enc.embeddings.tok_embeddings.weight'] = torch.cat([emb[keep], people])
    state.update({'tab': tab, 'tab_n': tab_n})
    spec = {'name': name, 'revision': revision, 'hf': hf, 'n_vocab': len(itos)}
    return spec, state

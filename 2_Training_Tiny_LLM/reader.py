"""The reader model: a question in, the fact language out (kinlogic). Vocabulary, example encoding, and fast
grammar-constrained greedy decoding with a key/value cache. Shared by train_reader.py and ask_reader.py."""
from __future__ import annotations

import gzip
import json
import random
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F

from kinlogic.facts import allowed_next, vocabulary
from qa_data import _heads, encode_prefix

SPECIALS = ['<pad>', '<unk>', '<bos>', '<c>', '<sep>', '<eos>']
N_ENT, N_NEW = 32, 16
ENTS = [f'@{i}' for i in range(N_ENT)]
NEWS = [f'#{i}' for i in range(1, N_NEW + 1)]


class Vocab:
    def __init__(self, itos: list[str]):
        self.itos = itos
        self.stoi = {t: i for i, t in enumerate(itos)}
        self.unk = self.stoi['<unk>']

    def encode(self, toks):
        return [self.stoi.get(t, self.unk) for t in toks]

    def __len__(self):
        return len(self.itos)


def build_vocab(n_generated: int = 100000, clutrr: str = 'data/clutrr_facts/train.jsonl.gz', seed: int = 0) -> Vocab:
    """Every word the generator writes, CLUTRR words seen twice, the fact language and placeholders."""
    from kinlogic import generate
    from kinlogic.prepare import LEXICON
    rng = random.Random(seed)
    gen_words = set()
    for _ in range(n_generated):
        ex = generate.sample(rng)
        gen_words.update(t for t in ex['tokens'] if t[0] not in '@#')
    counts = Counter()
    if Path(clutrr).exists():
        for line in gzip.open(clutrr, 'rt'):
            counts.update(t for t in json.loads(line)['tokens'] if t[0] not in '@#')
    words = gen_words | {w for w, c in counts.items() if c >= 2} | LEXICON
    fixed = SPECIALS + ENTS + NEWS + [t for t in vocabulary() if t != ';']
    return Vocab(fixed + sorted(words - set(fixed)))


def lf_tokens(lf: str) -> list[str]:
    return ['<sep>' if t == ';' else t for t in lf.split()]


def permute_entities(tokens, lf, rng: random.Random):
    """Random placeholder numbers (the model must not care which @k a person gets)."""
    perm = rng.sample(range(N_ENT), N_ENT)
    m = lambda t: f'@{perm[int(t[1:])]}' if t[0] == '@' and t[1:].isdigit() and int(t[1:]) < N_ENT else t
    return [m(t) for t in tokens], ' '.join(m(t) for t in lf.split())


def encode_example(vocab: Vocab, tokens, lf, block: int, rng: random.Random | None = None, unk_p: float = 0.0):
    """ids = <bos> question <c> facts <eos>; returns (ids, index of <c>) or None when too long."""
    if rng is not None:
        tokens, lf = permute_entities(tokens, lf, rng)
        if unk_p:
            tokens = [('<unk>' if t[0].isalpha() and rng.random() < unk_p else t) for t in tokens]
    ids = [vocab.stoi['<bos>']] + vocab.encode(tokens) + [vocab.stoi['<c>']]
    c = len(ids) - 1
    ids += vocab.encode(lf_tokens(lf)) + [vocab.stoi['<eos>']]
    return (ids, c) if len(ids) <= block else None


def collate(batch, pad: int, device, mult: int = 16):
    """Padded ids, fact targets (after <c>) and story targets (before it)."""
    T = max(len(ids) for ids, _ in batch)
    T = (T + mult - 1) // mult * mult
    idx = torch.full((len(batch), T), pad, dtype=torch.long)
    for i, (ids, _) in enumerate(batch):
        idx[i, :len(ids)] = torch.tensor(ids)
    nxt = torch.full_like(idx, -100)
    nxt[:, :-1] = idx[:, 1:]
    nxt[nxt == pad] = -100
    col = torch.arange(T)[None, :]
    c = torch.tensor([c for _, c in batch])[:, None]
    tgt = torch.where(col >= c, nxt, -100)          # the facts and <eos>
    lm = torch.where((col >= 1) & (col < c - 1), nxt, -100)   # the question's words
    return idx.to(device), tgt.to(device), lm.to(device)


@torch.no_grad()
def decode(model, vocab: Vocab, inputs: list[list[str]], device, max_new: int = 96, bs: int = 64) -> list[str]:
    """Greedy facts for each question (token lists from prepare), only ever writing valid statements."""
    model.eval()
    out = []
    sep, eos, pad = vocab.stoi['<sep>'], vocab.stoi['<eos>'], vocab.stoi['<pad>']
    for b in range(0, len(inputs), bs):
        chunk = inputs[b:b + bs]
        pre = [[vocab.stoi['<bos>']] + vocab.encode(t) + [vocab.stoi['<c>']] for t in chunk]
        pre = [p[-(model.block_size - max_new):] for p in pre]
        plen = torch.tensor([len(p) for p in pre], device=device)
        idx = torch.full((len(pre), int(plen.max())), pad, dtype=torch.long)
        for i, p in enumerate(pre):
            idx[i, :len(p)] = torch.tensor(p)
        idx = idx.to(device)
        cache, logits = encode_prefix(model, idx, plen)
        cache = [list(kv) for kv in cache]
        P = idx.size(1)
        ents = [list(dict.fromkeys(t for t in toks if t[0] == '@' and t in vocab.stoi)) for toks in chunk]
        written = [[] for _ in chunk]
        done = [False] * len(chunk)
        for step in range(max_new):
            lg = logits.float().cpu()
            nxt = []
            for i in range(len(chunk)):
                if done[i]:
                    nxt.append(pad)
                    continue
                allowed = [vocab.stoi['<sep>' if a == ';' else a] for a in allowed_next(written[i], ents[i])
                           if ('<sep>' if a == ';' else a) in vocab.stoi]
                best = allowed[int(lg[i, allowed].argmax())]
                if best == eos or step == max_new - 1:
                    done[i] = True
                else:
                    written[i].append(';' if best == sep else vocab.itos[best])
                nxt.append(best)
            if all(done):
                break
            tok = torch.tensor(nxt, device=device)[:, None]
            pos = (plen + step).clamp(max=model.block_size - 1)[:, None]
            x = model.tok_emb(tok) + model.pos_emb(pos)
            G = step + 1
            mask = torch.cat([torch.arange(P, device=device)[None, :] < plen[:, None],
                              torch.ones(len(chunk), G, dtype=torch.bool, device=device)], 1)[:, None, None, :]
            for li, blk in enumerate(model.blocks):
                q, k, v = _heads(blk.attn, blk.ln1(x))
                cache[li][0] = torch.cat([cache[li][0], k], 2)
                cache[li][1] = torch.cat([cache[li][1], v], 2)
                y = F.scaled_dot_product_attention(q, cache[li][0], cache[li][1], attn_mask=mask)
                x = x + blk.attn.proj(y.transpose(1, 2).reshape(x.shape))
                x = x + blk.ffn(blk.ln2(x))
            logits = model.head(model.ln_f(x[:, 0]))
        for w in written:
            while w and w[-1] == ';':
                w.pop()
            out.append(' '.join(w))
    return out

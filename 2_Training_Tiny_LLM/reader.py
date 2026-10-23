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
from qa_data import _heads, encode_prefix, hidden_states

SPECIALS = ['<pad>', '<unk>', '<bos>', '<c>', '<sep>', '<eos>']
N_ENT, N_NEW = 32, 16
ENTS = [f'@{i}' for i in range(N_ENT)]
NEWS = [f'#{i}' for i in range(1, N_NEW + 1)]


class Vocab:
    """Token ids. With `merges` (subword.py) a word outside the vocabulary becomes its sub-word pieces instead of
    <unk>; placeholders, the fact language and known words stay single tokens."""

    def __init__(self, itos: list[str], merges=None):
        self.itos = itos
        self.stoi = {t: i for i, t in enumerate(itos)}
        self.unk = self.stoi['<unk>']
        self.merges = [tuple(m) for m in merges] if merges else None
        self._rank = None
        self._cache = {}

    @classmethod
    def from_ck(cls, ck):
        return cls(ck['vocab'], ck.get('merges'))

    def word(self, t) -> list[int]:
        i = self.stoi.get(t)
        if i is not None or not self.merges or t[0] in '@#<':
            return [self.unk if i is None else i]
        got = self._cache.get(t)
        if got is None:
            from subword import pieces, ranks
            self._rank = self._rank or ranks(self.merges)
            got = self._cache[t] = [self.stoi.get(p, self.unk) for p in pieces(t, self._rank)]
        return got

    def encode(self, toks):
        if not self.merges:
            return [self.stoi.get(t, self.unk) for t in toks]
        return [i for t in toks for i in self.word(t)]

    def __len__(self):
        return len(self.itos)


def build_vocab(n_generated: int = 100000, clutrr: str = 'data/clutrr_facts/train.jsonl.gz', seed: int = 0,
                extra: str | None = None) -> Vocab:
    """Every word the generator writes, CLUTRR words seen twice, words of `extra` (reworded questions) seen twice,
    the fact language and placeholders."""
    from kinlogic import generate
    from kinlogic.prepare import LEXICON
    rng = random.Random(seed)
    gen_words = set()
    for _ in range(n_generated):
        ex = generate.sample(rng)
        gen_words.update(t for t in ex['tokens'] if t[0] not in '@#')
    counts = Counter()
    for path in (clutrr, extra):
        if path and Path(path).exists():
            for line in gzip.open(path, 'rt'):
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


def _allowed_ids(vocab: Vocab, written, ents):
    return [vocab.stoi['<sep>' if a == ';' else a] for a in allowed_next(written, ents) if ('<sep>' if a == ';' else a) in vocab.stoi]


def _prefix(model, vocab: Vocab, chunk, device, max_new):
    pre = [[vocab.stoi['<bos>']] + vocab.encode(t) + [vocab.stoi['<c>']] for t in chunk]
    pre = [p[-(model.block_size - max_new):] for p in pre]
    plen = torch.tensor([len(p) for p in pre], device=device)
    idx = torch.full((len(pre), int(plen.max())), vocab.stoi['<pad>'], dtype=torch.long)
    for i, p in enumerate(pre):
        idx[i, :len(p)] = torch.tensor(p)
    ents = [list(dict.fromkeys(t for t in toks if t[0] == '@' and t in vocab.stoi)) for toks in chunk]
    if getattr(model, 'arch', None) == 'v10':   # log-probabilities, copy head included
        cache, logp = model.prefill(idx.to(device), plen)
        return cache, logp, plen, idx.size(1), ents
    cache, logits = encode_prefix(model, idx.to(device), plen)
    return [list(kv) for kv in cache], logits, plen, idx.size(1), ents


def _step(model, cache, tok, pos, plen, P, G):
    """One new token per row, attending to the row's prefix (first plen of P positions) and its G written tokens."""
    if getattr(model, 'arch', None) == 'v10':
        return model.step(cache, tok, pos, plen, P, G)
    device = tok.device
    x = model.tok_emb(tok[:, None]) + model.pos_emb(pos.clamp(max=model.block_size - 1)[:, None])
    mask = torch.cat([torch.arange(P, device=device)[None, :] < plen[:, None],
                      torch.ones(len(tok), G, dtype=torch.bool, device=device)], 1)[:, None, None, :]
    for li, blk in enumerate(model.blocks):
        q, k, v = _heads(blk.attn, blk.ln1(x))
        cache[li][0] = torch.cat([cache[li][0], k], 2)
        cache[li][1] = torch.cat([cache[li][1], v], 2)
        y = F.scaled_dot_product_attention(q, cache[li][0], cache[li][1], attn_mask=mask)
        x = x + blk.attn.proj(y.transpose(1, 2).reshape(x.shape))
        x = x + blk.ffn(blk.ln2(x))
    return model.head(model.ln_f(x[:, 0])).float()


def _text(written):
    w = list(written)
    while w and w[-1] == ';':
        w.pop()
    return ' '.join(w)


@torch.no_grad()
def generate(model, vocab: Vocab, inputs: list[list[str]], device, k: int = 1, sample: bool = False,
             temperature: float = 1.0, max_new: int = 96, bs: int = 32, check=None) -> list[list[tuple[str, float]]]:
    """Fact readings for each question, only ever writing valid statements.
    Beam search (k best, by mean log-probability per token) or k samples (for reinforcement learning).
    check(facts) -> bool: beam readings it rejects (no family fits) do not count toward the k, so the search goes on
    to consistent ones (execution-guided decoding, Wang et al. 2018); they are used only if nothing passes.
    Returns, per question, [(facts, score)] best first (samples in drawing order)."""
    model.eval()
    sep, eos = vocab.stoi['<sep>'], vocab.stoi['<eos>']
    out = []
    for b in range(0, len(inputs), bs):
        chunk = inputs[b:b + bs]
        cache, logits, plen, P, ents = _prefix(model, vocab, chunk, device, max_new)
        width = k
        rows = torch.arange(len(chunk), device=device)
        if sample:   # k independent samples per question
            rows = rows.repeat_interleave(k)
        hyps = [(int(r), [], 0.0, 0) for r in rows.tolist()]   # (question, written, log-prob, tokens)
        cache = [[kv[0][rows], kv[1][rows]] for kv in cache]
        logits, plen_h = logits[rows], plen[rows]
        done = [[] for _ in chunk]
        spare = [[] for _ in chunk]   # finished readings `check` rejected
        for step in range(max_new):
            lp = torch.log_softmax(logits / (temperature if sample else 1.0), -1).cpu()
            cand = []   # (question, parent row, token, new log-prob)
            for h, (qi, written, score, n) in enumerate(hyps):
                allowed = _allowed_ids(vocab, written, ents[qi])
                a = torch.tensor(allowed)
                la = lp[h, a] - torch.logsumexp(lp[h, a], 0)   # renormalised over what the grammar allows
                if sample:
                    t = int(a[torch.multinomial(la.exp(), 1)])
                    cand.append((qi, h, t, score + float(la[allowed.index(t)])))
                else:
                    top = la.topk(min(width, len(allowed)))
                    cand += [(qi, h, int(a[i]), score + float(v)) for v, i in zip(top.values, top.indices)]
            keep = []
            if sample:
                keep = cand
            else:
                by_q = {}
                for c in cand:
                    by_q.setdefault(c[0], []).append(c)
                for qi, cs in by_q.items():
                    cs.sort(key=lambda c: -c[3])
                    keep += cs[:width]
            nxt_hyps, parents, toks = [], [], []
            for qi, h, t, sc in keep:
                written, n = hyps[h][1], hyps[h][3] + 1
                if t == eos or step == max_new - 1:
                    txt = _text(written)
                    (done if sample or check is None or check(txt) else spare)[qi].append((txt, sc / n))
                    continue
                nxt_hyps.append((qi, written + [';' if t == sep else vocab.itos[t]], sc, n))
                parents.append(h)
                toks.append(t)
            if not sample:   # a question stops once it has k finished readings
                full = {qi for qi in range(len(chunk)) if len(done[qi]) >= width}
                sel = [i for i, hy in enumerate(nxt_hyps) if hy[0] not in full]
                nxt_hyps, parents, toks = [nxt_hyps[i] for i in sel], [parents[i] for i in sel], [toks[i] for i in sel]
            if not nxt_hyps:
                break
            pidx = torch.tensor(parents, device=device)
            cache = [[kv[0][pidx], kv[1][pidx]] for kv in cache]
            plen_h = plen_h[pidx]
            hyps = nxt_hyps
            logits = _step(model, cache, torch.tensor(toks, device=device), plen_h + step, plen_h, P, step + 1)
        for qi in range(len(chunk)):
            res = done[qi] or spare[qi]
            if not sample:   # best first, one entry per distinct reading
                best = {}
                for f, sc in res:
                    best[f] = max(sc, best.get(f, float('-inf')))
                res = sorted(best.items(), key=lambda d: -d[1])[:width]
            out.append(res or [('', float('-inf'))])
    return out


def decode(model, vocab: Vocab, inputs: list[list[str]], device, max_new: int = 96, bs: int = 64) -> list[str]:
    """Greedy facts for each question."""
    return [r[0][0] for r in generate(model, vocab, inputs, device, k=1, max_new=max_new, bs=bs)]


def fact_losses(model, idx, tgt, lm, vocab):
    """Mean cross-entropy of the facts (copy head included for v10) and of the question's words."""
    if getattr(model, 'arch', None) == 'v10':
        mt = tgt != -100
        lp, h = model.fact_logprobs(idx, mt, vocab.stoi['<c>'])
        loss_f = F.nll_loss(lp, tgt[mt])
    else:
        h = hidden_states(model, idx)
        mt = tgt != -100
        loss_f = F.cross_entropy(model.head(h[mt]).float(), tgt[mt])
    ml = lm != -100
    loss_lm = F.cross_entropy(model.head(h[ml]).float(), lm[ml]) if ml.any() else torch.zeros((), device=idx.device)
    return loss_f, loss_lm


def fact_token_logprobs(model, idx, tgt, vocab):
    """Per-token log-probability of each target fact token (B, T), zero elsewhere; and the target mask."""
    mt = tgt != -100
    if getattr(model, 'arch', None) == 'v10':
        lp, _ = model.fact_logprobs(idx, mt, vocab.stoi['<c>'])
    else:
        lp = torch.log_softmax(model.head(hidden_states(model, idx)[mt]).float(), -1)
    out = torch.zeros(tgt.shape, device=idx.device)
    out[mt] = lp.gather(1, tgt[mt][:, None])[:, 0]
    return out, mt

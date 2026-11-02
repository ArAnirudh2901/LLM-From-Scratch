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

from kinlogic.facts import FOCUS, allowed_at, allowed_next, budget_filter, compact_people, vocabulary
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
        toks = _bare_symbols(list(toks))
        if not self.merges:
            return [self.stoi.get(t, self.unk) for t in toks]
        return [i for t in toks for i in self.word(t)]

    def __len__(self):
        return len(self.itos)


SPARE_SYMBOLS = ['¤', '§', '•', '☆', '∆', '¥', '€']


def _bare_symbols(toks):
    """A bare '@' or '#' is a code symbol ('P @ Q means ...'), but both mark placeholders (@3, #2) and have no token of their
    own: each becomes a spare symbol the question does not use, the same one throughout (2026-10-25: 167 real
    questions read these as <unk>, so two different codes looked identical)."""
    if '@' not in toks and '#' not in toks:
        return toks
    spare = iter(s for s in SPARE_SYMBOLS if s not in toks)
    sub = {b: next(spare) for b in ('@', '#') if b in toks}
    return [sub.get(t, t) for t in toks]


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


OF_FORM = False   # train_reader --of_form: targets in the text's order (facts.align_of)


def encode_example(vocab: Vocab, tokens, lf, block: int, rng: random.Random | None = None, unk_p: float = 0.0):
    """ids = <bos> question <c> facts <eos>; returns (ids, index of <c>) or None when too long."""
    got = compact_people(tokens, lf, N_ENT)   # ids past @31 have no token: renumber, or drop rows with too many people
    if got is None:
        return None
    tokens, lf = got
    if OF_FORM:
        from kinlogic.facts import align_of
        lf = align_of(list(tokens), lf)
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
    T = max(len(b[0]) for b in batch)
    T = (T + mult - 1) // mult * mult
    idx = torch.full((len(batch), T), pad, dtype=torch.long)
    for i, (ids, *_) in enumerate(batch):   # rows may carry a twin id (train_reader --cons_p)
        idx[i, :len(ids)] = torch.tensor(ids)
    nxt = torch.full_like(idx, -100)
    nxt[:, :-1] = idx[:, 1:]
    nxt[nxt == pad] = -100
    col = torch.arange(T)[None, :]
    c = torch.tensor([b[1] for b in batch])[:, None]
    tgt = torch.where(col >= c, nxt, -100)          # the facts and <eos>
    lm = torch.where((col >= 1) & (col < c - 1), nxt, -100)   # the question's words
    return idx.to(device), tgt.to(device), lm.to(device)


def _allowed_ids(vocab: Vocab, written, ents, left: int = 1 << 30, reserve: int = 0):
    toks = allowed_next(written, ents)
    if reserve:   # near the token budget: finish the statement and write the question (kinlogic.facts.budget_filter)
        toks = budget_filter(written, toks, left, reserve)
    return [vocab.stoi['<sep>' if a == ';' else a] for a in toks if ('<sep>' if a == ';' else a) in vocab.stoi]


AUTO_CAP = 256   # max_new <= 0: as many output tokens as fit, up to this (long stories need ~150)


def _budget(model, vocab: Vocab, chunk, max_new):
    """Output budget of a chunk: max_new when its questions fit, else never cutting a question more than 96 would.
    A chunk of sentence reads (FOCUS) is never cut: its budget shrinks to what fits (callers check it is enough)."""
    fit = model.block_size - (max(len(vocab.encode(t)) for t in chunk) + 2)
    if max_new <= 0:
        max_new = max(96, min(AUTO_CAP, fit))
    if any(FOCUS in t for t in chunk):
        return max(1, min(max_new, fit))
    return max(min(max_new, 96), min(max_new, fit))


FOCUS_ROOM = 64   # output tokens a sentence read needs: one sentence's facts are at most 31 in generated rows


def focus_fits(model, vocab: Vocab, room: int = FOCUS_ROOM):
    """tokens -> whether one sentence read fits the model with `room` tokens to write (exam.with_split_search)."""
    return lambda t: len(vocab.encode(t)) + 2 + room <= model.block_size


FOCUS_K = 4   # beam of each sentence read: the join keeps its 3 best (exam.with_split_search k)


def with_focus(read, model, vocab: Vocab, device, base=None, **kw):
    """read plus focus-search joins (exam.with_split_search focus=True); sentence reads by `base`, else a
    FOCUS_K-wide beam of this model. Shared by run_real, the suites and the verifier so all read the same way."""
    from kinlogic.exam import with_split_search
    base = base or (lambda b: generate(model, vocab, b, device, k=FOCUS_K, **kw))
    return with_split_search(read, focus=True, min_sentences=2, fits=focus_fits(model, vocab), base=base)


def _shared_question(model, vocab: Vocab, pre):
    """(question ids, per-row '<focus>' sentence ids) when every row is one question read at another sentence and the
    model keeps the question blind to the sentence (focus_mask): then the question is run once (prefill_focus)."""
    f = vocab.stoi.get(FOCUS)
    if (f is None or not getattr(model, 'focus_mask', False) or not model.prefix_lm or model.f_id != f
            or getattr(model, 'loop_noise', 0) or any(b.canon_a is not None for b in model.blocks)):
        return None   # causal readers, per-row loop noise and Canon need every row run in full
    cut = [p.index(f) if f in p else 0 for p in pre]
    if min(cut) < 1 or len({tuple(p[:i]) for p, i in zip(pre, cut)}) != 1:
        return None
    return pre[0][:cut[0]], [p[i:] for p, i in zip(pre, cut)]


def _prefix(model, vocab: Vocab, chunk, device, max_new):
    pre = [[vocab.stoi['<bos>']] + vocab.encode(t) + [vocab.stoi['<c>']] for t in chunk]
    ents = [list(dict.fromkeys(t for t in toks if t[0] == '@' and t in vocab.stoi)) for toks in chunk]
    shared = _shared_question(model, vocab, pre) if getattr(model, 'arch', None) == 'v10' else None
    if shared and max(len(p) for p in pre) + max_new <= model.block_size:
        q, fs = shared
        flen = torch.tensor([len(x) for x in fs], device=device)
        f = torch.full((len(fs), int(flen.max())), vocab.stoi['<pad>'], dtype=torch.long)
        for i, x in enumerate(fs):
            f[i, :len(x)] = torch.tensor(x)
        cache, logp = model.prefill_focus(torch.tensor([q], device=device), f.to(device), flen)
        return cache, logp, len(q) + flen, len(q) + f.size(1), ents
    pre = [p[-(model.block_size - max_new):] for p in pre]
    plen = torch.tensor([len(p) for p in pre], device=device)
    idx = torch.full((len(pre), int(plen.max())), vocab.stoi['<pad>'], dtype=torch.long)
    for i, p in enumerate(pre):
        idx[i, :len(p)] = torch.tensor(p)
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


_IDS = {}   # (vocabulary, question's people, grammar state) -> allowed token ids; readings and questions repeat states


def _state_ids(vocab: Vocab, ents: tuple, cur: tuple, n_new: int):
    key = (id(vocab), ents, cur, n_new)
    got = _IDS.get(key)
    if got is None:
        toks = allowed_at(cur, n_new, ents)
        got = torch.tensor([vocab.stoi['<sep>' if a == ';' else a] for a in toks if ('<sep>' if a == ';' else a) in vocab.stoi])
        if len(_IDS) >= 200_000:
            _IDS.clear()
        _IDS[key] = got
    return got


def _grow(cur: tuple, n_new: int, t: str):
    """Grammar state after writing t."""
    if t == ';':
        return (), n_new
    return cur + (t,), max(n_new, int(t[1:])) if t[0] == '#' else n_new


@torch.no_grad()
def generate(model, vocab: Vocab, inputs: list[list[str]], device, k: int = 1, sample: bool = False,
             temperature: float = 1.0, max_new: int = 96, bs: int = 32, check=None,
             reserve: int = 0) -> list[list[tuple[str, float]]]:
    """Fact readings for each question, only ever writing valid statements.
    Beam search (k best, by mean log-probability per token) or k samples (for reinforcement learning).
    check(facts) -> bool: beam readings it rejects (no family fits) do not count toward the k, so the search goes on
    to consistent ones (execution-guided decoding, Wang et al. 2018); they are used only if nothing passes.
    Returns, per question, [(facts, score)] best first (samples in drawing order).
    Each step scores every reading at once on the device (grammar mask, renormalised log-probs, top-k per reading,
    then per question); only the kept candidates come back to Python."""
    model.eval()
    sep, eos = vocab.stoi['<sep>'], vocab.stoi['<eos>']
    out = []
    for b in range(0, len(inputs), bs):
        chunk = inputs[b:b + bs]
        budget = _budget(model, vocab, chunk, max_new)
        if any(FOCUS in t for t in chunk) and budget < min(max_new if max_new > 0 else FOCUS_ROOM, FOCUS_ROOM):
            out += [[('', float('-inf'))] for _ in chunk]   # a sentence read with no room to write: none, never a cut question
            continue
        cache, logits, plen, P, ents = _prefix(model, vocab, chunk, device, budget)
        ents = [tuple(e) for e in ents]
        width = k
        rows = torch.arange(len(chunk), device=device)
        if sample:   # k independent samples per question
            rows = rows.repeat_interleave(k)
        hyps = [(int(r), [], 0.0, 0, (), 0) for r in rows.tolist()]   # (question, written, log-prob, tokens, statement, #max)
        cache = [[t[rows] for t in kv] for kv in cache]
        logits, plen_h = logits[rows], plen[rows]
        done = [[] for _ in chunk]
        spare = [[] for _ in chunk]   # finished readings `check` rejected
        for step in range(budget):
            left = budget - step
            ids = [_allowed_ids(vocab, hy[1], list(ents[hy[0]]), left, reserve) if reserve and left <= reserve
                   else _state_ids(vocab, ents[hy[0]], hy[4], hy[5]) for hy in hyps]
            ids = [torch.as_tensor(x, dtype=torch.long) for x in ids]
            lens = torch.tensor([len(x) for x in ids])
            mask = torch.zeros(logits.shape, dtype=torch.bool, device=device)
            mask[torch.repeat_interleave(torch.arange(len(hyps)), lens).to(device), torch.cat(ids).to(device)] = True
            lp = torch.log_softmax(logits.float() / (temperature if sample else 1.0), -1).masked_fill(~mask, float('-inf'))
            lp = lp - torch.logsumexp(lp, -1, keepdim=True)   # renormalised over what the grammar allows
            base = torch.tensor([hy[2] for hy in hyps], device=device)
            if sample:
                t = torch.multinomial(lp.exp(), 1)[:, 0]
                keep = list(zip([hy[0] for hy in hyps], range(len(hyps)), t.tolist(), (base + lp.gather(1, t[:, None])[:, 0]).tolist()))
            else:   # top `width` per reading, then per question over its readings' candidates
                w = min(width, lp.size(1))
                tv, ti = (lp + base[:, None]).topk(w, -1)
                qs = [hy[0] for hy in hyps]
                order = list(dict.fromkeys(qs))
                qpos, seen, sh = {qi: n for n, qi in enumerate(order)}, Counter(), []
                for qi in qs:   # each reading's slot among its question's readings
                    sh.append(seen[qi])
                    seen[qi] += 1
                m = max(sh) + 1
                dv = torch.full((len(order), m, w), float('-inf'), device=device)
                di = torch.zeros((len(order), m, w), dtype=torch.long, device=device)
                at = (torch.tensor([qpos[qi] for qi in qs], device=device), torch.tensor(sh, device=device))
                dv[at], di[at] = tv, ti
                hmap = [[0] * m for _ in order]
                for h, (qi, s_) in enumerate(zip(qs, sh)):
                    hmap[qpos[qi]][s_] = h
                vals, flat = dv.view(len(order), -1).topk(min(width, m * w), -1)
                tok = di.view(len(order), -1).gather(1, flat)
                keep = [(order[q], hmap[q][f // w], t, v)
                        for q, (fr, tr, vr) in enumerate(zip(flat.tolist(), tok.tolist(), vals.tolist()))
                        for f, t, v in zip(fr, tr, vr) if v != float('-inf')]
            nxt_hyps, parents, toks = [], [], []
            for qi, h, t, sc in keep:
                written, n = hyps[h][1], hyps[h][3] + 1
                if t == eos or step == budget - 1:
                    txt = _text(written)
                    (done if sample or check is None or check(txt) else spare)[qi].append((txt, sc / n))
                    continue
                word = ';' if t == sep else vocab.itos[t]
                nxt_hyps.append((qi, written + [word], sc, n, *_grow(hyps[h][4], hyps[h][5], word)))
                parents.append(h)
                toks.append(t)
            if not sample:   # a question stops once it has k finished readings
                full = {qi for qi in range(len(chunk)) if len(done[qi]) >= width}
                sel = [i for i, hy in enumerate(nxt_hyps) if hy[0] not in full]
                nxt_hyps, parents, toks = [nxt_hyps[i] for i in sel], [parents[i] for i in sel], [toks[i] for i in sel]
            if not nxt_hyps:
                break
            pidx = torch.tensor(parents, device=device)
            cache = [[t[pidx] for t in kv] for kv in cache]
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


PONDER_PRIOR, PONDER_BETA = 0.5, 0.01   # geometric prior on the stopping loop and its weight (PonderNet, Banino et al. 2021)


def fact_losses(model, idx, tgt, lm, vocab, pairs=None):
    """Mean cross-entropy of the facts (copy head included for v10) and of the question's words; with `pairs` (row
    indices of twins, v10) also their consistency loss, returned third."""
    cons = torch.zeros((), device=idx.device)
    if getattr(model, 'arch', None) == 'v10' and getattr(model, 'loops_now', 1) > 1:   # looped reader: PonderNet loss
        mt = tgt != -100
        lps, lams, h = model.fact_logprobs_loops(idx, mt, vocab.stoi['<c>'])
        y = tgt[mt][:, None]
        nll = torch.stack([-lp.gather(1, y)[:, 0] for lp in lps], 1)              # (N, R): loss if stopping after loop r
        lam = torch.stack(lams, 1).clamp(1e-4, 1 - 1e-4)
        lam = torch.cat([lam[:, :-1], torch.ones_like(lam[:, -1:])], 1)          # the last loop always stops
        keep = torch.cumprod(torch.cat([torch.ones_like(lam[:, :1]), 1 - lam[:, :-1]], 1), 1)
        p = keep * lam                                                            # p_r: stop exactly after loop r
        prior = torch.tensor([PONDER_PRIOR * (1 - PONDER_PRIOR) ** r for r in range(nll.size(1))], device=idx.device)
        prior = prior / prior.sum()
        kl = (p * ((p + 1e-9).log() - prior.log())).sum(1)
        loss_f = (p * nll).sum(1).mean() + PONDER_BETA * kl.mean()
    elif getattr(model, 'arch', None) == 'v10':
        mt = tgt != -100
        lp, h = model.fact_logprobs(idx, mt, vocab.stoi['<c>'])
        loss_f = F.nll_loss(lp, tgt[mt])
        if pairs:   # twins (same facts, another surface): their next-token distributions should match (symmetric KL)
            rows = mt.nonzero()[:, 0]
            ia = torch.cat([(rows == i).nonzero()[:, 0] for i, _ in pairs])
            ib = torch.cat([(rows == j).nonzero()[:, 0] for _, j in pairs])
            pa, pb = lp[ia], lp[ib]
            cons = 0.5 * ((pa.exp() * (pa - pb)).sum(-1) + (pb.exp() * (pb - pa)).sum(-1)).mean()
    else:
        h = hidden_states(model, idx)
        mt = tgt != -100
        loss_f = F.cross_entropy(model.head(h[mt]).float(), tgt[mt])
    ml = lm != -100
    if getattr(model, 'prefix_lm', False):   # the question attends both ways: predicting its next word would be copying it
        ml = torch.zeros_like(ml)
    loss_lm = F.cross_entropy(model.head(h[ml]).float(), lm[ml]) if ml.any() else torch.zeros((), device=idx.device)
    return (loss_f, loss_lm, cons) if pairs is not None else (loss_f, loss_lm)


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


@torch.no_grad()
def halt_prob(model, vocab: Vocab, tokens, lf: str, r: int, device) -> float:
    """The looped reader's own estimate that `lf` is finished after r loops: P(stopped by loop r), averaged over its tokens."""
    enc = encode_example(vocab, tokens, lf, model.block_size)
    if enc is None or getattr(model, 'exit_gate', None) is None:
        return 1.0
    idx, tgt, _ = collate([enc], vocab.stoi['<pad>'], device)
    _, lams = model.hidden_loops(idx)
    m = tgt != -100
    lam = torch.stack([x[m] for x in lams[:r]], 1)
    return float((1 - torch.prod(1 - lam, 1)).mean())


def answer_looped(text: str, model, vocab: Vocab, device, k: int = 8, tau_lambda: float = 0.5, tau_vote: float = 0.6):
    """Halting with two judges: after r loops the answer stands when the engine accepts it (a family fits, the vote is
    clear) and the exit gate agrees the reading is finished (lambda); otherwise one more loop, up to the model's maximum."""
    from kinlogic.exam import answer
    from kinlogic.prepare import prepare
    toks = prepare(text)['tokens']
    R, res = getattr(model, 'n_loop', 1), {}
    try:
        for r in range(1, R + 1):
            model.set_loops(r) if R > 1 else None
            res = answer(text, lambda b: generate(model, vocab, b, device, k=k))
            if r == R:
                break
            engine_ok = res.get('status') not in ('error', 'none', 'contradiction') and res.get('confidence', 1.0) >= tau_vote
            if engine_ok and res.get('facts') and halt_prob(model, vocab, toks, res['facts'], r, device) >= tau_lambda:
                break
        return {**res, 'loops': r}
    finally:
        if R > 1:
            model.set_loops(R)

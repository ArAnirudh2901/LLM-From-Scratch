from __future__ import annotations
import math, random

import torch
import torch.nn.functional as F

from qa_data import (EDGE_RELS, ENTS, N_ENT, NOFACT, RELATIONS, UNKNOWN, Vocab, anonymize, build_sequence, tokenize)

# Reference for the browser engine: the same constrained beam search as qa_data.beam_predict, computed without
# repeated work. Every token's keys/values are computed once and kept in a slot pool; a token attends to a list of
# slots (its story prefix, its hypothesis' earlier chain tokens, itself), so child hypotheses share their parent's
# tokens instead of recomputing the chain each round. Extra views (other person-id numberings) run only for
# questions the first view leaves unclear (adaptive views).

PERMS = [list(range(N_ENT))] + [random.Random(1000 + v).sample(range(N_ENT), N_ENT) for v in range(1, 8)]


def view_tokens(vocab: Vocab, ex: dict, view: int) -> list[int]:
    """<bos> <q> @a @b <s> story <c> with person ids renumbered by PERMS[view] (view 0: first-appearance order)."""
    story, m = anonymize(ex)
    p = PERMS[view]
    ren = {f'@{i}': f'@{p[i]}' for i in range(N_ENT)}
    toks = ['<bos>', '<q>', ren[m[ex['a']]], ren[m[ex['b']]], '<s>'] + [ren.get(t, t) for t in tokenize(story)] + ['<c>']
    return vocab.encode(toks)


class SlotEngine:
    """Transformer forward over rows that each attend to an explicit list of KV slots."""

    def __init__(self, model, cap: int = 16384):
        self.m, self.cap = model, cap
        L, C = len(model.blocks), model.tok_emb.weight.size(1)
        dev = model.tok_emb.weight.device
        self.K = torch.zeros(L, cap, C, device=dev)
        self.V = torch.zeros(L, cap, C, device=dev)
        self.n = 0

    def reset(self):
        self.n = 0

    @torch.no_grad()
    def forward(self, toks: list[int], pos: list[int], ctx: list[list[int]]) -> tuple[torch.Tensor, list[int]]:
        """Hidden states (after ln_f) of new rows; ctx[r] lists the slots row r attends to, which may name the
        rows' own new slots (returned) by index -1 - j for new row j."""
        m, R = self.m, len(toks)
        if self.n + R > self.cap:
            raise RuntimeError('slot pool full')
        slots = list(range(self.n, self.n + R))
        self.n += R
        dev = self.K.device
        M = max(map(len, ctx))
        idx = torch.zeros(R, M, dtype=torch.long)
        mask = torch.zeros(R, M, dtype=torch.bool)
        for r, c in enumerate(ctx):
            idx[r, :len(c)] = torch.tensor([s if s >= 0 else slots[-1 - s] for s in c])
            mask[r, :len(c)] = True
        idx, mask = idx.to(dev), mask.to(dev)
        H = m.blocks[0].attn.n_head
        C = self.K.size(2); D = C // H
        x = m.tok_emb(torch.tensor(toks, device=dev)) + m.pos_emb(torch.tensor(pos, device=dev))
        sl = torch.tensor(slots, device=dev)
        for l, blk in enumerate(m.blocks):
            q, k, v = blk.attn.qkv(blk.ln1(x)).split(C, -1)
            self.K[l, sl], self.V[l, sl] = k, v
            kc, vc = self.K[l][idx].view(R, M, H, D), self.V[l][idx].view(R, M, H, D)
            att = torch.einsum('rhd,rmhd->rhm', q.view(R, H, D), kc) / math.sqrt(D)
            att = att.masked_fill(~mask[:, None, :], -math.inf).softmax(-1)
            y = torch.einsum('rhm,rmhd->rhd', att, vc).reshape(R, C)
            x = x + blk.attn.proj(y)
            x = x + blk.ffn(blk.ln2(x))
        return m.ln_f(x), slots


@torch.no_grad()
def search(eng: SlotEngine, vocab: Vocab, seqs: list[list[int]], beam: int = 8, branch: int = 3,
           prune: float = 10.0, block_size: int = 320):
    """Constrained beam search (as qa_data._beam) for prefix token lists `seqs`. Returns per sequence:
    votes over RELATIONS (tensor), best chain (str), p_start, p_link, and the number of model rows computed."""
    m = eng.m
    sep, a_tok = vocab.stoi['<sep>'], vocab.stoi['<a>']
    nofact = vocab.stoi.get(NOFACT)
    rel_ids = torch.tensor(vocab.encode(RELATIONS))
    edge_ids = torch.tensor(vocab.encode(EDGE_RELS))
    ent_ids = torch.tensor(vocab.encode(ENTS))
    ent_base = ent_ids[0].item()
    n = len(seqs)
    eng.reset()
    # story prefixes: each token attends to its own sequence's earlier tokens
    toks, pos, ctx = [], [], []
    for s, p in enumerate(seqs):
        base = len(toks)
        for j, t in enumerate(p):
            toks.append(t); pos.append(j); ctx.append([-1 - (base + i) for i in range(j + 1)])
    h, slots = eng.forward(toks, pos, ctx)
    rows_done = len(toks)
    pre_slots, r0 = [], 0
    for p in seqs:
        pre_slots.append(slots[r0:r0 + len(p)]); r0 += len(p)
    last_rows = torch.tensor([sum(map(len, seqs[:s + 1])) - 1 for s in range(n)])
    p_start = torch.zeros(n)
    if UNKNOWN in vocab.stoi:
        lg = m.head(h[last_rows]).float()
        two = torch.stack([lg[torch.arange(n), torch.tensor([p[2] for p in seqs])], lg[:, a_tok]], -1)
        p_start = torch.softmax(two, -1)[:, 1].cpu()
    target = [p[3] for p in seqs]
    people = [sum(1 << (t - ent_base) for t in set(p[5:]) if ent_base <= t < ent_base + N_ENT) for p in seqs]
    votes = torch.zeros(n, len(RELATIONS))
    mass, valid_mass = [0.0] * n, [0.0] * n
    best_chain = [''] * n
    best_done = [-math.inf] * n
    # hypothesis: [seq, chain tokens, computed slots, visited bitmask, chain log-prob, phase, log-validity]
    hyps = [(s, [p[2]], [], 1 << (p[2] - ent_base), 0.0, 0, 0.0) for s, p in enumerate(seqs)]
    while hyps:
        toks, pos, ctx, where = [], [], [], []
        for hi, (s, ch, sl, *_rest) in enumerate(hyps):
            first = len(toks)
            for j in range(len(sl), len(ch)):
                toks.append(ch[j]); pos.append(len(seqs[s]) + j)
                ctx.append(pre_slots[s] + sl + [-1 - (first + i) for i in range(j - len(sl) + 1)])
            where.append((first, len(toks)))
        h, slots = eng.forward(toks, pos, ctx)
        rows_done += len(toks)
        want, new_hyps_slots = [], []
        for (first, end), (s, ch, sl, *_r) in zip(where, hyps):
            new_hyps_slots.append(sl + slots[first:end])
            want += [end - 1, max(first, end - 3)]
        lg = m.head(h[torch.tensor(want)]).float().view(len(hyps), 2, -1)
        lp = torch.log_softmax(lg[:, 0], -1)
        step_ok = (torch.log_softmax(lg[:, 1, [sep, nofact]], -1)[:, 0] if nofact is not None
                   else torch.zeros(len(hyps))).tolist()
        r_lp, r_ix = lp[:, edge_ids].topk(branch, -1)
        free = (torch.tensor([people[h_[0]] & ~h_[3] for h_ in hyps])[:, None] >> torch.arange(N_ENT) & 1).bool()
        e_lp, e_ix = lp[:, ent_ids].masked_fill(~free, -math.inf).topk(branch, -1)
        ans_p = torch.softmax(lp[:, rel_ids], -1)
        r_lp, r_ix, e_lp, e_ix = r_lp.tolist(), edge_ids[r_ix].tolist(), e_lp.tolist(), (e_ix + ent_base).tolist()
        new = []
        for j, ((s, ch, _, vis, score, ph, lv), sl) in enumerate(zip(hyps, new_hyps_slots)):
            if ph != 1 and len(ch) > 2:
                lv += step_ok[j]
            if ph == 2:
                votes[s] += math.exp(score + lv) * ans_p[j]
                mass[s] += math.exp(score); valid_mass[s] += math.exp(score + lv)
                if score + lv > best_done[s]:
                    best_done[s] = score + lv
                    best_chain[s] = ' '.join(vocab.itos[t] for t in ch[:-1])
            elif ph == 0:
                new += [(s, ch + [t], sl, vis, score + x, 1, lv) for t, x in zip(r_ix[j], r_lp[j])]
            else:
                for t, x in zip(e_ix[j], e_lp[j]):
                    if x == -math.inf:
                        continue
                    if t == target[s]:
                        new.append((s, ch + [t, sep, a_tok], sl, vis, score + x, 2, lv))
                    elif len(seqs[s]) + len(ch) < block_size - 8:
                        new.append((s, ch + [t, sep, t], sl, vis | 1 << (t - ent_base), score + x, 0, lv))
        new.sort(key=lambda h_: (h_[0], -h_[4] - h_[6]))
        hyps, count = [], {}
        for h_ in new:
            if count.get(h_[0], 0) < beam and h_[4] + h_[6] > best_done[h_[0]] - prune:
                hyps.append(h_); count[h_[0]] = count.get(h_[0], 0) + 1
    p_link = torch.tensor([v / m_ if m_ > 0 else 0.0 for v, m_ in zip(valid_mass, mass)])
    return votes, best_chain, p_start, p_link, rows_done


def predict(eng: SlotEngine, vocab: Vocab, exs: list[dict], views: int = 4, adaptive: tuple | None = None,
            unk_threshold: float = 0.5, **kw):
    """Like qa_data.beam_predict. adaptive=(min_share, max_unknown_margin): after view 0, run the other views only
    for questions whose single-view agreement is below min_share or whose p_unknown is within the margin of the
    threshold. Returns preds, chains, info (share, p_unknown, p_start, p_link, views used, model rows computed)."""
    n = len(exs)
    votes, p_start, p_link = torch.zeros(n, len(RELATIONS)), torch.zeros(n), torch.zeros(n)
    used, rows, chains = [0] * n, 0, [''] * n
    todo = list(range(n))
    for v in range(views):
        if not todo:
            break
        vt, ch, ps, pl, r = search(eng, vocab, [view_tokens(vocab, exs[i], v) for i in todo], **kw)
        rows += r
        for j, i in enumerate(todo):
            votes[i] += vt[j] / vt[j].sum().clamp_min(1e-30)
            p_start[i] += ps[j]; p_link[i] += pl[j]; used[i] += 1
            if v == 0:
                chains[i] = ch[j]
        if v == 0 and adaptive:
            min_share, margin = adaptive
            def clear(i):
                share = votes[i].max().item()
                pu = max(p_start[i].item(), 1 - p_link[i].item())
                return share >= min_share and abs(pu - unk_threshold) >= margin
            todo = [i for i in todo if not clear(i)]
    k = torch.tensor(used, dtype=torch.float).clamp_min(1)
    p_start, p_link = p_start / k, p_link / k
    share = votes.max(-1).values / k
    p_unknown = torch.maximum(p_start, 1 - p_link)
    preds = [UNKNOWN if p_unknown[i] >= unk_threshold or votes[i].sum() == 0 else RELATIONS[r]
             for i, r in enumerate(votes.argmax(-1).tolist())]
    return preds, chains, {'share': share, 'p_unknown': p_unknown, 'p_start': p_start, 'p_link': p_link,
                           'views': used, 'rows': rows}

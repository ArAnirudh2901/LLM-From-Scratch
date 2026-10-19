from __future__ import annotations
import ast, glob, math, random, re
from collections import Counter
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F

# CLUTRR kinship QA: story + query (A, B) -> how B is related to A.
# Names become entity tokens (@0, @1, ...) so the model reads gender from text, not from memorized names.
# Chain-of-thought: the model first writes the facts on the path from A to B ("@a r1 @x <sep> @x r2 @b <sep>"), then the answer.

RELATIONS = ['aunt', 'brother', 'daughter', 'daughter-in-law', 'father', 'father-in-law', 'granddaughter',
             'grandfather', 'grandmother', 'grandson', 'mother', 'mother-in-law', 'nephew', 'niece',
             'sister', 'son', 'son-in-law', 'uncle']
UNKNOWN = 'unknown'   # answer when the story gives no family link between A and B
NOFACT = '<nofact>'   # ends a chain step the story does not state (instead of <sep>)
SPECIALS = ['<pad>', '<unk>', '<bos>', '<q>', '<s>', '<c>', '<sep>', '<a>', '<eos>']
N_ENT = 32
ENTS = [f'@{i}' for i in range(N_ENT)]
EVAL_CFG = 'gen_train234_test2to10'
TOKEN_RE = re.compile(r"@\d+|'s|\w+(?:-\w+)*|[^\w\s]")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def parse_row(r) -> dict:
    names = [x.split(':')[0] for x in r.genders.split(',')]
    a, b = ast.literal_eval(r.query)
    edges = [(names[i], t, names[j]) for (i, j), t in zip(ast.literal_eval(r.story_edges), ast.literal_eval(r.edge_types))
             if i < len(names) and j < len(names)]
    rules = {}                                   # head triple -> (premise1, premise2)
    for step in ast.literal_eval(r.proof_state):
        for head, body in step.items():
            rules[head] = tuple(body)
    k = int(re.search(r'\.(\d+)$', r.task_name).group(1))
    genders = dict(x.split(':') for x in r.genders.split(','))
    links = [(names[i], names[j]) for i, j in ast.literal_eval(r.story_edges) if i < len(names) and j < len(names)]
    ex = {'story': r.story, 'a': a, 'b': b, 'rel': r.target_text, 'edges': edges, 'k': k, 'genders': genders,
          'links': links}   # links: all story facts, incl. untyped noise facts
    sub = set(edges) | set(rules) | {p for body in rules.values() for p in body}
    ex['sub'] = sorted(t for t in sub if path_facts(ex, t[0], t[2]) is not None)
    return ex


def story_key(ex: dict) -> str:
    """Story text with names replaced in first-appearance order: equal keys = same story up to names."""
    return ' '.join(tokenize(anonymize(ex)[0]))


def load_splits(data_dir: str = 'data/clutrr_v1', dev_size: int = 1000, seed: int = 0, other_tests: bool = False):
    """train: all other train/validation splits (+ other configs' test splits if other_tests);
    dev/val: gen_train234 validation; test: gen_train234 test.
    Training stories equal to a held-out story up to names are dropped. Held-out examples get
    'seen' = True when such a twin existed in the raw training data (excluded from clean accuracy)."""
    files = {Path(f).stem: f for f in glob.glob(f'{data_dir}/*.parquet')}
    val = [parse_row(r) for r in pd.read_parquet(files[f'{EVAL_CFG}__validation']).itertuples()]
    test = [parse_row(r) for r in pd.read_parquet(files[f'{EVAL_CFG}__test']).itertuples()]
    held_out = {story_key(e) for e in val + test}

    raw = []
    for stem, f in sorted(files.items()):
        cfg, split = stem.split('__')
        if cfg == EVAL_CFG and split in ('validation', 'test'):
            continue
        if split == 'test' and not other_tests:
            continue
        raw += [parse_row(r) for r in pd.read_parquet(f).itertuples()]
    raw_keys = {story_key(e) for e in raw}
    train = [e for e in raw if story_key(e) not in held_out]
    for e in val + test:
        e['seen'] = story_key(e) in raw_keys

    random.Random(seed).shuffle(val)
    return train, val[:dev_size], val[dev_size:], test


def anonymize(ex: dict, rng: random.Random | None = None) -> tuple[str, dict]:
    """Replace each distinct name with an entity token. rng -> random ids (augmentation), else first-appearance order."""
    extra = [n for t in ex.get('sub', []) for n in (t[0], t[2])]
    names = list(dict.fromkeys(re.findall(r'\[([^\]]+)\]', ex['story']) + [ex['a'], ex['b']] + extra))
    ids = rng.sample(range(N_ENT), len(names)) if rng else range(len(names))
    m = {n: f'@{i}' for n, i in zip(names, ids)}
    story = re.sub(r'\[([^\]]+)\]', lambda g: f' {m[g.group(1)]} ', ex['story'])
    return story, m


def path_facts(ex: dict, x: str, z: str):
    """Story facts on the path x -> z, following edges forward (y is x's r). None if no path. Cached per example."""
    cache = ex.setdefault('_paths', {})
    if (x, z) not in cache:
        cache[(x, z)] = _bfs_path(ex, x, z)
    return cache[(x, z)]


def _bfs_path(ex: dict, x: str, z: str):
    adj = {}
    for a, r, b in ex.get('edges', []):
        adj.setdefault(a, []).append((r, b))
    frontier, seen = [(x, [])], {x}
    while frontier:
        nxt = []
        for n, path in frontier:
            if n == z:
                return path
            for r, m in adj.get(n, []):
                if m not in seen:
                    seen.add(m); nxt.append((m, path + [(n, r, m)]))
        frontier = nxt
    return None


class Vocab:
    def __init__(self, itos: list[str]):
        self.itos = itos
        self.stoi = {t: i for i, t in enumerate(itos)}

    @classmethod
    def build(cls, train: list[dict], min_freq: int = 2) -> 'Vocab':
        c = Counter()
        for ex in train:
            c.update(tokenize(anonymize(ex)[0]))
        words = sorted(w for w, n in c.items() if n >= min_freq and not w.startswith('@'))
        rels = sorted(set(RELATIONS) | {t[1] for ex in train for t in ex['sub']})
        return cls(SPECIALS + ENTS + rels + [w for w in words if w not in rels])

    def encode(self, toks: list[str]) -> list[int]:
        unk = self.stoi['<unk>']
        return [self.stoi.get(t, unk) for t in toks]

    def __len__(self):
        return len(self.itos)


def build_sequence(vocab: Vocab, ex: dict, rng: random.Random | None = None, query=None, cot: bool = False):
    """<bos> <q> @a @b <s> story... [<c> steps...] <a> REL <eos>.
    A step the story does not state ends with <nofact> instead of <sep>, and the sequence stops there
    (ex['nofact']: chain given in ex['chain']).
    Returns ids, index where generation starts (after <s> story, i.e. the <c> or <a> token), index of <a>
    (for <nofact> chains: index of the step's last person, whose next token is <nofact>)."""
    story, m = anonymize(ex, rng)
    a, rel, b = query or (ex['a'], ex['rel'], ex['b'])
    toks = ['<bos>', '<q>', m[a], m[b], '<s>'] + tokenize(story)
    gen_start = len(toks)
    if cot:
        toks.append('<c>')                    # no link: empty chain, "<c> <a> unknown"
        steps = ex['chain'] if ex.get('nofact') else [] if rel == UNKNOWN else path_facts(ex, a, b) or []
        for x, r, y in steps:
            toks += [m[x], r, m[y], '<sep>']
        if ex.get('nofact'):
            toks[-1] = NOFACT
            return vocab.encode(toks + ['<eos>']), gen_start, len(toks) - 2
    ans_pos = len(toks)
    toks += ['<a>', rel, '<eos>']
    return vocab.encode(toks), gen_start, ans_pos


def make_batch(vocab: Vocab, exs: list[dict], device, rng: random.Random | None = None,
               sub_prob: float = 0.0, cot: bool = False):
    """Padded training batch. tgt: next-token targets for chain + answer; lm_tgt: next-token targets on the story."""
    seqs, starts, poss = [], [], []
    for ex in exs:
        query = rng.choice(ex['sub']) if rng and ex['sub'] and rng.random() < sub_prob else None
        ids, s, p = build_sequence(vocab, ex, rng, query, cot)
        seqs.append(ids); starts.append(s); poss.append(p)
    idx = pad_batch(seqs, vocab.stoi['<pad>'], 'cpu', mult=16)
    nxt = torch.full_like(idx, -100)
    nxt[:, :-1] = idx[:, 1:]
    col = torch.arange(idx.size(1))
    s, p = torch.tensor(starts)[:, None], torch.tensor(poss)[:, None]
    tgt = torch.where((col >= s - 1) & (col <= p), nxt, -100)      # chain tokens, <a>, and the answer
    lm_tgt = torch.where((col >= 4) & (col < s - 1), nxt, -100)    # story tokens
    ans = idx[torch.arange(len(seqs)), p[:, 0] + 1]
    ans[torch.tensor([bool(ex.get('nofact')) for ex in exs])] = -100   # <nofact> already in tgt: no double weight
    return idx.to(device), tgt.to(device), lm_tgt.to(device), ans.to(device), p[:, 0].to(device)


def pad_batch(seqs: list[list[int]], pad: int, device, mult: int = 16) -> torch.Tensor:
    T = -(-max(map(len, seqs)) // mult) * mult
    return torch.tensor([s + [pad] * (T - len(s)) for s in seqs], device=device)


def hidden_states(model, idx: torch.Tensor) -> torch.Tensor:
    """GPT body without the vocab head: (B, T, n_embd) after the final LayerNorm."""
    x = model.drop(model.tok_emb(idx) + model.pos_emb(torch.arange(idx.size(1), device=idx.device)))
    for blk in model.blocks:
        x = blk(x)
    return model.ln_f(x)


def logits_at(model, idx: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
    """Vocab logits only at one position per row: the head (n_embd x vocab) skips every other position."""
    return model.head(hidden_states(model, idx)[torch.arange(idx.size(0), device=idx.device), pos])


def _heads(attn, x):
    B, T, C = x.shape
    q, k, v = attn.qkv(x).view(B, T, 3, attn.n_head, attn.d_head).unbind(dim=2)
    return q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)


def encode_prefix(model, idx: torch.Tensor, plen: torch.Tensor) -> tuple[list, torch.Tensor]:
    """Run the shared story prefix once; keep each layer's keys/values. Right padding is harmless (causal).
    Also returns the logits at each row's last real token (plen - 1)."""
    x = model.tok_emb(idx) + model.pos_emb(torch.arange(idx.size(1), device=idx.device))
    cache = []
    for blk in model.blocks:
        q, k, v = _heads(blk.attn, blk.ln1(x))
        cache.append((k, v))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + blk.attn.proj(y.transpose(1, 2).reshape(x.shape))
        x = x + blk.ffn(blk.ln2(x))
    return cache, model.head(model.ln_f(x[torch.arange(idx.size(0), device=idx.device), plen - 1]))


def suffix_logits(model, cache: list, rows: torch.Tensor, plen: torch.Tensor, suf: torch.Tensor, last: torch.Tensor):
    """Logits at positions `last` (N, or N x j) of each suffix, attending to its example's cached prefix
    (rows, true lengths plen). Cost per call: suffix x (prefix + suffix) instead of a full (prefix + suffix)^2 pass."""
    S, P = suf.size(1), cache[0][0].size(2)
    ar = torch.arange(S, device=suf.device)
    x = model.tok_emb(suf) + model.pos_emb((plen[:, None] + ar).clamp(max=model.block_size - 1))   # padding past the window: never attended
    pre_ok = torch.arange(P, device=suf.device)[None, :] < plen[:, None]                 # (N, P)
    mask = torch.cat([pre_ok[:, None, :].expand(-1, S, -1), (ar[None, :] <= ar[:, None]).expand(len(suf), -1, -1)], -1)
    mask = mask[:, None]                                                                  # (N, 1, S, P+S)
    for (kp, vp), blk in zip(cache, model.blocks):
        q, k, v = _heads(blk.attn, blk.ln1(x))
        k, v = torch.cat([kp[rows], k], 2), torch.cat([vp[rows], v], 2)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        x = x + blk.attn.proj(y.transpose(1, 2).reshape(x.shape))
        x = x + blk.ffn(blk.ln2(x))
    ar = torch.arange(len(suf), device=suf.device)
    return model.head(model.ln_f(x[ar if last.dim() == 1 else ar[:, None], last]))


def _prefixes(vocab: Vocab, exs: list[dict], cot: bool, rng: random.Random | None = None) -> list[list[int]]:
    out = []
    for ex in exs:
        ids, s, _ = build_sequence(vocab, ex, rng)
        out.append(ids[:s] + [vocab.stoi['<c>' if cot else '<a>']])
    return out


@torch.no_grad()
def predict(model, vocab: Vocab, exs: list[dict], device, cot: bool = False, block_size: int = 320):
    """Greedy decoding. With cot, the model writes the chain until <a>; the answer is then restricted
    to the 18 relations (+ 'unknown' if the vocab has it). Finished rows leave the batch. Returns predicted relations and chains."""
    rel_ids = torch.tensor(vocab.encode(RELATIONS + [UNKNOWN] * (UNKNOWN in vocab.stoi)), device=device)
    a_id, pad = vocab.stoi['<a>'], vocab.stoi['<pad>']
    seqs = _prefixes(vocab, exs, cot)
    start = [len(s) for s in seqs]
    active = [i for i in range(len(seqs)) if cot]
    while active:
        idx = pad_batch([seqs[i] for i in active], pad, device)
        last = torch.tensor([len(seqs[i]) - 1 for i in active], device=device)
        for i, t in zip(active, logits_at(model, idx, last).argmax(-1).tolist()):
            seqs[i].append(t)
        active = [i for i in active if seqs[i][-1] != a_id and len(seqs[i]) < block_size - 1]
    for s in seqs:
        if s[-1] != a_id:   # chain ran too long: force the answer slot
            s.append(a_id)
    last = torch.tensor([len(s) - 1 for s in seqs], device=device)
    pred = rel_ids[logits_at(model, pad_batch(seqs, pad, device), last)[:, rel_ids].argmax(-1)].tolist()
    chains = [' '.join(vocab.itos[t] for t in s[st:-1]) for s, st in zip(seqs, start)]
    return [vocab.itos[t] for t in pred], chains


def prefix_fits(vocab: Vocab, ex: dict, block_size: int = 320, room: int = 16) -> bool:
    """Does the question + story leave at least `room` tokens for the chain within the model's window?"""
    return len(_prefixes(vocab, [ex], True)[0]) <= block_size - room


EDGE_RELS = ['aunt', 'brother', 'daughter', 'father', 'granddaughter', 'grandfather', 'grandmother',
             'grandson', 'husband', 'mother', 'sister', 'son', 'uncle', 'wife']


@torch.no_grad()
def beam_predict(model, vocab: Vocab, exs: list[dict], device, beam: int = 8, branch: int = 3,
                 block_size: int = 320, chunk: int = 256, views: int = 1, unk_threshold: float = 0.5):
    """Constrained beam search over the chain "@x r @y <sep>": x is the previous y (first is A), r is a
    family relation, y is a story person not yet visited, and the chain stops on reaching B.
    Each step is also checked: the model's P(<sep>) vs P(<nofact>) after it says whether the story states it.
    Every finished chain votes for its answer, weighted by its probability times its steps' validity.
    views > 1: repeat with random person ids (the model trained on random ids) and add the votes.
    Returns predictions, chains and a dict of per-example scores:
      share     agreement: share of all votes behind the answer (0-1)
      p_start   probability at <c> that A and B are not linked (writes "<a>" at once)
      p_link    probability-weighted validity of the finished chains (0 when no chain reaches B)
      p_unknown max(p_start, 1 - p_link); prediction is 'unknown' when it is >= unk_threshold."""
    votes, p_start, p_link, chains = torch.zeros(len(exs), len(RELATIONS)), torch.zeros(len(exs)), torch.zeros(len(exs)), None
    for v in range(views):
        vt, ch, ps, pl = _beam(model, vocab, exs, device, beam, branch, block_size, chunk, random.Random(v) if v else None)
        votes += vt / vt.sum(-1, keepdim=True).clamp_min(1e-30)
        p_start += ps / views; p_link += pl / views
        chains = chains or ch
    p_unknown = torch.maximum(p_start, 1 - p_link)
    preds = [UNKNOWN if p_unknown[i] >= unk_threshold or votes[i].sum() == 0 else RELATIONS[r]
             for i, r in enumerate(votes.argmax(-1).tolist())]
    return preds, chains, {'share': votes.max(-1).values / views, 'p_unknown': p_unknown,
                           'p_start': p_start, 'p_link': p_link}


def _beam(model, vocab, exs, device, beam, branch, block_size, chunk, rng):
    sep, a_tok, pad = vocab.stoi['<sep>'], vocab.stoi['<a>'], vocab.stoi['<pad>']
    nofact = vocab.stoi.get(NOFACT)
    rel_ids = torch.tensor(vocab.encode(RELATIONS), device=device)
    edge_ids = torch.tensor(vocab.encode(EDGE_RELS), device=device)
    ent_ids = torch.tensor(vocab.encode(ENTS), device=device)
    ent_base = ent_ids[0].item()                      # entity tokens are contiguous in the vocab
    bits = torch.arange(N_ENT, device=device)
    votes = torch.zeros(len(exs), len(RELATIONS))
    mass, valid_mass = [0.0] * len(exs), [0.0] * len(exs)
    best_chain = [''] * len(exs)
    pres = _prefixes(vocab, exs, True, rng)           # <bos> <q> @a @b <s> story <c>
    plen = torch.tensor([len(p) for p in pres], device=device)
    cache, first = encode_prefix(model, pad_batch(pres, pad, device), plen)
    p_start = torch.zeros(len(exs))
    if UNKNOWN in vocab.stoi:                         # at <c>: start the chain with A, or "<a>" at once (no link)
        start = torch.tensor([p[2] for p in pres], device=device)
        two = torch.stack([first[torch.arange(len(pres), device=device), start], first[:, a_tok]], -1)
        p_start = torch.softmax(two.float(), -1)[:, 1].cpu()
    target = [p[3] for p in pres]
    people = [sum(1 << (t - ent_base) for t in set(p[5:]) if ent_base <= t < ent_base + N_ENT) for p in pres]
    # hypothesis: (example, chain tokens, visited-people bitmask, chain log-prob, phase, steps' log-validity)
    # phase: 0 relation, 1 person, 2 answer. A step's validity is read one round later, at the person before <sep>.
    hyps = [(i, [p[2]], 1 << (p[2] - ent_base), 0.0, 0, 0.0) for i, p in enumerate(pres)]
    best_done = [-math.inf] * len(exs)
    while hyps:
        new = []
        for c in range(0, len(hyps), chunk):
            hs = hyps[c:c + chunk]
            rows = torch.tensor([h[0] for h in hs], device=device)
            last = torch.tensor([len(h[1]) - 1 for h in hs], device=device)
            both = suffix_logits(model, cache, rows, plen[rows], pad_batch([h[1] for h in hs], pad, device, 8),
                                 torch.stack([last, (last - 2).clamp_min(0)], 1)).float()
            lp = torch.log_softmax(both[:, 0], -1)
            step_ok = (torch.log_softmax(both[:, 1, [sep, nofact]], -1)[:, 0] if nofact is not None
                       else torch.zeros(len(hs), device=device)).tolist()
            r_lp, r_ix = lp[:, edge_ids].topk(branch, -1)
            free = (torch.tensor([people[h[0]] & ~h[2] for h in hs], device=device)[:, None] >> bits & 1).bool()
            e_lp, e_ix = lp[:, ent_ids].masked_fill(~free, -math.inf).topk(branch, -1)
            ans_p = torch.softmax(lp[:, rel_ids], -1).cpu()
            r_lp, r_ix, e_lp, e_ix = r_lp.tolist(), edge_ids[r_ix].tolist(), e_lp.tolist(), (e_ix + ent_base).tolist()
            for j, (i, toks, vis, score, ph, lv) in enumerate(hs):
                if ph != 1 and len(toks) > 2:     # a step just ended: add its validity
                    lv += step_ok[j]
                if ph == 2:
                    votes[i] += math.exp(score + lv) * ans_p[j]
                    mass[i] += math.exp(score); valid_mass[i] += math.exp(score + lv)
                    if score + lv > best_done[i]:
                        best_done[i] = score + lv
                        best_chain[i] = ' '.join(vocab.itos[t] for t in toks[:-1])
                elif ph == 0:
                    new += [(i, toks + [t], vis, score + s, 1, lv) for t, s in zip(r_ix[j], r_lp[j])]
                else:
                    for t, s in zip(e_ix[j], e_lp[j]):
                        if s == -math.inf:
                            continue
                        if t == target[i]:
                            new.append((i, toks + [t, sep, a_tok], vis, score + s, 2, lv))
                        elif len(pres[i]) + len(toks) < block_size - 8:
                            new.append((i, toks + [t, sep, t], vis | 1 << (t - ent_base), score + s, 0, lv))
        # keep the best `beam` per example; drop hypotheses far below a finished chain
        new.sort(key=lambda h: (h[0], -h[3] - h[5]))
        hyps, count = [], {}
        for h in new:
            if count.get(h[0], 0) < beam and h[3] + h[5] > best_done[h[0]] - 10:
                hyps.append(h); count[h[0]] = count.get(h[0], 0) + 1
    p_link = torch.tensor([v / m if m > 0 else 0.0 for v, m in zip(valid_mass, mass)])
    return votes, best_chain, p_start, p_link


MALE = {'son', 'father', 'brother', 'husband', 'grandfather', 'grandson', 'uncle', 'nephew'}
SENT_RE = re.compile(r'(?<=[.!?])\s+')


class StorySynth:
    """New stories from real crowd-written sentences. Each bank sentence states exactly one fact between two people;
    chains and answers come from the training data's chain -> answer table (ambiguous chains dropped)."""

    def __init__(self, train: list[dict]):
        self.bank = {}                                   # (relation, owner gender) -> sentence templates
        table = {}
        for ex in train:
            facts = {(x, y): r for x, r, y in ex['edges']}
            for sent in SENT_RE.split(ex['story']):
                names = list(dict.fromkeys(re.findall(r'\[([^\]]+)\]', sent)))
                if len(names) != 2:
                    continue
                x, y = names
                own, tgt = (x, y) if (x, y) in facts else (y, x)
                if (own, tgt) not in facts or own not in ex['genders']:
                    continue
                tpl = sent.replace(f'[{own}]', '[OWN]').replace(f'[{tgt}]', '[TGT]')
                self.bank.setdefault((facts[(own, tgt)], ex['genders'][own]), []).append(tpl)
            chain = tuple(r for _, r, _ in path_facts(ex, ex['a'], ex['b']) or [])
            table.setdefault(chain, Counter())[ex['rel']] += 1
        self.table = {c: t.most_common(1)[0][0] for c, t in table.items()
                      if c and t.most_common(1)[0][1] / sum(t.values()) >= 0.95}
        self.chains = {}                                  # hops -> [(chain, first person's gender)], real frequency
        for ex in train:
            c = tuple(r for _, r, _ in path_facts(ex, ex['a'], ex['b']) or [])
            if c in self.table and ex['a'] in ex['genders']:
                self.chains.setdefault(len(c), []).append((c, ex['genders'][ex['a']]))

    def sample(self, rng: random.Random) -> dict:
        while True:
            short = [k for k in self.chains if k <= 4]
            long = [k for k in self.chains if k > 4]
            k = rng.choice(long if long and rng.random() < 0.15 else short)    # hops: 85% 2-4 (uniform), 15% longer
            chain, g0 = rng.choice(self.chains[k])
            names = [f'P{i}' for i in range(len(chain) + 1)]
            genders = [g0] + ['male' if r in MALE else 'female' for r in chain]
            keys = [(r, genders[i]) for i, r in enumerate(chain)]
            if all(k in self.bank for k in keys):
                break
        sents = [rng.choice(self.bank[k]).replace('[OWN]', f'[{names[i]}]').replace('[TGT]', f'[{names[i + 1]}]')
                 for i, k in enumerate(keys)]
        rng.shuffle(sents)
        edges = [(names[i], r, names[i + 1]) for i, r in enumerate(chain)]
        sub = list(edges) + [(names[i], self.table[chain[i:j]], names[j])
                             for i in range(len(chain)) for j in range(i + 2, len(chain) + 1)
                             if chain[i:j] in self.table and (i, j) != (0, len(chain))]
        return {'story': ' '.join(sents), 'a': names[0], 'b': names[-1], 'rel': self.table[chain],
                'edges': edges, 'k': len(chain), 'sub': sub, 'genders': dict(zip(names, genders))}


# Unanswerable questions and distractors. Trained only on answerable questions, the model always names a
# relation and invents links when none exist. Templates are non-family sentences; EVAL_TEMPLATES are never
# trained on, so the checks also cover new wording. {Z}: new person, {X}: person already in the story.
PRON = {'male': ('he', 'him', 'his'), 'female': ('she', 'her', 'her')}
TRAIN_TEMPLATES = [
    '[{Z}] read a book in the park.', '[{Z}] went for a long run before breakfast.',
    '[{Z}] baked bread for the school bake sale.', '[{Z}] spent the whole afternoon fixing {zp} bike.',
    '[{Z}] is learning to play the guitar.', '[{Z}] won a prize at the science fair.',
    '[{Z}] walked {zp} dog around the lake.',
    '[{X}] went to the store with {xp} friend [{Z}].', '[{X}] and {xp} friend [{Z}] watched a movie together.',
    '[{X}] met {xp} coworker [{Z}] for lunch.', '[{Z}], a classmate of [{X}], helped {xo} with the homework.',
    '[{X}] shared a taxi with a stranger named [{Z}].', '[{Z}] works at the same office as [{X}].',
    '[{X}] played tennis with {xp} friend [{Z}] on Sunday.', '[{X}] asked {xp} neighbor [{Z}] to feed the cat.',
    '[{Z}] sold [{X}] a used car last spring.', '[{Z}] lives next door to [{X}] and often says hello.',
    '[{X}] took a cooking class taught by [{Z}].', '[{Z}] gave [{X}] directions to the museum.',
    '[{X}] and [{Z}] have been best friends since school.',
    "[{Z}] fixed the leaky faucet in [{X}]'s apartment.", '[{X}] waved to {xp} neighbor [{Z}] across the street.',
    '[{Z}] interviewed [{X}] for a job at the bakery.', '[{X}] studied for the exam with {xp} classmate [{Z}].',
    '[{Z}] delivered a package to [{X}] this morning.', '[{X}] hired a tutor named [{Z}] for the summer.',
    '[{Z}] coaches the basketball team that [{X}] plays on.', '[{X}] met [{Z}] at a concert and they became friends.',
    "[{Z}] is [{X}]'s landlord and lives upstairs.", '[{X}] and {xp} coworker [{Z}] finished the project early.',
    "[{Z}] cut [{X}]'s hair at the salon.", '[{X}] sat next to [{Z}] on the long flight.',
    '[{Z}] taught [{X}] how to swim at the community pool.', '[{Z}] went camping in the mountains.',
    '[{Z}] adopted a kitten from the shelter.', "[{Z}] repaired [{X}]'s car last week.",
    '[{X}] asked a police officer named [{Z}] for directions.', '[{Z}] sang at the open mic night on Friday.',
]
EVAL_TEMPLATES = [
    '[{Z}] painted a picture of the sea.', '[{Z}] planted tomatoes in the garden.',
    '[{Z}] took the train to work this morning.',
    '[{X}] carpooled to work with {xp} colleague [{Z}].', "[{Z}] has been [{X}]'s roommate since college.",
    '[{X}] hired [{Z}] to paint the kitchen.', '[{Z}] and [{X}] play on the same soccer team.',
    '[{X}] borrowed a ladder from {xp} neighbor [{Z}].', "[{Z}], [{X}]'s dentist, recommended a softer toothbrush.",
    '[{X}] chatted with [{Z}] at the bus stop.',
]


def people(story: str) -> list[str]:
    return list(dict.fromkeys(re.findall(r'\[([^\]]+)\]', story)))


def fits(ex: dict, block_size: int = 320) -> bool:
    """At most N_ENT people, and prefix + chain + answer within block_size."""
    names = set(people(ex['story'])) | {ex['a'], ex['b']} | {n for t in ex.get('sub', []) for n in (t[0], t[2])}
    return len(names) <= N_ENT and len(tokenize(anonymize(ex)[0])) + 4 * ex['k'] + 16 <= block_size


def _sentence(rng, templates, z, zg, genders: dict) -> str:
    xs = [p for p in genders if p != z]
    tpl = rng.choice(templates if xs else [t for t in templates if '{X}' not in t])
    x = rng.choice(xs) if xs else ''
    (zs, zo, zp), (xs_, xo, xp) = PRON[zg], PRON.get(genders.get(x), ('', '', ''))
    return tpl.format(Z=z, X=x, zs=zs, zo=zo, zp=zp, xs=xs_, xo=xo, xp=xp)


def _insert(story: str, new: list[str], rng) -> str:
    """Insert only before a sentence that starts with a name, or at the end: never between a sentence and a
    following 'His name is ...' that refers back to it."""
    sents = SENT_RE.split(story.strip())
    for s in new:
        sents.insert(rng.choice([i for i in range(len(sents) + 1) if i == len(sents) or sents[i].startswith('[')]), s)
    return ' '.join(sents)


def _rename(ex: dict, tag: str) -> dict:
    f = lambda n: f'{n}*{tag}'
    return {**ex, 'story': re.sub(r'\[([^\]]+)\]', lambda g: f'[{f(g.group(1))}]', ex['story']),
            'a': f(ex['a']), 'b': f(ex['b']), 'edges': [(f(x), r, f(y)) for x, r, y in ex['edges']],
            'links': [(f(x), f(y)) for x, y in ex.get('links', [])], 'sub': [(f(x), r, f(y)) for x, r, y in ex['sub']],
            'genders': {f(n): g for n, g in ex.get('genders', {}).items()}, '_paths': {}}


def _join(s1: str, s2: str, rng) -> str:
    return f'{s1} {s2}' if rng.random() < 0.5 else f'{s2} {s1}'


def unlinked(ex: dict) -> list[str]:
    """People in the story's facts with no chain of facts to A (CLUTRR's disconnected-noise stories)."""
    adj = {}
    for x, y in ex.get('links', []):
        adj.setdefault(x, set()).add(y); adj.setdefault(y, set()).add(x)
    seen, stack = {ex['a']}, [ex['a']]
    while stack:
        for m in adj.get(stack.pop(), ()):
            if m not in seen:
                seen.add(m); stack.append(m)
    return [n for n in adj if n not in seen]


def unanswerable(ex: dict, rng, templates, other: dict | None = None, kind: str | None = None):
    """Question about a person with no family link to the other: only a friend/stranger ('friend'), in another
    family ('two'; CLUTRR's own disconnected noise: 'disc'), or not in the story ('absent'). None if too long."""
    kind = kind or rng.choices(['friend', 'two', 'absent'], [0.5, 0.45, 0.05])[0]
    story, genders = ex['story'], dict(ex.get('genders', {}))
    who = people(story)
    if kind == 'disc':
        p, z = ex['a'], rng.choice(unlinked(ex))
    elif kind == 'friend':
        z, zg = 'New*0', rng.choice(['male', 'female'])
        new = [_sentence(rng, templates, z, zg, genders)]
        if rng.random() < 0.3:   # a second outsider
            new.append(_sentence(rng, templates, 'New*1', rng.choice(['male', 'female']), genders))
        story, genders[z], p = _insert(story, new, rng), zg, rng.choice(who)
    elif kind == 'two':
        o = _rename(other, 'o')
        story, p, z = _join(story, o['story'], rng), rng.choice(who), rng.choice(people(o['story']))
        genders.update(o['genders'])
    else:
        p, z = rng.choice(who), 'Nobody*'
    a, b = (p, z) if rng.random() < 0.5 else (z, p)
    edges = ex['edges'] + (o['edges'] if kind == 'two' else [])   # each family's facts; no path between them
    out = {'story': story, 'a': a, 'b': b, 'rel': UNKNOWN, 'edges': edges, 'links': [], 'k': 0, 'sub': [],
           'genders': genders, 'seen': False, 'kind': kind, '_paths': {}}
    return out if fits(out) else None


def add_distractors(ex: dict, rng, templates, other: dict | None = None):
    """Same question and answer, plus non-family sentences about new people or another family's story."""
    genders = dict(ex.get('genders', {}))
    if other is not None and rng.random() < 0.5:
        o = _rename(other, 'o')
        story = _join(ex['story'], o['story'], rng)
        genders.update(o['genders'])
    else:
        new = []
        for i in range(rng.randint(1, 2)):
            z, zg = f'New*{i}', rng.choice(['male', 'female'])
            new.append(_sentence(rng, templates, z, zg, genders)); genders[z] = zg
        story = _insert(ex['story'], new, rng)
    out = {**ex, 'story': story, 'genders': genders, '_paths': {}}
    return out if fits(out) else None


FEMALE_EDGES = [r for r in EDGE_RELS if r not in MALE]
MALE_EDGES = [r for r in EDGE_RELS if r in MALE]


def bad_chain(ex: dict, rng, templates, other: dict | None = None, table: dict | None = None):
    """The real chain from A, cut at a random step that is replaced by a false one. Target: <nofact> after it.
    Same story (needs table: relation chain -> answer, to know the true relation):
      'wrong'  a family member with a wrong relation of the right gender;
      'swap'   a real fact "w r y" with the subject swapped to the chain's x (false for x).
    Added people: 'cross' a person in another family or a friend/stranger with a random relation;
      'near'   a real fact of the other family with the subject swapped to x (looks plausible, is false).
    Half of the added-people ones also ask about that person, so the question itself has no answer."""
    steps = path_facts(ex, ex['a'], ex['b'])
    if not steps:
        return None
    i = rng.randrange(len(steps))
    x, genders = steps[i][0], dict(ex.get('genders', {}))
    out = {**ex, 'nofact': True, 'k': i + 1, 'sub': [], '_paths': {}}
    kind = rng.choice(['wrong', 'swap', 'cross', 'near'] if table is not None else ['cross', 'near'])
    if kind in ('wrong', 'swap'):
        seen = {s[0] for s in steps[:i + 1]}
        def true_rel(y):
            path = path_facts(ex, x, y)
            return path and (path[0][1] if len(path) == 1 else table.get(tuple(r for _, r, _ in path)))
        if kind == 'wrong':
            opts = [(y, r) for y in people(ex['story']) if y not in seen and y in genders and true_rel(y)
                    for r in (MALE_EDGES if genders[y] == 'male' else FEMALE_EDGES) if r != true_rel(y)]
        else:
            opts = [(y, r) for w, r, y in ex['edges'] if w != x and y not in seen and true_rel(y) and true_rel(y) != r]
        if not opts:
            return None
        y, r = rng.choice(opts)
        out['chain'] = steps[:i] + [(x, r, y)]
        return out if fits(out) else None
    if other is not None and (kind == 'near' or rng.random() < 0.5):
        o = _rename(other, 'o')
        story = _join(ex['story'], o['story'], rng)
        genders.update(o['genders'])
        r, z = (rng.choice(o['edges'])[1:] if kind == 'near' and o['edges']
                else (rng.choice(EDGE_RELS), rng.choice(people(o['story']))))
    else:
        z, zg = 'New*0', rng.choice(['male', 'female'])
        story = _insert(ex['story'], [_sentence(rng, templates, z, zg, genders)], rng)
        genders[z], r = zg, rng.choice(EDGE_RELS)
    out.update(story=story, genders=genders, chain=steps[:i] + [(x, r, z)])
    if rng.random() < 0.5:
        out.update(b=z, rel=UNKNOWN)
    return out if fits(out) else None


def step_truth(ex: dict, x: str, r: str, y: str, table: dict) -> bool | None:
    """Is "y is x's r" true? False across families or to an outsider; within a family, compared with the
    relation from the facts (relation chain -> table). None when it cannot be told."""
    comp = {}
    for a, _, b in ex['edges']:
        ca, cb = comp.setdefault(a, {a}), comp.setdefault(b, {b})
        if ca is not cb:
            ca |= cb
            for n in cb:
                comp[n] = ca
    if x not in comp or y not in comp or comp[x] is not comp[y]:
        return False
    path = path_facts(ex, x, y)
    true = path and (path[0][1] if len(path) == 1 else table.get(tuple(t for _, t, _ in path)))
    return None if not true else true == r


def mine_negatives(model, vocab: Vocab, exs: list[dict], device, table: dict, beam: int = 4, bs: int = 64) -> list[dict]:
    """Hard negatives: the model's own best chain for unanswerable questions, cut at its first false step,
    which is labelled <nofact>. These are the invented links beam search finds; random ones are too easy.
    bs questions per search: the prefix key/value cache is ~5.5 MB per question."""
    chains = []
    for i in range(0, len(exs), bs):
        chains += beam_predict(model, vocab, exs[i:i + bs], device, beam=beam, views=1, unk_threshold=math.inf)[1]
        if device.type == 'mps':
            torch.mps.empty_cache()
    out = []
    for ex, chain in zip(exs, chains):
        names = {tok: n for n, tok in anonymize(ex)[1].items()}
        steps = [tuple(names.get(w, w) for w in st.split()) for st in chain.split('<sep>') if st.strip()]
        for j, (x, r, y) in enumerate(steps):
            ok = step_truth(ex, x, r, y, table)
            if ok is None:
                break
            if not ok:
                neg = {**ex, 'chain': list(steps[:j + 1]), 'nofact': True, 'k': j + 1, '_paths': {}}
                if fits(neg):
                    out.append(neg)
                break
    return out


PRONOUN_START = re.compile(r"^(he|she|his|her|him|they|their|them)\b", re.I)


def linked_in_text(story: str, a: str, b: str) -> bool:
    """Rule check: are A and B connected through sentences that mention people together? A sentence starting
    with a pronoun joins the one before ("... her son. His name is [James]."). A story can only state a family
    link between people connected this way, so False means the story cannot answer."""
    groups, cur = [], []
    for sent in SENT_RE.split(story.strip()):
        if cur and not PRONOUN_START.match(sent.strip()):
            groups.append(cur); cur = []
        cur += re.findall(r'\[([^\]]+)\]', sent)
    groups.append(cur)
    comp = {}
    for g in groups:
        merged = set().union(*[comp.get(n, {n}) for n in g]) if g else set()
        for n in merged:
            comp[n] = merged
    return a in comp and b in comp and comp[a] is comp[b]


def unknown_eval_sets(exs: list[dict], n: int, seed: int = 0, templates=EVAL_TEMPLATES) -> dict:
    """Held-out checks from clean held-out stories (never trained on, no training twin) and EVAL_TEMPLATES."""
    rng = random.Random(seed)
    clean = [e for e in exs if not e.get('seen')]
    def build(make):
        out = []
        for _ in range(n * 20):
            e = rng.choice(clean)
            x = make(e, rng.choice([o for o in clean[:200] if o is not e]))
            if x is not None:
                out.append(x)
            if len(out) == n:
                break
        return out
    return {'answerable + distractors': build(lambda e, o: add_distractors(e, rng, templates, o)),
            'unknown: friend/stranger': build(lambda e, o: unanswerable(e, rng, templates, kind='friend')),
            'unknown: two families': build(lambda e, o: unanswerable(e, rng, templates, o, kind='two')),
            'unknown: not in story': build(lambda e, o: unanswerable(e, rng, templates, kind='absent'))}

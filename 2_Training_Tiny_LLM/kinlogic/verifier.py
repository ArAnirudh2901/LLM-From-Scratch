"""Learned verifier (LEVER, Ni et al. 2023): a logistic model scores each reading from what the engine made of it, not
only how likely the reader found it. Readings that are consistent but wrong are common (engine checks alone never
helped), so the verifier learns which signals mark a right one: vote share, log-probability gap, engine status, named
people left out, facts per sentence. Trained by train_verifier.py on keyed questions that are not evaluation
questions; exam._answer weights its vote by it when KIN_VERIFIER names a weights file.
"""
from __future__ import annotations

import json
import math
import os
import re

STATUS = ['unique', 'gender', 'conditional', 'ambiguous', 'assumed', 'side', 'unrelated', 'pair', 'claims']
NONE = re.compile(r'none of|cannot be|can.?t be|data (?:is )?inadequate|insufficient', re.I)
ROLE = {'father': 'line', 'mother': 'line', 'son': 'line', 'daughter': 'line', 'brother': 'sibling',   # 'son of' may be read as 'father'
        'sister': 'sibling', 'husband': 'spouse', 'wife': 'spouse'}


def coverage(toks, facts: str) -> tuple[float, float]:
    """Round-trip checks a careful reader makes: share of sentences whose people all appear in the facts, and share of
    the question's family roles (parent or child, sibling, spouse) that the facts use."""
    used = set(re.findall(r'[@#]\d+', facts))
    sents, cur = [], set()
    for t in toks:
        if re.fullmatch(r'@\d+', t):
            cur.add(t)
        if t in ('.', '?') and cur:
            sents.append(cur)
            cur = set()
    if cur:
        sents.append(cur)
    words = set(facts.split())
    asked = {ROLE[t] for t in toks if t in ROLE}
    have = {ROLE[w] for w in words if w in ROLE}
    return (sum(s <= used for s in sents) / len(sents) if sents else 1.0,
            len(asked & have) / len(asked) if asked else 1.0)


def features(p: dict, res: dict, score: float, top: float, share: float, rank: int, n: int) -> list[float]:
    toks = p['tokens']
    people = {t for t in toks if re.fullmatch(r'@\d+', t)}
    facts = res.get('facts') or ''
    used = set(re.findall(r'@\d+', facts))
    n_facts = sum(1 for f in facts.split(';') if f.strip() and f.split()[0] not in ('ask', 'who', 'count', 'gender', 'pair', 'prof'))
    sents = max(1, sum(t in ('.', '?') for t in toks))
    opt = dict(p.get('options') or []).get(res.get('option'), '')
    st = res.get('status') or ''
    return [1.0, score - top, share, rank / max(1, n), len(people - used), len(used & people) / max(1, len(people)),
            n_facts / sents, float('[relaxed' in facts), float(bool(NONE.search(opt or ''))),
            float(any(t == 'members' for t in facts.split())), float('means' in toks), float('"' in toks), len(toks) / 100.0,
            *[float(st.startswith(s)) for s in STATUS], *coverage(toks, facts)]   # new features last: older weights ignore them


class Verifier:
    def __init__(self, w):
        self.w = w

    @classmethod
    def load(cls, path):
        return cls(json.load(open(path))['w']) if path and os.path.exists(path) else None

    def prob(self, x) -> float:
        z = sum(a * b for a, b in zip(self.w, x))
        return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))


def fit(X, y, l2: float = 1e-3, steps: int = 400, lr: float = 0.5):
    """Logistic regression by full-batch gradient descent (features are few and small)."""
    import numpy as np
    X, y = np.asarray(X, float), np.asarray(y, float)
    mu, sd = X.mean(0), X.std(0) + 1e-6
    mu[0], sd[0] = 0.0, 1.0   # bias column
    Z = (X - mu) / sd
    w = np.zeros(X.shape[1])
    for _ in range(steps):
        p = 1 / (1 + np.exp(-np.clip(Z @ w, -30, 30)))
        w -= lr * (Z.T @ (p - y) / len(y) + l2 * w)
    w_raw = w / sd   # fold the scaling back so prob() takes raw features
    w_raw[0] = w[0] - (w[1:] * mu[1:] / sd[1:]).sum()
    return [float(v) for v in w_raw]

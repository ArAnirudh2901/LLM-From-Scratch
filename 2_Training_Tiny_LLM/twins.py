"""Twin views for consistency training (train_reader --cons_p): one question with its fact sentences in another order
and its sentence breaks typed differently (a comma for a full stop; no full stop before the question). The facts stay
token for token the same, so the reader is trained to read both the same way (paraphrase consistency: R-Drop, Liang
et al. 2021; UDA, Xie et al. 2020). Works on prepare()'s tokens."""
from __future__ import annotations

import random

ANAPHORA = {'he', 'she', 'his', 'her', 'him', 'hers', 'they', 'their', 'them', 'it', 'this', 'that', 'these', 'those',
            'former', 'latter', 'rest', 'remaining', 'other', 'others', 'both', 'also', 'same', 'such', 'too'}
DESC = {'man', 'woman', 'boy', 'girl', 'lady', 'gentleman', 'person', 'child', 'men', 'women'}   # "a man ... the man"
QUOTES = {'"', '“', '”', "'", '‘', '’'}


def _sentences(tokens):
    out, cur = [], []
    for t in tokens:
        cur.append(t)
        if t in ('.', '!', '?'):
            out.append(cur)
            cur = []
    return out + ([cur] if cur else [])


def twin(tokens: list[str], rng: random.Random) -> list[str] | None:
    """Another surface of the same question, or None when no safe change applies."""
    sents = _sentences(list(tokens))
    if not sents or sents[-1][-1] != '?':
        return None
    body, q = sents[:-1], sents[-1]
    if any(s[-1] != '.' for s in body):
        return None
    changed = False
    if len(body) >= 2 and not set(tokens) & (ANAPHORA | DESC | QUOTES) and rng.random() < 0.7:
        order = list(range(len(body)))
        while order == sorted(order):
            rng.shuffle(order)
        body, changed = [body[i] for i in order], True
    body = [list(s) for s in body]
    if len(body) >= 2 and rng.random() < 0.4:   # "A is B's son , C is A's wife ."
        body[rng.randrange(len(body) - 1)][-1] = ','
        changed = True
    if body and rng.random() < 0.4:   # "... of B how is A related to B ?"
        body[-1] = body[-1][:-1]
        changed = True
    return [t for s in body for t in s] + q if changed else None

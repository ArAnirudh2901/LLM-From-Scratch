"""Phrase-type balance for training (phrase_audit.py): rewrites verified generated questions into the clause shapes real
exams use far more than our generator (', who is' 18x, 'who is' without a comma 450x, 'whose' 7x), by merging adjacent
sentences. Meaning and facts are unchanged; sentence order is kept, so the fact order (text order) still holds. Each
rewrite is checked: when an exact reader reads it, its answer must equal the gold answer, or the row is dropped.

    python boost_phrases.py --pool data/dataset_v6 --n 60000 --out data/phrase_boost.jsonl.gz
"""
from __future__ import annotations

import argparse
import collections
import glob
import gzip
import json
import random
import re

E = r'([@#]\d+)'
REL = r'((?:the |a |an )?(?:only )?(?:paternal |maternal )?(?:elder |younger )?[a-z][a-z-]*)'


def sentences(tokens):
    """Statement sentences (token lists ending in '.') and the rest (the question) kept as one tail."""
    out, cur = [], []
    for t in tokens:
        cur.append(t)
        if t == '.':
            out.append(cur)
            cur = []
    return out, cur


def merge(s1, s2, rng):
    """One clause sentence from two adjacent statements, or None."""
    a, b = ' '.join(s1[:-1]), ' '.join(s2[:-1])
    m1 = re.fullmatch(rf'{E} (is|was) {REL} of {E}', a)
    if m1:
        x, v, r, y = m1.groups()
        m2 = re.fullmatch(rf'{re.escape(y)} (is|was) (.+)', b)   # "X is the R of Y . Y is the S of Z"
        if m2:
            return f'{x} {v} {r} of {y} , who {m2.group(1)} {m2.group(2)} .'.split()
        m2 = re.fullmatch(rf"{re.escape(y)} 's {REL} is {E}", b)   # "X is the R of Y . Y 's S is Z" -> "... Y , whose S is Z"
        if m2:
            return f'{x} {v} {r} of {y} , whose {m2.group(1)} is {m2.group(2)} .'.split()
        m2 = re.fullmatch(rf'{re.escape(y)} (has|had) (.+)', b)   # "X is the R of Y . Y has ..."
        if m2:
            return f'{x} {v} {r} of {y} , who {m2.group(1)} {m2.group(2)} .'.split()
        m2 = re.fullmatch(rf'{re.escape(x)} (is|was|has|had) (.+)', b)   # "X is the R of Y . X has ..."
        if m2:
            if rng.random() < 0.4:   # real exams: ', who is' 2:1 over 'who is' without commas; the generator wrote neither
                return f'{x} , who {v} {r} of {y} , {m2.group(1)} {m2.group(2)} .'.split()
            return f'{x} who {v} {r} of {y} {m2.group(1)} {m2.group(2)} .'.split()
    m1 = re.fullmatch(rf"{E} 's {REL} is {E}", a)   # "X 's R is Y . X is ..." -> "X , whose R is Y , is ..."
    if m1:
        x, r, y = m1.groups()
        m2 = re.fullmatch(rf'{re.escape(x)} (is|was|has|had) (.+)', b)
        if m2:
            return f'{x} , whose {r} is {y} , {m2.group(1)} {m2.group(2)} .'.split()
        m2 = re.fullmatch(rf"{re.escape(y)} 's {REL} is {E}", b)   # "X 's R is Y . Y 's S is Z" -> "... Y , whose S is Z"
        if m2:
            return f"{x} 's {r} is {y} , whose {m2.group(1)} is {m2.group(2)} .".split()
        m2 = re.fullmatch(rf'{re.escape(y)} (is|was|has|had) (.+)', b)   # "X 's R is Y . Y is ..." -> "... Y , who is ..."
        if m2:
            return f"{x} 's {r} is {y} , who {m2.group(1)} {m2.group(2)} .".split()
    return None


def boost(tokens, rng, p=0.8):
    sents, tail = sentences(tokens)
    out, i, changed = [], 0, False
    while i < len(sents):
        if i + 1 < len(sents) and rng.random() < p:
            m = merge(sents[i], sents[i + 1], rng)
            if m:
                out.append(m)
                i += 2
                changed = True
                continue
        out.append(sents[i])
        i += 1
    return ([t for s in out for t in s] + tail) if changed else None


def main():
    from kinlogic.codes import _fits
    from kinlogic.rules import read as read_rules
    from train_reader import answers
    ap = argparse.ArgumentParser()
    ap.add_argument('--pool', default='data/dataset_v6')
    ap.add_argument('--n', type=int, default=60000, help='source rows to try')
    ap.add_argument('--out', default='data/phrase_boost.jsonl.gz')
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    rows = []
    for f in sorted(glob.glob(f'{a.pool}/kinds/*.jsonl.gz')) + sorted(glob.glob(f'{a.pool}/deep/*.jsonl.gz')):
        try:
            got = [json.loads(l) for _, l in zip(range(a.n // 8), gzip.open(f, 'rt'))]
        except (EOFError, OSError):
            continue
        rows += got
    rng.shuffle(rows)
    st, out = collections.Counter(), []
    for r in rows[:a.n]:
        toks = r.get('tokens')
        if not isinstance(toks, list) or not r.get('lf'):
            continue
        new = boost(toks, rng)
        if not new:
            st['no adjacent pair to merge'] += 1
            continue
        people = sorted({t for t in toks if t[0] == '@'})
        try:
            lf = read_rules(new)
            if lf and _fits(lf) and answers(lf, people) != answers(r['lf'], people):
                st['exact reader disagrees (dropped)'] += 1
                continue
            st['checked by an exact reader' if lf else 'kept (by construction)'] += 1
        except Exception:
            st['kept (by construction)'] += 1
        out.append({**r, 'kind': 'phrase-' + str(r.get('kind', 'q')), 'tokens': new, 'text': ' '.join(new)})
    with gzip.open(a.out, 'wt') as f:
        f.write(''.join(json.dumps(x, ensure_ascii=False) + '\n' for x in out))
    from phrase_audit import PAT
    rare = ['paternal / maternal', 'of-chain (R of the S of)', 'count (has two sons)', 'negation (not / no)', 'code symbols (means)',
            'grand-/great-', 'only (only son / only child)', 'only daughter / son of my', 'profession', 'brother-/sister- (siblings)',
            'possessive chain x2 (X \'s R \'s S)', 'pointing / photo / picture']
    keep = []
    for r in rows:
        t = ' '.join(r['tokens']) if isinstance(r.get('tokens'), list) else ''
        if t and not PAT["X 's R is Y"].search(t) and any(PAT[k].search(t) for k in rare):
            keep.append({**r, 'kind': 'rare-' + str(r.get('kind', 'q'))})
    rare_out = a.out.replace('.jsonl.gz', '_rare.jsonl.gz')
    with gzip.open(rare_out, 'wt') as f:
        f.write(''.join(json.dumps(x, ensure_ascii=False) + '\n' for x in keep))
    print(len(keep), 'rows with rare meaning types (and no "X \'s R is Y") ->', rare_out)
    print(len(out), 'rows ->', a.out, dict(st))


if __name__ == '__main__':
    main()

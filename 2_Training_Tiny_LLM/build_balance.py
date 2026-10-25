"""Balanced extra rows for r8 (phrase_audit.py gaps): per-type quotas of verified generated rows for the meaning types
real exams use 2-5x more than our pool (paternal/maternal, "R of the S of", negation, codes, counts, grand-/great-,
only), the clause rewrites from boost_phrases.py, and "X 's R is Y" (8x over-used by the generator) rewritten to
"Y is the R of X" inside these rows (same fact; --of_form aligns the fact order with the text).

    python build_balance.py --pool data/dataset_v6 --out data/phrase_balance.jsonl.gz
"""
from __future__ import annotations

import argparse
import collections
import glob
import gzip
import json
import random
import re

from phrase_audit import PAT

QUOTA = {'paternal / maternal': 25000, 'of-chain (R of the S of)': 25000, 'negation (not / no)': 20000, 'code symbols (means)': 12000,
         'count (has two sons)': 10000, 'grand-/great-': 8000, 'only (only son / only child)': 8000, 'profession': 5000, "possessive chain x2 (X 's R 's S)": 15000, 'only daughter / son of my': 8000}
POSS = re.compile(r"(^|[.;,] )(@\d+) 's ((?:only )?(?:paternal |maternal )?[a-z][a-z-]*) is (@\d+)(?= [.;,?])")


def unposs(tokens):
    """'X 's R is Y' sentences -> 'Y is the R of X' (same fact)."""
    t = ' '.join(tokens)
    return POSS.sub(lambda m: f'{m.group(1)}{m.group(4)} is the {m.group(3)} of {m.group(2)}', t).split()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pool', default='data/dataset_v6')
    ap.add_argument('--per_file', type=int, default=120000)
    ap.add_argument('--boost', default='data/phrase_boost.jsonl.gz')
    ap.add_argument('--out', default='data/phrase_balance.jsonl.gz')
    a = ap.parse_args()
    rng = random.Random(1)
    rows = []
    for f in sorted(glob.glob(f'{a.pool}/kinds/*.jsonl.gz')) + sorted(glob.glob(f'{a.pool}/deep/*.jsonl.gz')):
        try:
            rows += [json.loads(l) for _, l in zip(range(a.per_file), gzip.open(f, 'rt'))]
        except (EOFError, OSError):
            pass
    rng.shuffle(rows)
    got, used, out = collections.Counter(), set(), []
    for r in rows:
        toks = r.get('tokens')
        if not isinstance(toks, list) or not r.get('lf'):
            continue
        t = ' '.join(toks)
        need = [k for k, q in QUOTA.items() if got[k] < q and PAT[k].search(t)]
        if not need or t in used:
            continue
        used.add(t)
        for k in need:
            got[k] += 1
        out.append({**r, 'kind': 'bal-' + str(r.get('kind', 'q')), 'tokens': unposs(toks)})
    boost = [json.loads(l) for l in gzip.open(a.boost, 'rt')]
    out += [{**r, 'tokens': unposs(r['tokens'])} for r in boost]
    rng.shuffle(out)
    with gzip.open(a.out, 'wt') as f:
        f.write(''.join(json.dumps(x, ensure_ascii=False) + '\n' for x in out))
    print(len(out), 'rows ->', a.out, '| quotas filled:', dict(got), '| clause rewrites:', len(boost))


if __name__ == '__main__':
    main()

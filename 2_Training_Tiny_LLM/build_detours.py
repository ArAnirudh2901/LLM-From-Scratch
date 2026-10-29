"""A detour supplement: questions that name people through longer exact paths ("mother's husband", "the only son of
my father's wife"), one to three detours, in statement, speech and question-embedded forms (kinlogic.generate.
obfuscation_set). The browser pipeline answers 93% of one-detour but 77% of two-detour questions (eval.gauntlet), and
renamed copies agree, so the gap is reading longer descriptions, which this trains.

    python build_detours.py --rows 4000 --seed 7 --out data/detour_v1.jsonl.gz
"""
from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import json
import random

from build_balanced import checked
from kinlogic.generate import obfuscation_set


def build(rows: int, seed: int):
    rng, out, seen = random.Random(seed), [], set()
    want = {n: rows // 3 + (n <= rows % 3) for n in (1, 2, 3)}
    got = collections.Counter()
    while any(got[n] < want[n] for n in want):
        n = rng.choice([k for k in want if got[k] < want[k]])
        ex = obfuscation_set(random.Random(rng.getrandbits(64)), (n,))
        ex = ex[0] if ex else None
        if not ex or ' '.join(ex['tokens']) in seen or not checked(ex):
            continue
        seen.add(' '.join(ex['tokens']))
        out.append({k: ex[k] for k in ('kind', 'text', 'tokens', 'lf', 'truth', 'detours')})
        got[n] += 1
    rng.shuffle(out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rows', type=int, default=4000)
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--out', default='data/detour_v1.jsonl.gz')
    a = ap.parse_args()
    rows = build(a.rows, a.seed)
    with gzip.open(a.out, 'wt') as f:
        f.writelines(json.dumps(r) + '\n' for r in rows)
    sha = hashlib.sha256(open(a.out, 'rb').read()).hexdigest()
    print(f'{len(rows)} rows -> {a.out} (sha256 {sha}):', dict(collections.Counter(r['detours'] for r in rows)),
          dict(collections.Counter(r['truth'] for r in rows).most_common(6)))


if __name__ == '__main__':
    main()

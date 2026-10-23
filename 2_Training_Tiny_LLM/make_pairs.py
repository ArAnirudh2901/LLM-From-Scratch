"""Minimal-pair training rows (kinlogic.generate.flip_pair): one relation word flipped in a generated question and its
facts, kept only when the answer changes. Rows have the pool's fields, so `train_reader --reworded` can mix them in.

    python make_pairs.py --out data/pairs_v1.jsonl.gz --n 20000 --workers 6
"""
from __future__ import annotations

import argparse
import gzip
import json
import random
from concurrent.futures import ProcessPoolExecutor


def _chunk(args):
    seed, n = args
    from kinlogic.generate import flip_pair, sample
    rng, out = random.Random(seed), []
    while len(out) < n:
        ex = sample(rng)
        pr = flip_pair(ex, rng)
        if pr:
            out.append({'kind': pr['kind'], 'text': pr['text'], 'tokens': pr['tokens'], 'lf': pr['lf']})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--n', type=int, default=20000)
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--seed', type=int, default=7001)
    a = ap.parse_args()
    per = 500
    jobs = [(a.seed + i, per) for i in range(-(-a.n // per))]
    rows = []
    with ProcessPoolExecutor(a.workers) as pool:
        for got in pool.map(_chunk, jobs):
            rows += got
            print(f'{len(rows)}/{a.n}', flush=True)
    with gzip.open(a.out, 'wt') as f:
        f.write(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows[:a.n]))


if __name__ == '__main__':
    main()

"""A test set no model trains on or is picked with: freshly generated, engine-verified questions (every kind, deep
chains 1-12 in five formats) whose people carry names outside every generator list, minus any question whose wording
(names masked) appears in a training or tuning file. Built once and frozen; only reported, never used to choose.

    python -m eval.unseen_set            -> data/unseen_test.jsonl.gz + data/unseen_test.manifest.json
"""
from __future__ import annotations

import argparse
import collections
import csv
import glob
import gzip
import hashlib
import json
import os
import random
from concurrent.futures import ProcessPoolExecutor

OUT = 'data/unseen_test.jsonl.gz'
SEED = 7_000_000_000   # far from build_dataset.py's hashed seeds; overlaps are removed anyway


def held_out_names(path='data/names/name_gender.tsv.gz', per=300, seed=0):
    """{'indian': (male, female), 'world': (male, female)}: well-evidenced clear-gender names no generator list has."""
    from build_names import common_words
    from kinlogic.names import INDIAN_F, INDIAN_M, UNISEX, WORLD_F, WORLD_M
    used = {n.lower() for n in INDIAN_M + INDIAN_F + WORLD_M + WORLD_F} | set(UNISEX) | common_words()
    pools = collections.defaultdict(list)
    with gzip.open(path, 'rt', newline='') as f:
        for r in csv.DictReader(f, delimiter='\t'):
            n, p, s, src = r['name'], float(r['p_female']), int(r['support']), r['sources']
            if s < 2000 or not n.isalpha() or not 4 <= len(n) <= 9 or n in used or 0.03 < p < 0.97:
                continue
            style = 'indian' if 'naampy' in src and 'ssa' not in src else 'world' if 'ssa' in src and 'naampy' not in src else None
            if style:
                pools[(style, 'F' if p >= 0.97 else 'M')].append(n.capitalize())
    rng = random.Random(seed)
    pick = lambda k: rng.sample(sorted(pools[k]), min(per, len(pools[k])))
    return {s: (pick((s, 'M')), pick((s, 'F'))) for s in ('indian', 'world')}


def _use_names(names):
    """Swaps the generator's name lists in place (every module shares the list objects)."""
    from kinlogic import names as nm
    for lst, new in ((nm.INDIAN_M, names['indian'][0]), (nm.INDIAN_F, names['indian'][1]),
                     (nm.WORLD_M, names['world'][0]), (nm.WORLD_F, names['world'][1])):
        lst[:] = new


def _job(args):
    from build_dataset import _deep_job, _kind_job
    kind, job = args
    return (_kind_job if kind == 'kind' else _deep_job)(job)


def key(tokens) -> bytes:
    return hashlib.sha1(' '.join(tokens).encode()).digest()


def seen_keys(pool='data/dataset_v2', extra=('data/reworded.jsonl.gz', 'data/reworded_dev.jsonl.gz')):
    """Question token sequences (names masked as @k) of every training and tuning file."""
    keys = set()
    for f in glob.glob(f'{pool}/**/*.jsonl.gz', recursive=True) + [e for e in extra if os.path.exists(e)]:
        for line in gzip.open(f, 'rt'):
            keys.add(key(json.loads(line)['tokens']))
    return keys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=OUT)
    ap.add_argument('--per_kind', type=int, default=100)
    ap.add_argument('--per_deep', type=int, default=10, help='per format and depth (5 formats x 12 depths)')
    a = ap.parse_args()
    from kinlogic import deep, generate as gen
    names = held_out_names()
    jobs = [('kind', (k, a.per_kind, SEED + i)) for i, k in enumerate(sorted(gen.KINDS))]
    jobs += [('deep', (fmt, d, a.per_deep, SEED + 1000 + 100 * i + d)) for i, fmt in enumerate(deep.FORMATS) for d in range(1, 13)]
    with ProcessPoolExecutor(initializer=_use_names, initargs=(names,)) as pool:
        res = list(pool.map(_job, jobs))
    rows = [x for r in res for x in r[2]]
    seen = seen_keys()
    fresh, overlap, dup = [], 0, set()
    for r in rows:
        k = key(r['tokens'])
        if k in seen or k in dup:
            overlap += 1
            continue
        dup.add(k)
        fresh.append(r)
    with gzip.open(a.out, 'wt') as f:
        f.write(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in fresh))
    man = {'rows': len(fresh), 'dropped_seen_wording': overlap, 'by_kind': dict(collections.Counter(r['kind'] for r in fresh)),
           'by_depth': dict(sorted(collections.Counter(r.get('depth') or 0 for r in fresh).items())),
           'names': {s: [len(m), len(f)] for s, (m, f) in names.items()}, 'name_examples': names['indian'][0][:5] + names['world'][1][:5],
           'sha256': hashlib.sha256(open(a.out, 'rb').read()).hexdigest()}
    json.dump(man, open(a.out.replace('.jsonl.gz', '.manifest.json'), 'w'), indent=1)
    print(json.dumps(man, indent=1))


if __name__ == '__main__':
    main()

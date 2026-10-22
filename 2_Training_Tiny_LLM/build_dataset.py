"""Builds the counted, verified question set: every question is built from a known family and re-solved by the
engine before it counts (kinlogic/deep.py check); the run keeps going until every quota is full.

    python build_dataset.py [--per 6000] [--test-per 80] [--per-kind 100000] [--workers 8]

data/dataset/deep/<format>/d<depth>.jsonl.gz   deep chains: 5 formats x depths 1-12, `per` each (training)
data/dataset/test/deep.jsonl.gz                `test-per` per format and depth (400 per depth by default), never in
                                               training (text-disjoint)
data/dataset/kinds/<kind>.jsonl.gz             every other generator kind, `per-kind` each
data/dataset/manifest.json                     counts, rejects and how many were tried
Resumable: existing files count toward the quotas.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import random
import re
import time
from multiprocessing import Pool

OUT = 'data/dataset'
FIELDS = ('kind', 'text', 'tokens', 'lf', 'truth', 'answer', 'depth', 'names')


def _key(text):
    return hashlib.sha1(re.sub(r'\s+', ' ', text.lower()).encode()).hexdigest()


def _deep_job(args):
    fmt, d, n, seed = args
    from kinlogic import deep
    rng = random.Random(seed)
    out, tried, why = [], 0, {}
    while len(out) < n and tried < n * 20:
        tried += 1
        ex = deep.make(rng, fmt, d)
        if ex is None:
            why['render'] = why.get('render', 0) + 1
            continue
        w = deep.check(ex)
        if w:
            k = w.split(',')[0]
            why[k] = why.get(k, 0) + 1
            continue
        out.append({k: ex.get(k) for k in FIELDS})
    return fmt, d, out, tried, why


def _kind_job(args):
    kind, n, seed = args
    from kinlogic import deep, generate as gen
    from kinlogic.engine import solve
    from kinlogic.facts import parse
    rng = random.Random(seed)
    out, tried, why = [], 0, {}
    while len(out) < n and tried < n * 20:
        tried += 1
        ex = gen.KINDS[kind](rng)
        if ex is None:
            why['render'] = why.get('render', 0) + 1
            continue
        st, bad = parse(ex['lf'])
        r = solve(st, people=sorted({t for t in ex['tokens'] if t[0] == '@'}), max_steps=20000) if not bad else None
        ok = r is not None and r['ok'] and (bool(r['questions']) or bool(r['claims']))
        if ok and isinstance(ex.get('truth'), str) and r['questions'] and r['questions'][-1]['answers']:
            q = r['questions'][-1]
            got = {a.get('key') for a in q['answers']} | {a.get('base') for a in q['answers']} | {q.get('neutral')}
            ok = ex['truth'] in got or q['status'] != 'unique' or deep._neutral(ex['truth']) in {deep._neutral(k) for k in got if k}
        if not ok:
            why['engine'] = why.get('engine', 0) + 1
            continue
        out.append({k: ex.get(k) for k in FIELDS} | {'kind': kind, 'truth': ex['truth'] if isinstance(ex.get('truth'), str) else None})
    return kind, None, out, tried, why


def _load(path):
    if not os.path.exists(path):
        return []
    with gzip.open(path, 'rt') as f:
        return [json.loads(l) for l in f]


def _save(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + '.tmp'
    with gzip.open(tmp, 'wt') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    os.replace(tmp, path)


def main():
    global OUT
    from kinlogic import deep, generate as gen
    ap = argparse.ArgumentParser()
    ap.add_argument('--per', type=int, default=6000)
    ap.add_argument('--test-per', type=int, default=80)
    ap.add_argument('--per-kind', type=int, default=100000)
    ap.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument('--depths', default='1-12')
    ap.add_argument('--root', default='data/dataset', help='where the dataset lives (e.g. on Google Drive, built once)')
    a = ap.parse_args()
    OUT = a.root
    lo, hi = map(int, a.depths.split('-'))
    depths = range(lo, hi + 1)
    kinds = [k for k in gen.MIX if k != 'deep']
    t0 = time.time()

    test_path = f'{OUT}/test/deep.jsonl.gz'
    buckets = {('test', f, d): [] for f in deep.FORMATS for d in depths}
    for r in _load(test_path):
        buckets.setdefault(('test', r['kind'].split('-', 1)[1], r['depth']), []).append(r)
    for f in deep.FORMATS:
        for d in depths:
            buckets[('deep', f, d)] = _load(f'{OUT}/deep/{f}/d{d:02d}.jsonl.gz')
    for k in kinds:
        buckets[('kind', k, None)] = _load(f'{OUT}/kinds/{k}.jsonl.gz')
    seen = {_key(r['text']) for rows in buckets.values() for r in rows}
    stats = {'tried': 0, 'rejected': {}}

    def want(b):
        return {'test': a.test_per, 'deep': a.per, 'kind': a.per_kind}[b[0]] - len(buckets[b])

    rnd = 0
    with Pool(a.workers) as pool:
        while True:   # until every quota is full
            open_ = [b for b in buckets if want(b) > 0]
            if not open_:
                break
            rnd += 1
            jobs = []
            for b in open_:
                n = min(want(b) + 5, 2000)
                seed = hash((b, rnd)) % 10**9 + (10**9 if b[0] == 'test' else 0)   # test seeds never meet train seeds
                jobs.append((_deep_job, (b[1], b[2], n, seed)) if b[0] != 'kind' else (_kind_job, (b[1], n, seed)))
            for (b, (fn, args)), res in zip([(b, j) for b, j in zip(open_, jobs)],
                                            pool.starmap(_call, [(fn, args) for fn, args in jobs])):
                _, _, rows, tried, why = res
                stats['tried'] += tried
                for k, v in why.items():
                    stats['rejected'][k] = stats['rejected'].get(k, 0) + v
                for r in rows:
                    h = _key(r['text'])
                    if h in seen or want(b) <= 0:
                        continue
                    seen.add(h)
                    buckets[b].append(r)
            for b in open_:   # saved every round: an interrupted build (sleep, crash) resumes from here
                _save(_path(b, test_path), [r for r in buckets[b]] if b[0] != 'test' else
                      [r for (t, _, _), rows in buckets.items() if t == 'test' for r in rows])
            done = sum(len(v) for v in buckets.values())
            print(f'round {rnd}: {done} questions, {len(open_)} buckets were open, {time.time() - t0:.0f}s', flush=True)
            if rnd > 200:
                print('stopping: some buckets cannot be filled', [b for b in buckets if want(b) > 0][:10])
                break

    _save(test_path, [r for (t, _, _), rows in buckets.items() if t == 'test' for r in rows])
    for (t, f, d), rows in buckets.items():
        if t == 'deep':
            _save(f'{OUT}/deep/{f}/d{d:02d}.jsonl.gz', rows)
        elif t == 'kind':
            _save(f'{OUT}/kinds/{f}.jsonl.gz', rows)
    manifest = {
        'deep_train': {f: {d: len(buckets[('deep', f, d)]) for d in depths} for f in deep.FORMATS},
        'deep_test': {f: {d: len(buckets[('test', f, d)]) for d in depths} for f in deep.FORMATS},
        'kinds': {k: len(buckets[('kind', k, None)]) for k in kinds},
        'total': sum(len(v) for v in buckets.values()), 'tried': stats['tried'], 'rejected': stats['rejected'],
        'seconds': round(time.time() - t0)}
    import glob
    manifest['sha256'] = {os.path.relpath(f, OUT): hashlib.sha256(open(f, 'rb').read()).hexdigest()
                          for f in sorted(glob.glob(f'{OUT}/**/*.jsonl.gz', recursive=True))}
    json.dump(manifest, open(f'{OUT}/manifest.json', 'w'), indent=1)
    print(json.dumps({k: manifest[k] for k in ('total', 'tried', 'rejected', 'seconds')}))


def _path(b, test_path):
    t, f, d = b
    return test_path if t == 'test' else f'{OUT}/deep/{f}/d{d:02d}.jsonl.gz' if t == 'deep' else f'{OUT}/kinds/{f}.jsonl.gz'


def _call(fn, args):
    return fn(args)


def verify(root):
    """Every file matches the manifest's SHA-256 and the counts add up; exit 1 otherwise."""
    import sys
    m = json.load(open(f'{root}/manifest.json'))
    bad = [f for f, h in m.get('sha256', {}).items()
           if not os.path.exists(f'{root}/{f}') or hashlib.sha256(open(f'{root}/{f}', 'rb').read()).hexdigest() != h]
    n = sum(sum(v.values()) for v in m['deep_train'].values()) + sum(sum(v.values()) for v in m['deep_test'].values()) + sum(m['kinds'].values())
    print(f"{len(m.get('sha256', {}))} files, {n} questions (manifest total {m['total']}); mismatched files: {bad or 'none'}")
    if bad or n != m['total'] or not m.get('sha256'):
        sys.exit(1)


if __name__ == '__main__':
    import sys
    if '--verify' in sys.argv:
        verify(sys.argv[sys.argv.index('--root') + 1] if '--root' in sys.argv else OUT)
    else:
        main()

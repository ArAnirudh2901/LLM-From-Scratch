"""Locked test v2: whole sites held out, and every locked score recorded once.

v1 (eval.run_real.split_of) halves one question pool by hash, so each site's templates sit on both sides and settings
tuned on dev meet test questions written the same way (selected setting: dev 73.9% -> test 68.8% fair). v2 keeps only
new questions from sites with no question in the tuned-on dev half (new scrapes: eval.sites into data/fresh_online),
and drops any whose people-masked 3-grams nearly match a dev question (eval.leak). It stores question hashes only, no
text. `eval.run_real --split locked2` scores them and appends the result to the ledger; another locked score is
refused unless --rescore_reason says why (a new model chosen on dev; never a setting tuned on locked2).

    python -m eval.locked --freeze --pattern 'data/fresh_online/*.jsonl' [--min_sites 3]
    python -m eval.locked --status
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import sys

MANIFEST = 'data/locked_v2.json'
LEDGER = 'data/locked_v2_ledger.jsonl'


def hashes(path: str = MANIFEST) -> set:
    if not os.path.exists(path):
        return set()
    with open(path) as f:
        return set(json.load(f)['hashes'])


def freeze(new_rows, dev_rows, near: float | None = None):
    """(kept rows, counts): questions from sites dev never saw, far from every dev question."""
    from eval.leak import NEAR, grams, mask
    from eval.run_real import as_text
    from kinlogic.prepare import prepare
    near = NEAR if near is None else near
    gram = lambda r: grams(mask(prepare(as_text(r))['tokens']))
    dev_sites, dev_grams = {r.get('site') for r in dev_rows}, [gram(r) for r in dev_rows]
    kept, counts = [], {'dev site': 0, 'near a dev question': 0}
    for r in new_rows:
        if r.get('site') in dev_sites:
            counts['dev site'] += 1
            continue
        g = gram(r)
        if any(len(g & d) / max(1, len(g | d)) >= near for d in dev_grams):
            counts['near a dev question'] += 1
            continue
        kept.append(r)
    return kept, counts


def check_once(reason: str | None, limit=None, ledger: str = LEDGER):
    """Stops a second (or partial) locked score unless a reason is recorded with it."""
    if limit:
        sys.exit('locked2 is scored whole: drop --limit.')
    if os.path.exists(ledger) and open(ledger).read().strip() and not reason:
        sys.exit(f'locked2 was already scored ({ledger}). Score it again only for a new model selected on dev, '
                 'with --rescore_reason.')


def record(score: dict, ckpt: str, argv, reason: str | None, ledger: str = LEDGER):
    sha = None
    if os.path.exists(ckpt):
        h = hashlib.sha256()
        with open(ckpt, 'rb') as f:
            for chunk in iter(lambda: f.read(1 << 20), b''):
                h.update(chunk)
        sha = h.hexdigest()
    with open(ledger, 'a') as f:
        f.write(json.dumps({'date': datetime.date.today().isoformat(), 'checkpoint_sha256': sha, 'args': list(argv),
                            'reason': reason, 'splits': score.get('splits')}) + '\n')


def main():
    from eval.run_real import load_real, split_of
    from eval.verify_keys import qhash
    ap = argparse.ArgumentParser()
    ap.add_argument('--freeze', action='store_true')
    ap.add_argument('--status', action='store_true')
    ap.add_argument('--pattern', nargs='+', default=['data/fresh_online/*.jsonl'], help='new question files')
    ap.add_argument('--base', default='data/real_questions/*.jsonl', help='the v1 pool (its dev half was tuned on)')
    ap.add_argument('--min_sites', type=int, default=3)
    a = ap.parse_args()
    if a.status or not a.freeze:
        info = json.load(open(MANIFEST)) if os.path.exists(MANIFEST) else None
        print(json.dumps({k: v for k, v in (info or {}).items() if k != 'hashes'} or 'not frozen', indent=1))
        if os.path.exists(LEDGER):
            print(open(LEDGER).read().strip() or 'never scored')
        return
    if os.path.exists(MANIFEST):
        sys.exit(f'{MANIFEST} exists: a frozen set is never rebuilt (delete it only if it was never scored).')
    base = load_real(a.base)
    dev, old = [r for r in base if split_of(r) == 'dev'], {qhash(r) for r in base}
    new = [r for p in a.pattern for r in load_real(p) if qhash(r) not in old]
    kept, counts = freeze(new, dev)
    sites = sorted({r.get('site') for r in kept})
    if len(sites) < a.min_sites:
        sys.exit(f'{len(kept)} questions from {len(sites)} new sites ({counts}): scrape more sites (eval.sites) first.')
    with open(MANIFEST, 'w') as f:
        json.dump({'created': datetime.date.today().isoformat(), 'n': len(kept), 'sites': sites, 'dropped': counts,
                   'rule': 'new questions, sites absent from the v1 dev half, 3-gram Jaccard < eval.leak.NEAR to dev',
                   'hashes': sorted(qhash(r) for r in kept)}, f, indent=1)
    print(f'froze {len(kept)} questions from {len(sites)} sites -> {MANIFEST}; dropped {counts}')


if __name__ == '__main__':
    main()

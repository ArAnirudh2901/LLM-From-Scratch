"""Audits (and with --clean, cleans) a verified pool before training:
  - conflicts: one wording (names masked: the token sequence) with two different fact sets -> both dropped
    (the reader cannot learn two answers to one input; usually two templates writing one sentence differently);
  - repeats: the same wording and fact set again, in any order -> the first kept (pool before rewrites, so the pool's
    text-order convention wins; repeats overweight a template);
  - leakage: training wording that also occurs in the locked deep test or the frozen unseen set -> dropped;
  - broken: '<unk>' in the input, facts that do not parse, no answer, longer than the training block -> dropped.
Placeholders are renumbered by first appearance before comparing, so "@3 son @1" and "@0 son @1" with the same
wording count as one reading only when they mean the same people.

    python audit_dataset.py --root data/dataset_v3                 # report only
    python audit_dataset.py --root data/dataset_v3 --clean --out data/dataset_v4
"""
from __future__ import annotations

import argparse
import collections
import glob
import gzip
import hashlib
import json
import os
import re

from kinlogic.facts import parse


def canon(tokens, lf):
    """Wording and reading with placeholders renumbered by first appearance in the wording; the reading as a sorted
    set of facts inside each group (the same facts in another order are the same reading)."""
    m = {}
    for t in tokens:
        if re.fullmatch(r'@\d+', t) and t not in m:
            m[t] = f'@{len(m)}'
    sub = lambda x: m.get(x, x)
    groups, cur = [], []
    for f in ' '.join(sub(t) for t in lf.split()).split(' ; '):
        if re.match(r'(claim|part) \d+$', f):   # group markers keep their place
            groups.append(' ; '.join(sorted(cur)))
            groups.append(f)
            cur = []
        else:
            cur.append(f)
    groups.append(' ; '.join(sorted(cur)))
    return ' '.join(sub(t) for t in tokens), ' | '.join(groups)


def rows_of(path):
    with gzip.open(path, 'rt') as f:
        for line in f:
            yield json.loads(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='data/dataset_v3')
    ap.add_argument('--extra', default='data/reworded.jsonl.gz', help='training rewrites, audited with the pool')
    ap.add_argument('--block', type=int, default=320)
    ap.add_argument('--clean', action='store_true')
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    from reader import Vocab, build_vocab, lf_tokens
    files = sorted(glob.glob(f'{a.root}/**/*.jsonl.gz', recursive=True))
    train_files = [f for f in files if '/test/' not in f] + ([a.extra] if a.extra and os.path.exists(a.extra) else [])
    held = [f for f in files if '/test/' in f] + [f for f in ('data/unseen_test.jsonl.gz', 'data/reworded_dev.jsonl.gz') if os.path.exists(f)]
    held_words = {canon(r['tokens'], r['lf'])[0] for f in held for r in rows_of(f)}

    readings = collections.defaultdict(set)   # wording -> readings, over all training files
    for f in train_files:
        for r in rows_of(f):
            w, l = canon(r['tokens'], r['lf'])
            readings[w].add(l)
    conflict = {w for w, ls in readings.items() if len(ls) > 1}

    stats = collections.Counter()
    by_file = {}
    examples = collections.defaultdict(list)
    seen = set()
    for f in train_files:
        keep = []
        for r in rows_of(f):
            w, l = canon(r['tokens'], r['lf'])
            why = None
            if '<unk>' in r['tokens']:
                why = 'unk token'
            elif parse(r['lf'])[1]:
                why = 'facts do not parse'
            elif r.get('truth') is None and not re.search(r'\b(claim|part|count|pair|gender|who)\b', r['lf']):
                why = 'no answer'
            elif len(r['tokens']) + len(lf_tokens(r['lf'])) + 3 > a.block:
                why = 'too long'
            elif w in held_words:
                why = 'wording in a held-out set'
            elif w in conflict:
                why = 'conflicting readings'
            elif (w, l) in seen:
                why = 'repeat'
            if why:
                stats[why] += 1
                if len(examples[why]) < 2:
                    examples[why].append((os.path.relpath(f, a.root) if f.startswith(a.root) else f, r['text'][:160], r['lf'][:120]))
                continue
            seen.add((w, l))
            keep.append(r)
        stats['kept'] += len(keep)
        by_file[f] = keep
    print(json.dumps({'files': len(train_files), 'held-out wordings': len(held_words), 'wordings': len(readings),
                      'conflicting wordings': len(conflict), **stats}, indent=1))
    for why, ex in examples.items():
        for e in ex:
            print(f'  [{why}] {e[0]}: {e[1]} || {e[2]}')
    if a.clean:
        out = a.out or a.root + '_clean'
        for f, keep in by_file.items():
            if f == a.extra:
                dst = os.path.join(out, 'reworded.jsonl.gz')
            else:
                dst = os.path.join(out, os.path.relpath(f, a.root))
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            with gzip.open(dst + '.tmp', 'wt') as g:
                g.write(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in keep))
            os.replace(dst + '.tmp', dst)
        for f in files:   # the locked test is copied unchanged
            if '/test/' in f:
                dst = os.path.join(out, os.path.relpath(f, a.root))
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                with open(f, 'rb') as s, open(dst, 'wb') as d:
                    d.write(s.read())
        old = json.load(open(f'{a.root}/manifest.json'))
        man = {k: old[k] for k in ('deep_test',) if k in old}
        man['deep_train'] = {}
        man['kinds'] = {}
        for f, keep in by_file.items():
            rel = os.path.relpath(f, a.root)
            m = re.match(r'deep/(\w+)/d(\d+)\.jsonl\.gz', rel)
            if m:
                man['deep_train'].setdefault(m.group(1), {})[int(m.group(2))] = len(keep)
            elif rel.startswith('kinds/'):
                man['kinds'][rel[6:-9]] = len(keep)
        man['total'] = sum(sum(v.values()) for v in man['deep_train'].values()) + sum(man['kinds'].values()) + \
            sum(sum(v.values()) for v in man.get('deep_test', {}).values())
        man['cleaned'] = {'from': a.root, 'dropped': {k: v for k, v in stats.items() if k != 'kept'}}
        man['sha256'] = {os.path.relpath(f, out): hashlib.sha256(open(f, 'rb').read()).hexdigest()
                         for f in sorted(glob.glob(f'{out}/**/*.jsonl.gz', recursive=True)) if not f.endswith('reworded.jsonl.gz')}
        json.dump(man, open(f'{out}/manifest.json', 'w'), indent=1)
        print('written', out, man['total'])


if __name__ == '__main__':
    main()

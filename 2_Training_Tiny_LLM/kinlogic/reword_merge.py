"""Merges every rewording run into the training file and a held-out dev file: one row per source question (first
kept), dev chosen by source hash (no source on both sides; dev words never reach the vocabulary). Every row is
re-checked against its source question:
  - the engine still gives the source's answer;
  - the facts keep the source's structure (statement parts, member/generation counts, "has N", "only"): the reading
    schema of the rewriters has none of these, so sufficiency and puzzle rewrites lose them and would teach the reader
    to drop them;
  - every gender fact is implied by the relations, stated in the source, or cued in a sentence that names the person
    ("her mother Elizabeth"); a gender read from a name alone would teach the reader to guess genders from names.

    python -m kinlogic.reword_merge data/reworded_parts --src data/dataset_v2 --out data/reworded.jsonl.gz
"""
from __future__ import annotations

import argparse
import collections
import glob
import gzip
import json
import hashlib
import os
import re
from concurrent.futures import ProcessPoolExecutor

from .kin import KIN
from .prepare import prepare
from .reword import accept

GENDER = re.compile(r'([@#]\d+) (male|female)')
CUE = re.compile(r"\b(he|she|his|her|him|hers|himself|herself|man|woman|men|women|lady|ladies|gentleman|boy|girl|"
                 r"mr|mrs|ms|miss|male|female|sir|madam)\b", re.I)


def rows_of(path):
    files = sorted(glob.glob(os.path.join(path, '**', '*.jsonl.gz'), recursive=True)) if os.path.isdir(path) else [path]
    for f in files:
        try:
            for line in gzip.open(f, 'rt'):
                yield json.loads(line)
        except (EOFError, OSError, ValueError) as e:   # a truncated tail loses only its last block
            print(f'{f}: stopped at {type(e).__name__}')


def structure(lf):
    """Fact-language words other than relations, genders and people: parts, counts, 'only', 'members' ..."""
    return collections.Counter(t for t in lf.split() if t[0] not in '@#' and t != ';' and t not in KIN and t not in ('male', 'female'))


def _names(text):
    return {e['id']: e['text'].lower() for e in prepare(text)['ents']}


def genders_supported(r, src):
    from eval.gender_set import gender_of
    fs = [f.strip() for f in r['lf'].split(';')]
    claimed = [tuple(f.split()) for f in fs if GENDER.fullmatch(f)]
    if not claimed:
        return True
    implied = gender_of(' ; '.join(f for f in fs if not GENDER.fullmatch(f)), sorted({t for t in r['tokens'] if t[0] == '@'})) or {}
    rest = [(p, g) for p, g in claimed if str(implied.get(p, ''))[:1].upper() != g[0].upper()]
    if not rest:
        return True
    names, src_names = _names(r['text']), _names(src['text'])
    stated = {(src_names.get(p), g) for p, g in GENDER.findall(src['lf'])}
    sentences = re.split(r'(?<=[.?!])\s+', r['text'])
    for p, g in rest:
        n = names.get(p)
        if not n or ((n, g) not in stated and not any(n in s.lower() and CUE.search(s) for s in sentences)):
            return False
    return True


def check(job):
    """'kept' or why the row is dropped."""
    r, src = job
    try:
        if not (r.get('text') and r.get('tokens') and r.get('lf')) or not accept(r['lf'], r['truth'], r['tokens']):
            return 'engine answer'
        if src is None:
            return 'source not found'
        if structure(r['lf']) != structure(src['lf']):
            return 'structure lost'
        return 'kept' if genders_supported(r, src) else 'gender from name'
    except Exception as e:
        return 'error ' + type(e).__name__


def sources(src, need):
    """{sha1 of text: row} for the needed source questions (never the locked test)."""
    out = {}
    for f in glob.glob(os.path.join(src, '**', '*.jsonl.gz'), recursive=True):
        if '/test/' in f:
            continue
        for line in gzip.open(f, 'rt'):
            r = json.loads(line)
            h = hashlib.sha1(r['text'].encode()).hexdigest()
            if h in need:
                out[h] = r
    return out


def is_dev(source: str, every: int) -> bool:
    return int(source[:8], 16) % every == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('inputs', nargs='+', help='part folders or .jsonl.gz files')
    ap.add_argument('--src', default='data/dataset_v2', help='the verified pool the rewrites came from')
    ap.add_argument('--out', default='data/reworded.jsonl.gz')
    ap.add_argument('--dev_out', default='data/reworded_dev.jsonl.gz')
    ap.add_argument('--dev_every', type=int, default=25, help='1 in N sources held out (4%%)')
    a = ap.parse_args()
    seen, rows, dup = set(), [], 0
    for path in a.inputs:
        for r in rows_of(path):
            if r['source'] in seen:
                dup += 1
                continue
            seen.add(r['source'])
            rows.append(r)
    src = sources(a.src, seen)
    with ProcessPoolExecutor() as pool:
        why = list(pool.map(check, [(r, src.get(r['source'])) for r in rows], chunksize=64))
    kept = [r for r, w in zip(rows, why) if w == 'kept']
    dropped = collections.Counter(w for w in why if w != 'kept')
    by_kind = collections.Counter((r['kind'].removeprefix('reworded-'), w) for r, w in zip(rows, why) if w != 'kept')
    dev = [r for r in kept if is_dev(r['source'], a.dev_every)]
    train = [r for r in kept if not is_dev(r['source'], a.dev_every)]
    for path, part in ((a.out, train), (a.dev_out, dev)):
        tmp = path + '.tmp'
        with gzip.open(tmp, 'wt') as f:
            f.write(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in part))
        os.replace(tmp, path)
    kinds = collections.Counter(r['kind'].removeprefix('reworded-') for r in kept)
    depth = collections.Counter(r.get('depth') or 0 for r in kept)
    print(json.dumps({'read': len(rows) + dup, 'duplicate sources': dup, 'dropped': dict(dropped), 'dropped by kind': {f'{k} {w}': n for (k, w), n in by_kind.most_common(12)},
                      'train': len(train), 'dev': len(dev), 'kinds': dict(kinds.most_common()),
                      'depths': dict(sorted(depth.items())),
                      'max tokens': max(len(r['tokens']) for r in kept)}, indent=1))


if __name__ == '__main__':
    main()

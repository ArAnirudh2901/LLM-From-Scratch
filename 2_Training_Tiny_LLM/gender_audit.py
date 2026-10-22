"""Gender facts the reader cannot learn: a fact "@k male|female" whose person is not given that gender by the relations
and appears in no sentence with a gender word (he, his, she, her, man, woman, Mr, Mrs ...). The reader sees
placeholders, not names, so such a fact can only come from the name: noise that teaches it to invent genders.

    python gender_audit.py --root data/dataset_v3          # counts by kind, examples
"""
from __future__ import annotations

import argparse
import collections
import glob
import gzip
import json
import re
from concurrent.futures import ProcessPoolExecutor

CUE = {'he', 'his', 'him', 'himself', 'she', 'her', 'hers', 'herself', 'man', 'woman', 'men', 'women', 'boy', 'girl', 'lady',
       'ladies', 'gentleman', 'mr', 'mrs', 'ms', 'miss', 'male', 'female', 'sir', 'madam', 'lad', 'brother', 'sister'}
GENDER = re.compile(r'([@#]\d+) (male|female)')
SENT = {'.', '?', '!', ';'}


def unsupported(row) -> list[str]:
    facts = [f.strip() for f in row['lf'].split(';')]
    claimed = [f for f in facts if GENDER.fullmatch(f)]
    if not claimed:
        return []
    cued, sent = set(), []
    for t in row['tokens'] + ['.']:
        if t in SENT:
            if CUE & set(sent):
                cued |= {x for x in sent if x[0] == '@'}
            sent = []
        else:
            sent.append(t)
    rest = [f for f in claimed if f.split()[0] not in cued]
    if not rest:
        return []
    from eval.gender_set import gender_of
    others = ' ; '.join(f for f in facts if not GENDER.fullmatch(f))
    implied = gender_of(others, sorted({t for t in row['tokens'] if t[0] == '@'})) or {}
    return [f for f in rest if str(implied.get(f.split()[0], ''))[:1].upper() != f.split()[1][0].upper()]


def _check(line):
    r = json.loads(line)
    return r['kind'], bool(unsupported(r)), (r['text'][:150], r['lf'][:110])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='data/dataset_v3')
    a = ap.parse_args()
    files = [f for f in sorted(glob.glob(f'{a.root}/**/*.jsonl.gz', recursive=True))]
    tot, bad, eg = collections.Counter(), collections.Counter(), {}
    with ProcessPoolExecutor() as pool:
        for f in files:
            for kind, b, ex in pool.map(_check, gzip.open(f, 'rt'), chunksize=512):
                tot[kind] += 1
                if b:
                    bad[kind] += 1
                    eg.setdefault(kind, ex)
    print(json.dumps({k: f'{bad[k]}/{tot[k]}' for k in sorted(tot, key=lambda k: -bad[k]) if bad[k]}, indent=1))
    for k, (t, l) in eg.items():
        print(f'  [{k}] {t} || {l}')


if __name__ == '__main__':
    main()

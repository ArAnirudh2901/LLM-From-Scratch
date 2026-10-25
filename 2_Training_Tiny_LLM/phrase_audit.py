"""Phrase-type coverage: how often each way of saying a family fact appears in our training text versus real exam
questions. A type that is common in exams but rare in training is where the reader memorized the generator's style
instead of learning the language (r6a: who/which-clause questions 52% vs 73% for others). boost_phrases.py uses the
gaps to upsample those types with engine-verified generated questions.

    python phrase_audit.py --train 'data/dataset_v6/kinds/*.jsonl.gz' --n 60000
"""
from __future__ import annotations

import argparse
import collections
import glob
import gzip
import json
import random
import re

E = r'@\d+'
TYPES = {   # on prepared token text (people are @k)
    'X is the R of Y': rf'{E} is (?:the |a |an )?(?:only )?[a-z-]+ of {E}',
    "X 's R is Y": rf"{E} 's [a-z-]+ is {E}",
    "possessive chain x2 (X 's R 's S)": rf"'s [a-z-]+ 's [a-z-]+",
    'possessive chain x3+': rf"'s [a-z-]+ 's [a-z-]+ 's [a-z-]+",
    'of-chain (R of the S of)': r'\b[a-z-]+ of (?:the |my |his |her )?(?:only )?[a-z-]+ of\b',
    ', who is/was/has (relative clause)': r', who (?:is|was|has|had)\b',
    'who is (no comma)': rf'{E} who (?:is|was|has)\b',
    ', whose R is': r', whose [a-z-]+ (?:is|was)\b',
    'whose (any)': r'\bwhose\b',
    ', which/that': r', (?:which|that) (?:is|was)\b',
    'apposition (X , the R of Y ,)': rf"{E} , (?:the |a )?[a-z-]+ of {E} ,",
    'pointing / photo / picture': r'\b(?:pointing|photograph|photo|picture|portrait)\b',
    'introducing / introduced': r'\bintroduc',
    'said / says / told / tells': r'\b(?:said|says|told|tells|asked|replied)\b',
    'first person (my / i / me)': r'\b(?:my|i|me|mine)\b',
    'pronouns (he / she / his / her)': r'\b(?:he|she|his|her|him)\b',
    'only (only son / only child)': r'\bonly\b',
    'negation (not / no)': r"\b(?:not|no|n't|neither|nor)\b",
    'count (has two sons)': r'\bhas (?:two|three|four|five|\d+) [a-z-]+',
    'in-law': r'-in-law\b',
    'married / husband / wife': r'\b(?:married|husband|wife|spouse|couple)\b',
    'paternal / maternal': r'\b(?:paternal|maternal)\b',
    'generation': r'\bgenerations?\b',
    'members / family of N': r'\b(?:members?|persons|people)\b',
    'profession': r'\b(?:doctor|lawyer|engineer|teacher|professor|architect|accountant|salesman|manager|clerk|pilot|nurse|businessman|contractor|housewife|farmer|student)\b',
    'code symbols (means)': r'\b(?:means|denotes|implies|represents|indicates)\b',
    'which of the following true': r'which of the following .*?(?:true|correct|false)',
    'cannot be determined / data': r'(?:cannot be|can not be|data (?:is )?(?:in)?adequate|insufficient)',
    'question first then facts': r'^(?:how|who|what) [^?]*\? ',
    'only daughter / son of my': r'only (?:son|daughter|child|brother|sister) of (?:my|his|her)',
    'grand-/great-': r'\b(?:grand|great)',
    'cousin / nephew / niece': r'\b(?:cousin|nephew|niece)\b',
    'brother-/sister- (siblings)': r'\b(?:siblings?|brothers|sisters)\b',
}
PAT = {k: re.compile(v) for k, v in TYPES.items()}


def shares(texts):
    c = collections.Counter()
    for t in texts:
        for k, p in PAT.items():
            if p.search(t):
                c[k] += 1
    n = max(1, len(texts))
    return {k: c[k] / n for k in TYPES}


def train_texts(patterns, n, seed=0):
    rows = []
    for pat in patterns:
        for f in glob.glob(pat):
            try:
                for i, l in enumerate(gzip.open(f, 'rt')):
                    r = json.loads(l)
                    if isinstance(r.get('tokens'), list):
                        rows.append(' '.join(r['tokens']))
                    if i > n:
                        break
            except (EOFError, OSError):
                pass
    random.Random(seed).shuffle(rows)
    return rows[:n]


def real_texts():
    from eval.real_clean import broken
    from eval.run_real import as_text, load_real
    from kinlogic.prepare import prepare
    return [' '.join(prepare(as_text(r))['tokens']) for r in load_real() if not broken(r)]   # dev + test, counts only


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train', nargs='+', default=['data/dataset_v6/kinds/*.jsonl.gz', 'data/dataset_v6/deep/*.jsonl.gz', 'data/*.jsonl.gz'])
    ap.add_argument('--n', type=int, default=60000)
    ap.add_argument('--out', default='data/phrase_audit.json')
    a = ap.parse_args()
    tr, rl = shares(train_texts(a.train, a.n)), shares(real_texts())
    rep = sorted(((k, rl[k], tr[k], rl[k] / max(tr[k], 1e-4)) for k in TYPES), key=lambda x: -x[3])
    print(f"{'phrase type':40s} {'real':>6s} {'train':>6s} {'real/train':>10s}")
    for k, r, t, ratio in rep:
        print(f'{k:40s} {r:6.1%} {t:6.1%} {ratio:10.1f}' + ('   <- under-represented' if ratio > 1.5 and r > 0.02 else ''))
    json.dump({k: {'real': r, 'train': t} for k, r, t, _ in rep}, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()

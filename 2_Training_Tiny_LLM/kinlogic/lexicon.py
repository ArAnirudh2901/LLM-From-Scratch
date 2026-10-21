"""Builds kinlogic/data/lexicon.txt: ordinary words (never names) from generated questions and CLUTRR stories.
prepare() treats a capitalized sentence-initial word as a name only if it is not in this list.

    python -m kinlogic.lexicon [n_questions]
"""
from __future__ import annotations

import glob
import random
import re
import sys
from pathlib import Path

from . import generate as gen
from .generate import PLACES as _P
PLACES = {p.lower() for p in _P}
from .names import INDIAN_F, INDIAN_M, SURNAMES, WORLD_F, WORLD_M
from .prepare import AMBIGUOUS_NAMES, TOKEN_RE

OUT = Path(__file__).parent / 'data' / 'lexicon.txt'
NAMES = {n.lower() for n in INDIAN_M + INDIAN_F + WORLD_M + WORLD_F + SURNAMES} - AMBIGUOUS_NAMES


def words_from_generator(n: int, seed: int = 0) -> set[str]:
    rng = random.Random(seed)
    out = set()
    kinds = [k for k in KINDS]
    made = 0
    while made < n:
        ex = KINDS[rng.choice(kinds)](rng)
        if ex is None:
            continue
        made += 1
        for t in TOKEN_RE.findall(ex['text']):   # sentence-initial words too; names and places are dropped later
            if len(t) > 1 and t.lower() not in PLACES:
                out.add(t.lower())
    return out


def words_from_clutrr(data_dir='data/clutrr_v1') -> set[str]:
    import pandas as pd
    out = set()
    for f in glob.glob(f'{data_dir}/*train.parquet'):
        for s in pd.read_parquet(f).story:
            for t in TOKEN_RE.findall(re.sub(r'\[[^\]]+\]', ' ', s)):
                if t[0].islower():   # names never appear lowercase
                    out.add(t)
    return out


KINDS = {k[2:]: getattr(gen, k) for k in dir(gen) if k.startswith('q_')}

if __name__ == '__main__':
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
    words = words_from_generator(n) | words_from_clutrr()
    words = {w for w in words if re.fullmatch(r"[a-z][a-z'-]*", w) and w not in NAMES}
    words |= {'i', 'ii', 'iii', 'iv', 'v', 'vi'}
    OUT.write_text('\n'.join(sorted(words)) + '\n')
    print(len(words), 'words ->', OUT)

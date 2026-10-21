"""CLUTRR stories as reader-model examples: the story plus a question, and its facts in text order.

A fact "b is a's REL" comes from each story edge; it is placed at the first sentence that names both people.
Gender facts come from pronouns (his/her...) referring to the nearest earlier name of that gender.

    python -m kinlogic.clutrr  ->  data/clutrr_facts/{train,validation,test}.jsonl.gz
"""
from __future__ import annotations

import ast
import gzip
import json
import random
import re
from pathlib import Path

from .facts import format_stmt
from .generate import ask_text
from .prepare import prepare

PRONOUNS = {'he': 'M', 'his': 'M', 'him': 'M', 'himself': 'M', 'she': 'F', 'her': 'F', 'hers': 'F', 'herself': 'F'}


def convert(row, rng: random.Random) -> dict | None:
    names = [x.split(':')[0] for x in row.genders.split(',')]
    gender = {n: ('M' if x.split(':')[1] == 'male' else 'F') for n, x in zip(names, row.genders.split(','))}
    edges = ast.literal_eval(row.story_edges) if isinstance(row.story_edges, str) else list(row.story_edges)
    types = ast.literal_eval(row.edge_types) if isinstance(row.edge_types, str) else list(row.edge_types)
    qa, qb = ast.literal_eval(row.query_edge) if isinstance(row.query_edge, str) else tuple(row.query_edge)
    sentences = re.split(r'(?<=[.!?])\s+', row.story.strip())
    # mention positions per sentence
    mentions = [[(m.start(), m.group(1)) for m in re.finditer(r'\[([^\]]+)\]', s)] for s in sentences]
    facts = []   # (sentence, position, stmt)
    for (a, b), r in zip(edges, types):
        na, nb = names[a], names[b]
        where = None
        for i, ms in enumerate(mentions):
            got = {n for _, n in ms}
            if na in got and nb in got:
                pos = max(p for p, n in ms if n in (na, nb))
                where = (i, pos)
                break
        if where is None:
            first = {}
            for i, ms in enumerate(mentions):
                for p, n in ms:
                    first.setdefault(n, (i, p))
            if na not in first or nb not in first:
                return None
            where = max(first[na], first[nb])
        facts.append((*where, {'t': 'rel', 'x': na, 'r': r, 'y': nb, 'side': None, 'only': False, 'neg': False}))
    # gender evidence from pronouns: nearest earlier name of that gender (same or previous sentence)
    seen, last = set(), []
    for i, s in enumerate(sentences):
        cur = []
        for m in re.finditer(r'\[([^\]]+)\]|\b([A-Za-z]+)\b', s):
            if m.group(1):
                cur.append(m.group(1))
                continue
            g = PRONOUNS.get(m.group(2).lower())
            if not g:
                continue
            cands = [n for n in reversed(cur) if gender.get(n) == g] or [n for n in reversed(last) if gender.get(n) == g]
            if cands and cands[0] not in seen:
                seen.add(cands[0])
                facts.append((i, -1, {'t': 'gender', 'x': cands[0], 'g': g}))
        last = cur or last
    facts.sort(key=lambda f: (f[0], f[1]))
    a, b = names[qa], names[qb]
    text = re.sub(r'\[([^\]]+)\]', r'\1', row.story.strip()) + ' ' + ask_text(rng, a, b)
    p = prepare(text, names=names)
    ids = {e['text'].lower(): e['id'] for e in p['ents']}
    try:
        out = [{**f, 'x': ids[f['x'].lower()], **({'y': ids[f['y'].lower()]} if 'y' in f else {})} for _, _, f in facts]
        out.append({'t': 'ask', 'x': ids[a.lower()], 'y': ids[b.lower()]})
    except KeyError:
        return None
    return {'kind': 'clutrr', 'tokens': p['tokens'], 'lf': ' ; '.join(format_stmt(s) for s in out), 'text': text,
            'answer': row.target_text, 'k': len(edges)}


def main(data_dir='data/clutrr_v1', out_dir='data/clutrr_facts', config='gen_train234_test2to10'):
    import pandas as pd
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    rng = random.Random(0)
    for split in ('train', 'validation', 'test'):
        df = pd.read_parquet(f'{data_dir}/{config}__{split}.parquet')
        made = dropped = 0
        with gzip.open(f'{out_dir}/{split}.jsonl.gz', 'wt') as f:
            for row in df.itertuples():
                ex = convert(row, rng)
                if ex is None:
                    dropped += 1
                    continue
                made += 1
                f.write(json.dumps(ex) + '\n')
        print(f'{split}: {made} examples, {dropped} dropped')


if __name__ == '__main__':
    main()

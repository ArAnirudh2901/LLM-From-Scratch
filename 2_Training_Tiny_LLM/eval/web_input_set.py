"""Fixed question set for the website's v10 input comparison (Kinthread tools/input-compare): generated relation
questions with known answers, never the private real dev/test questions. 12 each of statements, nested, story and
detour (obfuscated) questions about two named people, plus 4 each of speech, identity and introduce questions about a
described person ("a woman"). a/b are the picker's people (b's relation to a, the text as written); gold_option is the
option whose text is the truth.

    python -m eval.web_input_set ../kinthread/tools/input-compare/questions.json

The generator iterates sets, so the rows only repeat under one hash seed: this script re-runs itself with
PYTHONHASHSEED=0.
"""
from __future__ import annotations

import json
import os
import random
import re
import sys

PLAN = [('statements', 12), ('nested', 12), ('story', 12), ('obfuscated', 12), ('speech', 4), ('identity', 4), ('introduce', 4)]
NAMED = {'statements', 'nested', 'story', 'obfuscated'}
SEED = 20261028


def build():
    from eval.consistency import agrees
    from kinlogic.generate import KINDS
    from kinlogic.prepare import prepare
    rng, rows, seen = random.Random(SEED), [], set()
    for kind, n in PLAN:
        got = 0
        while got < n:
            ex = KINDS[kind](random.Random(rng.getrandbits(64)))
            if not ex or not ex.get('truth') or ex['text'] in seen:
                continue
            ask = re.search(r'\bask (\S+) (\S+)', ex['lf'])
            if not ask:
                continue
            x, y = ask.groups()
            p = prepare(ex['text'])
            ents, names = {e['id']: e['text'] for e in p['ents']}, ex.get('names', {})
            if kind in NAMED:
                if x not in names or y not in names:   # the picker needs both people named
                    continue
                lower = {t.lower(): t for t in ents.values()}   # as written ("nina"): what the picker offers
                a, b = lower.get(names[x].lower(), names[x]), lower.get(names[y].lower(), names[y])
            elif x in ents and y in ents:
                a, b = ents[x], ents[y]
            else:
                continue
            gold = [lab for lab, o in p['options'] if agrees(o, ex['truth'])]
            seen.add(ex['text'])
            got += 1
            rows.append({'id': f'{kind}-{got}', 'kind': kind, 'text': ex['text'], 'a': a, 'b': b, 'truth': ex['truth'],
                         'options': bool(p['options']), 'gold_option': gold[0] if len(gold) == 1 else None, 'lf': ex['lf']})
    return rows


def main():
    if os.environ.get('PYTHONHASHSEED') != '0':
        os.execvpe(sys.executable, [sys.executable, '-m', 'eval.web_input_set', *sys.argv[1:]], {**os.environ, 'PYTHONHASHSEED': '0'})
    out = sys.argv[1] if len(sys.argv) > 1 else 'web_input_set.json'
    rows = build()
    with open(out, 'w') as f:
        json.dump(rows, f, indent=1)
        f.write('\n')
    print(f'{len(rows)} rows ({sum(r["options"] for r in rows)} with options) -> {out}')


if __name__ == '__main__':
    main()

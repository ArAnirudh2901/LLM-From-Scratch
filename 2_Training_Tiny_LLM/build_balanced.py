"""A balanced synthetic supplement: every question format (eval.balance_report.fmt) filled to the same count, with
answers and fact depths spread inside each format. Rows come from the generator's exact facts (plus --extra files,
e.g. the balanced long-story candidate) and are kept only when the engine reads their facts to the stored answer and
the reader can write them (at most 32 people, within the output budget).

    python build_balanced.py --per_format 3000 --seed 11 --out data/balanced_mix.jsonl.gz \\
        --extra data/balanced_question_formats_v1.jsonl.gz
"""
from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import json
import math
import random
import sys

from eval.balance_report import bucket, depth, fmt, profile, rows_of, show
from kinlogic.engine import solve
from kinlogic.facts import compact_people, parse
from kinlogic.generate import KINDS

FORMATS = ('statements', 'possessive chain', 'nested clauses', 'direct speech', 'indirect speech', 'long story',
           'coded', 'family puzzle', 'data sufficiency')
ASK_KINDS = ('statements', 'nested', 'speech', 'identity', 'introduce', 'story', 'deep', 'obfuscated')   # truth = engine key
QSTART = ('ask', 'who', 'count', 'gender', 'pair', 'prof')


def checked(ex, max_people=32, max_fact=190, max_input=300) -> bool:
    """Facts the reader can write whose engine answer is the stored truth (only one answer)."""
    lf, toks = ex.get('lf') or '', ex.get('tokens') or []
    if not lf or len(lf.split()) + 1 > max_fact or len(toks) > max_input or compact_people(toks, lf, max_people) is None:
        return False
    try:
        res = solve(parse(lf)[0])
    except Exception:
        return False
    if not res.get('ok'):
        return False
    if ex.get('kind') in ASK_KINDS and ex.get('truth'):
        q = (res.get('questions') or [{}])[-1]
        if ex['truth'] == 'none':   # two families in one story (the engine finds no link: status 'none')
            return q.get('status') in ('none', 'unrelated')
        return q.get('status') == 'unique' and (q.get('answers') or [{}])[0].get('key') == ex['truth']
    return True


def long_story(rng):
    """80-240 word stories (generate_balanced_question_types); the generator's own kinds stay under 80 words."""
    from generate_balanced_question_types import ANSWER_TYPES, STORY_LENGTHS, connected_long_story, disconnected_long_story
    spec = STORY_LENGTHS[rng.choice(list(STORY_LENGTHS))]
    want, (lo, hi) = rng.choice(ANSWER_TYPES), spec['words']
    ex = (disconnected_long_story(rng, lo, spec['per_component']) if want == 'no_link'
          else connected_long_story(rng, want, lo, spec['clues']))
    return ex if ex and len(ex['text'].split()) <= hi else None


SOURCES = {**KINDS, 'long_story': long_story}


def answer_of(ex) -> str:
    if ex.get('truth') or ex.get('answer'):
        return str(ex.get('truth') or ex.get('answer'))
    q = [s.split()[0] for s in (ex.get('lf') or '').split(';') if s.split() and s.split()[0] in QSTART]
    return q[-1] if q else 'none'


def build(per_format: int, seed: int, extra=(), patience: int = 20000):
    rng = random.Random(seed)
    caps = [math.ceil(per_format / 12), math.ceil(per_format / 3), math.ceil(per_format / 2),   # per answer, depth, kind,
            math.ceil(per_format / 3)]                                                         # question type
    rows, seen, stats = [], set(), collections.Counter()
    n_fmt, n_ans, n_dep, n_kind, n_q = (collections.Counter() for _ in range(5))

    def offer(ex, src) -> bool:
        f = fmt(ex)
        a, dp, kd = (f, answer_of(ex)), (f, bucket(depth(ex.get('lf') or ''), (1, 2, 3, 4, 6, 8, 12))), (f, src.split('-')[-1])
        qs = [s.split()[0] for s in (ex.get('lf') or '').split(';') if s.split() and s.split()[0] in QSTART]
        qt, key = (f, qs[-1] if qs else 'none'), ' '.join(ex.get('tokens') or [])
        if f not in FORMATS or n_fmt[f] >= per_format or key in seen or n_ans[a] >= caps[0] or n_dep[dp] >= caps[1] \
                or n_kind[kd] >= caps[2] or n_q[qt] >= caps[3]:
            stats['skipped (cell full or duplicate)'] += 1
            return False
        if not checked(ex):
            stats[f'failed check: {src}'] += 1
            return False
        seen.add(key)
        n_fmt[f] += 1; n_ans[a] += 1; n_dep[dp] += 1; n_kind[kd] += 1; n_q[qt] += 1
        rows.append({**{k: ex[k] for k in ('kind', 'text', 'tokens', 'lf', 'truth') if k in ex}, 'format': f, 'source': src})
        return True

    kinds = list(SOURCES)
    useful = dict.fromkeys(kinds, 1.0)   # moving share of a kind's rows that still find room
    for phase in range(2):   # then caps x3: some formats have few answers or depths (long stories, sufficiency)
        for path in extra:
            for ex in rows_of(path):
                offer(ex, ex.get('kind') or path.rsplit('/', 1)[-1])
        attempts = last = 0
        while attempts - last < patience and any(n_fmt[f] < per_format for f in FORMATS):
            attempts += 1
            k = rng.choices(kinds, [useful[x] for x in kinds])[0]
            ex = SOURCES[k](random.Random(rng.getrandbits(64)))
            got = ex is not None and offer(ex, k)
            useful[k] = 0.98 * useful[k] + 0.02 * (2.0 if got else 0.2)
            if got:
                last = attempts
        caps = [3 * c for c in caps]
    rng.shuffle(rows)
    short = {f: per_format - n_fmt[f] for f in FORMATS if n_fmt[f] < per_format}
    return rows, stats, short


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--per_format', type=int, default=3000)
    ap.add_argument('--seed', type=int, default=11)
    ap.add_argument('--out', required=True)
    ap.add_argument('--extra', nargs='*', default=[], help='jsonl(.gz) rows offered first (same checks and caps)')
    a = ap.parse_args()
    rows, stats, short = build(a.per_format, a.seed, a.extra)
    with gzip.open(a.out, 'wt') as f:
        f.writelines(json.dumps(r) + '\n' for r in rows)
    prof = profile(rows)
    with open(a.out, 'rb') as f:
        sha = hashlib.sha256(f.read()).hexdigest()
    manifest = {'rows': len(rows), 'sha256': sha, 'command': ' '.join(['python', *sys.argv]), 'short': short,
                'rejected': dict(stats.most_common()), 'note': 'candidate: not trained on; exact generator facts',
                **{k: dict(prof[k].most_common()) for k in ('format', 'question', 'facts', 'people', 'input tokens',
                                                            'fact tokens', 'options', 'kind')}}
    with open(a.out.replace('.jsonl.gz', '.manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=1)
    print(f'{len(rows)} rows -> {a.out} (sha256 {sha}); short of the target: {short or "none"}')
    print(dict(stats.most_common()))
    print(show(prof, len(rows), 1 / (2 * len(FORMATS))))


if __name__ == '__main__':
    main()

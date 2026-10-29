"""Re-score an `eval.run_real --trace` file under answer-time settings that need no model (--closure, --core_repair,
--lever, --trace_pow): every question is answered again from the readings the traced run voted on (its last pass, so
the adaptive re-read when there was one). One Kaggle trace thus prices every combination on any CPU in minutes.

Para votes and sub-reads (option expressions, drafts) are not replayed, so compare each setting with the replayed
baseline, not with the traced score, and check the reproduction line first.

    python -m eval.rescore_trace runs/v10-r8p-final/trace_dev.jsonl --verifier runs/v10-r8p-final/verifier.json --cover
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import os

SETTINGS = {
    'base': {},
    'closure': {'CLOSURE': True},
    'core_repair': {'CORE_REPAIR': True},
    'lever': {'LEVER': True},
    'trace_pow1': {'TRACE_POW': 1.0},
    'closure+core': {'CLOSURE': True, 'CORE_REPAIR': True},
    'all': {'CLOSURE': True, 'CORE_REPAIR': True, 'LEVER': True, 'TRACE_POW': 1.0},
}
FLAGS = ('CLOSURE', 'CORE_REPAIR', 'LEVER', 'TRACE_POW')
CLI = {'closure': '--closure', 'core_repair': '--core_repair', 'lever': '--lever', 'trace_pow1': '--trace_pow 1',
       'closure+core': '--closure --core_repair', 'all': '--closure --core_repair --lever --trace_pow 1'}


def replay_reader(rec):
    """read() returning the readings the traced run voted on (exact routes are read again by the exact readers)."""
    main = [pa for pa in rec['passes'] if pa['original']]
    cands = [(c['facts'].split('  [')[0], c['score']) for c in main[-1]['cands']] if main else []
    return lambda batch: [list(cands) for _ in batch]


def row_of(rec):
    return {'question': rec['text'], 'options': rec.get('options') or [], 'answer': rec.get('answer'),
            'answer_text': rec.get('answer_text')}


def mcnemar(b: int, c: int) -> float:
    """Exact two-sided p-value for b wins against c losses."""
    n = b + c
    if not n:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def rescore(recs, settings=SETTINGS):
    import kinlogic.exam as E
    from eval.run_real import correct
    saved = {f: getattr(E, f) for f in FLAGS}
    ok, same = {}, 0
    try:
        for name, flags in settings.items():
            for f in FLAGS:
                setattr(E, f, flags.get(f, 0.0 if f == 'TRACE_POW' else False))
            got = []
            for rec in recs:
                res = E.answer(rec['text'], replay_reader(rec))
                got.append(bool(correct(row_of(rec), res)))
                if name == 'base':
                    same += (res.get('option') or res.get('answer')) == (rec['final'].get('option') or rec['final'].get('answer'))
            ok[name] = got
    finally:
        for f, v in saved.items():
            setattr(E, f, v)
    return ok, same


def report(recs, ok, same) -> str:
    n = len(recs)
    traced = sum(r['ok'] for r in recs)
    lines = [f'reproduction: replayed base gives the traced answer on {same}/{n}; traced {traced}/{n}, '
             f'replayed base {sum(ok["base"])}/{n}']
    cats = collections.Counter(r.get('category') or '?' for r in recs)
    for name, got in ok.items():
        b = sum(g and not o for g, o in zip(got, ok['base']))
        c = sum(o and not g for g, o in zip(got, ok['base']))
        per = ' | '.join(f'{k} {sum(g for g, r in zip(got, recs) if (r.get("category") or "?") == k)}/{v}'
                         for k, v in cats.most_common())
        lines.append(f'{name:13s} {sum(got)}/{n} = {sum(got) / max(1, n):.1%} | vs base +{b} -{c} '
                     f'(McNemar p={mcnemar(b, c):.3f}) | {per}')
    return '\n'.join(lines)


def decide(ok, alpha: float = 0.05) -> dict:
    """Per setting: correct, wins and losses against base, McNemar p, and adopt (more wins than losses, p < alpha);
    'adopted_cli' is the run_real flags of the best adopted setting ('' when none wins)."""
    out = {'questions': len(ok['base']), 'settings': {}}
    for name, got in ok.items():
        b = sum(g and not o for g, o in zip(got, ok['base']))
        c = sum(o and not g for g, o in zip(got, ok['base']))
        p = mcnemar(b, c)
        out['settings'][name] = {'correct': sum(got), 'wins': b, 'losses': c, 'p': round(p, 4),
                                 'adopt': name != 'base' and b > c and p < alpha}
    won = [n for n, s in out['settings'].items() if s['adopt']]
    best = max(won, key=lambda n: out['settings'][n]['correct'], default=None)
    out['adopted'], out['adopted_cli'] = best, CLI.get(best, '')
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('trace')
    ap.add_argument('--verifier', default=None, help='the verifier weights the traced run used')
    ap.add_argument('--cover', action='store_true', help='the traced run used --cover')
    ap.add_argument('--all', action='store_true', help='include broken (unfair) questions')
    ap.add_argument('--out', default=None, help='JSON with every setting and the flags to adopt (no question text)')
    a = ap.parse_args()
    if a.verifier:
        os.environ['KIN_VERIFIER'] = a.verifier
    import kinlogic.exam as E
    E.COVER = E.COVER or a.cover
    recs = [json.loads(line) for line in open(a.trace) if line.strip()]
    recs = [r for r in recs if a.all or r.get('fair', True)]
    ok, same = rescore(recs)
    print(report(recs, ok, same))
    if a.out:
        with open(a.out, 'w') as f:
            json.dump({**decide(ok), 'reproduced': same}, f, indent=1)


if __name__ == '__main__':
    main()

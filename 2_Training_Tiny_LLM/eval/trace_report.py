"""Where the points are lost, from an `eval.run_real --trace` file: per question category, the exact-reader and model
routes, how often some reading reached the key (oracle), and every miss in exactly one bucket with the work it calls for.

    python -m eval.trace_report runs/v10-r8p-final/trace_dev.jsonl [--all] [--out report.json]
"""
from __future__ import annotations

import argparse
import collections
import json

BUCKETS = {   # miss bucket -> what would fix it
    'exact reader wrong': 'rules: tighten them, or cross-check with the model before trusting them',
    'no reading right': 'reading coverage: training data in this wording',
    'right reading not picked': 'selection: verifier / vote weights',
}
SAID_NONE = ('unrelated', 'none')   # the engine found no family link


def bucket(rec) -> str | None:
    if rec['ok']:
        return None
    if rec['route'] == 'exact':
        return 'exact reader wrong'
    return 'right reading not picked' if rec['oracle'] else 'no reading right'


def summarize(recs: list[dict]) -> dict:
    groups = collections.defaultdict(list)
    for r in recs:
        groups[r.get('category') or '?'].append(r)
        groups['ALL'].append(r)
    out = {}
    for name, rs in sorted(groups.items(), key=lambda g: (g[0] == 'ALL', -len(g[1]))):
        n = len(rs)
        ex = [r for r in rs if r['route'] == 'exact']
        mo = [r for r in rs if r['route'] != 'exact']
        miss = collections.Counter(bucket(r) for r in rs if not r['ok'])
        sh = [r['shadow'] for r in ex if r.get('shadow')]
        out[name] = {
            'n': n, 'right': sum(r['ok'] for r in rs),
            'exact': {'n': len(ex), 'right': sum(r['ok'] for r in ex),
                      'shadow_n': len(sh), 'shadow_right': sum(s['ok'] for s in sh),
                      'wrong_but_model_right': sum(1 for r in ex if not r['ok'] and r.get('shadow') and r['shadow']['ok']),
                      'disagree_with_model': sum(1 for s in sh if not s['agrees'])},
            'model': {'n': len(mo), 'right': sum(r['ok'] for r in mo), 'oracle': sum(r['oracle'] for r in mo)},
            'misses': {k: miss.get(k, 0) for k in BUCKETS},
            # overlapping tags: misses where the engine said "no link" although the key names a relation
            'said_no_link_but_key_relation': sum(1 for r in rs if not r['ok'] and r['final']['status'] in SAID_NONE
                                                 and not r['key_is_none']),
            'errors': sum(1 for r in rs if r['final']['status'] in ('error', 'contradiction')),
            'verified': {'n': sum(r['ok_verified'] is not None for r in rs),
                         'right': sum(bool(r['ok_verified']) for r in rs)},
        }
    return out


def pct(a, b):
    return f'{a}/{b} = {a / b:.1%}' if b else '0/0'


def show(summary: dict) -> str:
    lines = []
    for name, s in summary.items():
        e, m = s['exact'], s['model']
        lines.append(f"\n{name}: {pct(s['right'], s['n'])} | verified keys {pct(s['verified']['right'], s['verified']['n'])}")
        lines.append(f"  exact readers {pct(e['right'], e['n'])}" + (
            f" | shadow model alone {pct(e['shadow_right'], e['shadow_n'])}, disagrees {e['disagree_with_model']},"
            f" rescues {e['wrong_but_model_right']} exact misses" if e['shadow_n'] else ''))
        lines.append(f"  model route   {pct(m['right'], m['n'])} | some reading right {pct(m['oracle'], m['n'])}")
        for k, v in s['misses'].items():
            lines.append(f"  miss: {k:26s} {v:4d}  ({v / max(1, s['n']):.1%} of points) -> {BUCKETS[k]}")
        lines.append(f"  tag: engine said no link, key is a relation: {s['said_no_link_but_key_relation']} "
                     f"(solver conventions, e.g. the one-family assumption) | errors/contradictions: {s['errors']}")
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('trace')
    ap.add_argument('--all', action='store_true', help='include broken (unfair) questions')
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    recs = [json.loads(line) for line in open(a.trace) if line.strip()]
    recs = [r for r in recs if a.all or r.get('fair', True)]
    summary = summarize(recs)
    print(show(summary))
    if a.out:
        json.dump(summary, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()

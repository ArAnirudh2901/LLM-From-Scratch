"""Paired comparison of two `eval.run_real --out` results on the same questions (the dev split): fair scores,
McNemar on the per-question results, and the adoption rule of TRAINING_V11_CODEX.md: the new run is adopted only if
its fair score is at least --min_gain points higher, it wins more questions than it loses, and p < --alpha.
The output holds counts only, never question text.

    python -m eval.compare_runs OLD.json NEW.json --split dev --out compare.json
"""
from __future__ import annotations

import argparse
import json
import re

from eval.rescore_trace import mcnemar


def fair(score: dict, split: str) -> tuple[int, int]:
    m = re.match(r'\s*(\d+)/(\d+)', score['splits'].get(f'{split} fair', '0/0'))
    return (int(m.group(1)), int(m.group(2))) if m else (0, 0)


def compare(old: dict, new: dict, split: str = 'dev', min_gain: float = 2.0, alpha: float = 0.05) -> dict:
    a, b = old['score'], new['score']
    if len(a['ok']) != len(b['ok']):
        raise SystemExit(f"STOP: the runs scored different question sets ({len(a['ok'])} vs {len(b['ok'])})")
    if a.get('ids') and b.get('ids') and a['ids'] != b['ids']:
        raise SystemExit('STOP: the runs scored different questions')
    if a.get('fair') and b.get('fair') and a['fair'] != b['fair']:
        raise SystemExit('STOP: the runs disagree on which questions are fair')
    # wins/losses on the fair questions only, like the fair score; runs older than the mask: same files, same order
    mask = b.get('fair') or a.get('fair') or [True] * len(a['ok'])
    pairs = [(o, n) for o, n, f in zip(a['ok'], b['ok'], mask) if f]
    wins = sum(n and not o for o, n in pairs)
    losses = sum(o and not n for o, n in pairs)
    (fo, no), (fn, nn) = fair(a, split), fair(b, split)
    po, pn = 100 * fo / max(no, 1), 100 * fn / max(nn, 1)
    p = mcnemar(wins, losses)
    return {'split': split, 'old_fair': f'{fo}/{no} = {po:.1f}%', 'new_fair': f'{fn}/{nn} = {pn:.1f}%',
            'gain_points': round(pn - po, 2), 'wins': wins, 'losses': losses, 'p': round(p, 4),
            'adopt': pn - po >= min_gain and wins > losses and p < alpha,
            'rule': f'gain >= {min_gain} points, wins > losses, McNemar p < {alpha}'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('old')
    ap.add_argument('new')
    ap.add_argument('--split', default='dev')
    ap.add_argument('--min_gain', type=float, default=2.0)
    ap.add_argument('--alpha', type=float, default=0.05)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    res = compare(json.load(open(a.old)), json.load(open(a.new)), a.split, a.min_gain, a.alpha)
    print(json.dumps(res, indent=1))
    if a.out:
        with open(a.out, 'w') as f:
            json.dump(res, f, indent=1)


if __name__ == '__main__':
    main()

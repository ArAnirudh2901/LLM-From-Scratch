import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from eval.compare_runs import compare  # noqa: E402
from eval.rescore_trace import decide  # noqa: E402


def run(ok, fair):
    return {'score': {'ok': ok, 'splits': {'dev fair': f'{fair}/100 = {fair}.0%'}}}


def test_adopt_needs_gain_wins_and_significance():
    old = run([True] * 60 + [False] * 40, 60)
    better = run([True] * 60 + [True] * 15 + [False] * 25, 75)
    r = compare(old, better)
    assert r['adopt'] and r['wins'] == 15 and r['losses'] == 0 and r['gain_points'] == 15.0
    small = run([True] * 61 + [False] * 39, 61)
    assert not compare(old, small)['adopt']   # +1 point, not significant
    with pytest.raises(SystemExit):
        compare(old, run([True] * 10, 10))


def test_rescore_decision_picks_the_best_significant_setting():
    base = [True] * 50 + [False] * 50
    ok = {'base': base, 'closure': [True] * 62 + [False] * 38, 'lever': [True] * 51 + [False] * 49,
          'all': [True] * 66 + [False] * 34}
    d = decide(ok)
    assert d['settings']['closure']['adopt'] and not d['settings']['lever']['adopt']
    assert d['adopted'] == 'all' and d['adopted_cli'] == '--closure --core_repair --lever --trace_pow 1'
    assert decide({'base': base, 'lever': base})['adopted_cli'] == ''

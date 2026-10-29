import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from build_balanced import checked  # noqa: E402
from eval.check_authored import check  # noqa: E402
from eval.gauntlet import CATEGORIES, build, correct, gold_answer  # noqa: E402
from eval.twisted_set import T  # noqa: E402
from kinlogic.generate import Lineage, q_generations  # noqa: E402


def test_lineage_ends_are_cousins_or_ancestors():
    fam = Lineage(random.Random(1), 4, 3)
    a, b = fam.ends
    assert 'cousin' in fam.truth(a, b)['key']
    fam = Lineage(random.Random(2), 7, 0)
    assert fam.truth(*fam.ends)['key'].endswith('great-great-great-great-great-grandfather') or \
        fam.truth(*fam.ends)['key'].endswith('great-great-great-great-great-grandmother')


def test_ten_generation_rows_read_to_their_answer():
    rows = [r for r in (q_generations(random.Random(s), 10, 0) for s in range(40)) if r and checked(r)]
    assert len(rows) > 10 and all(r['truth'].count('great-') == 8 for r in rows)


def test_twisted_set_verifies():
    assert check(T, verbose=False) == []


def test_gauntlet_rows_score_with_gold_facts(monkeypatch):
    import eval.gauntlet as g
    monkeypatch.setattr(g, 'CATEGORIES', {k: CATEGORIES[k] for k in ('generations 6-10', 'detour 2', 'puzzle')})
    rows = [r for r in build(per=2, seed=3) if r['lf']]
    assert len(rows) == 6 and all(correct(r, gold_answer(r)) for r in rows)

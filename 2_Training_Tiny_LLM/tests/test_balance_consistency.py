import collections
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from build_balanced import build, checked  # noqa: E402
from eval.consistency import groups, rename, reorder, shuffle_options  # noqa: E402
from kinlogic.engine import solve  # noqa: E402
from kinlogic.facts import parse  # noqa: E402
from kinlogic.generate import _exact, obfuscation_set, q_statements  # noqa: E402


def answer(lf):
    q = solve(parse(lf)[0])['questions'][-1]
    return q['status'], [a['key'] for a in q['answers']]


def test_detour_levels_keep_names_and_answer():
    sets = [s for s in (obfuscation_set(random.Random(i)) for i in range(400)) if s]
    assert len(sets) > 20
    for rows in sets:
        assert [r['detours'] for r in rows] == [0, 1, 2]
        assert len({tuple(sorted(r['names'].values())) for r in rows}) == 1
        assert all(answer(r['lf']) == ('unique', [rows[0]['truth']]) for r in rows)
        assert len(rows[2]['lf'].split(';')) > len(rows[0]['lf'].split(';'))


def test_detour_steps_name_one_person():
    assert _exact('husband', None, False) and _exact('son', None, True) and _exact('grandfather', 'paternal', False)
    assert not _exact('son', None, False) and not _exact('grandfather', None, False)


def test_variants_keep_the_facts():
    rng, made = random.Random(5), collections.Counter()
    for i in range(300):
        ex = q_statements(random.Random(i))
        if not ex or not ex.get('truth'):
            continue
        for name, fn in (('rename', rename), ('options', shuffle_options), ('reorder', reorder)):
            v = fn(ex, rng)
            if v:
                made[name] += 1
                assert v['text'] != ex['text'] and answer(v['lf']) == answer(ex['lf'])
    assert all(made[k] for k in ('rename', 'options', 'reorder'))


def test_groups_start_with_the_base():
    gs = groups(10, 1)
    assert len(gs) == 10 and all(g[0][0] == 'base' and len(g) > 1 for g in gs)


def test_balanced_build_fills_formats_with_checked_rows():
    rows, _, short = build(6, 1, patience=3000)
    by = collections.Counter(r['format'] for r in rows)
    assert max(by.values()) <= 6 and len(by) >= 6
    assert len({' '.join(r['tokens']) for r in rows}) == len(rows)
    assert all(checked(r) for r in rows)

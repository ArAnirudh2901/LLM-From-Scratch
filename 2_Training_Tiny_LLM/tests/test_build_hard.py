"""Hard constructions: every added sentence is implied, so the engine's answers never change and the facts parse."""
import random

from build_dataset import engine_ok
from build_hard import _answers, _solve, augment
from kinlogic import generate as gen
from kinlogic.facts import parse


def test_augmented_questions_keep_their_answers():
    rng, kept = random.Random(7), 0
    for _ in range(40):
        ex = gen.KINDS['puzzle'](rng)
        if not ex or not engine_ok(ex):
            continue
        got = augment(ex, rng)
        if not got:
            continue
        kept += 1
        people = sorted({t for t in ex['tokens'] if t.startswith('@')}, key=lambda t: int(t[1:]))
        assert _answers(_solve(got['lf'], people)) == _answers(_solve(ex['lf'], people))
        assert len(parse(got['lf'])[0]) > len(parse(ex['lf'])[0]) or 'coordination' in got['constructs']
        assert got['tokens'][-1] == '?'
    assert kept >= 20

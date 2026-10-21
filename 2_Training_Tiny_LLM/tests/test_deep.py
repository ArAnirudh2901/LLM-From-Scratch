import random

from kinlogic import deep
from kinlogic.facts import parse


def test_every_format_and_depth_is_exact():
    rng = random.Random(3)
    for fmt in deep.FORMATS:
        for d in range(1, 13):
            made = 0
            for _ in range(12):
                ex = deep.make(rng, fmt, d)
                if ex is None:
                    continue
                made += 1
                assert deep.check(ex) is None, (fmt, d, ex['text'], ex['lf'])
            assert made >= 10, (fmt, d)


def test_chains_have_their_length_and_fit_the_window():
    rng = random.Random(4)
    for d in range(1, 13):
        mv = deep.moves_for(rng, d)
        assert len(mv) == d and all(p not in deep.BAD for p in zip(mv, mv[1:]))
    for _ in range(300):
        ex = deep.sample(rng)
        assert len(ex['tokens']) + len(ex['lf'].split()) + 3 <= 320
        assert not parse(ex['lf'])[1]


def test_cousin_chart_names():
    from kinlogic.kin import name_moves

    def key(p):
        mv = []
        for c in p:
            mv += [('U', 'M'), ('D', 'M')] if c == 'B' else [(c, 'M')]
        return name_moves(mv)['key']
    assert key('UUBDD') == 'second-cousin'
    assert key('UUBDDD') == 'second-cousin once removed'
    assert key('UUUBDDDDD') == 'third-cousin twice removed'
    assert key('UUUUUU') == 'great-great-great-great-grandfather'

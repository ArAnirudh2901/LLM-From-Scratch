import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import kinlogic.exam as E  # noqa: E402
from kinlogic.facts import allowed_next, budget_filter  # noqa: E402
from kinlogic.prepare import prepare  # noqa: E402

PUZZLE = ('Barkha and Chandresh are siblings. Manish has two children and he is son of Suresh, who is father-in-law of '
          'Hansa. Hansa has only one son. Chandresh is not son-in-law of Suresh. How Barkha is related to Suresh?')
FACTS = '@0 sibling @1 ; @2 has 2 child ; @3 son @2 ; @4 father-in-law @3 ; @4 has 1 son ; @3 not son-in-law @1 ; @2 male ; @0 female'   # stated, so no test depends on the name-gender table
READ = FACTS + ' ; ask @3 @0'
CYCLE = '@0 sibling @1 ; @2 has 2 child ; @3 son @2 ; @3 father @2 ; @4 father-in-law @3 ; @4 has 1 son ; @2 male ; @0 female ; ask @3 @0'


def reader(*readings):
    return lambda batch: [list(readings) for _ in batch]


def flags(**kw):
    saved = {k: getattr(E, k) for k in kw}
    for k, v in kw.items():
        setattr(E, k, v)
    return saved


def test_stray_period_is_removed():
    p = prepare('Pointing to Falguni, Hardik said, "The son of her only brother is the brother of my. wife." '
                'How is Falguni related to Hardik?')
    assert 'my wife .' in ' '.join(p['tokens'])


def test_closure_reads_everyone_named_as_one_family():
    saved = flags(CLOSURE=False)
    try:
        assert E.answer(PUZZLE, reader((READ, -0.2)))['status'] == 'unrelated'
        E.CLOSURE = True
        r = E.answer(PUZZLE, reader((READ, -0.2)))
        assert (r['answer'], r.get('via')) == ('granddaughter', 'closure')
        r = E.answer(PUZZLE + ' (a) Granddaughter (b) No relation (c) Daughter', reader((READ, -0.2)))
        assert r['option'] == 'b' and r.get('via') is None   # "no relation" offered: the engine's verdict stands
    finally:
        flags(**saved)


def test_core_repair_drops_one_fact_of_a_minimal_conflict():
    saved = flags(CORE_REPAIR=False, CLOSURE=False)
    try:
        assert E.answer(PUZZLE, reader((CYCLE, -0.2)))['status'] == 'contradiction'
        E.CORE_REPAIR = E.CLOSURE = True
        r = E.answer(PUZZLE, reader((CYCLE, -0.2)))
        assert r['via'] == 'core-repair' and sorted(r['core']) == ['@3 father @2', '@3 son @2']
        assert (r['answer'], r['dropped']) == ('granddaughter', '@3 father @2')
    finally:
        flags(**saved)


class Const:
    def prob(self, x):
        return 0.5


def test_lever_keeps_the_reader_weight():
    readings = ((FACTS + ' ; members 5 ; ask @3 @0', -0.5), ('@3 daughter @0 ; ask @3 @0', -0.1))
    saved, cached = flags(LEVER=False), list(E._VERIFIER)
    E._VERIFIER[:] = [Const()]
    try:
        assert E.answer(PUZZLE, reader(*readings))['answer'] == 'granddaughter'   # flat verifier: first reading wins
        E.LEVER = True
        assert E.answer(PUZZLE, reader(*readings))['answer'] == 'daughter'   # times the reader weight: likelier one
    finally:
        E._VERIFIER[:] = cached
        flags(**saved)


def test_traceability_ignores_the_question_sentence():
    p = prepare('Ravi is the father of Sita. Sita is the mother of Mohan. How is Ravi related to Mohan?')
    assert E.traceability(p, '@1 father @0 ; @2 mother @1 ; ask @2 @0') == 1.0
    assert E.traceability(p, '@1 father @0 ; @2 son @0 ; ask @2 @0') == 0.5
    assert E.traceability(p, 'ask @2 @0') is None


def test_budget_filter_finishes_with_a_question():
    w, ents = ['@1', 'father', '@0'], ['@0', '@1', '@2']
    assert budget_filter(w, allowed_next(w, ents), 50, 8) == allowed_next(w, ents)   # far from the end: unchanged
    assert budget_filter(w, allowed_next(w, ents), 3, 8) == [';']                  # no question yet
    nxt = w + [';']
    assert set(budget_filter(nxt, allowed_next(nxt, ents), 2, 8)) <= {'ask', 'who', 'count', 'gender', 'pair', 'prof'}
    done = nxt + ['ask', '@2', '@0']
    assert budget_filter(done, allowed_next(done, ents), 2, 8) == ['<eos>']


def test_rescore_replays_the_traced_answers():
    from eval.rescore_trace import rescore
    from eval.run_real import run
    from eval.trace import Tracer
    rows = [{'site': 's', 'options': [], 'answer_text': 'granddaughter', 'question': PUZZLE}]
    tr = Tracer().install()
    try:
        run(reader((READ, -0.2)), rows, tracer=tr)
    finally:
        tr.uninstall()
    ok, same = rescore(tr.records, {'base': {}, 'closure': {'CLOSURE': True}})
    assert same == 1 and ok == {'base': [False], 'closure': [True]}

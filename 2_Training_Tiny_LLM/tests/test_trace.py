import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from eval.run_real import as_text, correct, run  # noqa: E402
from eval.trace import Tracer  # noqa: E402
from eval.trace_report import summarize  # noqa: E402
import kinlogic.exam as exam  # noqa: E402

SPEECH = {'site': 's', 'answer': '1', 'question': 'Pointing to Falguni, Hardik said, "The son of her only brother is the '
          'brother of my wife." How is Falguni related to Hardik?',
          'options': [['1', 'Sister of father-in-law'], ['2', 'Niece'], ['3', 'Maternal aunt'], ['4', 'Grandmother'],
                      ['5', 'None of the above']]}
PUZZLE = {'site': 's', 'options': [], 'answer_text': 'granddaughter',
          'question': 'Barkha and Chandresh are siblings. Manish has two children and he is son of Suresh, who is '
                      'father-in-law of Hansa. Hansa has only one son. Chandresh is not son-in-law of Suresh. How Barkha '
                      'is related to Suresh?'}
RIGHT = ('@0 sibling @1 ; @2 has 2 child ; @3 son @2 ; @4 father-in-law @3 ; @4 has 1 son ; @3 not son-in-law @1 ; '
         '@2 male ; members 5 ; ask @3 @0')
WRONG = '@0 sibling @1 ; @2 has 2 child ; @3 son @2 ; @4 father-in-law @3 ; @4 has 1 daughter ; ask @3 @0'


def model(readings):
    calls = []

    def read(batch):
        calls.append(batch)
        return [list(readings) for _ in batch]
    return read, calls


def traced(r, read, shadow=False):
    tr = Tracer(shadow=shadow).install()
    try:
        text = as_text(r)
        tr.begin(text)
        res = exam.answer(text, read)
        return tr.end(r, res, correct(r, res), lambda t: exam.answer(t, read), category='test')
    finally:
        tr.uninstall()


def test_exact_route():
    read, calls = model([('ask @1 @0', -0.1)])
    rec = traced(SPEECH, read)
    assert not calls, 'an exactly read question must not reach the model'
    assert (rec['route'], rec['exact_reader'], rec['ok'], rec['oracle']) == ('exact', 'speech', True, True)
    assert rec['shadow'] is None


def test_model_route_oracle_without_selection():
    read, calls = model([(WRONG, -0.1), (RIGHT, -0.5)])   # the wrong reading scores higher and wins the vote
    rec = traced(PUZZLE, read)
    assert calls and rec['route'] == 'model'
    assert not rec['ok'] and rec['oracle'] and rec['hits'] == 1 and rec['n_cands'] == 2
    assert [c['hit'] for c in rec['passes'][0]['cands']] == [False, True]


def test_shadow_answers_with_exact_readers_off():
    read, calls = model([('@0 sister @1 ; ask @1 @0', -0.2)])
    rec = traced(SPEECH, read, shadow=True)
    assert rec['route'] == 'exact' and len(calls) == 1   # only the shadow pass used the model
    assert rec['shadow'] is not None and rec['shadow']['ok'] is False and rec['shadow']['agrees'] is False


def test_hooks_are_removed():
    before = (exam.candidates, exam.prepare)
    Tracer().install().uninstall()
    import kinlogic.codes as codes
    import kinlogic.rules as rules
    assert (exam.candidates, exam.prepare) == before
    assert codes.read.__module__ == 'kinlogic.codes' and rules.read.__module__ == 'kinlogic.rules'


def test_report_buckets():
    base = {'category': 'c', 'final': {'status': 'unique'}, 'key_is_none': False, 'ok_verified': None, 'shadow': None}
    recs = [{**base, 'route': 'exact', 'ok': True, 'oracle': True},
            {**base, 'route': 'exact', 'ok': False, 'oracle': False},
            {**base, 'route': 'model', 'ok': False, 'oracle': True},
            {**base, 'route': 'model', 'ok': False, 'oracle': False, 'final': {'status': 'unrelated'}}]
    s = summarize(recs)['ALL']
    assert (s['n'], s['right']) == (4, 1)
    assert s['misses'] == {'exact reader wrong': 1, 'no reading right': 1, 'right reading not picked': 1}
    assert s['said_no_link_but_key_relation'] == 1 and s['model']['oracle'] == 1


def test_run_writes_one_record_per_question():
    read, _ = model([(WRONG, -0.1), (RIGHT, -0.5)])
    tr = Tracer().install()
    try:
        score, misses = run(read, [SPEECH, PUZZLE], tracer=tr)
    finally:
        tr.uninstall()
    assert [r['route'] for r in tr.records] == ['exact', 'model']
    assert [r['ok'] for r in tr.records] == score['ok'] == [True, False]
    assert all(r['fair'] and r['split'] in ('dev', 'test') and r['category'] for r in tr.records)

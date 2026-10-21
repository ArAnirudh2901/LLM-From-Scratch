import random

from kinlogic import codes, deep, generate as gen
from kinlogic.prepare import prepare
from train_reader import answers


def _same(a, b, toks):
    people = sorted({t for t in toks if t[0] == '@'})
    return answers(a, people) == answers(b, people)


def test_decoded_coded_questions_have_the_gold_answer():
    rng = random.Random(11)
    read = 0
    for make in (lambda: gen.q_coded(rng), lambda: deep.make(rng, 'coded', rng.randint(1, 12))):
        for _ in range(150):
            ex = make()
            if ex is None:
                continue
            lf = codes.read(ex['tokens'])
            if lf is None:
                continue
            read += 1
            assert _same(lf, ex['lf'], ex['tokens']), (ex['text'], lf, ex['lf'])
    assert read > 250


def test_reversed_definitions_relatives_and_genders():
    q = ("'X @ Y' means 'Y is the father of X', 'X # Y' means 'Y is the sister of X' and 'X $ Y' means 'Y is the son "
         "of X'. In the expression K @ L # M $ N, M is a woman. How is N related to K?")
    assert codes.read(prepare(q)['tokens']) == '@2 father @3 ; @3 sister @4 ; @4 son @5 ; @4 female ; ask @2 @5'
    q2 = 'If A + B means A is the son of B and A - B means A is married to B, how is R\'s mother related to P in P + Q - R?'
    assert codes.read(prepare(q2)['tokens']) == '@4 son @3 ; @2 spouse @4 ; @2 mother #1 ; ask @3 #1'   # R=@2, P=@3, Q=@4


def test_not_coded_text_is_left_to_the_model():
    assert codes.read(prepare('A is the father of B. How is B related to A?')['tokens']) is None

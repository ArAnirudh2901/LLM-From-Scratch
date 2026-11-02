"""A friend links two families but is not family: every wording of one story gets the same exact reading and answers
(website bug 2026-11-02: 'Raju is friend of Ashok, Sita is the sister of Raju' crashed the page for Sita and Ashok)."""
import pytest

from kinlogic import exam, rules
from kinlogic.prepare import mark_question, prepare

WORDINGS = ['Raju is friend of Ashok, Sita is the sister of Raju', 'Raju is friend of Ashok, Sita is the sister of Raju.',
            'Raju is a friend of Ashok. Sita is the sister of Raju.', 'Sita is the sister of Raju. Raju and Ashok are friends.',
            "Raju is Ashok's friend, Sita is the sister of Raju", "Ashok's friend is Raju. Sita is Raju's sister.",
            "Ashok is a close friend of Raju. Raju's sister is Sita."]
PAIRS = {('Sita', 'Ashok'): 'not related', ('Ashok', 'Sita'): 'not related', ('Sita', 'Raju'): 'brother',
         ('Raju', 'Sita'): 'sister', ('Raju', 'Ashok'): 'not related (friend)'}


@pytest.mark.parametrize('story', WORDINGS)
def test_every_wording_reads_exactly_with_the_same_answers(story):
    for (a, b), want in PAIRS.items():
        t = f'{story} How is {b} related to {a}?'
        lf = rules.read(prepare(t)['tokens'])
        assert lf is not None, t
        assert exam.answer(t, lambda batch, lf=lf: [[(lf, 0.0)] for _ in batch])['answer'] == want, t


def test_glued_question_gets_its_sentence_break():
    assert mark_question('Sita is the sister of Raju How is Ashok related to Sita?') == \
        'Sita is the sister of Raju. How is Ashok related to Sita?'
    assert mark_question('If A is the father of B how is B related to A?') == 'If A is the father of B how is B related to A?'
    assert prepare('Sita is the sister of Raju How is Raju related to Sita?')['tokens'] == \
        prepare('Sita is the sister of Raju. How is Raju related to Sita?')['tokens']


def test_plain_question_never_answers_with_a_statement_check():
    junk = '@0 friend of @1 ; claim 2 ; @2 sister of @0'   # a misreading: statements the question does not have
    t = 'Raju went to the fair with Ashok and Sita. How is Ashok related to Sita?'   # no exact reading: the model's counts
    assert exam.answer(t, lambda b: [[(junk, -0.1)] for _ in b])['status'] == 'none'
    t = 'Raju is friend of Ashok, Sita is the sister of Raju. Who is Sita to Ashok?'   # the exact reading wins over the junk
    assert exam.answer(t, lambda b: [[(junk, -0.1)] for _ in b])['answer'] == 'not related'

from eval.batched import Miss, Rounds
from kinlogic import exam


def _gen(calls):
    def gen(batch, k=1, sample=False, bs=32, **kw):
        calls.append(len(batch))
        return [[(f'@0 son @1 ; ask @0 @1', -0.1 * len(t))] * k for t in batch]
    return gen


def test_rounds_give_the_one_by_one_answers_in_few_batched_calls():
    texts = [f'Meena told Arun that the boy on the stage is the grandson of her mother-in-law{"." * i} How is the boy related to Meena?'
             for i in range(1, 6)]   # read by the model: the exact readers leave it
    calls = []
    R = Rounds(_gen(calls), rows=64)
    read = lambda b: R(b, k=2)
    got = R.answer_all(lambda t: exam.answer(t, exam.with_split_search(read, focus=True, min_sentences=2)), texts)
    calls2 = []
    one = _gen(calls2)
    want = [exam.answer(t, exam.with_split_search(lambda b: one(b, k=2), focus=True, min_sentences=2)) for t in texts]
    assert [g.get('answer') for g in got] == [w.get('answer') for w in want]
    assert calls and len(calls) < len(calls2)   # all questions' reads of one kind in one call


def test_miss_is_not_swallowed_by_answer():
    R = Rounds(_gen([]))
    try:
        exam.answer('Meena told Arun that the boy on the stage is the grandson of her mother-in-law. How is the boy related to Meena?',
                    lambda b: R(b, k=1))
    except Miss:
        return
    raise AssertionError('Miss was caught as an error answer')

import random
import re

from build_focus import local, rows_of
from kinlogic import generate as gen
from kinlogic.exam import FOCUS, _sentences, with_split_search


def _ex(kind, seed):
    rng = random.Random(seed)
    gen.UNITS = True
    try:
        for _ in range(200):
            ex = gen.KINDS[kind](rng)
            if ex and ex.get('units'):
                return ex
    finally:
        gen.UNITS = False
    raise AssertionError(f'no {kind} question with units in 200 tries')


def test_units_line_up_with_sentences():
    for kind in ('puzzle', 'statements', 'speech'):
        for seed in range(5):
            ex = _ex(kind, seed)
            ends = set()
            i = 0
            for t in _sentences(list(ex['tokens'])):
                i += len(t)
                ends.add(i)
            for s, e, lf in ex['units']:
                assert e in ends and (s == 0 or s in ends), (kind, ex['tokens'][s:e])
                assert set(re.findall(r'@\d+', lf)) <= set(ex['tokens'])


def test_focus_rows_number_unnamed_people_locally():
    assert local('@0 mother #3 ; #3 brother #4') == '@0 mother #1 ; #1 brother #2'
    ex = _ex('puzzle', 1)
    for r in rows_of(ex, random.Random(0), 9):
        i = r['tokens'].index(FOCUS)
        assert r['tokens'][:i] == ex['tokens'] and r['lf']


def test_focus_search_reads_each_sentence_in_context():
    toks = '@0 is the father of @1 . @1 is the mother of @2 . how is @2 related to @0 ?'.split()
    seen = []

    def read(batch):
        seen.extend(batch)
        out = []
        for b in batch:
            sent = b[b.index(FOCUS) + 1:] if FOCUS in b else b
            lf = {'@0': '@1 father @0', '@1': '@2 mother @1', 'how': 'ask @0 @2'}[sent[0]]
            out.append([(lf, -0.1)])
        return out

    got = with_split_search(read, focus=True, min_sentences=2)([toks])[0]
    assert all(b[:len(toks)] == toks for b in seen[1:])
    assert any(f == '@1 father @0 ; @2 mother @1 ; ask @0 @2' and abs(sc + 0.1) < 1e-9 for f, sc in got)


def test_focus_search_keeps_the_question_sentence_facts():
    # "A is the brother of B . if C is the husband of B , how is C related to A ?": 'husband' lives in the question
    toks = '@0 is the brother of @1 . if @2 is the husband of @1 , how is @2 related to @0 ?'.split()
    lfs = {'@0': '@1 brother @0 ; @1 son #1', 'if': '@1 husband @2 ; @2 son #1 ; ask @0 @2'}

    def read(batch):
        return [[(lfs[b[b.index(FOCUS) + 1]] if FOCUS in b else 'ask @0 @2', -0.1)] for b in batch]

    got = [f for f, _ in with_split_search(read, focus=True, min_sentences=2)([toks])[0]]
    assert '@1 brother @0 ; @1 son #1 ; @1 husband @2 ; @2 son #2 ; ask @0 @2' in got   # unnamed people numbered apart


def test_focus_search_skips_a_question_when_a_sentence_read_finishes_nothing():
    toks = '@0 is the father of @1 . how is @1 related to @0 ?'.split()
    read = lambda batch: [[('', float('-inf'))] if FOCUS in b else [('ask @0 @1', -0.1)] for b in batch]
    assert with_split_search(read, focus=True, min_sentences=2)([toks])[0] == [('ask @0 @1', -0.1)]


def _tiny(focus_mask=True, n_loop=1):
    import torch
    from model_reader import Reader
    torch.manual_seed(0)
    m = Reader(vocab_size=40, block_size=96, n_layer=2, n_head=2, n_embd=32, prefix_lm=True, focus_mask=focus_mask, n_loop=n_loop).eval()
    m.c_id, m.f_id = 3, 4
    return m


def test_shared_question_prefill_matches_joined_rows():
    import torch
    for n_loop in (1, 2):
        m = _tiny(n_loop=n_loop)
        q = [2] + list(range(10, 30))   # <bos> question
        sents = [[4, 11, 12, 13, 3], [4, 20, 21, 3], [4, 15, 16, 17, 18, 19, 3]]
        rows = [q + s for s in sents]
        P = max(map(len, rows))
        idx = torch.zeros(len(rows), P, dtype=torch.long)
        for i, r in enumerate(rows):
            idx[i, :len(r)] = torch.tensor(r)
        plen = torch.tensor([len(r) for r in rows])
        with torch.no_grad():
            c1, lp1 = m.prefill(idx, plen)
            S = max(map(len, sents))
            f = torch.zeros(len(sents), S, dtype=torch.long)
            for i, s in enumerate(sents):
                f[i, :len(s)] = torch.tensor(s)
            c2, lp2 = m.prefill_focus(torch.tensor([q]), f, torch.tensor([len(s) for s in sents]))
            assert torch.allclose(lp1, lp2, atol=1e-5)
            for (k1, v1), (k2, v2) in zip(c1[:-1], c2[:-1]):
                for i, n in enumerate(plen.tolist()):
                    assert torch.allclose(k1[i, :, :n], k2[i, :, :n], atol=1e-5) and torch.allclose(v1[i, :, :n], v2[i, :, :n], atol=1e-5)
            tok = torch.tensor([7, 8, 9])   # one decoded token each: same next-token log-probs
            assert torch.allclose(m.step(c1, tok, plen, plen, P, 1), m.step(c2, tok, plen, plen, len(q) + S, 1), atol=1e-5)
            # training path (teacher forcing) agrees with the cached prefill on the first fact token
            mask = torch.zeros_like(idx, dtype=torch.bool)
            for i, n in enumerate(plen.tolist()):
                mask[i, n - 1] = True
            lp3, _ = m.fact_logprobs(idx, mask, 3)
            assert torch.allclose(lp1, lp3, atol=1e-5)


def test_question_is_blind_to_its_sentence_only_with_focus_mask():
    import torch
    for fm, same in ((True, True), (False, False)):
        m = _tiny(focus_mask=fm)
        a = torch.tensor([[2, 10, 11, 12, 4, 13, 3]])
        b = torch.tensor([[2, 10, 11, 12, 4, 25, 3]])
        with torch.no_grad():
            ha, hb = m.hidden(a), m.hidden(b)
        assert torch.allclose(ha[:, :4], hb[:, :4], atol=1e-6) == same

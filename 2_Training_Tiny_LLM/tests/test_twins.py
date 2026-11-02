"""Twins for consistency training: same facts token for token, another surface; the loss is zero for identical rows."""
import random

import torch

from reader import collate, encode_example, fact_losses, Vocab
from twins import twin
from model_reader import Reader


def test_twin_keeps_the_question_and_the_words():
    toks = '@0 is the father of @1 . @2 is the wife of @0 . @3 is the son of @2 . how is @3 related to @1 ?'.split()
    rng = random.Random(1)
    for _ in range(50):
        tw = twin(toks, rng)
        if tw is None:
            continue
        assert tw[-7:] == toks[-7:]   # the question stays last
        assert sorted(t for t in tw if t not in '.,') == sorted(t for t in toks if t not in '.,')


def test_no_reorder_when_sentences_lean_on_each_other():
    toks = '@0 is the father of @1 . his wife is @2 . how is @2 related to @1 ?'.split()
    rng = random.Random(0)
    for _ in range(50):
        tw = twin(toks, rng)
        assert tw is None or [t for t in tw if t not in '.,'] == [t for t in toks if t not in '.,']


def test_consistency_loss_is_zero_for_identical_rows_and_trains_otherwise():
    itos = ['<pad>', '<unk>', '<bos>', '<c>', '<eos>', '.', ',', '?', 'is', 'the', 'father', 'of', 'how', 'related', 'to',
            'son', ';', 'ask'] + [f'@{i}' for i in range(32)]
    vocab = Vocab(itos)
    torch.manual_seed(0)
    m = Reader(vocab_size=len(itos), block_size=64, n_layer=2, n_head=2, n_embd=32, prefix_lm=True)
    m.c_id = vocab.stoi['<c>']
    a = '@0 is the father of @1 . how is @1 related to @0 ?'.split()
    b = '@0 is the father of @1 how is @1 related to @0 ?'.split()
    lf = '@1 father @0 ; ask @0 @1'
    ea, eb = (encode_example(vocab, t, lf, 64, random.Random(5)) for t in (a, b))
    assert ea[0][ea[1]:] == eb[0][eb[1]:]
    idx, tgt, lm = collate([(*ea, 7), (*ea, 7)], 0, 'cpu')
    assert fact_losses(m, idx, tgt, lm, vocab, pairs=[(0, 1)])[2].item() < 1e-6
    idx, tgt, lm = collate([(*ea, 7), (*eb, 7)], 0, 'cpu')
    cons = fact_losses(m, idx, tgt, lm, vocab, pairs=[(0, 1)])[2]
    assert cons.item() > 0
    cons.backward()
    assert m.tok_emb.weight.grad is not None

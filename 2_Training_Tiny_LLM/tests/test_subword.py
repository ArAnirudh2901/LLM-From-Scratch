from collections import Counter

from reader import Vocab
from subword import alphabet, pieces, ranks, train


def test_merges_rebuild_frequent_words():
    words = Counter({'grandmother': 50, 'grandfather': 40, 'granddaughter': 30, 'mother': 60})
    m = train(words, 40)
    assert pieces('grandmother', ranks(m)) == ['▁grandmother']
    assert ''.join(p.lstrip('▁').replace('##', '') for p in pieces('grandniece', ranks(m))) == 'grandniece'


def test_vocab_never_unknown_for_known_letters():
    words = Counter({'kinship': 5, 'relation': 7})
    m = train(words, 10)
    itos = ['<pad>', '<unk>', '<bos>', '<c>', '<sep>', '<eos>', '@0', 'mother'] + alphabet(words) + sorted({a + b[2:] for a, b in m})
    v = Vocab(itos, m)
    ids = v.encode(['mother', '@0', 'kinship', 'relation', 'ration'])
    assert v.unk not in ids
    assert ids[:2] == [v.stoi['mother'], v.stoi['@0']]   # known words and placeholders stay single tokens


def test_plain_vocab_unchanged():
    v = Vocab(['<pad>', '<unk>', 'a'])
    assert v.encode(['a', 'zzz']) == [2, 1]

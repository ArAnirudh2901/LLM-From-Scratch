import torch

from name_model import MAX_LEN, NameModel, batch_ids, decide, encode, pick_thresholds


def test_encode_keeps_start_and_ending():
    assert encode('Ram')[0] == encode('ram')[0] and len(encode('ram')) == 5
    long = encode('a' * 40 + 'preet')
    assert len(long) == MAX_LEN and long[-6:] == encode('preet')[-6:]


def test_model_shapes_and_padding_invariance():
    m = NameModel().eval()
    x = batch_ids(['asha', 'venkataramanan'], 'cpu')
    one = m(batch_ids(['asha'], 'cpu'))
    assert m(x).shape == (2,) and torch.allclose(m(x)[0], one[0], atol=1e-5)
    assert sum(p.numel() for p in m.parameters()) < 150_000


def test_thresholds_respect_the_wrong_budget():
    ps = [0.02, 0.03, 0.4, 0.6, 0.97, 0.99, 0.2]
    gold = ['M', 'M', 'F', 'M', 'F', 'F', 'F']
    tm, tf = pick_thresholds(ps, gold, 0.0)
    dec = [decide(p, tm, tf) for p in ps]
    assert not any(d is not None and d != g for d, g in zip(dec, gold)) and dec[:2] == ['M', 'M'] and dec[4:6] == ['F', 'F']

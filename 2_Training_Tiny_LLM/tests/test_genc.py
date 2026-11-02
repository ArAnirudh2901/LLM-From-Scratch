"""Grammar encoder (genc.py): starts as the reader alone, and training and cached decoding see the same question."""
import pytest
import torch

pytest.importorskip('transformers')
from genc import W, GrammarEncoder  # noqa: E402
from model_reader import Reader  # noqa: E402

V, C = 40, 3   # vocabulary, id of '<c>'
SPEC = {'hf': {'vocab_size': 30, 'hidden_size': 16, 'intermediate_size': 24, 'num_hidden_layers': 3,
               'num_attention_heads': 2, 'pad_token_id': 0, 'cls_token_id': 1, 'sep_token_id': 2, 'bos_token_id': 1,
               'eos_token_id': 2, 'local_attention': 8, 'reference_compile': False}, 'n_vocab': V}


def reader(genc=None):
    torch.manual_seed(0)
    m = Reader(vocab_size=V, block_size=64, n_layer=2, n_head=2, n_embd=32, prefix_lm=True, genc=genc).eval()
    m.c_id = C
    if genc:
        g = torch.Generator().manual_seed(1)
        m.genc.tab_n.copy_(torch.randint(0, 3, (V,), generator=g))
        m.genc.tab.copy_(torch.randint(3, 30, (V, W), generator=g))
    return m


def rows():
    q = [[4, 9, 11, 5, 17, C], [4, 22, 8, 30, 12, 7, 5, 19, 25, C]]   # questions ending at '<c>'
    return q, [x + [13, 14, 15] for x in q]                         # then some facts


def batch(seqs):
    idx = torch.zeros(len(seqs), max(map(len, seqs)), dtype=torch.long)
    for i, s in enumerate(seqs):
        idx[i, :len(s)] = torch.tensor(s)
    return idx


def at_c(m, seqs):
    """fact_logprobs at each row's '<c>' (the first fact's distribution)."""
    idx = batch(seqs)
    mask = idx == C
    return m.fact_logprobs(idx, mask, C)[0]


def test_zero_projection_is_the_reader_alone():
    base, fused = reader(), reader(SPEC)
    fused.load_state_dict(base.state_dict(), strict=False)
    q, full = rows()
    with torch.no_grad():
        assert torch.equal(at_c(base, full), at_c(fused, full))
        plen = torch.tensor([len(x) for x in q])
        assert torch.equal(base.prefill(batch(q), plen)[1], fused.prefill(batch(q), plen)[1])


def test_training_and_cached_decoding_agree_and_padding_does_not_leak():
    m = reader(SPEC)
    torch.nn.init.normal_(m.genc.proj.weight, 0, 0.5)
    q, full = rows()
    with torch.no_grad():
        train = at_c(m, full)
        dec = m.prefill(batch(q), torch.tensor([len(x) for x in q]))[1]
        one = m.prefill(batch(q[:1]), torch.tensor([len(q[0])]))[1]
    assert torch.allclose(train, dec, atol=1e-5)
    assert torch.allclose(dec[:1], one, atol=1e-5)
    base = reader()
    with torch.no_grad():
        assert not torch.allclose(at_c(base, full), train, atol=1e-3)   # the encoder is really read


def test_encoder_learns_only_after_freeze():
    m = reader(SPEC).train()
    torch.nn.init.normal_(m.genc.proj.weight, 0, 0.5)
    _, full = rows()
    for frozen in (True, False):
        m.zero_grad(set_to_none=True)
        m.genc.enc.requires_grad_(not frozen)
        at_c(m, full).logsumexp(-1).sum().backward()
        assert m.genc.proj.weight.grad is not None
        assert (m.genc.enc.embeddings.tok_embeddings.weight.grad is None) == frozen

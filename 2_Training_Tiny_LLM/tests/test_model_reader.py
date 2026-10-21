import torch

from model_reader import Reader, make_model


def _model(copy=True):
    torch.manual_seed(0)
    return Reader(vocab_size=50, block_size=64, n_layer=2, n_head=2, n_embd=32, copy=copy, d_copy=16).eval()


def test_cached_decoding_matches_full_forward():
    for copy in (True, False):
        m = _model(copy)
        C = 3   # the <c> token id in this test
        q = [1, 7, 8, 9, 10, 8, C]          # <bos> question <c>
        facts = [8, 11, 9, 12]
        idx = torch.tensor([q + facts])
        mask = torch.zeros_like(idx, dtype=torch.bool)
        mask[0, len(q) - 1:len(q) - 1 + len(facts)] = True   # positions that predict the facts
        full, _ = m.fact_logprobs(idx, mask, C)
        P = len(q)
        cache, lp = m.prefill(torch.tensor([q]), torch.tensor([P]))
        got = [lp[0]]
        for g, t in enumerate(facts[:-1]):
            lp = m.step(cache, torch.tensor([t]), torch.tensor([P + g]), torch.tensor([P]), P, g + 1)
            got.append(lp[0])
        assert torch.allclose(torch.stack(got), full, atol=1e-4), (copy, (torch.stack(got) - full).abs().max())


def test_copy_puts_mass_on_question_tokens_and_trains():
    m = _model(True).train()
    idx = torch.tensor([[1, 7, 8, 3, 8, 2]])
    mask = torch.tensor([[False, False, False, True, True, False]])
    lp, _ = m.fact_logprobs(idx, mask, 3)
    assert torch.allclose(lp.exp().sum(-1), torch.ones(2), atol=1e-4)
    (-lp[torch.arange(2), torch.tensor([8, 2])]).mean().backward()
    assert m.copy_q.weight.grad is not None and m.blocks[0].attn.qkv.weight.grad is not None


def test_make_model_and_size():
    m = make_model({'arch': 'v10', 'vocab_size': 3235, 'block_size': 320})
    n = sum(p.numel() for p in m.parameters())
    assert 10e6 < n < 14e6, n
    assert type(make_model({'vocab_size': 10, 'block_size': 8, 'n_layer': 1, 'n_head': 1, 'n_embd': 8})).__name__ == 'GPT'

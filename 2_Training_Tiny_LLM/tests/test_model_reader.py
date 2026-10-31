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


def test_looped_reader_cached_decoding_matches_full_pass():
    import torch
    from model_reader import Reader
    torch.manual_seed(0)
    m = Reader(vocab_size=50, block_size=64, n_layer=2, n_head=2, n_embd=32, n_loop=3, copy=False).eval()
    idx = torch.randint(5, 50, (2, 12))
    full = torch.log_softmax(m.head(m.hidden(idx)), -1)
    plen = torch.tensor([8, 8])
    cache, lp0 = m.prefill(idx[:, :8], plen)
    lp = m.step(cache, idx[:, 8], torch.tensor([8, 8]), plen, 8, 1)
    assert len(cache) == 2 * 3 + 1   # one cache per layer application, plus the copy keys
    assert torch.allclose(lp0, full[:, 7], atol=1e-5) and torch.allclose(lp, full[:, 8], atol=1e-5)


def test_ponder_loss_trains_the_exit_gate():
    import torch
    from model_reader import Reader
    from reader import fact_losses

    class V:
        stoi = {'<c>': 3}
    torch.manual_seed(0)
    m = Reader(vocab_size=50, block_size=64, n_layer=2, n_head=2, n_embd=32, n_loop=3)
    idx = torch.randint(5, 50, (2, 12))
    idx[:, 6] = 3
    tgt = torch.full((2, 12), -100)
    tgt[:, 7:11] = idx[:, 8:12]
    loss, _ = fact_losses(m, idx, tgt, torch.full((2, 12), -100), V)
    loss.backward()
    assert torch.isfinite(loss) and m.exit_gate.weight.grad.abs().sum() > 0
    m.set_loops(1)
    assert len(m.passes()) == 2


def test_gate_and_canon_start_as_the_plain_model():
    torch.manual_seed(0)
    plain = Reader(vocab_size=50, block_size=64, n_layer=2, n_head=2, n_embd=32, d_copy=16).eval()
    new = Reader(vocab_size=50, block_size=64, n_layer=2, n_head=2, n_embd=32, d_copy=16, gate_attn=True, canon=True).eval()
    missing = new.load_state_dict(plain.state_dict(), strict=False).missing_keys
    assert missing and all('canon' in k or '.attn.g.' in k for k in missing)
    idx = torch.randint(4, 50, (2, 10))
    assert torch.equal(plain.hidden(idx), new.hidden(idx))


def test_gate_and_canon_cached_decoding_matches_full_forward():
    torch.manual_seed(1)
    m = Reader(vocab_size=50, block_size=64, n_layer=2, n_head=2, n_embd=32, d_copy=16, gate_attn=True, canon=True,
               prefix_lm=True).eval()
    C = 3
    m.c_id = C
    with torch.no_grad():   # trained-looking modules, not the identity start
        for n, p in m.named_parameters():
            if 'canon' in n or '.attn.g.' in n:
                p.normal_(0, 0.3)
    qs = [[1, 7, 8, 9, 10, 8, C], [1, 9, C]]   # different prefix lengths: the decode cache is gathered per row
    facts = [8, 11, 9, 12]
    P = max(len(q) for q in qs)
    pre = torch.zeros(2, P, dtype=torch.long)
    for i, q in enumerate(qs):
        pre[i, :len(q)] = torch.tensor(q)
    plen = torch.tensor([len(q) for q in qs])
    cache, lp = m.prefill(pre, plen)
    got = [lp]
    for g, t in enumerate(facts[:-1]):
        got.append(m.step(cache, torch.tensor([t, t]), plen + g, plen, P, g + 1))
    for i, q in enumerate(qs):
        idx = torch.tensor([q + facts])
        mask = torch.zeros_like(idx, dtype=torch.bool)
        mask[0, len(q) - 1:len(q) - 1 + len(facts)] = True
        full, _ = m.fact_logprobs(idx, mask, C)
        dec = torch.stack([s[i] for s in got])
        assert torch.allclose(dec, full, atol=1e-4), (i, (dec - full).abs().max())

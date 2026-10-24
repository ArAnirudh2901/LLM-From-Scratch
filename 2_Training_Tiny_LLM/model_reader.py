"""v10 reader: a small modern decoder (rotary positions, RMSNorm, SwiGLU, embeddings tied to the output) with a copy
head. Each fact token is a mix of the vocabulary and the question's own tokens (pointer-generator): relation words,
names and placeholders are copied from the text instead of re-guessed, which is where the v9 reader failed on
unseen phrasings. Rotary positions keep long (deep-chain) inputs and outputs on the same footing as short ones.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        return (x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)).to(x.dtype) * self.weight


def rope(x, cos, sin):
    """Rotates pairs of channels by position; cos/sin broadcast to x[..., ::2]."""
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), -1).flatten(-2)


class Attention(nn.Module):
    def __init__(self, d: int, n_head: int, dropout: float, qk_norm: bool = False):
        super().__init__()
        self.n_head, self.d_head, self.dropout = n_head, d // n_head, dropout
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        self.q_norm, self.k_norm = (RMSNorm(self.d_head), RMSNorm(self.d_head)) if qk_norm else (None, None)   # OLMo 2 / Qwen3: bounded attention logits

    def heads(self, x, cos, sin):
        B, T, C = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.n_head, self.d_head).unbind(2)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        if self.q_norm is not None:
            q, k = self.q_norm(q), self.k_norm(k)
        return rope(q, cos, sin), rope(k, cos, sin), v

    def forward(self, x, cos, sin, mask=None):
        q, k, v = self.heads(x, cos, sin)
        drop = self.dropout if self.training else 0.0
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=drop) if mask is None else \
            F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=drop)
        return self.proj(y.transpose(1, 2).reshape(x.shape)), (k, v)


class SwiGLU(nn.Module):
    def __init__(self, d: int, hidden: int, dropout: float):
        super().__init__()
        self.w1 = nn.Linear(d, hidden, bias=False)
        self.w3 = nn.Linear(d, hidden, bias=False)
        self.w2 = nn.Linear(hidden, d, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class Block(nn.Module):
    def __init__(self, d, n_head, hidden, dropout, qk_norm=False):
        super().__init__()
        self.ln1, self.ln2 = RMSNorm(d), RMSNorm(d)
        self.attn = Attention(d, n_head, dropout, qk_norm)
        self.ffn = SwiGLU(d, hidden, dropout)

    def forward(self, x, cos, sin, mask=None):
        a, kv = self.attn(self.ln1(x), cos, sin, mask)
        x = x + a
        return x + self.ffn(self.ln2(x)), kv


class Reader(nn.Module):
    arch = 'v10'

    def __init__(self, vocab_size: int, block_size: int, n_layer: int = 6, n_head: int = 6, n_embd: int = 384,
                 dropout: float = 0.0, copy: bool = True, d_copy: int = 128, rope_base: float = 10000.0, arch='v10',
                 n_loop: int = 1, qk_norm: bool = False, prefix_lm: bool = False):
        super().__init__()
        self.n_loop = n_loop   # looped reader: the whole stack runs n_loop times with shared weights (Universal Transformer)
        self.prefix_lm = prefix_lm   # the question attends both ways (prefix LM, T5/UL2); facts stay causal
        self.c_id = None   # id of '<c>' (end of the question); set by whoever loads the vocabulary
        self.block_size, self.copy, self.d_copy = block_size, copy, d_copy
        hidden = int(8 * n_embd / 3 + 63) // 64 * 64
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([Block(n_embd, n_head, hidden, dropout, qk_norm) for _ in range(n_layer)])
        self.loop_emb = nn.Parameter(torch.zeros(n_loop, n_embd)) if n_loop > 1 else None   # which pass the shared blocks are on
        self.exit_gate = nn.Linear(n_embd, 1) if n_loop > 1 else None   # lambda_r: probability of stopping after loop r (PonderNet)
        self.loops_now = n_loop   # loops in use (set_loops): fewer while a loop curriculum ramps up, or to decode faster
        self.grad_loops = 0   # training: only the last grad_loops loops get gradients (0: all); set by train_reader
        self.neft = 0.0   # NEFTune noise on input embeddings while training (Jain et al. 2023); set by train_reader
        self.loop_noise = 0.0   # relative noise added at each later loop's start at test time (reader.generate samples)
        self.ln_f = RMSNorm(n_embd)
        if copy:
            self.copy_q = nn.Linear(n_embd, d_copy, bias=False)
            self.copy_k = nn.Linear(n_embd, d_copy, bias=False)
            self.gate = nn.Linear(n_embd, 1)
        dh = n_embd // n_head
        inv = 1.0 / (rope_base ** (torch.arange(0, dh, 2).float() / dh))
        f = torch.outer(torch.arange(block_size).float(), inv)
        self.register_buffer('cos', f.cos(), persistent=False)
        self.register_buffer('sin', f.sin(), persistent=False)
        self.apply(self._init)
        for n, p in self.named_parameters():   # residual projections start small (GPT-2 scaling)
            if n.endswith(('proj.weight', 'w2.weight')):
                nn.init.normal_(p, 0.0, 0.02 / math.sqrt(2 * n_layer * n_loop))

    def attn_mask(self, idx):
        """None (causal) or, for a prefix LM, a (B, 1, T, T) mask: the question (up to '<c>') sees all of itself."""
        if not self.prefix_lm or self.c_id is None:
            return None
        T = idx.size(1)
        c = (idx == self.c_id).float().argmax(1)[:, None, None]
        pos = torch.arange(T, device=idx.device)
        i, j = pos[None, :, None], pos[None, None, :]
        return ((j <= i) | ((j <= c) & (i <= c)))[:, None]

    def passes(self):
        """(block, loop index, starts a loop) for every layer application, in order."""
        return [(blk, r, i == 0) for r in range(self.loops_now) for i, blk in enumerate(self.blocks)]

    def _loop_in(self, x, r):
        """Start of loop r: its embedding, plus small noise at test time when loop_noise is set (several different
        readings from one looped reader, for the engine to vote on)."""
        x = x + self.loop_emb[min(r, self.loop_emb.size(0) - 1)]   # past the trained depth: the last loop's embedding
        if self.loop_noise and r > 0 and not self.training:
            x = x + torch.randn_like(x) * self.loop_noise * x.float().pow(2).mean(-1, keepdim=True).sqrt().to(x.dtype)
        return x

    def set_loops(self, r: int):
        self.loops_now = max(1, min(int(r), self.n_loop))

    def _embed(self, idx):
        x = self.tok_emb(idx)
        if self.training and self.neft:
            x = x + (torch.rand_like(x) * 2 - 1) * (self.neft / math.sqrt(idx.size(1) * x.size(-1)))
        return self.drop(x)

    def hidden_loops(self, idx):
        """Normalised hidden states after every loop in use (deep supervision) and lambda_r per position."""
        T = idx.size(1)
        x = self._embed(idx)
        cos, sin = self.cos[:T], self.sin[:T]
        hs, lams = [], []
        am = self.attn_mask(idx)
        cut = self.loops_now - self.grad_loops if self.training and self.grad_loops else 0
        for r in range(self.loops_now):
            # truncated backprop through depth (Huginn; retrofitted recurrence): any depth costs grad_loops loops of memory
            with torch.set_grad_enabled(torch.is_grad_enabled() and r >= cut):
                if self.loop_emb is not None:
                    x = self._loop_in(x, r)
                for blk in self.blocks:
                    x, _ = blk(x, cos, sin, am)
                h = self.ln_f(x)
            hs.append(h)   # gate and heads below still learn from early loops
            lams.append(torch.sigmoid(self.exit_gate(h).float())[..., 0] if self.exit_gate is not None else torch.ones(h.shape[:-1], device=h.device))
        return hs, lams

    def fact_logprobs_loops(self, idx, mask, c_id: int):
        """fact_logprobs after every loop: ([log-probs (N, V)] per loop, [lambda (N,)] per loop, last hidden states)."""
        hs, lams = self.hidden_loops(idx)
        b = mask.nonzero()[:, 0]
        T = idx.size(1)
        c = (idx == c_id).float().argmax(1)
        pos = torch.arange(T, device=idx.device)
        ok = (pos[None, :] >= 1) & (pos[None, :] < c[b][:, None])
        out = []
        for h in hs:
            hv = h[mask]
            logp_v = torch.log_softmax(self.head(hv).float(), -1)
            out.append(self._mix(logp_v, hv, self.copy_k(h)[b], idx[b], ok) if self.copy else logp_v)
        return out, [lam[mask] for lam in lams], hs[-1]

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, 0.0, 0.02)

    def head(self, h):
        return F.linear(h, self.tok_emb.weight)

    def hidden(self, idx):
        T = idx.size(1)
        x = self._embed(idx)
        cos, sin = self.cos[:T], self.sin[:T]
        am = self.attn_mask(idx)
        for blk, r, first in self.passes():
            if first and self.loop_emb is not None:
                x = self._loop_in(x, r)
            x, _ = blk(x, cos, sin, am)
        return self.ln_f(x)

    def _mix(self, logp_v, h, keys, ids, ok):
        """log of g * vocab + (1 - g) * copy, copy attending over the question's tokens (ok marks them)."""
        if not self.copy:
            return logp_v
        s = torch.einsum('nd,ntd->nt', self.copy_q(h).float(), keys.float()) / math.sqrt(self.d_copy)
        pc = torch.softmax(s.masked_fill(~ok, float('-inf')), -1)
        g = torch.sigmoid(self.gate(h).float())
        p = (g * logp_v.exp()).scatter_add(1, ids, (1 - g) * pc)
        return (p + 1e-9).log()

    def fact_logprobs(self, idx, mask, c_id: int):
        """Log-probabilities over the vocabulary at the positions in `mask` (the facts), copy included.
        Returns (log-probs (N, V), hidden states (B, T, d))."""
        h = self.hidden(idx)
        hv = h[mask]
        logp_v = torch.log_softmax(self.head(hv).float(), -1)
        if not self.copy:
            return logp_v, h
        b = mask.nonzero()[:, 0]
        T = idx.size(1)
        c = (idx == c_id).float().argmax(1)
        pos = torch.arange(T, device=idx.device)
        ok = (pos[None, :] >= 1) & (pos[None, :] < c[b][:, None])
        return self._mix(logp_v, hv, self.copy_k(h)[b], idx[b], ok), h

    # ---- cached decoding (one question per row, right-padded prefixes of true lengths plen) ----

    def prefill(self, idx, plen):
        """Keys/values of every layer for the prefixes, the copy keys (one extra cache entry) and the
        log-probabilities of the first fact token (at each row's last real position)."""
        B, P = idx.shape
        x = self.tok_emb(idx)
        cos, sin = self.cos[:P], self.sin[:P]
        cache = []
        am = None
        if self.prefix_lm:   # the prompt is all question: every real position sees every other
            real = torch.arange(P, device=idx.device)[None, :] < plen[:, None]
            am = (real[:, None, :] & real[:, :, None]) | torch.eye(P, dtype=torch.bool, device=idx.device)[None]
            am = am[:, None]
        for blk, r, first in self.passes():
            if first and self.loop_emb is not None:
                x = self._loop_in(x, r)
            x, kv = blk(x, cos, sin, am)
            cache.append(list(kv))
        h = self.ln_f(x)
        rows = torch.arange(B, device=idx.device)
        last = h[rows, plen - 1]
        cache.append([self.copy_k(h) if self.copy else h[..., :1], idx])
        return cache, self._step_logp(last, cache, plen, P)

    def _step_logp(self, h, cache, plen, P):
        logp_v = torch.log_softmax(self.head(h).float(), -1)
        if not self.copy:
            return logp_v
        keys, ids = cache[-1]
        pos = torch.arange(P, device=h.device)
        ok = (pos[None, :] >= 1) & (pos[None, :] < plen[:, None] - 1)   # the question: after <bos>, before <c>
        return self._mix(logp_v, h, keys, ids, ok)

    def step(self, cache, tok, pos, plen, P, G):
        """One new token per row at position `pos`, attending to its prefix (first plen of P) and G written tokens.
        Returns the next token's log-probabilities; the layer caches grow in place."""
        x = self.tok_emb(tok[:, None])
        p = pos.clamp(max=self.block_size - 1)
        cos, sin = self.cos[p][:, None, None, :], self.sin[p][:, None, None, :]
        mask = torch.cat([torch.arange(P, device=tok.device)[None, :] < plen[:, None],
                          torch.ones(len(tok), G, dtype=torch.bool, device=tok.device)], 1)[:, None, None, :]
        for li, (blk, r, first) in enumerate(self.passes()):
            if first and self.loop_emb is not None:
                x = self._loop_in(x, r)
            q, k, v = blk.attn.heads(blk.ln1(x), cos, sin)
            cache[li][0] = torch.cat([cache[li][0], k], 2)
            cache[li][1] = torch.cat([cache[li][1], v], 2)
            y = F.scaled_dot_product_attention(q, cache[li][0], cache[li][1], attn_mask=mask)
            x = x + blk.attn.proj(y.transpose(1, 2).reshape(x.shape))
            x = x + blk.ffn(blk.ln2(x))
        return self._step_logp(self.ln_f(x)[:, 0], cache, plen, P)


def make_model(config: dict):
    """The model a checkpoint's config describes: the v10 Reader or the original GPT."""
    if config.get('arch') == 'v10':
        return Reader(**config)
    from model_gpt import GPT
    return GPT(**{k: v for k, v in config.items() if k != 'arch'})

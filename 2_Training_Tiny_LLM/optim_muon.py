"""Muon (Jordan et al. 2024; scaled up in Moonlight / Kimi K2): momentum, then the update of each 2-D hidden matrix is
orthogonalised by a Newton-Schulz iteration. Groups with use_muon=False (embeddings, norms, gates, heads) get AdamW.
One optimizer, so GradScaler and the learning-rate schedule treat it like AdamW."""
from __future__ import annotations

import torch


@torch.no_grad()
def orthogonalize(g: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Approximately U V^T of g = U S V^T (quintic Newton-Schulz, coefficients from Jordan et al.)."""
    a, b, c = 3.4445, -4.7750, 2.0315
    x = g.bfloat16() if g.is_cuda else g.float()
    tall = x.size(0) > x.size(1)
    if tall:
        x = x.T
    x = x / (x.norm() + 1e-7)
    for _ in range(steps):
        s = x @ x.T
        x = a * x + (b * s + c * s @ s) @ x
    return (x.T if tall else x).to(g.dtype)


class Muon(torch.optim.Optimizer):
    def __init__(self, groups, lr=0.02, momentum=0.95, weight_decay=0.0, betas=(0.9, 0.95), eps=1e-8):
        for g in groups:
            g.setdefault('use_muon', False)
        super().__init__(groups, dict(lr=lr, momentum=momentum, weight_decay=weight_decay, betas=betas, eps=eps))

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for g in self.param_groups:
            for p in g['params']:
                if p.grad is None:
                    continue
                st = self.state[p]
                if g['use_muon']:
                    buf = st.setdefault('momentum', torch.zeros_like(p))
                    buf.mul_(g['momentum']).add_(p.grad)
                    u = orthogonalize(p.grad.add(buf, alpha=g['momentum']))   # Nesterov
                    u *= max(1.0, p.size(0) / p.size(1)) ** 0.5
                    p.mul_(1 - g['lr'] * g['weight_decay']).add_(u, alpha=-g['lr'])
                else:
                    if not st:
                        st['step'], st['m'], st['v'] = 0, torch.zeros_like(p), torch.zeros_like(p)
                    st['step'] += 1
                    b1, b2 = g['betas']
                    st['m'].lerp_(p.grad, 1 - b1)
                    st['v'].lerp_(p.grad.square(), 1 - b2)
                    mh = st['m'] / (1 - b1 ** st['step'])
                    vh = st['v'] / (1 - b2 ** st['step'])
                    p.mul_(1 - g['lr'] * g['weight_decay']).addcdiv_(mh, vh.sqrt().add_(g['eps']), value=-g['lr'])
        return loss

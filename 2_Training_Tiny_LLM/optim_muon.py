"""NorMuon with cautious weight decay (the IMU-1 recipe, 2602.02522: -2.9% and -1% training loss at 70M): hidden
matrices get momentum, a Newton-Schulz orthogonalised update, per-neuron second-moment normalisation (NorMuon,
Li et al. 2025) and an RMS of 0.2 so AdamW's learning rate and weight decay carry over (Moonlight). Groups with
use_muon=False (embeddings, norms, gates, heads) get AdamW. Weight decay is cautious (Chen et al. 2025): applied only
where it points the same way as the update. One optimizer, so GradScaler and the schedule treat it like AdamW."""
from __future__ import annotations

import torch


@torch.no_grad()
def orthogonalize(g: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Approximately U V^T of g = U S V^T (quintic Newton-Schulz, coefficients from Jordan et al. 2024)."""
    a, b, c = 3.4445, -4.7750, 2.0315
    x = g.bfloat16() if g.is_cuda else g.float()
    tall = x.size(0) > x.size(1)
    if tall:
        x = x.T
    x = x / (x.norm() + 1e-7)
    for _ in range(steps):
        s = x @ x.T
        x = a * x + (b * s + c * s @ s) @ x
    return (x.T if tall else x).float()


class Muon(torch.optim.Optimizer):
    def __init__(self, groups, lr=3e-4, momentum=0.95, weight_decay=0.1, betas=(0.9, 0.95), eps=1e-8, cautious=True):
        for g in groups:
            g.setdefault('use_muon', False)
        super().__init__(groups, dict(lr=lr, momentum=momentum, weight_decay=weight_decay, betas=betas, eps=eps, cautious=cautious))

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for g in self.param_groups:
            b1, b2 = g['betas']
            for p in g['params']:
                if p.grad is None:
                    continue
                st = self.state[p]
                if g['use_muon']:
                    buf = st.setdefault('momentum', torch.zeros_like(p))
                    buf.mul_(g['momentum']).add_(p.grad)
                    u = orthogonalize(p.grad.add(buf, alpha=g['momentum']))   # Nesterov
                    v = st.setdefault('row_v', torch.zeros(p.size(0), 1, device=p.device))
                    v.lerp_(u.square().mean(1, keepdim=True), 1 - b2)   # NorMuon: per-neuron scale
                    u = u / (v.sqrt() + g['eps'])
                    u = (u * (0.2 * u.numel() ** 0.5 / (u.norm() + 1e-7))).to(p.dtype)   # RMS 0.2 = AdamW's
                else:
                    if 'm' not in st:
                        st['step'], st['m'], st['v'] = 0, torch.zeros_like(p), torch.zeros_like(p)
                    st['step'] += 1
                    st['m'].lerp_(p.grad, 1 - b1)
                    st['v'].lerp_(p.grad.square(), 1 - b2)
                    u = (st['m'] / (1 - b1 ** st['step'])) / ((st['v'] / (1 - b2 ** st['step'])).sqrt() + g['eps'])
                wd = g['weight_decay'] * p
                if g['cautious']:
                    wd = wd * (u * p > 0)   # decay only where the update already moves toward zero
                p.add_(u + wd, alpha=-g['lr'])
        return loss

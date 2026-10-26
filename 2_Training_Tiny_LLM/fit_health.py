"""Training health, checked at every evaluation from numbers the run already has: catches overfitting, underfitting
and style memorization (synthetic scores rising while real questions fall) while a run trains. Runs are judged by
real dev questions; synthetic held-out numbers only explain why a run moves.

Each evaluation appends {step, train_loss, heldout_loss, synth, real} to the history.
"""
from __future__ import annotations

import torch

NOISE = 0.02     # real dev of ~300 questions moves about +-2 points between evaluations by chance
HIGH_LOSS = 0.05  # fact loss (per token) still this high means the reader has not learnt the training data


def require_finite_loss(loss, step: int, facts=None, words=None) -> None:
    """Stop before backpropagating a non-finite scalar loss; report its components for diagnosis."""
    if torch.isfinite(loss.detach()).all().item():
        return
    parts = []
    for name, value in (("facts", facts), ("words", words)):
        if value is not None:
            parts.append(f"{name}={value.detach().float().item():.6g}")
    detail = f" ({', '.join(parts)})" if parts else ""
    raise FloatingPointError(f"non-finite training loss at step {step}{detail}")


def diagnose(hist: list[dict]) -> list[str]:
    """Problems visible in the last three evaluations (two in a row, so one noisy evaluation never triggers)."""
    if len(hist) < 3:
        return []
    a, b, c = hist[-3:]
    flags = []
    if all(h.get('train_loss') is not None and h.get('heldout_loss') is not None for h in (a, b, c)):
        if c['heldout_loss'] > b['heldout_loss'] > a['heldout_loss'] and c['train_loss'] < a['train_loss']:
            flags.append('overfitting: held-out loss rising while training loss falls')
        if c['train_loss'] > HIGH_LOSS and c['train_loss'] > 0.95 * a['train_loss']:
            flags.append('underfitting: training loss high and flat (more steps, a higher rate or more capacity)')
    reals = [h['real'] for h in hist if h.get('real') is not None]
    if c.get('real') is not None and b.get('real') is not None and None not in (a.get('synth'), c.get('synth')):
        top = max(reals)
        if c['real'] < top - NOISE and b['real'] < top - NOISE and c['synth'] >= a['synth']:
            flags.append('style memorization: synthetic held-out steady or up, real dev down')
    if c.get('real') is not None and c.get('synth') is not None and c['synth'] - c['real'] > 0.25:
        flags.append(f"generalization gap {c['synth'] - c['real']:.0%} (synthetic {c['synth']:.1%} vs real {c['real']:.1%})")
    return flags


def must_stop(hist: list[dict]) -> bool:
    """Stop (and keep the best real-dev checkpoint) after two evaluations in a row show style memorization."""
    if len(hist) < 4:
        return False
    return all(any(f.startswith(('style memorization', 'overfitting')) for f in diagnose(hist[:n])) for n in (len(hist) - 1, len(hist)))

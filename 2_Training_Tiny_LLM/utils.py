from __future__ import annotations
import torch

def top_k_top_p_filtering(logits: torch.Tensor, top_k: int | None = None, top_p:float | None = None):
    """
        Filter a distribution of logits using top-k and/or  top-p (nucleus filtering)
        logits: (B, vocab)
        Returns filtered logits with -inf for masked entries since softmax(-inf) = 0
    """

    B, V = logits.shape
    filtered = logits.clone()

    if top_k is not None and 0 < top_k < V:
        topk_vals, _ = torch.topk(filtered, top_k, dim=-1)
        kth = topk_vals[:, -1].unsqueeze(-1)
        filtered[filtered < kth] = float('-inf')

    if top_p is not None and 0 < top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(filtered, descending=True, dim=-1)
        probs = torch.softmax(sorted_logits, dim=-1)
        cumu_sum = torch.cumsum(probs, dim=-1)

        mask = cumu_sum > top_p
        # Shift right: keep the token that crosses top_p
        mask[..., 1:] = mask[..., :-1].clone()

        # Keep atleast 1 token
        mask[..., 0] = False
        sorted_logits[mask] = float('-inf')

        # Scatter back
        filtered = torch.full_like(filtered, float('-inf'))
        filtered.scatter_(1, sorted_idx, sorted_logits)

    return filtered
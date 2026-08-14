import torch
from attn_mask import causal_mask

def test_causal_mask_shape_and_values():
    T = 4
    mask = causal_mask(T)
    assert mask.shape == (1, 1, T, T)
    expected = torch.tensor([
        [False,  True,  True,  True],
        [False, False,  True,  True],
        [False, False, False,  True],
        [False, False, False, False]
    ])
    assert torch.equal(mask[0, 0], expected)

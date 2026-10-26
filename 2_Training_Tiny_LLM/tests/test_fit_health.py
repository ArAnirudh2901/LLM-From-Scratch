import pytest
import torch

from fit_health import require_finite_loss


def test_finite_loss_is_allowed():
    require_finite_loss(torch.tensor(0.25, requires_grad=True), step=12)


def test_nonfinite_loss_stops_with_component_diagnostics():
    with pytest.raises(FloatingPointError, match=r"step 12.*facts=nan.*words=0.25"):
        require_finite_loss(torch.tensor(float("nan")), step=12,
                            facts=torch.tensor(float("nan")), words=torch.tensor(0.25))


def test_gradient_clipping_rejects_nonfinite_gradients():
    p = torch.nn.Parameter(torch.tensor([1.0]))
    (p * torch.tensor(float("nan"))).sum().backward()
    with pytest.raises(RuntimeError, match="non-finite"):
        torch.nn.utils.clip_grad_norm_([p], 1.0, error_if_nonfinite=True)

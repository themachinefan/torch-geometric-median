import pytest
import torch
from geom_median.torch import (
    compute_geometric_median as original_compute_geometric_median,
)
from torch_geometric_median.geometric_median import (
    TerminationReason,
    geometric_median,
)


@pytest.mark.parametrize(
    "dim1, dim2, sample_mag, maxiter, use_weights",
    [
        (100, 768, 100, 100, False),
        (100, 768, 1, 10, True),
        (10000, 768, 5, 100, False),
        (10000, 768, 20, 10, True),
    ],
)
def test_geometric_median_output_matches_original_lib(
    dim1: int, dim2: int, sample_mag: int, maxiter: int, use_weights: bool
):
    TOLERANCE = 1e-3

    samples1 = torch.randn((dim1, dim2)) * sample_mag
    samples2 = samples1.clone().detach()
    samples1.requires_grad = True
    samples2.requires_grad = True
    weights = torch.randn((dim1,)) if use_weights else None

    new = geometric_median(samples1, weights=weights, maxiter=maxiter)
    old = original_compute_geometric_median(
        samples2,
        weights=weights,
        skip_typechecks=True,
        maxiter=maxiter,
        per_component=False,
    )

    torch.linalg.norm(old.median).backward()
    torch.linalg.norm(new.median).backward()

    assert torch.allclose(new.median, old.median, atol=TOLERANCE)
    assert torch.allclose(new.new_weights, old.new_weights, atol=TOLERANCE)
    assert samples1.grad is not None
    assert samples2.grad is not None
    assert samples1.grad.norm().item() > 0.001
    assert torch.allclose(samples1.grad, samples2.grad, atol=TOLERANCE)


def test_geometric_median_terminates_early_based_on_ftol():
    samples = torch.randn((10000, 768)) * 100
    high_ftol_result = geometric_median(samples, maxiter=10, ftol=1e-3)
    assert high_ftol_result.termination_reason == TerminationReason.CONVERGED


def test_geometric_median_minimizes_sum_of_l2_norms():
    samples = torch.randn((100, 768)) * 100
    median = geometric_median(samples, maxiter=10).median
    other_vecs = [median + torch.randn_like(median) for _ in range(100)]
    other_vecs.append(samples.mean(dim=0))

    def l2_norm_sum(vec: torch.Tensor) -> torch.Tensor:
        return torch.linalg.norm(samples - vec.view(1, -1), dim=1).sum()

    median_l2_norm_sum = l2_norm_sum(median)
    for vec in other_vecs:
        assert median_l2_norm_sum < l2_norm_sum(vec)

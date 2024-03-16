from dataclasses import dataclass
from enum import StrEnum

import torch
import tqdm


def weighted_average(points: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    weights = weights / weights.sum()
    return (points * weights.view(-1, 1)).sum(dim=0)


@torch.no_grad()
def geometric_median_objective(
    median: torch.Tensor, points: torch.Tensor, weights: torch.Tensor
) -> torch.Tensor:
    norms = torch.linalg.norm(points - median.view(1, -1), dim=1)
    return (norms * weights).sum()


class TerminationReason(StrEnum):
    CONVERGED = "converged"
    MAX_ITERATIONS = "max_iterations"


@dataclass
class GeometricMedianResult:
    median: torch.Tensor
    new_weights: torch.Tensor
    termination_reason: TerminationReason
    objective_values_log: list[torch.Tensor] | None


def geometric_median(
    points: torch.Tensor,
    weights: torch.Tensor | None = None,
    eps: float = 1e-6,
    maxiter: int = 100,
    ftol: float = 1e-20,
    log_objective_values: bool = False,
    show_progress: bool = False,
) -> GeometricMedianResult:
    """
    :param points: ``torch.Tensor`` of shape ``(n, d)``
    :param weights: Optional ``torch.Tensor`` of shape :math:``(n,)``.
    :param eps: Smallest allowed value of denominator, to avoid divide by zero.
        Equivalently, this is a smoothing parameter. Default 1e-6.
    :param maxiter: Maximum number of Weiszfeld iterations. Default 100
    :param ftol: If objective value does not improve by at least this `ftol` fraction, terminate the algorithm. Default 1e-20.
    :param log_objective_values: If true will return a log of function values encountered through the course of the algorithm
    :param show_progress: If True, will display a progress bar. Default False.
    :return: GeometricMedianResult object with fields
        - `median`: estimate of the geometric median, which is a ``torch.Tensor`` object of shape :math:``(d,)``
        - `new_weights`: updated weights after the algorithm, which is a ``torch.Tensor`` object of shape :math:``(n,)``
        - `termination_reason`: enum explaining how the algorithm terminated.
        - `objective_values_log`: function values encountered through the course of the algorithm in a list (None if log_objective_values is false).
    """
    with torch.no_grad():
        if weights is None:
            weights = torch.ones((points.shape[0],), device=points.device)
        # initialize median estimate at mean
        new_weights = weights
        median = weighted_average(points, weights)
        objective_value = geometric_median_objective(median, points, weights)
        if log_objective_values:
            logs = [objective_value]
        else:
            logs = None

        # Weiszfeld iterations
        early_termination = False
        pbar = tqdm.tqdm(range(maxiter), disable=not show_progress)
        for _ in pbar:
            prev_obj_value = objective_value

            norms = torch.linalg.norm(points - median.view(1, -1), dim=1)
            new_weights = weights / torch.clamp(norms, min=eps)
            median = weighted_average(points, new_weights)
            objective_value = geometric_median_objective(median, points, weights)

            if logs is not None:
                logs.append(objective_value)
            if abs(prev_obj_value - objective_value) <= ftol * objective_value:
                early_termination = True
                break

            pbar.set_description(f"Objective value: {objective_value:.4f}")

    median = weighted_average(points, new_weights)  # allow autodiff to track it
    return GeometricMedianResult(
        median=median,
        new_weights=new_weights,
        termination_reason=(
            TerminationReason.CONVERGED
            if early_termination
            else TerminationReason.MAX_ITERATIONS
        ),
        objective_values_log=logs,
    )

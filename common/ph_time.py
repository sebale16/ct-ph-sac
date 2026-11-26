"""Time and entropy conversions; all durations are in seconds."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Tuple

import numpy as np
import torch

from common.ph_config import ClockConfig

if TYPE_CHECKING:
    from torch import Tensor


def discount_weight(durations: Tensor, *, discount_rate: float) -> Tensor:
    """w_lambda(Delta) = (1 - exp(-lambda Delta)) / lambda."""
    return -torch.expm1(-discount_rate * durations) / discount_rate


def interval_temperature(
    alpha_c: Tensor, durations: Tensor, *, discount_rate: float
) -> Tensor:
    """Charge alpha_c times w_lambda(Delta) exactly once per decision."""
    return alpha_c * discount_weight(durations, discount_rate=discount_rate)


def clock_duration_pmf(clock: ClockConfig) -> Tuple[np.ndarray, np.ndarray]:
    """Exact law of the unshortened clock: whole physics steps and their probabilities.

    The two-tailed laws match environment.base.sample_dt_from_distribution: endpoint
    mass tail_probability, split by tail_split, and the rest on interior whole steps.
    """
    dt = clock["physics_dt"]
    low = int(round(clock["min_duration"] / dt))
    high = int(round(clock["max_duration"] / dt))
    if clock["kind"] == "fixed":
        return np.array([low], dtype=np.int64), np.ones(1)

    steps = np.arange(low, high + 1, dtype=np.int64)
    probabilities = np.zeros(len(steps))
    interior = high - low - 1
    tail = clock["tail_probability"] if interior > 0 else 1.0
    probabilities[0] = tail * clock["tail_split"]
    probabilities[-1] += tail * (1.0 - clock["tail_split"])
    if interior <= 0:
        return steps, probabilities

    if clock["kind"] == "two_tail_uniform":
        probabilities[1:-1] = (1.0 - tail) / interior
    else:
        from scipy.stats import beta

        # Beta(a, a) on [0, 1] is rounded onto the interior whole steps, as when sampling.
        shape = clock["beta_shape"]
        span = interior - 1
        if span == 0:
            probabilities[1] = 1.0 - tail
        else:
            edges = (np.arange(span + 2) - 0.5) / span
            edges[0], edges[-1] = 0.0, 1.0
            probabilities[1:-1] = (1.0 - tail) * np.diff(beta.cdf(edges, shape, shape))
    return steps, probabilities


def expected_discount_weight(clock: ClockConfig, *, discount_rate: float) -> float:
    """E[w_lambda(Delta)] in seconds under the unshortened clock law (clock_duration_pmf)."""
    steps, probabilities = clock_duration_pmf(clock)
    durations = steps * clock["physics_dt"]
    weights = -np.expm1(-discount_rate * durations) / discount_rate
    return float(np.dot(probabilities, weights))


def mean_decision_interval(clock: ClockConfig) -> float:
    """E[Delta] in seconds under clock_duration_pmf's law of the unshortened clock.

    The reference period on a fixed clock; 0.4912 T on the irregular rows' two-tailed law
    (14.736 ms at the lqr tasks' T = 30 ms). An episode's shortened last interval is not
    in the law.
    """
    steps, probabilities = clock_duration_pmf(clock)
    return float(np.dot(probabilities, steps * clock["physics_dt"]))


def period_discount(*, discount_rate: float, reference_period: float) -> float:
    """Convert to gamma = exp(-lambda T), keeping T distinct from resolvent tau."""
    return math.exp(-discount_rate * reference_period)

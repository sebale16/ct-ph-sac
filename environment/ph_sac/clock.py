"""Exogenous decision clocks quantized to whole MuJoCo physics steps."""

from __future__ import annotations

from typing import TYPE_CHECKING, Mapping, Optional

import numpy as np

from common.ph_time import clock_duration_pmf, expected_discount_weight

if TYPE_CHECKING:
    from common.ph_config import ClockConfig


class DiscreteIntervalClock:
    """Fixed Acrobot periods or the Windy Pendulum two-tailed duration law."""

    def __init__(self, config: ClockConfig, *, seed: Optional[int] = None) -> None:
        self.config = config
        self.physics_dt = float(config["physics_dt"])
        self.steps, self.probabilities = clock_duration_pmf(config)
        self._rng = np.random.default_rng(seed)

    def reset(self, *, seed: Optional[int] = None) -> None:
        if seed is not None:
            self._rng = np.random.default_rng(seed)

    def sample_steps(self, *, remaining_steps: Optional[int] = None) -> int:
        """Read no task state or decision; shorten only at the episode limit."""
        if remaining_steps is not None and remaining_steps < 1:
            raise ValueError("The clock needs at least one remaining physics step.")
        steps = int(self._rng.choice(self.steps, p=self.probabilities))
        return steps if remaining_steps is None else min(steps, int(remaining_steps))

    def sample_duration(self, *, remaining_steps: Optional[int] = None) -> float:
        return self.sample_steps(remaining_steps=remaining_steps) * self.physics_dt

    def expected_discount_weight(self, discount_rate: float) -> float:
        """Read E[(1-exp(-lambda*Delta))/lambda] from the unshortened clock law.

        Episode-end shortening still requires each actual interval's entropy charge.
        """
        return expected_discount_weight(self.config, discount_rate=discount_rate)

    def state_dict(self) -> Mapping[str, object]:
        return {"rng": self._rng.bit_generator.state}

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        self._rng.bit_generator.state = state["rng"]


def duration_in_physics_steps(duration: float, physics_dt: float) -> int:
    """Require a positive whole-step duration rather than silently rounding."""
    ratio = float(duration) / float(physics_dt)
    steps = int(round(ratio))
    if steps < 1 or abs(ratio - steps) > 1e-8:
        raise ValueError(
            f"Duration {duration} is not a positive whole number of "
            f"physics steps of {physics_dt} s."
        )
    return steps

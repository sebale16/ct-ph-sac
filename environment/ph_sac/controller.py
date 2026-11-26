"""Decision-to-torque laws shared by environments, plant fits and model paths.

FeedbackController is the bounded impedance law of the impedance-law ablation arms
(decision_mode "feedback"); TorqueController is the identity of the paper arms
(decision_mode "torque"), whose decision is the held torque. make_controller picks one
from ControllerConfig["decision_mode"]. Both expose the same interface (decision_dim,
refresh_steps, refresh_mode, continuous, state_feedback, resolve_decision, action,
virtual_spring_energy, their numpy versions, refresh_due and refresh_times), so every PH
component reads the decision dimension and the box through the controller and its config.

refresh_mode is ControllerConfig["refresh_mode"] (common.ph_config.REFRESH_MODES).
"continuous": a feedback law is evaluated from the current state at every integrator
stage, by the environment's MuJoCo control callback and by the model's RK4 stages
(models/ph_sac/rollouts.py), so refresh_due and refresh_times describe only the "held"
mode, whose torque is refreshed every refresh_steps physics steps and held between.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Tuple, Union

import numpy as np
import torch

from common.ph_config import decision_mode_setting, refresh_mode_setting

if TYPE_CHECKING:
    from numpy import ndarray
    from torch import Tensor

    from common.ph_config import ControllerConfig
    from common.ph_types import StateSpec


class FeedbackController:
    """Held decision with sampled refreshes and task-specific angle errors.

    A decision is (q_star, k_p, k_d), each of length m, so its last dimension is 3m.
    All methods accept any leading batch dimensions shared by states and decisions.

    Sampled cadence and failure handling need their own holding-consistency
    estimate; a continuously refreshed, nonterminating bound is not implied.

    state_feedback is True: the torque depends on the state. continuous (refresh_mode
    "continuous") says that the environment and the model integrators evaluate it at
    every integrator stage; otherwise ("held") they re-evaluate it at every refresh of
    the grid and hold it between.
    """

    state_feedback = True

    def __init__(self, config: ControllerConfig, state_spec: StateSpec) -> None:
        if decision_mode_setting(config) != "feedback":
            raise ValueError("FeedbackController needs a feedback-decision controller config.")
        self.config = config
        self.state_spec = state_spec
        self.action_dim = int(state_spec["action_dim"])
        self.decision_dim = 3 * self.action_dim
        self.refresh_steps = int(config["refresh_steps"])
        self.refresh_mode = refresh_mode_setting(config)
        self.continuous = self.refresh_mode == "continuous"
        self.setpoint_mode = config["setpoint_mode"]

        positions = tuple(int(i) for i in state_spec["actuated_position_indices"])
        if len(positions) != self.action_dim:
            raise ValueError("Need one actuated position per action dimension.")
        projection = np.asarray(state_spec["observation_projection"])
        dofs = [int(np.argmax(np.abs(projection[i]))) for i in positions]
        periodic = set(state_spec["periodic_position_indices"])

        self._positions = np.array(positions, dtype=np.int64)
        self._velocities = int(state_spec["position_dim"]) + np.array(
            dofs, dtype=np.int64
        )
        self._wrapped = np.array(
            [config["setpoint_error"] == "wrapped" and i in periodic for i in positions],
            dtype=bool,
        )
        self._torque_low = np.asarray(config["torque_low"], dtype=np.float64)
        self._torque_high = np.asarray(config["torque_high"], dtype=np.float64)
        # Index and limit tensors per device and dtype: the law runs at every RK4 stage of
        # every model substep under refresh_mode "continuous".
        self._tensors: dict = {}

    def _split(self, decisions: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        m = self.action_dim
        return decisions[..., :m], decisions[..., m : 2 * m], decisions[..., 2 * m :]

    def _cached(self, name: str, reference: Tensor, dtype: object = None) -> Tensor:
        key = (name, reference.device, dtype)
        tensor = self._tensors.get(key)
        if tensor is None:
            tensor = torch.as_tensor(getattr(self, name), dtype=dtype, device=reference.device)
            self._tensors[key] = tensor
        return tensor

    def _indices(self, array: ndarray, reference: Tensor) -> Tensor:
        for name in ("_positions", "_velocities", "_wrapped"):
            if array is getattr(self, name):
                return self._cached(name, reference)
        return torch.as_tensor(array, device=reference.device)

    def resolve_decision(self, states: Tensor, decisions: Tensor) -> Tensor:
        """Freeze an absolute target at interval start, including relative offsets."""
        if self.setpoint_mode == "absolute":
            return decisions
        setpoint, stiffness, damping = self._split(decisions)
        positions = states[..., self._indices(self._positions, states)]
        return torch.cat((positions + setpoint, stiffness, damping), dim=-1)

    def action(self, states: Tensor, resolved_decisions: Tensor) -> Tensor:
        """Differentiate through clipped PD torque using the fixed setpoint."""
        _, stiffness, damping = self._split(resolved_decisions)
        error = self.position_error(states, resolved_decisions)
        velocities = states[..., self._indices(self._velocities, states)]
        torque = -stiffness * error - damping * velocities
        low = self._cached("_torque_low", torque, torque.dtype)
        high = self._cached("_torque_high", torque, torque.dtype)
        return torch.min(torch.max(torque, low), high)

    def position_error(self, states: Tensor, resolved_decisions: Tensor) -> Tensor:
        """Wrap errors only for coordinates with periodic setpoint semantics."""
        setpoint, _, _ = self._split(resolved_decisions)
        error = states[..., self._indices(self._positions, states)] - setpoint
        if not self._wrapped.any():
            return error
        wrapped = torch.remainder(error + math.pi, 2.0 * math.pi) - math.pi
        return torch.where(self._indices(self._wrapped, states), wrapped, error)

    def virtual_spring_energy(
        self, states: Tensor, resolved_decisions: Tensor
    ) -> Tensor:
        """H_c = sum_i k_p,i e_i^2 / 2, returned with a trailing dimension of one."""
        _, stiffness, _ = self._split(resolved_decisions)
        error = self.position_error(states, resolved_decisions)
        return 0.5 * (stiffness * error.square()).sum(dim=-1, keepdim=True)

    def resolve_decision_numpy(self, state: ndarray, decision: ndarray) -> ndarray:
        decision = np.asarray(decision, dtype=np.float64)
        if self.setpoint_mode == "absolute":
            return decision.copy()
        resolved = decision.copy()
        resolved[..., : self.action_dim] += np.asarray(state)[..., self._positions]
        return resolved

    def action_numpy(self, state: ndarray, resolved_decision: ndarray) -> ndarray:
        state = np.asarray(state, dtype=np.float64)
        return self.actuated_action_numpy(
            state[..., self._positions], state[..., self._velocities], resolved_decision
        )

    @property
    def actuated_positions(self) -> ndarray:
        """Observation indices of the actuated positions the law reads."""
        return self._positions.copy()

    @property
    def actuated_velocities(self) -> ndarray:
        """Observation indices of the actuated velocities the law reads."""
        return self._velocities.copy()

    def actuated_action_numpy(
        self, positions: ndarray, velocities: ndarray, resolved_decision: ndarray
    ) -> ndarray:
        """action_numpy from the actuated coordinates alone, the same arithmetic.

        positions and velocities are state[..., actuated_positions] and
        state[..., actuated_velocities]; the environment's control callback reads them
        straight from MuJoCo's data at every RK4 stage.
        """
        resolved = np.asarray(resolved_decision, dtype=np.float64)
        m = self.action_dim
        setpoint = resolved[..., :m]
        error = positions - setpoint
        wrapped = np.mod(error + math.pi, 2.0 * math.pi) - math.pi
        error = np.where(self._wrapped, wrapped, error)
        torque = -resolved[..., m : 2 * m] * error - resolved[..., 2 * m :] * velocities
        return np.clip(torque, self._torque_low, self._torque_high)

    def refresh_due(self, physics_step: int) -> bool:
        """Apply the same controller cadence in real and learned dynamics ("held" mode).

        Under refresh_mode "continuous" every physics step is a refresh (and the
        environment's callback evaluates the law at every stage in between as well).
        """
        if self.continuous:
            return True
        return physics_step % self.refresh_steps == 0

    def refresh_times(self, duration: float, physics_dt: float) -> Tuple[float, ...]:
        """List refresh times for "held" model grids, including the interval start.

        Model substeps must contain these times, using unequal lengths when needed.
        Under refresh_mode "continuous" the law has no refresh points: only the interval
        start is returned, and the model grid splits at model_step alone.
        """
        if self.continuous:
            return (0.0,)
        total = int(round(duration / physics_dt))
        return tuple(
            step * physics_dt for step in range(0, total, self.refresh_steps)
        )


class TorqueController:
    """Identity law of the torque-decision mode: a decision is the held motor torque.

    A decision is the torque itself, of length m, in the box
    [torque_low, torque_high] that ControllerConfig carries as decision_low/high
    (torque_decision_controller). action returns the decision clipped to the motor
    limits whatever the state, so the torque is set at the interval's start and held,
    as the CT-SAC baseline holds its torque: no setpoint, no stiffness or damping, no
    virtual spring. state_feedback is False, so the model integrators
    (models/ph_sac/rollouts.py) evaluate the torque at each interval's first substep
    only and hold it through every later substep and every RK4 stage; refresh_steps and
    refresh_mode are the config's and only lay out the model's substep grid, which is
    then the feedback arms' grid (split at every refresh under "held", at the model step
    alone under "continuous"). continuous is False whatever the mode: the torque is never
    re-evaluated. All methods accept any leading batch dimensions shared by states and
    decisions.
    """

    state_feedback = False
    continuous = False

    def __init__(self, config: ControllerConfig, state_spec: StateSpec) -> None:
        if decision_mode_setting(config) != "torque":
            raise ValueError("TorqueController needs a torque-decision controller config.")
        self.config = config
        self.state_spec = state_spec
        self.action_dim = int(state_spec["action_dim"])
        self.decision_dim = self.action_dim
        self.refresh_steps = int(config["refresh_steps"])
        self.refresh_mode = refresh_mode_setting(config)
        self.setpoint_mode = config["setpoint_mode"]
        self._torque_low = np.asarray(config["torque_low"], dtype=np.float64).reshape(-1)
        self._torque_high = np.asarray(config["torque_high"], dtype=np.float64).reshape(-1)
        low = np.asarray(config["decision_low"], dtype=np.float64).reshape(-1)
        high = np.asarray(config["decision_high"], dtype=np.float64).reshape(-1)
        if low.size != self.action_dim or self._torque_low.size != self.action_dim:
            raise ValueError("A torque decision needs one torque per actuator.")
        if not (np.array_equal(low, self._torque_low) and np.array_equal(high, self._torque_high)):
            raise ValueError("A torque decision box must be the torque box.")

    def resolve_decision(self, states: Tensor, decisions: Tensor) -> Tensor:
        """A torque needs no resolution at the interval start: return it unchanged."""
        return decisions

    def action(self, states: Tensor, resolved_decisions: Tensor) -> Tensor:
        """The decision clipped to the motor limits, broadcast over the states' batch.

        It does not read the states, so it is constant along any path.
        """
        shape = torch.broadcast_shapes(states.shape[:-1], resolved_decisions.shape[:-1])
        torque = torch.broadcast_to(resolved_decisions, (*shape, self.action_dim))
        low = torch.as_tensor(self._torque_low, dtype=torque.dtype, device=torque.device)
        high = torch.as_tensor(self._torque_high, dtype=torque.dtype, device=torque.device)
        return torch.min(torch.max(torque, low), high)

    def virtual_spring_energy(
        self, states: Tensor, resolved_decisions: Tensor
    ) -> Tensor:
        """H_c = 0: a held torque has no virtual spring; [..., 1] zeros."""
        shape = torch.broadcast_shapes(states.shape[:-1], resolved_decisions.shape[:-1])
        return states.new_zeros((*shape, 1))

    def resolve_decision_numpy(self, state: ndarray, decision: ndarray) -> ndarray:
        return np.array(decision, dtype=np.float64)

    def action_numpy(self, state: ndarray, resolved_decision: ndarray) -> ndarray:
        resolved = np.asarray(resolved_decision, dtype=np.float64)
        return np.clip(resolved, self._torque_low, self._torque_high)

    def refresh_due(self, physics_step: int) -> bool:
        """Only the interval's first physics step sets the torque; it is then held."""
        return physics_step == 0

    def refresh_times(self, duration: float, physics_dt: float) -> Tuple[float, ...]:
        """The interval start only: a held torque is never refreshed."""
        return (0.0,)


Controller = Union[FeedbackController, TorqueController]


def make_controller(config: ControllerConfig, state_spec: StateSpec) -> Controller:
    """The controller of config["decision_mode"]: FeedbackController or TorqueController."""
    if decision_mode_setting(config) == "torque":
        return TorqueController(config, state_spec)
    return FeedbackController(config, state_spec)

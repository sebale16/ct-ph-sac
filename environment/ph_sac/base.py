"""MuJoCo interval environment shared by every testbed.

States are (observed positions, context, velocities) in the layout of StateSpec: the
first position_dim - context_dim positions are MuJoCo's qpos[:physical_position_dim]
(periodic ones wrapped), the last context_dim positions are an episode-constant context
held in self._context (zero rate, written by the task's _reset_state through _set_state),
and the velocities are qvel[:velocity_dim].

decision_mode "feedback" takes impedance decisions, mapped to a torque by the
FeedbackController; "torque" takes the torque itself and holds it over the interval. A
torque-mode environment's config is torque_decision_testbed(config), so its controller
config, its TorqueController and its action space all carry the torque box; the CT-SAC
baseline and the torque-decision CT-PH-SAC arms run on it.

When the feedback law is evaluated is the controller's refresh_mode
(common.ph_config.REFRESH_MODES):

  "continuous"  (the default) MuJoCo's control callback (mujoco.set_mjcb_control)
                evaluates the law from the state of every integrator stage, so under
                RK4 physics each of a physics step's four stages sees the torque of its
                own state, clipped to the motor limits, as a continuous-time controller
                would apply it. The callback is process-global: _stage_feedback installs
                a dispatcher around each mj_step of a continuous-feedback interval only
                and removes it after the step; the dispatcher writes data.ctrl only for
                the MjData of the environment stepping in the calling thread and passes
                any other MjData to the callback that was installed before (normally
                none), so other environments, dm_control's own physics and every
                mj_forward outside a step never see it. Under Euler physics (the windy
                pendulum; the swimmers since 2026-11-10) a step has one stage, at its
                start, where the law is the held torque, so the environment runs the
                "held" path (stage_feedback False) with a refresh at every physics step
                (FeedbackController.refresh_due is True at every step under
                "continuous"): the law is evaluated once per step from the step's start
                state, as MuJoCo's callback would at the step's one forward pass, and as
                the model's semi_implicit map evaluates it once per substep. The two
                modes then agree bitwise, reward included (both knots of a step read
                the step's torque).
  "held"        the torque is computed from the state at every refresh (every
                refresh_steps physics steps) and held through each physics step and its
                RK4 stages: the sample-and-hold of every run before 2026-11-09, bitwise.

The reward is integrated by trapezoids on the physics-step knots in both modes: the rate
at a knot reads the knot's state and the torque applied there. Under "continuous" (RK4
physics) that torque is the law at the knot's state at both ends of every physics step
(the rate is continuous across knots); under "held" (and one-stage physics) it is the
torque held over the step. A physics step counts as saturated in
last_interval["saturated_fraction"] when its start knot's torque is on a motor limit.
"""

from __future__ import annotations

import contextlib
import copy
import math
import threading
from typing import TYPE_CHECKING, Iterator, Literal, Mapping, Optional, Tuple

import mujoco
import numpy as np
from gymnasium.spaces import Box

from common.ph_config import (
    check_decision_mode,
    decision_mode_setting,
    torque_decision_testbed,
)
from common.ph_types import context_dim

from .clock import DiscreteIntervalClock, duration_in_physics_steps
from .controller import make_controller

if TYPE_CHECKING:
    from mujoco import MjData, MjModel
    from numpy import ndarray

    from common.ph_config import TestbedConfig
    from common.ph_types import StateSpec, Transition


# MuJoCo's control callback is one process-global function pointer. While any
# continuous-feedback physics step runs, it is _stage_control_dispatch; _STEPPING.env is
# the environment whose mj_step the calling thread is inside. _CALLBACK_USERS counts the
# steps in flight (across threads) and _CALLBACK_PREVIOUS keeps the callback installed
# before the first of them, restored after the last.
_CALLBACK_LOCK = threading.Lock()
_CALLBACK_USERS = 0
_CALLBACK_PREVIOUS = None
_STEPPING = threading.local()


def _stage_control_dispatch(model: MjModel, data: MjData) -> None:
    """mjcb_control: write the stepping environment's feedback torque into its own data.

    Called by MuJoCo at every integrator stage (mj_forward inside mj_step). Data of any
    other simulation goes to the previously installed callback, if there was one.
    """
    env = getattr(_STEPPING, "env", None)
    if env is not None and data is env.data:
        env._stage_control()
    elif _CALLBACK_PREVIOUS is not None:
        _CALLBACK_PREVIOUS(model, data)


@contextlib.contextmanager
def _stage_feedback(env: MujocoPHEnv) -> Iterator[None]:
    """Install the dispatcher for env's next mj_step; remove it (restoring the previous
    callback) when no continuous step is left in flight."""
    global _CALLBACK_USERS, _CALLBACK_PREVIOUS
    with _CALLBACK_LOCK:
        if _CALLBACK_USERS == 0:
            _CALLBACK_PREVIOUS = mujoco.get_mjcb_control()
            mujoco.set_mjcb_control(_stage_control_dispatch)
        _CALLBACK_USERS += 1
    outer = getattr(_STEPPING, "env", None)
    _STEPPING.env = env
    try:
        yield
    finally:
        _STEPPING.env = outer
        with _CALLBACK_LOCK:
            _CALLBACK_USERS -= 1
            if _CALLBACK_USERS == 0:
                mujoco.set_mjcb_control(_CALLBACK_PREVIOUS)
                _CALLBACK_PREVIOUS = None


class MujocoPHEnv:
    """MuJoCo interval physics, distinct from the model's observation-coordinate map.

    Gust, clock and reset draws use separate streams derived from the reset seed and
    never read the policy, so arms evaluated on one seed share common random numbers.
    """

    model: MjModel
    data: MjData
    action_space: Box
    observation_space: Box
    state_spec: StateSpec
    config: TestbedConfig
    wrapped_observation_indices: Optional[Tuple[int, ...]] = None

    def __init__(
        self,
        config: TestbedConfig,
        model: MjModel,
        state_spec: StateSpec,
        *,
        decision_mode: Literal["torque", "feedback"] = "feedback",
        seed: Optional[int] = None,
    ) -> None:
        if decision_mode not in ("torque", "feedback"):
            raise ValueError(f"Unknown decision_mode {decision_mode!r}.")
        if decision_mode == "torque":
            # The decision box becomes the torque box; a torque config is kept as is.
            config = torque_decision_testbed(config)
        elif decision_mode_setting(config["controller"]) != "feedback":
            raise ValueError(
                "A torque-decision testbed config needs decision_mode='torque'."
            )
        self.config = config
        self.model = model
        self.data = mujoco.MjData(model)
        self.state_spec = state_spec
        self.decision_mode = check_decision_mode(decision_mode)
        self.discount_rate = float(config["discount_rate"])
        self.physics_dt = float(config["physics_dt"])
        self.model.opt.timestep = self.physics_dt
        self.clock = DiscreteIntervalClock(config["clock"])
        self.controller = make_controller(config["controller"], state_spec)
        # A feedback law evaluated at every stage of a multi-stage (RK4) physics step
        # (refresh_mode "continuous"). Under one-stage physics (Euler) the law's only
        # evaluation is at the step's start, which is the held torque: the environment
        # then runs the "held" path, so both modes agree bitwise there.
        self.stage_feedback = (
            decision_mode == "feedback"
            and self.controller.continuous
            and model.opt.integrator == mujoco.mjtIntegrator.mjINT_RK4
        )
        # The resolved decision the control callback evaluates, set by step_dt for the
        # physics steps of a continuous-feedback interval only; None otherwise.
        self._stage_decision: Optional[ndarray] = None

        controller = config["controller"]
        # The torque box in torque mode (torque_decision_testbed), the feedback box otherwise.
        low, high = controller["decision_low"], controller["decision_high"]
        self.action_space = Box(
            np.asarray(low, dtype=np.float64),
            np.asarray(high, dtype=np.float64),
            dtype=np.float64,
        )
        self._torque_low = np.asarray(controller["torque_low"], dtype=np.float64)
        self._torque_high = np.asarray(controller["torque_high"], dtype=np.float64)

        self.position_dim = int(state_spec["position_dim"])
        self.velocity_dim = int(state_spec["velocity_dim"])
        self.state_dim = self.position_dim + self.velocity_dim
        # The soft-limited coordinates (StateSpec limited_position_ranges): an interval
        # whose state is past one of these ranges at any physics knot (start included)
        # sets the Transition's limit_touched (MuJoCo's limit row exists exactly there:
        # margin 0, position strictly past the range).
        ranges = tuple(state_spec.get("limited_position_ranges", ()))
        self._limit_index = np.array([int(entry[0]) for entry in ranges], dtype=np.int64)
        self._limit_low = np.array([float(entry[1]) for entry in ranges], dtype=np.float64)
        self._limit_high = np.array([float(entry[2]) for entry in ranges], dtype=np.float64)
        # Episode-constant context: the last context_dim positions, outside MuJoCo's qpos.
        self.context_dim = context_dim(state_spec)
        self.physical_position_dim = self.position_dim - self.context_dim
        if self.physical_position_dim > model.nq or self.velocity_dim > model.nv:
            raise ValueError("The state spec has more positions or velocities than the model.")
        self._context = np.zeros(self.context_dim)
        self.observation_space = Box(
            -np.inf, np.inf, shape=(self.state_dim,), dtype=np.float64
        )
        if self.wrapped_observation_indices is None:
            self.wrapped_observation_indices = tuple(
                state_spec["periodic_position_indices"]
            )
        if self.stage_feedback:
            # The control callback's reads: actuated observation positions are qpos
            # entries (wrapped as observe() wraps them), actuated velocities qvel entries.
            positions = self.controller.actuated_positions
            self._law_qpos = positions
            self._law_qvel = self.controller.actuated_velocities - self.position_dim
            self._law_wrap = np.array(
                [int(i) in self.wrapped_observation_indices for i in positions], dtype=bool
            )

        self.episode_steps = duration_in_physics_steps(
            config["episode_duration"], self.physics_dt
        )
        self._seed_rng = np.random.default_rng(seed)
        self._reset_rng = np.random.default_rng(0)
        self._gust_rng = np.random.default_rng(0)
        self._scratch = mujoco.MjData(model)
        self._renderer: Optional[mujoco.Renderer] = None

        self._steps_taken = 0
        self._done = True
        self._held_torque = np.zeros(int(state_spec["action_dim"]))
        self._resolved_decision: Optional[ndarray] = None
        self.last_interval: Mapping[str, float] = {}
        # Undiscounted trapezoid integral of the reward rate over the last interval,
        # without the failure charge; episode-level evaluation metrics read it.
        self.last_undiscounted_reward = 0.0

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Mapping[str, object]] = None,
    ) -> Tuple[ndarray, Mapping[str, object]]:
        if seed is not None:
            self._seed_rng = np.random.default_rng(seed)
        episode_seed = int(self._seed_rng.integers(2**63))
        reset_seed, gust_seed, clock_seed = np.random.SeedSequence(episode_seed).spawn(3)
        self._reset_rng = np.random.default_rng(reset_seed)
        self._gust_rng = np.random.default_rng(gust_seed)
        self.clock.reset(seed=clock_seed)

        mujoco.mj_resetData(self.model, self.data)
        self._set_context(np.zeros(self.context_dim))
        self._reset_state(options)
        mujoco.mj_forward(self.model, self.data)
        self._steps_taken = 0
        self._done = False
        self._held_torque = np.zeros_like(self._held_torque)
        self._resolved_decision = None
        self.last_interval = {}
        self.last_undiscounted_reward = 0.0
        return self.observe(), {"time": 0.0, "episode_seed": episode_seed}

    def sample_duration(self) -> float:
        """Draw the next interval from the exogenous clock, shortened at the episode end."""
        return self.clock.sample_duration(remaining_steps=self.remaining_steps)

    @property
    def remaining_steps(self) -> int:
        return self.episode_steps - self._steps_taken

    @property
    def time(self) -> float:
        return self._steps_taken * self.physics_dt

    @property
    def done(self) -> bool:
        return self._done

    def step(
        self, decision: ndarray
    ) -> Tuple[ndarray, float, bool, bool, Mapping[str, object]]:
        """Sample the exogenous clock; distinguish failure and time-limit truncation."""
        if self._done:
            raise RuntimeError("Call reset before stepping a finished episode.")
        transition = self.step_dt(decision, self.sample_duration())
        info = {
            "transition": transition,
            "dt": transition["duration"],
            **self.last_interval,
        }
        return (
            transition["next_state"],
            transition["reward"],
            transition["terminated"],
            transition["truncated"],
            info,
        )

    def step_dt(self, decision: ndarray, duration: float) -> Transition:
        """Run whole physics steps and report actual time and the final pre-reset state.

        Stop at a sampled failure or time limit; charge a failure once in the
        discounted interval reward and preserve truncation for bootstrapping. The
        scheduled duration is fixed before the interval runs, so a failure cannot
        shorten it; the episode end can, because that is known in advance.
        """
        if self._done:
            raise RuntimeError("Call reset before stepping a finished episode.")
        decision = np.asarray(decision, dtype=np.float64)
        if decision.shape != self.action_space.shape or not np.isfinite(decision).all():
            raise ValueError(
                f"Decision must be finite with shape {self.action_space.shape}."
            )
        decision = np.clip(decision, self.action_space.low, self.action_space.high)
        requested = duration_in_physics_steps(duration, self.physics_dt)
        planned = min(requested, self.remaining_steps)

        start_time = self.time
        state = self.observe()
        start_state = state
        if self.decision_mode == "feedback":
            self._resolved_decision = self.controller.resolve_decision_numpy(
                state, decision
            )
        else:
            self._resolved_decision = None
            self._held_torque = decision.copy()

        # A torque change makes the rate jump, so each physics step keeps its own knots.
        # Under refresh_mode "continuous" (RK4 physics) each knot reads the law at its own
        # state, so the two knots at one time agree; otherwise both read the step's torque.
        continuous = self.stage_feedback
        reward_rates = np.empty(2 * planned)
        local_times = np.empty(2 * planned)
        terminated = False
        steps_run = 0
        saturated = 0
        limited = self._limit_index.size > 0
        touched = limited and self._past_limit(state)
        self._stage_decision = self._resolved_decision if continuous else None
        try:
            for step in range(planned):
                torque = self._interval_torque(state, step)
                reward_rates[2 * step] = self.reward_rate(state, torque)
                local_times[2 * step] = step * self.physics_dt
                self._physics_step(torque)
                state = self.observe()
                if limited and not touched:
                    touched = self._past_limit(state)
                end_torque = torque
                if continuous:
                    end_torque = self.controller.action_numpy(state, self._resolved_decision)
                reward_rates[2 * step + 1] = self.reward_rate(state, end_torque)
                local_times[2 * step + 1] = (step + 1) * self.physics_dt
                steps_run += 1
                saturated += int(
                    np.any(torque <= self._torque_low) or np.any(torque >= self._torque_high)
                )
                if self.failed(state):
                    terminated = True
                    break
        finally:
            self._stage_decision = None

        reward = self.integrate_interval_reward(
            reward_rates[: 2 * steps_run],
            local_times[: 2 * steps_run],
            terminated=terminated,
            terminal_state=state,
        )
        self._steps_taken += steps_run
        truncated = (not terminated) and self.remaining_steps <= 0
        self._done = terminated or truncated
        actual = steps_run * self.physics_dt
        self.last_interval = {
            "physics_steps": steps_run,
            "saturated_fraction": saturated / steps_run,
        }
        return {
            "state": start_state,
            "decision": decision,
            "duration": actual,
            "scheduled_duration": planned * self.physics_dt,
            "reward": reward,
            "next_state": state,
            "terminated": terminated,
            "truncated": truncated,
            "time": start_time,
            "next_time": self.time,
            "limit_touched": bool(touched),
        }

    def _past_limit(self, state: ndarray) -> bool:
        """Whether a limited coordinate of state lies strictly past its range."""
        positions = state[self._limit_index]
        return bool(np.any((positions < self._limit_low) | (positions > self._limit_high)))

    def _mass_matrix(self, qpos: Optional[ndarray] = None) -> ndarray:
        """The joint-space inertia M(q) [nv, nv] (armature included) at qpos (MuJoCo's
        qpos, default the model's reference pose qpos0), from a fresh MjData: the
        episode is not touched."""
        model = self.model
        data = mujoco.MjData(model)
        if qpos is not None:
            data.qpos[:] = np.asarray(qpos, dtype=np.float64).reshape(model.nq)
        mujoco.mj_forward(model, data)
        mass = np.zeros((model.nv, model.nv))
        mujoco.mj_fullM(model, mass, data.qM)
        return mass

    def velocity_decay_rates(self, qpos: Optional[ndarray] = None) -> ndarray:
        """Per velocity coordinate i, k_i = d_i (M(q)^-1)_ii in 1/s [velocity_dim].

        d_i is the compiled model's joint damping (dof_damping) and 1/(M^-1)_ii the
        coordinate's effective inertia with every other coordinate free, at qpos
        (default qpos0, the convention of motor_acceleration_bounds): the rate at which
        the damping alone slows the coordinate, the inverse of its velocity time constant
        m_i / d_i (zero without damping). Contact exclusion's reach bound
        (models/ph_sac/rollouts.start_state_margins) reads it.
        """
        mass = self._mass_matrix(qpos)
        damping = np.asarray(self.model.dof_damping, dtype=np.float64).reshape(-1)
        inverse = np.diag(np.linalg.inv(mass))
        return (damping * inverse)[: self.velocity_dim]

    def motor_acceleration_bounds(self, qpos: Optional[ndarray] = None) -> ndarray:
        """Per velocity coordinate i, sum_j |(M(q0)^-1 G)_ij| u_max_j [velocity_dim].

        The generalized acceleration the motors alone (every actuator at its torque
        limit u_max_j = max(|torque_low_j|, |torque_high_j|), worst signs) give each
        coordinate at the model's reference pose qpos0 (or at qpos), at rest. G is the
        joint transmission of oracle_input_matrix (gear per actuated dof). Reads a fresh
        MjData: the episode is not touched. Contact exclusion's acceleration allowance
        (models/ph_sac/rollouts.ContactExclusion) is a multiple of it.
        """
        model = self.model
        mass = self._mass_matrix(qpos)
        inputs = np.zeros((model.nv, model.nu))
        for actuator in range(model.nu):
            joint = int(model.actuator_trnid[actuator, 0])
            dof = int(model.jnt_dofadr[joint])
            inputs[dof, actuator] = float(model.actuator_gear[actuator, 0])
        limits = np.maximum(np.abs(self._torque_low), np.abs(self._torque_high)).reshape(-1)
        response = np.linalg.solve(mass, inputs * limits[: model.nu])
        return np.abs(response).sum(axis=1)[: self.velocity_dim]

    def _interval_torque(self, state: ndarray, step: int) -> ndarray:
        """The torque at the knot starting physics step `step` (also written to ctrl).

        A feedback law is refreshed when refresh_due says so (every step under
        refresh_mode "continuous", where the callback then re-evaluates it at every
        stage); a torque decision is the held torque.
        """
        if self.decision_mode == "feedback" and self.controller.refresh_due(step):
            self._held_torque = self.controller.action_numpy(
                state, self._resolved_decision
            )
        return self._held_torque

    def observe(self) -> ndarray:
        """Read observation coordinates while retaining unwrapped state for failures.

        The context follows the physical positions; it is never wrapped.
        """
        positions = np.array(self.data.qpos[: self.physical_position_dim], dtype=np.float64)
        for index in self.wrapped_observation_indices:
            positions[index] = _wrap_angle(positions[index])
        velocities = np.array(self.data.qvel[: self.velocity_dim], dtype=np.float64)
        return np.concatenate((positions, self._context, velocities))

    @property
    def context(self) -> ndarray:
        """The episode-constant context (empty without one)."""
        return self._context.copy()

    def _reset_state(self, options: Optional[Mapping[str, object]] = None) -> None:
        raise NotImplementedError

    def _set_state(self, state: ndarray) -> None:
        """Write physical positions and velocities to MuJoCo and keep the context."""
        state = np.asarray(state, dtype=np.float64)
        if state.shape != (self.state_dim,):
            raise ValueError(f"State must have shape ({self.state_dim},).")
        physical = self.physical_position_dim
        self.data.qpos[:physical] = state[:physical]
        self._set_context(state[physical : self.position_dim])
        self.data.qvel[: self.velocity_dim] = state[self.position_dim :]

    def _set_context(self, context: ndarray) -> None:
        """Store the episode-constant context, then let the task mirror it in the model."""
        context = np.array(context, dtype=np.float64).reshape(-1)
        if context.shape != (self.context_dim,):
            raise ValueError(f"The context must have shape ({self.context_dim},).")
        self._context = context
        self._apply_context()

    def _apply_context(self) -> None:
        """Hook: mirror self._context in the model (the reacher's target geom); no-op."""
        return None

    def _physics_step(self, torque: ndarray) -> None:
        """Advance the configured MuJoCo integrator by one physics step with gusts.

        ctrl is set to torque. Outside a continuous-feedback interval it is held through
        the step (all RK4 stages). Inside one (step_dt sets _stage_decision) MuJoCo's
        control callback replaces it at every stage by the law at the stage's state
        (_stage_control); torque is then the law at the step's start, which the first
        stage reproduces.
        """
        self.data.ctrl[:] = torque
        self._apply_disturbance()
        if self._stage_decision is None:
            mujoco.mj_step(self.model, self.data)
            return
        with _stage_feedback(self):
            mujoco.mj_step(self.model, self.data)

    def _stage_control(self) -> None:
        """The control callback's work: ctrl = the clipped law at the current stage state.

        MuJoCo writes the stage state into data before each stage's forward pass. The law
        reads only the actuated positions and velocities, so they are taken from qpos and
        qvel directly, the positions wrapped as observe() wraps them; the torque is
        bitwise the one action_numpy(observe(), ...) gives.
        """
        data = self.data
        positions = data.qpos[self._law_qpos]
        if self._law_wrap.any():
            outside = self._law_wrap & ((positions < -math.pi) | (positions >= math.pi))
            if outside.any():
                positions = np.where(
                    outside, np.mod(positions + math.pi, 2.0 * math.pi) - math.pi, positions
                )
        data.ctrl[:] = self.controller.actuated_action_numpy(
            positions, data.qvel[self._law_qvel], self._stage_decision
        )

    def _apply_disturbance(self) -> None:
        return None

    def reward_rate(self, state: ndarray, torque: ndarray) -> float:
        raise NotImplementedError

    def end_reward_rate(self, transition: Transition) -> float:
        """rho_end = reward_rate(s', u_e): the task's rate at the interval's end state
        s' = transition["next_state"] (a terminated interval's first failing knot) with the
        torque u_e there (common.ph_config.REWARD_OBSERVATION_KEY "end_state").

        Torque mode: u_e is the transition's decision. step_dt clips the decision to the
        torque box and holds exactly that array over every physics step (_held_torque,
        written to ctrl), and evaluates its last trapezoid knot as reward_rate(s', that
        array), so this is bitwise the environment's own last knot. Feedback mode: u_e is the
        law at s', controller.action_numpy(s', controller.resolve_decision_numpy(s,
        decision)) with s = transition["state"], the torque a learner rebuilds from (s,
        decision, s'); it is the last knot's torque under refresh_mode "continuous" with RK4
        physics, and differs from it under "held" or one-stage (Euler) physics, whose last
        knot reads the torque held over the last physics step. Pure: reads nothing but the
        transition and changes no state or stream; never reads the interval reward.
        """
        next_state = transition["next_state"]
        decision = np.asarray(transition["decision"], dtype=np.float64)
        if self.decision_mode == "torque":
            torque = decision
        else:
            resolved = self.controller.resolve_decision_numpy(transition["state"], decision)
            torque = self.controller.action_numpy(next_state, resolved)
        return float(self.reward_rate(next_state, torque))

    def failed(self, state: ndarray) -> bool:
        """Apply the sampled state-limit predicate; no continuous crossing claim."""
        raise NotImplementedError

    def terminal_payoff(self, state: ndarray) -> float:
        """Return the failure boundary payoff, separate from time-limit truncation."""
        raise NotImplementedError

    def integrate_interval_reward(
        self,
        reward_rates: ndarray,
        local_times: ndarray,
        *,
        terminated: bool,
        terminal_state: ndarray,
    ) -> float:
        """Discounted trapezoids plus one failure charge at the detected failure time.

        Also sets last_undiscounted_reward to the same trapezoids undiscounted and
        without the failure charge.
        """
        rates = np.asarray(reward_rates, dtype=np.float64)
        times = np.asarray(local_times, dtype=np.float64)
        self.last_undiscounted_reward = 0.5 * float(
            np.sum((rates[1:] + rates[:-1]) * np.diff(times))
        )
        discounted = np.exp(-self.discount_rate * times) * rates
        reward = 0.5 * float(np.sum((discounted[1:] + discounted[:-1]) * np.diff(times)))
        if terminated:
            reward += math.exp(-self.discount_rate * float(times[-1])) * float(
                self.terminal_payoff(terminal_state)
            )
        return reward

    def mujoco_drift(self, state: ndarray, torque: ndarray) -> ndarray:
        """Read (P q_dot, qacc) from MuJoCo's forward dynamics without touching the episode.

        The context rows of P are zero, so the context's rate is zero; the context does
        not reach MuJoCo, whose dynamics must not depend on it.
        """
        state = np.asarray(state, dtype=np.float64)
        scratch = self._scratch
        scratch.qpos[: self.physical_position_dim] = state[: self.physical_position_dim]
        scratch.qvel[: self.velocity_dim] = state[self.position_dim :]
        scratch.ctrl[:] = torque
        scratch.qfrc_applied[:] = 0.0
        mujoco.mj_forward(self.model, scratch)
        projection = np.asarray(self.state_spec["observation_projection"])
        velocities = state[self.position_dim :]
        return np.concatenate(
            (projection @ velocities, np.array(scratch.qacc[: self.velocity_dim]))
        )

    def oracle_drift(self, state: ndarray, torque: ndarray) -> ndarray:
        """Read continuous mechanical drift in (P q, q_dot), excluding sampled gusts.

        This coefficient does not identify the finite-step MuJoCo endpoint law.
        """
        return self.mujoco_drift(state, torque)

    def oracle_diffusion(self, state: ndarray) -> ndarray:
        """Map true force noise to observation-coordinate SDE diffusion.

        Return Sigma before any discrete K_h transport used by model rollouts.
        """
        return np.zeros((self.state_dim, 1))

    def oracle_input_matrix(self) -> ndarray:
        """Read G mapping motor torque to generalized force, before inverse mass."""
        matrix = np.zeros((self.velocity_dim, self.model.nu))
        for actuator in range(self.model.nu):
            joint = int(self.model.actuator_trnid[actuator, 0])
            dof = int(self.model.jnt_dofadr[joint])
            matrix[dof, actuator] = float(self.model.actuator_gear[actuator, 0])
        return matrix

    def snapshot(self) -> Mapping[str, object]:
        """Capture physics, clock, resolved setpoint, held torque, phase and RNG."""
        data = self.data
        return copy.deepcopy(
            {
                "qpos": np.array(data.qpos),
                "qvel": np.array(data.qvel),
                "act": np.array(data.act),
                "ctrl": np.array(data.ctrl),
                "qfrc_applied": np.array(data.qfrc_applied),
                "qacc_warmstart": np.array(data.qacc_warmstart),
                "context": np.array(self._context),
                "time": float(data.time),
                "steps_taken": self._steps_taken,
                "done": self._done,
                "held_torque": np.array(self._held_torque),
                "resolved_decision": self._resolved_decision,
                "last_interval": dict(self.last_interval),
                "last_undiscounted_reward": self.last_undiscounted_reward,
                "seed_rng": self._seed_rng.bit_generator.state,
                "reset_rng": self._reset_rng.bit_generator.state,
                "gust_rng": self._gust_rng.bit_generator.state,
                "clock": self.clock.state_dict(),
            }
        )

    def restore(self, snapshot: Mapping[str, object]) -> None:
        snapshot = copy.deepcopy(snapshot)
        data = self.data
        data.qpos[:] = snapshot["qpos"]
        data.qvel[:] = snapshot["qvel"]
        data.act[:] = snapshot["act"]
        data.ctrl[:] = snapshot["ctrl"]
        data.qfrc_applied[:] = snapshot["qfrc_applied"]
        data.qacc_warmstart[:] = snapshot["qacc_warmstart"]
        data.time = snapshot["time"]
        self._set_context(snapshot.get("context", np.zeros(self.context_dim)))
        mujoco.mj_forward(self.model, data)
        self._steps_taken = snapshot["steps_taken"]
        self._done = snapshot["done"]
        self._held_torque = snapshot["held_torque"]
        self._resolved_decision = snapshot["resolved_decision"]
        self.last_interval = snapshot["last_interval"]
        self.last_undiscounted_reward = float(snapshot.get("last_undiscounted_reward", 0.0))
        self._seed_rng.bit_generator.state = snapshot["seed_rng"]
        self._reset_rng.bit_generator.state = snapshot["reset_rng"]
        self._gust_rng.bit_generator.state = snapshot["gust_rng"]
        self.clock.load_state_dict(snapshot["clock"])

    def render(self, *, width: int = 640, height: int = 480) -> ndarray:
        if (
            self._renderer is None
            or self._renderer.width != width
            or self._renderer.height != height
        ):
            self.close()
            self._renderer = mujoco.Renderer(self.model, height=height, width=width)
        self._renderer.update_scene(self.data)
        return self._renderer.render()

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


def validated_state_spec(config: TestbedConfig) -> StateSpec:
    """Check P, the velocity layout, the context and the periodic/actuated coordinates.

    The context (context_dim trailing positions) must have zero rows in P; periodic and
    actuated indices must name physical positions.
    """
    spec = config["state_spec"]
    positions, velocities = spec["position_dim"], spec["velocity_dim"]
    projection = np.asarray(spec["observation_projection"])
    if projection.shape != (positions, velocities):
        raise ValueError("The observation projection must map velocities to positions.")
    context = spec.get("context_dim", 0)
    if isinstance(context, bool) or not isinstance(context, (int, np.integer)):
        raise ValueError("context_dim must be an integer.")
    if not 0 <= context <= positions:
        raise ValueError("context_dim must lie between 0 and position_dim.")
    physical = positions - context
    if np.any(projection[physical:] != 0.0):
        raise ValueError("The context rows of the observation projection must be zero.")
    for name in ("periodic_position_indices", "actuated_position_indices"):
        if any(not 0 <= index < physical for index in spec[name]):
            raise ValueError(f"{name} must index observed positions, not the context.")
    if len(spec["actuated_position_indices"]) != spec["action_dim"]:
        raise ValueError("Need one actuated position per action dimension.")
    for index, scale in spec.get("travel_position_scales", ()):
        if not 0 <= index < positions or not scale > 0.0:
            raise ValueError("travel_position_scales needs observed positions and positive scales.")
    return spec


def _wrap_angle(angle: float) -> float:
    """Map to [-pi, pi); angles already inside are returned unchanged."""
    if -math.pi <= angle < math.pi:
        return float(angle)
    return float((angle + math.pi) % (2.0 * math.pi) - math.pi)

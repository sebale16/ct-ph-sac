"""Adaptation seam for the existing CT-SAC torque baseline in the paper suite."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

import gymnasium as gym
import numpy as np
import torch as th
from gymnasium.spaces import Box

from algorithms.ct_sac import CTSAC
from common.callbacks import BaseCallback
from common.ph_config import (
    DEFAULT_CT_SAC_REWARD_SOURCE,
    DEFAULT_CT_SAC_REWARD_UNITS,
    check_ct_sac_reward_source,
    check_ct_sac_reward_units,
)
from common.ph_time import period_discount
from environment.base import ContinuousEnv
from environment.ph_sac.tasks import reward_rate_tensor, terminal_payoff_tensor

if TYPE_CHECKING:
    from numpy import ndarray
    from torch import Tensor

    from common.ph_config import TemperatureConfig, TrainingConfig
    from common.ph_types import Metrics, Transition, TransitionBatch
    from environment.ph_sac.base import MujocoPHEnv

# The legacy loop dumps its logger on this modulus; the protocol keeps its own records.
_NO_LEGACY_LOG = 2**62
# How a stored row treats a time-limit truncation (TorqueCTSACBaseline time_limit):
# "bootstrap" stores done = terminated, so the critic target bootstraps through the time
# limit; "terminal" stores done = terminated or truncated, as CT-SAC's own loop does
# (algorithms/off_policy.py learn), so the target is the reward alone there.
TIME_LIMIT_MODES = ("bootstrap", "terminal")
# The arrays of common.buffers.ReplayBuffer that resume_state carries.
_LEGACY_REPLAY_FIELDS = (
    "observations",
    "next_observations",
    "actions",
    "rewards",
    "dones",
    "t",
    "next_t",
    "dt",
)


def end_state_reward_rate(env: MujocoPHEnv, transition: Transition) -> float:
    """rho(s', a): the task's reward rate at the interval's end state s' (next_state) with
    the torque a held over the interval (ct_sac_reward_source "end_state").

    In a torque-mode environment the transition's decision is that torque: step_dt clips
    the decision to the torque box, holds exactly that array over every physics step
    (MujocoPHEnv._held_torque, written to MuJoCo's ctrl) and evaluates its last trapezoid
    knot as reward_rate(next_state, that torque), so this is the environment's own rate at
    its last knot: the environment's end_reward_rate, the one definition the CT-PH-SAC arms'
    reward_observation "end_state" reads too. A feedback-mode decision is an impedance law,
    not a torque: refused.
    """
    if env.decision_mode != "torque":
        raise ValueError(
            "ct_sac_reward_source 'end_state' reads the held torque from the transition's "
            "decision, so it needs a torque-mode environment."
        )
    return float(env.end_reward_rate(transition))


def _adapt_transition(
    env: MujocoPHEnv,
    transition: Transition,
    reward_units: str = DEFAULT_CT_SAC_REWARD_UNITS,
    reward_source: str = DEFAULT_CT_SAC_REWARD_SOURCE,
) -> Mapping[str, object]:
    """Convert one PH interval to legacy fields, keeping the failure charge in Q units.

    reward_source (common.ph_config.CT_SAC_REWARD_SOURCES, checked by the callers):
    "interval" gives the reward (r - c) / u + c, about T times the per-second rate averaged
    over the interval, c the failure charge; "end_state" gives T rho(s', a) + c, T times the
    rate at the end state with the held torque (end_state_reward_rate), not scaled by the
    duration. reward_units (common.ph_config.CT_SAC_REWARD_UNITS, checked by the callers):
    "second" gives that value; "reference_period" that value divided by T.
    """
    duration = float(transition["duration"])
    normalized = duration / float(env.config["reference_period"])
    charge = 0.0
    if transition["terminated"]:
        payoff = float(env.terminal_payoff(transition["next_state"]))
        charge = math.exp(-env.discount_rate * duration) * payoff
    if reward_source == "end_state":
        reward = (
            float(env.config["reference_period"]) * end_state_reward_rate(env, transition)
            + charge
        )
    else:
        reward = (float(transition["reward"]) - charge) / normalized + charge
    if reward_units == "reference_period":
        reward = reward / float(env.config["reference_period"])
    return {
        "observation": transition["state"],
        "action": transition["decision"],
        "reward": reward,
        "next_observation": transition["next_state"],
        "terminated": bool(transition["terminated"]),
        "truncated": bool(transition["truncated"]),
        "done": bool(transition["terminated"]),
        "duration": duration,
        "normalized_duration": normalized,
        "time": float(transition["time"]),
        "next_time": float(transition["next_time"]),
    }


def symmetric_torque_limit(env: MujocoPHEnv) -> ndarray:
    """Return the motor limit [m] of a box [-limit, limit]; raise for any other box."""
    controller = env.config["controller"]
    high = np.asarray(controller["torque_high"], dtype=np.float64).reshape(-1)
    low = np.asarray(controller["torque_low"], dtype=np.float64).reshape(-1)
    if high.shape != low.shape or not np.allclose(low, -high) or np.any(high <= 0.0):
        raise ValueError(
            "The CT-SAC baseline acts in u in [-1, 1] times the motor limit, so it needs "
            f"a symmetric torque box; got [{low.tolist()}, {high.tolist()}]."
        )
    return high


class _LegacyEnvAdapter(ContinuousEnv):
    """Expose a torque-mode MuJoCo interval env through the ContinuousEnv tuple API.

    dt_default is the reference period T, so legacy time_rescale makes u=Delta/T.
    The exogenous clock and the episode limit stay inside the wrapped environment.
    The learner's action is the normalized u in [-1, 1]; the motor receives u times
    the symmetric torque limit, so the transition records torque.

    The learner's observation is network_input(state): the environment's state itself
    (features None, every arm before ct_sac_features), or a feature map of it, CT-PH-SAC's
    models.ph_sac.networks.StateFeatures (periodic positions as sine and cosine, other
    positions raw, velocities divided by the network's velocity_feature_scale), whose
    output then is the observation space. The loop, its replay and its last observation
    hold that vector; the transition in info keeps the state.

    reward_units (common.ph_config.CT_SAC_REWARD_UNITS) sets the reward the loop receives
    and stores (_adapt_transition): "second" (the default) about T times the per-second
    rate, "reference_period" that divided by T. reward_source
    (common.ph_config.CT_SAC_REWARD_SOURCES) sets which rate: "interval" (the default) the
    environment's interval reward per period, "end_state" the rate at the interval's end
    state with the torque this adapter sent and the environment held (the transition's
    decision, end_state_reward_rate).
    """

    def __init__(
        self,
        env: MujocoPHEnv,
        *,
        seed: Optional[int],
        features: Optional[Any] = None,
        reward_units: str = DEFAULT_CT_SAC_REWARD_UNITS,
        reward_source: str = DEFAULT_CT_SAC_REWARD_SOURCE,
    ) -> None:
        super().__init__(
            time_sampling="uniform",
            dt=float(env.config["reference_period"]),
            physics_dt=float(env.physics_dt),
        )
        self.ph_env = env
        self.reward_units = check_ct_sac_reward_units(reward_units)
        self.reward_source = check_ct_sac_reward_source(reward_source)
        self.torque_limit = symmetric_torque_limit(env)
        self.action_space = Box(
            -1.0, 1.0, shape=env.action_space.shape, dtype=np.float64
        )
        self.features = features
        if features is None:
            self.observation_space = env.observation_space
        else:
            self.observation_space = Box(
                -np.inf, np.inf, shape=(int(features.output_dim),), dtype=np.float64
            )
        self.action_space.seed(seed)
        self._np_random, _ = gym.utils.seeding.np_random(seed)
        self.resume_episode = False
        self.simulated_seconds = 0.0

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset, unless a later learn call is continuing the episode in progress."""
        resume, self.resume_episode = self.resume_episode, False
        if resume and seed is None and options is None and not self.ph_env.done:
            return self._last_obs.copy(), {}
        return super().reset(seed=seed, options=options)

    def _reset_physics(
        self,
        *,
        seed: Optional[int],
        options: Optional[Dict[str, Any]],
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        observation, info = self.ph_env.reset(seed=seed, options=options)
        return self.network_input(observation), dict(info)

    def network_input(self, states: np.ndarray) -> np.ndarray:
        """The learner's observation of states [..., D]: states itself (the same object)
        without features, else the feature map computed in float64."""
        if self.features is None:
            return states
        with th.no_grad():
            mapped = self.features(th.as_tensor(np.asarray(states, dtype=np.float64)))
        return mapped.numpy()

    def _next_dt_request(self) -> float:
        return self.ph_env.sample_duration()

    def torque(self, action: np.ndarray) -> np.ndarray:
        """Map normalized u in [-1, 1] to motor torque u times the limit."""
        return np.asarray(action, dtype=np.float64) * self.torque_limit

    def _step_physics(
        self, action: np.ndarray, dt: float
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any], float]:
        # transition["decision"] is the torque the environment held over the interval:
        # this torque, clipped to the environment's torque box (end_state_reward_rate).
        transition = self.ph_env.step_dt(self.torque(action), dt)
        adapted = _adapt_transition(
            self.ph_env, transition, self.reward_units, self.reward_source
        )
        self.simulated_seconds += adapted["duration"]
        info = {"transition": transition, "adapted": adapted}
        return (
            self.network_input(adapted["next_observation"]),
            adapted["reward"],
            adapted["terminated"],
            adapted["truncated"],
            info,
            adapted["duration"],
        )

    def render(self, *args: Any, **kwargs: Any) -> np.ndarray:
        return self.ph_env.render(*args, **kwargs)

    def close(self) -> None:
        self.ph_env.close()


class _PHCTSAC(CTSAC):
    """Legacy CT-SAC with termination-only done flags and cumulative interaction counts.

    Warm-up. warmup_decisions None (the default; every arm before ct_sac_released): the
    legacy learning_starts is held at an unreachable count until warmup_episodes episodes
    have ended (failure or time limit), then set to 0; warmup_episodes <= 0 starts at 0.
    warmup_decisions an integer: it is CT-SAC's own learning_starts, counted in decisions,
    and episode ends never change it. The legacy loop (algorithms/off_policy.py learn,
    _sample_action) then draws uniform actions from the [-1, 1] box on every decision
    taken while the cumulative decision count is below it (exactly warmup_decisions
    random decisions) and trains from the decision that brings the count to it (once
    the replay also holds a batch); warmup_episodes is ignored.

    terminal_time_limit False (the default) stores done = terminated: truncation
    bootstraps. True stores done = terminated or truncated, the legacy loop's own flag.
    """

    def __init__(
        self,
        *args: Any,
        warmup_episodes: int,
        warmup_decisions: Optional[int] = None,
        terminal_time_limit: bool = False,
        **kwargs: Any,
    ) -> None:
        if warmup_decisions is None:
            kwargs["learning_starts"] = 0 if warmup_episodes <= 0 else _NO_LEGACY_LOG
        else:
            if isinstance(warmup_decisions, bool) or int(warmup_decisions) != warmup_decisions:
                raise ValueError(f"warmup_decisions must be an integer, not {warmup_decisions!r}.")
            if warmup_decisions < 0:
                raise ValueError("warmup_decisions must be nonnegative.")
            kwargs["learning_starts"] = int(warmup_decisions)
        super().__init__(*args, **kwargs)
        self.warmup_episodes = int(warmup_episodes)
        self.warmup_decisions = None if warmup_decisions is None else int(warmup_decisions)
        self.terminal_time_limit = bool(terminal_time_limit)
        self.finished_episodes = 0
        self.gradient_updates = 0

    def _setup_learn(
        self, total_timesteps: int, callback: Optional[BaseCallback]
    ) -> Tuple[int, Optional[BaseCallback]]:
        interactions = self.num_timesteps
        total_timesteps, callback = super()._setup_learn(total_timesteps, callback)
        self.num_timesteps = interactions
        self.env.resume_episode = interactions > 0
        return total_timesteps, callback

    def _store_transition(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        done: bool,
        next_obs: np.ndarray,
        t: float,
        next_t: float,
        infos: Optional[Mapping[str, Any]],
    ) -> None:
        """Store termination (and truncation under terminal_time_limit); a local time keeps
        Delta exact in the float32 columns."""
        adapted = infos["adapted"]
        done = adapted["terminated"] or (self.terminal_time_limit and adapted["truncated"])
        super()._store_transition(
            obs=obs,
            action=action,
            reward=reward,
            done=done,
            next_obs=next_obs,
            t=0.0,
            next_t=adapted["duration"],
            infos=None,
        )
        if adapted["terminated"] or adapted["truncated"]:
            self.finished_episodes += 1
            # An episode warm-up ends here; a decision warm-up is the loop's own count.
            if self.warmup_decisions is None and self.finished_episodes >= self.warmup_episodes:
                self.learning_starts = 0

    def train(self, gradient_steps: int, batch_size: int) -> None:
        super().train(gradient_steps, batch_size)
        self.gradient_updates += gradient_steps


class _MetricsCallback(BaseCallback):
    """Report protocol counters from the legacy loop after every interaction.

    on_rollout_start, which the legacy loop calls at the top of every iteration (after
    the previous interaction's reset and training), is the baseline's safe point.
    """

    def __init__(
        self,
        owner: TorqueCTSACBaseline,
        callback: Optional[Callable[[TorqueCTSACBaseline, Metrics], None]],
    ) -> None:
        super().__init__()
        self.owner = owner
        self.callback = callback

    def _on_step(self) -> bool:
        if self.callback is not None:
            self.callback(self.owner, self.owner.counters())
        return True

    def _on_rollout_start(self) -> None:
        self.owner._resume_point()


class TorqueCTSACBaseline:
    """Run Algorithm 1 as implemented, converting its environment and target conventions.

    Physical Delta becomes u=Delta/T with gamma=exp(-lambda T), so gamma^u is
    exp(-lambda Delta). Interval rewards become T times a rate, the failure charge
    stays in value units, and truncation bootstraps (time_limit "bootstrap", the
    default; "terminal" ends the continuation there instead, done = terminated or
    truncated, as CT-SAC's own loop does). The legacy update has no gradient clipping,
    so training["gradient_clip"] does not apply to this arm.

    Reward units (reward_units, the protocol key ct_sac_reward_units;
    common.ph_config.CT_SAC_REWARD_UNITS). "second" (the default, every run before the key):
    the reward and temperature above, in the per-second units of the environment's reward
    rate and of alpha_c, CT-SAC's own per-period scale multiplied by T. "reference_period":
    every value the legacy learner sees divided by T, the reward ((r - c) / u + c) / T
    (about the rate itself: a rate of 1 per second is a reward of 1 per reference period,
    dm_control's per-step scale that CT-SAC was released on), the failure charge c / T with
    it, and the temperature alpha = alpha_c T / T = alpha_c, so initial_alpha_c 1 starts
    the learner at CT-SAC's default alpha 1. Its critics then hold values divided by T. The
    recorded alpha_c stays the per-second temperature in the CT-PH-SAC arms' units,
    alpha / (T s) with s the reward scale, 1 under "second" and 1 / T under
    "reference_period" (alpha / period_temperature(1.0)); alpha_decision is the learner's
    own alpha under either. The setting is recorded in resume_state and in the counters
    sidecar of save when it is "reference_period", and a resume state, load or warm start
    written under other units is refused.

    Reward source (reward_source, the protocol key ct_sac_reward_source;
    common.ph_config.CT_SAC_REWARD_SOURCES). "interval" (the default, every run before the
    key): the interval part of the reward is (r - c) / u above, T times the rate averaged
    over the interval. "end_state": it is T rho(s', a), T times the environment's reward
    rate at the interval's end state s' with the torque a held over the interval
    (end_state_reward_rate), not scaled by the duration, as CT-SAC's own wrapper rewards
    the state that ends a step. The failure charge and the reward units apply as above,
    so under "reference_period" an interval without failure gives rho(s', a). The setting
    is recorded and checked as the reward units are (only when it is "end_state").

    Hyperparameters: step size training["learning_rate"] (actor, critics and the
    temperature), replay training["replay_capacity"], batch training["batch_size"],
    target rate polyak_rate, one gradient step per decision (the legacy train_freq and
    gradient_steps defaults). Warm-up: training["warmup_episodes"] whole episodes of
    uniform random actions, or, when warmup_decisions is given, CT-SAC's learning_starts
    in decisions (_PHCTSAC), which replaces the episode count.

    CT-SAC conventions: the actor and critics work in the normalized action
    u in [-1, 1] and the motor receives u times the symmetric torque limit (an
    asymmetric box raises); predict returns that torque, so evaluation is unchanged.
    The target entropy is CT-SAC's "auto", -action_dim in units of u, whatever
    temperature["target_entropy"] says: that value is the CT-PH-SAC arms' -1 per
    coordinate of their normalized 3m-dimensional feedback-decision box, a different
    space; this is the same -1 per coordinate on the m torques. A learned temperature
    is tuned at every gradient step. The temperature alpha is charged per reference
    period, as the reward is: CT-SAC's target charges an interval of u periods u alpha
    of entropy, so alpha = alpha_c T for the per-second alpha_c (period_temperature), and
    alpha_c under reward_units "reference_period".

    Network inputs: state_features None (the default) feeds the actor and critics the
    environment's state (positions with periodic angles wrapped to [-pi, pi), context,
    velocities, unscaled); a network config (a mapping read for velocity_feature_scale,
    default 1) feeds them CT-PH-SAC's StateFeatures of the state instead, the input map of
    the CT-PH-SAC arms' actor, critics, value and rate networks (_LegacyEnvAdapter.
    network_input; predict applies it to the states it is given).
    """

    algorithm: CTSAC
    # The legacy learner that __init__ builds, and the extra keyword arguments it receives
    # (_legacy_kwargs): _PHCTSAC with none here. algorithms/ct_sac_implicit's subclass
    # (the implicit value step) replaces both; ct_sac_torque is built exactly as before.
    _legacy_class: type = _PHCTSAC

    def __init__(
        self,
        env: MujocoPHEnv,
        training: TrainingConfig,
        temperature: TemperatureConfig,
        *,
        polyak_rate: float,
        expectation_samples: int,
        net_arch: Sequence[int] = (400, 300),
        log_std_init: float = -3.0,
        warmup_decisions: Optional[int] = None,
        time_limit: Literal["bootstrap", "terminal"] = "bootstrap",
        state_features: Optional[Mapping[str, object]] = None,
        reward_units: Literal["second", "reference_period"] = DEFAULT_CT_SAC_REWARD_UNITS,
        reward_source: Literal["interval", "end_state"] = DEFAULT_CT_SAC_REWARD_SOURCE,
    ) -> None:
        if env.decision_mode != "torque":
            raise ValueError("The CT-SAC baseline needs a torque-mode environment.")
        if expectation_samples != 1:
            raise ValueError(
                "expectation_samples must be 1: legacy CTSAC.train evaluates its target "
                "critics at the unrepeated observations, so more samples mismatch shapes."
            )
        if time_limit not in TIME_LIMIT_MODES:
            raise ValueError(f"time_limit must be one of {TIME_LIMIT_MODES}, not {time_limit!r}.")
        self.env = env
        self.time_limit = time_limit
        # Before period_temperature below, which reads it.
        self.reward_units = check_ct_sac_reward_units(reward_units)
        self.reward_source = check_ct_sac_reward_source(reward_source)
        self.reference_period = float(env.config["reference_period"])
        self.torque_limit = symmetric_torque_limit(env)
        self._wall_seconds = 0.0
        # perf_counter at the start of the learn call in progress, else None.
        self._learn_start: Optional[float] = None
        self._resume_hook: Optional[Callable[[TorqueCTSACBaseline], None]] = None

        alpha = self.period_temperature(temperature["initial_alpha_c"])
        self._learned_temperature = bool(temperature["learn_temperature"])
        if self._learned_temperature and alpha <= 0.0:
            raise ValueError("A learned temperature needs a positive initial value.")
        # The networks' input map (buffers only: building it draws no random number).
        self.input_features = None
        if state_features is not None:
            from models.ph_sac.networks import StateFeatures

            self.input_features = StateFeatures(env.state_spec, state_features)
        self.algorithm = self._legacy_class(
            _LegacyEnvAdapter(
                env,
                seed=training["seed"],
                features=self.input_features,
                reward_units=self.reward_units,
                reward_source=self.reward_source,
            ),
            model="ActorQCriticModel",
            model_kwargs={
                "q_net_arch": list(net_arch),
                "pi_net_arch": list(net_arch),
                "log_std_init": log_std_init,
            },
            device=training["device"],
            seed=training["seed"],
            gamma=period_discount(
                discount_rate=env.discount_rate,
                reference_period=self.reference_period,
            ),
            buffer_size=training["replay_capacity"],
            learning_rate=training["learning_rate"],
            batch_size=training["batch_size"],
            alpha=f"auto_{alpha!r}" if self._learned_temperature else alpha,
            tau=polyak_rate,
            num_expectation_samples=expectation_samples,
            # -action_dim in units of u; the config's value is for the 3m-dim decision box.
            target_entropy="auto",
            warmup_episodes=training["warmup_episodes"],
            warmup_decisions=warmup_decisions,
            terminal_time_limit=time_limit == "terminal",
            **self._legacy_kwargs(),
        )

        self.solver_records: list = []
        self.admission_records: list = []
        self.oracle_records: list = []
        self.calibration_records: list = []

    def _legacy_kwargs(self) -> Dict[str, Any]:
        """Extra keyword arguments of _legacy_class: none for CT-SAC itself."""
        return {}

    def adapt_transition(self, transition: Transition) -> Mapping[str, object]:
        """Bridge the env API; retain final observation, termination, and Delta/T. The
        reward is in this baseline's reward_units and from its reward_source, as the legacy
        loop stores it; under "end_state" the transition's decision must be the torque held
        over the interval (as every transition of this torque-mode environment's step_dt)."""
        return _adapt_transition(self.env, transition, self.reward_units, self.reward_source)

    def normalized_durations(self, durations: Tensor) -> Tensor:
        """Convert physical Delta to u=Delta/T for gamma^u, gamma=exp(-lambda T)."""
        return durations / self.reference_period

    def reward_rate(self, batch: TransitionBatch) -> Tensor:
        """Convert interval rewards to T*rate: (r-c)/u+c, c the charge in a terminated row;
        under reward_source "end_state" T rho(s', a) + c instead, rho the task's tensor
        rate (reward_rate_tensor) at next_states with the held torques, decisions (needed
        then); divided by T under reward_units "reference_period" (about the rate itself),
        as adapt_transition converts one interval."""
        durations = batch["durations"]
        payoff = terminal_payoff_tensor(batch["next_states"], self.env.config)
        charge = (
            batch["terminated"] * th.exp(-self.env.discount_rate * durations) * payoff
        )
        if self.reward_source == "end_state":
            rates = reward_rate_tensor(batch["next_states"], batch["decisions"], self.env.config)
            converted = self.reference_period * rates + charge
        else:
            converted = (
                (batch["rewards"] - charge) / self.normalized_durations(durations) + charge
            )
        if self.reward_units == "reference_period":
            return converted / self.reference_period
        return converted

    def period_temperature(self, alpha_c: float) -> float:
        """The legacy alpha for the per-second alpha_c.

        reward_units "second": alpha = alpha_c T, the legacy alpha charged per reference
        period T (u alpha for an interval of u = Delta / T periods), alpha_c per second.
        "reference_period": the rewards are divided by T, and the temperature with them,
        alpha = alpha_c T / T = alpha_c, returned as alpha_c itself (exactly 1.0 for 1.0).
        Its inverse, alpha / period_temperature(1.0), is the per-second alpha_c that
        policy_diagnostics and warm_start record.
        """
        if self.reward_units == "reference_period":
            return float(alpha_c)
        return float(alpha_c) * self.reference_period

    def policy_diagnostics(self) -> Dict[str, Optional[float]]:
        """The temperature and the policy's spread, as every evaluation records them (the
        keys of the CT-PH-SAC arms' policy_diagnostics).

        alpha_decision: the legacy learner's temperature alpha (tuned or fixed), in
        CT-SAC's own units, per reference period T (u alpha for an interval of u periods),
        under either reward_units; alpha_c: the per-second temperature it corresponds to in
        the CT-PH-SAC arms' units (period_temperature's inverse): alpha / T under
        "second", alpha itself under "reference_period" (alpha_c T / T); log_std_mean: the
        actor's mean log standard deviation over u in [-1, 1]^m. entropy_estimate stays
        None (entropy_rows 0): the legacy actor samples from the global torch stream, which
        the run's resumable state carries, so it is not drawn from here.
        """
        alpha = float(self.algorithm.alpha)
        log_std = getattr(self.algorithm.model.actor, "log_std", None)
        return {
            "alpha_c": alpha / self.period_temperature(1.0),
            "alpha_decision": alpha,
            "log_std_mean": None if log_std is None else float(log_std.detach().mean()),
            "entropy_estimate": None,
            "entropy_rows": 0,
        }

    def predict(self, state: ndarray, deterministic: bool = True) -> ndarray:
        """Return torque, the policy's u times the limit, held until the next decision."""
        state = self.algorithm.env.network_input(state)
        action = self.algorithm._policy_act(
            np.asarray(state, dtype=np.float32), deterministic=deterministic
        )
        return np.asarray(action, dtype=np.float64) * self.torque_limit

    def learn(
        self,
        total_interactions: int,
        *,
        callback: Optional[Callable[[TorqueCTSACBaseline, Metrics], None]] = None,
    ) -> TorqueCTSACBaseline:
        """Run until the cumulative decision count reaches the budget, warm-up included."""
        bridge = None
        if callback is not None or self._resume_hook is not None:
            bridge = _MetricsCallback(self, callback)
        start = time.perf_counter()
        self._learn_start = start
        try:
            self.algorithm.learn(
                total_interactions, callback=bridge, log_interval=_NO_LEGACY_LOG
            )
        finally:
            self._wall_seconds += time.perf_counter() - start
            self._learn_start = None
        return self

    # --- exact restarts ------------------------------------------------------

    def set_resume_hook(self, hook: Optional[Callable[[TorqueCTSACBaseline], None]]) -> None:
        """Call hook(self) at the top of every iteration of the legacy loop (a safe point:
        the previous interaction's reset and training are done); None removes it."""
        self._resume_hook = hook

    def _resume_point(self) -> None:
        if self._resume_hook is not None:
            self._resume_hook(self)

    def resume_state(self) -> Mapping[str, object]:
        """Everything a restart at a safe point needs, beyond the global RNG streams.

        Networks, optimizers, temperature, the legacy replay's filled rows (tensor views
        of the live arrays: save before the loop moves on), its position, the loop's
        counters and last observation, the adapter's clock and action-space stream, and
        the MuJoCo environment. The legacy loop draws from the global torch (policy
        noise), numpy (replay indices) and python streams, which
        evaluations/ph_sac/resume.py saves beside this state. Under reward_units
        "reference_period" it also holds "reward_units" (absent: "second", every state
        written before the key), and under reward_source "end_state" "reward_source"
        (absent: "interval", every state written before that key), which
        load_resume_state checks.
        """
        algorithm = self.algorithm
        adapter = algorithm.env
        buffer = algorithm.replay_buffer
        model = algorithm.model
        filled = buffer.size()
        wall = self._wall_seconds
        if self._learn_start is not None:
            wall += time.perf_counter() - self._learn_start
        actor_target = getattr(model, "actor_target", None)
        state = {
            "model": {
                "actor": model.actor.state_dict(),
                "critics": model.q_nets.state_dict(),
                "critic_targets": model.q_target_nets.state_dict(),
                "actor_target": None if actor_target is None else actor_target.state_dict(),
            },
            "optimizers": {
                "actor": algorithm.actor_optimizer.state_dict(),
                "critic": algorithm.critic_optimizer.state_dict(),
                "alpha": None
                if algorithm.alpha_optimizer is None
                else algorithm.alpha_optimizer.state_dict(),
            },
            "log_alpha": None
            if algorithm.log_alpha is None
            else algorithm.log_alpha.detach().clone(),
            "alpha": algorithm.alpha,
            "replay": {
                "pos": buffer.pos,
                "full": buffer.full,
                "rows": filled,
                "arrays": {
                    name: th.from_numpy(getattr(buffer, name)[:filled])
                    for name in _LEGACY_REPLAY_FIELDS
                },
            },
            "algorithm": {
                "num_timesteps": algorithm.num_timesteps,
                "finished_episodes": algorithm.finished_episodes,
                "learning_starts": algorithm.learning_starts,
                "gradient_updates": algorithm.gradient_updates,
                "n_updates": algorithm._n_updates,
                "progress_remaining": algorithm._progress_remaining,
                "last_obs": None
                if algorithm._last_obs is None
                else np.array(algorithm._last_obs, copy=True),
                "ep_info_buffer": list(algorithm.ep_info_buffer),
            },
            "adapter": {
                "cur_t": adapter.cur_t,
                "step_index": adapter._step_index,
                "last_obs": None
                if adapter._last_obs is None
                else np.array(adapter._last_obs, copy=True),
                "time_points": adapter._time_points,
                "np_random": None
                if adapter._np_random is None
                else adapter._np_random.bit_generator.state,
                "action_space": adapter.action_space.np_random.bit_generator.state,
                "simulated_seconds": adapter.simulated_seconds,
            },
            "environment": self.env.snapshot(),
            "wall_seconds": wall,
        }
        # Only off the default, so that a default state is that of the code before the key.
        if self.reward_units != DEFAULT_CT_SAC_REWARD_UNITS:
            state["reward_units"] = self.reward_units
        # The same for the reward source.
        if self.reward_source != DEFAULT_CT_SAC_REWARD_SOURCE:
            state["reward_source"] = self.reward_source
        return state

    def _check_reward_units(self, stored: object, where: str) -> None:
        """Raise ValueError unless stored (a state's or sidecar's reward_units, the default
        when absent) is this baseline's: the critics, temperature and replay rewards it
        holds are in its units."""
        if stored != self.reward_units:
            raise ValueError(
                f"{where} was written with ct_sac_reward_units {stored!r}, but this baseline "
                f"runs with {self.reward_units!r}: its critics, temperature and rewards are "
                "in other units."
            )

    def _check_reward_source(self, stored: object, where: str) -> None:
        """Raise ValueError unless stored (a state's or sidecar's reward_source, the
        default when absent) is this baseline's: the critics and replay rewards it holds
        are of its reward."""
        if stored != self.reward_source:
            raise ValueError(
                f"{where} was written with ct_sac_reward_source {stored!r}, but this "
                f"baseline runs with {self.reward_source!r}: its critics and rewards are of "
                "another reward."
            )

    def _check_reward_settings(self, state: Mapping[str, object], where: str) -> None:
        """Both checks on a resume state or counters sidecar (absent keys: the defaults)."""
        self._check_reward_units(state.get("reward_units", DEFAULT_CT_SAC_REWARD_UNITS), where)
        self._check_reward_source(
            state.get("reward_source", DEFAULT_CT_SAC_REWARD_SOURCE), where
        )

    def load_resume_state(self, state: Mapping[str, object]) -> None:
        """Restore resume_state() into a baseline built from the same settings; a state
        written under other reward_units or reward_source raises ValueError before anything
        is restored."""
        self._check_reward_settings(state, "The resume state")
        algorithm = self.algorithm
        adapter = algorithm.env
        buffer = algorithm.replay_buffer
        model = algorithm.model
        saved = state["model"]
        model.actor.load_state_dict(saved["actor"])
        model.q_nets.load_state_dict(saved["critics"])
        model.q_target_nets.load_state_dict(saved["critic_targets"])
        if saved["actor_target"] is not None:
            model.actor_target.load_state_dict(saved["actor_target"])
        optimizers = state["optimizers"]
        algorithm.actor_optimizer.load_state_dict(optimizers["actor"])
        algorithm.critic_optimizer.load_state_dict(optimizers["critic"])
        if optimizers["alpha"] is not None:
            algorithm.alpha_optimizer.load_state_dict(optimizers["alpha"])
        if state["log_alpha"] is not None:
            with th.no_grad():
                algorithm.log_alpha.copy_(state["log_alpha"])
        algorithm.alpha = state["alpha"]
        replay = state["replay"]
        rows = int(replay["rows"])
        for name in _LEGACY_REPLAY_FIELDS:
            array = getattr(buffer, name)
            array[...] = 0
            array[:rows] = np.asarray(replay["arrays"][name])
        buffer.pos = int(replay["pos"])
        buffer.full = bool(replay["full"])
        loop = state["algorithm"]
        algorithm.num_timesteps = loop["num_timesteps"]
        algorithm.finished_episodes = loop["finished_episodes"]
        algorithm.learning_starts = loop["learning_starts"]
        algorithm.gradient_updates = loop["gradient_updates"]
        algorithm._n_updates = loop["n_updates"]
        algorithm._progress_remaining = loop["progress_remaining"]
        algorithm._last_obs = None if loop["last_obs"] is None else np.array(loop["last_obs"])
        algorithm.ep_info_buffer.clear()
        algorithm.ep_info_buffer.extend(loop["ep_info_buffer"])
        clock = state["adapter"]
        adapter.cur_t = clock["cur_t"]
        adapter._step_index = clock["step_index"]
        adapter._last_obs = None if clock["last_obs"] is None else np.array(clock["last_obs"])
        adapter._time_points = clock["time_points"]
        if clock["np_random"] is not None:
            if adapter._np_random is None:
                adapter._np_random = np.random.default_rng()
            adapter._np_random.bit_generator.state = clock["np_random"]
        adapter.action_space.np_random.bit_generator.state = clock["action_space"]
        adapter.simulated_seconds = clock["simulated_seconds"]
        self.env.restore(state["environment"])
        self._wall_seconds = float(state["wall_seconds"])

    def counters(self) -> Metrics:
        updates = self.algorithm.gradient_updates
        return {
            "interactions": self.algorithm.num_timesteps,
            "simulated_seconds": self.algorithm.env.simulated_seconds,
            "critic_gradient_steps": updates,
            "value_gradient_steps": 0,
            "rate_gradient_steps": 0,
            "actor_gradient_steps": updates,
            "temperature_gradient_steps": updates if self._learned_temperature else 0,
            "plant_gradient_steps": 0,
            "model_rollout_paths": 0,
            "gate_simulated_paths": 0,
            "gate_real_rows_acquired": 0,
            "wall_seconds": self._wall_seconds,
        }

    def _counters_path(self, path: Path) -> Path:
        path = Path(path)
        return path.with_name(path.name + ".counters.json")

    def save(self, path: Path, include_replay: bool = True) -> None:
        """Save legacy model parameters; a sidecar holds counters and the temperature.

        include_replay matches the CT-PH-SAC agents' save; this save never holds a replay.
        Under reward_units "reference_period" the sidecar also holds "reward_units" (absent:
        "second"), and under reward_source "end_state" "reward_source" (absent:
        "interval"), which load and warm_start check.
        """
        self.algorithm.save(str(path))
        log_alpha = self.algorithm.log_alpha
        state = {
            "interactions": self.algorithm.num_timesteps,
            "simulated_seconds": self.algorithm.env.simulated_seconds,
            "gradient_updates": self.algorithm.gradient_updates,
            "finished_episodes": self.algorithm.finished_episodes,
            "learning_starts": self.algorithm.learning_starts,
            "wall_seconds": self._wall_seconds,
            "alpha": self.algorithm.alpha,
            "log_alpha": None if log_alpha is None else float(log_alpha.item()),
        }
        # Only off the default, so that a default sidecar is that of the code before the key.
        if self.reward_units != DEFAULT_CT_SAC_REWARD_UNITS:
            state["reward_units"] = self.reward_units
        if self.reward_source != DEFAULT_CT_SAC_REWARD_SOURCE:
            state["reward_source"] = self.reward_source
        self._counters_path(path).write_text(json.dumps(state, indent=2))

    def warm_start(
        self, path: Path, *, require_interactions: Optional[int] = None
    ) -> Dict[str, object]:
        """Start this fresh baseline from a finished run's checkpoint.pt (what save wrote).

        Loaded: the actor, both critics and their targets (the file), and log alpha from
        the counters sidecar when the temperature is tuned (alpha follows it). Kept fresh:
        the optimizers (the file holds none), the replay (empty: the source's rows came
        from the robot before the change), the counters and the random streams. The
        warm-up is skipped: the policy acts from the first decision, and training starts
        once the replay holds one batch, as after any warm-up. require_interactions, when
        given, must be the source's interaction count; the source's reward_units and
        reward_source (its sidecar's, "second" and "interval" when absent) must be this
        baseline's. Returns what was loaded.
        """
        algorithm = self.algorithm
        if algorithm.num_timesteps or algorithm.replay_buffer.size():
            raise RuntimeError("A warm start needs a fresh baseline with an empty replay.")
        path = Path(path)
        state = json.loads(self._counters_path(path).read_text())
        self._check_reward_settings(state, str(path))
        interactions = int(state["interactions"])
        if require_interactions is not None and interactions != int(require_interactions):
            raise ValueError(
                f"{path} holds a run at {interactions} decisions, not the required "
                f"{int(require_interactions)}."
            )
        algorithm.model.load_state(str(path))
        if self._learned_temperature:
            if state.get("log_alpha") is None:
                raise ValueError("A tuned temperature needs a source run that tuned its own.")
            with th.no_grad():
                algorithm.log_alpha.fill_(float(state["log_alpha"]))
            algorithm.alpha = float(algorithm.log_alpha.detach().exp().item())
        algorithm.warmup_episodes = 0
        algorithm.learning_starts = 0
        return {
            "source_interactions": interactions,
            "networks": ["actor", "critics", "critic_targets"],
            "alpha_decision": float(algorithm.alpha),
            "alpha_c": float(algorithm.alpha) / self.period_temperature(1.0),
        }

    def load(self, path: Path) -> None:
        """Restore what save wrote; the legacy mechanism keeps no optimizer or replay state.
        A file saved under other reward_units or reward_source (its sidecar's, "second" and
        "interval" when absent) raises ValueError before anything is loaded."""
        state = json.loads(self._counters_path(path).read_text())
        self._check_reward_settings(state, str(path))
        self.algorithm.load(str(path))
        self.algorithm.num_timesteps = state["interactions"]
        self.algorithm.env.simulated_seconds = state["simulated_seconds"]
        self.algorithm.gradient_updates = state["gradient_updates"]
        self.algorithm.finished_episodes = state["finished_episodes"]
        self.algorithm.learning_starts = state["learning_starts"]
        self._wall_seconds = state["wall_seconds"]
        self.algorithm.alpha = state["alpha"]
        if state["log_alpha"] is not None and self.algorithm.log_alpha is not None:
            with th.no_grad():
                self.algorithm.log_alpha.fill_(state["log_alpha"])

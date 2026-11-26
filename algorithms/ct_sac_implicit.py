"""CT-SAC with an implicit value step: the arms ct_sac_implicit, ct_sac_implicit_schedule
and ct_sac_implicit_released (evaluations/ph_sac/protocol.baseline_implicit_arms).

CT-SAC (algorithms/ct_sac.py) regresses its twin critics on the bracket
    F(Q)(s, a) = r + (1 - d) [V(s) + (gamma^u V(s') - V(s)) / u],
    V(x) = E_{a ~ pi}[min_j Q_j(x, a) - alpha log pi(a | x)],
built from Polyak-averaged target critics (T the reference period, u = Delta / T, r the
reward per period, gamma = e^(-lambda T)). That is the relaxed explicit step
Q_(k+1) = (1 - eta) Q_k + eta F(Q_k), the explicit Euler step of size eta T of the flow
dQ/dsigma = (F(Q) - Q) / T. This learner takes the implicit Euler step of size tau of the
same flow,
    (1 + c) Q - c F(Q) = Q_k,   c = tau / T,
and solves it by K residual-gradient steps on
    R_i^(j) = (1 + c) Q_j(s, a) - Qbar_j(s, a) - c F_i,
    F_i     = r + (1 - d) [V_i(s) + (gamma^u V_i(s') - V_i(s)) / u],
    V_i(x)  = min_j Q_j(x, a_i(x)) - alpha_k log pi_k(a_i(x) | x),
with N joint draws a_i(s), a_i(s') ~ pi_k and the ONLINE critics in V_i: the gradient
reaches Q through Q_j(s, a), V_i(s) and V_i(s'). Detaching V_i(s') would leave an exact
solve contracting by about 1 - lambda Delta per copy, the plain interval backup; the
implicit gain lives in the gradient through the next-state value. The loss is
sum_j of the all-pairs U-statistic over the draws (algorithms/ph_resolvent.all_pairs_loss),
whose expectation and gradient expectation are those of (E_draws R^(j))^2: the action
draws carry no double-sampling bias; the clock's one Delta per stored row still does on an
irregular clock (about (tau^2 / 4) Var(Delta) times the squared second time derivative of
the discounted value along the held path), and nothing does on a fixed clock with a
deterministic plant.

The frozen operator: Qbar_j are the legacy model's target critics (q_target_nets), hard
copied from the critics every K training rows (never Polyak-averaged: the arm's
polyak_rate is unused); pi_k (frozen_actor) and alpha_k (operator_alpha) are copied from
the actor and the temperature at the same moment. Its fixed point is CT-SAC's, Q = F(Q);
on the policy average the exact step contracts by 1 / (1 + tau E[(1 - e^(-lambda Delta)) /
Delta]), about 1 / (1 + lambda tau), for every u, short intervals (u < 1) included, and the
action-dependent remainder by 1 / (1 + c).

Actor and temperature. actor_schedule "per_row" (ct_sac_implicit): one temperature step
and one actor step per training row, exactly as CT-SAC takes them (same losses, same
order around the critic step, the actor reading min_j Q_j of the online critics);
"outer" (ct_sac_implicit_schedule): the policy is frozen through each solve, and after
each copy the actor takes actor_steps steps and the temperature the steps owed, one per
temperature_every_rows training rows, as implicit CT-PH-SAC schedules them
(algorithms/implicit_ct_ph_sac.py run_outer_iteration, tune_temperature).

Diagnostics: one record per copy in solver_records (evaluations/ph_sac/diagnostics.
SolverMetrics with the optional fields residual_u_statistic, q_policy_minus_stored,
q_abs_max and interactions), from the K-th step of the solve, before its gradient step.

Everything else is ct_sac_torque's (algorithms/ph_baseline.TorqueCTSACBaseline): the
adapter, the normalized actions, the reward per period, the discount from the task, the
done flag (bootstrapping through a time limit), the networks, step size, batch, replay,
warm-up and the temperature's initial value and target. The parent CT-SAC class is not
changed; this learner never reads num_expectation_samples (CT-SAC's own draws), so the
repeat issue of algorithms/ct_sac.py at N_e > 1 does not arise here.

Reward units (TorqueCTSACBaseline reward_units, the protocol key ct_sac_reward_units)
apply here unchanged: r is the stored replay reward (the adapter's, in those units) and
alpha_k the learner's temperature (period_temperature), and nothing else reads a reward
scale. F is linear in r and V_i, V_i is positively homogeneous in (Q, alpha_k), and R is
linear in (Q, Qbar, F); c = tau / T, u, gamma^u and the contraction factor are ratios of
times. So "reference_period" (r and alpha divided by T) divides every residual by T at
critics divided by T, and the solve's fixed point is the default's divided by T, as for
CT-SAC's own update. The solver records' residual_u_statistic (squared residuals),
q_policy_minus_stored and q_abs_max are in the learner's units, so they scale by 1 / T^2,
1 / T and 1 / T against "second". The reward source (TorqueCTSACBaseline reward_source,
the protocol key ct_sac_reward_source) applies the same way: r is the stored replay reward,
the adapter's from that source.
"""

from __future__ import annotations

import copy
import math
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional

import torch as th

from algorithms.ph_baseline import TorqueCTSACBaseline, _PHCTSAC
from algorithms.ph_resolvent import all_pairs_loss

if TYPE_CHECKING:
    from pathlib import Path

    from torch import Tensor

    from common.buffers import ReplayBatch
    from common.ph_types import Metrics

# How the actor and the temperature step (common.ph_config.CT_SAC_ACTOR_SCHEDULES).
ACTOR_SCHEDULES = ("per_row", "outer")
# CT-SAC's own guard on the interval ratio u in its target (algorithms/ct_sac.py).
_U_EPSILON = 1e-8


def implicit_bracket(
    soft_here: Tensor,
    soft_next: Tensor,
    rewards: Tensor,
    dones: Tensor,
    u: Tensor,
    gamma_u: Tensor,
) -> Tensor:
    """CT-SAC's critic target per draw, [B,N]: r + (1-d)[V_i(s) + (gamma^u V_i(s') - V_i(s))/u].

    soft_here and soft_next are V_i(s) and V_i(s') [B,N]; rewards, dones, u and gamma_u
    [B,1]. The expression of algorithms/ct_sac.py's q_fast_target, with V from draw i.
    """
    fraction = (gamma_u * soft_next - soft_here) / (u + _U_EPSILON)
    return rewards + (1.0 - dones) * (soft_here + fraction)


def resolvent_residuals(
    stored: Tensor, anchors: Tensor, bracket: Tensor, proximal_weight: float
) -> Tensor:
    """R_i^(j) = (1 + c) Q_j(s, a) - Qbar_j(s, a) - c F_i, [C,B,N].

    stored Q_j(s, a) and anchors Qbar_j(s, a) are [C,B,1], bracket F_i [B,N] and
    c = proximal_weight = tau / T. At c = 0 the residual is Q_j - Qbar_j for every draw.
    """
    c = float(proximal_weight)
    return (1.0 + c) * stored - anchors - c * bracket.unsqueeze(0)


def implicit_loss(residuals: Tensor) -> Tensor:
    """sum_j of the all-pairs U-statistic of critic j's residuals [C,B,N] over the N draws."""
    return sum(all_pairs_loss(residuals[j]) for j in range(residuals.shape[0]))


class _ImplicitPHCTSAC(_PHCTSAC):
    """The legacy learner of ct_sac_torque (_PHCTSAC) with the implicit value step.

    resolvent_step tau (s), inner_steps K (training rows per solve), policy_draws N (>= 2),
    actor_schedule ("per_row" or "outer"), and for "outer" actor_steps (actor steps per
    copy) and temperature_every_rows (training rows per temperature step). The other
    arguments are _PHCTSAC's.
    """

    def __init__(
        self,
        *args: Any,
        resolvent_step: float,
        inner_steps: int,
        policy_draws: int,
        actor_schedule: str = "per_row",
        actor_steps: int = 40,
        temperature_every_rows: float = 40.0,
        **kwargs: Any,
    ) -> None:
        if actor_schedule not in ACTOR_SCHEDULES:
            raise ValueError(
                f"actor_schedule must be one of {ACTOR_SCHEDULES}, not {actor_schedule!r}."
            )
        if isinstance(resolvent_step, bool) or not math.isfinite(float(resolvent_step)) or (
            float(resolvent_step) <= 0.0
        ):
            raise ValueError("resolvent_step must be a positive number of seconds.")
        for name, value, minimum in (
            ("inner_steps", inner_steps, 1),
            ("policy_draws", policy_draws, 2),
            ("actor_steps", actor_steps, 0),
        ):
            if isinstance(value, bool) or int(value) != value or int(value) < minimum:
                raise ValueError(f"{name} must be an integer of at least {minimum}.")
        if isinstance(temperature_every_rows, bool) or not float(temperature_every_rows) > 0.0:
            raise ValueError("temperature_every_rows must be positive.")
        super().__init__(*args, **kwargs)
        self.resolvent_step = float(resolvent_step)
        self.steps_per_solve = int(inner_steps)
        self.policy_draws = int(policy_draws)
        self.actor_schedule = str(actor_schedule)
        self.actor_steps_per_solve = int(actor_steps)
        self.temperature_every_rows = float(temperature_every_rows)
        # c = tau / T: CT-SAC's update is the explicit step of size eta T of
        # dQ/dsigma = (F(Q) - Q) / T, and this is the implicit step of size tau.
        self.proximal_weight = self.resolvent_step * self.time_rescale
        # lambda = beta / T (beta = -log gamma = lambda T); the nominal contraction factor.
        self.discount_rate = self.beta * self.time_rescale
        self.contraction = 1.0 / (1.0 + self.discount_rate * self.resolvent_step)
        # The frozen operator: anchors Qbar (the target critics), policy pi_k, temperature.
        self.model.q_target_nets.load_state_dict(self.model.q_nets.state_dict())
        self.frozen_actor = copy.deepcopy(self.model.actor)
        self.frozen_actor.requires_grad_(False)
        self.operator_alpha = float(self._current_alpha().item())
        # Training rows in the solve in progress, completed solves, actor steps taken,
        # training rows counted for the temperature and temperature steps taken ("outer").
        self.inner_steps = 0
        self.outer_iterations = 0
        self.actor_updates = 0
        self.tuning_rows = 0
        self.tuning_steps = 0
        self.solver_records: List[Dict[str, object]] = []

    # --- the value step ------------------------------------------------------------

    def _current_alpha(self) -> Tensor:
        """The temperature in force: exp(log alpha) when tuned, else the fixed value."""
        if self.log_alpha is not None:
            return th.exp(self.log_alpha.detach())
        return self.alpha_tensor

    def implicit_terms(self, batch: ReplayBatch) -> Dict[str, Tensor]:
        """The residuals of one batch and their parts, with the graph to the critics.

        values [C, B + 2BN, 1]: the online critics at (s, a), at (s, a_i(s)) and at
        (s', a_i(s')), one pass each; stored [C,B,1], q_here and q_next [C,B,N] views of it;
        anchors [C,B,1] the target critics at (s, a); log_here, log_next [B,N] the frozen
        policy's log densities of its draws; soft_here, soft_next [B,N]; bracket [B,N];
        residuals [C,B,N]. Draws, log densities and anchors carry no graph.
        """
        obs = batch.observations
        actions = batch.actions
        next_obs = batch.next_observations
        rows = obs.shape[0]
        draws = self.policy_draws
        model = self.model
        with th.no_grad():
            here = obs.repeat_interleave(draws, dim=0)
            there = next_obs.repeat_interleave(draws, dim=0)
            actions_here, log_here, _ = self.frozen_actor(here)
            actions_there, log_there, _ = self.frozen_actor(there)
            anchors = th.stack(model.target_q_values(obs, actions), dim=0)
        values = th.stack(
            model.q_values(
                th.cat([obs, here, there], dim=0),
                th.cat([actions, actions_here, actions_there], dim=0),
            ),
            dim=0,
        )
        critics = values.shape[0]
        stored = values[:, :rows]
        q_here = values[:, rows : rows + rows * draws].reshape(critics, rows, draws)
        q_next = values[:, rows + rows * draws :].reshape(critics, rows, draws)
        alpha = th.as_tensor(self.operator_alpha, dtype=values.dtype, device=values.device)
        log_here = log_here.reshape(rows, draws)
        log_next = log_there.reshape(rows, draws)
        soft_here = q_here.min(dim=0).values - alpha * log_here
        soft_next = q_next.min(dim=0).values - alpha * log_next
        u = batch.dt * self.time_rescale
        gamma_u = th.exp(-self.beta * u)
        bracket = implicit_bracket(soft_here, soft_next, batch.rewards, batch.dones, u, gamma_u)
        residuals = resolvent_residuals(stored, anchors, bracket, self.proximal_weight)
        return {
            "values": values,
            "stored": stored,
            "q_here": q_here,
            "q_next": q_next,
            "anchors": anchors,
            "log_here": log_here,
            "log_next": log_next,
            "soft_here": soft_here,
            "soft_next": soft_next,
            "bracket": bracket,
            "residuals": residuals,
        }

    @staticmethod
    def _solve_monitor(terms: Mapping[str, Tensor], loss: Tensor) -> Dict[str, float]:
        """The K-th step's diagnostics, before its gradient step: the loss per critic (the
        U-statistic, an unbiased estimate of mean_s (E_draws R)^2, so it can be negative);
        the batch mean of [mean_i min_j Q_j(s, a_i) - min_j Q_j(s, a)], the critics at the
        frozen policy's draws minus at the stored actions; max |Q| over every critic value
        the step evaluated."""
        with th.no_grad():
            critics = terms["residuals"].shape[0]
            policy = terms["q_here"].min(dim=0).values.mean(dim=1, keepdim=True)
            stored = terms["stored"].min(dim=0).values
            return {
                "residual_u_statistic": float(loss.detach()) / critics,
                "q_policy_minus_stored": float((policy - stored).mean()),
                "q_abs_max": float(terms["values"].detach().abs().max()),
            }

    def train(self, gradient_steps: int, batch_size: int) -> None:
        """One inner step per training row; the K-th step of a solve ends it (_end_solve)."""
        for _ in range(gradient_steps):
            batch = self.replay_buffer.sample(batch_size)
            obs = batch.observations
            per_row = self.actor_schedule == "per_row"
            if per_row:
                # As CT-SAC: the actor's draw on the batch, then the temperature step.
                actions_pi, log_prob_pi = self.model.act(obs)
                alpha_tensor = self._temperature_step(log_prob_pi)
            terms = self.implicit_terms(batch)
            loss = implicit_loss(terms["residuals"])
            monitor = None
            if self.inner_steps + 1 >= self.steps_per_solve:
                monitor = self._solve_monitor(terms, loss)
            self.critic_optimizer.zero_grad()
            loss.backward()
            self.critic_optimizer.step()
            if per_row:
                self._actor_step(obs, actions_pi, log_prob_pi, alpha_tensor)
                self.actor_updates += 1
            elif self.log_alpha is not None:
                self.tuning_rows += 1
            self.inner_steps += 1
            self.gradient_updates += 1
            if self.inner_steps >= self.steps_per_solve:
                self._end_solve(batch_size, monitor)

    # --- actor and temperature (CT-SAC's losses) ---------------------------------------

    def _temperature_step(self, log_prob: Tensor) -> Tensor:
        """CT-SAC's temperature step (algorithms/ct_sac.py); returns the alpha before it."""
        alpha_loss = None
        if self.alpha_optimizer is not None and self.log_alpha is not None:
            alpha_tensor = th.exp(self.log_alpha.detach())
            alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        else:
            alpha_tensor = self.alpha_tensor
        self.alpha = float(alpha_tensor.detach().item())
        if alpha_loss is not None:
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
        return alpha_tensor

    def _actor_step(
        self, obs: Tensor, actions_pi: Tensor, log_prob_pi: Tensor, alpha_tensor: Tensor
    ) -> None:
        """CT-SAC's actor step: alpha log pi - min_j Q_j on the online critics, held fixed."""
        for parameter in self.model.critic_parameters:
            parameter.requires_grad = False
        try:
            q_values_pi = self.model.min_q(obs, actions_pi)
            actor_loss = (alpha_tensor * log_prob_pi - q_values_pi).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
        finally:
            for parameter in self.model.critic_parameters:
                parameter.requires_grad = True

    def _outer_policy_steps(self, batch_size: int) -> None:
        """actor_steps actor steps, then the temperature steps owed (schedule "outer")."""
        alpha_tensor = self._current_alpha()
        for _ in range(self.actor_steps_per_solve):
            obs = self.replay_buffer.sample(batch_size).observations
            actions_pi, log_prob_pi = self.model.act(obs)
            self._actor_step(obs, actions_pi, log_prob_pi, alpha_tensor)
            self.actor_updates += 1
        if self.log_alpha is None:
            return
        owed = int(
            math.floor(
                self.tuning_rows / self.temperature_every_rows - self.tuning_steps + 1e-9
            )
        )
        for _ in range(owed):
            obs = self.replay_buffer.sample(batch_size).observations
            with th.no_grad():
                _, log_prob = self.model.act(obs)
            self._temperature_step(log_prob)
            self.tuning_steps += 1
        self.alpha = float(self._current_alpha().item())

    # --- copies -------------------------------------------------------------------------

    def _freeze_operator_policy(self) -> None:
        """pi_k <- the actor, alpha_k <- the temperature in force."""
        self.frozen_actor.load_state_dict(self.model.actor.state_dict())
        self.operator_alpha = float(self._current_alpha().item())

    def _end_solve(self, batch_size: int, monitor: Optional[Mapping[str, float]]) -> None:
        """Hard copy Qbar <- Q; the outer schedule's actor and temperature steps; then
        pi_k and alpha_k; one solver record."""
        self.model.q_target_nets.load_state_dict(self.model.q_nets.state_dict())
        if self.actor_schedule == "outer":
            self._outer_policy_steps(batch_size)
        self._freeze_operator_policy()
        record: Dict[str, object] = {
            "outer_iteration": self.outer_iterations,
            "inner_iterations": self.inner_steps,
            "status": "completed",
            "contraction_factor": self.contraction,
            # No plant: the operator moves only with the policy and the temperature.
            "fixed_point_motion_input": 0.0,
            "interactions": int(self.num_timesteps),
        }
        record.update(monitor or {})
        self.solver_records.append(record)
        self.inner_steps = 0
        self.outer_iterations += 1

    def reset_operator(self, *, copy_critics: bool = True) -> None:
        """Start a fresh solve from the networks as they stand (warm start, load)."""
        if copy_critics:
            self.model.q_target_nets.load_state_dict(self.model.q_nets.state_dict())
        self._freeze_operator_policy()
        self.inner_steps = 0

    # --- exact restarts -----------------------------------------------------------------

    def implicit_state(self) -> Dict[str, object]:
        """The frozen policy and temperature, the solve's position and the records (the
        anchors are the target critics, which the adapter's resume state carries)."""
        return {
            "frozen_actor": {
                key: tensor.detach().clone()
                for key, tensor in self.frozen_actor.state_dict().items()
            },
            "operator_alpha": self.operator_alpha,
            "inner_steps": self.inner_steps,
            "outer_iterations": self.outer_iterations,
            "actor_updates": self.actor_updates,
            "tuning_rows": self.tuning_rows,
            "tuning_steps": self.tuning_steps,
            "solver_records": copy.deepcopy(self.solver_records),
        }

    def load_implicit_state(self, state: Mapping[str, object]) -> None:
        self.frozen_actor.load_state_dict(state["frozen_actor"])
        self.operator_alpha = float(state["operator_alpha"])
        self.inner_steps = int(state["inner_steps"])
        self.outer_iterations = int(state["outer_iterations"])
        self.actor_updates = int(state["actor_updates"])
        self.tuning_rows = int(state["tuning_rows"])
        self.tuning_steps = int(state["tuning_steps"])
        # In place: the adapter's solver_records is this list.
        self.solver_records[:] = copy.deepcopy(list(state["solver_records"]))


class ImplicitTorqueCTSACBaseline(TorqueCTSACBaseline):
    """ct_sac_torque's adapter (TorqueCTSACBaseline) around _ImplicitPHCTSAC.

    implicit: the resolved settings (common.ph_config.ct_sac_implicit_setting):
    resolvent_step, inner_steps, policy_draws, actor_steps, temperature_every_rows.
    actor_schedule: "per_row" or "outer". Every other argument is TorqueCTSACBaseline's
    (polyak_rate is accepted and unused: the anchors are hard copies).

    solver_records is the learner's list (one record per copy). counters() reports the
    actor steps taken, and under "outer" the temperature steps taken; the critic steps
    are one per training row, as CT-SAC's. The resume state adds the learner's
    implicit_state under "implicit_ct_sac".
    """

    _legacy_class = _ImplicitPHCTSAC
    algorithm: _ImplicitPHCTSAC

    def __init__(
        self,
        env: Any,
        training: Mapping[str, object],
        temperature: Mapping[str, object],
        *,
        implicit: Mapping[str, object],
        actor_schedule: str = "per_row",
        **kwargs: Any,
    ) -> None:
        self._implicit_kwargs = {
            "resolvent_step": implicit["resolvent_step"],
            "inner_steps": implicit["inner_steps"],
            "policy_draws": implicit["policy_draws"],
            "actor_steps": implicit["actor_steps"],
            "temperature_every_rows": implicit["temperature_every_rows"],
            "actor_schedule": actor_schedule,
        }
        super().__init__(env, training, temperature, **kwargs)
        self.solver_records = self.algorithm.solver_records

    def _legacy_kwargs(self) -> Dict[str, Any]:
        return dict(self._implicit_kwargs)

    def counters(self) -> Metrics:
        counters = dict(super().counters())
        algorithm = self.algorithm
        counters["actor_gradient_steps"] = algorithm.actor_updates
        if algorithm.actor_schedule == "outer":
            counters["temperature_gradient_steps"] = (
                algorithm.tuning_steps if self._learned_temperature else 0
            )
        return counters

    def resume_state(self) -> Mapping[str, object]:
        state = dict(super().resume_state())
        state["implicit_ct_sac"] = self.algorithm.implicit_state()
        return state

    def load_resume_state(self, state: Mapping[str, object]) -> None:
        super().load_resume_state(state)
        self.algorithm.load_implicit_state(state["implicit_ct_sac"])
        self.solver_records = self.algorithm.solver_records

    def warm_start(
        self, path: Path, *, require_interactions: Optional[int] = None
    ) -> Dict[str, object]:
        """TorqueCTSACBaseline.warm_start, then a fresh solve from the loaded networks."""
        loaded = super().warm_start(path, require_interactions=require_interactions)
        self.algorithm.reset_operator()
        return loaded

    def load(self, path: Path) -> None:
        """TorqueCTSACBaseline.load; the frozen policy and temperature follow the loaded
        ones, the anchors are the loaded target critics."""
        super().load(path)
        self.algorithm.reset_operator(copy_critics=False)

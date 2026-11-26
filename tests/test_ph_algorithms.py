import contextlib
import copy
import io
import math
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

import algorithms.explicit_ct_ph_sac as explicit_module
import algorithms.implicit_ct_ph_sac as implicit_module
from algorithms.explicit_ct_ph_sac import ExplicitCTPHSAC
from algorithms.implicit_ct_ph_sac import ImplicitCTPHSAC
from algorithms.ph_targets import td_continuation
from algorithms.ph_training import validate_training_lineage
from common.ph_config import (
    IMPLICIT_DEFAULTS,
    INPUT_MATRIX_START_SEED_OFFSET,
    acrobot_xk_config,
    actor_log_std_init_setting,
    input_matrix_start_setting,
    network_start_settings,
    validate_configuration,
    warmup_decisions_setting,
    windy_pendulum_config,
)
from common.ph_time import discount_weight
from environment.ph_sac.tasks import make_env, make_oracle_plant, reward_rate_tensor
from evaluations.ph_sac.diagnostics import (
    ImplicitTrainingCounters,
    SolverMetrics,
    TrainingCounters,
)
from models.ph_sac.networks import LOG_STD_INIT
from models.ph_sac.operators import value_derivatives
from models.ph_sac.reward import RewardRate, reward_fit_rows, reward_rate_loss
from tests.test_ph_replay import assert_same_partitions


def windy():
    return windy_pendulum_config(
        reward_parameters={
            "angle_weight": 10.0,
            "velocity_weight": 1.0,
            "velocity_scale": 2.0 * math.pi,
            "effort_weight": 0.01,
        },
        reset_parameters={"angle_radius": 0.3, "velocity_radius": 0.5},
        tail_probability=0.8,
        tail_split=0.5,
        clock_kind="two_tail_uniform",
    )


def acrobot():
    return acrobot_xk_config(
        failure_limits={
            "elbow_angle": 4.0 * math.pi,
            "elbow_rate": 4.0 * math.pi,
            "shoulder_rate": 24.0,
        },
        reward_parameters={
            "energy_scale": 49.0,
            "upright_energy": 24.5,
            "lyapunov_k_p": 61.2,
            "reward_floor": 1e-6,
            "failure_rate": -1.0,
        },
        reset_parameters={"release_angle_low": 0.05, "release_angle_high": 0.5},
        plant_parameters={
            "m1": 1.0,
            "m2": 1.0,
            "l1": 1.0,
            "l2": 2.0,
            "lc1": 0.5,
            "lc2": 1.0,
            "I1": 0.083,
            "I2": 0.33,
            "gravity": 9.8,
        },
    )


def shared(testbed, *, seed=3, velocity_dim=1):
    return {
        "testbed": testbed,
        "network": {
            "hidden_sizes": (16, 16),
            "plant_hidden_sizes": (8, 8),
            "activation": "tanh",
            "mass_floor": 0.01,
            "diffusion_floor": 0.001,
            "damping_rank": velocity_dim,
            "velocity_feature_scale": 2.0 * math.pi,
        },
        "training": {
            "seed": seed,
            "device": "cpu",
            "replay_capacity": 5000,
            "batch_size": 8,
            "learning_rate": 3e-4,
            "gradient_clip": 10.0,
            "warmup_episodes": 0,
            "validation_fraction": 0.1,
        },
        "temperature": {
            "initial_alpha_c": 0.01,
            "learn_temperature": False,
            "target_entropy": -3.0,
        },
        "plant_source": "learned",
        "model_step": testbed["physics_dt"],
    }


def explicit_config(testbed=None, *, seed=3, continuation="generator", **overrides):
    testbed = testbed or windy()
    config = shared(
        testbed, seed=seed, velocity_dim=testbed["state_spec"]["velocity_dim"]
    )
    config.update(
        continuation=continuation, polyak_rate=0.25, rollout_paths=3, **overrides
    )
    validate_configuration(config)
    return config


def implicit_config(testbed=None, *, seed=3, **overrides):
    testbed = testbed or windy()
    config = shared(
        testbed, seed=seed, velocity_dim=testbed["state_spec"]["velocity_dim"]
    )
    config.update(
        solver={
            "resolvent_step": 1.0,
            "policy_draws": 3,
            "inner_steps": 6,
            "residual_estimator": "all_pairs",
        },
        rate={
            "replay_weight_max": 1.0,
            "noise_variance_floor": 1.0,
            "rollout_weight": 1.0,
            "model_weight": 1.0,
            "rollout_paths": 2,
            "model_points": 8,
            "rate_steps": 2,
            "actor_steps": 2,
        },
        admission={
            "efolds": 0.1,
            "margin_z": 1.645,
            "initial_rows": 8,
            "initial_paths": 2,
            "pilot_batches": 2,
            "real_row_budget": 64,
            "simulated_path_budget": 4096,
            "allocation": "pilot",
            "inference": "empirical",
            "plant_update": "admission",
            "polyak_rate": None,
        },
        # The repository defaults except the warm fits, which tests take explicitly.
        reward_source="learned",
        reward_warm_steps=0,
        plant_warm_steps=0,
        plant_fit_covariance="leading_order",
        temperature_every_rows=40,
    )
    config.update(overrides)
    validate_configuration(config)
    return config


def make_implicit(config=None, *, oracle=None, seed=11):
    config = config or implicit_config()
    env = make_env(config["testbed"], seed=seed)
    return ImplicitCTPHSAC(env, config, oracle=oracle)


def make_explicit(config=None, *, oracle=None, seed=11):
    config = config or explicit_config()
    env = make_env(config["testbed"], seed=seed)
    return ExplicitCTPHSAC(env, config, oracle=oracle)


def snapshot(module):
    return [p.detach().clone() for p in module.parameters()]


def same(module, frozen):
    return all(torch.equal(p, q) for p, q in zip(module.parameters(), frozen))


def collect(agent, count):
    """Fill replay directly so a test can drive individual loop stages."""
    for _ in range(count):
        agent.collect_transition()


def all_parameters(agent):
    return [
        p.detach().clone()
        for module in agent._network_modules().values()
        if module is not None
        for p in module.parameters()
    ]


def watch_decisions(agent):
    """Record (kind, interactions before it, episode id, gradient steps so far by counter)
    for every decision the agent draws: "uniform" (the warm-up's) or "policy"."""
    drawn = []
    uniform = agent._uniform_decision
    behavior = agent._behavior_decision

    def record(kind):
        steps = {
            key: int(value)
            for key, value in agent._counters.items()
            if key.endswith("_gradient_steps")
        }
        drawn.append((kind, agent.interactions, agent._episode_id, steps))

    def uniform_decision():
        record("uniform")
        return uniform()

    def behavior_decision(state):
        record("policy")
        return behavior(state)

    agent._uniform_decision = uniform_decision
    agent._behavior_decision = behavior_decision
    return drawn


def spy_transitions(agent):
    """Every transition the agent's environment returns, in order, unchanged."""
    transitions = []
    step_dt = agent.env.step_dt

    def wrapped(decision, duration):
        transition = step_dt(decision, duration)
        transitions.append(transition)
        return transition

    agent.env.step_dt = wrapped
    return transitions


def ended(transition):
    return bool(transition["terminated"] or transition["truncated"])


class TestConstruction(unittest.TestCase):
    def test_predict_works_before_learn_and_initialize_is_idempotent(self):
        for agent in (make_implicit(), make_explicit()):
            state = agent.env.reset(seed=0)[0]
            decision = agent.predict(state)
            self.assertEqual(decision.shape, (3,))
            self.assertTrue(agent.env.action_space.contains(decision))
            actor = agent.actor
            weights = snapshot(actor)
            agent.initialize()
            self.assertIs(agent.actor, actor)
            self.assertTrue(same(agent.actor, weights))

    def test_mean_and_stochastic_predictions(self):
        agent = make_implicit()
        state = agent.env.reset(seed=0)[0]
        np.testing.assert_array_equal(agent.predict(state), agent.predict(state))
        self.assertFalse(
            np.array_equal(agent.predict(state, False), agent.predict(state, False))
        )

    def test_learned_plant_starts_at_true_input_matrix(self):
        agent = make_implicit()
        np.testing.assert_allclose(
            agent.online_plant.input_matrix().detach().numpy(),
            agent.env.oracle_input_matrix(),
        )
        # Admitted and online plants begin as the same plant, as separate parameters.
        self.assertTrue(same(agent.admitted_plant, snapshot(agent.online_plant)))
        self.assertFalse(any(p.requires_grad for p in agent.admitted_plant.parameters()))

    def test_noiseless_plant_uses_the_conditioning_floor(self):
        agent = make_implicit(implicit_config(acrobot()))
        covariance = agent.online_plant.covariance(torch.zeros(1, 4)).detach()
        self.assertGreater(float(covariance[0, 2, 2]), 0.1**2 * 0.99)

    def test_value_obeys_terminal_boundary(self):
        agent = make_implicit()
        payoff = -1.0 / agent.discount_rate
        limit = agent.testbed["failure_limits"]["angle"]
        boundary = torch.tensor([[limit, 0.0], [-limit, 1.0]])
        for value in (agent.value, agent.target_value):
            np.testing.assert_allclose(
                value(boundary).detach().numpy(), payoff, atol=1e-5
            )


class TestImplicitInnerLoop(unittest.TestCase):
    def setUp(self):
        self.agent = make_implicit()
        collect(self.agent, 40)
        self.assertTrue(self.agent._train_ready())

    def test_only_value_and_online_plant_change(self):
        agent = self.agent
        frozen = {
            "actor": snapshot(agent.actor),
            "target": snapshot(agent.target_value),
            "admitted": snapshot(agent.admitted_plant),
            "rate": snapshot(agent.rate),
        }
        value = snapshot(agent.value)
        plant = snapshot(agent.online_plant)
        operator = agent.freeze_operator()
        result = agent.solve_value(operator, remaining_interactions=6)
        self.assertTrue(result["completed"])
        self.assertTrue(same(agent.actor, frozen["actor"]))
        self.assertTrue(same(agent.target_value, frozen["target"]))
        self.assertTrue(same(agent.admitted_plant, frozen["admitted"]))
        self.assertTrue(same(agent.rate, frozen["rate"]))
        self.assertFalse(same(agent.value, value))
        self.assertFalse(same(agent.online_plant, plant))

    def test_value_step_gives_gradients_only_to_the_value(self):
        agent = self.agent
        operator = agent.freeze_operator()
        for module in (
            agent.online_plant,
            agent.admitted_plant,
            agent.target_value,
            agent.actor,
            agent.rate,
            operator["policy"],
            operator["plant"],
            operator["target_value"],
        ):
            module.zero_grad(set_to_none=True)
        agent.update_value(agent.replay.sample("train", 8), operator)
        self.assertTrue(any(p.grad is not None for p in agent.value.parameters()))
        for module in (
            agent.online_plant,
            agent.admitted_plant,
            agent.target_value,
            agent.actor,
            agent.rate,
            operator["policy"],
            operator["plant"],
            operator["target_value"],
        ):
            self.assertTrue(all(p.grad is None for p in module.parameters()))

    def test_operator_holds_frozen_components(self):
        agents = [
            self.agent,
            make_implicit(
                implicit_config(
                    admission={
                        **self.agent.admission,
                        "plant_update": "polyak",
                        "polyak_rate": 0.5,
                    }
                )
            ),
            make_implicit(implicit_config(plant_source="oracle")),
        ]
        for agent in agents:
            operator = agent.freeze_operator()
            for name in ("plant", "target_value", "policy"):
                self.assertFalse(
                    any(p.requires_grad for p in operator[name].parameters()), name
                )
            self.assertIsNot(agent._reward_frozen, agent.reward_net)
            self.assertFalse(any(p.requires_grad for p in agent._reward_frozen.parameters()))

    def test_copy_equals_value_after_the_inner_loop(self):
        agent = self.agent
        before = snapshot(agent.target_value)
        agent.run_outer_iteration(remaining_interactions=50)
        self.assertFalse(same(agent.target_value, before))
        self.assertTrue(same(agent.target_value, snapshot(agent.value)))

    def test_value_head_never_reads_the_rate_head(self):
        agent = self.agent
        with torch.no_grad():
            for p in agent.rate.parameters():
                p.mul_(0.0).add_(float("nan"))
        operator = agent.freeze_operator()
        metrics = agent.update_value(agent.replay.sample("train", 8), operator)
        self.assertTrue(math.isfinite(metrics["value_loss"]))
        self.assertTrue(all(torch.isfinite(p).all() for p in agent.value.parameters()))

    def test_budget_stop_copies_and_fits_nothing_else(self):
        agent = self.agent
        rate = snapshot(agent.rate)
        actor = snapshot(agent.actor)
        result = agent.run_outer_iteration(remaining_interactions=1)
        self.assertFalse(result["completed"])
        self.assertTrue(same(agent.target_value, snapshot(agent.value)))
        self.assertTrue(same(agent.rate, rate))
        self.assertTrue(same(agent.actor, actor))
        self.assertEqual(agent.budget_stopped_outer_iterations, 1)
        self.assertEqual(agent.outer_in_cycle, 0)
        self.assertEqual(len(agent.solver_records), 1)
        self.assertEqual(agent.solver_records[0]["status"], "budget_exit")


class TestRateAndActor(unittest.TestCase):
    def setUp(self):
        self.agent = make_implicit()
        collect(self.agent, 40)

    def _others(self, *excluded):
        return {
            name: snapshot(module)
            for name, module in self.agent._network_modules().items()
            if name not in excluded and module is not None
        }

    def _unchanged(self, others):
        for name, weights in others.items():
            self.assertTrue(same(self.agent._network_modules()[name], weights), name)

    def test_rate_steps_change_only_omega(self):
        others = self._others("rate")
        before = snapshot(self.agent.rate)
        steps = self.agent.counters()["rate_gradient_steps"]
        self.agent.fit_rate_head(2)
        self._unchanged(others)
        self.assertFalse(same(self.agent.rate, before))
        self.assertEqual(self.agent.counters()["rate_gradient_steps"], steps + 2)

    def test_actor_steps_change_only_chi_and_leave_rate_gradients(self):
        agent = self.agent
        others = self._others("actor")
        before = snapshot(agent.actor)
        agent.rate.zero_grad(set_to_none=True)
        agent.update_actor(2)
        self._unchanged(others)
        self.assertFalse(same(agent.actor, before))
        self.assertTrue(all(p.grad is None for p in agent.rate.parameters()))
        self.assertTrue(all(p.requires_grad for p in agent.rate.parameters()))

    def test_rollout_weight_zero_runs_no_rate_rollouts(self):
        config = implicit_config()
        config["rate"]["rollout_weight"] = 0.0
        agent = make_implicit(config)
        collect(agent, 40)
        agent.fit_rate_head(2)
        self.assertEqual(agent.counters()["model_rollout_paths"], 0)
        self.assertEqual(agent.counters()["rate_gradient_steps"], 2)
        self.agent.fit_rate_head(2)
        self.assertEqual(self.agent.counters()["model_rollout_paths"], 2 * 8 * 2)

    def test_temperature_tuning_owes_one_step_per_r_rows(self):
        config = implicit_config(temperature_every_rows=40)
        config["temperature"]["learn_temperature"] = True
        agent = make_implicit(config)
        collect(agent, 8)
        ready_at = agent.tuning_rows
        collect(agent, 120)
        self.assertEqual(agent.tuning_rows - ready_at, 120)
        rows = agent.tuning_rows
        before = float(agent.log_temperature.detach())
        seen = []
        real = agent.update_temperature

        def recording(batch):
            seen.append(batch["states"])
            return real(batch)

        agent.update_temperature = recording
        agent.run_outer_iteration(remaining_interactions=50)
        # K=6 more rows are collected during the solve, then the burst runs.
        owed = (rows + 6) // 40
        self.assertEqual(agent.counters()["temperature_gradient_steps"], owed)
        self.assertEqual(len(seen), owed)
        self.assertFalse(torch.equal(seen[0], seen[1]))
        self.assertNotEqual(float(agent.log_temperature.detach()), before)
        self.assertLess(agent.tuning_due, 1.0)
        self.assertAlmostEqual(agent.tuning_due, (rows + 6) / 40 - owed)

    def test_tuning_counts_rows_only_once_training_is_ready(self):
        config = implicit_config(temperature_every_rows=4)
        config["temperature"]["learn_temperature"] = True
        agent = make_implicit(config)
        for _ in range(30):
            ready = agent._train_ready()
            rows = agent.tuning_rows
            agent.collect_transition()
            self.assertEqual(agent.tuning_rows - rows, int(agent._train_ready()))
            if not ready and not agent._train_ready():
                self.assertEqual(agent.tuning_rows, 0)
        state = agent._loop_state()
        restored = make_implicit(config)
        restored._load_loop_state(state)
        self.assertEqual(restored.tuning_rows, agent.tuning_rows)
        self.assertEqual(restored.tuning_due, agent.tuning_due)

    def test_tuning_does_not_count_warmup_rows(self):
        config = implicit_config(temperature_every_rows=4)
        config["temperature"]["learn_temperature"] = True
        agent = make_implicit(config)
        self.assertFalse(agent._collecting_warmup)
        flags = []
        real = agent._uniform_decision

        def recording():
            flags.append(agent._collecting_warmup)
            return real()

        agent._uniform_decision = recording
        # No learn(): no budget, so the warm-up runs its one episode (a 10 s horizon).
        agent.warmup(1)
        self.assertEqual(agent._warmup_episodes_done, 1)
        self.assertTrue(flags and all(flags))
        # Training became ready inside the warm-up (one episode fills a batch, as in
        # test_warm_fit_after_warm_up_episodes), so counting there would have counted rows.
        self.assertTrue(agent._train_ready())
        self.assertFalse(agent._collecting_warmup)
        self.assertEqual(agent.tuning_rows, 0)
        collect(agent, 5)
        self.assertEqual(agent.tuning_rows, 5)
        # A warm-up already done returns at once and leaves the flag down.
        interactions = agent.interactions
        agent.warmup(1)
        self.assertEqual(agent.interactions, interactions)
        self.assertFalse(agent._collecting_warmup)
        self.assertEqual(agent.tuning_rows, 5)

    def test_fixed_temperature_takes_no_tuning_steps(self):
        agent = make_implicit()
        collect(agent, 100)
        self.assertEqual(agent.tuning_rows, 0)
        agent.run_outer_iteration(remaining_interactions=50)
        self.assertEqual(agent.counters()["temperature_gradient_steps"], 0)


class TestAdmission(unittest.TestCase):
    def _agent(self, rows=60, **overrides):
        agent = make_implicit(implicit_config(**overrides) if overrides else None)
        collect(agent, rows)
        self.assertGreaterEqual(agent.replay.partition_size("gate"), 2)
        return agent

    def test_rejection_leaves_admitted_and_acceptance_copies(self):
        agent = self._agent()
        with torch.no_grad():
            for p in agent.online_plant.parameters():
                p.add_(0.01)
        admitted = snapshot(agent.admitted_plant)
        online = snapshot(agent.online_plant)
        with mock.patch.object(
            implicit_module, "gate_branch", return_value="evidence_against"
        ):
            agent.update_admitted_plant(agent.test_admission())
        self.assertTrue(same(agent.admitted_plant, admitted))
        self.assertFalse(agent.admission_records[-1]["accepted"])
        self.assertEqual(agent.admission_records[-1]["status"], "evidence_against")
        self.assertTrue(math.isnan(agent.admission_records[-1]["transfer_diagnostic"]))
        self.assertEqual(agent.cycle_length, agent.base_spacing)

        with mock.patch.object(
            implicit_module, "gate_branch", return_value="supported_improvement"
        ):
            result = agent.test_admission()
            agent.update_admitted_plant(result)
        self.assertTrue(same(agent.admitted_plant, online))
        self.assertTrue(agent.admission_records[-1]["accepted"])
        self.assertEqual(agent.cycle_length, agent.base_spacing)
        self.assertEqual(
            agent.admission_records[-1]["next_outer_iterations"], agent.base_spacing
        )
        self.assertEqual(agent.admission_index, 2)

    def test_spacing_after_an_admission_ignores_j(self):
        # M=ceil(n_e/log(1+lambda tau)) whatever J; efolds 0.1 gives M=2 at lambda tau 0.1.
        agent = self._agent()
        expected = math.ceil(
            agent.admission["efolds"]
            / math.log1p(agent.discount_rate * agent.resolvent_step)
        )
        self.assertEqual(expected, 2)
        self.assertEqual(agent.base_spacing, expected)
        for transfer in (0.0, 1e-3, 1.0, 1e6):
            with mock.patch.object(
                implicit_module, "gate_branch", return_value="supported_improvement"
            ), mock.patch.object(
                implicit_module, "transfer_diagnostic", return_value=transfer
            ):
                result = agent.test_admission()
                agent.update_admitted_plant(result)
            self.assertTrue(result["accepted"])
            self.assertEqual(result["transfer_diagnostic"], transfer)
            self.assertEqual(agent.cycle_length, expected)
            self.assertEqual(agent.admission_records[-1]["next_outer_iterations"], expected)
            # J still reaches the next solver record as the sampled motion input.
            self.assertEqual(agent._pending_motion, transfer)

    def test_fit_terms_read_the_plant_that_was_tested(self):
        agent = self._agent()
        with torch.no_grad():
            for p in agent.online_plant.noise_net.parameters():
                p.add_(0.5)
        real = implicit_module.admitted_fit_terms
        terms = []

        def recording(*args, **kwargs):
            terms.append(real(*args, **kwargs))
            return terms[-1]

        with mock.patch.object(
            implicit_module, "gate_branch", return_value="supported_improvement"
        ), mock.patch.object(implicit_module, "admitted_fit_terms", recording):
            result = agent.test_admission()
        agent.update_admitted_plant(result)
        record = agent.admission_records[-1]
        self.assertAlmostEqual(record["admitted_mean_square"], terms[0][0].numpy().mean())
        self.assertAlmostEqual(
            record["admitted_monte_carlo_variance"], terms[0][1].numpy().mean()
        )

    def test_gate_reports_return_components(self):
        # The windy rate is a cost, so the known reward makes every running reward <= 0.
        agent = self._agent(reward_source="known")
        result = agent.test_admission()
        agent.update_admitted_plant(result)
        record = agent.admission_records[-1]
        for name in (
            "candidate_failure_rate",
            "admitted_failure_rate",
            "observed_failure_rate",
        ):
            self.assertTrue(0.0 <= record[name] <= 1.0, name)
        self.assertLessEqual(record["admitted_running_reward"], 0.0)
        self.assertLessEqual(record["candidate_failure_payoff"], 0.0)
        self.assertGreater(record["mean_scheduled_horizon"], 0.0)

    def test_real_gate_prefers_a_correct_plant_over_a_broken_admitted_copy(self):
        """Start the online plant as the oracle-like fit and break the admitted copy."""
        config = implicit_config()
        config["admission"].update(initial_rows=40, real_row_budget=80)
        config["training"]["validation_fraction"] = 0.5
        agent = make_implicit(config)
        collect(agent, 300)
        self.assertGreaterEqual(agent.replay.partition_size("gate"), 80)

        # A value with a strong velocity slope makes the endpoint error visible.
        class Slope(torch.nn.Module):
            def forward(self, states):
                linear = states[..., :1] + 5.0 * states[..., 1:]
                return linear + 1e-3 * states.square().sum(dim=-1, keepdim=True)

        agent.target_value = Slope()

        # The admitted copy pushes hard in the wrong direction; the candidate does not.
        oracle = make_oracle_plant(agent.env)

        def broken_drift(states, actions):
            return oracle.drift(states, actions) + torch.tensor([0.0, 80.0])

        padding = agent.online_plant.noise_dim - oracle.noise_dim

        def wide(drift):
            return type(oracle)(
                drift,
                lambda states: torch.nn.functional.pad(
                    oracle.diffusion(states), (0, padding)
                ),
                oracle.input_matrix(),
                state_spec=oracle.state_spec,
            )

        agent.online_plant = wide(oracle.drift)
        agent.admitted_plant = wide(broken_drift)
        result = agent.test_admission()
        self.assertTrue(result["accepted"], result)
        self.assertLess(result["corrected_mean"], 0.0)

        # Swapping the roles must fail the same test.
        agent.online_plant, agent.admitted_plant = (
            agent.admitted_plant,
            agent.online_plant,
        )
        self.assertFalse(agent.test_admission()["accepted"])

    def test_no_rows_and_no_budget_ends_inconclusive_without_a_score(self):
        config = implicit_config()
        config["training"]["validation_fraction"] = 0.0
        agent = make_implicit(config)
        collect(agent, 20)
        agent._budget = agent.interactions
        admitted = snapshot(agent.admitted_plant)
        with torch.no_grad():
            for p in agent.online_plant.parameters():
                p.add_(0.01)
        result = agent.test_admission()
        self.assertFalse(result["accepted"])
        self.assertEqual(result["status"], "inconclusive_at_validation_cap")
        self.assertEqual(result["validation_rows"], 0)
        self.assertEqual(result["simulated_paths_used"], 0)
        agent.update_admitted_plant(result)
        self.assertTrue(same(agent.admitted_plant, admitted))
        record = agent.admission_records[-1]
        self.assertEqual(record["validation_rows"], 0)
        self.assertFalse(record["accepted"])
        self.assertTrue(math.isnan(record["corrected_mean"]))

    def test_rows_are_acquired_held_out_and_consume_interactions(self):
        agent = make_implicit()
        collect(agent, 20)
        unused = agent.replay.unused_rows("gate")
        self.assertLess(unused, 16)
        train = agent.replay.partition_size("train")
        before = agent.interactions
        with mock.patch.object(
            implicit_module, "gate_branch", return_value="evidence_against"
        ):
            result = agent.test_admission()
        # An 8-row pilot and an 8-row production batch, with nothing inspected beforehand.
        self.assertEqual(result["real_rows_acquired"], 16 - unused)
        self.assertEqual(result["real_rows_used"], 16)
        self.assertEqual(agent.interactions, before + 16 - unused)
        self.assertEqual(agent.counters()["gate_real_rows_acquired"], 16 - unused)
        self.assertEqual(agent.replay.partition_size("train"), train)
        self.assertEqual(agent.replay.unused_rows("gate"), 0)

    def test_a_stage_the_interaction_budget_cannot_cover_never_starts(self):
        agent = make_implicit()
        collect(agent, 20)
        needed = 16 - agent.replay.unused_rows("gate")
        agent._budget = agent.interactions + needed - 1
        before = agent.interactions
        result = agent.test_admission()
        self.assertEqual(result["status"], "inconclusive_at_validation_cap")
        self.assertEqual(result["stages"], 0)
        self.assertEqual(agent.interactions, before)

    def test_a_stage_the_partition_cannot_hold_never_starts(self):
        config = implicit_config()
        config["training"]["validation_fraction"] = 0.0
        agent = make_implicit(config)
        collect(agent, 20)
        before = agent.interactions
        result = agent.test_admission()
        self.assertEqual(result["status"], "inconclusive_at_validation_cap")
        self.assertEqual(agent.interactions, before)

    def _scored_batches(self, agent):
        """Run an attempt, recording every batch handed to the simulator."""
        real = agent._gate_samples
        seen = []

        def recording(batch, paths):
            seen.append((batch["rewards"].reshape(-1).tolist(), paths))
            return real(batch, paths)

        agent._gate_samples = recording
        return seen

    def test_inconclusive_stages_grow_rows_within_the_real_budget(self):
        agent = self._agent(rows=400)
        seen = self._scored_batches(agent)
        with mock.patch.object(implicit_module, "gate_branch", return_value="inconclusive"), mock.patch.object(
            implicit_module, "pilot_variance_components", return_value=(0.0, 1.0)
        ):
            result = agent.test_admission()
        # Budget 64: pilot 8 once, then production 8, 16 and 32.
        self.assertEqual(result["status"], "inconclusive_at_validation_cap")
        self.assertFalse(result["accepted"])
        self.assertEqual(result["real_rows_used"], 64)
        self.assertEqual([stage["rows"] for stage in result["stage_log"]], [8, 16, 32])
        self.assertEqual(result["stages"], 3)
        self.assertEqual(result["validation_rows"], 32)
        # Each stage repeats the pilot on the same rows, then scores a fresh batch.
        pilots = [rows for rows, _ in seen[0::3]] + [rows for rows, _ in seen[1::3]]
        self.assertEqual(len({tuple(rows) for rows in pilots}), 1)
        scoring = [set(rows) for rows, _ in seen[2::3]]
        self.assertEqual([len(rows) for rows in scoring], [8, 16, 32])
        everything = set(pilots[0])
        for rows in scoring:
            self.assertFalse(rows & everything)
            everything |= rows

    def test_inconclusive_stages_grow_paths_within_the_simulation_budget(self):
        agent = self._agent(rows=400)
        with mock.patch.object(implicit_module, "gate_branch", return_value="inconclusive"), mock.patch.object(
            implicit_module, "pilot_variance_components", return_value=(1.0, 0.0)
        ):
            before = agent.counters()["gate_simulated_paths"]
            result = agent.test_admission()
        paths = [stage["paths"] for stage in result["stage_log"]]
        self.assertEqual(paths[:3], [2, 4, 8])
        self.assertLessEqual(result["simulated_paths_used"], 4096)
        self.assertEqual(
            agent.counters()["gate_simulated_paths"] - before, result["simulated_paths_used"]
        )
        self.assertEqual(
            result["simulated_paths_used"],
            sum(2 * s["paths"] * (2 * s["pilot_rows"] + s["rows"]) for s in result["stage_log"]),
        )

    def test_scored_rows_stay_used_across_attempts(self):
        agent = self._agent(rows=400)
        seen = self._scored_batches(agent)
        with mock.patch.object(implicit_module, "gate_branch", return_value="evidence_against"):
            agent.test_admission()
            agent.test_admission()
        first, second = seen[2][0], seen[5][0]
        self.assertFalse(set(first) & set(second))

    def test_transfer_is_computed_only_for_supported_improvement(self):
        agent = self._agent()
        with mock.patch.object(implicit_module, "transfer_diagnostic") as transfer, mock.patch.object(
            implicit_module, "gate_branch", return_value="evidence_against"
        ):
            result = agent.test_admission()
        transfer.assert_not_called()
        self.assertTrue(math.isnan(result["transfer_diagnostic"]))

    def test_non_finite_transfer_rejects(self):
        agent = self._agent()
        with mock.patch.object(
            implicit_module, "transfer_diagnostic", return_value=torch.tensor(float("nan"))
        ), mock.patch.object(implicit_module, "gate_branch", return_value="supported_improvement"):
            result = agent.test_admission()
        self.assertFalse(result["accepted"])
        self.assertEqual(result["status"], "rejected_nonfinite")

    def test_fixed_allocation_drops_pilots_and_only_doubles_rows(self):
        config = implicit_config()
        config["admission"]["allocation"] = "fixed"
        agent = make_implicit(config)
        collect(agent, 400)
        with mock.patch.object(implicit_module, "gate_branch", return_value="inconclusive"), mock.patch.object(
            implicit_module, "pilot_variance_components"
        ) as pilot:
            result = agent.test_admission()
        pilot.assert_not_called()
        self.assertEqual([stage["rows"] for stage in result["stage_log"]], [8, 16, 32])
        self.assertEqual({stage["paths"] for stage in result["stage_log"]}, {2})
        self.assertEqual(result["real_rows_used"], 56)
        self.assertEqual(result["simulated_paths_used"], 2 * 2 * 56)

    def test_confidence_budgeted_mode_uses_its_width_over_global_looks(self):
        config = implicit_config()
        config["admission"].update(
            inference="confidence_budgeted",
            gate_failure_budget=0.1,
            return_bound=1e6,
            min_horizon=1e-6,
        )
        agent = make_implicit(config)
        collect(agent, 400)
        first = agent.test_admission()
        self.assertEqual(first["inference_mode"], "confidence_budgeted")
        self.assertEqual(first["status"], "inconclusive_at_validation_cap")
        looks = agent.gate_look_index
        self.assertEqual(looks, first["stages"])
        expected = implicit_module.budgeted_gate_width(
            rows=first["stage_log"][0]["rows"],
            look=1,
            gate_failure_budget=0.1,
            return_bound=1e6,
            min_horizon=1e-6,
        )
        self.assertAlmostEqual(first["stage_log"][0]["width"], expected)
        agent.test_admission()
        self.assertGreater(agent.gate_look_index, looks)

    def test_a_row_breaking_a_declared_bound_raises(self):
        config = implicit_config()
        config["admission"].update(
            inference="confidence_budgeted",
            gate_failure_budget=0.1,
            return_bound=1e-9,
            min_horizon=1e-6,
        )
        agent = make_implicit(config)
        collect(agent, 200)
        with self.assertRaisesRegex(ValueError, "return_bound"):
            agent.test_admission()

    def test_interrupted_attempt_repeats_after_a_resume(self):
        agent = self._agent()
        agent.cycle_length = 1
        agent.outer_in_cycle = 1
        calls = []

        def interrupted():
            calls.append(agent.outer_in_cycle)
            raise RuntimeError("stopped mid attempt")

        agent.test_admission = interrupted
        with self.assertRaises(RuntimeError):
            agent.run_admission_cycle(remaining_interactions=10)
        self.assertEqual(agent.outer_in_cycle, 1)
        self.assertEqual(calls, [1])

    def test_gate_runs_after_the_scheduled_outer_iterations(self):
        agent = self._agent()
        agent.cycle_length = 2
        calls = []
        real = agent.test_admission

        def counted():
            calls.append(agent.outer_in_cycle)
            return real()

        agent.test_admission = counted
        outcome = agent.run_admission_cycle(remaining_interactions=500)
        self.assertEqual(outcome["gate_tested"], 1.0)
        self.assertEqual(len(calls), 1)
        self.assertEqual(agent.outer_in_cycle, 0)
        self.assertEqual(agent.outer_iteration, 2)
        self.assertEqual(len(agent.solver_records), 2)
        self.assertEqual(len(agent.admission_records), 1)

    def test_initial_spacing_matches_the_paper(self):
        config = implicit_config()
        config["admission"]["efolds"] = 3.0
        self.assertEqual(make_implicit(config).cycle_length, 32)


class TestArms(unittest.TestCase):
    def test_oracle_arm_keeps_its_plant_and_runs_no_plant_steps(self):
        config = implicit_config(plant_source="oracle")
        agent = make_implicit(config)
        self.assertIsNone(agent.online_plant)
        agent.learn(120)
        counters = agent.counters()
        self.assertEqual(counters["plant_gradient_steps"], 0)
        self.assertGreater(counters["value_gradient_steps"], 0)
        self.assertEqual(agent.admission_records, [])
        states = torch.tensor([[0.2, 0.3]])
        actions = torch.tensor([[1.0]])
        reference = make_oracle_plant(agent.env)
        self.assertTrue(
            torch.equal(
                agent.admitted_plant.drift(states, actions),
                reference.drift(states, actions),
            )
        )
        agent.record_plant_diagnostics(agent.replay.sample("train", 8))
        self.assertEqual(agent.oracle_records, [])
        self.assertEqual(agent.calibration_records, [])

    def test_polyak_ablation_moves_the_plant_with_no_gate(self):
        base = implicit_config()
        admission = {**base["admission"], "plant_update": "polyak", "polyak_rate": 0.25}
        agent = make_implicit(implicit_config(admission=admission))
        collect(agent, 40)
        admitted = snapshot(agent.admitted_plant)
        online = snapshot(agent.online_plant)
        agent.update_online_plant(agent.replay.sample("train", 8))
        for slow, old, fast in zip(
            agent.admitted_plant.parameters(), admitted, agent.online_plant.parameters()
        ):
            torch.testing.assert_close(slow, 0.75 * old + 0.25 * fast.detach())
        self.assertFalse(same(agent.online_plant, online))
        agent.test_admission = mock.Mock(side_effect=AssertionError("No gate here."))
        spacing = agent.cycle_length
        agent.learn(agent.interactions + 60)
        self.assertEqual(agent.admission_records, [])
        self.assertEqual(agent.cycle_length, spacing)
        self.assertFalse(same(agent.admitted_plant, admitted))

    def test_a_solve_takes_exactly_k_inner_steps(self):
        for steps in (1, 5, 9):
            config = implicit_config()
            config["solver"]["inner_steps"] = steps
            agent = make_implicit(config)
            collect(agent, 40)
            values = agent.counters()["value_gradient_steps"]
            interactions = agent.interactions
            result = agent.run_outer_iteration(remaining_interactions=100)
            self.assertEqual(agent.counters()["value_gradient_steps"], values + steps)
            self.assertEqual(agent.interactions, interactions + steps)
            self.assertEqual(result, {"inner_steps": steps, "completed": True})
            self.assertEqual(agent.inner_steps, 0)
            record = agent.solver_records[-1]
            self.assertEqual(record["status"], "completed")
            self.assertEqual(record["inner_iterations"], steps)

    def test_loop_counters_survive_a_checkpoint(self):
        agent = make_implicit()
        agent.gate_look_index = 4
        agent.inner_steps = 3
        agent.warm_fit_done = True
        agent.tuning_rows, agent.tuning_steps = 17, 2
        agent.reward_records.append({"steps": 5})
        state = agent._loop_state()
        restored = make_implicit(implicit_config())
        restored._load_loop_state(state)
        self.assertEqual(restored.gate_look_index, 4)
        self.assertEqual(restored.inner_steps, 3)
        self.assertTrue(restored.warm_fit_done)
        self.assertEqual((restored.tuning_rows, restored.tuning_steps), (17, 2))
        self.assertEqual(restored.reward_records, [{"steps": 5}])

    def test_a_checkpoint_without_the_later_loop_fields_still_loads(self):
        agent = make_implicit()
        state = dict(agent._loop_state())
        for name in agent._LATER_LOOP_FIELDS + ("reward_records", "reward_diagnostic_rng"):
            state.pop(name)
        restored = make_implicit()
        restored._load_loop_state(state)
        self.assertFalse(restored.warm_fit_done)
        self.assertEqual(restored.tuning_rows, 0)

    def test_averaged_draw_ablation_uses_its_own_loss(self):
        config = implicit_config()
        config["solver"]["residual_estimator"] = "averaged_draw"
        agent = make_implicit(config)
        collect(agent, 40)
        with mock.patch.object(
            implicit_module, "averaged_draw_loss", wraps=implicit_module.averaged_draw_loss
        ) as averaged, mock.patch.object(
            implicit_module, "all_pairs_loss", wraps=implicit_module.all_pairs_loss
        ) as pairs:
            agent.update_value(agent.replay.sample("train", 8), agent.freeze_operator())
        averaged.assert_called_once()
        pairs.assert_not_called()

    def test_all_pairs_is_the_default_loss_and_monitors_the_solver(self):
        agent = make_implicit()
        collect(agent, 40)
        with mock.patch.object(
            implicit_module, "averaged_draw_loss", wraps=implicit_module.averaged_draw_loss
        ) as averaged:
            agent.update_value(agent.replay.sample("train", 8), agent.freeze_operator())
        averaged.assert_not_called()


class TestBudgetAndCounters(unittest.TestCase):
    def _check_budget(self, agent, budget):
        seen = []
        agent.learn(budget, callback=lambda _, metrics: seen.append(metrics["interactions"]))
        self.assertEqual(agent.interactions, budget)
        self.assertEqual(seen, [float(i + 1) for i in range(budget)])

    def test_exact_budget_including_warm_up(self):
        implicit = implicit_config()
        implicit["training"]["warmup_episodes"] = 1
        self._check_budget(make_implicit(implicit), 150)
        explicit = explicit_config()
        explicit["training"]["warmup_episodes"] = 1
        self._check_budget(make_explicit(explicit), 150)

    def test_budget_inside_warm_up_resumes_it(self):
        config = implicit_config()
        config["training"]["warmup_episodes"] = 2
        agent = make_implicit(config)
        agent.learn(3)
        self.assertEqual(agent.interactions, 3)
        self.assertEqual(agent.counters()["value_gradient_steps"], 0)
        self.assertEqual(agent._warmup_episodes_done, 0)

    def test_budget_is_cumulative(self):
        agent = make_explicit()
        agent.learn(30)
        agent.learn(50)
        self.assertEqual(agent.interactions, 50)
        agent.learn(50)
        self.assertEqual(agent.interactions, 50)

    def test_counters_have_exactly_the_protocol_keys(self):
        known = make_implicit(implicit_config(reward_source="known"))
        for agent, schema in (
            (make_implicit(), ImplicitTrainingCounters),
            (known, ImplicitTrainingCounters),
            (make_explicit(), TrainingCounters),
        ):
            agent.learn(40)
            self.assertEqual(set(agent.counters()), set(schema.__annotations__))
            self.assertEqual(set(agent.counters()), set(schema.__required_keys__))
            self.assertGreater(agent.counters()["simulated_seconds"], 0.0)
        self.assertEqual(known.counters()["reward_gradient_steps"], 0)

    def test_simulated_seconds_sum_actual_durations(self):
        agent = make_explicit()
        total = 0.0
        for _ in range(30):
            total += agent.collect_transition()["duration"]
        self.assertAlmostEqual(agent.counters()["simulated_seconds"], total)

    def test_wall_time_includes_time_before_learn(self):
        agent = make_explicit()
        first = agent.counters()["wall_seconds"]
        self.assertGreater(first, 0.0)
        self.assertGreater(agent.counters()["wall_seconds"], first)

    def test_explicit_counts_one_step_per_interaction_per_component(self):
        agent = make_explicit()
        agent.learn(60)
        counters = agent.counters()
        self.assertGreater(counters["critic_gradient_steps"], 0)
        for key in (
            "value_gradient_steps",
            "actor_gradient_steps",
            "plant_gradient_steps",
        ):
            self.assertEqual(counters[key], counters["critic_gradient_steps"])
        # Training begins as soon as a batch exists and repeats every interaction.
        self.assertEqual(
            counters["critic_gradient_steps"],
            60 - agent.training["batch_size"] + 1,
        )
        self.assertEqual(counters["model_rollout_paths"], 0)

    def test_collection_resets_after_episode_end(self):
        testbed = windy()
        testbed["episode_duration"] = 0.05
        agent = make_explicit(explicit_config(testbed))
        ends = 0
        for _ in range(40):
            transition = agent.collect_transition()
            if transition["terminated"] or transition["truncated"]:
                ends += 1
                self.assertIsNone(agent._state)
        self.assertGreater(ends, 0)
        ids = agent.replay.provenance("train")["episode_ids"]
        self.assertGreater(ids.max(), 0)


class TestExplicit(unittest.TestCase):
    def setUp(self):
        self.agent = make_explicit()
        collect(self.agent, 30)
        self.batch = self.agent.replay.sample("train", 8)

    def test_update_order(self):
        agent = self.agent
        order = []
        names = [
            "update_temperature",
            "update_plant",
            "update_value",
            "build_critic_target",
            "update_critics",
            "update_actor",
            "update_targets",
        ]

        def recording(name, real):
            def call(*args, **kwargs):
                order.append(name)
                return real(*args, **kwargs)

            return call

        for name in names:
            setattr(agent, name, recording(name, getattr(agent, name)))
        agent.train(1, 8)
        self.assertEqual(order, names)

    def test_polyak_arithmetic(self):
        agent = self.agent
        with torch.no_grad():
            for module in (agent.critics, agent.value, agent.online_plant):
                for p in module.parameters():
                    p.add_(0.5)
        targets = {
            "critics": snapshot(agent.target_critics),
            "value": snapshot(agent.target_value),
            "plant": snapshot(agent.target_plant),
        }
        agent.update_targets()
        pairs = (
            (agent.target_critics, agent.critics, targets["critics"]),
            (agent.target_value, agent.value, targets["value"]),
            (agent.target_plant, agent.online_plant, targets["plant"]),
        )
        for target, online, before in pairs:
            for slow, fast, old in zip(target.parameters(), online.parameters(), before):
                torch.testing.assert_close(slow, 0.75 * old + 0.25 * fast)

    def test_oracle_target_stays_fixed(self):
        agent = make_explicit(explicit_config(plant_source="oracle"))
        self.assertIsNone(agent.online_plant)
        agent.learn(30)
        self.assertEqual(agent.counters()["plant_gradient_steps"], 0)
        self.assertGreater(agent.counters()["critic_gradient_steps"], 0)

    def test_generator_target_matches_hand_formula(self):
        agent = self.agent
        batch = self.batch
        states = batch["states"]
        decisions = batch["decisions"]
        resolved = agent.controller.resolve_decision(states, decisions)
        actions = agent.controller.action(states, resolved)
        diffusion = agent.target_plant.diffusion(states)
        derivatives = value_derivatives(
            agent.target_value, states, diffusion, create_graph=False
        )
        drift = agent.target_plant.drift(states, actions)
        generated = (drift * derivatives["gradients"]).sum(-1, keepdim=True) + 0.5 * (
            derivatives["diffusion_traces"]
        )
        values = derivatives["values"]
        # One explicit Euler step over each row's own interval (the irregular clock).
        durations = batch["durations"]
        expected = batch["rewards"] + (1.0 - batch["terminated"]) * (
            values + durations * (generated - agent.discount_rate * values)
        )
        torch.testing.assert_close(
            agent.build_critic_target(batch), expected.detach(), atol=1e-5, rtol=1e-4
        )

    def test_truncation_does_not_mask_continuation(self):
        agent = self.agent
        base = {key: value[:2].clone() for key, value in self.batch.items()}
        terminated = {**base, "terminated": torch.tensor([[1.0], [0.0]])}
        truncated = {
            **base,
            "terminated": torch.zeros(2, 1),
            "truncated": torch.ones(2, 1),
        }
        running = {
            **base,
            "terminated": torch.zeros(2, 1),
            "truncated": torch.zeros(2, 1),
        }
        target_truncated = agent.build_critic_target(truncated)
        target_running = agent.build_critic_target(running)
        torch.testing.assert_close(target_truncated, target_running)
        target_terminated = agent.build_critic_target(terminated)
        torch.testing.assert_close(target_terminated[0], base["rewards"][0])
        torch.testing.assert_close(target_terminated[1], target_running[1])
        self.assertNotAlmostEqual(
            float(target_running[0]), float(base["rewards"][0]), places=3
        )

    def test_rollout_target_counts_paths_and_adds_reward_once(self):
        agent = make_explicit(explicit_config(continuation="rollout"))
        collect(agent, 30)
        batch = agent.replay.sample("train", 8)
        targets = agent.build_critic_target(batch)
        self.assertEqual(targets.shape, (8, 1))
        self.assertEqual(agent.counters()["model_rollout_paths"], 8 * 3)
        shifted = {**batch, "rewards": batch["rewards"] + 1.0}
        # Reseed the rollout stream so both targets read the same paths.
        state = copy.deepcopy(agent._rollout_rng.bit_generator.state)
        first = agent.build_critic_target(batch)
        agent._rollout_rng.bit_generator.state = state
        torch.testing.assert_close(
            agent.build_critic_target(shifted), first + 1.0, atol=1e-5, rtol=1e-5
        )

    def test_td_target_leaves_the_environment_clock_alone(self):
        agent = make_explicit(explicit_config(continuation="td"))
        collect(agent, 30)
        before = copy.deepcopy(agent.env.clock.state_dict())
        batch = agent.replay.sample("train", 8)
        self.assertEqual(agent.build_critic_target(batch).shape, (8, 1))
        self.assertEqual(agent.env.clock.state_dict(), before)
        agent.train(2, 8)
        self.assertEqual(agent.env.clock.state_dict(), before)

    def test_value_target_charges_interval_entropy(self):
        agent = self.agent
        batch = self.batch
        seen = {}

        def capture(value, states, targets):
            seen["targets"] = targets.clone()
            return value(states).square().mean()

        generator_state = agent._generator.get_state()
        with mock.patch.object(explicit_module, "soft_value_loss", capture):
            agent.update_value(batch)
        agent._generator.set_state(generator_state)
        with torch.no_grad():
            sample = agent._policy_sample(batch["states"])
            expected = agent.target_critics.minimum(
                batch["states"], sample["decisions"][:, 0]
            ) - agent.alpha_c * discount_weight(
                batch["durations"], discount_rate=agent.discount_rate
            ) * sample["log_probabilities"][:, 0]
        torch.testing.assert_close(seen["targets"], expected)

    def test_explicit_default_replay_holds_nothing_out(self):
        agent = make_explicit()
        collect(agent, 50)
        self.assertEqual(agent.replay.partition_size("train"), 50)
        self.assertEqual(agent.replay.partition_size("gate"), 0)

    def test_reward_rate_matches_task(self):
        states = self.batch["states"]
        torques = torch.ones(8, 1)
        torch.testing.assert_close(
            self.agent._reward_rate(states, torques),
            reward_rate_tensor(states, torques, self.agent.testbed),
        )


def unit_box_twin(actor):
    """The actor's networks on [-1,1]^D, where the physical and normalized densities agree."""
    twin = copy.deepcopy(actor)
    with torch.no_grad():
        twin.decision_low.fill_(-1.0)
        twin.decision_high.fill_(1.0)
    return twin


def box_log_half_widths(actor):
    return float(torch.log(0.5 * (actor.decision_high - actor.decision_low)).sum())


@contextlib.contextmanager
def recorded_draws(actor):
    """Record every actor.sample call; a draw without explicit noise gets recorded noise."""
    draws = []
    real = actor.sample

    def sample(states, *, sample_count=1, noise=None, deterministic=False):
        if noise is None and not deterministic:
            noise = torch.randn(states.shape[0], sample_count, actor.decision_dim)
        draws.append(
            {"states": states.detach().clone(), "noise": noise, "sample_count": sample_count}
        )
        return real(
            states, sample_count=sample_count, noise=noise, deterministic=deterministic
        )

    with mock.patch.object(actor, "sample", sample):
        yield draws


@contextlib.contextmanager
def captured(module, name, pick):
    """Wrap module.name and record pick(*args, **kwargs) of every call."""
    seen = []
    real = getattr(module, name)

    def wrapper(*args, **kwargs):
        seen.append(pick(*args, **kwargs))
        return real(*args, **kwargs)

    with mock.patch.object(module, name, wrapper):
        yield seen


def twin_draw(twin, draw):
    """The twin's draw with a recorded draw's noise: decisions in [-1,1]^D and log pi_u."""
    with torch.no_grad():
        return twin.sample(
            draw["states"], sample_count=draw["sample_count"], noise=draw["noise"]
        )


class _PeakedRate(torch.nn.Module):
    """q = -|u - centre|^2 on the normalized decision u: a rate peaked inside the box."""

    def __init__(self, low, high, centre):
        super().__init__()
        dtype = torch.get_default_dtype()
        self.register_buffer("low", torch.as_tensor(low, dtype=dtype))
        self.register_buffer("high", torch.as_tensor(high, dtype=dtype))
        self.register_buffer("centre", torch.as_tensor(centre, dtype=dtype))

    def forward(self, states, decisions):
        normalized = 2.0 * (decisions - self.low) / (self.high - self.low) - 1.0
        return -(normalized - self.centre).square().sum(dim=-1, keepdim=True)


class TestNormalizedEntropy(unittest.TestCase):
    """Every log pi the agents charge or tune on is the density over the normalized box.

    The reference is the same actor on [-1,1]^D (unit_box_twin) driven by the same noise.
    On the windy box sum(log h) = 5.5, so a density over the physical box would differ.
    """

    def _assert_twin_density(self, twin, draws, received):
        self.assertEqual(len(draws), len(received))
        for draw, log_densities in zip(draws, received):
            torch.testing.assert_close(log_densities, twin_draw(twin, draw)["log_probabilities"])

    def test_explicit_targets_actor_and_temperature_charge_the_normalized_density(self):
        config = explicit_config()
        config["temperature"]["learn_temperature"] = True
        agent = make_explicit(config)
        collect(agent, 30)
        batch = agent.replay.sample("train", 8)
        self.assertGreater(box_log_half_widths(agent.actor), 5.0)
        twin = unit_box_twin(agent.actor)
        weights = discount_weight(batch["durations"], discount_rate=agent.discount_rate)

        with recorded_draws(agent.actor) as draws, captured(
            explicit_module, "soft_value_loss", lambda value, states, targets: targets.clone()
        ) as targets:
            agent.update_value(batch)
        (draw,) = draws
        with torch.no_grad():
            decisions = agent.actor.sample(draw["states"], noise=draw["noise"])["decisions"]
            expected = agent.target_critics.minimum(
                batch["states"], decisions[:, 0]
            ) - agent.alpha_c * weights * twin_draw(twin, draw)["log_probabilities"][:, 0]
        torch.testing.assert_close(targets[0], expected)

        with recorded_draws(agent.actor) as draws, captured(
            explicit_module,
            "temperature_loss",
            lambda log_temperature, log_probabilities, *args, **kwargs: log_probabilities.clone(),
        ) as received:
            agent.update_temperature(batch)
        self._assert_twin_density(twin, draws, received)

        with recorded_draws(agent.actor) as draws, captured(
            explicit_module,
            "explicit_actor_loss",
            lambda critics, states, sample, temperature: sample["log_probabilities"].detach().clone(),
        ) as received:
            agent.update_actor(batch)
        self._assert_twin_density(twin, draws, received)

        # The optional TD continuation charges the next decision's density the same way.
        twin = unit_box_twin(agent.actor)
        next_durations = torch.full_like(batch["durations"], 0.01)
        with recorded_draws(agent.actor) as draws:
            continuation = td_continuation(
                agent.target_critics,
                agent.actor,
                batch["next_states"],
                batch["durations"],
                next_durations,
                alpha_c=agent.alpha_c,
                discount_rate=agent.discount_rate,
            )
        (draw,) = draws
        with torch.no_grad():
            decisions = agent.actor.sample(draw["states"], noise=draw["noise"])["decisions"]
            charge = agent.alpha_c * discount_weight(
                next_durations, discount_rate=agent.discount_rate
            )
            expected = torch.exp(-agent.discount_rate * batch["durations"]) * (
                agent.target_critics.minimum(batch["next_states"], decisions[:, 0])
                - charge * twin_draw(twin, draw)["log_probabilities"][:, 0]
            )
        torch.testing.assert_close(continuation, expected)

    def test_implicit_residual_actor_and_temperature_charge_the_normalized_density(self):
        config = implicit_config()
        config["temperature"]["learn_temperature"] = True
        agent = make_implicit(config)
        collect(agent, 40)
        batch = agent.replay.sample("train", 8)
        operator = agent.freeze_operator()
        self.assertGreater(box_log_half_widths(operator["policy"]), 5.0)

        with recorded_draws(operator["policy"]) as draws, captured(
            implicit_module,
            "resolvent_residuals",
            lambda value, operator, controller, states, samples, **kwargs: samples[
                "log_probabilities"
            ].clone(),
        ) as received:
            agent.update_value(batch, operator)
        self.assertEqual(draws[0]["sample_count"], config["solver"]["policy_draws"])
        self._assert_twin_density(unit_box_twin(operator["policy"]), draws, received)

        twin = unit_box_twin(agent.actor)
        with recorded_draws(agent.actor) as draws, captured(
            implicit_module,
            "implicit_actor_loss",
            lambda rate, states, sample, temperature: sample["log_probabilities"].detach().clone(),
        ) as received:
            agent.update_actor(1)
        self._assert_twin_density(twin, draws, received)

        twin = unit_box_twin(agent.actor)
        with recorded_draws(agent.actor) as draws, captured(
            implicit_module,
            "temperature_loss",
            lambda log_temperature, log_probabilities, *args, **kwargs: log_probabilities.clone(),
        ) as received:
            agent.update_temperature(batch)
        self._assert_twin_density(twin, draws, received)

    def test_tuned_temperature_reaches_minus_one_per_coordinate_in_any_units(self):
        """Tuning against a rate peaked inside the box brings the entropy to -3m.

        The windy box (sum log h = 5.5) and the same box with gains times 0.005 (sum log h
        = -5.1, reacher-like) end at the same entropy and temperature. A density over the
        physical box would have aimed the normalized entropy at -3 - sum(log h) instead:
        -8.5 and +2.1, the latter above the box's maximum 3 log 2.
        """
        temperatures = []
        for scale in (1.0, 0.005):
            testbed = windy()
            controller = testbed["controller"]
            factor = np.array([1.0, scale, scale])
            controller["decision_low"] = np.asarray(controller["decision_low"]) * factor
            controller["decision_high"] = np.asarray(controller["decision_high"]) * factor
            config = implicit_config(testbed)
            config["temperature"]["learn_temperature"] = True
            config["training"]["learning_rate"] = 0.02
            config["training"]["batch_size"] = 32
            agent = make_implicit(config)
            collect(agent, 40)
            agent.rate = _PeakedRate(
                controller["decision_low"], controller["decision_high"], [0.2, -0.3, 0.1]
            )
            for _ in range(500):
                agent.update_actor(1)
                agent.update_temperature(agent.replay.sample("train", 32))
            with torch.no_grad():
                states = agent.replay.sample("train", 256)["states"]
                log_densities = agent._policy_sample(states, sample_count=16)["log_probabilities"]
            target = config["temperature"]["target_entropy"]
            self.assertEqual(target, -3.0)
            self.assertAlmostEqual(-float(log_densities.mean()), target, delta=0.25)
            temperatures.append(float(agent.alpha_c))
        self.assertAlmostEqual(temperatures[0], temperatures[1], delta=0.05 * temperatures[0])


class TestReproducibility(unittest.TestCase):
    def test_identical_seeds_give_identical_parameters(self):
        for build in (make_implicit, make_explicit):
            first = build()
            first.learn(90)
            second = build()
            second.learn(90)
            for p, q in zip(all_parameters(first), all_parameters(second)):
                self.assertTrue(torch.equal(p, q))

    def test_different_seeds_differ(self):
        first = make_explicit(explicit_config(seed=1))
        second = make_explicit(explicit_config(seed=2))
        self.assertFalse(
            all(
                torch.equal(p, q)
                for p, q in zip(all_parameters(first), all_parameters(second))
            )
        )

    def _continue_identically(self, build, first_budget, final_budget, *, callback=None):
        reference = build()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "checkpoint.pt"
            reference.learn(first_budget)
            reference.save(path)
            restored = build()
            restored.load(path)
            self.assertEqual(restored.counters()["interactions"], first_budget)
            state = reference.env.observe()
            np.testing.assert_array_equal(restored.env.observe(), state)
            # Both continue from the checkpoint on the same global stream.
            global_state = torch.get_rng_state()
            restored.learn(final_budget)
            torch.set_rng_state(global_state)
            reference.learn(final_budget)
        for p, q in zip(all_parameters(reference), all_parameters(restored)):
            self.assertTrue(torch.equal(p, q))
        counts, other = reference.counters(), restored.counters()
        counts.pop("wall_seconds"), other.pop("wall_seconds")
        self.assertEqual(counts, other)
        return reference, restored

    def test_save_load_continuation_implicit(self):
        reference, restored = self._continue_identically(make_implicit, 70, 130)
        self.assertEqual(len(reference.solver_records), len(restored.solver_records))

    def test_save_load_continuation_explicit(self):
        self._continue_identically(make_explicit, 50, 80)

    def test_save_inside_callback_resumes_mid_inner_loop(self):
        """A checkpoint between inner iterations must not lose the inner count."""
        build = make_implicit
        saved = {}

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "checkpoint.pt"

            def save_once(agent, metrics):
                if agent.inner_steps == 3 and "inner" not in saved:
                    saved["inner"] = agent.inner_steps
                    saved["interactions"] = agent.interactions
                    agent.save(path)
                    saved["rng"] = torch.get_rng_state()

            reference = build()
            reference.learn(100, callback=save_once)
            self.assertEqual(saved["inner"], 3)

            restored = build()
            restored.load(path)
            self.assertEqual(restored.inner_steps, 3)
            restored.learn(100)
        for p, q in zip(all_parameters(reference), all_parameters(restored)):
            self.assertTrue(torch.equal(p, q))

    def test_observer_records_without_changing_results(self):
        plain = make_implicit()
        plain.learn(80)
        observed = make_implicit()
        traces = []
        observed.set_optimizer_observer(traces.append)
        observed.learn(80)
        for p, q in zip(all_parameters(plain), all_parameters(observed)):
            self.assertTrue(torch.equal(p, q))
        self.assertEqual(
            len(traces),
            sum(
                plain.counters()[key]
                for key in (
                    "value_gradient_steps",
                    "rate_gradient_steps",
                    "actor_gradient_steps",
                    "plant_gradient_steps",
                )
            ),
        )
        blocks = {trace["block"] for trace in traces}
        self.assertEqual(blocks, {"value", "rate", "actor", "model"})

    def test_observer_records_actual_increment_and_moments(self):
        agent = make_explicit()
        collect(agent, 30)
        traces = []
        agent.set_optimizer_observer(traces.append)
        before = torch.cat([p.detach().reshape(-1).clone() for p in agent.value.parameters()])
        agent.update_value(agent.replay.sample("train", 8))
        after = torch.cat([p.detach().reshape(-1) for p in agent.value.parameters()])
        trace = traces[0]
        self.assertEqual(trace["block"], "value")
        torch.testing.assert_close(trace["parameter_increment"], after - before)
        torch.testing.assert_close(trace["parameters_before"], before)
        self.assertIsNone(trace["empirical_gradient"])
        self.assertEqual(trace["evidence"]["kind"], "sampled_diagnostic")
        # First Adam step: the bias-corrected moment is the clipped gradient itself,
        # and d=-lr*P*m reproduces the actual increment.
        torch.testing.assert_close(trace["adam_first_moment"], trace["clipped_gradient"])
        reconstructed = (
            -trace["learning_rate"]
            * trace["adam_preconditioner"]
            * trace["adam_first_moment"]
        )
        torch.testing.assert_close(reconstructed, after - before, atol=1e-6, rtol=1e-3)

    def test_outer_step_trace_satisfies_lineage(self):
        agent = make_implicit()
        collect(agent, 40)
        agent.set_optimizer_observer(lambda trace: None)
        agent.run_outer_iteration(remaining_interactions=50)
        trace = agent.outer_step_trace()
        self.assertEqual(trace["input_context"]["candidate_id"], "V_0")
        self.assertEqual(trace["solve_output_context"]["candidate_id"], "V_1")
        self.assertEqual(trace["input_context"]["policy_id"], "pi_0")
        self.assertEqual(trace["solve_output_context"]["policy_id"], "pi_0")
        self.assertEqual(trace["actor_output_context"]["policy_id"], "pi_1")
        self.assertGreater(trace["value_increment_norm_sum"], 0.0)
        self.assertGreater(trace["actor_increment_norm_sum"], 0.0)
        evidence = {
            "kind": "assumed_uniform",
            "domain": "test",
            "justification": "test",
            "failure_probability": None,
        }
        solved = trace["solve_output_context"]
        improved = trace["actor_output_context"]
        accuracy = {
            "context": trace["input_context"],
            "residual_context": solved,
            "model_context": solved,
            "evidence": evidence,
            "raw_resolvent_residual": 0.0,
            "rate_error": 0.0,
            "actor_gap": 0.0,
            "model_error": 0.0,
            "rate_target_bias": 0.0,
            "admitted_model_lag": 0.0,
        }
        validate_training_lineage(trace, accuracy, purpose="original_value_iteration")
        post = {**accuracy, "context": improved, "model_context": improved}
        validate_training_lineage(trace, post, purpose="post_actor_performance")

    def test_outer_step_trace_requires_an_observer(self):
        with self.assertRaises(RuntimeError):
            make_implicit().outer_step_trace()


def assert_same_state(case, left, right, path="state"):
    """Nested dicts, sequences and tensors agree exactly."""
    if isinstance(left, torch.Tensor):
        case.assertTrue(torch.equal(left, right), path)
    elif isinstance(left, dict):
        case.assertEqual(set(left), set(right), path)
        for key in left:
            assert_same_state(case, left[key], right[key], f"{path}.{key}")
    elif isinstance(left, (list, tuple)):
        case.assertEqual(len(left), len(right), path)
        for index, (a, b) in enumerate(zip(left, right)):
            assert_same_state(case, a, b, f"{path}[{index}]")
    else:
        case.assertEqual(left, right, path)


def saved_bytes(state):
    buffer = io.BytesIO()
    torch.save(state, buffer)
    return buffer.getvalue()


def full_capacity_replay(replay):
    """The replay state as earlier versions saved it: every array at full capacity."""
    state = copy.deepcopy(replay.state_dict())
    for name, buffer in replay.buffers.items():
        state["buffers"][name]["arrays"] = {
            field: array.copy() for field, array in buffer._arrays.items()
        }
    state["episode_ids"] = {name: ids.copy() for name, ids in replay.episode_ids.items()}
    state["reuse_counts"] = {name: uses.copy() for name, uses in replay.reuse_counts.items()}
    return state


class TestCompactCheckpoints(unittest.TestCase):
    def test_checkpoint_keeps_only_filled_replay_rows(self):
        agent = make_implicit()
        collect(agent, 300)
        state = agent.state_dict()
        for name, buffer in agent.replay.buffers.items():
            size = buffer.size()
            self.assertGreater(size, 0)
            self.assertLess(size, buffer.capacity)
            stored = state["replay"]["buffers"][name]["arrays"]
            for field, array in stored.items():
                self.assertEqual(array.shape[0], size, field)
            self.assertEqual(state["replay"]["episode_ids"][name].shape, (size,))
            self.assertEqual(state["replay"]["reuse_counts"][name].shape, (size,))
        full = dict(state, replay=full_capacity_replay(agent.replay))
        compact_bytes, full_bytes = saved_bytes(state), saved_bytes(full)
        self.assertLess(len(compact_bytes), len(full_bytes) / 3)
        # Both forms load, earlier full-capacity checkpoints included, to the same replay.
        for data in (compact_bytes, full_bytes):
            restored = make_implicit()
            restored.load_state_dict(torch.load(io.BytesIO(data), weights_only=False))
            self.assertIs(restored.replay_restored, True)
            assert_same_partitions(self, agent.replay, restored.replay)

    def test_checkpoint_without_replay(self):
        for build, budget in ((make_implicit, 70), (make_explicit, 50)):
            with self.subTest(build=build.__name__):
                reference = build()
                reference.learn(budget)
                with tempfile.TemporaryDirectory() as folder:
                    path = Path(folder) / "checkpoint.pt"
                    reference.save(path, include_replay=False)
                    self.assertNotIn("replay", torch.load(path, weights_only=False))
                    restored = build()
                    self.assertIsNone(restored.replay_restored)
                    restored.load(path)
                self.assertIs(restored.replay_restored, False)
                for name in ("train", "gate"):
                    self.assertEqual(restored.replay.partition_size(name), 0)
                for p, q in zip(all_parameters(reference), all_parameters(restored)):
                    self.assertTrue(torch.equal(p, q))
                optimizers = restored._optimizers()
                for name, optimizer in reference._optimizers().items():
                    if optimizer is not None:
                        assert_same_state(
                            self, optimizer.state_dict(), optimizers[name].state_dict(), name
                        )
                counts, other = reference.counters(), restored.counters()
                counts.pop("wall_seconds"), other.pop("wall_seconds")
                self.assertEqual(counts, other)
                self.assertGreater(counts["interactions"], 0)


class TestDiagnostics(unittest.TestCase):
    def test_plant_diagnostics_append_records(self):
        for agent in (make_implicit(), make_explicit()):
            collect(agent, 60)
            agent.record_plant_diagnostics(agent.replay.sample("train", 32))
            self.assertEqual(len(agent.oracle_records), 1)
            self.assertEqual(agent.oracle_records[0]["sampled_states"], 32)
            self.assertGreater(agent.oracle_records[0]["drift_error_max"], 0.0)
            self.assertGreater(len(agent.calibration_records), 0)
            self.assertEqual(
                sum(item["sample_count"] for item in agent.calibration_records), 32
            )

    def test_plant_diagnostics_are_spaced_by_interactions(self):
        agent = make_explicit()
        agent.plant_diagnostics_every = 25
        agent.learn(60)
        self.assertEqual(
            [record["interactions"] for record in agent.oracle_records], [25, 50]
        )

    def test_solver_records_are_complete(self):
        agent = make_implicit()
        agent.learn(90)
        self.assertGreater(len(agent.solver_records), 0)
        record = agent.solver_records[0]
        self.assertEqual(set(record), set(SolverMetrics.__required_keys__))
        self.assertNotIn("running_error_bound_or_estimate", record)
        self.assertAlmostEqual(
            record["contraction_factor"], 1.0 / (1.0 + agent.discount_rate)
        )
        self.assertEqual(
            record["fixed_point_motion_input"], 0.0
        )

    def test_accepted_transfer_enters_the_next_solver_record_only(self):
        agent = make_implicit()
        collect(agent, 60)
        result = agent.test_admission()
        agent.update_admitted_plant(
            {**result, "accepted": True, "transfer_diagnostic": 0.75}
        )
        agent.run_outer_iteration(remaining_interactions=50)
        agent.run_outer_iteration(remaining_interactions=50)
        self.assertEqual(agent.solver_records[0]["fixed_point_motion_input"], 0.75)
        self.assertEqual(agent.solver_records[1]["fixed_point_motion_input"], 0.0)

def synthetic_reward_batch(agent, rows=64, *, seed=0, failed=(), truncated=()):
    """Rows whose rewards are the trapezoid of the task's true rate at both ends."""
    generator = torch.Generator().manual_seed(seed)
    dim = agent.env.state_dim
    states = 0.6 * (2.0 * torch.rand(rows, dim, generator=generator) - 1.0)
    next_states = states + 0.02 * torch.randn(rows, dim, generator=generator)
    low = torch.as_tensor(agent.env.action_space.low, dtype=torch.get_default_dtype())
    high = torch.as_tensor(agent.env.action_space.high, dtype=torch.get_default_dtype())
    decisions = low + (high - low) * torch.rand(rows, low.numel(), generator=generator)
    durations = torch.full((rows, 1), 0.01)
    resolved = agent.controller.resolve_decision(states, decisions)
    start = agent.controller.action(states, resolved)
    end = agent.controller.action(next_states, resolved)
    lam = agent.discount_rate
    rewards = 0.5 * durations * (
        reward_rate_tensor(states, start, agent.testbed)
        + math.exp(-lam * 0.01) * reward_rate_tensor(next_states, end, agent.testbed)
    )
    terminated = torch.zeros(rows, 1)
    terminated[list(failed)] = 1.0
    truncation = torch.zeros(rows, 1)
    truncation[list(truncated)] = 1.0
    return {
        "states": states,
        "decisions": decisions,
        "durations": durations,
        "horizons": durations.clone(),
        "rewards": rewards,
        "next_states": next_states,
        "terminated": terminated,
        "truncated": truncation,
    }


class TestLearnedReward(unittest.TestCase):
    def test_learned_reward_is_the_default(self):
        self.assertEqual(IMPLICIT_DEFAULTS["reward_source"], "learned")
        config = implicit_config()
        for name in IMPLICIT_DEFAULTS:
            config.pop(name)
        agent = make_implicit(config)
        self.assertIsInstance(agent.reward_net, RewardRate)
        self.assertIs(agent._network_modules()["reward"], agent.reward_net)
        self.assertIs(agent._optimizers()["reward"], agent.reward_optimizer)
        self.assertEqual(agent.reward_warm_steps, 2000)
        self.assertEqual(agent.plant_warm_steps, 2000)
        self.assertEqual(agent.covariance_mode, "leading_order")
        self.assertEqual(agent.temperature_every_rows, 40.0)

    def test_known_reward_reads_the_task_function(self):
        agent = make_implicit(implicit_config(reward_source="known"))
        self.assertIsNone(agent.reward_net)
        self.assertIsNone(agent._network_modules()["reward"])
        operator = agent.freeze_operator()
        states, torques = torch.randn(5, 2), torch.randn(5, 1)
        expected = reward_rate_tensor(states, torques, agent.testbed)
        torch.testing.assert_close(agent._reward_rate(states, torques), expected)
        torch.testing.assert_close(operator["reward_rate"](states, torques), expected)
        self.assertEqual(agent.update_reward(synthetic_reward_batch(agent)), {})

    def test_reward_network_leaves_the_other_initializations_alone(self):
        learned = make_implicit()
        known = make_implicit(implicit_config(reward_source="known"))
        for name in ("actor", "value", "rate", "online_plant"):
            self.assertTrue(
                same(learned._network_modules()[name], snapshot(known._network_modules()[name])),
                name,
            )
        again = make_implicit()
        self.assertTrue(same(again.reward_net, snapshot(learned.reward_net)))

    def test_reward_rate_returns_the_copy_frozen_with_the_operator(self):
        agent = make_implicit()
        collect(agent, 40)
        states = agent.replay.sample("train", 8)["states"]
        torques = torch.linspace(-5.0, 5.0, 8).reshape(8, 1)
        operator = agent.freeze_operator()
        frozen = agent._reward_rate(states, torques).detach()
        torch.testing.assert_close(operator["reward_rate"](states, torques), frozen)
        with torch.no_grad():
            for p in agent.reward_net.parameters():
                p.add_(1.0)
        live = agent.reward_net(states, torques).detach()
        self.assertFalse(torch.allclose(live, frozen))
        torch.testing.assert_close(agent._reward_rate(states, torques), frozen)
        torch.testing.assert_close(operator["reward_rate"](states, torques), frozen)
        # The next outer iteration's operator copies the moved network.
        agent.freeze_operator()
        torch.testing.assert_close(agent._reward_rate(states, torques), live)

    def test_every_consumer_reads_the_frozen_copy(self):
        agent = make_implicit()
        collect(agent, 40)
        operator = agent.freeze_operator()
        with torch.no_grad():
            for p in agent.reward_net.parameters():
                p.fill_(float("nan"))
        metrics = agent.update_value(agent.replay.sample("train", 8), operator)
        self.assertTrue(math.isfinite(metrics["value_loss"]))
        self.assertTrue(all(torch.isfinite(p).all() for p in agent.value.parameters()))
        self.assertTrue(math.isfinite(agent.fit_rate_head(1)["rate_loss"]))
        samples = agent._gate_samples(agent.replay.sample("train", 4), 2)
        self.assertTrue(torch.isfinite(samples["candidate_returns"]).all())

    def test_one_reward_step_per_value_step_on_the_same_batch(self):
        agent = make_implicit()
        collect(agent, 40)
        batches = []
        real = agent.update_reward

        def recording(batch):
            batches.append(batch)
            return real(batch)

        agent.update_reward = recording
        operator = agent.freeze_operator()
        batch = agent.replay.sample("train", 8)
        metrics = agent.update_value(batch, operator)
        self.assertEqual(len(batches), 1)
        self.assertIs(batches[0], batch)
        self.assertIn("reward_fit_loss", metrics)
        self.assertEqual(agent.last_reward_fit_loss, metrics["reward_fit_loss"])
        agent.run_outer_iteration(remaining_interactions=50)
        counters = agent.counters()
        self.assertEqual(counters["reward_gradient_steps"], counters["value_gradient_steps"])

    def test_reward_loss_decreases_on_a_synthetic_batch(self):
        agent = make_implicit()
        batch = synthetic_reward_batch(agent, rows=128)
        losses = [agent.update_reward(batch)["reward_fit_loss"] for _ in range(300)]
        self.assertLess(losses[-1], 0.2 * losses[0])
        self.assertEqual(agent.counters()["reward_gradient_steps"], 300)

    def test_the_loss_is_the_trapezoid_over_rows_that_did_not_fail(self):
        agent = make_implicit()
        failed, truncated = (0, 3, 5), (1, 6)
        batch = synthetic_reward_batch(agent, rows=8, failed=failed, truncated=truncated)
        batch["rewards"][list(failed)] = 1e6
        states, start, next_states, end, durations, rewards = reward_fit_rows(
            agent.controller, batch
        )
        self.assertEqual(states.shape[0], 5)
        self.assertLess(float(rewards.abs().max()), 1e3)
        net = copy.deepcopy(agent.reward_net)
        with torch.no_grad():
            hand = 0.5 * durations * (
                net(states, start)
                + torch.exp(-agent.discount_rate * durations) * net(next_states, end)
            )
            expected = (((hand - rewards) / durations) ** 2).mean()
            computed = reward_rate_loss(
                net, states, start, next_states, end, durations, rewards,
                discount_rate=agent.discount_rate,
            )
        torch.testing.assert_close(computed, expected)
        self.assertAlmostEqual(
            agent.update_reward(batch)["reward_fit_loss"], float(expected), places=5
        )
        everything_failed = synthetic_reward_batch(agent, rows=4, failed=range(4))
        steps = agent.counters()["reward_gradient_steps"]
        self.assertEqual(agent.update_reward(everything_failed), {})
        self.assertEqual(agent.counters()["reward_gradient_steps"], steps)

    def test_diagnostics_are_recorded_without_touching_training_streams(self):
        agent = make_implicit()
        agent.reward_diagnostics_every = 3
        batch = synthetic_reward_batch(agent, rows=16)
        global_state = torch.get_rng_state()
        policy_state = agent._generator.get_state()
        for _ in range(7):
            agent.update_reward(batch)
        self.assertTrue(torch.equal(torch.get_rng_state(), global_state))
        self.assertTrue(torch.equal(agent._generator.get_state(), policy_state))
        self.assertEqual([r["steps"] for r in agent.reward_records], [3, 6])
        record = agent.reward_records[-1]
        for key in ("rmse_taken", "rmse_policy_draws", "true_rate_sd_taken", "true_rate_sd_draws"):
            self.assertTrue(math.isfinite(record[key]) and record[key] >= 0.0, key)
        states = batch["states"]
        torques = agent.controller.action(
            states, agent.controller.resolve_decision(states, batch["decisions"])
        )
        direct = agent.reward_fit_diagnostics(states, torques)
        with torch.no_grad():
            error = agent.reward_net(states, torques) - reward_rate_tensor(
                states, torques, agent.testbed
            )
        self.assertAlmostEqual(direct["rmse_taken"], float(error.square().mean().sqrt()), places=6)
        with self.assertRaises(RuntimeError):
            make_implicit(implicit_config(reward_source="known")).reward_fit_diagnostics(
                states, torques
            )


class TestWarmFit(unittest.TestCase):
    def test_warm_fit_counts_and_admits_the_fitted_plant(self):
        agent = make_implicit(implicit_config(plant_warm_steps=5, reward_warm_steps=7))
        collect(agent, 40)
        online, reward = snapshot(agent.online_plant), snapshot(agent.reward_net)
        agent.warm_fit()
        counters = agent.counters()
        self.assertEqual(counters["plant_gradient_steps"], 5)
        self.assertEqual(counters["reward_gradient_steps"], 7)
        self.assertEqual(counters["value_gradient_steps"], 0)
        self.assertFalse(same(agent.online_plant, online))
        self.assertFalse(same(agent.reward_net, reward))
        self.assertTrue(same(agent.admitted_plant, snapshot(agent.online_plant)))
        self.assertEqual(agent.plant_version, 1)
        self.assertEqual(agent.admission_records, [])
        agent.warm_fit()
        self.assertEqual(agent.counters()["plant_gradient_steps"], 5)
        self.assertEqual(agent.plant_version, 1)

    def test_warm_fit_waits_for_a_batch_and_precedes_the_first_value_step(self):
        agent = make_implicit(implicit_config(plant_warm_steps=5, reward_warm_steps=7))
        self.assertEqual(agent.warm_fit(), {})
        self.assertFalse(agent.warm_fit_done)
        initial = snapshot(agent.reward_net)
        first = {}
        real = agent.update_value

        def recording(batch, operator):
            if not first:
                first["counters"] = dict(agent.counters())
                first["frozen_is_live"] = same(agent._reward_frozen, snapshot(agent.reward_net))
                first["frozen_is_initial"] = same(agent._reward_frozen, initial)
            return real(batch, operator)

        agent.update_value = recording
        agent.learn(30)
        self.assertTrue(agent.warm_fit_done)
        self.assertEqual(first["counters"]["plant_gradient_steps"], 5)
        self.assertEqual(first["counters"]["reward_gradient_steps"], 7)
        self.assertEqual(first["counters"]["value_gradient_steps"], 0)
        # The operator frozen before a batch existed reads the warm-fitted reward.
        self.assertTrue(first["frozen_is_live"])
        self.assertFalse(first["frozen_is_initial"])

    def test_warm_fit_after_warm_up_episodes(self):
        config = implicit_config(plant_warm_steps=3, reward_warm_steps=2)
        config["training"]["warmup_episodes"] = 1
        agent = make_implicit(config)
        calls = []
        real = agent.warm_fit

        class Stop(Exception):
            pass

        def recording():
            calls.append((agent._warmup_episodes_done, agent.warm_fit_done))
            real()
            raise Stop

        agent.warm_fit = recording
        with self.assertRaises(Stop):
            agent.learn(5000)
        # The first call follows the warm-up episode, before any value step.
        self.assertEqual(calls, [(1, False)])
        self.assertTrue(agent.warm_fit_done)
        counters = agent.counters()
        self.assertEqual(counters["value_gradient_steps"], 0)
        self.assertEqual(counters["reward_gradient_steps"], 2)
        self.assertEqual(counters["plant_gradient_steps"], 3)

    def test_oracle_plant_is_not_warm_fitted(self):
        agent = make_implicit(
            implicit_config(plant_source="oracle", plant_warm_steps=5, reward_warm_steps=4)
        )
        collect(agent, 40)
        agent.warm_fit()
        self.assertEqual(agent.counters()["plant_gradient_steps"], 0)
        self.assertEqual(agent.counters()["reward_gradient_steps"], 4)
        self.assertEqual(agent.plant_version, 0)


class TestCovarianceMode(unittest.TestCase):
    def test_plant_fit_reads_its_covariance_setting(self):
        for mode in ("leading_order", "propagated"):
            agent = make_implicit(implicit_config(plant_fit_covariance=mode))
            self.assertEqual(agent.covariance_mode, mode)
            collect(agent, 40)
            with mock.patch.object(
                implicit_module,
                "integrate_endpoint_moments",
                wraps=implicit_module.integrate_endpoint_moments,
            ) as moments:
                agent.update_online_plant(agent.replay.sample("train", 8))
            moments.assert_called_once()
            self.assertEqual(moments.call_args.kwargs["covariance_mode"], mode)

    def test_unknown_settings_are_rejected(self):
        for override in (
            {"plant_fit_covariance": "exact"},
            {"reward_source": "oracle"},
            {"plant_warm_steps": -1},
            {"reward_warm_steps": 1.5},
            {"temperature_every_rows": 0},
        ):
            with self.assertRaises(ValueError, msg=str(override)):
                implicit_config(**override)
        base = implicit_config()
        with self.assertRaises(ValueError):
            implicit_config(admission={**base["admission"], "plant_update": "sometimes"})
        with self.assertRaises(ValueError):
            implicit_config(network={**base["network"], "diffusion": "tiny"})
        implicit_config(network={**base["network"], "diffusion": "none"})


class TestAlwaysAdmission(unittest.TestCase):
    def _agent(self, rows=60):
        base = implicit_config()
        agent = make_implicit(
            implicit_config(admission={**base["admission"], "plant_update": "always"})
        )
        collect(agent, rows)
        return agent

    def test_every_cycle_end_admits_the_online_plant_untested(self):
        agent = self._agent()
        agent.test_admission = mock.Mock(side_effect=AssertionError("No gate here."))
        agent._reserve_gate_rows = mock.Mock(side_effect=AssertionError("No gate rows."))
        agent.cycle_length = 2
        before = snapshot(agent.admitted_plant)
        version = agent.plant_version
        outcome = agent.run_admission_cycle(remaining_interactions=500)
        self.assertEqual(outcome, {"gate_tested": 0.0, "gate_accepted": 1.0})
        self.assertFalse(same(agent.admitted_plant, before))
        self.assertTrue(same(agent.admitted_plant, snapshot(agent.online_plant)))
        self.assertEqual(agent.plant_version, version + 1)
        record = agent.admission_records[-1]
        self.assertEqual(record["status"], "admitted_without_test")
        self.assertTrue(record["accepted"])
        self.assertTrue(math.isfinite(record["transfer_diagnostic"]))
        self.assertEqual(record["validation_rows"], 0)
        self.assertEqual(record["real_rows_used"], 0)
        self.assertEqual(record["next_outer_iterations"], agent.base_spacing)
        self.assertEqual(agent.cycle_length, agent.base_spacing)
        self.assertEqual(agent._pending_motion, record["transfer_diagnostic"])
        counters = agent.counters()
        self.assertEqual(counters["gate_simulated_paths"], 0)
        self.assertEqual(counters["gate_real_rows_acquired"], 0)
        self.assertEqual(int(agent.replay.provenance("gate")["reuse_counts"].sum()), 0)

    def test_j_decides_nothing(self):
        agent = self._agent()
        for transfer in (0.0, 1e6, float("nan")):
            online = snapshot(agent.online_plant)
            with mock.patch.object(
                implicit_module, "transfer_diagnostic", return_value=torch.tensor(transfer)
            ):
                agent.update_admitted_plant(agent.admit_without_test())
            self.assertTrue(same(agent.admitted_plant, online))
            self.assertTrue(agent.admission_records[-1]["accepted"])
            self.assertEqual(agent.cycle_length, agent.base_spacing)

    def test_cycles_repeat_at_the_fixed_spacing_in_a_run(self):
        agent = self._agent(rows=0)
        agent.cycle_length = agent.base_spacing = 1
        agent.learn(60)
        self.assertGreater(len(agent.admission_records), 1)
        self.assertEqual(len(agent.admission_records), agent.outer_iteration)
        self.assertTrue(all(r["status"] == "admitted_without_test" for r in agent.admission_records))

    def test_oracle_has_nothing_to_admit(self):
        base = implicit_config()
        agent = make_implicit(
            implicit_config(
                plant_source="oracle",
                admission={**base["admission"], "plant_update": "always"},
            )
        )
        agent.cycle_length = 1
        agent.learn(40)
        self.assertEqual(agent.admission_records, [])


class TestBothTestbeds(unittest.TestCase):
    def test_all_algorithms_run_on_both_testbeds(self):
        for testbed in (windy(), acrobot()):
            with self.subTest(testbed=testbed["name"]):
                base = implicit_config(testbed)
                base["training"]["warmup_episodes"] = 0
                base["training"]["validation_fraction"] = 0.2
                base["admission"]["efolds"] = 0.05
                agent = make_implicit(base)
                agent.learn(200)
                self.assertEqual(agent.interactions, 200)
                self.assertGreater(agent.counters()["value_gradient_steps"], 0)
                self.assertGreater(agent.counters()["actor_gradient_steps"], 0)
                self.assertGreater(len(agent.admission_records), 0)
                for parameter in all_parameters(agent):
                    self.assertTrue(torch.isfinite(parameter).all())
                state = agent.env.reset(seed=1)[0]
                self.assertTrue(agent.env.action_space.contains(agent.predict(state)))

                for continuation in ("generator", "rollout", "td"):
                    explicit = make_explicit(
                        explicit_config(testbed, continuation=continuation)
                    )
                    explicit.learn(60)
                    self.assertEqual(explicit.interactions, 60)
                    for parameter in all_parameters(explicit):
                        self.assertTrue(torch.isfinite(parameter).all())

    def test_oracle_runs_on_acrobot(self):
        agent = make_implicit(implicit_config(acrobot(), plant_source="oracle"))
        agent.learn(80)
        self.assertEqual(agent.counters()["plant_gradient_steps"], 0)
        self.assertGreater(agent.counters()["rate_gradient_steps"], 0)


class TestDecisionWarmUp(unittest.TestCase):
    """The training key warmup_decisions: exactly that many uniform decisions, counted
    across episode ends (warmup_episodes ignored), before any gradient step; then the
    episode in progress continues under the policy, and the warm fit, the first training
    step and the temperature's row count follow as after an episode warm-up. The windy
    pendulum with 50 ms episodes (about five decisions each)."""

    BUDGET = 60
    # Far more episodes than any decision warm-up here spans: ignored under the key.
    EPISODES = 1000

    def config(self, kind, warmup, *, learned_temperature=False, **overrides):
        testbed = windy()
        testbed["episode_duration"] = 0.05
        build = implicit_config if kind == "implicit" else explicit_config
        config = build(testbed, **overrides)
        config["temperature"]["learn_temperature"] = learned_temperature
        config["training"]["warmup_episodes"] = self.EPISODES
        config["training"]["warmup_decisions"] = warmup
        validate_configuration(config)
        return config

    @staticmethod
    def make(kind, config):
        return make_implicit(config) if kind == "implicit" else make_explicit(config)

    def warmup_length(self, kind, **options):
        """A warm-up of 20 to 40 decisions whose last transition ends no episode, so that
        the policy has an episode in progress to continue. A probe with a longer warm-up
        draws the same decisions and intervals up to there (the same seeds and streams)."""
        probe = self.make(kind, self.config(kind, 10**6, **options))
        transitions = spy_transitions(probe)
        probe.learn(40)
        self.assertEqual(len(transitions), 40)
        for index in range(19, 40):
            if not ended(transitions[index]):
                return index + 1
        self.fail("every transition from the 20th to the 40th ended an episode")

    def run_agent(self, kind, **options):
        warmup = self.warmup_length(kind, **options)
        agent = self.make(kind, self.config(kind, warmup, **options))
        self.assertEqual(agent.warmup_decisions, warmup)
        drawn = watch_decisions(agent)
        transitions = spy_transitions(agent)
        agent.learn(self.BUDGET)
        # The budget includes the warm-up.
        self.assertEqual(agent.interactions, self.BUDGET)
        self.assertEqual(
            [entry[0] for entry in drawn],
            ["uniform"] * warmup + ["policy"] * (self.BUDGET - warmup),
        )
        self.assertEqual([entry[1] for entry in drawn], list(range(self.BUDGET)))
        # No gradient step of any kind before or during the warm-up.
        for entry in drawn[:warmup]:
            self.assertEqual(set(entry[3].values()), {0})
        # Episodes ended inside the warm-up and were reset; warmup_episodes counted none.
        ends = [ended(transition) for transition in transitions]
        self.assertEqual(len(ends), self.BUDGET)
        self.assertGreater(sum(ends[:warmup]), 0)
        self.assertEqual(agent._warmup_episodes_done, 0)
        # Nothing is reset but after an episode end: a decision opens a new episode exactly
        # when the transition before it ended one, also across the warm-up's end, where the
        # policy continues the episode of the warm-up's last decision.
        episodes = [entry[2] for entry in drawn]
        for index in range(1, self.BUDGET):
            self.assertEqual(episodes[index], episodes[index - 1] + int(ends[index - 1]), index)
        self.assertFalse(ends[warmup - 1])
        self.assertEqual(episodes[warmup], episodes[warmup - 1])
        return agent, drawn, warmup

    def test_explicit_agent(self):
        agent, drawn, warmup = self.run_agent("explicit")
        # As after an episode warm-up: the first policy decision, then the first gradient
        # step, then one per decision (the replay holds a batch).
        self.assertEqual(set(drawn[warmup][3].values()), {0})
        self.assertEqual(drawn[warmup + 1][3]["critic_gradient_steps"], 1)
        self.assertEqual(agent.counters()["critic_gradient_steps"], self.BUDGET - warmup)

    def test_implicit_agent(self):
        agent, drawn, warmup = self.run_agent(
            "implicit", learned_temperature=True, plant_warm_steps=3, reward_warm_steps=2
        )
        # The warm fit follows the warm-up, before the first policy decision and the first
        # value step, as after an episode warm-up.
        first = drawn[warmup][3]
        self.assertEqual(first["plant_gradient_steps"], 3)
        self.assertEqual(first["reward_gradient_steps"], 2)
        self.assertEqual(first["value_gradient_steps"], 0)
        self.assertEqual(first["temperature_gradient_steps"], 0)
        self.assertTrue(agent.warm_fit_done)
        self.assertGreater(agent.counters()["value_gradient_steps"], 0)
        # The temperature counts exactly the rows after the warm-up (training was ready).
        self.assertFalse(agent._collecting_warmup)
        self.assertEqual(agent.tuning_rows, self.BUDGET - warmup)

    def test_budget_inside_the_warm_up_resumes_it(self):
        agent = make_explicit(self.config("explicit", 30))
        drawn = watch_decisions(agent)
        agent.learn(10)
        self.assertEqual(agent.interactions, 10)
        self.assertEqual(agent.counters()["critic_gradient_steps"], 0)
        agent.learn(self.BUDGET)
        self.assertEqual(
            [entry[0] for entry in drawn], ["uniform"] * 30 + ["policy"] * (self.BUDGET - 30)
        )
        self.assertEqual(agent.counters()["critic_gradient_steps"], self.BUDGET - 30)

    def test_zero_decisions_start_with_the_policy(self):
        agent = make_explicit(self.config("explicit", 0))
        drawn = watch_decisions(agent)
        agent.learn(30)
        self.assertEqual({entry[0] for entry in drawn}, {"policy"})
        # Training once the replay holds a batch: decisions 8 to 30.
        self.assertEqual(agent.counters()["critic_gradient_steps"], 30 - 8 + 1)

    def test_warm_start_skips_it_and_checkpoints_carry_that(self):
        source = make_implicit()
        collect(source, 10)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "source.pt"
            source.save(path)
            agent = make_implicit(self.config("implicit", 20))
            agent.warm_start(path)
        self.assertTrue(agent._warmup_skipped)
        self.assertEqual(agent._warmup_episodes_done, self.EPISODES)
        drawn = watch_decisions(agent)
        agent.learn(5)
        self.assertEqual([entry[0] for entry in drawn], ["policy"] * 5)
        # A restart of the warm-started run keeps the warm-up skipped.
        state = agent.state_dict(include_replay=False)
        self.assertIs(state["warmup_skipped"], True)
        restored = make_implicit(self.config("implicit", 20))
        self.assertFalse(restored._warmup_skipped)
        restored.load_state_dict(state)
        self.assertTrue(restored._warmup_skipped)
        # Without a budget, a warm-up not skipped would collect up to decision 20.
        restored.warmup(restored.training["warmup_episodes"])
        self.assertEqual(restored.interactions, 5)
        # Under the key and not warm-started, checkpoints carry False.
        fresh = make_explicit(self.config("explicit", 20))
        collect(fresh, 1)
        self.assertIs(fresh.state_dict(include_replay=False)["warmup_skipped"], False)

    def test_absent_or_null_key_is_the_episode_warm_up(self):
        runs = []
        for null in (False, True):
            testbed = windy()
            testbed["episode_duration"] = 0.05
            config = explicit_config(testbed)
            config["training"]["warmup_episodes"] = 3
            if null:
                config["training"]["warmup_decisions"] = None
                validate_configuration(config)
            agent = make_explicit(config)
            self.assertIsNone(agent.warmup_decisions)
            drawn = watch_decisions(agent)
            agent.learn(80)
            self.assertEqual(agent._warmup_episodes_done, 3)
            # Checkpoints without the key are as before it.
            self.assertNotIn("warmup_skipped", agent.state_dict(include_replay=False))
            runs.append((agent, drawn))
        (plain, plain_drawn), (null, null_drawn) = runs
        self.assertEqual(plain_drawn, null_drawn)
        # The episode warm-up ends with its third episode: the policy opens a fourth.
        warmup = [entry[0] for entry in plain_drawn].count("uniform")
        self.assertEqual(plain_drawn[warmup][0], "policy")
        self.assertEqual(plain_drawn[warmup][2], 3)
        for left, right in zip(all_parameters(plain), all_parameters(null)):
            self.assertTrue(torch.equal(left, right))
        self.assertEqual(
            {k: v for k, v in plain.counters().items() if k != "wall_seconds"},
            {k: v for k, v in null.counters().items() if k != "wall_seconds"},
        )

    def test_setting_and_validation(self):
        self.assertIsNone(warmup_decisions_setting({}))
        self.assertIsNone(warmup_decisions_setting({"warmup_decisions": None}))
        self.assertEqual(warmup_decisions_setting({"warmup_decisions": 0}), 0)
        self.assertEqual(warmup_decisions_setting({"warmup_decisions": 10000}), 10000)
        value = warmup_decisions_setting({"warmup_decisions": np.int64(7)})
        self.assertEqual(value, 7)
        self.assertIs(type(value), int)
        for bad in (-1, True, False, 2.5, 10000.0, "10000", [5]):
            with self.subTest(bad=bad):
                with self.assertRaisesRegex(ValueError, "warmup_decisions"):
                    warmup_decisions_setting({"warmup_decisions": bad})
                for build, make in (
                    (implicit_config, make_implicit),
                    (explicit_config, make_explicit),
                ):
                    config = build()
                    config["training"]["warmup_decisions"] = bad
                    with self.assertRaisesRegex(ValueError, "warmup_decisions"):
                        validate_configuration(config)
                    with self.assertRaisesRegex(ValueError, "warmup_decisions"):
                        make(config)


# The random input-matrix start at training seed 0, as the runs of 2026-11-20 drew it
# (float32 default dtype): a 1x1 G (the windy pendulum) and a 2x1 G (the Acrobot).
RANDOM_START_SEED_0 = {
    "windy_pendulum": [[-0.936945915222168]],
    "acrobot_xk": [[-0.936945915222168], [1.187308430671692]],
}


def with_network(config, **keys):
    """config with the network keys added to a new network dict, validated."""
    config = dict(config, network={**config["network"], **keys})
    validate_configuration(config)
    return config


def quiet(make, config):
    """(make(config), what its construction printed)."""
    printed = io.StringIO()
    with contextlib.redirect_stdout(printed):
        agent = make(config)
    return agent, printed.getvalue()


@contextlib.contextmanager
def float32_default():
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)
    try:
        yield
    finally:
        torch.set_default_dtype(previous)


def plant_copies(agent):
    """The copies of the online plant made at construction: admitted or target."""
    return [agent.admitted_plant] if isinstance(agent, ImplicitCTPHSAC) else [agent.target_plant]


def module_states(agent, skip):
    """Every network's parameters and buffers, by module and name, except entries named skip."""
    return {
        f"{name}.{key}": value.detach().clone()
        for name, module in agent._network_modules().items()
        if module is not None
        for key, value in module.state_dict().items()
        if key != skip
    }


def streams(agent):
    """Every random stream's position, comparable with ==."""
    state = {
        "global": torch.get_rng_state().tolist(),
        "generator": agent._generator.get_state().tolist(),
        "predict": agent._predict_generator.get_state().tolist(),
        "numpy": agent._rng.bit_generator.state,
        "rollout": agent._rollout_rng.bit_generator.state,
        "routing": agent.replay._rng.bit_generator.state,
        "buffers": {name: b._rng.bit_generator.state for name, b in agent.replay.buffers.items()},
    }
    if isinstance(agent, ImplicitCTPHSAC):
        state["gate"] = agent._gate_rng.bit_generator.state
    else:
        state["next_clock"] = agent._next_clock.state_dict()
    return state


class TestNetworkStartKeys(unittest.TestCase):
    """The column-less network keys actor_log_std_init and input_matrix_start
    (common.ph_config.NETWORK_COLUMNLESS_KEYS). Building only, apart from one short
    training run that moves G off the zero start."""

    BUILDS = ((implicit_config, make_implicit), (explicit_config, make_explicit))

    def test_absent_keys_give_the_current_start_and_stated_defaults_the_same_agent(self):
        for build, make in self.BUILDS:
            for testbed in (windy(), acrobot()):
                with self.subTest(make=make.__name__, testbed=testbed["name"]):
                    agent, printed = quiet(make, build(testbed, seed=0))
                    self.assertNotIn("[input-start]", printed)
                    log_std = agent.actor.log_std.detach()
                    self.assertTrue(torch.equal(log_std, torch.full_like(log_std, LOG_STD_INIT)))
                    np.testing.assert_allclose(
                        agent.online_plant.input_matrix().detach().numpy(),
                        agent.env.oracle_input_matrix(),
                    )
                    reference = (module_states(agent, None), streams(agent))
                    stated, printed = quiet(
                        make,
                        with_network(
                            build(testbed, seed=0),
                            actor_log_std_init=LOG_STD_INIT,
                            input_matrix_start="true",
                        ),
                    )
                    self.assertNotIn("[input-start]", printed)
                    states = module_states(stated, None)
                    self.assertEqual(sorted(states), sorted(reference[0]))
                    for name, value in reference[0].items():
                        self.assertTrue(torch.equal(value, states[name]), name)
                    self.assertEqual(streams(stated), reference[1])

    def test_actor_log_std_init_sets_every_entry_and_moves_nothing_else(self):
        for build, make in self.BUILDS:
            with self.subTest(make=make.__name__):
                plain = make(build(seed=0))
                plain_streams = streams(plain)
                plain_states = module_states(plain, "log_std")
                keyed = make(with_network(build(seed=0), actor_log_std_init=-0.5))
                self.assertEqual(
                    keyed.actor.log_std.detach().tolist(), [-0.5] * keyed.actor.decision_dim
                )
                self.assertEqual(keyed.actor.log_std.dtype, plain.actor.log_std.dtype)
                self.assertEqual(streams(keyed), plain_streams)
                states = module_states(keyed, "log_std")
                self.assertEqual(sorted(states), sorted(plain_states))
                for name, value in plain_states.items():
                    self.assertTrue(torch.equal(value, states[name]), name)

    def test_random_input_matrix_start_is_the_seeded_draw(self):
        with float32_default():
            for build, make in self.BUILDS:
                for testbed in (windy(), acrobot()):
                    with self.subTest(make=make.__name__, testbed=testbed["name"]):
                        plain = make(build(testbed, seed=0))
                        true = plain.online_plant.force_matrix.detach().tolist()
                        agent, printed = quiet(
                            make, with_network(build(testbed, seed=0), input_matrix_start="random")
                        )
                        start = agent.online_plant.force_matrix
                        expected = RANDOM_START_SEED_0[testbed["name"]]
                        self.assertEqual(start.detach().tolist(), expected)
                        self.assertEqual(start.dtype, torch.float32)
                        self.assertTrue(start.requires_grad)
                        # The formula: float64 draws of a private CPU generator, cast to G.
                        generator = torch.Generator(device="cpu").manual_seed(
                            INPUT_MATRIX_START_SEED_OFFSET + 0
                        )
                        draw = torch.randn(
                            tuple(start.shape), generator=generator, dtype=torch.float64
                        )
                        self.assertTrue(torch.equal(start.detach(), draw.to(start)))
                        lines = [
                            line for line in printed.splitlines() if line.startswith("[input-start]")
                        ]
                        self.assertEqual(
                            lines, [f"[input-start] random seed 0 true {true} start {expected}"]
                        )
            # Another training seed draws another start, from its own generator.
            agent, printed = quiet(
                make_implicit, with_network(implicit_config(seed=1), input_matrix_start="random")
            )
            generator = torch.Generator(device="cpu").manual_seed(INPUT_MATRIX_START_SEED_OFFSET + 1)
            draw = torch.randn((1, 1), generator=generator, dtype=torch.float64).to(torch.float32)
            self.assertTrue(torch.equal(agent.online_plant.force_matrix.detach(), draw))
            self.assertNotEqual(draw.tolist(), RANDOM_START_SEED_0["windy_pendulum"])
            self.assertIn("[input-start] random seed 1 true ", printed)

    def test_random_start_leaves_every_other_value_and_stream_unchanged(self):
        for build, make in self.BUILDS:
            with self.subTest(make=make.__name__):
                plain = make(build(acrobot(), seed=0))
                plain_streams = streams(plain)
                plain_states = module_states(plain, "force_matrix")
                keyed, _ = quiet(
                    make, with_network(build(acrobot(), seed=0), input_matrix_start="random")
                )
                self.assertEqual(streams(keyed), plain_streams)
                states = module_states(keyed, "force_matrix")
                self.assertEqual(sorted(states), sorted(plain_states))
                for name, value in plain_states.items():
                    self.assertTrue(torch.equal(value, states[name]), name)
                self.assertFalse(
                    torch.equal(plain.online_plant.force_matrix, keyed.online_plant.force_matrix)
                )

    def test_every_construction_copy_starts_at_the_random_start(self):
        for build, make in self.BUILDS:
            with self.subTest(make=make.__name__):
                agent, _ = quiet(
                    make, with_network(build(acrobot(), seed=0), input_matrix_start="random")
                )
                online = agent.online_plant.force_matrix
                for copy_ in plant_copies(agent):
                    self.assertTrue(torch.equal(copy_.force_matrix, online.detach()))
                    self.assertNotEqual(copy_.force_matrix.data_ptr(), online.data_ptr())
                    self.assertFalse(copy_.force_matrix.requires_grad)
                    self.assertTrue(same(copy_, snapshot(agent.online_plant)))
                # The plant optimizer holds the online G and has taken no step.
                held = [p for group in agent.plant_optimizer.param_groups for p in group["params"]]
                self.assertTrue(any(p is online for p in held))
                self.assertEqual(len(agent.plant_optimizer.state), 0)

    def test_loading_a_state_overwrites_the_start_in_every_plant(self):
        # A resumed run builds the agent (drawing the start again) and loads its state.
        for build, make in self.BUILDS:
            with self.subTest(make=make.__name__):
                config = with_network(build(acrobot(), seed=0), input_matrix_start="random")
                source, _ = quiet(make, config)
                collect(source, 2)
                with torch.no_grad():
                    source.online_plant.force_matrix.add_(0.25)
                    for copy_ in plant_copies(source):
                        copy_.force_matrix.sub_(0.5)
                saved = source.state_dict(include_replay=False)
                fresh, _ = quiet(make, config)
                fresh.load_state_dict(saved)
                self.assertTrue(
                    torch.equal(fresh.online_plant.force_matrix, source.online_plant.force_matrix)
                )
                for loaded, kept in zip(plant_copies(fresh), plant_copies(source)):
                    self.assertTrue(torch.equal(loaded.force_matrix, kept.force_matrix))
                    self.assertFalse(
                        torch.equal(loaded.force_matrix, fresh.online_plant.force_matrix)
                    )

    def test_known_plant_arms_refuse_the_random_start(self):
        for build, make in self.BUILDS:
            with self.subTest(make=make.__name__):
                config = with_network(build(plant_source="oracle"), input_matrix_start="random")
                with self.assertRaisesRegex(ValueError, "input_matrix_start"):
                    make(config)
                # The default, stated, builds the known-plant arm.
                agent = make(with_network(build(plant_source="oracle"), input_matrix_start="true"))
                self.assertIsNone(agent.online_plant)

    def test_zero_input_matrix_start_is_exactly_zero(self):
        for build, make in self.BUILDS:
            for testbed in (windy(), acrobot()):
                with self.subTest(make=make.__name__, testbed=testbed["name"]):
                    plain = make(build(testbed, seed=0))
                    true_matrix = plain.online_plant.force_matrix.detach()
                    true = true_matrix.tolist()
                    zeros = torch.zeros_like(true_matrix)
                    agent, printed = quiet(
                        make, with_network(build(testbed, seed=0), input_matrix_start="zero")
                    )
                    start = agent.online_plant.force_matrix
                    self.assertIsInstance(start, torch.nn.Parameter)
                    self.assertTrue(torch.equal(start.detach(), zeros))
                    self.assertEqual(start.dtype, true_matrix.dtype)
                    self.assertEqual(start.device, true_matrix.device)
                    self.assertTrue(start.requires_grad)
                    lines = [
                        line for line in printed.splitlines() if line.startswith("[input-start]")
                    ]
                    self.assertEqual(
                        lines, [f"[input-start] zero seed 0 true {true} start {zeros.tolist()}"]
                    )
        # Another training seed prints its own seed; the start is zero all the same.
        agent, printed = quiet(
            make_implicit, with_network(implicit_config(seed=1), input_matrix_start="zero")
        )
        start = agent.online_plant.force_matrix.detach()
        self.assertTrue(torch.equal(start, torch.zeros_like(start)))
        self.assertIn("[input-start] zero seed 1 true ", printed)

    def test_zero_start_leaves_every_other_value_and_stream_unchanged(self):
        for build, make in self.BUILDS:
            with self.subTest(make=make.__name__):
                plain = make(build(acrobot(), seed=0))
                plain_streams = streams(plain)
                plain_states = module_states(plain, "force_matrix")
                keyed, _ = quiet(
                    make, with_network(build(acrobot(), seed=0), input_matrix_start="zero")
                )
                self.assertEqual(streams(keyed), plain_streams)
                states = module_states(keyed, "force_matrix")
                self.assertEqual(sorted(states), sorted(plain_states))
                for name, value in plain_states.items():
                    self.assertTrue(torch.equal(value, states[name]), name)
                self.assertFalse(
                    torch.equal(plain.online_plant.force_matrix, keyed.online_plant.force_matrix)
                )

    def test_every_construction_copy_starts_at_the_zero_start(self):
        for build, make in self.BUILDS:
            with self.subTest(make=make.__name__):
                agent, _ = quiet(
                    make, with_network(build(acrobot(), seed=0), input_matrix_start="zero")
                )
                online = agent.online_plant.force_matrix
                zeros = torch.zeros_like(online.detach())
                self.assertTrue(torch.equal(online.detach(), zeros))
                for copy_ in plant_copies(agent):
                    self.assertTrue(torch.equal(copy_.force_matrix, zeros))
                    self.assertNotEqual(copy_.force_matrix.data_ptr(), online.data_ptr())
                    self.assertFalse(copy_.force_matrix.requires_grad)
                    self.assertTrue(same(copy_, snapshot(agent.online_plant)))
                # The plant optimizer holds the online G and has taken no step.
                held = [p for group in agent.plant_optimizer.param_groups for p in group["params"]]
                self.assertTrue(any(p is online for p in held))
                self.assertEqual(len(agent.plant_optimizer.state), 0)

    def test_known_plant_arms_refuse_the_zero_start(self):
        for build, make in self.BUILDS:
            for start in ("random", "zero"):
                with self.subTest(make=make.__name__, start=start):
                    config = with_network(build(plant_source="oracle"), input_matrix_start=start)
                    # The message names the key and the refused value.
                    with self.assertRaisesRegex(ValueError, f"input_matrix_start '{start}'"):
                        make(config)

    def test_zero_start_input_matrix_leaves_zero_under_training(self):
        # The explicit arm takes one plant step per interaction once a batch exists
        # (TestBudgetAndCounters.test_explicit_counts_one_step_per_interaction_per_component).
        agent, _ = quiet(
            make_explicit, with_network(explicit_config(), input_matrix_start="zero")
        )
        start = agent.online_plant.force_matrix.detach()
        self.assertTrue(torch.equal(start, torch.zeros_like(start)))
        agent.learn(30)
        self.assertGreater(agent.counters()["plant_gradient_steps"], 0)
        trained = agent.online_plant.force_matrix.detach()
        self.assertTrue(bool(torch.isfinite(trained).all()))
        self.assertTrue(bool((trained != 0).any()))

    def test_settings_and_validation(self):
        self.assertIsNone(actor_log_std_init_setting({}))
        self.assertEqual(actor_log_std_init_setting({"actor_log_std_init": -2}), -2.0)
        value = actor_log_std_init_setting({"actor_log_std_init": np.float32(0.5)})
        self.assertEqual(value, 0.5)
        self.assertIs(type(value), float)
        self.assertEqual(input_matrix_start_setting({}), "true")
        self.assertEqual(input_matrix_start_setting({"input_matrix_start": "random"}), "random")
        self.assertEqual(input_matrix_start_setting({"input_matrix_start": "zero"}), "zero")
        self.assertEqual(network_start_settings({}), {})
        self.assertEqual(
            network_start_settings({"actor_log_std_init": -1, "input_matrix_start": "true"}), {}
        )
        self.assertEqual(
            network_start_settings({"actor_log_std_init": 0, "input_matrix_start": "random"}),
            {"actor_log_std_init": 0.0, "input_matrix_start": "random"},
        )
        self.assertEqual(
            network_start_settings({"input_matrix_start": "zero"}), {"input_matrix_start": "zero"}
        )
        bad_values = {
            "actor_log_std_init": (-20.5, float("nan"), float("inf"), 2.5, True, "x", None, [0.0]),
            "input_matrix_start": ("Random", "learned", "", 1, None, True, "Zero", "zeros", 0),
        }
        for key, values in bad_values.items():
            for index, bad in enumerate(values):
                for build, make in self.BUILDS:
                    with self.subTest(key=key, bad=bad, make=make.__name__):
                        config = build()
                        config["network"] = {**config["network"], key: bad}
                        with self.assertRaisesRegex(ValueError, key):
                            validate_configuration(config)
                        if index == 0:
                            # The agents check it too, before building anything from it.
                            with self.assertRaisesRegex(ValueError, key):
                                make(config)


if __name__ == "__main__":
    unittest.main()

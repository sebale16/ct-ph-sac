"""The value-step ablation (common.ph_config.STEP_ABLATION_DEFAULTS, protocol.step_ablation_arms).

Checks: the keys at their defaults leave every other arm's config, resume fingerprint and
run unchanged (bitwise against the keys spelled out at their defaults); the explicit step's
target is V_k + h E_pi[rate(V_k)] with the model operator written out by hand, for h = tau
and h = each row's Delta, and the observed-interval rate at contact-excluded rows; the
critic operator's rate is (Q - V) / Delta_ref - alpha_c log pi, its resolvent residual has
the stated root, Delta_ref is the clock's mean interval and the critic target is the
explicit generator arm's; every new arm learns with finite losses and monitors in both
decision modes (torque, the arms' own as the paper arms', and feedback) and restarts
exactly; the pilot configs are the lqr table rows on the irregular clock with 2 N motors.
Small networks on the windy pendulum and tiny protocol runs on pendulum-swingup keep each
test to seconds on a CPU; the agents take torque decisions unless a test says feedback.
"""

import copy
import json
import math
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

import algorithms.implicit_ct_ph_sac as implicit_module
from algorithms.explicit_ct_ph_sac import ExplicitCTPHSAC
from algorithms.implicit_ct_ph_sac import ImplicitCTPHSAC
from algorithms.ph_resolvent import (
    critic_soft_rates,
    explicit_step_residuals,
    operator_anchor_rates,
)
from algorithms.ph_targets import generator_critic_target
from benchmarks import run_ph_sac
from common.ph_config import (
    STEP_ABLATION_CHOICES,
    STEP_ABLATION_DEFAULTS,
    step_ablation_setting,
    torque_decision_config,
    validate_configuration,
)
from common.ph_time import discount_weight, mean_decision_interval
from environment.ph_sac.tasks import make_env
from evaluations.ph_sac import protocol as P
from evaluations.ph_sac import resume as R
from evaluations.ph_sac.diagnostics import SolverMetrics
from models.ph_sac.networks import TwinQCritic
from models.ph_sac.operators import value_derivatives
from tests.test_ph_algorithms import (
    all_parameters,
    collect,
    implicit_config,
    make_implicit,
    same,
    snapshot,
)
from tests.test_ph_resume import ExactRestartMixin, tiny_run

ROOT = Path(__file__).resolve().parents[1]
PILOTS = ROOT / "benchmarks" / "configs" / "ph_sac" / "step_ablation"
# Each step-ablation arm's choices (protocol.OptionalArmFields).
STEP_ARMS = {
    "implicit_explicit_step": {"value_step": "explicit", "explicit_step_size": "resolvent_step"},
    "implicit_explicit_step_interval": {"value_step": "explicit", "explicit_step_size": "interval"},
    "implicit_critic": {"decision_head": "critic", "value_operator": "critic"},
    "implicit_critic_head": {"decision_head": "critic", "value_operator": "model"},
}
VALUE_MONITORS = ("value_loss", "residual_mean_square", "value_abs_max", "value_gradient_norm_max")


def ablation_config(choices, **overrides):
    """test_ph_algorithms' implicit config (windy pendulum, 16-16 networks, K 6, N 3,
    tau 1 s) with the given choices and the arms' always-admit plant."""
    config = implicit_config(**overrides)
    config["admission"] = {**config["admission"], "plant_update": "always"}
    config.update(choices)
    validate_configuration(config)
    return config


def make_ablation(choices, *, decision_mode="torque", seed=11, **overrides):
    """An agent of ablation_config's; torque decisions (the arms' own) by default."""
    config = ablation_config(choices, **overrides)
    env = make_env(config["testbed"], decision_mode=decision_mode, seed=seed)
    if decision_mode == "torque":
        config = torque_decision_config(config)
    return ImplicitCTPHSAC(env, config)


def frozen_draws(agent, operator, states):
    with torch.no_grad():
        return agent._policy_sample(
            states, sample_count=int(agent.solver["policy_draws"]), actor=operator["policy"]
        )


def hand_model_rates(agent, operator, states, sample):
    """rho_i + G^(kappa_i) V_k - lambda V_k - alpha_c log pi_i [B,N,1], written out."""
    target, plant = operator["target_value"], operator["plant"]
    derivatives = value_derivatives(target, states, plant.diffusion(states), create_graph=False)
    decisions = sample["decisions"]
    batch, draws = decisions.shape[:2]
    repeated = states.unsqueeze(1).expand(-1, draws, -1)
    with torch.no_grad():
        torques = agent.controller.action(
            repeated, agent.controller.resolve_decision(repeated, decisions)
        )
        flat_states = repeated.reshape(batch * draws, -1)
        flat_torques = torques.reshape(batch * draws, -1)
        drift = plant.drift(flat_states, flat_torques).reshape(batch, draws, -1)
        rewards = operator["reward_rate"](flat_states, flat_torques).reshape(batch, draws, 1)
    generated = (drift * derivatives["gradients"].unsqueeze(1)).sum(-1, keepdim=True) + 0.5 * (
        derivatives["diffusion_traces"].unsqueeze(1)
    )
    values = derivatives["values"].unsqueeze(1)
    alpha = operator["temperature_per_second"]
    return (
        rewards + generated - agent.discount_rate * values - alpha * sample["log_probabilities"]
    ).detach()


# ------------------------------------------------------------------ defaults and plumbing


class TestDefaultsLeaveEveryOtherArmAlone(unittest.TestCase):
    def test_settings_and_validation(self):
        config = implicit_config()
        self.assertEqual(dict(step_ablation_setting(config)), dict(STEP_ABLATION_DEFAULTS))
        for name, choices in STEP_ARMS.items():
            with self.subTest(arm=name):
                setting = step_ablation_setting(ablation_config(choices))
                for key, value in choices.items():
                    self.assertEqual(setting[key], value)
        setting = step_ablation_setting(ablation_config(STEP_ARMS["implicit_critic"]))
        self.assertAlmostEqual(setting["critic_reference_interval"], 0.010, places=12)
        given = ablation_config(STEP_ARMS["implicit_critic"], critic_reference_interval=0.02)
        self.assertEqual(step_ablation_setting(given)["critic_reference_interval"], 0.02)
        bad = (
            {"value_step": "semi_implicit"},
            {"explicit_step_size": "interval"},
            {"value_operator": "critic"},
            {"decision_head": 1},
            {"critic_reference_interval": 0.0},
            {"critic_reference_interval": True},
            {"critic_reference_interval": float("nan")},
        )
        for choices in bad:
            with self.subTest(bad=choices), self.assertRaises(ValueError):
                ablation_config(choices)
        # test_ph_algorithms' config runs the gate (plant_update "admission").
        gated = implicit_config()
        self.assertEqual(gated["admission"]["plant_update"], "admission")
        gated.update(STEP_ARMS["implicit_critic"])
        with self.assertRaisesRegex(ValueError, "admission"):
            validate_configuration(gated)

    def test_mean_decision_interval(self):
        windy = implicit_config()["testbed"]["clock"]
        # 2 and 18 ms with 0.4 each, 3..17 ms with 0.2 / 15 each: 10 ms.
        self.assertAlmostEqual(mean_decision_interval(windy), 0.010, places=12)
        fixed = dict(windy, kind="fixed", min_duration=0.03, max_duration=0.03)
        self.assertAlmostEqual(mean_decision_interval(fixed), 0.03, places=12)

    def test_existing_arm_configs_and_fingerprints_carry_no_new_key(self):
        run = tiny_run("pendulum-swingup", "implicit_learned_admissions", "/nonexistent", 10)
        proto = copy.deepcopy(run["protocol"])
        proto["arms"] = list(P.all_arms())
        P.validate_matched_quantities(proto)
        runs = {spec["arm"]["name"]: spec for spec in P.build_run_matrix(proto)}
        self.assertEqual(
            [arm["name"] for arm in P.step_ablation_arms()], list(STEP_ARMS)
        )
        self.assertFalse(set(STEP_ARMS) & {arm["name"] for arm in P.paper_arms()})
        # Each arm is the implicit paper arm, implicit_torque (torque decisions), with its
        # choices.
        catalog = {arm["name"]: arm for arm in P.all_arms()}
        for name, choices in STEP_ARMS.items():
            arm = catalog[name]
            self.assertEqual(arm["decision_mode"], "torque")
            self.assertEqual(
                {k: v for k, v in arm.items() if k not in choices and k != "name"},
                {k: v for k, v in catalog["implicit_torque"].items() if k != "name"},
            )
        reference = P._implicit_arm_config(runs["implicit_torque"])
        for name, spec in runs.items():
            with self.subTest(arm=name):
                choices = {key for key in STEP_ABLATION_CHOICES if key in spec["arm"]}
                resolved = R.run_settings(spec)["resolved"]["implicit_config"]
                if spec["arm"]["algorithm"] == "explicit":
                    config = P._explicit_arm_config(spec)
                    self.assertEqual(set(config), set(proto["explicit_config"]))
                if name not in STEP_ARMS:
                    self.assertEqual(choices, set())
                    self.assertNotIn("step_ablation", resolved)
                    if spec["arm"]["algorithm"] == "implicit":
                        config = P._implicit_arm_config(spec)
                        self.assertFalse(set(STEP_ABLATION_DEFAULTS) & set(config))
                        self.assertEqual(
                            dict(step_ablation_setting(config)), dict(STEP_ABLATION_DEFAULTS)
                        )
                    continue
                config = P._implicit_arm_config(spec)
                self.assertEqual(choices, set(STEP_ARMS[name]))
                for key, value in STEP_ARMS[name].items():
                    self.assertEqual(config[key], value)
                rest = {key: value for key, value in config.items() if key not in choices}
                self.assertTrue(P._same_settings(rest, reference))
                # run_settings holds the canonical form (floats as their hex).
                self.assertEqual(
                    resolved["step_ablation"], R._canonical(dict(step_ablation_setting(config)))
                )
        critic = R.run_settings(runs["implicit_critic"])["resolved"]["implicit_config"]
        clock = proto["implicit_config"]["testbed"]["clock"]
        self.assertEqual(
            critic["step_ablation"]["critic_reference_interval"],
            R._canonical(mean_decision_interval(clock)),
        )

    def test_default_keys_spelled_out_run_bitwise(self):
        """Every key at its default (here written into the config) is the earlier run."""
        plain = make_implicit(implicit_config())
        plain.learn(90)
        keyed = make_implicit(implicit_config(**STEP_ABLATION_DEFAULTS))
        keyed.learn(90)
        self.assertTrue(plain.default_value_update and not plain.step_monitors)
        self.assertIsNone(plain.critics)
        self.assertNotIn("critics", plain.state_dict(include_replay=False)["networks"])
        for p, q in zip(all_parameters(plain), all_parameters(keyed)):
            self.assertTrue(torch.equal(p, q))
        self.assertEqual(plain.solver_records, keyed.solver_records)
        self.assertGreater(len(plain.solver_records), 0)
        for record in plain.solver_records:
            self.assertEqual(set(record), set(SolverMetrics.__required_keys__))
        counts, other = plain.counters(), keyed.counters()
        counts.pop("wall_seconds"), other.pop("wall_seconds")
        self.assertEqual(counts, other)

    def test_arms_option_and_list(self):
        path = str(PILOTS / "lqr_2_1.json")
        args = run_ph_sac.parse_args(["--config", path, "--arms", *STEP_ARMS, "--seeds", "0"])
        proto = run_ph_sac.apply_overrides(run_ph_sac.load_protocol(path), args)
        self.assertEqual([arm["name"] for arm in proto["arms"]], list(STEP_ARMS))
        self.assertEqual(
            sorted(run["run_id"] for run in P.build_run_matrix(proto)),
            sorted(f"lqr_2_1__{name}__seed0" for name in STEP_ARMS),
        )
        listed = {line.split()[0]: line for line in run_ph_sac.list_arms()[1:-1]}
        for name in STEP_ARMS:
            self.assertIn("ablation", listed[name])

    def test_json_implicit_section_takes_only_the_reference_interval(self):
        raw = json.loads((PILOTS / "lqr_2_1.json").read_text())
        raw["implicit"]["critic_reference_interval"] = 0.03
        proto = run_ph_sac.protocol_from_config(raw, "x")
        self.assertEqual(proto["implicit_config"]["critic_reference_interval"], 0.03)
        raw["implicit"]["value_step"] = "explicit"
        with self.assertRaisesRegex(ValueError, "unknown settings"):
            run_ph_sac.protocol_from_config(raw, "x")


class TestPilotConfigs(unittest.TestCase):
    def test_pilots_are_the_table_rows_on_the_irregular_clock_with_2_n(self):
        for task in ("lqr_2_1", "lqr_6_2"):
            with self.subTest(task=task):
                pilot = run_ph_sac.load_protocol(str(PILOTS / f"{task}.json"))
                row = run_ph_sac.load_protocol_from_table(f"lqr-{task}", "top")
                for name in ("explicit_config", "implicit_config"):
                    self.assertTrue(P._same_settings(
                        {k: v for k, v in pilot[name].items() if k != "testbed"},
                        {k: v for k, v in row[name].items() if k != "testbed"},
                    ), name)
                testbed = pilot["matched"]["testbed_config"]
                self.assertTrue(P._same_settings(
                    {k: v for k, v in testbed.items() if k not in ("controller", "clock")},
                    {
                        k: v
                        for k, v in row["matched"]["testbed_config"].items()
                        if k not in ("controller", "clock")
                    },
                ))
                actuators = int(testbed["state_spec"]["action_dim"])
                np.testing.assert_array_equal(testbed["controller"]["torque_high"], [2.0] * actuators)
                clock = testbed["clock"]
                self.assertEqual(clock["kind"], "two_tail_uniform")
                self.assertAlmostEqual(clock["min_duration"], 0.006, places=12)
                self.assertAlmostEqual(clock["max_duration"], 0.09, places=12)
                self.assertAlmostEqual(mean_decision_interval(clock), 0.014736, places=9)
                self.assertEqual(pilot["matched"]["interaction_budget"], 300000)
                self.assertEqual(pilot["training_seeds"], list(range(10)))
                self.assertEqual(
                    pilot["evaluation_every_interactions"], row["evaluation_every_interactions"]
                )
                self.assertEqual(
                    pilot["matched"]["evaluation_reset_seeds"],
                    row["matched"]["evaluation_reset_seeds"],
                )
                self.assertEqual(pilot["implicit_config"]["solver"]["resolvent_step"], 1.0)


# ------------------------------------------------------------- arm A: the explicit step


class TestExplicitStep(unittest.TestCase):
    def setUp(self):
        self.agent = make_ablation(STEP_ARMS["implicit_explicit_step"])
        collect(self.agent, 40)
        self.operator = self.agent.freeze_operator()
        self.batch = self.agent.replay.sample("train", 8)
        self.sample = frozen_draws(self.agent, self.operator, self.batch["states"])

    def test_anchor_rates_are_the_model_operator_at_v_k(self):
        agent, operator, batch, sample = self.agent, self.operator, self.batch, self.sample
        self.assertNotIn("critics", operator)
        rates = operator_anchor_rates(operator, agent.controller, batch, sample)
        expected = hand_model_rates(agent, operator, batch["states"], sample)
        self.assertEqual(tuple(rates.shape), (8, 3, 1))
        self.assertFalse(rates.requires_grad)
        torch.testing.assert_close(rates, expected, atol=1e-5, rtol=1e-4)

    def test_target_is_v_k_plus_h_times_the_averaged_rate_for_both_steps(self):
        agent, operator, batch, sample = self.agent, self.operator, self.batch, self.sample
        states = batch["states"]
        rates = hand_model_rates(agent, operator, states, sample)
        with torch.no_grad():
            values = agent.value(states)
            anchor = operator["target_value"](states)
        steps = {
            "resolvent_step": torch.full_like(anchor, agent.resolvent_step),
            "interval": batch["durations"].reshape(-1, 1),
        }
        self.assertEqual(agent.resolvent_step, 1.0)
        for size, h in steps.items():
            with self.subTest(explicit_step_size=size):
                agent.explicit_step_size = size
                given = agent._explicit_steps(batch)
                if size == "resolvent_step":
                    self.assertEqual(given, agent.resolvent_step)
                else:
                    torch.testing.assert_close(given, h.unsqueeze(1))
                residuals = agent._ablation_residuals(batch, operator, sample)
                target = anchor + h * rates.mean(dim=1)
                torch.testing.assert_close(
                    residuals.detach().mean(dim=1), values - target, atol=1e-5, rtol=1e-4
                )
                per_draw = values.unsqueeze(1) - (anchor.unsqueeze(1) + h.unsqueeze(1) * rates)
                torch.testing.assert_close(residuals.detach(), per_draw, atol=1e-5, rtol=1e-4)

    def test_target_does_not_move_with_the_candidate(self):
        """The operator is read at V_k: moving V moves every residual by V's change."""
        agent, operator, batch, sample = self.agent, self.operator, self.batch, self.sample
        states = batch["states"]
        before = agent._ablation_residuals(batch, operator, sample).detach()
        with torch.no_grad():
            old = agent.value(states)
            agent.value.network[-1].bias.add_(0.25)
            agent.value.network[0].weight.mul_(1.1)
            change = agent.value(states) - old
        after = agent._ablation_residuals(batch, operator, sample).detach()
        torch.testing.assert_close(after - before, change.unsqueeze(1).expand_as(before),
                                   atol=1e-5, rtol=1e-4)

    def test_flagged_rows_take_the_observed_interval_rate(self):
        agent, operator, batch, sample = self.agent, self.operator, self.batch, self.sample
        flags = torch.zeros(8, dtype=torch.bool)
        flags[[1, 4]] = True
        rates = operator_anchor_rates(operator, agent.controller, batch, sample, flags)
        model = hand_model_rates(agent, operator, batch["states"], sample)
        torch.testing.assert_close(rates[~flags], model[~flags], atol=1e-5, rtol=1e-4)
        lam = agent.discount_rate
        with torch.no_grad():
            here = operator["target_value"](batch["states"])
            there = operator["target_value"](batch["next_states"])
        durations = batch["durations"].reshape(-1, 1)
        interval = (
            batch["rewards"] + (1.0 - batch["terminated"]) * torch.exp(-lam * durations) * there
            - here
        ) / durations
        charge = operator["temperature_per_second"] * (
            discount_weight(durations, discount_rate=lam) / durations
        ).unsqueeze(1) * sample["log_probabilities"]
        observed = interval.unsqueeze(1) - charge
        torch.testing.assert_close(rates[flags], observed[flags], atol=1e-2, rtol=1e-3)

    def test_explicit_step_residuals_take_a_float_or_a_per_row_step(self):
        agent, operator, batch = self.agent, self.operator, self.batch
        states = batch["states"]
        rates = torch.ones(8, 3, 1)
        with torch.no_grad():
            values = agent.value(states)
            anchor = operator["target_value"](states)
        scalar = explicit_step_residuals(agent.value, operator, states, rates, 0.5, create_graph=False)
        torch.testing.assert_close(scalar, (values - anchor - 0.5).unsqueeze(1).expand(-1, 3, -1))
        rows = torch.linspace(0.1, 0.8, 8).reshape(-1, 1, 1)
        per_row = explicit_step_residuals(agent.value, operator, states, rates, rows, create_graph=False)
        torch.testing.assert_close(per_row, (values.unsqueeze(1) - anchor.unsqueeze(1) - rows).expand(-1, 3, -1))

    def test_value_step_moves_only_the_value(self):
        agent, operator = self.agent, self.operator
        modules = (agent.online_plant, agent.admitted_plant, agent.target_value, agent.actor,
                   agent.rate, operator["policy"], operator["plant"])
        for module in modules:
            module.zero_grad(set_to_none=True)
        before = snapshot(agent.value)
        metrics = agent.update_value(self.batch, operator)
        self.assertTrue(math.isfinite(metrics["value_loss"]))
        self.assertFalse(same(agent.value, before))
        for module in modules:
            self.assertTrue(all(p.grad is None for p in module.parameters()))


# ------------------------------------------------------------ arm B: the critic operator


class TestCriticOperator(unittest.TestCase):
    def setUp(self):
        self.agent = make_ablation(STEP_ARMS["implicit_critic"])
        collect(self.agent, 40)
        self.operator = self.agent.freeze_operator()
        self.batch = self.agent.replay.sample("train", 8)
        self.sample = frozen_draws(self.agent, self.operator, self.batch["states"])

    def _scores(self, states):
        decisions = self.sample["decisions"]
        repeated = states.unsqueeze(1).expand(-1, decisions.shape[1], -1)
        with torch.no_grad():
            return self.operator["critics"].minimum(repeated, decisions)

    def test_components(self):
        agent, operator = self.agent, self.operator
        self.assertIsNone(agent.rate)
        self.assertIsNone(agent.rate_optimizer)
        self.assertIsInstance(agent.critics, TwinQCritic)
        self.assertIsNone(agent.reward_net)
        self.assertIs(agent._network_modules()["critics"], agent.critics)
        self.assertIsNone(agent._network_modules()["rate"])
        self.assertIs(agent._optimizers()["critics"], agent.critic_optimizer)
        self.assertIsNot(operator["critics"], agent.critics)
        self.assertTrue(same(operator["critics"], snapshot(agent.critics)))
        self.assertFalse(any(p.requires_grad for p in operator["critics"].parameters()))
        self.assertAlmostEqual(operator["reference_interval"], 0.010, places=12)
        self.assertEqual(
            operator["reference_interval"], mean_decision_interval(agent.testbed["clock"])
        )
        self.assertEqual(agent.update_reward(self.batch), {})

    def test_averaged_rate_is_q_minus_v_over_the_reference_interval(self):
        operator, sample = self.operator, self.sample
        states = self.batch["states"]
        with torch.no_grad():
            values = self.agent.value(states)
        scores = self._scores(states)
        reference = operator["reference_interval"]
        alpha = operator["temperature_per_second"]
        rates = critic_soft_rates(operator, states, sample, values)
        expected = (scores - values.unsqueeze(1)) / reference - alpha * sample["log_probabilities"]
        torch.testing.assert_close(rates, expected)
        torch.testing.assert_close(
            rates.mean(dim=1),
            (scores.mean(dim=1) - values) / reference
            - alpha * sample["log_probabilities"].mean(dim=1),
            atol=1e-4,
            rtol=1e-5,
        )
        with torch.no_grad():
            anchor = operator["target_value"](states)
        torch.testing.assert_close(
            operator_anchor_rates(operator, self.agent.controller, self.batch, sample),
            critic_soft_rates(operator, states, sample, anchor),
        )

    def test_resolvent_residual_and_its_root(self):
        agent, operator, sample = self.agent, self.operator, self.sample
        states = self.batch["states"]
        residuals = agent._ablation_residuals(self.batch, operator, sample).detach().double()
        with torch.no_grad():
            values = agent.value(states).double()
            anchor = operator["target_value"](states).double()
        scores = self._scores(states).double()
        log_pi = sample["log_probabilities"].double()
        tau = agent.resolvent_step
        ratio = tau / operator["reference_interval"]
        alpha = float(operator["temperature_per_second"])
        expected = ((1.0 + ratio) * values - anchor).unsqueeze(1) - ratio * scores + tau * alpha * log_pi
        torch.testing.assert_close(residuals, expected, atol=1e-3, rtol=1e-4)
        root = (anchor + ratio * (scores.mean(dim=1) - operator["reference_interval"] * alpha
                                  * log_pi.mean(dim=1))) / (1.0 + ratio)
        torch.testing.assert_close(
            residuals.mean(dim=1), (1.0 + ratio) * (values - root), atol=1e-3, rtol=1e-4
        )

    def test_critic_target_is_the_explicit_generator_arms(self):
        agent, batch = self.agent, self.batch
        targets = generator_critic_target(
            agent.admitted_plant, agent.target_value, agent.controller, batch,
            discount_rate=agent.discount_rate,
        )
        explicit = types.SimpleNamespace(
            exclusion_active=False,
            end_state=False,
            continuation="generator",
            target_plant=agent.admitted_plant,
            target_value=agent.target_value,
            controller=agent.controller,
            discount_rate=agent.discount_rate,
        )
        self.assertTrue(torch.equal(targets, ExplicitCTPHSAC.build_critic_target(explicit, batch)))
        states, decisions = batch["states"], batch["decisions"]
        with torch.no_grad():
            torques = agent.controller.action(
                states, agent.controller.resolve_decision(states, decisions)
            )
            drift = agent.admitted_plant.drift(states, torques)
        derivatives = value_derivatives(
            agent.target_value, states, agent.admitted_plant.diffusion(states), create_graph=False
        )
        generated = (drift * derivatives["gradients"]).sum(-1, keepdim=True) + 0.5 * (
            derivatives["diffusion_traces"]
        )
        values = derivatives["values"]
        expected = batch["rewards"] + (1.0 - batch["terminated"]) * (
            values + batch["durations"] * (generated - agent.discount_rate * values)
        )
        torch.testing.assert_close(targets, expected.detach(), atol=1e-5, rtol=1e-4)

    def test_critic_steps_take_the_rate_heads_place(self):
        agent = self.agent
        others = {
            name: snapshot(module)
            for name, module in agent._network_modules().items()
            if module is not None and name != "critics"
        }
        before = snapshot(agent.critics)
        counters = agent.counters()
        with mock.patch.object(
            implicit_module, "generator_critic_target", wraps=generator_critic_target
        ) as target:
            metrics = agent.fit_rate_head(2)
        self.assertEqual(target.call_count, 2)
        self.assertIs(target.call_args.args[0], agent.admitted_plant)
        self.assertIs(target.call_args.args[1], agent.target_value)
        self.assertTrue(math.isfinite(metrics["critic_loss"]))
        self.assertFalse(same(agent.critics, before))
        for name, weights in others.items():
            self.assertTrue(same(agent._network_modules()[name], weights), name)
        after = agent.counters()
        self.assertEqual(after["critic_gradient_steps"], counters["critic_gradient_steps"] + 2)
        self.assertEqual(after["rate_gradient_steps"], counters["rate_gradient_steps"])

    def test_actor_reads_the_critic_with_the_interval_temperature(self):
        agent = self.agent
        batches = []
        draw = agent.replay.sample

        def recording(partition, size):
            batches.append(draw(partition, size))
            return batches[-1]

        before = snapshot(agent.critics)
        agent.critics.zero_grad(set_to_none=True)
        with mock.patch.object(agent.replay, "sample", recording), mock.patch.object(
            implicit_module, "explicit_actor_loss", wraps=implicit_module.explicit_actor_loss
        ) as critic_loss, mock.patch.object(
            implicit_module, "implicit_actor_loss", wraps=implicit_module.implicit_actor_loss
        ) as rate_loss:
            agent.update_actor(1)
        rate_loss.assert_not_called()
        critic_loss.assert_called_once()
        self.assertIs(critic_loss.call_args.args[0], agent.critics)
        expected = agent.alpha_c * discount_weight(
            batches[-1]["durations"], discount_rate=agent.discount_rate
        )
        torch.testing.assert_close(critic_loss.call_args.args[3], expected)
        self.assertTrue(same(agent.critics, before))
        self.assertTrue(all(p.grad is None for p in agent.critics.parameters()))

    def test_value_step_moves_only_the_value(self):
        agent, operator = self.agent, self.operator
        modules = (agent.critics, operator["critics"], agent.target_value, agent.actor,
                   agent.online_plant, agent.admitted_plant)
        for module in modules:
            module.zero_grad(set_to_none=True)
        before = snapshot(agent.value)
        metrics = agent.update_value(self.batch, operator)
        self.assertTrue(math.isfinite(metrics["value_loss"]))
        self.assertFalse(same(agent.value, before))
        for module in modules:
            self.assertTrue(all(p.grad is None for p in module.parameters()))


# --------------------------------------------------------------------- every arm learns


class TestArmsLearn(unittest.TestCase):
    def test_each_arm_learns_with_finite_losses_and_monitors(self):
        for name, choices in STEP_ARMS.items():
            # The arms' own decisions, then the impedance law (the code reads neither).
            for mode in ("torque", "feedback"):
                with self.subTest(arm=name, decision_mode=mode):
                    agent = make_ablation(choices, decision_mode=mode)
                    self.assertEqual(agent.decision_mode, mode)
                    agent.learn(90)
                    self.assertTrue(agent.step_monitors)
                    for p in all_parameters(agent):
                        self.assertTrue(torch.isfinite(p).all())
                    critic = choices.get("decision_head") == "critic"
                    head = "critic_loss" if critic else "rate_loss"
                    completed = [r for r in agent.solver_records if r["status"] == "completed"]
                    self.assertGreater(len(completed), 3)
                    keys = set(SolverMetrics.__required_keys__) | set(VALUE_MONITORS) | {head}
                    for record in agent.solver_records:
                        self.assertEqual(set(record), keys)
                    for record in completed:
                        for key in (*VALUE_MONITORS, head):
                            self.assertTrue(math.isfinite(record[key]), key)
                        self.assertGreaterEqual(record["residual_mean_square"], 0.0)
                    counters = agent.counters()
                    self.assertGreater(counters["value_gradient_steps"], 0)
                    self.assertGreater(counters["actor_gradient_steps"], 0)
                    if critic:
                        self.assertGreater(counters["critic_gradient_steps"], 0)
                        self.assertEqual(counters["rate_gradient_steps"], 0)
                    else:
                        self.assertGreater(counters["rate_gradient_steps"], 0)
                        self.assertEqual(counters["critic_gradient_steps"], 0)
                    # Only the critic operator (implicit_critic) learns no reward rate.
                    if choices.get("value_operator") == "critic":
                        self.assertEqual(counters["reward_gradient_steps"], 0)
                    else:
                        self.assertGreater(counters["reward_gradient_steps"], 0)

    def test_critic_head_keeps_the_resolvent_step_and_the_learned_reward(self):
        """implicit_critic_head: the resolvent residual of implicit_learned_admissions with
        the learned reward copy, no Delta_ref; the critic only in the rate head's place."""
        agent = make_ablation(STEP_ARMS["implicit_critic_head"])
        self.assertTrue(agent.default_value_update and agent.step_monitors)
        self.assertIsNone(agent.rate)
        self.assertIsInstance(agent.critics, TwinQCritic)
        self.assertIsNotNone(agent.reward_net)
        self.assertIsNone(agent.critic_reference_interval)
        collect(agent, 40)
        operator = agent.freeze_operator()
        self.assertNotIn("critics", operator)
        self.assertNotIn("reference_interval", operator)
        self.assertIsNotNone(agent._reward_frozen)
        batch = agent.replay.sample("train", 8)
        with mock.patch.object(
            implicit_module, "resolvent_residuals", wraps=implicit_module.resolvent_residuals
        ) as model, mock.patch.object(
            implicit_module, "critic_resolvent_residuals",
            wraps=implicit_module.critic_resolvent_residuals,
        ) as critic, mock.patch.object(
            implicit_module, "explicit_step_residuals",
            wraps=implicit_module.explicit_step_residuals,
        ) as explicit:
            metrics = agent.update_value(batch, operator)
        model.assert_called_once()
        self.assertIs(model.call_args.args[0], agent.value)
        critic.assert_not_called()
        explicit.assert_not_called()
        self.assertTrue(math.isfinite(metrics["value_loss"]))
        self.assertIn("reward_fit_loss", metrics)
        # The critic takes the rate head's steps on explicit_generator's target.
        with mock.patch.object(
            implicit_module, "generator_critic_target", wraps=generator_critic_target
        ) as target:
            fitted = agent.fit_rate_head(2)
        self.assertEqual(target.call_count, 2)
        self.assertIs(target.call_args.args[0], agent.admitted_plant)
        self.assertIs(target.call_args.args[1], agent.target_value)
        self.assertTrue(math.isfinite(fitted["critic_loss"]))
        with mock.patch.object(
            implicit_module, "explicit_actor_loss", wraps=implicit_module.explicit_actor_loss
        ) as actor:
            agent.update_actor(1)
        actor.assert_called_once()
        self.assertIs(actor.call_args.args[0], agent.critics)

    def test_checkpoint_round_trip(self):
        for name, choices in STEP_ARMS.items():
            with self.subTest(arm=name):
                reference = make_ablation(choices)
                reference.learn(70)
                state = reference.state_dict()
                self.assertEqual(
                    "critics" in state["networks"], choices.get("decision_head") == "critic"
                )
                self.assertEqual(
                    "rate" in state["networks"], choices.get("decision_head") != "critic"
                )
                self.assertEqual(
                    "reward" in state["networks"], choices.get("value_operator") != "critic"
                )
                restored = make_ablation(choices)
                restored.load_state_dict(state)
                global_state = torch.get_rng_state()
                restored.learn(110)
                torch.set_rng_state(global_state)
                reference.learn(110)
                for p, q in zip(all_parameters(reference), all_parameters(restored)):
                    self.assertTrue(torch.equal(p, q))


class TestExactRestart(ExactRestartMixin, unittest.TestCase):
    """Each new arm, stopped mid-run and resumed in a disturbed fresh agent, ends bitwise
    where the uninterrupted run ends (tests/test_ph_resume.py's protocol and checks)."""

    def test_every_step_ablation_arm(self):
        for name in STEP_ARMS:
            with self.subTest(arm=name):
                reference, agent, _, _ = self.check_exact_restart("pendulum-swingup", name, 100, 70)
                self.assertEqual(agent.decision_mode, "torque")
                self.assertEqual(reference["decision_mode"], "torque")
                completed = [r for r in reference["solver"] if r["status"] == "completed"]
                self.assertGreater(len(completed), 0)
                for record in completed:
                    for key in VALUE_MONITORS:
                        self.assertTrue(math.isfinite(record[key]), key)
                self.assertEqual(
                    agent.decision_head == "critic",
                    name in ("implicit_critic", "implicit_critic_head"),
                )
                self.assertEqual(agent.reward_net is None, name == "implicit_critic")


if __name__ == "__main__":
    unittest.main()

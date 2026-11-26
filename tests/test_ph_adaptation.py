"""The adaptation experiment: changed robots, trainable plant parts and warm starts.

docs/ct_ph_sac_layout.md ("Adaptation experiment"). A physical change (testbed_settings
physical_change, common.ph_dmc._common.with_physical_change) must change the MuJoCo model
and the oracle as declared and leave the reward and every config without it untouched;
an implicit arm with trainable_plant_parts must move only those parts of its plant; a warm
start (protocol warm_start, evaluations/ph_sac/adaptation.py) must reproduce the source
run's networks and temperature with an empty replay, fresh counters and optimizers, no
warm-up and no warm fit, so the evaluation at 0 scores the source policy on the changed
robot. Tiny configs of the acrobot-swingup and cartpole-swingup table rows (mode top) with
1 s episodes; each test takes a few seconds.
"""

from __future__ import annotations

import contextlib
import copy
import math
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import mujoco
import numpy as np
import torch

from benchmarks import ph_hyperparams as table
from benchmarks import run_ph_sac
from common.ph_dmc._common import (
    PHYSICAL_CHANGE_KINDS,
    check_physical_change,
    nominal_testbed,
    physical_plant_parameters,
)
from common.ph_dmc.cartpole import POLE_RADIUS, capsule_transverse_inertia
from environment.ph_sac import tasks
from environment.ph_sac.dmc import acrobot as dmc_acrobot
from environment.ph_sac.dmc import cartpole as dmc_cartpole
from evaluations.ph_sac import adaptation as AD
from evaluations.ph_sac import protocol as P
from evaluations.ph_sac.reporting import load_run_records

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "benchmarks" / "configs" / "ph_sac" / "adaptation"
EPISODE_SECONDS = 1.0
BATCH = 16
LENGTH = {"kind": "length", "scale": 1.25}
DAMPING = {"kind": "damping", "scale": 1.5}
BUILDERS = {
    "acrobot-swingup": dmc_acrobot.build_acrobot_model,
    "cartpole-swingup": dmc_cartpole.build_cartpole_model,
}
MODEL_ARRAYS = (
    "body_mass", "body_inertia", "body_ipos", "body_pos", "dof_damping", "geom_size",
    "site_pos", "jnt_range", "actuator_gear",
)


def row_settings(env_id, change=None):
    settings = copy.deepcopy(table.find_ph_entry(env_id, "top").config["testbed_settings"])
    if change is not None:
        settings["physical_change"] = dict(change)
    return settings


def row_testbed(env_id, change=None):
    """The row's TestbedConfig through the protocol loader, with the change when given."""
    name = table.find_ph_entry(env_id, "top").config["testbed"]
    return run_ph_sac._build_testbed(name, row_settings(env_id, change))


def model_arrays(model):
    return {name: np.array(getattr(model, name)) for name in MODEL_ARRAYS}


def body(model, name):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)


@contextlib.contextmanager
def float64_default():
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        yield
    finally:
        torch.set_default_dtype(previous)


def relative_error(left, right):
    left, right = np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
    return float(np.max(np.abs(left - right) / np.maximum(np.abs(right), 1.0)))


def random_states(env_id, rng, count):
    """Acrobot: any angles and rates; cart-pole: the cart well inside the rail."""
    if env_id == "acrobot-swingup":
        return np.concatenate(
            (rng.uniform(-math.pi, math.pi, (count, 2)), rng.uniform(-6.0, 6.0, (count, 2))), axis=1
        )
    return np.stack(
        (
            rng.uniform(-1.0, 1.0, count),
            rng.uniform(-math.pi, math.pi, count),
            rng.uniform(-2.0, 2.0, count),
            rng.uniform(-6.0, 6.0, count),
        ),
        axis=1,
    )


def tiny_run(arm_name, output, budget, *, seed=0, change=None, parts=None, warm=None,
             env_id="acrobot-swingup", protocol=None):
    """A tiny protocol of a table row (mode top), one arm and seed; evaluations at 0 and at
    the end only. change, parts and warm are the adaptation keys (absent when None);
    protocol: extra protocol keys when given."""
    raw = copy.deepcopy(table.find_ph_entry(env_id, "top").config)
    if change is not None:
        raw["testbed_settings"]["physical_change"] = dict(change)
    raw["network"].update(hidden_sizes=[16, 16], plant_hidden_sizes=[8, 8])
    training = raw["training"]
    training.update(batch_size=BATCH, replay_capacity=2000, warmup_episodes=1)
    for key in ("plant_step_dense_until", "plant_step_every"):
        training.pop(key, None)
    raw["explicit"].update(rollout_paths=2)
    implicit = raw["implicit"]
    implicit.update(reward_warm_steps=5, plant_warm_steps=5, temperature_every_rows=4)
    if parts is not None:
        implicit["trainable_plant_parts"] = list(parts)
    implicit["solver"].update(inner_steps=8, policy_draws=2)
    implicit["rate"].update(rollout_paths=2, model_points=8, rate_steps=2, actor_steps=2)
    implicit["admission"].update(
        initial_rows=8, initial_paths=2, pilot_batches=2, real_row_budget=64,
        simulated_path_budget=4096, efolds=0.5,
    )
    raw["protocol"].update(
        interaction_budget=budget,
        training_seeds=[seed],
        evaluation_reset_seeds=[3],
        evaluation_every_interactions=10 * budget,
        output_directory=str(output),
    )
    if warm is not None:
        raw["protocol"]["warm_start"] = warm
    if protocol is not None:
        raw["protocol"].update(protocol)
    proto = run_ph_sac.protocol_from_config(raw, f"tiny adaptation {env_id}")
    for config in (
        proto["matched"]["testbed_config"],
        proto["explicit_config"]["testbed"],
        proto["implicit_config"]["testbed"],
    ):
        config["episode_duration"] = EPISODE_SECONDS
    proto["arms"] = [arm for arm in P.all_arms() if arm["name"] == arm_name]
    P.validate_matched_quantities(proto)
    (run,) = P.build_run_matrix(proto)
    return run


def build_agent(run):
    config = run["protocol"]["matched"]["testbed_config"]
    env = tasks.make_env(config, decision_mode=run["arm"]["decision_mode"], seed=run["training_seed"])
    return P.make_training_agent(run, env)


def collect(agent, rows):
    """Rows of uniform decisions into the agent's replay (a PH agent's own collection)."""
    for _ in range(rows):
        agent._collect(lambda state: agent._uniform_decision())


def perturb(modules, seed=7, scale=0.05):
    """Move every parameter away from its initialization (a private generator)."""
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for module in modules:
            for parameter in module.parameters():
                noise = torch.randn(parameter.shape, generator=generator, dtype=parameter.dtype)
                parameter.add_(scale * noise)


def same_state(left, right):
    left, right = left.state_dict(), right.state_dict()
    return left.keys() == right.keys() and all(torch.equal(left[k], right[k]) for k in left)


def part_values(plant):
    return {
        part: [parameter.detach().clone() for parameter in parameters]
        for part, parameters in plant.part_parameters().items()
    }


def same_values(left, right):
    return all(torch.equal(a, b) for a, b in zip(left, right))


def run_captured(run):
    """run_seed, also returning the agent it built."""
    agents = []
    build = P.make_training_agent

    def capture(spec, env):
        agent = build(spec, env)
        agents.append(agent)
        return agent

    with mock.patch.object(P, "make_training_agent", capture):
        result = P.run_seed(run)
    return result, agents[-1]


def first_returns(result):
    return [episode["discounted_return"] for episode in result["evaluations"][0]["episodes"]]


class TestPhysicalChange(unittest.TestCase):
    def test_default_and_unit_scale_leave_the_model_unchanged(self):
        for env_id, build in BUILDERS.items():
            nominal = row_testbed(env_id)
            self.assertNotIn("physical_change", nominal)
            self.assertNotIn("physical_plant_parameters", nominal)
            self.assertIs(nominal_testbed(nominal), nominal)
            self.assertIs(physical_plant_parameters(nominal), nominal["plant_parameters"])
            reference = model_arrays(build(nominal))
            for kind in PHYSICAL_CHANGE_KINDS:
                with self.subTest(env_id=env_id, kind=kind):
                    unit = row_testbed(env_id, {"kind": kind, "scale": 1.0})
                    self.assertEqual(unit["physical_plant_parameters"], unit["plant_parameters"])
                    arrays = model_arrays(build(unit))
                    for name, values in reference.items():
                        np.testing.assert_array_equal(arrays[name], values, err_msg=name)

    def test_acrobot_length(self):
        nominal, changed = row_testbed("acrobot-swingup"), row_testbed("acrobot-swingup", LENGTH)
        self.assertEqual(changed["plant_parameters"], nominal["plant_parameters"])
        self.assertEqual(changed["physical_plant_parameters"]["l2"], 1.25)
        before = dmc_acrobot.build_acrobot_model(nominal)
        after = dmc_acrobot.build_acrobot_model(changed)
        lower, upper = body(after, "lower_arm"), body(after, "upper_arm")
        self.assertAlmostEqual(float(after.body_ipos[lower][2]), 0.625, places=12)
        self.assertEqual(float(after.body_mass[lower]), float(before.body_mass[lower]))
        radius = float(nominal["plant_parameters"]["radius2"])
        expected = capsule_transverse_inertia(float(after.body_mass[lower]), radius, 1.25)
        self.assertLess(abs(float(np.max(after.body_inertia[lower])) - expected), 1e-12)
        self.assertGreater(np.max(after.body_inertia[lower]), np.max(before.body_inertia[lower]))
        tip = mujoco.mj_name2id(after, mujoco.mjtObj.mjOBJ_SITE, "tip")
        self.assertAlmostEqual(float(after.site_pos[tip][2]), 1.25, places=12)
        np.testing.assert_array_equal(after.dof_damping, before.dof_damping)
        for name in ("body_mass", "body_inertia", "body_ipos", "body_pos"):
            np.testing.assert_array_equal(getattr(after, name)[upper], getattr(before, name)[upper])
        self.assertEqual(
            AD.physical_change_record(changed),
            {"kind": "length", "scale": 1.25, "changed": {"l2": {"nominal": 1.0, "physical": 1.25}}},
        )

    def test_acrobot_damping(self):
        nominal, changed = row_testbed("acrobot-swingup"), row_testbed("acrobot-swingup", DAMPING)
        before = dmc_acrobot.build_acrobot_model(nominal)
        after = dmc_acrobot.build_acrobot_model(changed)
        np.testing.assert_allclose(after.dof_damping, 1.5 * before.dof_damping, rtol=1e-15)
        self.assertGreater(float(np.min(after.dof_damping)), 0.0)
        for name in ("body_mass", "body_inertia", "body_ipos", "geom_size", "site_pos"):
            np.testing.assert_array_equal(getattr(after, name), getattr(before, name))
        self.assertEqual(
            sorted(AD.physical_change_record(changed)["changed"]),
            ["elbow_damping", "shoulder_damping"],
        )

    def test_capsule_inertia_is_mujocos(self):
        # The compiled value of dm_control's pole, which the cart-pole rows carry.
        self.assertLess(
            abs(capsule_transverse_inertia(0.1, POLE_RADIUS, 1.0) - 0.009424592767295598), 1e-16
        )
        for length in (0.8, 1.1, 1.25, 1.5):
            xml = (
                '<mujoco><worldbody><body name="pole"><geom type="capsule" '
                f'fromto="0 0 0 0 0 {length}" size="{POLE_RADIUS}" mass="0.1"/>'
                "</body></worldbody></mujoco>"
            )
            compiled = mujoco.MjModel.from_xml_string(xml)
            expected = float(np.max(compiled.body_inertia[body(compiled, "pole")]))
            self.assertLess(
                abs(capsule_transverse_inertia(0.1, POLE_RADIUS, length) - expected), 1e-12 * expected
            )

    def test_cartpole_length_and_damping(self):
        nominal = row_testbed("cartpole-swingup")
        before = dmc_cartpole.build_cartpole_model(nominal)
        changed = row_testbed("cartpole-swingup", LENGTH)
        after = dmc_cartpole.build_cartpole_model(changed)
        pole = body(after, "pole")
        xml = (
            '<mujoco><worldbody><body name="pole"><geom type="capsule" '
            f'fromto="0 0 0 0 0 1.25" size="{POLE_RADIUS}" mass="0.1"/>'
            "</body></worldbody></mujoco>"
        )
        compiled = mujoco.MjModel.from_xml_string(xml)
        expected = float(np.max(compiled.body_inertia[body(compiled, "pole")]))
        self.assertLess(abs(float(np.max(after.body_inertia[pole])) - expected), 1e-12 * expected)
        self.assertAlmostEqual(float(after.body_ipos[pole][2]), 0.625, places=12)
        self.assertEqual(float(after.body_mass[pole]), float(before.body_mass[pole]))
        np.testing.assert_array_equal(after.dof_damping, before.dof_damping)
        self.assertEqual(
            sorted(AD.physical_change_record(changed)["changed"]),
            ["pole_com", "pole_inertia", "pole_length"],
        )
        # The state spec (the limit port's start) stays the nominal model's.
        self.assertEqual(
            changed["state_spec"]["limit_port_init"], nominal["state_spec"]["limit_port_init"]
        )
        damped = dmc_cartpole.build_cartpole_model(row_testbed("cartpole-swingup", DAMPING))
        np.testing.assert_allclose(damped.dof_damping, 1.5 * before.dof_damping, rtol=1e-15)
        for name in ("body_mass", "body_inertia", "body_ipos"):
            np.testing.assert_array_equal(getattr(damped, name), getattr(before, name))

    def test_rewards_stay_the_nominal_functions(self):
        rng = np.random.default_rng(0)
        for env_id in BUILDERS:
            for change in (LENGTH, DAMPING):
                with self.subTest(env_id=env_id, change=change):
                    nominal, changed = row_testbed(env_id), row_testbed(env_id, change)
                    nominal_env = tasks.make_env(nominal, seed=0)
                    changed_env = tasks.make_env(changed, seed=0)
                    states = random_states(env_id, rng, 16)
                    torques = rng.uniform(-1.0, 1.0, (16, 1))
                    for state, torque in zip(states, torques):
                        self.assertEqual(
                            changed_env.reward_rate(state, torque),
                            nominal_env.reward_rate(state, torque),
                        )
                    if env_id == "acrobot-swingup":
                        for state in states:
                            self.assertEqual(
                                changed_env.lyapunov_ratio(state), nominal_env.lyapunov_ratio(state)
                            )
                    states_t, torques_t = torch.as_tensor(states), torch.as_tensor(torques)
                    self.assertTrue(
                        torch.equal(
                            tasks.reward_rate_tensor(states_t, torques_t, changed),
                            tasks.reward_rate_tensor(states_t, torques_t, nominal),
                        )
                    )

    def test_oracle_follows_the_changed_robot(self):
        rng = np.random.default_rng(1)
        for env_id in BUILDERS:
            for change in (LENGTH, DAMPING):
                with self.subTest(env_id=env_id, change=change):
                    config = row_testbed(env_id, change)
                    env = tasks.make_env(config, seed=0)
                    nominal_env = tasks.make_env(row_testbed(env_id), seed=0)
                    states = random_states(env_id, rng, 32)
                    torques = rng.uniform(-1.0, 1.0, (32, 1))
                    with float64_default():
                        drift, _ = tasks.oracle_callbacks(config)
                        tensor = drift(torch.as_tensor(states), torch.as_tensor(torques)).numpy()
                    moved = 0.0
                    for state, torque, row in zip(states, torques, tensor):
                        truth = env.mujoco_drift(state, torque)
                        self.assertLess(relative_error(env.oracle_drift(state, torque), truth), 1e-8)
                        self.assertLess(relative_error(row, truth), 1e-8)
                        moved = max(moved, relative_error(nominal_env.mujoco_drift(state, torque), truth))
                    # The cart-pole's damping (5e-4 and 2e-6) moves the drift by far less.
                    floor = 1e-9 if (env_id, change["kind"]) == ("cartpole-swingup", "damping") else 1e-3
                    self.assertGreater(moved, floor)

    def test_invalid_changes_raise(self):
        for change in (
            {"kind": "mass", "scale": 1.1},
            {"kind": "length", "scale": 0.0},
            {"kind": "length", "scale": -1.0},
            {"kind": "length", "scale": float("nan")},
            {"kind": "length", "scale": True},
            {"kind": "length"},
            {"kind": "length", "scale": 1.1, "joint": "elbow"},
        ):
            with self.subTest(change=change), self.assertRaises(ValueError):
                check_physical_change(change)
        sparse = copy.deepcopy(
            table.find_ph_entry("acrobot-swingup_sparse", "top").config["testbed_settings"]
        )
        sparse["physical_change"] = dict(LENGTH)
        with self.assertRaisesRegex(ValueError, "unknown settings"):
            run_ph_sac._build_testbed("acrobot_swingup_sparse", sparse)


class TestTrainablePlantParts(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.root = Path(self.folder.name)

    def tearDown(self):
        self.folder.cleanup()

    def test_default_trains_every_parameter(self):
        agent = build_agent(tiny_run("implicit_learned_admissions", self.root, 40))
        plant = agent.online_plant
        self.assertIsNone(agent.trainable_plant_parts)
        self.assertEqual(
            [id(p) for p in agent.plant_optimizer.param_groups[0]["params"]],
            [id(p) for p in plant.parameters()],
        )
        self.assertTrue(all(p.requires_grad for p in plant.parameters()))
        # A drift-only plant without a limit port.
        self.assertEqual(set(plant.part_parameters()), {"potential", "mass", "damping", "input"})

    def test_frozen_parts_keep_their_values(self):
        # The implicit paper arm (torque decisions) and the impedance-law arm: the plant's
        # parts do not depend on the decision mode.
        for arm in ("implicit_torque", "implicit_learned_admissions"):
            for parts in (("damping",), ("mass", "potential")):
                with self.subTest(arm=arm, parts=parts):
                    self.check_frozen_parts(arm, parts)

    def check_frozen_parts(self, arm, parts):
        agent = build_agent(tiny_run(arm, self.root / arm / "_".join(parts), 40, parts=parts))
        self.assertEqual(
            agent.decision_mode, "torque" if arm == "implicit_torque" else "feedback"
        )
        plant = agent.online_plant
        self.assertEqual(agent.trainable_plant_parts, parts)
        trainable = {id(p) for part in parts for p in plant.part_parameters()[part]}
        self.assertEqual(
            {id(p) for p in agent.plant_optimizer.param_groups[0]["params"]}, trainable
        )
        self.assertEqual({id(p) for p in plant.parameters() if p.requires_grad}, trainable)
        collect(agent, 40)
        before = part_values(plant)
        for _ in range(3):
            agent.update_online_plant(agent.replay.sample("train", BATCH))
        after = part_values(plant)
        for part in before:
            self.assertEqual(same_values(before[part], after[part]), part not in parts, part)
        self.assertLessEqual({id(p) for p in agent.plant_optimizer.state}, trainable)
        self.assertEqual(agent.counters()["plant_gradient_steps"], 3)

    def test_unknown_or_absent_parts_raise(self):
        with self.assertRaises(ValueError):
            tiny_run("implicit_learned_admissions", self.root, 40, parts=("mass", "wheels"))
        with self.assertRaises(ValueError):
            tiny_run("implicit_learned_admissions", self.root, 40, parts=())
        # The acrobot's plant is drift-only: no diffusion part.
        run = tiny_run("implicit_learned_admissions", self.root, 40, parts=("diffusion",))
        with self.assertRaisesRegex(ValueError, "plant's parts"):
            build_agent(run)


class TestImplicitWarmStart(unittest.TestCase):
    SOURCE_ROWS = 30

    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.root = Path(self.folder.name)

    def tearDown(self):
        self.folder.cleanup()

    def source(self):
        """A nominal-robot implicit agent with rows, moved networks and temperature, saved."""
        agent = build_agent(tiny_run("implicit_learned_admissions", self.root / "source", 40))
        collect(agent, self.SOURCE_ROWS)
        perturb(
            [agent.actor, agent.rate, agent.value, agent.target_value, agent.online_plant,
             agent.admitted_plant, agent.reward_net]
        )
        with torch.no_grad():
            agent.log_temperature.fill_(math.log(0.037))
        path = self.root / "source.pt"
        agent.save(path)
        return agent, path

    def warm(self, path, **extra):
        return {"checkpoints": {"implicit_learned_admissions": str(path)}, **extra}

    def test_warm_start_reproduces_the_source(self):
        source, path = self.source()
        run = tiny_run(
            "implicit_learned_admissions", self.root / "target", 40, seed=1, change=LENGTH,
            warm=self.warm(path, require_interactions=self.SOURCE_ROWS),
        )
        agent = build_agent(run)
        record = AD.apply_warm_start(agent, run)
        self.assertEqual(record["checkpoint"], str(path))
        self.assertEqual(record["source_interactions"], self.SOURCE_ROWS)
        self.assertEqual(
            record["networks"],
            ["actor", "rate", "value", "target_value", "online_plant", "admitted_plant", "reward"],
        )
        for name in record["networks"]:
            self.assertTrue(same_state(getattr(agent, AGENT_ATTRIBUTES[name]),
                                       getattr(source, AGENT_ATTRIBUTES[name])), name)
        self.assertEqual(float(agent.alpha_c), float(source.alpha_c))
        self.assertAlmostEqual(record["alpha_c"], 0.037, places=6)
        # Empty replay, fresh counters, optimizers and schedules; no warm-up, no warm fit.
        self.assertEqual(agent.interactions, 0)
        for partition in ("train", "gate"):
            self.assertEqual(agent.replay.partition_size(partition), 0)
        self.assertTrue(agent.warm_fit_done)
        self.assertEqual(agent._warmup_episodes_done, agent.training["warmup_episodes"])
        self.assertIsNone(agent._reward_frozen)
        for name, optimizer in agent._optimizers().items():
            if optimizer is not None:
                self.assertEqual(len(optimizer.state), 0, name)
        self.assertEqual((agent.outer_iteration, agent.inner_steps, agent.outer_in_cycle), (0, 0, 0))
        # The policy's decisions and the value are the source's.
        states = random_states("acrobot-swingup", np.random.default_rng(2), 8)
        for state in states:
            np.testing.assert_array_equal(agent.predict(state), source.predict(state))
        with torch.no_grad():
            points = torch.as_tensor(states, dtype=torch.get_default_dtype())
            self.assertTrue(torch.equal(agent.value(points), source.value(points)))

    def test_required_interactions_guard(self):
        _, path = self.source()
        run = tiny_run(
            "implicit_learned_admissions", self.root / "target", 40, seed=1,
            warm=self.warm(path, require_interactions=500000),
        )
        agent = build_agent(run)
        actor = copy.deepcopy(agent.actor.state_dict())
        with self.assertRaisesRegex(ValueError, "required"):
            AD.apply_warm_start(agent, run)
        self.assertTrue(all(torch.equal(actor[k], v) for k, v in agent.actor.state_dict().items()))

    def test_run_starts_from_the_source_and_records_it(self):
        source, path = self.source()
        parts = ("mass", "potential")
        run = tiny_run(
            "implicit_learned_admissions", self.root / "target", 40, seed=1, change=LENGTH,
            parts=parts, warm=self.warm(path),
        )
        result, agent = run_captured(run)
        config = run["protocol"]["matched"]["testbed_config"]
        expected = P.evaluate_fixed_seeds(
            source.predict,
            lambda seed: tasks.make_env(config, seed=seed),
            [3],
            discount_rate=float(config["discount_rate"]),
            deterministic=True,
        )
        self.assertEqual(result["evaluations"][0]["interactions"], 0)
        self.assertEqual(first_returns(result), [e["discounted_return"] for e in expected])
        counters = result["counters"]
        self.assertEqual(counters["interactions"], 40)
        self.assertGreater(counters["value_gradient_steps"], 0)
        # No warm fit: one plant and one reward step per value step, none before.
        self.assertEqual(counters["plant_gradient_steps"], counters["value_gradient_steps"])
        self.assertEqual(counters["reward_gradient_steps"], counters["value_gradient_steps"])
        # The frozen parts are still the source's; the trainable ones moved.
        now, then = part_values(agent.online_plant), part_values(source.online_plant)
        for part in now:
            self.assertEqual(same_values(now[part], then[part]), part not in parts, part)
        # The run record.
        self.assertEqual(result["warm_start"]["checkpoint"], str(path))
        self.assertEqual(result["warm_start"]["source_interactions"], self.SOURCE_ROWS)
        self.assertEqual(result["physical_change"]["changed"], {"l2": {"nominal": 1.0, "physical": 1.25}})
        self.assertEqual(result["trainable_plant_parts"], list(parts))
        (on_disk,) = load_run_records([str(P.run_output_directory(run) / "run_record.json")])
        self.assertEqual(on_disk["warm_start"]["source_interactions"], self.SOURCE_ROWS)
        self.assertEqual(on_disk["trainable_plant_parts"], list(parts))

    def test_runs_without_the_keys_record_nothing_new(self):
        run = tiny_run("implicit_learned_admissions", self.root / "plain", 40)
        self.assertNotIn("warm_start", run["protocol"])
        self.assertNotIn("trainable_plant_parts", run["protocol"]["implicit_config"])
        self.assertNotIn("physical_change", run["protocol"]["matched"]["testbed_config"])
        self.assertEqual(AD.adaptation_record(run, None), {})

    def test_warm_start_refuses_another_reward_observation(self):
        """A source run under the other reward_observation (its reward network fitted to the
        other reward datum) is refused before anything is loaded; the same datum loads."""
        end_state = {"reward_observation": "end_state"}
        _, plain_path = self.source()
        keyed_source = build_agent(
            tiny_run("implicit_learned_admissions", self.root / "keyed", 40, protocol=end_state)
        )
        self.assertEqual(keyed_source.reward_observation, "end_state")
        collect(keyed_source, self.SOURCE_ROWS)
        perturb([keyed_source.actor, keyed_source.reward_net])
        keyed_path = self.root / "keyed.pt"
        keyed_source.save(keyed_path)
        for path, keys in ((plain_path, end_state), (keyed_path, None)):
            with self.subTest(source=path.name):
                run = tiny_run(
                    "implicit_learned_admissions", self.root / f"target_{path.stem}", 40,
                    seed=1, warm=self.warm(path), protocol=keys,
                )
                agent = build_agent(run)
                actor = copy.deepcopy(agent.actor.state_dict())
                with self.assertRaisesRegex(ValueError, "reward_observation"):
                    AD.apply_warm_start(agent, run)
                self.assertTrue(
                    all(torch.equal(actor[k], v) for k, v in agent.actor.state_dict().items())
                )
                self.assertEqual(agent._warmup_skipped, False)
        run = tiny_run(
            "implicit_learned_admissions", self.root / "target_same", 40, seed=1,
            warm=self.warm(keyed_path), protocol=end_state,
        )
        agent = build_agent(run)
        record = AD.apply_warm_start(agent, run)
        self.assertIn("reward", record["networks"])
        self.assertTrue(same_state(agent.reward_net, keyed_source.reward_net))
        self.assertTrue(same_state(agent.actor, keyed_source.actor))


# The agent attribute of each network a warm start loads.
AGENT_ATTRIBUTES = {
    "actor": "actor",
    "rate": "rate",
    "value": "value",
    "target_value": "target_value",
    "online_plant": "online_plant",
    "admitted_plant": "admitted_plant",
    "reward": "reward_net",
}


class TestCTSACWarmStart(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.root = Path(self.folder.name)

    def tearDown(self):
        self.folder.cleanup()

    def source(self):
        """A CT-SAC baseline with moved actor, critics, targets and temperature, saved."""
        agent = build_agent(tiny_run("ct_sac_torque", self.root / "source", 40))
        model = agent.algorithm.model
        perturb([model.actor, model.q_nets, model.q_target_nets])
        with torch.no_grad():
            agent.algorithm.log_alpha.fill_(math.log(0.0123))
        agent.algorithm.alpha = float(agent.algorithm.log_alpha.detach().exp().item())
        path = self.root / "ct_sac.pt"
        agent.save(path)
        return agent, path

    def target(self, path, output):
        warm = {"checkpoints": {"ct_sac_torque": str(path)}}
        return tiny_run("ct_sac_torque", output, 40, seed=1, change=LENGTH, warm=warm)

    def test_warm_start_loads_actor_critics_and_temperature(self):
        source, path = self.source()
        run = self.target(path, self.root / "target")
        agent = build_agent(run)
        record = AD.apply_warm_start(agent, run)
        algorithm, model = agent.algorithm, agent.algorithm.model
        reference = source.algorithm.model
        self.assertEqual(record["networks"], ["actor", "critics", "critic_targets"])
        self.assertTrue(same_state(model.actor, reference.actor))
        self.assertTrue(same_state(model.q_nets, reference.q_nets))
        self.assertTrue(same_state(model.q_target_nets, reference.q_target_nets))
        self.assertTrue(torch.equal(algorithm.log_alpha, source.algorithm.log_alpha))
        self.assertEqual(algorithm.alpha, source.algorithm.alpha)
        self.assertEqual(algorithm.replay_buffer.size(), 0)
        self.assertEqual(algorithm.num_timesteps, 0)
        self.assertEqual(algorithm.learning_starts, 0)
        for optimizer in (algorithm.actor_optimizer, algorithm.critic_optimizer, algorithm.alpha_optimizer):
            self.assertEqual(len(optimizer.state), 0)
        rng = np.random.default_rng(3)
        states = random_states("acrobot-swingup", rng, 8)
        for state in states:
            np.testing.assert_array_equal(agent.predict(state), source.predict(state))
        observations = torch.as_tensor(states, dtype=torch.float32)
        actions = torch.as_tensor(rng.uniform(-1.0, 1.0, (8, 1)), dtype=torch.float32)
        with torch.no_grad():
            for ours, theirs in zip(model.q_values(observations, actions),
                                    reference.q_values(observations, actions)):
                self.assertTrue(torch.equal(ours, theirs))

    def test_run_skips_the_warm_up_and_trains_from_the_first_batch(self):
        source, path = self.source()
        run = self.target(path, self.root / "target")
        result, _ = run_captured(run)
        config = run["protocol"]["matched"]["testbed_config"]
        expected = P.evaluate_fixed_seeds(
            source.predict,
            lambda seed: tasks.make_env(config, decision_mode="torque", seed=seed),
            [3],
            discount_rate=float(config["discount_rate"]),
            deterministic=True,
        )
        self.assertEqual(first_returns(result), [e["discounted_return"] for e in expected])
        # One update per decision once the replay holds a batch (no random warm-up).
        self.assertEqual(result["counters"]["critic_gradient_steps"], 40 - BATCH + 1)
        self.assertEqual(result["warm_start"]["checkpoint"], str(path))
        self.assertEqual(result["physical_change"]["kind"], "length")
        self.assertNotIn("trainable_plant_parts", result)


class TestTorqueArmWarmStart(unittest.TestCase):
    """implicit_torque (the implicit algorithm acting in torques) takes the same warm start."""

    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.root = Path(self.folder.name)

    def tearDown(self):
        self.folder.cleanup()

    def test_torque_arm_starts_from_its_source_and_records_it(self):
        source = build_agent(tiny_run("implicit_torque", self.root / "source", 40))
        self.assertEqual(source.decision_mode, "torque")
        collect(source, 30)
        perturb(
            [source.actor, source.rate, source.value, source.target_value, source.online_plant,
             source.admitted_plant, source.reward_net]
        )
        with torch.no_grad():
            source.log_temperature.fill_(math.log(0.021))
        path = self.root / "torque.pt"
        source.save(path)
        warm = {"checkpoints": {"implicit_torque": str(path)}, "require_interactions": 30}
        run = tiny_run("implicit_torque", self.root / "target", 40, seed=1, change=DAMPING, warm=warm)
        AD.validate_warm_start_arms(run["protocol"])
        self.assertEqual(AD.warm_start_checkpoint(run), path)
        result, agent = run_captured(run)
        config = run["protocol"]["matched"]["testbed_config"]
        expected = P.evaluate_fixed_seeds(
            source.predict,
            lambda seed: tasks.make_env(config, decision_mode="torque", seed=seed),
            [3],
            discount_rate=float(config["discount_rate"]),
            deterministic=True,
        )
        self.assertEqual(first_returns(result), [e["discounted_return"] for e in expected])
        counters = result["counters"]
        self.assertGreater(counters["value_gradient_steps"], 0)
        self.assertEqual(counters["plant_gradient_steps"], counters["value_gradient_steps"])
        self.assertEqual(result["decision_mode"], "torque")
        self.assertEqual(result["warm_start"]["source_interactions"], 30)
        self.assertEqual(result["physical_change"]["kind"], "damping")
        self.assertNotIn("trainable_plant_parts", result)
        self.assertEqual(agent.decision_mode, "torque")

    def test_a_feedback_source_cannot_start_the_torque_arm(self):
        feedback = build_agent(tiny_run("implicit_learned_admissions", self.root / "feedback", 40))
        path = self.root / "feedback.pt"
        feedback.save(path)
        run = tiny_run(
            "implicit_torque", self.root / "target", 40, seed=1,
            warm={"checkpoints": {"implicit_torque": str(path)}},
        )
        with self.assertRaisesRegex(ValueError, "same decisions"):
            AD.apply_warm_start(build_agent(run), run)


class TestWarmStartSettings(unittest.TestCase):
    def test_arms_need_a_checkpoint_and_a_warm_start(self):
        with tempfile.TemporaryDirectory() as folder:
            warm = {"checkpoints": {"implicit_learned_admissions": "runs/s{seed}/{arm}/checkpoint.pt"}}
            run = tiny_run("implicit_learned_admissions", folder, 40, seed=2, warm=warm)
        self.assertEqual(
            AD.warm_start_checkpoint(run), Path("runs/s2/implicit_learned_admissions/checkpoint.pt")
        )
        protocol = copy.deepcopy(run["protocol"])
        AD.validate_warm_start_arms(protocol)
        catalog = {arm["name"]: arm for arm in P.all_arms()}
        for name, message in (("explicit_rollout", "takes no warm start"), ("ct_sac_torque", "no checkpoint")):
            protocol["arms"] = [catalog[name]]
            with self.subTest(arm=name), self.assertRaisesRegex(ValueError, message):
                AD.validate_warm_start_arms(protocol)

    def test_malformed_settings_raise(self):
        for warm in (
            {"checkpoints": {"implicit_learned_admissions": "runs/{bogus}.pt"}},
            {"checkpoints": {}},
            {"checkpoints": {"implicit_learned_admissions": "a.pt"}, "require_interactions": 0},
            {"checkpoints": {"implicit_learned_admissions": "a.pt"}, "optimizers": "carried"},
            {"checkpoint": "a.pt"},
        ):
            with self.subTest(warm=warm), tempfile.TemporaryDirectory() as folder:
                with self.assertRaises(ValueError):
                    tiny_run("implicit_learned_admissions", folder, 40, warm=warm)


class TestPilotConfigs(unittest.TestCase):
    PARTS = {"length": ["mass", "potential"], "damping": ["damping"]}

    def test_pilot_configs_follow_the_irregular_row(self):
        row = run_ph_sac.load_protocol_from_table("acrobot-swingup", "irregular")
        paths = sorted(CONFIGS.glob("*.json"))
        self.assertEqual(len(paths), 6)
        for path in paths:
            with self.subTest(config=path.name):
                protocol = run_ph_sac.load_protocol(str(path))
                kind, variant = path.stem.split("_")[1], path.stem.split("_")[-1]
                config = protocol["matched"]["testbed_config"]
                self.assertEqual(config["physical_change"], {"kind": kind, "scale": 1.25})
                self.assertTrue(
                    P._same_settings(nominal_testbed(config), row["matched"]["testbed_config"])
                )
                self.assertEqual(protocol["matched"]["interaction_budget"], 100000)
                self.assertEqual(protocol["training_seeds"], [0, 1, 2])
                for name in ("explicit_config", "implicit_config"):
                    ours = {k: v for k, v in protocol[name].items()
                            if k not in ("testbed", "trainable_plant_parts")}
                    theirs = {k: v for k, v in row[name].items() if k != "testbed"}
                    self.assertTrue(P._same_settings(ours, theirs), name)
                parts = protocol["implicit_config"].get("trainable_plant_parts")
                self.assertEqual(parts, self.PARTS[kind] if variant == "structured" else None)
                # The arms each file runs: the paper arms that take a warm start (its --arms
                # list: implicit_torque, and ct_sac_torque but in the structured files) and
                # the impedance-law implicit arm (--arms may add it); output folders differ
                # by arm.
                arms = ["implicit_torque", "implicit_learned_admissions"]
                if variant != "structured":
                    arms += ["ct_sac_torque"]
                selected = copy.deepcopy(protocol)
                selected["arms"] = [arm for arm in P.all_arms() if arm["name"] in arms]
                runs = P.build_run_matrix(selected)
                self.assertEqual(len({str(P.run_output_directory(r)) for r in runs}), 3 * len(arms))
                # The structured fine-tune reaches the torque arm's config unchanged.
                for run in runs:
                    if run["arm"]["algorithm"] == "implicit":
                        implicit = P._implicit_arm_config(run)
                        self.assertEqual(implicit.get("trainable_plant_parts"), parts)
                        self.assertEqual(
                            AD.adaptation_record(run, None).get("trainable_plant_parts"), parts
                        )
                if variant == "scratch":
                    self.assertNotIn("warm_start", protocol)
                    continue
                self.assertEqual(protocol["warm_start"]["require_interactions"], 500000)
                self.assertEqual(sorted(protocol["warm_start"]["checkpoints"]), sorted(arms))
                AD.validate_warm_start_arms(selected)
                tails = {
                    "implicit_learned_admissions": "g_acrobot_swingup/implicit_s2/acrobot_swingup/"
                    "implicit_learned_admissions/seed_2/checkpoint.pt",
                    "ct_sac_torque": "g_acrobot_swingup/ct_sac_s2/acrobot_swingup/ct_sac_torque/"
                    "seed_2/checkpoint.pt",
                    "implicit_torque": "gt_acrobot_swingup/implicit_torque_s2/acrobot_swingup/"
                    "implicit_torque/seed_2/checkpoint.pt",
                }
                for run in runs:
                    if run["training_seed"] == 2:
                        self.assertTrue(
                            str(AD.warm_start_checkpoint(run)).endswith(tails[run["arm"]["name"]]),
                            run["arm"]["name"],
                        )


if __name__ == "__main__":
    unittest.main()

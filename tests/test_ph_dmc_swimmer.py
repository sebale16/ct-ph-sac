"""The dm_control ports swimmer_swimmer6 and swimmer_swimmer15, and the plant options they use.

1. Dynamics: the analytic oracle (torch and numpy) against our MuJoCo model's forward
   dynamics, its mass matrix, centrifugal and fluid forces term by term (MuJoCo's
   inertia-box fluid model, with and without viscosity), MuJoCo's soft joint-limit rows
   (efc_pos, efc_vel, impedance, efc_aref, efc_R, efc_D, efc_force, qfrc_constraint) on
   states with joints past their ranges, against dm_control's compiled model, and its
   drift Jacobian against central differences.
2. Joint limits (environment/ph_sac/dmc/joint_limits.py) on a toy model with MuJoCo's
   default solref/solimp: rows, forces and accelerations against mj_forward, numpy twin,
   complementarity.
3. Task: the nose against MuJoCo's and dm_control's nose, the reward against
   dm_control's get_reward and the long_tail tolerance against rewards.tolerance, the
   tensor/numpy twins, no failure set (a joint passes its range and returns), the reset
   law against dm_control's resets, and the target as episode context.
4. Trajectories: our environment at the table's physics (dm_control's 2 ms Euler, since
   2026-11-10) against dm_control's own stepping (to rounding), no dof damping in either;
   our environment at 1 ms RK4 (the physics before) against dm_control's model stepped by
   MuJoCo at 1 ms RK4 (to rounding), and against dm_control's 2 ms Euler loosely, through
   the joint limits.
5. Model integrator: the oracle plant's semi_implicit map at the 2 ms Euler physics (15
   substeps) and its rk4 map at 1 ms RK4 physics (30 substeps) reproduce env.step_dt
   over a 30 ms interval, endpoint means and noise-free rollout paths, damping near the
   box maximum and joints past their ranges included; the semi_implicit closed-loop
   Jacobian (the propagated covariance's) with D(y, v) and the limit port against
   central differences; uniform-decision episodes at 2 ms Euler stay bounded.
6. Table: both rows build, their boxes follow the rule, the entropy targets are -15 and
   -42, the rows set network damping_velocity and limit_port, no failure set, the state
   spec's cyclic positions and limited ranges, 2 ms Euler physics and the semi_implicit
   map at the 2 ms model step.
7. Learned plant options: network damping_velocity (D(y, v), analytic Jacobian with the
   dD/dv terms against torch.func, a short fit of the swimmer's drift better than D(y)),
   cyclic_position_indices (features without x, y and the target) and limit_port (zero
   inside the ranges, passive, analytic Jacobian, and a short fit past the limits far
   better than without it).
8. Regression: without the new keys the plant and short training runs of existing rows
   reproduce values computed with the code before these changes (with contact
   substepping off, as the code they pin); short swimmer runs at the 2 ms Euler
   convention reproduce REFERENCE_SWIMMER_ARMS (generated on the cluster by
   benchmarks/ph_swimmer_references.py; a key set to None is skipped).
"""

import copy
import math
import tempfile
import unittest
from pathlib import Path

import mujoco
import numpy as np
import torch

from benchmarks import ph_hyperparams as table
from benchmarks import run_ph_sac
from common.ph_config import decision_box_entropy_target
from common.ph_dmc._common import fixed_clock
from common.ph_dmc.swimmer import swimmer_joint_range
from environment.ph_sac import tasks
from environment.ph_sac.dmc import _common as dmc_common
from environment.ph_sac.dmc import joint_limits
from environment.ph_sac.dmc import swimmer as dmc_swimmer
from environment.ph_sac.registry import registered_testbeds, testbed_spec
from evaluations.ph_sac import protocol
from models.ph_sac.plant import PortHamiltonianPlant
from models.ph_sac.rollouts import (
    augmented_closed_loop_step,
    semi_implicit_step,
    stochastic_rollout,
    wrapped_endpoint_residual,
)
from tests.test_ph_model_integrator import interval_means

ENV_IDS = {"swimmer_swimmer6": "swimmer-swimmer6", "swimmer_swimmer15": "swimmer-swimmer15"}
LINKS = {"swimmer_swimmer6": 6, "swimmer_swimmer15": 15}
TESTBEDS = tuple(ENV_IDS)
GEAR = 5e-4  # dm_control's control = our torque / gear
# Joints drawn inside their ranges stay within this fraction of them; random_states puts
# a share of them past the range by up to PAST_DEPTH rad (the impedance ramp is 0.1 rad).
SAFE = 0.95
PAST_DEPTH = 0.15


def table_entry(name):
    return table.find_ph_entry(ENV_IDS[name], "top")


def table_config(name):
    return testbed_spec(name).config_factory(**table_entry(name).config["testbed_settings"])


def euler_config(name):
    """The table config, which has dm_control's physics since 2026-11-10: 2 ms Euler steps."""
    config = copy.deepcopy(table_config(name))
    assert (config["physics_dt"], config["integrator"]) == (0.002, "Euler")
    return config


def rk4_config(name):
    """The table config with the swimmers' physics before 2026-11-10: 1 ms RK4 steps."""
    config = copy.deepcopy(table_config(name))
    config["physics_dt"] = 0.001
    config["integrator"] = "RK4"
    config["clock"] = fixed_clock(0.03, 0.001)
    return config


def load_dm(name, seed=0):
    from dm_control import suite

    env = suite.load("swimmer", f"swimmer{LINKS[name]}", task_kwargs={"random": seed})
    env.reset()
    return env


def random_states(config, rng, count, *, speed=1.0, past=0.0):
    """States with each joint past its range with probability past (by up to PAST_DEPTH
    rad), else inside it; a quarter with the target within 0.1 m of the nose (reward 1),
    a quarter within 0.6 m, the rest anywhere in +-2 m."""
    n = int(config["plant_parameters"]["n_links"])
    limit = SAFE * swimmer_joint_range(n)
    positions = np.concatenate(
        (
            rng.uniform(-2.0, 2.0, (count, 2)),
            rng.uniform(-np.pi, np.pi, (count, 1)),
            rng.uniform(-limit, limit, (count, n - 1)),
        ),
        axis=-1,
    )
    if past:
        beyond = rng.random((count, n - 1)) < past
        sign = np.where(rng.random((count, n - 1)) < 0.5, -1.0, 1.0)
        depth = swimmer_joint_range(n) + rng.uniform(0.0, PAST_DEPTH, (count, n - 1))
        positions[:, 3:] = np.where(beyond, sign * depth, positions[:, 3:])
    velocities = np.concatenate(
        (rng.uniform(-0.3, 0.3, (count, 2)), rng.uniform(-3.0, 3.0, (count, n))), axis=-1
    ) * speed
    noses = np.stack([nose_of(config, row) for row in positions])
    targets = rng.uniform(-2.0, 2.0, (count, 2))
    draw = rng.random(count)
    near, at = (draw < 0.5) & (draw >= 0.25), draw < 0.25
    targets[near] = noses[near] + rng.uniform(-0.6, 0.6, (int(near.sum()), 2))
    targets[at] = noses[at] + rng.uniform(-0.07, 0.07, (int(at.sum()), 2))
    return np.concatenate((positions, targets, velocities), axis=-1)


def nose_of(config, positions):
    """The nose from the definition: 0.01 m ahead of the root hinge, 0.05 m ahead of the head."""
    x, y, yaw = positions[0], positions[1], positions[2]
    return np.array([x + 0.01 * math.sin(yaw), y - 0.05 - 0.01 * math.cos(yaw)])


def split(state, n):
    nv = n + 2
    return state[:nv], state[nv : nv + 2], state[nv + 2 :]


def set_dm_state(dm_env, state, torque, n):
    physics = dm_env.physics
    q, target, v = split(state, n)
    physics.data.qpos[:] = q
    physics.data.qvel[:] = v
    physics.data.ctrl[:] = np.asarray(torque) / GEAR
    physics.named.model.geom_pos["target", :2] = target
    physics.forward()


def scaled_error(left, right):
    """max |left - right| relative to max(1, max |right|)."""
    left, right = np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
    return float(np.max(np.abs(left - right)) / max(1.0, float(np.max(np.abs(right)))))


def wrap(angle):
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def ks_statistic(left, right):
    left, right = np.sort(left), np.sort(right)
    points = np.concatenate((left, right))
    return float(
        np.max(
            np.abs(
                np.searchsorted(left, points, side="right") / len(left)
                - np.searchsorted(right, points, side="right") / len(right)
            )
        )
    )


class Float64:
    def __enter__(self):
        self.previous = torch.get_default_dtype()
        torch.set_default_dtype(torch.float64)

    def __exit__(self, *args):
        torch.set_default_dtype(self.previous)


# ---------------------------------------------------------------------------- dynamics


class TestDynamics(unittest.TestCase):
    def test_oracle_matches_mujoco(self):
        for name in TESTBEDS:
            with self.subTest(testbed=name):
                config = table_config(name)
                n = LINKS[name]
                env = tasks.make_env(config, seed=0)
                rng = np.random.default_rng(0)
                # A third of the joints past their ranges: the limit forces are included.
                states = random_states(config, rng, 200, past=0.3)
                torques = rng.uniform(-GEAR, GEAR, (200, n - 1))
                drift, diffusion = tasks.oracle_callbacks(config)
                tensor = drift(torch.as_tensor(states), torch.as_tensor(torques)).numpy()
                batched = env.oracle_drift(states, torques)
                for state, torque, from_tensor, from_batch in zip(states, torques, tensor, batched):
                    truth = env.mujoco_drift(state, torque)
                    self.assertLess(scaled_error(env.oracle_drift(state, torque), truth), 1e-10)
                    self.assertLess(scaled_error(from_tensor, truth), 1e-10)
                    self.assertLess(scaled_error(from_batch, truth), 1e-10)
                    np.testing.assert_array_equal(truth[: n + 4], np.concatenate((state[n + 4 :], [0.0, 0.0])))
                    for value in (from_tensor, from_batch):
                        np.testing.assert_array_equal(value[n + 2 : n + 4], [0.0, 0.0])
                sigma = diffusion(torch.as_tensor(states))
                self.assertEqual(sigma.shape, (200, env.state_dim, 1))
                self.assertTrue(torch.equal(sigma, torch.zeros_like(sigma)))
                expected_g = np.zeros((n + 2, n - 1))
                expected_g[3:] = np.eye(n - 1)
                np.testing.assert_array_equal(env.oracle_input_matrix(), expected_g)
                np.testing.assert_array_equal(env.dynamics.input_matrix, expected_g)

    def test_terms_match_mujoco(self):
        """M(q), the centrifugal force and the fluid force each equal MuJoCo's."""
        for name in TESTBEDS:
            with self.subTest(testbed=name):
                config = table_config(name)
                n = LINKS[name]
                env = tasks.make_env(config, seed=0)
                model, data = env.model, mujoco.MjData(env.model)
                dynamics = env.dynamics
                rng = np.random.default_rng(1)
                for state in random_states(config, rng, 50):
                    q, _, v = split(state, n)
                    data.qpos[:], data.qvel[:] = q, v
                    mujoco.mj_forward(model, data)
                    mass = np.zeros((model.nv, model.nv))
                    mujoco.mj_fullM(model, mass, data.qM)
                    with Float64():
                        ours = dmc_swimmer.swimmer_terms(dynamics, torch.as_tensor(q), torch.as_tensor(v))
                    ours = {key: value.numpy() for key, value in ours.items()}
                    self.assertLess(scaled_error(ours["mass"], mass), 1e-12)
                    self.assertLess(scaled_error(ours["bias"], data.qfrc_bias), 1e-12)
                    self.assertLess(scaled_error(ours["fluid"], data.qfrc_fluid), 1e-12)
                    np.testing.assert_array_equal(data.qfrc_passive, data.qfrc_fluid)
                    self.assertEqual(data.nefc, 0)
                # The root is free, the joints keep dm_control's limits, the solver is on.
                self.assertEqual(model.jnt_limited.tolist(), [0, 0, 0] + [1] * (n - 1))
                self.assertFalse(model.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_CONSTRAINT)
                self.assertTrue(model.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_CONTACT)

    def test_limit_rows_match_mujoco(self):
        """Every efc_* field of the limit rows, the forces and qacc, with joints past their
        ranges, near the impedance ramp and deep beyond it, and several limits at once.

        MuJoCo's default solver settings (Newton, 100 iterations, tolerance 1e-8) reach the
        complementarity solution to rounding."""
        for name in TESTBEDS:
            with self.subTest(testbed=name):
                config = table_config(name)
                n = LINKS[name]
                env = tasks.make_env(config, seed=0)
                model, data = env.model, mujoco.MjData(env.model)
                self.assertEqual(model.opt.solver, mujoco.mjtSolver.mjSOL_NEWTON)
                self.assertEqual((model.opt.iterations, model.opt.tolerance), (100, 1e-8))
                limits = env.dynamics.limits
                rng = np.random.default_rng(21)
                states = random_states(config, rng, 150, past=0.4)
                # Small penetrations, where the impedance starts from its floor.
                states[:40, 3:4] = swimmer_joint_range(n) + rng.uniform(0.0, 1e-3, (40, 1))
                counts = []
                for state in states:
                    q, _, v = split(state, n)
                    torque = rng.uniform(-GEAR, GEAR, n - 1)
                    data.qpos[:], data.qvel[:], data.ctrl[:] = q, v, torque
                    mujoco.mj_forward(model, data)
                    theirs = joint_limits.mujoco_limit_rows(model, data, limits)
                    mass = np.zeros((model.nv, model.nv))
                    mujoco.mj_fullM(model, mass, data.qM)
                    with Float64():
                        ours = joint_limits.constrained_accelerations(
                            limits, torch.linalg.cholesky(torch.as_tensor(mass)),
                            torch.as_tensor(np.array(data.qacc_smooth)), torch.as_tensor(q), torch.as_tensor(v),
                        )
                    ours = {key: value.numpy() for key, value in ours.items()}
                    twin = joint_limits.constrained_accelerations_numpy(limits, mass, data.qacc_smooth, q, v)
                    active = theirs["active"]
                    np.testing.assert_array_equal(ours["active"], active)
                    np.testing.assert_array_equal(twin["active"], active)
                    counts.append((int(active.sum()), int((theirs["force"][active] > 0).sum())))
                    if active.any():
                        for key in ("side", "pos", "vel", "imp", "aref", "R", "D", "force"):
                            for values in (ours[key], twin[key]):
                                self.assertLess(scaled_error(values[active], theirs[key][active]), 1e-10, key)
                        np.testing.assert_allclose(theirs["margin"][active], 0.0)
                        np.testing.assert_allclose(theirs["diagApprox"][active], limits.invweight[active], rtol=1e-14)
                    for values in (ours, twin):
                        self.assertLess(scaled_error(values["qfrc_constraint"], data.qfrc_constraint), 1e-10)
                        self.assertLess(scaled_error(values["qacc"], data.qacc), 1e-10)
                        # Complementarity: f >= 0 and pushing rows are MuJoCo's active rows.
                        self.assertTrue(np.all(values["force"] >= 0.0))
                    # The active set is where MuJoCo's force is positive (its efc_state is not
                    # updated after the last Newton step, so it is no guide).
                    np.testing.assert_array_equal(ours["rows_active"][active], theirs["force"][active] > 0.0)
                counts = np.array(counts)
                self.assertGreater((counts[:, 0] >= 2).sum(), 20)  # several limits at once
                self.assertGreater((counts[:, 1] < counts[:, 0]).sum(), 0)  # a row that pulls back off

    def test_fluid_dissipates_and_scales_quadratically(self):
        config = table_config("swimmer_swimmer6")
        env = tasks.make_env(config, seed=0)
        rng = np.random.default_rng(2)
        with Float64():
            for state in random_states(config, rng, 30):
                q, _, v = split(state, 6)
                q, v = torch.as_tensor(q), torch.as_tensor(v)
                fluid = dmc_swimmer.swimmer_terms(env.dynamics, q, v)["fluid"]
                self.assertLessEqual(float(fluid @ v), 0.0)
                doubled = dmc_swimmer.swimmer_terms(env.dynamics, q, 2.0 * v)["fluid"]
                torch.testing.assert_close(doubled, 4.0 * fluid, rtol=1e-12, atol=1e-15)

    def test_viscosity_matches_mujoco(self):
        """The inertia-box model's viscous terms (unused by dm_control's swimmer) are right too."""
        config = copy.deepcopy(table_config("swimmer_swimmer6"))
        config["plant_parameters"] = {**config["plant_parameters"], "viscosity": 0.01}
        env = tasks.make_env(config, seed=0)
        drift, _ = tasks.oracle_callbacks(config)
        rng = np.random.default_rng(3)
        for state in random_states(config, rng, 50):
            torque = rng.uniform(-GEAR, GEAR, 5)
            truth = env.mujoco_drift(state, torque)
            self.assertLess(scaled_error(env.oracle_drift(state, torque), truth), 1e-10)
            got = drift(torch.as_tensor(state[None]), torch.as_tensor(torque[None]))[0].numpy()
            self.assertLess(scaled_error(got, truth), 1e-10)

    def test_dynamics_constants_match_dm_control(self):
        for name in TESTBEDS:
            with self.subTest(testbed=name):
                n = LINKS[name]
                ours = tasks.make_env(table_config(name), seed=0).dynamics
                theirs = dmc_swimmer.swimmer_dynamics(load_dm(name).physics, n)
                for field in ("masses", "inertias", "com", "lengths", "armature", "drag_x",
                              "drag_y", "drag_spin", "viscous_linear", "viscous_spin",
                              "head_anchor"):
                    np.testing.assert_allclose(getattr(ours, field), getattr(theirs, field), rtol=1e-14, atol=0, err_msg=field)
                for field in ("dofs", "lower", "upper", "timeconst", "dampratio", "d0", "dmax",
                              "width", "midpoint", "power", "margin"):
                    np.testing.assert_array_equal(getattr(ours.limits, field), getattr(theirs.limits, field), err_msg=field)
                # dof_invweight0 depends on the timestep-free mass matrix at qpos0 only.
                np.testing.assert_allclose(ours.limits.invweight, theirs.limits.invweight, rtol=1e-13)
                np.testing.assert_allclose(ours.limits.upper, 2.0 * math.pi / n, rtol=1e-15)
                np.testing.assert_array_equal(ours.limits.timeconst, 0.05)
                np.testing.assert_array_equal(ours.limits.dmax, 0.8)
                np.testing.assert_array_equal(ours.limits.width, 0.1)
                np.testing.assert_allclose(ours.input_matrix * GEAR, theirs.input_matrix, rtol=1e-15)
                self.assertAlmostEqual(ours.nose_offset, 0.01, places=15)
                np.testing.assert_allclose(ours.masses, 0.01)
                # The inertia box: full sizes (0.002, 0.1, 0.02) m, rho/2 b_y b_z = 3, rho/2 b_x b_z = 0.06.
                np.testing.assert_allclose(ours.drag_x, 3.0, rtol=1e-12)
                np.testing.assert_allclose(ours.drag_y, 0.06, rtol=1e-12)
                np.testing.assert_allclose(ours.drag_spin, 3000 * 0.02 * (0.002**4 + 0.1**4) / 64, rtol=1e-12)

    def test_forward_dynamics_match_dm_control(self):
        for name in TESTBEDS:
            with self.subTest(testbed=name):
                config = table_config(name)
                n = LINKS[name]
                env = tasks.make_env(config, seed=0)
                dm_env = load_dm(name)
                rng = np.random.default_rng(4)
                rows = 0
                for state in random_states(config, rng, 100, past=0.3):
                    torque = rng.uniform(-GEAR, GEAR, n - 1)
                    set_dm_state(dm_env, state, torque, n)
                    rows += dm_env.physics.data.nefc
                    got = env.mujoco_drift(state, torque)[n + 4 :]
                    self.assertLess(scaled_error(got, dm_env.physics.data.qacc), 1e-12)
                self.assertGreater(rows, 50)

    def test_oracle_plant_jacobian(self):
        for name in TESTBEDS:
            with self.subTest(testbed=name), Float64():
                config = table_config(name)
                n = LINKS[name]
                env = tasks.make_env(config, seed=0)
                plant = tasks.make_oracle_plant(env)
                rng = np.random.default_rng(5)
                states = torch.as_tensor(random_states(config, rng, 4, speed=0.5, past=0.3))
                actions = torch.as_tensor(rng.uniform(-GEAR, GEAR, (4, n - 1)))
                values, by_state, by_action = plant.drift_jacobian(states, actions)
                torch.testing.assert_close(values, plant.drift(states, actions))
                for argument, jacobian, size, epsilon in (
                    (0, by_state, env.state_dim, 1e-6),
                    (1, by_action, n - 1, 1e-8),
                ):
                    for i in range(size):
                        bump = torch.zeros(4, size, dtype=torch.float64)
                        bump[:, i] = epsilon
                        if argument == 0:
                            high, low = plant.drift(states + bump, actions), plant.drift(states - bump, actions)
                        else:
                            high, low = plant.drift(states, actions + bump), plant.drift(states, actions - bump)
                        numeric = (high - low) / (2.0 * epsilon)
                        scale = 1.0 + numeric.abs().max().item()
                        # Central differences: the drag |v| v jumps in curvature at v = 0.
                        self.assertLess((jacobian[:, :, i] - numeric).abs().max().item() / scale, 2e-5)
                # x, y and the target are cyclic: no column, and the target has no rate.
                cyclic = [0, 1, n + 2, n + 3]
                self.assertEqual(by_state[:, :, cyclic].abs().max().item(), 0.0)
                self.assertEqual(by_state[:, n + 2 : n + 4].abs().max().item(), 0.0)


# ------------------------------------------------------------------- joint-limit module

# A slide carrying a two-link arm under gravity: MuJoCo's default solref (0.02 1) and
# solimp (0.9 0.95 0.001 0.5 2) on the slide and the first hinge, dm_control's swimmer
# values on the second, and a time constant below 2 timesteps (refsafe) on the third.
_TOY_LIMITS_XML = """
<mujoco>
  <option timestep="0.004"><flag contact="disable"/></option>
  <worldbody>
    <body name="cart">
      <joint name="slide" type="slide" axis="1 0 0" limited="true" range="-0.5 0.5" damping="0.1"/>
      <geom type="box" size=".1 .1 .1" mass="1"/>
      <body name="arm" pos="0 0 .1">
        <joint name="hinge1" type="hinge" axis="0 1 0" limited="true" range="-30 45"/>
        <geom type="capsule" fromto="0 0 0 0 0 .5" size=".02" mass=".3"/>
        <body name="hand" pos="0 0 .5">
          <joint name="hinge2" type="hinge" axis="0 1 0" limited="true" range="-60 60"
                 solreflimit=".05 1" solimplimit="0 .8 .1"/>
          <geom type="capsule" fromto="0 0 0 0 0 .4" size=".02" mass=".2"/>
          <body name="tip" pos="0 0 .4">
            <joint name="hinge3" type="hinge" axis="0 1 0" limited="true" range="-20 20"
                   solreflimit=".001 .5"/>
            <geom type="sphere" size=".05" mass=".1"/>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


class TestJointLimitsModule(unittest.TestCase):
    """joint_limits.py on its own: any model's limited hinges and slides, MuJoCo defaults."""

    def test_rows_forces_and_accelerations_match_mujoco(self):
        model = mujoco.MjModel.from_xml_string(_TOY_LIMITS_XML)
        data = mujoco.MjData(model)
        limits = joint_limits.joint_limits_from_model(model, ["slide", "hinge1", "hinge2", "hinge3"])
        np.testing.assert_array_equal(limits.dampratio, [1.0, 1.0, 1.0, 0.5])
        np.testing.assert_array_equal(limits.timeconst, [0.02, 0.02, 0.05, 0.008])  # refsafe
        np.testing.assert_array_equal(limits.d0, [0.9, 0.9, 0.0, 0.9])
        np.testing.assert_array_equal(limits.width, [0.001, 0.001, 0.1, 0.001])
        rng = np.random.default_rng(31)
        lower, upper = model.jnt_range[:, 0], model.jnt_range[:, 1]
        seen = 0
        for trial in range(300):
            q = rng.uniform(lower, upper)
            past = rng.random(4) < 0.5
            depth = rng.choice([1e-4, 5e-4, 2e-3, 0.05, 0.2]) * rng.random(4)
            q = np.where(past & (rng.random(4) < 0.5), upper + depth, np.where(past, lower - depth, q))
            v = rng.normal(0.0, 2.0, 4)
            data.qpos[:], data.qvel[:] = q, v
            mujoco.mj_forward(model, data)
            mass = np.zeros((4, 4))
            mujoco.mj_fullM(model, mass, data.qM)
            theirs = joint_limits.mujoco_limit_rows(model, data, limits)
            with Float64():
                ours = joint_limits.constrained_accelerations(
                    limits, torch.linalg.cholesky(torch.as_tensor(mass)),
                    torch.as_tensor(np.array(data.qacc_smooth)), torch.as_tensor(q), torch.as_tensor(v),
                )
            ours = {key: value.numpy() for key, value in ours.items()}
            twin = joint_limits.constrained_accelerations_numpy(limits, mass, data.qacc_smooth, q, v)
            active = theirs["active"]
            seen += int(active.sum())
            for values in (ours, twin):
                np.testing.assert_array_equal(values["active"], active)
                if active.any():
                    for key in ("pos", "vel", "imp", "aref", "R", "D", "force"):
                        self.assertLess(scaled_error(values[key][active], theirs[key][active]), 1e-10, key)
                self.assertLess(scaled_error(values["qfrc_constraint"], data.qfrc_constraint), 1e-10)
                self.assertLess(scaled_error(values["qacc"], data.qacc), 1e-10)
        self.assertGreater(seen, 300)

    def test_no_rows_inside_the_ranges_and_the_derivative(self):
        model = mujoco.MjModel.from_xml_string(_TOY_LIMITS_XML)
        limits = joint_limits.joint_limits_from_model(model, ["slide", "hinge1", "hinge2", "hinge3"])
        with Float64():
            mass = torch.eye(4) + 0.1 * torch.ones(4, 4)
            factor = torch.linalg.cholesky(mass)
            smooth = torch.tensor([0.3, -1.0, 2.0, 0.5])
            inside = torch.tensor([0.1, 0.2, -0.3, 0.1])
            out = joint_limits.constrained_accelerations(limits, factor, smooth, inside, torch.ones(4))
            self.assertTrue(torch.equal(out["qacc"], smooth))
            self.assertFalse(out["active"].any())
            # torch.func: the Jacobian of qacc in the positions past the limits.
            past = torch.tensor([0.52, 0.9, -1.2, 0.36])
            rates = torch.tensor([0.4, -0.3, 0.2, 0.1])

            def accel(positions):
                return joint_limits.constrained_accelerations(limits, factor, smooth, positions, rates)["qacc"]

            jacobian = torch.func.jacfwd(accel)(past)
            numeric = torch.stack(
                [(accel(past + 1e-7 * e) - accel(past - 1e-7 * e)) / 2e-7 for e in torch.eye(4)], dim=-1
            )
            torch.testing.assert_close(jacobian, numeric, rtol=1e-6, atol=1e-6)


# -------------------------------------------------------------------------------- task


class TestTask(unittest.TestCase):
    def test_nose_matches_mujoco_and_dm_control(self):
        for name in TESTBEDS:
            with self.subTest(testbed=name):
                config = table_config(name)
                n = LINKS[name]
                env = tasks.make_env(config, seed=0)
                dm_env = load_dm(name)
                data = mujoco.MjData(env.model)
                rng = np.random.default_rng(6)
                states = random_states(config, rng, 100)
                tensor = dmc_swimmer.swimmer_nose_tensor(torch.as_tensor(states), config).numpy()
                for state, from_tensor in zip(states, tensor):
                    q, _, _ = split(state, n)
                    data.qpos[:] = q
                    mujoco.mj_forward(env.model, data)
                    sensor = data.sensordata[:3]
                    set_dm_state(dm_env, state, np.zeros(n - 1), n)
                    for value in (env.nose(state), from_tensor, nose_of(config, q)):
                        np.testing.assert_allclose(value, sensor[:2], rtol=0, atol=1e-14)
                        np.testing.assert_allclose(value, dm_env.physics.named.data.sensordata["nose_pos"][:2], rtol=0, atol=1e-14)
                        np.testing.assert_allclose(value, dm_env.physics.named.data.geom_xpos["nose", :2], rtol=0, atol=1e-14)
                    self.assertAlmostEqual(sensor[2], 0.05, places=14)

    def test_long_tail_tolerance_matches_dm_control(self):
        from dm_control.utils import rewards

        values = np.linspace(-3.0, 3.0, 601)
        for bounds, margin, value_at_margin in (
            ((0.0, 0.1), 0.5, 0.1),
            ((-0.5, 0.2), 1.0, 0.3),
            ((0.0, 0.0), 2.0, 0.05),
        ):
            expected = rewards.tolerance(values, bounds=bounds, margin=margin, sigmoid="long_tail", value_at_margin=value_at_margin)
            got = dmc_common.tolerance(values, bounds, margin, sigmoid="long_tail", value_at_margin=value_at_margin)
            np.testing.assert_allclose(got, expected, rtol=1e-14, atol=0)
            tensor = dmc_common.tolerance_tensor(torch.as_tensor(values), bounds, margin, sigmoid="long_tail", value_at_margin=value_at_margin)
            np.testing.assert_allclose(tensor.numpy(), expected, rtol=1e-14, atol=0)
        self.assertAlmostEqual(dmc_common.tolerance(0.6, (0.0, 0.1), 0.5, sigmoid="long_tail"), 0.1, places=15)
        with self.assertRaises(ValueError):
            dmc_common.tolerance(0.6, (0.0, 0.1), 0.5, sigmoid="long_tail", value_at_margin=0.0)
        # A long tail: 1 / (1 + 9 (d / m)^2), far above the gaussian's value away from the target.
        self.assertAlmostEqual(dmc_common.tolerance(2.1, (0.0, 0.1), 0.5, sigmoid="long_tail"), 1.0 / 145.0, places=15)

    def test_reward_matches_dm_control(self):
        for name in TESTBEDS:
            with self.subTest(testbed=name):
                config = table_config(name)
                n = LINKS[name]
                env = tasks.make_env(config, seed=0)
                dm_env = load_dm(name)
                rng = np.random.default_rng(7)
                states = random_states(config, rng, 300)
                torques = rng.uniform(-GEAR, GEAR, (300, n - 1))
                tensor = tasks.reward_rate_tensor(torch.as_tensor(states), torch.as_tensor(torques), config)
                self.assertEqual(tensor.shape, (300, 1))
                rewards = []
                for state, torque, from_tensor in zip(states, torques, tensor[:, 0].tolist()):
                    set_dm_state(dm_env, state, torque, n)
                    expected = float(dm_env.task.get_reward(dm_env.physics))
                    self.assertLessEqual(abs(env.reward_rate(state, torque) - expected), 1e-12)
                    self.assertLessEqual(abs(from_tensor - expected), 1e-12)
                    rewards.append(expected)
                rewards = np.array(rewards)
                self.assertEqual(dm_env.physics.named.model.geom_size["target", 0], 0.1)
                self.assertGreater((rewards == 1.0).sum(), 10)
                self.assertGreater(((rewards > 0.0) & (rewards < 0.5)).sum(), 50)
                self.assertTrue(np.all((rewards > 0.0) & (rewards <= 1.0)))
                self.assertEqual(testbed_spec(name).reward_rate_bounds(config), (0.0, 1.0))
                self.assertIsNone(testbed_spec(name).upright)

    def test_no_failure_set(self):
        for name in TESTBEDS:
            with self.subTest(testbed=name):
                config = table_config(name)
                env = tasks.make_env(config, seed=0)
                rng = np.random.default_rng(8)
                states = random_states(config, rng, 100, past=0.5)
                mask = tasks.failure_mask_tensor(torch.as_tensor(states), config)
                self.assertFalse(mask.any())
                self.assertFalse(any(env.failed(s) for s in states))
                payoff = tasks.terminal_payoff_tensor(torch.as_tensor(states), config)
                self.assertTrue(torch.equal(payoff, torch.zeros_like(payoff)))
                lift, vanishing = tasks.terminal_boundary_functions(config)
                self.assertTrue(torch.equal(vanishing(torch.as_tensor(states)), torch.ones(100, 1, dtype=torch.float64)))
                self.assertTrue(torch.equal(lift(torch.as_tensor(states)), torch.zeros(100, 1, dtype=torch.float64)))
                self.assertEqual(config["failure_limits"], {})

    def test_joints_pass_their_ranges_and_return(self):
        """A full-torque fold drives the joints past their ranges; the soft limits stop
        them a fraction of a radian beyond and the episode goes on."""
        for name in TESTBEDS:
            with self.subTest(testbed=name):
                config = table_config(name)
                n = LINKS[name]
                limit = swimmer_joint_range(n)
                env = tasks.make_env(config, decision_mode="torque", seed=0)
                state = np.zeros(env.state_dim)
                state[n + 2 : n + 4] = [1.0, 1.0]
                env.reset(options={"state": state})
                deepest, passed = 0.0, False
                for step in range(100):
                    torque = np.full(n - 1, GEAR) if step < 60 else np.full(n - 1, -GEAR)
                    transition = env.step_dt(torque, 0.03)
                    self.assertFalse(transition["terminated"])
                    joints = np.abs(transition["next_state"][3 : n + 2])
                    passed = passed or bool(np.any(joints > limit))
                    deepest = max(deepest, float(np.max(joints)) - limit)
                self.assertTrue(passed)
                self.assertLess(deepest, 0.3)
                self.assertFalse(env.done)

    def test_reset_law(self):
        """Our resets against dm_control's: yaw, joints, target box mixture, root at rest."""
        for name in TESTBEDS:
            with self.subTest(testbed=name):
                config = table_config(name)
                n = LINKS[name]
                limit = swimmer_joint_range(n)
                env = tasks.make_env(config, seed=11)
                ours = np.array([env.reset()[0] for _ in range(400)])
                np.testing.assert_array_equal(ours[:, :2], 0.0)
                np.testing.assert_array_equal(ours[:, n + 4 :], 0.0)
                self.assertTrue(np.all(np.abs(ours[:, 3 : n + 2]) < limit))
                self.assertTrue(np.all((ours[:, 2] >= -math.pi) & (ours[:, 2] < math.pi)))
                targets = ours[:, n + 2 : n + 4]
                self.assertTrue(np.all(np.abs(targets) <= 2.0))
                dm_env = load_dm(name, seed=12)
                theirs, dm_targets = [], []
                for _ in range(400):
                    dm_env.reset()
                    theirs.append(np.array(dm_env.physics.data.qpos))
                    dm_targets.append(np.array(dm_env.physics.named.model.geom_pos["target", :2]))
                theirs, dm_targets = np.array(theirs), np.array(dm_targets)
                np.testing.assert_array_equal(theirs[:, :2], 0.0)
                self.assertTrue(np.all(np.abs(theirs[:, 3:]) <= limit))
                # KS at 400 + 400 draws: the 1% critical value is 1.63 sqrt(2 / 400) = 0.115.
                self.assertLess(ks_statistic(ours[:, 2], theirs[:, 2]), 0.115)
                for joint in (3, n + 1):
                    self.assertLess(ks_statistic(ours[:, joint], theirs[:, joint]), 0.115)
                for axis in (0, 1):
                    self.assertLess(ks_statistic(targets[:, axis], dm_targets[:, axis]), 0.115)
                close = np.all(np.abs(targets) <= 0.3, axis=-1).mean()
                dm_close = np.all(np.abs(dm_targets) <= 0.3, axis=-1).mean()
                expected = 0.2 + 0.8 * (0.3 / 2.0) ** 2
                for fraction in (close, dm_close):
                    self.assertLess(abs(fraction - expected), 4.0 * math.sqrt(expected * (1 - expected) / 400))

    def test_context(self):
        """The target is episode-constant, carried by reset options and snapshots, read by the reward."""
        config = table_config("swimmer_swimmer6")
        env = tasks.make_env(config, seed=3)
        state, _ = env.reset()
        target = env.context
        np.testing.assert_array_equal(state[8:10], target)
        np.testing.assert_array_equal(env.model.geom_pos[env._target_geom, :2], target)
        decision = np.zeros(15)
        for _ in range(5):
            transition = env.step_dt(decision, 0.03)
            np.testing.assert_array_equal(transition["next_state"][8:10], target)
            if transition["terminated"]:
                break
        snapshot = env.snapshot()
        moved = env.observe().copy()
        moved[8:10] = [0.5, -0.25]
        env.reset(options={"state": moved})
        np.testing.assert_array_equal(env.context, [0.5, -0.25])
        np.testing.assert_array_equal(env.observe(), moved)
        env.restore(snapshot)
        np.testing.assert_array_equal(env.context, target)
        # The reward reads the context: a target at the nose gives 1, a far one less.
        at_nose = moved.copy()
        at_nose[8:10] = env.nose(moved)
        self.assertEqual(env.reward_rate(at_nose, np.zeros(5)), 1.0)
        far = moved.copy()
        far[8:10] = env.nose(moved) + [1.0, 0.0]
        self.assertAlmostEqual(env.reward_rate(far, np.zeros(5)), 1.0 / (1.0 + 9.0 * (0.9 / 0.5) ** 2), places=14)
        # The dynamics do not read it.
        torque = np.full(5, 1e-4)
        np.testing.assert_array_equal(env.mujoco_drift(at_nose, torque)[10:], env.mujoco_drift(far, torque)[10:])


# ------------------------------------------------------------------------ trajectories


def open_loop_torques(n, steps, dt, scale=0.9, *, square=False):
    """A travelling wave of joint torques, as a swimming stroke; square=True is a
    0.5 Hz square wave at full torque, which folds the joints past their ranges."""
    times = np.arange(steps)[:, None] * dt
    if square:
        return GEAR * np.sign(np.sin(2.0 * np.pi * 0.5 * times + 0.8 * np.arange(n - 1)[None, :]))
    return scale * GEAR * np.sin(2.0 * np.pi * 1.5 * times + 0.8 * np.arange(n - 1)[None, :])


class TestTrajectories(unittest.TestCase):
    def start(self, n, rng):
        limit = swimmer_joint_range(n)
        q = np.concatenate(([0.0, 0.0], [rng.uniform(-np.pi, np.pi)], rng.uniform(-0.3, 0.3, n - 1) * limit))
        return q, np.concatenate((q, [1.0, 1.0], np.zeros(n + 2)))

    def test_rk4_matches_dm_control_model_at_rk4(self):
        """Our environment at 1 ms RK4 (the swimmers' physics before 2026-11-10) against
        dm_control's model stepped by MuJoCo RK4 at 1 ms, through the joint limits (a
        strong stroke folds the joints past their ranges)."""
        for name in TESTBEDS:
            with self.subTest(testbed=name):
                n = LINKS[name]
                env = tasks.make_env(rk4_config(name), decision_mode="torque", seed=0)
                model = load_dm(name).physics.model.ptr
                model.opt.timestep = 0.001
                model.opt.integrator = mujoco.mjtIntegrator.mjINT_RK4
                data = mujoco.MjData(model)
                q, state = self.start(n, np.random.default_rng(9))
                env.reset(options={"state": state})
                data.qpos[:] = q
                limited = 0
                for torque in open_loop_torques(n, 2000, 0.001, square=True):
                    data.ctrl[:] = torque / GEAR
                    mujoco.mj_step(model, data)
                    transition = env.step_dt(torque, 0.001)
                    self.assertFalse(transition["terminated"])
                    ours = transition["next_state"]
                    theirs = np.concatenate((data.qpos, data.qvel))
                    theirs[2] = wrap(theirs[2])
                    np.testing.assert_allclose(np.concatenate((ours[: n + 2], ours[n + 4 :])), theirs, rtol=0, atol=1e-10)
                    limited += int(data.nefc > 0)
                self.assertGreater(limited, 50)

    def test_euler_matches_dm_control_stepping(self):
        """With dm_control's 2 ms Euler (the table's physics since 2026-11-10) our
        environment reproduces env.step of dm_control; dm_control's model and ours have
        no dof damping, so MuJoCo's Euler has no implicit damping term in either."""
        for name in TESTBEDS:
            with self.subTest(testbed=name):
                n = LINKS[name]
                env = tasks.make_env(euler_config(name), decision_mode="torque", seed=0)
                self.assertEqual(env.model.opt.integrator, mujoco.mjtIntegrator.mjINT_EULER)
                self.assertEqual(env.model.opt.timestep, 0.002)
                np.testing.assert_array_equal(env.model.dof_damping, 0.0)
                dm_model = load_dm(name).physics.model
                np.testing.assert_array_equal(np.asarray(dm_model.dof_damping), 0.0)
                self.assertEqual(int(dm_model.opt.integrator), mujoco.mjtIntegrator.mjINT_EULER.value)
                self.assertEqual(float(dm_model.opt.timestep), 0.002)
                dm_env = load_dm(name)
                physics = dm_env.physics
                q, state = self.start(n, np.random.default_rng(10))
                env.reset(options={"state": state})
                with physics.reset_context():
                    physics.data.qpos[:] = q
                    physics.data.qvel[:] = 0.0
                past = 0
                # One dm_control step is 15 physics steps of 2 ms under a held control.
                for torque in open_loop_torques(n, 60, 0.03, square=True):
                    transition = env.step_dt(torque, 0.03)
                    dm_env.step(torque / GEAR)
                    self.assertEqual(env.last_interval["physics_steps"], 15)
                    ours = transition["next_state"]
                    theirs = np.concatenate((physics.data.qpos, physics.data.qvel))
                    theirs[2] = wrap(theirs[2])
                    np.testing.assert_allclose(np.concatenate((ours[: n + 2], ours[n + 4 :])), theirs, rtol=0, atol=1e-10)
                    past += int(np.any(np.abs(physics.data.qpos[3:]) > swimmer_joint_range(n)))
                self.assertGreater(past, 3)

    def test_rk4_against_dm_control_euler_loosely(self):
        """Our 1 ms RK4 (before 2026-11-10) against dm_control's own 2 ms Euler: close, not equal."""
        for name in TESTBEDS:
            with self.subTest(testbed=name):
                n = LINKS[name]
                env = tasks.make_env(rk4_config(name), decision_mode="torque", seed=0)
                dm_env = load_dm(name)
                physics = dm_env.physics
                q, state = self.start(n, np.random.default_rng(11))
                env.reset(options={"state": state})
                with physics.reset_context():
                    physics.data.qpos[:] = q
                    physics.data.qvel[:] = 0.0
                worst, steps = 0.0, 0
                for torque in open_loop_torques(n, 34, 0.03, scale=0.4):
                    transition = env.step_dt(torque, 0.03)
                    dm_env.step(torque / GEAR)
                    ours = transition["next_state"][: n + 2]
                    theirs = np.array(physics.data.qpos)
                    theirs[2] = wrap(theirs[2])
                    worst = max(worst, float(np.max(np.abs(ours - theirs))))
                    steps += 1
                self.assertEqual(steps, 34)
                self.assertGreater(worst, 1e-6)  # different integrators
                self.assertLess(worst, 0.05)  # rad and m, over 1 s


# --------------------------------------------------------------------- model integrator


class TestModelIntegrator(unittest.TestCase):
    """The oracle's interval map against the environment over 30 ms intervals.

    The table's convention since 2026-11-10: dm_control's 2 ms Euler physics and the
    semi_implicit map at the 2 ms model step (15 substeps), MuJoCo's Euler step since no
    joint has damping, with the feedback law once per step at its start in both. The
    convention before: 1 ms RK4 physics and the rk4 map (30 substeps). With the limits
    active MuJoCo's Newton solver stops at its tolerance (1e-8 on the cost, dm_control's
    default and ours), which leaves up to about 5e-9 in an RK4 endpoint and, at 2 ms
    Euler, where each step's constraint force enters the velocity directly, up to 3.8e-6
    (swimmer15; 2.4e-13 on swimmer6; 2000 rows, cluster job 1135351); with the tolerance
    at 1e-12 the solve is exact and the endpoints agree to 1e-11 in both.

    MuJoCo's Euler step treats the quadratic fluid drag explicitly, so it is unstable for
    a link moving fast enough (the drag's velocity derivative 600 |u| per second on a
    0.01 kg link against 2 / h = 1000 per second: |u| above about 1.7 m/s); a start with
    joints far past their ranges and a hard stroke can get there (1 of the 16 swimmer15
    rows below). The simulator then flags non-finite or huge accelerations
    (mjWARN_BADQACC) and resets; the semi_implicit map diverges on the same row, since it
    is the same step. Such rows are checked to diverge in the model and left out of the
    agreement; at least TRIALS - 2 rows must remain.
    """

    TRIALS = 16

    def test_semi_implicit_interval_map_is_the_euler_simulator(self):
        for name in TESTBEDS:
            for tolerance, bound in ((None, 1e-5), (1e-12, 1e-10)):
                with self.subTest(testbed=name, solver_tolerance=tolerance), Float64():
                    self.check(name, tolerance, bound, euler_config(name), "semi_implicit", 15)

    def test_rk4_interval_map_is_the_simulator(self):
        for name in TESTBEDS:
            for tolerance, bound in ((None, 1e-8), (1e-12, 1e-10)):
                with self.subTest(testbed=name, solver_tolerance=tolerance), Float64():
                    self.check(name, tolerance, bound, rk4_config(name), "rk4", 30)

    def check(self, name, tolerance, bound, config, integrator, steps):
        n = LINKS[name]
        env = tasks.make_env(config, decision_mode="feedback", seed=0)
        if tolerance is not None:
            env.model.opt.tolerance = tolerance
        plant = tasks.make_oracle_plant(env).double()
        self.assertEqual(round(config["reference_period"] / env.physics_dt), steps)
        rng = np.random.default_rng(12)
        low, high = env.action_space.low, env.action_space.high
        m = n - 1
        rows = []
        while len(rows) < self.TRIALS:
            stiff = len(rows) % 2 == 0
            # Half the trials start with joints past their ranges (limit forces on).
            state = random_states(config, rng, 1, speed=0.3, past=0.3 if len(rows) % 4 < 2 else 0.0)[0]
            decision = rng.uniform(low, high)
            if stiff:
                decision[:m] = 0.05 * rng.uniform(-1.0, 1.0, m) * high[:m]
                decision[m : 2 * m] = 0.02 * high[m : 2 * m] * rng.uniform(0.0, 1.0, m)
                decision[2 * m :] = high[2 * m :] * rng.uniform(0.9, 1.0, m)
                state[n + 4 :] *= 0.2
            env.reset(options={"state": state})
            warnings = int(env.data.warning[mujoco.mjtWarning.mjWARN_BADQACC].number)
            transition = env.step_dt(decision, config["reference_period"])
            unstable = int(env.data.warning[mujoco.mjtWarning.mjWARN_BADQACC].number) > warnings
            if transition["terminated"]:
                continue
            rows.append((state, decision, transition["next_state"], unstable))
        stable = torch.as_tensor([not row[3] for row in rows])
        self.assertGreaterEqual(int(stable.sum()), self.TRIALS - 2)
        states, decisions, observed = (torch.as_tensor(np.array([row[k] for row in rows])) for k in range(3))
        durations = torch.full((self.TRIALS, 1), config["reference_period"], dtype=torch.float64)
        means = interval_means(env, plant, states, decisions, durations, integrator)
        # Where the simulator's own step went unstable, the model's (the same map) diverges.
        for row in torch.nonzero(~stable).reshape(-1).tolist():
            rates = means[row, n + 4 :]
            self.assertTrue(bool((~torch.isfinite(rates)).any() or rates.abs().max() > 1e6), row)
        states, decisions, observed, durations, means = (
            tensor[stable] for tensor in (states, decisions, observed, durations, means)
        )
        residual = wrapped_endpoint_residual(observed, means, env.state_spec)
        self.assertLess(residual.abs().max().item(), bound)
        np.testing.assert_array_equal(means[:, n + 2 : n + 4].numpy(), states[:, n + 2 : n + 4].numpy())
        # The noise-free rollout path (the rollout targets' and gate paths' map) as well.
        paths = stochastic_rollout(
            plant, env.controller, states, decisions, durations,
            lambda s: torch.zeros(s.shape[0], dtype=torch.bool),
            maximum_step=env.physics_dt, physics_dt=env.physics_dt,
            controller_refresh_steps=env.controller.refresh_steps, path_count=1,
            integrator=integrator,
        )
        residual = wrapped_endpoint_residual(observed, paths["endpoints"][:, 0], env.state_spec)
        self.assertLess(residual.abs().max().item(), bound)


class TestNativeEuler(unittest.TestCase):
    """The swimmers on dm_control's 2 ms Euler (since 2026-11-10): the semi_implicit map's
    closed-loop Jacobian with the swimmer plant's options, and a stability smoke test."""

    def test_semi_implicit_closed_loop_jacobian_with_the_plant_options(self):
        """augmented_closed_loop_step under the semi_implicit map and the continuous law
        (refreshed at every substep) against central differences of the map
        s -> semi_implicit_step(s, g(s)), for a learned plant with velocity-dependent
        damping D(y, v) and the limit port (joints inside, in the port's ramp and past
        it), at the 2 ms model step: the propagated plant-fit covariance's Jacobian."""
        for name in TESTBEDS:
            with self.subTest(testbed=name), Float64():
                config = table_config(name)
                n = LINKS[name]
                env = tasks.make_env(config, seed=0)
                controller = env.controller
                self.assertTrue(controller.continuous)
                plant = perturbed_plant(
                    config["state_spec"], network_config(damping_velocity=True, limit_port=True,
                                                         damping_rank=n + 2), 3
                )
                rng = np.random.default_rng(15)
                states = torch.as_tensor(random_states(config, rng, 6, speed=0.5, past=0.4))
                low, high = env.action_space.low, env.action_space.high
                decisions = torch.as_tensor(rng.uniform(low, high, (6, low.size)))
                resolved = controller.resolve_decision(states, decisions)
                h = torch.full((6,), 0.002, dtype=torch.float64)
                held = torch.zeros(6, n - 1, dtype=torch.float64)
                mask = torch.zeros(6, dtype=torch.bool)  # forced to every substep under continuous
                following, torque, jacobian = augmented_closed_loop_step(
                    plant, controller, states, held, resolved, h, mask, integrator="semi_implicit"
                )

                def step(points):
                    return semi_implicit_step(plant, points, controller.action(points, resolved), h)

                # drift_jacobian's drift value equals plant.drift to rounding (not bitwise).
                torch.testing.assert_close(following, step(states), rtol=1e-12, atol=1e-14)
                torch.testing.assert_close(torque, controller.action(states, resolved), rtol=0, atol=0)
                size = env.state_dim
                epsilon = 1e-6
                numeric = torch.stack(
                    [
                        (step(states + epsilon * e) - step(states - epsilon * e)) / (2.0 * epsilon)
                        for e in torch.eye(size, dtype=torch.float64)
                    ],
                    dim=-1,
                )
                scale = 1.0 + numeric.abs().max().item()
                error = (jacobian[:, :size, :size] - numeric).abs().max().item() / scale
                # Central differences: the drag |v| v and the port's ramp are C1 only.
                self.assertLess(error, 2e-5)

    def test_uniform_decisions_stay_bounded_at_euler_2ms(self):
        """Two whole episodes of decisions uniform in the box per testbed at the table's
        2 ms Euler: every state finite, the rates bounded (the closed loop at the box's
        stiffness and damping maxima is inside the semi-implicit step's stability region;
        the measurement over 2000 intervals and 20 episodes is
        benchmarks/ph_contact_substep.py swimmer)."""
        for name in TESTBEDS:
            with self.subTest(testbed=name):
                config = table_config(name)
                env = tasks.make_env(config, seed=21)
                rng = np.random.default_rng(22)
                low, high = env.action_space.low, env.action_space.high
                fastest = 0.0
                for episode in range(2):
                    env.reset(seed=100 + episode)
                    while not env.done:
                        transition = env.step_dt(rng.uniform(low, high), env.sample_duration())
                        state = transition["next_state"]
                        self.assertTrue(np.isfinite(state).all())
                        fastest = max(fastest, float(np.abs(state[env.position_dim :]).max()))
                self.assertLess(fastest, 100.0)


# ------------------------------------------------------------------------------- table


class TestTable(unittest.TestCase):
    def test_rows_build_and_list(self):
        self.assertTrue(set(TESTBEDS) <= set(registered_testbeds(implemented_only=True)))
        lines = run_ph_sac.list_table()
        for name, env_id in ENV_IDS.items():
            with self.subTest(testbed=name):
                proto = run_ph_sac.load_protocol_from_table(env_id)
                self.assertEqual(proto["testbed"], name)
                (line,) = [line for line in lines if line.startswith(env_id + " ")]
                self.assertTrue(line.endswith("builds"), line)

    def test_box_and_settings(self):
        for name in TESTBEDS:
            with self.subTest(testbed=name):
                n = LINKS[name]
                entry = table_entry(name)
                config = table_config(name)
                self.assertEqual(table.declared_testbed_mismatches(entry.task, config), [])
                spec = config["state_spec"]
                self.assertEqual((spec["position_dim"], spec["velocity_dim"], spec["action_dim"]), (n + 4, n + 2, n - 1))
                self.assertEqual(spec["context_dim"], 2)
                self.assertEqual(tuple(spec["periodic_position_indices"]), (2,))
                self.assertEqual(tuple(spec["actuated_position_indices"]), tuple(range(3, n + 2)))
                self.assertEqual(tuple(spec["cyclic_position_indices"]), (0, 1, n + 2, n + 3))
                offset = 2.0 * math.pi / n
                self.assertEqual(
                    tuple(spec["limited_position_ranges"]),
                    tuple((i, -offset, offset) for i in range(3, n + 2)),
                )
                k_max = 10.0 * 5e-4 / offset
                controller = config["controller"]
                np.testing.assert_array_equal(controller["decision_low"], [-offset] * (n - 1) + [0.0] * (2 * (n - 1)))
                np.testing.assert_array_equal(controller["decision_high"], [offset] * (n - 1) + [k_max] * (n - 1) + [0.2 * k_max] * (n - 1))
                np.testing.assert_array_equal(controller["torque_high"], [5e-4] * (n - 1))
                self.assertEqual((controller["setpoint_mode"], controller["setpoint_error"]), ("offset", "unwrapped"))
                self.assertEqual(decision_box_entropy_target(controller), -3.0 * (n - 1))
                row = entry.config
                self.assertEqual(row["temperature"]["target_entropy"], {6: -15.0, 15: -42.0}[n])
                self.assertIs(row["network"]["damping_velocity"], True)
                self.assertIs(row["network"]["limit_port"], True)
                self.assertEqual(row["network"]["damping_rank"], n + 2)
                self.assertEqual(row["network"]["diffusion"], "none")
                # dm_control's physics, 2 ms Euler, and the model on that very map: the
                # semi_implicit step at the physics step (since 2026-11-10; 1 ms RK4 and rk4
                # before), no contact substepping.
                self.assertEqual(row["model_integrator"], "semi_implicit")
                self.assertEqual(row["model_step"], config["physics_dt"])
                self.assertNotIn("model_coarse_step", row)
                self.assertEqual((config["physics_dt"], config["integrator"]), (0.002, "Euler"))
                self.assertEqual(round(config["reference_period"] / config["physics_dt"]), 15)
                self.assertEqual((config["reference_period"], config["episode_duration"], config["discount_rate"]), (0.03, 30.0, 0.5))
                self.assertEqual(config["failure_limits"], {})
                plant = config["plant_parameters"]
                self.assertEqual(
                    [plant[k] for k in ("limit_timeconst", "limit_dampratio", "limit_d0", "limit_dmax",
                                        "limit_width", "limit_midpoint", "limit_power")],
                    [0.05, 1.0, 0.0, 0.8, 0.1, 0.5, 2.0],
                )
                self.assertIn("Joint limits KEPT", entry.comment)

    def test_factory_rejects_a_wrong_link_count_or_range(self):
        settings = copy.deepcopy(table_entry("swimmer_swimmer6").config["testbed_settings"])
        settings["plant_parameters"]["n_links"] = 15
        with self.assertRaises(ValueError):
            testbed_spec("swimmer_swimmer6").config_factory(**settings)
        for key, value in (("failure_limits", {"joint_angle": 1.0}), ("reset_parameters", None)):
            settings = copy.deepcopy(table_entry("swimmer_swimmer6").config["testbed_settings"])
            if value is None:
                settings["reset_parameters"]["joint_range"] = 1.0
            else:
                settings[key] = value
            with self.assertRaises(ValueError):
                testbed_spec("swimmer_swimmer6").config_factory(**settings)

    def test_other_rows_keep_their_plant(self):
        for entry in table.load_ph_table():
            if entry.env_id.startswith("swimmer-"):
                continue
            self.assertNotIn("damping_velocity", entry.config["network"], entry.env_id)
            # Since 2026-11-10 the cart-pole, reacher and point-mass rows keep dm_control's
            # soft joint limits too and set the limit port (tests/test_ph_dmc_joint_limits.py).
            if entry.env_id.startswith(("cartpole-", "reacher-", "point_mass-")):
                continue
            self.assertNotIn("limit_port", entry.config["network"], entry.env_id)


# --------------------------------------------------------------- learned plant options


def network_config(**changes):
    config = {
        "hidden_sizes": (16, 16),
        "plant_hidden_sizes": (32, 32),
        "activation": "tanh",
        "mass_floor": 0.01,
        "diffusion_floor": 0.001,
        "damping_rank": 8,
        "velocity_feature_scale": 2.0 * math.pi,
        "diffusion": "none",
    }
    config.update(changes)
    return config


def perturbed_plant(spec, network, seed):
    """A plant with random output layers, so that U, M and D vary over the state, and,
    with the limit port, random port constants of order one."""
    torch.manual_seed(seed)
    plant = PortHamiltonianPlant(spec, network, learn_diffusion=False).double()
    with torch.no_grad():
        for net in (plant.potential_net, plant.mass_net, plant.damping_net):
            net.layers[-1].weight.normal_(0.0, 0.3)
        if plant.limit_port:
            for parameter in (plant.limit_stiffness, plant.limit_damping, plant.limit_width):
                parameter.normal_(-1.0, 0.5)
    return plant


class TestPlantOptions(unittest.TestCase):
    def spec(self, *, cyclic=True):
        spec = copy.deepcopy(table_config("swimmer_swimmer6")["state_spec"])
        if not cyclic:
            spec.pop("cyclic_position_indices")
        return spec

    def test_inputs_of_the_networks(self):
        with Float64():
            plain = PortHamiltonianPlant(self.spec(cyclic=False), network_config(), learn_diffusion=False)
            # x, y, sin/cos psi, 5 joints, the 2 context coordinates.
            self.assertEqual(plain.damping_net.layers[0].in_features, 11)
            self.assertFalse(plain.damping_velocity)
            cyclic = PortHamiltonianPlant(self.spec(), network_config(), learn_diffusion=False)
            for net in (cyclic.potential_net, cyclic.mass_net, cyclic.damping_net):
                self.assertEqual(net.layers[0].in_features, 7)
            velocity = PortHamiltonianPlant(self.spec(), network_config(damping_velocity=True), learn_diffusion=False)
            self.assertEqual(velocity.damping_net.layers[0].in_features, 7 + 8)
            self.assertEqual(velocity.mass_net.layers[0].in_features, 7)
            bad = self.spec()
            bad["cyclic_position_indices"] = (12,)
            with self.assertRaises(ValueError):
                PortHamiltonianPlant(bad, network_config(), learn_diffusion=False)

    def test_cyclic_positions_do_not_move_the_plant(self):
        with Float64():
            plant = perturbed_plant(self.spec(), network_config(damping_velocity=True), 0)
            states = torch.randn(16, 18)
            actions = torch.randn(16, 5)
            shifted = states.clone()
            shifted[:, [0, 1, 8, 9]] += torch.randn(16, 4) * 3.0
            torch.testing.assert_close(plant.drift(states, actions), plant.drift(shifted, actions), rtol=0, atol=0)
            _, by_state, _ = plant.drift_jacobian(states, actions)
            self.assertEqual(by_state[:, :, [0, 1, 8, 9]].abs().max().item(), 0.0)

    def test_velocity_damping_is_dissipative_and_velocity_dependent(self):
        with Float64():
            plant = perturbed_plant(self.spec(), network_config(damping_velocity=True), 1)
            positions = torch.randn(32, 10)
            velocities = torch.randn(32, 8)
            damping = plant.damping_matrix(positions, velocities)
            self.assertGreaterEqual(torch.linalg.eigvalsh(damping).min().item(), -1e-12)
            other = plant.damping_matrix(positions, 3.0 * velocities)
            self.assertGreater((other - damping).abs().max().item(), 1e-3)
            with self.assertRaises(ValueError):
                plant.damping_matrix(positions)
            torch.testing.assert_close(
                plant.damping_factor(positions, velocities) @ plant.damping_factor(positions, velocities).mT,
                damping,
            )

    def test_analytic_jacobian_matches_torch_func(self):
        """drift_jacobian (with dD/dv and the limit port) against vmap(jacfwd(drift)), with
        and without each option; the joints of randn states lie inside, in the port's ramp
        and beyond it."""
        for velocity, cyclic, port in (
            (True, True, True), (True, True, False), (False, True, True),
            (True, False, False), (False, False, False), (False, False, True),
        ):
            with self.subTest(damping_velocity=velocity, cyclic=cyclic, limit_port=port), Float64():
                plant = perturbed_plant(
                    self.spec(cyclic=cyclic), network_config(damping_velocity=velocity, limit_port=port), 2
                )
                states = torch.randn(8, 18)
                actions = torch.randn(8, 5)
                values, by_state, by_action = plant.drift_jacobian(states, actions)
                torch.testing.assert_close(values, plant.drift(states, actions), rtol=1e-12, atol=1e-12)

                def row(state, action):
                    return plant.drift(state.unsqueeze(0), action.unsqueeze(0)).squeeze(0)

                expected = torch.func.vmap(torch.func.jacfwd(row, argnums=(0, 1)))(states, actions)
                torch.testing.assert_close(by_state, expected[0], rtol=1e-10, atol=1e-10)
                torch.testing.assert_close(by_action, expected[1], rtol=1e-10, atol=1e-10)

    def test_velocity_damping_fits_the_swimmer_better(self):
        """A short fit of the swimmer's drift: D(y, v) beats D(y) at the same budget.

        Each position appears at the speeds +-0.3, +-1 and +-3 times one velocity, and the
        loss weighs each speed block by its own scale. The drag is odd and quadratic in
        the speed; D(y) v is linear in it and the learned centrifugal terms are even, so
        D(y) stalls near the best linear fit of s|s| (about 0.4 of the loss), while
        D(y, v) can represent it.
        """
        config = table_config("swimmer_swimmer6")
        env = tasks.make_env(config, seed=0)
        rng = np.random.default_rng(13)
        speeds = (-3.0, -1.0, -0.3, 0.3, 1.0, 3.0)
        count = 128
        with Float64():
            oracle = tasks.make_oracle_plant(env).double()
            base = random_states(config, rng, count, speed=0.5)
            states = np.concatenate([base.copy() for _ in speeds])
            for block, factor in enumerate(speeds):
                states[block * count : (block + 1) * count, 10:] *= factor
            states = torch.as_tensor(states)
            actions = torch.as_tensor(rng.uniform(-GEAR, GEAR, (len(states), 5)))
            target = oracle.drift(states, actions)[:, 10:].unflatten(0, (len(speeds), count))
            scale = target.square().mean(1, keepdim=True).sqrt()
            losses = {}
            for velocity in (False, True):
                torch.manual_seed(0)
                plant = PortHamiltonianPlant(
                    config["state_spec"],
                    network_config(damping_velocity=velocity, mass_floor=1e-3),
                    learn_diffusion=False,
                ).double()
                plant.initialize_from_physics(torch.as_tensor(env.oracle_input_matrix()), torch.ones(8))
                optimizer = torch.optim.Adam(plant.parameters(), lr=1e-2)
                for _ in range(FIT_STEPS):
                    optimizer.zero_grad()
                    drift = plant.drift(states, actions)[:, 10:].unflatten(0, (len(speeds), count))
                    loss = ((drift - target) / scale).square().mean()
                    loss.backward()
                    optimizer.step()
                losses[velocity] = float(loss.detach())
            self.assertGreater(losses[False], 0.3, losses)
            self.assertLess(losses[True], 0.5 * losses[False], losses)


    def test_limit_port_is_zero_inside_the_ranges(self):
        """With the same seed, a plant with the port equals one without it inside the
        ranges (drift and Jacobian, bitwise), and differs past them."""
        spec = self.spec()
        with Float64():
            torch.manual_seed(5)
            plain = PortHamiltonianPlant(spec, network_config(damping_velocity=True), learn_diffusion=False).double()
            torch.manual_seed(5)
            port = PortHamiltonianPlant(spec, network_config(damping_velocity=True, limit_port=True), learn_diffusion=False).double()
            self.assertEqual(
                {k for k in port.state_dict()} - {k for k in plain.state_dict()},
                {"limit_stiffness", "limit_damping", "limit_width", "_limit_positions",
                 "_limit_velocities", "_limit_low", "_limit_high", "_limit_position_map",
                 "_limit_velocity_map"},
            )
            for key, value in plain.state_dict().items():
                self.assertTrue(torch.equal(value, port.state_dict()[key]), key)
            states = torch.randn(64, 18)
            states[:, 3:8] = torch.rand(64, 5) * 2.0 - 1.0  # inside +-pi/3
            actions = torch.randn(64, 5)
            self.assertTrue(torch.equal(plain.drift(states, actions), port.drift(states, actions)))
            for left, right in zip(plain.drift_jacobian(states, actions), port.drift_jacobian(states, actions)):
                self.assertTrue(torch.equal(left, right))
            terms = port.limit_terms(states[:, :10])
            for key, value in terms.items():
                self.assertEqual(value.abs().max().item(), 0.0, key)
            past = states.clone()
            past[:, 3] = math.pi / 3 + 0.05
            self.assertGreater((plain.drift(past, actions) - port.drift(past, actions)).abs().max().item(), 1e-6)
            bad = copy.deepcopy(spec)
            bad.pop("limited_position_ranges")
            with self.assertRaises(ValueError):
                PortHamiltonianPlant(bad, network_config(limit_port=True), learn_diffusion=False)

    def test_limit_port_energy_and_passivity(self):
        """V is the potential of the port's force, joins U and H, and the port's damping is
        nonnegative, so it only dissipates; the force is continuous at the range and its
        stiffness ramps to k over the width."""
        with Float64():
            plant = perturbed_plant(self.spec(), network_config(limit_port=True), 7)
            positions = torch.randn(32, 10).requires_grad_(True)
            terms = plant.limit_terms(positions)
            (gradient,) = torch.autograd.grad(terms["potential"].sum(), positions)
            expected = torch.zeros_like(gradient)
            expected[:, 3:8] = terms["gradient"]
            torch.testing.assert_close(gradient, expected, rtol=1e-12, atol=1e-14)
            self.assertTrue(bool((terms["damping"] >= 0.0).all()))
            self.assertTrue(bool((terms["curvature"] >= 0.0).all()))
            # U + V in potential_energy, and in the Hamiltonian.
            plain = plant.potential_net(plant.position_features(positions))[0]
            torch.testing.assert_close(plant.potential_energy(positions), plain + terms["potential"])
            momenta = torch.randn(32, 8)
            kinetic = plant.hamiltonian(positions, momenta) - plant.potential_energy(positions)
            solved = torch.cholesky_solve(momenta.unsqueeze(-1), plant.mass_factor(positions))
            torch.testing.assert_close(kinetic, 0.5 * (momenta.unsqueeze(-2) @ solved).squeeze(-1))
            # The force is continuous across the range and the ramp's end.
            width = 1e-3 + torch.nn.functional.softplus(plant.limit_width.detach())
            limit = math.pi / 3
            for depth in (torch.zeros(5), width):
                below = torch.zeros(2, 10)
                below[:, 3:8] = limit + depth - 1e-9
                below[1, 3:8] = limit + depth + 1e-9
                force = plant.limit_terms(below)["gradient"]
                self.assertLess((force[0] - force[1]).abs().max().item(), 1e-6)
            deep = torch.zeros(1, 10)
            deep[:, 3:8] = -(limit + 2.0 * width)
            stiffness = plant.limit_stiffness.detach().exp()
            torch.testing.assert_close(plant.limit_terms(deep)["curvature"][0], stiffness)
            self.assertTrue(bool((plant.limit_terms(deep)["gradient"] < 0.0).all()))

    def test_limit_port_fits_past_the_limits(self):
        """A short fit of the swimmer's drift at rest, joints inside and past their ranges:
        with the port the held-out error past the limits falls far below the error of the
        same plant without it.

        At rest the drift is M(q)^-1 (G u + limit force), so the test isolates what the
        port is for. Both plants get the true mass matrix (their mass network replaced by
        the oracle's Cholesky factor; at rest its derivatives do not enter the drift) and
        the true input matrix, and fit their potential network, and the port, to the
        oracle's drift.
        """
        config = table_config("swimmer_swimmer6")
        env = tasks.make_env(config, seed=0)
        dynamics = env.dynamics
        rng = np.random.default_rng(14)
        with Float64():
            oracle = tasks.make_oracle_plant(env).double()

            def sample(count):
                states = random_states(config, rng, count, speed=0.0, past=0.25)
                actions = rng.uniform(-GEAR, GEAR, (count, 5))
                states, actions = torch.as_tensor(states), torch.as_tensor(actions)
                return states, actions, oracle.drift(states, actions)[:, 10:]

            train, train_actions, train_target = sample(1024)
            test, test_actions, test_target = sample(1024)
            scale = train_target.square().mean(0).sqrt()
            past = (test[:, 3:8].abs() > math.pi / 3).any(-1)
            self.assertGreater(int(past.sum()), 500)
            self.assertGreater(int((~past).sum()), 150)

            def true_mass_jets(features, first, second):
                # Features: sin psi, cos psi, the five joint angles (x, y, target cyclic).
                yaw = torch.atan2(features[:, 0:1], features[:, 1:2])
                positions = torch.cat((torch.zeros_like(features[:, :2]), yaw, features[:, 2:7]), dim=-1)
                mass = dmc_swimmer.swimmer_terms(dynamics, positions, torch.zeros_like(positions))["mass"]
                tangent = None if first is None else mass.new_zeros(len(mass), first.shape[1], 8, 8)
                return torch.linalg.cholesky(mass), tangent, None

            errors = {}
            for label, port in (("without", False), ("with", True)):
                torch.manual_seed(0)
                network = network_config(damping_velocity=True, limit_port=port)
                plant = PortHamiltonianPlant(config["state_spec"], network, learn_diffusion=False).double()
                plant._mass_jets = true_mass_jets
                plant.force_matrix.data.copy_(torch.as_tensor(env.oracle_input_matrix()))
                trained = [p for name, p in plant.named_parameters() if not name.startswith(("mass_net", "force_matrix"))]
                optimizer = torch.optim.Adam(trained, lr=1e-2)
                for _ in range(PORT_FIT_STEPS):
                    optimizer.zero_grad()
                    loss = ((plant.drift(train, train_actions)[:, 10:] - train_target) / scale).square().mean()
                    loss.backward()
                    optimizer.step()
                with torch.no_grad():
                    error = ((plant.drift(test, test_actions)[:, 10:] - test_target) / scale).square().mean(-1)
                errors[label] = (float(error[past].mean()), float(error[~past].mean()), float(loss))
            self.assertLess(errors["with"][0], PORT_GAIN * errors["without"][0], errors)


FIT_STEPS = 800
PORT_FIT_STEPS = 800
PORT_GAIN = 0.35


# ------------------------------------------------------------------------- regression

# Values computed with the code before network damping_velocity, cyclic_position_indices
# and the swimmer existed (2026-11-09, a copy of the source tree as it was, with the
# functions below): a default plant of each row in float64 (sum and sum of squares of
# drift, drift_jacobian and the damping matrix), 40 Adam steps of that plant on a loss of
# its drift and Jacobian (parameter sum, sum of squares, last loss), and 300-decision tiny
# runs of two torque-decision arms (they exercise the learned plant's fit and rollouts and
# hold their torque, so they do not depend on how a feedback law is refreshed; regenerated
# 2026-11-10, see REFERENCE_ARMS). The plants are built without limit_port (default_network). The full
# check at the change, every paper arm and implicit_torque on cartpole-swingup and
# reacher-easy for 300 decisions, gave identical SHA-256 digests of every network.
REFERENCE_PLANTS = {'acrobot-swingup': {'da': [32.038232519264696, 32.061871651945545],
                     'damping': [0.6478789132248656, 0.006589433028535152],
                     'drift': [4.302332950768875, 105.94522199860967],
                     'ds': [63.34349478931247, 64.007381061692],
                     'plain_drift': [4.302332950768875, 105.94522199860967]},
 'cartpole-swingup': {'da': [32.005708524055926, 31.990647979384356],
                      'damping': [0.6540507435346388, 0.006560202510094757],
                      'drift': [4.264546811553252, 105.87895020880211],
                      'ds': [63.345573190566455, 64.00694242082312],
                      'plain_drift': [4.264546811553252, 105.87895020880211]},
 'lqr-lqr_6_2': {'da': [64.01973979169381, 63.97896491277247],
                 'damping': [1.8802855135357313, 0.019124081221928164],
                 'drift': [22.567763236331135, 274.7693246297832],
                 'ds': [190.2373367879349, 192.03005847062656],
                 'plain_drift': [22.567763236331135, 274.7693246297832]},
 'reacher-easy': {'da': [63.973016010357945, 63.948934210537125],
                  'damping': [0.6386407070165236, 0.006449123016685347],
                  'drift': [15.632880383507686, 148.7674680476042],
                  'ds': [63.35780352664503, 64.00751118674978],
                  'plain_drift': [15.632880383507686, 148.7674680476042]}}
REFERENCE_FITS = {'acrobot-swingup': [19.11967262566035, 176.72491783086397, 1.3964687599172445],
 'cartpole-swingup': [11.431171924435064, 177.91000317047113, 1.4239245591593948],
 'lqr-lqr_6_2': [76.18031156467758, 265.7883402549463, 1.0940682600302667],
 'reacher-easy': [30.52429961341355, 194.3601715631136, 1.3613479234492443]}
# Regenerated 2026-11-10 from the code of that day: the two rows changed convention (the
# cart-pole rail and the reacher wrist keep dm_control's soft limits, no failure set, the
# plant's limit port on), so the 2026-11-09 values no longer apply; these pin the new
# behaviour. Regenerated again later on 2026-11-10 (scratchpad port_init/regen_arms.py),
# for two changes these runs legitimately depend on: the limit port now starts at the
# compiled limit (state spec limit_port_init: the online plant's port parameters, hence
# both rows' online_plant sums, and through the plant's rollouts the reacher's value,
# rate and actor slightly), and cartpole-swingup's model step is back to 1 ms (tasks with
# joint limits keep the physics step: its explicit rollout targets, hence its actor,
# critics and value). Counters are unchanged; the reacher's returns too.
REFERENCE_ARMS = {'cartpole-swingup/explicit_rollout_torque': {'counters': {'actor_gradient_steps': 293,
                                                           'critic_gradient_steps': 293,
                                                           'gate_real_rows_acquired': 0,
                                                           'gate_simulated_paths': 0,
                                                           'interactions': 300,
                                                           'model_rollout_paths': 7032,
                                                           'plant_gradient_steps': 293,
                                                           'rate_gradient_steps': 0,
                                                           'simulated_seconds': 2.99999999999998,
                                                           'temperature_gradient_steps': 293,
                                                           'value_gradient_steps': 293},
                                              'networks': {'actor': [1.424726445125998,
                                                                     214.87778156615963],
                                                           'critics': [3.2399762747372733,
                                                                       225.93750317453706],
                                                           'online_plant': [10.693949182110373,
                                                                            174.0663528534974],
                                                           'value': [0.906717658508569,
                                                                     11.511592496318855]},
                                              'returns': [0.0694627264621497,
                                                          0.13573729289499412]},
 'reacher-easy/implicit_torque': {'counters': {'actor_gradient_steps': 96,
                                               'critic_gradient_steps': 0,
                                               'gate_real_rows_acquired': 0,
                                               'gate_simulated_paths': 0,
                                               'interactions': 300,
                                               'model_rollout_paths': 1536,
                                               'plant_gradient_steps': 297,
                                               'rate_gradient_steps': 96,
                                               'reward_gradient_steps': 297,
                                               'simulated_seconds': 5.99999999999996,
                                               'temperature_gradient_steps': 7,
                                               'value_gradient_steps': 292},
                                  'networks': {'actor': [-1.0392746140132658,
                                                         12.956306602000044],
                                               'online_plant': [-8.995907101460034,
                                                                179.10701813872853],
                                               'rate': [7.7524559701560065,
                                                        12.403068782564462],
                                               'reward': [-3.6987488467639196,
                                                          15.100756117354607],
                                               'value': [-1.5310074771987274,
                                                         12.333757176024335]},
                                  'returns': [0.0015692963273727465, 0.0]}}
TRAINING_TOLERANCE = {"rtol": 1e-7, "atol": 1e-9}


def tiny_protocol(env_id, output_directory, budget=300):
    raw = copy.deepcopy(table.find_ph_entry(env_id, "top").config)
    # The references below predate contact-triggered fine substepping (2026-11-10): with
    # it off every row integrates at the model step, bitwise as the code they pin.
    raw.pop("model_coarse_step", None)
    # And contact exclusion (cartpole-swingup's row since 2026-11-11): they pin the limit
    # port, as the row had it when they were regenerated.
    if raw.pop("contact_handling", None) == "exclude":
        raw["network"]["limit_port"] = True
    raw["network"].update(hidden_sizes=[16, 16], plant_hidden_sizes=[8, 8])
    raw["training"].update(replay_capacity=5000, batch_size=8, warmup_episodes=0)
    raw["explicit"].update(rollout_paths=3)
    implicit = raw["implicit"]
    implicit.update(reward_warm_steps=5, plant_warm_steps=5)
    implicit["solver"].update(policy_draws=3, inner_steps=6)
    implicit["rate"].update(rollout_paths=2, model_points=8, rate_steps=2, actor_steps=2)
    implicit["admission"].update(initial_rows=8, initial_paths=2)
    raw["protocol"].update(
        interaction_budget=budget,
        training_seeds=[0],
        evaluation_reset_seeds=[20000],
        evaluation_every_interactions=budget,
        output_directory=str(output_directory),
    )
    return run_ph_sac.protocol_from_config(raw, f"tiny {env_id}")


def run_tiny_arm(env_id, arm_name, output_directory, budget=300):
    proto = tiny_protocol(env_id, output_directory, budget)
    proto["arms"] = [arm for arm in protocol.all_arms() if arm["name"] == arm_name]
    protocol.validate_matched_quantities(proto)
    (run,) = protocol.build_run_matrix(proto)
    return run, protocol.run_seed(run)


def default_network(config):
    """The row's network without limit_port (on for the cart-pole and reacher rows since
    2026-11-10): the regression is of the default plant, the code before the new keys."""
    network = dict(config["network"])
    network.pop("limit_port", None)
    return network


def plant_sums(env_id):
    config = table.find_ph_entry(env_id, "top").config
    testbed = testbed_spec(config["testbed"]).config_factory(**config["testbed_settings"])
    torch.manual_seed(0)
    plant = PortHamiltonianPlant(testbed["state_spec"], default_network(config), learn_diffusion=False).double()
    generator = torch.Generator().manual_seed(1)
    states = torch.randn(32, plant.state_dim, generator=generator, dtype=torch.float64)
    actions = torch.randn(32, plant.action_dim, generator=generator, dtype=torch.float64)
    drift, ds, da = plant.drift_jacobian(states, actions)
    damping = plant.damping_matrix(states[:, : plant.position_dim])
    return {
        name: [float(value.sum()), float(value.square().sum())]
        for name, value in (("drift", drift), ("ds", ds), ("da", da), ("damping", damping),
                            ("plain_drift", plant.drift(states, actions)))
    }


def plant_fit_sums(env_id, steps=40):
    """Adam steps of a default plant on a loss of its drift and drift_jacobian."""
    config = table.find_ph_entry(env_id, "top").config
    testbed = testbed_spec(config["testbed"]).config_factory(**config["testbed_settings"])
    torch.manual_seed(0)
    plant = PortHamiltonianPlant(testbed["state_spec"], default_network(config), learn_diffusion=False).double()
    generator = torch.Generator().manual_seed(2)
    states = torch.randn(64, plant.state_dim, generator=generator, dtype=torch.float64)
    actions = torch.randn(64, plant.action_dim, generator=generator, dtype=torch.float64)
    target = torch.randn(64, plant.state_dim, generator=generator, dtype=torch.float64)
    optimizer = torch.optim.Adam(plant.parameters(), lr=1e-3)
    for _ in range(steps):
        optimizer.zero_grad()
        drift, ds, _ = plant.drift_jacobian(states, actions)
        loss = (
            (plant.drift(states, actions) - target).square().mean()
            + 1e-3 * ds.square().mean()
            + 1e-3 * drift.square().mean()
        )
        loss.backward()
        optimizer.step()
    flat = torch.cat([p.detach().reshape(-1) for _, p in sorted(plant.state_dict().items())])
    return [float(flat.sum()), float(flat.square().sum()), float(loss.detach())]


class TestExistingRowsUnchanged(unittest.TestCase):
    def test_default_plant_fits_reproduce_the_earlier_code(self):
        for env_id, expected in REFERENCE_FITS.items():
            with self.subTest(env_id=env_id):
                np.testing.assert_allclose(plant_fit_sums(env_id), expected, rtol=1e-10, atol=1e-12)

    def test_default_plants_reproduce_the_earlier_code(self):
        for env_id, expected in REFERENCE_PLANTS.items():
            with self.subTest(env_id=env_id):
                observed = plant_sums(env_id)
                self.assertEqual(set(observed), set(expected))
                for name, values in expected.items():
                    np.testing.assert_allclose(observed[name], values, rtol=1e-12, atol=1e-12, err_msg=name)

    def test_short_runs_reproduce_the_earlier_code(self):
        for key, expected in REFERENCE_ARMS.items():
            env_id, arm = key.split("/")
            with self.subTest(run=key), tempfile.TemporaryDirectory() as tmp:
                _, result = run_tiny_arm(env_id, arm, tmp)
                counters = {k: v for k, v in result["counters"].items() if k != "wall_seconds"}
                self.assertEqual(set(counters), set(expected["counters"]))
                for name, value in expected["counters"].items():
                    if isinstance(value, float):
                        self.assertAlmostEqual(counters[name], value, places=12)
                    else:
                        self.assertEqual(counters[name], value, name)
                (checkpoint,) = list(Path(tmp).rglob("checkpoint.pt"))
                state = torch.load(checkpoint, weights_only=False)
                for name, values in expected["networks"].items():
                    tensors = state["networks"][name]
                    flat = torch.cat([t.detach().double().reshape(-1) for k, t in sorted(tensors.items())
                                      if isinstance(t, torch.Tensor) and t.dtype.is_floating_point])
                    np.testing.assert_allclose([float(flat.sum()), float(flat.square().sum())], values,
                                               **TRAINING_TOLERANCE, err_msg=name)
                returns = [float(e["discounted_return"]) for c in result["evaluations"] for e in c["episodes"]]
                np.testing.assert_allclose(returns, expected["returns"], **TRAINING_TOLERANCE)


# Short swimmer runs at the convention of 2026-11-10 (dm_control's 2 ms Euler physics, the
# semi_implicit map at the 2 ms model step, damping_velocity and the limit port), as
# REFERENCE_ARMS: counters, (sum, sum of squares) of each network's parameters, and the
# evaluation returns of tiny_arm_summary. Generated by benchmarks/ph_swimmer_references.py
# in Vista job 1135351 (2026-11-10, aarch64, 2 threads) from this code and pasted here
# unchanged; a platform with other floating-point kernels may differ at the 1e-7 level, as
# REFERENCE_ARMS does between the workstation and the cluster. A key set to None is skipped
# (pending regeneration).
SWIMMER_BUDGET = 150
REFERENCE_SWIMMER_ARMS = {'swimmer-swimmer6/explicit_rollout': {'counters': {'actor_gradient_steps': 143,
                                                    'critic_gradient_steps': 143,
                                                    'gate_real_rows_acquired': 0,
                                                    'gate_simulated_paths': 0,
                                                    'interactions': 150,
                                                    'model_rollout_paths': 3432,
                                                    'plant_gradient_steps': 143,
                                                    'rate_gradient_steps': 0,
                                                    'simulated_seconds': 4.499999999999993,
                                                    'temperature_gradient_steps': 143,
                                                    'value_gradient_steps': 143},
                                       'networks': {'actor': [-7.7591672639900935,
                                                              42.553515698730266],
                                                    'critics': [0.044955379591556266,
                                                                34.39001835699024],
                                                    'online_plant': [-13.233773985986772,
                                                                     605.7030823491093],
                                                    'target_critics': [0.13323015018249862,
                                                                       34.365624448162755],
                                                    'target_plant': [-12.698899898890431,
                                                                     603.7391668448345],
                                                    'target_value': [-1.5275602357578464,
                                                                     10.787853307267273],
                                                    'value': [-1.275906987953931,
                                                              10.78094661105489]},
                                       'returns': [0.047119935463580516, 0.046683761302347276]},
 'swimmer-swimmer6/implicit_learned_admissions': {'counters': {'actor_gradient_steps': 46,
                                                               'critic_gradient_steps': 0,
                                                               'gate_real_rows_acquired': 0,
                                                               'gate_simulated_paths': 0,
                                                               'interactions': 150,
                                                               'model_rollout_paths': 736,
                                                               'plant_gradient_steps': 147,
                                                               'rate_gradient_steps': 46,
                                                               'reward_gradient_steps': 147,
                                                               'simulated_seconds': 4.499999999999993,
                                                               'temperature_gradient_steps': 3,
                                                               'value_gradient_steps': 142},
                                                  'networks': {'actor': [-8.004264960502042,
                                                                         41.26608246831069],
                                                               'admitted_plant': [-14.313087423375691,
                                                                                  604.5999520168459],
                                                               'online_plant': [-14.455228168255417,
                                                                                606.5774756595995],
                                                               'rate': [-4.082786593076889,
                                                                        22.542401381263964],
                                                               'reward': [-0.051954107999335974,
                                                                          13.115022022104544],
                                                               'reward_frozen': [-0.08875438058748841,
                                                                                 13.05599597342018],
                                                               'target_value': [5.805627790934523,
                                                                                12.18021794010329],
                                                               'value': [5.805627790934523,
                                                                         12.18021794010329]},
                                                  'returns': [0.047119935463580516,
                                                              0.04679216780231688]}}


def tiny_arm_summary(env_id, arm, budget):
    """counters, networks and returns of a tiny run (tiny_protocol), as REFERENCE_ARMS."""
    with tempfile.TemporaryDirectory() as tmp:
        _, result = run_tiny_arm(env_id, arm, tmp, budget)
        counters = {k: v for k, v in result["counters"].items() if k != "wall_seconds"}
        (checkpoint,) = list(Path(tmp).rglob("checkpoint.pt"))
        state = torch.load(checkpoint, weights_only=False)
    networks = {}
    for name, tensors in sorted(state["networks"].items()):
        if not isinstance(tensors, dict):
            continue
        values = [t.detach().double().reshape(-1) for k, t in sorted(tensors.items())
                  if isinstance(t, torch.Tensor) and t.dtype.is_floating_point]
        if values:
            flat = torch.cat(values)
            networks[name] = [float(flat.sum()), float(flat.square().sum())]
    returns = [float(e["discounted_return"]) for c in result["evaluations"] for e in c["episodes"]]
    return {"counters": counters, "networks": networks, "returns": returns}


class TestSwimmerRunsAtTheNativeStep(unittest.TestCase):
    def test_short_swimmer_runs_reproduce_their_references(self):
        for key, expected in REFERENCE_SWIMMER_ARMS.items():
            with self.subTest(run=key):
                if expected is None:
                    self.skipTest(
                        f"{key}: reference pending regeneration on the cluster "
                        "(benchmarks/ph_swimmer_references.py)"
                    )
                env_id, arm = key.split("/")
                observed = tiny_arm_summary(env_id, arm, SWIMMER_BUDGET)
                self.assertEqual(set(observed["counters"]), set(expected["counters"]))
                for name, value in expected["counters"].items():
                    if isinstance(value, float):
                        self.assertAlmostEqual(observed["counters"][name], value, places=12)
                    else:
                        self.assertEqual(observed["counters"][name], value, name)
                self.assertEqual(set(observed["networks"]), set(expected["networks"]))
                for name, values in expected["networks"].items():
                    np.testing.assert_allclose(observed["networks"][name], values,
                                               **TRAINING_TOLERANCE, err_msg=name)
                np.testing.assert_allclose(observed["returns"], expected["returns"],
                                           **TRAINING_TOLERANCE)


if __name__ == "__main__":
    unittest.main()

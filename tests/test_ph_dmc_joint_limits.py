"""dm_control's soft joint limits on the cart-pole slider, the reacher wrist and the point
mass's slides (since 2026-11-10).

Until 2026-11-10 these ports removed the limits and made them failure boundaries
(|x| >= 1.8 m on every cart-pole, |theta_w| >= 160 deg on the reachers,
max(|x|, |y|) >= 0.29 m on the point mass, payoff 0, with the boundary lift). They now
keep the limits as dm_control's MJCF has them, MuJoCo's constraint solver on, the oracle
plants add MuJoCo's constraint force (environment/ph_sac/dmc/joint_limits.py), the rows
set the learned plant's limit port, and no dm_control task has a failure set.

1. Compiled limits: jnt_limited, jnt_range, jnt_solref, jnt_solimp, jnt_margin and
   dof_invweight0 equal dm_control's compiled model's, the solver settings are
   dm_control's, the state spec declares the ranges, and inside every range the
   environment and the oracle are bitwise those of the model without the limit.
2. Constraint rows: MuJoCo's efc_* rows (pos, vel, impedance, aref, R, D, force),
   qfrc_constraint and qacc against joint_limits (torch and numpy), env.oracle_drift and
   the tensor oracle against env.mujoco_drift, limits active and inactive; the oracle
   plant's drift Jacobian against central differences past the limits.
3. Model integrator: the oracle plant's rk4 interval map at the physics step against
   env.step_dt through the limits (MuJoCo's Newton tolerance leaves up to a few 1e-9).
4. dm_control: forward dynamics past the limits, and open-loop trajectories into the
   limits at dm_control's own step (cart-pole 10 ms RK4; reacher and point mass 20 ms
   Euler, where MuJoCo's 2 x timestep floor makes the limit's time constant 0.04 s in
   both models).
5. No failure set: masks, payoffs and lifts; the factories reject the old failure_limits;
   episodes run into the limits, are pushed back and continue.
6. Table and limit port: the rows set network limit_port and the plant's limit keys; the
   learned port is exactly zero inside the ranges; it starts at the compiled limit (state
   spec limit_port_init, joint_limits.limit_port_init: stiffness and damping against
   MuJoCo's efc rows, within 10%), a spec without the key starts at the plant's generic
   constants; and a short fit past the limits beats the same plant without the port, from
   the compiled start at the training learning rate at least as well as from the generic
   start at a high one.
"""

import copy
import math
import unittest

import mujoco
import numpy as np
import torch

from benchmarks import ph_hyperparams as table
from common.ph_dmc._common import LIMIT_KEYS, MUJOCO_DEFAULT_LIMIT, fixed_clock
from environment.ph_sac import tasks
from environment.ph_sac.dmc import cartpole as dmc_cartpole
from environment.ph_sac.dmc import joint_limits
from environment.ph_sac.dmc.chain import chain_callbacks, chain_drift_numpy
from environment.ph_sac.dmc.limited_chain import limited_chain_drift_numpy
from environment.ph_sac.registry import testbed_spec
from models.ph_sac.plant import (
    LIMIT_DAMPING_INIT,
    LIMIT_STIFFNESS_INIT,
    LIMIT_WIDTH_FLOOR,
    LIMIT_WIDTH_INIT,
    PortHamiltonianPlant,
)
from models.ph_sac.rollouts import wrapped_endpoint_residual
from tests.test_ph_model_integrator import interval_means

# env_id: (testbed, dm_control (domain, task), limited joints, their state indices)
ROWS = {
    "cartpole-swingup": ("cartpole_swingup", ("cartpole", "swingup"), ("slider",), (0,)),
    "cartpole-balance": ("cartpole_balance", ("cartpole", "balance"), ("slider",), (0,)),
    "cartpole-balance_sparse": ("cartpole_balance_sparse", ("cartpole", "balance_sparse"), ("slider",), (0,)),
    "cartpole-swingup_sparse": ("cartpole_swingup_sparse", ("cartpole", "swingup_sparse"), ("slider",), (0,)),
    "cartpole-two_poles": ("cartpole_two_poles", ("cartpole", "two_poles"), ("slider",), (0,)),
    "cartpole-three_poles": ("cartpole_three_poles", ("cartpole", "three_poles"), ("slider",), (0,)),
    "reacher-easy": ("reacher_easy", ("reacher", "easy"), ("wrist",), (1,)),
    "reacher-hard": ("reacher_hard", ("reacher", "hard"), ("wrist",), (1,)),
    "point_mass-easy": ("point_mass_easy", ("point_mass", "easy"), ("root_x", "root_y"), (0, 1)),
}
# The rows under contact exclusion (no limit port) since 2026-11-11
# (make_ct_ph_sac_table.CONTACT_HANDLING); the reachers keep the port.
EXCLUDED = tuple(env_id for env_id in ROWS if env_id.startswith(("cartpole-", "point_mass-")))
# One row per distinct dynamics (the others share a measured row's plant).
FAMILIES = ("cartpole-swingup", "cartpole-two_poles", "cartpole-three_poles", "reacher-easy", "point_mass-easy")
GEAR = {"cartpole": 10.0, "reacher": 0.05, "point_mass": 0.1}  # dm_control's ctrl = ours / gear
# The old failure_limits each factory now rejects.
OLD_FAILURE = {
    "cartpole": {"cart_position": 1.8},
    "reacher": {"wrist_angle": math.radians(160.0)},
    "point_mass": {"position": 0.29},
}
# Penetration scales: the impedance width (MuJoCo's default solimp, 1 mm or 1 mrad) and a
# deeper reach for each family.
WIDTH = 1e-3
DEEP = {"cartpole": 0.15, "reacher": 0.1, "point_mass": 0.03}


def domain(env_id):
    return ROWS[env_id][1][0]


def table_config(env_id):
    entry = table.find_ph_entry(env_id, "top")
    return testbed_spec(ROWS[env_id][0]).config_factory(**copy.deepcopy(entry.config["testbed_settings"]))


def load_dm(env_id):
    from dm_control import suite

    name, task = ROWS[env_id][1]
    env = suite.load(name, task, task_kwargs={"random": 0})
    env.reset()
    return env


def env_limits(env, env_id):
    return joint_limits.joint_limits_from_model(env.model, ROWS[env_id][2])


def ranges(config):
    return {int(i): (float(low), float(high)) for i, low, high in config["state_spec"]["limited_position_ranges"]}


def limit_states(env_id, config, rng, count, *, past=0.5, speed=1.0):
    """States inside the ranges, each limited coordinate past its range with probability
    past: by up to the impedance width, up to ten widths, or up to DEEP, a third each."""
    family = domain(env_id)
    spec = config["state_spec"]
    n_o, n = spec["position_dim"], spec["velocity_dim"]
    states = np.zeros((count, n_o + n))
    if family == "cartpole":
        states[:, 0] = rng.uniform(-1.7, 1.7, count)
        states[:, 1:n_o] = rng.uniform(-math.pi, math.pi, (count, n_o - 1))
        states[:, n_o] = rng.uniform(-3.0, 3.0, count) * speed
        states[:, n_o + 1 :] = rng.uniform(-8.0, 8.0, (count, n - 1)) * speed
    elif family == "reacher":
        states[:, 0] = rng.uniform(-math.pi, math.pi, count)
        states[:, 1] = rng.uniform(-2.6, 2.6, count)
        radius, angle = rng.uniform(0.05, 0.2, count), rng.uniform(0.0, 2.0 * math.pi, count)
        states[:, 2], states[:, 3] = radius * np.sin(angle), radius * np.cos(angle)
        states[:, 4:] = rng.uniform(-5.0, 5.0, (count, 2)) * speed
    else:
        states[:, :2] = rng.uniform(-0.28, 0.28, (count, 2))
        states[:, 2:] = rng.uniform(-0.5, 0.5, (count, 2)) * speed
    for index, (low, high) in ranges(config).items():
        beyond = rng.random(count) < past
        kind = rng.integers(0, 3, count)
        reach = np.where(kind == 0, WIDTH, np.where(kind == 1, 10.0 * WIDTH, DEEP[family]))
        depth = rng.uniform(0.0, 1.0, count) * reach
        sign = np.where(rng.random(count) < 0.5, -1.0, 1.0)
        value = np.where(sign > 0, high + depth, low - depth)
        states[:, index] = np.where(beyond, value, states[:, index])
    return states


def random_torques(config, rng, count):
    high = np.asarray(config["controller"]["torque_high"], dtype=np.float64)
    return rng.uniform(-high, high, (count, len(high)))


def scaled_error(left, right):
    """max |left - right| relative to max(1, max |right|)."""
    left, right = np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
    return float(np.max(np.abs(left - right)) / max(1.0, float(np.max(np.abs(right)))))


def past_any(config, states):
    """Boolean [B]: some limited coordinate lies past its range."""
    states = np.atleast_2d(states)
    mask = np.zeros(len(states), dtype=bool)
    for index, (low, high) in ranges(config).items():
        mask |= (states[:, index] > high) | (states[:, index] < low)
    return mask


class Float64:
    def __enter__(self):
        self.previous = torch.get_default_dtype()
        torch.set_default_dtype(torch.float64)

    def __exit__(self, *args):
        torch.set_default_dtype(self.previous)


def set_mujoco(env, data, state, torque):
    physical, n_o = env.physical_position_dim, env.position_dim
    data.qpos[:physical] = state[:physical]
    data.qvel[:] = state[n_o:]
    data.ctrl[:] = torque
    mujoco.mj_forward(env.model, data)


# --------------------------------------------------------------------- compiled limits


class TestCompiledLimits(unittest.TestCase):
    def test_limits_are_dm_controls(self):
        """Every limited joint compiles to dm_control's range, solref, solimp and margin
        (bitwise), with dm_control's dof_invweight0 (the regularizer's diagApprox)."""
        for env_id, (_, _, joints, indices) in ROWS.items():
            with self.subTest(env_id=env_id):
                config = table_config(env_id)
                env = tasks.make_env(config, seed=0)
                ours, theirs = env.model, load_dm(env_id).physics.model.ptr
                np.testing.assert_array_equal(
                    ours.jnt_limited, theirs.jnt_limited[: ours.njnt]
                )
                for joint, index in zip(joints, indices):
                    a = mujoco.mj_name2id(ours, mujoco.mjtObj.mjOBJ_JOINT, joint)
                    b = mujoco.mj_name2id(theirs, mujoco.mjtObj.mjOBJ_JOINT, joint)
                    self.assertEqual(int(ours.jnt_limited[a]), 1)
                    for field in ("jnt_range", "jnt_solref", "jnt_solimp", "jnt_margin"):
                        np.testing.assert_array_equal(getattr(ours, field)[a], getattr(theirs, field)[b], field)
                    self.assertEqual(ours.dof_invweight0[ours.jnt_dofadr[a]], theirs.dof_invweight0[theirs.jnt_dofadr[b]])
                    self.assertEqual(ours.dof_damping[ours.jnt_dofadr[a]], theirs.dof_damping[theirs.jnt_dofadr[b]])
                    # The state spec's range is the compiled one.
                    np.testing.assert_array_equal(ranges(config)[index], ours.jnt_range[a])
                self.assertEqual(sorted(ranges(config)), sorted(indices))
                for model in (ours, theirs):
                    self.assertFalse(model.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_CONSTRAINT)
                    self.assertTrue(model.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_CONTACT)
                self.assertEqual(
                    (ours.opt.solver, ours.opt.iterations, ours.opt.tolerance),
                    (theirs.opt.solver, theirs.opt.iterations, theirs.opt.tolerance),
                )
                self.assertEqual(ours.opt.solver, mujoco.mjtSolver.mjSOL_NEWTON)
                plant = config["plant_parameters"]
                expected = (0.08, 1.0) if domain(env_id) == "cartpole" else (0.02, 1.0)
                self.assertEqual((plant["limit_timeconst"], plant["limit_dampratio"]), expected)
                for key in LIMIT_KEYS[2:]:
                    self.assertEqual(plant[key], MUJOCO_DEFAULT_LIMIT[key], key)

    def test_effective_time_constant(self):
        """MuJoCo floors solref's time constant at 2 x timestep: at our 1 ms the compiled
        values act as they read; at dm_control's 20 ms the reacher's and point mass's 0.02
        s acts as 0.04 s (the cart-pole's 0.08 s at 10 ms is above the floor)."""
        for env_id in ("cartpole-swingup", "reacher-easy", "point_mass-easy"):
            with self.subTest(env_id=env_id):
                env = tasks.make_env(table_config(env_id), seed=0)
                ours = env_limits(env, env_id)
                theirs = joint_limits.joint_limits_from_model(load_dm(env_id).physics, ROWS[env_id][2])
                if domain(env_id) == "cartpole":
                    np.testing.assert_array_equal(ours.timeconst, [0.08])
                    np.testing.assert_array_equal(theirs.timeconst, [0.08])
                else:
                    np.testing.assert_array_equal(ours.timeconst, [0.02] * ours.count)
                    np.testing.assert_array_equal(theirs.timeconst, [0.04] * ours.count)

    def test_unchanged_inside_the_ranges(self):
        """Inside every range the environment steps bitwise as the model without the limit
        (constraint solver off), and the oracle is bitwise the smooth one."""
        for env_id in FAMILIES:
            with self.subTest(env_id=env_id):
                config = table_config(env_id)
                limited = tasks.make_env(config, seed=0)
                free = tasks.make_env(config, seed=0)
                free.model.jnt_limited[:] = 0
                free.model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONSTRAINT
                rng = np.random.default_rng(1)
                state = limit_states(env_id, config, rng, 1, past=0.0, speed=0.2)[0]
                for env in (limited, free):
                    env.reset(options={"state": state})
                high = limited.action_space.high
                m = limited.state_spec["action_dim"]
                steps = 0
                for _ in range(60):
                    decision = rng.uniform(limited.action_space.low, high)
                    decision[:m] *= 0.1  # small setpoint offsets: stay inside the ranges
                    first = limited.step_dt(decision, config["reference_period"])
                    second = free.step_dt(decision, config["reference_period"])
                    if past_any(config, first["next_state"]).any():
                        break
                    np.testing.assert_array_equal(first["next_state"], second["next_state"])
                    self.assertEqual(first["reward"], second["reward"])
                    steps += 1
                self.assertGreater(steps, 20)
                # The oracles inside the ranges, against the code without limits.
                states = limit_states(env_id, config, rng, 64, past=0.0)
                torques = random_torques(config, rng, 64)
                drift, _ = tasks.oracle_callbacks(config)
                plant = config["plant_parameters"]
                if env_id == "cartpole-swingup":
                    smooth, _ = dmc_cartpole._cartpole_callbacks(plant)
                elif domain(env_id) == "cartpole":
                    smooth, _ = chain_callbacks(dmc_cartpole.cartpole_chain(plant, limited.n_poles))
                else:
                    smooth = None
                with Float64():
                    got = drift(torch.as_tensor(states), torch.as_tensor(torques))
                    if smooth is not None:
                        self.assertTrue(torch.equal(got, smooth(torch.as_tensor(states), torch.as_tensor(torques))))
                if domain(env_id) == "reacher":
                    for state, torque in zip(states, torques):
                        np.testing.assert_array_equal(
                            limited.oracle_drift(state, torque),
                            chain_drift_numpy(limited._chain, state, torque, context_dim=2),
                        )
                if domain(env_id) == "point_mass":
                    expected = np.concatenate((states[:, 2:], (torques - states[:, 2:]) / 0.3), axis=-1)
                    np.testing.assert_array_equal(got.numpy(), expected)


# --------------------------------------------------------------------- constraint rows


class TestConstraintRows(unittest.TestCase):
    def test_rows_forces_and_accelerations_match_mujoco(self):
        """efc_* rows, the limit forces, qfrc_constraint and qacc, on states with the
        limits inactive, in the impedance ramp and deep past it; for the point mass with
        both slides at once. MuJoCo's default Newton solver reaches the complementarity
        solution to rounding."""
        for env_id in FAMILIES:
            with self.subTest(env_id=env_id):
                config = table_config(env_id)
                env = tasks.make_env(config, seed=0)
                limits = env_limits(env, env_id)
                model, data = env.model, mujoco.MjData(env.model)
                rng = np.random.default_rng(2)
                states = limit_states(env_id, config, rng, 300, past=0.6)
                torques = random_torques(config, rng, 300)
                drift, _ = tasks.oracle_callbacks(config)
                with Float64():
                    tensor = drift(torch.as_tensor(states), torch.as_tensor(torques)).numpy()
                counts = []
                for state, torque, from_tensor in zip(states, torques, tensor):
                    set_mujoco(env, data, state, torque)
                    q = state[: env.physical_position_dim]
                    v = state[env.position_dim :]
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
                        np.testing.assert_allclose(theirs["diagApprox"][active], limits.invweight[active], rtol=1e-14)
                        np.testing.assert_array_equal(ours["rows_active"][active], theirs["force"][active] > 0.0)
                    for values in (ours, twin):
                        self.assertLess(scaled_error(values["qfrc_constraint"], data.qfrc_constraint), 1e-10)
                        self.assertLess(scaled_error(values["qacc"], data.qacc), 1e-10)
                        self.assertTrue(np.all(values["force"] >= 0.0))
                    truth = env.mujoco_drift(state, torque)
                    self.assertLess(scaled_error(env.oracle_drift(state, torque), truth), 1e-10)
                    self.assertLess(scaled_error(from_tensor, truth), 1e-10)
                counts = np.array(counts)
                self.assertGreater((counts[:, 1] > 0).sum(), 60)  # pushing rows
                self.assertGreater((counts[:, 0] == 0).sum(), 50)  # inactive states
                self.assertGreater((counts[:, 1] < counts[:, 0]).sum(), 0)  # a row that lets go
                if env_id == "point_mass-easy":
                    self.assertGreater((counts[:, 0] == 2).sum(), 20)  # both slides at once

    def test_oracle_plant_jacobian_past_the_limits(self):
        """make_oracle_plant(env).drift_jacobian (vmap(jacfwd)) against central differences,
        the limits pushing; penetrations kept off the impedance's kinks (0, half and one
        width) and velocities off the active set's switch."""
        for env_id in FAMILIES:
            with self.subTest(env_id=env_id), Float64():
                config = table_config(env_id)
                env = tasks.make_env(config, seed=0)
                plant = tasks.make_oracle_plant(env)
                rng = np.random.default_rng(3)
                states = limit_states(env_id, config, rng, 8, past=1.0, speed=0.0)
                for index, (low, high) in ranges(config).items():
                    depth = np.where(np.arange(8) % 2 == 0, rng.uniform(1.5e-3, 0.02, 8), rng.uniform(1e-4, 4e-4, 8))
                    states[:, index] = np.where(states[:, index] > 0, high + depth, low - depth)
                torques = random_torques(config, rng, 8)
                # Half the rows push into the limit at the full motor: the row then pushes.
                high = np.asarray(config["controller"]["torque_high"], dtype=np.float64)
                for index, (low, high_q) in ranges(config).items():
                    dof = 1 if domain(env_id) == "reacher" else index
                    torques[::2, min(dof, len(high) - 1)] = np.sign(states[::2, index]) * high[min(dof, len(high) - 1)]
                states, actions = torch.as_tensor(states), torch.as_tensor(torques)
                values, by_state, by_action = plant.drift_jacobian(states, actions)
                torch.testing.assert_close(values, plant.drift(states, actions))
                data = mujoco.MjData(env.model)
                pushing = 0
                for state, torque in zip(states.numpy(), torques):
                    set_mujoco(env, data, state, torque)
                    pushing += int(np.abs(data.qfrc_constraint).max() > 0.0)
                self.assertGreaterEqual(pushing, 4)
                epsilon = 1e-7
                for i in range(env.state_dim):
                    bump = torch.zeros_like(states)
                    bump[:, i] = epsilon
                    numeric = (plant.drift(states + bump, actions) - plant.drift(states - bump, actions)) / (2.0 * epsilon)
                    scale = 1.0 + numeric.abs().max().item()
                    self.assertLess((by_state[:, :, i] - numeric).abs().max().item() / scale, 1e-5, i)
                for i in range(actions.shape[1]):
                    bump = torch.zeros_like(actions)
                    bump[:, i] = epsilon
                    numeric = (plant.drift(states, actions + bump) - plant.drift(states, actions - bump)) / (2.0 * epsilon)
                    scale = 1.0 + numeric.abs().max().item()
                    self.assertLess((by_action[:, :, i] - numeric).abs().max().item() / scale, 1e-5)


# -------------------------------------------------------------------- model integrator


def crossing_trials(env_id, config, env, rng, trials):
    """Interval starts at the limits: half already past their range (pushing in), half
    inside and moving toward it fast enough to cross within the interval."""
    family = domain(env_id)
    duration = config["reference_period"]
    states = limit_states(env_id, config, rng, trials, past=0.0, speed=0.3)
    (index, (low, high)), *_ = ranges(config).items()
    n_o = config["state_spec"]["position_dim"]
    dof = int(np.argmax(np.abs(np.asarray(config["state_spec"]["observation_projection"])[index])))
    speed = {"cartpole": 2.0, "reacher": 4.0, "point_mass": 0.3}[family]
    for row in range(trials):
        sign = 1.0 if row % 4 < 2 else -1.0
        bound = high if sign > 0 else low
        if row % 2 == 0:
            states[row, index] = bound + sign * rng.uniform(0.0, 0.01)
        else:
            states[row, index] = bound - sign * 0.5 * speed * duration * rng.uniform(0.2, 1.0)
        states[row, n_o + dof] = sign * speed * rng.uniform(0.5, 1.0)
    low_box, high_box = env.action_space.low, env.action_space.high
    decisions = rng.uniform(low_box, high_box, (trials, low_box.size))
    m = config["state_spec"]["action_dim"]
    decisions[1::3, m : 2 * m] = high_box[m : 2 * m] * rng.uniform(0.9, 1.0, (len(decisions[1::3]), m))
    decisions[1::3, 2 * m :] = high_box[2 * m :] * rng.uniform(0.9, 1.0, (len(decisions[1::3]), m))
    return states, decisions


class TestModelIntegratorThroughTheLimits(unittest.TestCase):
    """The oracle's rk4 interval map at the physics step against env.step_dt, intervals
    starting past the limits or crossing into them. MuJoCo's Newton solver inside mj_step
    is warm-started and stops at opt.tolerance (1e-8, dm_control's default and ours); with
    one limit row per joint (these tasks; the swimmer's several coupled rows left up to
    5e-9) its exact line search lands on the solution, and the endpoints agree to rounding
    (measured 2026-11-10: at most 1.8e-15 at either tolerance)."""

    TRIALS = 16

    def test_rk4_interval_map_is_the_simulator_through_the_limits(self):
        for env_id in FAMILIES:
            for tolerance, bound in ((None, 1e-10), (1e-12, 1e-10)):
                with self.subTest(env_id=env_id, solver_tolerance=tolerance), Float64():
                    self.check(env_id, tolerance, bound)

    def check(self, env_id, tolerance, bound):
        config = table_config(env_id)
        env = tasks.make_env(config, decision_mode="feedback", seed=0)
        if tolerance is not None:
            env.model.opt.tolerance = tolerance
        plant = tasks.make_oracle_plant(env).double()
        rng = np.random.default_rng(4)
        states, decisions = crossing_trials(env_id, config, env, rng, self.TRIALS)
        duration = config["reference_period"]
        observed, engaged = [], 0
        for state, decision in zip(states, decisions):
            env.reset(options={"state": state})
            transition = env.step_dt(decision, duration)
            self.assertFalse(transition["terminated"])
            observed.append(transition["next_state"])
            engaged += int(past_any(config, state)[0] or past_any(config, transition["next_state"])[0])
        self.assertGreaterEqual(engaged, self.TRIALS // 2)
        durations = torch.full((self.TRIALS, 1), duration, dtype=torch.float64)
        means = interval_means(env, plant, torch.as_tensor(states), torch.as_tensor(decisions), durations, "rk4")
        residual = wrapped_endpoint_residual(torch.as_tensor(np.array(observed)), means, env.state_spec)
        self.assertLess(residual.abs().max().item(), bound)


# ---------------------------------------------------------------------- dm_control


class TestAgainstDmControl(unittest.TestCase):
    def test_forward_dynamics_past_the_limits(self):
        """Our MuJoCo model's qacc (and the oracle's) against dm_control's compiled model's
        after physics.forward(), with the limits active. The limit force depends on the
        timestep through MuJoCo's time-constant floor (2 x timestep): dm_control's model is
        compared at our 1 ms step, and our model at dm_control's step; at their own steps
        the reacher's and point mass's limits differ (0.02 s against 0.04 s)."""
        for env_id in FAMILIES:
            with self.subTest(env_id=env_id):
                config = table_config(env_id)
                env = tasks.make_env(config, seed=0)
                physics = load_dm(env_id).physics
                dm_step = float(physics.model.opt.timestep)
                rng = np.random.default_rng(5)
                states = limit_states(env_id, config, rng, 150, past=0.7)
                torques = random_torques(config, rng, 150)
                engaged, differ = 0, 0
                for state, torque in zip(states, torques):
                    physical = env.physical_position_dim
                    physics.data.qpos[:] = state[:physical]
                    physics.data.qvel[:] = state[env.position_dim :]
                    physics.data.ctrl[:] = torque / GEAR[domain(env_id)]
                    physics.model.opt.timestep = dm_step
                    physics.forward()
                    at_their_step = np.array(physics.data.qacc)
                    physics.model.opt.timestep = env.physics_dt
                    physics.forward()
                    expected = np.array(physics.data.qacc)
                    engaged += int(physics.data.nefc > 0)
                    self.assertLess(scaled_error(env.mujoco_drift(state, torque)[env.position_dim :], expected), 1e-9)
                    self.assertLess(scaled_error(env.oracle_drift(state, torque)[env.position_dim :], expected), 1e-9)
                    env.model.opt.timestep = dm_step
                    self.assertLess(scaled_error(env.mujoco_drift(state, torque)[env.position_dim :], at_their_step), 1e-9)
                    env.model.opt.timestep = env.physics_dt
                    differ += int(scaled_error(expected, at_their_step) > 1e-6)
                physics.model.opt.timestep = dm_step
                self.assertGreater(engaged, 50)
                if domain(env_id) == "cartpole":
                    self.assertEqual(differ, 0)  # 0.08 s is above both floors
                else:
                    self.assertGreater(differ, 50)

    def test_trajectories_into_the_limits_at_dm_controls_step(self):
        """Open-loop full-force pushes into the limits and back, our environment at
        dm_control's physics step and integrator against dm_control's own stepping: the
        same compiled limit (and time-constant floor), so the paths agree to the solver's
        tolerance."""
        for env_id in FAMILIES:
            with self.subTest(env_id=env_id):
                family = domain(env_id)
                config = copy.deepcopy(table_config(env_id))
                dt, integrator = (0.01, "RK4") if family == "cartpole" else (0.02, "Euler")
                config.update(physics_dt=dt, integrator=integrator, clock=fixed_clock(dt, dt))
                env = tasks.make_env(config, decision_mode="torque", seed=0)
                dm_env = load_dm(env_id)
                physics = dm_env.physics
                rng = np.random.default_rng(6)
                start = limit_states(env_id, config, rng, 1, past=0.0, speed=0.0)[0]
                (index, (low, high)), *_ = ranges(config).items()
                start[index] = 0.8 * high
                env.reset(options={"state": start})
                with physics.reset_context():
                    physics.data.qpos[:] = start[: env.physical_position_dim]
                    physics.data.qvel[:] = start[env.position_dim :]
                high_torque = np.asarray(config["controller"]["torque_high"], dtype=np.float64)
                push = np.zeros_like(high_torque)
                dof = 0 if family == "cartpole" else (1 if family == "reacher" else 0)
                push[min(dof, len(push) - 1)] = high_torque[min(dof, len(push) - 1)]
                if family == "point_mass":
                    push[:] = high_torque
                worst, deepest = 0.0, 0.0
                for step in range(150):
                    torque = push if step < 100 else -push
                    transition = env.step_dt(torque, dt)
                    self.assertFalse(transition["terminated"])
                    physics.data.ctrl[:] = torque / GEAR[family]
                    physics.step()
                    worst = max(worst, float(np.abs(env.data.qpos - physics.data.qpos).max()))
                    worst = max(worst, float(np.abs(env.data.qvel - physics.data.qvel).max()))
                    deepest = max(deepest, float(physics.data.qpos[index] - high))
                self.assertGreater(deepest, 0.0)  # dm_control's own path passes the range
                self.assertLess(worst, 1e-10)  # measured 2026-11-10: 0 (bitwise)


# ------------------------------------------------------------------- no failure set


class TestNoFailureSet(unittest.TestCase):
    def test_masks_payoffs_and_lifts(self):
        for env_id in ROWS:
            with self.subTest(env_id=env_id):
                config = table_config(env_id)
                self.assertEqual(config["failure_limits"], {})
                env = tasks.make_env(config, seed=0)
                rng = np.random.default_rng(7)
                states = limit_states(env_id, config, rng, 100, past=0.7)
                self.assertGreater(int(past_any(config, states).sum()), 30)
                tensor = torch.as_tensor(states)
                self.assertFalse(tasks.failure_mask_tensor(tensor, config).any())
                self.assertFalse(any(env.failed(state) for state in states))
                payoff = tasks.terminal_payoff_tensor(tensor, config)
                self.assertTrue(torch.equal(payoff, torch.zeros(100, 1, dtype=torch.float64)))
                self.assertEqual(testbed_spec(ROWS[env_id][0]).terminal_payoff_value(config), 0.0)
                lift, vanishing = tasks.terminal_boundary_functions(config)
                self.assertTrue(torch.equal(lift(tensor), torch.zeros(100, 1, dtype=torch.float64)))
                self.assertTrue(torch.equal(vanishing(tensor), torch.ones(100, 1, dtype=torch.float64)))

    def test_factories_and_environments_reject_the_old_failure_sets(self):
        for env_id in ROWS:
            with self.subTest(env_id=env_id):
                settings = copy.deepcopy(table.find_ph_entry(env_id, "top").config["testbed_settings"])
                settings["failure_limits"] = dict(OLD_FAILURE[domain(env_id)])
                with self.assertRaisesRegex(ValueError, "no failure set"):
                    testbed_spec(ROWS[env_id][0]).config_factory(**settings)
                config = table_config(env_id)
                config["failure_limits"] = dict(OLD_FAILURE[domain(env_id)])
                with self.assertRaisesRegex(ValueError, "no failure set"):
                    tasks.make_env(config)

    def test_episodes_run_into_the_limits_and_continue(self):
        """Full force into the limit: the coordinate passes its range by a bounded depth,
        is pushed back, and the episode goes on to its time limit."""
        depth_bounds = {"cartpole": 0.3, "reacher": 0.15, "point_mass": 0.01}
        for env_id in FAMILIES:
            with self.subTest(env_id=env_id):
                family = domain(env_id)
                config = table_config(env_id)
                env = tasks.make_env(config, decision_mode="torque", seed=0)
                rng = np.random.default_rng(8)
                start = limit_states(env_id, config, rng, 1, past=0.0, speed=0.0)[0]
                (index, (low, high)), *_ = ranges(config).items()
                start[index] = 0.9 * high
                env.reset(options={"state": start})
                high_torque = np.asarray(config["controller"]["torque_high"], dtype=np.float64)
                push = np.zeros_like(high_torque)
                push[1 if family == "reacher" else 0] = high_torque[0]
                if family == "point_mass":
                    push[:] = high_torque
                deepest, last = 0.0, None
                while not env.done:
                    elapsed = env.time
                    torque = push if elapsed < 0.5 * config["episode_duration"] else -push
                    transition = env.step_dt(torque, config["reference_period"])
                    self.assertFalse(transition["terminated"])
                    last = transition["next_state"]
                    deepest = max(deepest, float(last[index] - high))
                self.assertTrue(transition["truncated"])
                self.assertAlmostEqual(env.time, config["episode_duration"], places=9)
                self.assertGreater(deepest, 0.0)
                self.assertLess(deepest, depth_bounds[family])
                self.assertLess(float(last[index]), high)  # pushed back off the limit


# ---------------------------------------------------------------- table and limit port


def port_network(**changes):
    config = {
        "hidden_sizes": (16, 16),
        "plant_hidden_sizes": (32, 32),
        "activation": "tanh",
        "mass_floor": 0.01,
        "diffusion_floor": 0.001,
        "damping_rank": 2,
        "velocity_feature_scale": 2.0 * math.pi,
        "diffusion": "none",
    }
    config.update(changes)
    return config


class TestTableAndLimitPort(unittest.TestCase):
    def test_rows_set_the_limit_port(self):
        """Every row with a kept joint limit either sets the limit port (the reachers and
        swimmers) or contact exclusion without it (the cart-poles and point mass, since
        2026-11-11, EXCLUDED); no other row sets either."""
        for entry in table.load_ph_table():
            with self.subTest(env_id=entry.env_id, mode=entry.mode):
                network = entry.config["network"]
                if entry.env_id in EXCLUDED:
                    self.assertNotIn("limit_port", network)
                    self.assertEqual(entry.config["contact_handling"], "exclude")
                    self.assertEqual(entry.config["testbed_settings"]["failure_limits"], {})
                elif entry.env_id in ROWS or entry.env_id.startswith("swimmer-"):
                    self.assertIs(network.get("limit_port"), True)
                    self.assertNotIn("contact_handling", entry.config)
                    self.assertEqual(entry.config["testbed_settings"]["failure_limits"], {})
                else:
                    self.assertNotIn("limit_port", network)
                    self.assertNotIn("contact_handling", entry.config)
        for env_id in ROWS:
            with self.subTest(env_id=env_id):
                entry = table.find_ph_entry(env_id, "top")
                self.assertIn("KEPT", entry.comment)
                self.assertEqual(table.declared_testbed_mismatches(entry.task, table_config(env_id)), [])
                # The port as the row has it, or as --contact-handling port restores it.
                network = {**entry.config["network"], "limit_port": True}
                plant = PortHamiltonianPlant(table_config(env_id)["state_spec"], network, learn_diffusion=False)
                self.assertTrue(plant.limit_port)
                self.assertEqual(plant.limit_stiffness.shape, (len(ROWS[env_id][3]),))

    def test_port_is_zero_inside_the_ranges(self):
        """With the same seed the plant with the port equals the plant without it inside
        the ranges (drift and Jacobian, bitwise) and differs past them."""
        for env_id in FAMILIES:
            with self.subTest(env_id=env_id), Float64():
                config = table_config(env_id)
                spec = config["state_spec"]
                torch.manual_seed(5)
                plain = PortHamiltonianPlant(spec, port_network(), learn_diffusion=False).double()
                torch.manual_seed(5)
                port = PortHamiltonianPlant(spec, port_network(limit_port=True), learn_diffusion=False).double()
                rng = np.random.default_rng(9)
                inside = torch.as_tensor(limit_states(env_id, config, rng, 32, past=0.0))
                actions = torch.as_tensor(random_torques(config, rng, 32))
                self.assertTrue(torch.equal(plain.drift(inside, actions), port.drift(inside, actions)))
                for left, right in zip(plain.drift_jacobian(inside, actions), port.drift_jacobian(inside, actions)):
                    self.assertTrue(torch.equal(left, right))
                past = torch.as_tensor(limit_states(env_id, config, rng, 32, past=1.0))
                self.assertGreater((plain.drift(past, actions) - port.drift(past, actions)).abs().max().item(), 1e-8)

    def test_port_starts_at_the_compiled_limit(self):
        """Every row with joint limits (the cart-poles, reachers, point mass and swimmers)
        declares limit_port_init = joint_limits.limit_port_init of its compiled limits, and
        the plant's port starts there: its stiffness and damping within 10% of MuJoCo's own
        k d^2 / diagApprox and b d / diagApprox (efc_KBIP, efc_KBIP[2] = d and
        efc_diagApprox = dof_invweight0 of the limit rows, every limited coordinate 3
        impedance widths past its range, so d = dmax), its width LIMIT_WIDTH_FLOOR + solimp
        width (1 - d0 / dmax). A state spec without limit_port_init (configs and records
        before 2026-11-10) starts at the plant's generic constants; a malformed one raises."""
        rows = {env_id: (ROWS[env_id][0], ROWS[env_id][2]) for env_id in ROWS}
        for env_id, links in (("swimmer-swimmer6", 6), ("swimmer-swimmer15", 15)):
            rows[env_id] = (env_id.replace("-", "_"), tuple(f"joint_{i}" for i in range(links - 1)))
        for env_id, (testbed, joints) in rows.items():
            with self.subTest(env_id=env_id), Float64():
                entry = table.find_ph_entry(env_id, "top")
                config = testbed_spec(testbed).config_factory(**copy.deepcopy(entry.config["testbed_settings"]))
                spec = config["state_spec"]
                env = tasks.make_env(config, seed=0)
                limits = joint_limits.joint_limits_from_model(env.model, joints)
                initial = spec["limit_port_init"]
                self.assertEqual(initial, joint_limits.limit_port_init(limits))
                self.assertEqual(len(initial), len(spec["limited_position_ranges"]))
                plant = PortHamiltonianPlant(spec, {**entry.config["network"], "limit_port": True}, learn_diffusion=False).double()
                stiffness = plant.limit_stiffness.detach().exp().numpy()
                damping = plant.limit_damping.detach().exp().numpy()
                width = (LIMIT_WIDTH_FLOOR + torch.nn.functional.softplus(plant.limit_width.detach())).numpy()
                np.testing.assert_allclose(np.array(initial), np.stack((stiffness, damping, width), -1), rtol=1e-12)
                d0 = np.clip(limits.d0, 1e-4, 0.9999)
                dmax = np.clip(limits.dmax, 1e-4, 0.9999)
                np.testing.assert_allclose(width, LIMIT_WIDTH_FLOOR + limits.width * (1.0 - d0 / dmax), rtol=1e-12)
                # MuJoCo's rows, each limited joint 3 widths past its upper limit.
                data = mujoco.MjData(env.model)
                for joint in joints:
                    joint_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, joint)
                    address = env.model.jnt_qposadr[joint_id]
                    data.qpos[address] = env.model.jnt_range[joint_id, 1] + 3.0 * env.model.jnt_solimp[joint_id, 2]
                mujoco.mj_forward(env.model, data)
                kbip = np.array(data.efc_KBIP).reshape(-1, 4)
                predicted = {}
                for row in range(data.nefc):
                    if data.efc_type[row] == mujoco.mjtConstraint.mjCNSTR_LIMIT_JOINT:
                        k, b, d = kbip[row, 0], kbip[row, 1], kbip[row, 2]
                        name = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_JOINT, int(data.efc_id[row]))
                        predicted[name] = (k * d * d / data.efc_diagApprox[row], b * d / data.efc_diagApprox[row])
                self.assertEqual(set(predicted), set(joints))
                expected = np.array([predicted[joint] for joint in joints])
                np.testing.assert_allclose(stiffness, expected[:, 0], rtol=0.1)
                np.testing.assert_allclose(damping, expected[:, 1], rtol=0.1)
                env.close()
        spec = table_config("point_mass-easy")["state_spec"]
        legacy = {key: value for key, value in spec.items() if key != "limit_port_init"}
        plant = PortHamiltonianPlant(legacy, port_network(limit_port=True), learn_diffusion=False).double()
        np.testing.assert_allclose(plant.limit_stiffness.detach().exp().numpy(), [LIMIT_STIFFNESS_INIT] * 2, rtol=1e-6)
        np.testing.assert_allclose(plant.limit_damping.detach().exp().numpy(), [LIMIT_DAMPING_INIT] * 2, rtol=1e-6)
        np.testing.assert_allclose(
            (LIMIT_WIDTH_FLOOR + torch.nn.functional.softplus(plant.limit_width.detach())).numpy(),
            [LIMIT_WIDTH_INIT] * 2, rtol=1e-6,
        )
        for bad in ((spec["limit_port_init"][0],), ((750.0, 30.0, LIMIT_WIDTH_FLOOR),) * 2,
                    ((-1.0, 30.0, 0.002),) * 2, ((750.0, float("nan"), 0.002),) * 2):
            with self.assertRaises(ValueError):
                PortHamiltonianPlant({**legacy, "limit_port_init": bad}, port_network(limit_port=True), learn_diffusion=False)

    def test_port_fits_past_the_limits(self):
        """A short fit of the drift at rest, coordinates inside and past their ranges, three
        plants: without the port; with the port from the plant's generic start (stiffness
        1e-2, damping 1e-3, width 0.1; the state spec without limit_port_init) at a high
        port learning rate PORT_LR; and with the port from the compiled limit
        (limit_port_init) at the training learning rate TRAINING_LR. Both ports' held-out
        error past the limits falls below the plant's without (on the point mass well
        below), and the compiled start at the training rate does at least as well as the
        generic start at the high rate. Both ports' stiffness comes to the limit's static
        stiffness k d^2 / invweight0 (MuJoCo's spring k d over A + R, with A + R =
        invweight0 / d past the impedance width at qpos0's A): about 161 N/m on the
        cart-pole and 750 N/m on the point mass. All plants get the true mass matrix and
        input matrix (at rest the mass's derivatives do not enter the drift) and fit their
        potential network at 1e-2 (the same for all three, so only the port's start and
        learning rate differ), and the port, to the oracle's drift (point mass: no other
        force; cart-pole: gravity).

        From the generic start the port needs PORT_LR: these limits need about 160 N/m
        (cart-pole) and 750 N/m (point mass), log-distances of 10 from 1e-2 that 800 steps of
        TRAINING_LR cannot cover (the port stays inert there: 0.02 after 2000 steps). From
        the compiled start TRAINING_LR suffices (2026-11-10, 800 steps: point mass 2.7e-4,
        3.8e-4 after 400, against 5.9e-4 from the generic start and 5.2e-2 without;
        cart-pole 5.65e-3, 5.69e-3 after 400, against 5.64e-3 and 7.3e-3).

        The cart-pole's forces are a tenth of the motor's: MuJoCo's constraint also cancels
        about d = 0.95 of every other force along the limited joint (its added inertia),
        which no potential or damping of the position can do, and at the full +-10 N that
        part is as large as the spring's (with and without the port the error past the rail
        then stays at 0.21). With a tenth that part is the error both ports leave (5.6e-3),
        so on the cart-pole the compiled start may trail the generic one by COMPILED_SLACK
        (0.1% at 800 steps, 0.9% at 400). On the point mass the spring dominates the 0.1 N
        motor."""
        results = {}
        for env_id in ("point_mass-easy", "cartpole-swingup"):
            with self.subTest(env_id=env_id), Float64():
                config = table_config(env_id)
                env = tasks.make_env(config, seed=0)
                oracle = tasks.make_oracle_plant(env).double()
                spec = config["state_spec"]
                n_o, n = spec["position_dim"], spec["velocity_dim"]
                rng = np.random.default_rng(10)

                force_scale = 0.1 if domain(env_id) == "cartpole" else 1.0

                def sample(count):
                    states = limit_states(env_id, config, rng, count, past=0.5, speed=0.0)
                    actions = force_scale * random_torques(config, rng, count)
                    states, actions = torch.as_tensor(states), torch.as_tensor(actions)
                    return states, actions, oracle.drift(states, actions)[:, n_o:]

                train, train_actions, train_target = sample(1024)
                test, test_actions, test_target = sample(1024)
                scale = train_target.square().mean(0).sqrt()
                past = torch.as_tensor(past_any(config, test.numpy()))
                self.assertGreater(int(past.sum()), 300)

                plant_parameters = config["plant_parameters"]
                if env_id == "point_mass-easy":
                    root = math.sqrt(plant_parameters["mass"])

                    def true_mass_jets(features, first, second):
                        factor = root * torch.eye(2, dtype=features.dtype).expand(len(features), 2, 2)
                        tangent = None if first is None else features.new_zeros(len(features), first.shape[1], 2, 2)
                        return factor, tangent, None
                else:
                    total, coupling, inertia, _ = dmc_cartpole._cartpole_coefficients(plant_parameters)

                    def true_mass_jets(features, first, second):
                        # Features: x, sin theta, cos theta.
                        off = coupling * features[:, 2:3]
                        mass = torch.stack(
                            (torch.cat((torch.full_like(off, total), off), -1),
                             torch.cat((off, torch.full_like(off, inertia)), -1)),
                            dim=-2,
                        )
                        tangent = None if first is None else features.new_zeros(len(features), first.shape[1], 2, 2)
                        return torch.linalg.cholesky(mass), tangent, None

                self.assertEqual(table.find_ph_entry(env_id, "top").config["training"]["learning_rate"], TRAINING_LR)
                generic = {key: value for key, value in spec.items() if key != "limit_port_init"}
                errors = {}
                for label, with_port, plant_spec, port_lr in (
                    ("without", False, generic, None),
                    ("generic", True, generic, PORT_LR),
                    ("compiled", True, spec, TRAINING_LR),
                ):
                    torch.manual_seed(0)
                    plant = PortHamiltonianPlant(plant_spec, port_network(limit_port=with_port), learn_diffusion=False).double()
                    plant._mass_jets = true_mass_jets
                    plant.force_matrix.data.copy_(torch.as_tensor(env.oracle_input_matrix()))
                    named = [(name, p) for name, p in plant.named_parameters() if not name.startswith(("mass_net", "force_matrix"))]
                    groups = [{"params": [p for name, p in named if not name.startswith("limit_")], "lr": 1e-2}]
                    port_parameters = [p for name, p in named if name.startswith("limit_")]
                    if port_parameters:
                        groups.append({"params": port_parameters, "lr": port_lr})
                    optimizer = torch.optim.Adam(groups)
                    for _ in range(PORT_FIT_STEPS):
                        optimizer.zero_grad()
                        loss = ((plant.drift(train, train_actions)[:, n_o:] - train_target) / scale).square().mean()
                        loss.backward()
                        optimizer.step()
                    with torch.no_grad():
                        error = ((plant.drift(test, test_actions)[:, n_o:] - test_target) / scale).square().mean(-1)
                    errors[label] = (float(error[past].mean()), float(error[~past].mean()), float(loss.detach()))
                    if with_port:
                        errors[label + "_port"] = {
                            "stiffness": plant.limit_stiffness.detach().exp().tolist(),
                            "damping": plant.limit_damping.detach().exp().tolist(),
                            "width": (1e-3 + torch.nn.functional.softplus(plant.limit_width.detach())).tolist(),
                        }
                results[env_id] = errors
                for label in ("generic", "compiled"):
                    self.assertLess(errors[label][0], errors["without"][0], errors)
                    if domain(env_id) == "point_mass":
                        self.assertLess(errors[label][0], PORT_GAIN * errors["without"][0], errors)
                slack = COMPILED_SLACK[domain(env_id)]
                self.assertLessEqual(errors["compiled"][0], (1.0 + slack) * errors["generic"][0], errors)
                limits = env_limits(env, env_id)
                dmax = np.clip(limits.dmax, 1e-4, 0.9999)
                static = limits.stiffness_coefficient * dmax**2 / limits.invweight
                for label in ("generic", "compiled"):
                    np.testing.assert_allclose(errors[label + "_port"]["stiffness"], static, rtol=0.15, err_msg=str(errors))


PORT_FIT_STEPS = 800
PORT_GAIN = 0.35
PORT_LR = 5e-2
# The learning rate of every table row (training.learning_rate).
TRAINING_LR = 3e-4
# How far the compiled start's error past the limits may exceed the generic start's (the
# cart-pole: both reach the error of the constraint's added inertia; test docstring).
COMPILED_SLACK = {"point_mass": 0.0, "cartpole": 0.01}


if __name__ == "__main__":
    unittest.main()

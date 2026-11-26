"""The model integrator (config key model_integrator, models/ph_sac/rollouts.model_substep).

1. Oracle against environment: with a dm_control port's oracle drift, the "rk4" interval
   map reproduces the MuJoCo environment's next state after one decision interval, to
   rounding error, on reacher-easy (also at damping decisions near the box maximum,
   where the closed loop is stiff and "semi_implicit" is far off), cartpole-swingup
   and acrobot-swingup, in both controller refresh modes: "continuous" (the default;
   the law evaluated at every RK4 stage, by MuJoCo's control callback in the
   environment) to 1e-14 of the state's size, and "held" (the torque refreshed at
   each substep's start and held through the four stages).
2. Regression: "semi_implicit" reproduces the outputs of the code before 2026-11-09
   (endpoint moments, path Jacobian norms, rollouts with returns and failures) on a
   fixed analytic plant; the reference values were computed with that code.
3. Gradients: the RK4 plant-fit loss is finite and differentiable, and its autograd
   gradient matches central differences, for a drift-only and a noisy plant.
4. Plumbing: default "rk4", "semi_implicit" accepted, anything else rejected; configs
   and table rows without the key read "rk4"; every table row integrates its model as
   its simulator does; both arms pass their integrator to every model integration.
"""

import copy
import json
import math
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

import algorithms.explicit_ct_ph_sac as explicit_module
import algorithms.implicit_ct_ph_sac as implicit_module
import algorithms.ph_base as base_module
from benchmarks import ph_hyperparams as table
from benchmarks.run_ph_sac import protocol_from_config
from common.ph_config import (
    DEFAULT_MODEL_INTEGRATOR,
    MODEL_INTEGRATORS,
    check_model_integrator,
    model_integrator_setting,
)
from environment.ph_sac import tasks
from environment.ph_sac.controller import FeedbackController
from environment.ph_sac.registry import testbed_spec
from evaluations.ph_sac.protocol import validate_matched_quantities
from models.ph_sac.losses import plant_likelihood_loss
from models.ph_sac.plant import OraclePlant, PortHamiltonianPlant
from models.ph_sac.rollouts import (
    integrate_endpoint_moments,
    model_substep,
    path_jacobian_norms,
    physical_noise_injection,
    rk4_step,
    semi_implicit_step,
    stochastic_rollout,
    wrapped_endpoint_residual,
)
from tests.test_ph_algorithms import (
    collect,
    explicit_config,
    implicit_config,
    make_explicit,
    make_implicit,
)

CONFIGS = Path(__file__).resolve().parents[1] / "benchmarks" / "configs" / "ph_sac"

# ------------------------------------------------------------- oracle vs environment

# Testbed -> (table row, position half-ranges, velocity half-range). Positions stay
# clear of failure sets (reacher wrist stop 160 deg, cart rail 1.8 m) for one interval.
ORACLE_TASKS = {
    "reacher_easy": ("reacher-easy", (math.pi, 2.4), 3.0),
    "cartpole_swingup": ("cartpole-swingup", (1.0, math.pi), 3.0),
    "acrobot_swingup": ("acrobot-swingup", (math.pi, math.pi), 5.0),
}
TRIALS = 24
# rk4 against MuJoCo RK4: the oracle drift equals MuJoCo's to rounding, and the two
# maps are the same arithmetic, so the endpoints agree to rounding (about 1e-15).
ORACLE_TOLERANCE = 1e-10


def table_testbed(name, refresh_mode=None):
    """The table row's TestbedConfig; refresh_mode replaces the factory's when given."""
    env_id = ORACLE_TASKS[name][0]
    settings = table.find_ph_entry(env_id, "top").config["testbed_settings"]
    config = testbed_spec(name).config_factory(**settings)
    if refresh_mode is not None:
        config["controller"] = dict(config["controller"], refresh_mode=refresh_mode)
    return config


def oracle_trials(name, *, seed=0, refresh_mode=None):
    """Random states and decisions, half of them stiff, with the environment's endpoints.

    The stiff half holds damping in [0.9, 1] of the box maximum with a weak spring and
    slow joints, so the motor stays inside its clip and the damping acts at full
    strength (on the reacher's wrist h c / m_eff reaches 2). Returns the environment,
    its oracle plant and double tensors of states, decisions, durations, observed next
    states and the stiff-row mask.
    """
    _, position_ranges, speed = ORACLE_TASKS[name]
    config = table_testbed(name, refresh_mode)
    env = tasks.make_env(config, decision_mode="feedback", seed=seed)
    plant = tasks.make_oracle_plant(env).double()
    rng = np.random.default_rng(seed)
    physical = len(position_ranges)
    velocities = env.state_spec["velocity_dim"]
    low, high = env.action_space.low, env.action_space.high
    m = env.state_spec["action_dim"]
    duration = config["reference_period"]
    rows = []
    for trial in range(TRIALS):
        stiff = trial % 2 == 0
        env.reset(seed=seed + trial)
        state = env.observe()
        state[:physical] = rng.uniform(-1.0, 1.0, physical) * np.asarray(position_ranges)
        state[-velocities:] = rng.uniform(-1.0, 1.0, velocities) * (0.3 if stiff else speed)
        decision = rng.uniform(low, high)
        if stiff:
            decision[:m] = 0.05 * rng.uniform(-1.0, 1.0, m) * high[:m]
            decision[m : 2 * m] = 0.02 * high[m : 2 * m] * rng.uniform(0.0, 1.0, m)
            decision[2 * m :] = high[2 * m :] * rng.uniform(0.9, 1.0, m)
        env.reset(options={"state": state})
        state = env.observe()
        transition = env.step_dt(decision, duration)
        if transition["terminated"]:
            raise AssertionError(f"{name}: trial {trial} failed inside the interval.")
        rows.append((state, decision, transition["next_state"], stiff))
    states, decisions, observed, stiff = (np.array(column) for column in zip(*rows))
    as_tensor = lambda array: torch.as_tensor(array, dtype=torch.float64)  # noqa: E731
    return (
        env,
        plant,
        as_tensor(states),
        as_tensor(decisions),
        torch.full((TRIALS, 1), duration, dtype=torch.float64),
        as_tensor(observed),
        torch.as_tensor(stiff),
    )


def interval_means(env, plant, states, decisions, durations, integrator):
    """Noise-free model endpoints on the environment's own grid and refresh cadence."""
    return integrate_endpoint_moments(
        plant,
        env.controller,
        states,
        decisions,
        durations,
        maximum_step=env.physics_dt,
        physics_dt=env.physics_dt,
        controller_refresh_steps=env.controller.refresh_steps,
        covariance_mode="leading_order",
        create_graph=False,
        integrator=integrator,
    )["means"]


class TestOracleReproducesEnvironment(unittest.TestCase):
    def test_rk4_interval_map_is_the_simulator(self):
        for name in ORACLE_TASKS:
            with self.subTest(testbed=name):
                env, plant, states, decisions, durations, observed, _ = oracle_trials(name)
                self.assertEqual(env.config["integrator"], "RK4")
                means = interval_means(env, plant, states, decisions, durations, "rk4")
                residual = wrapped_endpoint_residual(observed, means, env.state_spec)
                self.assertLess(residual.abs().max().item(), ORACLE_TOLERANCE)

    def test_both_refresh_modes_reproduce_their_environment_to_rounding(self):
        """Continuous law at every stage (callback / model stages) and held torque alike."""
        for name in ORACLE_TASKS:
            for mode in ("continuous", "held"):
                with self.subTest(testbed=name, refresh_mode=mode):
                    env, plant, states, decisions, durations, observed, _ = oracle_trials(
                        name, seed=3, refresh_mode=mode
                    )
                    self.assertEqual(env.stage_feedback, mode == "continuous")
                    self.assertEqual(env.controller.refresh_mode, mode)
                    means = interval_means(env, plant, states, decisions, durations, "rk4")
                    residual = wrapped_endpoint_residual(observed, means, env.state_spec)
                    scale = max(1.0, observed.abs().max().item())
                    self.assertLess(residual.abs().max().item(), 1e-14 * scale)
        # The two environments differ well beyond rounding on the same rows.
        continuous = oracle_trials("acrobot_swingup", seed=3, refresh_mode="continuous")
        held = oracle_trials("acrobot_swingup", seed=3, refresh_mode="held")
        self.assertTrue(torch.equal(continuous[2], held[2]))
        self.assertGreater((continuous[5] - held[5]).abs().max().item(), 1e-6)

    def test_semi_implicit_misses_the_stiff_reacher(self):
        """At h c / m_eff near 2 the semi-implicit endpoint is far from the simulator's."""
        env, plant, states, decisions, durations, observed, stiff = oracle_trials(
            "reacher_easy"
        )
        means = interval_means(env, plant, states, decisions, durations, "semi_implicit")
        residual = wrapped_endpoint_residual(observed, means, env.state_spec).abs()
        self.assertGreater(residual[stiff].max().item(), 1e-3)
        exact = interval_means(env, plant, states, decisions, durations, "rk4")
        self.assertLess(
            wrapped_endpoint_residual(observed, exact, env.state_spec)[stiff].abs().max().item(),
            ORACLE_TOLERANCE,
        )

    def test_rk4_rollout_paths_are_the_simulator(self):
        """Noise-free stochastic_rollout paths end where the environment does; the context stays."""
        env, plant, states, decisions, durations, observed, _ = oracle_trials(
            "reacher_easy", seed=7
        )
        steps = int(round(env.config["reference_period"] / env.physics_dt))
        paths = stochastic_rollout(
            plant,
            env.controller,
            states,
            decisions,
            durations,
            lambda s: tasks.failure_mask_tensor(s, env.config),
            maximum_step=env.physics_dt,
            physics_dt=env.physics_dt,
            controller_refresh_steps=env.controller.refresh_steps,
            path_count=2,
            brownian_increments=torch.zeros(TRIALS, 2, steps, plant.noise_dim).double(),
            integrator="rk4",
        )
        self.assertTrue(paths["survived"].all())
        for path in range(2):
            endpoints = paths["endpoints"][:, path]
            residual = wrapped_endpoint_residual(observed, endpoints, env.state_spec)
            self.assertLess(residual.abs().max().item(), ORACLE_TOLERANCE)
            # Context coordinates have zero drift: RK4 leaves them bit for bit.
            self.assertTrue(torch.equal(endpoints[:, 2:4], states[:, 2:4]))


# ------------------------------------------------------------ semi-implicit regression

COUPLED_SPEC = {
    "position_dim": 2,
    "velocity_dim": 2,
    "action_dim": 1,
    "periodic_position_indices": (0, 1),
    "actuated_position_indices": (1,),
    "observation_projection": np.eye(2),
}
COUPLED_CONTROLLER = {
    "decision_low": np.array([-1.0, 0.0, 0.0]),
    "decision_high": np.array([1.0, 100.0, 20.0]),
    "torque_low": np.array([-4.0]),
    "torque_high": np.array([4.0]),
    "setpoint_mode": "offset",
    "setpoint_error": "wrapped",
    "refresh_steps": 2,
    # The regressions below are of the sample-and-hold code (every run before 2026-11-09).
    "refresh_mode": "held",
}
# Torque refreshed every second 1 ms substep, so held torques enter the regression.
GRID = {"maximum_step": 0.001, "physics_dt": 0.001, "controller_refresh_steps": 2}


def coupled_drift(states, actions):
    """Two coupled damped pendulums, the second actuated, with an angle-dependent inertia."""
    q1, q2, v1, v2 = states.unbind(-1)
    torque = actions[..., 0]
    inertia = 1.0 + 0.3 * torch.cos(q2)
    a1 = (-9.81 * torch.sin(q1) - 2.0 * torch.sin(q1 - q2) - 0.4 * v1 + 0.2 * v2 * v1) / inertia
    a2 = 4.0 * (2.0 * torch.sin(q1 - q2) - 0.3 * torch.sin(q2) - 0.6 * v2 + torque)
    return torch.stack((v1, v2, a1, a2), dim=-1)


def coupled_diffusion(states):
    """State-dependent velocity noise and a small direct position noise, [B,4,2]."""
    q1 = states[..., 0]
    zero = torch.zeros_like(q1)
    small = torch.full_like(q1, 0.01)
    columns = (
        torch.stack((small, zero, 0.3 * (1.0 + 0.2 * torch.sin(q1)), 0.05 + zero), dim=-1),
        torch.stack((zero, small, zero, 0.2 + 0.1 * torch.cos(q1)), dim=-1),
    )
    return torch.stack(columns, dim=-1)


def coupled_plant():
    return OraclePlant(
        coupled_drift,
        coupled_diffusion,
        torch.tensor([[0.0], [1.0]]),
        state_spec=COUPLED_SPEC,
    ).double()


def coupled_controller():
    return FeedbackController(dict(COUPLED_CONTROLLER), COUPLED_SPEC)


def coupled_batch():
    """Three rows: a clipped stiff decision, one that fails and an unclipped damped one."""
    states = torch.tensor(
        [[0.4, -0.3, 1.0, -2.0], [2.9, 0.2, 6.0, 0.3], [-1.2, 0.8, 0.1, 0.4]],
        dtype=torch.float64,
    )
    decisions = torch.tensor(
        [[0.5, 100.0, 5.0], [-0.2, 10.0, 1.0], [0.05, 30.0, 20.0]], dtype=torch.float64
    )
    durations = torch.tensor([[0.003], [0.010], [0.007]], dtype=torch.float64)
    return states, decisions, durations


def coupled_increments():
    """Already-scaled dW [3 rows, 2 paths, 10 substeps, 2], zero past each row's count."""
    rng = np.random.default_rng(20261109)
    standard = rng.standard_normal((3, 2, 10, 2))
    for row, steps in enumerate((3, 10, 7)):
        standard[row, :, steps:] = 0.0
    return torch.as_tensor(standard * math.sqrt(0.001), dtype=torch.float64)


def coupled_return_terms():
    return {
        "reward_rate": lambda s, a: -(s[:, :2].square().sum(-1, keepdim=True)) - 0.01 * a.square(),
        "terminal_payoff": lambda s: torch.full((s.shape[0], 1), -1.0, dtype=s.dtype),
        "discount_rate": 0.5,
    }


def coupled_failure(states):
    return states[:, 0].abs() >= 2.95


# Outputs of models/ph_sac/rollouts.py as it was before 2026-11-09 (semi-implicit only)
# on the fixture above: integrate_endpoint_moments (propagated), path_jacobian_norms and
# stochastic_rollout with coupled_increments, coupled_return_terms and coupled_failure.
# shape (3, 4)
SEMI_IMPLICIT_MEANS = (
    0.40297242857895643, -0.3058423209028218, 0.9862100950745089, -1.9212032663325695,
    2.95978496145726, 0.20262732291648097, 5.961608349500242, 0.23209068775430194,
    -1.1990470319819122, 0.8021024705187116, 0.16322074554716698, 0.2273501590873888,
)
# shape (3, 4, 4)
SEMI_IMPLICIT_COVARIANCES = (
    3.4024823024958147e-07, 3.226358381211959e-09, 1.0320760170298966e-05,
    1.5951825700280816e-06, 3.2263583812119596e-09, 3.3621196613317883e-07,
    9.879704333142556e-08, 9.262868105350753e-06, 1.0320760170298968e-05,
    9.879704333142555e-08, 0.0003132651732102972, 4.846931434301598e-05,
    1.5951825700280813e-06, 9.262868105350751e-06, 4.846931434301597e-05,
    0.0002620366298177221, 1.3827385427146095e-06, 3.2913321998878246e-08,
    3.67125463825968e-05, 5.652875648797982e-06, 3.2913321998878246e-08,
    1.1146078835779194e-06, 8.800977822876296e-07, 1.0486288028123602e-05,
    3.6712546382596804e-05, 8.800977822876298e-07, 0.000979987750093318,
    0.00015149744314841396, 5.652875648797979e-06, 1.0486288028123598e-05,
    0.00015149744314841402, 0.00011992722757869548, 8.449111802042629e-07,
    1.5352245935129287e-08, 1.8731144676443623e-05, 3.5393074200279016e-06,
    1.5352245935129287e-08, 8.371039602910556e-07, 3.368415109903209e-07,
    1.6670924225227127e-05, 1.8731144676443617e-05, 3.3684151099032103e-07,
    0.0004161057108187351, 7.877519752120373e-05, 3.539307420027901e-06,
    1.6670924225227127e-05, 7.877519752120374e-05, 0.0003462676963927794,
)
# shape (3, 1)
SEMI_IMPLICIT_SATURATION = (
    1.0, 0.0, 0.75,
)
# shape (3, 1)
SEMI_IMPLICIT_JACOBIAN_NORMS = (
    0.03624585548399286, 0.35294066912375044, 1.0048686347840043,
)
# shape (3, 2, 4)
SEMI_IMPLICIT_ENDPOINTS = (
    0.4029880662981847, -0.30567663560667313, 0.986754289669278, -1.916521542378322,
    0.4036056819223811, -0.3054301719343924, 1.0054556054172687, -1.9068776603347795,
    2.9534350669637357, 0.20196583001887497, 5.95406684799295, 0.23251601852909645,
    2.953726195338745, 0.20261482317631982, 5.9627139306090555, 0.2409214763111291,
    -1.1980351408399434, 0.803943136480896, 0.1863567947989457, 0.2681311785812689,
    -1.1978553486640144, 0.801497576650647, 0.1892592643391052, 0.22052431338876113,
)
# shape (3, 2, 1)
SEMI_IMPLICIT_RUNNING_REWARDS = (
    -0.0012378198962494282, -0.0012384606451742348, -0.07776135492340915,
    -0.07775959773777798, -0.015666874351154377, -0.015598306285626784,
)
# shape (3, 2, 1)
SEMI_IMPLICIT_FAILURE_PAYOFFS = (
    0.0, 0.0, -0.9955101098295706, -0.9955101098295706, 0.0, 0.0,
)
# shape (3, 2, 1)
SEMI_IMPLICIT_SURVIVED = (
    1.0, 1.0, 0.0, 0.0, 1.0, 1.0,
)
# float64 throughout: the same arithmetic in the same order, so only BLAS rounding differs.
REGRESSION_TOLERANCE = {"atol": 1e-13, "rtol": 1e-11}


def reference(values, shape):
    return torch.tensor(values, dtype=torch.float64).reshape(shape)


class TestSemiImplicitRegression(unittest.TestCase):
    def setUp(self):
        self.plant, self.controller = coupled_plant(), coupled_controller()
        self.states, self.decisions, self.durations = coupled_batch()

    def test_endpoint_moments(self):
        moments = integrate_endpoint_moments(
            self.plant,
            self.controller,
            self.states,
            self.decisions,
            self.durations,
            covariance_mode="propagated",
            create_graph=False,
            integrator="semi_implicit",
            **GRID,
        )
        torch.testing.assert_close(
            moments["means"], reference(SEMI_IMPLICIT_MEANS, (3, 4)), **REGRESSION_TOLERANCE
        )
        torch.testing.assert_close(
            moments["covariances"],
            reference(SEMI_IMPLICIT_COVARIANCES, (3, 4, 4)),
            **REGRESSION_TOLERANCE,
        )
        torch.testing.assert_close(
            moments["saturation_fractions"],
            reference(SEMI_IMPLICIT_SATURATION, (3, 1)),
            **REGRESSION_TOLERANCE,
        )

    def test_path_jacobian_norms(self):
        norms = path_jacobian_norms(
            self.plant,
            self.controller,
            self.states,
            self.decisions,
            self.durations,
            integrator="semi_implicit",
            **GRID,
        )
        torch.testing.assert_close(
            norms, reference(SEMI_IMPLICIT_JACOBIAN_NORMS, (3, 1)), **REGRESSION_TOLERANCE
        )

    def test_rollouts_with_returns_and_failures(self):
        paths = stochastic_rollout(
            self.plant,
            self.controller,
            self.states,
            self.decisions,
            self.durations,
            coupled_failure,
            brownian_increments=coupled_increments(),
            return_terms=coupled_return_terms(),
            integrator="semi_implicit",
            **GRID,
        )
        for key, values, shape in (
            ("endpoints", SEMI_IMPLICIT_ENDPOINTS, (3, 2, 4)),
            ("running_rewards", SEMI_IMPLICIT_RUNNING_REWARDS, (3, 2, 1)),
            ("failure_payoffs", SEMI_IMPLICIT_FAILURE_PAYOFFS, (3, 2, 1)),
        ):
            torch.testing.assert_close(
                paths[key], reference(values, shape), **REGRESSION_TOLERANCE, msg=key
            )
        self.assertEqual(
            paths["survived"].double().reshape(-1).tolist(), list(SEMI_IMPLICIT_SURVIVED)
        )

    def test_rk4_differs_from_the_regression(self):
        """The default map is not the old one: the two means differ well beyond rounding."""
        means = integrate_endpoint_moments(
            self.plant,
            self.controller,
            self.states,
            self.decisions,
            self.durations,
            covariance_mode="leading_order",
            create_graph=False,
            **GRID,
        )["means"]
        gap = (means - reference(SEMI_IMPLICIT_MEANS, (3, 4))).abs().max().item()
        self.assertGreater(gap, 1e-6)


# ------------------------------------------------------------------ the substep maps


class TestSubstepMaps(unittest.TestCase):
    def setUp(self):
        self.plant = coupled_plant()
        self.states = torch.tensor(
            [[0.4, -0.3, 1.0, -2.0], [2.0, 0.2, -1.0, 0.3]], dtype=torch.float64
        )
        self.actions = torch.tensor([[1.5], [-0.5]], dtype=torch.float64)
        self.h = torch.tensor([0.002, 0.001], dtype=torch.float64)

    def test_rk4_is_the_classical_formula(self):
        h = self.h[:, None]
        k1 = coupled_drift(self.states, self.actions)
        k2 = coupled_drift(self.states + 0.5 * h * k1, self.actions)
        k3 = coupled_drift(self.states + 0.5 * h * k2, self.actions)
        k4 = coupled_drift(self.states + h * k3, self.actions)
        expected = self.states + h * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        torch.testing.assert_close(
            rk4_step(self.plant, self.states, self.actions, self.h),
            expected,
            atol=1e-15,
            rtol=1e-14,
        )

    def test_both_maps_add_the_same_noise_increment(self):
        """s' = noise-free map + K_h Sigma(s) dW, Sigma at the substep's start."""
        kicks = torch.tensor([[0.03, -0.02], [-0.01, 0.04]], dtype=torch.float64)
        injection = physical_noise_injection(self.plant, self.states, self.h)
        increment = (injection @ kicks.unsqueeze(-1)).squeeze(-1)
        for integrator in MODEL_INTEGRATORS:
            with self.subTest(integrator=integrator):
                noisy = model_substep(
                    self.plant,
                    self.states,
                    self.actions,
                    self.h,
                    integrator=integrator,
                    brownian_increments=kicks,
                )
                clean = model_substep(
                    self.plant, self.states, self.actions, self.h, integrator=integrator
                )
                torch.testing.assert_close(noisy - clean, increment, atol=1e-14, rtol=1e-12)

    def test_dispatch_and_rejection(self):
        torch.testing.assert_close(
            model_substep(self.plant, self.states, self.actions, self.h),
            rk4_step(self.plant, self.states, self.actions, self.h),
        )
        torch.testing.assert_close(
            model_substep(
                self.plant, self.states, self.actions, self.h, integrator="semi_implicit"
            ),
            semi_implicit_step(self.plant, self.states, self.actions, self.h),
        )
        controller = coupled_controller()
        states, decisions, durations = coupled_batch()
        for name in ("euler", "RK4", "semi-implicit", None):
            with self.subTest(integrator=name):
                with self.assertRaisesRegex(ValueError, "model_integrator"):
                    model_substep(
                        self.plant, self.states, self.actions, self.h, integrator=name
                    )
                with self.assertRaisesRegex(ValueError, "model_integrator"):
                    integrate_endpoint_moments(
                        self.plant,
                        controller,
                        states,
                        decisions,
                        durations,
                        covariance_mode="leading_order",
                        create_graph=False,
                        integrator=name,
                        **GRID,
                    )

    def test_propagated_covariance_matches_monte_carlo_under_rk4(self):
        """The RK4 recursion F Lambda F^T + h K_h A K_h^T against RK4 noisy paths."""
        plant, controller = coupled_plant(), coupled_controller()
        state = torch.tensor([[0.3, -0.2, 0.5, 0.4]], dtype=torch.float64)
        decision = torch.tensor([[0.0, 15.0, 3.0]], dtype=torch.float64)
        duration = torch.tensor([[0.02]], dtype=torch.float64)
        moments = integrate_endpoint_moments(
            plant,
            controller,
            state,
            decision,
            duration,
            covariance_mode="propagated",
            create_graph=False,
            integrator="rk4",
            **GRID,
        )
        torch.manual_seed(5)
        paths = stochastic_rollout(
            plant,
            controller,
            state,
            decision,
            duration,
            lambda s: torch.zeros(len(s), dtype=torch.bool),
            path_count=100000,
            integrator="rk4",
            **GRID,
        )
        endpoints = paths["endpoints"][0]
        torch.testing.assert_close(
            torch.cov(endpoints.T), moments["covariances"][0], atol=0.0, rtol=0.03
        )
        self.assertLess((endpoints.mean(0) - moments["means"][0]).abs().max().item(), 1e-3)


# ------------------------------------------------------------------------- gradients

PLANT_NETWORK = {
    "hidden_sizes": (8,),
    "plant_hidden_sizes": (8, 8),
    "activation": "tanh",
    "mass_floor": 1e-2,
    "diffusion_floor": 1e-3,
    "damping_rank": 2,
}


class TestRK4FitGradient(unittest.TestCase):
    """The RK4 plant-fit loss on reacher rows: finite, and autograd equals central differences."""

    def setUp(self):
        torch.manual_seed(4)
        config = table_testbed("reacher_easy")
        self.spec = config["state_spec"]
        self.controller = FeedbackController(config["controller"], self.spec)
        rng = np.random.default_rng(3)
        rows = 6
        positions = rng.uniform(-1.0, 1.0, (rows, 2)) * np.array([math.pi, 2.0])
        targets = rng.uniform(-0.2, 0.2, (rows, 2))
        velocities = rng.uniform(-0.3, 0.3, (rows, 2))
        self.states = torch.as_tensor(
            np.concatenate((positions, targets, velocities), axis=-1), dtype=torch.float64
        )
        # Stiff damping, weak springs: the clip stays inactive, so the loss is smooth.
        self.decisions = torch.as_tensor(
            np.concatenate(
                (
                    rng.uniform(-0.05, 0.05, (rows, 2)),
                    rng.uniform(0.0, 0.01, (rows, 2)),
                    rng.uniform(0.09, 0.1, (rows, 2)),
                ),
                axis=-1,
            ),
            dtype=torch.float64,
        )
        self.durations = torch.full((rows, 1), 0.005, dtype=torch.float64)
        noise = torch.as_tensor(rng.normal(0.0, 0.01, (rows, 6)), dtype=torch.float64)
        noise[:, 2:4] = 0.0
        self.batch = {"next_states": self.states + noise, "durations": self.durations}

    def plant(self, learn_diffusion):
        plant = PortHamiltonianPlant(
            self.spec, PLANT_NETWORK, learn_diffusion=learn_diffusion
        ).double()
        # Inertias near 0.02 and position-dependent networks; much smaller random
        # inertias let the untrained potential drive the path to overflow.
        with torch.no_grad():
            plant.mass_net.layers[-1].bias.fill_(-2.0)
            for net in (plant.potential_net, plant.mass_net, plant.damping_net):
                net.layers[-1].weight.mul_(5.0)
        return plant

    def loss(self, plant, mode):
        moments = integrate_endpoint_moments(
            plant,
            self.controller,
            self.states,
            self.decisions,
            self.durations,
            maximum_step=0.001,
            physics_dt=0.001,
            controller_refresh_steps=1,
            covariance_mode=mode,
            create_graph=True,
            integrator="rk4",
        )
        return plant_likelihood_loss(self.batch, moments, self.spec)

    def test_gradient_is_finite_and_matches_central_differences(self):
        for learn_diffusion, mode in ((False, "leading_order"), (True, "leading_order"), (True, "propagated")):
            with self.subTest(learn_diffusion=learn_diffusion, mode=mode):
                plant = self.plant(learn_diffusion)
                loss = self.loss(plant, mode)
                self.assertTrue(torch.isfinite(loss))
                loss.backward()
                for name, parameter in plant.named_parameters():
                    if name == "potential_net.layers.2.bias":
                        # The drift reads only grad U, so the constant of U has no effect.
                        self.assertIsNone(parameter.grad)
                        continue
                    self.assertIsNotNone(parameter.grad, name)
                    self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                    self.assertGreater(parameter.grad.abs().sum().item(), 0.0, name)
                entries = (
                    ("force_matrix", (1, 1)),
                    ("mass_net.layers.2.bias", (2,)),
                    ("damping_net.layers.0.weight", (0, 1)),
                    ("potential_net.layers.0.weight", (3, 0)),
                )
                parameters = dict(plant.named_parameters())
                for name, index in entries:
                    parameter = parameters[name]
                    analytic = parameter.grad[index].item()
                    epsilon = 1e-6
                    with torch.no_grad():
                        parameter[index] += epsilon
                        upper = self.loss(plant, mode).item()
                        parameter[index] -= 2.0 * epsilon
                        lower = self.loss(plant, mode).item()
                        parameter[index] += epsilon
                    numeric = (upper - lower) / (2.0 * epsilon)
                    self.assertAlmostEqual(
                        analytic, numeric, delta=1e-5 * max(1.0, abs(numeric)), msg=name
                    )


# --------------------------------------------------------------------------- plumbing


def raw_config(name):
    with open(CONFIGS / name, encoding="utf-8") as handle:
        return json.load(handle)


class TestPlumbing(unittest.TestCase):
    def test_setting_default_and_values(self):
        self.assertEqual(DEFAULT_MODEL_INTEGRATOR, "rk4")
        self.assertEqual(MODEL_INTEGRATORS, ("rk4", "semi_implicit"))
        self.assertEqual(model_integrator_setting({}), "rk4")
        for value in MODEL_INTEGRATORS:
            self.assertEqual(model_integrator_setting({"model_integrator": value}), value)
            self.assertEqual(check_model_integrator(value), value)
        for value in ("euler", "RK4", "semi-implicit", "", None, 4):
            with self.assertRaisesRegex(ValueError, "model_integrator", msg=repr(value)):
                model_integrator_setting({"model_integrator": value})

    def test_arm_configs_validate_the_key(self):
        for build in (explicit_config, implicit_config):
            with self.subTest(build=build.__name__):
                self.assertNotIn("model_integrator", build())
                build(model_integrator="semi_implicit")
                build(model_integrator="rk4")
                with self.assertRaisesRegex(ValueError, "model_integrator"):
                    build(model_integrator="midpoint")

    def test_agents_pass_their_integrator_to_every_integration(self):
        for value in (None, "semi_implicit"):
            expected = value or "rk4"
            overrides = {} if value is None else {"model_integrator": value}
            with self.subTest(integrator=expected):
                agent = make_implicit(implicit_config(**overrides))
                self.assertEqual(agent.model_integrator, expected)
                collect(agent, 40)
                batch = agent.replay.sample("train", 8)
                with mock.patch.object(
                    implicit_module,
                    "integrate_endpoint_moments",
                    wraps=implicit_module.integrate_endpoint_moments,
                ) as moments, mock.patch.object(
                    implicit_module, "stochastic_rollout", wraps=implicit_module.stochastic_rollout
                ) as rollout, mock.patch.object(
                    implicit_module,
                    "paired_gate_samples",
                    wraps=implicit_module.paired_gate_samples,
                ) as gate, mock.patch.object(
                    base_module,
                    "integrate_endpoint_moments",
                    wraps=base_module.integrate_endpoint_moments,
                ) as diagnostics, mock.patch.object(
                    base_module, "path_jacobian_norms", wraps=base_module.path_jacobian_norms
                ) as norms:
                    agent.update_online_plant(batch)
                    agent.fit_rate_head(1)
                    agent._gate_samples(agent.replay.sample("train", 4), 2)
                    agent.record_plant_diagnostics(batch)
                for spy in (moments, rollout, gate, diagnostics, norms):
                    spy.assert_called()
                    self.assertEqual(spy.call_args.kwargs["integrator"], expected)

                agent = make_explicit(explicit_config(continuation="rollout", **overrides))
                self.assertEqual(agent.model_integrator, expected)
                collect(agent, 20)
                batch = agent.replay.sample("train", 8)
                with mock.patch.object(
                    explicit_module,
                    "integrate_endpoint_moments",
                    wraps=explicit_module.integrate_endpoint_moments,
                ) as moments, mock.patch.object(
                    explicit_module, "stochastic_rollout", wraps=explicit_module.stochastic_rollout
                ) as rollout:
                    agent.update_plant(batch)
                    agent.build_critic_target(batch)
                for spy in (moments, rollout):
                    spy.assert_called_once()
                    self.assertEqual(spy.call_args.kwargs["integrator"], expected)

    def test_protocol_reads_the_key_and_defaults_old_configs(self):
        raw = raw_config("cartpole_swingup.json")
        self.assertEqual(raw["model_integrator"], "rk4")
        old = copy.deepcopy(raw)
        del old["model_integrator"]
        for config, expected in ((raw, "rk4"), (old, "rk4")):
            protocol = protocol_from_config(config, "x")
            self.assertEqual(protocol["explicit_config"]["model_integrator"], expected)
            self.assertEqual(protocol["implicit_config"]["model_integrator"], expected)
        semi = dict(raw, model_integrator="semi_implicit")
        protocol = protocol_from_config(semi, "x")
        self.assertEqual(protocol["explicit_config"]["model_integrator"], "semi_implicit")
        self.assertEqual(protocol["implicit_config"]["model_integrator"], "semi_implicit")
        for bad in ("Euler", "rk2"):
            with self.assertRaisesRegex(ValueError, "model_integrator"):
                protocol_from_config(dict(raw, model_integrator=bad), "x")
        typo = dict(old, model_integrater="rk4")
        with self.assertRaisesRegex(ValueError, "unknown settings"):
            protocol_from_config(typo, "x")
        mixed = copy.deepcopy(protocol)
        mixed["implicit_config"]["model_integrator"] = "rk4"
        with self.assertRaisesRegex(ValueError, "model_integrator"):
            validate_matched_quantities(mixed)

    def test_table_cells(self):
        """A cell round-trips; an empty cell (a row written before the column) reads rk4."""
        entries = {(entry.env_id, entry.mode): entry for entry in table.load_ph_table()}
        key = ("cartpole-swingup", "top")
        raw = raw_config("cartpole_swingup.json")
        for value in MODEL_INTEGRATORS:
            row = table.config_to_row(
                dict(raw, model_integrator=value), env_id=key[0], mode=key[1],
                task=entries[key].task,
            )
            self.assertEqual(row["model_integrator"], value)
            self.assertEqual(table.parse_ph_row(row).config["model_integrator"], value)
        row["model_integrator"] = ""
        absent = table.parse_ph_row(row).config
        self.assertNotIn("model_integrator", absent)
        protocol = protocol_from_config(absent, "x")
        self.assertEqual(protocol["implicit_config"]["model_integrator"], "rk4")

    def test_every_row_integrates_its_model_as_its_simulator(self):
        """RK4 physics rows use rk4; the windy pendulum's Euler physics rows semi_implicit."""
        matching = {"RK4": "rk4", "Euler": "semi_implicit"}
        for entry in table.load_ph_table():
            with self.subTest(row=(entry.env_id, entry.mode)):
                self.assertEqual(
                    entry.config["model_integrator"], matching[entry.task["integrator"]]
                )

    def test_json_configs_state_the_key_and_its_choice(self):
        for path in sorted(CONFIGS.glob("*.json")):
            with self.subTest(config=path.name):
                raw = raw_config(path.name)
                self.assertIn(raw["model_integrator"], MODEL_INTEGRATORS)
                self.assertIn("model_integrator", raw["_experiment_choices"])


if __name__ == "__main__":
    unittest.main()

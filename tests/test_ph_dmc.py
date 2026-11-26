"""Registry, dm_control helpers, planar-chain dynamics and the episode-constant context."""

import math
import unittest

import mujoco
import numpy as np
import torch

import common.ph_config as ph_config
import common.ph_dmc.cartpole as ph_dmc_cartpole
from benchmarks import ph_hyperparams as table
from common.ph_dmc.swimmer import INTEGRATOR as SWIMMER_INTEGRATOR
from common.ph_dmc.swimmer import PHYSICS_DT as SWIMMER_PHYSICS_DT
from common.ph_dmc._common import (
    CLOCK_SETTINGS_KEYS,
    box_rule_controller,
    dmc_clock,
    dmc_state_spec,
    dmc_testbed_config,
    fixed_clock,
    irregular_clock_bounds,
)
from environment.ph_sac import tasks
from environment.ph_sac.base import MujocoPHEnv, validated_state_spec
from environment.ph_sac.dmc import cartpole as dmc_cartpole
from environment.ph_sac.dmc._common import (
    body_inertials,
    failure_rate_payoff,
    limit_boundary_functions,
    limit_failed,
    limit_failure_mask,
    planar_chain_from_model,
    tolerance,
    tolerance_tensor,
    zero_diffusion,
)
from environment.ph_sac.dmc.chain import (
    PlanarChain,
    chain_accelerations,
    chain_accelerations_numpy,
    chain_callbacks,
    chain_drift,
    chain_drift_numpy,
    chain_energy,
    chain_energy_numpy,
    chain_mass_matrix,
    chain_mass_matrix_numpy,
    chain_points,
    chain_points_numpy,
)
from environment.ph_sac.registry import (
    TestbedSpec,
    register_testbed,
    registered_testbeds,
    testbed_spec,
)
from models.ph_sac.plant import OraclePlant
from tests.test_ph_env import acrobot_config, cartpole_config, random_states

IN_HOUSE = {"windy_pendulum", "acrobot_xk"}
DMC = {
    "pendulum_swingup",
    "acrobot_swingup",
    "acrobot_swingup_sparse",
    "cartpole_balance",
    "cartpole_balance_sparse",
    "cartpole_swingup",
    "cartpole_swingup_sparse",
    "cartpole_two_poles",
    "cartpole_three_poles",
    "reacher_easy",
    "reacher_hard",
    "point_mass_easy",
    "lqr_2_1",
    "lqr_6_2",
    "swimmer_swimmer6",
    "swimmer_swimmer15",
}


def table_rows():
    return [entry for entry in table.load_ph_table() if entry.mode == "top"]


def relative_error(left, right):
    left, right = np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
    return float(np.max(np.abs(left - right) / np.maximum(np.abs(right), 1.0)))


class TestRegistry(unittest.TestCase):
    def test_every_testbed_is_registered(self):
        self.assertEqual(set(registered_testbeds()), IN_HOUSE | DMC)
        implemented = set(registered_testbeds(implemented_only=True))
        self.assertLessEqual({"windy_pendulum", "acrobot_xk", "cartpole_swingup"}, implemented)
        for name in registered_testbeds():
            spec = testbed_spec(name)
            self.assertEqual(spec.name, name)
            self.assertTrue(issubclass(spec.env_class, MujocoPHEnv))
        with self.assertRaisesRegex(ValueError, "Unknown testbed"):
            testbed_spec("no_such_testbed")

    def test_settings_keys_cover_the_table_rows(self):
        for entry in table.load_ph_table():
            with self.subTest(row=(entry.env_id, entry.mode)):
                spec = testbed_spec(entry.config["testbed"])
                given = set(entry.config["testbed_settings"])
                self.assertLessEqual(set(spec.settings_keys), given)
                self.assertLessEqual(given, set(spec.settings_keys) | set(spec.optional_settings_keys))

    def test_stubs_raise_naming_the_testbed(self):
        stubs = [name for name in registered_testbeds() if not testbed_spec(name).implemented]
        for entry in table_rows():
            name = entry.config["testbed"]
            if name not in stubs:
                continue
            with self.subTest(testbed=name):
                spec = testbed_spec(name)
                with self.assertRaisesRegex(NotImplementedError, name):
                    spec.config_factory(**entry.config["testbed_settings"])
                config = {"name": name}
                with self.assertRaisesRegex(NotImplementedError, name):
                    spec.env_class(config)
                with self.assertRaisesRegex(NotImplementedError, name):
                    tasks.make_env(config)
                states = torch.zeros(2, 4)
                for call in (
                    lambda: spec.reward_rate_tensor(states, torch.zeros(2, 1), config),
                    lambda: spec.failure_mask_tensor(states, config),
                    lambda: spec.terminal_payoff_value(config),
                    lambda: spec.terminal_boundary_functions(config),
                    lambda: spec.oracle_callbacks(config),
                ):
                    with self.assertRaisesRegex(NotImplementedError, name):
                        call()

    def test_registration_rules(self):
        spec = testbed_spec("cartpole_swingup")
        with self.assertRaisesRegex(ValueError, "already registered"):
            register_testbed(spec)
        self.assertIs(register_testbed(spec, replace=True), spec)
        self.assertIs(testbed_spec("cartpole_swingup"), spec)
        fields = dict(
            name="toy",
            env_class=dmc_cartpole.CartpoleSwingupEnv,
            config_factory=lambda **kwargs: kwargs,
            settings_keys=("a",),
            reward_rate_tensor=lambda s, u, c: s,
            failure_mask_tensor=lambda s, c: s,
            terminal_payoff_value=lambda c: 0.0,
            terminal_boundary_functions=lambda c: (None, None),
            oracle_callbacks=lambda c: (None, None),
        )
        TestbedSpec(**fields)
        with self.assertRaises(TypeError):
            TestbedSpec(**dict(fields, env_class=object))
        with self.assertRaises(ValueError):
            TestbedSpec(**dict(fields, optional_settings_keys=("a",)))
        with self.assertRaises(TypeError):
            TestbedSpec(**dict(fields, settings_keys=["a"]))

    def test_dispatchers_read_the_registry(self):
        config = cartpole_config()
        spec = testbed_spec("cartpole_swingup")
        states = torch.as_tensor(random_states(config, np.random.default_rng(0), 5))
        torques = torch.ones(5, 1, dtype=torch.float64)
        self.assertTrue(
            torch.equal(
                tasks.reward_rate_tensor(states, torques, config),
                spec.reward_rate_tensor(states, torques, config),
            )
        )
        self.assertEqual(tasks._terminal_payoff_value(config), spec.terminal_payoff_value(config))
        self.assertIsInstance(tasks.make_env(config), spec.env_class)

    def test_cartpole_swingup_moved_with_its_old_names(self):
        self.assertIs(tasks.CartpoleSwingupEnv, dmc_cartpole.CartpoleSwingupEnv)
        self.assertIs(tasks.build_cartpole_model, dmc_cartpole.build_cartpole_model)
        self.assertIs(ph_config.cartpole_swingup_config, ph_dmc_cartpole.cartpole_swingup_config)
        self.assertIs(testbed_spec("cartpole_swingup").env_class, dmc_cartpole.CartpoleSwingupEnv)


class TestTolerance(unittest.TestCase):
    def test_matches_dm_control(self):
        from dm_control.utils import rewards

        x = np.random.default_rng(0).uniform(-3.0, 3.0, 400)
        x[:4] = [0.0, 0.25, -0.25, 1.0]
        cases = (
            dict(bounds=(0.0, 0.0), margin=2.0),
            dict(bounds=(0.0, 0.2), margin=1.0),
            dict(bounds=(-0.25, 0.25), margin=0.0),
            dict(bounds=(math.cos(math.radians(8.0)), 1.0), margin=0.0),
            # pendulum-swingup's row since 2026-11-09 (dm_control's task has margin 0).
            dict(bounds=(math.cos(math.radians(8.0)), 1.0), margin=2.0),
            dict(bounds=(0.0, 0.0), margin=1.0, sigmoid="quadratic", value_at_margin=0.0),
            dict(bounds=(-0.5, 1.0), margin=0.7, sigmoid="quadratic", value_at_margin=0.1),
            dict(bounds=(-math.inf, 0.5), margin=0.3),
        )
        for case in cases:
            with self.subTest(**{k: str(v) for k, v in case.items()}):
                expected = rewards.tolerance(x, **case)
                np.testing.assert_allclose(tolerance(x, **case), expected, rtol=1e-12, atol=1e-14)
                tensor = tolerance_tensor(torch.as_tensor(x), **case)
                np.testing.assert_allclose(tensor.numpy(), expected, rtol=1e-12, atol=1e-14)
                self.assertIsInstance(tolerance(0.1, **case), float)

    def test_rejects_bad_arguments(self):
        # long_tail is supported since the swimmer port (2026-11-09); cosine is not.
        with self.assertRaises(ValueError):
            tolerance(0.0, margin=1.0, sigmoid="cosine")
        with self.assertRaises(ValueError):
            tolerance(0.0, margin=1.0, value_at_margin=0.0)
        with self.assertRaises(ValueError):
            tolerance(0.0, bounds=(1.0, 0.0))


class TestBoundaryHelpers(unittest.TestCase):
    def test_failure_mask_payoff_and_lift(self):
        limited = ((0, 1.8), (2, math.inf), (1, 0.5))
        states = torch.tensor([[0.0, 0.0, 9.0], [1.8, 0.0, 0.0], [0.1, -0.5, 0.0], [1.0, 0.4, 1e9]])
        mask = limit_failure_mask(states, limited)
        self.assertEqual(mask.tolist(), [False, True, True, False])
        for state, failed in zip(states.numpy(), mask.tolist()):
            self.assertEqual(limit_failed(state, limited), failed)
        self.assertEqual(limit_failure_mask(states, ()).tolist(), [False] * 4)
        lift, vanishing = limit_boundary_functions(-2.0, limited)
        self.assertTrue(torch.equal(lift(states), torch.full((4, 1), -2.0)))
        values = vanishing(states)[:, 0]
        self.assertGreater(values[0].item(), 0.0)
        self.assertEqual(values[1].item(), 0.0)
        self.assertEqual(values[2].item(), 0.0)
        _, free = limit_boundary_functions(0.0, ())
        self.assertTrue(torch.equal(free(states), torch.ones(4, 1)))
        config = {"reward_parameters": {"failure_rate": -1.0}, "discount_rate": 0.5}
        self.assertEqual(failure_rate_payoff(config), -2.0)
        self.assertEqual(failure_rate_payoff({"reward_parameters": {}, "discount_rate": 0.5}), 0.0)
        self.assertEqual(zero_diffusion(6)(torch.zeros(3, 6)).shape, (3, 6, 1))


class TestConfigHelpers(unittest.TestCase):
    def test_helpers_reproduce_every_dm_control_row(self):
        """The box rule and builders give each row's declared TestbedConfig exactly."""
        for entry in table_rows():
            name = entry.config["testbed"]
            if name not in DMC:
                continue
            with self.subTest(testbed=name):
                task = entry.task
                nu = int(task["nu"])
                # The swimmers run dm_control's own physics (2 ms Euler, since 2026-11-10);
                # every other row the builder's shared 1 ms RK4.
                physics = {}
                if name.startswith("swimmer_"):
                    physics = {"physics_dt": SWIMMER_PHYSICS_DT, "integrator": SWIMMER_INTEGRATOR}
                config = dmc_testbed_config(
                    **physics,
                    name=name,
                    state_spec=dmc_state_spec(
                        physical_dim=int(task["nq"]),
                        action_dim=nu,
                        periodic=task["periodic_position_indices"],
                        actuated=task["actuated_position_indices"],
                        context_dim=int(task["context_dim"]),
                    ),
                    reference_period=task["reference_period"],
                    episode_duration=task["episode_duration"],
                    controller=box_rule_controller(
                        offsets=task["decision_high"][:nu],
                        torque_limit=task["torque_high"][0],
                        setpoint_mode=task["setpoint_mode"],
                        setpoint_error=task["setpoint_error"],
                    ),
                    **entry.config["testbed_settings"],
                )
                self.assertEqual(table.declared_testbed_mismatches(task, config), [])
                self.assertIs(validated_state_spec(config), config["state_spec"])
                ph_config._validate_testbed(config)
        # The pendulum's offset is pi, written as math.pi.
        pendulum = box_rule_controller(
            offsets=(math.pi,), torque_limit=1.0, setpoint_mode="absolute", setpoint_error="wrapped"
        )
        self.assertEqual(pendulum["decision_high"].tolist(), [math.pi, 3.183098861837907, 0.6366197723675815])

    def test_reacher_state_spec_counts_the_context(self):
        spec = dmc_state_spec(physical_dim=2, action_dim=2, periodic=(0, 1), actuated=(0, 1), context_dim=2)
        self.assertEqual((spec["position_dim"], spec["velocity_dim"], spec["context_dim"]), (4, 2, 2))
        np.testing.assert_array_equal(spec["observation_projection"], np.vstack((np.eye(2), np.zeros((2, 2)))))
        self.assertEqual(table._TASK_FIELDS["nq"]({"state_spec": spec}), 2)
        self.assertEqual(table._TASK_FIELDS["context_dim"]({"state_spec": spec}), 2)


# ------------------------------------------------------------------------------ chain


def cartpole_chain(plant):
    return PlanarChain(
        link_masses=(plant["pole_mass"],),
        link_com_distances=(plant["pole_com"],),
        link_inertias=(plant["pole_inertia"],),
        link_lengths=(plant["pole_length"],),
        joint_damping=(plant["pole_damping"],),
        input_matrix=[[1.0], [0.0]],
        gravity=plant["gravity"],
        cart_mass=plant["cart_mass"],
        cart_damping=plant["cart_damping"],
    )


def acrobot_xk_chain(plant):
    """The Acrobot-XK links in the chain's convention, with the XK torque on the elbow.

    Acrobot-XK measures the shoulder q1 from +x about -y (upright pi/2), so the chain's
    theta_1 = pi/2 - q1 and theta_2 = -q2, and the elbow torque enters with the sign -1.
    """
    return PlanarChain(
        link_masses=(plant["m1"], plant["m2"]),
        link_com_distances=(plant["lc1"], plant["lc2"]),
        link_inertias=(plant["I1"], plant["I2"]),
        link_lengths=(plant["l1"], plant["l2"]),
        joint_damping=(0.0, 0.0),
        input_matrix=[[0.0], [-1.0]],
        gravity=plant["gravity"],
    )


def xk_to_chain(states):
    return np.stack(
        (0.5 * np.pi - states[:, 0], -states[:, 1], -states[:, 2], -states[:, 3]), axis=-1
    )


def chain_to_xk_drift(drift):
    return -drift


class TestChainDynamics(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(11)

    def test_matches_the_cartpole_callbacks(self):
        for damping in (True, False):
            config = cartpole_config(damping=damping)
            chain = cartpole_chain(config["plant_parameters"])
            drift, _ = dmc_cartpole._cartpole_callbacks(config["plant_parameters"])
            states = self.rng.uniform(-6.0, 6.0, (300, 4))
            torques = self.rng.uniform(-10.0, 10.0, (300, 1))
            expected = drift(torch.as_tensor(states), torch.as_tensor(torques)).numpy()
            got = chain_drift(chain, torch.as_tensor(states), torch.as_tensor(torques)).numpy()
            self.assertLess(relative_error(got, expected), 1e-10)
            self.assertLess(relative_error(chain_drift_numpy(chain, states, torques), expected), 1e-10)

    def test_matches_the_acrobot_xk_callbacks(self):
        config = acrobot_config()
        chain = acrobot_xk_chain(config["plant_parameters"])
        drift, _ = tasks.oracle_callbacks(config)
        states = self.rng.uniform(-8.0, 8.0, (300, 4))
        torques = self.rng.uniform(-10.0, 10.0, (300, 1))
        expected = drift(torch.as_tensor(states), torch.as_tensor(torques)).numpy()
        converted = torch.as_tensor(xk_to_chain(states))
        got = chain_to_xk_drift(chain_drift(chain, converted, torch.as_tensor(torques)).numpy())
        self.assertLess(relative_error(got, expected), 1e-10)
        got = chain_to_xk_drift(chain_drift_numpy(chain, xk_to_chain(states), torques))
        self.assertLess(relative_error(got, expected), 1e-10)

    def test_matches_the_cartpole_environment(self):
        config = cartpole_config()
        env = tasks.make_env(config, seed=0)
        chain = cartpole_chain(config["plant_parameters"])
        for state in random_states(config, self.rng, 100):
            torque = self.rng.uniform(-10.0, 10.0, 1)
            self.assertLess(
                relative_error(chain_drift_numpy(chain, state, torque), env.mujoco_drift(state, torque)),
                1e-6,
            )

    def three_pole_chain(self):
        return PlanarChain(
            link_masses=(0.1, 0.2, 0.15),
            link_com_distances=(0.5, 0.4, 0.3),
            link_inertias=(0.009, 0.01, 0.004),
            link_lengths=(1.0, 0.8, 0.6),
            joint_damping=(1e-3, 2e-3, 3e-3),
            input_matrix=[[1.0], [0.0], [0.0], [0.0]],
            gravity=9.81,
            cart_mass=1.0,
            cart_damping=5e-4,
        )

    def test_power_balance(self):
        """dE/dt = qd^T (G u - D qd): the mass matrix, Coriolis and gravity terms agree."""
        for chain in (self.three_pole_chain(), acrobot_xk_chain(acrobot_config()["plant_parameters"])):
            nv = chain.nv
            states = torch.as_tensor(self.rng.uniform(-3.0, 3.0, (50, 2 * nv)), dtype=torch.float64)
            torques = torch.as_tensor(self.rng.uniform(-5.0, 5.0, (50, 1)), dtype=torch.float64)
            states.requires_grad_(True)
            energy = chain_energy(chain, states[:, :nv], states[:, nv:])
            gradient = torch.autograd.grad(energy.sum(), states)[0]
            drift = chain_drift(chain, states, torques)
            rate = (gradient * drift).sum(-1)
            velocities = states[:, nv:]
            damping = torch.as_tensor(chain.damping)
            power = (velocities * (torques @ torch.as_tensor(chain.input_matrix).T - damping * velocities)).sum(-1)
            torch.testing.assert_close(rate, power, rtol=1e-9, atol=1e-9)
            np.testing.assert_allclose(
                chain_energy_numpy(chain, states[:, :nv].detach().numpy(), velocities.detach().numpy()),
                energy.detach().numpy(),
                rtol=1e-12,
            )

    def test_numpy_and_torch_agree_and_context_has_zero_rate(self):
        chain = self.three_pole_chain()
        states = self.rng.uniform(-3.0, 3.0, (20, 8))
        torques = self.rng.uniform(-10.0, 10.0, (20, 1))
        torch_drift = chain_drift(chain, torch.as_tensor(states), torch.as_tensor(torques)).numpy()
        np.testing.assert_allclose(chain_drift_numpy(chain, states, torques), torch_drift, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(
            chain_mass_matrix_numpy(chain, states[:, :4]),
            chain_mass_matrix(chain, torch.as_tensor(states[:, :4])).numpy(),
            rtol=1e-12,
        )
        np.testing.assert_allclose(
            chain_accelerations_numpy(chain, states[:, :4], states[:, 4:], torques),
            chain_accelerations(chain, torch.as_tensor(states[:, :4]), torch.as_tensor(states[:, 4:]), torch.as_tensor(torques)).numpy(),
            rtol=1e-12,
            atol=1e-12,
        )
        single = chain_drift_numpy(chain, states[0], torques[0])
        np.testing.assert_allclose(single, torch_drift[0], rtol=1e-12, atol=1e-12)

        reacher = PlanarChain(
            link_masses=(0.042, 0.04),
            link_com_distances=(0.06, 0.06),
            link_inertias=(5e-5, 6e-5),
            link_lengths=(0.12, 0.12),
            joint_damping=(0.01, 0.01),
            input_matrix=np.eye(2),
        )
        states = self.rng.uniform(-3.0, 3.0, (20, 6))
        torques = self.rng.uniform(-0.05, 0.05, (20, 2))
        drift = chain_drift(reacher, torch.as_tensor(states), torch.as_tensor(torques), context_dim=2)
        self.assertTrue(torch.equal(drift[:, 2:4], torch.zeros(20, 2, dtype=torch.float64)))
        self.assertTrue(torch.equal(drift[:, :2], torch.as_tensor(states[:, 4:])))
        without = chain_drift(reacher, torch.as_tensor(states[:, [0, 1, 4, 5]]), torch.as_tensor(torques))
        torch.testing.assert_close(drift[:, 4:], without[:, 2:], rtol=0, atol=0)
        np.testing.assert_allclose(
            chain_drift_numpy(reacher, states, torques, context_dim=2), drift.numpy(), rtol=1e-12, atol=1e-15
        )
        with self.assertRaises(ValueError):
            chain_drift(reacher, torch.as_tensor(states), torch.as_tensor(torques))

    def test_torch_func_and_oracle_plant(self):
        chain = self.three_pole_chain()
        drift, diffusion = chain_callbacks(chain)
        states = torch.as_tensor(self.rng.uniform(-2.0, 2.0, (4, 8)))
        actions = torch.as_tensor(self.rng.uniform(-5.0, 5.0, (4, 1)))

        def single(state, action):
            return drift(state[None], action[None])[0]

        jacobian = torch.func.vmap(torch.func.jacfwd(single))(states, actions)
        epsilon = 1e-6
        for i in range(8):
            bump = torch.zeros_like(states)
            bump[:, i] = epsilon
            numeric = (drift(states + bump, actions) - drift(states - bump, actions)) / (2 * epsilon)
            self.assertLess((jacobian[:, :, i] - numeric).abs().max().item(), 1e-6)
        spec = dmc_state_spec(physical_dim=4, action_dim=1, periodic=(1, 2, 3), actuated=(0,))
        previous = torch.get_default_dtype()
        torch.set_default_dtype(torch.float64)
        try:
            plant = OraclePlant(drift, diffusion, torch.as_tensor(chain.input_matrix), state_spec=spec)
            values, by_state, by_action = plant.drift_jacobian(states, actions)
        finally:
            torch.set_default_dtype(previous)
        torch.testing.assert_close(by_state, jacobian)
        torch.testing.assert_close(values, drift(states, actions))
        self.assertEqual(by_action.shape, (4, 8, 1))
        self.assertEqual(diffusion(states).shape, (4, 8, 1))
        self.assertEqual(plant.noise_dim, 1)

    def test_points(self):
        chain = self.three_pole_chain()
        positions = np.array([[0.3, 0.0, 0.0, 0.0], [0.0, math.pi, 0.0, 0.0], [0.0, 0.5 * math.pi, -0.5 * math.pi, 0.0]])
        points = chain_points_numpy(chain, positions)
        self.assertEqual(points.shape, (3, 4, 2))
        np.testing.assert_allclose(points[0], [[0.3, 0.0], [0.3, 1.0], [0.3, 1.8], [0.3, 2.4]], atol=1e-12)
        np.testing.assert_allclose(points[1, -1], [0.0, -2.4], atol=1e-12)
        np.testing.assert_allclose(points[2, -1], [1.0, 1.4], atol=1e-12)
        np.testing.assert_allclose(chain_points(chain, torch.as_tensor(positions)).numpy(), points, atol=1e-12)

    def test_rejects_inconsistent_parameters(self):
        with self.assertRaises(ValueError):
            PlanarChain(
                link_masses=(1.0, 1.0),
                link_com_distances=(0.5,),
                link_inertias=(0.1, 0.1),
                link_lengths=(1.0, 1.0),
                joint_damping=(0.0, 0.0),
                input_matrix=[[0.0], [1.0]],
            )
        with self.assertRaises(ValueError):
            PlanarChain(
                link_masses=(1.0,),
                link_com_distances=(0.5,),
                link_inertias=(0.1,),
                link_lengths=(1.0,),
                joint_damping=(0.0,),
                input_matrix=[[1.0]],
                cart_mass=1.0,
            )


class TestChainAgainstDmControl(unittest.TestCase):
    """The chain with parameters read from dm_control's compiled models reproduces qacc."""

    @classmethod
    def setUpClass(cls):
        from dm_control import suite

        cls.suite = suite

    def check(self, domain, task, links, *, along, joint_axis, cart=None, sample, count=60):
        env = self.suite.load(domain, task, task_kwargs={"random": 0})
        physics = env.physics
        chain = planar_chain_from_model(
            physics, links, along=along, joint_axis=joint_axis, cart_body=cart
        )
        rng = np.random.default_rng(5)
        model = physics.model.ptr
        for _ in range(count):
            positions, velocities = sample(rng)
            control = rng.uniform(-1.0, 1.0, model.nu)
            physics.data.qpos[:] = positions
            physics.data.qvel[:] = velocities
            physics.data.ctrl[:] = control
            physics.forward()
            expected = np.array(physics.data.qacc)
            got = chain_accelerations_numpy(chain, positions, velocities, control)
            self.assertLess(relative_error(got, expected), 1e-8)
        return chain

    def test_pendulum(self):
        chain = self.check(
            "pendulum", "swingup", ["pole"], along=2, joint_axis=1,
            sample=lambda rng: (rng.uniform(-np.pi, np.pi, 1), rng.uniform(-8.0, 8.0, 1)),
        )
        self.assertAlmostEqual(chain.link_masses[0], 1.0)
        self.assertAlmostEqual(chain.link_com_distances[0], 0.5)
        self.assertAlmostEqual(chain.link_inertias[0], 0.4 * 0.05**2)
        self.assertAlmostEqual(chain.gravity, 9.81)

    def test_acrobot(self):
        chain = self.check(
            "acrobot", "swingup", ["upper_arm", "lower_arm"], along=2, joint_axis=1,
            sample=lambda rng: (rng.uniform(-np.pi, np.pi, 2), rng.uniform(-6.0, 6.0, 2)),
        )
        self.assertEqual(chain.link_lengths[0], 1.0)
        np.testing.assert_array_equal(chain.input_matrix, [[0.0], [2.0]])

    def test_cartpole_three_poles(self):
        chain = self.check(
            "cartpole", "three_poles", ["pole_1", "pole_2", "pole_3"], along=2, joint_axis=1,
            cart="cart",
            sample=lambda rng: (
                np.concatenate((rng.uniform(-1.5, 1.5, 1), rng.uniform(-np.pi, np.pi, 3))),
                rng.uniform(-5.0, 5.0, 4),
            ),
        )
        self.assertAlmostEqual(chain.link_inertias[0], 0.00942459, places=8)
        self.assertEqual(chain.cart_mass, 1.0)
        np.testing.assert_array_equal(chain.input_matrix, [[10.0], [0.0], [0.0], [0.0]])

    def test_reacher_merges_the_finger(self):
        limit = math.radians(150.0)
        chain = self.check(
            "reacher", "easy", ["arm", "hand"], along=0, joint_axis=2,
            sample=lambda rng: (
                np.array([rng.uniform(-np.pi, np.pi), rng.uniform(-limit, limit)]),
                rng.uniform(-5.0, 5.0, 2),
            ),
        )
        self.assertEqual(chain.gravity, 0.0)
        inertials = body_inertials(self.suite.load("reacher", "easy").physics, ["hand", "finger"])
        alone = body_inertials(
            self.suite.load("reacher", "easy").physics, ["hand"], include_welded=False
        )["hand"]
        self.assertAlmostEqual(
            chain.link_masses[1], alone.mass + inertials["finger"].mass, places=12
        )
        self.assertGreater(chain.link_com_distances[1], alone.com[0])


# ---------------------------------------------------------------------------- context

_CONTEXT_XML = """
<mujoco model="context-toy">
  <option timestep="0.001" integrator="RK4" gravity="0 0 0">
    <flag contact="disable" constraint="disable"/>
  </option>
  <worldbody>
    <body name="mass">
      <joint name="x" type="slide" axis="1 0 0" damping="0.5"/>
      <joint name="y" type="slide" axis="0 1 0" damping="0.5"/>
      <geom type="sphere" size="0.05" mass="2.0"/>
    </body>
    <geom name="target" type="sphere" size="0.02" pos="0 0 0" contype="0" conaffinity="0"/>
  </worldbody>
  <actuator>
    <motor joint="x" gear="1" ctrllimited="true" ctrlrange="-1 1"/>
    <motor joint="y" gear="1" ctrllimited="true" ctrlrange="-1 1"/>
  </actuator>
</mujoco>
"""


def context_config():
    return dmc_testbed_config(
        name="context_toy",
        state_spec=dmc_state_spec(physical_dim=2, action_dim=2, periodic=(), actuated=(0, 1), context_dim=1),
        reference_period=0.01,
        episode_duration=1.0,
        controller=box_rule_controller(
            offsets=(0.3, 0.3), torque_limit=1.0, setpoint_mode="offset", setpoint_error="unwrapped"
        ),
        plant_parameters={},
        failure_limits={},
        reward_parameters={},
        reset_parameters={},
    )


class ContextToyEnv(MujocoPHEnv):
    """A point mass whose reward reads a target x drawn each episode as context."""

    def __init__(self, config, *, decision_mode="feedback", seed=None):
        super().__init__(
            config,
            mujoco.MjModel.from_xml_string(_CONTEXT_XML),
            validated_state_spec(config),
            decision_mode=decision_mode,
            seed=seed,
        )
        self.applied = []

    def _reset_state(self, options=None):
        if options and "state" in options:
            self._set_state(options["state"])
            return
        target = self._reset_rng.uniform(-1.0, 1.0)
        self._set_state(np.array([0.1, -0.2, target, 0.0, 0.0]))

    def _apply_context(self):
        target = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "target")
        if self.context_dim:
            self.model.geom_pos[target, 0] = self._context[0]
        if hasattr(self, "applied"):
            self.applied.append(np.array(self._context))

    def reward_rate(self, state, torque):
        return -float((state[0] - state[2]) ** 2)

    def failed(self, state):
        return False

    def terminal_payoff(self, state):
        return 0.0


class TestContextEnvironment(unittest.TestCase):
    def test_reset_observe_and_steps_keep_the_context(self):
        env = ContextToyEnv(context_config(), seed=3)
        self.assertEqual((env.physical_position_dim, env.context_dim, env.state_dim), (2, 1, 5))
        state, _ = env.reset(seed=3)
        target = state[2]
        self.assertNotEqual(target, 0.0)
        self.assertEqual(env.context.tolist(), [target])
        self.assertEqual(env.model.nq, 2)
        geom = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, "target")
        self.assertEqual(env.model.geom_pos[geom, 0], target)
        for _ in range(5):
            transition = env.step_dt(np.array([0.2, -0.1, 1.0, 1.0, 0.1, 0.1]), 0.01)
            self.assertEqual(transition["next_state"][2], target)
        self.assertNotEqual(transition["next_state"][0], 0.1)
        other, _ = env.reset(seed=4)
        self.assertNotEqual(other[2], target)

    def test_drift_has_zero_context_rate(self):
        env = ContextToyEnv(context_config(), seed=0)
        env.reset(seed=0)
        before = env.observe()
        state = np.array([0.3, -0.1, 0.7, 0.5, -1.0])
        drift = env.mujoco_drift(state, np.array([0.4, -0.6]))
        np.testing.assert_allclose(
            drift, [0.5, -1.0, 0.0, (0.4 - 0.5 * 0.5) / 2.0, (-0.6 + 0.5 * 1.0) / 2.0], atol=1e-12
        )
        np.testing.assert_array_equal(env.observe(), before)
        np.testing.assert_array_equal(env.oracle_input_matrix(), np.eye(2))

    def test_explicit_state_and_snapshot_carry_the_context(self):
        env = ContextToyEnv(context_config(), seed=0)
        state, _ = env.reset(options={"state": np.array([0.0, 0.0, 0.25, 0.0, 0.0])})
        self.assertEqual(state[2], 0.25)
        snapshot = env.snapshot()
        env.reset(options={"state": np.array([0.0, 0.0, -0.75, 0.0, 0.0])})
        self.assertEqual(env.observe()[2], -0.75)
        env.restore(snapshot)
        self.assertEqual(env.observe()[2], 0.25)
        self.assertEqual(env.applied[-1].tolist(), [0.25])
        with self.assertRaises(ValueError):
            env.reset(options={"state": np.zeros(4)})

    def test_state_spec_validation(self):
        config = context_config()
        self.assertIs(validated_state_spec(config), config["state_spec"])
        self.assertIs(tasks.state_spec(config), config["state_spec"])
        for change in (
            {"observation_projection": np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])},
            {"periodic_position_indices": (2,)},
            {"actuated_position_indices": (0, 2)},
            {"context_dim": 4},
            {"context_dim": True},
        ):
            with self.subTest(change=str(change)):
                broken = dict(config, state_spec=dict(config["state_spec"], **change))
                with self.assertRaises(ValueError):
                    validated_state_spec(broken)
        with self.assertRaisesRegex(ValueError, "more positions"):
            ContextToyEnv(dict(config, state_spec=dict(
                config["state_spec"], position_dim=4, context_dim=1,
                observation_projection=np.vstack((np.eye(2), np.zeros((2, 2)))),
            )))


class TestIrregularClock(unittest.TestCase):
    """The optional clock settings of the dm_control testbeds of the final run (since
    2026-11-11; common.ph_dmc._common.with_clock_settings, the rows of mode irregular)."""

    WRAPPED = {
        "pendulum_swingup": "pendulum-swingup",
        "acrobot_swingup": "acrobot-swingup",
        "acrobot_swingup_sparse": "acrobot-swingup_sparse",
        "cartpole_swingup": "cartpole-swingup",
        "cartpole_balance": "cartpole-balance",
        "cartpole_balance_sparse": "cartpole-balance_sparse",
        "cartpole_swingup_sparse": "cartpole-swingup_sparse",
        "cartpole_two_poles": "cartpole-two_poles",
        "cartpole_three_poles": "cartpole-three_poles",
        "point_mass_easy": "point_mass-easy",
    }
    LAW = {"clock_kind": "two_tail_uniform", "tail_probability": 0.99, "tail_split": 0.9}

    def test_bounds_and_laws(self):
        self.assertEqual(irregular_clock_bounds(0.01), (0.002, 0.03))
        self.assertEqual(irregular_clock_bounds(0.02), (0.004, 0.06))
        self.assertEqual(irregular_clock_bounds(0.03), (0.006, 0.09))
        self.assertEqual(dmc_clock(0.01), fixed_clock(0.01))
        self.assertEqual(dmc_clock(0.01, clock_kind="fixed"), fixed_clock(0.01))
        clock = dmc_clock(0.02, **self.LAW)
        self.assertEqual((clock["min_duration"], clock["max_duration"]), (0.004, 0.06))
        self.assertEqual((clock["kind"], clock["reference_period"]), ("two_tail_uniform", 0.02))
        beta = dmc_clock(0.01, clock_kind="two_tail_beta", tail_probability=0.5, tail_split=0.5,
                         beta_shape=2.0)
        self.assertEqual(beta["beta_shape"], 2.0)
        for bad in (
            dict(clock_kind="fixed", tail_probability=0.8),
            dict(clock_kind="two_tail_uniform", tail_probability=0.8),
            dict(clock_kind="two_tail_uniform", tail_probability=0.8, tail_split=0.5, beta_shape=2.0),
            dict(clock_kind="poisson", tail_probability=0.8, tail_split=0.5),
        ):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                dmc_clock(0.01, **bad)
        # T, T/5 and 3T must be whole physics steps (no rounding, unlike CT-SAC's sampler,
        # so the stated law is the one drawn) and the minimum at least one step.
        self.assertEqual(irregular_clock_bounds(0.005), (0.001, 0.015))
        self.assertEqual(irregular_clock_bounds(0.025, 0.0025), (0.005, 0.075))  # CT-SAC's walker
        for period, physics_dt in ((0.001, 0.001), (0.002, 0.001), (0.012, 0.001),
                                   (0.0105, 0.001), (0.01, 0.003)):
            with self.subTest(period=period, physics_dt=physics_dt), self.assertRaises(ValueError):
                irregular_clock_bounds(period, physics_dt)

    def test_wrapped_factories_take_the_clock_and_keep_the_rest(self):
        from common.ph_time import clock_duration_pmf

        for testbed, env_id in self.WRAPPED.items():
            with self.subTest(testbed=testbed):
                spec = testbed_spec(testbed)
                self.assertEqual(spec.optional_settings_keys, CLOCK_SETTINGS_KEYS)
                settings = table.find_ph_entry(env_id, "top").config["testbed_settings"]
                fixed = spec.config_factory(**settings)
                self.assertEqual(fixed["clock"]["kind"], "fixed")
                irregular = spec.config_factory(**settings, **self.LAW)
                ph_config._validate_testbed(irregular)
                period = fixed["reference_period"]
                self.assertEqual(irregular["clock"], dmc_clock(period, fixed["physics_dt"], **self.LAW))
                self.assertEqual(sorted(irregular), sorted(fixed))
                for key in fixed:
                    if key not in ("clock", "state_spec", "controller"):
                        self.assertEqual(repr(irregular[key]), repr(fixed[key]), key)
                steps, probabilities = clock_duration_pmf(irregular["clock"])
                self.assertAlmostEqual(float((steps * probabilities).sum()) * fixed["physics_dt"],
                                       0.4912 * period, places=12)
                # The environment draws whole physics steps within the bounds, mean about 0.4912 T.
                env = tasks.make_env(irregular, decision_mode="torque", seed=0)
                env.reset(seed=0)
                # 40000 draws: the mean's standard error is ~0.9% (sd ~0.84 T against a mean of 0.4912 T).
                draws = np.array([env.clock.sample_duration(remaining_steps=10**6) for _ in range(40000)])
                ratio = draws / fixed["physics_dt"]
                np.testing.assert_allclose(ratio, np.round(ratio), atol=1e-9)
                self.assertGreaterEqual(draws.min(), irregular["clock"]["min_duration"] - 1e-12)
                self.assertLessEqual(draws.max(), irregular["clock"]["max_duration"] + 1e-12)
                self.assertLess(abs(draws.mean() / (0.4912 * period) - 1.0), 0.05)
                env.close()


if __name__ == "__main__":
    unittest.main()

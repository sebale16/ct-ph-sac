import copy
import math
import unittest

import numpy as np
import torch
from torch import func

from algorithms.explicit_ct_ph_sac import ExplicitCTPHSAC
from algorithms.ph_targets import data_rate_weights
from common.ph_config import acrobot_xk_config, windy_pendulum_config
from environment.ph_sac.controller import FeedbackController
from environment.ph_sac.tasks import (
    make_env,
    make_oracle_plant,
    oracle_callbacks,
    terminal_boundary_functions,
)
from models.ph_sac.losses import (
    critic_loss,
    explicit_actor_loss,
    implicit_actor_loss,
    plant_drift_loss,
    plant_likelihood_loss,
    rate_regression_loss,
    soft_value_loss,
    temperature_loss,
)
from models.ph_sac.networks import (
    LOG_STD_INIT,
    LOG_STD_MAX,
    LOG_STD_MIN,
    DecisionActor,
    RateHead,
    SmoothValueHead,
    StateFeatures,
    TwinQCritic,
)
from models.ph_sac.operators import (
    compress_diffusion_factor,
    generator,
    hessian_vector_product,
    policy_generator,
    policy_generator_terms,
    soft_reward_rate,
    value_derivatives,
)
from models.ph_sac.plant import OraclePlant, PortHamiltonianPlant, diffusion_is_learned
from models.ph_sac.rollouts import (
    augment_noise_injection,
    augmented_closed_loop_step,
    integrate_endpoint_moments,
    integration_grid,
    path_continuations,
    kinematic_noise_transport,
    model_substep,
    path_jacobian_norms,
    physical_noise_injection,
    project_physical_covariance,
    propagate_linearized_covariance,
    refresh_aligned_grid,
    sample_brownian_increments,
    semi_implicit_step,
    stochastic_rollout,
    survivor_continuation,
    wrapped_endpoint_residual,
)
from tests.test_ph_env import acrobot_config, cartpole_config, windy_config

NETWORK = {
    "hidden_sizes": (32, 32),
    "plant_hidden_sizes": (16, 16),
    "activation": "tanh",
    "mass_floor": 1e-2,
    "diffusion_floor": 1e-3,
    "damping_rank": 1,
}
PENDULUM_SPEC = {
    "position_dim": 1,
    "velocity_dim": 1,
    "action_dim": 1,
    "periodic_position_indices": (0,),
    "actuated_position_indices": (0,),
    "observation_projection": np.eye(1),
}
ACROBOT_SPEC = {
    "position_dim": 2,
    "velocity_dim": 2,
    "action_dim": 1,
    "periodic_position_indices": (0, 1),
    "actuated_position_indices": (1,),
    "observation_projection": np.eye(2),
}
# Two dofs with only the first observed: the second position is cyclic.
CYCLIC_SPEC = {
    "position_dim": 1,
    "velocity_dim": 2,
    "action_dim": 1,
    "periodic_position_indices": (0,),
    "actuated_position_indices": (0,),
    "observation_projection": np.array([[1.0, 0.0]]),
}
PENDULUM_CONTROLLER = {
    "decision_low": np.array([-0.5, 0.0, 0.0]),
    "decision_high": np.array([0.5, 100.0, 20.0]),
    "torque_low": np.array([-12.0]),
    "torque_high": np.array([12.0]),
    "setpoint_mode": "absolute",
    "setpoint_error": "wrapped",
    "refresh_steps": 1,
    # These tests cover the sample-and-hold machinery (refresh grids, held torques, the
    # augmented held-torque Jacobian); tests/test_ph_refresh_mode.py covers "continuous".
    "refresh_mode": "held",
}


def _controller(refresh_steps=1, spec=PENDULUM_SPEC, config=PENDULUM_CONTROLLER):
    return FeedbackController(dict(config, refresh_steps=refresh_steps), spec)


def _pendulum_oracle(sigma=3.0, mass=1.0, length=1.0, gravity=9.81):
    inertia = mass * length**2

    def drift(states, actions):
        theta, omega = states[:, :1], states[:, 1:]
        return torch.cat(
            (omega, (mass * gravity * length * theta.sin() + actions) / inertia), dim=-1
        )

    def diffusion(states):
        return torch.stack(
            (torch.zeros_like(states[:, 0]), sigma / inertia * torch.ones_like(states[:, 0])),
            dim=1,
        ).unsqueeze(-1)

    return OraclePlant(drift, diffusion, torch.ones(1, 1), state_spec=PENDULUM_SPEC)


def _free_motion_oracle(sigma=1.0):
    def drift(states, actions):
        return torch.cat((states[:, 1:], torch.zeros_like(states[:, 1:])), dim=-1)

    def diffusion(states):
        return torch.stack(
            (torch.zeros_like(states[:, 0]), sigma * torch.ones_like(states[:, 0])), dim=1
        ).unsqueeze(-1)

    return OraclePlant(drift, diffusion, torch.ones(1, 1), state_spec=PENDULUM_SPEC)


def _exact_pendulum_plant(mass=1.0, length=1.0, gravity=9.81, damping=0.2):
    """Plant with one-layer networks set to the rigid pendulum's exact coefficients."""
    config = dict(NETWORK, plant_hidden_sizes=())
    plant = PortHamiltonianPlant(PENDULUM_SPEC, config).double()
    inertia = mass * length**2
    with torch.no_grad():
        for net in (plant.potential_net, plant.mass_net, plant.damping_net):
            net.layers[-1].weight.zero_()
        # Features are (sin, cos), and U = m g l cos(theta) from upright.
        plant.potential_net.layers[-1].weight[0, 1] = mass * gravity * length
        plant.potential_net.layers[-1].bias.zero_()
        plant.mass_net.layers[-1].bias.fill_(
            math.log(math.expm1(math.sqrt(inertia) - plant.mass_floor))
        )
        plant.damping_net.layers[-1].bias.fill_(math.sqrt(damping))
        plant.force_matrix.fill_(1.0)
    return plant


class TestPlant(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def test_untrained_plant_matches_appendix(self):
        plant = PortHamiltonianPlant(ACROBOT_SPEC, dict(NETWORK, damping_rank=2)).double()
        y = torch.randn(32, 2).double()
        eye = torch.eye(2).double()
        self.assertLess((plant.mass_matrix(y) - eye).abs().max().item(), 0.05)
        self.assertLess((plant.damping_matrix(y) - 0.01 * eye).abs().max().item(), 0.01)
        self.assertLess(plant.potential_energy(y).abs().max().item(), 0.05)
        scale = plant.covariance(torch.randn(32, 4).double()).diagonal(dim1=-2, dim2=-1)
        self.assertLess(scale[:, :2].max().item(), 1e-3)
        self.assertGreater(scale[:, 2:].min().item(), 0.5)

    def test_initialize_from_physics(self):
        plant = PortHamiltonianPlant(PENDULUM_SPEC, NETWORK).double()
        plant.initialize_from_physics(torch.tensor([[2.0]]), torch.tensor(3.0))
        self.assertEqual(plant.input_matrix().item(), 2.0)
        covariance = plant.covariance(torch.zeros(1, 2).double())
        self.assertAlmostEqual(covariance[0, 1, 1].item(), 9.0, delta=1.0)

    def _hamilton_residual(self, spec, states, actions, plant):
        """Compare the drift with finite differences of H at fixed momentum."""
        n_o = spec["position_dim"]
        projection = torch.as_tensor(spec["observation_projection"]).double()
        positions, velocities = states[:, :n_o], states[:, n_o:]
        epsilon = 1e-5
        mass = plant.mass_matrix(positions)
        momenta = (mass @ velocities.unsqueeze(-1)).squeeze(-1)

        grad_y = []
        mass_rate = torch.zeros_like(mass)
        for j in range(n_o):
            step = torch.zeros(n_o).double()
            step[j] = epsilon
            grad_y.append(
                (
                    plant.hamiltonian(positions + step, momenta)
                    - plant.hamiltonian(positions - step, momenta)
                )
                / (2 * epsilon)
            )
            dmass = (plant.mass_matrix(positions + step) - plant.mass_matrix(positions - step)) / (
                2 * epsilon
            )
            mass_rate = mass_rate + (velocities @ projection.T)[:, j, None, None] * dmass
        grad_q = torch.cat(grad_y, dim=-1) @ projection
        force = (
            -grad_q
            - (plant.damping_matrix(positions) @ velocities.unsqueeze(-1)).squeeze(-1)
            + actions @ plant.input_matrix().T
        )
        rhs = force - (mass_rate @ velocities.unsqueeze(-1)).squeeze(-1)
        acceleration = torch.linalg.solve(mass, rhs.unsqueeze(-1)).squeeze(-1)
        expected = torch.cat((velocities @ projection.T, acceleration), dim=-1)
        return (plant.drift(states, actions) - expected).abs().max().item()

    def test_drift_is_hamiltons_equations(self):
        for spec, rank in ((PENDULUM_SPEC, 1), (ACROBOT_SPEC, 2), (CYCLIC_SPEC, 2)):
            plant = PortHamiltonianPlant(spec, dict(NETWORK, damping_rank=rank)).double()
            # Move away from the near-constant initialization.
            with torch.no_grad():
                for net in (plant.potential_net, plant.mass_net, plant.damping_net):
                    net.layers[-1].weight.mul_(100.0)
            states = torch.randn(16, spec["position_dim"] + spec["velocity_dim"]).double()
            actions = torch.randn(16, spec["action_dim"]).double()
            self.assertLess(self._hamilton_residual(spec, states, actions, plant), 1e-7)

    def test_known_pendulum_recovered(self):
        plant = _exact_pendulum_plant(mass=1.0, length=2.0, gravity=9.81, damping=0.2)
        states = torch.tensor([[0.3, -0.7], [-1.1, 2.0], [0.0, 0.0]]).double()
        actions = torch.tensor([[1.5], [-2.0], [0.0]]).double()
        drift = plant.drift(states, actions)
        inertia = 4.0
        expected = (
            1.0 * 9.81 * 2.0 * states[:, 0].sin() - 0.2 * states[:, 1] + actions[:, 0]
        ) / inertia
        torch.testing.assert_close(drift[:, 0], states[:, 1])
        torch.testing.assert_close(drift[:, 1], expected)

    def test_analytic_jacobian_matches_autodiff(self):
        for spec, rank in ((PENDULUM_SPEC, 1), (ACROBOT_SPEC, 2), (CYCLIC_SPEC, 2)):
            plant = PortHamiltonianPlant(spec, dict(NETWORK, damping_rank=rank)).double()
            with torch.no_grad():
                for net in (plant.potential_net, plant.mass_net, plant.damping_net):
                    net.layers[-1].weight.mul_(100.0)
            states = torch.randn(8, spec["position_dim"] + spec["velocity_dim"]).double()
            actions = torch.randn(8, spec["action_dim"]).double()
            drift, by_state, by_action = plant.drift_jacobian(states, actions)

            def row(state, action):
                return plant.drift(state.unsqueeze(0), action.unsqueeze(0)).squeeze(0)

            expect_s, expect_a = func.vmap(func.jacfwd(row, argnums=(0, 1)))(states, actions)
            torch.testing.assert_close(drift, plant.drift(states, actions))
            torch.testing.assert_close(by_state, expect_s, atol=1e-9, rtol=1e-7)
            torch.testing.assert_close(by_action, expect_a, atol=1e-9, rtol=1e-7)

    def test_torch_func_composes_and_reaches_parameters(self):
        plant = PortHamiltonianPlant(ACROBOT_SPEC, dict(NETWORK, damping_rank=2))
        states, actions = torch.randn(4, 4), torch.randn(4, 1)
        jacobians = func.vmap(
            func.jacrev(
                lambda s, a: plant.drift(s.unsqueeze(0), a.unsqueeze(0)).squeeze(0)
            )
        )(states, actions)
        jacobians.square().sum().backward()
        self.assertGreater(plant.potential_net.layers[0].weight.grad.abs().sum().item(), 0.0)
        diffusions = func.vmap(
            lambda s: plant.diffusion(s.unsqueeze(0)).squeeze(0)
        )(states)
        self.assertEqual(diffusions.shape, (4, 4, 8))

    def test_diffusion_and_covariance_agree_and_are_floored(self):
        plant = PortHamiltonianPlant(ACROBOT_SPEC, dict(NETWORK, damping_rank=2)).double()
        states = torch.randn(5, 4).double()
        factor = plant.diffusion(states)
        self.assertEqual(factor.shape, (5, 4, plant.noise_dim))
        torch.testing.assert_close(factor @ factor.mT, plant.covariance(states))
        eigenvalues = torch.linalg.eigvalsh(plant.covariance(states))
        self.assertGreaterEqual(eigenvalues.min().item(), 1e-6 * 0.99)

    def test_frozen_copy_detaches(self):
        plant = PortHamiltonianPlant(PENDULUM_SPEC, NETWORK)
        frozen = plant.frozen_copy()
        self.assertFalse(any(p.requires_grad for p in frozen.parameters()))
        self.assertTrue(all(p.requires_grad for p in plant.parameters()))
        with torch.no_grad():
            plant.force_matrix.add_(1.0)
        self.assertNotEqual(frozen.input_matrix().item(), plant.input_matrix().item())
        oracle = _pendulum_oracle()
        self.assertEqual(oracle.frozen_copy().noise_dim, 1)


class TestNetworks(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.low = torch.tensor([-0.5, 0.0, 0.0])
        self.high = torch.tensor([0.5, 100.0, 20.0])

    def test_acrobot_features_include_the_unwrapped_elbow(self):
        """sin/cos of both angles, scaled rates, then the elbow over its travel limit; one turn apart differ."""
        limit = 4.0 * math.pi
        spec = {
            "position_dim": 2, "velocity_dim": 2, "action_dim": 1,
            "periodic_position_indices": (0, 1), "actuated_position_indices": (1,),
            "observation_projection": np.eye(2), "travel_position_scales": ((1, limit),),
        }
        features = StateFeatures(spec, {"velocity_feature_scale": 2.0 * math.pi})
        self.assertEqual(features.output_dim, 7)
        states = torch.tensor([[0.3, 0.2, 1.0, -2.0], [0.3, 0.2 + 2.0 * math.pi, 1.0, -2.0]], dtype=torch.float64)
        out = features(states)
        self.assertEqual(out.shape, (2, 7))
        self.assertTrue(torch.equal(out[:, -1:], states[:, 1:2] / limit))
        self.assertTrue(torch.allclose(out[0, :-1], out[1, :-1]))
        self.assertNotEqual(out[0, -1].item(), out[1, -1].item())
        plain = {key: value for key, value in spec.items() if key != "travel_position_scales"}
        self.assertTrue(torch.equal(out[:, :-1], StateFeatures(plain, {"velocity_feature_scale": 2.0 * math.pi})(states)))

    def test_actor_log_std_starts_at_the_default_or_the_network_key(self):
        """Absent: every entry at LOG_STD_INIT; network actor_log_std_init sets every entry,
        and nothing else of the actor or of the global stream moves."""
        torch.manual_seed(5)
        plain = DecisionActor(PENDULUM_SPEC, self.low, self.high, NETWORK)
        plain_stream = torch.get_rng_state()
        self.assertTrue(torch.equal(plain.log_std.detach(), torch.full((3,), LOG_STD_INIT)))
        for start, expected in ((-0.25, -0.25), (0, 0.0), (LOG_STD_MIN, LOG_STD_MIN), (LOG_STD_MAX, LOG_STD_MAX)):
            with self.subTest(start=start):
                torch.manual_seed(5)
                keyed = DecisionActor(
                    PENDULUM_SPEC, self.low, self.high, dict(NETWORK, actor_log_std_init=start)
                )
                self.assertTrue(torch.equal(torch.get_rng_state(), plain_stream))
                self.assertEqual(keyed.log_std.dtype, plain.log_std.dtype)
                self.assertEqual(keyed.log_std.detach().tolist(), [expected] * 3)
                self.assertTrue(keyed.log_std.requires_grad)
                named = dict(keyed.named_parameters())
                for name, parameter in plain.named_parameters():
                    if name != "log_std":
                        self.assertTrue(torch.equal(parameter, named[name]), name)
        for bad in (float("nan"), float("inf"), LOG_STD_MIN - 0.5, LOG_STD_MAX + 0.5, True, "-1", None):
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, "actor_log_std_init"):
                DecisionActor(PENDULUM_SPEC, self.low, self.high, dict(NETWORK, actor_log_std_init=bad))

    def test_actor_shapes_and_box(self):
        actor = DecisionActor(PENDULUM_SPEC, self.low, self.high, NETWORK)
        sample = actor.sample(torch.randn(8, 2), sample_count=5)
        self.assertEqual(sample["decisions"].shape, (8, 5, 3))
        self.assertEqual(sample["log_probabilities"].shape, (8, 5, 1))
        self.assertTrue(((sample["decisions"] >= self.low) & (sample["decisions"] <= self.high)).all())
        self.assertEqual(actor.log_standard_deviation().shape, (3,))

    def test_sample_and_log_prob_agree(self):
        actor = DecisionActor(PENDULUM_SPEC, self.low, self.high, NETWORK).double()
        states = torch.randn(8, 2).double()
        sample = actor.sample(states, sample_count=4)
        torch.testing.assert_close(
            actor.log_prob(states, sample["decisions"]),
            sample["log_probabilities"],
            atol=1e-6,
            rtol=1e-6,
        )
        flat = actor.log_prob(states, sample["decisions"][:, 0])
        torch.testing.assert_close(flat, sample["log_probabilities"][:, 0], atol=1e-6, rtol=1e-6)

    def test_log_prob_normalizes_over_the_normalized_box(self):
        """exp(log pi) integrates to one over u=(kappa-centre)/h; less log h, over kappa."""
        low, high = torch.tensor([-2.0]), torch.tensor([3.0])
        actor = DecisionActor(PENDULUM_SPEC, low, high, NETWORK).double()
        state = torch.zeros(1, 2).double()
        grid = torch.linspace(-2.0 + 1e-6, 3.0 - 1e-6, 40001).double().reshape(1, -1, 1)
        log_density = actor.log_prob(state, grid).reshape(-1)
        decisions = grid.reshape(-1)
        normalized = (decisions - 0.5) / 2.5
        integral = torch.trapezoid(log_density.exp(), normalized).item()
        self.assertAlmostEqual(integral, 1.0, places=3)
        physical = (log_density - math.log(2.5)).exp()
        self.assertAlmostEqual(torch.trapezoid(physical, decisions).item(), 1.0, places=3)

    def _twin(self, actor, low, high):
        """The same networks on another box."""
        twin = DecisionActor(PENDULUM_SPEC, low, high, NETWORK).double()
        twin.mean_network.load_state_dict(actor.mean_network.state_dict())
        twin.log_std.data.copy_(actor.log_std.data)
        return twin

    def test_log_density_is_over_the_normalized_box(self):
        """log pi_u(u) = log pi_kappa(kappa) + sum(log h): the tanh-Gaussian density of u.

        It equals the same networks' density on [-1,1]^3, where sum(log h) = 0, and the
        tanh-Gaussian density written out without the affine Jacobian.
        """
        actor = DecisionActor(PENDULUM_SPEC, self.low, self.high, NETWORK).double()
        unit = self._twin(actor, -torch.ones(3), torch.ones(3))
        states = torch.randn(8, 2).double()
        noise = torch.randn(8, 4, 3).double()
        sample = actor.sample(states, sample_count=4, noise=noise)
        unit_sample = unit.sample(states, sample_count=4, noise=noise)
        centre = 0.5 * (self.high + self.low).double()
        half = 0.5 * (self.high - self.low).double()
        normalized = (sample["decisions"] - centre) / half
        torch.testing.assert_close(unit_sample["decisions"], normalized)
        torch.testing.assert_close(sample["log_probabilities"], unit_sample["log_probabilities"])
        torch.testing.assert_close(
            actor.log_prob(states, sample["decisions"]),
            unit.log_prob(states, normalized),
            atol=1e-6,
            rtol=1e-6,
        )
        # Written out: a Gaussian at atanh(u) with the tanh Jacobian, no affine Jacobian.
        mean = actor(states).unsqueeze(1)
        deviation = actor.log_standard_deviation().exp()
        gaussian = torch.distributions.Normal(mean, deviation).log_prob(torch.atanh(normalized))
        expected = (gaussian - torch.log1p(-normalized.square())).sum(dim=-1, keepdim=True)
        torch.testing.assert_close(sample["log_probabilities"], expected, atol=1e-6, rtol=1e-6)

    def test_box_units_leave_the_density_and_entropy_charge_unchanged(self):
        """Gains in other units (times 100) move the decisions but not log pi or its charge."""
        scale = torch.tensor([1.0, 100.0, 100.0])
        actor = DecisionActor(PENDULUM_SPEC, self.low, self.high, NETWORK).double()
        scaled = self._twin(actor, self.low * scale, self.high * scale)
        states = torch.randn(8, 2).double()
        noise = torch.randn(8, 4, 3).double()
        first = actor.sample(states, sample_count=4, noise=noise)
        second = scaled.sample(states, sample_count=4, noise=noise)
        torch.testing.assert_close(second["decisions"], first["decisions"] * scale.double())
        torch.testing.assert_close(second["log_probabilities"], first["log_probabilities"])
        torch.testing.assert_close(
            scaled.log_prob(states, second["decisions"]),
            actor.log_prob(states, first["decisions"]),
            atol=1e-6,
            rtol=1e-6,
        )
        # Physical-box densities would differ by log(100) per rescaled coordinate.
        shift = torch.log(0.5 * (scaled.decision_high - scaled.decision_low)).sum() - torch.log(
            0.5 * (actor.decision_high - actor.decision_low)
        ).sum()
        self.assertAlmostEqual(shift.item(), 2.0 * math.log(100.0), places=6)
        # The charge: the implicit actor loss on a rate head over each box, and the
        # temperature loss, are the same in both units.
        rate = RateHead(
            PENDULUM_SPEC, 3, NETWORK, decision_low=self.low, decision_high=self.high
        ).double()
        scaled_rate = RateHead(
            PENDULUM_SPEC,
            3,
            NETWORK,
            decision_low=self.low * scale,
            decision_high=self.high * scale,
        ).double()
        scaled_rate.network.load_state_dict(rate.network.state_dict())
        alpha = torch.tensor(0.1, dtype=torch.float64)
        torch.testing.assert_close(
            implicit_actor_loss(scaled_rate, states, second, alpha),
            implicit_actor_loss(rate, states, first, alpha),
        )
        log_alpha = torch.tensor(math.log(0.1), dtype=torch.float64)
        weights = torch.ones(8, 1, dtype=torch.float64)
        torch.testing.assert_close(
            temperature_loss(log_alpha, second["log_probabilities"], -3.0, interval_weights=weights),
            temperature_loss(log_alpha, first["log_probabilities"], -3.0, interval_weights=weights),
        )

    def test_deterministic_uses_mean(self):
        actor = DecisionActor(PENDULUM_SPEC, self.low, self.high, NETWORK)
        states = torch.randn(4, 2)
        first = actor.sample(states, deterministic=True)["decisions"]
        second = actor.sample(states, deterministic=True)["decisions"]
        torch.testing.assert_close(first, second)
        expected = self.low + 0.5 * (torch.tanh(actor(states)) + 1.0) * (self.high - self.low)
        torch.testing.assert_close(first[:, 0], expected)

    def test_reparameterized_gradient_reaches_actor(self):
        actor = DecisionActor(PENDULUM_SPEC, self.low, self.high, NETWORK)
        actor.sample(torch.randn(8, 2))["decisions"].sum().backward()
        self.assertGreater(actor.log_std.grad.abs().sum().item(), 0.0)
        self.assertIsNotNone(actor.mean_network[0].weight.grad)

    def test_value_boundary_is_hard(self):
        limit = math.pi / 2.0
        payoff = -10.0
        value = SmoothValueHead(PENDULUM_SPEC, NETWORK)
        value.configure_terminal_boundary(
            lambda s: torch.full_like(s[..., :1], payoff),
            lambda s: (1.0 - (s[..., :1] / limit) ** 2)
            * (1.0 - (s[..., 1:] / (4 * math.pi)) ** 2),
        )
        boundary = torch.tensor([[limit, 1.0], [-limit, -3.0], [0.2, 4 * math.pi]])
        torch.testing.assert_close(
            value(boundary), torch.full((3, 1), payoff), atol=1e-4, rtol=1e-4
        )
        copy = value.frozen_copy()
        torch.testing.assert_close(copy(boundary), value(boundary))
        self.assertFalse(any(p.requires_grad for p in copy.parameters()))
        lifted = value.boundary_lift(boundary, torch.zeros(3, 1))
        torch.testing.assert_close(lifted, torch.full((3, 1), payoff))
        plain = SmoothValueHead(PENDULUM_SPEC, NETWORK)
        torch.testing.assert_close(
            plain.boundary_lift(boundary, torch.ones(3, 1)), torch.ones(3, 1)
        )

    def test_features_are_periodic(self):
        value = SmoothValueHead(PENDULUM_SPEC, NETWORK)
        state = torch.tensor([[0.3, 1.0]])
        shifted = torch.tensor([[0.3 + 2 * math.pi, 1.0]])
        torch.testing.assert_close(value(state), value(shifted), atol=1e-5, rtol=1e-5)

    def test_rate_and_critics(self):
        states, decisions = torch.randn(6, 4, 2), torch.rand(6, 4, 3)
        rate = RateHead(PENDULUM_SPEC, 3, NETWORK, decision_low=self.low, decision_high=self.high)
        self.assertEqual(rate(states, decisions).shape, (6, 4, 1))
        critics = TwinQCritic(PENDULUM_SPEC, 3, NETWORK)
        first, second = critics(states, decisions)
        self.assertFalse(torch.equal(first, second))
        torch.testing.assert_close(critics.minimum(states, decisions), torch.min(first, second))
        self.assertFalse(any(p.requires_grad for p in critics.frozen_copy().parameters()))
        # The box scaling sends the corners to +-1.
        self.assertTrue(torch.equal(rate.scaling(self.low), -torch.ones(3)))
        self.assertTrue(torch.equal(rate.scaling(self.high), torch.ones(3)))


class _QuadraticValue(torch.nn.Module):
    """V(s)=s^T Q s/2 + c^T s, with known derivatives."""

    def __init__(self, quadratic, linear):
        super().__init__()
        self.quadratic = quadratic
        self.linear = linear

    def forward(self, states):
        return (
            0.5 * (states @ self.quadratic * states).sum(-1, keepdim=True)
            + (states * self.linear).sum(-1, keepdim=True)
        )


class TestOperators(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.quadratic = torch.tensor([[2.0, 0.5], [0.5, 3.0]]).double()
        self.linear = torch.tensor([0.1, -0.2]).double()
        self.value = _QuadraticValue(self.quadratic, self.linear)

    def test_compress_preserves_covariance(self):
        for rows, columns in ((4, 8), (4, 4), (4, 2), (2, 1)):
            factor = torch.randn(5, rows, columns).double()
            compact = compress_diffusion_factor(factor)
            self.assertLessEqual(compact.shape[-1], rows)
            torch.testing.assert_close(compact @ compact.mT, factor @ factor.mT)
        singular = torch.randn(3, 4, 1).double().repeat(1, 1, 6)
        compact = compress_diffusion_factor(singular)
        self.assertEqual(compact.shape, (3, 4, 4))
        torch.testing.assert_close(compact @ compact.mT, singular @ singular.mT)
        zero = torch.zeros(3, 4, 6).double()
        torch.testing.assert_close(compress_diffusion_factor(zero), torch.zeros(3, 4, 4).double())

    def test_derivatives_on_quadratic(self):
        states = torch.randn(6, 2).double()
        diffusion = torch.randn(6, 2, 4).double()
        result = value_derivatives(self.value, states, diffusion, create_graph=False)
        expected_gradient = states @ self.quadratic + self.linear
        torch.testing.assert_close(result["gradients"], expected_gradient)
        covariance = diffusion @ diffusion.mT
        trace = (covariance * self.quadratic).sum((-2, -1), keepdim=False).unsqueeze(-1)
        torch.testing.assert_close(result["diffusion_traces"], trace)
        self.assertFalse(result["values"].requires_grad)
        vectors = torch.randn(6, 2).double()
        torch.testing.assert_close(
            hessian_vector_product(self.value, states, vectors, create_graph=False),
            vectors @ self.quadratic,
        )

    def test_generator_on_quadratic(self):
        plant = _pendulum_oracle(sigma=3.0)
        plant = plant.double()
        controller = _controller()
        states = torch.tensor([[0.2, -0.3], [-0.1, 0.5]]).double()
        decisions = torch.tensor([[0.0, 10.0, 2.0], [0.1, 5.0, 1.0]]).double()
        result = generator(
            self.value, plant, controller, states, decisions, create_graph=False
        )
        torques = controller.action(states, decisions)
        drift = plant.drift(states, torques)
        gradient = states @ self.quadratic + self.linear
        expected = (drift * gradient).sum(-1, keepdim=True) + 0.5 * 9.0 * self.quadratic[1, 1]
        torch.testing.assert_close(result, expected)

    def test_gradient_reaches_value_parameters(self):
        value = SmoothValueHead(PENDULUM_SPEC, NETWORK)
        plant = _pendulum_oracle()
        controller = _controller()
        states = torch.randn(4, 2)
        decisions = torch.tensor([0.0, 10.0, 2.0]).expand(4, 1, 3)
        output = policy_generator(
            value, plant, controller, states, decisions, create_graph=True
        )
        output.sum().backward()
        self.assertGreater(value.network[0].weight.grad.abs().sum().item(), 0.0)
        detached = policy_generator(
            value, plant, controller, states, decisions, create_graph=False
        )
        self.assertFalse(detached.requires_grad)

    def test_policy_generator_per_draw_and_averaged(self):
        plant = _pendulum_oracle().double()
        controller = _controller()
        states = torch.randn(3, 2).double()
        draws = torch.rand(3, 5, 3).double() * torch.tensor([0.5, 50.0, 10.0]).double()
        per_draw = policy_generator(
            self.value, plant, controller, states, draws, create_graph=False
        )
        self.assertEqual(per_draw.shape, (3, 5, 1))
        for i in range(5):
            single = generator(
                self.value, plant, controller, states, draws[:, i], create_graph=False
            )
            torch.testing.assert_close(per_draw[:, i], single)
        averaged = policy_generator(
            self.value,
            plant,
            controller,
            states,
            draws,
            create_graph=False,
            average_plant_terms_first=True,
        )
        self.assertEqual(averaged.shape, (3, 1, 1))
        torch.testing.assert_close(averaged[:, 0], per_draw.mean(1))
        derivatives, generated = policy_generator_terms(
            self.value, plant, controller, states, draws, create_graph=False
        )
        torch.testing.assert_close(generated, per_draw)
        self.assertEqual(derivatives["values"].shape, (3, 1))

    def test_offset_setpoint_resolves_at_start_state(self):
        controller = _controller(
            spec=ACROBOT_SPEC,
            config=dict(
                PENDULUM_CONTROLLER,
                decision_low=np.array([-1.0, 0.0, 0.0]),
                decision_high=np.array([1.0, 100.0, 20.0]),
                setpoint_mode="offset",
                setpoint_error="unwrapped",
            ),
        )
        plant = PortHamiltonianPlant(ACROBOT_SPEC, dict(NETWORK, damping_rank=2))
        value = SmoothValueHead(ACROBOT_SPEC, NETWORK)
        states = torch.randn(2, 4)
        draws = torch.tensor([[[0.5, 10.0, 1.0]], [[-0.5, 5.0, 2.0]]])
        by_draws = policy_generator(value, plant, controller, states, draws, create_graph=False)
        resolved = controller.resolve_decision(states, draws[:, 0])
        direct = generator(value, plant, controller, states, resolved, create_graph=False)
        torch.testing.assert_close(by_draws[:, 0], direct)

    def test_soft_reward_rate(self):
        controller = _controller()
        states = torch.randn(3, 2)
        draws = torch.tensor([0.1, 10.0, 2.0]).expand(3, 2, 3)
        logs = torch.randn(3, 2, 1)

        def rate(flat_states, torques):
            return -flat_states[:, :1] ** 2 - 0.1 * torques**2

        result = soft_reward_rate(states, draws, logs, controller, rate, torch.tensor(0.5))
        torques = controller.action(states.unsqueeze(1).expand(-1, 2, -1), draws)
        expected = -states[:, None, :1] ** 2 - 0.1 * torques**2 - 0.5 * logs
        torch.testing.assert_close(result, expected)


class TestGrid(unittest.TestCase):
    def test_equal_steps_when_h_is_physics_step(self):
        durations = torch.tensor([[0.002], [0.018], [0.001]], dtype=torch.float32)
        counts, sizes = integration_grid(
            durations, 0.001, physics_dt=0.001, controller_refresh_steps=1
        )
        self.assertEqual(counts.tolist(), [2, 18, 1])
        self.assertEqual(sizes.shape, (3, 18))
        self.assertTrue(torch.all(sizes[1] == sizes[1][0]))
        self.assertEqual(sizes[0, 2:].abs().sum().item(), 0.0)
        torch.testing.assert_close(sizes.sum(-1, keepdim=True), durations)

    def test_grid_contains_every_refresh_when_h_exceeds_delta(self):
        for steps, refresh, h_steps in ((10, 5, 2.5), (10, 3, 2.0), (7, 2, 3.0), (12, 4, 4.0)):
            duration = torch.tensor([steps * 0.001], dtype=torch.float64)
            counts, sizes, flags = refresh_aligned_grid(
                duration, h_steps * 0.001, physics_dt=0.001, controller_refresh_steps=refresh
            )
            grid = torch.cat((torch.zeros(1).double(), sizes[0, : counts[0]].cumsum(0)))
            for time in range(0, steps, refresh):
                self.assertTrue(((grid - time * 0.001).abs() < 1e-12).any())
            self.assertAlmostEqual(grid[-1].item(), steps * 0.001, places=12)
            self.assertLessEqual(sizes.max().item(), h_steps * 0.001 + 1e-12)
            starts = grid[:-1][flags[0, : counts[0]]]
            expected = torch.arange(0, steps, refresh).double() * 0.001
            torch.testing.assert_close(starts, expected, atol=1e-12, rtol=0)

    def test_unequal_substeps_only_where_needed(self):
        duration = torch.tensor([0.010], dtype=torch.float64)
        _, sizes, flags = refresh_aligned_grid(
            duration, 0.002, physics_dt=0.001, controller_refresh_steps=3
        )
        expected = [0.0015] * 6 + [0.001]
        torch.testing.assert_close(
            sizes[0], torch.tensor(expected, dtype=torch.float64), atol=1e-12, rtol=0
        )
        self.assertEqual(flags[0].tolist(), [True, False, True, False, True, False, True])

    def test_rejects_fractional_physics_steps(self):
        with self.assertRaises(ValueError):
            integration_grid(
                torch.tensor([0.0025]), 0.001, physics_dt=0.001, controller_refresh_steps=1
            )

    def test_without_clock_arguments(self):
        counts, sizes = integration_grid(torch.tensor([0.01, 0.015]).double(), 0.004)
        self.assertEqual(counts.tolist(), [3, 4])
        torch.testing.assert_close(sizes.sum(-1), torch.tensor([0.01, 0.015]).double())
        self.assertEqual(sizes[0, 3].item(), 0.0)

    def test_brownian_scaling_seed_and_padding(self):
        durations = torch.tensor([0.002, 0.018]).double()
        kwargs = dict(
            path_count=20000,
            noise_dim=2,
            maximum_step=0.001,
            physics_dt=0.001,
            controller_refresh_steps=1,
        )
        first = sample_brownian_increments(durations, seed=3, **kwargs)
        second = sample_brownian_increments(durations, seed=3, **kwargs)
        self.assertTrue(torch.equal(first, second))
        self.assertEqual(first.shape, (2, 20000, 18, 2))
        self.assertEqual(first[0, :, 2:].abs().sum().item(), 0.0)
        self.assertAlmostEqual(first[1].var().item(), 0.001, delta=0.00003)


class TestSemiImplicit(unittest.TestCase):
    def test_step_matches_hand_formula(self):
        plant = _pendulum_oracle(sigma=3.0).double()
        states = torch.tensor([[0.3, -0.5]]).double()
        actions = torch.tensor([[2.0]]).double()
        h = torch.tensor([0.01]).double()
        kick = torch.tensor([[0.07]]).double()

        acceleration = 9.81 * math.sin(0.3) + 2.0
        velocity = -0.5 + 0.01 * acceleration
        result = semi_implicit_step(plant, states, actions, h)
        torch.testing.assert_close(
            result, torch.tensor([[0.3 + 0.01 * velocity, velocity]]).double()
        )

        noisy_velocity = velocity + 3.0 * 0.07
        noisy = semi_implicit_step(plant, states, actions, h, brownian_increments=kick)
        torch.testing.assert_close(
            noisy, torch.tensor([[0.3 + 0.01 * noisy_velocity, noisy_velocity]]).double()
        )

    def test_step_equals_psi_plus_transported_noise(self):
        plant = PortHamiltonianPlant(ACROBOT_SPEC, dict(NETWORK, damping_rank=2)).double()
        states = torch.randn(6, 4).double()
        actions = torch.randn(6, 1).double()
        h = torch.full((6,), 0.01, dtype=torch.float64)
        kicks = torch.randn(6, plant.noise_dim).double() * math.sqrt(0.01)

        drift = plant.drift(states, actions)
        transport = kinematic_noise_transport(ACROBOT_SPEC, h)
        kinematic = (transport - torch.eye(4).double()) / 0.01
        psi = states + 0.01 * drift + 0.01**2 * (kinematic @ drift.unsqueeze(-1)).squeeze(-1)
        injection = physical_noise_injection(plant, states, h)
        expected = psi + (injection @ kicks.unsqueeze(-1)).squeeze(-1)
        torch.testing.assert_close(
            semi_implicit_step(plant, states, actions, h, brownian_increments=kicks),
            expected,
            atol=1e-12,
            rtol=1e-10,
        )

    def test_zero_step_is_exact_no_op(self):
        plant = PortHamiltonianPlant(PENDULUM_SPEC, NETWORK)
        states = torch.randn(4, 2)
        result = semi_implicit_step(
            plant,
            states,
            torch.randn(4, 1),
            torch.zeros(4),
            brownian_increments=torch.zeros(4, plant.noise_dim),
        )
        self.assertTrue(torch.equal(result, states))

    def test_noise_injection_blocks(self):
        injection = torch.randn(3, 2, 4)
        augmented = augment_noise_injection(injection, 1)
        self.assertEqual(augmented.shape, (3, 3, 4))
        self.assertTrue(torch.equal(augmented[:, :2], injection))
        self.assertEqual(augmented[:, 2].abs().sum().item(), 0.0)

    def test_projection_keeps_physical_block(self):
        covariance = torch.randn(2, 5, 5)
        projected = project_physical_covariance(covariance, ACROBOT_SPEC)
        self.assertTrue(torch.equal(projected, covariance[:, :4, :4]))

    def test_wrapped_residual_only_on_periodic(self):
        spec = dict(PENDULUM_SPEC)
        observed = torch.tensor([[math.pi - 0.1, 5.0]]).double()
        predicted = torch.tensor([[-math.pi + 0.1, -5.0]]).double()
        residual = wrapped_endpoint_residual(observed, predicted, spec)
        torch.testing.assert_close(residual, torch.tensor([[-0.2, 10.0]]).double())
        unwrapped = dict(spec, periodic_position_indices=())
        residual = wrapped_endpoint_residual(observed, predicted, unwrapped)
        torch.testing.assert_close(
            residual, torch.tensor([[2 * math.pi - 0.2, 10.0]]).double()
        )


def _numeric_jacobian(function, point, epsilon=1e-6):
    columns = []
    for i in range(point.shape[-1]):
        step = torch.zeros_like(point)
        step[:, i] = epsilon
        columns.append((function(point + step) - function(point - step)) / (2 * epsilon))
    return torch.stack(columns, dim=-1)


class TestStepJacobian(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1)

    def _check(self, plant, spec, mask_value, integrator="rk4"):
        controller = _controller(spec=spec) if spec is PENDULUM_SPEC else _controller(
            spec=spec,
            config=dict(
                PENDULUM_CONTROLLER, setpoint_mode="absolute", setpoint_error="unwrapped"
            ),
        )
        size = spec["position_dim"] + spec["velocity_dim"]
        states = 0.2 * torch.randn(5, size).double()
        held = 0.5 * torch.randn(5, spec["action_dim"]).double()
        # Small gains keep the clip inactive, where the law is differentiable.
        resolved = torch.tensor([0.05, 2.0, 0.5]).double().expand(5, 3)
        h = torch.full((5,), 0.002).double()
        mask = torch.full((5,), mask_value)
        augmented = torch.cat((states, held), dim=-1)

        def step(point):
            next_states, next_actions, _ = augmented_closed_loop_step(
                plant,
                controller,
                point[:, :size],
                point[:, size:],
                resolved,
                h,
                mask,
                integrator=integrator,
            )
            return torch.cat((next_states, next_actions), dim=-1)

        with torch.no_grad():
            numeric = _numeric_jacobian(step, augmented)
        following, _, analytic = augmented_closed_loop_step(
            plant, controller, states, held, resolved, h, mask, integrator=integrator
        )
        torch.testing.assert_close(analytic, numeric, atol=1e-6, rtol=1e-5)
        # The state block's mean is the noise-free model_substep at the step's torque.
        torque = torch.where(
            mask.reshape(-1, 1), controller.action(states, resolved), held
        )
        torch.testing.assert_close(
            following, model_substep(plant, states, torque, h, integrator=integrator)
        )
        return analytic

    def test_refresh_and_hold_on_oracle(self):
        plant = _pendulum_oracle().double()
        for integrator in ("rk4", "semi_implicit"):
            with self.subTest(integrator=integrator):
                refreshed = self._check(plant, PENDULUM_SPEC, True, integrator)
                # A refresh discards the stored torque.
                self.assertEqual(refreshed[:, :, 2:].abs().max().item(), 0.0)
                held = self._check(plant, PENDULUM_SPEC, False, integrator)
                # Between refreshes, the stored torque is unchanged and ignores the state.
                torch.testing.assert_close(held[:, 2, 2], torch.ones(5).double())
                self.assertEqual(held[:, 2, :2].abs().max().item(), 0.0)

    def test_refresh_and_hold_on_learned_plant(self):
        plant = PortHamiltonianPlant(ACROBOT_SPEC, dict(NETWORK, damping_rank=2)).double()
        with torch.no_grad():
            for net in (plant.potential_net, plant.mass_net, plant.damping_net):
                net.layers[-1].weight.mul_(50.0)
        for integrator in ("rk4", "semi_implicit"):
            with self.subTest(integrator=integrator):
                self._check(plant, ACROBOT_SPEC, True, integrator)
                self._check(plant, ACROBOT_SPEC, False, integrator)

    def test_propagate_formula(self):
        covariance = torch.randn(2, 3, 3).double()
        covariance = covariance @ covariance.mT
        jacobian = torch.randn(2, 3, 3).double()
        injection = torch.randn(2, 3, 4).double()
        h = torch.tensor([0.1, 0.2]).double()
        result = propagate_linearized_covariance(covariance, jacobian, injection, h)
        expected = jacobian @ covariance @ jacobian.mT + h[:, None, None] * (
            injection @ injection.mT
        )
        torch.testing.assert_close(result, expected)


class TestEndpointMoments(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2)
        # Zero gains leave the free particle unforced.
        self.resting = torch.tensor([[0.0, 0.0, 0.0]]).double()

    def test_free_motion_closed_form(self):
        sigma = 1.3
        plant = _free_motion_oracle(sigma=sigma).double()
        controller = _controller()
        for steps, h in ((10, 0.001), (5, 0.002), (18, 0.001), (1, 0.001)):
            duration = steps * h
            moments = integrate_endpoint_moments(
                plant,
                controller,
                torch.tensor([[0.2, 0.5]]).double(),
                self.resting,
                torch.tensor([[duration]]).double(),
                maximum_step=h,
                physics_dt=h,
                controller_refresh_steps=1,
                covariance_mode="propagated",
                create_graph=False,
            )
            covariance = moments["covariances"][0]
            variance = sigma**2
            self.assertAlmostEqual(
                covariance[0, 0].item(),
                variance * (duration**3 / 3 + duration**2 * h / 2 + duration * h**2 / 6),
                places=12,
            )
            self.assertAlmostEqual(
                covariance[0, 1].item(),
                variance * (duration**2 / 2 + duration * h / 2),
                places=12,
            )
            self.assertAlmostEqual(covariance[1, 1].item(), variance * duration, places=12)
            torch.testing.assert_close(
                moments["means"], torch.tensor([[0.2 + 0.5 * duration, 0.5]]).double()
            )

    def test_mixed_durations_match_single_rows(self):
        plant = PortHamiltonianPlant(PENDULUM_SPEC, NETWORK).double()
        controller = _controller()
        states = torch.randn(4, 2).double()
        decisions = torch.tensor([0.1, 10.0, 1.0]).double().expand(4, 3)
        durations = torch.tensor([[0.002], [0.018], [0.010], [0.002]]).double()
        kwargs = dict(
            maximum_step=0.001,
            physics_dt=0.001,
            controller_refresh_steps=1,
            covariance_mode="propagated",
            create_graph=False,
        )
        together = integrate_endpoint_moments(
            plant, controller, states, decisions, durations, **kwargs
        )
        for i in range(4):
            alone = integrate_endpoint_moments(
                plant, controller, states[i:i + 1], decisions[i:i + 1], durations[i:i + 1], **kwargs
            )
            torch.testing.assert_close(alone["means"][0], together["means"][i])
            torch.testing.assert_close(alone["covariances"][0], together["covariances"][i])

    def test_matches_monte_carlo_with_held_torque(self):
        """Compare the augmented recursion with the nonlinear step on a pendulum."""
        sigma = 0.05
        plant = _pendulum_oracle(sigma=sigma).double()
        for refresh in (1, 5):
            controller = _controller(refresh_steps=refresh)
            state = torch.tensor([[0.3, 0.5]]).double()
            decision = torch.tensor([[0.0, 15.0, 3.0]]).double()
            duration = torch.tensor([[0.05]]).double()
            kwargs = dict(
                maximum_step=0.001,
                physics_dt=0.001,
                controller_refresh_steps=refresh,
            )
            moments = integrate_endpoint_moments(
                plant,
                controller,
                state,
                decision,
                duration,
                covariance_mode="propagated",
                create_graph=False,
                **kwargs,
            )
            torch.manual_seed(10 + refresh)
            paths = stochastic_rollout(
                plant,
                controller,
                state,
                decision,
                duration,
                lambda s: torch.zeros(len(s), dtype=torch.bool),
                path_count=200000,
                **kwargs,
            )
            endpoints = paths["endpoints"][0]
            empirical = torch.cov(endpoints.T)
            propagated = moments["covariances"][0]
            torch.testing.assert_close(
                empirical,
                propagated,
                atol=0.0,
                rtol=0.02,
                msg=lambda m: f"refresh {refresh}: {m}",
            )
            self.assertLess(
                (endpoints.mean(0) - moments["means"][0]).abs().max().item(),
                1e-3,
            )

    def test_held_torque_changes_covariance(self):
        plant = _pendulum_oracle(sigma=1.0).double()
        state = torch.tensor([[0.3, 0.5]]).double()
        decision = torch.tensor([[0.0, 30.0, 5.0]]).double()
        duration = torch.tensor([[0.05]]).double()
        results = [
            integrate_endpoint_moments(
                plant,
                _controller(refresh_steps=refresh),
                state,
                decision,
                duration,
                maximum_step=0.001,
                physics_dt=0.001,
                controller_refresh_steps=refresh,
                covariance_mode="propagated",
                create_graph=False,
            )["covariances"][0]
            for refresh in (1, 25)
        ]
        self.assertGreater((results[0] - results[1]).abs().max().item(), 1e-4)

    def test_lemma_endpoint_bound(self):
        sigma = 3.0
        plant = _pendulum_oracle(sigma=sigma).double()
        gravity = 9.81
        for steps, h in ((10, 0.001), (18, 0.001), (5, 0.002)):
            duration = steps * h
            controller = _controller()
            moments = integrate_endpoint_moments(
                plant,
                controller,
                torch.tensor([[0.4, -0.3]]).double(),
                self.resting,
                torch.tensor([[duration]]).double(),
                maximum_step=h,
                physics_dt=h,
                controller_refresh_steps=1,
                covariance_mode="propagated",
                create_graph=False,
            )
            leading = duration * plant.covariance(torch.tensor([[0.4, -0.3]]).double())
            gap = torch.linalg.matrix_norm(
                moments["covariances"] - leading, ord=2
            ).item()
            # Constant A, so L_A=0; K bounds the spectral norm of Db = [[0,1],[g cos,0]].
            lipschitz = max(1.0, gravity)
            c_h = 1.0 + h
            bound = duration * sigma**2 * (c_h**2 * math.exp(2 * c_h * lipschitz * duration) - 1.0)
            self.assertLessEqual(gap, bound)
            self.assertGreater(gap, 0.0)

    def test_leading_order_mode(self):
        plant = PortHamiltonianPlant(PENDULUM_SPEC, NETWORK).double()
        states = torch.randn(3, 2).double()
        decisions = torch.tensor([0.0, 5.0, 1.0]).double().expand(3, 3)
        durations = torch.tensor([[0.002], [0.01], [0.018]]).double()
        moments = integrate_endpoint_moments(
            plant,
            _controller(),
            states,
            decisions,
            durations,
            maximum_step=0.001,
            physics_dt=0.001,
            controller_refresh_steps=1,
            covariance_mode="leading_order",
            create_graph=False,
        )
        torch.testing.assert_close(
            moments["covariances"], durations[:, :, None] * plant.covariance(states)
        )
        propagated = integrate_endpoint_moments(
            plant,
            _controller(),
            states,
            decisions,
            durations,
            maximum_step=0.001,
            physics_dt=0.001,
            controller_refresh_steps=1,
            covariance_mode="propagated",
            create_graph=False,
        )
        torch.testing.assert_close(moments["means"], propagated["means"])

    def test_saturation_fraction_and_jacobian_norm(self):
        plant = _pendulum_oracle().double()
        controller = _controller()
        states = torch.tensor([[0.4, 0.0], [0.0, 0.0]]).double()
        decisions = torch.tensor([[-0.5, 100.0, 0.0], [0.0, 1.0, 0.0]]).double()
        durations = torch.tensor([[0.004], [0.004]]).double()
        kwargs = dict(maximum_step=0.001, physics_dt=0.001, controller_refresh_steps=1)
        moments = integrate_endpoint_moments(
            plant,
            controller,
            states,
            decisions,
            durations,
            covariance_mode="propagated",
            create_graph=False,
            **kwargs,
        )
        self.assertEqual(moments["saturation_fractions"].shape, (2, 1))
        self.assertAlmostEqual(moments["saturation_fractions"][0].item(), 1.0)
        self.assertAlmostEqual(moments["saturation_fractions"][1].item(), 0.0)
        norms = path_jacobian_norms(plant, controller, states, decisions, durations, **kwargs)
        self.assertEqual(norms.shape, (2, 1))
        self.assertGreater(norms.min().item(), 0.0)

    def test_nll_gradient_reaches_every_parameter_group(self):
        plant = PortHamiltonianPlant(ACROBOT_SPEC, dict(NETWORK, damping_rank=2))
        controller = _controller(
            spec=ACROBOT_SPEC,
            config=dict(
                PENDULUM_CONTROLLER, setpoint_mode="offset", setpoint_error="unwrapped"
            ),
        )
        states = 0.5 * torch.randn(8, 4)
        decisions = torch.tensor([0.1, 5.0, 1.0]).expand(8, 3)
        durations = torch.full((8, 1), 0.005)
        for mode in ("leading_order", "propagated"):
            plant.zero_grad()
            moments = integrate_endpoint_moments(
                plant,
                controller,
                states,
                decisions,
                durations,
                maximum_step=0.001,
                physics_dt=0.001,
                controller_refresh_steps=1,
                covariance_mode=mode,
                create_graph=True,
            )
            batch = {"next_states": states + 0.01 * torch.randn(8, 4)}
            loss = plant_likelihood_loss(batch, moments, ACROBOT_SPEC)
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            for name, parameter in plant.named_parameters():
                # The drift reads only grad U, so the constant of U has no effect.
                if name == "potential_net.layers.2.bias":
                    self.assertIsNone(parameter.grad)
                    continue
                self.assertIsNotNone(parameter.grad, f"{mode}: {name}")
                self.assertGreater(
                    parameter.grad.abs().sum().item(), 0.0, f"{mode}: {name}"
                )

    def test_create_graph_false_is_detached(self):
        plant = PortHamiltonianPlant(PENDULUM_SPEC, NETWORK)
        moments = integrate_endpoint_moments(
            plant,
            _controller(),
            torch.zeros(2, 2),
            torch.tensor([0.0, 5.0, 1.0]).expand(2, 3),
            torch.full((2, 1), 0.003),
            maximum_step=0.001,
            physics_dt=0.001,
            controller_refresh_steps=1,
            covariance_mode="propagated",
            create_graph=False,
        )
        self.assertFalse(moments["means"].requires_grad)
        self.assertFalse(moments["covariances"].requires_grad)

    def test_nll_formula(self):
        spec = PENDULUM_SPEC
        covariance = torch.tensor([[[2.0, 0.5], [0.5, 1.0]]]).double()
        residual = torch.tensor([[0.3, -0.4]]).double()
        moments = {"means": torch.zeros(1, 2).double(), "covariances": covariance}
        loss = plant_likelihood_loss({"next_states": residual}, moments, spec)
        whitened = residual @ torch.linalg.inv(covariance)[0] @ residual.T
        expected = 0.5 * whitened.item() + 0.5 * math.log(2.0 - 0.25)
        self.assertAlmostEqual(loss.item(), expected, places=10)

    def test_nll_survives_short_float32_durations(self):
        """The position and velocity blocks differ by many orders at one physics step."""
        plant = PortHamiltonianPlant(PENDULUM_SPEC, NETWORK)
        states = torch.zeros(4, 2)
        moments = integrate_endpoint_moments(
            plant,
            _controller(),
            states,
            torch.tensor([0.0, 5.0, 1.0]).expand(4, 3),
            torch.full((4, 1), 0.001),
            maximum_step=0.001,
            physics_dt=0.001,
            controller_refresh_steps=1,
            covariance_mode="propagated",
            create_graph=True,
        )
        loss = plant_likelihood_loss({"next_states": states}, moments, PENDULUM_SPEC)
        self.assertTrue(torch.isfinite(loss))


class TestRollouts(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(3)
        self.plant = _pendulum_oracle(sigma=3.0).double()
        self.controller = _controller()
        self.kwargs = dict(maximum_step=0.001, physics_dt=0.001, controller_refresh_steps=1)

    def test_shapes_and_shared_noise(self):
        states = torch.tensor([[0.1, 0.0], [0.2, 0.1]]).double()
        decisions = torch.tensor([[0.0, 10.0, 1.0], [0.1, 20.0, 2.0]]).double()
        durations = torch.tensor([[0.002], [0.018]], dtype=torch.float64)
        increments = sample_brownian_increments(
            durations, path_count=4, noise_dim=1, seed=5, **self.kwargs
        )
        first = stochastic_rollout(
            self.plant,
            self.controller,
            states,
            decisions,
            durations,
            lambda s: torch.zeros(len(s), dtype=torch.bool),
            brownian_increments=increments,
            **self.kwargs,
        )
        second = stochastic_rollout(
            self.plant,
            self.controller,
            states,
            decisions,
            durations,
            lambda s: torch.zeros(len(s), dtype=torch.bool),
            brownian_increments=increments,
            **self.kwargs,
        )
        self.assertEqual(first["endpoints"].shape, (2, 4, 2))
        self.assertEqual(first["survived"].shape, (2, 4, 1))
        self.assertEqual(first["failure_times"].shape, (2, 4, 1))
        self.assertEqual(first["brownian_increments"].shape, increments.shape)
        self.assertTrue(torch.equal(first["endpoints"], second["endpoints"]))
        self.assertTrue(first["survived"].all())
        self.assertTrue(torch.isinf(first["failure_times"]).all())
        torch.testing.assert_close(
            first["durations_used"][:, :, 0], durations.expand(2, 4), atol=1e-12, rtol=0
        )

    def test_rollout_matches_hand_loop_with_padding(self):
        """A short row in a padded batch evolves as it does alone, ending at its count.

        Under either integrator, with the torque refreshed at every substep's start.
        """
        states = torch.tensor([[0.1, 0.2], [0.2, 0.1]]).double()
        decisions = torch.tensor([[0.0, 10.0, 1.0], [0.1, 20.0, 2.0]]).double()
        durations = torch.tensor([[0.003], [0.010]]).double()
        increments = sample_brownian_increments(
            durations, path_count=2, noise_dim=1, seed=7, **self.kwargs
        )
        for integrator in ("rk4", "semi_implicit"):
            result = stochastic_rollout(
                self.plant,
                self.controller,
                states,
                decisions,
                durations,
                lambda s: torch.zeros(len(s), dtype=torch.bool),
                brownian_increments=increments,
                integrator=integrator,
                **self.kwargs,
            )
            for row, steps in ((0, 3), (1, 10)):
                for path in range(2):
                    current = states[row : row + 1]
                    for j in range(steps):
                        torque = self.controller.action(current, decisions[row : row + 1])
                        current = model_substep(
                            self.plant,
                            current,
                            torque,
                            torch.tensor([0.001]).double(),
                            integrator=integrator,
                            brownian_increments=increments[row, path, j][None],
                        )
                    torch.testing.assert_close(
                        result["endpoints"][row, path],
                        current[0],
                        atol=1e-10,
                        rtol=1e-8,
                        msg=lambda m: f"{integrator}: {m}",
                    )

    def test_paths_stop_at_first_failure(self):
        """A path that crosses keeps its crossing state and records the crossing time."""
        limit = 0.1

        def failure(states):
            return states[:, 0].abs() >= limit

        state = torch.tensor([[0.0, 30.0]]).double()
        result = stochastic_rollout(
            _free_motion_oracle(sigma=0.0).double(),
            self.controller,
            state,
            torch.tensor([[0.0, 0.0, 0.0]]).double(),
            torch.tensor([[0.018]], dtype=torch.float64),
            failure,
            path_count=2,
            **self.kwargs,
        )
        self.assertFalse(result["survived"].any())
        # 30 rad/s reaches 0.1 rad at the fourth 1 ms step.
        torch.testing.assert_close(
            result["failure_times"][0, :, 0],
            torch.full((2,), 0.004, dtype=torch.float64),
            atol=1e-12,
            rtol=0,
        )
        torch.testing.assert_close(
            result["durations_used"], result["failure_times"], atol=1e-12, rtol=0
        )
        torch.testing.assert_close(
            result["endpoints"][0, :, 0],
            torch.full((2,), 0.12, dtype=torch.float64),
            atol=1e-12,
            rtol=0,
        )

    def _constructed(self, survived):
        endpoints = torch.tensor(
            [[[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]], [[4.0, 0.0], [5.0, 0.0], [6.0, 0.0]]]
        ).double()
        return {"endpoints": endpoints, "survived": torch.tensor(survived).unsqueeze(-1)}

    def test_survivor_mean_and_fallback(self):
        value = lambda s: s[..., :1]
        rollouts = self._constructed([[True, False, True], [False, False, False]])
        observed = torch.tensor([[9.0, 0.0], [7.0, 0.0]]).double()
        durations = torch.tensor([[0.5], [0.25]]).double()
        result = survivor_continuation(value, rollouts, observed, durations, 0.1)
        expected = torch.tensor([[math.exp(-0.05) * 2.0], [math.exp(-0.025) * 7.0]]).double()
        torch.testing.assert_close(result, expected)

    def test_failed_paths_have_zero_continuation_and_no_renormalization(self):
        value = lambda s: s[..., :1]
        rollouts = self._constructed([[True, False, True], [False, False, False]])
        horizons = torch.tensor([[0.5], [0.25]]).double()
        result = path_continuations(value, rollouts, horizons, 0.1)
        expected = math.exp(-0.05) * torch.tensor([[1.0, 0.0, 3.0]]).double()
        torch.testing.assert_close(result[0].squeeze(-1), expected[0])
        self.assertEqual(result[1].abs().max().item(), 0.0)


class TestLosses(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(4)
        self.low = torch.tensor([-0.5, 0.0, 0.0])
        self.high = torch.tensor([0.5, 100.0, 20.0])

    def test_regressions(self):
        states, decisions = torch.randn(8, 2), torch.rand(8, 3)
        targets = torch.randn(8, 1)
        critics = TwinQCritic(PENDULUM_SPEC, 3, NETWORK)
        first, second = critics(states, decisions)
        expected = (first - targets).square().mean() + (second - targets).square().mean()
        torch.testing.assert_close(critic_loss(critics, states, decisions, targets), expected)
        value = SmoothValueHead(PENDULUM_SPEC, NETWORK)
        torch.testing.assert_close(
            soft_value_loss(value, states, targets), (value(states) - targets).square().mean()
        )

    def test_rate_loss_weights_only_data_term(self):
        rate = RateHead(PENDULUM_SPEC, 3, NETWORK)
        states, decisions = torch.randn(8, 2), torch.rand(8, 3)
        model_states, model_decisions = torch.randn(5, 2), torch.rand(5, 3)
        batch = {"states": states, "decisions": decisions}
        targets = {
            "data": torch.randn(8, 1),
            "rollout": torch.randn(8, 1),
            "generator": torch.randn(5, 1),
            "data_weights": torch.rand(8, 1),
        }
        loss = rate_regression_loss(
            rate, batch, targets, model_states, model_decisions, rollout_weight=0.5, model_weight=2.0
        )
        replay = rate(states, decisions)
        expected = (
            (targets["data_weights"] * (replay - targets["data"]).square()).mean()
            + 0.5 * (replay - targets["rollout"]).square().mean()
            + 2.0 * (rate(model_states, model_decisions) - targets["generator"]).square().mean()
        )
        torch.testing.assert_close(loss, expected)
        no_rollout = rate_regression_loss(
            rate, batch, targets, model_states, model_decisions, rollout_weight=0.0, model_weight=0.0
        )
        torch.testing.assert_close(
            no_rollout, (targets["data_weights"] * (replay - targets["data"]).square()).mean()
        )

    def test_implicit_actor_gradient_reaches_only_actor(self):
        actor = DecisionActor(PENDULUM_SPEC, self.low, self.high, NETWORK)
        rate = RateHead(PENDULUM_SPEC, 3, NETWORK, decision_low=self.low, decision_high=self.high)
        states = torch.randn(8, 2)
        sample = actor.sample(states, sample_count=2)
        loss = implicit_actor_loss(rate, states, sample, torch.tensor(0.2))
        loss.backward()
        self.assertGreater(actor.mean_network[0].weight.grad.abs().sum().item(), 0.0)
        self.assertGreater(actor.log_std.grad.abs().sum().item(), 0.0)
        for parameter in rate.parameters():
            self.assertIsNone(parameter.grad)
            self.assertTrue(parameter.requires_grad)

    def test_explicit_actor_gradient_reaches_only_actor(self):
        actor = DecisionActor(PENDULUM_SPEC, self.low, self.high, NETWORK)
        critics = TwinQCritic(PENDULUM_SPEC, 3, NETWORK)
        states = torch.randn(8, 2)
        sample = actor.sample(states)
        loss = explicit_actor_loss(critics, states, sample, torch.full((8, 1), 0.01))
        loss.backward()
        self.assertGreater(actor.mean_network[0].weight.grad.abs().sum().item(), 0.0)
        for parameter in critics.parameters():
            self.assertIsNone(parameter.grad)
            self.assertTrue(parameter.requires_grad)

    def test_temperature_loss(self):
        log_alpha = torch.tensor(0.5, requires_grad=True)
        logs = torch.tensor([[-1.0], [2.0]], requires_grad=True)
        weights = torch.tensor([[0.5], [1.5]])
        loss = temperature_loss(log_alpha, logs, 1.5, interval_weights=weights)
        expected = -(0.5 * ((0.5 * 0.5) + (1.5 * 3.5))) / 2.0
        self.assertAlmostEqual(loss.item(), expected, places=6)
        loss.backward()
        self.assertIsNone(logs.grad)
        self.assertAlmostEqual(log_alpha.grad.item(), expected / 0.5, places=6)
        sampled = temperature_loss(
            log_alpha, torch.zeros(2, 3, 1), 1.0, interval_weights=torch.ones(2, 1)
        )
        self.assertAlmostEqual(sampled.item(), -0.5, places=6)


class TestPresets(unittest.TestCase):
    def test_model_works_with_paper_presets(self):
        windy = windy_pendulum_config(
            reward_parameters={},
            reset_parameters={},
            tail_probability=0.8,
            tail_split=0.5,
            clock_kind="two_tail_uniform",
        )
        acrobot = acrobot_xk_config(
            failure_limits={"elbow_angle": 4.0 * math.pi},
            reward_parameters={},
            reset_parameters={},
            plant_parameters={},
        )
        for testbed, rank in ((windy, 1), (acrobot, 2)):
            spec = testbed["state_spec"]
            controller = FeedbackController(testbed["controller"], spec)
            plant = PortHamiltonianPlant(spec, dict(NETWORK, damping_rank=rank))
            size = spec["position_dim"] + spec["velocity_dim"]
            low = torch.as_tensor(testbed["controller"]["decision_low"], dtype=torch.float32)
            high = torch.as_tensor(testbed["controller"]["decision_high"], dtype=torch.float32)
            actor = DecisionActor(spec, low, high, NETWORK)
            states = torch.randn(6, size) * 0.2
            decisions = actor.sample(states)["decisions"][:, 0]
            moments = integrate_endpoint_moments(
                plant,
                controller,
                states,
                decisions,
                torch.full((6, 1), testbed["reference_period"]),
                maximum_step=testbed["physics_dt"],
                physics_dt=testbed["physics_dt"],
                controller_refresh_steps=testbed["controller"]["refresh_steps"],
                covariance_mode="propagated",
                create_graph=True,
            )
            self.assertEqual(moments["means"].shape, (6, size))
            self.assertTrue(torch.isfinite(moments["covariances"]).all())


ACROBOT_CONTROLLER = dict(
    PENDULUM_CONTROLLER, setpoint_mode="offset", setpoint_error="unwrapped"
)
NO_FAILURE = staticmethod(lambda states: torch.zeros(len(states), dtype=torch.bool))


def _drift_only_plant(spec=ACROBOT_SPEC, rank=2):
    return PortHamiltonianPlant(
        spec, dict(NETWORK, damping_rank=rank), learn_diffusion=False
    ).double()


def _same_drift_with_learned_noise(plant):
    """A learned-diffusion plant sharing every drift weight of a drift-only plant."""
    learned = PortHamiltonianPlant(
        plant.state_spec, dict(NETWORK, damping_rank=plant.damping_rank)
    ).double()
    missing, unexpected = learned.load_state_dict(plant.state_dict(), strict=False)
    assert not unexpected and all("noise" in name for name in missing), missing
    return learned


class TestDriftOnlyPlant(unittest.TestCase):
    """network["diffusion"] == "none": zero diffusion, the drift loss, no noise terms."""

    def setUp(self):
        torch.manual_seed(0)
        self.moment_grid = dict(maximum_step=0.001, physics_dt=0.001, controller_refresh_steps=1)

    def test_config_key(self):
        self.assertTrue(diffusion_is_learned(NETWORK))
        self.assertTrue(diffusion_is_learned(dict(NETWORK, diffusion="learned")))
        self.assertFalse(diffusion_is_learned(dict(NETWORK, diffusion="none")))
        with self.assertRaisesRegex(ValueError, "diffusion"):
            diffusion_is_learned(dict(NETWORK, diffusion="zero"))

    def test_zero_diffusion_and_covariance_without_a_noise_network(self):
        plant = _drift_only_plant()
        states = torch.randn(64, 4).double()
        diffusion = plant.diffusion(states)
        self.assertEqual(tuple(diffusion.shape), (64, 4, 1))
        self.assertFalse(diffusion.any())
        covariance = plant.covariance(states)
        self.assertEqual(tuple(covariance.shape), (64, 4, 4))
        self.assertFalse(covariance.any())
        self.assertEqual(plant.noise_dim, 1)
        self.assertFalse(plant.learns_diffusion)
        self.assertIsNone(plant.noise_net)
        self.assertFalse(any("noise" in name for name in plant.state_dict()))
        # The known Acrobot plant has the same noiseless layout.
        _, oracle_diffusion = oracle_callbacks(acrobot_config())
        self.assertEqual(oracle_diffusion(states).shape, diffusion.shape)
        mapped = func.vmap(lambda s: plant.diffusion(s.unsqueeze(0)).squeeze(0))(states)
        self.assertEqual(tuple(mapped.shape), (64, 4, 1))

        frozen = plant.frozen_copy()
        self.assertEqual(frozen.noise_dim, 1)
        self.assertEqual(set(frozen.state_dict()), set(plant.state_dict()))
        self.assertFalse(frozen.diffusion(states).any())

        # Only the noise model differs from a learned-diffusion plant.
        learned = _same_drift_with_learned_noise(plant)
        actions = torch.randn(64, 1).double()
        torch.testing.assert_close(learned.drift(states, actions), plant.drift(states, actions))
        self.assertTrue(learned.diffusion(states).any())

        plant.initialize_from_physics(torch.tensor([[0.0], [2.0]]), torch.tensor(3.0))
        self.assertEqual(plant.input_matrix().flatten().tolist(), [0.0, 2.0])
        self.assertFalse(plant.diffusion(states).any())

    def test_second_order_and_noise_terms_vanish(self):
        plant = _drift_only_plant(PENDULUM_SPEC, rank=1)
        value = SmoothValueHead(PENDULUM_SPEC, NETWORK).double()
        controller = _controller()
        states = torch.randn(16, 2).double()
        decisions = torch.tensor([0.1, 5.0, 1.0]).double().expand(16, 3)
        resolved = controller.resolve_decision(states, decisions)
        derivatives = value_derivatives(
            value, states, plant.diffusion(states), create_graph=False
        )
        self.assertFalse(derivatives["diffusion_traces"].any())
        generated = generator(value, plant, controller, states, resolved, create_graph=False)
        drift = plant.drift(states, controller.action(states, resolved))
        torch.testing.assert_close(
            generated, (drift * derivatives["gradients"]).sum(-1, keepdim=True)
        )
        durations = torch.full((16, 1), 0.01).double()
        weights = data_rate_weights(
            plant, value, states, durations, weight_max=5.0, noise_variance_floor=0.5
        )
        torch.testing.assert_close(weights, torch.full((16, 1), 2.0).double())

        # With the learned noise model both terms are present.
        learned = _same_drift_with_learned_noise(plant)
        traces = value_derivatives(
            value, states, learned.diffusion(states), create_graph=False
        )["diffusion_traces"]
        self.assertGreater(traces.abs().max().item(), 0.0)
        weights = data_rate_weights(
            learned, value, states, durations, weight_max=5.0, noise_variance_floor=0.5
        )
        self.assertLess(weights.max().item(), 2.0)

    def test_moments_are_noise_free_in_both_covariance_modes(self):
        plant = _drift_only_plant()
        controller = _controller(spec=ACROBOT_SPEC, config=ACROBOT_CONTROLLER)
        states = 0.5 * torch.randn(6, 4).double()
        decisions = torch.tensor([0.1, 5.0, 1.0]).double().expand(6, 3)
        durations = torch.tensor([[0.002], [0.01], [0.018], [0.005], [0.01], [0.001]]).double()
        results = {}
        for mode in ("leading_order", "propagated"):
            results[mode] = integrate_endpoint_moments(
                plant, controller, states, decisions, durations,
                covariance_mode=mode, create_graph=False, **self.moment_grid,
            )
            self.assertTrue(results[mode]["drift_only"])
            self.assertEqual(tuple(results[mode]["covariances"].shape), (6, 4, 4))
            self.assertFalse(results[mode]["covariances"].any())
        torch.testing.assert_close(
            results["leading_order"]["means"], results["propagated"]["means"]
        )
        # Skipping the propagation leaves the means the propagation would give.
        learned = integrate_endpoint_moments(
            _same_drift_with_learned_noise(plant), controller, states, decisions, durations,
            covariance_mode="propagated", create_graph=False, **self.moment_grid,
        )
        self.assertFalse(learned["drift_only"])
        torch.testing.assert_close(results["propagated"]["means"], learned["means"])

    def test_drift_loss_formula_and_dispatch(self):
        plant = _drift_only_plant()
        controller = _controller(spec=ACROBOT_SPEC, config=ACROBOT_CONTROLLER)
        states = 0.5 * torch.randn(8, 4).double()
        decisions = torch.tensor([0.1, 5.0, 1.0]).double().expand(8, 3)
        durations = torch.tensor([[0.002], [0.01], [0.018], [0.005]] * 2).double()
        batch = {
            "next_states": states + 0.01 * torch.randn(8, 4).double(),
            "durations": durations,
        }
        for mode in ("leading_order", "propagated"):
            moments = integrate_endpoint_moments(
                plant, controller, states, decisions, durations,
                covariance_mode=mode, create_graph=True, **self.moment_grid,
            )
            residual = wrapped_endpoint_residual(
                batch["next_states"], moments["means"], ACROBOT_SPEC
            )
            expected = (0.5 * residual.square().sum(-1) / durations.squeeze(-1)).mean()
            torch.testing.assert_close(plant_drift_loss(batch, moments, ACROBOT_SPEC), expected)
            torch.testing.assert_close(
                plant_likelihood_loss(batch, moments, ACROBOT_SPEC), expected
            )
        # Moments without the flag, as from a learned noise model, keep the Gaussian NLL.
        gaussian = {"means": moments["means"].detach(), "covariances": 0.1 * torch.eye(4).double().expand(8, 4, 4)}
        self.assertNotAlmostEqual(
            plant_likelihood_loss(batch, gaussian, ACROBOT_SPEC).item(),
            plant_drift_loss(batch, gaussian, ACROBOT_SPEC).item(),
            places=3,
        )

    def test_gradients_reach_only_the_drift_networks(self):
        plant = _drift_only_plant()
        controller = _controller(spec=ACROBOT_SPEC, config=ACROBOT_CONTROLLER)
        self.assertEqual(
            {name.split(".")[0] for name, _ in plant.named_parameters()},
            {"potential_net", "mass_net", "damping_net", "force_matrix"},
        )
        states = 0.5 * torch.randn(8, 4).double()
        decisions = torch.tensor([0.1, 5.0, 1.0]).double().expand(8, 3)
        durations = torch.full((8, 1), 0.005).double()
        batch = {"next_states": states + 0.01 * torch.randn(8, 4).double(), "durations": durations}
        for mode in ("leading_order", "propagated"):
            plant.zero_grad()
            moments = integrate_endpoint_moments(
                plant, controller, states, decisions, durations,
                covariance_mode=mode, create_graph=True, **self.moment_grid,
            )
            loss = plant_likelihood_loss(batch, moments, ACROBOT_SPEC)
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            for name, parameter in plant.named_parameters():
                if name == "potential_net.layers.2.bias":  # U's constant has no effect
                    self.assertIsNone(parameter.grad)
                    continue
                self.assertIsNotNone(parameter.grad, f"{mode}: {name}")
                self.assertGreater(parameter.grad.abs().sum().item(), 0.0, f"{mode}: {name}")

    def test_rollout_kicks_vanish(self):
        plant = _drift_only_plant(PENDULUM_SPEC, rank=1)
        controller = _controller()
        states = 0.3 * torch.randn(3, 2).double()
        decisions = torch.tensor([0.1, 5.0, 1.0]).double().expand(3, 3)
        durations = torch.tensor([[0.004], [0.01], [0.002]]).double()
        noise = sample_brownian_increments(
            durations, path_count=3, noise_dim=plant.noise_dim, seed=4, **self.moment_grid
        ).double()
        self.assertEqual(noise.shape[-1], 1)
        noisy, quiet = (
            stochastic_rollout(
                plant, controller, states, decisions, durations, NO_FAILURE,
                path_count=3, brownian_increments=increments, **self.moment_grid,
            )
            for increments in (noise, torch.zeros_like(noise))
        )
        self.assertGreater(noise.abs().max().item(), 0.0)
        torch.testing.assert_close(noisy["endpoints"], quiet["endpoints"])
        torch.testing.assert_close(
            noisy["endpoints"], noisy["endpoints"][:, :1].expand(-1, 3, -1)
        )


def _explicit_agent(testbed, *, diffusion, continuation="generator", seed=11):
    """A small explicit agent with a learned plant, as the protocol builds one."""
    config = {
        "testbed": testbed,
        "network": {
            "hidden_sizes": (16, 16),
            "plant_hidden_sizes": (8, 8),
            "activation": "tanh",
            "mass_floor": 0.01,
            "diffusion_floor": 0.001,
            "damping_rank": testbed["state_spec"]["velocity_dim"],
            "velocity_feature_scale": 2.0 * math.pi,
            "diffusion": diffusion,
        },
        "training": {
            "seed": 3,
            "device": "cpu",
            "replay_capacity": 2000,
            "batch_size": 8,
            "learning_rate": 3e-4,
            "gradient_clip": 10.0,
            "warmup_episodes": 0,
            "validation_fraction": 0.0,
        },
        "temperature": {
            "initial_alpha_c": 0.01,
            "learn_temperature": False,
            "target_entropy": -3.0,
        },
        "plant_source": "learned",
        "model_step": testbed["physics_dt"],
        "continuation": continuation,
        "polyak_rate": 0.25,
        "rollout_paths": 3,
    }
    return ExplicitCTPHSAC(make_env(testbed, seed=seed), config)


def _collected_batch(agent, rows=40, size=8):
    for _ in range(rows):
        agent.collect_transition()
    return agent.replay.sample("train", size)


class TestExplicitDriftOnlyPlant(unittest.TestCase):
    """Both explicit arms with the drift-only learned plant (cart-pole, no noise)."""

    def setUp(self):
        torch.manual_seed(0)

    def test_online_and_target_plants_follow_the_config(self):
        for continuation in ("generator", "rollout"):
            agent = _explicit_agent(cartpole_config(), diffusion="none", continuation=continuation)
            states = 0.3 * torch.randn(8, 4)
            for plant in (agent.online_plant, agent.target_plant):
                self.assertFalse(plant.learns_diffusion)
                self.assertEqual(plant.noise_dim, 1)
                self.assertEqual(tuple(plant.diffusion(states).shape), (8, 4, 1))
                self.assertFalse(plant.diffusion(states).any())
                self.assertFalse(plant.covariance(states).any())
        agent = _explicit_agent(cartpole_config(), diffusion="learned")
        self.assertTrue(agent.online_plant.learns_diffusion)
        self.assertEqual(agent.online_plant.noise_dim, 8)
        self.assertTrue(agent.online_plant.diffusion(torch.zeros(2, 4)).any())

    def test_plant_step_fits_the_drift_loss(self):
        agent = _explicit_agent(cartpole_config(), diffusion="none")
        batch = _collected_batch(agent)
        before = copy.deepcopy(agent.online_plant)
        moments = integrate_endpoint_moments(
            before, agent.controller, batch["states"], batch["decisions"], batch["durations"],
            maximum_step=agent.model_step, physics_dt=agent.physics_dt,
            controller_refresh_steps=agent.refresh_steps,
            covariance_mode="leading_order", create_graph=False,
        )
        expected = plant_drift_loss(batch, moments, agent.state_spec).item()
        metrics = agent.update_plant(batch)
        self.assertAlmostEqual(metrics["plant_loss"], expected, places=5)
        moved = [
            not torch.equal(p, q)
            for p, q in zip(before.parameters(), agent.online_plant.parameters())
        ]
        self.assertTrue(any(moved))

    def test_targets_equal_the_known_plant_with_the_learned_drift(self):
        for diffusion in ("none", "learned"):
            for continuation in ("generator", "rollout"):
                agent = _explicit_agent(
                    cartpole_config(), diffusion=diffusion, continuation=continuation
                )
                batch = _collected_batch(agent)
                learned = agent.target_plant
                oracle = make_oracle_plant(agent.env)
                known = OraclePlant(
                    learned.drift, oracle.diffusion, oracle.input_matrix(),
                    state_spec=agent.state_spec,
                )
                targets = agent.build_critic_target(batch)
                again = agent.build_critic_target(batch)  # fresh rollout noise
                traces = value_derivatives(
                    agent.target_value, batch["states"], learned.diffusion(batch["states"]),
                    create_graph=False,
                )["diffusion_traces"]
                agent.target_plant = known
                expected = agent.build_critic_target(batch)
                if diffusion == "none":
                    torch.testing.assert_close(targets, expected)
                    torch.testing.assert_close(again, targets)
                    self.assertFalse(traces.any())
                elif continuation == "generator":
                    # The learned noise adds the second-order term to the generator.
                    self.assertGreater(traces.abs().max().item(), 0.0)
                else:
                    # ... and kicks to the rollout paths, so fresh noise moves the target.
                    self.assertGreater((again - targets).abs().max().item(), 0.0)


class TestExplicitBoundaryLift(unittest.TestCase):
    """The explicit value head and its target carry the implicit agent's lift."""

    def setUp(self):
        torch.manual_seed(0)

    def _check(self, agent, boundary, interior, payoff):
        for module in (agent.value, agent.target_value):
            with torch.no_grad():
                on_limit = module(boundary)
                inside = module(interior)
            torch.testing.assert_close(on_limit, torch.full_like(on_limit, payoff))
            self.assertGreater((inside - payoff).abs().max().item(), 1e-6)

    def test_value_equals_the_payoff_on_the_failure_limits(self):
        # The windy pendulum keeps its failure set; the cart-pole has none since
        # 2026-11-10 (dm_control's soft rail limit; test_cartpole_value_has_no_boundary).
        for testbed in (windy_config(),):
            agent = _explicit_agent(testbed, diffusion="learned")
            limits = testbed["failure_limits"]
            boundary = torch.tensor(
                [
                    [limits["angle"], 0.0],
                    [-limits["angle"], 0.3],
                    [0.1, limits["velocity"]],
                    [0.1, -limits["velocity"]],
                ]
            )
            interior = torch.tensor([[0.1, 0.2], [0.0, 0.0]])
            payoff = agent.env.terminal_payoff(boundary[0].numpy())
            self.assertEqual(payoff, -10.0)
            self._check(agent, boundary, interior, payoff)
            lift, vanishing = terminal_boundary_functions(testbed)
            torch.testing.assert_close(agent.value._vanishing(interior), vanishing(interior))
            torch.testing.assert_close(agent.target_value._lift(interior), lift(interior))
            # Value steps and Polyak updates keep the boundary values.
            batch = _collected_batch(agent, rows=20)
            agent.train(2, 8)
            self._check(agent, boundary, interior, payoff)
            self.assertTrue(torch.isfinite(agent.build_critic_target(batch)).all())

    def test_cartpole_value_has_no_boundary(self):
        """Since 2026-11-10 the cart-pole keeps dm_control's soft rail limit and has no
        failure set: H = 0 and B = 1 everywhere, at and past the rail too."""
        testbed = cartpole_config()
        self.assertEqual(testbed["failure_limits"], {})
        agent = _explicit_agent(testbed, diffusion="learned")
        states = torch.tensor(
            [[1.8, 3.0, 0.5, -2.0], [-1.8, 0.1, 0.0, 1.0], [2.0, 0.0, 0.0, 0.0], [0.0, 3.0, 0.5, -2.0]]
        )
        lift, vanishing = terminal_boundary_functions(testbed)
        self.assertTrue(torch.equal(lift(states), torch.zeros(4, 1)))
        self.assertTrue(torch.equal(vanishing(states), torch.ones(4, 1)))
        self.assertEqual(agent.env.terminal_payoff(states[0].numpy()), 0.0)
        self.assertFalse(agent.env.failed(states[0].numpy()))
        torch.testing.assert_close(agent.value._vanishing(states), torch.ones(4, 1))
        with torch.no_grad():
            self.assertTrue(torch.isfinite(agent.value(states)).all())


# Two physical positions and one episode-constant context coordinate (a target), which
# is the last position and has a zero row in P.
CONTEXT_SPEC = {
    "position_dim": 3,
    "velocity_dim": 2,
    "action_dim": 1,
    "periodic_position_indices": (0,),
    "actuated_position_indices": (0,),
    "observation_projection": np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]),
    "context_dim": 1,
}
CONTEXT_CONTROLLER = dict(PENDULUM_CONTROLLER, setpoint_mode="offset")


class TestContextCoordinates(unittest.TestCase):
    """A context coordinate never moves: zero drift, no noise, constant along model paths."""

    def setUp(self):
        torch.manual_seed(7)
        self.states = torch.cat(
            (torch.randn(6, 2), torch.rand(6, 1) + 0.5, torch.randn(6, 2)), dim=-1
        ).double()
        self.decisions = torch.tensor([0.2, 10.0, 1.0]).double().expand(6, 3)
        self.controller = FeedbackController(CONTEXT_CONTROLLER, CONTEXT_SPEC)
        self.kwargs = dict(maximum_step=0.001, physics_dt=0.001, controller_refresh_steps=1)

    def plants(self):
        learned = PortHamiltonianPlant(CONTEXT_SPEC, dict(NETWORK, damping_rank=2)).double()
        drift_only = PortHamiltonianPlant(
            CONTEXT_SPEC, dict(NETWORK, damping_rank=2), learn_diffusion=False
        ).double()
        return {"learned": learned, "drift_only": drift_only}

    def test_drift_and_noise_vanish_on_the_context(self):
        actions = torch.randn(6, 1).double()
        for name, plant in self.plants().items():
            with self.subTest(plant=name):
                drift = plant.drift(self.states, actions)
                self.assertEqual(drift.shape, (6, 5))
                self.assertTrue(torch.equal(drift[:, 2], torch.zeros(6).double()))
                self.assertTrue(torch.all(drift[:, :2] == self.states[:, 3:]))
                _, jacobian, _ = plant.drift_jacobian(self.states, actions)
                self.assertTrue(torch.equal(jacobian[:, 2], torch.zeros(6, 5).double()))
                diffusion = plant.diffusion(self.states)
                self.assertTrue(torch.equal(diffusion[:, 2], torch.zeros_like(diffusion[:, 2])))
                covariance = plant.covariance(self.states)
                self.assertTrue(torch.equal(covariance[:, 2], torch.zeros(6, 5).double()))
                self.assertTrue(torch.equal(covariance[:, :, 2], torch.zeros(6, 5).double()))
        learned = self.plants()["learned"]
        moving = [0, 1, 3, 4]
        block = learned.covariance(self.states)[:, moving][:, :, moving]
        self.assertTrue(torch.all(torch.linalg.eigvalsh(block) > 0.0))

    def test_stochastic_rollout_keeps_the_context(self):
        durations = torch.full((6, 1), 0.01, dtype=torch.float64)
        for name, plant in self.plants().items():
            with self.subTest(plant=name):
                paths = stochastic_rollout(
                    plant,
                    self.controller,
                    self.states,
                    self.decisions,
                    durations,
                    lambda s: torch.zeros(len(s), dtype=torch.bool),
                    path_count=3,
                    **self.kwargs,
                )
                endpoints = paths["endpoints"]
                self.assertEqual(endpoints.shape, (6, 3, 5))
                self.assertTrue(
                    torch.equal(endpoints[..., 2], self.states[:, 2:3].expand(6, 3))
                )
                # The physical coordinates did move.
                self.assertGreater((endpoints[..., 0] - self.states[:, :1]).abs().max().item(), 0.0)

    def test_likelihood_skips_the_singular_context_rows(self):
        plant = self.plants()["learned"]
        durations = torch.full((6, 1), 0.01, dtype=torch.float64)
        next_states = self.states.clone()
        next_states[:, :2] += 0.01 * torch.randn(6, 2).double()
        batch = {"next_states": next_states, "durations": durations}
        for mode in ("leading_order", "propagated"):
            with self.subTest(mode=mode):
                moments = integrate_endpoint_moments(
                    plant,
                    self.controller,
                    self.states,
                    self.decisions,
                    durations,
                    covariance_mode=mode,
                    create_graph=True,
                    **self.kwargs,
                )
                self.assertTrue(torch.equal(moments["means"][:, 2], self.states[:, 2]))
                self.assertTrue(
                    torch.equal(moments["covariances"][:, 2], torch.zeros(6, 5).double())
                )
                loss = plant_likelihood_loss(batch, moments, CONTEXT_SPEC)
                self.assertTrue(torch.isfinite(loss))
                loss.backward()
                gradients = [p.grad for p in plant.parameters() if p.grad is not None]
                self.assertTrue(gradients)
                self.assertTrue(all(torch.isfinite(g).all() for g in gradients))
                plant.zero_grad()

    def test_networks_read_the_context(self):
        features = StateFeatures(CONTEXT_SPEC, NETWORK)
        # sin, cos of the periodic angle; the second position and the context raw;
        # the two velocities.
        self.assertEqual(features.output_dim, 2 + 2 + 2)
        moved = self.states.clone()
        moved[:, 2] += 1.0
        self.assertFalse(torch.equal(features(moved.float()), features(self.states.float())))


if __name__ == "__main__":
    unittest.main()

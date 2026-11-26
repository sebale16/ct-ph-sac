import math
import unittest

import numpy as np
import torch
from torch import nn

from algorithms import ph_admission as adm
from algorithms import ph_resolvent as res
from algorithms import ph_targets as tgt
from environment.ph_sac.controller import FeedbackController
from models.ph_sac.networks import DecisionActor, SmoothValueHead
from models.ph_sac.plant import OraclePlant, PortHamiltonianPlant
from models.ph_sac.rollouts import sample_brownian_increments

SPEC = {
    "position_dim": 1,
    "velocity_dim": 1,
    "action_dim": 1,
    "periodic_position_indices": (),
    "actuated_position_indices": (0,),
    "observation_projection": np.eye(1),
}
CONTROLLER = {
    "decision_low": np.array([-1.0, 0.0, 0.0]),
    "decision_high": np.array([1.0, 100.0, 20.0]),
    "torque_low": np.array([-1e6]),
    "torque_high": np.array([1e6]),
    "setpoint_mode": "absolute",
    "setpoint_error": "unwrapped",
    "refresh_steps": 1,
}
NETWORK = {
    "hidden_sizes": (16,),
    "plant_hidden_sizes": (8,),
    "activation": "tanh",
    "mass_floor": 0.01,
    "diffusion_floor": 0.001,
    "damping_rank": 1,
}


def controller():
    return FeedbackController(CONTROLLER, SPEC)


def double(values):
    return torch.as_tensor(values, dtype=torch.float64)


class Quadratic(nn.Module):
    """V(s)=s^T Q s+c, a value with closed-form derivatives."""

    def __init__(self, quadratic, constant=0.0):
        super().__init__()
        self.quadratic = nn.Parameter(double(quadratic))
        self.constant = nn.Parameter(double(constant))

    def forward(self, states):
        form = (states @ self.quadratic * states).sum(dim=-1, keepdim=True)
        return form + self.constant


def linear_plant(force=0.0, sigma=0.5):
    """y'=v, v'=force+a, with noise column (0, sigma)."""

    def drift(states, actions):
        return torch.cat((states[:, 1:], force + actions), dim=-1)

    def diffusion(states):
        column = states.new_zeros(states.shape[0], 2, 1)
        column[:, 1, 0] = sigma
        return column

    return OraclePlant(drift, diffusion, torch.eye(1), state_spec=SPEC)


def state_reward(states, torques):
    return -states[:, :1].square()


def operator(target, *, tau=2.0, discount=0.1, alpha=2.0, plant=None):
    return {
        "target_value": target,
        "policy": None,
        "plant": plant if plant is not None else linear_plant(),
        "temperature_per_second": double(alpha),
        "discount_rate": discount,
        "resolvent_step": tau,
        "reward_rate": state_reward,
    }


def decisions_like(gains, batch):
    """Draws [B,N,3] with zero setpoint and the given (k_p, k_d) gains."""
    row = double([[0.0, kp, kd] for kp, kd in gains])
    return row.unsqueeze(0).expand(batch, -1, -1)


def transition_batch(**columns):
    batch = {
        "states": double([[1.0, 2.0]]),
        "decisions": double([[0.0, 2.0, 1.0]]),
        "durations": double([[0.01]]),
        "rewards": double([[0.0]]),
        "next_states": double([[0.0, 0.0]]),
        "terminated": double([[0.0]]),
        "truncated": double([[0.0]]),
    }
    batch.update({key: double(value) for key, value in columns.items()})
    return batch


class TestResolventResidual(unittest.TestCase):
    def test_hand_value_on_a_wrong_value(self):
        # V=5 at s=(1,2), grad V=(2,4); b=(2,-4), generator=-12+sigma^2=-11.75;
        # soft rate=-1-2*(-0.5)=0; R=1.2*5-1-2*(-11.75)=28.5.
        value = Quadratic(np.eye(2))
        target = Quadratic(np.zeros((2, 2)), 1.0)
        samples = {
            "decisions": decisions_like([(2.0, 1.0)], 1),
            "log_probabilities": double([[[-0.5]]]),
        }
        residual = res.resolvent_residuals(
            value,
            operator(target),
            controller(),
            double([[1.0, 2.0]]),
            samples,
            create_graph=False,
        )
        self.assertEqual(residual.shape, (1, 1, 1))
        self.assertAlmostEqual(residual.item(), 28.5, places=10)
        self.assertFalse(residual.requires_grad)

    def test_averaged_residual_vanishes_at_the_closed_form_solution(self):
        # Linear closed loop with draws of different gains: Q solves the Lyapunov system
        # (1+lambda tau)Q-tau(Q F+F^T Q)+tau W=Qbar under the averaged F.
        tau, discount, alpha, sigma = 0.7, 0.3, 0.4, 0.5
        gains = [(2.0, 1.0), (4.0, 0.5), (6.0, 3.0)]
        log_pi = np.array([-0.2, 0.1, -0.9])
        loops = [np.array([[0.0, 1.0], [-kp, -kd]]) for kp, kd in gains]
        average = np.mean(loops, axis=0)
        reward = np.diag([1.0, 0.0])
        target_q = np.array([[0.5, 0.1], [0.1, -0.2]])
        target_c = 0.3

        identity = np.eye(2)
        system = (1.0 + discount * tau) * np.eye(4) - tau * (
            np.kron(average.T, identity) + np.kron(identity, average.T)
        )
        solved = np.linalg.solve(
            system, (target_q - tau * reward).reshape(-1, order="F")
        ).reshape(2, 2, order="F")
        solved = 0.5 * (solved + solved.T)
        diffusion = np.diag([0.0, sigma**2])
        constant = (
            target_c
            + tau * (np.trace(diffusion @ solved) - alpha * log_pi.mean())
        ) / (1.0 + discount * tau)

        states = double(np.random.default_rng(0).normal(size=(32, 2)))
        samples = {
            "decisions": decisions_like(gains, 32),
            "log_probabilities": double(log_pi).reshape(1, 3, 1).expand(32, -1, -1),
        }
        residuals = res.resolvent_residuals(
            Quadratic(solved, constant),
            operator(
                Quadratic(target_q, target_c),
                tau=tau,
                discount=discount,
                alpha=alpha,
                plant=linear_plant(sigma=sigma),
            ),
            controller(),
            states,
            samples,
            create_graph=False,
        )
        self.assertEqual(residuals.shape, (32, 3, 1))
        self.assertLess(residuals.mean(dim=1).abs().max().item(), 1e-10)
        # Individual draws do not vanish: only the policy average is the update equation.
        self.assertGreater(residuals.abs().max().item(), 1e-2)

    def test_gradient_reaches_only_psi(self):
        value = SmoothValueHead(SPEC, NETWORK)
        target = value.frozen_copy()
        actor = DecisionActor(
            SPEC,
            torch.as_tensor(CONTROLLER["decision_low"]).float(),
            torch.as_tensor(CONTROLLER["decision_high"]).float(),
            NETWORK,
        )
        plant = PortHamiltonianPlant(SPEC, NETWORK)
        states = torch.randn(8, 2)
        samples = actor.sample(states, sample_count=4)
        frozen = {
            "target_value": target,
            "policy": actor,
            "plant": plant.frozen_copy(),
            "temperature_per_second": torch.tensor(0.1),
            "discount_rate": 0.1,
            "resolvent_step": 1.0,
            "reward_rate": lambda s, a: -s[:, :1].square(),
        }
        loss = res.all_pairs_loss(
            res.resolvent_residuals(
                value, frozen, controller(), states, samples, create_graph=True
            )
        )
        loss.backward()
        self.assertTrue(all(p.grad is not None for p in value.parameters()))
        self.assertGreater(sum(p.grad.norm() for p in value.parameters()).item(), 0.0)
        self.assertTrue(all(p.grad is None for p in actor.parameters()))
        self.assertTrue(all(p.grad is None for p in plant.parameters()))


class TestResidualLosses(unittest.TestCase):
    def test_all_pairs_identity(self):
        draws = torch.randn(5, 4, 1, dtype=torch.float64, generator=torch.Generator().manual_seed(1))
        pairs = 0.0
        for row in draws.squeeze(-1):
            total = sum(
                row[i] * row[j] for i in range(4) for j in range(4) if i != j
            )
            pairs = pairs + total / 12.0
        self.assertAlmostEqual(
            res.all_pairs_loss(draws).item(), (pairs / 5.0).item(), places=10
        )

    def test_all_pairs_can_be_negative_and_needs_two_draws(self):
        self.assertLess(res.all_pairs_loss(double([[1.0, -1.0]])).item(), 0.0)
        with self.assertRaises(ValueError):
            res.all_pairs_loss(double([[1.0]]))

    def test_moments_and_gradient_of_the_estimators(self):
        # R_i=psi(1+s xi_i)+c: m=psi+c, v=psi^2 s^2, dm/dpsi=1.
        psi = torch.tensor(1.5, dtype=torch.float64, requires_grad=True)
        offset, noise, count, rows = 0.25, 0.6, 4, 400000
        generator = torch.Generator().manual_seed(2)
        xi = torch.randn(rows, count, dtype=torch.float64, generator=generator)
        residuals = psi * (1.0 + noise * xi) + offset
        mean, variance = psi.item() + offset, (psi.item() * noise) ** 2

        per_row = residuals.mean(dim=1).square() - residuals.var(dim=1) / count
        exact_variance = 4 * mean**2 * variance / count + 2 * variance**2 / (
            count * (count - 1)
        )
        margin = 5.0 * math.sqrt(exact_variance / rows)
        loss = res.all_pairs_loss(residuals)
        self.assertLess(abs(loss.item() - mean**2), margin)
        self.assertAlmostEqual(per_row.var().item() / exact_variance, 1.0, delta=0.02)

        loss.backward()
        self.assertAlmostEqual(psi.grad.item(), 2.0 * mean, delta=0.02)

        # One draw per state keeps the full excess v, averaging keeps v/N.
        one_draw = res.averaged_draw_loss(residuals[:, :1].detach())
        averaged = res.averaged_draw_loss(residuals.detach())
        self.assertAlmostEqual(one_draw.item() - mean**2, variance, delta=0.03)
        self.assertAlmostEqual(averaged.item() - mean**2, variance / count, delta=0.01)


class TestStopping(unittest.TestCase):
    def test_a_solve_completes_after_exactly_k_steps(self):
        for steps in (0, 3, 4):
            running = res.solver_stopping_result(inner_steps=steps, steps_per_solve=5)
            self.assertEqual(running, {"inner_steps": steps, "completed": False})
            self.assertEqual(res.solver_status(running), "budget_exit")
        done = res.solver_stopping_result(inner_steps=5, steps_per_solve=5)
        self.assertEqual(done, {"inner_steps": 5, "completed": True})
        self.assertEqual(res.solver_status(done), "completed")

    def test_the_audit_and_tolerance_are_gone(self):
        for name in (
            "AuditRead",
            "audit_indicators",
            "audit_branch",
            "refined_draw_count",
            "solver_tolerance",
            "solver_should_stop",
        ):
            self.assertFalse(hasattr(res, name), name)

    def test_moving_operator_bound(self):
        bound = res.moving_operator_bound(
            2.0, contraction=0.5, solver_error_bound=0.25, fixed_point_motion_bound=0.5
        )
        self.assertEqual(bound, 1.75)
        with self.assertRaises(ValueError):
            res.moving_operator_bound(
                2.0, contraction=1.0, solver_error_bound=0.0, fixed_point_motion_bound=0.0
            )


class ConstantCritic:
    def minimum(self, states, decisions):
        return states.new_full((states.shape[0], 1), 3.0)


class ConstantActor:
    def __init__(self, log_probability=-2.0):
        self.log_probability = log_probability

    def sample(self, states, *, sample_count=1, **kwargs):
        decisions = states.new_zeros(states.shape[0], sample_count, 3)
        return {
            "decisions": decisions,
            "log_probabilities": states.new_full(
                (states.shape[0], sample_count, 1), self.log_probability
            ),
        }


class TestContinuationTargets(unittest.TestCase):
    def test_generator_continuation(self):
        # V+Delta(GV-lambda V)=5+0.01*(-11.75-0.5), using the decision's initial torque.
        result = tgt.generator_continuation(
            linear_plant(),
            Quadratic(np.eye(2)),
            controller(),
            double([[1.0, 2.0]]),
            double([[0.0, 2.0, 1.0]]),
            double([[0.01]]),
            discount_rate=0.1,
        )
        self.assertEqual(result.shape, (1, 1))
        self.assertAlmostEqual(result.item(), 4.8775, places=10)

    def test_generator_continuation_steps_each_row_by_its_own_interval(self):
        # Rows of 0.2 T, T and 3 T (T=0.01), all at s=(1,2): V=5, GV-lambda V=-12.25.
        period = 0.01
        durations = double([[0.2 * period], [period], [3.0 * period]])
        terms = (
            linear_plant(),
            Quadratic(np.eye(2)),
            controller(),
            double([[1.0, 2.0]] * 3),
            double([[0.0, 2.0, 1.0]] * 3),
        )
        result = tgt.generator_continuation(*terms, durations, discount_rate=0.1)
        self.assertEqual(result.shape, (3, 1))
        for row, duration in enumerate(durations.reshape(-1).tolist()):
            self.assertAlmostEqual(result[row, 0].item(), 5.0 - 12.25 * duration, places=10)
        # A [B] duration vector reads the same rows.
        flat = tgt.generator_continuation(*terms, durations.reshape(-1), discount_rate=0.1)
        torch.testing.assert_close(flat, result, rtol=0, atol=0)
        # Where Delta=T the row is bitwise the fixed-step value V+T(GV-lambda V).
        values, generated = tgt._value_and_generator(*terms)
        fixed = values + period * (generated - 0.1 * values)
        self.assertEqual(result[1, 0].item(), fixed[1, 0].item())
        self.assertNotEqual(result[0, 0].item(), fixed[0, 0].item())
        self.assertNotEqual(result[2, 0].item(), fixed[2, 0].item())

    def test_generator_target_error_is_second_order_in_each_rows_interval(self):
        # Coasting y'=v, v'=0 (zero gains, no noise), V=y^2+v^2 at s=(1,2) and a rate-1
        # reward. The exact one-interval backup of V over Delta is
        #   r_Delta + e^(-lambda Delta) V(s_Delta),  r_Delta=(1-e^(-lambda Delta))/lambda,
        # s_Delta=(1+2 Delta, 2), so V(s_Delta)=5+4 Delta+4 Delta^2. With V=5, GV=2yv=4
        # and lambda=0.1, the generator target r_Delta + V + Delta(GV-lambda V) misses it by
        # c2 Delta^2 + O(Delta^3), c2=4-lambda*4+lambda^2*5/2=3.625, on every row; a step
        # of the fixed T would miss it by (T-Delta)(GV-lambda V)=3.5(T-Delta) to first order.
        rate, period = 0.1, 0.01
        durations = [0.2 * period, period, 3.0 * period]
        rewards = [(1.0 - math.exp(-rate * d)) / rate for d in durations]
        batch = transition_batch(
            states=[[1.0, 2.0]] * 3,
            decisions=[[0.0, 0.0, 0.0]] * 3,
            durations=[[d] for d in durations],
            rewards=[[r] for r in rewards],
            terminated=[[0.0]] * 3,
        )
        continuation = tgt.generator_continuation(
            linear_plant(sigma=0.0),
            Quadratic(np.eye(2)),
            controller(),
            batch["states"],
            batch["decisions"],
            batch["durations"],
            discount_rate=rate,
        )
        targets = tgt.interval_critic_target(batch, continuation)
        self.assertEqual(targets.shape, (3, 1))
        for row, (duration, reward) in enumerate(zip(durations, rewards)):
            exact = reward + math.exp(-rate * duration) * (
                (1.0 + 2.0 * duration) ** 2 + 4.0
            )
            error = exact - targets[row, 0].item()
            self.assertAlmostEqual(error / duration**2, 3.625, delta=0.01 * 3.625)
            if row != 1:
                fixed = reward + 5.0 + period * (4.0 - rate * 5.0)
                self.assertAlmostEqual(
                    (fixed - exact) / ((period - duration) * 3.5), 1.0, delta=0.1
                )
                self.assertLess(abs(error), 0.1 * abs(fixed - exact))

    def test_rollout_uses_survivors_and_falls_back(self):
        # V(s)=y^2+v^2: endpoints 1 and 4 survive while 100 fails in row one; none in two.
        endpoints = double(
            [
                [[1.0, 0.0], [2.0, 0.0], [10.0, 0.0]],
                [[5.0, 0.0], [6.0, 0.0], [7.0, 0.0]],
            ]
        )
        paths = {
            "endpoints": endpoints,
            "survived": torch.tensor(
                [[[True], [True], [False]], [[False], [False], [False]]]
            ),
        }
        batch = transition_batch(
            durations=[[1.0], [2.0]],
            next_states=[[0.0, 0.0], [3.0, 0.0]],
        )
        result = tgt.rollout_continuation(
            Quadratic(np.eye(2)), paths, batch, discount_rate=0.5
        )
        self.assertAlmostEqual(result[0, 0].item(), math.exp(-0.5) * 2.5, places=10)
        self.assertAlmostEqual(result[1, 0].item(), math.exp(-1.0) * 9.0, places=10)

    def test_td_charges_the_next_interval(self):
        result = tgt.td_continuation(
            ConstantCritic(),
            ConstantActor(-2.0),
            double([[0.0, 0.0]]),
            double([[1.0]]),
            double([[0.5]]),
            alpha_c=double(0.2),
            discount_rate=0.4,
        )
        weight = (1.0 - math.exp(-0.2)) / 0.4
        expected = math.exp(-0.4) * (3.0 + 0.2 * weight * 2.0)
        self.assertAlmostEqual(result.item(), expected, places=10)

    def test_interval_target_masks_only_termination(self):
        batch = transition_batch(
            rewards=[[1.0], [1.0], [1.0]],
            terminated=[[0.0], [1.0], [0.0]],
            truncated=[[0.0], [0.0], [1.0]],
        )
        target = tgt.interval_critic_target(batch, double([[2.0], [2.0], [2.0]]))
        self.assertEqual(target.squeeze(-1).tolist(), [3.0, 1.0, 3.0])


class TestRateTargets(unittest.TestCase):
    def setUp(self):
        self.value = Quadratic(np.eye(2))

    def test_data_rate_target(self):
        # V(s)=5, V(s')=4, reward 0.5, duration 0.5; the terminated row drops V(s').
        batch = transition_batch(
            states=[[1.0, 2.0], [1.0, 2.0]],
            next_states=[[2.0, 0.0], [2.0, 0.0]],
            durations=[[0.5], [0.5]],
            rewards=[[0.5], [0.5]],
            terminated=[[0.0], [1.0]],
        )
        rates = tgt.data_rate_target(self.value, batch, discount_rate=0.2)
        self.assertAlmostEqual(
            rates[0, 0].item(), (0.5 + math.exp(-0.1) * 4.0 - 5.0) / 0.5, places=10
        )
        self.assertAlmostEqual(rates[1, 0].item(), (0.5 - 5.0) / 0.5, places=10)

    def test_rollout_rate_target(self):
        batch = transition_batch(
            durations=[[0.5]], rewards=[[0.5]], terminated=[[0.0]]
        )
        rate = tgt.rollout_rate_target(self.value, batch, double([[4.0]]))
        self.assertAlmostEqual(rate.item(), (0.5 + 4.0 - 5.0) / 0.5, places=10)

    def test_generator_rate_target(self):
        # rho+GV-lambda V=-1-11.75-0.5, with no entropy term.
        rate = tgt.generator_rate_target(
            linear_plant(),
            self.value,
            controller(),
            double([[1.0, 2.0]]),
            double([[0.0, 2.0, 1.0]]),
            state_reward,
            discount_rate=0.1,
        )
        self.assertAlmostEqual(rate.item(), -13.25, places=10)

    def test_data_weights_and_clip(self):
        # Sigma^T grad V=2 at s=(1,2), so 1/(4/0.01+1)=1/401; zero gradient at the origin.
        states = double([[1.0, 2.0], [0.0, 0.0]])
        weights = tgt.data_rate_weights(
            linear_plant(),
            self.value,
            states,
            double([[0.01], [0.01]]),
            weight_max=0.5,
            noise_variance_floor=1.0,
        )
        self.assertAlmostEqual(weights[0, 0].item(), 1.0 / 401.0, places=10)
        self.assertEqual(weights[1, 0].item(), 0.5)
        self.assertEqual(
            tgt.data_rate_weights(
                linear_plant(),
                self.value,
                states,
                double([[0.01], [0.01]]),
                weight_max=0.001,
                noise_variance_floor=1.0,
            )[0, 0].item(),
            0.001,
        )
        # A zero floor at zero gradient must clip rather than return infinity.
        self.assertEqual(
            tgt.data_rate_weights(
                linear_plant(),
                self.value,
                double([[0.0, 0.0]]),
                double([[0.01]]),
                weight_max=0.5,
                noise_variance_floor=0.0,
            ).item(),
            0.5,
        )

    def test_build_targets_are_detached_with_expected_shapes(self):
        batch = transition_batch(
            states=[[1.0, 2.0], [0.5, 0.0]],
            decisions=[[0.0, 2.0, 1.0], [0.0, 1.0, 1.0]],
            next_states=[[1.0, 1.0], [0.0, 0.0]],
            durations=[[0.01], [0.01]],
            rewards=[[0.0], [0.0]],
            terminated=[[0.0], [0.0]],
        )
        paths = {
            "endpoints": batch["next_states"].unsqueeze(1).repeat(1, 2, 1),
            "survived": torch.ones(2, 2, 1, dtype=torch.bool),
        }
        config = {
            "replay_weight_max": 1.0,
            "noise_variance_floor": 1.0,
        }
        targets = tgt.build_rate_targets(
            linear_plant(),
            self.value,
            controller(),
            batch,
            paths,
            double([[1.0, 0.0]] * 3),
            double([[0.0, 1.0, 1.0]] * 3),
            state_reward,
            config=config,
            discount_rate=0.1,
        )
        self.assertEqual(targets["data"].shape, (2, 1))
        self.assertEqual(targets["rollout"].shape, (2, 1))
        self.assertEqual(targets["data_weights"].shape, (2, 1))
        self.assertEqual(targets["generator"].shape, (3, 1))
        for name, target in targets.items():
            self.assertFalse(target.requires_grad, name)
        # With endpoints equal to the observed ones, the two interval rates coincide.
        self.assertLess((targets["data"] - targets["rollout"]).abs().max().item(), 1e-9)


def velocity_value():
    """V(s)=v, linear in the coordinate carrying the constant force."""

    class Velocity(nn.Module):
        def forward(self, states):
            return states[..., 1:]

    return Velocity()


def never_fails(states):
    return torch.zeros(states.shape[0], dtype=torch.bool)


def gate_batch(rows, *, seed, true_force, true_sigma, duration=0.01):
    """Observed endpoints with zero gains: v'=v+Delta*f+sigma*sqrt(Delta)*xi, exactly."""
    generator = torch.Generator().manual_seed(seed)
    states = torch.randn(rows, 2, dtype=torch.float64, generator=generator)
    noise = torch.randn(rows, 1, dtype=torch.float64, generator=generator)
    next_states = states.clone()
    next_states[:, 1:] += duration * true_force + true_sigma * math.sqrt(duration) * noise
    return {
        "states": states,
        "decisions": torch.zeros(rows, 3, dtype=torch.float64),
        "durations": torch.full((rows, 1), duration, dtype=torch.float64),
        "horizons": torch.full((rows, 1), duration, dtype=torch.float64),
        "rewards": torch.zeros(rows, 1, dtype=torch.float64),
        "next_states": next_states,
        "terminated": torch.zeros(rows, 1, dtype=torch.float64),
        "truncated": torch.zeros(rows, 1, dtype=torch.float64),
    }


def no_task_terms(discount_rate=0.1):
    """Zero reward and payoff leave only the continuation in each return."""
    return {
        "reward_rate": lambda states, torques: torch.zeros_like(states[..., :1]),
        "terminal_payoff": lambda states: torch.zeros_like(states[..., :1]),
        "discount_rate": discount_rate,
    }


def gate_samples(candidate, admitted, batch, *, paths, seed, discount_rate=0.1):
    noise = sample_brownian_increments(
        batch["durations"],
        path_count=paths,
        noise_dim=candidate.noise_dim,
        maximum_step=0.001,
        physics_dt=0.001,
        controller_refresh_steps=1,
        seed=seed,
    )
    return adm.paired_gate_samples(
        candidate,
        admitted,
        velocity_value(),
        controller(),
        batch,
        failure_predicate=never_fails,
        return_terms=no_task_terms(discount_rate),
        maximum_step=0.001,
        physics_dt=0.001,
        controller_refresh_steps=1,
        path_count=paths,
        brownian_increments=noise,
    )


def hand_samples(observed, candidate, admitted, horizons):
    return {
        "observed_returns": double(observed),
        "candidate_returns": double(candidate),
        "admitted_returns": double(admitted),
        "horizons": double(horizons),
    }


class TestGateRows(unittest.TestCase):
    def test_hand_rows(self):
        # e_c=0 and e_a=1 at H=1, so raw=-1; S_c^2=2, S_a^2=0 at L=2, so the correction
        # is 2/(2*1)=1 and D=-2.
        samples = hand_samples(
            [[2.0], [2.0]],
            [[1.0, 3.0], [1.0, 3.0]],
            [[1.0, 1.0], [1.0, 1.0]],
            [[1.0], [1.0]],
        )
        rows = adm.corrected_gate_rows(samples)
        self.assertAlmostEqual(rows["raw_differences"][0, 0].item(), -1.0)
        self.assertAlmostEqual(rows["variance_corrections"][0, 0].item(), 1.0)
        self.assertAlmostEqual(rows["corrected_differences"][0, 0].item(), -2.0)
        self.assertEqual(rows["corrected_differences"].shape, (2, 1))

    def test_rows_scale_by_the_horizon(self):
        samples = hand_samples(
            [[2.0], [2.0]],
            [[1.0, 3.0], [1.0, 3.0]],
            [[1.0, 1.0], [1.0, 1.0]],
            [[0.5], [0.5]],
        )
        rows = adm.corrected_gate_rows(samples)
        self.assertAlmostEqual(rows["corrected_differences"][0, 0].item(), -8.0)

    def test_minimum_rows_and_paths(self):
        two = double([[1.0, 2.0], [1.0, 2.0]])
        base = {
            "observed_returns": double([[1.0], [1.0]]),
            "candidate_returns": two,
            "admitted_returns": two,
            "horizons": double([[1.0], [1.0]]),
        }
        with self.assertRaises(ValueError):
            adm.corrected_gate_rows(
                {**base, "candidate_returns": two[:, :1], "admitted_returns": two[:, :1]}
            )
        with self.assertRaises(ValueError):
            adm.corrected_gate_rows(
                {
                    **base,
                    "observed_returns": base["observed_returns"][:1],
                    "candidate_returns": two[:1],
                    "admitted_returns": two[:1],
                    "horizons": base["horizons"][:1],
                }
            )

    def test_identical_plants_vanish_pathwise_and_do_not_pass(self):
        plant = PortHamiltonianPlant(SPEC, NETWORK)
        value = SmoothValueHead(SPEC, NETWORK)
        batch = {
            key: tensor.float() for key, tensor in gate_batch(32, seed=3, true_force=0.0, true_sigma=0.5).items()
        }
        noise = sample_brownian_increments(
            batch["durations"],
            path_count=4,
            noise_dim=plant.noise_dim,
            maximum_step=0.001,
            physics_dt=0.001,
            controller_refresh_steps=1,
            seed=4,
        )
        samples = adm.paired_gate_samples(
            plant,
            plant.frozen_copy(),
            value,
            controller(),
            batch,
            failure_predicate=never_fails,
            return_terms=no_task_terms(),
            maximum_step=0.001,
            physics_dt=0.001,
            controller_refresh_steps=1,
            path_count=4,
            brownian_increments=noise,
        )
        rows = adm.corrected_gate_rows(samples)
        self.assertEqual(rows["raw_differences"].abs().max().item(), 0.0)
        self.assertEqual(rows["corrected_differences"].abs().max().item(), 0.0)
        interval = adm.gate_interval(rows, margin_z=1.645)
        self.assertEqual(adm.gate_branch(interval), "inconclusive")

    def test_corrected_mean_is_the_difference_of_exact_squared_errors(self):
        # c_a=v+Delta f_a, so beta_a=Delta(f_true-f_a) exactly in this linear case. The
        # plants have different noise, so the uncorrected mean is biased at L=2.
        duration, lam, paths, rows = 0.01, 0.1, 2, 200000
        true_force, candidate_force, admitted_force = 1.0, 1.5, 0.0
        candidate_sigma, admitted_sigma = 2.0, 0.5
        batch = gate_batch(rows, seed=5, true_force=true_force, true_sigma=1.0)
        samples = gate_samples(
            linear_plant(candidate_force, candidate_sigma),
            linear_plant(admitted_force, admitted_sigma),
            batch,
            paths=paths,
            seed=6,
            discount_rate=lam,
        )
        out = adm.corrected_gate_rows(samples)
        scale = math.exp(-lam * duration) / duration
        exact = scale**2 * duration**2 * (
            (true_force - candidate_force) ** 2 - (true_force - admitted_force) ** 2
        )
        corrected = out["corrected_differences"]
        margin = 4.0 * corrected.std().item() / math.sqrt(rows)
        self.assertLess(abs(corrected.mean().item() - exact), margin)

        bias = scale**2 * duration * (candidate_sigma**2 - admitted_sigma**2) / paths
        raw = out["raw_differences"]
        self.assertLess(
            abs(raw.mean().item() - (exact + bias)),
            4.0 * raw.std().item() / math.sqrt(rows),
        )
        self.assertGreater(abs(bias), 3.0 * margin)

    def gate_on_failing_paths(self, batch, *, lam=0.1, payoff=-5.0, rate=-1.0, paths=2):
        noise = sample_brownian_increments(
            batch["horizons"],
            path_count=paths,
            noise_dim=1,
            maximum_step=0.001,
            physics_dt=0.001,
            controller_refresh_steps=1,
            seed=8,
        )
        return adm.paired_gate_samples(
            linear_plant(),
            linear_plant(),
            velocity_value(),
            controller(),
            batch,
            failure_predicate=lambda s: torch.ones(s.shape[0], dtype=torch.bool),
            return_terms={
                "reward_rate": lambda s, a: torch.full_like(s[..., :1], rate),
                "terminal_payoff": lambda s: torch.full_like(s[..., :1], payoff),
                "discount_rate": lam,
            },
            maximum_step=0.001,
            physics_dt=0.001,
            controller_refresh_steps=1,
            path_count=paths,
            brownian_increments=noise,
        )

    def test_failed_paths_keep_their_reward_and_charge_once(self):
        lam, h = 0.1, 0.001
        batch = gate_batch(4, seed=7, true_force=0.0, true_sigma=0.1)
        batch["next_states"][:, 1] = 2.0
        batch["rewards"][:, 0] = -7.0
        batch["terminated"][0, 0] = 1.0
        samples = self.gate_on_failing_paths(batch, lam=lam)
        # Every path fails on its first step: a trapezoid of rate -1, then one charge.
        running = 0.5 * h * (-1.0 - math.exp(-lam * h))
        expected = running + math.exp(-lam * h) * -5.0
        for name in ("candidate_returns", "admitted_returns"):
            torch.testing.assert_close(
                samples[name], torch.full((4, 2), expected, dtype=torch.float64)
            )
        # A failed observed row has no continuation; its charge is inside r_i.
        discount = math.exp(-lam * 0.01)
        self.assertAlmostEqual(samples["observed_returns"][0, 0].item(), -7.0)
        self.assertAlmostEqual(
            samples["observed_returns"][1, 0].item(), -7.0 + discount * 2.0
        )
        summary = samples["candidate_summary"]
        self.assertEqual(summary["failure_rate"], 1.0)
        self.assertAlmostEqual(summary["running_reward"], running)
        self.assertAlmostEqual(summary["failure_payoff"], math.exp(-lam * h) * -5.0)
        self.assertEqual(summary["continuation"], 0.0)
        self.assertEqual(samples["observed_failure_rate"], 0.25)

    def test_all_failed_paths_stay_finite_with_no_fallback(self):
        batch = gate_batch(4, seed=7, true_force=0.0, true_sigma=0.1)
        rows = adm.corrected_gate_rows(self.gate_on_failing_paths(batch))
        self.assertTrue(torch.isfinite(rows["corrected_differences"]).all())
        self.assertEqual(rows["variance_corrections"].abs().max().item(), 0.0)

    def test_scheduled_horizon_survives_an_observed_failure(self):
        # The observed row failed after 2 ms of a 10 ms schedule. The model still runs
        # the schedule, and the residual divides by it.
        batch = gate_batch(2, seed=7, true_force=0.0, true_sigma=0.0)
        batch["durations"][0, 0] = 0.002
        batch["terminated"][0, 0] = 1.0
        batch["rewards"][0, 0] = -3.0
        samples = gate_samples(
            linear_plant(0.0, 0.0), linear_plant(0.0, 0.0), batch, paths=2, seed=1
        )
        torch.testing.assert_close(samples["horizons"], batch["horizons"])
        start = batch["states"][0, 1].item()
        self.assertAlmostEqual(
            samples["candidate_returns"][0, 0].item(),
            math.exp(-0.1 * 0.01) * start,
            places=9,
        )
        errors = (samples["observed_returns"] - samples["admitted_returns"]) / 0.01
        terms, _ = adm.admitted_fit_terms(samples)
        self.assertAlmostEqual(terms[0, 0].item(), errors[0, 0].item() ** 2, places=6)

    def test_plants_disagreeing_on_failure_are_scored_by_the_charge(self):
        # A surviving path near the limit has V close to F, and the failed path pays F.
        # With the charge the two returns are close instead of differing by |F|.
        lam, payoff = 0.1, -10.0
        batch = gate_batch(2, seed=3, true_force=0.0, true_sigma=0.0, duration=0.001)
        batch["states"][:, 1] = payoff

        def rollout_returns(fails):
            noise = sample_brownian_increments(
                batch["horizons"],
                path_count=2,
                noise_dim=1,
                maximum_step=0.001,
                physics_dt=0.001,
                controller_refresh_steps=1,
                seed=1,
            )
            predicate = (
                (lambda s: torch.ones(s.shape[0], dtype=torch.bool)) if fails else never_fails
            )
            return adm.paired_gate_samples(
                linear_plant(0.0, 0.0),
                linear_plant(0.0, 0.0),
                velocity_value(),
                controller(),
                batch,
                failure_predicate=predicate,
                return_terms={
                    **no_task_terms(lam),
                    "terminal_payoff": lambda s: torch.full_like(s[..., :1], payoff),
                },
                maximum_step=0.001,
                physics_dt=0.001,
                controller_refresh_steps=1,
                path_count=2,
                brownian_increments=noise,
            )["candidate_returns"]

        gap = (rollout_returns(True) - rollout_returns(False)).abs().max().item()
        self.assertLess(gap, 1e-6)


class TestAdmissionDecision(unittest.TestCase):
    def rows(self, values):
        tensor = double(values).reshape(-1, 1)
        return {
            "raw_differences": tensor + 1.0,
            "variance_corrections": torch.ones_like(tensor),
            "corrected_differences": tensor,
        }

    def test_hand_interval(self):
        # mean=-2, std=sqrt 2, n=2: the half-width is z.
        interval = adm.gate_interval(self.rows([-3.0, -1.0]), margin_z=1.645)
        self.assertAlmostEqual(interval["upper"], -2.0 + 1.645)
        self.assertAlmostEqual(interval["lower"], -2.0 - 1.645)
        self.assertAlmostEqual(interval["corrected_mean"], -2.0)
        self.assertAlmostEqual(interval["corrected_std"], math.sqrt(2.0))
        self.assertEqual(interval["rows"], 2)
        self.assertEqual(adm.gate_branch(interval), "supported_improvement")

        wide = adm.gate_interval(self.rows([-3.0, -1.0]), margin_z=2.1)
        self.assertEqual(adm.gate_branch(wide), "inconclusive")
        against = adm.gate_interval(self.rows([3.0, 1.0]), margin_z=1.645)
        self.assertEqual(adm.gate_branch(against), "evidence_against")

    def test_width_overrides_the_margin(self):
        interval = adm.gate_interval(self.rows([-3.0, -1.0]), margin_z=1.645, width=5.0)
        self.assertAlmostEqual(interval["upper"], 3.0)
        self.assertEqual(adm.gate_branch(interval), "inconclusive")

    def test_non_finite_statistics_are_rejected(self):
        for values in ([-3.0, float("nan")], [-float("inf"), -1.0]):
            interval = adm.gate_interval(self.rows(values), margin_z=1.645)
            self.assertEqual(adm.gate_branch(interval), "rejected_nonfinite")

    def test_budgeted_width_by_hand(self):
        # g=1 spends beta/2; C=5*4/0.25=80 at M=2, H_min=0.5.
        width = adm.budgeted_gate_width(
            rows=8, look=1, gate_failure_budget=0.5, return_bound=2.0, min_horizon=0.5
        )
        self.assertAlmostEqual(width, 80.0 * math.sqrt(2.0 * math.log(8.0) / 8.0))
        later = adm.budgeted_gate_width(
            rows=8, look=5, gate_failure_budget=0.5, return_bound=2.0, min_horizon=0.5
        )
        self.assertGreater(later, width)
        with self.assertRaises(ValueError):
            adm.budgeted_gate_width(
                rows=8, look=0, gate_failure_budget=0.5, return_bound=2.0, min_horizon=0.5
            )

    def test_budgeted_row_bound_holds(self):
        # |D_i|<=5M^2/H_min^2 for any returns within M, at L>=2.
        generator = torch.Generator().manual_seed(3)
        bound, horizon = 2.0, 0.25
        for paths in (2, 4):
            samples = hand_samples(
                ((2.0 * torch.rand(4000, 1, generator=generator) - 1.0) * bound).tolist(),
                ((2.0 * torch.rand(4000, paths, generator=generator) - 1.0) * bound).tolist(),
                (bound * torch.sign(torch.rand(4000, paths, generator=generator) - 0.5)).tolist(),
                [[horizon]] * 4000,
            )
            rows = adm.corrected_gate_rows(samples)["corrected_differences"]
            self.assertLessEqual(rows.abs().max().item(), 5.0 * bound**2 / horizon**2)

    def test_better_passes_worse_fails(self):
        batch = gate_batch(2000, seed=9, true_force=1.0, true_sigma=0.1)
        close, far = linear_plant(1.05, 0.1), linear_plant(5.0, 0.1)
        better = adm.corrected_gate_rows(gate_samples(close, far, batch, paths=4, seed=10))
        worse = adm.corrected_gate_rows(gate_samples(far, close, batch, paths=4, seed=10))
        same = adm.corrected_gate_rows(
            gate_samples(close, linear_plant(1.05, 0.1), batch, paths=4, seed=10)
        )
        branch = lambda rows: adm.gate_branch(adm.gate_interval(rows, margin_z=1.645))
        self.assertEqual(branch(better), "supported_improvement")
        self.assertEqual(branch(worse), "evidence_against")
        self.assertEqual(branch(same), "inconclusive")


class TestPilotAllocation(unittest.TestCase):
    def test_variance_components_by_hand(self):
        # W^2=2 at each row; row means 0 and 4 have variance 8; v_real=8-2/2=7.
        simulation, real = adm.pilot_variance_components(double([[-1.0, 1.0], [3.0, 5.0]]))
        self.assertAlmostEqual(simulation, 2.0)
        self.assertAlmostEqual(real, 7.0)

    def test_real_component_clips_at_zero(self):
        simulation, real = adm.pilot_variance_components(double([[-3.0, 3.0], [3.0, -3.0]]))
        self.assertAlmostEqual(simulation, 18.0)
        self.assertEqual(real, 0.0)

    def test_pilot_needs_two_rows_and_two_batches(self):
        with self.assertRaises(ValueError):
            adm.pilot_variance_components(double([[1.0], [2.0]]))
        with self.assertRaises(ValueError):
            adm.pilot_variance_components(double([[1.0, 2.0]]))

    def test_components_estimate_the_total_variance_law(self):
        # D=z+e: real variance 4, simulation variance 1 per batch.
        generator = torch.Generator().manual_seed(5)
        rows = 20000
        real = 2.0 * torch.randn(rows, 1, generator=generator, dtype=torch.float64)
        noise = torch.randn(rows, 2, generator=generator, dtype=torch.float64)
        simulation, across = adm.pilot_variance_components(real + noise)
        self.assertAlmostEqual(simulation, 1.0, delta=0.05)
        self.assertAlmostEqual(across, 4.0, delta=0.15)

    def test_stage_cost_counts_both_plants_and_the_pilot(self):
        self.assertEqual(adm.gate_stage_paths(32, 4, pilot_rows=32, pilot_batches=2), 768)
        self.assertEqual(adm.gate_stage_paths(32, 4, pilot_rows=0, pilot_batches=2), 256)

    def test_next_counts_follow_the_larger_component(self):
        always = lambda n, l: True
        self.assertEqual(
            adm.next_gate_counts(8, 4, simulation_variance=2.0, real_variance=1.0, fits=always),
            (8, 8),
        )
        self.assertEqual(
            adm.next_gate_counts(8, 4, simulation_variance=1.0, real_variance=2.0, fits=always),
            (16, 4),
        )
        # Equal components prefer real rows, which reduce both.
        self.assertEqual(
            adm.next_gate_counts(8, 4, simulation_variance=1.0, real_variance=1.0, fits=always),
            (16, 4),
        )

    def test_next_counts_try_the_other_increase_then_stop(self):
        only_rows = lambda n, l: l == 4
        self.assertEqual(
            adm.next_gate_counts(8, 4, simulation_variance=2.0, real_variance=1.0, fits=only_rows),
            (16, 4),
        )
        self.assertIsNone(
            adm.next_gate_counts(
                8, 4, simulation_variance=2.0, real_variance=1.0, fits=lambda n, l: False
            )
        )


class TestTransferAndSpacing(unittest.TestCase):
    def test_transfer_hand_value(self):
        # V=s^T s: (f_c-f_a)*2v+(sigma_c^2-sigma_a^2), at zero torque; largest at v=2.
        states = double([[0.0, 2.0], [1.0, -1.0], [0.0, 0.5]])
        value = Quadratic(np.eye(2))
        result = adm.transfer_diagnostic(
            linear_plant(3.0, 0.5),
            linear_plant(1.0, 1.0),
            value,
            ConstantActor(),
            controller(),
            states,
            discount_rate=0.5,
            policy_draws=2,
        )
        expected = abs(2.0 * 2.0 * 2.0 + (0.25 - 1.0)) / 0.5
        self.assertEqual(result.dim(), 0)
        self.assertAlmostEqual(result.item(), expected, places=10)
        self.assertFalse(result.requires_grad)

    def test_transfer_is_zero_for_identical_plants_and_grows_with_their_gap(self):
        states = double(np.random.default_rng(1).normal(size=(16, 2)))
        value = Quadratic(np.eye(2))
        arguments = dict(discount_rate=0.1, policy_draws=2)
        same = adm.transfer_diagnostic(
            linear_plant(1.0),
            linear_plant(1.0),
            value,
            ConstantActor(),
            controller(),
            states,
            **arguments,
        )
        self.assertEqual(same.item(), 0.0)
        near, far = (
            adm.transfer_diagnostic(
                linear_plant(1.0 + gap),
                linear_plant(1.0),
                value,
                ConstantActor(),
                controller(),
                states,
                **arguments,
            ).item()
            for gap in (0.1, 0.2)
        )
        self.assertAlmostEqual(far / near, 2.0, places=8)

    def test_spacing(self):
        # M=ceil(n_e/log(1+lambda tau)): 3/log(1.1) is 31.5, so 32 outer iterations.
        base = dict(discount_rate=0.1, resolvent_step=1.0)
        self.assertEqual(adm.admission_spacing(efolds=3.0, **base), 32)
        self.assertEqual(
            adm.admission_spacing(efolds=6.0, **base), math.ceil(6.0 / math.log(1.1))
        )
        self.assertEqual(
            adm.admission_spacing(discount_rate=0.5, resolvent_step=2.0, efolds=3.0),
            math.ceil(3.0 / math.log(2.0)),
        )
        for bad in (
            dict(efolds=0.0, **base),
            dict(efolds=3.0, discount_rate=0.0, resolvent_step=1.0),
            dict(efolds=3.0, discount_rate=0.1, resolvent_step=-1.0),
        ):
            with self.assertRaises(ValueError):
                adm.admission_spacing(**bad)
        # Neither the transfer diagnostic J nor delta_V is an input any more.
        with self.assertRaises(TypeError):
            adm.admission_spacing(accepted_transfer=1.0, efolds=3.0, **base)
        with self.assertRaises(TypeError):
            adm.admission_spacing(value_accuracy=0.1, efolds=3.0, **base)

    def test_admitted_fit_terms(self):
        samples = hand_samples(
            [[2.0], [2.0]],
            [[1.0, 1.0], [1.0, 1.0]],
            [[0.0, 2.0], [0.0, 2.0]],
            [[0.5], [0.5]],
        )
        squares, variances = adm.admitted_fit_terms(samples)
        # e=(2-1)/0.5=2; S^2=2 at L=2, so S^2/(L H^2)=2/(2*0.25)=4.
        self.assertAlmostEqual(squares[0, 0].item(), 4.0)
        self.assertAlmostEqual(variances[0, 0].item(), 4.0)


if __name__ == "__main__":
    unittest.main()

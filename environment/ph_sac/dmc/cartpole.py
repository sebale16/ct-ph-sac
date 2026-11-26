"""dm_control's cart-pole domain: six tasks on one cart, motor and rail.

cartpole_swingup (CartpoleSwingupEnv, moved here from environment/ph_sac/tasks.py, which
re-exports CartpoleSwingupEnv and build_cartpole_model):

    plant_parameters: cart_mass, pole_mass, pole_com, pole_inertia (about the centre of
    mass), pole_length (drawing only), cart_damping, pole_damping, gravity, and the rail:
    cart_range (1.8 m) and the soft-limit parameters limit_timeconst, limit_dampratio
    (dm_control's solreflimit 0.08 1), limit_d0, limit_dmax, limit_width,
    limit_midpoint, limit_power (MuJoCo's default solimplimit 0.9 0.95 0.001 0.5 2).
    failure_limits: {} (none). Since 2026-11-10 the slider keeps dm_control's soft rail
    limit: limited to +-cart_range by MuJoCo's constraint solver, as dm_control's
    cartpole.xml; a cart past the rail is pushed back and the episode continues. (Until
    2026-11-10 the limit was removed and |x| >= 1.8 m was a failure, payoff 0, lift
    B = 1 - (x / 1.8)^2.)
    reward_parameters: cart_margin, velocity_margin, failure_rate. The rate is
    dm_control's smooth swing-up reward with the control read as force / force limit:
        r = (1 + cos theta)/2 * (4 + 1 - u^2)/5 * (1 + g(theta_dot/velocity_margin))/2
            * (1 + g(x/cart_margin))/2,   g(d) = 0.1^(d^2),
    in [0, 1], with theta = 0 upright.
    reset_parameters: start_angle (pi, hanging); position_noise, angle_noise,
    velocity_noise, the standard deviations of dm_control's Gaussian start.
    The adaptation experiment's testbed_settings key physical_change
    (common.ph_dmc.cartpole.cartpole_physical_change; this task only): MuJoCo and the
    oracle simulate physical_plant_parameters, the pole's length (pole_length, pole_com and
    pole_inertia from dm_control's capsule geometry, mass kept) or the slider's and the
    hinge's damping times the scale. The reward reads no plant parameter, so it is
    unchanged; the state spec's limit_port_init stays the nominal model's.

cartpole_balance, cartpole_balance_sparse, cartpole_swingup_sparse, cartpole_two_poles,
cartpole_three_poles (CartpoleTaskEnv; CARTPOLE_TASKS gives each its reset law, reward
law and pole count). All keep the swing-up's cart, first pole, motor (+-10 N on the cart,
gear 1 in newtons, the reward's control u = force / 10), dm_control's soft rail limit at
|x| = 1.8 m (no failure set: payoff failure_rate / lambda = 0 is never charged, H = 0,
B = 1), decision box, clock, 1 ms RK4 physics and lambda = 0.5
(common/ph_dmc/cartpole.py). As dm_control's suite/cartpole.py (Balance, _make_model):

  state      (x, theta_1..theta_n, x_dot, theta_dot_1..theta_dot_n); theta_1 from upright,
             theta_k (k > 1) relative to pole k-1 as MuJoCo's nested hinges give it; every
             hinge periodic and wrapped; the cart actuated, its position a raw feature.
  plant      plant_parameters as the swing-up, plus n_poles (2 or 3) for the multi-pole
             rows: identical poles, pole k+1 hinged pole_length up pole k (dm_control:
             body pole_k at "0 0 1", 0.1 kg capsules 0..1 m, hinge damping 2e-6).
  reset      balance (balance, balance_sparse): cart uniform on +-position_range (0.1),
             every hinge uniform on +-angle_range (0.034), velocities velocity_noise
             (0.01) N(0,1). swingup (swingup_sparse, two_poles, three_poles): x
             position_noise N(0,1), theta_1 = start_angle + angle_noise N(0,1), further
             hinges upper_angle_noise (0.1) N(0,1), velocities velocity_noise N(0,1).
  reward     smooth (balance, two_poles, three_poles; cart_margin, velocity_margin):
                 mean_k (1 + cos phi_k)/2 * (4 + max(0, 1 - u^2))/5
                 * (1 + min_k g(theta_dot_k / velocity_margin))/2 * (1 + g(x/cart_margin))/2
             with phi_k = theta_1 + ... + theta_k the absolute angles (dm_control's xmat
             zz) and the RELATIVE hinge rates (dm_control's qvel[1:]); one pole gives the
             swing-up's reward. sparse (balance_sparse, swingup_sparse; cart_bound,
             cosine_bound): 1[|x| <= cart_bound] prod_k 1[cos phi_k >= cosine_bound]
             (0.25 m, 0.995), dm_control's tolerance with margin 0. Optional Lyapunov
             rewards ("orbit" rewards, no default row; selected by the reward_parameters'
             keys, cartpole_reward_law, in place of the task's law; the table rows
             cartpole-two_poles and cartpole-three_poles of modes lyapunov_cart and
             lyapunov_cart_straight carry them), the negative-log transform of the
             acrobot's Xin--Kaneda reward (environment/ph_sac/dmc/acrobot.py) applied to an
             energy-shaping function of the cart and poles:
               lyapunov_cart (LYAPUNOV_CART_KEYS energy_scale, upright_energy,
                 lyapunov_k_cart, reward_floor, failure_rate):
                   V = (E - upright_energy)^2 / 2 + lyapunov_k_cart x^2 / 2,
               lyapunov_cart_straight (LYAPUNOV_STRAIGHT_KEYS: those and
                 lyapunov_k_straight):
                   V = that + lyapunov_k_straight sum_{k >= 2} wrap(theta_k)^2 / 2,
               r = -log(max(V / V_down, reward_floor)),  V_down = energy_scale^2 / 2,
             with E = chain.chain_energy of cartpole_chain (kinetic energy of the cart and
             every pole, potential g sum_j h_j cos phi_j, zero with every pole horizontal;
             the cart's potential is constant), upright_energy = E with every pole upright
             at rest = g sum_j h_j and energy_scale = upright_energy - E(hanging rest) =
             2 g sum_j h_j, so V_down is V at hanging rest (theta_1 = pi, theta_k = 0 for
             k >= 2, x = 0, at rest) and r = 0 there; r = -log(reward_floor) where
             V <= reward_floor V_down (the upright at rest and a small neighbourhood).
             The first is the faithful mapping of the acrobot's reward, which penalizes
             only the actuated coordinate besides the energy (here the cart position x,
             the only actuated coordinate). Its zero set {E = upright_energy, x = 0}
             holds, for n >= 2 poles with the cart near rest, a free multi-pendulum at the
             upright energy that need not approach the upright; a straight chain at that
             energy instead moves as one pendulum on its homoclinic orbit toward the
             upright. The acrobot's elbow term makes that collapse with its actuated joint;
             here it takes the unactuated relative angles theta_k, k >= 2, hence the
             straightness term of the second (with one pole it adds nothing). No theory
             fixes the weights (Xin and Kaneda's bound covers only the acrobot's elbow);
             the scale rule (cartpole_lyapunov_reward_terms) gives the cart at the rail
             limit (x = cart_range) and one relative angle of pi each the cost V_down:
             lyapunov_k_cart = energy_scale^2 / cart_range^2, lyapunov_k_straight =
             energy_scale^2 / pi^2 (two poles: energy_scale 3.924 J, 4.7524 and 1.5601;
             three poles: 8.829 J, 24.059 and 7.8981). Unbounded below, logarithmically,
             so reward_rate_bounds is None for them (a linear reward head); state-only;
             the straightness terms are continuous across theta_k = +-pi but not
             differentiable there. failure_rate stays 0 (no failure set).
  oracle     one pole: the swing-up's closed form (_cartpole_callbacks); several: the
             planar chain on a cart (cartpole_chain), Cholesky; both then add MuJoCo's
             soft rail-limit force (joint_limits.constrained_accelerations with the
             compiled model's slider limit: impedance, reference acceleration and
             regularizer of the row, f >= 0), exactly 0 within +-cart_range.
  limit      the slider: range +-1.8 m, solreflimit 0.08 1 (time constant 0.08 s, above
             MuJoCo's 2 x timestep floor both at our 1 ms and dm_control's 10 ms),
             solimplimit 0.9 0.95 0.001 0.5 2 (impedance from 0.9 to 0.95 over the first
             mm), dof_invweight0 as dm_control's (same masses), MuJoCo's default Newton
             solver (tolerance 1e-8); the state spec declares it as
             limited_position_ranges ((0, -1.8, 1.8),) for the learned plant's limit port,
             which starts at this limit (limit_port_init, the factory's, from the model).
  upright    every pole's absolute angle phi_k within the tube of 0.

Validated against dm_control in tests/test_ph_dmc_cartpole.py (forward dynamics, rewards,
trajectories, resets). The table's pole_inertia 0.00942459 rounds dm_control's compiled
0.0094245927672956 (relative 3e-7); the dm_control comparisons at 1e-8 use the compiled
value.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Callable, Dict, Literal, Mapping, Optional, Tuple

import mujoco
import numpy as np
import torch

from common.ph_dmc._common import LIMIT_KEYS, physical_plant_parameters, require_no_failure_set
from common.ph_dmc.cartpole import (
    PLANT_KEYS,
    cartpole_physical_change,
    cartpole_balance_config,
    cartpole_balance_sparse_config,
    cartpole_swingup_config,
    cartpole_swingup_sparse_config,
    cartpole_three_poles_config,
    cartpole_two_poles_config,
)

from ..base import MujocoPHEnv, validated_state_spec
from ..registry import TestbedSpec, register_testbed
# The irregular clocks (optional testbed_settings, since 2026-11-11; _common.py).
from common.ph_dmc._common import CLOCK_SETTINGS_KEYS, with_clock_settings  # noqa: E402
from ._common import (
    NO_FAILURE,
    TOLERANCE_SCALE_SQUARED,
    failure_rate_payoff,
    limit_attributes,
    limit_boundary_functions,
    limit_failure_mask,
    require_keys,
    tolerance_tensor,
    torque_limit,
    unit_interval_bounds,
)
from .chain import PlanarChain, chain_energy, chain_energy_numpy
from .joint_limits import (
    JointLimits,
    constrained_accelerations,
    constrained_accelerations_numpy,
    joint_limits_from_model,
)
from .limited_chain import limited_chain_callbacks, limited_chain_drift_numpy, limits_engaged_numpy

if TYPE_CHECKING:
    from mujoco import MjModel
    from numpy import ndarray
    from torch import Tensor

    from common.ph_config import TestbedConfig

SETTINGS_KEYS = ("failure_limits", "reward_parameters", "reset_parameters", "plant_parameters")

# dm_control's cart-pole with its compiled masses and pole inertia, the motor in newtons
# (gear 1) and dm_control's soft rail limit on the slider (range +-cart_range, solreflimit
# and solimplimit from plant_parameters, MuJoCo's constraint solver on). The pole turns
# about +y, so its centre of mass sits at (x + l sin theta, l cos theta).
_CARTPOLE_XML = """
<mujoco model="cartpole-swingup">
  <option timestep="{timestep}" integrator="{integrator}" gravity="0 0 -{gravity}">
    <flag contact="disable"/>
  </option>
  <visual><global offwidth="1280" offheight="960"/></visual>
  <worldbody>
    <light pos="0 -2 6"/>
    <camera name="fixed" pos="0 -5 1" xyaxes="1 0 0 0 0 1"/>
    <geom type="capsule" fromto="-{rail} 0.07 1 {rail} 0.07 1" size="0.02" mass="0"/>
    <body name="cart" pos="0 0 1">
      <joint name="slider" type="slide" axis="1 0 0" damping="{cart_damping}" limited="true"
             range="-{rail} {rail}" {limit}/>
      <inertial pos="0 0 0" mass="{cart_mass}" diaginertia="0.01083 0.01667 0.02083"/>
      <geom type="box" size="0.2 0.15 0.1" mass="0"/>
      <body name="pole">
        <joint name="hinge" type="hinge" axis="0 1 0" damping="{pole_damping}" limited="false"/>
        <inertial pos="0 0 {pole_com}" mass="{pole_mass}"
                  diaginertia="{pole_inertia} {pole_inertia} 0.0001"/>
        <geom type="capsule" fromto="0 0 0 0 0 {pole_length}" size="0.045" mass="0"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor name="slide" joint="slider" gear="1" ctrllimited="true"
           ctrlrange="{force_low} {force_high}"/>
  </actuator>
</mujoco>
"""

# The cart-pole's tolerance constant, kept under its old name.
_TOLERANCE_SCALE_SQUARED = TOLERANCE_SCALE_SQUARED


def _wrap(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def _cartpole_coefficients(plant: Mapping[str, float]) -> Tuple[float, float, float, float]:
    """(total mass, m_p l, pole inertia about the hinge J, m_p l g) of M(theta) and the potential.

    M = [[total, m_p l cos theta], [m_p l cos theta, J]], potential m_p l g cos theta.
    """
    coupling = plant["pole_mass"] * plant["pole_com"]
    hinge_inertia = plant["pole_inertia"] + plant["pole_mass"] * plant["pole_com"] ** 2
    return (
        plant["cart_mass"] + plant["pole_mass"],
        coupling,
        hinge_inertia,
        coupling * plant["gravity"],
    )


def _cartpole_force_limit(config: TestbedConfig) -> float:
    return torque_limit(config)


def _cartpole_drift_numpy(
    coefficients: Tuple[float, float, float, float],
    cart_damping: float,
    pole_damping: float,
    state: ndarray,
    torque: ndarray,
) -> ndarray:
    """The single pole's closed-form drift (x_dot, theta_dot, x_ddot, theta_ddot)."""
    _, angle, position_rate, angle_rate = (float(x) for x in state)
    total, coupling, hinge_inertia, gravity_torque = coefficients
    cosine, sine = math.cos(angle), math.sin(angle)
    force_cart = (
        float(np.asarray(torque).reshape(-1)[0])
        - cart_damping * position_rate
        + coupling * sine * angle_rate**2
    )
    force_pole = gravity_torque * sine - pole_damping * angle_rate
    determinant = total * hinge_inertia - (coupling * cosine) ** 2
    return np.array(
        [
            position_rate,
            angle_rate,
            (hinge_inertia * force_cart - coupling * cosine * force_pole) / determinant,
            (total * force_pole - coupling * cosine * force_cart) / determinant,
        ]
    )


def _cartpole_mass_numpy(
    coefficients: Tuple[float, float, float, float], angle: float
) -> ndarray:
    """M(theta) = [[total, m_p l cos theta], [m_p l cos theta, J]] of the single pole."""
    total, coupling, hinge_inertia, _ = coefficients
    off = coupling * math.cos(angle)
    return np.array([[total, off], [off, hinge_inertia]])


def _limited_cartpole_drift_numpy(
    coefficients: Tuple[float, float, float, float],
    cart_damping: float,
    pole_damping: float,
    limits: JointLimits,
    state: ndarray,
    torque: ndarray,
) -> ndarray:
    """The closed form plus the soft rail-limit force; unchanged within the rail."""
    drift = _cartpole_drift_numpy(coefficients, cart_damping, pole_damping, state, torque)
    state = np.asarray(state, dtype=np.float64)
    if not limits_engaged_numpy(limits, state[:2]):
        return drift
    result = constrained_accelerations_numpy(
        limits, _cartpole_mass_numpy(coefficients, float(state[1])), drift[2:], state[:2], state[2:]
    )
    return np.concatenate((drift[:2], result["qacc"]))


def _require_cartpole_settings(config: TestbedConfig, plant_keys: Tuple[str, ...]) -> None:
    require_keys(config["plant_parameters"], plant_keys, "Cart-pole plant_parameters")
    require_no_failure_set(config["name"], config["failure_limits"])


def cartpole_rail_limits(model: MjModel) -> JointLimits:
    """The slider's soft limit of a compiled cart-pole (ours or dm_control's)."""
    return joint_limits_from_model(model, ("slider",))


class CartpoleSwingupEnv(MujocoPHEnv):
    """dm_control's cart-pole swing-up: force on the cart, the pole wrapped, no noise.

    The state is (x, theta, x_dot, theta_dot) with theta = 0 upright; the cart keeps
    dm_control's soft rail limit at +-cart_range.
    """

    def __init__(
        self,
        config: TestbedConfig,
        *,
        decision_mode: Literal["torque", "feedback"] = "feedback",
        seed: Optional[int] = None,
    ) -> None:
        _require_cartpole_settings(config, PLANT_KEYS)
        require_keys(
            config["reward_parameters"],
            ("cart_margin", "velocity_margin", "failure_rate"),
            "Cart-pole reward_parameters",
        )
        require_keys(
            config["reset_parameters"],
            ("start_angle", "position_noise", "angle_noise", "velocity_noise"),
            "Cart-pole reset_parameters",
        )
        super().__init__(
            config,
            build_cartpole_model(config),
            validated_state_spec(config),
            decision_mode=decision_mode,
            seed=seed,
        )
        # The simulated robot's (a physical change's when the config carries one).
        plant = physical_plant_parameters(config)
        self._coefficients = _cartpole_coefficients(plant)
        self._cart_damping = plant["cart_damping"]
        self._pole_damping = plant["pole_damping"]
        self._force_limit = _cartpole_force_limit(config)
        self._limits = cartpole_rail_limits(self.model)

    @property
    def rail_limits(self) -> JointLimits:
        """The slider's soft limit, read from the compiled model."""
        return self._limits

    def _reset_state(self, options: Optional[Mapping[str, object]] = None) -> None:
        """dm_control's swing-up start: Gaussian noise about the hanging pole at rest."""
        if options and "state" in options:
            self._set_state(options["state"])
            return
        reset = self.config["reset_parameters"]
        position = reset["position_noise"] * self._reset_rng.standard_normal()
        angle = reset["start_angle"] + reset["angle_noise"] * self._reset_rng.standard_normal()
        velocities = reset["velocity_noise"] * self._reset_rng.standard_normal(2)
        self._set_state(np.array([position, angle, velocities[0], velocities[1]]))

    def reward_rate(self, state: ndarray, torque: ndarray) -> float:
        """dm_control's smooth swing-up reward; reward_rate_tensor must stay its copy."""
        reward = self.config["reward_parameters"]
        position, angle, _, angle_rate = (float(x) for x in state)
        control = float(np.asarray(torque).reshape(-1)[0]) / self._force_limit
        upright = 0.5 * (1.0 + math.cos(angle))
        centered = 0.5 * (1.0 + math.exp(
            -0.5 * _TOLERANCE_SCALE_SQUARED * (position / reward["cart_margin"]) ** 2
        ))
        small_control = (4.0 + max(0.0, 1.0 - control**2)) / 5.0
        small_velocity = 0.5 * (1.0 + math.exp(
            -0.5 * _TOLERANCE_SCALE_SQUARED * (angle_rate / reward["velocity_margin"]) ** 2
        ))
        return upright * centered * small_control * small_velocity

    def failed(self, state: ndarray) -> bool:
        """Never: a cart past the rail meets dm_control's soft limit and continues."""
        return False

    def terminal_payoff(self, state: ndarray) -> float:
        return self.config["reward_parameters"]["failure_rate"] / self.discount_rate

    def oracle_drift(self, state: ndarray, torque: ndarray) -> ndarray:
        """The closed form, plus the rail-limit force beyond +-cart_range."""
        return _limited_cartpole_drift_numpy(
            self._coefficients, self._cart_damping, self._pole_damping, self._limits, state, torque
        )

    def oracle_diffusion(self, state: ndarray) -> ndarray:
        """The cart-pole has no noise."""
        return np.zeros((self.state_dim, 1))


def build_cartpole_model(config: TestbedConfig) -> MjModel:
    """Build dm_control's cart-pole with its soft rail limit, with MuJoCo RK4 settings.

    The cart and pole are the physical robot's (physical_plant_parameters: a physical
    change's when the config carries one).
    """
    plant = physical_plant_parameters(config)
    controller = config["controller"]
    require_keys(plant, ("cart_range",) + LIMIT_KEYS, "Cart-pole plant_parameters")
    return mujoco.MjModel.from_xml_string(
        _CARTPOLE_XML.format(
            timestep=config["physics_dt"],
            integrator=config["integrator"],
            gravity=plant["gravity"],
            rail=float(plant["cart_range"]),
            limit=limit_attributes(plant),
            cart_mass=plant["cart_mass"],
            cart_damping=plant["cart_damping"],
            pole_mass=plant["pole_mass"],
            pole_com=plant["pole_com"],
            pole_inertia=plant["pole_inertia"],
            pole_length=plant["pole_length"],
            pole_damping=plant["pole_damping"],
            force_low=float(controller["torque_low"][0]),
            force_high=float(controller["torque_high"][0]),
        )
    )


def cartpole_swingup_reward_rate_tensor(
    states: Tensor, torques: Tensor, config: TestbedConfig
) -> Tensor:
    """The swing-up's smooth reward rate on tensor batches, [B,1]."""
    reward = config["reward_parameters"]
    position, angle, angle_rate = states[..., :1], states[..., 1:2], states[..., 3:4]
    control = torques[..., :1] / _cartpole_force_limit(config)
    upright = 0.5 * (1.0 + torch.cos(angle))
    centered = 0.5 * (1.0 + torch.exp(
        -0.5 * _TOLERANCE_SCALE_SQUARED * (position / reward["cart_margin"]).square()
    ))
    small_control = (4.0 + torch.clamp(1.0 - control.square(), min=0.0)) / 5.0
    small_velocity = 0.5 * (1.0 + torch.exp(
        -0.5 * _TOLERANCE_SCALE_SQUARED * (angle_rate / reward["velocity_margin"]).square()
    ))
    return upright * centered * small_control * small_velocity


def cartpole_failure_mask_tensor(states: Tensor, config: TestbedConfig) -> Tensor:
    """All False, [B]: no failure set (the rail is dm_control's soft limit)."""
    return limit_failure_mask(states, NO_FAILURE)


def cartpole_terminal_boundary_functions(
    config: TestbedConfig,
) -> Tuple[Callable[[Tensor], Tensor], Callable[[Tensor], Tensor]]:
    """No failure boundary: H = failure_rate / lambda (0) and B = 1."""
    return limit_boundary_functions(failure_rate_payoff(config), NO_FAILURE)


def _cartpole_callbacks(
    plant: Mapping[str, float],
    limits: Optional[JointLimits] = None,
) -> Tuple[Callable[[Tensor, Tensor], Tensor], Callable[[Tensor], Tensor]]:
    """The single pole's closed-form callbacks; with limits (the slider's soft limit, from
    the compiled model) the rail-limit force joins them, exactly 0 within the rail."""
    total, coupling, hinge_inertia, gravity_torque = _cartpole_coefficients(plant)
    cart_damping, pole_damping = plant["cart_damping"], plant["pole_damping"]

    def cartpole_drift(states: Tensor, actions: Tensor) -> Tensor:
        angle = states[..., 1:2]
        position_rate, angle_rate = states[..., 2:3], states[..., 3:4]
        cosine, sine = torch.cos(angle), torch.sin(angle)
        force_cart = actions - cart_damping * position_rate + coupling * sine * angle_rate.square()
        force_pole = gravity_torque * sine - pole_damping * angle_rate
        determinant = total * hinge_inertia - (coupling * cosine).square()
        accelerations = torch.cat(
            (
                (hinge_inertia * force_cart - coupling * cosine * force_pole) / determinant,
                (total * force_pole - coupling * cosine * force_cart) / determinant,
            ),
            dim=-1,
        )
        if limits is not None:
            off = coupling * cosine
            mass = torch.stack(
                (
                    torch.cat((torch.full_like(off, total), off), dim=-1),
                    torch.cat((off, torch.full_like(off, hinge_inertia)), dim=-1),
                ),
                dim=-2,
            )
            accelerations = constrained_accelerations(
                limits,
                torch.linalg.cholesky(mass),
                accelerations,
                states[..., 0:2],
                states[..., 2:4],
            )["qacc"]
        return torch.cat((position_rate, angle_rate, accelerations), dim=-1)

    def cartpole_diffusion(states: Tensor) -> Tensor:
        return torch.zeros(
            *states.shape[:-1], 4, 1, dtype=states.dtype, device=states.device
        )

    return cartpole_drift, cartpole_diffusion


def cartpole_oracle_callbacks(
    config: TestbedConfig,
) -> Tuple[Callable[[Tensor, Tensor], Tensor], Callable[[Tensor], Tensor]]:
    """The closed form with the compiled model's soft rail limit, both the physical robot's
    (a physical change's when the config carries one)."""
    return _cartpole_callbacks(
        physical_plant_parameters(config), cartpole_rail_limits(build_cartpole_model(config))
    )


def cartpole_upright(config: TestbedConfig, state: ndarray, tube: float) -> bool:
    """The pole, the second coordinate, within tube of upright."""
    return abs(_wrap(float(state[1]))) <= tube


register_testbed(
    TestbedSpec(
        name="cartpole_swingup",
        env_class=CartpoleSwingupEnv,
        config_factory=with_clock_settings(cartpole_swingup_config),
        settings_keys=SETTINGS_KEYS,
        optional_settings_keys=CLOCK_SETTINGS_KEYS,
        reward_rate_tensor=cartpole_swingup_reward_rate_tensor,
        failure_mask_tensor=cartpole_failure_mask_tensor,
        terminal_payoff_value=failure_rate_payoff,
        terminal_boundary_functions=cartpole_terminal_boundary_functions,
        oracle_callbacks=cartpole_oracle_callbacks,
        upright=cartpole_upright,
        reward_rate_bounds=unit_interval_bounds,
        # The adaptation experiment's change: the pole's length or both joints' damping.
        physical_change=cartpole_physical_change,
        # The testbed_settings key discount_rate (_common.py, with_discount_rate).
        discount_setting=True,
    )
)


# ------------------------------------------- balance, sparse rewards, two and three poles

# testbed: (reset law, reward law, number of poles). Reset "balance": cart uniform on
# +-position_range, every hinge uniform on +-angle_range; "swingup": Gaussian about the
# hanging first pole. Reward "smooth": dm_control's smooth cart-pole reward over n poles;
# "sparse": its indicator reward.
CARTPOLE_TASKS: Mapping[str, Tuple[str, str, int]] = {
    "cartpole_balance": ("balance", "smooth", 1),
    "cartpole_balance_sparse": ("balance", "sparse", 1),
    "cartpole_swingup_sparse": ("swingup", "sparse", 1),
    "cartpole_two_poles": ("swingup", "smooth", 2),
    "cartpole_three_poles": ("swingup", "smooth", 3),
}
_PLANT_KEYS = PLANT_KEYS
# The optional Lyapunov rewards (module docstring), selected by their keys in place of the
# task's law: lyapunov_cart (energy and cart) and lyapunov_cart_straight (plus the
# straightness of the poles).
LYAPUNOV_CART_KEYS = (
    "energy_scale", "upright_energy", "lyapunov_k_cart", "reward_floor", "failure_rate",
)
LYAPUNOV_STRAIGHT_KEYS = (
    "energy_scale", "upright_energy", "lyapunov_k_cart", "lyapunov_k_straight",
    "reward_floor", "failure_rate",
)
LYAPUNOV_LAWS = ("lyapunov_cart", "lyapunov_cart_straight")
_REWARD_KEYS = {
    "smooth": ("cart_margin", "velocity_margin", "failure_rate"),
    "sparse": ("cart_bound", "cosine_bound", "failure_rate"),
    "lyapunov_cart": LYAPUNOV_CART_KEYS,
    "lyapunov_cart_straight": LYAPUNOV_STRAIGHT_KEYS,
}
# Any of these keys selects a Lyapunov reward (failure_rate is every law's).
_LYAPUNOV_MARKERS = tuple(key for key in LYAPUNOV_STRAIGHT_KEYS if key != "failure_rate")
# The task laws' own keys, which must not join a Lyapunov reward's.
_TASK_LAW_MARKERS = ("cart_margin", "velocity_margin", "cart_bound", "cosine_bound")
# The tensor Lyapunov reward's chains by (pole count, plant_parameters).
_LYAPUNOV_CHAINS: Dict[Tuple[int, Tuple[Tuple[str, float], ...]], PlanarChain] = {}
_RESET_KEYS = {
    "balance": ("position_range", "angle_range", "velocity_noise"),
    "swingup": ("start_angle", "position_noise", "angle_noise", "velocity_noise"),
}

# dm_control's cart-pole with n identical poles, pole k+1 hinged at the tip of pole k
# (dm_control's _make_model: body pole_k at "0 0 1" in pole_{k-1}), the motor in newtons
# (gear 1) and dm_control's soft rail limit, as _CARTPOLE_XML; pole_length places the next
# hinge.
_CARTPOLE_POLES_XML = """
<mujoco model="cartpole-{n_poles}-poles">
  <option timestep="{timestep}" integrator="{integrator}" gravity="0 0 -{gravity}">
    <flag contact="disable"/>
  </option>
  <visual><global offwidth="1280" offheight="960"/></visual>
  <worldbody>
    <light pos="0 -2 6"/>
    <camera name="fixed" pos="0 -{camera_distance} {camera_height}" xyaxes="1 0 0 0 0 1"/>
    <geom type="capsule" fromto="-{rail} 0.07 1 {rail} 0.07 1" size="0.02" mass="0"/>
    <body name="cart" pos="0 0 1">
      <joint name="slider" type="slide" axis="1 0 0" damping="{cart_damping}" limited="true"
             range="-{rail} {rail}" {limit}/>
      <inertial pos="0 0 0" mass="{cart_mass}" diaginertia="0.01083 0.01667 0.02083"/>
      <geom type="box" size="0.2 0.15 0.1" mass="0"/>
      {poles}
    </body>
  </worldbody>
  <actuator>
    <motor name="slide" joint="slider" gear="1" ctrllimited="true"
           ctrlrange="{force_low} {force_high}"/>
  </actuator>
</mujoco>
"""
_POLE_XML = """<body name="pole_{index}" pos="0 0 {offset}">
        <joint name="hinge_{index}" type="hinge" axis="0 1 0" damping="{pole_damping}" limited="false"/>
        <inertial pos="0 0 {pole_com}" mass="{pole_mass}"
                  diaginertia="{pole_inertia} {pole_inertia} 0.0001"/>
        <geom type="capsule" fromto="0 0 0 0 0 {pole_length}" size="0.045" mass="0"/>
        {child}
      </body>"""


def cartpole_task_laws(config: TestbedConfig) -> Tuple[str, str, int]:
    """(reset law, reward law, number of poles) of one of the five later cart-pole tasks."""
    try:
        return CARTPOLE_TASKS[config["name"]]
    except KeyError:
        raise ValueError(
            f"{config['name']!r} is not one of the cart-pole tasks {sorted(CARTPOLE_TASKS)}."
        ) from None


def _pole_count(config: TestbedConfig) -> int:
    return cartpole_task_laws(config)[2]


def build_cartpole_poles_model(config: TestbedConfig) -> MjModel:
    """Build dm_control's n-pole cart-pole with its soft rail limit, the config's integrator."""
    plant = config["plant_parameters"]
    controller = config["controller"]
    require_keys(plant, ("cart_range",) + LIMIT_KEYS, "Cart-pole plant_parameters")
    n_poles = _pole_count(config)
    poles = ""
    for index in range(n_poles, 0, -1):
        poles = _POLE_XML.format(
            index=index,
            offset=0.0 if index == 1 else plant["pole_length"],
            pole_damping=plant["pole_damping"],
            pole_com=plant["pole_com"],
            pole_mass=plant["pole_mass"],
            pole_inertia=plant["pole_inertia"],
            pole_length=plant["pole_length"],
            child=poles,
        )
    return mujoco.MjModel.from_xml_string(
        _CARTPOLE_POLES_XML.format(
            n_poles=n_poles,
            timestep=config["physics_dt"],
            integrator=config["integrator"],
            gravity=plant["gravity"],
            camera_distance=3 + 2 * n_poles,
            camera_height=0.5 + 0.5 * n_poles,
            rail=float(plant["cart_range"]),
            limit=limit_attributes(plant),
            cart_mass=plant["cart_mass"],
            cart_damping=plant["cart_damping"],
            poles=poles,
            force_low=float(controller["torque_low"][0]),
            force_high=float(controller["torque_high"][0]),
        )
    )


def cartpole_chain(plant: Mapping[str, float], n_poles: int) -> PlanarChain:
    """The PlanarChain of n identical poles on the cart, the force entering the slider."""
    return PlanarChain(
        link_masses=(plant["pole_mass"],) * n_poles,
        link_com_distances=(plant["pole_com"],) * n_poles,
        link_inertias=(plant["pole_inertia"],) * n_poles,
        link_lengths=(plant["pole_length"],) * n_poles,
        joint_damping=(plant["pole_damping"],) * n_poles,
        input_matrix=[[1.0]] + [[0.0]] * n_poles,
        gravity=plant["gravity"],
        cart_mass=plant["cart_mass"],
        cart_damping=plant["cart_damping"],
    )


def cartpole_reward_law(config: Mapping[str, object]) -> str:
    """The reward law of one of the five later cart-pole tasks' configs.

    "lyapunov_cart_straight" when the reward_parameters carry lyapunov_k_straight,
    "lyapunov_cart" when they carry another Lyapunov key (energy_scale, upright_energy,
    lyapunov_k_cart, reward_floor), otherwise the task's own law ("smooth" or "sparse",
    CARTPOLE_TASKS by name; also for a config without reward_parameters). A Lyapunov key
    beside a task law's own key (cart_margin, velocity_margin, cart_bound, cosine_bound)
    raises ValueError; the missing keys of the law chosen are reported by the environment.
    """
    task_law = cartpole_task_laws(config)[1]
    reward = config.get("reward_parameters") or {}
    markers = [key for key in _LYAPUNOV_MARKERS if key in reward]
    if not markers:
        return task_law
    others = [key for key in _TASK_LAW_MARKERS if key in reward]
    if others:
        raise ValueError(
            "Cart-pole reward_parameters mix the keys of several rewards: "
            f"lyapunov {markers}; smooth/sparse {others}."
        )
    return "lyapunov_cart_straight" if "lyapunov_k_straight" in reward else "lyapunov_cart"


def cartpole_lyapunov_reward_terms(
    plant: Mapping[str, float], n_poles: int, *, straight: bool
) -> Dict[str, float]:
    """energy_scale, upright_energy, lyapunov_k_cart (and lyapunov_k_straight) of the plant.

    Through cartpole_chain: upright_energy = chain_energy with every pole upright at rest,
    energy_scale = that minus chain_energy at hanging rest (theta_1 = pi, the others 0,
    x = 0); the scale rule of the weights (no theory fixes them) gives the cart at the rail
    limit and one relative angle of pi each V_down = energy_scale^2 / 2:
    lyapunov_k_cart = energy_scale^2 / cart_range^2, lyapunov_k_straight =
    energy_scale^2 / pi^2. benchmarks/make_ct_ph_sac_table.py writes them into the rows of
    modes lyapunov_cart and lyapunov_cart_straight; the environment reads them from there.
    """
    chain = cartpole_chain(plant, n_poles)
    rest = np.zeros(1 + n_poles)
    hanging = np.zeros(1 + n_poles)
    hanging[1] = math.pi
    upright_energy = float(chain_energy_numpy(chain, rest, rest)[0])
    energy_scale = upright_energy - float(chain_energy_numpy(chain, hanging, rest)[0])
    terms = {
        "energy_scale": energy_scale,
        "upright_energy": upright_energy,
        "lyapunov_k_cart": energy_scale**2 / float(plant["cart_range"]) ** 2,
    }
    if straight:
        terms["lyapunov_k_straight"] = energy_scale**2 / math.pi**2
    return terms


def _validate_lyapunov_cart(reward: Mapping[str, float]) -> None:
    """The energies and weights positive and finite; reward_floor in (0, 1)."""
    for key in ("energy_scale", "upright_energy", "lyapunov_k_cart", "lyapunov_k_straight"):
        if key not in reward:
            continue
        value = float(reward[key])
        if not (math.isfinite(value) and value > 0.0):
            raise ValueError(f"Cart-pole reward_parameters {key} must be positive and finite.")
    if not 0.0 < float(reward["reward_floor"]) < 1.0:
        raise ValueError("Cart-pole reward_parameters reward_floor must lie in (0, 1).")


def cartpole_lyapunov_ratio(
    config: TestbedConfig, state: ndarray, chain: Optional[PlanarChain] = None
) -> float:
    """V / V_down of a Lyapunov reward (module docstring), state (x, theta, x_dot, theta_dot).

    V = (E - upright_energy)^2 / 2 + lyapunov_k_cart x^2 / 2 (+ lyapunov_k_straight
    sum_{k >= 2} wrap(theta_k)^2 / 2 with lyapunov_k_straight), V_down = energy_scale^2 / 2:
    1 at hanging rest, 0 on the reward's zero set. chain is cartpole_chain of the plant
    (built when None).
    """
    reward = config["reward_parameters"]
    n_poles = _pole_count(config)
    if chain is None:
        chain = cartpole_chain(config["plant_parameters"], n_poles)
    state = np.asarray(state, dtype=np.float64)
    nv = 1 + n_poles
    energy = float(chain_energy_numpy(chain, state[:nv], state[nv : 2 * nv])[0])
    position = float(state[0])
    lyapunov = (
        0.5 * (energy - float(reward["upright_energy"])) ** 2
        + 0.5 * float(reward["lyapunov_k_cart"]) * position**2
    )
    if "lyapunov_k_straight" in reward:
        bends = sum(_wrap(float(state[index])) ** 2 for index in range(2, 1 + n_poles))
        lyapunov = lyapunov + 0.5 * float(reward["lyapunov_k_straight"]) * bends
    return lyapunov / (0.5 * float(reward["energy_scale"]) ** 2)


def cartpole_lyapunov_reward_rate(
    config: TestbedConfig, state: ndarray, chain: Optional[PlanarChain] = None
) -> float:
    """-log(max(V / V_down, reward_floor)) with cartpole_lyapunov_ratio's V / V_down."""
    ratio = cartpole_lyapunov_ratio(config, state, chain)
    return -math.log(max(ratio, float(config["reward_parameters"]["reward_floor"])))


class CartpoleTaskEnv(MujocoPHEnv):
    """dm_control's cart-pole balance, sparse and multi-pole tasks: force on the cart, no noise.

    The state is (x, theta_1..theta_n, x_dot, theta_dot_1..theta_dot_n), theta_1 from
    upright and theta_k (k > 1) relative to pole k-1, every hinge wrapped. The task's
    reset law, reward law and pole count come from CARTPOLE_TASKS by the config's name.
    """

    def __init__(
        self,
        config: TestbedConfig,
        *,
        decision_mode: Literal["torque", "feedback"] = "feedback",
        seed: Optional[int] = None,
    ) -> None:
        reset_law, _, n_poles = cartpole_task_laws(config)
        reward_law = cartpole_reward_law(config)
        plant = config["plant_parameters"]
        _require_cartpole_settings(config, _PLANT_KEYS)
        if plant.get("n_poles", n_poles) != n_poles:
            raise ValueError(
                f"{config['name']} has {n_poles} pole(s), not n_poles = {plant['n_poles']!r}."
            )
        require_keys(
            config["reward_parameters"], _REWARD_KEYS[reward_law], "Cart-pole reward_parameters"
        )
        if reward_law in LYAPUNOV_LAWS:
            _validate_lyapunov_cart(config["reward_parameters"])
        reset_keys = _RESET_KEYS[reset_law]
        if reset_law == "swingup" and n_poles > 1:
            reset_keys = reset_keys + ("upper_angle_noise",)
        require_keys(config["reset_parameters"], reset_keys, "Cart-pole reset_parameters")
        spec = validated_state_spec(config)
        if spec["position_dim"] != 1 + n_poles or spec["velocity_dim"] != 1 + n_poles:
            raise ValueError(f"{config['name']} needs {1 + n_poles} positions and velocities.")
        super().__init__(
            config,
            build_cartpole_poles_model(config),
            spec,
            decision_mode=decision_mode,
            seed=seed,
        )
        self.n_poles = n_poles
        self.reset_law = reset_law
        self.reward_law = reward_law
        self._force_limit = _cartpole_force_limit(config)
        self._coefficients = _cartpole_coefficients(plant)
        self._cart_damping = plant["cart_damping"]
        self._pole_damping = plant["pole_damping"]
        self._chain = cartpole_chain(plant, n_poles)
        self._limits = cartpole_rail_limits(self.model)

    @property
    def rail_limits(self) -> JointLimits:
        """The slider's soft limit, read from the compiled model."""
        return self._limits

    def _reset_state(self, options: Optional[Mapping[str, object]] = None) -> None:
        """dm_control's Balance.initialize_episode for the task's reset law."""
        if options and "state" in options:
            self._set_state(options["state"])
            return
        reset = self.config["reset_parameters"]
        rng = self._reset_rng
        if self.reset_law == "balance":
            position = rng.uniform(-reset["position_range"], reset["position_range"])
            angles = rng.uniform(-reset["angle_range"], reset["angle_range"], self.n_poles)
        else:
            position = reset["position_noise"] * rng.standard_normal()
            first = reset["start_angle"] + reset["angle_noise"] * rng.standard_normal()
            angles = np.array([first])
            if self.n_poles > 1:
                upper = reset["upper_angle_noise"] * rng.standard_normal(self.n_poles - 1)
                angles = np.concatenate((angles, upper))
        velocities = reset["velocity_noise"] * rng.standard_normal(1 + self.n_poles)
        self._set_state(np.concatenate(([position], angles, velocities)))

    def _absolute_cosines(self, state: ndarray) -> list:
        angle, cosines = 0.0, []
        for index in range(1, 1 + self.n_poles):
            angle += float(state[index])
            cosines.append(math.cos(angle))
        return cosines

    def lyapunov_ratio(self, state: ndarray) -> Optional[float]:
        """V / V_down of a Lyapunov reward (cartpole_lyapunov_ratio), else None.

        For the evaluation's orbit diagnostics (evaluations/ph_sac/protocol.py).
        """
        if self.reward_law not in LYAPUNOV_LAWS:
            return None
        return cartpole_lyapunov_ratio(self.config, state, self._chain)

    def reward_rate(self, state: ndarray, torque: ndarray) -> float:
        """dm_control's smooth or sparse cart-pole reward, or a Lyapunov reward (module
        docstring); the tensor rewards are its copies."""
        if self.reward_law in LYAPUNOV_LAWS:
            return cartpole_lyapunov_reward_rate(self.config, state, self._chain)
        reward = self.config["reward_parameters"]
        position = float(state[0])
        cosines = self._absolute_cosines(state)
        if self.reward_law == "sparse":
            if abs(position) > reward["cart_bound"]:
                return 0.0
            return float(all(cosine >= reward["cosine_bound"] for cosine in cosines))
        control = float(np.asarray(torque).reshape(-1)[0]) / self._force_limit
        upright = sum(0.5 * (1.0 + cosine) for cosine in cosines) / self.n_poles
        centered = 0.5 * (1.0 + math.exp(
            -0.5 * TOLERANCE_SCALE_SQUARED * (position / reward["cart_margin"]) ** 2
        ))
        small_control = (4.0 + max(0.0, 1.0 - control**2)) / 5.0
        fastest = max(float(rate) ** 2 for rate in state[2 + self.n_poles :])
        small_velocity = 0.5 * (1.0 + math.exp(
            -0.5 * TOLERANCE_SCALE_SQUARED * fastest / reward["velocity_margin"] ** 2
        ))
        return upright * small_control * small_velocity * centered

    def failed(self, state: ndarray) -> bool:
        """Never: a cart past the rail meets dm_control's soft limit and continues."""
        return False

    def terminal_payoff(self, state: ndarray) -> float:
        return self.config["reward_parameters"]["failure_rate"] / self.discount_rate

    def oracle_drift(self, state: ndarray, torque: ndarray) -> ndarray:
        """The swing-up's closed form for one pole, the planar chain for several, each with
        the soft rail-limit force beyond +-cart_range."""
        if self.n_poles == 1:
            return _limited_cartpole_drift_numpy(
                self._coefficients, self._cart_damping, self._pole_damping, self._limits,
                state, torque,
            )
        return limited_chain_drift_numpy(
            self._chain, self._limits, state, np.asarray(torque, dtype=np.float64)
        )

    def oracle_diffusion(self, state: ndarray) -> ndarray:
        """The cart-pole has no noise."""
        return np.zeros((self.state_dim, 1))


def _absolute_angles_tensor(states: Tensor, n_poles: int) -> Tensor:
    return torch.cumsum(states[..., 1 : 1 + n_poles], dim=-1)


def cartpole_smooth_reward_rate_tensor(
    states: Tensor, torques: Tensor, config: TestbedConfig
) -> Tensor:
    """dm_control's smooth cart-pole reward over n poles on tensor batches, [B,1].

    upright = mean_k (1 + cos phi_k)/2 over the absolute angles, small_velocity from the
    fastest relative hinge rate (dm_control's min over qvel[1:]), as env.reward_rate.
    """
    reward = config["reward_parameters"]
    n_poles = _pole_count(config)
    position = states[..., :1]
    control = torques[..., :1] / _cartpole_force_limit(config)
    upright = (0.5 * (1.0 + torch.cos(_absolute_angles_tensor(states, n_poles)))).mean(
        dim=-1, keepdim=True
    )
    centered = 0.5 * (1.0 + tolerance_tensor(position, margin=reward["cart_margin"]))
    small_control = (
        4.0 + tolerance_tensor(control, margin=1.0, sigmoid="quadratic", value_at_margin=0.0)
    ) / 5.0
    rates = states[..., 2 + n_poles : 2 + 2 * n_poles]
    small_velocity = 0.5 * (
        1.0
        + tolerance_tensor(rates, margin=reward["velocity_margin"]).amin(dim=-1, keepdim=True)
    )
    return upright * small_control * small_velocity * centered


def cartpole_sparse_reward_rate_tensor(
    states: Tensor, torques: Tensor, config: TestbedConfig
) -> Tensor:
    """dm_control's sparse reward 1[|x| <= cart_bound] prod_k 1[cos phi_k >= cosine_bound], [B,1]."""
    reward = config["reward_parameters"]
    n_poles = _pole_count(config)
    bound = float(reward["cart_bound"])
    centered = tolerance_tensor(states[..., :1], bounds=(-bound, bound))
    cosines = torch.cos(_absolute_angles_tensor(states, n_poles))
    upright = tolerance_tensor(cosines, bounds=(float(reward["cosine_bound"]), 1.0)).prod(
        dim=-1, keepdim=True
    )
    return centered * upright


def _lyapunov_cart_chain(config: TestbedConfig) -> PlanarChain:
    """cartpole_chain of the config, kept in _LYAPUNOV_CHAINS by pole count and plant."""
    n_poles = _pole_count(config)
    key = (
        n_poles,
        tuple(sorted((str(name), float(value)) for name, value in config["plant_parameters"].items())),
    )
    chain = _LYAPUNOV_CHAINS.get(key)
    if chain is None:
        chain = _LYAPUNOV_CHAINS[key] = cartpole_chain(config["plant_parameters"], n_poles)
    return chain


def cartpole_lyapunov_ratio_tensor(states: Tensor, config: TestbedConfig) -> Tensor:
    """cartpole_lyapunov_ratio on tensor batches, [B,1]."""
    reward = config["reward_parameters"]
    n_poles = _pole_count(config)
    nv = 1 + n_poles
    energy = chain_energy(_lyapunov_cart_chain(config), states[..., :nv], states[..., nv : 2 * nv])
    lyapunov = (
        0.5 * (energy - float(reward["upright_energy"])).square()
        + 0.5 * float(reward["lyapunov_k_cart"]) * states[..., :1].square()
    )
    if "lyapunov_k_straight" in reward and n_poles > 1:
        bends = states[..., 2 : 1 + n_poles]
        wrapped = torch.atan2(torch.sin(bends), torch.cos(bends))
        lyapunov = lyapunov + 0.5 * float(reward["lyapunov_k_straight"]) * wrapped.square().sum(
            dim=-1, keepdim=True
        )
    return lyapunov / (0.5 * float(reward["energy_scale"]) ** 2)


def cartpole_lyapunov_reward_rate_tensor(
    states: Tensor, torques: Tensor, config: TestbedConfig
) -> Tensor:
    """-log(clamp(V / V_down, reward_floor)) on tensor batches, [B,1]; state-only, no
    gradient where the floor binds."""
    floor = float(config["reward_parameters"]["reward_floor"])
    return -torch.log(torch.clamp(cartpole_lyapunov_ratio_tensor(states, config), min=floor))


_LAW_REWARD_TENSORS = {
    "smooth": cartpole_smooth_reward_rate_tensor,
    "sparse": cartpole_sparse_reward_rate_tensor,
    "lyapunov_cart": cartpole_lyapunov_reward_rate_tensor,
    "lyapunov_cart_straight": cartpole_lyapunov_reward_rate_tensor,
}


def cartpole_task_reward_rate_tensor(
    states: Tensor, torques: Tensor, config: TestbedConfig
) -> Tensor:
    """env.reward_rate on tensor batches, [B,1], for the config's cartpole_reward_law: the
    smooth or sparse reward of the task, or a Lyapunov reward."""
    return _LAW_REWARD_TENSORS[cartpole_reward_law(config)](states, torques, config)


def cartpole_task_reward_rate_bounds(config: TestbedConfig) -> Optional[Tuple[float, float]]:
    """(0, 1) for the smooth and sparse rewards (unit_interval_bounds), None for a Lyapunov
    reward, unbounded below (a linear reward head)."""
    if cartpole_reward_law(config) in LYAPUNOV_LAWS:
        return None
    return unit_interval_bounds(config)


def cartpole_task_oracle_callbacks(
    config: TestbedConfig,
) -> Tuple[Callable[[Tensor, Tensor], Tensor], Callable[[Tensor], Tensor]]:
    """_cartpole_callbacks (closed form) for one pole, the chain for several, each with the
    compiled model's soft rail limit (limited_chain.limited_chain_callbacks)."""
    n_poles = _pole_count(config)
    plant = config["plant_parameters"]
    limits = cartpole_rail_limits(build_cartpole_poles_model(config))
    if n_poles == 1:
        return _cartpole_callbacks(plant, limits)
    return limited_chain_callbacks(cartpole_chain(plant, n_poles), limits)


def cartpole_poles_upright(config: TestbedConfig, state: ndarray, tube: float) -> bool:
    """Every pole's absolute angle theta_1 + ... + theta_k within tube of upright."""
    angle = 0.0
    for index in range(1, 1 + _pole_count(config)):
        angle += float(state[index])
        if abs(_wrap(angle)) > tube:
            return False
    return True


# Every one of the five tasks dispatches on cartpole_reward_law: its own law (smooth or
# sparse, CARTPOLE_TASKS) unless the reward_parameters select a Lyapunov reward. All five
# take the testbed_settings key discount_rate (_common.py, with_discount_rate).
for _name, _factory in (
    ("cartpole_balance", cartpole_balance_config),
    ("cartpole_balance_sparse", cartpole_balance_sparse_config),
    ("cartpole_swingup_sparse", cartpole_swingup_sparse_config),
    ("cartpole_two_poles", cartpole_two_poles_config),
    ("cartpole_three_poles", cartpole_three_poles_config),
):
    register_testbed(
        TestbedSpec(
            name=_name,
            env_class=CartpoleTaskEnv,
            config_factory=with_clock_settings(_factory),
            settings_keys=SETTINGS_KEYS,
            optional_settings_keys=CLOCK_SETTINGS_KEYS,
            reward_rate_tensor=cartpole_task_reward_rate_tensor,
            failure_mask_tensor=cartpole_failure_mask_tensor,
            terminal_payoff_value=failure_rate_payoff,
            terminal_boundary_functions=cartpole_terminal_boundary_functions,
            oracle_callbacks=cartpole_task_oracle_callbacks,
            upright=cartpole_poles_upright,
            reward_rate_bounds=cartpole_task_reward_rate_bounds,
            discount_setting=True,
        )
    )

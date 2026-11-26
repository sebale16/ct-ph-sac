"""dm_control's acrobot as acrobot_swingup (with Xin and Kaneda's Lyapunov reward) and
acrobot_swingup_sparse (dm_control's swingup_sparse).

dm_control (suite/acrobot.xml, acrobot.py): shoulder fixed at height 2 m, two links
along +z of length 1 m and mass 1 kg (capsules fromto 0..1, radii 0.05 and 0.049, so
centres of mass 0.5 m along each link and compiled inertias about them 0.09557 and
0.09530 about the hinge axis), both hinges about +y, damping 0.05 on both, gravity 9.81,
constraint solver disabled; only the elbow is actuated, gear 2 with control in [-1, 1]
(torque +-2 N m); RK4 at 0.01 s, one substep per control step. Reset: both angles
uniform on [-pi, pi), velocities 0. The target site (radius 0.2) is at height 4 m, where
the tip sits when both links point up. Reward: tolerance(|tip - target|, (0, 0.2),
margin) with gaussian margin 1 (swingup) or margin 0, the indicator (swingup_sparse).
Episodes of 10 s. This is NOT the in-house Acrobot-XK (environment/ph_sac/tasks.py),
which has other links, reward and failure set.

The port (table rows acrobot-swingup, acrobot-swingup_sparse; factories in
common/ph_dmc/acrobot.py):
  state       (theta_1, theta_2, theta_dot_1, theta_dot_2): the shoulder from upright and
              the elbow relative to the upper arm (dm_control's qpos), both periodic and
              wrapped, the elbow actuated; no context.
  settings    plant_parameters l1, l2, m1, m2, radius1, radius2, shoulder_damping,
              elbow_damping and optionally gravity (default dm_control's 9.81; the row has
              no key for it); reward_parameters LYAPUNOV_REWARD_KEYS (acrobot-swingup),
              LYAPUNOV_SHOULDER_KEYS (acrobot-swingup, mode lyapunov_shoulder),
              TOLERANCE_REWARD_KEYS (acrobot-swingup_sparse) or ZHANG_REWARD_KEYS (no row
              since 2026-11-08), see "reward";
              reset_parameters angle_range, velocity_noise; failure_limits {} (no
              failure set).
  model       dm_control's geometry from plant_parameters: capsule geoms carrying the
              link masses, so MuJoCo compiles dm_control's centres of mass and inertias;
              the elbow motor in N m (gear 1, ctrlrange the torque limits); contact and
              constraints disabled; timestep and integrator from the config (1 ms RK4).
              The target site (rendering only) has radius target_radius, or dm_control's
              0.2 when the reward has none.
  reward      chosen by the reward_parameters' keys (acrobot_reward_kind; the keys of two
              rewards raise ValueError; lyapunov_k_shoulder turns "lyapunov" into
              "lyapunov_shoulder"):
              "lyapunov" (LYAPUNOV_REWARD_KEYS energy_scale, upright_energy, lyapunov_k_p,
                reward_floor; acrobot-swingup since the second change of 2026-11-08): the
                log-reciprocal of Xin and Kaneda's energy-and-elbow Lyapunov function
                without its elbow-rate term (X. Xin and M. Kaneda, "Analysis of the
                energy-based swing-up control of the Acrobot", Int. J. Robust Nonlinear
                Control 17(16):1503-1524, 2007, doi:10.1002/rnc.1184), the reward r1 of the
                in-house Acrobot-XK (environment/ph_sac/tasks.py) on this plant:
                  V(s) = (E(s) - upright_energy)^2 / 2 + lyapunov_k_p wrap(theta_2)^2 / 2,
                  r(s) = -log(max(V(s) / V_down, reward_floor)),
                  V_down = energy_scale^2 / 2,
                with E the mechanical energy of the PlanarChain the oracle uses
                (chain.chain_energy: kinetic plus g sum_j h_j cos phi_j, zero with both
                links horizontal, so g h_1 = b1 = g (m1 lc1 + m2 l1) and g h_2 = b2 =
                g m2 lc2 in Xin and Kaneda's notation). The constants are this plant's:
                upright_energy = E at upright rest = b1 + b2 = 19.62 J, energy_scale =
                E(upright rest) - E(hanging rest) = 2 (b1 + b2) = 39.24 J (dm_control's
                g 9.81, 1 kg and 1 m links with centres of mass at 0.5 m: b1 = 14.715 N m,
                b2 = 4.905 N m), lyapunov_k_p = LYAPUNOV_K_P_FACTOR (1.001) x Xin and
                Kaneda's lower bound (2/pi) min(b1^2, b2^2) = 15.3165 for this plant, so
                15.3318 (Acrobot-XK's 61.2 against its bound 61.141 is the same ratio),
                and reward_floor 1e-6 as Acrobot-XK. benchmarks/make_ct_ph_sac_table.py
                computes them through the chain (acrobot_lyapunov_reward_terms) and writes
                them into the row; the environment does not recompute them and checks
                only that they are positive and finite, with reward_floor in (0, 1). The
                elbow is wrapped, |theta_2| <= pi, because the port's state is periodic
                (Acrobot-XK's elbow is unwrapped and its failure set caps it at 4 pi); the
                elbow term is continuous across the folded elbow theta_2 = +-pi but not
                differentiable there. State-only: 0 at hanging rest (V = V_down) and at
                most -log(reward_floor) = 13.8155, reached where V <= reward_floor V_down
                (the upright at rest and a small neighbourhood); unbounded below, but only
                logarithmically in the energy, so reward_rate_bounds stays None (a linear
                reward head) and no failure set is added. The tensor version carries a
                graph except where the floor binds. Design intent: V = 0 on the whole set
                {E = upright_energy, theta_2 = 0}, on which the straight arm moves as one
                pendulum on its homoclinic orbit toward the upright; the reward asks for
                that orbit (from which an LQR can catch the arm), not for stabilization,
                and the unactuated shoulder theta_1 carries no penalty.
              "lyapunov_shoulder" (LYAPUNOV_SHOULDER_KEYS: LYAPUNOV_REWARD_KEYS plus
                lyapunov_k_shoulder; the table row acrobot-swingup of mode
                lyapunov_shoulder, an optional variant, no default row): the Lyapunov
                reward with a penalty on the unactuated shoulder,
                  V_C(s) = V(s) + lyapunov_k_shoulder wrap(theta_1)^2 / 2,
                  r(s) = -log(max(V_C(s) / V_C_down, reward_floor)),
                  V_C_down = energy_scale^2 / 2 + lyapunov_k_shoulder pi^2 / 2,
                V_C at hanging rest (theta_1 = pi, theta_2 = 0, at rest), so r = 0 there as
                for "lyapunov". V_C = 0 only at the upright at rest: this variant asks for
                stabilization, not for the orbit. No theory fixes the weight (Xin and
                Kaneda's bound covers only the elbow's k_P); the scale rule
                (acrobot_lyapunov_shoulder_reward_terms) gives theta_1 = pi alone the cost
                of the energy term at hanging rest, lyapunov_k_shoulder pi^2 / 2 =
                energy_scale^2 / 2, so lyapunov_k_shoulder = energy_scale^2 / pi^2 =
                156.012 here and V_C_down = energy_scale^2. The shoulder term is continuous
                across theta_1 = +-pi but not differentiable there; the other choices
                (wrapped elbow, no failure set, linear reward head) are "lyapunov"'s.
              "tolerance" (TOLERANCE_REWARD_KEYS target_radius, margin; acrobot-
                swingup_sparse, dm_control's reward): the tip relative to the shoulder
                is l1 (sin theta_1, cos theta_1) + l2 (sin phi_2, cos phi_2),
                phi_2 = theta_1 + theta_2, in (x, z) (dm_control's horizontal xz and
                vertical zz of the link frames); the target is the upright tip
                (0, l1 + l2); r = tolerance(distance, (0, target_radius), margin) (a
                positive margin takes dm_control's default gaussian shape, which no row
                uses now). State-only, in [0, 1].
              "zhang2023" (ZHANG_REWARD_KEYS; acrobot-swingup's reward from the first
                change of 2026-11-08 to the second, kept so that the runs which used it
                still load; no row carries it now): Zhang, Sathuluri and Zimmermann,
                "Solving the swing-up and balance task for the Acrobot and Pendubot with
                SAC", arXiv:2312.11311 (2023), Eq. 4 with Table I's acrobot constants. It
                replaced dm_control's swingup reward (whose tolerance is below 6e-4
                whenever the tip is more than 2 m from the target, so no arm learned from
                it) and was itself replaced because every curve stayed flat near a mean
                rate of -0.085 with no upright in 40 runs. With e = (wrap theta_1, wrap
                theta_2, theta_dot_1, theta_dot_2), the error from the upright
                x_g = (0, 0, 0, 0), h = l1 cos theta_1 + l2 cos(theta_1 + theta_2) the tip
                height above the shoulder and u the elbow torque in N m,
                  r = scale * ( - q_angle (e_1^2 + e_2^2) - q_velocity (e_3^2 + e_4^2)
                                - r_torque u^2
                                + line_bonus 1[h >= line_fraction (l1 + l2)]
                                + lqr_bonus 1[e^T S e <= rho]
                                - velocity_penalty (1[|e_3| >= velocity_threshold]
                                                    + 1[|e_4| >= velocity_threshold]) ),
                with the paper's q_angle 10, q_velocity 0.2, r_torque 1e-4, line_bonus
                500, line_fraction 0.8, lqr_bonus 1e4, velocity_penalty 1e4,
                velocity_threshold 8 rad/s. Choices (the paper leaves them open): the
                paper's coordinates put 0 hanging, ours 0 upright, so x_g is the origin;
                the angle errors are wrapped (the state is periodic, and the paper's
                observation is wrapped too); the region-of-attraction indicator is
                e^T S e <= rho (the paper prints >= but pays the bonus inside the
                region); the paper's per-step reward is read as the reward rate (per
                second), as the port reads dm_control's per-step reward, so a
                1000-step return is 1000 x the mean rate; scale 1e-4 makes lqr_bonus 1
                (a positive factor changes no optimal policy because the temperature is
                tuned to an entropy target). S and rho (s11, s12, ..., s44 and rho) are
                this plant's: S is the LQR cost-to-go of the upright for the paper's
                Q_lqr = diag(0.97, 0.93, 0.39, 0.26), R_lqr = 0.11 and rho a simulated
                region of attraction of the saturated LQR (+-2 N m), both computed by
                environment/ph_sac/dmc/acrobot_lqr.py (the values the row carried are in
                common/ph_dmc/acrobot.py), because the paper's sums-of-squares values are
                for its own robot and not given. Unbounded (the velocity terms), so
                reward_rate_bounds stays None. The indicators are piecewise constant (no
                graph in the tensor version).
  failure     none: an all-False mask, payoff 0, lift H = 0, B = 1.
  reset       both angles uniform on [-angle_range, angle_range), velocities
              velocity_noise N(0,1) (0 in the rows); options={"state": s} sets s.
  oracle      chain.chain_callbacks of the two-link PlanarChain read from the compiled
              model (_common.planar_chain_from_model: compiled masses, centres of mass,
              inertias, damping, gravity, G = [[0], [1]] in N m); zero diffusion.
  upright     both links' absolute angles theta_1 and theta_1 + theta_2 within tube of
              upright (README.md's "every link's absolute angle").
  change      acrobot_swingup only, the adaptation experiment's testbed_settings key
              physical_change (common.ph_dmc.acrobot.acrobot_physical_change): MuJoCo and
              the oracle simulate physical_plant_parameters, the lower link's length l2
              (capsule, centre of mass, inertia and tip site from the geometry, mass m2
              kept) or both joints' damping times the scale; every reward keeps the
              nominal plant_parameters (the Lyapunov rewards the nominal chain's energy and
              constants, the tolerance and Zhang rewards the nominal lengths), so it stays
              the task's function of the state; upright rest still has V = 0.

Deviations from dm_control: acrobot-swingup's reward is Xin and Kaneda's Lyapunov reward,
not dm_control's (acrobot-swingup_sparse keeps dm_control's indicator); 1 ms RK4 physics
instead of 10 ms RK4 (as every dm_control row of the table), with the decision clock at
dm_control's 10 ms; contact is disabled, which changes nothing because dm_control's
constraint solver is off, so its floor contacts exert no force.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Callable, Dict, Literal, Mapping, Optional, Tuple

import mujoco
import numpy as np
import torch

from common.ph_dmc.acrobot import (
    acrobot_physical_change,
    acrobot_swingup_config,
    acrobot_swingup_sparse_config,
)

from ..base import MujocoPHEnv, validated_state_spec
from ..registry import TestbedSpec, register_testbed
# The irregular clocks (optional testbed_settings, since 2026-11-11; _common.py).
from common.ph_dmc._common import CLOCK_SETTINGS_KEYS, with_clock_settings  # noqa: E402
# The adaptation experiment's changed robot (_common.py, with_physical_change).
from common.ph_dmc._common import nominal_testbed, physical_plant_parameters  # noqa: E402
from ._common import (
    failure_rate_payoff,
    limit_boundary_functions,
    limit_failed,
    limit_failure_mask,
    planar_chain_from_model,
    require_keys,
    tolerance,
    tolerance_tensor,
)
from .acrobot_lqr import cost_to_go_matrix, wrapped_errors
from .chain import (
    PlanarChain,
    chain_callbacks,
    chain_drift_numpy,
    chain_energy,
    chain_energy_numpy,
)

if TYPE_CHECKING:
    from mujoco import MjModel
    from numpy import ndarray
    from torch import Tensor

    from common.ph_config import TestbedConfig

SETTINGS_KEYS = ("failure_limits", "reward_parameters", "reset_parameters", "plant_parameters")
PLANT_KEYS = (
    "l1", "l2", "m1", "m2", "radius1", "radius2", "shoulder_damping", "elbow_damping",
)
# acrobot_swingup: Xin and Kaneda's Lyapunov reward (module docstring), in Acrobot-XK's order.
LYAPUNOV_REWARD_KEYS = ("energy_scale", "upright_energy", "lyapunov_k_p", "reward_floor")
# The optional variant "lyapunov_shoulder" (row acrobot-swingup, mode lyapunov_shoulder):
# the Lyapunov reward plus a penalty on the unactuated shoulder (module docstring).
LYAPUNOV_SHOULDER_KEYS = (
    "energy_scale", "upright_energy", "lyapunov_k_p", "lyapunov_k_shoulder", "reward_floor",
)
# The reward kinds whose energy function is a Lyapunov function V (lyapunov_ratio reports
# V / V_down for them).
LYAPUNOV_KINDS = ("lyapunov", "lyapunov_shoulder")
# No row since 2026-11-08: Zhang et al. (2023)'s reward (module docstring); the paper's
# constants, then rho and the upper triangle of S row by row (acrobot_lqr.COST_TO_GO_KEYS).
ZHANG_REWARD_KEYS = (
    "q_angle", "q_velocity", "r_torque", "line_bonus", "line_fraction", "lqr_bonus",
    "velocity_penalty", "velocity_threshold", "scale", "rho",
    "s11", "s12", "s13", "s14", "s22", "s23", "s24", "s33", "s34", "s44",
)
# acrobot_swingup_sparse: dm_control's tolerance reward.
TOLERANCE_REWARD_KEYS = ("target_radius", "margin")
_REWARD_KEYS = {
    "lyapunov": LYAPUNOV_REWARD_KEYS,
    "lyapunov_shoulder": LYAPUNOV_SHOULDER_KEYS,
    "zhang2023": ZHANG_REWARD_KEYS,
    "tolerance": TOLERANCE_REWARD_KEYS,
}
# Families of rewards whose keys must not mix; the Lyapunov family's extra key
# lyapunov_k_shoulder selects its variant "lyapunov_shoulder".
_REWARD_FAMILIES = {
    "lyapunov": LYAPUNOV_SHOULDER_KEYS,
    "zhang2023": ZHANG_REWARD_KEYS,
    "tolerance": TOLERANCE_REWARD_KEYS,
}
# lyapunov_k_p over Xin and Kaneda's lower bound (2/pi) min(b1^2, b2^2), the ratio of
# Acrobot-XK's 61.2 to its bound 61.141.
LYAPUNOV_K_P_FACTOR = 1.001
RESET_KEYS = ("angle_range", "velocity_noise")
LINK_BODIES = ("upper_arm", "lower_arm")
# dm_control's default gravity; plant_parameters["gravity"] overrides it.
DEFAULT_GRAVITY = 9.81
# No joint stop and no failure set: (state index, limit) pairs for the _common helpers.
_LIMITS: Tuple[Tuple[int, float], ...] = ()
# dm_control's shoulder height (rendering only; the shoulder is fixed).
_SHOULDER_HEIGHT = 2.0
# dm_control's target site radius, for rendering when the reward has no target_radius.
_TARGET_SITE_RADIUS = 0.2
# Floor on the squared tip distance before its square root, far inside any target ball:
# it changes no reward and keeps the gradient finite at the target itself.
_SQUARED_DISTANCE_FLOOR = 1e-24
# The tensor Lyapunov reward's chains by plant_parameters, on which alone the energy
# depends: compiling the model for every call would cost about a millisecond.
_LYAPUNOV_CHAINS: Dict[Tuple[Tuple[str, float], ...], PlanarChain] = {}

# dm_control's acrobot with the elbow motor in N m (gear 1). The links rest along +z and
# the hinges turn about +y, so link i points along (sin phi_i, cos phi_i) in (x, z).
_ACROBOT_XML = """
<mujoco model="acrobot-swingup">
  <option timestep="{timestep}" integrator="{integrator}" gravity="0 0 -{gravity}">
    <flag contact="disable" constraint="disable"/>
  </option>
  <visual><global offwidth="1280" offheight="960"/></visual>
  <default>
    <joint type="hinge" axis="0 1 0" limited="false"/>
    <geom type="capsule"/>
  </default>
  <worldbody>
    <light name="light" pos="0 0 6"/>
    <site name="target" type="sphere" pos="0 0 {target_height}" size="{target_radius}"
          rgba="0.6 0.3 0.3 0.4"/>
    <camera name="fixed" pos="0 -6 2" zaxis="0 -1 0"/>
    <body name="upper_arm" pos="0 0 {shoulder_height}">
      <joint name="shoulder" damping="{shoulder_damping}"/>
      <geom name="upper_arm_decoration" type="cylinder" fromto="0 -.06 0 0 .06 0"
            size="0.051" mass="0"/>
      <geom name="upper_arm" fromto="0 0 0 0 0 {l1}" size="{radius1}" mass="{m1}"/>
      <body name="lower_arm" pos="0 0 {l1}">
        <joint name="elbow" damping="{elbow_damping}"/>
        <geom name="lower_arm" fromto="0 0 0 0 0 {l2}" size="{radius2}" mass="{m2}"/>
        <site name="tip" pos="0 0 {l2}" size="0.01"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor name="elbow" joint="elbow" gear="1" ctrllimited="true"
           ctrlrange="{torque_low} {torque_high}"/>
  </actuator>
</mujoco>
"""


def _wrap(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def _gravity(plant: Mapping[str, float]) -> float:
    return float(plant.get("gravity", DEFAULT_GRAVITY))


def build_acrobot_model(config: TestbedConfig) -> MjModel:
    """dm_control's acrobot with the config's links, timestep, integrator and motor limits.

    The links are the physical robot's (physical_plant_parameters: a physical change's
    when the config carries one); the target site, drawn only, sits at its upright tip.
    """
    plant = physical_plant_parameters(config)
    controller = config["controller"]
    return mujoco.MjModel.from_xml_string(
        _ACROBOT_XML.format(
            timestep=config["physics_dt"],
            integrator=config["integrator"],
            gravity=_gravity(plant),
            shoulder_height=_SHOULDER_HEIGHT,
            target_height=_SHOULDER_HEIGHT + plant["l1"] + plant["l2"],
            target_radius=config["reward_parameters"].get("target_radius", _TARGET_SITE_RADIUS),
            l1=plant["l1"],
            l2=plant["l2"],
            m1=plant["m1"],
            m2=plant["m2"],
            radius1=plant["radius1"],
            radius2=plant["radius2"],
            shoulder_damping=plant["shoulder_damping"],
            elbow_damping=plant["elbow_damping"],
            torque_low=float(controller["torque_low"][0]),
            torque_high=float(controller["torque_high"][0]),
        )
    )


def acrobot_chain_from_model(model: object, config: TestbedConfig) -> PlanarChain:
    """The two-link PlanarChain of a compiled acrobot model (ours, or dm_control's physics)."""
    return planar_chain_from_model(
        model,
        LINK_BODIES,
        along=2,
        joint_axis=1,
        last_length=float(physical_plant_parameters(config)["l2"]),
    )


def acrobot_chain(config: TestbedConfig) -> PlanarChain:
    """The two-link PlanarChain of the config's own compiled model (input in N m): the
    physical robot's, a physical change's when the config carries one."""
    return acrobot_chain_from_model(build_acrobot_model(config), config)


def acrobot_reward_chain(config: TestbedConfig) -> PlanarChain:
    """The PlanarChain whose energy the Lyapunov rewards read: the nominal robot's
    (plant_parameters), whatever physical change the config carries, so the reward stays
    the task's function of the state. acrobot_chain(config) for a config without one."""
    return acrobot_chain(nominal_testbed(config))


def _tip_offset(plant: Mapping[str, float], shoulder: float, elbow: float) -> Tuple[float, float]:
    """The tip minus the target (the upright tip) in (x, z)."""
    l1, l2 = float(plant["l1"]), float(plant["l2"])
    absolute = shoulder + elbow
    return (
        l1 * math.sin(shoulder) + l2 * math.sin(absolute),
        l1 * math.cos(shoulder) + l2 * math.cos(absolute) - (l1 + l2),
    )


def acrobot_reward_kind(reward: Mapping[str, object]) -> str:
    """The reward whose keys the reward_parameters carry.

    "lyapunov" for any of LYAPUNOV_REWARD_KEYS, "lyapunov_shoulder" when lyapunov_k_shoulder
    is among them, "zhang2023" for any of ZHANG_REWARD_KEYS, otherwise "tolerance" (also
    with none of the keys). Keys of two rewards raise ValueError; the missing keys of the
    reward chosen are reported by the environment.
    """
    present = {
        kind: [key for key in keys if key in reward] for kind, keys in _REWARD_FAMILIES.items()
    }
    kinds = [kind for kind, keys in present.items() if keys]
    if len(kinds) > 1:
        mixed = "; ".join(f"{kind} {present[kind]}" for kind in kinds)
        raise ValueError(f"Acrobot reward_parameters mix the keys of several rewards: {mixed}.")
    if not kinds:
        return "tolerance"
    if kinds[0] == "lyapunov" and "lyapunov_k_shoulder" in reward:
        return "lyapunov_shoulder"
    return kinds[0]


def lyapunov_k_p_bound(chain: PlanarChain) -> float:
    """Xin and Kaneda's lower bound on k_P, (2/pi) min(b1^2, b2^2), for the chain's plant.

    (b1, b2) = g (h_1, h_2), the chain's gravity times its first moments.
    """
    b1, b2 = (chain.gravity * chain.first_moments).tolist()
    return 2.0 / math.pi * min(b1**2, b2**2)


def acrobot_lyapunov_reward_terms(config: TestbedConfig) -> Dict[str, float]:
    """energy_scale, upright_energy and lyapunov_k_p of the config's (nominal) plant, through
    its chain (acrobot_reward_chain).

    upright_energy is chain_energy at upright rest (both angles 0), energy_scale that
    minus chain_energy at hanging rest (theta_1 = pi, theta_2 = 0), lyapunov_k_p
    LYAPUNOV_K_P_FACTOR times lyapunov_k_p_bound. benchmarks/make_ct_ph_sac_table.py writes
    them into the row; the environment reads them from there and does not call this.
    """
    chain = acrobot_reward_chain(config)
    rest = np.zeros(2)
    upright = float(chain_energy_numpy(chain, rest, rest)[0])
    hanging = float(chain_energy_numpy(chain, np.array([math.pi, 0.0]), rest)[0])
    return {
        "energy_scale": upright - hanging,
        "upright_energy": upright,
        "lyapunov_k_p": LYAPUNOV_K_P_FACTOR * lyapunov_k_p_bound(chain),
    }


def acrobot_shoulder_weight(energy_scale: float) -> float:
    """The scale rule of lyapunov_k_shoulder: theta_1 = pi alone costs V_down.

    k pi^2 / 2 = energy_scale^2 / 2, so k = energy_scale^2 / pi^2 (no theory fixes it).
    """
    return float(energy_scale) ** 2 / math.pi**2


def acrobot_lyapunov_shoulder_reward_terms(config: TestbedConfig) -> Dict[str, float]:
    """acrobot_lyapunov_reward_terms plus lyapunov_k_shoulder by acrobot_shoulder_weight.

    In LYAPUNOV_SHOULDER_KEYS' order (without reward_floor); benchmarks/
    make_ct_ph_sac_table.py writes them into the row acrobot-swingup of mode
    lyapunov_shoulder.
    """
    terms = acrobot_lyapunov_reward_terms(config)
    return {
        "energy_scale": terms["energy_scale"],
        "upright_energy": terms["upright_energy"],
        "lyapunov_k_p": terms["lyapunov_k_p"],
        "lyapunov_k_shoulder": acrobot_shoulder_weight(terms["energy_scale"]),
    }


def _validate_lyapunov(reward: Mapping[str, float]) -> None:
    """energy_scale, upright_energy, lyapunov_k_p (and lyapunov_k_shoulder when present)
    positive and finite; reward_floor in (0, 1)."""
    weights = ("energy_scale", "upright_energy", "lyapunov_k_p")
    if "lyapunov_k_shoulder" in reward:
        weights += ("lyapunov_k_shoulder",)
    for key in weights:
        value = float(reward[key])
        if not (math.isfinite(value) and value > 0.0):
            raise ValueError(f"Acrobot reward_parameters {key} must be positive and finite.")
    if not 0.0 < float(reward["reward_floor"]) < 1.0:
        raise ValueError("Acrobot reward_parameters reward_floor must lie in (0, 1).")


def acrobot_lyapunov_ratio(
    config: TestbedConfig, state: ndarray, chain: Optional[PlanarChain] = None
) -> float:
    """V / V_down of the "lyapunov" or "lyapunov_shoulder" reward (module docstring).

    V = (E - upright_energy)^2 / 2 + lyapunov_k_p wrap(theta_2)^2 / 2 with E the chain's
    mechanical energy and V_down = energy_scale^2 / 2; with lyapunov_k_shoulder in the
    reward_parameters both gain the shoulder term, + lyapunov_k_shoulder wrap(theta_1)^2 / 2
    and + lyapunov_k_shoulder pi^2 / 2. 1 at hanging rest, 0 on the reward's zero set.
    chain is the nominal plant's PlanarChain (acrobot_reward_chain(config) when None).
    """
    reward = config["reward_parameters"]
    if chain is None:
        chain = acrobot_reward_chain(config)
    state = np.asarray(state, dtype=np.float64)
    energy = float(chain_energy_numpy(chain, state[:2], state[2:4])[0])
    elbow = _wrap(float(state[1]))
    lyapunov = (
        0.5 * (energy - float(reward["upright_energy"])) ** 2
        + 0.5 * float(reward["lyapunov_k_p"]) * elbow**2
    )
    down = 0.5 * float(reward["energy_scale"]) ** 2
    if "lyapunov_k_shoulder" in reward:
        weight = float(reward["lyapunov_k_shoulder"])
        lyapunov = lyapunov + 0.5 * weight * _wrap(float(state[0])) ** 2
        down = down + 0.5 * weight * math.pi**2
    return lyapunov / down


def acrobot_lyapunov_reward_rate(
    config: TestbedConfig, state: ndarray, chain: Optional[PlanarChain] = None
) -> float:
    """Xin and Kaneda's Lyapunov reward -log(max(V / V_down, reward_floor)) (module docstring).

    V / V_down is acrobot_lyapunov_ratio's, with the shoulder term for "lyapunov_shoulder";
    state-only. chain is the nominal plant's PlanarChain (acrobot_reward_chain(config)
    when None).
    """
    ratio = acrobot_lyapunov_ratio(config, state, chain)
    return -math.log(max(ratio, float(config["reward_parameters"]["reward_floor"])))


def _validated_cost_to_go(reward: Mapping[str, float]) -> ndarray:
    """S from s11, ..., s44, checked positive definite, with rho > 0."""
    cost_to_go = cost_to_go_matrix(reward)
    if not np.all(np.isfinite(cost_to_go)) or np.min(np.linalg.eigvalsh(cost_to_go)) <= 0.0:
        raise ValueError("Acrobot reward_parameters s11, ..., s44 must form a positive-definite S.")
    if not float(reward["rho"]) > 0.0:
        raise ValueError("Acrobot reward_parameters rho must be positive.")
    return cost_to_go


def acrobot_zhang_reward_rate(
    config: TestbedConfig, state: ndarray, torque: ndarray, cost_to_go: Optional[ndarray] = None
) -> float:
    """Zhang et al. (2023)'s reward (Eq. 4) times scale, as a rate (module docstring).

    e = wrapped_errors(state), h = l1 cos theta_1 + l2 cos(theta_1 + theta_2), u the elbow
    torque in N m; cost_to_go is S (built from the reward_parameters when None).
    """
    plant, reward = config["plant_parameters"], config["reward_parameters"]
    if cost_to_go is None:
        cost_to_go = cost_to_go_matrix(reward)
    state = np.asarray(state, dtype=np.float64)
    error = wrapped_errors(state[:4])
    torque = np.asarray(torque, dtype=np.float64).reshape(-1)
    l1, l2 = float(plant["l1"]), float(plant["l2"])
    shoulder = float(state[0])
    height = l1 * math.cos(shoulder) + l2 * math.cos(shoulder + float(state[1]))
    threshold = float(reward["velocity_threshold"])
    fast = float(abs(error[2]) >= threshold) + float(abs(error[3]) >= threshold)
    inside = float(error @ cost_to_go @ error) <= float(reward["rho"])
    on_line = height >= float(reward["line_fraction"]) * (l1 + l2)
    value = (
        -float(reward["q_angle"]) * (error[0] ** 2 + error[1] ** 2)
        - float(reward["q_velocity"]) * (error[2] ** 2 + error[3] ** 2)
        - float(reward["r_torque"]) * float(np.sum(torque**2))
        + float(reward["line_bonus"]) * float(on_line)
        + float(reward["lqr_bonus"]) * float(inside)
        - float(reward["velocity_penalty"]) * fast
    )
    return float(reward["scale"]) * float(value)


class AcrobotSwingupEnv(MujocoPHEnv):
    """dm_control's acrobot swing-up with any of its rewards (acrobot_reward_kind of the config).

    The state is (theta_1, theta_2, theta_dot_1, theta_dot_2) with both angles 0 upright.
    Under a physical change (the adaptation experiment) MuJoCo and the oracle simulate the
    changed robot (physical_plant_parameters) while every reward reads the nominal
    plant_parameters: the Lyapunov rewards the nominal chain's energy (_reward_chain,
    acrobot_reward_chain), the tolerance and Zhang rewards the nominal link lengths.
    """

    def __init__(
        self,
        config: TestbedConfig,
        *,
        decision_mode: Literal["torque", "feedback"] = "feedback",
        seed: Optional[int] = None,
    ) -> None:
        require_keys(config["plant_parameters"], PLANT_KEYS, "Acrobot plant_parameters")
        reward = config["reward_parameters"]
        self._reward_kind = acrobot_reward_kind(reward)
        require_keys(reward, _REWARD_KEYS[self._reward_kind], "Acrobot reward_parameters")
        self._cost_to_go = (
            _validated_cost_to_go(reward) if self._reward_kind == "zhang2023" else None
        )
        if self._reward_kind in LYAPUNOV_KINDS:
            _validate_lyapunov(reward)
        require_keys(config["reset_parameters"], RESET_KEYS, "Acrobot reset_parameters")
        super().__init__(
            config,
            build_acrobot_model(config),
            validated_state_spec(config),
            decision_mode=decision_mode,
            seed=seed,
        )
        self._chain = acrobot_chain_from_model(self.model, config)
        # The chain the rewards read: the simulated one unless a physical change differs.
        self._reward_chain = (
            self._chain if nominal_testbed(config) is config else acrobot_reward_chain(config)
        )

    def _reset_state(self, options: Optional[Mapping[str, object]] = None) -> None:
        """dm_control's start: both angles uniform on +-angle_range, at rest."""
        if options and "state" in options:
            self._set_state(options["state"])
            return
        reset = self.config["reset_parameters"]
        bound = float(reset["angle_range"])
        angles = self._reset_rng.uniform(-bound, bound, 2)
        velocities = float(reset["velocity_noise"]) * self._reset_rng.standard_normal(2)
        self._set_state(np.concatenate((angles, velocities)))

    def tip_distance(self, state: ndarray) -> float:
        """Distance from the tip to the target (the upright tip), m."""
        dx, dz = _tip_offset(self.config["plant_parameters"], float(state[0]), float(state[1]))
        return math.hypot(dx, dz)

    @property
    def reward_kind(self) -> str:
        """"lyapunov" (acrobot_swingup), "tolerance" (acrobot_swingup_sparse),
        "lyapunov_shoulder" (the optional variant) or "zhang2023"."""
        return self._reward_kind

    def lyapunov_ratio(self, state: ndarray) -> Optional[float]:
        """V / V_down of a Lyapunov reward (acrobot_lyapunov_ratio), else None.

        For the evaluation's orbit diagnostics (evaluations/ph_sac/protocol.py).
        """
        if self._reward_kind not in LYAPUNOV_KINDS:
            return None
        return acrobot_lyapunov_ratio(self.config, state, self._reward_chain)

    def reward_rate(self, state: ndarray, torque: ndarray) -> float:
        """The config's reward; acrobot_reward_rate_tensor must stay its copy."""
        if self._reward_kind in LYAPUNOV_KINDS:
            return acrobot_lyapunov_reward_rate(self.config, state, self._reward_chain)
        if self._reward_kind == "zhang2023":
            return acrobot_zhang_reward_rate(self.config, state, torque, self._cost_to_go)
        reward = self.config["reward_parameters"]
        return tolerance(
            self.tip_distance(state), (0.0, float(reward["target_radius"])), float(reward["margin"])
        )

    def failed(self, state: ndarray) -> bool:
        """No failure set."""
        return limit_failed(state, _LIMITS)

    def terminal_payoff(self, state: ndarray) -> float:
        return failure_rate_payoff(self.config)

    def oracle_drift(self, state: ndarray, torque: ndarray) -> ndarray:
        return chain_drift_numpy(self._chain, state, np.asarray(torque, dtype=np.float64).reshape(-1))

    def oracle_diffusion(self, state: ndarray) -> ndarray:
        """The acrobot has no noise."""
        return np.zeros((self.state_dim, 1))


def _zhang_reward_rate_tensor(states: Tensor, torques: Tensor, config: TestbedConfig) -> Tensor:
    """acrobot_zhang_reward_rate on tensor batches, [B,1]; the indicators carry no graph."""
    plant, reward = config["plant_parameters"], config["reward_parameters"]
    l1, l2 = float(plant["l1"]), float(plant["l2"])
    angles = torch.atan2(torch.sin(states[..., :2]), torch.cos(states[..., :2]))
    rates = states[..., 2:4]
    error = torch.cat((angles, rates), dim=-1)
    cost_to_go = torch.as_tensor(
        cost_to_go_matrix(reward), dtype=states.dtype, device=states.device
    )
    shoulder = states[..., :1]
    height = l1 * torch.cos(shoulder) + l2 * torch.cos(shoulder + states[..., 1:2])
    inside = ((error @ cost_to_go) * error).sum(dim=-1, keepdim=True) <= float(reward["rho"])
    on_line = height >= float(reward["line_fraction"]) * (l1 + l2)
    threshold = float(reward["velocity_threshold"])
    fast = (rates.abs() >= threshold).to(states.dtype).sum(dim=-1, keepdim=True)
    torques = torques.to(states.dtype)
    value = (
        -float(reward["q_angle"]) * angles.square().sum(dim=-1, keepdim=True)
        - float(reward["q_velocity"]) * rates.square().sum(dim=-1, keepdim=True)
        - float(reward["r_torque"]) * torques.square().sum(dim=-1, keepdim=True)
        + float(reward["line_bonus"]) * on_line.to(states.dtype)
        + float(reward["lqr_bonus"]) * inside.to(states.dtype)
        - float(reward["velocity_penalty"]) * fast
    )
    return float(reward["scale"]) * value


def _lyapunov_chain(config: TestbedConfig) -> PlanarChain:
    """acrobot_reward_chain(config), the nominal robot's, kept in _LYAPUNOV_CHAINS by the
    config's (nominal) plant_parameters."""
    key = tuple(
        sorted((str(name), float(value)) for name, value in config["plant_parameters"].items())
    )
    chain = _LYAPUNOV_CHAINS.get(key)
    if chain is None:
        chain = _LYAPUNOV_CHAINS[key] = acrobot_reward_chain(config)
    return chain


def acrobot_lyapunov_ratio_tensor(states: Tensor, config: TestbedConfig) -> Tensor:
    """acrobot_lyapunov_ratio on tensor batches, [B,1] (the shoulder term with
    lyapunov_k_shoulder). The chain comes from _LYAPUNOV_CHAINS, compiled once per
    plant_parameters."""
    reward = config["reward_parameters"]
    energy = chain_energy(_lyapunov_chain(config), states[..., :2], states[..., 2:4])
    elbow = torch.atan2(torch.sin(states[..., 1:2]), torch.cos(states[..., 1:2]))
    lyapunov = (
        0.5 * (energy - float(reward["upright_energy"])).square()
        + 0.5 * float(reward["lyapunov_k_p"]) * elbow.square()
    )
    down = 0.5 * float(reward["energy_scale"]) ** 2
    if "lyapunov_k_shoulder" in reward:
        weight = float(reward["lyapunov_k_shoulder"])
        shoulder = torch.atan2(torch.sin(states[..., :1]), torch.cos(states[..., :1]))
        lyapunov = lyapunov + 0.5 * weight * shoulder.square()
        down = down + 0.5 * weight * math.pi**2
    return lyapunov / down


def _lyapunov_reward_rate_tensor(states: Tensor, config: TestbedConfig) -> Tensor:
    """acrobot_lyapunov_reward_rate on tensor batches, [B,1]; no gradient where the floor binds."""
    floor = float(config["reward_parameters"]["reward_floor"])
    return -torch.log(torch.clamp(acrobot_lyapunov_ratio_tensor(states, config), min=floor))


def acrobot_reward_rate_tensor(states: Tensor, torques: Tensor, config: TestbedConfig) -> Tensor:
    """env.reward_rate on tensor batches, [B,1], for the config's acrobot_reward_kind.

    "lyapunov": Xin and Kaneda's Lyapunov reward ("lyapunov_shoulder": with the shoulder
    term); "zhang2023": Zhang et al.'s reward; "tolerance": tolerance(|tip - target|,
    (0, target_radius), margin).
    """
    plant, reward = config["plant_parameters"], config["reward_parameters"]
    kind = acrobot_reward_kind(reward)
    if kind in LYAPUNOV_KINDS:
        return _lyapunov_reward_rate_tensor(states, config)
    if kind == "zhang2023":
        return _zhang_reward_rate_tensor(states, torques, config)
    l1, l2 = float(plant["l1"]), float(plant["l2"])
    shoulder = states[..., :1]
    absolute = shoulder + states[..., 1:2]
    dx = l1 * torch.sin(shoulder) + l2 * torch.sin(absolute)
    dz = l1 * torch.cos(shoulder) + l2 * torch.cos(absolute) - (l1 + l2)
    squared = torch.clamp(dx.square() + dz.square(), min=_SQUARED_DISTANCE_FLOOR)
    return tolerance_tensor(
        torch.sqrt(squared), (0.0, float(reward["target_radius"])), float(reward["margin"])
    )


def acrobot_failure_mask_tensor(states: Tensor, config: TestbedConfig) -> Tensor:
    """No failure set: all False, [B]."""
    return limit_failure_mask(states, _LIMITS)


def acrobot_terminal_boundary_functions(
    config: TestbedConfig,
) -> Tuple[Callable[[Tensor], Tensor], Callable[[Tensor], Tensor]]:
    """No failure set: H = the payoff (0), B = 1."""
    return limit_boundary_functions(failure_rate_payoff(config), _LIMITS)


def acrobot_oracle_callbacks(
    config: TestbedConfig,
) -> Tuple[Callable[[Tensor, Tensor], Tensor], Callable[[Tensor], Tensor]]:
    """chain.chain_callbacks of the two-link PlanarChain of the compiled model (the
    physical robot's, a physical change's when the config carries one)."""
    return chain_callbacks(acrobot_chain(config))


def acrobot_upright(config: TestbedConfig, state: ndarray, tube: float) -> bool:
    """Both links' absolute angles, theta_1 and theta_1 + theta_2, within tube of upright."""
    shoulder, elbow = float(state[0]), float(state[1])
    return abs(_wrap(shoulder)) <= tube and abs(_wrap(shoulder + elbow)) <= tube


# acrobot_swingup takes the adaptation experiment's physical change (the lower link's
# length or both joints' damping, acrobot_physical_change); acrobot_swingup_sparse does
# not: its tolerance target is the nominal upright tip, which a longer link would overshoot.
# Both take the testbed_settings key discount_rate (_common.py, with_discount_rate).
for _name, _factory, _change in (
    ("acrobot_swingup", acrobot_swingup_config, acrobot_physical_change),
    ("acrobot_swingup_sparse", acrobot_swingup_sparse_config, None),
):
    register_testbed(
        TestbedSpec(
            name=_name,
            env_class=AcrobotSwingupEnv,
            config_factory=with_clock_settings(_factory),
            settings_keys=SETTINGS_KEYS,
            optional_settings_keys=CLOCK_SETTINGS_KEYS,
            reward_rate_tensor=acrobot_reward_rate_tensor,
            failure_mask_tensor=acrobot_failure_mask_tensor,
            terminal_payoff_value=failure_rate_payoff,
            terminal_boundary_functions=acrobot_terminal_boundary_functions,
            oracle_callbacks=acrobot_oracle_callbacks,
            upright=acrobot_upright,
            physical_change=_change,
            discount_setting=True,
        )
    )

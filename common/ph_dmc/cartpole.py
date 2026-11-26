"""Config factories of the dm_control cart-pole ports.

Testbeds: cartpole_swingup, cartpole_balance, cartpole_balance_sparse,
cartpole_swingup_sparse, cartpole_two_poles, cartpole_three_poles.

cartpole_swingup_config moved here from common/ph_config.py, which re-exports it. Each
factory takes the testbed_settings of its table row (benchmarks/hyperparams/
ct_ph_sac.csv, the env_* cells) as keyword arguments and returns the TestbedConfig whose
fields the row's task_* cells declare. The five later tasks share the swing-up's cart,
motor (+-10 N), decision box (offset +-1 m, stiffness 0..100, damping 0..20), fixed 10 ms
clock, 1 ms RK4 physics, discount 0.5 /s and 10 s episodes; they differ in the reset, the
reward and the number of poles, which environment/ph_sac/dmc/cartpole.py reads from the
settings. The environment, the tensor task functions and the registration live there;
environment/ph_sac/dmc/README.md describes the contract. Import common.ph_config only under
TYPE_CHECKING here (it imports this package).

The slider keeps dm_control's soft rail limit (since 2026-11-10): limited to +-cart_range
(1.8 m) with solreflimit (0.08, 1) and MuJoCo's default solimplimit, all in
plant_parameters (cart_range and LIMIT_KEYS). No task has a failure set (failure_limits
must be empty; require_no_failure_set), and the state spec declares the rail as
limited_position_ranges ((0, -cart_range, cart_range),), read by the learned plant's
limit port (network limit_port, set by the rows), and since 2026-11-10 the port's start
limit_port_init, read from the compiled model (with_limit_port_init: k 160.5 N/m,
c 25.7 N s/m on one pole, 161.0 N/m with two or three).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Mapping, Tuple

import numpy as np

from common.ph_dmc._common import (
    DMC_REFRESH_MODE,
    LIMIT_KEYS,
    box_rule_controller,
    dmc_state_spec,
    dmc_testbed_config,
    require_no_failure_set,
    with_limit_port_init,
)

if TYPE_CHECKING:
    from common.ph_config import ClockConfig, ControllerConfig, TestbedConfig
    from common.ph_types import StateSpec

# The cart-pole tasks' shared declaration: dm_control's 10 ms control step, 10 s episodes,
# +-10 N on the cart and setpoints offset up to 1 m from the cart's position.
_REFERENCE_PERIOD = 0.01
_EPISODE_DURATION = 10.0
_FORCE_LIMIT = 10.0
_CART_OFFSET = 1.0
# Required plant_parameters keys of every cart-pole task (the multi-pole rows add n_poles):
# the cart, the poles, and dm_control's rail limit, the slider's range +-cart_range and
# its soft-limit parameters (solreflimit 0.08 1, MuJoCo's default solimplimit).
PLANT_KEYS: Tuple[str, ...] = (
    "cart_mass", "pole_mass", "pole_com", "pole_inertia", "pole_length",
    "cart_damping", "pole_damping", "gravity", "cart_range",
) + LIMIT_KEYS


def cartpole_rail_range(plant_parameters: Mapping[str, float]) -> Tuple[Tuple[int, float, float], ...]:
    """limited_position_ranges of the cart-pole state: the cart (index 0) within +-cart_range."""
    if "cart_range" not in plant_parameters:
        raise ValueError("Cart-pole plant_parameters need cart_range, the slider's limit.")
    rail = float(plant_parameters["cart_range"])
    return ((0, -rail, rail),)


# dm_control's pole (cartpole.xml): a capsule of radius 0.045 m from the hinge to
# pole_length, carrying pole_mass; MuJoCo compiles pole_com and pole_inertia from it.
POLE_RADIUS = 0.045
# The joints whose damping a "damping" physical change scales: the slider and the hinge.
CARTPOLE_DAMPING_KEYS = ("cart_damping", "pole_damping")


def capsule_transverse_inertia(mass: float, radius: float, length: float) -> float:
    """MuJoCo's compiled inertia of a solid capsule about an axis through its centre of mass
    perpendicular to its axis (user_objects.cc, mjCGeom's capsule): the cylinder of the
    given length and radius plus two hemispheres, at uniform density, mass shared by
    volume. 0.009424592767295598 kg m^2 for dm_control's pole (0.1 kg, 0.045 m, 1 m)."""
    radius, length = float(radius), float(length)
    sphere_mass = float(mass) * 4.0 * radius / (4.0 * radius + 3.0 * length)
    cylinder_mass = float(mass) - sphere_mass
    cylinder = cylinder_mass * (3.0 * radius * radius + length * length) / 12.0
    sphere = 2.0 * sphere_mass * radius * radius / 5.0
    offset = sphere_mass * length * (3.0 * radius + 2.0 * length) / 8.0
    # MuJoCo's order: the cylinder, then the hemispheres' own and translated inertia.
    return cylinder + (sphere + offset)


def cartpole_physical_change(
    plant_parameters: Mapping[str, float], kind: str, scale: float
) -> dict:
    """The plant_parameters of cartpole_swingup's changed robot (the adaptation experiment;
    common.ph_dmc._common.with_physical_change).

    "length": the pole's length times scale with its mass and radius kept, the geometry
    MuJoCo compiles dm_control's pole from: pole_length (the drawn capsule) and pole_com
    (half the length) times scale, and pole_inertia times the ratio of the capsule's
    transverse inertia at the new and the old length (capsule_transverse_inertia, radius
    POLE_RADIUS), so the row's compiled value scales by the geometry: the mass matrix and
    the gravity potential change. "damping": the slider's and the hinge's damping times
    scale, nothing else.
    """
    changed = {key: value for key, value in plant_parameters.items()}
    scale = float(scale)
    if kind == "length":
        length = float(plant_parameters["pole_length"])
        ratio = capsule_transverse_inertia(1.0, POLE_RADIUS, scale * length) / (
            capsule_transverse_inertia(1.0, POLE_RADIUS, length)
        )
        changed["pole_length"] = scale * length
        changed["pole_com"] = scale * float(plant_parameters["pole_com"])
        changed["pole_inertia"] = ratio * float(plant_parameters["pole_inertia"])
    elif kind == "damping":
        for key in CARTPOLE_DAMPING_KEYS:
            changed[key] = float(plant_parameters[key]) * scale
    else:
        raise ValueError(f"cartpole_swingup has no physical change {kind!r}.")
    return changed


def _with_rail_port_init(config: TestbedConfig, *, poles: bool) -> TestbedConfig:
    """Add the slider's limit_port_init from the compiled model (the swing-up's, or the
    multi-task model with poles=True); environment/ph_sac/dmc/cartpole.py imports this
    module, so it is imported here."""
    from environment.ph_sac.dmc import cartpole as dmc_cartpole

    build = dmc_cartpole.build_cartpole_poles_model if poles else dmc_cartpole.build_cartpole_model
    return with_limit_port_init(config, dmc_cartpole.cartpole_rail_limits(build(config)))


def cartpole_swingup_config(
    *,
    failure_limits: Mapping[str, float],
    reward_parameters: Mapping[str, float],
    reset_parameters: Mapping[str, object],
    plant_parameters: Mapping[str, float],
) -> TestbedConfig:
    """Declare dm_control's cart-pole swing-up: 10 s episodes, 10 ms decisions, +-10 N on the cart.

    The pole angle is periodic and wrapped; the cart position is not, and is a raw network
    feature. Decisions are offsets from the cart's position, as Acrobot's are from the elbow's,
    in the same box, so the two tasks share a decision density. The cart is limited to
    +-cart_range by dm_control's soft rail limit (limited_position_ranges, with the limit
    port's start limit_port_init from the compiled model); failure_limits must be empty.
    """
    require_no_failure_set("cartpole_swingup", failure_limits)
    physics_dt = 0.001
    reference_period = 0.01
    state_spec: StateSpec = {
        "position_dim": 2,
        "velocity_dim": 2,
        "action_dim": 1,
        "periodic_position_indices": (1,),
        "actuated_position_indices": (0,),
        "observation_projection": np.eye(2),
        "limited_position_ranges": cartpole_rail_range(plant_parameters),
    }
    clock: ClockConfig = {
        "kind": "fixed",
        "physics_dt": physics_dt,
        "reference_period": reference_period,
        "min_duration": reference_period,
        "max_duration": reference_period,
        "tail_probability": None,
        "tail_split": None,
        "beta_shape": None,
    }
    controller: ControllerConfig = {
        "decision_low": np.array([-1.0, 0.0, 0.0]),
        "decision_high": np.array([1.0, 100.0, 20.0]),
        "torque_low": np.array([-10.0]),
        "torque_high": np.array([10.0]),
        "setpoint_mode": "offset",
        "setpoint_error": "unwrapped",
        "refresh_steps": 1,
        "refresh_mode": DMC_REFRESH_MODE,
    }
    config: TestbedConfig = {
        "name": "cartpole_swingup",
        "state_spec": state_spec,
        "physics_dt": physics_dt,
        "reference_period": reference_period,
        "episode_duration": 10.0,
        "discount_rate": 0.5,
        "clock": clock,
        "controller": controller,
        "integrator": "RK4",
        "plant_parameters": plant_parameters,
        "failure_limits": failure_limits,
        "reward_parameters": reward_parameters,
        "reset_parameters": reset_parameters,
    }
    return _with_rail_port_init(config, poles=False)


def _pole_count(name: str, plant_parameters: Mapping[str, float], expected: int) -> int:
    """The number of poles, checked against plant_parameters["n_poles"] when it is given.

    The single-pole rows leave n_poles out; the two- and three-pole rows must state it.
    """
    if expected > 1 and "n_poles" not in plant_parameters:
        raise ValueError(f"{name} needs plant_parameters['n_poles'] = {expected}.")
    count = plant_parameters.get("n_poles", expected)
    if isinstance(count, bool) or count != expected:
        raise ValueError(
            f"{name} has {expected} pole(s), not plant_parameters['n_poles'] = {count!r}."
        )
    return expected


def _cartpole_task_config(
    name: str,
    n_poles: int,
    *,
    failure_limits: Mapping[str, float],
    reward_parameters: Mapping[str, float],
    reset_parameters: Mapping[str, object],
    plant_parameters: Mapping[str, float],
) -> TestbedConfig:
    """State (x, theta_1..theta_n, x_dot, theta_dot_1..theta_dot_n): every hinge wrapped,
    the cart actuated, a raw feature and limited to +-cart_range, as the swing-up declares it."""
    require_no_failure_set(name, failure_limits)
    n_poles = _pole_count(name, plant_parameters, n_poles)
    config = dmc_testbed_config(
        name=name,
        state_spec=dmc_state_spec(
            physical_dim=1 + n_poles,
            action_dim=1,
            periodic=tuple(range(1, 1 + n_poles)),
            actuated=(0,),
            limited_position_ranges=cartpole_rail_range(plant_parameters),
        ),
        reference_period=_REFERENCE_PERIOD,
        episode_duration=_EPISODE_DURATION,
        controller=box_rule_controller(
            offsets=(_CART_OFFSET,),
            torque_limit=_FORCE_LIMIT,
            setpoint_mode="offset",
            setpoint_error="unwrapped",
        ),
        plant_parameters=plant_parameters,
        failure_limits=failure_limits,
        reward_parameters=reward_parameters,
        reset_parameters=reset_parameters,
    )
    return _with_rail_port_init(config, poles=True)


def cartpole_balance_config(
    *,
    failure_limits: Mapping[str, float],
    reward_parameters: Mapping[str, float],
    reset_parameters: Mapping[str, object],
    plant_parameters: Mapping[str, float],
) -> TestbedConfig:
    """Declare dm_control's cartpole/balance (table row cartpole-balance).

    Settings (the row's env_* cells; the environment checks the keys):
        failure_limits     {} (none: the rail is dm_control's soft limit)
        reward_parameters  cart_margin (2), velocity_margin (5), failure_rate (0): the
                           smooth reward
        reset_parameters   position_range (0.1), angle_range (0.034), velocity_noise
                           (0.01): cart and angle uniform on +-range, velocities Gaussian
        plant_parameters   the swing-up's cart-pole (cart_mass, pole_mass, pole_com,
                           pole_inertia, pole_length, cart_damping, pole_damping, gravity)
                           and its rail limit (cart_range 1.8, limit_timeconst 0.08,
                           limit_dampratio 1, limit_d0 0.9, limit_dmax 0.95, limit_width
                           0.001, limit_midpoint 0.5, limit_power 2)
    Declares nq = nv = 2, nu = 1, periodic (1,), actuated (0,), the shared cart-pole
    clock, box and physics (module docstring).
    """
    return _cartpole_task_config(
        "cartpole_balance",
        1,
        failure_limits=failure_limits,
        reward_parameters=reward_parameters,
        reset_parameters=reset_parameters,
        plant_parameters=plant_parameters,
    )


def cartpole_balance_sparse_config(
    *,
    failure_limits: Mapping[str, float],
    reward_parameters: Mapping[str, float],
    reset_parameters: Mapping[str, object],
    plant_parameters: Mapping[str, float],
) -> TestbedConfig:
    """Declare dm_control's cartpole/balance_sparse (table row cartpole-balance_sparse).

    As cartpole_balance_config, with reward_parameters cart_bound (0.25), cosine_bound
    (0.995) and failure_rate (0): the reward 1[|x| <= cart_bound] 1[cos theta >=
    cosine_bound].
    """
    return _cartpole_task_config(
        "cartpole_balance_sparse",
        1,
        failure_limits=failure_limits,
        reward_parameters=reward_parameters,
        reset_parameters=reset_parameters,
        plant_parameters=plant_parameters,
    )


def cartpole_swingup_sparse_config(
    *,
    failure_limits: Mapping[str, float],
    reward_parameters: Mapping[str, float],
    reset_parameters: Mapping[str, object],
    plant_parameters: Mapping[str, float],
) -> TestbedConfig:
    """Declare dm_control's cartpole/swingup_sparse (table row cartpole-swingup_sparse).

    The swing-up's settings with the sparse reward of cartpole_balance_sparse_config:
    reset_parameters start_angle (pi), position_noise, angle_noise, velocity_noise (0.01),
    reward_parameters cart_bound (0.25), cosine_bound (0.995), failure_rate (0).
    """
    return _cartpole_task_config(
        "cartpole_swingup_sparse",
        1,
        failure_limits=failure_limits,
        reward_parameters=reward_parameters,
        reset_parameters=reset_parameters,
        plant_parameters=plant_parameters,
    )


def cartpole_two_poles_config(
    *,
    failure_limits: Mapping[str, float],
    reward_parameters: Mapping[str, float],
    reset_parameters: Mapping[str, object],
    plant_parameters: Mapping[str, float],
) -> TestbedConfig:
    """Declare dm_control's cartpole/two_poles (table row cartpole-two_poles).

    The swing-up's settings, with plant_parameters n_poles = 2 (identical poles chained
    at the tips, pole_length apart) and reset_parameters upper_angle_noise (0.1), the
    standard deviation of the second hinge's start. Declares nq = nv = 3, periodic (1, 2).
    """
    return _cartpole_task_config(
        "cartpole_two_poles",
        2,
        failure_limits=failure_limits,
        reward_parameters=reward_parameters,
        reset_parameters=reset_parameters,
        plant_parameters=plant_parameters,
    )


def cartpole_three_poles_config(
    *,
    failure_limits: Mapping[str, float],
    reward_parameters: Mapping[str, float],
    reset_parameters: Mapping[str, object],
    plant_parameters: Mapping[str, float],
) -> TestbedConfig:
    """Declare dm_control's cartpole/three_poles (table row cartpole-three_poles).

    As cartpole_two_poles_config with n_poles = 3. Declares nq = nv = 4, periodic (1, 2, 3).
    """
    return _cartpole_task_config(
        "cartpole_three_poles",
        3,
        failure_limits=failure_limits,
        reward_parameters=reward_parameters,
        reset_parameters=reset_parameters,
        plant_parameters=plant_parameters,
    )

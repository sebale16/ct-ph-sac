"""MuJoCo models and task declarations of the in-house testbeds; name-based dispatch.

The in-house testbeds live here: the paper's windy pendulum (balance and swing-up) and
Acrobot-XK. The dm_control ports, the cart-pole swing-up among them, live in
environment/ph_sac/dmc/ (CartpoleSwingupEnv and build_cartpole_model are re-exported
here). Every testbed registers a TestbedSpec in environment/ph_sac/registry.py, and the
dispatchers below (make_env, reward_rate_tensor, failure_mask_tensor,
terminal_payoff_tensor, terminal_boundary_functions, oracle_callbacks, make_oracle_plant)
look the testbed up by config["name"].

Experiment settings the paper leaves open arrive through the testbed configuration:

Windy pendulum:
    reward_parameters: angle_weight, velocity_weight, velocity_scale, effort_weight.
    reset_parameters: angle_radius, velocity_radius; optional start_angle (default 0).
    Balance and swing-up differ only in configuration, all read from the config: the
    swing-up starts at start_angle pi, sets failure_limits.angle to inf (no angle
    failure; the cost bound then takes the angle term at pi), and takes its wind
    intensity, controller decision box and torque limits from plant_parameters and
    controller.

Acrobot-XK:
    plant_parameters: m1, m2, l1, l2, lc1, lc2, I1, I2, gravity.
    failure_limits: elbow_angle (unwrapped), elbow_rate, shoulder_rate.
    reward_parameters: energy_scale, upright_energy, lyapunov_k_p, reward_floor,
    failure_rate. The reward rate is the saved CPHT-SAC runs' r1, the log-reciprocal
    of the Xin--Kaneda Lyapunov function relative to hanging rest:
        r1(x) = -log(max(V(x) / V_down, reward_floor)),
        V(x) = (E(x) - upright_energy)^2 / 2 + lyapunov_k_p q2^2 / 2,
        V_down = energy_scale^2 / 2,
    with E the mechanical energy and q2 the unwrapped elbow angle. It is 0 at hanging
    rest and -log(reward_floor) at upright rest.
    reset_parameters: release_angle_low, release_angle_high; optional angle_noise,
    velocity_noise (default 0).

Every task accepts options={"state": s} at reset to start from an explicit state.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Callable, Literal, Mapping, Optional, Tuple

import mujoco
import numpy as np
import torch

from common.ph_config import acrobot_xk_config, windy_pendulum_config

from .base import MujocoPHEnv
from .base import validated_state_spec as state_spec  # public name; the envs call it
from .dmc._common import failure_rate_payoff, limit_boundary_functions, require_keys
from .dmc.cartpole import CartpoleSwingupEnv, build_cartpole_model  # noqa: F401  (re-exported)
from .registry import TestbedSpec, register_testbed, testbed_spec

if TYPE_CHECKING:
    from mujoco import MjModel
    from numpy import ndarray
    from torch import Tensor

    from common.ph_config import TestbedConfig
    from models.ph_sac.plant import OraclePlant

_HANGING_SHOULDER = -0.5 * math.pi

# The point mass keeps a negligible rotational inertia that MuJoCo requires to compile.
_WINDY_XML = """
<mujoco model="windy-pendulum">
  <option timestep="{timestep}" integrator="{integrator}" gravity="0 0 -{gravity}">
    <flag contact="disable" constraint="disable"/>
  </option>
  <visual><global offwidth="1280" offheight="960"/></visual>
  <worldbody>
    <light pos="0 -2 4"/>
    <camera name="fixed" pos="0 -4 0" xyaxes="1 0 0 0 0 1"/>
    <body name="pole">
      <joint name="hinge" type="hinge" axis="0 1 0" damping="{damping}"/>
      <inertial pos="0 0 {length}" mass="{mass}" diaginertia="1e-9 1e-9 1e-9"/>
      <geom type="capsule" fromto="0 0 0 0 0 {length}" size="0.02" mass="0"/>
    </body>
  </worldbody>
  <actuator>
    <motor name="motor" joint="hinge" gear="1" ctrllimited="true"
           ctrlrange="{torque_low} {torque_high}"/>
  </actuator>
</mujoco>
"""

# Links rest along +x and the hinges turn about -y, so qpos is Xin--Kaneda's (q1, q2):
# upright is q1=+pi/2, hanging is q1=-pi/2, with the potential b1 sin q1 + b2 sin(q1+q2).
_ACROBOT_XML = """
<mujoco model="acrobot-xk">
  <option timestep="{timestep}" integrator="{integrator}" gravity="0 0 -{gravity}">
    <flag contact="disable" constraint="disable"/>
  </option>
  <visual><global offwidth="1280" offheight="960"/></visual>
  <default>
    <joint type="hinge" axis="0 -1 0" damping="0" limited="false"/>
    <geom type="capsule" mass="0"/>
  </default>
  <worldbody>
    <light pos="0 -2 6"/>
    <camera name="fixed" pos="0 -8 0" xyaxes="1 0 0 0 0 1"/>
    <body name="upper_arm">
      <joint name="shoulder"/>
      <inertial pos="{lc1} 0 0" mass="{m1}" diaginertia="0.001 {i1} {i1}"/>
      <geom fromto="0 0 0 {l1} 0 0" size="0.05"/>
      <body name="lower_arm" pos="{l1} 0 0">
        <joint name="elbow"/>
        <inertial pos="{lc2} 0 0" mass="{m2}" diaginertia="0.001 {i2} {i2}"/>
        <geom fromto="0 0 0 {l2} 0 0" size="0.049"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor name="elbow" joint="elbow" gear="1" ctrllimited="true"
           ctrlrange="{torque_low} {torque_high}"/>
  </actuator>
</mujoco>
"""

_require = require_keys


def _windy_cost_bound(config: TestbedConfig) -> float:
    """c_max: the largest unscaled cost rate inside the limits with the motor saturated."""
    reward = config["reward_parameters"]
    limits = config["failure_limits"]
    controller = config["controller"]
    torque_limit = float(
        max(np.max(np.abs(controller["torque_low"])), np.max(np.abs(controller["torque_high"])))
    )
    angle = min(float(limits["angle"]), math.pi)
    return (
        2.0 * reward["angle_weight"] * math.sin(0.5 * angle) ** 2
        + reward["velocity_weight"] * (limits["velocity"] / reward["velocity_scale"]) ** 2
        + 0.5 * reward["effort_weight"] * torque_limit**2
    )


def _acrobot_grouped(plant: Mapping[str, float]) -> Tuple[float, float, float, float, float]:
    """Xin--Kaneda's (a1, a2, a3, b1, b2) from the link parameters."""
    a1 = plant["I1"] + plant["m1"] * plant["lc1"] ** 2 + plant["m2"] * plant["l1"] ** 2
    a2 = plant["I2"] + plant["m2"] * plant["lc2"] ** 2
    a3 = plant["m2"] * plant["l1"] * plant["lc2"]
    b1 = (plant["m1"] * plant["lc1"] + plant["m2"] * plant["l1"]) * plant["gravity"]
    b2 = plant["m2"] * plant["lc2"] * plant["gravity"]
    return a1, a2, a3, b1, b2


class WindyPendulumEnv(MujocoPHEnv):
    """MuJoCo pendulum with upright-referenced angle and physics-step gust torque.

    MuJoCo remains the simulator.
    """

    def __init__(
        self,
        config: TestbedConfig,
        *,
        decision_mode: Literal["torque", "feedback"] = "feedback",
        seed: Optional[int] = None,
    ) -> None:
        _require(
            config["reward_parameters"],
            ("angle_weight", "velocity_weight", "velocity_scale", "effort_weight"),
            "Windy reward_parameters",
        )
        _require(
            config["reset_parameters"],
            ("angle_radius", "velocity_radius"),
            "Windy reset_parameters",
        )
        super().__init__(
            config,
            build_windy_pendulum_model(config),
            state_spec(config),
            decision_mode=decision_mode,
            seed=seed,
        )
        plant = config["plant_parameters"]
        self._inertia = plant["mass"] * plant["length"] ** 2
        self._gravity_torque = plant["mass"] * plant["gravity"] * plant["length"]
        self._damping = plant["damping"]
        self._wind_intensity = plant["wind_intensity"]
        self._cost_bound = _windy_cost_bound(config)

    def _reset_state(self, options: Optional[Mapping[str, object]] = None) -> None:
        if options and "state" in options:
            self._set_state(options["state"])
            return
        reset = self.config["reset_parameters"]
        start = float(reset.get("start_angle", 0.0))
        angle = start + self._reset_rng.uniform(-reset["angle_radius"], reset["angle_radius"])
        speed = self._reset_rng.uniform(-reset["velocity_radius"], reset["velocity_radius"])
        self._set_state(np.array([angle, speed]))

    def _apply_disturbance(self) -> None:
        """Apply external gust torque sigma_w*xi/sqrt(delta), outside motor clipping."""
        self.data.qfrc_applied[0] = (
            self._wind_intensity
            * self._gust_rng.standard_normal()
            / math.sqrt(self.physics_dt)
        )

    def reward_rate(self, state: ndarray, torque: ndarray) -> float:
        """Normalized angle, speed and motor cost using the configured weights."""
        reward = self.config["reward_parameters"]
        angle, speed = float(state[0]), float(state[1])
        cost = (
            2.0 * reward["angle_weight"] * math.sin(0.5 * angle) ** 2
            + reward["velocity_weight"] * (speed / reward["velocity_scale"]) ** 2
            + 0.5 * reward["effort_weight"] * float(np.sum(np.square(torque)))
        )
        return -cost / self._cost_bound

    def failed(self, state: ndarray) -> bool:
        limits = self.config["failure_limits"]
        return bool(abs(state[0]) >= limits["angle"] or abs(state[1]) >= limits["velocity"])

    def terminal_payoff(self, state: ndarray) -> float:
        """Prescribe the failure boundary value F=-1/lambda."""
        return -1.0 / self.discount_rate

    def oracle_drift(self, state: ndarray, torque: ndarray) -> ndarray:
        angle, speed = float(state[0]), float(state[1])
        acceleration = (
            self._gravity_torque * math.sin(angle)
            - self._damping * speed
            + float(np.asarray(torque).reshape(-1)[0])
        ) / self._inertia
        return np.array([speed, acceleration])

    def oracle_diffusion(self, state: ndarray) -> ndarray:
        """Return the continuous column (0, sigma_w/I), before discrete transport."""
        return np.array([[0.0], [self._wind_intensity / self._inertia]])


class AcrobotXKEnv(MujocoPHEnv):
    """Xin–Kaneda geometry, elbow motor, no damping and MuJoCo RK4 physics.

    The model's rk4 substep map (model_integrator "rk4", models/ph_sac/rollouts.py)
    reproduces these endpoints from the true drift; its semi_implicit map does not.
    """

    wrapped_observation_indices = (0,)

    def __init__(
        self,
        config: TestbedConfig,
        *,
        decision_mode: Literal["torque", "feedback"] = "feedback",
        seed: Optional[int] = None,
    ) -> None:
        _require(
            config["plant_parameters"],
            ("m1", "m2", "l1", "l2", "lc1", "lc2", "I1", "I2", "gravity"),
            "Acrobot plant_parameters",
        )
        _require(
            config["failure_limits"],
            ("elbow_angle", "elbow_rate", "shoulder_rate"),
            "Acrobot failure_limits",
        )
        _require(
            config["reward_parameters"],
            ("energy_scale", "upright_energy", "lyapunov_k_p", "reward_floor", "failure_rate"),
            "Acrobot reward_parameters",
        )
        _require(
            config["reset_parameters"],
            ("release_angle_low", "release_angle_high"),
            "Acrobot reset_parameters",
        )
        super().__init__(
            config,
            build_acrobot_xk_model(config),
            state_spec(config),
            decision_mode=decision_mode,
            seed=seed,
        )
        self._grouped = _acrobot_grouped(config["plant_parameters"])

    def _reset_state(self, options: Optional[Mapping[str, object]] = None) -> None:
        """Use the configured release start."""
        if options and "state" in options:
            self._set_state(options["state"])
            return
        reset = self.config["reset_parameters"]
        displacement = self._reset_rng.uniform(
            reset["release_angle_low"], reset["release_angle_high"]
        )
        if self._reset_rng.uniform() < 0.5:
            displacement = -displacement
        angle_noise = float(reset.get("angle_noise", 0.0))
        velocity_noise = float(reset.get("velocity_noise", 0.0))
        positions = np.array([_HANGING_SHOULDER + displacement, 0.0])
        positions += self._reset_rng.uniform(-angle_noise, angle_noise, 2)
        velocities = self._reset_rng.uniform(-velocity_noise, velocity_noise, 2)
        self._set_state(np.concatenate((positions, velocities)))

    def _mass_matrix(self, elbow: float) -> Tuple[float, float, float]:
        a1, a2, a3, _, _ = self._grouped
        cosine = math.cos(elbow)
        return a1 + a2 + 2.0 * a3 * cosine, a2 + a3 * cosine, a2

    def reward_rate(self, state: ndarray, torque: ndarray) -> float:
        """Use the task's state-only r1 rate; reward_rate_tensor must stay its copy."""
        reward = self.config["reward_parameters"]
        shoulder, elbow, shoulder_rate, elbow_rate = (float(x) for x in state)
        m11, m12, m22 = self._mass_matrix(elbow)
        kinetic = 0.5 * (
            m11 * shoulder_rate**2
            + 2.0 * m12 * shoulder_rate * elbow_rate
            + m22 * elbow_rate**2
        )
        _, _, _, b1, b2 = self._grouped
        potential = b1 * math.sin(shoulder) + b2 * math.sin(shoulder + elbow)
        energy_error = kinetic + potential - reward["upright_energy"]
        lyapunov = 0.5 * energy_error**2 + 0.5 * reward["lyapunov_k_p"] * elbow**2
        return -math.log(
            max(lyapunov / (0.5 * reward["energy_scale"] ** 2), reward["reward_floor"])
        )

    def failed(self, state: ndarray) -> bool:
        """Check unwrapped elbow travel and both joint-speed caps."""
        limits = self.config["failure_limits"]
        return bool(
            abs(state[1]) >= limits["elbow_angle"]
            or abs(state[2]) >= limits["shoulder_rate"]
            or abs(state[3]) >= limits["elbow_rate"]
        )

    def terminal_payoff(self, state: ndarray) -> float:
        return self.config["reward_parameters"]["failure_rate"] / self.discount_rate

    def oracle_drift(self, state: ndarray, torque: ndarray) -> ndarray:
        shoulder, elbow, shoulder_rate, elbow_rate = (float(x) for x in state)
        a3, b1, b2 = self._grouped[2], self._grouped[3], self._grouped[4]
        m11, m12, m22 = self._mass_matrix(elbow)
        sine = math.sin(elbow)
        force1 = (
            a3 * sine * (2.0 * shoulder_rate * elbow_rate + elbow_rate**2)
            - b1 * math.cos(shoulder)
            - b2 * math.cos(shoulder + elbow)
        )
        force2 = (
            float(np.asarray(torque).reshape(-1)[0])
            - a3 * sine * shoulder_rate**2
            - b2 * math.cos(shoulder + elbow)
        )
        determinant = m11 * m22 - m12**2
        return np.array(
            [
                shoulder_rate,
                elbow_rate,
                (m22 * force1 - m12 * force2) / determinant,
                (m11 * force2 - m12 * force1) / determinant,
            ]
        )

    def oracle_diffusion(self, state: ndarray) -> ndarray:
        """Return the task's continuous diffusion; the default plant has no gusts."""
        return np.zeros((self.state_dim, 1))


def build_windy_pendulum_model(config: TestbedConfig) -> MjModel:
    """Build MuJoCo geometry and Euler settings; retain simulator numerical bias."""
    plant = config["plant_parameters"]
    controller = config["controller"]
    return mujoco.MjModel.from_xml_string(
        _WINDY_XML.format(
            timestep=config["physics_dt"],
            integrator=config["integrator"],
            gravity=plant["gravity"],
            damping=plant["damping"],
            length=plant["length"],
            mass=plant["mass"],
            torque_low=float(controller["torque_low"][0]),
            torque_high=float(controller["torque_high"][0]),
        )
    )


def build_acrobot_xk_model(config: TestbedConfig) -> MjModel:
    """Build the customized DMC Acrobot geometry with MuJoCo RK4 settings."""
    plant = config["plant_parameters"]
    controller = config["controller"]
    return mujoco.MjModel.from_xml_string(
        _ACROBOT_XML.format(
            timestep=config["physics_dt"],
            integrator=config["integrator"],
            gravity=plant["gravity"],
            m1=plant["m1"],
            m2=plant["m2"],
            l1=plant["l1"],
            l2=plant["l2"],
            lc1=plant["lc1"],
            lc2=plant["lc2"],
            i1=plant["I1"],
            i2=plant["I2"],
            torque_low=float(controller["torque_low"][0]),
            torque_high=float(controller["torque_high"][0]),
        )
    )


# ------------------------------------------------------------------------ dispatchers


def make_env(
    config: TestbedConfig,
    *,
    decision_mode: Literal["torque", "feedback"] = "feedback",
    seed: Optional[int] = None,
) -> MujocoPHEnv:
    """Construct a testbed with a MuJoCo backend in both modes."""
    return testbed_spec(config["name"]).env_class(
        config, decision_mode=decision_mode, seed=seed
    )


def reward_rate_tensor(
    states: Tensor, torques: Tensor, config: TestbedConfig
) -> Tensor:
    """Evaluate the known task rate on tensor batches for model operators."""
    return testbed_spec(config["name"]).reward_rate_tensor(states, torques, config)


def failure_mask_tensor(states: Tensor, config: TestbedConfig) -> Tensor:
    """Apply the same sampled task predicate and unwrapped travel convention."""
    return testbed_spec(config["name"]).failure_mask_tensor(states, config)


def _terminal_payoff_value(config: TestbedConfig) -> float:
    return testbed_spec(config["name"]).terminal_payoff_value(config)


def terminal_payoff_tensor(states: Tensor, config: TestbedConfig) -> Tensor:
    """Supply boundary values; observed terminal rewards already carry their charge."""
    return torch.full_like(states[..., :1], _terminal_payoff_value(config))


def terminal_boundary_functions(
    config: TestbedConfig,
) -> Tuple[Callable[[Tensor], Tensor], Callable[[Tensor], Tensor]]:
    """Return (H, B) for V=H+B*network: H is the payoff and B vanishes on every limit.

    Each limited coordinate contributes the smooth factor 1-(x/x_lim)^2, so B is
    positive strictly inside the failure limits. The elbow uses its unwrapped angle.
    An infinite limit (the windy swing-up's angle) bounds nothing and adds no factor.
    """
    return testbed_spec(config["name"]).terminal_boundary_functions(config)


def oracle_callbacks(
    config: TestbedConfig,
) -> Tuple[Callable[[Tensor, Tensor], Tensor], Callable[[Tensor], Tensor]]:
    """Analytic drift and diffusion in observation coordinates, compatible with torch.func."""
    return testbed_spec(config["name"]).oracle_callbacks(config)


# ------------------------------------------------------------- in-house task tensors


def _windy_reward_rate_tensor(
    states: Tensor, torques: Tensor, config: TestbedConfig
) -> Tensor:
    reward = config["reward_parameters"]
    cost = (
        2.0 * reward["angle_weight"] * torch.sin(0.5 * states[..., :1]).square()
        + reward["velocity_weight"] * (states[..., 1:2] / reward["velocity_scale"]).square()
        + 0.5 * reward["effort_weight"] * torques.square().sum(dim=-1, keepdim=True)
    )
    return -cost / _windy_cost_bound(config)


def _windy_limits(config: TestbedConfig) -> Tuple[Tuple[int, float], ...]:
    limits = config["failure_limits"]
    return ((0, float(limits["angle"])), (1, float(limits["velocity"])))


def _windy_failure_mask_tensor(states: Tensor, config: TestbedConfig) -> Tensor:
    limits = config["failure_limits"]
    return (states[..., 0].abs() >= limits["angle"]) | (
        states[..., 1].abs() >= limits["velocity"]
    )


def _windy_terminal_payoff_value(config: TestbedConfig) -> float:
    return -1.0 / config["discount_rate"]


def _windy_terminal_boundary_functions(
    config: TestbedConfig,
) -> Tuple[Callable[[Tensor], Tensor], Callable[[Tensor], Tensor]]:
    return limit_boundary_functions(_windy_terminal_payoff_value(config), _windy_limits(config))


def _windy_oracle_callbacks(
    config: TestbedConfig,
) -> Tuple[Callable[[Tensor, Tensor], Tensor], Callable[[Tensor], Tensor]]:
    plant = config["plant_parameters"]
    inertia = plant["mass"] * plant["length"] ** 2
    gravity_torque = plant["mass"] * plant["gravity"] * plant["length"]
    damping = plant["damping"]
    noise = plant["wind_intensity"] / inertia

    def windy_drift(states: Tensor, actions: Tensor) -> Tensor:
        angle, speed = states[..., :1], states[..., 1:2]
        acceleration = (
            gravity_torque * torch.sin(angle) - damping * speed + actions
        ) / inertia
        return torch.cat((speed, acceleration), dim=-1)

    def windy_diffusion(states: Tensor) -> Tensor:
        column = torch.as_tensor(
            [[0.0], [noise]], dtype=states.dtype, device=states.device
        )
        return column.expand(*states.shape[:-1], 2, 1)

    return windy_drift, windy_diffusion


def _wrap(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def _windy_upright(config: TestbedConfig, state: ndarray, tube: float) -> bool:
    return abs(_wrap(float(state[0]))) <= tube


def _acrobot_reward_rate_tensor(
    states: Tensor, torques: Tensor, config: TestbedConfig
) -> Tensor:
    reward = config["reward_parameters"]
    a1, a2, a3, b1, b2 = _acrobot_grouped(config["plant_parameters"])
    shoulder, elbow = states[..., :1], states[..., 1:2]
    shoulder_rate, elbow_rate = states[..., 2:3], states[..., 3:4]
    cosine = torch.cos(elbow)
    kinetic = 0.5 * (
        (a1 + a2 + 2.0 * a3 * cosine) * shoulder_rate.square()
        + 2.0 * (a2 + a3 * cosine) * shoulder_rate * elbow_rate
        + a2 * elbow_rate.square()
    )
    potential = b1 * torch.sin(shoulder) + b2 * torch.sin(shoulder + elbow)
    energy_error = kinetic + potential - reward["upright_energy"]
    lyapunov = 0.5 * energy_error.square() + 0.5 * reward["lyapunov_k_p"] * elbow.square()
    return -torch.log(
        torch.clamp(
            lyapunov / (0.5 * reward["energy_scale"] ** 2), min=reward["reward_floor"]
        )
    )


def _acrobot_limits(config: TestbedConfig) -> Tuple[Tuple[int, float], ...]:
    limits = config["failure_limits"]
    return (
        (1, float(limits["elbow_angle"])),
        (2, float(limits["shoulder_rate"])),
        (3, float(limits["elbow_rate"])),
    )


def _acrobot_failure_mask_tensor(states: Tensor, config: TestbedConfig) -> Tensor:
    limits = config["failure_limits"]
    return (
        (states[..., 1].abs() >= limits["elbow_angle"])
        | (states[..., 2].abs() >= limits["shoulder_rate"])
        | (states[..., 3].abs() >= limits["elbow_rate"])
    )


def _acrobot_terminal_boundary_functions(
    config: TestbedConfig,
) -> Tuple[Callable[[Tensor], Tensor], Callable[[Tensor], Tensor]]:
    return limit_boundary_functions(failure_rate_payoff(config), _acrobot_limits(config))


def _acrobot_oracle_callbacks(
    config: TestbedConfig,
) -> Tuple[Callable[[Tensor, Tensor], Tensor], Callable[[Tensor], Tensor]]:
    a1, a2, a3, b1, b2 = _acrobot_grouped(config["plant_parameters"])

    def acrobot_drift(states: Tensor, actions: Tensor) -> Tensor:
        shoulder, elbow = states[..., :1], states[..., 1:2]
        shoulder_rate, elbow_rate = states[..., 2:3], states[..., 3:4]
        cosine, sine = torch.cos(elbow), torch.sin(elbow)
        m11 = a1 + a2 + 2.0 * a3 * cosine
        m12 = a2 + a3 * cosine
        gravity_outer = b2 * torch.cos(shoulder + elbow)
        force1 = (
            a3 * sine * (2.0 * shoulder_rate * elbow_rate + elbow_rate.square())
            - b1 * torch.cos(shoulder)
            - gravity_outer
        )
        force2 = actions - a3 * sine * shoulder_rate.square() - gravity_outer
        determinant = m11 * a2 - m12.square()
        return torch.cat(
            (
                shoulder_rate,
                elbow_rate,
                (a2 * force1 - m12 * force2) / determinant,
                (m11 * force2 - m12 * force1) / determinant,
            ),
            dim=-1,
        )

    def acrobot_diffusion(states: Tensor) -> Tensor:
        return torch.zeros(
            *states.shape[:-1], 4, 1, dtype=states.dtype, device=states.device
        )

    return acrobot_drift, acrobot_diffusion


def _acrobot_upright(config: TestbedConfig, state: ndarray, tube: float) -> bool:
    """Shoulder at pi/2 (hanging is -pi/2) and elbow at 0."""
    return (
        abs(_wrap(float(state[0]) - 0.5 * math.pi)) <= tube
        and abs(_wrap(float(state[1]))) <= tube
    )


_WINDY_SETTINGS = (
    "task",
    "reward_parameters",
    "reset_parameters",
    "tail_probability",
    "tail_split",
    "clock_kind",
    "beta_shape",
)
_ACROBOT_SETTINGS = ("failure_limits", "reward_parameters", "reset_parameters", "plant_parameters")


def _register_in_house() -> None:
    register_testbed(
        TestbedSpec(
            name="windy_pendulum",
            env_class=WindyPendulumEnv,
            config_factory=windy_pendulum_config,
            settings_keys=_WINDY_SETTINGS,
            # torque_limit is the swing-up motor limit; the balance task has none.
            optional_settings_keys=("torque_limit",),
            reward_rate_tensor=_windy_reward_rate_tensor,
            failure_mask_tensor=_windy_failure_mask_tensor,
            terminal_payoff_value=_windy_terminal_payoff_value,
            terminal_boundary_functions=_windy_terminal_boundary_functions,
            oracle_callbacks=_windy_oracle_callbacks,
            upright=_windy_upright,
        )
    )
    register_testbed(
        TestbedSpec(
            name="acrobot_xk",
            env_class=AcrobotXKEnv,
            config_factory=acrobot_xk_config,
            settings_keys=_ACROBOT_SETTINGS,
            reward_rate_tensor=_acrobot_reward_rate_tensor,
            failure_mask_tensor=_acrobot_failure_mask_tensor,
            terminal_payoff_value=failure_rate_payoff,
            terminal_boundary_functions=_acrobot_terminal_boundary_functions,
            oracle_callbacks=_acrobot_oracle_callbacks,
            upright=_acrobot_upright,
        )
    )


_register_in_house()


def make_oracle_plant(env: MujocoPHEnv) -> OraclePlant:
    """Bridge continuous true coefficients, generalized-force G and env.state_spec.

    Tensor callbacks preserve observation coordinates and derivative access.
    Simulator endpoints and sampled checks do not certify an exact interval
    operator, a holding-consistency bound or uniform model accuracy.
    """
    from models.ph_sac.plant import OraclePlant

    drift, diffusion = oracle_callbacks(env.config)
    input_matrix = torch.as_tensor(
        env.oracle_input_matrix(), dtype=torch.get_default_dtype()
    )
    return OraclePlant(drift, diffusion, input_matrix, state_spec=env.state_spec)

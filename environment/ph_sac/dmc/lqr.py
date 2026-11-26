"""dm_control's lqr/lqr_2_1 and lqr/lqr_6_2 as the testbeds lqr_2_1 and lqr_6_2.

dm_control (suite/lqr.xml, lqr.py): n bodies (2 or 6), spheres of radius 0.1 at density
1000 (4.188790204786391 kg), each on a slide joint along +y NESTED in the previous body,
so qpos are relative displacements: body k sits at y_k = q_1 + ... + q_k and the constant
mass matrix is M = m S^T S with S the lower-triangular ones, M_ij = m (n - max(i, j))
(0-based). Each joint has a spring of stiffness drawn U[15, 25] and damping drawn U[0, 0]
when the model is built, body by body (stiffness, then damping) from the task's numpy
RandomState; the first m bodies (1 or 2) carry motors of gear 1 and NO control limit;
constraints disabled, gravity orthogonal to the motion; Euler at 0.03 s. Reset: qpos a
uniform point on the sphere of radius sqrt(2) (randn normalized), velocities 0. Reward
1 - |q|^2 / 2 - control_cost |u|^2 / 2 (control_cost 0.1) with u = physics.control(),
the force itself. No time limit; dm_control ends an episode when |(q, qd)| < 1e-6.

The port (rows lqr-lqr_2_1, lqr-lqr_6_2; settings and TestbedConfig in common/ph_dmc/lqr.py):
  state       (q_1..q_n, qd_1..qd_n), all travel coordinates (none wrapped), the first m
              actuated; no context.
  model       an MJCF string of n nested slides with plant_parameters' stiffness list
              (default dm_control's seed-0 draw), body_mass and damping, motors in newtons
              (gear 1, ctrlrange +-20 N, the table's limit: dm_control's are unlimited),
              contact and constraints disabled, the config's 1 ms RK4.
  reward      1 - |q|^2 / 2 - control_cost |u|^2 / 2 with u the torque (gear 1); it can be
              negative; no failure set (all-False mask, payoff 0, H = 0, B = 1).
  episode     capped at 10 s; no termination near the origin (dm_control's |s| < 1e-6).
  clock       dm_control's 30 ms, fixed (every row); the optional testbed_settings
              CLOCK_SETTINGS_KEYS (common/ph_dmc/_common.py, with_clock_settings) give a
              two-tailed clock on [T/5, 3T] = [6 ms, 90 ms], with or without torque_limit.
  oracle      linear and closed form: qdd = M^-1 (G u - K q - D qd), K = diag(stiffness),
              D = damping I, G the first m unit columns, M^-1 = (I - N)(I - N)^T / m with N
              the subdiagonal shift (S^-1 = I - N); zero diffusion. No solve in the
              callbacks, so torch.func's forward mode is exact.
  upright     None: no upright pose (dm_control's sparse success |s| <= 0.01 is not used).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Literal, Mapping, Optional, Tuple

import mujoco
import numpy as np
import torch

from common.ph_dmc.lqr import (
    PLANT_KEYS,
    RESET_KEYS,
    REWARD_KEYS,
    lqr_2_1_config,
    lqr_6_2_config,
    lqr_stiffness,
)
# The irregular clocks (optional testbed_settings, as the other dm_control ports; _common.py).
from common.ph_dmc._common import CLOCK_SETTINGS_KEYS, with_clock_settings

from ..base import MujocoPHEnv, validated_state_spec
from ..registry import TestbedSpec, register_testbed
from ._common import (
    failure_rate_payoff,
    limit_boundary_functions,
    limit_failure_mask,
    require_keys,
    zero_diffusion,
)

if TYPE_CHECKING:
    from mujoco import MjModel
    from numpy import ndarray
    from torch import Tensor

    from common.ph_config import TestbedConfig

SETTINGS_KEYS = ("failure_limits", "reward_parameters", "reset_parameters", "plant_parameters")
SPHERE_RADIUS = 0.1

# dm_control's lqr.xml with explicit inertials (mass from plant_parameters), the springs
# written per joint, and the motors limited to the table's torque range.
_LQR_XML = """
<mujoco model="{name}">
  <option timestep="{timestep}" integrator="{integrator}" gravity="0 0 -9.81">
    <flag contact="disable" constraint="disable"/>
  </option>
  <visual><global offwidth="1280" offheight="960"/></visual>
  <default>
    <joint type="slide" axis="0 1 0" limited="false"/>
    <geom type="sphere" size="{radius}" mass="0" contype="0" conaffinity="0"/>
  </default>
  <worldbody>
    <light name="light" pos="0 0 2"/>
    <camera name="cam0" pos="-1.428 -0.311 0.856" xyaxes="0.099 -0.995 0.000 0.350 0.035 0.936"/>
    <geom name="floor" type="plane" size="4 1 .2" rgba=".2 .3 .4 1"/>
{bodies}
  </worldbody>
  <actuator>
{motors}
  </actuator>
</mujoco>
"""


def _bodies_xml(stiffness, damping: float, mass: float) -> str:
    """The nested bodies: body_0 at (.25, 0, .1) in the world, each next at (.25, 0, 0)."""
    inertia = 0.4 * mass * SPHERE_RADIUS**2
    text = ""
    for body in reversed(range(len(stiffness))):
        position = ".25 0 .1" if body == 0 else ".25 0 0"
        text = (
            f'<body name="body_{body}" pos="{position}">'
            f'<joint name="joint_{body}" stiffness="{float(stiffness[body])!r}" '
            f'damping="{float(damping)!r}"/>'
            f'<inertial pos="0 0 0" mass="{float(mass)!r}" '
            f'diaginertia="{inertia!r} {inertia!r} {inertia!r}"/>'
            f'<geom name="geom_{body}"/>{text}</body>'
        )
    return "    " + text


def build_lqr_model(config: TestbedConfig) -> MjModel:
    """Compile the nested-slide chain of the config, motors on the first m joints."""
    plant = config["plant_parameters"]
    controller = config["controller"]
    n_bodies, n_actuated = int(plant["n_bodies"]), int(plant["n_actuated"])
    stiffness = lqr_stiffness(plant, n_bodies)
    motors = "\n".join(
        f'    <motor name="motor_{index}" joint="joint_{index}" gear="1" ctrllimited="true" '
        f'ctrlrange="{float(controller["torque_low"][index])!r} '
        f'{float(controller["torque_high"][index])!r}"/>'
        for index in range(n_actuated)
    )
    return mujoco.MjModel.from_xml_string(
        _LQR_XML.format(
            name=config["name"],
            timestep=config["physics_dt"],
            integrator=config["integrator"],
            radius=SPHERE_RADIUS,
            bodies=_bodies_xml(stiffness, float(plant["damping"]), float(plant["body_mass"])),
            motors=motors,
        )
    )


def lqr_mass_matrix(n_bodies: int, body_mass: float) -> ndarray:
    """M = m S^T S, M_ij = m (n - max(i, j)) for 0-based i, j."""
    index = np.arange(n_bodies)
    return body_mass * (n_bodies - np.maximum(index[:, None], index[None, :])).astype(np.float64)


def lqr_inverse_mass_matrix(n_bodies: int, body_mass: float) -> ndarray:
    """M^-1 = S^-1 S^-T / m with S^-1 = I - N (N the subdiagonal shift): tridiagonal."""
    difference = np.eye(n_bodies) - np.eye(n_bodies, k=-1)
    return difference @ difference.T / body_mass


def lqr_linear_dynamics(config: TestbedConfig) -> Tuple[ndarray, ndarray, ndarray]:
    """(A_q, A_v, B) with qdd = A_q q + A_v qd + B u: -M^-1 K, -M^-1 D, M^-1 G."""
    plant = config["plant_parameters"]
    n_bodies, n_actuated = int(plant["n_bodies"]), int(plant["n_actuated"])
    inverse = lqr_inverse_mass_matrix(n_bodies, float(plant["body_mass"]))
    stiffness = np.diag(lqr_stiffness(plant, n_bodies))
    damping = float(plant["damping"]) * np.eye(n_bodies)
    inputs = np.eye(n_bodies)[:, :n_actuated]
    return -inverse @ stiffness, -inverse @ damping, inverse @ inputs


def lqr_state_matrices(config: TestbedConfig) -> Tuple[ndarray, ndarray]:
    """(A, B) of the first-order system d(q, qd)/dt = A (q, qd) + B u."""
    position_gain, velocity_gain, inputs = lqr_linear_dynamics(config)
    n_bodies, n_actuated = inputs.shape
    state = np.block(
        [[np.zeros((n_bodies, n_bodies)), np.eye(n_bodies)], [position_gain, velocity_gain]]
    )
    return state, np.vstack((np.zeros((n_bodies, n_actuated)), inputs))


class LQREnv(MujocoPHEnv):
    """dm_control's LQR chain of nested slides, its motors limited to the table's +-20 N.

    The state is (q_1..q_n, qd_1..qd_n), relative displacements and their rates.
    """

    def __init__(
        self,
        config: TestbedConfig,
        *,
        decision_mode: Literal["torque", "feedback"] = "feedback",
        seed: Optional[int] = None,
    ) -> None:
        require_keys(config["plant_parameters"], PLANT_KEYS, "LQR plant_parameters")
        require_keys(config["reward_parameters"], REWARD_KEYS, "LQR reward_parameters")
        require_keys(config["reset_parameters"], RESET_KEYS, "LQR reset_parameters")
        plant = config["plant_parameters"]
        self.n_bodies, self.n_actuated = int(plant["n_bodies"]), int(plant["n_actuated"])
        spec = validated_state_spec(config)
        if (spec["position_dim"], spec["velocity_dim"], spec["action_dim"]) != (
            self.n_bodies, self.n_bodies, self.n_actuated
        ):
            raise ValueError("The LQR state spec must match n_bodies and n_actuated.")
        super().__init__(
            config, build_lqr_model(config), spec, decision_mode=decision_mode, seed=seed
        )
        self._position_gain, self._velocity_gain, self._input_gain = lqr_linear_dynamics(config)
        self._control_cost = float(config["reward_parameters"]["control_cost"])

    def _reset_state(self, options: Optional[Mapping[str, object]] = None) -> None:
        """dm_control's start: q uniform on the sphere of position_radius, qd noise (0)."""
        if options and "state" in options:
            self._set_state(options["state"])
            return
        reset = self.config["reset_parameters"]
        direction = self._reset_rng.standard_normal(self.n_bodies)
        positions = reset["position_radius"] * direction / np.linalg.norm(direction)
        velocities = reset["velocity_noise"] * self._reset_rng.standard_normal(self.n_bodies)
        self._set_state(np.concatenate((positions, velocities)))

    def reward_rate(self, state: ndarray, torque: ndarray) -> float:
        """dm_control's 1 - |q|^2 / 2 - control_cost |u|^2 / 2, u the force (gear 1)."""
        positions = np.asarray(state, dtype=np.float64)[: self.n_bodies]
        control = np.asarray(torque, dtype=np.float64).reshape(-1)
        return float(
            1.0
            - 0.5 * float(positions @ positions)
            - 0.5 * self._control_cost * float(control @ control)
        )

    def failed(self, state: ndarray) -> bool:
        """No failure set."""
        return False

    def terminal_payoff(self, state: ndarray) -> float:
        return lqr_terminal_payoff_value(self.config)

    def oracle_drift(self, state: ndarray, torque: ndarray) -> ndarray:
        state = np.asarray(state, dtype=np.float64)
        positions, velocities = state[: self.n_bodies], state[self.n_bodies :]
        control = np.asarray(torque, dtype=np.float64).reshape(-1)
        accelerations = (
            self._position_gain @ positions
            + self._velocity_gain @ velocities
            + self._input_gain @ control
        )
        return np.concatenate((velocities, accelerations))

    def oracle_diffusion(self, state: ndarray) -> ndarray:
        """The LQR has no noise."""
        return np.zeros((self.state_dim, 1))


def lqr_reward_rate_tensor(states: Tensor, torques: Tensor, config: TestbedConfig) -> Tensor:
    """1 - |q|^2 / 2 - control_cost |u|^2 / 2 on tensor batches, [B,1]."""
    n_bodies = int(config["plant_parameters"]["n_bodies"])
    control_cost = float(config["reward_parameters"]["control_cost"])
    positions = states[..., :n_bodies]
    return (
        1.0
        - 0.5 * positions.square().sum(dim=-1, keepdim=True)
        - 0.5 * control_cost * torques.square().sum(dim=-1, keepdim=True)
    )


def lqr_failure_mask_tensor(states: Tensor, config: TestbedConfig) -> Tensor:
    """No failure set: all False [B]."""
    return limit_failure_mask(states, ())


def lqr_terminal_payoff_value(config: TestbedConfig) -> float:
    """0: no failure set (failure_rate_payoff without a failure_rate)."""
    return failure_rate_payoff(config)


def lqr_terminal_boundary_functions(
    config: TestbedConfig,
) -> Tuple[Callable[[Tensor], Tensor], Callable[[Tensor], Tensor]]:
    """H = 0, B = 1."""
    return limit_boundary_functions(lqr_terminal_payoff_value(config), ())


def lqr_oracle_callbacks(
    config: TestbedConfig,
) -> Tuple[Callable[[Tensor, Tensor], Tensor], Callable[[Tensor], Tensor]]:
    """The linear drift (qd, A_q q + A_v qd + B u) and a zero diffusion; torch.func-safe."""
    position_gain, velocity_gain, input_gain = lqr_linear_dynamics(config)
    n_bodies = position_gain.shape[0]

    def lqr_drift(states: Tensor, actions: Tensor) -> Tensor:
        def matrix(values: ndarray) -> Tensor:
            return torch.as_tensor(values, dtype=states.dtype, device=states.device)

        positions, velocities = states[..., :n_bodies], states[..., n_bodies:]
        accelerations = (
            positions @ matrix(position_gain).T
            + velocities @ matrix(velocity_gain).T
            + actions.to(states.dtype) @ matrix(input_gain).T
        )
        return torch.cat((velocities, accelerations), dim=-1)

    return lqr_drift, zero_diffusion(2 * n_bodies)


for _name, _factory in (("lqr_2_1", lqr_2_1_config), ("lqr_6_2", lqr_6_2_config)):
    register_testbed(
        TestbedSpec(
            name=_name,
            env_class=LQREnv,
            config_factory=with_clock_settings(_factory),
            settings_keys=SETTINGS_KEYS,
            # Optional motor limit (default common.ph_dmc.lqr.LQR_TORQUE_LIMIT, every row;
            # the lqr_6_2 diagnostic of 2026-11-11 runs 2 N) and the optional irregular
            # clock (absent: the fixed 30 ms clock, every row).
            optional_settings_keys=("torque_limit",) + CLOCK_SETTINGS_KEYS,
            reward_rate_tensor=lqr_reward_rate_tensor,
            failure_mask_tensor=lqr_failure_mask_tensor,
            terminal_payoff_value=lqr_terminal_payoff_value,
            terminal_boundary_functions=lqr_terminal_boundary_functions,
            oracle_callbacks=lqr_oracle_callbacks,
            upright=None,
            # 1 - |q|^2/2 - c|u|^2/2 is unbounded below: the learned reward stays linear.
            reward_rate_bounds=None,
            # The testbed_settings key discount_rate (_common.py, with_discount_rate).
            discount_setting=True,
        )
    )

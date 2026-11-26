"""Config factories of the dm_control acrobot ports (acrobot_swingup, acrobot_swingup_sparse).

Each factory takes the testbed_settings of its table row (benchmarks/hyperparams/
ct_ph_sac.csv, the env_* cells) as keyword arguments and returns the TestbedConfig whose
fields the row's task_* cells declare. The two tasks differ only in the reward
(reward_parameters): acrobot_swingup carries Xin and Kaneda's Lyapunov reward, whose
plant constants the row records, acrobot_swingup_sparse dm_control's indicator; they
share every other field. The environment, the tensor task
functions and the registration live in environment/ph_sac/dmc/acrobot.py;
environment/ph_sac/dmc/README.md describes the contract. Import common.ph_config only
under TYPE_CHECKING here (it imports this package).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Mapping

from common.ph_dmc._common import box_rule_controller, dmc_state_spec, dmc_testbed_config

if TYPE_CHECKING:
    from common.ph_config import TestbedConfig

# dm_control's acrobot: the elbow motor has gear 2 with control in [-1, 1], so +-2 N m.
ACROBOT_TORQUE_LIMIT = 2.0
# The elbow setpoint offset of the decision box (rad).
ACROBOT_SETPOINT_OFFSET = 1.0
# dm_control's control step (its RK4 physics step, one substep) and time limit.
ACROBOT_CONTROL_PERIOD = 0.01
ACROBOT_EPISODE_DURATION = 10.0
# The joints whose damping a "damping" physical change scales: both hinges.
ACROBOT_DAMPING_KEYS = ("shoulder_damping", "elbow_damping")


def acrobot_physical_change(
    plant_parameters: Mapping[str, float], kind: str, scale: float
) -> dict:
    """The plant_parameters of acrobot_swingup's changed robot (the adaptation experiment;
    common.ph_dmc._common.with_physical_change).

    "length": the lower link's length l2 times scale, its mass m2 and radius radius2 kept.
    The model builds the link as dm_control does, a capsule from the elbow to l2 carrying
    the mass m2, so MuJoCo compiles its centre of mass (l2 / 2) and its inertia from the
    new geometry, and the tip site sits at l2: the mass matrix and the gravity potential
    change. "damping": the shoulder's and the elbow's damping times scale, nothing else.
    """
    changed = {key: value for key, value in plant_parameters.items()}
    if kind == "length":
        changed["l2"] = float(plant_parameters["l2"]) * float(scale)
    elif kind == "damping":
        for key in ACROBOT_DAMPING_KEYS:
            changed[key] = float(plant_parameters[key]) * float(scale)
    else:
        raise ValueError(f"acrobot_swingup has no physical change {kind!r}.")
    return changed


def _acrobot_config(
    name: str,
    *,
    failure_limits: Mapping[str, float],
    reward_parameters: Mapping[str, float],
    reset_parameters: Mapping[str, object],
    plant_parameters: Mapping[str, float],
) -> TestbedConfig:
    """The TestbedConfig both acrobot rows declare, under the testbed name given."""
    return dmc_testbed_config(
        name=name,
        state_spec=dmc_state_spec(physical_dim=2, action_dim=1, periodic=(0, 1), actuated=(1,)),
        reference_period=ACROBOT_CONTROL_PERIOD,
        episode_duration=ACROBOT_EPISODE_DURATION,
        controller=box_rule_controller(
            offsets=(ACROBOT_SETPOINT_OFFSET,),
            torque_limit=ACROBOT_TORQUE_LIMIT,
            setpoint_mode="offset",
            setpoint_error="wrapped",
        ),
        plant_parameters=plant_parameters,
        failure_limits=failure_limits,
        reward_parameters=reward_parameters,
        reset_parameters=reset_parameters,
    )


def acrobot_swingup_config(
    *,
    failure_limits: Mapping[str, float],
    reward_parameters: Mapping[str, float],
    reset_parameters: Mapping[str, object],
    plant_parameters: Mapping[str, float],
) -> TestbedConfig:
    """Declare acrobot/swingup with the Xin--Kaneda Lyapunov reward (row acrobot-swingup, top).

    dm_control's acrobot/swingup plant, motor, reset and clock, with the log-reciprocal of
    Xin and Kaneda's energy-and-elbow Lyapunov function without its elbow-rate term (X. Xin
    and M. Kaneda, Int. J. Robust Nonlinear Control 17(16):1503-1524, 2007), the reward of
    the in-house Acrobot-XK, in place of dm_control's tolerance reward (the formula and
    the choices are in environment/ph_sac/dmc/acrobot.py's docstring). The row passes
    these testbed_settings (its env_* cells):
        failure_limits     {}
        reward_parameters  energy_scale=39.24;upright_energy=19.62;
                           lyapunov_k_p=15.331767469904836;reward_floor=1e-06
        reset_parameters   angle_range=3.141592653589793;velocity_noise=0.0
        plant_parameters   l1=1.0;l2=1.0;m1=1.0;m2=1.0;radius1=0.05;radius2=0.049;
                           shoulder_damping=0.05;elbow_damping=0.05
    (plant_parameters may add gravity, default dm_control's 9.81). The reward is
    r(s) = -log(max(V(s) / V_down, reward_floor)) with
    V = (E - upright_energy)^2 / 2 + lyapunov_k_p wrap(theta_2)^2 / 2 and
    V_down = energy_scale^2 / 2, E the mechanical energy of the plant's two-link chain
    (potential zero with both links horizontal). The constants are this plant's:
    upright_energy = E at upright rest = b1 + b2, energy_scale = E(upright rest) -
    E(hanging rest) = 2 (b1 + b2), with b1 = g (m1 lc1 + m2 l1) = 14.715 N m and
    b2 = g m2 lc2 = 4.905 N m, and lyapunov_k_p = 1.001 x Xin and Kaneda's lower bound
    (2/pi) min(b1^2, b2^2) = 15.3165; reward_floor 1e-6 as Acrobot-XK.
    benchmarks/make_ct_ph_sac_table.py computed them through the chain
    (environment/ph_sac/dmc/acrobot.py, acrobot_lyapunov_reward_terms) and wrote them into
    the row: the row is the record, so this factory reads them and the environment only
    checks them. The TestbedConfig reproduces the row's task_* cells: state (theta_1,
    theta_2, theta_dot_1, theta_dot_2), both angles periodic (wrapped), the elbow
    actuated; 1 ms RK4 physics; a fixed clock at dm_control's 10 ms control step; 10 s
    episodes; discount rate 0.5 /s; the box rule's decisions (elbow setpoint offset
    within +-1 rad with a wrapped error, stiffness 0-20, damping 0-4) and torques within
    +-2 N m; refresh every physics step. The settings' keys are checked by the
    environment (AcrobotSwingupEnv).

    From the first change of 2026-11-08 to the second the row carried instead the reward
    of Zhang, Sathuluri and Zimmermann, "Solving the swing-up and balance task for the
    Acrobot and Pendubot with SAC", arXiv:2312.11311 (2023), Eq. 4 and Table I (acrobot
    column), reward kind "zhang2023", which the environment still accepts. Its
    reward_parameters were
        q_angle=10.0;q_velocity=0.2;r_torque=0.0001;line_bonus=500.0;
        line_fraction=0.8;lqr_bonus=10000.0;velocity_penalty=10000.0;
        velocity_threshold=8.0;scale=0.0001;rho=0.1468853616463693;
        s11=2460.646551158671;s12=682.4822519369063;
        s13=907.6439942361637;s14=282.0701102754566;
        s22=190.36706529392856;s23=251.85979713893715;
        s24=78.34549651978381;s33=334.8981452316153;
        s34=104.08074616907096;s44=32.36377880183468
    (rho and the upper triangle of S, row by row): the paper's Q = diag(q_angle, q_angle,
    q_velocity, q_velocity) on the wrapped error from the upright, R = r_torque on the
    elbow torque in N m, the line bonus while the tip height is at least line_fraction
    (l1 + l2), the LQR bonus while e^T S e <= rho, the penalty for each rate of magnitude
    at least velocity_threshold, all times scale = 1e-4. S is this plant's LQR cost-to-go
    at the upright for the paper's Q_lqr = diag(0.97, 0.93, 0.39, 0.26), R_lqr = 0.11, and
    rho a simulated region of attraction of the saturated LQR u = clip(-K e, -2, 2): 0.9 x
    the largest level of e^T S e at which 512 seeded starts on and inside the level set
    all converge within 5 s of 1 ms RK4 on the port's dynamics, confirmed on 2048
    independent starts (environment/ph_sac/dmc/acrobot_lqr.py, acrobot_lqr_reward_terms).
    Every curve stayed flat near a mean rate of -0.085 with no upright in 40 runs.
    """
    return _acrobot_config(
        "acrobot_swingup",
        failure_limits=failure_limits,
        reward_parameters=reward_parameters,
        reset_parameters=reset_parameters,
        plant_parameters=plant_parameters,
    )


def acrobot_swingup_sparse_config(
    *,
    failure_limits: Mapping[str, float],
    reward_parameters: Mapping[str, float],
    reset_parameters: Mapping[str, object],
    plant_parameters: Mapping[str, float],
) -> TestbedConfig:
    """Declare dm_control's acrobot/swingup_sparse (table row acrobot-swingup_sparse, mode top).

    The row passes the settings of acrobot_swingup_config except
        reward_parameters  target_radius=0.2;margin=0.0
    (dm_control's reward, tolerance(|tip - target|, (0, target_radius), margin) with
    margin 0: the indicator of the tip within the target ball), and the TestbedConfig
    equals acrobot_swingup_config's apart from the name and the reward parameters.
    """
    return _acrobot_config(
        "acrobot_swingup_sparse",
        failure_limits=failure_limits,
        reward_parameters=reward_parameters,
        reset_parameters=reset_parameters,
        plant_parameters=plant_parameters,
    )

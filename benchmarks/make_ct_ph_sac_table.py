"""Write benchmarks/hyperparams/ct_ph_sac.csv: python benchmarks/make_ct_ph_sac_table.py.

Run from any folder; the repository is this file's parent's parent (until 2026-11-10 a
fixed local path, so the generator ran only on the workstation). The CSV is generated,
never edited by hand.
"""

import copy
import csv
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from benchmarks.ph_hyperparams import TABLE_COLUMNS, config_to_row, table_path  # noqa: E402
from benchmarks.run_ph_sac import _build_testbed  # noqa: E402
from common.ph_config import (  # noqa: E402
    decision_box_entropy_target,
    refresh_mode_setting,
    testbed_model_coarse_step,
    testbed_model_step,
)

# The model integrator that is each physics integrator's own step (tests/
# test_ph_model_integrator.py checks every row against it).
MODEL_INTEGRATOR_OF = {"RK4": "rk4", "Euler": "semi_implicit"}

CONFIGS = ROOT / "benchmarks" / "configs" / "ph_sac"
PI = math.pi

# The learned plant's online-step schedule of every dm_control row (training keys
# plant_step_dense_until and plant_step_every; common.ph_config.plant_step_schedule): a
# step on every row until 100k decisions, then on every 4th row. Proposed 2026-11-10,
# CONFIRMED 2026-11-11 after the exclude_thin experiment (Vista job 1135664: no measurable
# change in plant error or score from 100k to 200k decisions on cartpole-swingup,
# point_mass-easy and lqr_6_2, about a third less time per decision). To revert, set it to
# {} (a step on every row, the default) and delete the two keys from
# cartpole_swingup.json's training, which must equal it (checked below; the
# cartpole-swingup row is that file). The windy and Acrobot-XK rows never carry it (dense).
PLANT_STEP_SCHEDULE = {"plant_step_dense_until": 100_000, "plant_step_every": 4}
PLANT_STEP_KEYS = ("plant_step_dense_until", "plant_step_every")
# Contact exclusion per row (top-level key contact_handling, column model_contact_handling;
# common.ph_config.CONTACT_HANDLINGS): env_id -> "exclude". A listed row gets
# contact_handling "exclude" and no limit port (the plant is the free mechanism); the
# other rows with joint limits (reachers, swimmers) keep the limit port ("port", the
# default, an empty cell). Decided 2026-11-11 for the cart-pole family and the point mass
# after the exclusion experiment of 2026-11-10 (Vista $S/exclude/2026-11-10, job 1135664):
# the same final scores as the port at 3-5x less time per decision. cartpole-swingup's row
# is cartpole_swingup.json, which must agree (checked below). A run selects the other
# scheme with run_ph_sac.py --contact-handling port|exclude.
CONTACT_HANDLING = {
    env_id: "exclude"
    for env_id in (
        "cartpole-balance", "cartpole-balance_sparse", "cartpole-swingup",
        "cartpole-swingup_sparse", "cartpole-two_poles", "cartpole-three_poles",
        "point_mass-easy",
    )
}
# Top rows that carry an optional reward variant's reward in place of their task's own
# (env_id -> variant mode of VARIANTS below). That variant then gets no row of its own (it
# would repeat the top row but for its output directory). Empty: every top row keeps its
# task's reward. (On 2026-11-11 the two- and three-pole rows briefly took lyapunov_cart's;
# this was reverted the same day after re-scoring showed variant A came no closer to
# balancing: both keep dm_control's smooth reward, and lyapunov_cart is an unused variant.)
TOP_REWARD_VARIANT = {}
# Status of rows left out of the final run's launch plan (decided 2026-11-11); the rows
# stay complete and runnable.
ROW_STATUS = {
    "swimmer-swimmer6": "Status 2026-11-11: dropped from the final run (left out of its "
    "launch plan); the row stays complete and runnable.",
    "swimmer-swimmer15": "Status 2026-11-11: dropped from the final run (left out of its "
    "launch plan); the row stays complete and runnable.",
    "reacher-easy": "Status 2026-11-11: set aside, not in the final run's launch plan; the "
    "row stays complete and runnable.",
    "reacher-hard": "Status 2026-11-11: set aside, not in the final run's launch plan; the "
    "row stays complete and runnable.",
}


def plant_step_note():
    if not PLANT_STEP_SCHEDULE:
        return "Plant step on every row (the default schedule)."
    return (
        "Plant-step schedule (confirmed 2026-11-11): the learned plant takes its online "
        "step on every row until "
        f"{PLANT_STEP_SCHEDULE['plant_step_dense_until']} decisions, then on every k-th "
        f"row, k = {PLANT_STEP_SCHEDULE['plant_step_every']} (training "
        "plant_step_dense_until, plant_step_every; make_ct_ph_sac_table.PLANT_STEP_SCHEDULE)."
    )


def with_plant_step_schedule(training):
    """training without its plant-step keys, then with PLANT_STEP_SCHEDULE's."""
    training = {k: v for k, v in training.items() if k not in PLANT_STEP_KEYS}
    training.update(PLANT_STEP_SCHEDULE)
    return training


def load(name):
    with open(CONFIGS / name, encoding="utf-8") as handle:
        raw = json.load(handle)
    raw.pop("_experiment_choices", None)
    return raw


def declared_from_factory(raw):
    t = _build_testbed(raw["testbed"], raw["testbed_settings"])
    spec, clock, ctl = t["state_spec"], t["clock"], t["controller"]
    return {
        "nq": spec["position_dim"],
        "nv": spec["velocity_dim"],
        "nu": spec["action_dim"],
        "context_dim": 0,
        "periodic_position_indices": list(spec["periodic_position_indices"]),
        "actuated_position_indices": list(spec["actuated_position_indices"]),
        "physics_dt": t["physics_dt"],
        "integrator": t["integrator"],
        "reference_period": t["reference_period"],
        "clock_kind": clock["kind"],
        "min_duration": clock["min_duration"],
        "max_duration": clock["max_duration"],
        "episode_duration": t["episode_duration"],
        "discount_rate": t["discount_rate"],
        "decision_low": [float(x) for x in ctl["decision_low"]],
        "decision_high": [float(x) for x in ctl["decision_high"]],
        "torque_low": [float(x) for x in ctl["torque_low"]],
        "torque_high": [float(x) for x in ctl["torque_high"]],
        "setpoint_mode": ctl["setpoint_mode"],
        "setpoint_error": ctl["setpoint_error"],
        "refresh_steps": ctl["refresh_steps"],
        "refresh_mode": refresh_mode_setting(ctl),
    }


SEE = "its _experiment_choices document every setting"
IN_HOUSE = [
    (
        "windy_pendulum-balance", "top", "windy_pendulum.json",
        "Ported (testbed windy_pendulum, task balance). Equals "
        f"benchmarks/configs/ph_sac/windy_pendulum.json; {SEE}. The paper's in-house windy "
        "pendulum: two-tailed uniform clock on [2, 18] ms, 1 ms Euler physics (model "
        "integrator semi_implicit, the same step), 12 N m motor, "
        "angle failure at pi/2; not a dm_control task.",
    ),
    (
        "windy_pendulum-balance", "smoke", "smoke_windy.json",
        "Equals benchmarks/configs/ph_sac/smoke_windy.json: the windy balance task at "
        "smoke-test sizes (1500 decisions, one training seed, two reset seeds, small "
        "networks, few steps); a pipeline check, not an experiment.",
    ),
    (
        "windy_pendulum-swingup_tq5", "top", "windy_swingup_tq5.json",
        "Ported (testbed windy_pendulum, task swingup, torque_limit 5 N m). Equals "
        f"benchmarks/configs/ph_sac/windy_swingup_tq5.json; {SEE}. In-house task, not "
        "dm_control's pendulum (see pendulum-swingup). Model integrator semi_implicit, the "
        "1 ms Euler physics' own step.",
    ),
    (
        "windy_pendulum-swingup_tq12", "top", "windy_swingup_tq12.json",
        "Ported (testbed windy_pendulum, task swingup, torque_limit 12 N m). Equals "
        f"benchmarks/configs/ph_sac/windy_swingup_tq12.json; {SEE}. In-house task, not "
        "dm_control's pendulum (see pendulum-swingup). Model integrator semi_implicit, the "
        "1 ms Euler physics' own step.",
    ),
    (
        "acrobot_xk-swingup", "top", "acrobot_xk.json",
        "Ported (testbed acrobot_xk). Equals benchmarks/configs/ph_sac/acrobot_xk.json; "
        f"{SEE}. The in-house Xin--Kaneda Acrobot (second link 2 m, r1 Lyapunov reward, "
        "failure limits on the elbow angle and both rates, 20 s episodes, 800k budget), not "
        "dm_control's acrobot (see acrobot-swingup).",
    ),
    (
        "cartpole-swingup", "top", "cartpole_swingup.json",
        "Ported (testbed cartpole_swingup). Equals "
        f"benchmarks/configs/ph_sac/cartpole_swingup.json; {SEE}. dm_control "
        "cartpole/swingup with dm_control's soft rail limit KEPT (slider limited to "
        "+-1.8 m, solreflimit 0.08 1, MuJoCo's default solimplimit; since 2026-11-10, no "
        "failure set: before, |x| = 1.8 m was a failure with payoff 0) and contact "
        "exclusion in place of the learned plant's limit port (model_contact_handling "
        "exclude, no network limit_port, since 2026-11-11; below); 1 ms RK4 physics, fixed "
        "10 ms clock (dm_control's control step), 1000 decisions "
        "per 10 s episode. Model step 1 ms, the physics step (ten substeps per decision; "
        "common.ph_config.MODEL_STEPS, as every row's: the model step is coarsened only on "
        "tasks without joint limits), and model_coarse_step 0.005 "
        "(common.ph_config.MODEL_COARSE_STEPS), the step of every model integration under "
        "contact exclusion (below; contact-triggered fine substepping, which re-integrates "
        "intervals near the rail at 1 ms, applies only with the limit port). The method "
        "columns of every other dm_control row copy this row except model_step, "
        "model_integrator, model_coarse_step, model_contact_handling and model_limit_port, "
        "which are per task.",
    ),
]

# ----------------------------------------------------------------- dm_control rows

# Decision box per actuator: setpoint (offset or absolute) within +-OFFSET, stiffness in
# [0, k_max] and damping in [0, 0.2 k_max], with k_max = 10 tau_max / OFFSET so that the
# spring torque at the full offset is ten times the motor limit. This reproduces the
# reference boxes (cart-pole and Acrobot: +-1, 0-100, 0-20 at +-10) and scales the others
# to their motors; the damping-to-stiffness ratio 0.2 s is the reference boxes' as well.
STIFFNESS_OVER_LIMIT = 10.0
DAMPING_RATIO = 0.2


def scaled_box(offset, torque, actuators=1):
    k_max = STIFFNESS_OVER_LIMIT * torque / offset
    low = [-offset] * actuators + [0.0] * actuators + [0.0] * actuators
    high = [offset] * actuators + [k_max] * actuators + [DAMPING_RATIO * k_max] * actuators
    return (low, high)


def box_note(offset, torque, unit_q, unit_tau):
    k_max = STIFFNESS_OVER_LIMIT * torque / offset
    return (
        f"Decision box scaled to the motor: setpoint within +-{offset:g} {unit_q}, stiffness "
        f"0-{k_max:g} {unit_tau}/{unit_q}, damping 0-{DAMPING_RATIO * k_max:g} {unit_tau} s/{unit_q} "
        f"(k_max = 10 x {torque:g} {unit_tau} / {offset:g} {unit_q}, damping 0.2 s x k_max, the "
        "rule that gives the cart-pole and Acrobot reference boxes at their +-10 limit)"
    )

CARTPOLE_PLANT = load("cartpole_swingup.json")["testbed_settings"]["plant_parameters"]
CARTPOLE_SMOOTH = {"cart_margin": 2.0, "velocity_margin": 5.0, "failure_rate": 0.0}
CARTPOLE_SPARSE = {"cart_bound": 0.25, "cosine_bound": 0.995, "failure_rate": 0.0}
CARTPOLE_SWINGUP_RESET = load("cartpole_swingup.json")["testbed_settings"]["reset_parameters"]
CARTPOLE_BALANCE_RESET = {"position_range": 0.1, "angle_range": 0.034, "velocity_noise": 0.01}
WRIST = math.radians(160.0)
# MuJoCo's default solreflimit and solimplimit, which dm_control's reacher and point_mass
# XML leave in place for their limited joints (common.ph_dmc._common).
from common.ph_dmc._common import MUJOCO_DEFAULT_LIMIT  # noqa: E402

def kept_limit_note(env_id):
    """The kept soft limit, and how the row's learned plant treats it (CONTACT_HANDLING)."""
    head = (
        "No failure set since 2026-11-10 (before, the stop was a failure with payoff 0 and "
        "the boundary lift): the limit stays MuJoCo's soft constraint, the oracle reproduces "
        "its force (environment/ph_sac/dmc/joint_limits.py) and "
    )
    if CONTACT_HANDLING.get(env_id) == "exclude":
        return head + (
            "the learned plant has no limit port: contact exclusion keeps the plant off the "
            "limit (model_contact_handling exclude, since 2026-11-11; below)."
        )
    return head + (
        "the learned plant carries the limit port (network limit_port, "
        "limited_position_ranges)."
    )

COMMON = (
    "Method and protocol columns as cartpole-swingup (cartpole_swingup.json) but the "
    "model step, the model integrator and the contact scheme's coarse step: K 400, tau "
    "1 s, N 4, rate/actor steps 200/40, drift-only plant "
    "(diffusion none: the task is deterministic) integrated by the simulator's own step "
    "(model integrator rk4 on the RK4 rows, semi_implicit, MuJoCo's Euler step, on the "
    "swimmers' Euler), learned reward, always-admit, warm steps 2000/2000, "
    "leading_order, "
    "alpha_c tuned from 0.1 /s every 40 rows, 10 seeds, evaluation every 20k decisions on "
    "reset seeds 20000-20031; damping_rank is the velocity dimension as there, and the "
    "target entropy is decision_box_entropy_target of this row's box. Discount rate "
    "0.5 /s as the swing-up tasks (dm_control and the paper set none). Feedback law "
    "evaluated at every RK4 stage, once per physics step at its start under Euler "
    "(task_refresh_mode continuous, since 2026-11-09; the "
    "runs before held it through each physics step) and terminal payoff 0 (failure_rate "
    "0) as cartpole-swingup."
)
BUDGET_10 = (
    "Budget 500k decisions as cartpole-swingup (500 episodes of 1000 decisions)."
)
BUDGET_20 = (
    "Budget 500k decisions, the same number of decisions as cartpole-swingup (500 "
    "episodes of 1000 decisions; at 20 ms per decision this is 10000 s of interaction "
    "against the 10-s tasks' 5000 s)."
)
CLOCK_10 = "1 ms RK4 physics as the cart-pole port; fixed clock at dm_control's 10 ms control step (the reference period), 1000 decisions per 10 s episode."
CLOCK_20 = "1 ms RK4 physics as the cart-pole port (dm_control: 20 ms); fixed clock at dm_control's 20 ms control step (the reference period), 1000 decisions per 20 s episode."


def ported(testbed, compared="forward dynamics, reward, reset law and open-loop trajectories"):
    domain = testbed.split("_")[0] if not testbed.startswith("point_mass") else "point_mass"
    return (
        f"Ported: testbed {testbed} is environment/ph_sac/dmc/{domain}.py with its factory "
        f"in common/ph_dmc/{domain}.py; the row builds, its task_* columns are checked "
        "against the factory, and tests/test_ph_dmc_*.py compare the port with dm_control "
        f"({compared}). Task values from "
        "docs/dmc_environments/dmc_environments.tex and docs/paper/sections/E_experiment_details.tex."
    )


def dmc(env_id, testbed, *, nq, nu, periodic, actuated, period, episode, box, torque,
        setpoint_mode, setpoint_error, failure, reward, reset, plant, context=0,
        notes, budget_note, compared=None, network=None, physics_dt=0.001,
        integrator="RK4"):
    return {
        "env_id": env_id,
        "testbed": testbed,
        "network": dict(network or {}),
        "settings": {
            "failure_limits": failure,
            "reward_parameters": reward,
            "reset_parameters": reset,
            "plant_parameters": plant,
        },
        "task": {
            "nq": nq,
            "nv": nq,
            "nu": nu,
            "context_dim": context,
            "periodic_position_indices": periodic,
            "actuated_position_indices": actuated,
            "physics_dt": physics_dt,
            "integrator": integrator,
            "reference_period": period,
            "clock_kind": "fixed",
            "min_duration": period,
            "max_duration": period,
            "episode_duration": episode,
            "discount_rate": 0.5,
            "decision_low": list(box[0]),
            "decision_high": list(box[1]),
            "torque_low": [-torque] * nu,
            "torque_high": [torque] * nu,
            "setpoint_mode": setpoint_mode,
            "setpoint_error": setpoint_error,
            "refresh_steps": 1,
            "refresh_mode": "continuous",
        },
        "comment": " ".join([
            ported(testbed) if compared is None else ported(testbed, compared),
            notes, COMMON, budget_note,
        ]),
    }



# pendulum-swingup's reward since 2026-11-09: dm_control's own tolerance with margin 2
# (gaussian, value 0.1 at the margin) in place of its margin 0 (the indicator). The
# environment computes it through _common.tolerance as before; only the row's margin changed.
PENDULUM_COSINE_BOUND = math.cos(math.radians(8.0))
PENDULUM_MARGIN = 2.0


def pendulum_reward_at(angle):
    """The row's reward rate at a pole angle (0 upright), through the port's tolerance."""
    from environment.ph_sac.dmc._common import tolerance

    return tolerance(math.cos(angle), (PENDULUM_COSINE_BOUND, 1.0), PENDULUM_MARGIN)


PENDULUM_REWARD_NOTE = (
    "Reward dense since 2026-11-09, so NOT dm_control's pendulum/swingup reward: "
    "tol(cos theta; [cos 8 deg, 1], margin 2) with dm_control's default gaussian shape "
    "(value 0.1 at the margin) where dm_control sets margin 0, the form of dm_control's "
    "own dense tolerance rewards (acrobot/swingup is acrobot/swingup_sparse's bounds with "
    "margin 1), so no new code path. It is 1 within 8 deg of upright, "
    f"{pendulum_reward_at(0.5 * PI):.3g} with the pole horizontal and "
    f"{pendulum_reward_at(PI):.3g} hanging (the margin spans the whole range of cos "
    "theta, so the reward falls to about the value at the margin at the bottom). "
    "dm_control's margin 0 (the indicator cos(theta) >= cos 8 deg) taught no arm "
    "anything in 500k decisions: with "
    "the 1 N m motor against 4.9 N m of gravity torque, a swing-up needs about 6.6 s of "
    "coherent pumping, random exploration gains 1-3% of the energy needed, and in 14,600 "
    "training episodes none entered the window from beyond 30 deg, so the value stayed "
    "flat outside the upright basin (the learned reward and plant were accurate). The "
    "reward still lies in [0, 1], so the sigmoid reward head (unit_interval_bounds) stays "
    "valid. The row remains dm_control's plant, motor, reset, clock and evaluation; its "
    "scores do not compare with published dm_control pendulum/swingup returns."
)

DMC = [
    dmc(
        "pendulum-swingup", "pendulum_swingup", nq=1, nu=1, periodic=[0], actuated=[0],
        period=0.02, episode=20.0, box=scaled_box(PI, 1.0), torque=1.0,
        setpoint_mode="absolute", setpoint_error="wrapped",
        failure={},
        reward={"cosine_bound": PENDULUM_COSINE_BOUND, "margin": PENDULUM_MARGIN},
        reset={"angle_range": PI, "velocity_noise": 0.0},
        plant={"mass": 1.0, "length": 0.5, "bob_radius": 0.05, "damping": 0.1,
               "gravity": 9.81},
        notes=(
            "dm_control pendulum/swingup: one hinge, massless 0.5 m pole with a 1 kg bob "
            "(radius 0.05 m), damping 0.1, g 9.81; motor gear 1, torque +-1 N m. "
            f"{CLOCK_20} Reset: angle uniform on [-pi, pi), velocity 0. "
            f"{PENDULUM_REWARD_NOTE} No stop, no failure "
            f"set. {box_note(PI, 1.0, 'rad', 'N m')}; the setpoint is absolute with a wrapped "
            "error, as the windy swing-up's, which is how the companion document's 'decision "
            "box as the windy pendulum's' reads for a swing-up; flagged as a choice. The port "
            "reads the bob's inertia from dm_control's compiled model."
        ),
        budget_note=BUDGET_20,
        compared=(
            "forward dynamics, reset law and open-loop trajectories, and the reward at "
            "margin 0 against dm_control's get_reward; the row's margin-2 reward is tested "
            "against dm_control's rewards.tolerance"
        ),
    ),
]

ACROBOT_PLANT = {"l1": 1.0, "l2": 1.0, "m1": 1.0, "m2": 1.0, "radius1": 0.05,
                 "radius2": 0.049, "shoulder_damping": 0.05, "elbow_damping": 0.05}
ACROBOT_RESET = {"angle_range": PI, "velocity_noise": 0.0}
# acrobot-swingup's reward from the first change of 2026-11-08 to the second, no row's
# since (reward kind zhang2023): Zhang, Sathuluri, Zimmermann (2023), arXiv:2312.11311,
# Eq. 4 and Table I (acrobot column), scaled by 1e-4; rho and S (s11..s44) are appended
# from this plant's Riccati solution and simulated region of attraction (acrobot_lqr.py).
# acrobot_zhang_reward is no longer called; it rebuilds the reward_parameters the row
# carried (recorded in common/ph_dmc/acrobot.py).
ACROBOT_ZHANG_CONSTANTS = {
    "q_angle": 10.0, "q_velocity": 0.2, "r_torque": 1e-4, "line_bonus": 500.0,
    "line_fraction": 0.8, "lqr_bonus": 1e4, "velocity_penalty": 1e4,
    "velocity_threshold": 8.0, "scale": 1e-4,
}


def acrobot_zhang_reward():
    """The paper's constants plus rho and S of the port's plant (about 40 s of rollouts)."""
    from common.ph_dmc.acrobot import acrobot_swingup_config
    from environment.ph_sac.dmc.acrobot_lqr import acrobot_lqr_reward_terms

    provisional = acrobot_swingup_config(
        failure_limits={}, reward_parameters=dict(ACROBOT_ZHANG_CONSTANTS),
        reset_parameters=dict(ACROBOT_RESET), plant_parameters=dict(ACROBOT_PLANT),
    )
    return {**ACROBOT_ZHANG_CONSTANTS, **acrobot_lqr_reward_terms(provisional)}


# acrobot-swingup's reward since the second change of 2026-11-08 (reward kind lyapunov):
# the log-reciprocal Xin--Kaneda Lyapunov reward of acrobot_xk on this plant; the floor is
# acrobot_xk's, and upright_energy, energy_scale and lyapunov_k_p come from the port's chain.
ACROBOT_LYAPUNOV_FLOOR = 1e-6


def acrobot_lyapunov_reward():
    """energy_scale, upright_energy and lyapunov_k_p of the port's plant, then reward_floor.

    Also returns Xin and Kaneda's (b1, b2) = g (h_1, h_2) of the chain, for the comment.
    """
    from common.ph_dmc.acrobot import acrobot_swingup_config
    from environment.ph_sac.dmc.acrobot import acrobot_chain, acrobot_lyapunov_reward_terms

    provisional = acrobot_swingup_config(
        failure_limits={}, reward_parameters={"reward_floor": ACROBOT_LYAPUNOV_FLOOR},
        reset_parameters=dict(ACROBOT_RESET), plant_parameters=dict(ACROBOT_PLANT),
    )
    chain = acrobot_chain(provisional)
    grouped = tuple((chain.gravity * chain.first_moments).tolist())
    terms = acrobot_lyapunov_reward_terms(provisional)
    return {**terms, "reward_floor": ACROBOT_LYAPUNOV_FLOOR}, grouped


ACROBOT_LYAPUNOV_REWARD, (ACROBOT_B1, ACROBOT_B2) = acrobot_lyapunov_reward()
ACROBOT_PLANT_NOTE = (
    "two 1 m, 1 kg links (capsule radii 0.05 and 0.049 m), damping 0.05 on both hinges, "
    "elbow motor gear 2, torque +-2 N m; the model has no constraint to remove. "
    f"{CLOCK_10} Reset: both angles uniform on [-pi, pi], velocities 0."
)
ACROBOT_BOX_NOTE = (
    f"No stop, no failure set. {box_note(1.0, 2.0, 'rad', 'N m')}, an elbow offset as "
    "acrobot_xk's; setpoint_error wrapped because both angles are periodic, wrapped "
    "observations here (acrobot_xk's unwrapped elbow served its travel limit, which "
    "dm_control lacks); flagged as a choice. The port reads the centres of mass, inertias "
    "and gravity from dm_control's compiled model (the documents do not state them)."
)
ACROBOT_LYAPUNOV_NOTE = (
    "Reward NOT dm_control's: dm_control's tol(|tip - target|; [0, 0.2], gaussian margin "
    "1) is below 6e-4 whenever the tip is more than 2 m from the target, and no arm "
    "learned from it in 500k decisions; the reward of Zhang, Sathuluri and Zimmermann "
    "(2023), arXiv:2312.11311, Eq. 4, which replaced it on 2026-11-08, stayed flat near a "
    "mean rate of -0.085 with no upright in 40 runs (it remains reward kind zhang2023 of "
    "the port, but no row uses it). Since the second change of 2026-11-08 this row takes "
    "acrobot_xk-swingup's reward on this plant: the log-reciprocal of Xin and Kaneda's "
    "energy-and-elbow Lyapunov function without its elbow-rate term (X. Xin and "
    "M. Kaneda, Int. J. Robust Nonlinear Control 17(16):1503-1524, 2007, "
    "doi:10.1002/rnc.1184), r(s) = -log(max(V(s) / V_down, reward_floor)) with "
    "V = (E - upright_energy)^2 / 2 + lyapunov_k_p wrap(theta_2)^2 / 2 and "
    "V_down = energy_scale^2 / 2, E the mechanical energy of the port's chain (kinetic "
    "plus potential, zero with both links horizontal). The constants are this plant's, "
    "computed through the chain (environment/ph_sac/dmc/acrobot.py, "
    "acrobot_lyapunov_reward_terms; the environment reads them from this row and checks "
    "only that they are positive and finite): upright_energy = E at upright rest = "
    f"b1 + b2 = {ACROBOT_LYAPUNOV_REWARD['upright_energy']:.6g} J and energy_scale = "
    "E(upright rest) - E(hanging rest) = 2 (b1 + b2) = "
    f"{ACROBOT_LYAPUNOV_REWARD['energy_scale']:.6g} J, with b1 = g (m1 lc1 + m2 l1) = "
    f"{ACROBOT_B1:.6g} N m and b2 = g m2 lc2 = {ACROBOT_B2:.6g} N m; lyapunov_k_p = "
    "1.001 x Xin and Kaneda's lower bound (2/pi) min(b1^2, b2^2) = "
    f"{2.0 / PI * min(ACROBOT_B1, ACROBOT_B2) ** 2:.6g}, so "
    f"{ACROBOT_LYAPUNOV_REWARD['lyapunov_k_p']:.6g} (acrobot_xk's 61.2 against its "
    f"bound 61.141 is the same ratio); reward_floor {ACROBOT_LYAPUNOV_FLOOR:g} as "
    "acrobot_xk. Choices: the elbow is wrapped (|theta_2| <= pi) because the port's "
    "state is periodic, where acrobot_xk's elbow is unwrapped and capped at 4 pi by its "
    "failure set; no failure set is added, the log keeping runaway energies tame. The "
    "rate is state-only, 0 at hanging rest, at most -log(reward_floor) = "
    f"{-math.log(ACROBOT_LYAPUNOV_FLOOR):.4g} (upright at rest) and unbounded below, "
    "logarithmically in the energy, so the learned reward head stays linear "
    "(reward_rate_bounds None)."
)
ACROBOT_SCOPE = {"swingup": " (its plant, motor, reset and clock)", "swingup_sparse": ""}
for task, reward, reward_note in (
    ("swingup", ACROBOT_LYAPUNOV_REWARD, ACROBOT_LYAPUNOV_NOTE),
    ("swingup_sparse", {"target_radius": 0.2, "margin": 0.0},
     "Reward: tol(|tip - target|; [0, 0.2], margin 0, the indicator) with the target at "
     "the upright tip (height 4 m)."),
):
    DMC.append(dmc(
        f"acrobot-{task}", f"acrobot_{task}", nq=2, nu=1, periodic=[0, 1], actuated=[1],
        period=0.01, episode=10.0, box=scaled_box(1.0, 2.0), torque=2.0,
        setpoint_mode="offset", setpoint_error="wrapped",
        failure={},
        reward=dict(reward),
        reset=dict(ACROBOT_RESET),
        plant=dict(ACROBOT_PLANT),
        notes=(
            f"dm_control acrobot/{task}{ACROBOT_SCOPE[task]}: "
            f"{ACROBOT_PLANT_NOTE} {reward_note} {ACROBOT_BOX_NOTE}"
        ),
        budget_note=BUDGET_10,
        compared=(
            "forward dynamics, reset law and open-loop trajectories; the reward is not "
            "dm_control's and is tested against its definition"
        ) if task == "swingup" else None,
    ))

for task, reward, reset, reward_note, reset_note in (
    ("balance", CARTPOLE_SMOOTH, CARTPOLE_BALANCE_RESET, "the smooth reward",
     "balance reset: cart uniform on [-0.1, 0.1] m, angle uniform on [-0.034, 0.034] rad, velocities 0.01 N(0,1)"),
    ("balance_sparse", CARTPOLE_SPARSE, CARTPOLE_BALANCE_RESET,
     "the sparse reward 1[|x| <= 0.25] 1[cos theta >= 0.995]",
     "balance reset: cart uniform on [-0.1, 0.1] m, angle uniform on [-0.034, 0.034] rad, velocities 0.01 N(0,1)"),
    ("swingup_sparse", CARTPOLE_SPARSE, CARTPOLE_SWINGUP_RESET,
     "the sparse reward 1[|x| <= 0.25] 1[cos theta >= 0.995]",
     "swing-up reset as cartpole-swingup"),
):
    DMC.append(dmc(
        f"cartpole-{task}", f"cartpole_{task}", nq=2, nu=1, periodic=[1], actuated=[0],
        period=0.01, episode=10.0, box=scaled_box(1.0, 10.0), torque=10.0,
        setpoint_mode="offset", setpoint_error="unwrapped",
        failure={},
        reward=dict(reward), reset=dict(reset), plant=dict(CARTPOLE_PLANT),
        network={"limit_port": True},
        notes=(
            f"dm_control cartpole/{task}: cartpole-swingup's plant, motor (+-10 N), soft "
            "rail limit at |x| = 1.8 m KEPT (solreflimit 0.08 1, MuJoCo's default "
            f"solimplimit). {kept_limit_note('cartpole-' + task)} Decision box and clock as cartpole-swingup; it differs in "
            f"{'the reset and ' if 'balance' in task else ''}{reward_note} ({reset_note}). "
            f"{CLOCK_10} The existing cartpole_swingup factory hard-codes the smooth "
            "reward and the Gaussian swing-up reset, so this task needs its own testbed."
        ),
        budget_note=BUDGET_10,
    ))

# dm_control's reward of the two- and three-pole tasks; a row of TOP_REWARD_VARIANT replaces
# this sentence of its comment by its variant's reward (VARIANTS below).
MULTIPOLE_SMOOTH_NOTE = (
    "Reward: the smooth cart-pole reward with the upright term averaged and the velocity "
    "term the minimum over the poles."
)
for task, poles in (("two_poles", 2), ("three_poles", 3)):
    DMC.append(dmc(
        f"cartpole-{task}", f"cartpole_{task}", nq=1 + poles, nu=1,
        periodic=list(range(1, 1 + poles)), actuated=[0],
        period=0.01, episode=10.0, box=scaled_box(1.0, 10.0), torque=10.0,
        setpoint_mode="offset", setpoint_error="unwrapped",
        failure={},
        reward=dict(CARTPOLE_SMOOTH),
        reset={**CARTPOLE_SWINGUP_RESET, "upper_angle_noise": 0.1},
        plant={**CARTPOLE_PLANT, "n_poles": poles},
        network={"limit_port": True},
        notes=(
            f"dm_control cartpole/{task}: cartpole-swingup's cart and first pole with "
            f"{poles - 1} further pole(s) of the same class (0.1 kg, 1 m, hinge damping "
            "2e-6) chained at the tips; same motor (+-10 N), soft rail limit at 1.8 m KEPT "
            f"(solreflimit 0.08 1). {kept_limit_note('cartpole-' + task)} Decision box and clock as "
            f"cartpole-swingup. {CLOCK_10} Reset: first hinge at pi + 0.01 N(0,1), "
            f"the others 0.1 N(0,1), cart and velocities 0.01 N(0,1). {MULTIPOLE_SMOOTH_NOTE} "
            "Not in dm_control's benchmarking set."
        ),
        budget_note=BUDGET_10,
        compared=(
            "forward dynamics, dm_control's smooth reward, reset law and open-loop "
            "trajectories; the row's Lyapunov reward is not dm_control's and is tested "
            "against its definition (tests/test_ph_orbit_rewards.py)"
        ) if f"cartpole-{task}" in TOP_REWARD_VARIANT else None,
    ))

REACHER_PLANT = {"arm_length": 0.12, "hand_length": 0.1, "finger_offset": 0.12,
                 "arm_mass": 0.04188790204786391, "hand_mass": 0.03560471674068432, "finger_mass": 0.004188790204786391,
                 "link_radius": 0.01, "finger_radius": 0.01, "damping": 0.01,
                 # dm_control's wrist limit range="-160 160" (degrees), default solref/solimp.
                 "wrist_range_degrees": 160.0, **MUJOCO_DEFAULT_LIMIT}
for task, radius in (("easy", 0.06), ("hard", 0.025)):
    DMC.append(dmc(
        f"reacher-{task}", f"reacher_{task}", nq=2, nu=2, periodic=[0, 1],
        actuated=[0, 1], context=2,
        period=0.02, episode=20.0, box=scaled_box(1.0, 0.05, 2), torque=0.05,
        setpoint_mode="offset", setpoint_error="wrapped",
        failure={},
        reward={"reach_radius": radius, "margin": 0.0, "failure_rate": 0.0},
        reset={"shoulder_range": PI, "wrist_range": WRIST, "target_radius_low": 0.05,
               "target_radius_high": 0.2, "velocity_noise": 0.0},
        plant=dict(REACHER_PLANT),
        network={"limit_port": True},
        notes=(
            f"dm_control reacher/{task}: planar two-link arm (links 0.12 and 0.10 m, "
            "masses 0.042, 0.036 and a 0.004 kg fingertip 0.12 m from the wrist), no "
            "gravity work, damping 0.01, both joints actuated with gear 0.05, torque "
            f"+-0.05 N m each. {CLOCK_20} Reset: shoulder uniform on [-pi, pi], wrist "
            "uniform within +-160 deg, velocities 0; target at radius U[0.05, 0.20] m, "
            "angle U[0, 2 pi]. Reward: the indicator |finger - target| <= "
            f"{radius} m (finger plus target radius, margin 0). The target is an "
            "episode-constant context (task_context_dim 2). The wrist's soft joint limit "
            "at +-160 deg is KEPT (MuJoCo's default solreflimit 0.02 1 and solimplimit "
            "0.9 0.95 0.001 0.5 2, as dm_control's XML; MuJoCo floors the time constant at "
            "2 x timestep, 0.04 s at dm_control's 20 ms but 0.02 s at our 1 ms). "
            f"{kept_limit_note('reacher-' + task)} {box_note(1.0, 0.05, 'rad', 'N m')}; the offset range is the reachable disc's joint range. "
            "Inertias are rounded in the documents; the port reads them from the compiled "
            "model."
        ),
        budget_note=BUDGET_20,
    ))

DMC.append(dmc(
    "point_mass-easy", "point_mass_easy", nq=2, nu=2, periodic=[], actuated=[0, 1],
    period=0.02, episode=20.0, box=scaled_box(0.3, 0.1, 2), torque=0.1,
    setpoint_mode="offset", setpoint_error="unwrapped",
    failure={},
    reward={"target_radius": 0.015, "margin": 0.015, "failure_rate": 0.0},
    reset={"position_range": 0.29, "velocity_noise": 0.0},
    plant={"mass": 0.3, "damping": 1.0, "slide_range": 0.29, **MUJOCO_DEFAULT_LIMIT},
    network={"limit_port": True},
    notes=(
        "dm_control point_mass/easy: a 0.3 kg point on two slides with damping 1, motors "
        "of gear 0.1 through fixed tendons, force +-0.1 N per axis; linear dynamics "
        f"(M = 0.3 I, D = I, U = 0). {CLOCK_20} Reset: position uniform within +-0.29 m, "
        "velocity 0. Reward: tol(|p|; [0, 0.015], 0.015) (gaussian) times "
        "(4 + mean_i max(0, 1 - u_i^2))/5. The slides' soft joint limits at +-0.29 m are "
        "KEPT (MuJoCo's default solreflimit 0.02 1 and solimplimit 0.9 0.95 0.001 0.5 2, as "
        "dm_control's XML; MuJoCo floors the time constant at 2 x timestep, 0.04 s at "
        f"dm_control's 20 ms but 0.02 s at our 1 ms). {kept_limit_note('point_mass-easy')} "
        f"{box_note(0.3, 0.1, 'm', 'N')}; the offset range covers the +-0.29 m box."
    ),
    budget_note=BUDGET_20,
))

LQR_SEED0_STIFFNESS = (20.488135039273246, 21.02763376071644, 19.236547993389046,
                       19.375872112626926, 24.636627605010293, 22.917250380826644)
for task, bodies, actuators in (("lqr_2_1", 2, 1), ("lqr_6_2", 6, 2)):
    DMC.append(dmc(
        f"lqr-{task}", task, nq=bodies, nu=actuators, periodic=[],
        actuated=list(range(actuators)),
        period=0.03, episode=10.0, box=scaled_box(1.0, 20.0, actuators), torque=20.0,
        setpoint_mode="offset", setpoint_error="unwrapped",
        failure={},
        reward={"control_cost": 0.1},
        reset={"position_radius": math.sqrt(2.0), "velocity_noise": 0.0},
        plant={"n_bodies": bodies, "n_actuated": actuators,
               "body_mass": 4.188790204786391,
               "stiffness_low": 15.0, "stiffness_high": 25.0, "damping": 0.0,
               # dm_control's draw with RandomState(0): the compiled jnt_stiffness values.
               "stiffness": ",".join(repr(k) for k in LQR_SEED0_STIFFNESS[:bodies])},
        notes=(
            f"dm_control lqr/{task}: {bodies} nested bodies of 4.18879 kg (compiled mass) on slides "
            "(relative displacements, constant non-diagonal mass matrix), springs with "
            "stiffness drawn from U[15, 25] when the model is built, zero damping, the "
            f"first {actuators} actuated with gear 1 and no control limit. 1 ms RK4 "
            "physics; fixed clock at dm_control's 30 ms control step (the reference "
            "period). Reset: positions on the sphere of radius sqrt(2), velocities 0. "
            "Reward 1 - |q|^2/2 - 0.1 |u|^2/2 (can be negative). No stop, no failure set. "
            "Episode capped at 10 s because dm_control's is unbounded: 333 decisions of "
            "30 ms and a last one of 10 ms. OPEN: torque limit +-20 N, the companion "
            "document's suggestion, not checked against the optimal closed loop; whether "
            "the port keeps dm_control's termination at |s| < 1e-6 (this row does not); "
            "the stiffness draw depends on the task's random seed, which the port must fix "
            f"(the drawn values are not in the documents). {box_note(1.0, 20.0, 'm', 'N')}, with the chosen 20 N limit. Diagnostic "
            "task, not in dm_control's benchmarking set."
        ),
        budget_note=(
            "Budget 500k decisions, the same number of decisions as cartpole-swingup "
            "(about 1497 episodes of 334 decisions)."
        ),
    ))

# swimmer-swimmer6 and swimmer-swimmer15 (2026-11-09): dm_control's geometry, fluid and
# motors; the joint range 2 pi / n is the failure limit, the reset range and the box offset.
from common.ph_dmc.swimmer import swimmer_joint_range  # noqa: E402

SWIMMER_PLANT = {"link_mass": 0.01, "link_length": 0.1, "box_half_width": 0.001,
                 "box_half_height": 0.01, "armature": 1e-06, "density": 3000.0,
                 "viscosity": 0.0, "nose_offset": 0.01,
                 # dm_control's solreflimit ".05 1" and solimplimit "0 .8 .1" (midpoint and
                 # power MuJoCo's defaults 0.5 and 2), kept: the limits stay soft constraints.
                 "limit_timeconst": 0.05, "limit_dampratio": 1.0, "limit_d0": 0.0,
                 "limit_dmax": 0.8, "limit_width": 0.1, "limit_midpoint": 0.5,
                 "limit_power": 2.0}
SWIMMER_TORQUE = 5e-4
# dm_control's physics step and integrator (common/ph_dmc/swimmer.py, the factory's).
from common.ph_dmc.swimmer import INTEGRATOR as SWIMMER_INTEGRATOR  # noqa: E402
from common.ph_dmc.swimmer import PHYSICS_DT as SWIMMER_PHYSICS_DT  # noqa: E402

BUDGET_30 = (
    "Budget 500k decisions, the same number of decisions as cartpole-swingup (500 "
    "episodes of 1000 decisions at full length; at 30 ms per decision this is 15000 s of "
    "interaction against the 10-s tasks' 5000 s)."
)
for links in (6, 15):
    joint = swimmer_joint_range(links)
    DMC.append(dmc(
        f"swimmer-swimmer{links}", f"swimmer_swimmer{links}", nq=links + 2, nu=links - 1,
        periodic=[2], actuated=list(range(3, links + 2)), context=2,
        period=0.03, episode=30.0, box=scaled_box(joint, SWIMMER_TORQUE, links - 1),
        torque=SWIMMER_TORQUE, setpoint_mode="offset", setpoint_error="unwrapped",
        failure={},
        reward={"target_radius": 0.1, "margin": 0.5, "failure_rate": 0.0},
        reset={"yaw_range": PI, "joint_range": joint, "close_target_probability": 0.2,
               "close_target_range": 0.3, "far_target_range": 2.0, "velocity_noise": 0.0},
        plant={"n_links": links, **SWIMMER_PLANT},
        network={"damping_velocity": True, "limit_port": True},
        physics_dt=SWIMMER_PHYSICS_DT,
        integrator=SWIMMER_INTEGRATOR,
        notes=(
            f"dm_control swimmer/swimmer{links}: a planar chain of {links} bodies (head plus "
            f"{links - 1} segments, 0.1 m apart, an inertial box of half-sizes 0.001 x 0.05 x "
            "0.01 m and 0.01 kg each) on a free planar root (slides x, y and the yaw hinge, "
            f"unactuated), {links - 1} actuated hinges (armature 1e-6, no damping, gear 5e-4, "
            "torque +-5e-4 N m) in a fluid of density 3000 and viscosity 0 (MuJoCo's "
            "inertia-box drag, quadratic in velocity), contact off, gravity normal to the "
            "plane. dm_control's own physics, Euler at 2 ms (swimmer.xml: timestep 0.002, "
            "MuJoCo's default integrator; since 2026-11-10, 1 ms RK4 before): no joint has "
            "damping, so MuJoCo's Euler step is the plain velocity-first semi-implicit "
            "step, and the model integrates with that map (model integrator "
            "semi_implicit) at the 2 ms physics step, one drift evaluation per step, the "
            "feedback law evaluated once per step at its start in both; fixed clock at "
            "dm_control's 30 ms control step (the reference period, 15 physics steps), "
            "1000 decisions per 30 s episode. "
            "Reset: root at the origin, yaw uniform on [-pi, pi], each joint uniform within "
            f"+-360/{links} deg, velocities 0; target uniform in +-0.3 m with probability 0.2, "
            "else in +-2 m. Reward: dm_control's tol(|nose - target|; [0, 0.1], margin 0.5, "
            "long_tail). The target is an episode-constant context (task_context_dim 2) "
            "against the absolute nose pose; the root x, y and the target are cyclic "
            "(cyclic_position_indices), so the learned plant leaves them out of its "
            "features. Joint limits KEPT as dm_control models them: range "
            f"+-360/{links} = +-{math.degrees(joint):g} deg, solreflimit 0.05 1, solimplimit 0 "
            "0.8 0.1, MuJoCo's default Newton solver; no failure set (a joint past its range "
            "meets the soft limit and continues), and the oracle reproduces MuJoCo's "
            "limit forces (environment/ph_sac/dmc/joint_limits.py). Plant: damping_velocity "
            "True, a damping D(y, v) = L(y, v) L(y, v)^T that the quadratic fluid drag needs "
            "(D_fl is linear in |v|); damping_rank nv because the drag is full rank in v "
            f"(three drag directions per body, {3 * links} >= nv = {links + 2}); limit_port "
            "True, a learned one-sided spring and damper beyond each joint's known range "
            "(limited_position_ranges), exactly zero inside, for the limits' kink. "
            "Departures from dm_control: the controller convention, the decision box, and "
            "the target as context (the physics is dm_control's since 2026-11-10). "
            f"{box_note(joint, SWIMMER_TORQUE, 'rad', 'N m')}; the offset is the joint range."
        ),
        budget_note=BUDGET_30,
        compared=(
            "forward dynamics with MuJoCo's fluid force and joint-limit constraint, reward "
            "with dm_control's nose, reset law and open-loop trajectories through the limits"
        ),
    ))


# ------------------------------------------------- optional reward variants (extra rows)
# Rows of other modes than top, each a copy of its top row with another reward (selected by
# the reward_parameters' keys) and its own output directory. Run one with
# benchmarks/run_ph_sac.py --env-id <env_id> --mode <mode>. A top row listed in
# TOP_REWARD_VARIANT takes its variant's reward itself, and that variant gets no row. The
# Lyapunov ("orbit") rewards: the negative log of an energy-shaping Lyapunov function, as
# acrobot-swingup's.
LYAPUNOV_FLOOR = ACROBOT_LYAPUNOV_FLOOR
VARIANT_NOTE = (
    "Optional reward variant of {env_id} (mode top), selected with --env-id {env_id} --mode "
    "{mode}; every column equals that row's but env_reward_parameters and "
    "protocol_output_directory. Not a default: the top row keeps its reward. "
)
# Variants decided against for the final run (2026-11-11); their rows stay runnable.
UNUSED_VARIANTS = ("lyapunov_cart", "lyapunov_cart_straight", "lyapunov_shoulder")
UNUSED_VARIANT_NOTE = "Not used in the final run (decided 2026-11-11). "
# The top row's own reward sentence when it takes a variant's (TOP_REWARD_VARIANT); it
# replaces MULTIPOLE_SMOOTH_NOTE in the row's comment. TOP_VARIANT_STATUS: an optional
# status sentence per env_id after it.
TOP_VARIANT_NOTE = (
    "Reward NOT dm_control's: the row takes the Lyapunov reward of the optional row of "
    "mode {mode}, in place of dm_control's smooth reward (the smooth cart-pole reward with "
    "the upright term averaged and the velocity term the minimum over the poles; it "
    "remains the testbed's own law). "
)
TOP_VARIANT_STATUS = {}


def cartpole_lyapunov_reward(poles, straight):
    """The Lyapunov reward_parameters of the n-pole cart-pole row, through the port's chain."""
    from environment.ph_sac.dmc.cartpole import (
        LYAPUNOV_CART_KEYS,
        LYAPUNOV_STRAIGHT_KEYS,
        cartpole_lyapunov_reward_terms,
    )

    terms = cartpole_lyapunov_reward_terms(
        {**CARTPOLE_PLANT, "n_poles": poles}, poles, straight=straight
    )
    values = {**terms, "reward_floor": LYAPUNOV_FLOOR, "failure_rate": 0.0}
    keys = LYAPUNOV_STRAIGHT_KEYS if straight else LYAPUNOV_CART_KEYS
    return {key: float(values[key]) for key in keys}


CART_LYAPUNOV_COMMON = (
    "r(s) = -log(max(V(s) / V_down, reward_floor)) with V_down = energy_scale^2 / 2 (V at "
    "hanging rest, x = 0, so r = 0 there; at most -log(reward_floor) = 13.8155 at the "
    "upright at rest), E the mechanical energy of the cart and poles (the port's chain: "
    "kinetic energy of cart and poles, potential g sum_j h_j cos phi_j, zero with every "
    "pole horizontal), upright_energy = E with every pole upright at rest and "
    "energy_scale = upright_energy - E(hanging rest), computed through the chain "
    "(environment/ph_sac/dmc/cartpole.py, cartpole_lyapunov_reward_terms). No theory fixes "
    "the weights; the scale rule gives the cart at the rail limit (1.8 m) the cost V_down, "
    "lyapunov_k_cart = energy_scale^2 / 1.8^2"
)
CART_LYAPUNOV_TAIL = (
    "; reward_floor 1e-6 as acrobot-swingup. State-only and unbounded below "
    "(logarithmically), so the learned reward head is linear (reward_rate_bounds None); "
    "no failure set (failure_rate 0)."
)
VARIANTS = []
for task, poles in (("two_poles", 2), ("three_poles", 3)):
    for mode, straight in (("lyapunov_cart", False), ("lyapunov_cart_straight", True)):
        reward = cartpole_lyapunov_reward(poles, straight)
        if straight:
            formula = (
                "Reward: energy, cart and straightness, V = (E - upright_energy)^2 / 2 + "
                "lyapunov_k_cart x^2 / 2 + lyapunov_k_straight sum_{k>=2} wrap(theta_k)^2 / 2 "
                "over the relative angles of pole k to pole k-1 (unactuated): with the cart "
                "at rest at 0, n >= 2 poles at the upright energy remain a free "
                "multi-pendulum that need not approach the upright, whereas a straight chain "
                "at that energy moves as one pendulum on its homoclinic orbit; the acrobot's "
                "elbow term does this collapse with its actuated joint, here it takes "
                "unactuated coordinates. "
            )
            weights = (
                f" = {reward['lyapunov_k_cart']:.6g} and one relative angle of pi the cost "
                f"V_down, lyapunov_k_straight = energy_scale^2 / pi^2 = "
                f"{reward['lyapunov_k_straight']:.6g}"
            )
        else:
            formula = (
                "Reward: the faithful mapping of acrobot-swingup's Xin--Kaneda reward to the "
                "cart-pole, energy and the actuated coordinate only, V = (E - "
                "upright_energy)^2 / 2 + lyapunov_k_cart x^2 / 2, zero on {E = "
                "upright_energy, x = 0}. "
            )
            weights = f" = {reward['lyapunov_k_cart']:.6g}"
        reward_note = (
            formula + CART_LYAPUNOV_COMMON
            + f" (energy_scale {reward['energy_scale']:.6g} J, upright_energy "
            f"{reward['upright_energy']:.6g} J, V_down {0.5 * reward['energy_scale'] ** 2:.6g})"
            + weights + CART_LYAPUNOV_TAIL
        )
        base = next(spec for spec in DMC if spec["env_id"] == f"cartpole-{task}")
        if TOP_REWARD_VARIANT.get(base["env_id"]) == mode:
            # The top row takes this reward; its comment's reward sentence changes with it.
            if MULTIPOLE_SMOOTH_NOTE not in base["comment"]:
                raise ValueError(f"{base['env_id']}: the reward sentence to replace is missing.")
            base["settings"]["reward_parameters"] = reward
            base["comment"] = base["comment"].replace(
                MULTIPOLE_SMOOTH_NOTE,
                TOP_VARIANT_NOTE.format(mode=mode) + reward_note
                + TOP_VARIANT_STATUS.get(base["env_id"], ""),
            )
            continue
        variant = copy.deepcopy(base)
        variant["mode"] = mode
        variant["settings"]["reward_parameters"] = reward
        variant["comment"] = (
            VARIANT_NOTE.format(env_id=base["env_id"], mode=mode)
            + (UNUSED_VARIANT_NOTE if mode in UNUSED_VARIANTS else "")
            + reward_note
            + " Otherwise as the top row: " + base["comment"]
        )
        VARIANTS.append(variant)
for env_id, mode in TOP_REWARD_VARIANT.items():
    if env_id not in ("cartpole-two_poles", "cartpole-three_poles") or mode not in (
        "lyapunov_cart", "lyapunov_cart_straight"
    ):
        raise ValueError(f"TOP_REWARD_VARIANT {env_id}: {mode} is not a variant this file builds.")


def acrobot_lyapunov_shoulder_reward():
    """The top row's Lyapunov constants plus lyapunov_k_shoulder (acrobot_shoulder_weight)."""
    from environment.ph_sac.dmc.acrobot import LYAPUNOV_SHOULDER_KEYS, acrobot_shoulder_weight

    values = {
        **ACROBOT_LYAPUNOV_REWARD,
        "lyapunov_k_shoulder": acrobot_shoulder_weight(ACROBOT_LYAPUNOV_REWARD["energy_scale"]),
    }
    return {key: float(values[key]) for key in LYAPUNOV_SHOULDER_KEYS}


ACROBOT_SHOULDER_REWARD = acrobot_lyapunov_shoulder_reward()
_acrobot_base = next(spec for spec in DMC if spec["env_id"] == "acrobot-swingup")
_acrobot_variant = copy.deepcopy(_acrobot_base)
_acrobot_variant["mode"] = "lyapunov_shoulder"
_acrobot_variant["settings"]["reward_parameters"] = ACROBOT_SHOULDER_REWARD
_acrobot_variant["comment"] = (
    VARIANT_NOTE.format(env_id="acrobot-swingup", mode="lyapunov_shoulder")
    + UNUSED_VARIANT_NOTE
    + "Reward: the top row's Lyapunov reward with a penalty on the unactuated shoulder, "
    "V_C = V + lyapunov_k_shoulder wrap(theta_1)^2 / 2 (theta_1 from upright), r = "
    "-log(max(V_C / V_C_down, reward_floor)) with V_C_down = energy_scale^2 / 2 + "
    "lyapunov_k_shoulder pi^2 / 2, V_C at hanging rest, so r = 0 there. V_C = 0 only at "
    "the upright at rest, so this variant asks for stabilization, where the top row's V = 0 "
    "on the whole set {E = upright_energy, theta_2 = 0} (the straight arm on its homoclinic "
    "orbit). No theory fixes the weight (Xin and Kaneda's bound covers only the elbow's "
    "k_P); the scale rule gives theta_1 = pi alone the cost energy_scale^2 / 2: "
    "lyapunov_k_shoulder = energy_scale^2 / pi^2 = "
    f"{ACROBOT_SHOULDER_REWARD['lyapunov_k_shoulder']:.6g} "
    "(environment/ph_sac/dmc/acrobot.py, acrobot_shoulder_weight), so V_C_down = "
    f"energy_scale^2 = {ACROBOT_SHOULDER_REWARD['energy_scale'] ** 2:.6g}. The other "
    "constants and choices are the top row's. Otherwise as the top row: "
    + _acrobot_base["comment"]
)
VARIANTS.append(_acrobot_variant)
DMC.extend(VARIANTS)


def dmc_config(spec, method):
    config = copy.deepcopy(method)
    config["testbed"] = spec["testbed"]
    config["model_step"] = testbed_model_step(spec["testbed"], spec["task"]["physics_dt"])
    # The simulator's own step (rk4 on RK4 physics, semi_implicit on Euler: the swimmers).
    config["model_integrator"] = MODEL_INTEGRATOR_OF[spec["task"]["integrator"]]
    # Contact-triggered fine substepping where MODEL_COARSE_STEPS lists the testbed; the
    # method row's (cartpole-swingup's) value is not copied to the others.
    config.pop("model_coarse_step", None)
    coarse = testbed_model_coarse_step(spec["testbed"])
    if coarse is not None:
        config["model_coarse_step"] = coarse
    config["testbed_settings"] = spec["settings"]
    config["network"]["damping_rank"] = spec["task"]["nv"]
    # A row carries the limit port only when its own spec has a kept joint limit and
    # CONTACT_HANDLING does not list it; the method row's (cartpole_swingup.json's)
    # contact_handling and limit_port are not copied to the others.
    config.pop("contact_handling", None)
    config["network"].pop("limit_port", None)
    config["network"].update(spec["network"])
    if CONTACT_HANDLING.get(spec["env_id"]) == "exclude":
        config["contact_handling"] = "exclude"
        config["network"].pop("limit_port", None)
    controller = {
        "decision_low": np.array(spec["task"]["decision_low"]),
        "decision_high": np.array(spec["task"]["decision_high"]),
    }
    config["temperature"]["target_entropy"] = decision_box_entropy_target(controller)
    # A variant row (mode other than top) writes into its own folder.
    mode = spec.get("mode", "top")
    suffix = "" if mode == "top" else "_" + mode
    config["protocol"]["output_directory"] = "out/ph_sac/" + spec["env_id"].replace("-", "_") + suffix
    config["training"] = with_plant_step_schedule(config["training"])
    return config


MODEL_STEP_NOTE = (
    "Model step {step:g} ms ({pieces} substeps per {interval:g} ms decision): "
    "common.ph_config.MODEL_STEPS, the largest step tried that divides the interval at "
    "which the oracle plant's one-interval model endpoint stays within 1e-3 relative error "
    "of the simulator's (99th percentile, decisions uniform in the box; 2026-11-09)."
)
# Rows with joint limits (network limit_port): the model step is coarsened only on tasks
# without joint limits (common.ph_config.MODEL_STEPS, since 2026-11-10).
LIMITED_STEP_NOTE = (
    "Model step {step:g} ms, the physics step ({pieces} substeps per {interval:g} ms "
    "decision; common.ph_config.MODEL_STEPS): the model step is coarsened only on tasks "
    "without joint limits, since with the limit in the oracle an interval that touches a "
    "limit stays within 1e-3 relative endpoint error only at the physics step (2026-11-10)."
)


# Rows with joint limits and a MODEL_COARSE_STEPS entry: contact-triggered fine substepping.
CONTACT_STEP_NOTE = (
    " Contact-triggered fine substepping (model_coarse_step {coarse:g} ms, "
    "common.ph_config.MODEL_COARSE_STEPS, since 2026-11-10): every interval is integrated "
    "at {coarse:g} ms ({coarse_pieces} substeps, the task's free-state bound) first; an "
    "interval whose coarse path has, at any knot, a limited coordinate within the margin "
    "1% of its range + 2 x {coarse:g} ms x |its rate| of a range edge (or past it) is "
    "integrated again at the {step:g} ms model step, and that result replaces the coarse "
    "one (models/ph_sac/rollouts.contact_flags)."
)
# Rows of CONTACT_HANDLING (contact exclusion): in place of CONTACT_STEP_NOTE, which needs
# the limit port (common.ph_config.contact_substepping_setting is None under exclusion).
EXCLUSION_NOTE = (
    " Contact exclusion (model_contact_handling exclude, common.ph_config.CONTACT_HANDLINGS; "
    "decided 2026-11-11 in place of the limit port, after the experiment of 2026-11-10: "
    "the same final scores as the port on cartpole-swingup and point_mass-easy at 3-5x "
    "less time per decision, and a plant error on free intervals of 0.2% against the "
    "port's 0.9% on the cart-pole): the learned plant is the free mechanism (no limit "
    "port); a replay transition whose start state lies within the reach margin of a range "
    "edge (1% of the range plus the damped travel bound over the interval, rule version 2) "
    "or whose interval touched a limit is left out of the plant fit (warm fit included), "
    "every use of the plant at such a state falls back to the observed transition, and "
    "model paths are truncated at their last knot clear of a limit; every model "
    "integration runs at the coarse step {coarse:g} ms ({coarse_pieces} substeps; "
    "common.ph_config.exclusion_model_step), with no fine re-integration (the plant has "
    "no limit force)."
)


def exclusion_note(testbed, interval):
    """EXCLUSION_NOTE for a testbed with a MODEL_COARSE_STEPS entry."""
    coarse = testbed_model_coarse_step(testbed)
    if coarse is None:
        raise ValueError(f"{testbed}: contact exclusion here expects a model_coarse_step.")
    return EXCLUSION_NOTE.format(coarse=1e3 * coarse, coarse_pieces=round(interval / coarse))


# Rows simulated with Euler at their model step (the swimmers): the model is the simulator.
NATIVE_STEP_NOTE = (
    "Model step {step:g} ms, the physics step ({pieces} substeps per {interval:g} ms "
    "decision; common.ph_config.MODEL_STEPS): the model integrator semi_implicit at the "
    "physics step is MuJoCo's Euler step (no dof damping), so the oracle drift reproduces "
    "the simulator's endpoints up to MuJoCo's constraint-solver tolerance, limits "
    "included, with one drift evaluation per physics step."
)


def model_step_note(spec):
    step = testbed_model_step(spec["testbed"], spec["task"]["physics_dt"])
    interval = spec["task"]["min_duration"]
    limited = bool(spec.get("network", {}).get("limit_port"))
    if limited and step != spec["task"]["physics_dt"]:
        raise ValueError(f"{spec['env_id']}: a task with joint limits keeps the physics step.")
    coarse = testbed_model_coarse_step(spec["testbed"])
    if coarse is not None and not limited:
        raise ValueError(f"{spec['env_id']}: a coarse step needs joint limits to watch.")
    template = MODEL_STEP_NOTE
    if spec["task"]["integrator"] == "Euler":
        template = NATIVE_STEP_NOTE
    elif limited:
        template = LIMITED_STEP_NOTE
    note = template.format(
        step=1e3 * step, pieces=round(interval / step), interval=1e3 * interval
    )
    if CONTACT_HANDLING.get(spec["env_id"]) == "exclude":
        note += exclusion_note(spec["testbed"], interval)
    elif coarse is not None:
        note += CONTACT_STEP_NOTE.format(
            coarse=1e3 * coarse, coarse_pieces=round(interval / coarse), step=1e3 * step
        )
    return note


# Units of the limit port's (stiffness, damping, width) on a slide and on a hinge.
PORT_UNITS = {"slide": ("N/m", "N s/m", "m"), "hinge": ("N m/rad", "N m s/rad", "rad")}
PORT_JOINTS = {"cartpole": "slide", "point_mass": "slide", "reacher": "hinge", "swimmer": "hinge"}


def port_init_note(testbed, settings):
    """The limit port's start, as the factory's state spec (limit_port_init) has it."""
    init = _build_testbed(testbed, settings)["state_spec"]["limit_port_init"]
    kind = next(PORT_JOINTS[prefix] for prefix in PORT_JOINTS if testbed.startswith(prefix))
    units = PORT_UNITS[kind]

    def span(values):
        low, high = min(values), max(values)
        if math.isclose(low, high, rel_tol=1e-3):
            return f"{low:.4g}"
        return f"{low:.4g} to {high:.4g}"

    stiffness, damping, width = zip(*init)
    return (
        "Limit port initialised from the compiled limit (state spec limit_port_init, "
        "environment/ph_sac/dmc/joint_limits.limit_port_init: stiffness "
        "1/(dof_invweight0 tc^2 dampratio^2), damping 2/(dof_invweight0 tc), width "
        "1e-3 + solimp width (1 - d0/dmax)): "
        f"k {span(stiffness)} {units[0]}, c {span(damping)} {units[1]}, "
        f"w {span(width)} {units[2]}."
    )


for spec in DMC:
    notes = [spec["comment"], model_step_note(spec), plant_step_note()]
    excluded = CONTACT_HANDLING.get(spec["env_id"]) == "exclude"
    if spec.get("network", {}).get("limit_port") and not excluded:
        notes.append(port_init_note(spec["testbed"], spec["settings"]))
    if spec.get("mode", "top") == "top" and spec["env_id"] in ROW_STATUS:
        notes.append(ROW_STATUS[spec["env_id"]])
    spec["comment"] = " ".join(notes)
for env_id in list(CONTACT_HANDLING) + list(ROW_STATUS):
    if env_id not in {spec["env_id"] for spec in DMC} | {"cartpole-swingup"}:
        raise ValueError(f"{env_id}: no dm_control row of that name.")


# ---------------------------------------------- irregular clocks (rows of mode irregular)
# The final run's clock (decided 2026-11-11): on a fixed clock at the reference period the
# CT-SAC baseline is exactly SAC. A row of mode "irregular" copies its top row and adds the
# testbed_settings IRREGULAR_CLOCK (common.ph_dmc._common.CLOCK_SETTINGS_KEYS, read by the
# factory through with_clock_settings): the CT-SAC benchmark's law (Nguyen 2026,
# arXiv:2602.14587, Sec. 5.1 and App. B.1.3 Table 1), as this repository's CT-SAC code draws
# it (environment/base.py sample_dt_from_distribution, dist two_tail_uniform; the irregular
# rows of benchmarks/hyperparams/ct_sac.csv: tail_p=0.99;tail_split=0.9): a whole number of
# physics steps, the minimum with probability 0.99 x 0.9 = 0.891, the maximum with 0.99 x
# 0.1 = 0.099, uniform on the steps strictly between with 0.01; the bounds T/5 and 3T of
# cheetah-run and walker-run (irregular_clock_bounds), T the task's dm_control control step,
# CT-SAC's dt_default. Its mean is 0.4912 T on these grids, so an episode has about twice the
# top row's decisions. Run one with --env-id <env_id> --mode irregular. The top rows keep
# dm_control's fixed clock. (2026-11-11 to 11-12: tail 0.8 split 0.5 on [T/5, 9T/5].)
from common.ph_dmc._common import CLOCK_SETTINGS_KEYS, dmc_clock  # noqa: E402
from common.ph_time import clock_duration_pmf  # noqa: E402

IRREGULAR_MODE = "irregular"
IRREGULAR_MIN_FRACTION_NOTE = "T/5 and 3T"
IRREGULAR_CLOCK = {"clock_kind": "two_tail_uniform", "tail_probability": 0.99, "tail_split": 0.9}
IRREGULAR_ROWS = (
    "pendulum-swingup", "acrobot-swingup", "acrobot-swingup_sparse", "cartpole-balance",
    "cartpole-balance_sparse", "cartpole-swingup", "cartpole-swingup_sparse",
    "cartpole-two_poles", "cartpole-three_poles", "point_mass-easy",
)


def irregular_clock(period, physics_dt):
    """The IRREGULAR_CLOCK ClockConfig of a task with reference period and physics step."""
    return dmc_clock(period, physics_dt, **IRREGULAR_CLOCK)


def irregular_note(env_id, period, physics_dt, episode, model_step, coarse, excluded,
                   base_mode="top"):
    """The comment that heads a row of mode irregular (or <base_mode>_irregular)."""
    mode = IRREGULAR_MODE if base_mode == "top" else f"{base_mode}_{IRREGULAR_MODE}"
    base = "top row" if base_mode == "top" else f"row of mode {base_mode}"
    clock = irregular_clock(period, physics_dt)
    steps, probabilities = clock_duration_pmf(clock)
    mean = float(np.dot(steps, probabilities)) * physics_dt
    tail, split = IRREGULAR_CLOCK["tail_probability"], IRREGULAR_CLOCK["tail_split"]
    text = (
        f"Row of mode {mode}: the final run's clock (decided 2026-11-11; on a fixed "
        f"clock at the reference period the CT-SAC baseline is exactly SAC). The {base} "
        f"{env_id} on an irregular clock, selected with --env-id {env_id} --mode "
        f"{mode}; every column equals the {base}'s but env_clock_kind, "
        "env_tail_probability, env_tail_split, task_clock_kind, task_min_duration, "
        "task_max_duration and protocol_output_directory. Clock: the CT-SAC benchmark's "
        "irregular law (Nguyen 2026, arXiv:2602.14587, App. B.1.3 Table 1; "
        "environment/base.py sample_dt_from_distribution, two_tail_uniform with "
        f"tail_p={tail:g};tail_split={split:g} as benchmarks/hyperparams/ct_sac.csv's "
        "irregular rows): a whole number of "
        f"{1e3 * physics_dt:g} ms physics steps, {1e3 * clock['min_duration']:g} ms with "
        f"probability {tail * split:.3f}, {1e3 * clock['max_duration']:g} ms with "
        f"{tail * (1.0 - split):.3f}, and uniform on the {len(steps) - 2} steps between with "
        f"{1.0 - tail:.3f}; the bounds {IRREGULAR_MIN_FRACTION_NOTE} of CT-SAC's cheetah-run "
        "and walker-run (common.ph_dmc._common.irregular_clock_bounds), T the reference "
        f"period, {1e3 * period:g} ms (dm_control's control step and CT-SAC's dt_default, "
        f"unchanged). Mean interval {1e3 * mean:.4g} ms = {mean / period:.4f} T, so about "
        f"{episode / mean:.0f} decisions per {episode:g} s episode (the top row: "
        f"{episode / period:.0f}), and the budget, in decisions as in the top row, covers "
        f"{mean / period:.4f} times its simulated time (the evaluation cadence is in "
        "decisions too; the last interval of an episode is shortened at the time limit). A "
        "CHOICE: CT-SAC's quadruped-run used T/4 and 2.5T, "
        "humanoid-walk 0.4T and 1.6T with tail_p 0.8, tail_split 0.5. The model step need "
        "not divide an interval: an interval of n "
        "physics steps is integrated in ceil(n delta / h) equal substeps of at most h "
        "(models/ph_sac/rollouts._continuous_grid), h the model step "
        f"{1e3 * model_step:g} ms"
    )
    if excluded:
        text += (
            f", under contact exclusion the coarse step {1e3 * coarse:g} ms, which then "
            "needs no fixed clock (common.ph_config._check_contact_substepping)"
        )
    return text + "."


IRREGULAR_SPECS = []
for spec in DMC:
    if spec.get("mode", "top") != "top" or spec["env_id"] not in IRREGULAR_ROWS:
        continue
    period, physics_dt = spec["task"]["reference_period"], spec["task"]["physics_dt"]
    clock = irregular_clock(period, physics_dt)
    irregular = copy.deepcopy(spec)
    irregular["mode"] = IRREGULAR_MODE
    irregular["settings"].update(IRREGULAR_CLOCK)
    irregular["task"].update(
        clock_kind=clock["kind"],
        min_duration=clock["min_duration"],
        max_duration=clock["max_duration"],
    )
    irregular["comment"] = (
        irregular_note(
            spec["env_id"], period, physics_dt, spec["task"]["episode_duration"],
            testbed_model_step(spec["testbed"], physics_dt),
            testbed_model_coarse_step(spec["testbed"]),
            CONTACT_HANDLING.get(spec["env_id"]) == "exclude",
        )
        + " Otherwise as the top row: " + spec["comment"]
    )
    IRREGULAR_SPECS.append(irregular)
if {spec["env_id"] for spec in IRREGULAR_SPECS} | {"cartpole-swingup"} != set(IRREGULAR_ROWS):
    raise ValueError("IRREGULAR_ROWS names a row this file does not build.")
# Exploratory variant rows on the final run's clock (mode <variant>_irregular, 2026-11-11):
# not part of the reported set. Variant A on two poles was added beside the final
# run, to see whether a longer budget separates the arms more.
IRREGULAR_VARIANT_ROWS = (("cartpole-two_poles", "lyapunov_cart"),)
EXPLORATORY_NOTE = (
    "EXPLORATORY row (2026-11-11), not part of the final run's reported set: the unused "
    "reward variant {mode} on the final run's clock and contact convention, run beside it "
    "only to see whether a longer budget separates the arms. "
)
for env_id, base_mode in IRREGULAR_VARIANT_ROWS:
    spec = next(s for s in DMC if s["env_id"] == env_id and s.get("mode", "top") == base_mode)
    period, physics_dt = spec["task"]["reference_period"], spec["task"]["physics_dt"]
    clock = irregular_clock(period, physics_dt)
    irregular = copy.deepcopy(spec)
    irregular["mode"] = f"{base_mode}_{IRREGULAR_MODE}"
    irregular["settings"].update(IRREGULAR_CLOCK)
    irregular["task"].update(
        clock_kind=clock["kind"],
        min_duration=clock["min_duration"],
        max_duration=clock["max_duration"],
    )
    irregular["comment"] = (
        EXPLORATORY_NOTE.format(mode=base_mode)
        + irregular_note(
            env_id, period, physics_dt, spec["task"]["episode_duration"],
            testbed_model_step(spec["testbed"], physics_dt),
            testbed_model_coarse_step(spec["testbed"]),
            CONTACT_HANDLING.get(env_id) == "exclude",
            base_mode=base_mode,
        )
        + f" Otherwise as the row of mode {base_mode}: " + spec["comment"]
    )
    IRREGULAR_SPECS.append(irregular)
DMC.extend(IRREGULAR_SPECS)

rows = []
for env_id, mode, name, comment in IN_HOUSE:
    raw = load(name)
    # A JSON config states its own model_step; it must be the central table's.
    step = testbed_model_step(raw["testbed"], declared_from_factory(raw)["physics_dt"])
    if raw["model_step"] != step:
        raise ValueError(
            f"{name}: model_step {raw['model_step']} differs from {step}, the testbed's in "
            "common.ph_config.MODEL_STEPS; update one of them."
        )
    # And its model_coarse_step (absent: off) the central MODEL_COARSE_STEPS entry.
    coarse = testbed_model_coarse_step(raw["testbed"])
    if raw.get("model_coarse_step") != coarse:
        raise ValueError(
            f"{name}: model_coarse_step {raw.get('model_coarse_step')} differs from {coarse}, "
            "the testbed's in common.ph_config.MODEL_COARSE_STEPS; update one of them."
        )
    # And its contact_handling (absent: "port") CONTACT_HANDLING's entry for the row; an
    # excluded row has no limit port.
    handling = raw.get("contact_handling")
    if handling != CONTACT_HANDLING.get(env_id) or (
        handling == "exclude" and raw["network"].get("limit_port")
    ):
        raise ValueError(
            f"{name}: contact_handling {handling!r} (limit_port "
            f"{raw['network'].get('limit_port')!r}) differs from CONTACT_HANDLING's "
            f"{CONTACT_HANDLING.get(env_id)!r} for {env_id} (exclude: no limit port); update "
            "one of them."
        )
    if raw["network"].get("limit_port"):
        comment = " ".join([comment, port_init_note(raw["testbed"], raw["testbed_settings"])])
    if handling == "exclude":
        interval = declared_from_factory(raw)["min_duration"]
        comment = comment + exclusion_note(raw["testbed"], interval)
    # The dm_control row read from a JSON (cartpole-swingup) carries the dm_control rows'
    # plant-step schedule, so the file must state it; the in-house rows carry none.
    schedule = {k: raw["training"][k] for k in PLANT_STEP_KEYS if k in raw["training"]}
    expected = PLANT_STEP_SCHEDULE if env_id == "cartpole-swingup" else {}
    if schedule != expected:
        raise ValueError(
            f"{name}: training plant-step keys {schedule} differ from {expected} "
            "(PLANT_STEP_SCHEDULE for cartpole-swingup, none for the in-house rows); update "
            "one of them."
        )
    if schedule:
        comment = " ".join([comment, plant_step_note()])
    rows.append(config_to_row(raw, env_id=env_id, mode=mode,
                              task=declared_from_factory(raw), comment=comment))
    if mode == "top" and env_id in IRREGULAR_ROWS:
        # cartpole-swingup's row of mode irregular: its JSON with IRREGULAR_CLOCK.
        if set(CLOCK_SETTINGS_KEYS) & set(raw["testbed_settings"]):
            raise ValueError(f"{name}: the top row keeps the fixed clock.")
        irregular = copy.deepcopy(raw)
        irregular["testbed_settings"].update(IRREGULAR_CLOCK)
        irregular["protocol"]["output_directory"] += "_" + IRREGULAR_MODE
        declared = declared_from_factory(irregular)
        note = irregular_note(
            env_id, declared["reference_period"], declared["physics_dt"],
            declared["episode_duration"], raw["model_step"],
            raw.get("model_coarse_step"), raw.get("contact_handling") == "exclude",
        )
        rows.append(config_to_row(irregular, env_id=env_id, mode=IRREGULAR_MODE, task=declared,
                                  comment=note + " Otherwise as the top row: " + comment))

method = load("cartpole_swingup.json")
for spec in DMC:
    rows.append(config_to_row(dmc_config(spec, method), env_id=spec["env_id"],
                              mode=spec.get("mode", "top"), task=spec["task"],
                              comment=spec["comment"]))

order = [
    "pendulum-swingup", "acrobot-swingup", "acrobot-swingup_sparse", "cartpole-balance",
    "cartpole-balance_sparse", "cartpole-swingup", "cartpole-swingup_sparse",
    "cartpole-two_poles", "cartpole-three_poles", "reacher-easy", "reacher-hard",
    "point_mass-easy", "lqr-lqr_2_1", "lqr-lqr_6_2", "swimmer-swimmer6",
    "swimmer-swimmer15", "windy_pendulum-balance",
    "windy_pendulum-swingup_tq5", "windy_pendulum-swingup_tq12", "acrobot_xk-swingup",
]
rank = {env_id: i for i, env_id in enumerate(order)}
rows.sort(key=lambda r: (rank[r["env_id"]], r["mode"] != "top", r["mode"]))

path = table_path()
with path.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(TABLE_COLUMNS), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
print(path, len(rows))

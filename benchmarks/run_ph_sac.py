"""Entry point for the MuJoCo PH-SAC comparison suite.

A protocol comes from a JSON config (--config) or from a row of the hyperparameter
table benchmarks/hyperparams/ct_ph_sac.csv (--env-id, --mode; see
benchmarks/ph_hyperparams.py). Testbeds, their settings keys and their config factories
come from the registry (environment/ph_sac/registry.py); rows whose testbed is a
registered stub parse but do not build; --list shows every row and whether it builds,
then every arm.

A protocol runs the four paper arms unless --arms names others: ct_sac_torque,
implicit_torque, explicit_rollout_torque and explicit_generator_torque (the CT-SAC
baseline, then CT-PH-SAC's implicit, rollout and generator arms), all deciding the torque
itself, held over the interval (evaluations/ph_sac/protocol.paper_arms). The ablation
arms run only when named: the impedance-law ablation (implicit_learned_admissions,
explicit_rollout, explicit_generator: the same three CT-PH-SAC arms deciding impedance
laws, setpoint, stiffness and damping, on the feedback box; protocol.impedance_decision_arms),
the implicit-setting ablations of implicit_learned_admissions (implicit_oracle,
implicit_averaged_draw, implicit_no_rollout_rate, implicit_polyak_plant,
implicit_gated_admissions, implicit_fixed_gate_allocation; protocol.implicit_ablation_arms)
and the value-step ablation (implicit_explicit_step, implicit_explicit_step_interval,
implicit_critic, implicit_critic_head, with torque decisions: protocol.step_ablation_arms),
whose pilots are benchmarks/configs/ph_sac/step_ablation/ (add implicit_torque
explicit_generator_torque to --arms for the two reference arms on the same settings):
    python benchmarks/run_ph_sac.py --config benchmarks/configs/ph_sac/step_ablation/lqr_2_1.json --arms implicit_explicit_step implicit_explicit_step_interval implicit_critic implicit_critic_head --resume

Examples:
    python benchmarks/run_ph_sac.py --config benchmarks/configs/ph_sac/windy_pendulum.json
    python benchmarks/run_ph_sac.py --env-id cartpole-swingup --mode top
    python benchmarks/run_ph_sac.py --env-id cartpole-swingup --mode top --arms implicit_torque --seeds 0
    python benchmarks/run_ph_sac.py --env-id cartpole-swingup --mode top --arms implicit_learned_admissions --seeds 0
    python benchmarks/run_ph_sac.py --env-id cartpole-swingup --refresh-mode held
    python benchmarks/run_ph_sac.py --list

Two more CT-SAC baseline arms run only when named (protocol.baseline_released_arms; not in
the paper or ablation sets): ct_sac_released is ct_sac_torque with CT-SAC's released
hyperparameters (protocol.RELEASED_CT_SAC_SETTINGS: step size 7.3e-4, replay 300000, a
warm-up of 10000 decisions of uniform random actions, target rate 0.02) in place of the
explicit config's, and ct_sac_released_timeout is the same with CT-SAC's own time-limit
convention (done = terminated or truncated, no bootstrap through the time limit):
    python benchmarks/run_ph_sac.py --config <config.json> --arms ct_sac_released ct_sac_released_timeout --seeds 0 --resume

The CT-SAC arms with an implicit value step also run only when named
(protocol.baseline_implicit_arms; algorithms/ct_sac_implicit.py): ct_sac_implicit is
ct_sac_torque with its critic update replaced by the implicit step of size tau, solved by
K residual-gradient steps over N policy draws per row and a hard copy; ct_sac_implicit_schedule
adds implicit CT-PH-SAC's outer actor and temperature schedule; ct_sac_implicit_released
takes ct_sac_released's hyperparameters. tau, K, N (1 s, 400, 4 by default) are the optional
"ct_sac_implicit" object of a JSON config's "explicit" section:
    python benchmarks/run_ph_sac.py --config <config.json> --arms ct_sac_implicit ct_sac_implicit_schedule --seeds 0 --resume

So do the CT-SAC arms whose actor and critics read CT-PH-SAC's state features
(protocol.baseline_feature_arms: periodic positions as sine and cosine, other positions
raw, velocities divided by the network's velocity_feature_scale) in place of the state:
ct_sac_features, ct_sac_released_features and ct_sac_released_timeout_features are
ct_sac_torque, ct_sac_released and ct_sac_released_timeout with that input map:
    python benchmarks/run_ph_sac.py --config <config.json> --arms ct_sac_features ct_sac_released_features --seeds 0 --resume
ct_sac_implicit_features, ct_sac_implicit_schedule_features and
ct_sac_implicit_released_features are ct_sac_implicit, ct_sac_implicit_schedule and
ct_sac_implicit_released with the same input map (protocol.baseline_implicit_feature_arms).

Every CT-SAC arm (ct_sac_torque and all of the above) takes its reward and temperature
units from the protocol key ct_sac_reward_units (common.ph_config.CT_SAC_REWARD_UNITS; a
JSON config's "protocol" section or --ct-sac-reward-units, no table column): "second"
(absent, every earlier run) or "reference_period", which divides the learner's rewards,
failure charge and temperature by T, so a reward rate of 1 per second is a reward of 1 per
period (CT-SAC's dm_control scale) and initial_alpha_c 1 is CT-SAC's alpha 1. The output
path does not hold it: give such runs their own --output.
    python benchmarks/run_ph_sac.py --env-id cartpole-swingup --arms ct_sac_released_features --ct-sac-reward-units reference_period --output <dir> --resume
They take the reward itself from the protocol key ct_sac_reward_source
(common.ph_config.CT_SAC_REWARD_SOURCES; a JSON config's "protocol" section or
--ct-sac-reward-source, no table column): "interval" (absent, every earlier run), the
environment's interval reward per period, or "end_state", T times the task's reward rate at
the state that ends the interval with the torque held over it, not scaled by the interval's
duration, as CT-SAC's own wrapper rewards a step (with reference_period units: the rate
itself). The failure charge is the same under both. Again the output path does not hold it.
    python benchmarks/run_ph_sac.py --env-id cartpole-swingup --arms ct_sac_released_features --ct-sac-reward-units reference_period --ct-sac-reward-source end_state --output <dir> --resume

Every arm takes its reward datum from the protocol key reward_observation
(common.ph_config.REWARD_OBSERVATIONS; a JSON config's "protocol" section or
--reward-observation, no table column): "interval" (absent, every earlier run), the
environment's interval reward, or "end_state", the task's reward rate at each interval's end
state alone. Under "end_state" the CT-SAC arms run ct_sac_reward_source end_state (an explicit
ct_sac_reward_source interval is refused), and every CT-PH-SAC arm learns a reward-rate network
from the end-state rates and integrates it over each interval in its targets (the implicit
arms need implicit reward_source learned). The output path does not hold it either.
    python benchmarks/run_ph_sac.py --env-id cartpole-swingup --reward-observation end_state --output <dir> --resume

Under contact_handling exclude the implicit arms' value residual at a flagged row reads the
online value at s and s', and its gradient reaches both by default. The optional key
flagged_value_gradient of a JSON config's "implicit" section
(common.ph_config.FLAGGED_VALUE_GRADIENTS; no table column, no command-line flag) set to
"stop_successor" detaches V(s') there (the residuals keep their values); it needs
contact_handling exclude, and "full" stated is the absent key. The output path does not
hold it.
    "implicit": {..., "flagged_value_gradient": "stop_successor"}

--report-only rebuilds the reports from the run records already in the output folder, of
every arm found there unless --arms names some (the default arm set does not limit it).

The impedance-law arms' feedback law is evaluated at every RK4 stage (controller
refresh_mode "continuous", every table row); --refresh-mode held (or the config key
refresh_mode) runs the sample-and-hold controller of every run before 2026-11-09. A torque
decision is held over its interval in either mode.

Restarts (runs longer than a queue's wall limit): add --resume to the command and submit
the same command again after a stop. With --resume each run keeps one resumable
checkpoint, <output>/<testbed>/<arm>/seed_<s>/resume.pt, rewritten every --resume-every
decisions (default 10000), after every evaluation and on SIGUSR1 or SIGTERM, after which
the process exits with code 75 (evaluations/ph_sac/resume.RESUMABLE_EXIT_CODE). The same
command with --resume skips finished runs (resume_complete.json), continues a stopped one
exactly, bitwise as if never stopped, and starts the others afresh; a checkpoint written
for other settings is refused. Without --resume nothing changes. A run's fingerprint holds
its arm's spec, not the protocol's arm list, so the default arm set does not enter it.
    python benchmarks/run_ph_sac.py --env-id cartpole-swingup --arms implicit_torque --seeds 0 --resume

The adaptation experiment (docs/ct_ph_sac_layout.md, "Adaptation experiment") runs from
JSON configs only: testbed_settings physical_change changes the simulated robot
(acrobot_swingup, cartpole_swingup), protocol warm_start starts every run from a finished
run's checkpoint.pt (evaluations/ph_sac/adaptation.py; select the CT-SAC and implicit arms
with --arms: the explicit paper arms take no warm start), and implicit
trainable_plant_parts restricts the plant fit to some parts (add
implicit_learned_admissions for the impedance-law arm):
    python benchmarks/run_ph_sac.py --config benchmarks/configs/ph_sac/adaptation/acrobot_length_1.25_warm.json --arms ct_sac_torque implicit_torque --resume
    python benchmarks/run_ph_sac.py --config benchmarks/configs/ph_sac/adaptation/acrobot_length_1.25_structured.json --arms implicit_torque --resume

A JSON config may also set testbed_settings discount_rate (per second) on the testbeds that
take it (TestbedSpec.discount_setting): the run's discount in place of the row's 0.5 /s,
for every arm, the environment's rewards and the evaluation alike.

A JSON config's network section may also set actor_log_std_init (the start of every entry
of the CT-PH-SAC actor's log standard deviation, in place of -1) and input_matrix_start
("true", the default; "random": the learned plant's input matrix starts at standard
normal draws seeded by the training seed; or "zero": it starts at exactly zero;
common.ph_config.NETWORK_COLUMNLESS_KEYS). Neither has a table column; the CT-SAC arms
ignore both.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, List, Mapping, Optional, Sequence, Union

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from benchmarks.ph_hyperparams import (  # noqa: E402
    DEFAULT_HYPERPARAMS_DIR,
    declared_testbed_mismatches,
    find_ph_entry,
    load_ph_table,
)
from common.ph_config import (  # noqa: E402
    CONTACT_HANDLINGS,
    CT_SAC_REWARD_SOURCE_KEY,
    CT_SAC_REWARD_SOURCES,
    CT_SAC_REWARD_UNITS,
    CT_SAC_REWARD_UNITS_KEY,
    DEFAULT_FLAGGED_VALUE_GRADIENT,
    DEFAULT_MODEL_INTEGRATOR,
    FLAGGED_VALUE_GRADIENT_KEY,
    NETWORK_COLUMNLESS_KEYS,
    REFRESH_MODES,
    REWARD_OBSERVATION_KEY,
    REWARD_OBSERVATIONS,
    AdmissionConfig,
    NetworkConfig,
    RateConfig,
    SolverConfig,
    TemperatureConfig,
    TrainingConfig,
    check_refresh_mode,
    flagged_value_gradient_setting,
    validate_configuration,
)
from common.ph_dmc._common import (  # noqa: E402
    DISCOUNT_RATE_KEY,
    PHYSICAL_CHANGE_KEY,
    with_discount_rate,
    with_physical_change,
)
import environment.ph_sac.dmc  # noqa: E402,F401  (registers the dm_control ports)
import environment.ph_sac.tasks  # noqa: E402,F401  (registers the in-house testbeds)
from environment.ph_sac.registry import registered_testbeds, testbed_spec  # noqa: E402
from evaluations.ph_sac.protocol import (  # noqa: E402
    TheoryDiagnosticsConfig,
    baseline_feature_arms,
    baseline_implicit_arms,
    baseline_implicit_feature_arms,
    baseline_released_arms,
    build_protocol,
    impedance_decision_arms,
    implicit_ablation_arms,
    paper_arms,
    run_protocol,
    selectable_arms,
    step_ablation_arms,
    validate_matched_quantities,
)
from evaluations.ph_sac.reporting import (  # noqa: E402
    RECORD_NAME,
    load_run_records,
    plot_diagnostics,
    plot_learning_curves,
    plot_theory_diagnostics,
    write_comparison_report,
    write_theory_records,
)
from evaluations.ph_sac.resume import (  # noqa: E402
    DEFAULT_RESUME_EVERY,
    RESUMABLE_EXIT_CODE,
    STOP,
    ResumableStop,
    resume_options,
    stop_on_signals,
)

if TYPE_CHECKING:
    from argparse import Namespace

    from evaluations.ph_sac.protocol import EvaluationProtocol, RunResult
    from evaluations.ph_sac.resume import ResumeOptions

_TOP_LEVEL = (
    "testbed",
    "testbed_settings",
    "model_step",
    "network",
    "training",
    "temperature",
    "explicit",
    "implicit",
    "protocol",
)
# model_integrator (default DEFAULT_MODEL_INTEGRATOR, "rk4") sets the substep map of every
# model integration of both CT-PH-SAC arms; configs and table rows written before
# 2026-11-09 have none. refresh_mode (common.ph_config.REFRESH_MODES) replaces the
# testbed factory's controller refresh_mode ("continuous" in every factory): "held"
# reproduces the sample-and-hold controller of every run before 2026-11-09.
# model_coarse_step (since 2026-11-10; absent or null: off) is the coarse step of
# contact-triggered fine substepping (common.ph_config.model_coarse_step_setting), given
# to both CT-PH-SAC arms; table column model_coarse_step.
# contact_handling (since 2026-11-10; absent: "port") is contact exclusion
# (common.ph_config.CONTACT_HANDLINGS), given to both CT-PH-SAC arms; table column
# model_contact_handling ("exclude" on the cart-pole and point-mass rows since 2026-11-11);
# --contact-handling sets it on any protocol.
_OPTIONAL_TOP_LEVEL = ("model_integrator", "refresh_mode", "model_coarse_step", "contact_handling")
# Snapshots of the registry at import, by testbed name: the required and optional
# testbed_settings keys of every registered testbed, stubs included, and the config
# factories of the implemented ones. _build_testbed reads the registry itself.
_TESTBED_SETTINGS = {name: testbed_spec(name).settings_keys for name in registered_testbeds()}
_OPTIONAL_TESTBED_SETTINGS = {
    name: testbed_spec(name).optional_settings_keys
    for name in registered_testbeds()
    if testbed_spec(name).optional_settings_keys
}
_TESTBED_FACTORIES = {
    name: testbed_spec(name).config_factory
    for name in registered_testbeds(implemented_only=True)
}
_EXPLICIT = ("plant_source", "continuation", "polyak_rate", "rollout_paths")
# The implicit CT-SAC arms' optional section of "explicit" (no table column; absent: its
# defaults, common.ph_config.CT_SAC_IMPLICIT_DEFAULTS), e.g.
# "ct_sac_implicit": {"resolvent_step": 1.0, "inner_steps": 400, "policy_draws": 4}.
_CT_SAC_IMPLICIT_EXPLICIT = ("ct_sac_implicit",)
_IMPLICIT = (
    "plant_source",
    "reward_source",
    "reward_warm_steps",
    "plant_warm_steps",
    "plant_fit_covariance",
    "temperature_every_rows",
    "solver",
    "rate",
    "admission",
)
_PROTOCOL = (
    "interaction_budget",
    "temperature_per_second",
    "training_seeds",
    "evaluation_reset_seeds",
    "evaluation_every_interactions",
    "deterministic_evaluation",
    "ablation_polyak_rate",
    "baseline_expectation_samples",
    "output_directory",
)
# checkpoint_replay (default True) only sets what checkpoint.pt stores.
_OPTIONAL_PROTOCOL = ("theory_diagnostics", "checkpoint_replay")
# Settings of the adaptation experiment (docs/ct_ph_sac_layout.md), which no table row
# carries (no column): the protocol's warm_start (evaluations/ph_sac/adaptation.py) and the
# implicit arms' trainable_plant_parts (common.ph_config.trainable_plant_parts_setting).
# testbed_settings physical_change and discount_rate (_build_testbed) are column-less too.
# Absent: every earlier run.
_EXPERIMENT_PROTOCOL = ("warm_start",)
_EXPERIMENT_IMPLICIT = ("trainable_plant_parts",)
# The CT-SAC arms' reward units (common.ph_config.CT_SAC_REWARD_UNITS; no table column, so
# JSON configs and --ct-sac-reward-units only): "second" (absent, every earlier run) or
# "reference_period", the learner's rewards and temperature divided by T. Their reward
# source (common.ph_config.CT_SAC_REWARD_SOURCES; likewise JSON configs and
# --ct-sac-reward-source only): "interval" (absent, every earlier run) or "end_state", T
# times the reward rate at each interval's end state with the held torque.
_CT_SAC_PROTOCOL = (CT_SAC_REWARD_UNITS_KEY, CT_SAC_REWARD_SOURCE_KEY)
# Every arm's reward datum (common.ph_config.REWARD_OBSERVATIONS; no table column, so JSON
# configs and --reward-observation only): "interval" (absent, every earlier run) or
# "end_state", the reward rate at each interval's end state alone.
_REWARD_OBSERVATION_PROTOCOL = (REWARD_OBSERVATION_KEY,)
# The training key warmup_decisions (common.ph_config.warmup_decisions_setting; absent or
# null: the warmup_episodes episodes of every earlier run): a warm-up of exactly that many
# uniform decisions for every CT-PH-SAC arm, and the learning_starts of the CT-SAC arms
# without released settings. No table column and no TrainingConfig field (each of those
# has a column), so JSON configs only.
_COLUMNLESS_TRAINING = ("warmup_decisions",)
# The network keys actor_log_std_init and input_matrix_start
# (common.ph_config.NETWORK_COLUMNLESS_KEYS; absent: every earlier run): the CT-PH-SAC
# actor's starting log standard deviation and the learned plant's starting input matrix
# ("true", "random" or "zero"). No table column and no NetworkConfig field, so JSON
# configs only.
_COLUMNLESS_NETWORK = tuple(NETWORK_COLUMNLESS_KEYS)
# The value-step ablation's one protocol-level key (common.ph_config.STEP_ABLATION_DEFAULTS;
# no table column): Delta_ref of the implicit_critic arm, absent or null for the clock's
# mean interval. Its other keys are the step-ablation arms' own (protocol.OptionalArmFields).
_STEP_ABLATION_IMPLICIT = ("critic_reference_interval",)
# The value gradient at contact exclusion's flagged rows (common.ph_config.
# FLAGGED_VALUE_GRADIENTS; no table column, so JSON configs only): "full" (absent, every
# earlier run) or "stop_successor" (needs contact_handling "exclude"). Written into the
# implicit config only off its default, so that a stated "full" is the absent key.
_FLAGGED_VALUE_GRADIENT_IMPLICIT = (FLAGGED_VALUE_GRADIENT_KEY,)
# A local run of an ablation arm (the impedance-law implicit arm), shown by --list and its help.
ABLATION_ARM_EXAMPLE = (
    "python benchmarks/run_ph_sac.py --env-id cartpole-swingup --mode top "
    f"--arms {impedance_decision_arms()[0]['name']} --seeds 0"
)
# The arm sets list_arms names, in all_arms() order after the paper arms.
_ABLATION_SETS = (
    ("impedance ablation", impedance_decision_arms),
    ("implicit ablation", implicit_ablation_arms),
    ("step ablation", step_ablation_arms),
)
# Every set that runs only when --arms names it: the ablations, then the CT-SAC arms with
# CT-SAC's released hyperparameters (selectable_arms() order).
_NAMED_ONLY_SETS = (
    *_ABLATION_SETS,
    ("released baseline", baseline_released_arms),
    ("implicit baseline", baseline_implicit_arms),
    ("feature baseline", baseline_feature_arms),
    ("implicit feature baseline", baseline_implicit_feature_arms),
)


def _section(
    mapping: Mapping[str, object],
    keys: Sequence[str],
    where: str,
    *,
    optional: Sequence[str] = (),
) -> dict:
    """Return exactly the declared settings; a missing or unknown key is an error."""
    if not isinstance(mapping, Mapping):
        raise ValueError(f"{where} must be an object.")
    missing = [key for key in keys if key not in mapping]
    if missing:
        raise ValueError(f"{where} is missing required settings: {missing}.")
    unknown = sorted(set(mapping) - set(keys) - set(optional))
    if unknown:
        raise ValueError(f"{where} has unknown settings: {unknown}.")
    return dict(mapping)


def _typed_section(mapping: Mapping[str, object], schema: type, where: str) -> dict:
    return _section(
        mapping,
        tuple(sorted(schema.__required_keys__)),
        where,
        optional=tuple(sorted(schema.__optional_keys__)),
    )


def parse_args(argv: Optional[Sequence[str]] = None) -> Namespace:
    """Select protocols, optional arm/seed/budget overrides and report-only mode."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        action="append",
        help="Protocol JSON file; repeat to run several testbeds.",
    )
    parser.add_argument(
        "--env-id",
        action="append",
        help=(
            "Row of the hyperparameter table, e.g. cartpole-swingup, instead of a JSON "
            "file; repeat to run several testbeds."
        ),
    )
    parser.add_argument(
        "--mode", default="top", help="Table row mode for --env-id (default: top)."
    )
    parser.add_argument(
        "--hyperparams-dir",
        default=str(DEFAULT_HYPERPARAMS_DIR),
        help="Directory holding ct_ph_sac.csv (default: benchmarks/hyperparams).",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help=(
            "List the table rows and whether each builds a protocol, then the arms, then "
            f"exit. An ablation run: {ABLATION_ARM_EXAMPLE}"
        ),
    )
    parser.add_argument(
        "--arms",
        nargs="+",
        choices=[arm["name"] for arm in selectable_arms()],
        help=(
            "Run only these arms (default: the four paper arms, "
            f"{', '.join(arm['name'] for arm in paper_arms())}, all with torque "
            "decisions). The ablation arms run only when named here: the impedance-law "
            f"arms ({', '.join(arm['name'] for arm in impedance_decision_arms())}), the "
            "implicit-setting ablations and the value-step ablation arms (implicit_explicit_step, "
            "implicit_explicit_step_interval, implicit_critic, implicit_critic_head); so do "
            "the CT-SAC arms with CT-SAC's released hyperparameters "
            f"({', '.join(arm['name'] for arm in baseline_released_arms())}) and those "
            "whose networks read CT-PH-SAC's state features "
            f"({', '.join(arm['name'] for arm in baseline_feature_arms())}), the CT-SAC "
            "arms with an implicit value step "
            f"({', '.join(arm['name'] for arm in baseline_implicit_arms())}) and those with "
            "the state features too "
            f"({', '.join(arm['name'] for arm in baseline_implicit_feature_arms())})."
        ),
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, help="Override the training seeds."
    )
    parser.add_argument("--output", help="Override the output directory.")
    parser.add_argument(
        "--interactions", type=int, help="Override the matched interaction budget."
    )
    parser.add_argument(
        "--refresh-mode",
        choices=REFRESH_MODES,
        help=(
            "Override the controller refresh_mode of every arm (default: the config's, "
            "'continuous' in every table row); 'held' reproduces the runs before "
            "2026-11-09."
        ),
    )
    parser.add_argument(
        "--contact-handling",
        choices=CONTACT_HANDLINGS,
        help=(
            "Override the CT-PH-SAC arms' contact_handling (default: the config's; "
            "'exclude' in the cart-pole and point-mass table rows since 2026-11-11, 'port' "
            "in the others). 'exclude' (contact exclusion) also turns the learned plant's "
            "limit port off; 'port' removes the key (the table's convention) and, on a "
            "config that had 'exclude', turns the limit port back on."
        ),
    )
    parser.add_argument(
        "--ct-sac-reward-units",
        choices=CT_SAC_REWARD_UNITS,
        help=(
            "Override the protocol's ct_sac_reward_units, the reward and temperature units "
            "of every CT-SAC arm's learner (default: the config's, 'second' when absent, "
            "every earlier run): 'reference_period' divides its rewards, failure charge and "
            "temperature by the reference period T, so a reward rate of 1 per second is a "
            "reward of 1 per period and initial_alpha_c is its alpha. Other arms ignore it."
        ),
    )
    parser.add_argument(
        "--ct-sac-reward-source",
        choices=CT_SAC_REWARD_SOURCES,
        help=(
            "Override the protocol's ct_sac_reward_source, the reward every CT-SAC arm's "
            "learner receives per interval (default: the config's, 'interval' when absent, "
            "every earlier run: the environment's interval reward per reference period): "
            "'end_state' gives T times the task's reward rate at the state that ends the "
            "interval with the torque held over it, not scaled by the interval's duration "
            "(CT-SAC's own convention); with --ct-sac-reward-units reference_period that is "
            "the rate itself. The failure charge is unchanged. Other arms ignore it."
        ),
    )
    parser.add_argument(
        "--reward-observation",
        choices=REWARD_OBSERVATIONS,
        help=(
            "Override the protocol's reward_observation, the reward datum every arm's "
            "learner may use (default: the config's, 'interval' when absent, every earlier "
            "run: the environment's interval reward): 'end_state' leaves each learner only "
            "the task's reward rate at the interval's end state; the CT-SAC arms then take "
            "it as ct_sac_reward_source end_state (an explicit 'interval' there is refused) "
            "and every CT-PH-SAC arm learns a reward-rate network from it."
        ),
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help=(
            "Skip training and rebuild reports from the existing run records under "
            "<output>/<testbed>: every arm found there (whatever the default arm set), or "
            "only the arms --arms names; the protocol's training seeds (--seeds). No "
            "matching record is an error."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Make the runs restartable: continue a run from its resumable checkpoint "
            "(resume.pt) when there is one, skip finished runs, start the others afresh; "
            "write resume.pt while running, and on SIGUSR1/SIGTERM write it and exit "
            f"with code {RESUMABLE_EXIT_CODE}. Resubmit the same command to continue."
        ),
    )
    parser.add_argument(
        "--resume-every",
        type=int,
        default=DEFAULT_RESUME_EVERY,
        help=(
            "With --resume: also rewrite resume.pt every this many decisions (default "
            f"{DEFAULT_RESUME_EVERY}; 0: only after evaluations and signals)."
        ),
    )
    parser.add_argument(
        "--stop-after-interactions",
        type=int,
        help=(
            "With --resume, a testing aid: stop each run as a signal would once it "
            "crosses this many decisions (exit code "
            f"{RESUMABLE_EXIT_CODE}); a run resumed beyond it is not stopped again."
        ),
    )
    args = parser.parse_args(argv)
    if not args.list and not (args.config or args.env_id):
        parser.error("give --config or --env-id (or --list).")
    if args.resume_every < 0:
        parser.error("--resume-every must be nonnegative.")
    if args.stop_after_interactions is not None and not args.resume:
        parser.error("--stop-after-interactions needs --resume.")
    return args


def resume_settings(args: Namespace) -> Optional[ResumeOptions]:
    """The ResumeOptions of --resume, or None without it (runs as always)."""
    if not getattr(args, "resume", False):
        return None
    return resume_options(args.resume_every, args.stop_after_interactions)


def _build_testbed(name: str, settings: Mapping[str, object]) -> dict:
    """Check the settings against the registered keys and call the testbed's factory.

    A stub's factory raises NotImplementedError. Two settings have no table column and
    are applied to the factory's config: on a testbed whose TestbedSpec declares
    physical_change (the adaptation experiment), PHYSICAL_CHANGE_KEY changes the simulated
    robot (with_physical_change); on one with discount_setting, DISCOUNT_RATE_KEY replaces
    the discount rate (with_discount_rate). On any other testbed each is an unknown setting.
    """
    spec = testbed_spec(name)
    extra = []
    if spec.physical_change is not None:
        extra.append(PHYSICAL_CHANGE_KEY)
    if spec.discount_setting:
        extra.append(DISCOUNT_RATE_KEY)
    given = {}
    if extra and isinstance(settings, Mapping):
        given = {key: settings[key] for key in extra if key in settings}
        settings = {key: value for key, value in settings.items() if key not in extra}
    values = _section(
        settings,
        spec.settings_keys,
        f"testbed_settings for {name}",
        optional=spec.optional_settings_keys,
    )
    config = spec.config_factory(**values)
    if given.get(PHYSICAL_CHANGE_KEY) is not None:
        config = with_physical_change(config, given[PHYSICAL_CHANGE_KEY], spec.physical_change)
    if DISCOUNT_RATE_KEY in given:
        config = with_discount_rate(config, given[DISCOUNT_RATE_KEY])
    return config


def _implemented(name: object) -> bool:
    """Whether a table row's testbed is registered and implemented."""
    return isinstance(name, str) and name in registered_testbeds(implemented_only=True)


def load_protocol(config_path: str) -> EvaluationProtocol:
    """Parse explicit experiment settings without inventing unspecified defaults."""
    with open(config_path, encoding="utf-8") as handle:
        raw = json.load(handle)
    return protocol_from_config(raw, str(config_path))


def protocol_from_config(raw: Mapping[str, object], where: str) -> EvaluationProtocol:
    """Build the protocol of a config dict shaped like a JSON config; where names it."""
    raw = copy.deepcopy(dict(raw))
    raw.pop("_experiment_choices", None)
    top = _section(raw, _TOP_LEVEL, where, optional=_OPTIONAL_TOP_LEVEL)
    model_integrator = top.get("model_integrator", DEFAULT_MODEL_INTEGRATOR)

    testbed = _build_testbed(top["testbed"], top["testbed_settings"])
    if "refresh_mode" in top:
        testbed["controller"] = {
            **testbed["controller"],
            "refresh_mode": check_refresh_mode(top["refresh_mode"]),
        }
    network = _section(
        top["network"],
        tuple(sorted(NetworkConfig.__required_keys__)),
        "network",
        optional=tuple(sorted(NetworkConfig.__optional_keys__)) + _COLUMNLESS_NETWORK,
    )
    network["hidden_sizes"] = tuple(network["hidden_sizes"])
    network["plant_hidden_sizes"] = tuple(network["plant_hidden_sizes"])
    training = _section(
        top["training"],
        tuple(sorted(TrainingConfig.__required_keys__)),
        "training",
        optional=tuple(sorted(TrainingConfig.__optional_keys__)) + _COLUMNLESS_TRAINING,
    )
    temperature = _typed_section(top["temperature"], TemperatureConfig, "temperature")

    explicit = _section(
        top["explicit"], _EXPLICIT, "explicit", optional=_CT_SAC_IMPLICIT_EXPLICIT
    )
    explicit_config = {
        "testbed": testbed,
        "network": network,
        "training": training,
        "temperature": temperature,
        "model_step": top["model_step"],
        "model_integrator": model_integrator,
        **explicit,
    }

    implicit = _section(
        top["implicit"],
        _IMPLICIT,
        "implicit",
        optional=_EXPERIMENT_IMPLICIT + _STEP_ABLATION_IMPLICIT + _FLAGGED_VALUE_GRADIENT_IMPLICIT,
    )
    implicit_config = {
        "testbed": testbed,
        "network": network,
        "training": training,
        "temperature": temperature,
        "model_step": top["model_step"],
        "model_integrator": model_integrator,
        "plant_source": implicit["plant_source"],
        "reward_source": implicit["reward_source"],
        "reward_warm_steps": implicit["reward_warm_steps"],
        "plant_warm_steps": implicit["plant_warm_steps"],
        "plant_fit_covariance": implicit["plant_fit_covariance"],
        "temperature_every_rows": implicit["temperature_every_rows"],
        "solver": _typed_section(implicit["solver"], SolverConfig, "implicit.solver"),
        "rate": _typed_section(implicit["rate"], RateConfig, "implicit.rate"),
        "admission": _typed_section(
            implicit["admission"], AdmissionConfig, "implicit.admission"
        ),
    }
    if "model_coarse_step" in top:
        explicit_config["model_coarse_step"] = top["model_coarse_step"]
        implicit_config["model_coarse_step"] = top["model_coarse_step"]
    if "contact_handling" in top:
        explicit_config["contact_handling"] = top["contact_handling"]
        implicit_config["contact_handling"] = top["contact_handling"]
    for key in _EXPERIMENT_IMPLICIT + _STEP_ABLATION_IMPLICIT:
        if key in implicit:
            implicit_config[key] = implicit[key]
    # Checked here (a bad value raises); only off its default, so that a stated "full"
    # leaves the config, records and fingerprint those of a config without the key.
    gradient = flagged_value_gradient_setting(implicit)
    if gradient != DEFAULT_FLAGGED_VALUE_GRADIENT:
        implicit_config[FLAGGED_VALUE_GRADIENT_KEY] = gradient
    validate_configuration(explicit_config)
    validate_configuration(implicit_config)

    settings = _section(
        top["protocol"],
        _PROTOCOL,
        "protocol",
        optional=(
            _OPTIONAL_PROTOCOL
            + _EXPERIMENT_PROTOCOL
            + _CT_SAC_PROTOCOL
            + _REWARD_OBSERVATION_PROTOCOL
        ),
    )
    theory = settings.get("theory_diagnostics")
    if theory is not None:
        theory = _typed_section(
            theory, TheoryDiagnosticsConfig, "protocol.theory_diagnostics"
        )
    return build_protocol(
        testbed=top["testbed"],
        testbed_config=testbed,
        explicit_config=explicit_config,
        implicit_config=implicit_config,
        interaction_budget=settings["interaction_budget"],
        discount_rate=testbed["discount_rate"],
        temperature_per_second=settings["temperature_per_second"],
        training_seeds=settings["training_seeds"],
        evaluation_reset_seeds=settings["evaluation_reset_seeds"],
        evaluation_every_interactions=settings["evaluation_every_interactions"],
        deterministic_evaluation=settings["deterministic_evaluation"],
        ablation_polyak_rate=settings["ablation_polyak_rate"],
        output_directory=settings["output_directory"],
        theory_diagnostics=theory,
        baseline_expectation_samples=settings["baseline_expectation_samples"],
        checkpoint_replay=settings.get("checkpoint_replay", True),
        warm_start=settings.get("warm_start"),
        ct_sac_reward_units=settings.get(CT_SAC_REWARD_UNITS_KEY),
        ct_sac_reward_source=settings.get(CT_SAC_REWARD_SOURCE_KEY),
        reward_observation=settings.get(REWARD_OBSERVATION_KEY),
    )


def load_protocol_from_table(
    env_id: str,
    mode: str = "top",
    hyperparams_dir: Union[str, Path] = DEFAULT_HYPERPARAMS_DIR,
) -> EvaluationProtocol:
    """Build the protocol of one ct_ph_sac.csv row, as load_protocol does for a JSON.

    A row whose testbed is a registered stub (or not registered) raises
    NotImplementedError. The row's task_* columns must agree with the TestbedConfig that
    the factory declares.
    """
    entry = find_ph_entry(env_id, mode, hyperparams_dir)
    name = entry.config.get("testbed")
    if not _implemented(name):
        where = (
            "is registered as a stub (environment/ph_sac/dmc/) and not implemented yet"
            if name in registered_testbeds()
            else "is not registered in environment/ph_sac/registry.py"
        )
        raise NotImplementedError(
            f"{env_id} (mode {mode}) needs testbed {name!r}, which {where}; its row in "
            f"{entry.source} parses but cannot be built. Implemented testbeds: "
            f"{list(registered_testbeds(implemented_only=True))}."
        )
    where = f"{entry.source} ({env_id}, mode {mode})"
    protocol = protocol_from_config(entry.config, where)
    mismatches = declared_testbed_mismatches(
        entry.task, protocol["matched"]["testbed_config"]
    )
    if mismatches:
        raise ValueError(
            f"{where}: the task_* columns disagree with the {name} factory: "
            + "; ".join(mismatches)
        )
    return protocol


def list_table(hyperparams_dir: Union[str, Path] = DEFAULT_HYPERPARAMS_DIR) -> List[str]:
    """One line per table row: its key, testbed, task size, clock, budget and status."""
    lines = [
        f"{'env_id':<28} {'mode':<6} {'testbed':<24} {'nq/nv/nu':<9} "
        f"{'H_e':>5} {'T':>6} {'budget':>8}  status"
    ]
    for entry in load_ph_table(hyperparams_dir):
        task, config = entry.task, entry.config
        name = config.get("testbed")
        if _implemented(name):
            load_protocol_from_table(entry.env_id, entry.mode, hyperparams_dir)
            status = "builds"
        elif name in registered_testbeds():
            status = "parses; not implemented"
        else:
            status = "parses; not registered"
        size = f"{task.get('nq')}/{task.get('nv')}/{task.get('nu')}"
        budget = config.get("protocol", {}).get("interaction_budget")
        lines.append(
            f"{entry.env_id:<28} {entry.mode:<6} {name:<24} {size:<9} "
            f"{task.get('episode_duration'):>5} {task.get('reference_period'):>6} "
            f"{budget:>8}  {status}"
        )
    return lines


def list_arms() -> List[str]:
    """One line per arm, as --list prints them: name, decisions, algorithm and arm set.

    "paper" arms (torque decisions) run by default; the ablation arms ("impedance
    ablation", "implicit ablation", "step ablation") and the CT-SAC arms with CT-SAC's
    released hyperparameters ("released baseline"), each "(only with --arms)", run only
    when --arms names them. The last line is an example invocation of one.
    """
    sets = {arm["name"]: "paper" for arm in paper_arms()}
    for label, arms in _NAMED_ONLY_SETS:
        sets.update({arm["name"]: f"{label} (only with --arms)" for arm in arms()})
    lines = [f"{'arm':<32} {'decisions':<10} {'algorithm':<10} set"]
    for arm in selectable_arms():
        lines.append(
            f"{arm['name']:<32} {arm['decision_mode']:<10} {arm['algorithm']:<10} "
            f"{sets[arm['name']]}"
        )
    lines.append(f"example: {ABLATION_ARM_EXAMPLE}")
    return lines


def load_protocols(args: Namespace) -> List[EvaluationProtocol]:
    """The protocols of every --config file, then of every --env-id row, overridden."""
    protocols = [load_protocol(path) for path in args.config or ()]
    protocols.extend(
        load_protocol_from_table(env_id, args.mode, args.hyperparams_dir)
        for env_id in args.env_id or ()
    )
    return [apply_overrides(protocol, args) for protocol in protocols]


def apply_overrides(protocol: EvaluationProtocol, args: Namespace) -> EvaluationProtocol:
    """Apply command-line selections, then recheck the matched quantities."""
    protocol = copy.deepcopy(protocol)
    if args.arms:
        # Selected from every declared arm, so an ablation or released-baseline arm can
        # join a protocol.
        protocol["arms"] = [arm for arm in selectable_arms() if arm["name"] in args.arms]
    if args.seeds:
        protocol["training_seeds"] = list(args.seeds)
    if args.output:
        protocol["output_directory"] = args.output
    if args.interactions is not None:
        protocol["matched"]["interaction_budget"] = int(args.interactions)
    handling = getattr(args, "contact_handling", None)
    if handling is not None:
        for name in ("explicit_config", "implicit_config"):
            config = protocol[name]
            if handling == "exclude":
                config["contact_handling"] = "exclude"
                config["network"] = {**config["network"], "limit_port": False}
            elif config.pop("contact_handling", "port") == "exclude":
                # An exclusion row (no limit port) back to the port convention.
                config["network"] = {**config["network"], "limit_port": True}
            validate_configuration(config)
    units = getattr(args, "ct_sac_reward_units", None)
    if units is not None:
        protocol[CT_SAC_REWARD_UNITS_KEY] = units
    source = getattr(args, "ct_sac_reward_source", None)
    if source is not None:
        protocol[CT_SAC_REWARD_SOURCE_KEY] = source
    observation = getattr(args, "reward_observation", None)
    if observation is not None:
        protocol[REWARD_OBSERVATION_KEY] = observation
    mode = getattr(args, "refresh_mode", None)
    if mode is not None:
        for testbed in (
            protocol["matched"]["testbed_config"],
            protocol["explicit_config"]["testbed"],
            protocol["implicit_config"]["testbed"],
        ):
            testbed["controller"] = {**testbed["controller"], "refresh_mode": mode}
    validate_matched_quantities(protocol)
    return protocol


def run_experiments(
    protocols: Sequence[EvaluationProtocol], resume: Optional[ResumeOptions] = None
) -> Sequence[RunResult]:
    """Run the selected arms on each MuJoCo testbed and seed, in sequence."""
    results: list[RunResult] = []
    for protocol in protocols:
        results.extend(run_protocol(protocol, resume))
    return results


def _existing_results(
    protocol: EvaluationProtocol, arms: Optional[Sequence[str]] = None
) -> Sequence[RunResult]:
    """The run records under <output>/<testbed> of the protocol's training seeds
    (--report-only).

    arms None (no --arms): every arm with a record there, whatever the default arm set, so
    a folder of impedance-law or other ablation runs reports all of them; otherwise only
    the named arms. Raises FileNotFoundError when there is no record, or none of the
    selection, naming the (arm, seed) pairs present: a report is never written empty.
    """
    root = Path(protocol["output_directory"]) / protocol["testbed"]
    paths = sorted(str(path) for path in root.rglob(RECORD_NAME))
    if not paths:
        raise FileNotFoundError(f"No run records under {root}.")
    seeds = set(protocol["training_seeds"])
    loaded = load_run_records(paths)
    selected = [
        result
        for result in loaded
        if (arms is None or result["run"]["arm"]["name"] in arms)
        and result["run"]["training_seed"] in seeds
    ]
    if not selected:
        present = sorted(
            {(result["run"]["arm"]["name"], result["run"]["training_seed"]) for result in loaded}
        )
        wanted = "any arm" if arms is None else f"arms {sorted(arms)}"
        raise FileNotFoundError(
            f"No run records under {root} for {wanted} and training seeds {sorted(seeds)}; "
            f"the records there are of (arm, seed) {present}."
        )
    return selected


def write_reports(protocol: EvaluationProtocol, results: Sequence[RunResult]) -> Sequence[str]:
    """Write the comparison report and every figure next to the run records."""
    output = str(Path(protocol["output_directory"]) / protocol["testbed"] / "report")
    paths = [write_comparison_report(protocol, results, output)]
    for x_axis in ("interactions", "gradient_steps", "wall_seconds"):
        paths.extend(plot_learning_curves(results, output, x_axis=x_axis))
    paths.extend(plot_diagnostics(results, output))
    records = [record for result in results for record in result.get("theory_records", [])]
    if records:
        paths.extend(write_theory_records(records, output))
        paths.extend(plot_theory_diagnostics(records, output))
    return paths


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Dispatch the experiments, then generate each protocol's report.

    With --resume a stop (signal, or --stop-after-interactions) ends the process with
    exit code RESUMABLE_EXIT_CODE once the resumable checkpoint is written.
    """
    args = parse_args(argv)
    if args.list:
        for line in [*list_table(args.hyperparams_dir), *list_arms()]:
            print(line)
        return
    protocols = load_protocols(args)
    resume = None if args.report_only else resume_settings(args)
    if resume is None:
        for protocol in protocols:
            results = (
                _existing_results(protocol, args.arms)
                if args.report_only
                else run_protocol(protocol)
            )
            for path in write_reports(protocol, results):
                print(path)
        return
    try:
        with stop_on_signals():
            for protocol in protocols:
                if STOP.requested:
                    raise ResumableStop(f"stop requested (signal {STOP.signal})")
                results = run_protocol(protocol, resume)
                for path in write_reports(protocol, results):
                    print(path)
    except ResumableStop as stop:
        print(f"Stopped for a restart: {stop}. Rerun the same command (with --resume).",
              file=sys.stderr, flush=True)
        sys.exit(RESUMABLE_EXIT_CODE)


if __name__ == "__main__":
    main()

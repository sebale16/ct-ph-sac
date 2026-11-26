# CT-PH-SAC

## What this repository contains

This repository holds the code of four continuous-time soft actor-critic methods for control on an irregular decision clock, where each torque action is held over an interval of random length. The tasks are dm_control tasks ported to MuJoCo models. CT-SAC is the model-free baseline. The other three methods are variants of CT-PH-SAC, which learns a port-Hamiltonian model of the plant and a network for the reward rate. Generator CT-PH-SAC and rollout CT-PH-SAC take an explicit value step through the learned model: their critic targets continue the value with the model's generator or with model paths integrated over the interval. Implicit CT-PH-SAC replaces the explicit value step by an implicit (resolvent) step. The repository also holds the configurations, task lists and job scripts of the paper's experiments (`experiments/paper/`).

## Relation to the CT-SAC code base

This repository builds on the CT-SAC code base [mpnguyen2/ct-rl](https://github.com/mpnguyen2/ct-rl). Its files are kept unchanged, with three exceptions: `.gitignore` and `requirements.txt` are extended (the latter by `scipy` and `mujoco`), and its README is moved, unchanged, to `README_ct-rl.md`. The MuJoCo models of the dm_control ports are adapted from dm_control (see `NOTICE`). The CT-SAC baseline of the experiments runs that code base's CT-SAC through an adapter (`algorithms/ph_baseline.py`).

## Layout

Files of the CT-PH-SAC code:

| Path | Contents |
|---|---|
| `benchmarks/run_ph_sac.py` | Entry point for every training run of the four methods and the ablation arms |
| `algorithms/implicit_ct_ph_sac.py`, `algorithms/explicit_ct_ph_sac.py` | Implicit CT-PH-SAC; generator and rollout CT-PH-SAC |
| `algorithms/ph_baseline.py` | Adapter that runs CT-SAC in the same protocol |
| `algorithms/ph_*.py`, `algorithms/ct_sac_implicit.py` | Shared training loop, targets, resolvent residuals, admission statistics; CT-SAC with an implicit value step (not part of the paper's comparison) |
| `models/ph_sac/` | Networks, port-Hamiltonian plant, generators, model paths, reward-rate network, losses |
| `environment/ph_sac/` | MuJoCo interval environment, decision clock, task registry; `dmc/` holds the dm_control ports |
| `common/ph_config.py`, `common/ph_dmc/`, `common/ph_*.py` | Configuration schemas and per-task config factories, replay, time and type definitions |
| `evaluations/ph_sac/` | Evaluation protocol and arms, run records and reports, exact restarts (`--resume`) |
| `benchmarks/hyperparams/ct_ph_sac.csv`, `benchmarks/ph_hyperparams.py`, `benchmarks/make_ct_ph_sac_table.py` | Hyperparameter table of the task rows and the script that writes it |
| `benchmarks/configs/ph_sac/` | Example protocol configs (not the paper's; those are in `experiments/paper/configs/`) |
| `benchmarks/ph_plant_checkpoint_accuracy.py` | Accuracy of a finished run's learned plant on its own replay |
| `benchmarks/ph_*.py` (others) | Diagnostics and summaries used during development |
| `tests/test_ph_*.py`, `tests/test_ct_sac_implicit.py` | Tests of the CT-PH-SAC code |
| `tests/verify_ph_suite.sh` | Runs the `tests/test_ph_*.py` modules in batches, and an end-to-end restart check |
| `experiments/paper/` | Configurations, task lists, node sets and job scripts of the paper's experiments |

The other files belong to the CT-SAC code base. The code is released as it ran. It therefore also contains tasks, arms and settings that the paper does not use (for example the windy pendulum, reacher and swimmer tasks, and the example configs in `benchmarks/configs/ph_sac/`), and its comments, example configs and hyperparameter table carry development notes and refer to files that are not part of this release: documents under `docs/`, helper scripts, and sibling folders named by relative paths. The paper's experiments are exactly those of `experiments/paper/`.

## Installation

```bash
pip install -r requirements.txt
```

`requirements.txt` lists the packages without versions. `README_ct-rl.md` creates a Python 3.9 environment for the CT-SAC code. The CT-PH-SAC modules import `NotRequired` from `typing` (`common/ph_types.py`, `common/ph_config.py`), which Python provides from version 3.11 on. The job scripts check the environment with `import torch, mujoco, numpy, scipy, dm_control` and set `MALLOC_ARENA_MAX=2 MUJOCO_GL=disable MPLBACKEND=Agg PYTHONUNBUFFERED=1` (`experiments/paper/all20_worker_b.sh`). Every config in `experiments/paper/` sets `"device": "cpu"`. The paper's runs used Python 3.11.8, PyTorch 2.12.0 (CPU build), MuJoCo 3.8.1, dm_control 1.0.41 and NumPy 2.4.6.

## One training run of each method

Every run of the paper was one line of a task list in `experiments/paper/tasks/`, of the form `name|threads|arguments`. The worker started each line from the repository root as (`experiments/paper/final_lib.sh`, function `launch_group`):

```bash
env OMP_NUM_THREADS=<threads> MKL_NUM_THREADS=<threads> OPENBLAS_NUM_THREADS=1 \
    python -u benchmarks/run_ph_sac.py <arguments> --output <run folder> --resume
```

In the task lists the config paths start with `<RUN_ROOT>/final/2026-11-23_r20/configs/` (in `experiments/paper/fgtest/tasks/`, with `<RUN_ROOT>/final/2026-11-23_r20/fgtest/configs/`), the cluster folders that `experiments/paper/configs/` and `experiments/paper/fgtest/configs/` reproduce. Written with the configs of this repository, seed 0 of each method on `lqr_2_1` is (arguments from `experiments/paper/tasks/c20_lqr_lqr_2_1.txt`, `g20_lqr_lqr_2_1.txt`, `r20_lqr_lqr_2_1.txt`, `i20_lqr_lqr_2_1.txt`):

```bash
# CT-SAC (1 thread)
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    python -u benchmarks/run_ph_sac.py --config experiments/paper/configs/m19_lqr_lqr_2_1.json \
    --arms ct_sac_released_features --seeds 0 --output <run folder> --resume
# Generator CT-PH-SAC (2 threads)
env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 \
    python -u benchmarks/run_ph_sac.py --config experiments/paper/configs/r20_lqr_lqr_2_1.json \
    --arms explicit_generator_torque --seeds 0 --output <run folder> --resume
# Rollout CT-PH-SAC (2 threads)
env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 \
    python -u benchmarks/run_ph_sac.py --config experiments/paper/configs/r20_lqr_lqr_2_1.json \
    --arms explicit_rollout_torque --seeds 0 --output <run folder> --resume
# Implicit CT-PH-SAC (2 threads)
env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 \
    python -u benchmarks/run_ph_sac.py --config experiments/paper/configs/r20_lqr_lqr_2_1.json \
    --arms implicit_torque --seeds 0 --output <run folder> --resume
```

A run writes its records under `<run folder>/<testbed>/<arm>/seed_<seed>/`. With `--resume` a stopped run (SIGUSR1 or SIGTERM, exit code 75) continues exactly when the same command is given again, and a finished run is skipped. The arms of the value-step ablation take the same form with `--config experiments/paper/configs/r20abl_lqr_lqr_2_1.json` and `--arms implicit_critic_head`, `implicit_explicit_step_interval` or `implicit_explicit_step` (`experiments/paper/tasks/a20h_*.txt`, `a20i_*.txt`, `a20t_*.txt`). The plant-accuracy measurement ran, per task (`experiments/paper/plantacc/plantacc_dev.slurm`):

```bash
python -u benchmarks/ph_plant_checkpoint_accuracy.py --runs <run folder>/<testbed>/implicit_torque/seed_0 ... \
    --out <csv file> --rows 20000 --threads 8
```

## The paper's experiments

On the cluster, `experiments/paper/` was the folder `<RUN_ROOT>/final/2026-11-23_r20`. The configs and task lists are the files that ran, with the run root replaced by `<RUN_ROOT>`. Each task list has a one-line header written for this release. The job scripts are kept as examples, with site-specific values replaced by placeholders; they expect a staged code archive, an environment archive and gate files that are not part of this release. The runs of the main comparison and of the ablation ran on the nodes of one job (`all20b.slurm`), which packs the node sets `sets/v1.txt` to `sets/v9.txt` (lists of task lists) onto nodes (`all20_driver_b.sh`) and starts one worker per node (`all20_worker_b.sh`, `final_lib.sh`). The runs of implicit CT-PH-SAC on the tasks with joint limits were started as further workers on nodes of the same job (`fgtest/`). The job also ran groups that the paper does not report; they are removed from the node sets here, and their task lists and configs are not included. One run (`fg_cartpole_three_poles`, seed 5) was stopped at 480k decisions by the end of the job and was finished by a later job with the same command and `--resume`.

| Experiment | Configs (`experiments/paper/`) | Task lists (`experiments/paper/`) | Launched by |
|---|---|---|---|
| Main comparison, CT-SAC, 12 task rows, 10 seeds, 500k decisions | `configs/m19_<task>.json`, `configs/m19dmc_<task>.json` | `tasks/c20_<task>.txt`, `tasks/c20d_<task>_dmc.txt` | `all20b.slurm` |
| Main comparison, generator and rollout CT-PH-SAC | `configs/r20_<task>.json`, `configs/r20dmc_<task>.json` | `tasks/g20_*`, `tasks/g20d_*`, `tasks/r20_*`, `tasks/r20d_*` | `all20b.slurm` |
| Main comparison, implicit CT-PH-SAC on the five tasks without joint limits | `configs/r20_acrobot_swingup.json`, `configs/r20_lqr_lqr_2_1.json`, `configs/r20_lqr_lqr_6_2.json`, `configs/r20dmc_<task>.json` | `tasks/i20_acrobot_swingup.txt`, `tasks/i20_lqr_lqr_*.txt`, `tasks/i20d_*` | `all20b.slurm` |
| Main comparison, implicit CT-PH-SAC on the seven tasks with joint limits | `fgtest/configs/r20fg_<task>.json` | `fgtest/tasks/fg_<task>.txt`, grouped by `fgtest/sets/v1.txt` and `fgtest/sets/v2.txt` | `fgtest/start_on_node.sh` (set `fgtest/sets/v1.txt`), `fgtest/guest_start.sh` (set `fgtest/sets/v2.txt`), on nodes of the job of `all20b.slurm` |
| Value-step ablation on `lqr_2_1` and `lqr_6_2`, 3 arms, 10 seeds, 300k decisions | `configs/r20abl_lqr_lqr_<n>.json` | `tasks/a20h_*` (critic head), `tasks/a20i_*` (explicit step, step size the interval), `tasks/a20t_*` (explicit step, step size the resolvent step) | `all20b.slurm`; implicit CT-PH-SAC enters through `tasks/i20_lqr_lqr_*.txt` |
| Accuracy of the learned plant at the end of training | the runs above | the runs above | `plantacc/plantacc_dev.slurm` (three cart-pole tasks), `plantacc/plantacc9_dev.slurm` (nine other tasks), both with `benchmarks/ph_plant_checkpoint_accuracy.py` |
| Plant error recorded during training | the runs above | the runs above | written by the runs themselves (`oracle.csv` in each run folder) |
| Run times | the runs above | the runs above | the `wall_seconds` counter in each run's record |

Task rows and file names: `pendulum/swingup` and `acrobot/swingup` with dm_control's rewards are `<task>_dmc` (configs `m19dmc_`, `r20dmc_`); `acrobot/swingup` with the Lyapunov reward is `acrobot_swingup`; the six `cartpole` tasks are `cartpole_<task>`; `point_mass/easy` is `point_mass_easy`; the two `lqr` tasks are `lqr_lqr_2_1` and `lqr_lqr_6_2`. The joint-limit tasks are the six `cartpole` tasks and `point_mass_easy`. On these tasks implicit CT-PH-SAC ran with `r20fg_<task>.json`, which is `r20_<task>.json` with `"flagged_value_gradient": "stop_successor"` changed to `"full"`. The argument line with which `fgtest/guest_start.sh` was called is not recorded.

## Tests

The cluster checks compiled the code and ran each test module in its own process, from the repository root:

```bash
python -m compileall -q algorithms common benchmarks evaluations environment models tests
env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python -m unittest tests.<module>
```

with one such command for every `tests/test_*.py`. In the cluster check of this code, `tests.test_ph_dmc_swimmer`, `tests.test_ph_refresh_mode` and `tests.test_ph_torque_decisions` did not pass, and the other modules did. `tests/verify_ph_suite.sh` runs the CT-PH-SAC tests in batches and an end-to-end restart check (`bash tests/verify_ph_suite.sh [PYTHON] [LOG]`).

## Revision used for the paper's runs

The code is the archive that produced the paper's training runs, file for file, with these differences:

- `tests/test_ph_flagged_gradient.py` passes `retain_graph=True` in one test helper. This is the version of the archive that the cluster check ran. No training run executes this file.
- `benchmarks/ph_plant_checkpoint_accuracy.py` is added. The plant-accuracy jobs copied this file into the archive's code before running it.
- Four example configs in `benchmarks/configs/ph_sac/adaptation/` and `environment/ph_sac/dmc/README.md` have a path prefix and a cluster name replaced. The paper's experiments do not use them.
- Comment text is reworded in five places (`benchmarks/make_ct_ph_sac_table.py`, `environment/ph_sac/tasks.py`, `tests/test_ph_hyperparams.py`), and with it one sentence in the comment column of eleven rows of `benchmarks/hyperparams/ct_ph_sac.csv`. No code and no value changes.
- `README.md` is rewritten for this release, and `README_ct-rl.md`, `NOTICE`, `experiments/` and `sample_images/` are not in the archive.

## Citation

To be added.

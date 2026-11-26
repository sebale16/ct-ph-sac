#!/bin/bash
# Verify exact restarts (--resume), the plant-step schedule and the whole CT-PH-SAC suite,
# in batches at 2 threads, from the root of an unpacked code tar (no git needed). Usage:
#
#   bash tests/verify_ph_suite.sh [PYTHON] [LOG]
#
# PYTHON: the interpreter of the environment (default $PYTHON, else python3).
# LOG: the log file (default verify_ph_suite_<host>_<date>.log in the repository root).
# Steps, each logged with its time, exit code and unittest summary:
#   A  tests.test_ph_resume (restarts and plant-step schedule), the new tests
#   B1-B3  every other tests/test_ph_*.py file, in three batches (any file not listed
#      below is run in an extra batch B4, so nothing is skipped)
#   C  end to end through the command line: a smoke run stopped by
#      --stop-after-interactions must exit 75 with resume.pt; rerun with --resume it must
#      finish (exit 0, resume_complete.json, no resume.pt) with the evaluation returns and
#      counters of an uninterrupted run without --resume
# Exit code: 0 if every step passed, 1 otherwise. Runs one Python process at a time.
set -u
cd "$(dirname "$0")/.."
ROOT=$(pwd)
PY=${1:-${PYTHON:-python3}}
LOG=${2:-$ROOT/verify_ph_suite_$(hostname -s)_$(date +%Y%m%d_%H%M%S).log}
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONUNBUFFERED=1
SCRATCH=$(mktemp -d "${TMPDIR:-/tmp}/verify_ph_suite.XXXXXX")
trap 'rm -rf "$SCRATCH"' EXIT

A="tests.test_ph_resume"
B1="tests.test_ph_algorithms tests.test_ph_baseline tests.test_ph_dmc tests.test_ph_dmc_acrobot_reward tests.test_ph_dmc_cartpole tests.test_ph_dmc_joint_limits tests.test_ph_dmc_lqr tests.test_ph_dmc_pendulum_acrobot"
B2="tests.test_ph_dmc_reacher_point_mass tests.test_ph_dmc_swimmer tests.test_ph_env tests.test_ph_evaluation tests.test_ph_hyperparams tests.test_ph_model_integrator tests.test_ph_model_step tests.test_ph_models"
B3="tests.test_ph_operators tests.test_ph_refresh_mode tests.test_ph_replay tests.test_ph_reward_head tests.test_ph_theory tests.test_ph_torque_decisions tests.test_ph_value_update_stability"
LISTED=" $A $B1 $B2 $B3 "
B4=""
for file in tests/test_ph_*.py; do
  module="tests.$(basename "$file" .py)"
  case "$LISTED" in *" $module "*) ;; *) B4="$B4 $module" ;; esac
done

log() { echo "$@" | tee -a "$LOG"; }
FAILED=0
SUMMARY=()

run_batch() {
  local name=$1; shift
  log ""
  log "=== batch $name $(date +%T): $*"
  local start=$(date +%s)
  "$PY" -m unittest "$@" >"$SCRATCH/batch.log" 2>&1
  local code=$?
  cat "$SCRATCH/batch.log" >>"$LOG"
  local ran=$(grep -E "^Ran [0-9]+ tests" "$SCRATCH/batch.log" | tail -1)
  local verdict=$(grep -E "^(OK|FAILED)" "$SCRATCH/batch.log" | tail -1)
  local seconds=$(( $(date +%s) - start ))
  log "=== batch $name exit $code after ${seconds}s: ${ran:-no summary} ${verdict:-}"
  SUMMARY+=("$name exit=$code ${seconds}s ${ran:-no summary} ${verdict:-}")
  [ $code -eq 0 ] || FAILED=1
}

: >"$LOG"
log "verify_ph_suite $(date '+%F %T') on $(hostname) in $ROOT"
log "python: $PY"
"$PY" -c "import sys, torch, numpy, mujoco; print('python', sys.version.split()[0], 'torch', torch.__version__, 'numpy', numpy.__version__, 'mujoco', mujoco.__version__, 'threads', torch.get_num_threads())" 2>&1 | tee -a "$LOG"

run_batch A $A
run_batch B1 $B1
run_batch B2 $B2
run_batch B3 $B3
if [ -n "$B4" ]; then run_batch B4 $B4; fi

# --- C: the command line, a real process: stop (exit 75), resume, compare -----------
log ""
log "=== step C $(date +%T): command-line stop and resume"
CLI=(benchmarks/run_ph_sac.py --config benchmarks/configs/ph_sac/smoke_windy.json
     --arms explicit_rollout --seeds 0 --interactions 1200)
"$PY" "${CLI[@]}" --output "$SCRATCH/reference" >>"$LOG" 2>&1
reference=$?
"$PY" "${CLI[@]}" --output "$SCRATCH/restarted" --resume --resume-every 300 \
    --stop-after-interactions 700 >>"$LOG" 2>&1
stopped=$?
SEED_DIR="$SCRATCH/restarted/windy_pendulum/explicit_rollout/seed_0"
[ -f "$SEED_DIR/resume.pt" ] && had_checkpoint=1 || had_checkpoint=0
"$PY" "${CLI[@]}" --output "$SCRATCH/restarted" --resume >>"$LOG" 2>&1
resumed=$?
"$PY" - "$SCRATCH/reference/windy_pendulum/explicit_rollout/seed_0/run_record.json" \
    "$SEED_DIR/run_record.json" "$SEED_DIR" >>"$LOG" 2>&1 <<'EOF'
import json, sys
from pathlib import Path

def load(path):
    result = json.load(open(path))["result"]
    returns = [e["discounted_return"] for c in result["evaluations"] for e in c["episodes"]]
    counters = {k: v for k, v in result["counters"].items() if k != "wall_seconds"}
    return returns, counters

reference, restarted, folder = sys.argv[1], sys.argv[2], Path(sys.argv[3])
same = load(reference) == load(restarted)
complete = (folder / "resume_complete.json").exists() and not (folder / "resume.pt").exists()
print("C compare: returns and counters equal:", same, "| completion marker, no resume.pt:", complete)
sys.exit(0 if same and complete else 1)
EOF
compared=$?
log "=== step C: reference exit $reference (want 0), stop exit $stopped (want 75), resume.pt after the stop $had_checkpoint (want 1), resume exit $resumed (want 0), comparison exit $compared (want 0)"
if [ $reference -eq 0 ] && [ $stopped -eq 75 ] && [ $had_checkpoint -eq 1 ] && [ $resumed -eq 0 ] && [ $compared -eq 0 ]; then
  SUMMARY+=("C pass")
else
  SUMMARY+=("C FAIL")
  FAILED=1
fi

log ""
log "=== summary $(date +%T)"
for line in "${SUMMARY[@]}"; do log "  $line"; done
log "=== overall: $([ $FAILED -eq 0 ] && echo PASS || echo FAIL) (log $LOG)"
exit $FAILED

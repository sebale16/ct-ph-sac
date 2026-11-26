#!/bin/bash
# EXAMPLE of how the paper's runs were launched on a SLURM cluster, kept as it ran except that site-specific values
# (allocation, partition, node names, paths, cluster names in comments) are replaced by <placeholders> and the RUN_ROOT variable.
# It expects files that are not part of this release (a staged code archive, an environment archive, gate and check files).
#SBATCH --job-name=plantacc
#SBATCH --partition=<dev_partition>
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --time=01:30:00
#SBATCH --account=<allocation>
#SBATCH --output=<RUN_ROOT>/final/2026-11-23_r20/logs/%x_%j.out
#SBATCH --error=<RUN_ROOT>/final/2026-11-23_r20/logs/%x_%j.err
# 2026-11-24. READ-ONLY diagnostic: loads the saved checkpoints of the full-gradient implicit runs on cartpole_three_poles,
# cartpole_two_poles and cartpole_swingup (10 seeds each) and prints the learned input matrix beside the true one
# and the drift error relative to the true drift (plantacc/ph_plant_checkpoint_accuracy.py, a new script that has
# never run: the job first compiles it and runs it on ONE run, and stops if that fails). Code: the archive the
# runs used, unpacked in /tmp. Writes only under plantacc/out_<job> (new folder) and node-local /tmp.
set -uo pipefail
S=${RUN_ROOT:?set RUN_ROOT to the run root}
F=$S/final/2026-11-23_r20
P=$F/plantacc
O=$P/out_$SLURM_JOB_ID
LOCAL=/tmp/plantacc_$SLURM_JOB_ID
trap 'cd /; rm -rf "$LOCAL"' EXIT
echo "start $(date) $(hostname) job $SLURM_JOB_ID"
[ -e "$O" ] && { echo "$O exists"; exit 1; }
mkdir -p "$O" "$LOCAL/tmp" || exit 1
TAR=$S/stage/$(cat "$F/code_tar.name")
[ "$(md5sum < "$TAR" | cut -d' ' -f1)" = "$(cat "$F/code_tar.md5")" ] || { echo "tar md5 mismatch"; exit 1; }
tar -C "$LOCAL" -xf "$TAR" && tar -C "$LOCAL" -xf "$S/stage/venv.tar" || { echo "could not unpack"; exit 1; }
CODE=$LOCAL/ct-ph-rl
PY=$LOCAL/.venv/bin/python
cp "$P/ph_plant_checkpoint_accuracy.py" "$CODE/benchmarks/" || exit 1
md5sum "$TAR" "$CODE/benchmarks/ph_plant_checkpoint_accuracy.py" | tee "$O/md5.txt"
export MALLOC_ARENA_MAX=2 MUJOCO_GL=disable MPLBACKEND=Agg PYTHONUNBUFFERED=1 TMPDIR=$LOCAL/tmp
cd "$CODE" || exit 1
"$PY" -m py_compile benchmarks/ph_plant_checkpoint_accuracy.py || { echo "COMPILE FAILED"; exit 1; }
echo "compiled"
run_dir() { echo "$F/fgtest/runs/fg_$1/implicit_torque_s$2/$1/implicit_torque/seed_$2"; }
# smoke: one run, small sample
"$PY" -u benchmarks/ph_plant_checkpoint_accuracy.py --runs "$(run_dir cartpole_swingup 0)" --out "$O/smoke.csv" --rows 2000 --threads 8 > "$O/smoke.log" 2>&1
RC=$?
echo "smoke rc=$RC; csv lines $(wc -l < "$O/smoke.csv" 2>/dev/null)"
tail -25 "$O/smoke.log" | cut -c1-400
if [ "$RC" -ne 0 ] || [ ! -s "$O/smoke.csv" ]; then echo "SMOKE FAILED: full pass not started"; grep -n "Traceback" -A30 "$O/smoke.log" | head -60 | cut -c1-300; exit 2; fi
for t in cartpole_three_poles cartpole_two_poles cartpole_swingup; do
  ( dirs=""; for s in 0 1 2 3 4 5 6 7 8 9; do dirs="$dirs $(run_dir $t $s)"; done
    "$PY" -u benchmarks/ph_plant_checkpoint_accuracy.py --runs $dirs --out "$O/acc_$t.csv" --rows 20000 --threads 8 > "$O/acc_$t.log" 2>&1
    echo "$t rc=$? csv lines $(wc -l < "$O/acc_$t.csv" 2>/dev/null) +$SECONDS s" ) &
done
wait
grep -c "Traceback" "$O"/acc_*.log
echo "end $(date)"

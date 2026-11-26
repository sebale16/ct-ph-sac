#!/bin/bash
# EXAMPLE of how the paper's runs were launched on a SLURM cluster, kept as it ran except that site-specific values
# (allocation, partition, node names, paths, cluster names in comments) are replaced by <placeholders> and the RUN_ROOT variable.
# It expects files that are not part of this release (a staged code archive, an environment archive, gate and check files).
#SBATCH --job-name=plantacc9
#SBATCH --partition=<dev_partition>
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --time=01:00:00
#SBATCH --account=<allocation>
#SBATCH --output=<RUN_ROOT>/final/2026-11-23_r20/logs/%x_%j.out
#SBATCH --error=<RUN_ROOT>/final/2026-11-23_r20/logs/%x_%j.err
# 2026-11-24. READ-ONLY diagnostic, second part: the plant-accuracy measurement of job 1167548
# (plantacc/ph_plant_checkpoint_accuracy.py, md5 2398c6bf...) on the saved checkpoints of implicit CT-PH-SAC on the
# nine other tasks of the paper, 10 seeds each, 20,000 replay rows per run. Code: the archive the runs used,
# unpacked in /tmp. Writes only under plantacc/out_<job> (new folder) and node-local /tmp.
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
# label:group folder under F
for spec in pendulum_swingup:runs/i20d_pendulum_swingup_dmc acrobot_swingup:runs/i20d_acrobot_swingup_dmc \
            acrobot_swingup_lyapunov:runs/i20_acrobot_swingup lqr_lqr_2_1:runs/i20_lqr_lqr_2_1 \
            lqr_lqr_6_2:runs/i20_lqr_lqr_6_2 cartpole_balance:fgtest/runs/fg_cartpole_balance \
            cartpole_balance_sparse:fgtest/runs/fg_cartpole_balance_sparse \
            cartpole_swingup_sparse:fgtest/runs/fg_cartpole_swingup_sparse \
            point_mass_easy:fgtest/runs/fg_point_mass_easy; do
  t=${spec%%:*}; g=${spec#*:}
  ( dirs=""; for s in 0 1 2 3 4 5 6 7 8 9; do d=$(ls -d "$F/$g"/*/*/*/seed_$s 2>/dev/null | head -1); dirs="$dirs $d"; done
    echo "$t: $(echo $dirs | wc -w) run folders"
    "$PY" -u benchmarks/ph_plant_checkpoint_accuracy.py --runs $dirs --out "$O/acc_$t.csv" --rows 20000 --threads 8 > "$O/acc_$t.log" 2>&1
    echo "$t rc=$? csv lines $(wc -l < "$O/acc_$t.csv" 2>/dev/null) +$SECONDS s; $(grep -h '^done:' "$O/acc_$t.log" | tail -1)" ) &
done
wait
grep -c "Traceback" "$O"/acc_*.log
echo "end $(date)"

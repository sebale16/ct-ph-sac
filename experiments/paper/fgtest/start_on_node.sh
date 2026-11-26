#!/bin/bash
# EXAMPLE of how the paper's runs were launched on a SLURM cluster, kept as it ran except that site-specific values
# (allocation, partition, node names, paths, cluster names in comments) are replaced by <placeholders> and the RUN_ROOT variable.
# It expects files that are not part of this release (a staged code archive, an environment archive, gate and check files).
# 2026-11-23. Run ON node <node> (n8 of the job 1163331): starts the test's worker detached with the environment of
# the job's own worker on this node (read from /proc), so the test's runs see what the job's runs see. The first
# start (label fg) ended at the import check: a plain ssh shell has no library path for the environment's python.
set -uo pipefail
F=${RUN_ROOT:?set RUN_ROOT to the run root}/final/2026-11-23_r20
R=$F/fgtest
LABEL=fgb
[ -e "$R/all_driver/${LABEL}_worker.pid" ] && { echo "already started"; exit 1; }
P=$(pgrep -u "$USER" -f "all20_worker_b.sh n8" | head -1)
[ -n "$P" ] || { echo "the job's worker is not on this node"; exit 1; }
N=0
while IFS= read -r -d '' kv; do
    case "$kv" in
        ALL20_*|SLURM_JOB_NAME=*|END_EPOCH=*|MIN_CPUS=*|_=*|SHLVL=*|PWD=*|OLDPWD=*|TMPDIR=*) ;;
        *=*) export "$kv"; N=$((N + 1)) ;;
    esac
done < "/proc/$P/environ"
export SLURM_JOB_NAME=fgtest END_EPOCH=1795607128 MIN_CPUS=128 ALL20_MARGIN=2700 ALL20_ROOT=$R ALL20_MIN_MEM_MB=150000 ALL20_HOW=ssh
[ "${SLURM_JOB_ID:-}" = 1163331 ] || { echo "SLURM_JOB_ID is ${SLURM_JOB_ID:-unset}"; exit 1; }
cd "$F" || exit 1
setsid nohup bash "$F/all20_worker_b.sh" "$LABEL" v1 >> "$R/logs/fgtest_1163331_$LABEL.out" 2>&1 < /dev/null &
echo $! > "$R/all_driver/${LABEL}_worker.pid"
echo "started worker pid $! with $N variables of pid $P on $(hostname) at $(date)"

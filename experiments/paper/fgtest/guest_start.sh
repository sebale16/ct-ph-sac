#!/bin/bash
# EXAMPLE of how the paper's runs were launched on a SLURM cluster, kept as it ran except that site-specific values
# (allocation, partition, node names, paths, cluster names in comments) are replaced by <placeholders> and the RUN_ROOT variable.
# It expects files that are not part of this release (a staged code archive, an environment archive, gate and check files).
# 2026-11-23. Run ON a node of the job 1163331: starts a guest worker detached in the node's idle cores, with the
# environment of a worker already on the node (read from /proc), so the guest's runs see what the job's runs see.
#   bash guest_start.sh <pattern of the worker to copy the environment from> <guest label> <root> <worker script> <set> [<set> ...]
# The guest ends with the job at the latest (watchdog at the job's end - 2700 s; runs then stop resumable).
set -uo pipefail
[ $# -ge 5 ] || { echo "usage: guest_start.sh <pattern> <label> <root> <worker> <set>..."; exit 1; }
PATTERN=$1; LABEL=$2; R=$3; WORKER=$4; shift 4
F=${RUN_ROOT:?set RUN_ROOT to the run root}/final/2026-11-23_r20
[ -d "$R/sets" ] && [ -f "$WORKER" ] || { echo "no $R/sets or no $WORKER"; exit 1; }
mkdir -p "$R/all_driver" "$R/logs"
[ -e "$R/all_driver/${LABEL}_worker.pid" ] && { echo "label $LABEL already started"; exit 1; }
P=$(pgrep -u "$USER" -f "$PATTERN" | head -1)
[ -n "$P" ] || { echo "no process matches '$PATTERN' on $(hostname)"; exit 1; }
N=0
while IFS= read -r -d '' kv; do
    case "$kv" in
        ALL20_*|SLURM_JOB_NAME=*|END_EPOCH=*|MIN_CPUS=*|_=*|SHLVL=*|PWD=*|OLDPWD=*|TMPDIR=*|BASH_FUNC_*) ;;
        *=*) export "$kv"; N=$((N + 1)) ;;
    esac
done < "/proc/$P/environ"
export SLURM_JOB_NAME=fgtest END_EPOCH=1795607128 MIN_CPUS=128 ALL20_MARGIN=2700 ALL20_ROOT=$R ALL20_MIN_MEM_MB=100000 ALL20_HOW=ssh
[ "${SLURM_JOB_ID:-}" = 1163331 ] || { echo "SLURM_JOB_ID is ${SLURM_JOB_ID:-unset}"; exit 1; }
cd "$F" || exit 1
setsid nohup bash "$WORKER" "$LABEL" "$@" >> "$R/logs/fgtest_1163331_$LABEL.out" 2>&1 < /dev/null &
echo $! > "$R/all_driver/${LABEL}_worker.pid"
echo "started worker pid $! (label $LABEL, sets $*) with $N variables of pid $P on $(hostname) at $(date)"

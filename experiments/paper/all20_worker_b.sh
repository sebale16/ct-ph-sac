#!/bin/bash
# EXAMPLE of how the paper's runs were launched on a SLURM cluster, kept as it ran except that site-specific values
# (allocation, partition, node names, paths, cluster names in comments) are replaced by <placeholders> and the RUN_ROOT variable.
# It expects files that are not part of this release (a staged code archive, an environment archive, gate and check files).
# 2026-11-23. SECOND VERSION (all20_worker_b.sh): the usable processors are counted without OMP_NUM_THREADS, which
# made the first version refuse every node in the check acheck_1123_031526. Nothing else changed.
# 2026-11-23. WORKER of all20_driver.sh: one node's part of the single <cluster> job that holds every run of the
# rerun.
#   bash all20_worker.sh <label> <set> [<set> ...]
# It starts every incomplete, unlocked run of the named sets (sets/<set>.txt lists groups, tasks/<group>.txt the
# runs: name|threads|run_ph_sac.py arguments) through final_lib.sh, exactly as final_runs_r20.slurm does on one
# node: each run with --resume, output in runs/<group>/<name>. A watchdog on the job's own clock sends SIGUSR1
# MARGIN seconds before the job's end (END_EPOCH, from the driver), so the runs write resume.pt and exit 75; the
# same command in a later job continues them exactly and skips the finished ones. A set with no incomplete run
# gets all_<set>/COMPLETE. Exit 4 without starting anything when fewer than MIN_CPUS processors are usable here
# (the driver then starts the worker another way). Nothing starts if STOP_all20 exists in the folder.
# Environment from the driver: SLURM_JOB_ID, SLURM_JOB_NAME, END_EPOCH; optional ALL20_ROOT (sets, tasks, runs,
# jobs and state under another root: used only by the check), ALL20_HOW (how it was started), MIN_CPUS,
# ALL20_MARGIN. On every exit it leaves its exit code in all_driver/done_<jobname>_<job>_<label>_<how>.
# Writes only under the root (runs/, jobs/, all_<set>/, all_driver/) and node-local /tmp. Nothing is moved or
# deleted.
set -uo pipefail
[ $# -ge 2 ] || { echo "usage: bash all20_worker.sh <label> <set> [<set> ...] (inside all20_driver.sh)"; exit 1; }
LABEL=$1; shift
SETS_HERE=("$@")
S=${RUN_ROOT:?set RUN_ROOT to the run root}
F=$S/final/2026-11-23_r20
ROOT=${ALL20_ROOT:-$F}
LOCAL=/tmp/all20_${SLURM_JOB_ID:-nojob}_$LABEL
# On every exit: the node-local folder goes, and the exit code is left for the driver (it waits for this file
# when the worker was started detached through ssh).
trap 'rc=$?; cd /; rm -rf "$LOCAL"; mkdir -p "$ROOT/all_driver"; echo "$rc" > "$ROOT/all_driver/done_${SLURM_JOB_NAME:-noname}_${SLURM_JOB_ID:-nojob}_${LABEL}_${ALL20_HOW:-direct}"' EXIT
MIN_CPUS=${MIN_CPUS:-128}
MARGIN=${ALL20_MARGIN:-2700}
# nproc alone answers OMP_NUM_THREADS (1 in every job here), so both variables are removed for the count
CPUS=$(env -u OMP_NUM_THREADS -u OMP_THREAD_LIMIT nproc)
if [ "$CPUS" -lt "$MIN_CPUS" ]; then
    echo "CPUS RESTRICTED: nproc=$CPUS on $(hostname), fewer than $MIN_CPUS: nothing started"
    exit 4
fi
: "${SLURM_JOB_ID:?no SLURM_JOB_ID}" "${SLURM_JOB_NAME:?no SLURM_JOB_NAME}" "${END_EPOCH:?no END_EPOCH}"
command -v squeue > /dev/null || { echo "NO SQUEUE in PATH on $(hostname): nothing started"; exit 5; }
TAR=$S/stage/$(cat "$F/code_tar.name")
EXPECTED_MD5=$(cat "$F/code_tar.md5")
RUNS=$ROOT/runs
WORKRES=$ROOT/no_work_copy          # final_lib.sh also looks here for finished runs; never exists
J=$ROOT/jobs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}_$LABEL
T0=$(date +%s)
el() { echo $(( $(date +%s) - T0 )); }
echo "start $(date) $(hostname) job $SLURM_JOB_ID label $LABEL sets ${SETS_HERE[*]}; $CPUS processors usable; started ${ALL20_HOW:-direct}"
GROUPS_LIST=()
for s in "${SETS_HERE[@]}"; do
    [ -f "$ROOT/sets/$s.txt" ] || { echo "no $ROOT/sets/$s.txt"; exit 1; }
    while read -r g; do
        [ -z "$g" ] && continue
        [[ $g == \#* ]] && continue
        [ -f "$ROOT/tasks/$g.txt" ] || { echo "no $ROOT/tasks/$g.txt"; exit 1; }
        GROUPS_LIST+=("$g")
    done < "$ROOT/sets/$s.txt"
done
[ "${#GROUPS_LIST[@]}" -ge 1 ] || { echo "no group in ${SETS_HERE[*]}"; exit 1; }
[ -e "$F/STOP_all20" ] && { echo "$F/STOP_all20 present: nothing started"; exit 0; }
mkdir -p "$J" "$RUNS" "$LOCAL/tmp" || exit 1
set_incomplete() {  # SET: its runs with no resume_complete.json
    local n=0 g name rest
    while read -r g; do
        [ -z "$g" ] && continue
        [[ $g == \#* ]] && continue
        while IFS='|' read -r name rest; do
            [ -z "$name" ] && continue
            [[ $name == \#* ]] && continue
            compgen -G "$RUNS/$g/$name/*/*/seed_*/resume_complete.json" > /dev/null || n=$((n + 1))
        done < "$ROOT/tasks/$g.txt"
    done < "$ROOT/sets/$1.txt"
    echo $n
}
mark_complete() {  # every set of this worker with no incomplete run; prints the incomplete total
    local total=0 s n
    for s in "${SETS_HERE[@]}"; do
        n=$(set_incomplete "$s")
        total=$((total + n))
        if [ "$n" -eq 0 ]; then mkdir -p "$ROOT/all_$s"; touch "$ROOT/all_$s/COMPLETE"; fi
    done
    echo $total
}
INCOMPLETE=$(mark_complete)
echo "$INCOMPLETE runs incomplete in ${SETS_HERE[*]}"
[ "$INCOMPLETE" -eq 0 ] && { echo "every run is finished: nothing to start"; exit 0; }

mem_avail_mb() { awk '/^MemAvailable/{print int($2 / 1024)}' /proc/meminfo; }
echo "MemAvailable_mb: $(mem_avail_mb); cpus allowed: $(grep Cpus_allowed_list /proc/self/status | cut -f2)" | tee "$J/node_health.txt"
if [ "$(mem_avail_mb)" -lt "${ALL20_MIN_MEM_MB:-150000}" ]; then
    echo "NODE CHECK FAILED: only $(mem_avail_mb) MB available on $(hostname)"; touch "$J/NODE_CHECK_FAILED"; exit 3
fi
[ "$(md5sum < "$TAR" | cut -d' ' -f1)" = "$EXPECTED_MD5" ] || { echo "tar md5 mismatch"; exit 1; }
tar -C "$LOCAL" -xf "$TAR" && tar -C "$LOCAL" -xf "$S/stage/venv.tar" || { echo "could not unpack"; exit 1; }
echo "unpacked +$(el)s"
CODE=$LOCAL/ct-ph-rl
PY=$LOCAL/.venv/bin/python
[ "$(md5sum < "$CODE/benchmarks/hyperparams/ct_ph_sac.csv" | cut -d' ' -f1)" = "$(cat "$F/verified/ct_ph_sac.csv.md5")" ] \
    || { echo "the tar's table is not the verified one"; exit 1; }
export MALLOC_ARENA_MAX=2 MUJOCO_GL=disable MPLBACKEND=Agg PYTHONUNBUFFERED=1 TMPDIR=$LOCAL/tmp
cd "$CODE" || exit 1
"$PY" -c "import torch, mujoco, numpy, scipy, dm_control; print('imports ok', torch.__version__, mujoco.__version__, numpy.__version__)" \
    || { echo "import check failed"; exit 1; }
md5sum "$TAR" > "$J/tar_md5.txt"
# shellcheck source=/dev/null
source "$F/final_lib.sh"
# A lock holds only while its job is pending, running or completing, so a continuation that starts when its
# predecessor has just ended takes over that job's runs (as in final_runs_r20.slurm).
run_locked() {
    local dir=$1 owner
    [ -f "$dir/LOCK" ] || return 1
    owner=$(cat "$dir/LOCK" 2>/dev/null)
    [ -n "$owner" ] && [ "$owner" != "${SLURM_JOB_ID:-none}" ] || return 1
    [ -n "$(squeue -h -j "$owner" -t PENDING,RUNNING,COMPLETING -o %i 2>/dev/null)" ]
}
trap 'stop_requested; date +%F_%T > "$J/STOP_FORWARDED"' USR1
trap 'echo "$(date +%F_%T) TERM: forwarding"; forward_signal TERM' TERM

LEFT_S=$(( END_EPOCH - $(date +%s) ))
STOP_IN=$(( LEFT_S - MARGIN ))
[ "$STOP_IN" -lt 30 ] && STOP_IN=30
DEADLINE=$(( END_EPOCH - 60 ))
echo "time left at start: $LEFT_S s; watchdog SIGUSR1 at +$STOP_IN s"
# stdin and output closed: a leftover sleep must not keep an ssh or srun channel open after the worker ends
STOP_AT=$(( $(date +%s) + STOP_IN ))
( while [ "$(date +%s)" -lt "$STOP_AT" ]; do sleep 15; done; kill -USR1 $$ ) > /dev/null 2>&1 < /dev/null &
WATCHDOG=$!

for GROUP in "${GROUPS_LIST[@]}"; do
    launch_group "$GROUP" "$ROOT/tasks/$GROUP.txt"
done
echo "launched ${#RUN_NAMES[@]} runs, skipped $SKIPPED_COMPLETE complete and $SKIPPED_LOCKED held elsewhere +$(el)s"
printf '%s\n' "${RUN_NAMES[@]}" > "$J/runs.txt"
( while true; do
    echo "$(date +%F_%T) +$(el)s load $(cut -d' ' -f1-3 /proc/loadavg) $(free -m | awk '/Mem:/{print "used_mb="$3, "avail_mb="$7}') py=$(pgrep -c -f "^$PY ")" >> "$J/progress.txt"
    sleep 1800
  done ) > /dev/null 2>&1 < /dev/null &
SAMPLER=$!

wait_runs "$DEADLINE"
echo "all runs ended (or the limit is near) +$(el)s"
collect_status "$J/status.txt"
kill "$WATCHDOG" "$SAMPLER" 2>/dev/null
INCOMPLETE=$(mark_complete)
echo "label $LABEL sets ${SETS_HERE[*]}: finished $(grep -c ' finished$' "$J/status.txt"), stopped_resumable $(grep -c ' stopped_resumable$' "$J/status.txt"), failed $(grep -c ' failed$' "$J/status.txt"), still_running $(grep -c ' still_running$' "$J/status.txt"); $INCOMPLETE runs incomplete"
grep -v ' finished$' "$J/status.txt" | grep -v ' stopped_resumable$' | head -20
echo "end $(date) +$(el)s"

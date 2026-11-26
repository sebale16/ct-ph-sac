#!/bin/bash
# EXAMPLE of how the paper's runs were launched on a SLURM cluster, kept as it ran except that site-specific values
# (allocation, partition, node names, paths, cluster names in comments) are replaced by <placeholders> and the RUN_ROOT variable.
# It expects files that are not part of this release (a staged code archive, an environment archive, gate and check files).
# 2026-11-23. SECOND VERSION (all20_driver_b.sh): worker all20_worker_b.sh, the continuation id in
# all_driver/next_job_b, and the driver works from the shared folder. Nothing else changed.
# 2026-11-23. DRIVER of the single <cluster> job that holds every run of the rerun. Started by all20.slurm after its gate check (and by the check
# check_vista_all20.sh on two small test sets under another root).
#   * Sets sets/v<n>.txt (groups of run lists): v1..v8 = the eight node sets of the <second cluster> chain
#     (<second-cluster folder>: four methods on eleven tasks, the value-step ablation, three poles with the second
#     replay size) plus the eleven groups of CT-SAC with the implicit value step in their free cores; v9 = the four
#     methods on acrobot and pendulum with dm_control's original reward. 730 runs, 1,210 threads.
#   * At its start it counts the threads of each set's incomplete runs and packs the sets onto nodes, first fit,
#     at most CAP threads a node: nine nodes in the first job, fewer in a continuation. Node groups beyond the
#     job's nodes are left for a later job and named in the log.
#   * One worker (all20_worker.sh) per node: on this node directly, on the others with srun; a worker that srun
#     did not start, or that found too few usable processors, is started detached through ssh instead, and the
#     driver then waits for the exit-code file the worker leaves.
#   * Slurm's SIGUSR1 (MARGIN seconds before the limit) is forwarded to the worker on this node; every worker has
#     its own watchdog on the job's clock, so every run is told to stop either way, writes resume.pt and exits 75.
#   * When every set is complete it cancels the continuation job named in all_driver/next_job_b, if any.
# Nothing starts if STOP_all20 exists in the folder. Writes only under the root (logs/, all_driver/, and what the
# workers write) and node-local /tmp. Nothing is moved or deleted.
set -uo pipefail
S=${RUN_ROOT:?set RUN_ROOT to the run root}
F=$S/final/2026-11-23_r20
ROOT=${ALL20_ROOT:-$F}
WORKER=$F/all20_worker_b.sh
D=$ROOT/all_driver
CAP=${ALL20_CAP:-140}
MARGIN=${ALL20_MARGIN:-2700}
export ALL20_MARGIN=$MARGIN
export ALL20_ROOT=$ROOT
export MIN_CPUS=${MIN_CPUS:-128}
T0=$(date +%s)
el() { echo $(( $(date +%s) - T0 )); }
mkdir -p "$D" "$ROOT/logs" || exit 1
cd "$F" || exit 1          # a folder every node has: srun starts its steps in the current one
note() { echo "$(date +%F_%T) job $SLURM_JOB_ID: $*" | tee -a "$D/driver.log"; }
echo "driver start $(date) $(hostname) job $SLURM_JOB_ID nodes $SLURM_JOB_NODELIST root $ROOT"
[ -f "$WORKER" ] || { echo "no $WORKER"; exit 1; }
[ -e "$F/STOP_all20" ] && { note "$F/STOP_all20 present: nothing started"; exit 0; }
# shellcheck source=/dev/null
source "$F/final_lib.sh"      # run_complete, to_seconds
mapfile -t SETS < <(ls "$ROOT/sets" | grep -E '^v[0-9]+\.txt$' | sed 's/\.txt$//' | sort -V)
[ "${#SETS[@]}" -ge 1 ] || { note "no set in $ROOT/sets"; exit 1; }
if [ -n "${ALL20_END_EPOCH:-}" ]; then
    END_EPOCH=$ALL20_END_EPOCH
else
    LEFT_S=$(to_seconds "$(squeue -h -j "$SLURM_JOB_ID" -o %L)")
    [ "${LEFT_S:-0}" -gt 0 ] 2>/dev/null || { note "could not read the job's remaining time"; exit 1; }
    END_EPOCH=$(( $(date +%s) + LEFT_S ))
fi
export END_EPOCH
set_threads() {  # SET: threads of its incomplete runs
    local n=0 g name threads rest
    while read -r g; do
        [ -z "$g" ] && continue
        [[ $g == \#* ]] && continue
        while IFS='|' read -r name threads rest; do
            [ -z "$name" ] && continue
            [[ $name == \#* ]] && continue
            run_complete "$ROOT/runs/$g/$name" || n=$((n + threads))
        done < "$ROOT/tasks/$g.txt"
    done < "$ROOT/sets/$1.txt"
    echo $n
}
pack() {  # fills BIN_SETS and BIN_T: first fit of the sets with incomplete runs, at most CAP threads a bin
    local s t i placed
    BIN_SETS=(); BIN_T=(); STATE=""
    for s in "${SETS[@]}"; do
        t=$(set_threads "$s")
        STATE="$STATE $s=$t"
        if [ "$t" -eq 0 ]; then mkdir -p "$ROOT/all_$s"; touch "$ROOT/all_$s/COMPLETE"; continue; fi
        placed=0
        for i in "${!BIN_T[@]}"; do
            if [ $(( BIN_T[i] + t )) -le "$CAP" ]; then
                BIN_SETS[$i]="${BIN_SETS[$i]} $s"; BIN_T[$i]=$(( BIN_T[i] + t )); placed=1; break
            fi
        done
        [ "$placed" -eq 1 ] || { BIN_SETS+=("$s"); BIN_T+=("$t"); }
    done
}
cancel_next() {
    local next
    next=$(cat "$D/next_job_b" 2>/dev/null || true)
    [ -n "$next" ] && [ "$next" != "$SLURM_JOB_ID" ] || return 0
    [ -n "$(squeue -h -j "$next" -t PENDING -o %i 2>/dev/null)" ] || return 0
    scancel "$next" 2>/dev/null
    note "continuation $next cancelled (nothing left to run)"
}
pack
note "threads of incomplete runs per set:$STATE; node groups needed: ${#BIN_T[@]}; job ends at $(date -d "@$END_EPOCH" +%F_%T)"
if [ "${#BIN_T[@]}" -eq 0 ]; then
    note "every set is complete"
    touch "$D/COMPLETE"
    cancel_next
    exit 0
fi
mapfile -t NODES < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
BATCH_NODE=$(hostname -s)
ORDERED=("$BATCH_NODE")
for n in "${NODES[@]}"; do [ "$n" = "$BATCH_NODE" ] || ORDERED+=("$n"); done
if [ "${#ORDERED[@]}" -lt "${#BIN_T[@]}" ]; then
    LEFT_OUT=""
    for i in "${!BIN_T[@]}"; do [ "$i" -ge "${#ORDERED[@]}" ] && LEFT_OUT="$LEFT_OUT [${BIN_SETS[$i]}]"; done
    note "only ${#ORDERED[@]} nodes for ${#BIN_T[@]} node groups: left for a later job:$LEFT_OUT"
fi

PIDS=(); WNODE=(); WLABEL=(); WSETS=(); WHOW=()
forward_usr1() {
    local k
    for k in "${!PIDS[@]}"; do   # only to the worker that is a child shell; the others stop on their own watchdog
        [ "${WHOW[$k]:-}" = direct ] && kill -USR1 "${PIDS[$k]}" 2>/dev/null
    done
    date +%F_%T > "$D/USR1_forwarded_$SLURM_JOB_ID"
}
trap forward_usr1 USR1
ssh_worker() {  # NODE LABEL SETS LOG: detached on the node, so a dropped connection does not end it
    ssh -o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=20 "$1" \
        "export SLURM_JOB_ID=$SLURM_JOB_ID SLURM_JOB_NAME=$SLURM_JOB_NAME END_EPOCH=$END_EPOCH MIN_CPUS=$MIN_CPUS ALL20_MARGIN=$MARGIN ALL20_ROOT=$ROOT RUN_ROOT=$RUN_ROOT ALL20_MIN_MEM_MB=${ALL20_MIN_MEM_MB:-150000} ALL20_HOW=ssh PATH=$PATH LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}; setsid nohup bash $WORKER $2 $3 >> $4 2>&1 < /dev/null &" \
        >> "$4" 2>&1
}
for i in "${!BIN_T[@]}"; do
    [ "$i" -lt "${#ORDERED[@]}" ] || break
    node=${ORDERED[$i]}; label=n$((i + 1)); sets=${BIN_SETS[$i]}
    log=$ROOT/logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}_$label.out
    if [ "$node" = "$BATCH_NODE" ]; then
        # shellcheck disable=SC2086
        ALL20_HOW=direct bash "$WORKER" "$label" $sets > "$log" 2>&1 &
        PIDS+=($!); WHOW+=(direct)
    else
        # shellcheck disable=SC2086
        ALL20_HOW=srun srun --overlap -N1 -n1 --cpu-bind=none -w "$node" bash "$WORKER" "$label" $sets > "$log" 2>&1 &
        PIDS+=($!); WHOW+=(srun)
    fi
    WNODE+=("$node"); WLABEL+=("$label"); WSETS+=("$sets")
    note "$label: sets $sets (${BIN_T[$i]} threads) on $node (${WHOW[-1]})"
done
sleep 60
for k in "${!PIDS[@]}"; do
    [ "${WHOW[$k]}" = srun ] || continue
    log=$ROOT/logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}_${WLABEL[$k]}.out
    if ! kill -0 "${PIDS[$k]}" 2>/dev/null && { ! grep -q '^start ' "$log" 2>/dev/null || grep -q 'CPUS RESTRICTED' "$log" 2>/dev/null; }; then
        note "${WLABEL[$k]}: srun did not give a usable worker ($(tail -1 "$log" 2>/dev/null | cut -c1-160)); starting it detached through ssh on ${WNODE[$k]}"
        ssh_worker "${WNODE[$k]}" "${WLABEL[$k]}" "${WSETS[$k]}" "$log"
        note "${WLABEL[$k]}: ssh returned $?"
        PIDS[$k]=0; WHOW[$k]=ssh
    fi
done
worker_alive() {  # index
    if [ "${WHOW[$1]}" = ssh ]; then
        [ ! -e "$D/done_${SLURM_JOB_NAME}_${SLURM_JOB_ID}_${WLABEL[$1]}_ssh" ]
    else
        kill -0 "${PIDS[$1]}" 2>/dev/null
    fi
}
while :; do
    alive=0
    for k in "${!PIDS[@]}"; do worker_alive "$k" && alive=$((alive + 1)); done
    [ "$alive" -eq 0 ] && break
    [ "$(date +%s)" -ge $(( END_EPOCH - 30 )) ] && { note "the limit is reached with $alive workers not reported"; break; }
    sleep 15
done
RCS=""
for k in "${!PIDS[@]}"; do
    if [ "${WHOW[$k]}" = ssh ]; then rc=$(cat "$D/done_${SLURM_JOB_NAME}_${SLURM_JOB_ID}_${WLABEL[$k]}_ssh" 2>/dev/null || echo "?")
    else wait "${PIDS[$k]}"; rc=$?; fi
    RCS="$RCS ${WLABEL[$k]}(${WHOW[$k]})=$rc"
done
note "workers ended +$(el)s, exit codes$RCS"

pack
note "threads of incomplete runs per set:$STATE; node groups needed: ${#BIN_T[@]}"
if [ "${#BIN_T[@]}" -eq 0 ]; then
    note "every set is complete"
    touch "$D/COMPLETE"
    cancel_next
else
    note "runs remain: the continuation named in $D/next_job_b ($(cat "$D/next_job_b" 2>/dev/null || echo none)) continues them"
fi
echo "driver end $(date) +$(el)s"

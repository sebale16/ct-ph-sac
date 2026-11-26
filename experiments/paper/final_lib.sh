#!/bin/bash
# EXAMPLE of how the paper's runs were launched on a SLURM cluster, kept as it ran except that site-specific values
# (allocation, partition, node names, paths, cluster names in comments) are replaced by <placeholders> and the RUN_ROOT variable.
# It expects files that are not part of this release (a staged code archive, an environment archive, gate and check files).
# Shared by final_runs.slurm, lqr_diag.slurm and verify.slurm (phase 3, the launcher test).
# Written 2026-11-11 from exclude/2026-11-10/exclude_lib.sh, with the two job-script fixes
# of <project log> section 4 item 6:
#   * the skip check looks in BOTH scratch and work: a run is complete when its folder in
#     $RUNS/<group>/<name> or in $WORKRES/<group>/<name> holds a resume_complete.json
#     (benchmarks/run_ph_sac.py --resume writes it when the run has finished);
#   * SIGUSR1 is forwarded to every run (forward_signal), which writes resume.pt at its next
#     safe point and exits with code 75, so the same command continues it after the 48 h
#     limit (--resume).
# A run being trained by another live job (a lock file with that job's id, still in squeue)
# is skipped as well, so two jobs never write one run.
#
# Functions (the caller sets RUNS, WORKRES, J, PY, CODE and installs the traps):
#   run_complete DIR...         0 if any DIR holds */*/seed_*/resume_complete.json
#   run_locked DIR              0 if DIR/LOCK names a job other than this one still queued
#   launch_group GROUP FILE     start every incomplete, unlocked run of a task file
#   forward_signal SIG          send SIG to every launched run still alive
#   alive_runs                  how many launched runs are alive
#   wait_runs DEADLINE_EPOCH    wait (interruptibly) until every run ended or the deadline
#   collect_status FILE         "group/name rc verdict" per launched run
# Task file lines: name|threads|arguments of benchmarks/run_ph_sac.py without --output and
# --resume (the launcher adds --output $RUNS/<group>/<name> --resume), no quotes.
declare -A RUN_PID=()
declare -A RUN_RC=()
declare -A RUN_DIR=()
RUN_NAMES=()
SKIPPED_COMPLETE=0
SKIPPED_LOCKED=0
STOP_FORWARDED=0

run_complete() {
    local dir
    for dir in "$@"; do
        compgen -G "$dir/*/*/seed_*/resume_complete.json" > /dev/null && return 0
    done
    return 1
}

run_locked() {
    local dir=$1 owner
    [ -f "$dir/LOCK" ] || return 1
    owner=$(cat "$dir/LOCK" 2>/dev/null)
    [ -n "$owner" ] && [ "$owner" != "${SLURM_JOB_ID:-none}" ] || return 1
    [ -n "$(squeue -h -j "$owner" -o %i 2>/dev/null)" ]
}

launch_group() {  # GROUP TASKFILE
    local group=$1 file=$2 name threads arguments out args
    while IFS='|' read -r name threads arguments; do
        [ -z "$name" ] && continue
        [[ $name == \#* ]] && continue
        out=$RUNS/$group/$name
        if run_complete "$out" "$WORKRES/$group/$name"; then
            SKIPPED_COMPLETE=$((SKIPPED_COMPLETE + 1))
            echo "$(date +%F_%T) $group/$name complete (scratch or work): skipped"
            continue
        fi
        if run_locked "$out"; then
            SKIPPED_LOCKED=$((SKIPPED_LOCKED + 1))
            echo "$(date +%F_%T) $group/$name held by job $(cat "$out/LOCK"): skipped"
            continue
        fi
        mkdir -p "$out"
        echo "${SLURM_JOB_ID:-none}" > "$out/LOCK"
        read -r -a args <<< "$arguments"
        env OMP_NUM_THREADS="$threads" MKL_NUM_THREADS="$threads" OPENBLAS_NUM_THREADS=1 \
            "$PY" -u benchmarks/run_ph_sac.py "${args[@]}" --output "$out" --resume \
            >> "$out/stdout.log" 2>&1 &
        RUN_PID["$group/$name"]=$!
        RUN_DIR["$group/$name"]=$out
        RUN_NAMES+=("$group/$name")
    done < "$file"
}

forward_signal() {  # SIG
    local key
    for key in "${RUN_NAMES[@]}"; do
        kill -0 "${RUN_PID[$key]}" 2>/dev/null && kill "-$1" "${RUN_PID[$key]}" 2>/dev/null
    done
}

stop_requested() {  # the USR1 trap: forward once
    if [ "$STOP_FORWARDED" -eq 0 ]; then
        STOP_FORWARDED=1
        echo "$(date +%F_%T) SIGUSR1: forwarding to $(alive_runs) runs"
        forward_signal USR1
    fi
}

alive_runs() {
    local n=0 key
    for key in "${RUN_NAMES[@]}"; do
        kill -0 "${RUN_PID[$key]}" 2>/dev/null && n=$((n + 1))
    done
    echo $n
}

reap_finished() {  # record the exit code of every launched run that has ended
    # (its LOCK stays: a lock whose job has left the queue is ignored by run_locked)
    local key rc
    for key in "${RUN_NAMES[@]}"; do
        [ -n "${RUN_RC[$key]:-}" ] && continue
        kill -0 "${RUN_PID[$key]}" 2>/dev/null && continue
        wait "${RUN_PID[$key]}"
        rc=$?
        RUN_RC[$key]=$rc
        echo "$(date +%F_%T) $key rc=$rc" >> "$J/events.txt"
    done
}

wait_runs() {  # DEADLINE_EPOCH
    local deadline=$1
    while [ "$(alive_runs)" -gt 0 ] && [ "$(date +%s)" -lt "$deadline" ]; do
        sleep 20 &
        wait $!
        reap_finished
    done
    reap_finished
}

collect_status() {  # FILE
    local file=$1 key rc verdict
    reap_finished
    : > "$file"
    for key in "${RUN_NAMES[@]}"; do
        rc=${RUN_RC[$key]:-running}
        case "$rc" in
            0) verdict=finished ;;
            75) verdict=stopped_resumable ;;
            running) verdict=still_running ;;
            *) verdict=failed ;;
        esac
        echo "$key rc=$rc $verdict" >> "$file"
    done
}

to_seconds() {  # [D-]HH:MM:SS, MM:SS or SS
    local t=$1 d=0 h=0 m=0 s=0
    [[ $t == *-* ]] && { d=${t%%-*}; t=${t#*-}; }
    IFS=: read -r -a parts <<< "$t"
    case ${#parts[@]} in
        3) h=${parts[0]}; m=${parts[1]}; s=${parts[2]} ;;
        2) m=${parts[0]}; s=${parts[1]} ;;
        1) s=${parts[0]} ;;
    esac
    echo $(( 10#$d * 86400 + 10#$h * 3600 + 10#$m * 60 + 10#$s ))
}

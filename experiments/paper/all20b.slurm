#!/bin/bash
# EXAMPLE of how the paper's runs were launched on a SLURM cluster, kept as it ran except that site-specific values
# (allocation, partition, node names, paths, cluster names in comments) are replaced by <placeholders> and the RUN_ROOT variable.
# It expects files that are not part of this release (a staged code archive, an environment archive, gate and check files).
#SBATCH --job-name=all20
#SBATCH --partition=<partition>
#SBATCH -N 9
#SBATCH --ntasks=9
#SBATCH --time=48:00:00
#SBATCH --account=<allocation>
#SBATCH --signal=B:USR1@2700
#SBATCH --exclude=<node>
#SBATCH --output=<RUN_ROOT>/final/2026-11-23_r20/logs/%x_%j.out
#SBATCH --error=<RUN_ROOT>/final/2026-11-23_r20/logs/%x_%j.err
# 2026-11-23. ONE <cluster> job for every run of the rerun: the 540 runs of the <second cluster> chain <second-cluster folder> (four methods on eleven tasks, the value-step
# ablation, three poles with the second replay size), the 80 runs with dm_control's original acrobot and pendulum
# rewards, and the 110 runs of CT-SAC with the implicit value step: 730 runs, 1,210 threads, nine node sets
# sets/v1..v9.txt of at most 140 threads on 144-core nodes. Code archive named in code_tar.name (r20), configs
# md5-equal to the <second cluster> folder's. The work is done by all20_driver_b.sh (one worker all20_worker_b.sh per node).
#   sbatch all20b.slurm                                             (nine nodes)
#   sbatch -N 3 --ntasks=3 --dependency=afterany:<job> all20b.slurm (continuation; its id goes in all_driver/next_job_b)
# GATE: no run starts unless verified/ALL20_VISTA_OK exists. If it does not exist when the job starts, the job runs
# check_vista_all20b.sh on its own nodes first; a check that ends "VERDICT PASS" writes the gate. After a check that
# does not pass, the job keeps the nodes for up to GATE_WAIT_S seconds in case the gate is written by hand after
# the check output has been read, and then ends without starting anything. Nothing starts if STOP_all20 exists.
# SIGUSR1 (from Slurm 45 min before the limit, and from each worker's watchdog on the job's own clock) makes every
# run write resume.pt at its next safe point and exit 75; a later job continues those runs exactly and skips the
# finished ones. Results stay in scratch (runs/); nothing is moved or deleted.
set -uo pipefail
S=${RUN_ROOT:?set RUN_ROOT to the run root}
F=$S/final/2026-11-23_r20
GATE=$F/verified/ALL20_VISTA_OK
GATE_WAIT_S=7200
T0=$(date +%s)
el() { echo $(( $(date +%s) - T0 )); }
echo "start $(date) $(hostname) job $SLURM_JOB_ID nodes $SLURM_JOB_NODELIST"
[ -e "$F/STOP_all20" ] && { echo "$F/STOP_all20 present: nothing started"; exit 0; }
if [ ! -f "$GATE" ]; then
    echo "$(date +%F_%T) no $GATE: running check_vista_all20b.sh on this job's nodes first"
    WRITE_GATE=1 bash "$F/check_vista_all20b.sh"
    echo "$(date +%F_%T) check ended rc=$? +$(el)s: $(tail -1 "$(ls -d "$F"/acheck_[0-9]* | tail -1)/check.out" | cut -c1-600)"
    WAIT0=$(date +%s)
    while [ ! -f "$GATE" ] && [ ! -e "$F/STOP_all20" ] && [ $(( $(date +%s) - WAIT0 )) -lt "$GATE_WAIT_S" ]; do
        sleep 120
    done
fi
[ -e "$F/STOP_all20" ] && { echo "$F/STOP_all20 present: nothing started"; exit 0; }
[ -f "$GATE" ] || { echo "no $GATE after the check and $GATE_WAIT_S s of waiting: nothing started"; exit 1; }
echo "$(date +%F_%T) gate present +$(el)s: $(cut -c1-200 "$GATE")"
exec bash "$F/all20_driver_b.sh"

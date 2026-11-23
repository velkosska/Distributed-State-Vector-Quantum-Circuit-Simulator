#!/usr/bin/env bash
#SBATCH --job-name=qsim-stream
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64       # course QOS cap; 32 physical cores
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=slurm-%x-%j.out
##SBATCH --partition=CHANGE_ME
##SBATCH --account=CHANGE_ME
#
# Measured memory bandwidth of one node (the roof in the roofline plot), at
# 1, 2, 4, .. threads with the same pinning as the OpenMP runs.
# Submit from the repo root:  sbatch slurm/stream.sbatch
# Course QOS "student": at most 64 CPUs per user (32 physical cores with 2 hardware
# threads each) and no --exclusive, or the job waits forever (QOSMaxCpuPerUserLimit).
# Output: results/stream/<jobid>/{meta.txt,stream.csv}
set -euo pipefail

ARRAY_MIB=${ARRAY_MIB:-2048}      # per array; 3 arrays. Must be >> last-level cache
REPEATS=${REPEATS:-10}
MAX_T=$(python3 "${SLURM_SUBMIT_DIR:-.}/slurm/lib/physical_cores.py" | tr ',' '\n' | wc -l)   # physical cores
export QSIM_BIN=${STREAM_BIN:-build/stream_triad}
export BACKEND=stream
export OMP_PROC_BIND=${OMP_PROC_BIND:-close}
export OMP_PLACES=${OMP_PLACES:-cores}

HARNESS_SCRIPT=slurm/stream.sbatch
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}/slurm/lib/run_harness.sh"
if [ -z "${SLURM_JOB_ID:-}" ] && [ "${ALLOW_LOCAL:-0}" != 1 ]; then
  ARRAY_MIB=${LOCAL_ARRAY_MIB:-64}; MAX_T=2
  echo "not in a Slurm job: small local check only (${ARRAY_MIB} MiB, <= 2 threads)" >&2
fi
harness_init stream
rm -f "$HARNESS_OUT/runs.csv"     # no qsim runs in this job
echo "job_id,host,threads,array_mib,copy_gbs,triad_gbs" > "$HARNESS_OUT/stream.csv"

for t in $(harness_pow2_upto "$MAX_T"); do
  export OMP_NUM_THREADS=$t
  cmd=("$QSIM_BIN" "$ARRAY_MIB" "$REPEATS")
  echo "OMP_NUM_THREADS=$t ${cmd[*]}" >> "$HARNESS_OUT/meta.txt"
  line=$("${cmd[@]}" | grep '^STREAM')
  get() { sed -n "s/.* $1=\([^ ]*\).*/\1/p" <<<"$line"; }
  echo "$HARNESS_JOBID,$(hostname),$t,$ARRAY_MIB,$(get copy_GBs),$(get triad_GBs)" >> "$HARNESS_OUT/stream.csv"
  echo "threads=$t triad=$(get triad_GBs) GB/s"
done

harness_finish

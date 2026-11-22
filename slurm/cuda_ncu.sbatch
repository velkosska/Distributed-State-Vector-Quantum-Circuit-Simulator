#!/usr/bin/env bash
#SBATCH --job-name=qsim-cuda-ncu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx6000bb:1   # same card as slurm/cuda.sbatch (RTX PRO 6000 Blackwell)
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --requeue
#
# Nsight Compute profile of the qsim_cuda gate kernels: DRAM bytes moved, kernel
# time and achieved memory bandwidth (also as % of the GPU's peak), per kernel.
# Nsight Systems (nsys) is not installed on SciTech; Nsight Compute (ncu) is in
# the course CUDA container, but reading GPU performance counters is restricted
# to admins there (ERR_NVGPUCTRPERM, job 34556), so this job only works where
# counters are enabled. The host<->device transfer vs kernel split comes
# from the CUDA event timers of slurm/cuda.sbatch (analysis/cuda_summary.py).
#
#   sbatch slurm/cuda_ncu.sbatch
#   N=26 LAUNCHES=60 sbatch slurm/cuda_ncu.sbatch
#
# Output: results/cuda_ncu/<jobid>/
#   meta.txt        commit, modules, nvidia-smi, ncu version, command
#   ncu_raw.csv     ncu --csv output, one row per (kernel launch, metric)
#   ncu_kernels.csv per kernel: launches, mean time, DRAM GB moved, GB/s, % of peak
# The state must be much larger than the L2 cache so kernels really stream from
# DRAM: 28 qubits = 4 GiB. ncu replays each profiled kernel, so only the first
# LAUNCHES kernel launches of a random circuit (all gate types) are profiled.

set -euo pipefail

N=${N:-28}
LAUNCHES=${LAUNCHES:-40}
SEED=${SEED:-1}
RANDOM_DEPTH=${RANDOM_DEPTH:-10}

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$ROOT"
set +u; source env/modules.sh cuda; set -u
export CUDA_MODULE_LOADING=EAGER

JOBID=${SLURM_JOB_ID:-local-$(date +%Y%m%d-%H%M%S)}
OUT="results/cuda_ncu/$JOBID"
mkdir -p "$OUT" data/cuda_ncu

source slurm/lib/build_cuda.sh               # -> build-cuda/qsim_cuda

circ="data/cuda_ncu/random_${N}_s${SEED}.txt"
python3 src/tools/gen_circuits.py random -n "$N" --seed "$SEED" --depth "$RANDOM_DEPTH" -o "$circ" >/dev/null

METRICS=gpu__time_duration.sum,dram__bytes_read.sum,dram__bytes_write.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed
cmd=(cuda-exec ncu --csv --print-units base --log-file "$OUT/ncu_raw.csv"
     --kernel-name regex:k_apply --launch-count "$LAUNCHES" --metrics "$METRICS"
     build-cuda/qsim_cuda "$circ" --no-gate-timing --quiet)
{
  echo "job_id:      $JOBID"
  echo "date:        $(date -Iseconds)"
  echo "host:        $(hostname)"
  echo "git_commit:  $(git rev-parse --verify -q HEAD || echo unknown)"
  echo "git_dirty:   $(git status --porcelain -- . ':(exclude)results' | grep -q . && echo yes || echo no)"
  echo "cuda_arch:   sm_$CUDA_ARCH"
  echo "ncu:         $(cuda-exec ncu --version 2>&1 | tail -1)"
  echo "circuit:     $circ (random, $N qubits, depth $RANDOM_DEPTH, seed $SEED)"
  echo "launches:    first $LAUNCHES gate kernels"
  echo "--- module list"; module list 2>&1
  echo "--- nvidia-smi"; cuda-exec nvidia-smi
  echo "--- command"; echo "${cmd[*]}"
} > "$OUT/meta.txt"

echo "--- profiling $LAUNCHES kernel launches of $circ"
# ncu writes its own messages (==PROF==, ==ERROR==) into --log-file, so a
# failure is checked there; a non-zero exit must not end the job silently.
"${cmd[@]}" 2>&1 | tee "$OUT/ncu_stdout.txt" || echo "ncu exited with an error, see $OUT/ncu_raw.csv" >&2
grep -h "==ERROR==" "$OUT/ncu_raw.csv" "$OUT/ncu_stdout.txt" 2>/dev/null >&2 || true

if grep -q "ERR_NVGPUCTRPERM" "$OUT/ncu_stdout.txt" "$OUT/ncu_raw.csv" 2>/dev/null; then
  echo "ERROR: no permission to read GPU performance counters (ERR_NVGPUCTRPERM)." >&2
  echo "Only the cluster admins can enable them (driver option NVreg_RestrictProfilingToAdminUsers=0)." >&2
  echo "Use the CUDA event timings instead: python3 analysis/cuda_summary.py" >&2
  exit 1
fi

# ---------------------------------------------------------------- per-kernel table
python3 - "$OUT" <<'EOF'
import csv, re, statistics, sys
from collections import defaultdict
out = sys.argv[1]
lines = open(f"{out}/ncu_raw.csv").read().splitlines()
start = next((i for i, l in enumerate(lines) if l.startswith('"ID"')), None)
if start is None:
    sys.exit("no ncu CSV table found in ncu_raw.csv (see ncu_stdout.txt)")
launch = defaultdict(dict)   # launch id -> {metric: value}
name = {}
for r in csv.DictReader(lines[start:]):
    try:
        v = float(r["Metric Value"].replace(",", ""))
    except ValueError:
        continue
    launch[r["ID"]][r["Metric Name"]] = v
    m = re.search(r"k_apply\w*", r["Kernel Name"])
    name[r["ID"]] = m.group(0) if m else r["Kernel Name"]
per = defaultdict(list)
for i, m in launch.items():
    per[name[i]].append(m)
cols = ["kernel", "launches", "mean_time_ms", "mean_gbytes", "gbps", "pct_of_peak_dram"]
with open(f"{out}/ncu_kernels.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(cols)
    print(f"{'kernel':28}{'launches':>9}{'time_ms':>10}{'GB':>8}{'GB/s':>9}{'%peak':>7}")
    for k, ms in sorted(per.items()):
        t = statistics.mean(m["gpu__time_duration.sum"] for m in ms) * 1e-9          # ns -> s
        b = statistics.mean(m["dram__bytes_read.sum"] + m["dram__bytes_write.sum"] for m in ms)
        p = statistics.mean(m["dram__throughput.avg.pct_of_peak_sustained_elapsed"] for m in ms)
        w.writerow([k, len(ms), f"{t*1e3:.3f}", f"{b*1e-9:.3f}", f"{b/t*1e-9:.1f}", f"{p:.1f}"])
        print(f"{k:28}{len(ms):9d}{t*1e3:10.3f}{b*1e-9:8.3f}{b/t*1e-9:9.1f}{p:7.1f}")
EOF
echo "done: $OUT"

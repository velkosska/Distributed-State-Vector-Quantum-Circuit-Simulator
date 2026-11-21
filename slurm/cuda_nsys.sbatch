#!/usr/bin/env bash
#SBATCH --job-name=qsim-cuda-nsys
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx6000bb:1   # same card as slurm/cuda.sbatch (RTX PRO 6000 Blackwell)
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G                # 30 qubits = 16 GiB state on the host (+ profiler buffers)
#SBATCH --time=00:30:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --requeue
#
# Nsight Systems profile of qsim_cuda: how much time goes to host<->device
# copies vs gate kernels. One run per circuit under `nsys profile`, then
# `nsys stats` turns each report into CSV tables.
#
#   sbatch slurm/cuda_nsys.sbatch
#   N=28 KINDS="qft" sbatch slurm/cuda_nsys.sbatch
#
# Output: results/cuda_nsys/<jobid>/
#   meta.txt                       commit, modules, nvidia-smi, nsys version, commands
#   transfer_vs_kernel.csv         per circuit: H2D, kernels, D2H time (s) and share of the total
#   <kind>_<n>q_*.csv              nsys stats tables (kernel summary, memcpy time/size, CUDA API)
#   raw/<kind>_<n>q.nsys-rep       full timeline: open in the Nsight Systems GUI (ignored by git)
#
# Profiling adds overhead, so take timings from slurm/cuda.sbatch and use this
# job for the split between copies and kernels. No runs.csv on purpose: the
# analysis scripts only read the timing experiments.

set -euo pipefail

N=${N:-30}
KINDS=${KINDS:-"qft random"}
SEED=${SEED:-1}
RANDOM_DEPTH=${RANDOM_DEPTH:-10}

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$ROOT"
set +u; source env/modules.sh cuda; set -u
export CUDA_MODULE_LOADING=EAGER             # same as slurm/cuda.sbatch

JOBID=${SLURM_JOB_ID:-local-$(date +%Y%m%d-%H%M%S)}
OUT="results/cuda_nsys/$JOBID"
mkdir -p "$OUT/raw" data/cuda_nsys

# ---------------------------------------------------------------- nsys available?
if ! cuda-exec bash -c 'command -v nsys' >/dev/null 2>&1; then
  echo "ERROR: nsys (Nsight Systems) is not in the CUDA container." >&2
  echo "Look for it with: module spider nsight ; module spider Nsight-Systems ; ls /opt/nvidia" >&2
  echo "If it is a separate module, load it and rerun with NSYS=/path/to/nsys." >&2
  [ -n "${NSYS:-}" ] || exit 1
fi
NSYS=${NSYS:-nsys}

source slurm/lib/build_cuda.sh               # -> build-cuda/qsim_cuda

{
  echo "job_id:      $JOBID"
  echo "date:        $(date -Iseconds)"
  echo "host:        $(hostname)"
  echo "git_commit:  $(git rev-parse --verify -q HEAD || echo unknown)"
  echo "git_dirty:   $(git status --porcelain -- . ':(exclude)results' | grep -q . && echo yes || echo no)"
  echo "cuda_arch:   sm_$CUDA_ARCH"
  echo "nsys:        $(cuda-exec "$NSYS" --version 2>&1 | head -1)"
  echo "qubits:      $N"
  echo "kinds:       $KINDS"
  echo "seed:        $SEED"
  echo "CUDA_MODULE_LOADING=$CUDA_MODULE_LOADING"
  echo "--- module list"; module list 2>&1
  echo "--- nvidia-smi"; cuda-exec nvidia-smi
  echo "--- commands"
} > "$OUT/meta.txt"

# ---------------------------------------------------------------- profile
for kind in $KINDS; do
  circ="data/cuda_nsys/${kind}_${N}_s${SEED}.txt"
  case $kind in
    qft)    python3 src/tools/gen_circuits.py qft -n "$N" --seed "$SEED" -o "$circ" >/dev/null ;;
    grover) python3 src/tools/gen_circuits.py grover -n "$N" --seed "$SEED" --iterations 1 -o "$circ" >/dev/null ;;
    random) python3 src/tools/gen_circuits.py random -n "$N" --seed "$SEED" --depth "$RANDOM_DEPTH" -o "$circ" >/dev/null ;;
    *) echo "unknown kind $kind" >&2; exit 1 ;;
  esac
  rep="$OUT/raw/${kind}_${N}q"
  # -t cuda: CUDA API + kernels + memcpy only (low overhead). --no-gate-timing:
  # no per-gate event syncs, so kernels run back to back as in the benchmark.
  cmd=(cuda-exec "$NSYS" profile -t cuda --force-overwrite true -o "$rep"
       build-cuda/qsim_cuda "$circ" --no-gate-timing)
  echo "${cmd[*]}" >> "$OUT/meta.txt"
  echo "--- profiling $kind n=$N"
  "${cmd[@]}" | tee "$OUT/raw/${kind}_${N}q.stdout"

  stats=(cuda-exec "$NSYS" stats --force-export true --format csv
         --report cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,cuda_gpu_mem_size_sum,cuda_api_sum
         --output "$OUT/${kind}_${N}q" "$rep.nsys-rep")
  echo "${stats[*]}" >> "$OUT/meta.txt"
  "${stats[@]}" > "$OUT/raw/${kind}_${N}q.stats.log" 2>&1 || echo "nsys stats failed, see raw/${kind}_${N}q.stats.log" >&2
done

# ---------------------------------------------------------------- transfer vs kernel table
python3 - "$OUT" $KINDS <<EOF
import csv, glob, sys
out, kinds, n = sys.argv[1], sys.argv[2:], $N

def total_s(path, label_col, match=None):
    """Sum 'Total Time (ns)' over rows (optionally only rows whose label contains match)."""
    try:
        rows = list(csv.DictReader(open(path)))
    except FileNotFoundError:
        return None
    if not rows:
        return 0.0
    tcol = next(c for c in rows[0] if c.startswith("Total Time"))
    lcol = next((c for c in rows[0] if c.lower() in (label_col, "name", "operation")), None)
    s = sum(float(r[tcol]) for r in rows if match is None or (lcol and match in r[lcol]))
    return s * 1e-9

hdr = ["kind", "qubits", "h2d_s", "kernels_s", "d2h_s", "h2d_pct", "kernels_pct", "d2h_pct"]
with open(f"{out}/transfer_vs_kernel.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(hdr)
    print(f"{'kind':8}{'qubits':>7}{'h2d_s':>10}{'kernels_s':>11}{'d2h_s':>10}   share h2d/kern/d2h")
    for k in kinds:
        p = f"{out}/{k}_{n}q"
        kern = total_s(glob.glob(p + "_cuda_gpu_kern_sum*.csv")[0] if glob.glob(p + "_cuda_gpu_kern_sum*.csv") else "", "name")
        mem = (glob.glob(p + "_cuda_gpu_mem_time_sum*.csv") or [""])[0]
        h2d = total_s(mem, "operation", "Host-to-Device")
        d2h = total_s(mem, "operation", "Device-to-Host")
        if None in (kern, h2d, d2h):
            print(f"{k:8}{n:7d}   (nsys stats missing)"); continue
        tot = kern + h2d + d2h
        pct = [100 * x / tot for x in (h2d, kern, d2h)]
        w.writerow([k, n, f"{h2d:.4f}", f"{kern:.4f}", f"{d2h:.4f}", *[f"{x:.1f}" for x in pct]])
        print(f"{k:8}{n:7d}{h2d:10.3f}{kern:11.3f}{d2h:10.3f}   {pct[0]:.1f}% / {pct[1]:.1f}% / {pct[2]:.1f}%")
EOF

echo "done: $OUT  (open raw/*.nsys-rep in the Nsight Systems GUI for the timeline)"

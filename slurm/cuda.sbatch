#!/usr/bin/env bash
#SBATCH --job-name=qsim-cuda
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx6000bb:1   # RTX PRO 6000 Blackwell, 96 GB (julia/fortran): same card for every run
#SBATCH --cpus-per-task=64       # CPU side. Course QOS "student" caps a user at 64 CPUs, so no --exclusive
#SBATCH --mem=80G                # 32 qubits = 64 GiB state, also held on the host
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --requeue                # student jobs can be preempted; Slurm requeues the job ...
#SBATCH --open-mode=append       # ... and it resumes (finished runs are skipped), keeping the log
#
# CUDA experiment: 1 GPU vs the CPU cores of the same node, through the shared
# run harness (slurm/lib/run_harness.sh). QFT, Grover and random circuits,
# fixed seeds, REPEATS runs each.
#   GPU (backend=cuda):   build-cuda/qsim_cuda, built here for this card, 20-32 qubits
#   CPU (backend=openmp): build-omp/qsim, one thread per allocated CPU, pinned, 20-30 qubits
# A full CPU node is not possible on this cluster (QOSMaxCpuPerUserLimit, cpu=64).
#
#   sbatch slurm/cuda.sbatch
#   GPU_QUBITS="20 24" CPU_QUBITS="20 24" REPEATS=1 sbatch --time=00:30:00 slurm/cuda.sbatch   # quick test
#   RUN_CPU=0 sbatch slurm/cuda.sbatch                                                         # GPU only
#
# Output: results/cuda/<jobid>/ (harness layout: meta.txt, runs.csv, raw/) plus
#   speedup.csv          median CPU total_s / median GPU total_s per circuit and size
#   raw/gpu_*.gates.csv  one per-gate timing run per circuit and size (GB/s per gate),
#                        kept out of runs.csv because per-gate syncs inflate total_s
#   raw/*.out            full stdout of every run, incl. the GPU copy times h2d_s / d2h_s
# total_s is gate time only on both backends (the GPU copies are not included).
#
# Preemption: Slurm requeues the job under the same id. The previous attempt is
# moved to results/cuda/<jobid>_partN/ and its finished runs are not repeated;
# the analysis reads every runs.csv, so all parts together form the result.

set -euo pipefail

GPU_QUBITS=${GPU_QUBITS:-"20 22 24 26 28 30 31 32"}   # 32 qubits = 64 GiB, fits in 96 GB
CPU_QUBITS=${CPU_QUBITS:-"20 22 24 26 28 30"}         # the CPU side is much slower; 30 is enough to compare
KINDS=${KINDS:-"qft grover random"}
REPEATS=${REPEATS:-3}
SEED=${SEED:-1}
RANDOM_DEPTH=${RANDOM_DEPTH:-10}
GROVER_ITERS=${GROVER_ITERS:-1}   # as in baseline.sbatch: 1 iteration is enough for time per gate
RUN_CPU=${RUN_CPU:-1}

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$ROOT"
set +u; source env/modules.sh cuda; set -u   # GCC, CMake, Python + venv, course CUDA container

# Load every kernel when the GPU context starts, so the first gate of a run is
# not charged for lazy kernel loading (seen as a ~1 ms first gate in validation).
export CUDA_MODULE_LOADING=EAGER

# ---------------------------------------------------------------- build
source slurm/lib/build_cuda.sh             # -> build-cuda/qsim_cuda, sets CUDA_ARCH, QSIM_CUDA_BIN
CPU_THREADS=${SLURM_CPUS_PER_TASK:-$(nproc)}
if [ "$RUN_CPU" = 1 ]; then
  # Fresh build on every job: -march=native must match this node's CPU (a binary
  # from another node can die with "Illegal instruction"), and a stale build
  # can survive when the file server's clock is ahead of the node's.
  echo "--- building OpenMP qsim (build-omp) on $(hostname)"
  rm -rf build-omp
  cmake -S . -B build-omp -DQSIM_OPENMP=ON -DQSIM_NATIVE=ON -DQSIM_TESTS=OFF >/dev/null
  cmake --build build-omp -j "$CPU_THREADS" --target qsim >/dev/null
fi

# ---------------------------------------------------------------- resume after preemption
JOBDIR="results/cuda/${SLURM_JOB_ID:-none}"
if [ -n "${SLURM_JOB_ID:-}" ] && [ -d "$JOBDIR" ]; then
  part=1; while [ -d "${JOBDIR}_part$part" ]; do part=$((part + 1)); done
  mv "$JOBDIR" "${JOBDIR}_part$part"
  echo "--- requeued: previous attempt moved to ${JOBDIR}_part$part"
fi
shopt -s nullglob; PREV=("${JOBDIR}"_part*/runs.csv); shopt -u nullglob

# done_already BACKEND KIND QUBITS REPEAT -> true if an earlier attempt finished that run
# (runs.csv columns: 4 backend, 5 kind, 6 qubits, 11 repeat, 23 status)
done_already() {
  [ ${#PREV[@]} -gt 0 ] || return 1
  awk -F, -v b="$1" -v k="$2" -v q="$3" -v r="$4" \
    'FNR > 1 && $4 == b && $5 == k && $6 == q && $11 == r && $23 == "ok" { f = 1 } END { exit !f }' "${PREV[@]}"
}

# ---------------------------------------------------------------- harness
export QSIM_BIN=$QSIM_CUDA_BIN BACKEND=cuda GATE_TIMING=0
export HARNESS_SKIP_MODULES=1 HARNESS_SCRIPT=slurm/cuda.sbatch   # modules are already loaded above
source slurm/lib/run_harness.sh
max_q=$(tr ' ' '\n' <<<"$GPU_QUBITS $CPU_QUBITS" | sort -n | tail -1)
harness_guard_local "$max_q"
harness_init cuda
OUT=$HARNESS_OUT
{
  echo
  echo "--- cuda experiment"
  echo "cuda_arch:       sm_$CUDA_ARCH"
  echo "gpu_binary:      $QSIM_CUDA_BIN (backend=cuda)"
  echo "cpu_binary:      build-omp/qsim (backend=openmp, $CPU_THREADS threads)"
  echo "gpu_qubits:      $GPU_QUBITS"
  echo "cpu_qubits:      ${CPU_QUBITS} (run_cpu=$RUN_CPU)"
  echo "resumed_from:    ${PREV[*]:-none}"
} >> "$OUT/meta.txt"
cuda-exec nvidia-smi > "$OUT/raw/nvidia-smi.txt" 2>&1 || true

# circuit KIND N -> sets $circ and $extra (harness keys for the circuit parameters)
circuit() {
  case $1 in
    qft)    circ=$(harness_gen_circuit qft "$2" --seed "$SEED"); extra=() ;;
    grover) circ=$(harness_gen_circuit grover "$2" --seed "$SEED" --iterations "$GROVER_ITERS"); extra=(iterations="$GROVER_ITERS") ;;
    random) circ=$(harness_gen_circuit random "$2" --seed "$SEED" --depth "$RANDOM_DEPTH"); extra=(depth="$RANDOM_DEPTH") ;;
    *) echo "unknown kind $1" >&2; exit 1 ;;
  esac
}

# ---------------------------------------------------------------- GPU
export OMP_NUM_THREADS=1
for kind in $KINDS; do
  for n in $GPU_QUBITS; do
    circuit "$kind" "$n"
    gates_csv="$OUT/raw/gpu_${kind}_${n}q.gates.csv"
    if ! ls "${JOBDIR}"_part*/raw/"gpu_${kind}_${n}q.gates.csv" >/dev/null 2>&1; then
      echo "gates  $QSIM_CUDA_BIN $circ --quiet --timing $gates_csv" >> "$OUT/meta.txt"
      "$QSIM_CUDA_BIN" "$circ" --quiet --timing "$gates_csv" > /dev/null || echo "per-gate run failed: $kind $n" >&2
    fi
    for r in $(seq 1 "$REPEATS"); do
      done_already cuda "$kind" "$n" "$r" && { echo "skip (done earlier): cuda $kind n=$n rep=$r"; continue; }
      harness_run kind="$kind" qubits="$n" repeat="$r" seed="$SEED" gpus=1 threads=1 circuit="$circ" ${extra[@]+"${extra[@]}"}
    done
  done
done

# ---------------------------------------------------------------- CPU (allocated cores of the same node)
if [ "$RUN_CPU" = 1 ]; then
  export QSIM_BIN=build-omp/qsim BACKEND=openmp
  export OMP_NUM_THREADS=$CPU_THREADS OMP_PROC_BIND=spread OMP_PLACES=cores
  echo "cpu runs: OMP_NUM_THREADS=$OMP_NUM_THREADS OMP_PROC_BIND=$OMP_PROC_BIND OMP_PLACES=$OMP_PLACES" >> "$OUT/meta.txt"
  for kind in $KINDS; do
    for n in $CPU_QUBITS; do
      circuit "$kind" "$n"
      for r in $(seq 1 "$REPEATS"); do
        done_already openmp "$kind" "$n" "$r" && { echo "skip (done earlier): openmp $kind n=$n rep=$r"; continue; }
        harness_run kind="$kind" qubits="$n" repeat="$r" seed="$SEED" gpus=0 threads="$CPU_THREADS" circuit="$circ" ${extra[@]+"${extra[@]}"}
      done
    done
  done
fi

# ---------------------------------------------------------------- speedup table (this job, all parts)
python3 - "$OUT" ${PREV[@]+"${PREV[@]}"} <<'EOF'
import csv, statistics, sys
from collections import defaultdict
out, prev = sys.argv[1], sys.argv[2:]
t = defaultdict(list)
for path in [f"{out}/runs.csv", *prev]:
    for row in csv.DictReader(open(path)):
        if row["status"] == "ok" and row["total_s"]:
            t[(row["kind"], int(row["qubits"]), row["backend"])].append(float(row["total_s"]))
keys = sorted({(k, n) for k, n, _ in t})
with open(f"{out}/speedup.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["kind", "qubits", "gpu_median_s", "cpu_median_s", "speedup"])
    print(f"{'kind':8}{'qubits':>7}{'gpu_s':>12}{'cpu_s':>12}{'speedup':>9}")
    for k, n in keys:
        g = statistics.median(t[(k, n, "cuda")]) if t[(k, n, "cuda")] else None
        c = statistics.median(t[(k, n, "openmp")]) if t[(k, n, "openmp")] else None
        s = c / g if g and c else None
        w.writerow([k, n, g or "", c or "", f"{s:.2f}" if s else ""])
        gs = f"{g:12.4g}" if g is not None else "-".rjust(12)
        cs = f"{c:12.4g}" if c is not None else "-".rjust(12)
        ss = f"{s:9.1f}" if s else "-".rjust(9)
        print(f"{k:8}{n:7d}{gs}{cs}{ss}")
EOF

harness_finish

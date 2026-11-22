#!/usr/bin/env bash
#SBATCH --job-name=qsim-cuda-validate
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=slurm-%x-%j.out
#
# Build qsim_cuda on the GPU node and validate it against Qiskit (tol 1e-10).
#
#   sbatch slurm/cuda_validate.sbatch
#   MAX_QUBITS=12 sbatch slurm/cuda_validate.sbatch
#   sbatch --nodelist=haskell slurm/cuda_validate.sbatch   # pin a node (haskell = RTX 6000 Ada)
#
# Runs on any free GPU node; the CUDA arch is detected from the GPU it gets
# (override with CUDA_ARCH=89 etc.). For benchmarks, pin one node so all
# numbers come from the same card.
#
# One-time setup on the login node first: bash env/setup_venv.sh (numpy + qiskit).
# CUDA on this cluster is an Apptainer container: every CUDA command runs via cuda-exec.

set -euo pipefail

MAX_QUBITS=${MAX_QUBITS:-20}

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$ROOT"
set +u; source env/modules.sh cuda; set -u   # GCC, CMake, Python + venv, course CUDA container

echo "--- node $(hostname), commit $(git rev-parse --short HEAD)"
module list 2>&1
cuda-exec nvidia-smi

source slurm/lib/build_cuda.sh             # sets CUDA_ARCH, CUDA_BUILD, QSIM_CUDA_BIN

# ---------------------------------------------------------------- smoke test
python3 src/tools/gen_circuits.py qft -n 10 --seed 1 -o "$CUDA_BUILD/smoke_qft10.txt"
"$QSIM_CUDA_BIN" "$CUDA_BUILD/smoke_qft10.txt"

# ---------------------------------------------------------------- validate
echo "--- validating $QSIM_CUDA_BIN up to $MAX_QUBITS qubits"
python3 src/tools/validate_qiskit.py --qsim "$QSIM_CUDA_BIN" --max-qubits "$MAX_QUBITS" \
  --csv "$CUDA_BUILD/validation.csv"
echo "done: validation passed, see $CUDA_BUILD/validation.csv"

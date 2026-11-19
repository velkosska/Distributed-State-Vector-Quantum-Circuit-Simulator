#!/usr/bin/env bash
#SBATCH --job-name=qsim-cuda-validate
#SBATCH --partition=gpu
#SBATCH --nodelist=haskell       # RTX 6000 Ada (sm_89); keep all CUDA runs on the same card
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
#
# One-time setup on the login node first: bash env/setup_venv.sh (numpy + qiskit).
# CUDA on this cluster is an Apptainer container: every CUDA command runs via cuda-exec.

set -euo pipefail

MAX_QUBITS=${MAX_QUBITS:-20}
CUDA_ARCH=${CUDA_ARCH:-89}
BUILD=build-cuda

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$ROOT"
source env/modules.sh                     # GCC, CMake, Python + venv (does module purge)
module load hpc-course-cuda/12.8.1

echo "--- node $(hostname), commit $(git rev-parse --short HEAD)"
module list 2>&1
cuda-exec nvidia-smi

# ---------------------------------------------------------------- build
if cuda-exec bash -c 'command -v cmake' >/dev/null 2>&1; then
  echo "--- building with CMake inside the container"
  cuda-exec cmake -S . -B "$BUILD" -DQSIM_CUDA=ON -DQSIM_NATIVE=OFF -DQSIM_TESTS=OFF \
    -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH"
  cuda-exec cmake --build "$BUILD" -j 4 --target qsim_cuda
else
  echo "--- no cmake in the container, building with nvcc directly"
  mkdir -p "$BUILD"
  cuda-exec nvcc -O3 -DNDEBUG -std=c++17 -arch=sm_"$CUDA_ARCH" -Isrc \
    src/apps/qsim_cuda_main.cpp src/qsim/cuda/cuda_sim.cu \
    src/qsim/circuit.cpp src/qsim/gates.cpp src/qsim/state.cpp src/qsim/simulator.cpp src/qsim/fusion.cpp \
    -o "$BUILD/qsim_cuda"
fi

# ---------------------------------------------------------------- smoke test
python3 src/tools/gen_circuits.py qft -n 10 --seed 1 -o "$BUILD/smoke_qft10.txt"
cuda-exec "$BUILD/qsim_cuda" "$BUILD/smoke_qft10.txt"

# ---------------------------------------------------------------- validate
# The binary was built inside the container, so it may need the container's
# libraries to start. validate_qiskit.py runs it as a subprocess, so give it a
# small wrapper that goes through cuda-exec.
WRAP="$ROOT/$BUILD/qsim_cuda_wrapped.sh"
cat > "$WRAP" <<EOF
#!/usr/bin/env bash
exec cuda-exec "$ROOT/$BUILD/qsim_cuda" "\$@"
EOF
chmod +x "$WRAP"
if "$BUILD/qsim_cuda" --help >/dev/null 2>&1; then QSIM="$BUILD/qsim_cuda"; else QSIM="$WRAP"; fi
echo "--- validating $QSIM up to $MAX_QUBITS qubits"

python3 src/tools/validate_qiskit.py --qsim "$QSIM" --max-qubits "$MAX_QUBITS" --csv "$BUILD/validation.csv"
echo "done: validation passed, see $BUILD/validation.csv"

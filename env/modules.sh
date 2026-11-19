# Source this on the cluster (login node for building, and inside every job):
#   source env/modules.sh
# Module versions below are the ones we plan to use; confirm with `module avail`.
module purge
module load GCC/12.3.0
module load CMake/3.26.3-GCCcore-12.3.0
module load Python/3.11.3-GCCcore-12.3.0

# Python venv with numpy + qiskit (created once by env/setup_venv.sh)
QSIM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ -f "$QSIM_ROOT/env/.venv/bin/activate" ]; then
  source "$QSIM_ROOT/env/.venv/bin/activate"
fi

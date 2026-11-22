# shellcheck shell=bash
# Software environment for building and running qsim. Source it on the login
# node before building, and inside every job (the run harness does this):
#
#   source env/modules.sh            # serial / OpenMP: GCC, CMake, Python
#   source env/modules.sh mpi        # + OpenMPI (same GCC 12.3.0 toolchain, foss/2023a)
#   source env/modules.sh cuda       # + CUDA
#   source env/modules.sh mpi cuda   # both
#
# All versions live here and nowhere else. They are EasyBuild 2023a names;
# confirm them with `module avail` on a new system and record the change in
# REPRODUCE.md. The harness writes `module list` into every meta.txt, so a
# mismatch is visible in the results.
#
# Works under `set -u` and outside a module system (then it only activates the venv).

QSIM_MOD_GCC=${QSIM_MOD_GCC:-GCC/12.3.0}
QSIM_MOD_CMAKE=${QSIM_MOD_CMAKE:-CMake/3.26.3-GCCcore-12.3.0}
QSIM_MOD_PYTHON=${QSIM_MOD_PYTHON:-Python/3.11.3-GCCcore-12.3.0}
QSIM_MOD_MPI=${QSIM_MOD_MPI:-OpenMPI/4.1.5-GCC-12.3.0}
# SciTech: CUDA 12.8.1 comes as an Apptainer container; nvcc etc. run via `cuda-exec`
# (see slurm/lib/build_cuda.sh). Only usable on GPU nodes.
QSIM_MOD_CUDA=${QSIM_MOD_CUDA:-hpc-course-cuda/12.8.1}

_qsim_env_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if command -v module >/dev/null 2>&1; then
  module purge
  module load "$QSIM_MOD_GCC" "$QSIM_MOD_CMAKE" "$QSIM_MOD_PYTHON"
  for _qsim_extra in "$@" ${QSIM_MODULES_EXTRA:-}; do
    case $_qsim_extra in
      mpi)  module load "$QSIM_MOD_MPI" ;;
      cuda) module load "$QSIM_MOD_CUDA" ;;
      "")   ;;
      *)    module load "$_qsim_extra" ;;   # any other module name
    esac
  done
  unset _qsim_extra
else
  echo "env/modules.sh: no module command; using the system compilers and Python" >&2
fi

# Python venv with the pinned packages (created once by env/setup_venv.sh)
if [ -f "$_qsim_env_root/env/.venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "$_qsim_env_root/env/.venv/bin/activate"
fi
export QSIM_ROOT="$_qsim_env_root"
unset _qsim_env_root

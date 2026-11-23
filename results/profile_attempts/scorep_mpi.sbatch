#!/usr/bin/env bash
#SBATCH --job-name=qsim-scorep
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/home/hpcb37/qsim_profile/slurm-%x-%j.out
# Score-P MPI profile of the 30-qubit random circuit on 16 ranks over 2 nodes (commit fabbd99).
set -eo pipefail; source /etc/profile; [ -f /usr/share/lmod/lmod/init/bash ] && source /usr/share/lmod/lmod/init/bash; set -u
cd ~/qsim
module purge
module load Score-P/8.4-gompi-2023b CMake/3.27.6-GCCcore-13.2.0
module list 2>&1
git rev-parse HEAD
OUT=~/qsim_profile/$SLURM_JOB_ID; mkdir -p $OUT
cmake -S . -B build-scorep -DQSIM_MPI=ON -DQSIM_OPENMP=ON -DQSIM_NATIVE=ON \
  -DCMAKE_CXX_COMPILER="$(command -v scorep-mpicxx || echo mpicxx)" -DCMAKE_BUILD_TYPE=Release > $OUT/cmake.log 2>&1 || true
if ! command -v scorep-mpicxx >/dev/null; then
  rm -rf build-scorep
  SCOREP_WRAPPER=off cmake -S . -B build-scorep -DQSIM_MPI=ON -DQSIM_OPENMP=ON -DCMAKE_CXX_COMPILER=mpicxx \
    -DCMAKE_CXX_COMPILER_LAUNCHER="scorep;--nocompiler;--mpp=mpi" > $OUT/cmake.log 2>&1
fi
SCOREP_WRAPPER_INSTRUMENTER_FLAGS="--nocompiler --mpp=mpi" cmake --build build-scorep -j 16 --target qsim_mpi >> $OUT/cmake.log 2>&1
export OMP_NUM_THREADS=2 OMP_PROC_BIND=close OMP_PLACES=cores
export SCOREP_ENABLE_PROFILING=true SCOREP_ENABLE_TRACING=false SCOREP_EXPERIMENT_DIRECTORY=$OUT/scorep_random30_p16
for c in random_30_seed_1_depth_10 qft_30_seed_1; do
  export SCOREP_EXPERIMENT_DIRECTORY=$OUT/scorep_${c}_p16
  /usr/bin/time -v srun --cpu-bind=cores build-scorep/qsim_mpi data/mpi_strong/$c.txt > $OUT/$c.out 2> $OUT/$c.time
  scorep-score -r $SCOREP_EXPERIMENT_DIRECTORY/profile.cubex > $OUT/${c}_score.txt
done
srun -N1 -n1 bash -c "papi_avail 2>&1 | head -20" > $OUT/papi_avail.txt 2>&1 || true
echo done

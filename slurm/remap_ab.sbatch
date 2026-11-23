#!/usr/bin/env bash
#SBATCH --job-name=qsim-remap-ab
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/home/hpcb37/qsim_profile/slurm-%x-%j.out
# A/B test of qubit remapping under identical network conditions: same allocation,
# remap off and on interleaved, 3 pairs per circuit, 28 qubits on 16 ranks over 2 nodes (commit fabbd99).
set -eo pipefail; source /etc/profile; set -u
cd ~/qsim && source env/modules.sh mpi
OUT=~/qsim_profile/remap_ab_$SLURM_JOB_ID; mkdir -p $OUT
git rev-parse HEAD > $OUT/git_commit.txt; scontrol show job $SLURM_JOB_ID > $OUT/scontrol.txt; module list > $OUT/modules.txt 2>&1
export OMP_NUM_THREADS=4 OMP_PROC_BIND=close OMP_PLACES=cores
for c in qft random; do
  f=data/remap_ab/${c}_28.txt; mkdir -p data/remap_ab
  [ $c = qft ] && python3 src/tools/gen_circuits.py qft -n 28 --seed 1 -o $f || python3 src/tools/gen_circuits.py random -n 28 --seed 1 --depth 10 -o $f
  for rep in 1 2 3; do
    for mode in --no-remap --remap; do
      line=$(srun --cpu-bind=cores build-mpi/qsim_mpi $f $mode | grep SUMMARY)
      echo "kind=$c rep=$rep mode=$mode $line" | tee -a $OUT/runs.txt
    done
  done
done
echo done

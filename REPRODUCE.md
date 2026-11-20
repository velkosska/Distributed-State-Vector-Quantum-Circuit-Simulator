# Reproducing every result

From a fresh clone to every table and figure in the report. Commands are run
from the repository root. **Everything heavier than building runs through
`sbatch`, never on the login node.** The harness refuses runs above 18 qubits
outside a Slurm job (override with `ALLOW_LOCAL=1` only on a compute node).

Lines marked *(site)* contain cluster-specific values (partition, account,
cores per node). Set them once in the `#SBATCH` headers or on the command line
(`sbatch --partition=... --account=...`).

## 0. What you need

| | Version used | Where it is set |
|---|---|---|
| GCC | 12.3.0 | `env/modules.sh` |
| CMake | 3.26.3 | `env/modules.sh` |
| OpenMPI | 4.1.5 (MPI build only) | `env/modules.sh` (`mpi`) |
| CUDA | 12.1.1 (CUDA build only) | `env/modules.sh` (`cuda`) |
| Python | 3.11.3 | `env/modules.sh` |
| Python packages | exact pins | `env/requirements.txt`, full set in `env/requirements.lock` |
| Slurm | any recent | — |

If a module name differs on your system, override it without editing the file,
e.g. `QSIM_MOD_GCC=GCC/12.2.0 source env/modules.sh`, and note it in your
results. Every `meta.txt` contains the `module list` of the job.

## 1. Clone and set up (login node)

```bash
git clone https://github.com/velkosska/Distributed-State-Vector-Quantum-Circuit-Simulator.git qsim
cd qsim
git checkout <commit or tag of the report>      # the commit printed in results/*/*/meta.txt

source env/modules.sh
env/setup_venv.sh                               # once: env/.venv from requirements.lock
source env/modules.sh                           # again, to activate the venv
```

## 2. Build (login node is fine, it is light)

Build on the CPU type you run on (`-march=native`); on a cluster whose login
nodes differ from the compute nodes, add `-DQSIM_NATIVE=OFF` or build inside a
short job (`srun -p <partition> -c 16 -t 10 cmake --build ...`).

```bash
cmake -S . -B build                              # serial + stream_triad
cmake --build build -j
cmake -S . -B build-omp -DQSIM_OPENMP=ON         # OpenMP
cmake --build build-omp -j
ctest --test-dir build                           # kernel unit tests
```

MPI (`build-mpi/qsim_mpi`) and CUDA (`build-cuda/qsim_cuda`) builds: follow the
build notes of those backends (Edard / Emma), with `source env/modules.sh mpi`
or `source env/modules.sh cuda` first. The job scripts expect exactly these
binary paths; override with `QSIM_BIN=...`.

## 3. Check correctness

```bash
sbatch slurm/validate.sbatch                     # Qiskit comparison, 2..20 qubits, tol 1e-10
```

Every backend must pass before its timings are used (`QSIM_BIN=build-omp/qsim sbatch slurm/validate.sbatch`, ...).

## 4. Run the experiments

Submit from the repo root (the harness uses `$SLURM_SUBMIT_DIR`). Each job
writes `results/<experiment>/<jobid>/{meta.txt,runs.csv,raw/}`. All runs use
3 repeats and seed 1 unless stated. Circuits are regenerated at job start from
`src/tools/gen_circuits.py`; the generator command and a hash of each circuit
are in `meta.txt`.

| # | Experiment | Command | Figures / tables |
|---|---|---|---|
| 4.1 | Memory bandwidth (roof) | `sbatch slurm/stream.sbatch` | roofline, bandwidth |
| 4.2 | Serial baseline, 20–28 q | `sbatch slurm/baseline.sbatch` | time per gate vs qubits, bandwidth vs qubits |
| 4.3 | OpenMP strong, 28 q | `MODE=strong N=28 sbatch slurm/openmp_scaling.sbatch` | strong scaling, Amdahl, roofline |
| 4.4 | OpenMP weak, 24 q at 1 thread | `MODE=weak N=24 sbatch slurm/openmp_scaling.sbatch` | weak scaling |
| 4.5 | MPI strong, 30 q | `MODE=strong N=30 sbatch --nodes=4 slurm/mpi_scaling.sbatch` | strong scaling, Amdahl |
| 4.6 | MPI weak, 28 q at 1 rank | `MODE=weak N=28 sbatch --nodes=8 slurm/mpi_scaling.sbatch` | weak scaling, scale-up plan |
| 4.7 | Communication sensitivity | `sbatch slurm/sensitivity.sbatch` (N=30, P=16) | sensitivity |
| 4.8 | CUDA, 20–30 q | `sbatch slurm/cuda.sbatch` | time per gate vs qubits |

Set `--cpus-per-task` in `openmp_scaling.sbatch` and `stream.sbatch` to the
core count of one node *(site)*. For the sensitivity job, turn the MPI qubit
remapping off (pass its flag in `QSIM_ARGS`) so the static communication count
matches what runs; see the header of `slurm/sensitivity.sbatch`.

Quick dry run of any script on tiny sizes (seconds; fine on the login node):

```bash
QUBITS="10 12" REPEATS=1 bash slurm/baseline.sbatch
THREADS="1 2" N=12 bash slurm/openmp_scaling.sbatch
```

Check a finished job: `tail -3 results/<experiment>/<jobid>/meta.txt` shows
`runs:` and `failed_runs:`; the job exits non-zero if any run failed.

## 5. Make the tables and figures

```bash
source env/modules.sh
python3 -m pytest analysis -q                    # analysis self-test (synthetic data)
python3 analysis/make_figures.py --clean-only --peak-gflops <per-core FP64 peak>
python3 analysis/scaleup.py --system mn5-gpp --mem-bw <STREAM triad of one node>
```

`--peak-gflops` is cores' FP64 peak: GHz × FLOP/cycle (e.g. AVX-512 with
2 FMA units: 2.0 GHz × 32 = 64; AVX2 with 2 FMA units: 16 FLOP/cycle).
Without it the roofline shows only the memory roof.

| Output | Content |
|---|---|
| `results/summary/runs_summary.csv` | every configuration: median, min, max, range, n, time per gate |
| `results/summary/scaling_strong.csv`, `scaling_weak.csv` | speedup, efficiency (with repeat range), Karp–Flatt |
| `results/summary/amdahl.csv` | serial fraction f per strong-scaling curve, with range |
| `results/summary/sensitivity.csv` | time vs % communicating gates; fitted cost of a comm gate |
| `results/summary/gate_bandwidth.csv` | GB/s and GFLOP/s per gate type |
| `results/summary/scaleup_*.csv` | memory per qubit count, node-hours |
| `results/figures/time_per_gate_vs_qubits` | figure 1 |
| `results/figures/strong_<experiment>` | strong scaling + efficiency + Amdahl fit |
| `results/figures/weak_<experiment>` | weak scaling (time per gate) + efficiency |
| `results/figures/bandwidth_vs_qubits_serial`, `bandwidth_vs_threads_roof`, `roofline` | bandwidth vs roofline |
| `results/figures/sensitivity_comm_share` | MPI communication sensitivity |

Filters: `--experiments mpi_strong mpi_weak`, `--commit <hash>`,
`--clean-only` (drop runs from a dirty tree). The script warns when one
experiment mixes commits, when a configuration has fewer than 3 repeats, or
when the repeat range exceeds 10 % of the median; rerun those before using the
figure.

## 6. Writing a new job script (teammates)

Copy the pattern of `slurm/baseline.sbatch`:

```bash
set -euo pipefail
HARNESS_SCRIPT=slurm/my_job.sbatch
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}/slurm/lib/run_harness.sh"
export QSIM_BIN=build-xyz/qsim_xyz BACKEND=openmp      # serial|openmp|mpi|cuda
harness_guard_local 28                                  # largest qubit count of the job
harness_init my_experiment                              # results/my_experiment/<jobid>/
circ=$(harness_gen_circuit qft 28 --seed 1)
harness_run kind=qft qubits=28 repeat=1 seed=1 threads=8 circuit="$circ" \
            [ranks=4 nodes=1 launch="srun -n 4 --exact"] [tag=...] [-- extra qsim args]
harness_finish
```

Rules: generate circuits with `harness_gen_circuit` (records the command and
hash), set every `OMP_*` variable *before* `harness_init` so it is recorded,
keep the `SUMMARY` line format in new binaries (extra `key=value` pairs such as
`ranks=` or `h2d_s=` are welcome), and report per-rank bytes in the MPI timing CSV.

## Updating pins

```bash
# edit env/requirements.txt, then regenerate the lock for the cluster's Python
python3.11 -m venv /tmp/lock && /tmp/lock/bin/pip install -r env/requirements.txt
{ echo "# Generated from env/requirements.txt for CPython 3.11, Linux x86_64. Do not edit by hand."; \
  /tmp/lock/bin/pip freeze | tr 'A-Z' 'a-z' | sort; } > env/requirements.lock
```

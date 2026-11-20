# Distributed State-Vector Quantum Circuit Simulator

A state-vector simulator for quantum circuits in C++17, built as an HPC case
study: a serial reference, then OpenMP, MPI (distributed state with a
global/local qubit split) and CUDA versions, with scaling and bottleneck
analysis on a Slurm cluster.

The state of `n` qubits is a vector of `2^n` complex doubles (16 B each:
28 qubits = 4 GiB, 32 qubits = 64 GiB). Every gate is a full pass over that
vector, so the simulator is memory-bandwidth bound, which is the central
theme of the project.

## Repository layout

```
src/qsim/        simulator library: state vector, gates, kernels, circuit reader, fusion hook
src/apps/        qsim command-line driver
src/tools/       Python: circuit generators, Qiskit validation
tests/           kernel unit tests (ctest)
env/             module loads and pinned Python requirements for the cluster
slurm/           batch scripts; slurm/lib/run_harness.sh is sourced by all of them
analysis/        pandas/matplotlib: summary tables, scaling fits, figures, scale-up estimate
data/            generated circuits (not committed; examples/ only)
results/         job output: metadata, CSVs, plots
docs/            circuit format, kernel API for the parallel backends
```

## Build

Requirements: a C++17 compiler (GCC ≥ 9), CMake ≥ 3.20, and for the Python
tools Python ≥ 3.9 with `numpy` and `qiskit`.

```bash
# on the cluster
source env/modules.sh
env/setup_venv.sh          # once: creates env/.venv with numpy + qiskit
source env/modules.sh      # again, to activate the venv

cmake -S . -B build                    # Release, -O3 -march=native
cmake --build build -j
ctest --test-dir build                 # kernel unit tests
```

CMake options:

| Option | Default | Meaning |
|---|---|---|
| `QSIM_NATIVE` | `ON` | add `-march=native` (build on the same CPU type you run on) |
| `QSIM_OPENMP` | `OFF` | parallelise kernel loops with OpenMP |
| `QSIM_TESTS` | `ON` | build `test_kernels` |

Example: `cmake -S . -B build-omp -DQSIM_OPENMP=ON`.

## Run

```bash
python3 src/tools/gen_circuits.py qft    -n 20 --seed 1 -o data/qft_20.txt
python3 src/tools/gen_circuits.py grover -n 20 --seed 1 --iterations 2 -o data/grover_20.txt
python3 src/tools/gen_circuits.py random -n 20 --seed 1 --depth 20 -o data/random_20.txt

./build/qsim data/qft_20.txt --timing qft_20_gates.csv
```

`qsim` prints a per-gate-type table (count, time, mean time per gate,
effective GB/s) and a `SUMMARY key=value` line for scripts. Options:
`--timing FILE` (per-gate CSV), `--dump FILE` (final state as raw
`complex<double>`), `--fusion none|1q`, `--no-gate-timing`, `--quiet`.

Circuit format: [`docs/circuit_format.md`](docs/circuit_format.md).
Gates: H X Y Z S Sdg T Tdg P RX RY RZ U (any 2×2), CX CY CZ CP SWAP.

## Validation

```bash
python3 src/tools/validate_qiskit.py --qsim build/qsim --max-qubits 20
```

Runs QFT (with a seeded random input layer), Grover and random circuits from
2 to 20 qubits and compares the final state with Qiskit's `Statevector`
(max absolute amplitude difference ≤ 1e-10). On the cluster use
`sbatch slurm/validate.sbatch` instead of the login node.

## Baseline (1 core)

```bash
sbatch slurm/baseline.sbatch                       # 20..28 qubits, 3 repeats
QUBITS="20 24" REPEATS=1 sbatch slurm/baseline.sbatch
```

Each job writes `results/baseline/<jobid>/meta.txt` (git commit, `module
list`, node, CPU, seeds, exact commands), `runs.csv` (one row per run) and
per-gate timing CSVs in `raw/`.

## Experiments and figures

All job scripts (`baseline`, `openmp_scaling`, `mpi_scaling`, `sensitivity`,
`cuda`, `stream`) source the same harness, so every run of every backend lands
in the same `runs.csv` format ([`results/README.md`](results/README.md)).
`python3 analysis/make_figures.py` turns all of them into tables and figures.
[`REPRODUCE.md`](REPRODUCE.md) goes from a fresh clone to every figure.

## For the parallel backends

Read [`docs/kernel_api.md`](docs/kernel_api.md): it describes the pair-index
kernel loop, where the OpenMP pragma goes, how MPI ranks map to global qubits
and which kernel to call after an exchange, and how the CUDA port mirrors the
same kernels.

## Team

| Area | Owner |
|---|---|
| Serial baseline, repo, gate fusion | Marija |
| OpenMP, CPU profiling | Hurshneet |
| MPI, qubit remapping | Edard |
| CUDA | Emma |
| Slurm harness, experiments, reproducibility | Ana-Maria |

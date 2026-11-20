# results/

Every Slurm job writes to `results/<experiment>/<jobid>/` through the shared
harness [`slurm/lib/run_harness.sh`](../slurm/lib/run_harness.sh):

| File | Content |
|---|---|
| `meta.txt` | provenance: git commit, branch, dirty flag (+ `raw/git_diff.patch` when dirty), binary sha256 and CMake options, `module list`, node list, CPU model / sockets / caches / SIMD flags, GPU, every `OMP_*`, `SLURM_*`, MPI/UCX and `QSIM_*` variable, the generator command and hash of every circuit, and the exact command of every run (`r0001 ...`) |
| `runs.csv` | one row per qsim run, parsed from the `SUMMARY` line (schema below) |
| `raw/` | per-gate timing CSVs (`<run>.gates.csv`: `index,gate,q0,q1,theta,seconds,bytes,gbps`), stdout/stderr of every run, generator log. Ignored by git |
| `circuit_analysis.csv` | sensitivity job only: `analyze_circuit.py` table per circuit (local / global_free / comm gates) |
| `stream.csv` | standalone STREAM job bandwidth per thread count |
| `speedup.csv`, `efficiency.csv` | OpenMP strong-scaling medians and repeat bounds by circuit, binding and thread count |
| `profile_runs.csv`, `gate_bandwidth.csv`, `stream.csv`, `roofline.pdf`, `roofline.png` | `omp_profile` job outputs for matched LIKWID and STREAM triad measurements |

Experiments: `baseline`, `openmp_strong`, `openmp_weak`, `omp_profile`,
`mpi_strong`, `mpi_weak`, `sensitivity`, `cuda`, `stream`. Jobs started outside Slurm get the
id `local-<date>-<pid>`; the analysis skips them unless `--include-local`.

## runs.csv columns

| Column | Meaning |
|---|---|
| `run_id` | `r0001`... within the job; matches the command line in `meta.txt` and the files in `raw/` |
| `experiment`, `job_id`, `backend` | `backend` = serial, openmp, mpi, cuda |
| `kind` | qft, grover, random, random_gf (random with `--global-fraction`) |
| `qubits`, `nodes`, `ranks`, `threads`, `gpus` | resources of this run (`threads` = OpenMP threads per rank) |
| `repeat`, `seed`, `depth`, `iterations`, `global_fraction`, `global_qubits` | circuit parameters |
| `gates_in`, `gates_run`, `fusion` | gates in the file / after fusion |
| `total_s`, `mean_gate_s`, `norm_err` | from the `SUMMARY` line |
| `status` | `ok`, `fail:<exit code>` or `nosummary`; failed runs stay in the file and are filtered by the analysis |
| `tag` | free label (strong / weak / kind / global_fraction) |
| `circuit`, `timing_csv` | circuit path (relative to repo root), per-gate CSV (relative to the job directory) |

## Derived output (from `python3 analysis/make_figures.py`)

`results/summary/*.csv` (median / min / max / range over repeats, speedup,
efficiency, Amdahl fit, sensitivity, bandwidth) and `results/figures/*.pdf|png`.
Commit the summary CSVs and figures that go into the report, never `raw/`.

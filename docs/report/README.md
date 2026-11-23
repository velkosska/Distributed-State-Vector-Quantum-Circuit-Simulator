# Final report: sections, owners and what is left

`GroupXX_Part2_Final.pdf` = 5–7 page technical report + 1–2 page EuroHPC
scale-up plan. Marija merges the sections. Every number must come from a job in
`results/` whose `meta.txt` shows `git_dirty: no`.

| # | Section | File | Owner | Draft | Missing |
|---|---|---|---|---|---|
| 1 | Problem | `problem_architecture.md` | Marija | ✅ | — |
| 2 | Architecture | `problem_architecture.md` | Marija | ✅ | architecture figure (optional) |
| 3 | Methodology | `methodology.md` | Ana-Maria | ✅ | final ⟨⟩ values; **cut to ≈ 0.5 page** |
| 4 | OpenMP | `openmp_cpu.md` | Hurshneet | ✅ | strong/weak scaling numbers + figure |
| 5 | Bottleneck analysis (CPU) | `openmp_cpu.md` | Hurshneet | ✅ | LIKWID counters, roofline figure |
| 6 | MPI + communication bottleneck | `mpi_communication.md` | Edard | ✅ starting draft | check text, all ⟨⟩ numbers, scaling + sensitivity figures, hybrid table |
| 7 | Technology exploration (CUDA) + table | `technology_exploration.md` | Emma | ✅ with results | MPI and STREAM cells of the table |
| 8 | Optimisation (fusion + remapping) | `optimization.md` | Marija + Edard | ✅ | fusion timings, remapping timings |
| 9 | Conclusions | `conclusions.md` | Marija | ✅ | fill ⟨⟩ once 4–8 are final |
| — | EuroHPC scale-up plan | `scaleup_plan.md` | Ana-Maria | ✅ | rerun `analysis/scaleup.py` on Edard's weak-scaling data |

**Length.** The drafts together are ≈ 5 500 words without the scale-up plan;
5–7 pages with figures is about 3 000–3 500. Methodology (≈ 1 500 words) and
the per-section setup paragraphs are the places to cut.

## Cluster jobs still to run (none of these results are in `results/` yet)

All scripts fit the course QOS limit (≤ 64 CPUs per user, no `--exclusive`).
Submit from the repo root on a clean, pushed commit, then commit the job's
`results/<experiment>/<jobid>/` (raw/ is git-ignored) in a PR.

| Order | Command | Owner | Feeds |
|---|---|---|---|
| 1 | `sbatch slurm/stream.sbatch` | Hurshneet | roof of the roofline, tech table |
| 2 | `sbatch slurm/baseline.sbatch` | Marija / Ana-Maria | serial reference, time per gate vs qubits |
| 3 | `sbatch slurm/omp_scaling.sbatch` | Hurshneet | OpenMP strong scaling (1–32 threads) |
| 4 | `N=24 sbatch slurm/openmp_scaling.sbatch` | Hurshneet | OpenMP weak scaling |
| 5 | `QSIM_MODULES_EXTRA=<likwid module> sbatch slurm/omp_profile.sbatch` | Hurshneet | LIKWID bandwidth, roofline |
| 6 | `sbatch slurm/fusion.sbatch` | Marija | Optimisation 1 table + figure |
| 7 | `MODE=strong N=30 sbatch slurm/mpi_scaling.sbatch` | Edard | MPI strong scaling |
| 8 | `MODE=strong N=30 sbatch --ntasks-per-node=8 slurm/mpi_scaling.sbatch` | Edard | P = 16 on 2 nodes vs 1 |
| 9 | `MODE=weak N=28 sbatch slurm/mpi_scaling.sbatch` | Edard | weak scaling, EuroHPC plan |
| 10 | `QSIM_ARGS=--no-remap sbatch slurm/sensitivity.sbatch` and again without `QSIM_ARGS` | Ana-Maria + Edard | sensitivity, Optimisation 2 |
| 11 | hybrid: `MODE=strong N=30 RANKS=<P> sbatch --nodes=2 --ntasks-per-node=<P/2> --cpus-per-task=<64/P> slurm/mpi_scaling.sbatch` for P = 16, 8, 4, 2 | Edard + Hurshneet | hybrid table |

Then: `python3 analysis/make_figures.py` (all figures), `python3
analysis/scaleup.py` (paste the new tables into `scaleup_plan.md`), and
the reproducibility check from a fresh clone (Ana-Maria + one other person).

## Slides (5 × 1 minute)

1. Problem + architecture (Marija) · 2. OpenMP scaling + roofline (Hurshneet) ·
3. MPI scaling + remapping (Edard) · 4. CUDA vs CPU (Emma) ·
5. Methodology, reproducibility + EuroHPC (Ana-Maria)

# results/

Each Slurm job writes to `results/<experiment>/<jobid>/`:

- `meta.txt` – git commit, `module list`, node, CPU model, command lines, seeds
- `runs.csv` – one row per run (parsed from the `SUMMARY` line of `qsim`)
- `raw/` – per-gate timing CSVs (ignored by git; copy what you plot elsewhere)

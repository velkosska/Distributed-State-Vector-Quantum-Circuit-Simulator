"""Write per-binding strong-scaling CSVs for one completed OpenMP job."""

import argparse
from pathlib import Path

import pandas as pd

import qsimres as qr


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job", type=Path)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    runs = pd.read_csv(args.job / "runs.csv")
    if runs.empty or runs["status"].ne("ok").any() or runs["total_s"].le(0).any() or runs["total_s"].isna().any():
        parser.error("expected successful runs with positive timings")
    runs["workers"] = runs["threads"]
    runs["time_per_gate_s"] = runs["mean_gate_s"]
    summary = qr.summarize(runs)
    scaling = qr.add_speedup(summary, "strong").rename(
        columns={"tag": "omp_proc_bind", "median": "median_s", "min": "min_s", "max": "max_s", "n": "repeats"}
    )
    if scaling["repeats"].ne(args.repeats).any() or scaling["p0"].ne(1).any():
        parser.error(f"each configuration needs {args.repeats} runs and each series needs a one-thread baseline")
    common = ["kind", "qubits", "omp_proc_bind", "threads", "repeats", "median_s", "min_s", "max_s"]
    for metric in ("speedup", "efficiency"):
        columns = [*common, metric, f"{metric}_lo", f"{metric}_hi"]
        scaling.to_csv(args.job / f"{metric}.csv", columns=columns, index=False)


if __name__ == "__main__":
    main()

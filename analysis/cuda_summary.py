#!/usr/bin/env python3
"""CUDA experiment summary: GPU vs CPU time, transfer vs kernel time, achieved GB/s.

  python3 analysis/cuda_summary.py                      # every job under results/cuda/
  python3 analysis/cuda_summary.py --jobs 33869         # only these job ids (and their _partN)

Reads what slurm/cuda.sbatch writes to results/cuda/<jobid>/:
  runs.csv                   total_s per run (backend cuda = 1 GPU, openmp = CPU cores)
  raw/<run_id>_*.out         full stdout of each run; the GPU SUMMARY line has h2d_s / d2h_s
  raw/gpu_<kind>_<n>q.gates.csv   per-gate timing run: bytes moved per gate (traffic model)

Writes results/summary/cuda_summary.csv, one row per circuit kind and size:
  gpu_s, cpu_s, speedup      median total_s over repeats (gate time only, both backends)
  h2d_s, d2h_s, transfer_pct host<->device copies of the state and their share of
                             copies + kernels (CUDA event timers, see slurm/cuda.sbatch)
  gbytes                     bytes read + written by all gates of the circuit (same model
                             as the GB/s column of the per-gate CSVs, qsim::bytes_touched)
  gpu_gbps, cpu_gbps         gbytes / median total_s: achieved memory bandwidth
Only the Python standard library is needed, so it runs on the login node.
"""
from __future__ import annotations

import argparse
import csv
import re
import statistics
from collections import defaultdict
from pathlib import Path


def summary_fields(path: Path) -> dict[str, str]:
    """key=value pairs of the first SUMMARY line in a qsim stdout file."""
    try:
        for line in path.read_text().splitlines():
            if line.startswith("SUMMARY"):
                return dict(kv.split("=", 1) for kv in line.split()[1:] if "=" in kv)
    except OSError:
        pass
    return {}


def job_dirs(results: Path, jobs: list[str] | None, include_local: bool) -> list[Path]:
    dirs = []
    for d in sorted((results / "cuda").glob("*")):
        if not (d / "runs.csv").is_file():
            continue
        base = re.sub(r"_part\d+$", "", d.name)
        if jobs and base not in jobs:
            continue
        if base.startswith("local-") and not include_local:
            continue
        dirs.append(d)
    return dirs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="results")
    ap.add_argument("--jobs", nargs="*", help="job ids to use (default: all under results/cuda)")
    ap.add_argument("--include-local", action="store_true", help="keep local-* test jobs (not for the report)")
    ap.add_argument("--out", default=None, help="output CSV (default: <results>/summary/cuda_summary.csv)")
    args = ap.parse_args()

    results = Path(args.results)
    dirs = job_dirs(results, args.jobs, args.include_local)
    if not dirs:
        print(f"no CUDA jobs found under {results}/cuda")
        return 1

    total = defaultdict(list)     # (kind, n, backend) -> [total_s]
    h2d = defaultdict(list)       # (kind, n) -> [h2d_s]
    d2h = defaultdict(list)
    threads = {}                  # (kind, n) -> CPU threads
    gbytes = {}                   # (kind, n) -> bytes of the whole circuit / 1e9
    for d in dirs:
        for row in csv.DictReader(open(d / "runs.csv")):
            if row.get("status") != "ok" or not row.get("total_s") or row.get("tag"):
                continue
            key = (row["kind"], int(row["qubits"]))
            total[key + (row["backend"],)].append(float(row["total_s"]))
            if row["backend"] == "openmp":
                threads[key] = row["threads"]
            if row["backend"] == "cuda":
                out = next(iter((d / "raw").glob(f"{row['run_id']}_*.out")), None)
                f = summary_fields(out) if out else {}
                if "h2d_s" in f:
                    h2d[key].append(float(f["h2d_s"]))
                    d2h[key].append(float(f["d2h_s"]))
        for g in (d / "raw").glob("gpu_*_*q.gates.csv"):
            m = re.match(r"gpu_(.+)_(\d+)q\.gates\.csv$", g.name)
            if m:
                rows = list(csv.DictReader(open(g)))
                if rows:
                    gbytes[(m.group(1), int(m.group(2)))] = sum(float(r["bytes"]) for r in rows) * 1e-9

    med = lambda xs: statistics.median(xs) if xs else None  # noqa: E731
    keys = sorted({(k, n) for k, n, _ in total})
    out = Path(args.out) if args.out else results / "summary" / "cuda_summary.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    cols = ["kind", "qubits", "state_gib", "gpu_s", "cpu_s", "cpu_threads", "speedup", "h2d_s", "d2h_s",
            "transfer_pct", "gbytes", "gpu_gbps", "cpu_gbps", "gpu_repeats", "cpu_repeats"]
    fmt = lambda x, p=4: "" if x is None else f"{x:.{p}g}"  # noqa: E731
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        print(f"{'kind':7}{'n':>4}{'gpu_s':>11}{'cpu_s':>11}{'speedup':>9}{'h2d_s':>11}{'d2h_s':>11}"
              f"{'xfer%':>7}{'gpuGB/s':>9}{'cpuGB/s':>9}")
        for k, n in keys:
            g, c = med(total[(k, n, "cuda")]), med(total[(k, n, "openmp")])
            hi, lo = med(h2d[(k, n)]), med(d2h[(k, n)])
            gb = gbytes.get((k, n))
            xfer = 100 * (hi + lo) / (hi + lo + g) if None not in (hi, lo, g) else None
            row = {
                "kind": k, "qubits": n, "state_gib": fmt(16 * 2**n / 2**30),
                "gpu_s": fmt(g), "cpu_s": fmt(c), "cpu_threads": threads.get((k, n), ""),
                "speedup": fmt(c / g if g and c else None, 3), "h2d_s": fmt(hi), "d2h_s": fmt(lo),
                "transfer_pct": fmt(xfer, 3), "gbytes": fmt(gb),
                "gpu_gbps": fmt(gb / g if gb and g else None), "cpu_gbps": fmt(gb / c if gb and c else None),
                "gpu_repeats": len(total[(k, n, "cuda")]), "cpu_repeats": len(total[(k, n, "openmp")]),
            }
            w.writerow([row[c_] for c_ in cols])
            print(f"{k:7}{n:4d}{row['gpu_s']:>11}{row['cpu_s'] or '-':>11}{row['speedup'] or '-':>9}"
                  f"{row['h2d_s'] or '-':>11}{row['d2h_s'] or '-':>11}{row['transfer_pct'] or '-':>7}"
                  f"{row['gpu_gbps'] or '-':>9}{row['cpu_gbps'] or '-':>9}")
    print(f"wrote {out} (from {len(dirs)} job dir(s): {', '.join(d.name for d in dirs)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Split MPI run time into gates that exchange data between ranks and local gates.

Reads the per-gate timing files the MPI binary writes with --timing
(results/<experiment>/<jobid>/raw/*.gates.csv: one row per gate with its
MPI_Wtime duration and the bytes it sent) and writes one row per run.

  python3 analysis/comm_breakdown.py results/mpi_strong/36515 results/mpi_strong/36586 ... \
      -o results/summary/mpi_comm_breakdown.csv
"""
import argparse, csv, glob, os, re

ap = argparse.ArgumentParser()
ap.add_argument("jobdirs", nargs="+")
ap.add_argument("-o", "--out", required=True)
a = ap.parse_args()
rows = []
for jd in a.jobdirs:
    runs = {r["run_id"]: r for r in csv.DictReader(open(os.path.join(jd, "runs.csv")))}
    for f in sorted(glob.glob(os.path.join(jd, "raw", "*.gates.csv"))):
        rid = os.path.basename(f).split("_")[0]
        meta = runs.get(rid, {})
        tot = comm = comm_b = 0.0; ncomm = n = 0
        for g in csv.DictReader(open(f)):
            s = float(g["seconds"]); tot += s; n += 1
            b = float(g.get("comm_bytes") or 0)
            if b > 0:
                comm += s; comm_b += b; ncomm += 1
        rows.append({"job_id": os.path.basename(jd.rstrip("/")), "run_id": rid, "kind": meta.get("kind"),
                     "qubits": meta.get("qubits"), "ranks": meta.get("ranks"), "nodes": meta.get("nodes"),
                     "threads": meta.get("threads"), "repeat": meta.get("repeat"), "gates": n,
                     "comm_gates": ncomm, "gate_time_s": round(tot, 3), "comm_time_s": round(comm, 3),
                     "comm_share": round(comm / tot, 4) if tot else 0, "comm_GiB_per_rank": round(comm_b / 2**30, 3),
                     "comm_GBps_per_rank": round(comm_b / comm / 1e9, 4) if comm else ""})
os.makedirs(os.path.dirname(a.out), exist_ok=True)
with open(a.out, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print(f"wrote {len(rows)} runs to {a.out}")

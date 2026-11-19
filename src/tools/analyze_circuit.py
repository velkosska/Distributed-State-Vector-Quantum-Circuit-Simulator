#!/usr/bin/env python3
"""Classify the gates of a circuit for a distributed run on P = 2^p ranks.

With P ranks, qubits 0..n-p-1 are local and n-p..n-1 are global (bits of the
rank number). Each gate falls into one of three classes:

  local       all qubits it acts on are local: no communication
  global_free touches a global qubit but needs no data exchange:
              diagonal gates (Z S Sdg T Tdg P RZ CZ CP), and CX/CY whose
              control is global and target local (the rank knows the
              control bit and either applies X/Y locally or does nothing)
  comm        needs an exchange with a partner rank: a non-diagonal 1q gate
              on a global qubit, CX/CY with a global target, SWAP with
              exactly one global qubit
              (SWAP of two global qubits is a rank relabel: global_free)

`comm_MiB_per_rank` is an upper bound: every comm gate is counted as
exchanging the whole local slice (2^(n-p) * 16 bytes).

  analyze_circuit.py data/random_28.txt --ranks 1,2,4,8,16
  analyze_circuit.py data/*.txt --ranks 16 --csv results/analysis.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from qsim_circuit import Circuit  # noqa: E402

DIAGONAL = {"z", "s", "sdg", "t", "tdg", "p", "rz", "cz", "cp"}
CONTROLLED = {"cx", "cy"}


def classify(name: str, qubits: tuple[int, ...], n_local: int) -> str:
    is_global = [q >= n_local for q in qubits]
    if not any(is_global):
        return "local"
    if name in DIAGONAL:
        return "global_free"
    if name in CONTROLLED:
        control_global, target_global = is_global
        return "comm" if target_global else "global_free"
    if name == "swap":
        return "global_free" if all(is_global) else "comm"
    return "comm"  # dense 1q gate (h x y rx ry u) on a global qubit


def analyze(c: Circuit, ranks: int) -> dict:
    p = ranks.bit_length() - 1
    if ranks < 1 or 2 ** p != ranks:
        raise ValueError(f"ranks must be a power of two, got {ranks}")
    if p >= c.num_qubits:
        raise ValueError(f"{ranks} ranks need more than {c.num_qubits} qubits")
    n_local = c.num_qubits - p
    counts = {"local": 0, "global_free": 0, "comm": 0}
    comm_by_qubit: dict[int, int] = {}
    for name, qs, _ in c.ops:
        cls = classify(name, qs, n_local)
        counts[cls] += 1
        if cls == "comm":
            for q in qs:
                if q >= n_local:
                    comm_by_qubit[q] = comm_by_qubit.get(q, 0) + 1
    total = len(c.ops)
    slice_mib = 2 ** n_local * 16 / 2 ** 20
    touching_global = counts["global_free"] + counts["comm"]
    return {
        "qubits": c.num_qubits,
        "ranks": ranks,
        "local_qubits": n_local,
        "gates": total,
        "local": counts["local"],
        "global_free": counts["global_free"],
        "comm": counts["comm"],
        "pct_touching_global": 100.0 * touching_global / total if total else 0.0,
        "pct_comm": 100.0 * counts["comm"] / total if total else 0.0,
        "comm_MiB_per_rank": counts["comm"] * slice_mib,
        "comm_by_global_qubit": " ".join(f"q{q}:{k}" for q, k in sorted(comm_by_qubit.items())),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("circuits", nargs="+", help="circuit files in qsim text format")
    ap.add_argument("--ranks", default="1,2,4,8,16", help="comma-separated powers of two")
    ap.add_argument("--csv", help="also write the table to this CSV file")
    args = ap.parse_args(argv)

    rows = []
    for path in args.circuits:
        c = Circuit.read(path)
        for r in (int(x) for x in args.ranks.split(",")):
            if (r.bit_length() - 1) >= c.num_qubits:
                continue
            row = {"circuit": Path(path).name, **analyze(c, r)}
            rows.append(row)

    hdr = f"{'circuit':24}{'n':>4}{'ranks':>6}{'gates':>8}{'local':>8}{'g_free':>8}{'comm':>7}{'%glob':>7}{'%comm':>7}{'MiB/rank':>11}"
    print(hdr)
    for r in rows:
        print(f"{r['circuit'][:23]:24}{r['qubits']:>4}{r['ranks']:>6}{r['gates']:>8}{r['local']:>8}"
              f"{r['global_free']:>8}{r['comm']:>7}{r['pct_touching_global']:>7.1f}{r['pct_comm']:>7.1f}"
              f"{r['comm_MiB_per_rank']:>11.1f}")
    if args.csv and rows:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()

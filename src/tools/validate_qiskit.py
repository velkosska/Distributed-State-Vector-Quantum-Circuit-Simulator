#!/usr/bin/env python3
"""Compare qsim's final state vector with Qiskit's Statevector.

For each circuit kind and qubit count the script generates a circuit, runs
`qsim --dump`, and checks max_i |psi_qsim[i] - psi_qiskit[i]| <= tolerance.
Exit status is non-zero if any case fails.

  python3 src/tools/validate_qiskit.py --qsim build/qsim --max-qubits 20
"""
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gen_circuits as gen  # noqa: E402
from qsim_circuit import Circuit  # noqa: E402


def make_cases(kinds, sizes, seed):
    for kind in kinds:
        for n in sizes:
            if kind == "qft":
                yield kind, n, gen.qft(n, prep_seed=seed)
            elif kind == "grover":
                # cap iterations so large n stays quick; correctness does not need the optimum
                c, _, _ = gen.grover(n, seed, iterations=None if n <= 12 else 2)
                yield kind, n, c
            elif kind == "random":
                yield kind, n, gen.random_circuit(n, depth=10, seed=seed + n)


def qiskit_state(c: Circuit) -> np.ndarray:
    from qiskit.quantum_info import Statevector
    return np.asarray(Statevector(c.to_qiskit()).data)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--qsim", default="build/qsim", help="path to the qsim binary")
    p.add_argument("--min-qubits", type=int, default=2)
    p.add_argument("--max-qubits", type=int, default=20)
    p.add_argument("--step", type=int, default=1)
    p.add_argument("--kinds", default="qft,grover,random")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--tol", type=float, default=1e-10)
    p.add_argument("--csv", help="also write results to this CSV file")
    p.add_argument("--keep", help="keep generated circuits and dumps in this directory")
    args = p.parse_args(argv)

    qsim = Path(args.qsim).resolve()
    if not qsim.exists():
        sys.exit(f"qsim binary not found: {qsim} (build first, see README)")

    sizes = list(range(args.min_qubits, args.max_qubits + 1, args.step))
    workdir = Path(args.keep) if args.keep else Path(tempfile.mkdtemp(prefix="qsim_val_"))
    workdir.mkdir(parents=True, exist_ok=True)

    rows, failed = [], 0
    print(f"{'kind':8}{'n':>4}{'gates':>8}{'max_abs_err':>14}{'qsim_s':>10}{'qiskit_s':>10}  result")
    for kind, n, circ in make_cases(args.kinds.split(","), sizes, args.seed):
        cfile = workdir / f"{kind}_{n}.txt"
        dfile = workdir / f"{kind}_{n}.bin"
        circ.write(str(cfile))

        t0 = time.perf_counter()
        subprocess.run([str(qsim), str(cfile), "--dump", str(dfile), "--quiet", "--no-gate-timing"],
                       check=True, stdout=subprocess.DEVNULL)
        t_qsim = time.perf_counter() - t0
        ours = np.fromfile(dfile, dtype=np.complex128)

        t0 = time.perf_counter()
        ref = qiskit_state(Circuit.read(str(cfile)))  # re-read: also checks the file round-trip
        t_qk = time.perf_counter() - t0

        err = float(np.max(np.abs(ours - ref))) if ours.shape == ref.shape else float("inf")
        ok = err <= args.tol
        failed += not ok
        print(f"{kind:8}{n:>4}{len(circ.ops):>8}{err:>14.3e}{t_qsim:>10.3f}{t_qk:>10.3f}  {'ok' if ok else 'FAIL'}")
        rows.append(dict(kind=kind, qubits=n, gates=len(circ.ops), max_abs_err=err,
                         qsim_s=t_qsim, qiskit_s=t_qk, ok=ok))
        if not args.keep:
            dfile.unlink(); cfile.unlink()

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)

    print(f"\n{len(rows) - failed}/{len(rows)} cases within tolerance {args.tol:g}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Generate benchmark circuits in the qsim text format.

All generators are deterministic: the same arguments (including --seed)
always produce byte-identical files.

  gen_circuits.py qft    -n 24 -o data/qft_24.txt
  gen_circuits.py grover -n 24 --seed 1 --iterations 2 -o data/grover_24.txt
  gen_circuits.py random -n 24 --depth 20 --seed 1 -o data/random_24.txt
"""
from __future__ import annotations

import argparse
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from qsim_circuit import Circuit  # noqa: E402


# --------------------------------------------------------------------- QFT
def qft(n: int, with_swaps: bool = True, prep_seed: int | None = None) -> Circuit:
    """Textbook QFT. Qubit n-1 is the most significant bit.

    If prep_seed is given, a seeded layer of RY rotations is prepended so the
    input is not |0...0> (QFT of |0> is trivially uniform, which makes a weak
    correctness test).
    """
    c = Circuit(n)
    if prep_seed is not None:
        rng = random.Random(prep_seed)
        for q in range(n):
            c.ry(q, theta=rng.uniform(0, 2 * math.pi))
    for j in reversed(range(n)):
        c.h(j)
        for k in reversed(range(j)):
            c.cp(k, j, theta=math.pi / 2 ** (j - k))
    if with_swaps:
        for q in range(n // 2):
            c.swap(q, n - 1 - q)
    return c


# ------------------------------------------------------------------ Grover
def _toffoli(c: Circuit, a: int, b: int, t: int):
    """Exact Toffoli from H, T, Tdg, CX (no global phase)."""
    c.h(t); c.cx(b, t); c.tdg(t); c.cx(a, t); c.t(t); c.cx(b, t); c.tdg(t); c.cx(a, t)
    c.t(b); c.t(t); c.h(t); c.cx(a, b); c.t(a); c.tdg(b); c.cx(a, b)


def _mcx(c: Circuit, controls: list[int], target: int, anc: list[int]):
    """Multi-controlled X using the V-chain construction (len(anc) >= k-2)."""
    k = len(controls)
    if k == 1:
        c.cx(controls[0], target); return
    if k == 2:
        _toffoli(c, controls[0], controls[1], target); return
    chain = [(controls[0], controls[1], anc[0])]
    for i in range(2, k - 1):
        chain.append((controls[i], anc[i - 2], anc[i - 1]))
    for a, b, t in chain:
        _toffoli(c, a, b, t)
    _toffoli(c, controls[k - 1], anc[k - 3], target)
    for a, b, t in reversed(chain):
        _toffoli(c, a, b, t)


def _mcz(c: Circuit, data: list[int], anc: list[int]):
    if len(data) == 2:
        c.cz(data[0], data[1]); return
    c.h(data[-1]); _mcx(c, data[:-1], data[-1], anc); c.h(data[-1])


def grover(n: int, seed: int, iterations: int | None = None) -> tuple[Circuit, int, int]:
    """Grover search for one seeded marked item.

    Uses m = (n+3)//2 data qubits (0..m-1) and m-3 ancillas (m..2m-4) for the
    multi-controlled Z; any remaining qubit stays idle. Returns
    (circuit, marked_item, iterations).
    """
    if n < 2:
        raise ValueError("grover needs n >= 2")
    m = (n + 3) // 2 if n >= 3 else 2
    data = list(range(m))
    anc = list(range(m, m + max(0, m - 3)))
    marked = random.Random(seed).randrange(2 ** m)
    if iterations is None:
        iterations = max(1, int(math.floor(math.pi / 4 * math.sqrt(2 ** m))))
    c = Circuit(n)
    for q in data:
        c.h(q)
    zero_bits = [q for q in data if not (marked >> q) & 1]
    for _ in range(iterations):
        for q in zero_bits: c.x(q)          # oracle: phase flip on |marked>
        _mcz(c, data, anc)
        for q in zero_bits: c.x(q)
        for q in data: c.h(q)               # diffusion: 2|s><s| - I
        for q in data: c.x(q)
        _mcz(c, data, anc)
        for q in data: c.x(q)
        for q in data: c.h(q)
    return c, marked, iterations


# ------------------------------------------------------------------ random
ONE_Q_FIXED = ["h", "x", "y", "z", "s", "t"]
ONE_Q_ROT = ["rx", "ry", "rz"]
TWO_Q = ["cx", "cz", "swap", "cp"]


def random_circuit(n: int, depth: int, seed: int, two_qubit_prob: float = 0.5) -> Circuit:
    """`depth` layers; each layer puts one random 1q gate on every qubit, then
    pairs qubits at random and applies a random 2q gate to each pair with
    probability two_qubit_prob."""
    rng = random.Random(seed)
    c = Circuit(n)
    for _ in range(depth):
        for q in range(n):
            g = rng.choice(ONE_Q_FIXED + ONE_Q_ROT)
            if g in ONE_Q_ROT:
                c.add(g, q, params=(rng.uniform(0, 2 * math.pi),))
            else:
                c.add(g, q)
        perm = list(range(n)); rng.shuffle(perm)
        for a, b in zip(perm[0::2], perm[1::2]):
            if rng.random() < two_qubit_prob:
                g = rng.choice(TWO_Q)
                params = (rng.uniform(0, 2 * math.pi),) if g == "cp" else ()
                c.add(g, a, b, params=params)
    return c


# --------------------------------------------------------------------- CLI
def build(args) -> tuple[Circuit, str]:
    if args.kind == "qft":
        c = qft(args.n, with_swaps=not args.no_swaps, prep_seed=args.seed)
        info = f"qft n={args.n} prep_seed={args.seed} swaps={not args.no_swaps}"
    elif args.kind == "grover":
        c, marked, it = grover(args.n, args.seed, args.iterations)
        info = f"grover n={args.n} seed={args.seed} marked={marked} iterations={it}"
    else:
        c = random_circuit(args.n, args.depth, args.seed, args.two_qubit_prob)
        info = (f"random n={args.n} depth={args.depth} seed={args.seed} "
                f"two_qubit_prob={args.two_qubit_prob}")
    return c, f"{info} gates={len(c.ops)}"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("kind", choices=["qft", "grover", "random"])
    p.add_argument("-n", type=int, required=True, help="number of qubits")
    p.add_argument("-o", "--output", help="output file (default: stdout)")
    p.add_argument("--seed", type=int, default=None,
                   help="RNG seed (grover/random default 1; qft: seeded RY input layer if given)")
    p.add_argument("--depth", type=int, default=20, help="random: number of layers")
    p.add_argument("--two-qubit-prob", type=float, default=0.5, help="random: chance of a 2q gate per pair")
    p.add_argument("--iterations", type=int, default=None, help="grover: iterations (default: optimal)")
    p.add_argument("--no-swaps", action="store_true", help="qft: omit final bit-reversal swaps")
    args = p.parse_args(argv)
    if args.kind != "qft" and args.seed is None:
        args.seed = 1

    c, info = build(args)
    text = c.to_text(f"generated by gen_circuits.py: {info}")
    if args.output:
        Path(args.output).write_text(text)
        print(info, file=sys.stderr)
    else:
        sys.stdout.write(text)


if __name__ == "__main__":
    main()

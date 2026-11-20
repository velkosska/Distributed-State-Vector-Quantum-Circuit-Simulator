#!/usr/bin/env python3
"""EuroHPC scale-up estimate: memory per qubit count and node-hours.

  python3 analysis/scaleup.py                                   # MN5 GPP defaults, model only
  python3 analysis/scaleup.py --weak-experiment mpi_weak        # use Edard's weak-scaling runs
  python3 analysis/scaleup.py --system lumi-c --targets 34,36,38,40

Memory.  A state of n qubits is 16 * 2^n bytes. With P = 2^p ranks each rank
holds 16 * 2^(n-p) bytes plus the receive buffer of the global-qubit exchange
(--buffer-factor: 1.0 = full slice, as in a plain MPI_Sendrecv of the slice;
0.0625 = exchange in 16 chunks). One rank per NUMA domain or per node does not
change the per-node total, so the table is per node.

Time.  Weak scaling keeps 2^(n-p) amplitudes per node, so the time per gate
stays roughly constant except for communication. Two sources, in this order:
  1. measured: median time per gate from the weak-scaling experiment, fitted as
     t_gate(P) = a + b * log2(P) per circuit kind and extrapolated in log2(P);
  2. model (no runs yet): local gates move 32 B per amplitude at the node's
     memory bandwidth; a communicating gate additionally sends and receives the
     whole slice over the node's network link.
The campaign (--per-node-qubits b, up to 2^K nodes):
  weak         b + k qubits on 2^k nodes, k = 0..K, kinds from --kinds
  strong       b + s qubits on 2^s .. 2^K nodes (s = --strong-start-log2)
  sensitivity  b + s qubits on 2^s nodes: qft, grover, random, and random with
               global fraction 0 .. 1 (gen_circuits.py --global-fraction)
Node-hours = sum over planned runs of gates(n) * t_gate * nodes * repeats / 3600,
times --overhead (job start-up, circuit generation, failed runs; default 1.5).
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "src" / "tools"))
import qsimres as qr  # noqa: E402

GiB = 2 ** 30

# Node data from the system documentation (checked 2026-11, see docs/report/scaleup_plan.md).
# usable_gib leaves room for the OS and MPI; mem_bw_gbs is a planning value
# to be replaced by our STREAM measurement (slurm/stream.sbatch) on the system.
SYSTEMS = {
    "mn5-gpp": dict(name="MareNostrum 5 GPP (BSC)", nodes=6192, cores=112, mem_gib=256, usable_gib=224,
                    net_gbs=12.5, mem_bw_gbs=400.0,
                    note="2x Xeon 8480+ (56c), DDR5; NDR200 shared by 2 nodes = 100 Gb/s per node; "
                         "216 HighMem nodes with 1024 GiB"),
    "mn5-gpp-highmem": dict(name="MareNostrum 5 GPP HighMem", nodes=216, cores=112, mem_gib=1024, usable_gib=990,
                            net_gbs=12.5, mem_bw_gbs=400.0, note="same CPUs, 1 TiB DDR5"),
    "lumi-c": dict(name="LUMI-C (CSC)", nodes=1888, cores=128, mem_gib=256, usable_gib=224,
                   net_gbs=25.0, mem_bw_gbs=300.0,
                   note="2x EPYC 7763 (64c), Slingshot-11 200 Gb/s; 128 nodes 512 GiB, 32 nodes 1 TiB"),
}


def memory_table(targets, sysd, buffer_factor) -> pd.DataFrame:
    rows = []
    usable = sysd["usable_gib"] * GiB
    per_amp = 16 * (1 + buffer_factor)
    for n in targets:
        state = 16 * 2.0 ** n
        # smallest power-of-two node count whose per-node share fits
        p = max(0, math.ceil(math.log2(per_amp * 2.0 ** n / usable)))
        nodes = 2 ** p
        rows.append(dict(qubits=n, state_GiB=state / GiB, nodes_min=nodes,
                         per_node_GiB=per_amp * 2.0 ** n / nodes / GiB,
                         fits_system=nodes <= sysd["nodes"],
                         share_of_system=nodes / sysd["nodes"]))
    return pd.DataFrame(rows)


def gate_count(kind: str, n: int, depth: int, seed: int) -> int:
    import gen_circuits as gc  # noqa: E402  (src/tools)
    if kind == "qft":
        return len(gc.qft(n, prep_seed=seed).ops)
    if kind == "grover":
        return len(gc.grover(n, seed, iterations=1)[0].ops)
    return len(gc.random_circuit(n, depth, seed).ops)


def comm_share(kind: str, n: int, p: int, depth: int, seed: int) -> float:
    """Share of gates needing an exchange with 2^p ranks (analyze_circuit.py rules)."""
    import gen_circuits as gc  # noqa: E402
    import analyze_circuit as ac  # noqa: E402
    if p == 0:
        return 0.0
    c = (gc.qft(n, prep_seed=seed) if kind == "qft" else
         gc.grover(n, seed, iterations=1)[0] if kind == "grover" else gc.random_circuit(n, depth, seed))
    return ac.analyze(c, 2 ** p)["pct_comm"] / 100


def fit_weak(results: Path, experiment: str) -> pd.DataFrame:
    """Per kind: a, b in t_gate = a + b*log2(P) from the weak-scaling medians (per node)."""
    runs = qr.load_runs(results, [experiment])
    if runs.empty:
        return pd.DataFrame()
    s = qr.summarize(runs)
    if s["nodes"].max() <= 1:
        print("warning: the weak-scaling runs use a single node, so they contain no inter-node "
              "communication; the extrapolation to many nodes is optimistic. Run mpi_scaling.sbatch "
              "MODE=weak with --nodes >= 4.", file=sys.stderr)
    rows = []
    for kind, g in s.groupby("kind"):
        x = np.log2(g["nodes"].clip(lower=1).astype(float))
        y = g["tpg_median"].to_numpy()
        if x.nunique() >= 2:
            b, a = np.polyfit(x, y, 1)
        else:
            a, b = float(y.mean()), 0.0
        n_per_node = int((g["qubits"] - np.log2(g["nodes"].clip(lower=1))).round().mode()[0])
        rows.append(dict(kind=kind, a=a, b=max(b, 0.0), local_qubits_per_node=n_per_node,
                         max_nodes_measured=int(g["nodes"].max()), points=len(g)))
    return pd.DataFrame(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--system", default="mn5-gpp", choices=list(SYSTEMS))
    ap.add_argument("--targets", default="32,34,36,38,40", help="qubit counts to plan for")
    ap.add_argument("--kinds", default="qft,random")
    ap.add_argument("--depth", type=int, default=10, help="random circuit depth")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--per-node-qubits", type=int, default=32,
                    help="weak scaling: qubits on one node (32 = 64 GiB state + 64 GiB buffer)")
    ap.add_argument("--weak-max-log2-nodes", type=int, default=8, help="weak/strong up to 2^k nodes (8 = 256)")
    ap.add_argument("--strong-start-log2", type=int, default=4,
                    help="strong scaling and sensitivity at per-node-qubits + k qubits from 2^k nodes")
    ap.add_argument("--buffer-factor", type=float, default=1.0, help="exchange buffer / slice (1 = full slice)")
    ap.add_argument("--overhead", type=float, default=1.5)
    ap.add_argument("--results", default="results")
    ap.add_argument("--weak-experiment", default="mpi_weak")
    ap.add_argument("--mem-bw", type=float, help="override node memory bandwidth [GB/s] (use STREAM)")
    ap.add_argument("--net-bw", type=float, help="override node network bandwidth [GB/s] per direction")
    ap.add_argument("--out", default=None, help="directory for CSVs (default results/summary)")
    args = ap.parse_args(argv)

    sysd = dict(SYSTEMS[args.system])
    if args.mem_bw: sysd["mem_bw_gbs"] = args.mem_bw
    if args.net_bw: sysd["net_gbs"] = args.net_bw
    targets = [int(x) for x in args.targets.split(",")]
    kinds = args.kinds.split(",")
    out = Path(args.out or Path(args.results) / "summary")
    out.mkdir(parents=True, exist_ok=True)

    mem = memory_table(targets, sysd, args.buffer_factor)
    mem.to_csv(out / f"scaleup_memory_{args.system}.csv", index=False)
    print(f"# {sysd['name']}: {sysd['nodes']} nodes x {sysd['mem_gib']} GiB ({sysd['usable_gib']} usable), "
          f"buffer factor {args.buffer_factor}")
    print(mem.to_string(index=False, float_format=lambda v: f"{v:.3g}"))

    fit = fit_weak(Path(args.results), args.weak_experiment)
    source = "measured weak scaling" if not fit.empty else "bandwidth model (no weak-scaling runs found)"

    def t_gate(kind, n, nodes, f_comm):
        """(seconds per gate, how): fit if we have one for this kind, else the model."""
        p = int(math.log2(nodes))
        slice_bytes = 16 * 2.0 ** (n - p)
        t_model = 2 * slice_bytes / (sysd["mem_bw_gbs"] * 1e9) + f_comm * slice_bytes / (sysd["net_gbs"] * 1e9)
        fk = fit[fit["kind"] == kind.replace("random_gf", "random")] if not fit.empty else fit
        if fk.empty:
            return t_model, t_model, "model"
        r = fk.iloc[0]
        scale = 2.0 ** ((n - p) - r["local_qubits_per_node"])
        return (r["a"] + r["b"] * p) * scale, t_model, f"fit (measured to {int(r['max_nodes_measured'])} nodes)"

    # The campaign: (phase, kind, qubits, nodes, global_fraction)
    base = args.per_node_qubits
    plan = []
    for k in range(0, args.weak_max_log2_nodes + 1):                                   # weak
        for kind in kinds:
            plan.append(("weak", kind, base + k, 2 ** k, None))
    n_s = base + args.strong_start_log2
    for k in range(args.strong_start_log2, args.weak_max_log2_nodes + 1):              # strong
        for kind in kinds:
            plan.append(("strong", kind, n_s, 2 ** k, None))
    for kind in ["qft", "grover", "random"]:                                            # sensitivity
        plan.append(("sensitivity", kind, n_s, 2 ** args.strong_start_log2, None))
    for f in [0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1]:
        plan.append(("sensitivity", "random_gf", n_s, 2 ** args.strong_start_log2, f))

    import gen_circuits as gc  # noqa: E402
    import analyze_circuit as ac  # noqa: E402
    rows = []
    for phase, kind, n, nodes, f in plan:
        if 16 * (1 + args.buffer_factor) * 2.0 ** n / nodes > sysd["usable_gib"] * GiB:
            continue  # does not fit: skip (strong scaling below the memory limit)
        p = int(math.log2(nodes))
        if kind == "random_gf":
            c = gc.random_global_fraction(n, args.depth, args.seed, f, max(p, 1))
        elif kind == "qft":
            c = gc.qft(n, prep_seed=args.seed)
        elif kind == "grover":
            c = gc.grover(n, args.seed, iterations=1)[0]
        else:
            c = gc.random_circuit(n, args.depth, args.seed)
        f_comm = ac.analyze(c, nodes)["pct_comm"] / 100 if p > 0 else 0.0
        tg, tm, how = t_gate(kind, n, nodes, f_comm)
        gates = len(c.ops)
        rows.append(dict(phase=phase, kind=kind, global_fraction=f, qubits=n, nodes=nodes, gates=gates,
                         pct_comm=100 * f_comm, t_gate_s=tg, t_gate_model_s=tm, source=how,
                         run_minutes=gates * tg / 60,
                         node_hours=gates * tg * nodes * args.repeats / 3600 * args.overhead))
    nh = pd.DataFrame(rows)
    nh.to_csv(out / f"scaleup_nodehours_{args.system}.csv", index=False)
    print(f"\n# campaign node-hours ({source}; {args.repeats} repeats, x{args.overhead} overhead; "
          f"{base} qubits per node)")
    print(nh.to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    by = nh.groupby("phase")["node_hours"].sum()
    print("\n" + "\n".join(f"{ph:12s} {h:8.0f} node-hours" for ph, h in by.items()))
    print(f"{'total':12s} {nh['node_hours'].sum():8.0f} node-hours "
          f"(= {nh['node_hours'].sum() * sysd['cores']:.0f} core-hours)")
    if not fit.empty:
        fit.to_csv(out / "scaleup_weak_fit.csv", index=False)

if __name__ == "__main__":
    main()

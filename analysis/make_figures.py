#!/usr/bin/env python3
"""Build every summary table and figure from the results/ tree.

  python3 analysis/make_figures.py                       # all jobs under results/
  python3 analysis/make_figures.py --clean-only          # skip runs from a dirty git tree
  python3 analysis/make_figures.py --commit 1a2b3c4      # one code version only
  python3 analysis/make_figures.py --peak-gflops 35.2    # per-core FP64 peak for the roofline

Writes
  results/summary/runs_summary.csv    median/min/max/range over repeats, every configuration
  results/summary/scaling_strong.csv  speedup, efficiency, Karp-Flatt per point
  results/summary/scaling_weak.csv
  results/summary/amdahl.csv          serial fraction f per strong-scaling curve
  results/summary/sensitivity.csv     time vs share of communicating gates (MPI)
  results/summary/gate_bandwidth.csv  median GB/s and GFLOP/s per gate type
  results/figures/*.pdf and *.png

Figures are static (report) plots. Every series has a legend, a marker shape as
well as a colour, and colour is fixed per entity (circuit kind, gate type), so a
figure that drops a series never repaints the others.
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qsimres as qr  # noqa: E402

# Categorical palette, fixed order (validated: CVD-safe adjacent pairs on white).
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
MARKERS = ["o", "s", "^", "D", "v", "P"]
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3de"
KIND_ORDER = ["qft", "grover", "random", "random_gf"]
GATE_GROUPS = {  # gate types grouped by kernel (same memory traffic pattern)
    "dense 1q (H X Y RX RY U)": ["h", "x", "y", "rx", "ry", "u"],
    "diagonal 1q (Z S T P RZ)": ["z", "s", "sdg", "t", "tdg", "p", "rz"],
    "controlled (CX CY)": ["cx", "cy"],
    "controlled phase (CZ CP)": ["cz", "cp"],
    "swap": ["swap"],
}


def style(name: str, order: list[str]):
    i = order.index(name) if name in order else len(order) % len(SLOTS)
    return dict(color=SLOTS[i % len(SLOTS)], marker=MARKERS[i % len(MARKERS)])


def setup_matplotlib():
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 200, "font.size": 9, "axes.titlesize": 10,
        "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
        "axes.spines.right": False, "lines.linewidth": 2, "lines.markersize": 6, "legend.frameon": False,
        "text.color": INK,
    })


def save(fig, outdir: Path, name: str):
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(outdir / f"{name}.{ext}")
    plt.close(fig)
    print(f"  figure  {outdir / name}.pdf")


def errbar(ax, x, y, lo, hi, label, **kw):
    yerr = np.vstack([np.clip(y - lo, 0, None), np.clip(hi - y, 0, None)])
    ax.errorbar(x, y, yerr=yerr, label=label, capsize=2, elinewidth=1, **kw)


def log2_axis(ax):
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))


# ------------------------------------------------------------------ figures
def fig_time_per_gate(summary: pd.DataFrame, outdir: Path):
    """Time per gate vs qubits, one panel per backend/worker count with >= 2 sizes."""
    s = summary[~summary["experiment"].str.contains("weak|sensitivity")]
    series = [(k, g) for k, g in s.groupby(["experiment", "backend", "workers"]) if g["qubits"].nunique() >= 2]
    if not series:
        return
    fig, axes = plt.subplots(1, len(series), figsize=(4.2 * len(series), 3.4), squeeze=False, sharey=True)
    for ax, ((exp, backend, workers), g) in zip(axes[0], series):
        for kind, gk in sorted(g.groupby("kind"), key=lambda t: KIND_ORDER.index(t[0]) if t[0] in KIND_ORDER else 99):
            gk = gk.sort_values("qubits")
            errbar(ax, gk["qubits"], gk["tpg_median"], gk["tpg_min"], gk["tpg_max"], kind, **style(kind, KIND_ORDER))
        # 2^n reference through the largest point of the first series
        big = g.sort_values("qubits").iloc[-1]
        n = np.array(sorted(g["qubits"].unique()))
        ax.plot(n, big["tpg_median"] * 2.0 ** (n - big["qubits"]), ls="--", lw=1, color=MUTED, label=r"$\propto 2^n$")
        ax.set_yscale("log")
        ax.set_xlabel("qubits n")
        ax.set_title(f"{exp} ({backend}, {workers} {'GPU' if backend == 'cuda' else 'cores'})")
    axes[0][0].set_ylabel("median time per gate [s]")
    axes[0][-1].legend(loc="upper left")
    save(fig, outdir, "time_per_gate_vs_qubits")


def fig_strong(strong: pd.DataFrame, outdir: Path) -> pd.DataFrame:
    rows = []
    if strong.empty:
        return pd.DataFrame()
    for exp, ge in strong.groupby("experiment"):
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.4, 3.4))
        pmax = ge["workers"].max()
        p = np.unique(np.r_[ge["workers"].unique(), pmax])
        a1.plot(p, p, ls="--", lw=1, color=MUTED, label="ideal")
        curves = list(ge.groupby(["kind", "qubits", "git_commit"]))
        names = [f"{k} {n}q" for (k, n, _), _ in curves]
        for ((kind, n, commit), g), name in zip(curves, names):
            g = g.sort_values("workers")
            st = style(name, names)
            errbar(a1, g["workers"], g["speedup"], g["speedup_lo"], g["speedup_hi"], name, **st)
            errbar(a2, g["workers"], g["efficiency"], g["efficiency_lo"], g["efficiency_hi"], name, **st)
            f, rmse = qr.amdahl_fit(g["workers"], g["speedup"])
            f_lo, _ = qr.amdahl_fit(g["workers"], g["speedup_hi"])
            f_hi, _ = qr.amdahl_fit(g["workers"], g["speedup_lo"])
            if np.isfinite(f):
                pp = np.geomspace(1, pmax, 50)
                a1.plot(pp, qr.amdahl_speedup(pp, f), ls=":", lw=1.2, color=st["color"])
            rows.append(dict(experiment=exp, kind=kind, qubits=n, git_commit=commit, p0=g["p0"].iloc[0],
                             max_workers=g["workers"].max(), serial_fraction=f, serial_fraction_lo=f_lo,
                             serial_fraction_hi=f_hi, amdahl_max_speedup=1 / f if f > 0 else np.inf,
                             fit_rmse=rmse, speedup_at_max=g["speedup"].iloc[-1],
                             efficiency_at_max=g["efficiency"].iloc[-1]))
        for a in (a1, a2):
            log2_axis(a)
            a.set_xlabel("workers (ranks x threads)" if "mpi" in exp else "threads")
        a1.set_yscale("log", base=2)
        a1.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        a1.set_ylabel("speedup (dotted: Amdahl fit)")
        a2.set_ylabel("parallel efficiency")
        a2.set_ylim(0, 1.15)
        a2.axhline(1, ls="--", lw=1, color=MUTED)
        a1.set_title(f"strong scaling: {exp}")
        a2.set_title("efficiency (bars: repeat range)")
        a2.legend(loc="lower left")
        save(fig, outdir, f"strong_{exp}")
    return pd.DataFrame(rows)


def fig_weak(weak: pd.DataFrame, outdir: Path):
    for exp, ge in weak.groupby("experiment"):
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.4, 3.4))
        curves = list(ge.groupby(["kind", "git_commit"]))
        names = [k for (k, _), _ in curves]
        for ((kind, _), g), name in zip(curves, names):
            g = g.sort_values("workers")
            st = style(name, KIND_ORDER)
            errbar(a1, g["workers"], g["tpg_median"], g["tpg_min"], g["tpg_max"], name, **st)
            errbar(a2, g["workers"], g["efficiency"], g["efficiency_lo"], g["efficiency_hi"], name, **st)
            for _, r in g.iterrows():
                a1.annotate(f"{r['qubits']}q", (r["workers"], r["tpg_median"]), textcoords="offset points",
                            xytext=(4, 4), fontsize=7, color=MUTED)
        for a in (a1, a2):
            log2_axis(a)
            a.set_xlabel("workers (ranks x threads)" if "mpi" in exp else "threads")
        a1.set_ylabel("median time per gate [s]")
        a1.set_ylim(bottom=0)
        a2.set_ylabel("weak efficiency  t_gate(p0) / t_gate(p)")
        a2.set_ylim(0, 1.15)
        a2.axhline(1, ls="--", lw=1, color=MUTED)
        a1.set_title(f"weak scaling: {exp} (+1 qubit per doubling)")
        a2.set_title("efficiency (bars: repeat range)")
        a2.legend(loc="lower left")
        save(fig, outdir, f"weak_{exp}")


def gate_group(g: str) -> str:
    for name, members in GATE_GROUPS.items():
        if g in members:
            return name
    return "other"


def fig_bandwidth(gates: pd.DataFrame, stream: pd.DataFrame, outdir: Path, peak_gflops_core: float | None):
    """(1) GB/s vs qubits per gate group (cache -> DRAM transition) with STREAM;
    (2) GB/s vs threads with the STREAM curve; (3) roofline at the largest size."""
    if gates.empty:
        return pd.DataFrame()
    gates = gates.assign(group=gates["gate"].map(gate_group))
    groups = list(GATE_GROUPS)
    agg = (gates.groupby(["experiment", "backend", "qubits", "ranks", "threads", "group"])
           .agg(gbps=("gbps", "median"), gflops=("gflops", "median"), ai=("ai", "median"), n=("gbps", "size"))
           .reset_index())

    def stream_at(t: int) -> float:
        if stream.empty:
            return np.nan
        s = stream[stream["threads"] <= t]
        return s.groupby("threads")["triad_gbs"].max().iloc[-1] if not s.empty else np.nan

    # (1) serial: GB/s vs qubits
    ser = agg[(agg["threads"] == 1) & (agg["ranks"] == 1) & (agg["backend"] != "cuda")]
    if ser["qubits"].nunique() >= 2:
        fig, ax = plt.subplots(figsize=(5.2, 3.4))
        for grp, g in ser.groupby("group"):
            g = g.groupby("qubits")["gbps"].median().reset_index()
            ax.plot(g["qubits"], g["gbps"], label=grp, **style(grp, groups))
        bw1 = stream_at(1)
        if np.isfinite(bw1):
            ax.axhline(bw1, ls="--", lw=1, color=MUTED)
            ax.annotate(f"STREAM triad, 1 core: {bw1:.1f} GB/s", (ax.get_xlim()[1], bw1), xytext=(-2, 3),
                        textcoords="offset points", fontsize=7, color=MUTED, ha="right")
        ax.set_xlabel("qubits n  (state = 16 B x 2^n)")
        ax.set_ylabel("effective bandwidth [GB/s]")
        ax.set_title("serial: bandwidth per gate type vs size")
        ax.legend(fontsize=7)
        save(fig, outdir, "bandwidth_vs_qubits_serial")

    # (2) OpenMP: GB/s vs threads at the largest size
    omp = agg[(agg["backend"] == "openmp") & (agg["ranks"] == 1)]
    if not omp.empty and omp["threads"].nunique() >= 2:
        nmax = omp.groupby("threads")["qubits"].max().min()
        o = omp[omp["qubits"] == nmax]
        fig, ax = plt.subplots(figsize=(5.2, 3.4))
        for grp, g in o.groupby("group"):
            g = g.groupby("threads")["gbps"].median().reset_index()
            ax.plot(g["threads"], g["gbps"], label=grp, **style(grp, groups))
        if not stream.empty:
            s = stream.groupby("threads")["triad_gbs"].max().reset_index()
            ax.plot(s["threads"], s["triad_gbs"], ls="--", lw=1.5, color=INK, marker="x", label="STREAM triad (roof)")
        log2_axis(ax)
        ax.set_xlabel("OpenMP threads")
        ax.set_ylabel("effective bandwidth [GB/s]")
        ax.set_title(f"bandwidth vs measured roof ({nmax} qubits)")
        ax.legend(fontsize=7)
        save(fig, outdir, "bandwidth_vs_threads_roof")

    # (3) roofline: CPU configurations at their largest size
    # single-process CPU runs only: the roof is one node's STREAM bandwidth
    cpu = agg[agg["backend"].isin(["serial", "openmp"]) & (agg["ranks"] == 1) & (agg["ai"] > 0)]
    if not cpu.empty:
        cfgs = cpu.groupby(["threads", "ranks"])["qubits"].max().reset_index().sort_values("threads")
        cfgs = cfgs[cfgs["threads"].isin([cfgs["threads"].min(), cfgs["threads"].max()])]
        fig, ax = plt.subplots(figsize=(5.6, 3.8))
        ai = np.geomspace(0.05, 50, 200)
        for i, (_, c) in enumerate(cfgs.iterrows()):
            t = int(c["threads"])
            pts = cpu[(cpu["threads"] == t) & (cpu["ranks"] == c["ranks"]) & (cpu["qubits"] == c["qubits"])]
            bw = stream_at(t)
            ls = "-" if i == 0 else "-."
            if np.isfinite(bw):
                roof = bw * ai
                if peak_gflops_core:
                    roof = np.minimum(roof, peak_gflops_core * t)
                ax.plot(ai, roof, color=INK if i == 0 else MUTED, lw=1.2, ls=ls, label=f"roof, {t} thread(s)")
            for grp, g in pts.groupby("group"):
                st = style(grp, groups)
                ax.scatter(g["ai"], g["gflops"], s=36, marker=st["marker"], color=st["color"],
                           edgecolors="white", linewidths=1,
                           label=grp if i == 0 else None, alpha=1.0 if i == len(cfgs) - 1 else 0.55)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("arithmetic intensity [flop/byte]")
        ax.set_ylabel("GFLOP/s")
        title = "roofline (memory roof = STREAM triad"
        title += f", compute roof = {peak_gflops_core:g} GFLOP/s/core)" if peak_gflops_core else "; no compute roof given)"
        ax.set_title(title, fontsize=8)
        ax.legend(fontsize=7, loc="upper left")
        if len(cfgs) > 1:
            ax.text(0.99, 0.02, f"solid points: {int(cfgs['threads'].max())} threads, faded: {int(cfgs['threads'].min())}",
                    transform=ax.transAxes, ha="right", fontsize=7, color=MUTED)
        save(fig, outdir, "roofline")
    return agg


def fig_sensitivity(runs: pd.DataFrame, analysis: pd.DataFrame, outdir: Path) -> pd.DataFrame:
    sens = runs[runs["experiment"] == "sensitivity"]
    if sens.empty or analysis.empty:
        return pd.DataFrame()
    a = analysis.drop_duplicates(["circuit_path", "ranks"])[
        ["circuit_path", "ranks", "local", "global_free", "comm", "pct_comm", "pct_touching_global", "comm_MiB_per_rank"]]
    m = sens.merge(a, left_on=["circuit", "ranks"], right_on=["circuit_path", "ranks"], how="left")
    s = (m.groupby(["kind", "qubits", "ranks", "global_fraction", "pct_comm", "pct_touching_global",
                    "local", "global_free", "comm", "git_commit"], dropna=False)
         .agg(t_median=("total_s", "median"), t_min=("total_s", "min"), t_max=("total_s", "max"),
              tpg_median=("time_per_gate_s", "median"), tpg_min=("time_per_gate_s", "min"),
              tpg_max=("time_per_gate_s", "max"), gates=("gates_run", "first"), n=("total_s", "size"))
         .reset_index())
    # Linear cost model per configuration: T = t_local * (local + global_free) + t_comm * comm
    for (n, p), g in s.groupby(["qubits", "ranks"]):
        X = np.c_[g["local"] + g["global_free"], g["comm"]].astype(float)
        if len(g) >= 2 and np.linalg.matrix_rank(X) == 2:
            (t_loc, t_comm), *_ = np.linalg.lstsq(X, g["t_median"].to_numpy(), rcond=None)
            s.loc[g.index, "fit_t_local_gate_s"] = t_loc
            s.loc[g.index, "fit_t_comm_gate_s"] = t_comm
            s.loc[g.index, "fit_comm_over_local"] = t_comm / t_loc if t_loc > 0 else np.nan

    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    gf = s[s["kind"] == "random_gf"].sort_values("pct_comm")
    if not gf.empty:
        errbar(ax, gf["pct_comm"], gf["tpg_median"], gf["tpg_min"], gf["tpg_max"],
               "random, swept global fraction", **style("random_gf", KIND_ORDER))
        for _, r in gf.iterrows():
            ax.annotate(f"F={r['global_fraction']:g}", (r["pct_comm"], r["tpg_median"]), xytext=(4, -10),
                        textcoords="offset points", fontsize=7, color=MUTED)
        if "fit_t_comm_gate_s" in gf and gf["fit_t_comm_gate_s"].notna().any():
            r0 = gf.iloc[0]
            x = np.linspace(0, max(100, s["pct_comm"].max()), 50) / 100
            ax.plot(x * 100, r0["fit_t_local_gate_s"] * (1 - x) + r0["fit_t_comm_gate_s"] * x, ls=":", lw=1.2,
                    color=MUTED, label=f"fit: comm gate = {r0['fit_comm_over_local']:.1f}x local gate")
    for kind in ["qft", "grover", "random"]:
        g = s[s["kind"] == kind]
        if not g.empty:
            ax.errorbar(g["pct_comm"], g["tpg_median"], yerr=[g["tpg_median"] - g["tpg_min"], g["tpg_max"] - g["tpg_median"]],
                        ls="none", capsize=2, ms=8, label=f"{kind} (benchmark)", **style(kind, KIND_ORDER))
    n, p = s["qubits"].iloc[0], s["ranks"].iloc[0]
    ax.set_xlabel(f"gates that need an exchange [% of all gates]  (analyze_circuit.py, P={p})")
    ax.set_ylabel("median time per gate [s]")
    ax.set_ylim(bottom=0)
    ax.set_title(f"MPI communication sensitivity, {n} qubits, {p} ranks")
    ax.legend(fontsize=7)
    save(fig, outdir, "sensitivity_comm_share")
    return s


# ------------------------------------------------------------------ main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default=None, help="output root (default: <results>)")
    ap.add_argument("--experiments", nargs="*", help="only these experiment names")
    ap.add_argument("--commit", help="only runs whose git commit starts with this")
    ap.add_argument("--clean-only", action="store_true", help="drop runs from a dirty working tree")
    ap.add_argument("--include-local", action="store_true", help="keep local-* test jobs (not for the report)")
    ap.add_argument("--peak-gflops", type=float, default=None,
                    help="FP64 peak per core [GFLOP/s] for the compute roof (cores x GHz x flops/cycle)")
    ap.add_argument("--max-gate-files", type=int, default=None, help="cap per-gate CSVs read (speed)")
    args = ap.parse_args(argv)

    setup_matplotlib()
    out = Path(args.out or args.results)
    figdir, sumdir = out / "figures", out / "summary"
    figdir.mkdir(parents=True, exist_ok=True); sumdir.mkdir(parents=True, exist_ok=True)

    runs = qr.load_runs(args.results, args.experiments, include_dirty=not args.clean_only,
                        include_local=args.include_local)
    if args.commit is not None and not runs.empty:
        runs = runs[runs["git_commit"].str.startswith(args.commit[:10])]
    if runs.empty:
        sys.exit(f"no runs found under {args.results} (local-* jobs need --include-local)")
    qr.warn_mixed_commits(runs)
    print(f"{len(runs)} runs, {runs['job_id'].nunique()} jobs, experiments: {sorted(runs['experiment'].unique())}")

    summary = qr.summarize(runs)
    summary.to_csv(sumdir / "runs_summary.csv", index=False)
    print(f"  table   {sumdir / 'runs_summary.csv'}")
    noisy = summary[summary["rel_range"] > 0.10]
    if not noisy.empty:
        warnings.warn(f"{len(noisy)} configurations have a repeat range > 10% of the median (see rel_range)")
    few = summary[summary["n"] < 3]
    if not few.empty:
        warnings.warn(f"{len(few)} configurations have fewer than 3 repeats")

    fig_time_per_gate(summary, figdir)

    strong = qr.add_speedup(summary[summary["experiment"].str.contains("strong")], "strong") \
        if summary["experiment"].str.contains("strong").any() else pd.DataFrame()
    weak = qr.add_speedup(summary[summary["experiment"].str.contains("weak")], "weak") \
        if summary["experiment"].str.contains("weak").any() else pd.DataFrame()
    if not strong.empty:
        strong.to_csv(sumdir / "scaling_strong.csv", index=False)
        amd = fig_strong(strong, figdir)
        amd.to_csv(sumdir / "amdahl.csv", index=False)
        print(f"  table   {sumdir / 'amdahl.csv'}")
    if not weak.empty:
        weak.to_csv(sumdir / "scaling_weak.csv", index=False)
        fig_weak(weak, figdir)

    gates = qr.load_gate_timings(runs, args.max_gate_files)
    bw = fig_bandwidth(gates, qr.load_stream(args.results), figdir, args.peak_gflops)
    if not bw.empty:
        bw.to_csv(sumdir / "gate_bandwidth.csv", index=False)

    sens = fig_sensitivity(runs, qr.load_circuit_analysis(args.results), figdir)
    if not sens.empty:
        sens.to_csv(sumdir / "sensitivity.csv", index=False)
        print(f"  table   {sumdir / 'sensitivity.csv'}")


if __name__ == "__main__":
    main()

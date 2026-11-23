"""Before/after table (and figure) for gate fusion, from a slurm/fusion.sbatch job.

    python3 analysis/fusion.py results/fusion/<jobid> [--no-plot]

Writes <job>/fusion.csv with one row per (kind, qubits, threads):
gates without/with fusion, median time without/with fusion (over repeats),
min-max ranges, and the speedup none/1q. Also writes <job>/fusion.png.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

KEYS = ["kind", "qubits", "threads"]


def fusion_table(runs: pd.DataFrame) -> pd.DataFrame:
    runs = runs[runs["status"] == "ok"].copy()
    if runs.empty:
        raise ValueError("no successful runs")
    agg = (runs.groupby([*KEYS, "fusion"])
               .agg(gates=("gates_run", "first"), median_s=("total_s", "median"),
                    min_s=("total_s", "min"), max_s=("total_s", "max"), repeats=("total_s", "size"))
               .reset_index())
    none = agg[agg["fusion"] == "none"].drop(columns="fusion")
    fused = agg[agg["fusion"] == "1q"].drop(columns="fusion")
    t = none.merge(fused, on=KEYS, suffixes=("_none", "_1q"))
    t["gate_reduction_pct"] = 100.0 * (1.0 - t["gates_1q"] / t["gates_none"])
    t["speedup"] = t["median_s_none"] / t["median_s_1q"]
    # conservative range: slowest fused run vs fastest unfused run and vice versa
    t["speedup_lo"] = t["min_s_none"] / t["max_s_1q"]
    t["speedup_hi"] = t["max_s_none"] / t["min_s_1q"]
    return t.sort_values(KEYS)


def plot(t: pd.DataFrame, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    threads = sorted(t["threads"].unique())
    fig, axes = plt.subplots(1, len(threads), figsize=(5 * len(threads), 3.6), squeeze=False)
    for ax, th in zip(axes[0], threads):
        sub = t[t["threads"] == th]
        for kind, g in sub.groupby("kind"):
            ax.errorbar(g["qubits"], g["speedup"],
                        yerr=[g["speedup"] - g["speedup_lo"], g["speedup_hi"] - g["speedup"]],
                        marker="o", capsize=3, label=kind)
        ax.axhline(1.0, color="grey", lw=0.8, ls="--")
        ax.set_title(f"{th} thread{'s' if th != 1 else ''}")
        ax.set_xlabel("qubits")
        ax.set_ylabel("speedup from 1q fusion (median)")
        ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("job", type=Path, help="results/fusion/<jobid>")
    p.add_argument("--no-plot", action="store_true")
    args = p.parse_args()

    t = fusion_table(pd.read_csv(args.job / "runs.csv"))
    t.to_csv(args.job / "fusion.csv", index=False, float_format="%.6g")
    cols = [*KEYS, "gates_none", "gates_1q", "gate_reduction_pct", "median_s_none", "median_s_1q", "speedup"]
    print(t[cols].to_string(index=False, float_format=lambda x: f"{x:.3g}"))
    if not args.no_plot:
        plot(t, args.job / "fusion.png")


if __name__ == "__main__":
    main()

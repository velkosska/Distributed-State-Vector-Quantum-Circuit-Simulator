"""Summarize matched LIKWID gate and STREAM runs and plot their CPU roofline."""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


KERNELS = {"h": "H", "rz": "RZ", "cx": "CX", "stream": "STREAM"}
VOLUME = "Memory data volume [GBytes]"


def read_profile(path: Path, kernel: str, cpus: set[int]) -> dict:
    lines = path.read_text().splitlines()
    summaries = [line for line in lines if line.startswith("PROFILE ")]
    if len(summaries) != 1:
        raise ValueError(f"{path}: expected one PROFILE summary")
    summary = dict(item.split("=", 1) for item in summaries[0].split()[1:])
    if summary["kernel"] != kernel or int(summary["threads"]) != len(cpus):
        raise ValueError(f"{path}: benchmark configuration differs from the manifest")

    table = {row[0].strip(): [item.strip() for item in row[1:] if item.strip()]
             for row in csv.reader(lines) if row}
    for label in ("Region Info", "Metric"):
        measured_cpus = {int(entry.split()[-1]) for entry in table.get(label, [])
                         if re.fullmatch(r"(?:HWThread|Core)\s+\d+", entry)}
        if measured_cpus != cpus:
            raise ValueError(f"{path}: LIKWID {label} reports CPUs {sorted(measured_cpus)}, expected {sorted(cpus)}")
    if table.get("call count") != ["1"] * len(cpus):
        raise ValueError(f"{path}: the marked region did not run once on each CPU")

    # The total-volume metric in MEM/MEM_DP uses memory-controller counts.
    # LIKWID attributes those counts to the controller's reporting CPU.
    volumes = [float(value) for value in table.get(VOLUME, [])]
    if len(volumes) != len(cpus) or any(not math.isfinite(v) or v < 0 for v in volumes):
        raise ValueError(f"{path}: expected valid per-CPU {VOLUME} values")
    dram_bytes = sum(volumes) * 1e9
    seconds, model_bytes, model_flops = (float(summary[key]) for key in ("seconds", "model_bytes", "model_flops"))
    if any(not math.isfinite(v) or v <= 0 for v in (seconds, dram_bytes, model_bytes, model_flops)):
        raise ValueError(f"{path}: invalid timing, FLOP model, or memory counter volume")
    return {
        "passes": int(summary["passes"]),
        "seconds": seconds,
        "model_bytes": model_bytes,
        "model_flops": model_flops,
        "dram_bytes": dram_bytes,
        "effective_gbs": model_bytes / seconds / 1e9,
        "dram_gbs": dram_bytes / seconds / 1e9,
        "model_gflops": model_flops / seconds / 1e9,
        "model_flop_per_byte": model_flops / model_bytes,
        "flop_per_dram_byte": model_flops / dram_bytes,
    }


def load_runs(job: Path) -> pd.DataFrame:
    with (job / "profile_manifest.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        cpus = set(map(int, row["cpu_ids"].split(",")))
        for key in ("threads", "repeat"):
            row[key] = int(row[key])
        if len(cpus) != row["threads"]:
            raise ValueError("expected one distinct sampled CPU per worker")
        row.update(read_profile(job / row["likwid_file"], row["kernel"], cpus))
    return pd.DataFrame(rows)


def plot_roofline(gates: pd.DataFrame, job: Path, peak_per_core: float | None) -> None:
    panels = list(gates.groupby(["threads", "omp_proc_bind"], sort=True))
    fig, axes = plt.subplots((len(panels) + 1) // 2, 2, figsize=(9.4, 3.9 * ((len(panels) + 1) // 2)),
                             sharex=True, sharey=True, squeeze=False, layout="constrained")
    low = gates["flop_per_dram_byte"].min() / 2
    high = gates["flop_per_dram_byte"].max() * 2.5
    if peak_per_core:
        high = max(high, 2 * (gates["threads"] * peak_per_core / gates["stream_dram_gbs"]).max())
    colors = {"h": "#135c95", "rz": "#b34618", "cx": "#227c62"}
    offsets = {"h": (7, -13), "rz": (-19, -13), "cx": (7, 6)}
    for ax, ((threads, bind), data) in zip(axes.flat, panels):
        bandwidth = data["stream_dram_gbs"].iloc[0]
        ridge = threads * peak_per_core / bandwidth if peak_per_core else high
        xs = sorted({low, high, min(high, max(low, ridge))})
        roof = [bandwidth * min(x, ridge) for x in xs]
        ax.loglog(xs, roof, "--", color="#303b48", label=f"STREAM triad · {bandwidth:.1f} GB/s")
        for row in data.itertuples():
            x, y = row.flop_per_dram_byte, row.model_gflops
            ax.scatter(x, y, color=colors[row.kernel], s=46, zorder=3)
            ax.annotate(KERNELS[row.kernel], (x, y), xytext=offsets[row.kernel],
                        textcoords="offset points", color=colors[row.kernel])
        ax.set_title(f"{threads} threads · {bind}")
        ax.grid(which="major", alpha=0.23)
        ax.legend(loc="upper left", frameon=False, fontsize=8)
    for ax in list(axes.flat)[len(panels):]:
        ax.set_visible(False)
    fig.suptitle("CPU roofline · state-vector gate kernels")
    fig.supxlabel("Modeled FLOP per measured DRAM byte")
    fig.supylabel("Modeled throughput (GFLOP/s)")
    for suffix in ("pdf", "png"):
        fig.savefig(job / f"roofline.{suffix}", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job", type=Path)
    parser.add_argument("--peak-gflops-per-core", type=float, help="optional FP64 compute ceiling")
    args = parser.parse_args()
    peak = args.peak_gflops_per_core
    if peak is not None and (not math.isfinite(peak) or peak <= 0):
        parser.error("FP64 peak must be positive")
    try:
        runs = load_runs(args.job)
        if runs.empty:
            raise ValueError("the manifest contains no profiler runs")
        matched = ["omp_proc_bind", "threads", "repeat"]
        for _, group in runs.groupby(matched):
            if sorted(group["kernel"]) != sorted(KERNELS) or group["cpu_ids"].nunique() != 1:
                raise ValueError("each repeat needs H, RZ, CX and STREAM on the same CPUs")
    except (OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))

    config = ["omp_proc_bind", "threads"]
    stream = runs[runs["kernel"] == "stream"].groupby(config + ["array_mib"], as_index=False).agg(
        repeats=("repeat", "size"), stream_dram_gbs=("dram_gbs", "median"),
        stream_triad_gbs=("effective_gbs", "median"),
    )
    gates = runs[runs["kernel"] != "stream"].groupby(config + ["kernel", "qubits"], as_index=False).agg(
        repeats=("repeat", "size"), dram_gbs=("dram_gbs", "median"),
        effective_gbs=("effective_gbs", "median"), model_gflops=("model_gflops", "median"),
        model_flop_per_byte=("model_flop_per_byte", "median"),
    )
    # Compute the plotted intensity from the same median rates used on the y-axis
    # so the point's slope equals the reported median DRAM bandwidth.
    gates["flop_per_dram_byte"] = gates["model_gflops"] / gates["dram_gbs"]
    gates = gates.merge(stream.drop(columns=["repeats", "array_mib"]), on=config, validate="many_to_one")
    gates["dram_fraction_of_triad"] = gates["dram_gbs"] / gates["stream_dram_gbs"]
    gates["effective_fraction_of_triad"] = gates["effective_gbs"] / gates["stream_triad_gbs"]
    runs.to_csv(args.job / "profile_runs.csv", index=False)
    stream.to_csv(args.job / "stream.csv", index=False)
    gates.to_csv(args.job / "gate_bandwidth.csv", index=False)
    plot_roofline(gates, args.job, peak)


if __name__ == "__main__":
    main()

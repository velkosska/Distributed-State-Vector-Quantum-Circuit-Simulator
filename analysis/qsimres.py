"""Load and summarise qsim benchmark results written by slurm/lib/run_harness.sh.

Everything the plotting scripts need lives here so that numbers in tables and
figures come from one place:

  load_runs()          every results/<experiment>/<jobid>/runs.csv, plus meta.txt fields
  summarize()          median / min / max / range over repeats
  add_speedup()        speedup and parallel efficiency (strong and weak)
  amdahl_fit()         serial fraction f from 1/S = f + (1-f)/p
  karp_flatt()         experimentally determined serial fraction per p
  load_gate_timings()  per-gate CSVs (raw/*.gates.csv) with a flop model
  load_stream()        measured memory bandwidth (results/stream/*/stream.csv)
  load_circuit_analysis()  analyze_circuit.py tables from the sensitivity job
"""
from __future__ import annotations

import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

AMP_BYTES = 16  # complex<double>

# Columns that identify one configuration; repeats differ only in `repeat`
# (and job_id when the same configuration was run by several jobs).
CONFIG_COLS = ["experiment", "backend", "kind", "qubits", "nodes", "ranks", "threads", "gpus",
               "fusion", "depth", "iterations", "global_fraction", "global_qubits", "tag", "git_commit"]

META_FIELDS = {"git_commit": "git_commit", "git_dirty": "git_dirty", "batch_host": "host",
               "Model name": "cpu_model", "num_nodes": "job_nodes", "harness_version": "harness_version"}


# ------------------------------------------------------------------ loading
def parse_meta(path: Path) -> dict:
    out = {}
    if not path.exists():
        return out
    for line in path.read_text(errors="replace").splitlines():
        m = re.match(r"^\s*([A-Za-z_ ()]+?):\s*(.*)$", line)
        if m and m.group(1) in META_FIELDS and META_FIELDS[m.group(1)] not in out:
            out[META_FIELDS[m.group(1)]] = m.group(2).strip()
    return out


def load_runs(results: str | Path = "results", experiments: list[str] | None = None,
              include_failed: bool = False, include_dirty: bool = True,
              include_local: bool = False) -> pd.DataFrame:
    """All runs.csv rows under `results`, one row per qsim run.

    Adds: git_commit, git_dirty, host, cpu_model (from meta.txt), workers
    (= ranks * threads, or gpus for CUDA), time_per_gate_s, result_dir.
    Local test jobs (job_id local-*) are skipped unless include_local.
    """
    results = Path(results)
    frames = []
    for csv in sorted(results.glob("*/*/runs.csv")):
        df = pd.read_csv(csv, dtype={"job_id": str, "tag": str, "fusion": str, "status": str})
        if df.empty:
            continue
        meta = parse_meta(csv.parent / "meta.txt")
        for k, v in meta.items():
            df[k] = v
        df["result_dir"] = str(csv.parent)
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    for c in ["git_commit", "git_dirty", "host", "cpu_model"]:
        if c not in df:
            df[c] = "unknown"
    if experiments:
        df = df[df["experiment"].isin(experiments)]
    if not include_local:
        df = df[~df["job_id"].astype(str).str.startswith("local-")]
    if not include_failed:
        df = df[df["status"] == "ok"]
    if not include_dirty:
        df = df[df["git_dirty"] != "yes"]
    df = df.copy()
    for c in ["ranks", "threads", "gpus", "nodes", "qubits", "repeat", "gates_run", "gates_in"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
    for c in ["total_s", "mean_gate_s", "norm_err", "global_fraction", "depth", "iterations", "global_qubits"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["ranks"] = df["ranks"].clip(lower=1)
    df["threads"] = df["threads"].clip(lower=1)
    df["workers"] = np.where(df["backend"] == "cuda", df["gpus"].clip(lower=1), df["ranks"] * df["threads"])
    df["time_per_gate_s"] = df["total_s"] / df["gates_run"].replace(0, np.nan)
    df["git_commit"] = df["git_commit"].astype(str).str[:10]
    for c in ["tag", "fusion"]:
        df[c] = df[c].fillna("")
    return df


def warn_mixed_commits(df: pd.DataFrame) -> None:
    for exp, g in df.groupby("experiment"):
        commits = sorted(g["git_commit"].unique())
        if len(commits) > 1:
            warnings.warn(f"{exp}: runs from {len(commits)} commits {commits}; they are kept apart "
                          "(git_commit is part of the configuration key). Filter with --commit.")
        if (g["git_dirty"] == "yes").any():
            warnings.warn(f"{exp}: some runs come from a dirty tree (see raw/git_diff.patch); "
                          "use --clean-only for final figures.")


# ------------------------------------------------------------------ statistics
def summarize(df: pd.DataFrame, value: str = "total_s", by: list[str] | None = None) -> pd.DataFrame:
    """Median, min, max, range and count of `value` over repeats of each configuration.

    `range` is max - min; `rel_range` = range / median (the spread we report as
    error bars). Also gives the median time per gate.
    """
    by = by or CONFIG_COLS
    by = [c for c in by if c in df.columns]
    g = df.groupby(by, dropna=False)
    out = g[value].agg(median="median", min="min", max="max", n="count").reset_index()
    out["range"] = out["max"] - out["min"]
    out["rel_range"] = out["range"] / out["median"]
    tpg = g["time_per_gate_s"].agg(tpg_median="median", tpg_min="min", tpg_max="max").reset_index()
    out = out.merge(tpg, on=by, how="left")
    gates = g["gates_run"].first().reset_index()
    out = out.merge(gates, on=by, how="left")
    out["workers"] = np.where(out["backend"] == "cuda", out["gpus"].clip(lower=1), out["ranks"] * out["threads"])
    return out


def scaling_family_cols(summary: pd.DataFrame) -> list[str]:
    """Columns that stay fixed along one scaling curve (workers and, for weak scaling, qubits vary)."""
    return [c for c in ["experiment", "backend", "kind", "fusion", "depth", "iterations", "tag", "git_commit"]
            if c in summary.columns]


def add_speedup(summary: pd.DataFrame, mode: str) -> pd.DataFrame:
    """Speedup and efficiency relative to the smallest worker count p0 of each curve.

    strong: S(p) = T(p0) / T(p) * p0  (= T(1)/T(p) when p0 = 1; otherwise assumes
            ideal scaling from 1 to p0, stated in the table as `p0`)
            E(p) = S(p) / p
    weak:   problem grows with p, so compare *time per gate* (the number of gates
            also grows with n):  E(p) = t_gate(p0) / t_gate(p);  S(p) = E(p) * p / p0 * p0
    Lower / upper bounds use the repeat min/max so the range carries through.
    """
    fam = scaling_family_cols(summary)
    if mode == "strong":
        fam = fam + ["qubits"]
    rows = []
    for _, g in summary.groupby(fam, dropna=False):
        g = g.sort_values("workers").copy()
        ref = g.iloc[0]
        p0 = ref["workers"]
        if mode == "strong":
            g["speedup"] = ref["median"] / g["median"] * p0
            g["speedup_lo"] = ref["min"] / g["max"] * p0
            g["speedup_hi"] = ref["max"] / g["min"] * p0
        else:
            e = ref["tpg_median"] / g["tpg_median"]
            g["speedup"] = e * g["workers"]
            g["speedup_lo"] = ref["tpg_min"] / g["tpg_max"] * g["workers"]
            g["speedup_hi"] = ref["tpg_max"] / g["tpg_min"] * g["workers"]
        g["efficiency"] = g["speedup"] / g["workers"]
        g["efficiency_lo"] = g["speedup_lo"] / g["workers"]
        g["efficiency_hi"] = g["speedup_hi"] / g["workers"]
        g["p0"] = p0
        g["karp_flatt"] = karp_flatt(g["speedup"].to_numpy(), g["workers"].to_numpy())
        rows.append(g)
    return pd.concat(rows, ignore_index=True) if rows else summary.assign(speedup=np.nan)


def amdahl_fit(p, speedup) -> tuple[float, float]:
    """Least-squares serial fraction f in S(p) = 1 / (f + (1 - f) / p).

    Linear in f after writing y = 1/S, x = 1/p:  y - x = f (1 - x).
    Points with p = 1 carry no information and are ignored. Returns (f, rmse of S).
    f is clipped to [0, 1].
    """
    p = np.asarray(p, float); s = np.asarray(speedup, float)
    m = (p > 1) & np.isfinite(s) & (s > 0)
    if m.sum() < 1:
        return float("nan"), float("nan")
    x, y = 1.0 / p[m], 1.0 / s[m]
    f = float(np.clip(np.sum((1 - x) * (y - x)) / np.sum((1 - x) ** 2), 0.0, 1.0))
    pred = 1.0 / (f + (1 - f) / p[m])
    return f, float(np.sqrt(np.mean((pred - s[m]) ** 2)))


def amdahl_speedup(p, f):
    p = np.asarray(p, float)
    return 1.0 / (f + (1 - f) / p)


def karp_flatt(speedup, p):
    """e(p) = (1/S - 1/p) / (1 - 1/p); NaN at p = 1. Rising e(p) means overhead
    grows with p (communication, bandwidth saturation), not a fixed serial part."""
    s = np.asarray(speedup, float); p = np.asarray(p, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        e = (1 / s - 1 / p) / (1 - 1 / p)
    return np.where(p > 1, e, np.nan)


# ------------------------------------------------------------------ per-gate data
# flops per amplitude of the full state vector, matching the kernels in
# src/qsim/kernels.hpp (complex mul = 6 flops, complex add = 2 flops):
#   dense 2x2 on a pair: 4 mul + 2 add = 28 flops per 2 amplitudes
#   diagonal phase: 1 mul (6 flops) per touched amplitude
FLOPS_PER_AMP = {
    "h": 14, "x": 14, "y": 14, "rx": 14, "ry": 14, "u": 14,      # every amplitude
    "rz": 6,                                                    # diag, d0 != 1: all amplitudes
    "z": 3, "s": 3, "sdg": 3, "t": 3, "tdg": 3, "p": 3,          # |1> half only
    "cx": 7, "cy": 7,                                           # dense on the control=1 half
    "cz": 1.5, "cp": 1.5,                                       # |11> quarter
    "swap": 0,
}


def load_gate_timings(runs: pd.DataFrame, max_files: int | None = None) -> pd.DataFrame:
    """Concatenate the per-gate CSVs of the given runs, with run context columns.

    Adds flops (model above), gflops and arithmetic intensity (flops/byte).
    """
    frames = []
    sel = runs[runs["timing_csv"].notna() & (runs["timing_csv"].astype(str) != "")]
    if max_files:
        sel = sel.head(max_files)
    keep = ["experiment", "backend", "kind", "qubits", "ranks", "threads", "gpus", "repeat", "job_id", "git_commit"]
    for _, r in sel.iterrows():
        path = Path(r["result_dir"]) / r["timing_csv"]
        if not path.exists():
            continue
        g = pd.read_csv(path)
        for k in keep:
            g[k] = r[k]
        frames.append(g)
    if not frames:
        return pd.DataFrame()
    g = pd.concat(frames, ignore_index=True)
    g["gate"] = g["gate"].str.lower()
    n_local = g["qubits"] - np.log2(g["ranks"].clip(lower=1)).round().astype(int)
    g["flops"] = g["gate"].map(FLOPS_PER_AMP).fillna(14) * np.power(2.0, n_local)
    g["gflops"] = np.where(g["seconds"] > 0, g["flops"] / g["seconds"] * 1e-9, np.nan)
    g["ai"] = np.where(g["bytes"] > 0, g["flops"] / g["bytes"], np.nan)
    return g


def load_stream(results: str | Path = "results") -> pd.DataFrame:
    frames = [pd.read_csv(p, dtype={"job_id": str}) for p in sorted(Path(results).glob("stream/*/stream.csv"))]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_circuit_analysis(results: str | Path = "results") -> pd.DataFrame:
    frames = []
    for p in sorted(Path(results).glob("*/*/circuit_analysis.csv")):
        d = pd.read_csv(p)
        d["result_dir"] = str(p.parent)
        frames.append(d)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def state_bytes(n: int, ranks: int = 1) -> float:
    """Bytes of the state vector per rank."""
    return AMP_BYTES * 2.0 ** n / ranks

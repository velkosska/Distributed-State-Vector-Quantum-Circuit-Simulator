"""Tests for the analysis code on synthetic results with known answers.

  python3 -m pytest analysis/ -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qsimres as qr  # noqa: E402
import make_figures  # noqa: E402

COLS = ("run_id,experiment,job_id,backend,kind,qubits,nodes,ranks,threads,gpus,repeat,seed,depth,iterations,"
        "global_fraction,global_qubits,gates_in,gates_run,fusion,total_s,mean_gate_s,norm_err,status,tag,"
        "circuit,timing_csv").split(",")


def fake_job(root: Path, experiment: str, rows: list[dict], job="123", commit="abc1234", dirty="no"):
    d = root / experiment / job
    (d / "raw").mkdir(parents=True)
    (d / "meta.txt").write_text(f"experiment: {experiment}\njob_id: {job}\ngit_commit:      {commit}\n"
                                f"git_dirty:       {dirty}\n  Model name:   Fake CPU\n")
    full = []
    for i, r in enumerate(rows):
        base = {c: "" for c in COLS}
        base.update(run_id=f"r{i:04d}", experiment=experiment, job_id=job, backend="mpi", kind="qft",
                    nodes=1, ranks=1, threads=1, gpus=0, repeat=1, seed=1, gates_in=100, gates_run=100,
                    fusion="none", norm_err=1e-15, status="ok", circuit="c.txt")
        base.update(r)
        base["mean_gate_s"] = float(base["total_s"]) / int(base["gates_run"])
        full.append(base)
    pd.DataFrame(full, columns=COLS).to_csv(d / "runs.csv", index=False)


def test_amdahl_recovers_serial_fraction():
    p = np.array([1, 2, 4, 8, 16, 32, 64])
    for f in (0.0, 0.01, 0.05, 0.3):
        est, rmse = qr.amdahl_fit(p, qr.amdahl_speedup(p, f))
        assert est == pytest.approx(f, abs=1e-12)
        assert rmse == pytest.approx(0, abs=1e-9)


def test_karp_flatt_constant_for_amdahl():
    p = np.array([2, 4, 8, 16])
    e = qr.karp_flatt(qr.amdahl_speedup(p, 0.07), p)
    assert np.allclose(e, 0.07)


def test_strong_scaling_median_speedup_and_fit(tmp_path):
    f, t1 = 0.05, 100.0
    rows = []
    for p in (1, 2, 4, 8, 16):
        for rep, noise in enumerate((0.98, 1.0, 1.03), start=1):  # median = noise-free value
            rows.append(dict(qubits=30, ranks=p, repeat=rep, total_s=t1 * (f + (1 - f) / p) * noise))
    fake_job(tmp_path, "mpi_strong", rows)
    runs = qr.load_runs(tmp_path)
    assert len(runs) == 15
    s = qr.add_speedup(qr.summarize(runs), "strong").sort_values("workers")
    assert s["n"].tolist() == [3] * 5
    assert s["speedup"].to_numpy() == pytest.approx(qr.amdahl_speedup(s["workers"], f))
    assert (s["speedup_lo"] <= s["speedup"]).all() and (s["speedup"] <= s["speedup_hi"]).all()
    est, _ = qr.amdahl_fit(s["workers"], s["speedup"])
    assert est == pytest.approx(f, abs=1e-9)


def test_weak_scaling_uses_time_per_gate(tmp_path):
    rows = []
    for p, n in ((1, 28), (2, 29), (4, 30)):
        gates = 10 * n  # the gate count grows with n; time per gate stays constant
        rows.append(dict(qubits=n, ranks=p, gates_run=gates, gates_in=gates, total_s=0.5 * gates))
    fake_job(tmp_path, "mpi_weak", rows)
    s = qr.add_speedup(qr.summarize(qr.load_runs(tmp_path)), "weak")
    assert s["efficiency"].to_numpy() == pytest.approx(1.0)


def test_filters_and_commit_key(tmp_path):
    fake_job(tmp_path, "baseline", [dict(qubits=20, total_s=1.0)], job="1", commit="aaaaaaa")
    fake_job(tmp_path, "baseline", [dict(qubits=20, total_s=2.0)], job="2", commit="bbbbbbb", dirty="yes")
    fake_job(tmp_path, "baseline", [dict(qubits=20, total_s=9.0, status="fail:1")], job="3")
    fake_job(tmp_path, "baseline", [dict(qubits=20, total_s=9.0)], job="local-x")
    runs = qr.load_runs(tmp_path)
    assert sorted(runs["job_id"]) == ["1", "2"]
    assert len(qr.summarize(runs)) == 2  # different commits are different configurations
    assert len(qr.load_runs(tmp_path, include_dirty=False)) == 1


def test_make_figures_end_to_end(tmp_path):
    rows = [dict(qubits=n, ranks=p, repeat=r, total_s=0.01 * 2 ** (n - 20) / p * (1 + 0.01 * r))
            for n in (20, 22) for p in (1, 2, 4) for r in (1, 2, 3)]
    fake_job(tmp_path, "mpi_strong", rows)
    make_figures.main(["--results", str(tmp_path)])
    assert (tmp_path / "figures" / "strong_mpi_strong.pdf").exists()
    amd = pd.read_csv(tmp_path / "summary" / "amdahl.csv")
    assert (amd["serial_fraction"] < 0.02).all()

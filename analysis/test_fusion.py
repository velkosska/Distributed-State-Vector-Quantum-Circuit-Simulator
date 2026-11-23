"""Tests for analysis/fusion.py (run with: python3 -m pytest analysis -q)."""
import pandas as pd
import pytest

from fusion import fusion_table


def _runs():
    rows = []
    for fusion, gates, times in (("none", 300, [2.0, 2.2, 2.1]), ("1q", 150, [1.0, 1.1, 0.9])):
        for r, t in enumerate(times, 1):
            rows.append(dict(kind="random", qubits=24, threads=1, fusion=fusion, gates_run=gates,
                             total_s=t, repeat=r, status="ok"))
    rows.append(dict(kind="random", qubits=24, threads=1, fusion="1q", gates_run=150,
                     total_s=99.0, repeat=4, status="fail:1"))  # failed runs are ignored
    return pd.DataFrame(rows)


def test_fusion_table_medians_and_speedup():
    t = fusion_table(_runs())
    assert len(t) == 1
    row = t.iloc[0]
    assert row["gates_none"] == 300 and row["gates_1q"] == 150
    assert row["gate_reduction_pct"] == pytest.approx(50.0)
    assert row["median_s_none"] == pytest.approx(2.1)
    assert row["median_s_1q"] == pytest.approx(1.0)
    assert row["speedup"] == pytest.approx(2.1)
    assert row["speedup_lo"] == pytest.approx(2.0 / 1.1)
    assert row["speedup_hi"] == pytest.approx(2.2 / 0.9)


def test_fusion_table_needs_successful_runs():
    runs = _runs()
    runs["status"] = "fail:1"
    with pytest.raises(ValueError):
        fusion_table(runs)

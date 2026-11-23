"""Charts and diagrams for the qsim deck (graphite + one amber accent), drawn from the real CSVs.
Run: python charts.py <repo results dir> <out dir>"""
import csv, os, sys, statistics as st
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import FancyBboxPatch, Circle

R, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
for f in ["/System/Library/Fonts/Avenir Next.ttc", "/System/Library/Fonts/Menlo.ttc"]:
    try: fm.fontManager.addfont(f)
    except Exception: pass

SURF, GRID = "#F4F4F5", "#E4E4E7"
TXT, SEC, MUTED, FAINT = "#18181B", "#52525B", "#71717A", "#A1A1AA"
AMBER = "#C2410C"
plt.rcParams.update({
    "font.family": ["Avenir Next", "Helvetica Neue", "DejaVu Sans"], "font.size": 14,
    "text.color": SEC, "axes.labelcolor": SEC, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.edgecolor": GRID, "axes.facecolor": "none", "figure.facecolor": "none",
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
    "legend.frameon": False, "legend.labelcolor": SEC, "lines.linewidth": 2.6,
    "lines.markersize": 7, "xtick.major.size": 0, "ytick.major.size": 0,
})
MONO = {"family": ["Menlo", "DejaVu Sans Mono"], "size": 12}


def mono_ticks(ax):
    for lab in ax.get_xticklabels() + ax.get_yticklabels():
        lab.set_fontfamily(MONO["family"]); lab.set_fontsize(MONO["size"])


def rows(p): return list(csv.DictReader(open(os.path.join(R, p))))


def save(fig, ax, name):
    if ax is not None: mono_ticks(ax)
    fig.savefig(os.path.join(OUT, name), dpi=200, transparent=True, bbox_inches="tight")
    plt.close(fig)


def label_end(ax, x, y, t, c, dy=0):
    ax.annotate(t, (x, y), xytext=(8, dy), textcoords="offset points", color=c, fontsize=13, va="center")


# 1 memory wall
fig, ax = plt.subplots(figsize=(7.6, 4.6))
q = list(range(20, 41)); gib = [16 * 2 ** n / 2 ** 30 for n in q]
ax.plot(q, gib, color=AMBER)
ax.set_yscale("log", base=2)
for y, lab in [(16, "laptop 16 GB"), (96, "one GPU 96 GB"), (503, "one CPU node 503 GiB")]:
    ax.axhline(y, color=FAINT, lw=1.2)
    ax.text(40.3, y, lab, color=SEC, fontsize=12, va="center")
for n in (28, 33):
    ax.plot(n, 16 * 2 ** n / 2 ** 30, "o", color=AMBER, ms=8)
    ax.annotate(f"{n} q  {16*2**n/2**30:g} GiB", (n, 16 * 2 ** n / 2 ** 30), xytext=(-12, 12),
                textcoords="offset points", ha="right", color=TXT, fontsize=12, fontfamily=MONO["family"])
ax.set_xlabel("Qubits"); ax.set_ylabel("State vector, GiB")
ax.set_yticks([2 ** k for k in range(-6, 15, 4)]); ax.set_yticklabels([f"{2**k:g}" for k in range(-6, 15, 4)])
save(fig, ax, "c_memory.png")

# 2 validation
fig, ax = plt.subplots(figsize=(7.6, 4.4))
styles = [("36497", "serial", AMBER, 2.6), ("36498", "OpenMP", SEC, 1.6), ("36499", "MPI, 4 ranks", MUTED, 1.6)]
for job, lab, c, lw in styles:
    d = defaultdict(list)
    for r in rows(f"validation/{job}/validation.csv"):
        d[int(r["qubits"])].append(max(float(r["max_abs_err"]), 1e-18))
    xs = sorted(d); ax.plot(xs, [max(d[x]) for x in xs], color=c, lw=lw, label=lab)
ax.axhline(1e-10, color=TXT, ls="--", lw=1.2)
ax.text(20.3, 1e-10, "tolerance 1e-10", color=TXT, fontsize=12, va="center")
ax.set_yscale("log"); ax.set_ylim(1e-18, 1e-6); ax.set_xlim(2, 20)
ax.set_xlabel("Qubits"); ax.set_ylabel("Largest error vs Qiskit"); ax.legend(loc="upper left", ncol=3, fontsize=12)
save(fig, ax, "c_validation.png")

# 3 serial
fig, ax = plt.subplots(figsize=(7.6, 4.6))
d = defaultdict(list)
for r in rows("baseline/36504/runs.csv"):
    d[(r["kind"], int(r["qubits"]))].append(float(r["mean_gate_s"]) * 1e3)
for k, c, lw in [("random", MUTED, 1.6), ("grover", SEC, 1.6), ("qft", AMBER, 2.8)]:
    xs = sorted(n for kk, n in d if kk == k); ys = [st.median(d[(k, x)]) for x in xs]
    ax.plot(xs, ys, color=c, lw=lw, marker="o" if k == "qft" else None)
    label_end(ax, xs[-1], ys[-1], {"qft": "QFT", "grover": "Grover", "random": "random"}[k], c,
              dy={"qft": -10, "grover": 4, "random": 16}[k])
ax.set_yscale("log"); ax.set_xlabel("Qubits"); ax.set_ylabel("Time per gate, ms"); ax.set_xlim(19.5, 29.6)
save(fig, ax, "c_serial.png")

# 4 stream
fig, ax = plt.subplots(figsize=(5.0, 3.0))
s = rows("stream/36503/stream.csv")
t = [int(r["threads"]) for r in s]; b = [float(r["triad_gbs"]) for r in s]
ax.plot(t, b, color=SEC, marker="o", lw=2); ax.set_xscale("log", base=2)
ax.set_xticks(t); ax.set_xticklabels(t); ax.set_ylim(0, 100)
for x, y in zip(t, b): ax.text(x, y + 6, f"{y:.0f}", ha="center", fontsize=11, color=TXT, fontfamily=MONO["family"])
ax.set_xlabel("Threads"); ax.set_ylabel("STREAM triad, GB/s")
save(fig, ax, "c_stream.png")

# 5 OpenMP strong
fig, ax = plt.subplots(figsize=(7.6, 4.6))
sp = rows("openmp_strong/36512/speedup.csv")
for (k, bnd), c, ls, lw in [(("random", "close"), FAINT, "--", 1.4), (("random", "spread"), MUTED, "-", 1.6),
                            (("qft", "close"), SEC, "--", 1.6), (("qft", "spread"), AMBER, "-", 2.8)]:
    rr = [r for r in sp if r["kind"] == k and r["omp_proc_bind"] == bnd]
    x = [int(r["threads"]) for r in rr]; y = [float(r["speedup"]) for r in rr]
    ax.plot(x, y, color=c, ls=ls, lw=lw, marker="o" if (k, bnd) == ("qft", "spread") else None)
    label_end(ax, x[-1], y[-1], f"{'QFT' if k=='qft' else 'random'}, {bnd}", c,
              dy={("qft", "spread"): 14, ("qft", "close"): 0, ("random", "spread"): -2, ("random", "close"): -16}[(k, bnd)])
ax.plot([1, 32], [1, 32], color=FAINT, ls=":", lw=1.4); ax.text(15, 19, "ideal", color=MUTED, fontsize=12)
ax.set_xscale("log", base=2); ax.set_yscale("log", base=2)
for f_ in (ax.set_xticks, ax.set_yticks): f_([1, 2, 4, 8, 16, 32])
ax.set_xticklabels([1, 2, 4, 8, 16, 32]); ax.set_yticklabels([1, 2, 4, 8, 16, 32])
ax.set_xlabel("Threads"); ax.set_ylabel("Speedup over 1 thread"); ax.set_xlim(0.9, 75)
save(fig, ax, "c_omp_strong.png")

# 6 OpenMP weak
fig, ax = plt.subplots(figsize=(7.6, 4.4))
d = defaultdict(list)
for r in rows("openmp_weak/36513/runs.csv"):
    d[(r["kind"], int(r["threads"]))].append(float(r["mean_gate_s"]))
ts = [1, 2, 4, 8, 16, 32]; xi = np.arange(len(ts)); w = 0.36
for off, k, c in [(-w / 2, "qft", AMBER), (w / 2, "random", FAINT)]:
    base = st.median(d[(k, 1)])
    eff = [100 * base / st.median(d[(k, t_)]) for t_ in ts]
    ax.bar(xi + off, eff, w, color=c, label="QFT" if k == "qft" else "random")
    if k == "qft":
        for xb, e in zip(xi + off, eff):
            ax.text(xb, e + 3, f"{e:.0f}", ha="center", fontsize=11, color=TXT, fontfamily=MONO["family"])
ax.axhline(100, color=MUTED, ls=":", lw=1.2)
ax.set_xticks(xi); ax.set_xticklabels([f"{t_} t\n{24+i} q" for i, t_ in enumerate(ts)])
ax.set_ylabel("Weak-scaling efficiency, %"); ax.set_ylim(0, 112); ax.legend(loc="upper right", fontsize=12)
ax.grid(axis="x", visible=False)
save(fig, ax, "c_omp_weak.png")

# 7 MPI strong
fig, ax = plt.subplots(figsize=(7.6, 4.6))
def mpi(kind):
    d = defaultdict(list)
    for f in ["mpi_strong/36515/runs.csv", "mpi_strong/36586/runs.csv", "mpi_strong/36593/runs.csv"]:
        if os.path.exists(os.path.join(R, f)):
            for r in rows(f):
                if r["kind"] == kind and r["status"] == "ok":
                    d[int(r["ranks"])].append(float(r["total_s"]))
    P_ = sorted(d); t1 = min(d[1])
    return P_, [t1 / min(d[p]) for p in P_], [t1 / max(d[p]) for p in P_]
ax.axvspan(22.6, 45, color=MUTED, alpha=0.15, lw=0)
ax.text(24, 17, "two nodes", color=SEC, fontsize=12)
for kind, c, lw, lab in [("random", SEC, 1.8, "random"), ("qft", AMBER, 2.8, "QFT")]:
    P_, best, lo = mpi(kind)
    ax.fill_between(P_, lo, best, color=c, alpha=0.12, lw=0)
    ax.plot(P_, best, color=c, marker="o", lw=lw); label_end(ax, P_[-1], best[-1], lab, c)
ax.plot([1, 32], [1, 32], color=FAINT, ls=":", lw=1.4); ax.text(9, 12, "ideal", color=MUTED, fontsize=12)
ax.set_xscale("log", base=2); ax.set_yscale("log", base=2)
ax.set_xticks([1, 2, 4, 8, 16, 32]); ax.set_xticklabels([1, 2, 4, 8, 16, 32])
ax.set_yticks([0.25, 0.5, 1, 2, 4, 8, 16]); ax.set_yticklabels(["0.25", "0.5", 1, 2, 4, 8, 16])
ax.set_xlim(0.9, 45); ax.set_ylim(0.2, 24)
ax.set_xlabel("MPI ranks, 2 threads each"); ax.set_ylabel("Speedup, 30 qubits, best of 3")
save(fig, ax, "c_mpi_strong.png")

# 7a MPI weak (teammate jobs 36584 + 36745)
fig, ax = plt.subplots(figsize=(7.6, 4.6))
d = defaultdict(list)
for f in ["mpi_weak/36584/runs.csv", "mpi_weak/36745/runs.csv"]:
    for r in rows(f):
        if r["status"] == "ok":
            d[(r["kind"], int(r["ranks"]))].append(float(r["mean_gate_s"]))
ax.axhline(100, color=FAINT, ls=":", lw=1.4); ax.text(1.05, 104, "ideal", color=MUTED, fontsize=12)
for kind, c, lw, lab in [("random", SEC, 1.8, "random"), ("qft", AMBER, 2.8, "QFT")]:
    P_ = sorted(p for k, p in d if k == kind); b = min(d[(kind, 1)])
    eff = [100 * b / min(d[(kind, p)]) for p in P_]
    ax.plot(P_, eff, color=c, marker="o", lw=lw); label_end(ax, P_[-1], eff[-1], lab, c)
    if kind == "qft":
        for p, e in zip(P_, eff):
            ax.annotate(f"{e:.0f}%", (p, e), xytext=(0, 10), textcoords="offset points", ha="center",
                        color=TXT, fontsize=11, fontfamily=MONO["family"])
ax.set_xscale("log", base=2); ax.set_xticks([1, 2, 4, 8, 16])
ax.set_xticklabels(["1\n28 q", "2\n29 q", "4\n30 q", "8\n31 q", "16\n32 q"])
ax.set_xlim(0.8, 30); ax.set_ylim(0, 115)
ax.set_xlabel("MPI ranks and qubits, 4 GiB per rank, one node"); ax.set_ylabel("Weak-scaling efficiency, %, best of 3")
save(fig, ax, "c_mpi_weak.png")

# 7b sensitivity (teammate job 36597)
fig, ax = plt.subplots(figsize=(7.6, 4.6))
ca = {r["circuit"]: int(r["comm"]) for r in rows("sensitivity/36597/circuit_analysis.csv")}
d = defaultdict(list)
for r in rows("sensitivity/36597/runs.csv"):
    if r["kind"] == "random_gf":
        f = r["global_fraction"]
        key = f"random_30_seed_1_depth_10_global-fraction_{f}_global-qubits_4.txt"
        d[(ca[key], f)].append(float(r["total_s"]))
ks = sorted(d); xs = [k[0] for k in ks]
med = [st.median(d[k]) for k in ks]; lo = [min(d[k]) for k in ks]; hi = [max(d[k]) for k in ks]
slope, icpt = np.polyfit(xs, med, 1)
xx = np.linspace(0, 160, 50)
ax.plot(xx, icpt + slope * xx, color=FAINT, ls=":", lw=1.6)
ax.vlines(xs, lo, hi, color=MUTED, lw=1.4)
ax.plot(xs, med, "o", color=AMBER, ms=9)
for (n, f), m in zip(ks, med):
    ax.annotate(f"{float(f)*100:.0f} % global\n{m:.0f} s", (n, m), xytext=(10, -6), textcoords="offset points",
                color=TXT, fontsize=11, fontfamily=MONO["family"])
ax.text(80, icpt + slope * 80 - 40, f"fit: {slope:.2f} s per exchange", color=SEC, fontsize=12)
ax.set_xlabel("Gates that need a cross-rank exchange (of 300)"); ax.set_ylabel("Time, s (median of 3)")
ax.set_xlim(-5, 190); ax.set_ylim(0, 270)
save(fig, ax, "c_sensitivity.png")
print("sens slope", slope, icpt)

# 8 GPU
fig, ax = plt.subplots(figsize=(7.6, 4.6))
cs = [r for r in rows("summary/cuda_summary.csv") if r["kind"] == "qft"]
qs = [int(r["qubits"]) for r in cs]; g = [float(r["gpu_s"]) for r in cs]
cq = [r for r in cs if r["cpu_s"]]; cx = [int(r["qubits"]) for r in cq]; cy = [float(r["cpu_s"]) for r in cq]
ax.plot(cx, cy, color=SEC, lw=1.8); label_end(ax, cx[-1], cy[-1], "CPU, 32 cores", SEC)
ax.plot(qs, g, color=AMBER, marker="o", lw=2.8); label_end(ax, qs[-1], g[-1], "GPU", AMBER)
for r in cq:
    if int(r["qubits"]) in (26, 30):
        n = int(r["qubits"])
        ax.annotate(f'{float(r["speedup"]):.1f}x', (n, float(r["cpu_s"])), xytext=(-4, -26), textcoords="offset points",
                    color=TXT, fontsize=12, fontfamily=MONO["family"], ha="right")
ax.set_yscale("log"); ax.set_xlabel("Qubits"); ax.set_ylabel("QFT time, s"); ax.set_xlim(19.5, 35)
save(fig, ax, "c_gpu.png")

# 9 fusion: speedup per configuration, horizontal bars
fig, ax = plt.subplots(figsize=(7.6, 4.4))
fu = rows("fusion/36514/fusion.csv")
fu = sorted(fu, key=lambda r: float(r["speedup"]))
lab = [f'{"QFT" if r["kind"]=="qft" else "random"}  {r["qubits"]} q  {r["threads"]} t' for r in fu]
sp_ = [float(r["speedup"]) for r in fu]
ax.barh(range(len(fu)), sp_, color=[AMBER if r["kind"] == "random" else FAINT for r in fu], height=0.62)
for i, (v, r) in enumerate(zip(sp_, fu)):
    ax.text(v + 0.04, i, f'{v:.2f}x   {r["gates_none"]} → {r["gates_1q"]} gates', va="center", fontsize=11,
            color=TXT, fontfamily=MONO["family"])
ax.axvline(1, color=MUTED, lw=1)
ax.set_yticks(range(len(fu))); ax.set_yticklabels(lab); ax.set_xlim(0, 3.9)
ax.set_xlabel("Speedup from fusion"); ax.grid(axis="y", visible=False)
save(fig, ax, "c_fusion.png")

# 10 cover circuit
fig, ax = plt.subplots(figsize=(7, 3.6)); ax.axis("off"); ax.grid(False)
n, cols = 4, 9.8
for i in range(n): ax.plot([0, cols], [i, i], color=FAINT, lw=1.6, zorder=0)
def box(x, y, t, c, fill):
    ax.add_patch(FancyBboxPatch((x - .3, y - .3), .6, .6, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=fill, ec=c, lw=1.6, zorder=2))
    ax.text(x, y, t, ha="center", va="center", color=TXT if fill != AMBER else "#111113", fontsize=14,
            fontfamily=MONO["family"], zorder=3)
x = 0.8
for tgt in range(n - 1, -1, -1):
    box(x, tgt, "H", AMBER, AMBER); x += 0.9
    for k, ctl in enumerate(range(tgt - 1, -1, -1)):
        ax.plot([x, x], [ctl, tgt], color=SEC, lw=1.6, zorder=1); ax.add_patch(Circle((x, ctl), .09, color=SEC, zorder=3))
        box(x, tgt, f"R{k+2}", SEC, SURF); x += 0.9
ax.set_xlim(-.2, cols); ax.set_ylim(-.6, n - .4)
for i in range(n): ax.text(-.15, i, f"q{i}", ha="right", va="center", color=MUTED, fontsize=12, fontfamily=MONO["family"])
save(fig, None, "d_circuit.png")

# 11 pair diagram
fig, ax = plt.subplots(figsize=(7.4, 2.7)); ax.axis("off"); ax.grid(False)
for i in range(8):
    hi = i in (0, 2)
    ax.add_patch(FancyBboxPatch((i * 0.9, 0), .74, .74, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=SURF, ec=AMBER if hi else FAINT, lw=1.8 if hi else 1.2))
    ax.text(i * .9 + .37, .37, f"a{i}", ha="center", va="center", color=TXT, fontsize=14, fontfamily=MONO["family"])
    ax.text(i * .9 + .37, -.25, format(i, "03b"), ha="center", color=MUTED, fontsize=11, fontfamily=MONO["family"])
for i in [0, 1, 4, 5]:
    x0, x1 = i * .9 + .37, (i + 2) * .9 + .37
    ax.annotate("", xy=(x1, .82), xytext=(x0, .82), arrowprops=dict(arrowstyle="<->", color=AMBER if i == 0 else MUTED,
                lw=1.8 if i == 0 else 1.2, connectionstyle="arc3,rad=-0.45"))
ax.set_xlim(-.1, 7.3); ax.set_ylim(-.45, 1.6)
save(fig, None, "d_pairs.png")
print("charts done")

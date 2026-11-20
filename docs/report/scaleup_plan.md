# EuroHPC scale-up plan (draft)

> Lead: Ana-Maria. Input: Edard's MPI weak-scaling runs (`results/mpi_weak/`).
> All numbers below come from `python3 analysis/scaleup.py`; rerun it when the
> weak-scaling data change and paste the new tables. **Status: the node-hour
> figures are currently from the bandwidth model, not yet from Edard's
> measurements** (the script prints which source it used).

## 1. Goal

Show that the distributed simulator keeps its weak-scaling efficiency from one
node to a few hundred nodes, and measure strong scaling and the cost of
communication at a size no single node can hold. Target: **40 qubits**
(16 TiB state) on 256 nodes, with 32 qubits per node.

## 2. System and access route

* **System: MareNostrum 5 GPP** (BSC, EuroHPC JU). 6,192 standard nodes with
  2 × Intel Xeon Platinum 8480+ (112 cores) and 256 GiB DDR5, plus 216
  HighMem nodes with 1 TiB. Same instruction set (x86, AVX-512) and the same
  Slurm/EasyBuild-style environment as our development cluster, so the harness
  and `env/modules.sh` carry over by changing module names.
  The node's network bandwidth is stated as 200 Gb/s in some BSC material and
  as one NDR200 port shared by two nodes (100 Gb/s per node) in others; we plan
  with the conservative 100 Gb/s and measure it in the first week (§5).
* **Alternative: LUMI-C** (CSC): 2 × AMD EPYC 7763 (128 cores), 256 GiB per
  standard node, Slingshot-11 at 200 Gb/s. Same plan with `--system lumi-c`.
* **Access mode: Benchmark Access.** It is meant for scalability tests whose
  results feed a later Regular or Extreme Scale proposal, gives a limited
  number of node hours for at most three months, and has a cut-off on the
  first day of every month. If the optimisation work (remapping, fusion) needs
  more time on the machine, follow up with **Development Access** (12 months,
  same monthly cut-offs). Check the per-system node-hour cap in the current
  Terms of Reference before submitting.

## 3. Memory per qubit count

State = 16 B × 2ⁿ. The current MPI version receives the partner's whole slice
before a global-qubit gate, so each node needs 2 × its slice (buffer factor 1).
With 224 GiB usable per node (≈ 2 GiB per core reserved for OS and MPI):

| Qubits | State | Nodes (min, power of 2) | Per node (slice + buffer) | Share of MN5 GPP |
|---:|---:|---:|---:|---:|
| 32 | 64 GiB | 1 | 128 GiB | 0.02 % |
| 34 | 256 GiB | 4 | 128 GiB | 0.06 % |
| 36 | 1 TiB | 16 | 128 GiB | 0.26 % |
| 38 | 4 TiB | 64 | 128 GiB | 1.0 % |
| 40 | 16 TiB | 256 | 128 GiB | 4.1 % |

Rule of thumb: **each extra qubit doubles the nodes**. Exchanging in chunks
(buffer factor 1/16) would fit 33 qubits per node and reach 41 qubits on 256
nodes; that is an optimisation to ask Edard about, not an assumption of the
plan.

## 4. Node-hour estimate

Per run: gates(n) × t_gate × nodes. t_gate comes from (in this order)

1. **measured**: Edard's weak-scaling medians, fitted as
   t_gate(P) = a + b · log₂ P per circuit kind and extrapolated in log₂ P;
2. **model** (used now): a local gate streams 32 B per local amplitude at the
   node's memory bandwidth (planning value 400 GB/s, to be replaced by our
   STREAM result); a communicating gate also moves the 128 GiB-per-node slice
   over the network at 12.5 GB/s. The share of communicating gates per circuit
   comes from `analyze_circuit.py` at the planned rank count (one rank per node).

The model makes one point clear before any measurement: on MN5 GPP **a
communicating gate costs about 17 local gates**: its own local pass plus an
exchange that takes 16 times as long (slice / 12.5 GB/s versus
2 × slice / 400 GB/s). Circuits with many dense gates on high qubits are network
bound, which is why the sensitivity experiment and qubit remapping matter.

Campaign (3 repeats, × 1.5 for start-up, generation and failed runs):

| Phase | Runs | Qubits / nodes | Node-hours (model) |
|---|---|---|---:|
| Weak scaling | QFT, random | 32 q / 1 node … 40 q / 256 nodes | 552 |
| Strong scaling | QFT, random | 36 q on 16 … 256 nodes | 73 |
| Sensitivity | QFT, Grover, random; random F = 0 … 1 | 36 q on 16 nodes | 113 |
| **Total** | | | **≈ 740** (≈ 83 k core-hours) |
| + sensitivity again with remapping on | | | ≈ 115 |

Largest single run: random circuit at 40 qubits on 256 nodes, ≈ 9 min.
**Request: 1,500 node-hours** (≈ 1.75 × the 850 node-hours above), because the strong-scaling
model ignores latency and network contention, and the weak-scaling extrapolation
beyond Edard's largest measured node count is uncertain. Revise down once the
measured fit replaces the model.

## 5. Work plan (3 months)

| Week | What | Node-hours |
|---|---|---:|
| 1 | Port: modules, build, `validate.sbatch`; STREAM (`stream.sbatch`) and an MPI point-to-point bandwidth test between two nodes; replace the planning bandwidths with the measured ones and rerun `scaleup.py` | < 10 |
| 2–3 | Weak scaling 1 → 64 nodes (32 → 38 q); go/no-go: weak efficiency at 64 nodes ≥ ⟨0.7⟩ | ≈ 110 |
| 4–6 | Weak scaling to 256 nodes (40 q); strong scaling at 36 q | ≈ 515 |
| 7–8 | Sensitivity at 36 q / 16 nodes, remapping off and on | ≈ 225 (115 each, remapping off and on) |
| 9–12 | Reserve: reruns of noisy points (> 10 % range), final runs from the tagged commit | rest |

## 6. What we need from Edard

* `results/mpi_weak/<jobid>/` from `MODE=weak sbatch --nodes=≥4 slurm/mpi_scaling.sbatch`
  (runs inside one node contain no network traffic, and `scaleup.py` warns about it);
* ranks per node used, and whether the exchange is chunked (buffer factor);
* the flag that turns qubit remapping off, for the sensitivity runs.

Then: `python3 analysis/scaleup.py --mem-bw <STREAM> --net-bw <measured>`
replaces the model column with measured values, and the tables above are updated
from `results/summary/scaleup_*.csv`.

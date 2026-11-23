# Optimisation (draft)

> Owners: Marija (gate fusion), Edard (qubit remapping). Target length: 1 page.
> Numbers marked "deterministic" are gate or byte counts that do not depend
> on the machine; ⟨angle brackets⟩ are filled from `results/fusion/` and
> `results/mpi_*` / `results/sensitivity/` once the cluster jobs have run.

Both optimisations attack the same cost: every gate is one pass over memory
(on a node) or one exchange of a slice (between nodes). Gate fusion removes
passes; qubit remapping removes exchanges.

## Optimisation 1: single-qubit gate fusion

`fuse()` (`src/qsim/fusion.cpp`, `--fusion 1q`) rewrites the gate list
before simulation. It keeps, for every qubit, the product of the
single-qubit gates applied to it since the last two-qubit gate on that qubit.
Gates on other qubits do not end a run because they commute with it. A
two-qubit gate flushes the pending product as one gate, with three
refinements:

* a run of diagonal gates commutes with CZ/CP and with the control of CX/CY,
  so it is carried past them instead of being flushed;
* a SWAP just exchanges the pending runs of its two qubits;
* a product that is diagonal is emitted as one phase gate P (a half-sweep
  that touches only the |1⟩ amplitudes and never needs MPI communication),
  and a product equal to the identity up to a phase is dropped.

The global phases dropped by the last two rules are multiplied back into one
dense gate, so the fused circuit reproduces the unfused state exactly
(validated against Qiskit to 10⁻¹⁰ in CI, serial and on 4 MPI ranks, plus a
unit test over 200 random circuits with every gate kind). The pass outputs
ordinary gates, so all three backends use it without changes.

Gates per circuit at the benchmark sizes (seed 1, deterministic):

| Circuit | 24 q | 26 q | 28 q | 30 q | Reduction |
|---|---:|---:|---:|---:|---:|
| QFT | 336 → 335 | 390 → 389 | 448 → 447 | 510 → 509 | < 0.5 % |
| Grover, 1 iteration | 715 → 571 | 782 → 625 | 847 → 679 | 912 → 733 | 20 % |
| Random, depth 10 | 298 → 143 | 323 → 154 | 349 → 158 | 376 → 178 | 52–55 % |

QFT gains nothing: its only single-qubit gates are one H per qubit between
controlled-phase gates, so there is nothing to merge. Random circuits halve
their gate count. Because the code is bandwidth bound, we expect the time to
fall roughly with the number of full passes; a fused U is a full pass where
the gates it replaces may have been half passes (P, Z, S, T), so the time
saving is smaller than the gate saving. Measured (median of 3, 28 qubits):

| Circuit | 1 thread: none → 1q | speedup | 32 threads: none → 1q | speedup |
|---|---|---:|---|---:|
| Grover | ⟨⟩ | ⟨⟩ | ⟨⟩ | ⟨⟩ |
| Random | ⟨⟩ | ⟨⟩ | ⟨⟩ | ⟨⟩ |

(`sbatch slurm/fusion.sbatch`, then `python3 analysis/fusion.py
results/fusion/<jobid>` writes this table and `fusion.png`.)

A natural next step is two-qubit fusion (absorbing the neighbouring 1q gates
into a 4×4 gate on CX/CZ), which would also help QFT; we did not implement it.

## Optimisation 2: qubit remapping (MPI)

A dense gate on a global qubit costs a full-slice exchange. The MPI backend
instead swaps the global qubit with a local one before such a gate (half a
slice), choosing the local qubit whose next use is furthest away (Belady),
and keeps the permutation as a relabelling until the end, where the layout is
restored. SWAP gates become free relabellings. Details: `docs/mpi_backend.md`.

Exchanges and data sent per rank, 18 qubits on 4 ranks (2 global qubits),
seed 1 (deterministic; a laptop run, so volumes only, no network time):

| Circuit | No remap: comm gates / MiB per rank | Remap: swaps / MiB per rank | + fusion, remap: swaps / MiB |
|---|---|---|---|
| Random, depth 10 | 16 / 13.0 | 11 / 7.5 | 4 / 5.5 |
| Random, 25 % gates on global qubits | 24 / 15.5 | 6 / 5.0 | 3 / 3.0 |
| Random, 50 % gates on global qubits | 48 / 35.0 | 5 / 4.0 | 5 / 5.0 |
| Grover, 1 iteration | 28 / 18.0 | 4 / 2.5 | 5 / 3.0 |

Remapping cuts the traffic by 1.7–9× on these circuits, and the cut grows
with the share of gates on global qubits, which is the case where the
unoptimised code suffers most. Fusion on top helps when it removes dense
gates on global qubits; in two cases it sends 0.5–1 MiB more per rank,
because the remapping heuristic then sees a different gate sequence. Measured time
with and without remapping on the cluster (16 ranks, 30 qubits, sensitivity
job): ⟨⟩.

The two optimisations combine: fusion reduces passes over memory on every
backend, remapping reduces the remaining exchanges on MPI.

# MPI and communication bottleneck (draft)

> Owner: Edard. This is a starting draft written from `docs/mpi_backend.md`
> and the job scripts; please correct anything that does not match what you
> built, and fill the ⟨angle brackets⟩ from `results/mpi_strong/`,
> `results/mpi_weak/` and `results/sensitivity/`. Target length: 1 page.

## MPI

With P = 2ᵖ ranks, rank *r* holds the 2ⁿ⁻ᵖ amplitudes whose top *p* index
bits equal *r*. Qubits 0 … n−p−1 are local and every gate on them runs the
serial (or OpenMP) kernel on the local slice with no communication. For a
gate on a global qubit the backend distinguishes:

* **diagonal gates** (Z, S, T, P, RZ, CZ, CP): every rank knows its global
  bits, so it scales its slice or applies a local diagonal: no communication;
* **CX/CY with a global control**: ranks whose control bit is 0 do nothing,
  the others apply X/Y locally: no communication;
* **dense 1q gates on a global qubit**: each rank exchanges its slice with
  partner `r XOR 2^(g−(n−p))` (`MPI_Sendrecv`, 64 MiB chunks so the extra
  buffer is ≈ 0.1 % of the slice) and applies the pair kernel to its own and
  the received half;
* **CX/CY with a local control and global target** exchanges only the half
  of the slice where the control is 1, and **SWAP** exchanges the half that
  changes rank.

Compute and communication are timed separately with `MPI_Wtime` per gate;
the binary reports the communicated volume per rank (`comm_GiB_per_rank`),
which matches the static count of `src/tools/analyze_circuit.py --ranks P`
when remapping is off. The final state (written collectively with MPI-IO)
matches the serial simulator on 1, 2, 4 and 8 ranks (unit tests) and Qiskit
to 10⁻¹⁰ on 4 ranks with remapping on and off (CI).

Setup: ⟨node type⟩, ⟨interconnect⟩; the course QOS limits a user to 64 CPUs,
so the experiments use 2 nodes × 16 ranks × 1 core (P ≤ 32) for scaling and
2 × 8 ranks × 2 cores for the sensitivity runs.

**Strong scaling** (30 qubits, 16 GiB state, QFT and random, median of 3):
⟨figure: time per gate and efficiency vs P = 1 … 32; P = 16 on 1 node vs
2 nodes (`--ntasks-per-node=8`)⟩. Expected shape: near-linear while all
ranks share one node's memory bandwidth only up to the point where the
socket's bandwidth saturates, then a step when ranks span two nodes and
exchanges cross the network.

**Weak scaling** (28 qubits at P = 1, +1 qubit per doubling, so 4 GiB per
rank): ⟨figure: time per gate vs P⟩. These are the numbers the EuroHPC plan
extrapolates from.

## Communication bottleneck

A dense gate on a global qubit moves the whole slice, 16 B × 2ⁿ⁻ᵖ per rank,
both ways. At 64 GiB per node and the ⟨X⟩ GB/s we measured between nodes
that is ⟨X⟩ s per such gate, against ⟨X⟩ s for the same gate on local
data, so every global-qubit gate costs as much as ⟨X⟩ local gates. The
communication share therefore depends on the circuit, not only on its size.
The sensitivity experiment makes that explicit: random circuits with
0–100 % of their single-qubit gates on global qubits, 30 qubits on 16 ranks.

| Share of gates on global qubits | Comm gates (static) | Time, remap off | Comm time share | Time, remap on |
|---|---:|---:|---:|---:|
| 0 % | ⟨⟩ | ⟨⟩ | ⟨⟩ | ⟨⟩ |
| 10 % | ⟨⟩ | ⟨⟩ | ⟨⟩ | ⟨⟩ |
| 50 % | ⟨⟩ | ⟨⟩ | ⟨⟩ | ⟨⟩ |
| 100 % | ⟨⟩ | ⟨⟩ | ⟨⟩ | ⟨⟩ |

Qubit remapping (Optimisation 2, see the Optimisation section) turns most
full-slice exchanges into one half-slice swap per run of gates on a global
qubit; on 18-qubit test circuits it cut the bytes sent per rank by 1.7–9×.
What remains: exchanges are blocking, so communication and computation do not
overlap (non-blocking `MPI_Isend`/`MPI_Irecv` on the next chunk would hide
part of it), and remapping swaps one qubit at a time where an all-to-all
could swap several.

**Hybrid MPI × OpenMP** (with Hurshneet): at a fixed 64 CPUs, ⟨16 × 2, 8 × 4,
4 × 8, 2 × 16⟩ ranks × threads at 30 qubits: ⟨table⟩. Fewer, fatter ranks mean
fewer global qubits and fewer exchanges, but each rank then needs the
bandwidth of several cores.

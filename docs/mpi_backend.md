# MPI backend (`qsim_mpi`)

Distributed state-vector simulation over `P = 2^p` MPI ranks, optionally with
OpenMP threads inside each rank. Code: `src/qsim/mpi/dist_sim.{hpp,cpp}`,
driver `src/apps/qsim_mpi_main.cpp`, tests `tests/test_mpi.cpp`.

## Build and run

```bash
source env/modules.sh mpi
cmake -S . -B build-mpi -DQSIM_MPI=ON -DQSIM_OPENMP=ON   # OPENMP=OFF for pure MPI
cmake --build build-mpi -j
ctest --test-dir build-mpi                               # kernels + distributed vs serial on 1/2/4/8 ranks

srun -n 16 build-mpi/qsim_mpi circuit.txt                # or: mpirun -np 16 ...
```

On a laptop with fewer cores than ranks, configure with
`-DMPIEXEC_PREFLAGS=--oversubscribe` so `ctest` can start 8 ranks.

Same command line and `SUMMARY` line as `qsim`, plus:

| Option | Meaning |
|---|---|
| `--no-remap` / `--remap` | qubit remapping off / on (default **on**) |
| `--chunk-mib N` | size of one exchange message in MiB (default 64) |
| `--timing FILE` | per-gate CSV of **rank 0**; serial columns plus `comm_bytes,remapped` |
| `--dump FILE` | final state in index order, written collectively with MPI-IO |

Extra `SUMMARY` fields: `ranks threads local_qubits remap comm_gates remaps
comm_GiB_per_rank chunk_MiB buffer_factor restore_s`. The run harness picks up
`ranks=` and `threads=` automatically.

## Data layout

Rank `r` holds amplitudes `r · 2^(n-p) … (r+1) · 2^(n-p) - 1`. Physical qubit
positions `0 … n-p-1` are local, `n-p … n-1` are global (bit `k - (n-p)` of the
rank number), as in `docs/kernel_api.md`. The rank count must be a power of two
and leave at least one local qubit.

How each gate is applied (`a`, `b` = physical positions):

| Gate | Both local | Global qubit involved |
|---|---|---|
| H X Y RX RY U | `apply_1q` | swap the whole slice with partner `rank ^ bit`, `apply_1q_pairs` |
| Z S T P RZ … | `apply_1q_diag` | scale the whole slice by `d0` or `d1`: **no communication** |
| CX / CY | `apply_controlled_1q` | control global: apply X/Y locally or do nothing (no comm). Target global, control local: exchange **only the half** with control = 1. Both global: ranks with control = 1 exchange whole slices |
| CZ / CP | `apply_controlled_phase` | `apply_1q_diag` on the local qubit, or scale the slice: **no communication** |
| SWAP | `apply_swap` | local–global: exchange the half that changes rank. global–global: ranks whose two bits differ trade slices |

Exchanges use blocking `MPI_Sendrecv` in chunks of `--chunk-mib`. The send and
receive buffers take `2 × chunk` per rank, so the **buffer factor** is
`2 × chunk / slice`. At the default 64 MiB that is 0.002 for a 64 GiB slice
(32 qubits per node) and 0.001 for a 128 GiB slice (33 qubits per node). In
practice a node needs **one slice**, not two, so `scaleup_plan.md` §3 can
assume buffer factor ≈ 0: 33 qubits per node (128 GiB) fit in the 224 GiB
usable, which is one more qubit per node than the current table.

## Qubit remapping (optimisation 2)

The simulator keeps a layout (logical qubit → physical position). With
remapping on:

* Before a gate that would need an exchange (a dense 1q gate on a global qubit,
  or CX/CY with a global target), the global qubit is swapped with a local
  one. The local qubit chosen is the one whose next use is furthest in the future
  (Belady). For CX/CY the control is never chosen. The swap exchanges **half**
  a slice, compared with the full slice the gate itself would exchange, and
  later gates on that qubit are local.
* A SWAP gate is just a relabelling: no data moves.
* At the end of the run the layout is restored to the identity so the final
  state (and `--dump`) is in standard order. That time is included in
  `total_s` and also reported as `restore_s`.

With `--no-remap` the layout stays the identity and every gate is executed as
in the table above. `comm_gates` then equals the `comm` column of
`src/tools/analyze_circuit.py --ranks P`. One exception: the static analysis counts
a SWAP of **two global** qubits as free (rank relabel), but the binary
exchanges data for it. QFT never does this. A random circuit can, when both
qubits of a `swap` land on global positions.

First local numbers (8 ranks on one laptop, so they show traffic volume, not
network time):

| Circuit | comm gates, off | GiB sent per rank, off | remaps, on | GiB sent per rank, on |
|---|---:|---:|---:|---:|
| qft 18 | 9 | 0.0037 | 6 | 0.0034 |
| grover 18, 2 iterations | 120 | 0.041 | 16 | 0.0044 |
| random 22, depth 20 | 49 | 0.33 | 42 | 0.20 |

## Answers for `scaleup_plan.md` §6

* **Ranks per node:** the code supports any power of two. For MareNostrum 5 GPP
  plan one rank per node with 112 OpenMP threads (`--ntasks-per-node=1
  --cpus-per-task=112`) as in the plan. Two ranks per node (one per socket,
  56 threads each) is worth measuring in week 1, because it keeps memory
  accesses on the local socket.
* **Chunked exchange:** yes. Buffer factor = `2 × chunk / slice`, ≈ 0.002 at
  32 qubits per node and ≈ 0.001 at 33 with the default 64 MiB chunk, so
  33 qubits per node and 41 qubits on 256 nodes fit (see above).
* **Flag that turns remapping off:** `--no-remap`, e.g.
  `QSIM_ARGS=--no-remap sbatch slurm/sensitivity.sbatch`.
* **Weak-scaling data:** `MODE=weak N=<q> sbatch --nodes=4 slurm/mpi_scaling.sbatch`.
  Not run yet: it needs the cluster.

## Not done yet / ideas

* Communication and computation do not overlap: the next chunk could be sent
  with `MPI_Isend`/`MPI_Irecv` while the current one is computed.
* Remapping swaps one qubit at a time. Swapping several global qubits in one
  all-to-all is the next step for circuits that use many global qubits.
* Per-gate timings are rank 0's wall time, which includes waiting for the
  partner.

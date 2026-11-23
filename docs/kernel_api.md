# Gate kernel API

This is the contract the OpenMP, MPI and CUDA versions build on. The serial
code is the reference: every other backend must reproduce its final state to
1e-10 (`src/tools/validate_qiskit.py`).

## Layers

```
circuit file ──read_circuit()──▶ Circuit{num_qubits, vector<Gate>}
                                   │
                                   ▼  fuse(gates, n, mode)        src/qsim/fusion.*   (optimisation 1)
                              vector<Gate>
                                   │
                                   ▼  run(state, gates)           src/qsim/simulator.*  (timing per gate)
                              apply_gate(amp, nq, gate)           single dispatch Gate → kernel
                                   │
                                   ▼
                              kern::apply_*(amp, nq, ...)         src/qsim/kernels.hpp  (hot loops, header-only)
```

* `StateVector` (`state.hpp`) owns 2^n `std::complex<double>` amplitudes in
  a 4 KiB-aligned allocation. `set_zero_state()` constructs the amplitudes
  in place and first touches pages using the kernels' static tile partition.
* `Gate` (`gates.hpp`) is a small POD: `kind`, `q0`, `q1`, `theta`, `m` (2×2
  matrix, only for `U`). `matrix_1q(g)` gives the 2×2 matrix of any one-qubit
  gate and the target action of `CX`/`CY`/`CP`.
* `bytes_touched(g, n)` is the memory traffic model used for the GB/s column.

## Index convention

Qubit `q` is bit `q` of the amplitude index (little-endian, like Qiskit).

Every kernel is a loop over a *pair index* `k` and builds the amplitude indices
from it with bit insertion:

```cpp
// 1-qubit gate on q: 2^(n-1) independent pairs
for (index k = 0; k < 2^(n-1); ++k) {
  i0 = insert_zero_bit(k, q);   // bit q = 0
  i1 = i0 | (1 << q);           // bit q = 1
  (amp[i0], amp[i1]) = M * (amp[i0], amp[i1]);
}
// 2-qubit gate on (lo, hi): 2^(n-2) independent groups
base = insert_two_zero_bits(k, lo, hi);
```

Iterations are fully independent, there is no reduction and no branch on data.

## Kernels (`src/qsim/kernels.hpp`, namespace `qsim::kern`)

| Function | Used for | Amplitudes touched |
|---|---|---|
| `apply_1q(amp, nq, q, M)` | H X Y RX RY U | all |
| `apply_1q_diag(amp, nq, q, d0, d1)` | Z S Sdg T Tdg P RZ | half if `d0 == 1`, else all |
| `apply_controlled_1q(amp, nq, c, t, M)` | CX CY | half |
| `apply_controlled_phase(amp, nq, a, b, phase)` | CZ (phase −1), CP | quarter |
| `apply_swap(amp, nq, a, b)` | SWAP | half |
| `apply_1q_pairs(lo, hi, count, M)` | MPI global-qubit gates | `count` pairs |

All take a raw `cplx*` and the number of qubits *in that buffer*, never a
`StateVector`, so they run unchanged on a rank-local slice.

## How each backend plugs in

**OpenMP.** Build with `-DQSIM_OPENMP=ON`. State initialization and in-place
gates use `state_loop.hpp`, which assigns each worker a contiguous static
range of 4 KiB-equivalent tiles. A tile holds 256 amplitude indices, 128
pair indices or 64 two-qubit group indices. Inserting the omitted bits
preserves the cache-line boundaries of each range on an aligned state.
First-touch and gate ownership match for qubits within a worker's aligned
range in the power-of-two thread sweep. High qubits can pair amplitudes across
ranges. The split-buffer MPI kernel uses `schedule(static)` for its separately
allocated inputs. Set `OMP_DYNAMIC=false` for a fixed team and use
`OMP_PROC_BIND` / `OMP_PLACES` for runtime affinity.

**MPI.** With `P = 2^p` ranks each rank holds `2^(n-p)` amplitudes. Qubits
`0 .. n-p-1` are *local*, qubits `n-p .. n-1` are *global* (they are bits of
the rank number).
* Gate only on local qubits: call `apply_gate(local_amp, n - p, g)` as is.
* 1q gate on global qubit `g`: partner rank is `rank ^ (1 << (g - (n-p)))`.
  Exchange the slice (`MPI_Sendrecv`, in chunks), then the rank with bit 0
  calls `apply_1q_pairs(mine, theirs, ...)` and keeps the `lo` half, the
  other rank keeps `hi` (or both compute and each keeps its own half).
* Diagonal gates (Z, S, T, P, RZ, CZ, CP) on global qubits need **no
  communication**: the rank knows the global bit value and just scales.
* Qubit remapping (optimisation 2) = a SWAP between a global and a local qubit
  followed by relabelling the remaining gates.

**CUDA.** Port each kernel 1:1: the thread id is `k`, the body is identical
(`insert_zero_bit` is `constexpr` and works in device code if marked
`__host__ __device__`). Keep the state on the device for the whole circuit and
dispatch with the same `switch` as `apply_gate`; time host↔device copies
separately.
Implemented in `src/qsim/cuda/cuda_sim.cu` (app `qsim_cuda`, CMake option
`-DQSIM_CUDA=ON`, default arch 89). Its `SUMMARY` line adds `h2d_s` and
`d2h_s`; `total_s` is gate time only. Build + validate on the cluster with
`sbatch slurm/cuda_validate.sbatch`.

**Gate fusion.** `fuse()` in `fusion.cpp` runs before `run()` (all three
apps: `--fusion none|1q`). Mode `none` returns the list unchanged. Mode `1q`
merges every run of single-qubit gates on the same qubit into one gate; gates
on other qubits do not end a run. Diagonal runs are carried past CZ/CP and the
control of CX/CY (they commute), and SWAP moves pending runs across. A merged
run becomes a `U` with the 2x2 product, a `P` if the product is diagonal (so
it stays a half-sweep and needs no MPI exchange on a global qubit), or nothing
if it is the identity up to a phase. The global phases dropped this way are
folded back into one dense gate, so the final state matches the unfused
circuit exactly (validated against Qiskit at 1e-10 in CI, serial and on 4 MPI
ranks). Because fusion outputs ordinary `Gate`s, every backend benefits
without changes. Tests: `tests/test_fusion.cpp`; experiment:
`slurm/fusion.sbatch` + `analysis/fusion.py`.

## Timing output

`qsim CIRCUIT --timing t.csv` writes one row per gate:
`index,gate,q0,q1,theta,seconds,bytes,gbps`. Stdout gets a per-gate-type table
and one machine-readable line:

```
SUMMARY qubits=24 gates_in=90 gates_run=90 fusion=none total_s=3.69 mean_gate_s=0.041 norm_err=8.7e-13
```

Use `--no-gate-timing` to time only the whole circuit (timer calls cost
microseconds, which matters below ~16 qubits).

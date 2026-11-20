# Gate kernel API

This is the contract the OpenMP, MPI and CUDA versions build on. The serial
code is the reference: every other backend must reproduce its final state to
1e-10 (`src/tools/validate_qiskit.py`).

## Layers

```
circuit file ──read_circuit()──▶ Circuit{num_qubits, vector<Gate>}
                                   │
                                   ▼  fuse(gates, n, mode)        src/qsim/fusion.*   (optimisation 1 hook)
                              vector<Gate>
                                   │
                                   ▼  run(state, gates)           src/qsim/simulator.*  (timing per gate)
                              apply_gate(amp, nq, gate)           single dispatch Gate → kernel
                                   │
                                   ▼
                              kern::apply_*(amp, nq, ...)         src/qsim/kernels.hpp  (hot loops, header-only)
```

* `StateVector` (`state.hpp`) owns 2^n `std::complex<double>` amplitudes,
  64-byte aligned, allocated without construction. Pages are first touched in
  `set_zero_state()`, which uses the same static schedule as the kernels.
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

**OpenMP.** Build with `-DQSIM_OPENMP=ON`. That defines `QSIM_USE_OPENMP` and
turns `QSIM_PRAGMA_FOR` (already placed on every kernel loop and on the
initialisation loop) into `#pragma omp parallel for schedule(static)`. Tune
there: schedule, `collapse`, blocking for low `q`, SIMD. Pinning is set at run
time with `OMP_PROC_BIND` / `OMP_PLACES`.

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

**Gate fusion.** `fuse()` in `fusion.cpp` runs before `run()`. Mode `none`
returns the list unchanged; mode `1q` is reserved for merging consecutive
single-qubit gates on the same qubit into one `GateKind::U`. Because fusion
outputs ordinary `Gate`s, every backend benefits without changes.

## Timing output

`qsim CIRCUIT --timing t.csv` writes one row per gate:
`index,gate,q0,q1,theta,seconds,bytes,gbps`. Stdout gets a per-gate-type table
and one machine-readable line:

```
SUMMARY qubits=24 gates_in=90 gates_run=90 fusion=none total_s=3.69 mean_gate_s=0.041 norm_err=8.7e-13
```

Use `--no-gate-timing` to time only the whole circuit (timer calls cost
microseconds, which matters below ~16 qubits).

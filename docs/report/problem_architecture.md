# Problem and architecture (draft)

> Owner: Marija. Target length: 0.5 page each. Values in ⟨angle brackets⟩
> come from the final cluster runs (see `docs/report/README.md`).

## Problem

A quantum circuit on *n* qubits acts on a state of 2ⁿ complex amplitudes.
A state-vector simulator stores all of them (16 B each in double precision)
and applies every gate as an update of that vector: a one-qubit gate mixes
the 2ⁿ⁻¹ pairs of amplitudes whose indices differ only in the target bit, a
controlled gate does the same on the half where the control bit is 1, and a
diagonal gate only rescales. The method is exact and general, and it is how
quantum algorithms are checked before they run on hardware. Its cost is
exponential: 28 qubits need 4 GiB, 32 qubits 64 GiB, 40 qubits 16 TiB, and
each extra qubit doubles both memory and time per gate.

Each gate performs only a few floating-point operations per amplitude it
reads and writes (0.19–0.44 flop/B, see the CPU bottleneck section), so the
simulator is limited by memory bandwidth on one device and by network
bandwidth once the state is split over several. The project asks how far
three parallel technologies get on that problem: OpenMP on one node, MPI
across nodes (the only way past one node's memory), and CUDA on one GPU,
and which optimisations reduce the number of passes over memory and the
data sent over the network.

## Architecture

The code is one C++17 library with one binary per backend, built with CMake
(`QSIM_OPENMP`, `QSIM_MPI`, `QSIM_CUDA`, `QSIM_LIKWID` options).

```
circuit file ─▶ read_circuit() ─▶ fuse() ─▶ backend ─▶ SUMMARY line + per-gate CSV
 gen_circuits.py                  opt. 1     │
 (QFT, Grover,                               ├─ qsim       serial / OpenMP kernels
  random, seeded)                            ├─ qsim_mpi   2^p ranks, qubit remapping (opt. 2)
                                             └─ qsim_cuda  one kernel launch per gate
```

* **Gates and layout.** Qubit *q* is bit *q* of the amplitude index (the
  Qiskit convention). A gate is a small record (kind, qubits, angle, optional
  2×2 matrix), and one `apply_gate` switch maps it to a kernel.
* **Kernels.** Every kernel is a loop over a pair index *k*; inserting a zero
  bit at the target position gives the two amplitudes of pair *k*. The loop
  body has no dependencies between iterations, so the same kernel is the
  OpenMP loop (static schedule, cache-line-aligned tiles, matching first
  touch), the CUDA kernel (thread id = *k*), and the per-rank kernel of the
  MPI backend (applied to the rank's local slice).
* **Distribution.** With P = 2ᵖ ranks, the top *p* qubits are *global*: they
  select the rank. Gates on local qubits run without communication; diagonal
  gates never communicate; a dense gate on a global qubit exchanges slices
  with a partner rank in 64 MiB `MPI_Sendrecv` chunks.
* **Correctness.** Every backend is checked against Qiskit state vectors
  (max |Δψᵢ| ≤ 10⁻¹⁰, 2–12 qubits in CI on every pull request, serial,
  OpenMP, and MPI on 4 ranks with and without remapping and fusion). Unit tests cover the kernels, the fusion pass and
  distributed vs serial results on 1–8 ranks.
* **Reproducibility.** All runs go through one Slurm harness that records
  the git commit, modules, nodes, CPU model, environment and exact command,
  and writes one CSV row per run; the analysis scripts read only those CSVs.
  `REPRODUCE.md` goes from a fresh clone to every figure.

Size: ≈ 1 800 lines of C++ for the library and apps, of which the OpenMP
changes are a few lines per kernel, the MPI backend ≈ 470 and the CUDA backend
≈ 250.

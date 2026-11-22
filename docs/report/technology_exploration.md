# Technology exploration: CUDA (draft)

> Draft for the report (Emma). GPU and CPU numbers come from job 33869
> (`results/cuda/33869/`, commit `ead5cef`, `git_dirty: no`) and
> `results/summary/cuda_summary.csv` (`python3 analysis/cuda_summary.py`).
> Values in ⟨angle brackets⟩ are still to be filled in.

## Port

The CUDA backend (`src/qsim/cuda/`) is a 1:1 port of the CPU kernels. Each CPU
kernel is a loop over a pair index *k*, so the loop was removed and every GPU
thread handles one *k* with the same index arithmetic (`insert_zero_bit`, now
`__host__ __device__`). The state is copied to the GPU once, every gate is one
kernel launch dispatched like `apply_gate()`, and the state is copied back at
the end. Gate time and the two copies are timed separately with CUDA events.
The port passed the same Qiskit check as the CPU code: 57/57 circuits from 2 to
20 qubits within 10⁻¹⁰, largest error 4.4 × 10⁻¹⁴.

Most of the effort went into the platform, not the kernels. On SciTech CUDA
12.8.1 is an Apptainer container without CMake, so the job compiles with `nvcc`
inside it and detects the GPU architecture at run time (sm_120 on the RTX PRO
6000 Blackwell). Neither NVIDIA profiler could be used: Nsight Systems is not
installed, and Nsight Compute needs GPU performance counters, which are
restricted to administrators (`ERR_NVGPUCTRPERM`). The CUDA event timers are
used for the split between copies and kernels instead.

## Setup

One NVIDIA RTX PRO 6000 Blackwell Server Edition (96 GB) against the CPU cores
of the same node, using the OpenMP build with pinned threads. The course limits
a user to 64 CPUs, which Slurm gave as 32 physical cores with two hardware
threads each, on one socket of a 2 × 128-core AMD EPYC 9755 node. QFT, Grover
(one iteration) and random (depth 10) circuits, seed 1, 3 repeats, median
reported. The GPU ran 20 to 32 qubits and the CPU 20 to 30 qubits. Times are
gate time only on both sides; the GPU copies are reported separately.

## Results

| Qubits | Speedup, GPU vs 32 cores | GPU GB/s | CPU GB/s |
|---|---|---|---|
| 20–24 | 3.6–5.1× | 1 500–2 950 | 300–690 |
| 26–30 | 5.6–6.7× | 1 400–1 450 | 210–260 |

From 26 qubits on the GPU sustains 1 400–1 460 GB/s and the CPU allocation
210–260 GB/s, using the traffic model of the per-gate timings (bytes read and
written by each kernel). At 30 qubits the bandwidth ratio is 6.0–6.7, the same
as the measured speedup (6.0× Grover, 6.7× QFT, 6.0× random). The simulator is
memory-bandwidth bound on both devices, and the GPU wins by roughly the factor
by which its memory is faster. For small states the speedup is lower and both
devices report more than their DRAM bandwidth. At 20–22 qubits the state is
16–64 MiB and largely stays in cache, and each GPU kernel has little work to
amortise its launch.

The GPU reaches 32 qubits: a 64 GiB state in 96 GB of device memory, while 33
qubits would need 128 GiB. One extra qubit doubles the time (QFT: 4.3 s at 30,
9.1 s at 31, 19.2 s at 32 qubits), as expected for a bandwidth-bound code.
Beyond 32 qubits the state has to be split across devices, which is what the
MPI backend does.

**Transfers.** Copying the 16 GiB state at 30 qubits takes 0.62 s to the GPU
(≈ 27 GB/s) and 1.5–2.3 s back (7–12 GB/s). For these short circuits the copies
are 9–34 % of the GPU's time (copies + kernels) at 26–32 qubits. They happen once
per circuit, so their share falls for deeper circuits. Pinned host memory
(`cudaHostRegister`) would likely speed up both directions; we have not tried it.

**Repeatability.** GPU repeats differ by at most 0.22 %, and the same circuits
gave the same times in an earlier job on another node (QFT at 31 qubits:
9.086 s in both). CPU repeats differ by a median of 4 % and up to 19 %, likely
because other jobs share the node's memory system.

**Caveat.** The CPU baseline is 32 of the node's 256 cores and uses one of its
two sockets, so it has half the node's memory channels. The plan was a full
node, which the per-user limit does not allow. Because the code is bandwidth
bound, a full node would likely narrow the gap to the GPU considerably. We
could not measure that.

## Technology comparison

| | CPU / OpenMP | MPI | CUDA (1 GPU) |
|---|---|---|---|
| Programming effort | Lowest: one pragma per kernel loop; tuning is pinning, first touch and NUMA placement | Highest: data split over 2^p ranks, exchange of slices for gates on global qubits, qubit remapping (≈ 470 lines in `src/qsim/mpi/` and the app) | Low for the kernels (1:1 port, ≈ 220 lines in `cuda_sim.cu`); moderate for build and tooling on the cluster |
| Peak memory / network bandwidth | ⟨STREAM triad, 32 cores of one socket (Hurshneet)⟩ | ⟨interconnect bandwidth (Edard)⟩ | ⟨GPU memory bandwidth, vendor datasheet⟩ |
| Achieved bandwidth | 210–260 GB/s (32 cores, 26–30 qubits) | ⟨GB/s per rank incl. exchanges (Edard)⟩ | 1 400–1 460 GB/s (26–32 qubits) |
| Max qubits | 35 by memory (755 GiB node; 512 GiB state); measured to 30 | ⟨by total memory of the ranks used (Edard)⟩ | 32 (96 GB device memory) |

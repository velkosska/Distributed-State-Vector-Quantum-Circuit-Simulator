# OpenMP

Each gate updates independent amplitude pairs or groups. The OpenMP build
divides the loop statically into contiguous ranges and synchronizes at the
end of the gate. Static assignment fits the uniform work per iteration
and avoids distributing work at runtime. Initialization and the gate loops
use the same partition over 4 KiB-equivalent tiles. One tile contains
256 amplitudes, 128 one-qubit pair indices, or 64 two-qubit group indices.
The state allocation is aligned to 4 KiB. Gates use a direct loop below
64 KiB or when running on one thread.

A 64-byte cache line holds four complex-double amplitudes. Targets 0
and 1 can place both members of a pair in one line. Target 2 puts them in
adjacent lines. With the power-of-two thread counts in the 28-qubit sweep,
even divisions of the original pair loops already fell on cache-line
boundaries. The tiled partition also prevents workers from splitting a
cache line when the team size does not divide the loop count. At 64 threads
the state ranges are 64 MiB and first-touch ownership matches gate ownership
when all gate qubits are below 22. Higher qubits can couple ranges and
cause remote NUMA accesses.

The OpenMP build passed the Qiskit state-vector comparison at a maximum
absolute amplitude tolerance of 10⁻¹⁰ in a local Linux container. QFT, Grover
and random circuits passed all 57 cases from 2 to 20 qubits with four threads
and close binding, and all 39 cases from 2 to 14 qubits with three threads
and spread binding. The largest amplitude error was 4.33 × 10⁻¹⁴ across
both sweeps.

The strong-scaling experiment uses one exclusive Slurm node, fixed 28-qubit
QFT and seeded random circuits, and 1, 2, 4, 8, 16, 32 and 64 threads. Each
thread count uses three repeats under both close and spread binding with
`OMP_PLACES=cores`. The reported gate-loop time is the median. For each
circuit and binding, speedup is `T(1) / T(p)` and efficiency is speedup
divided by `p`. The two timing extremes supply the plotted ranges. Cluster
speedups and efficiencies are pending.

# Bottleneck analysis (CPU)

At 28 qubits, the state occupies 4 GiB. H and RZ read and write the entire
state. CX updates the half where the control bit is set. The current CX
implementation calls the generic controlled matrix kernel, so its modeled
work counts a dense multiply on each updated pair.

| Gate and qubits | Modeled FLOP per state amplitude | Modeled bytes per state amplitude | Modeled FLOP/B |
| --- | ---: | ---: | ---: |
| H on 0 | 14 | 32 | 0.4375 |
| RZ(0.37) on 0 | 6 | 32 | 0.1875 |
| CX with control 1 and target 0 | 7 | 16 | 0.4375 |

The byte model counts one load and one store per updated amplitude. With
CX control 1, each 64-byte cache line contains two updated amplitudes and
two untouched amplitudes. A cache miss or dirty eviction moves the whole
line, so CX can generate substantially more DRAM traffic than its logical
byte count predicts. STREAM triad also distinguishes its nominal 24 bytes
per element from measured DRAM traffic, which can include write allocation.
These low modeled intensities suggest a bandwidth limit. The measured
fraction of STREAM and the scaling curve are needed to assess where the
simulator actually saturates.

In the profiling job, LIKWID measures memory-controller volume over warmed
batches of H, RZ, CX and STREAM triad. STREAM uses three 2 GiB arrays on the
same exclusive node and the same cores, binding and thread count as each gate
batch. The roofline uses modeled FLOP divided by measured DRAM bytes on the
horizontal axis and modeled GFLOP/s on the vertical axis. Its memory slope is
the measured STREAM triad DRAM bandwidth. The bandwidth table also compares
effective gate GB/s with conventional STREAM triad GB/s. This keeps the
comparison consistent with the byte definition in each case. Counter data,
attained bandwidth fractions and the measured roofline are pending.
[LIKWID marker measurements](https://github.com/RRZE-HPC/likwid/wiki/likwid-perfctr#using-the-marker-api)

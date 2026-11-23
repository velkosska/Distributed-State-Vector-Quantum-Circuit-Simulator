# Conclusions (draft)

> Owner: Marija. Target length: 0.5 page. Rewrite once the ⟨angle bracket⟩
> numbers are in; keep one claim per paragraph and point to the figure.

State-vector simulation is memory-bandwidth bound on every device we used,
and every result follows from that. On the GPU the code sustains
1 400–1 460 GB/s and is 5.6–6.7× faster than 32 CPU cores from 26 qubits on,
which is the ratio of the two memory bandwidths. On the CPU, OpenMP scaling
⟨flattens at about X threads, when the socket's measured STREAM bandwidth of
X GB/s is reached⟩, and the roofline places all three gate types on the
bandwidth slope.

Across nodes the limit becomes the network. MPI lets us go past one device's
memory (⟨X⟩ qubits on ⟨X⟩ nodes, against 32 on the GPU), but each gate on a
global qubit costs a full-slice exchange; the sensitivity experiment shows the
time ⟨grows linearly⟩ with the share of such gates. Qubit remapping makes the
cost depend much less on the circuit: it cut the data sent per rank by
1.7–9× on test circuits and ⟨X⟩× in time on the cluster.

Gate fusion halves the gate count of random circuits and removes 20 % of
Grover's, at no cost in accuracy (the fused state matches Qiskit to 10⁻¹⁰);
the measured time saving is ⟨X⟩. It does nothing for QFT, which would need
two-qubit fusion.

Technology choice: CUDA gave the largest speedup for the least code, as long
as the state fits on one GPU (32 qubits here). OpenMP is the cheapest to add
but is capped by one node's bandwidth and memory. MPI is the most code and the
only route to 40+ qubits, which is why the EuroHPC plan builds on it.
Next steps: overlapping communication with computation, multi-qubit swaps in
the remapper, two-qubit fusion, and a multi-GPU (MPI + CUDA) backend.

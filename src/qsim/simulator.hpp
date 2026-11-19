#pragma once
// Serial driver: dispatches gates to kernels and records per-gate timings.
#include <iosfwd>
#include <vector>

#include "qsim/circuit.hpp"
#include "qsim/state.hpp"

namespace qsim {

// Apply one gate to a full (non-distributed) amplitude buffer of 2^nq entries.
// This is the single dispatch point from Gate to kernel; backends that keep the
// whole state in one address space (serial, OpenMP, CUDA via its own
// dispatcher) mirror this function.
void apply_gate(cplx* amp, int nq, const Gate& g);
inline void apply_gate(StateVector& s, const Gate& g) { apply_gate(s.data(), s.num_qubits(), g); }

struct GateTiming {
  Gate gate;
  double seconds;
  double bytes;
};

struct RunOptions {
  bool time_each_gate = true;  // false: only total time (removes timer overhead for tiny states)
};

struct RunResult {
  double total_seconds = 0.0;
  std::vector<GateTiming> timings;  // empty if time_each_gate == false
};

RunResult run(StateVector& s, const std::vector<Gate>& gates, const RunOptions& opt = {});

// CSV: index,gate,q0,q1,theta,seconds,bytes,gbps
void write_timings_csv(std::ostream& out, const RunResult& r);
// Human-readable per-gate-type table (count, total s, mean s, mean GB/s).
void print_summary(std::ostream& out, const RunResult& r, int nq);

}  // namespace qsim

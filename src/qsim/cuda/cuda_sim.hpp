#pragma once
// CUDA backend: runs a whole circuit on one GPU.
//
// The state is copied to the device once, every gate is one kernel launch
// (same dispatch as apply_gate() in simulator.cpp), and the state is copied
// back at the end. This header is plain C++ so the app can be compiled by the
// host compiler; all CUDA code lives in cuda_sim.cu.
#include <string>
#include <vector>

#include "qsim/gates.hpp"
#include "qsim/simulator.hpp"
#include "qsim/state.hpp"

namespace qsim::cuda {

struct CudaRunResult {
  RunResult run;             // run.total_seconds = gate kernels only (comparable to the CPU total_s)
  double h2d_seconds = 0.0;  // host -> device copy of the initial state
  double d2h_seconds = 0.0;  // device -> host copy of the final state
  std::string device_name;
};

// Simulate `gates` on `s` (which holds the initial state) and write the final
// state back into `s`. Timings use cudaEvents.
CudaRunResult run(StateVector& s, const std::vector<Gate>& gates, const RunOptions& opt = {});

}  // namespace qsim::cuda

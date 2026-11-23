// Run with likwid-perfctr -m. Each process measures one warmed kernel.
#include <chrono>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <likwid-marker.h>
#include <omp.h>
#include <sched.h>

#include "qsim/simulator.hpp"
#include "stream_loops.hpp"

namespace {

std::vector<int> worker_cpus() {
  std::vector<int> cpus(omp_get_max_threads(), -1);
#pragma omp parallel
  {
    cpus[omp_get_thread_num()] = sched_getcpu();
  }
  for (int cpu : cpus)
    if (cpu < 0) throw std::runtime_error("OpenMP team/CPU detection failed");
  return cpus;
}

template<class Kernel>
void measure(const char* region, int passes, double bytes_per_pass, double flops_per_pass,
             const Kernel& kernel) {
  const auto cpus = worker_cpus();
  LIKWID_MARKER_INIT;
#pragma omp parallel
  {
    LIKWID_MARKER_THREADINIT;
    LIKWID_MARKER_REGISTER(region);
#pragma omp barrier
    LIKWID_MARKER_START(region);
  }
  using clock = std::chrono::steady_clock;
  const auto start = clock::now();
  for (int i = 0; i < passes; ++i) kernel();
  const double seconds = std::chrono::duration<double>(clock::now() - start).count();
#pragma omp parallel
  {
    LIKWID_MARKER_STOP(region);
  }
  LIKWID_MARKER_CLOSE;
  std::cout << std::setprecision(12) << "PROFILE kernel=" << region
            << " threads=" << cpus.size() << " passes=" << passes << " seconds=" << seconds
            << " model_bytes=" << bytes_per_pass * passes
            << " model_flops=" << flops_per_pass * passes << '\n';
}

void profile_stream(int array_mib, int passes) {
  const long n = static_cast<long>(array_mib) * (1L << 20) / sizeof(double);
  std::unique_ptr<double[]> a(new double[n]), b(new double[n]), c(new double[n]);
  stream::initialize(a.get(), b.get(), c.get(), n);
  auto triad = [&]() {
    stream::triad(a.get(), b.get(), c.get(), n);
    a.swap(c);
  };
  triad();
  measure("stream", passes, 3.0 * sizeof(double) * n, 2.0 * n, triad);
  volatile double sink = c[n / 2];
  (void)sink;
}

void profile_gate(const std::string& name, int nq, int passes) {
  using namespace qsim;
  Gate gate{};
  double flops_per_amplitude;
  if (name == "h") {
    gate.kind = GateKind::H;
    flops_per_amplitude = 14.0;
  } else if (name == "rz") {
    gate.kind = GateKind::RZ;
    gate.theta = 0.37;
    flops_per_amplitude = 6.0;
  } else if (name == "cx") {
    gate.kind = GateKind::CX;
    gate.q0 = 1;
    gate.q1 = 0;
    flops_per_amplitude = 7.0;
  } else {
    throw std::invalid_argument("expected h, rz, cx or stream");
  }
  StateVector state(nq);
  Gate prepare{};
  prepare.kind = GateKind::H;
  prepare.q0 = 1;
  apply_gate(state, prepare);
  auto apply = [&]() { apply_gate(state, gate); };
  apply();
  measure(name.c_str(), passes, bytes_touched(gate, nq), flops_per_amplitude * state.size(), apply);
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc == 2 && std::string(argv[1]) == "--cpus") {
      const auto cpus = worker_cpus();
      for (std::size_t i = 0; i < cpus.size(); ++i) std::cout << (i == 0 ? "" : ",") << cpus[i];
      std::cout << '\n';
      return 0;
    }
    if (argc != 4) throw std::invalid_argument("usage: qsim_cpu_profile h|rz|cx|stream QUBITS_OR_ARRAY_MIB PASSES");
    const std::string name = argv[1];
    const int size = std::stoi(argv[2]), passes = std::stoi(argv[3]);
    if (size < 2 || passes < 1) throw std::invalid_argument("size must be at least 2 and passes must be positive");
    if (name == "stream") profile_stream(size, passes);
    else profile_gate(name, size, passes);
  } catch (const std::exception& error) {
    std::cerr << "qsim_cpu_profile: " << error.what() << '\n';
    return 1;
  }
}

// qsim_cuda: single-GPU state-vector simulator command-line driver.
// Same options and SUMMARY line as qsim (src/apps/qsim_main.cpp), plus the
// host<->device copy times h2d_s / d2h_s. total_s is gate time only.
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

#include "qsim/circuit.hpp"
#include "qsim/cuda/cuda_sim.hpp"
#include "qsim/fusion.hpp"
#include "qsim/simulator.hpp"
#include "qsim/state.hpp"

namespace {

void usage(const char* argv0) {
  std::cerr << "usage: " << argv0 << " CIRCUIT [options]\n"
            << "  --timing FILE      write per-gate timings as CSV\n"
            << "  --dump FILE        write final state (raw complex<double>, index order)\n"
            << "  --fusion MODE      none (default) | 1q\n"
            << "  --no-gate-timing   only time the whole circuit\n"
            << "  --quiet            print only the SUMMARY line\n";
}

}  // namespace

int main(int argc, char** argv) {
  std::string circuit_path, timing_path, dump_path, fusion_str = "none";
  bool gate_timing = true, quiet = false;

  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&]() -> std::string {
      if (i + 1 >= argc) { usage(argv[0]); std::exit(2); }
      return argv[++i];
    };
    if (a == "--timing") timing_path = next();
    else if (a == "--dump") dump_path = next();
    else if (a == "--fusion") fusion_str = next();
    else if (a == "--no-gate-timing") gate_timing = false;
    else if (a == "--quiet") quiet = true;
    else if (a == "-h" || a == "--help") { usage(argv[0]); return 0; }
    else if (!a.empty() && a[0] == '-') { std::cerr << "unknown option " << a << "\n"; usage(argv[0]); return 2; }
    else if (circuit_path.empty()) circuit_path = a;
    else { usage(argv[0]); return 2; }
  }
  if (circuit_path.empty()) { usage(argv[0]); return 2; }

  try {
    qsim::FusionMode fmode;
    if (!qsim::parse_fusion_mode(fusion_str, fmode)) throw std::runtime_error("bad --fusion value: " + fusion_str);

    const qsim::Circuit circ = qsim::read_circuit_file(circuit_path);
    const auto gates = qsim::fuse(circ.gates, circ.num_qubits, fmode);

    qsim::StateVector state(circ.num_qubits);
    qsim::RunOptions opt;
    opt.time_each_gate = gate_timing;
    const auto cres = qsim::cuda::run(state, gates, opt);
    const auto& res = cres.run;

    if (!quiet) std::cout << "device " << cres.device_name << "\n";

    if (!quiet) qsim::print_summary(std::cout, res, circ.num_qubits);
    if (!timing_path.empty()) {
      std::ofstream f(timing_path);
      if (!f) throw std::runtime_error("cannot open " + timing_path);
      qsim::write_timings_csv(f, res);
    }
    if (!dump_path.empty()) state.dump_binary(dump_path);

    const double norm_err = std::abs(state.norm2() - 1.0);
    // One machine-readable line for the run harness / result parsers.
    std::cout << std::setprecision(9) << "SUMMARY qubits=" << circ.num_qubits
              << " gates_in=" << circ.gates.size() << " gates_run=" << gates.size()
              << " fusion=" << fusion_str << " total_s=" << res.total_seconds
              << " mean_gate_s=" << (gates.empty() ? 0.0 : res.total_seconds / gates.size())
              << " norm_err=" << norm_err << " h2d_s=" << cres.h2d_seconds
              << " d2h_s=" << cres.d2h_seconds << "\n";
    if (norm_err > 1e-8) {
      std::cerr << "warning: state norm drifted by " << norm_err << "\n";
      return 3;
    }
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
  return 0;
}

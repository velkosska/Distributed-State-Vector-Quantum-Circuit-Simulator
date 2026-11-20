#include <mpi.h>

#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

#if defined(QSIM_USE_OPENMP)
#include <omp.h>
#endif

#include "qsim/circuit.hpp"
#include "qsim/fusion.hpp"
#include "qsim/mpi/dist_sim.hpp"

namespace {

void usage(const char* argv0) {
  std::cerr << "usage: mpirun -np P " << argv0 << " CIRCUIT [options]     (P a power of two)\n"
            << "  --timing FILE        write per-gate timings of rank 0 as CSV\n"
            << "  --dump FILE          write final state (raw complex<double>, index order; MPI-IO)\n"
            << "  --fusion MODE        none (default) | 1q\n"
            << "  --no-remap           turn qubit remapping off (sensitivity runs)\n"
            << "  --remap              turn qubit remapping on (default)\n"
            << "  --chunk-mib N        exchange chunk per message in MiB (default 64)\n"
            << "  --no-gate-timing     only time the whole circuit\n"
            << "  --quiet              print only the SUMMARY line\n";
}

int run(int argc, char** argv, int rank) {
  std::string circuit_path, timing_path, dump_path, fusion_str = "none";
  bool quiet = false;
  double chunk_mib = 64.0;
  qsim::dist::Options opt;

  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&]() -> std::string {
      if (i + 1 >= argc) { if (rank == 0) usage(argv[0]); std::exit(2); }
      return argv[++i];
    };
    if (a == "--timing") timing_path = next();
    else if (a == "--dump") dump_path = next();
    else if (a == "--fusion") fusion_str = next();
    else if (a == "--no-remap") opt.remap = false;
    else if (a == "--remap") opt.remap = true;
    else if (a == "--chunk-mib") chunk_mib = std::stod(next());
    else if (a == "--no-gate-timing") opt.time_each_gate = false;
    else if (a == "--quiet") quiet = true;
    else if (a == "-h" || a == "--help") { if (rank == 0) usage(argv[0]); return 0; }
    else if (!a.empty() && a[0] == '-') {
      if (rank == 0) { std::cerr << "unknown option " << a << "\n"; usage(argv[0]); }
      return 2;
    }
    else if (circuit_path.empty()) circuit_path = a;
    else { if (rank == 0) usage(argv[0]); return 2; }
  }
  if (circuit_path.empty()) { if (rank == 0) usage(argv[0]); return 2; }
  if (!(chunk_mib > 0)) throw std::runtime_error("--chunk-mib must be positive");
  opt.chunk_amps = static_cast<qsim::index>(chunk_mib * (1 << 20) / sizeof(qsim::cplx));

  qsim::FusionMode fmode;
  if (!qsim::parse_fusion_mode(fusion_str, fmode)) throw std::runtime_error("bad --fusion value: " + fusion_str);

  const qsim::Circuit circ = qsim::read_circuit_file(circuit_path);
  const auto gates = qsim::fuse(circ.gates, circ.num_qubits, fmode);

  qsim::dist::DistState state(circ.num_qubits, MPI_COMM_WORLD, opt);
  const auto st = state.run(gates);

  double comm_max = 0.0;
  MPI_Reduce(&st.comm_bytes, &comm_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

  if (rank == 0 && !quiet) qsim::print_summary(std::cout, qsim::dist::to_run_result(st), circ.num_qubits);
  if (rank == 0 && !timing_path.empty()) {
    std::ofstream f(timing_path);
    if (!f) throw std::runtime_error("cannot open " + timing_path);
    qsim::dist::write_timings_csv(f, st);
  }
  if (!dump_path.empty()) state.dump_binary(dump_path);

  const double norm_err = std::abs(state.norm2() - 1.0);
  int threads = 1;
#if defined(QSIM_USE_OPENMP)
  threads = omp_get_max_threads();
#endif
  if (rank == 0) {
    const double buffer_factor =
        (state.num_ranks() > 1 ? 2.0 : 0.0) * double(state.chunk_amps()) / double(qsim::pow2(state.num_local()));
    std::cout << std::setprecision(9) << "SUMMARY qubits=" << circ.num_qubits
              << " gates_in=" << circ.gates.size() << " gates_run=" << gates.size()
              << " fusion=" << fusion_str << " total_s=" << st.total_seconds
              << " mean_gate_s=" << (gates.empty() ? 0.0 : st.total_seconds / gates.size())
              << " norm_err=" << norm_err << " ranks=" << state.num_ranks() << " threads=" << threads
              << " local_qubits=" << state.num_local() << " remap=" << (opt.remap ? "on" : "off")
              << " comm_gates=" << st.comm_gates << " remaps=" << st.remaps
              << " comm_GiB_per_rank=" << comm_max / double(1 << 30)
              << " chunk_MiB=" << double(state.chunk_amps()) * sizeof(qsim::cplx) / (1 << 20)
              << " buffer_factor=" << buffer_factor << " restore_s=" << st.restore_seconds << "\n";
  }
  if (norm_err > 1e-8) {
    if (rank == 0) std::cerr << "warning: state norm drifted by " << norm_err << "\n";
    return 3;
  }
  return 0;
}

}

int main(int argc, char** argv) {
  int provided = 0;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  int rc = 0;
  try {
    rc = run(argc, argv, rank);
  } catch (const std::exception& e) {
    std::cerr << "error (rank " << rank << "): " << e.what() << "\n";
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  MPI_Finalize();
  return rc;
}

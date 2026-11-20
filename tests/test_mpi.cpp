#include <mpi.h>

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "qsim/mpi/dist_sim.hpp"
#include "qsim/simulator.hpp"

using namespace qsim;

namespace {

std::vector<Gate> random_circuit(int n, int depth, unsigned seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> ang(-3.0, 3.0);
  const GateKind kinds[] = {GateKind::H,  GateKind::X,  GateKind::Y,  GateKind::Z,   GateKind::S,
                            GateKind::SDG, GateKind::T, GateKind::TDG, GateKind::P,  GateKind::RX,
                            GateKind::RY, GateKind::RZ, GateKind::CX, GateKind::CY,  GateKind::CZ,
                            GateKind::CP, GateKind::SWAP, GateKind::U};
  std::uniform_int_distribution<int> pick_kind(0, int(std::size(kinds)) - 1), pick_q(0, n - 1);
  std::vector<Gate> gates;
  for (int q = 0; q < n; ++q) gates.push_back({GateKind::H, q, -1, 0.0, {}});
  for (int d = 0; d < depth; ++d) {
    Gate g{kinds[pick_kind(rng)], pick_q(rng), -1, 0.0, {}};
    if (num_qubits(g.kind) == 2) {
      do g.q1 = pick_q(rng); while (g.q1 == g.q0);
    }
    if (num_params(g.kind) == 1) g.theta = ang(rng);
    if (g.kind == GateKind::U) {
      Gate a{GateKind::RY, 0, -1, ang(rng), {}}, b{GateKind::RZ, 0, -1, ang(rng), {}};
      const Mat2 ma = matrix_1q(a), mb = matrix_1q(b);
      g.m = {mb[0] * ma[0] + mb[1] * ma[2], mb[0] * ma[1] + mb[1] * ma[3],
             mb[2] * ma[0] + mb[3] * ma[2], mb[2] * ma[1] + mb[3] * ma[3]};
    }
    gates.push_back(g);
  }
  return gates;
}

}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int failures = 0, cases = 0;
  double worst = 0.0;
  for (int n : {4, 5, 7, 9}) {
    if ((1 << (n - 1)) < size) continue;
    for (unsigned seed = 1; seed <= 6; ++seed) {
      const auto gates = random_circuit(n, 120, seed * 31 + unsigned(n));
      StateVector ref(n);
      run(ref, gates, {false});
      for (bool remap : {false, true}) {
        for (qsim::index chunk : {qsim::index{1}, qsim::index{3}, qsim::index{1} << 20}) {
          dist::Options opt;
          opt.remap = remap;
          opt.time_each_gate = false;
          opt.chunk_amps = chunk;
          dist::DistState st(n, MPI_COMM_WORLD, opt);
          st.run(gates);
          std::vector<cplx> all(rank == 0 ? ref.size() : 0);
          MPI_Gather(st.local().data(), int(2 * st.local().size()), MPI_DOUBLE, all.data(),
                     int(2 * st.local().size()), MPI_DOUBLE, 0, MPI_COMM_WORLD);
          if (rank == 0) {
            double err = 0.0;
            for (qsim::index i = 0; i < ref.size(); ++i) err = std::max(err, std::abs(all[i] - ref[i]));
            worst = std::max(worst, err);
            ++cases;
            if (!(err < 1e-12)) {
              ++failures;
              std::printf("FAIL n=%d seed=%u remap=%d chunk=%llu ranks=%d err=%.3e\n", n, seed, remap,
                          (unsigned long long)chunk, size, err);
            }
          }
        }
      }
    }
  }
  if (rank == 0) {
    if (failures == 0) std::printf("test_mpi: %d cases on %d ranks ok (max err %.2e)\n", cases, size, worst);
    else std::printf("test_mpi: %d of %d cases FAILED on %d ranks\n", failures, cases, size);
  }
  MPI_Bcast(&failures, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Finalize();
  return failures == 0 ? 0 : 1;
}

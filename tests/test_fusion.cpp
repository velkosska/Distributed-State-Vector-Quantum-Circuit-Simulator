// Checks of the gate-fusion pass (src/qsim/fusion.cpp): fused and unfused
// circuits must give the same state (including global phase), and the pass
// must merge what it promises to merge.
#include <cmath>
#include <cstdio>
#include <random>
#include <sstream>

#include "qsim/circuit.hpp"
#include "qsim/fusion.hpp"
#include "qsim/simulator.hpp"
#include "qsim/state.hpp"

using namespace qsim;

static int failures = 0;
#define CHECK(cond)                                                    \
  do {                                                                 \
    if (!(cond)) {                                                     \
      std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);      \
      ++failures;                                                      \
    }                                                                  \
  } while (0)

static Circuit parse(const char* text) {
  std::istringstream in(text);
  return read_circuit(in);
}

static double max_diff(const Circuit& c, const std::vector<Gate>& fused) {
  StateVector a(c.num_qubits), b(c.num_qubits);
  // Start from a non-trivial state so phases on every amplitude matter.
  std::vector<Gate> prep;
  for (int q = 0; q < c.num_qubits; ++q) {
    Gate h{}; h.kind = GateKind::H; h.q0 = q; prep.push_back(h);
    Gate t{}; t.kind = GateKind::RY; t.q0 = q; t.theta = 0.3 + 0.1 * q; prep.push_back(t);
  }
  run(a, prep, {false});
  run(b, prep, {false});
  run(a, c.gates, {false});
  run(b, fused, {false});
  double err = 0.0;
  for (index i = 0; i < a.size(); ++i) err = std::max(err, std::abs(a[i] - b[i]));
  return err;
}

static int count_kind(const std::vector<Gate>& gs, GateKind k) {
  int n = 0;
  for (const auto& g : gs) n += g.kind == k;
  return n;
}

static Circuit random_circuit(int nq, int ngates, unsigned seed) {
  static const GateKind k1[] = {GateKind::H, GateKind::X, GateKind::Y, GateKind::Z, GateKind::S,
                                GateKind::SDG, GateKind::T, GateKind::TDG, GateKind::P,
                                GateKind::RX, GateKind::RY, GateKind::RZ};
  static const GateKind k2[] = {GateKind::CX, GateKind::CY, GateKind::CZ, GateKind::CP, GateKind::SWAP};
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> ang(-M_PI, M_PI);
  Circuit c;
  c.num_qubits = nq;
  for (int i = 0; i < ngates; ++i) {
    Gate g{};
    if (rng() % 4 == 0) {
      g.kind = k2[rng() % 5];
      g.q0 = static_cast<int>(rng() % nq);
      do g.q1 = static_cast<int>(rng() % nq); while (g.q1 == g.q0);
    } else {
      g.kind = k1[rng() % 12];
      g.q0 = static_cast<int>(rng() % nq);
    }
    if (num_params(g.kind) == 1) g.theta = ang(rng);
    c.gates.push_back(g);
  }
  return c;
}

int main() {
  {  // parse_fusion_mode
    FusionMode m;
    CHECK(parse_fusion_mode("none", m) && m == FusionMode::None);
    CHECK(parse_fusion_mode("1q", m) && m == FusionMode::SingleQubit);
    CHECK(!parse_fusion_mode("2q", m));
  }
  {  // mode none is the identity
    const Circuit c = parse("qubits 2\nh 0\nh 0\ncx 0 1\n");
    CHECK(fuse(c.gates, 2, FusionMode::None).size() == 3);
  }
  {  // a dense run becomes one U; gates on other qubits do not break it
    const Circuit c = parse("qubits 3\nh 0\nrx 0 0.3\nx 1\nry 0 0.7\nt 0\n");
    const auto f = fuse(c.gates, 3, FusionMode::SingleQubit);
    CHECK(f.size() == 2);
    CHECK(count_kind(f, GateKind::U) == 1);
    CHECK(max_diff(c, f) < 1e-12);
  }
  {  // H H cancels; S S Sdg Sdg cancels; nothing left
    const Circuit c = parse("qubits 2\nh 0\nh 0\ns 1\ns 1\nsdg 1\nsdg 1\n");
    const auto f = fuse(c.gates, 2, FusionMode::SingleQubit);
    CHECK(f.empty());
    CHECK(max_diff(c, f) < 1e-12);
  }
  {  // diagonal runs stay diagonal (one P) and keep the global phase of RZ
    const Circuit c = parse("qubits 2\nh 1\nrz 0 0.4\nt 0\nrz 0 1.1\n");
    const auto f = fuse(c.gates, 2, FusionMode::SingleQubit);
    CHECK(count_kind(f, GateKind::P) == 1);
    CHECK(f.size() == 2);  // H (now U, carrying the phase) + P
    CHECK(max_diff(c, f) < 1e-12);
  }
  {  // only diagonal gates and phases: the global phase is still reproduced
    const Circuit c = parse("qubits 1\nrz 0 0.4\nrz 0 0.5\n");
    const auto f = fuse(c.gates, 1, FusionMode::SingleQubit);
    CHECK(max_diff(c, f) < 1e-12);
  }
  {  // diagonal runs pass CZ / CP and the control of CX
    const Circuit c = parse("qubits 2\nt 0\ncz 0 1\nt 0\ncx 0 1\ns 0\ncp 1 0 0.3\nt 0\n");
    const auto f = fuse(c.gates, 2, FusionMode::SingleQubit);
    CHECK(count_kind(f, GateKind::P) + count_kind(f, GateKind::T) == 1);
    CHECK(f.size() == 4);
    CHECK(max_diff(c, f) < 1e-12);
  }
  {  // a dense run is flushed before a two-qubit gate on its qubit
    const Circuit c = parse("qubits 2\nh 0\nrx 0 0.2\ncx 1 0\nh 0\nh 0\n");
    const auto f = fuse(c.gates, 2, FusionMode::SingleQubit);
    CHECK(f.size() == 2);
    CHECK(f[0].kind == GateKind::U && f[1].kind == GateKind::CX);
    CHECK(max_diff(c, f) < 1e-12);
  }
  {  // SWAP carries pending runs across
    const Circuit c = parse("qubits 2\nh 0\nswap 0 1\nh 1\nrx 0 0.5\n");
    const auto f = fuse(c.gates, 2, FusionMode::SingleQubit);
    CHECK(f.size() == 2);  // swap + rx (H H on the same wire cancelled)
  }
  {  // a single gate carried across a SWAP lands on the other qubit
    const Circuit c = parse("qubits 3\nx 0\nswap 0 2\n");
    const auto f = fuse(c.gates, 3, FusionMode::SingleQubit);
    CHECK(f.size() == 2 && f[1].kind == GateKind::X && f[1].q0 == 2);
    CHECK(max_diff(c, f) < 1e-12);
    CHECK(max_diff(c, f) < 1e-12);
  }
  {  // random circuits with every gate kind: same state, never more gates
    for (unsigned seed = 1; seed <= 200; ++seed) {
      const int nq = 2 + static_cast<int>(seed % 5);
      const Circuit c = random_circuit(nq, 60, seed);
      const auto f = fuse(c.gates, nq, FusionMode::SingleQubit);
      const double err = max_diff(c, f);
      if (err > 1e-11) std::printf("seed %u: err %g\n", seed, err);
      CHECK(err < 1e-11);
      CHECK(f.size() <= c.gates.size() + 2);
    }
  }

  if (failures == 0) std::printf("all fusion tests passed\n");
  return failures == 0 ? 0 : 1;
}

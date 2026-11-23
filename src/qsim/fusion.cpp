#include "qsim/fusion.hpp"

#include <cmath>
#include <stdexcept>
#include <utility>

namespace qsim {

namespace {

constexpr double kEps = 1e-14;

// a * b (apply b first, then a).
Mat2 mul(const Mat2& a, const Mat2& b) {
  return {a[0] * b[0] + a[1] * b[2], a[0] * b[1] + a[1] * b[3],
          a[2] * b[0] + a[3] * b[2], a[2] * b[1] + a[3] * b[3]};
}

bool is_dense_1q_kind(GateKind k) {
  using K = GateKind;
  return k == K::H || k == K::X || k == K::Y || k == K::RX || k == K::RY || k == K::U;
}

// Single-qubit gates still waiting to be emitted on one qubit.
struct Pending {
  int count = 0;
  bool diag = true;  // every gate in the run is diagonal
  Mat2 m{cplx(1.0), cplx(0.0), cplx(0.0), cplx(1.0)};
  Gate first{};
};

class Fuser {
 public:
  explicit Fuser(int nq) : pend_(static_cast<std::size_t>(nq)) {}

  void add(const Gate& g) {
    using K = GateKind;
    if (num_qubits(g.kind) == 1) {
      Pending& p = pend_.at(static_cast<std::size_t>(g.q0));
      if (p.count == 0) p.first = g;
      p.m = mul(matrix_1q(g), p.m);
      p.diag = p.diag && is_diagonal(g.kind);
      ++p.count;
      return;
    }
    switch (g.kind) {
      case K::CZ: case K::CP:  // diagonal: commutes with diagonal runs on both qubits
        flush_unless_diag(g.q0);
        flush_unless_diag(g.q1);
        break;
      case K::CX: case K::CY:  // diagonal runs commute with the control only
        flush_unless_diag(g.q0);
        flush(g.q1);
        break;
      case K::SWAP:  // U_a then SWAP == SWAP then U_b
        std::swap(pend_.at(static_cast<std::size_t>(g.q0)), pend_.at(static_cast<std::size_t>(g.q1)));
        break;
      default:
        flush(g.q0);
        flush(g.q1);
        break;
    }
    out_.push_back(g);
  }

  std::vector<Gate> finish() {
    for (int q = 0; q < static_cast<int>(pend_.size()); ++q) flush(q);
    apply_global_phase();
    return std::move(out_);
  }

 private:
  void flush_unless_diag(int q) {
    if (!pend_.at(static_cast<std::size_t>(q)).diag) flush(q);
  }

  void flush(int q) {
    Pending& p = pend_.at(static_cast<std::size_t>(q));
    if (p.count == 0) return;
    const Mat2 m = p.m;
    const bool diag = p.diag;
    const int count = p.count;
    Gate first = p.first;
    p = Pending{};

    if (count == 1) {  // nothing to merge: keep the gate as written
      first.q0 = q;    // a SWAP may have moved the run to another qubit
      out_.push_back(first);
      return;
    }
    const bool numerically_diag = diag || (std::abs(m[1]) < kEps && std::abs(m[2]) < kEps);
    if (numerically_diag) {
      // diag(d0, d1) = d0 * diag(1, d1/d0); d0 has modulus 1 for a unitary run.
      const cplx d0 = m[0], ratio = m[3] / m[0];
      phase_ *= d0 / std::abs(d0);
      if (std::abs(ratio - cplx(1.0)) < kEps) return;  // identity up to phase
      Gate g{};
      g.kind = GateKind::P;
      g.q0 = q;
      g.theta = std::arg(ratio);
      out_.push_back(g);
      return;
    }
    Gate g{};
    g.kind = GateKind::U;
    g.q0 = q;
    g.m = m;
    out_.push_back(g);
  }

  // Put the collected global phase back so the state is reproduced exactly.
  void apply_global_phase() {
    if (std::abs(phase_ - cplx(1.0)) < kEps) return;
    for (auto it = out_.rbegin(); it != out_.rend(); ++it) {
      if (!is_dense_1q_kind(it->kind)) continue;
      const Mat2 m = matrix_1q(*it);
      it->kind = GateKind::U;
      it->theta = 0.0;
      for (int k = 0; k < 4; ++k) it->m[k] = phase_ * m[k];
      return;
    }
    // No dense gate to carry it: RZ(-2a) P(2a) = e^{ia} I on qubit 0.
    const double a = std::arg(phase_);
    Gate rz{};
    rz.kind = GateKind::RZ;
    rz.q0 = 0;
    rz.theta = -2.0 * a;
    Gate p{};
    p.kind = GateKind::P;
    p.q0 = 0;
    p.theta = 2.0 * a;
    out_.push_back(rz);
    out_.push_back(p);
  }

  std::vector<Pending> pend_;
  std::vector<Gate> out_;
  cplx phase_{1.0, 0.0};
};

}  // namespace

bool parse_fusion_mode(const std::string& s, FusionMode& out) {
  if (s == "none") { out = FusionMode::None; return true; }
  if (s == "1q") { out = FusionMode::SingleQubit; return true; }
  return false;
}

std::vector<Gate> fuse(const std::vector<Gate>& gates, int num_qubits, FusionMode mode) {
  switch (mode) {
    case FusionMode::None:
      return gates;
    case FusionMode::SingleQubit: {
      if (num_qubits < 1) throw std::invalid_argument("fuse: num_qubits must be >= 1");
      Fuser f(num_qubits);
      for (const auto& g : gates) f.add(g);
      return f.finish();
    }
  }
  return gates;
}

}  // namespace qsim

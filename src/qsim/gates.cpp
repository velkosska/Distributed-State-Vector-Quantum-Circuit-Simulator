#include "qsim/gates.hpp"

#include <cmath>
#include <stdexcept>

namespace qsim {

namespace {
struct Info {
  GateKind kind;
  const char* name;
  int nq;
  int np;
  bool diag;
};
constexpr Info kInfo[] = {
    {GateKind::H, "h", 1, 0, false},     {GateKind::X, "x", 1, 0, false},
    {GateKind::Y, "y", 1, 0, false},     {GateKind::Z, "z", 1, 0, true},
    {GateKind::S, "s", 1, 0, true},      {GateKind::SDG, "sdg", 1, 0, true},
    {GateKind::T, "t", 1, 0, true},      {GateKind::TDG, "tdg", 1, 0, true},
    {GateKind::P, "p", 1, 1, true},      {GateKind::RX, "rx", 1, 1, false},
    {GateKind::RY, "ry", 1, 1, false},   {GateKind::RZ, "rz", 1, 1, true},
    {GateKind::U, "u", 1, 0, false},     {GateKind::CX, "cx", 2, 0, false},
    {GateKind::CY, "cy", 2, 0, false},   {GateKind::CZ, "cz", 2, 0, true},
    {GateKind::CP, "cp", 2, 1, true},    {GateKind::SWAP, "swap", 2, 0, false},
};
const Info& info(GateKind k) {
  for (const auto& i : kInfo)
    if (i.kind == k) return i;
  throw std::logic_error("unknown gate kind");
}
}  // namespace

int num_qubits(GateKind k) { return info(k).nq; }
int num_params(GateKind k) { return info(k).np; }
bool is_diagonal(GateKind k) { return info(k).diag; }
const char* gate_name(GateKind k) { return info(k).name; }

bool parse_gate_name(const std::string& s, GateKind& out) {
  for (const auto& i : kInfo)
    if (s == i.name) {
      out = i.kind;
      return true;
    }
  if (s == "cnot") { out = GateKind::CX; return true; }
  return false;
}

Mat2 matrix_1q(const Gate& g) {
  const double r = 1.0 / std::sqrt(2.0);
  const cplx I(0.0, 1.0);
  const double h = g.theta / 2.0;
  switch (g.kind) {
    case GateKind::H: return {r, r, r, -r};
    case GateKind::X: case GateKind::CX: return {0.0, 1.0, 1.0, 0.0};
    case GateKind::Y: case GateKind::CY: return {0.0, -I, I, 0.0};
    case GateKind::Z: case GateKind::CZ: return {1.0, 0.0, 0.0, -1.0};
    case GateKind::S: return {1.0, 0.0, 0.0, I};
    case GateKind::SDG: return {1.0, 0.0, 0.0, -I};
    case GateKind::T: return {1.0, 0.0, 0.0, std::polar(1.0, M_PI / 4)};
    case GateKind::TDG: return {1.0, 0.0, 0.0, std::polar(1.0, -M_PI / 4)};
    case GateKind::P: case GateKind::CP: return {1.0, 0.0, 0.0, std::polar(1.0, g.theta)};
    case GateKind::RX: return {std::cos(h), -I * std::sin(h), -I * std::sin(h), std::cos(h)};
    case GateKind::RY: return {std::cos(h), -std::sin(h), std::sin(h), std::cos(h)};
    case GateKind::RZ: return {std::polar(1.0, -h), 0.0, 0.0, std::polar(1.0, h)};
    case GateKind::U: return g.m;
    case GateKind::SWAP: break;
  }
  throw std::logic_error("matrix_1q: gate has no 2x2 matrix");
}

double bytes_touched(const Gate& g, int nq) {
  const double all = 2.0 * sizeof(cplx) * static_cast<double>(pow2(nq));  // read + write every amplitude
  switch (g.kind) {
    case GateKind::Z: case GateKind::S: case GateKind::SDG:
    case GateKind::T: case GateKind::TDG: case GateKind::P:
      return all / 2;  // only the |1> half
    case GateKind::CX: case GateKind::CY: case GateKind::SWAP:
      return all / 2;  // half of the amplitudes move
    case GateKind::CZ: case GateKind::CP:
      return all / 4;  // only |11>
    default:
      return all;
  }
}

}  // namespace qsim

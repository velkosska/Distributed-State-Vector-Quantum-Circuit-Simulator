#include "qsim/circuit.hpp"

#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace qsim {

namespace {
[[noreturn]] void fail(const std::string& src, int line, const std::string& msg) {
  throw std::runtime_error(src + ":" + std::to_string(line) + ": " + msg);
}
}  // namespace

Circuit read_circuit(std::istream& in, const std::string& src) {
  Circuit c;
  std::string line;
  int lineno = 0;
  while (std::getline(in, line)) {
    ++lineno;
    if (auto pos = line.find('#'); pos != std::string::npos) line.erase(pos);
    std::istringstream ss(line);
    std::string word;
    if (!(ss >> word)) continue;

    if (word == "qubits") {
      if (c.num_qubits != 0) fail(src, lineno, "duplicate 'qubits' line");
      if (!(ss >> c.num_qubits) || c.num_qubits < 1 || c.num_qubits > 40)
        fail(src, lineno, "expected 'qubits N' with 1 <= N <= 40");
      continue;
    }
    if (c.num_qubits == 0) fail(src, lineno, "'qubits N' must come before the first gate");

    Gate g{};
    if (!parse_gate_name(word, g.kind)) fail(src, lineno, "unknown gate '" + word + "'");

    const int nq = num_qubits(g.kind);
    if (!(ss >> g.q0)) fail(src, lineno, "missing qubit index");
    if (nq == 2 && !(ss >> g.q1)) fail(src, lineno, "missing second qubit index");
    for (int q : {g.q0, g.q1}) {
      if (q == -1 && nq == 1) continue;
      if (q < 0 || q >= c.num_qubits) fail(src, lineno, "qubit index out of range");
    }
    if (nq == 2 && g.q0 == g.q1) fail(src, lineno, "two-qubit gate on the same qubit");

    if (num_params(g.kind) == 1 && !(ss >> g.theta)) fail(src, lineno, "missing angle");
    if (g.kind == GateKind::U) {
      for (auto& e : g.m) {
        double re, im;
        if (!(ss >> re >> im)) fail(src, lineno, "'u' needs 8 numbers: re/im of m00 m01 m10 m11");
        e = cplx(re, im);
      }
    }
    std::string extra;
    if (ss >> extra) fail(src, lineno, "unexpected token '" + extra + "'");
    c.gates.push_back(g);
  }
  if (c.num_qubits == 0) fail(src, lineno, "no 'qubits N' line");
  return c;
}

Circuit read_circuit_file(const std::string& path) {
  std::ifstream f(path);
  if (!f) throw std::runtime_error("cannot open circuit file: " + path);
  return read_circuit(f, path);
}

void write_circuit(std::ostream& out, const Circuit& c) {
  out << std::setprecision(17);
  out << "qubits " << c.num_qubits << "\n";
  for (const auto& g : c.gates) {
    out << gate_name(g.kind) << ' ' << g.q0;
    if (num_qubits(g.kind) == 2) out << ' ' << g.q1;
    if (num_params(g.kind) == 1) out << ' ' << g.theta;
    if (g.kind == GateKind::U)
      for (const auto& e : g.m) out << ' ' << e.real() << ' ' << e.imag();
    out << "\n";
  }
}

}  // namespace qsim

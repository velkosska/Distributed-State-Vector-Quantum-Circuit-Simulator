#include "qsim/simulator.hpp"

#include <chrono>
#include <iomanip>
#include <map>
#include <ostream>
#include <stdexcept>

#include "qsim/kernels.hpp"

namespace qsim {

void apply_gate(cplx* amp, int nq, const Gate& g) {
  using K = GateKind;
  switch (g.kind) {
    case K::H: case K::X: case K::Y: case K::RX: case K::RY: case K::U:
      kern::apply_1q(amp, nq, g.q0, matrix_1q(g));
      return;
    case K::Z: case K::S: case K::SDG: case K::T: case K::TDG: case K::P: case K::RZ: {
      const Mat2 m = matrix_1q(g);
      kern::apply_1q_diag(amp, nq, g.q0, m[0], m[3]);
      return;
    }
    case K::CX: case K::CY:
      kern::apply_controlled_1q(amp, nq, g.q0, g.q1, matrix_1q(g));
      return;
    case K::CZ:
      kern::apply_controlled_phase(amp, nq, g.q0, g.q1, cplx(-1.0, 0.0));
      return;
    case K::CP:
      kern::apply_controlled_phase(amp, nq, g.q0, g.q1, std::polar(1.0, g.theta));
      return;
    case K::SWAP:
      kern::apply_swap(amp, nq, g.q0, g.q1);
      return;
  }
  throw std::logic_error("apply_gate: unhandled gate");
}

RunResult run(StateVector& s, const std::vector<Gate>& gates, const RunOptions& opt) {
  using clock = std::chrono::steady_clock;
  RunResult r;
  const int nq = s.num_qubits();
  if (opt.time_each_gate) r.timings.reserve(gates.size());
  const auto t_start = clock::now();
  for (const auto& g : gates) {
    if (opt.time_each_gate) {
      const auto t0 = clock::now();
      apply_gate(s, g);
      const auto t1 = clock::now();
      r.timings.push_back({g, std::chrono::duration<double>(t1 - t0).count(), bytes_touched(g, nq)});
    } else {
      apply_gate(s, g);
    }
  }
  r.total_seconds = std::chrono::duration<double>(clock::now() - t_start).count();
  return r;
}

void write_timings_csv(std::ostream& out, const RunResult& r) {
  out << "index,gate,q0,q1,theta,seconds,bytes,gbps\n";
  out << std::setprecision(9);
  for (std::size_t i = 0; i < r.timings.size(); ++i) {
    const auto& t = r.timings[i];
    out << i << ',' << gate_name(t.gate.kind) << ',' << t.gate.q0 << ',' << t.gate.q1 << ','
        << t.gate.theta << ',' << t.seconds << ',' << t.bytes << ','
        << (t.seconds > 0 ? t.bytes / t.seconds * 1e-9 : 0.0) << '\n';
  }
}

void print_summary(std::ostream& out, const RunResult& r, int nq) {
  struct Agg { std::size_t n = 0; double sec = 0, bytes = 0; };
  std::map<std::string, Agg> by_kind;
  for (const auto& t : r.timings) {
    auto& a = by_kind[gate_name(t.gate.kind)];
    ++a.n;
    a.sec += t.seconds;
    a.bytes += t.bytes;
  }
  out << "qubits " << nq << ", state " << std::fixed << std::setprecision(3)
      << double(pow2(nq)) * sizeof(cplx) / (1 << 20) << " MiB\n";
  if (!by_kind.empty()) {
    out << std::left << std::setw(6) << "gate" << std::right << std::setw(9) << "count"
        << std::setw(14) << "total_s" << std::setw(14) << "mean_s" << std::setw(10) << "GB/s\n";
    for (const auto& [name, a] : by_kind) {
      out << std::left << std::setw(6) << name << std::right << std::setw(9) << a.n
          << std::setw(14) << std::setprecision(6) << a.sec << std::setw(14) << std::scientific
          << std::setprecision(3) << a.sec / a.n << std::fixed << std::setw(10) << std::setprecision(2)
          << (a.sec > 0 ? a.bytes / a.sec * 1e-9 : 0.0) << '\n';
    }
  }
  out << std::defaultfloat;
}

}  // namespace qsim

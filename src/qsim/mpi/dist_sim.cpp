#include "qsim/mpi/dist_sim.hpp"

#include <algorithm>
#include <iomanip>
#include <ostream>
#include <stdexcept>

#include "qsim/kernels.hpp"

namespace qsim::dist {

namespace {

int log2_exact(int x) {
  int l = 0;
  while ((1 << l) < x) ++l;
  return (1 << l) == x ? l : -1;
}

int comm_size(MPI_Comm c) { int s; MPI_Comm_size(c, &s); return s; }
int comm_rank(MPI_Comm c) { int r; MPI_Comm_rank(c, &r); return r; }

int local_qubits(int n, MPI_Comm c) {
  const int p = log2_exact(comm_size(c));
  if (p < 0) throw std::invalid_argument("number of MPI ranks must be a power of two");
  if (n - p < 1)
    throw std::invalid_argument(std::to_string(comm_size(c)) + " ranks need at least " +
                                std::to_string(p + 1) + " qubits");
  return n - p;
}

bool is_dense_1q(GateKind k) {
  using K = GateKind;
  return k == K::H || k == K::X || k == K::Y || k == K::RX || k == K::RY || k == K::U;
}

constexpr index kMaxMsgAmps = index{1} << 29;

}

DistState::DistState(int num_qubits, MPI_Comm comm, const Options& opt)
    : comm_(comm),
      rank_(comm_rank(comm)),
      size_(comm_size(comm)),
      p_(log2_exact(size_)),
      n_(num_qubits),
      nl_(local_qubits(num_qubits, comm)),
      opt_(opt),
      chunk_(std::min({std::max<index>(opt.chunk_amps, 1), pow2(nl_), kMaxMsgAmps})),
      local_(nl_),
      pos_(num_qubits),
      lq_(num_qubits) {
  if (rank_ != 0) local_[0] = cplx(0.0, 0.0);
  for (int q = 0; q < n_; ++q) pos_[q] = lq_[q] = q;
  if (p_ > 0) {
    sbuf_.resize(chunk_);
    rbuf_.resize(chunk_);
  }
}

void DistState::sendrecv(const cplx* send, cplx* recv, index count, int peer) {
  MPI_Sendrecv(send, static_cast<int>(2 * count), MPI_DOUBLE, peer, 0, recv, static_cast<int>(2 * count),
               MPI_DOUBLE, peer, 0, comm_, MPI_STATUS_IGNORE);
  comm_bytes_ += double(count) * sizeof(cplx);
}

void DistState::exchange_dense_1q(int gpos, const Mat2& m) {
  const int peer = partner(gpos);
  const bool hi = bit(gpos);
  cplx* amp = local_.data();
  const index total = local_.size();
  ++exchanges_;
  for (index s = 0; s < total; s += chunk_) {
    const index c = std::min(chunk_, total - s);
    sendrecv(amp + s, rbuf_.data(), c, peer);
    if (hi) kern::apply_1q_pairs(rbuf_.data(), amp + s, c, m);
    else    kern::apply_1q_pairs(amp + s, rbuf_.data(), c, m);
  }
}

void DistState::exchange_controlled_local(int cpos, int tpos, const Mat2& m) {
  const int peer = partner(tpos);
  const bool hi = bit(tpos);
  const index cbit = pow2(cpos);
  cplx* amp = local_.data();
  cplx* sb = sbuf_.data();
  cplx* rb = rbuf_.data();
  const index total = local_.size() / 2;
  const cplx m00 = m[0], m01 = m[1], m10 = m[2], m11 = m[3];
  ++exchanges_;
  for (index s = 0; s < total; s += chunk_) {
    const index c = std::min(chunk_, total - s);
    QSIM_PRAGMA_FOR
    for (index k = 0; k < c; ++k) sb[k] = amp[insert_zero_bit(s + k, cpos) | cbit];
    sendrecv(sb, rb, c, peer);
    QSIM_PRAGMA_FOR
    for (index k = 0; k < c; ++k) {
      const index i = insert_zero_bit(s + k, cpos) | cbit;
      amp[i] = hi ? m10 * rb[k] + m11 * sb[k] : m00 * sb[k] + m01 * rb[k];
    }
  }
}

void DistState::exchange_swap_local_global(int lpos, int gpos) {
  const int peer = partner(gpos);
  const index sel = bit(gpos) ? 0 : pow2(lpos);
  cplx* amp = local_.data();
  cplx* sb = sbuf_.data();
  cplx* rb = rbuf_.data();
  const index total = local_.size() / 2;
  ++exchanges_;
  for (index s = 0; s < total; s += chunk_) {
    const index c = std::min(chunk_, total - s);
    QSIM_PRAGMA_FOR
    for (index k = 0; k < c; ++k) sb[k] = amp[insert_zero_bit(s + k, lpos) | sel];
    sendrecv(sb, rb, c, peer);
    QSIM_PRAGMA_FOR
    for (index k = 0; k < c; ++k) amp[insert_zero_bit(s + k, lpos) | sel] = rb[k];
  }
}

void DistState::exchange_swap_global_global(int a, int b) {
  if (bit(a) == bit(b)) return;
  const int peer = rank_ ^ (1 << (a - nl_)) ^ (1 << (b - nl_));
  cplx* amp = local_.data();
  const index total = local_.size();
  ++exchanges_;
  for (index s = 0; s < total; s += chunk_) {
    const index c = std::min(chunk_, total - s);
    sendrecv(amp + s, rbuf_.data(), c, peer);
    std::copy(rbuf_.begin(), rbuf_.begin() + static_cast<std::ptrdiff_t>(c), amp + s);
  }
}

void DistState::scale_all(cplx f) {
  if (f == cplx(1.0, 0.0)) return;
  cplx* amp = local_.data();
  const index total = local_.size();
  QSIM_PRAGMA_FOR
  for (index i = 0; i < total; ++i) amp[i] *= f;
}

void DistState::apply_physical(const Gate& g, int a, int b) {
  using K = GateKind;
  cplx* amp = local_.data();
  const bool la = a < nl_, lb = b >= 0 && b < nl_;
  switch (g.kind) {
    case K::H: case K::X: case K::Y: case K::RX: case K::RY: case K::U: {
      const Mat2 m = matrix_1q(g);
      if (la) kern::apply_1q(amp, nl_, a, m);
      else exchange_dense_1q(a, m);
      return;
    }
    case K::Z: case K::S: case K::SDG: case K::T: case K::TDG: case K::P: case K::RZ: {
      const Mat2 m = matrix_1q(g);
      if (la) kern::apply_1q_diag(amp, nl_, a, m[0], m[3]);
      else scale_all(bit(a) ? m[3] : m[0]);
      return;
    }
    case K::CX: case K::CY: {
      const Mat2 m = matrix_1q(g);
      if (la && lb) kern::apply_controlled_1q(amp, nl_, a, b, m);
      else if (!la && lb) { if (bit(a)) kern::apply_1q(amp, nl_, b, m); }
      else if (!la && !lb) { if (bit(a)) exchange_dense_1q(b, m); }
      else exchange_controlled_local(a, b, m);
      return;
    }
    case K::CZ: case K::CP: {
      const cplx ph = g.kind == K::CZ ? cplx(-1.0, 0.0) : std::polar(1.0, g.theta);
      if (la && lb) kern::apply_controlled_phase(amp, nl_, a, b, ph);
      else if (la) { if (bit(b)) kern::apply_1q_diag(amp, nl_, a, cplx(1.0, 0.0), ph); }
      else if (lb) { if (bit(a)) kern::apply_1q_diag(amp, nl_, b, cplx(1.0, 0.0), ph); }
      else if (bit(a) && bit(b)) scale_all(ph);
      return;
    }
    case K::SWAP:
      swap_data(a, b);
      return;
  }
  throw std::logic_error("dist::apply_physical: unhandled gate");
}

void DistState::swap_data(int a, int b) {
  if (a == b) return;
  if (a > b) std::swap(a, b);
  if (b < nl_) kern::apply_swap(local_.data(), nl_, a, b);
  else if (a < nl_) exchange_swap_local_global(a, b);
  else exchange_swap_global_global(a, b);
}

void DistState::swap_positions(int a, int b) {
  swap_data(a, b);
  std::swap(lq_[a], lq_[b]);
  pos_[lq_[a]] = a;
  pos_[lq_[b]] = b;
}

bool DistState::is_comm(const Gate& g, int a, int b) const {
  using K = GateKind;
  if (is_dense_1q(g.kind)) return a >= nl_;
  if (g.kind == K::CX || g.kind == K::CY) return b >= nl_;
  if (g.kind == K::SWAP) return a != b && (a >= nl_ || b >= nl_);
  return false;
}

int DistState::needs_comm_qubit(const Gate& g) const {
  using K = GateKind;
  if (is_dense_1q(g.kind)) return pos_[g.q0] >= nl_ ? g.q0 : -1;
  if (g.kind == K::CX || g.kind == K::CY) return pos_[g.q1] >= nl_ ? g.q1 : -1;
  return -1;
}

int DistState::pick_victim(const std::vector<Gate>& gates, std::size_t i, int protect_logical) const {
  std::vector<char> live(n_, 0);
  int remaining = 0;
  for (int v = 0; v < nl_; ++v) {
    if (lq_[v] == protect_logical) continue;
    live[lq_[v]] = 1;
    ++remaining;
  }
  if (remaining == 0) return -1;
  for (std::size_t j = i + 1; j < gates.size() && remaining > 1; ++j) {
    for (int q : {gates[j].q0, gates[j].q1}) {
      if (q >= 0 && live[q]) {
        live[q] = 0;
        --remaining;
        if (remaining == 1) break;
      }
    }
  }
  for (int q = 0; q < n_; ++q)
    if (live[q]) return pos_[q];
  return -1;
}

Stats DistState::run(const std::vector<Gate>& gates) {
  Stats st;
  comm_bytes_ = 0.0;
  exchanges_ = 0;
  if (opt_.time_each_gate) st.timings.reserve(gates.size());

  MPI_Barrier(comm_);
  const double t_start = MPI_Wtime();
  for (std::size_t i = 0; i < gates.size(); ++i) {
    const Gate& g = gates[i];
    const double t0 = opt_.time_each_gate ? MPI_Wtime() : 0.0;
    const double c0 = comm_bytes_;
    int remapped = 0;

    if (opt_.remap && g.kind == GateKind::SWAP) {
      std::swap(pos_[g.q0], pos_[g.q1]);
      lq_[pos_[g.q0]] = g.q0;
      lq_[pos_[g.q1]] = g.q1;
    } else {
      if (opt_.remap && p_ > 0) {
        const int q = needs_comm_qubit(g);
        if (q >= 0) {
          const bool controlled = g.kind == GateKind::CX || g.kind == GateKind::CY;
          const int v = pick_victim(gates, i, controlled ? g.q0 : -1);
          if (v >= 0) {
            swap_positions(v, pos_[q]);
            ++remapped;
          }
        }
      }
      const int a = pos_[g.q0], b = g.q1 >= 0 ? pos_[g.q1] : -1;
      if (is_comm(g, a, b)) ++st.comm_gates;
      apply_physical(g, a, b);
    }
    st.remaps += remapped;

    if (opt_.time_each_gate) {
      GateRecord r;
      r.gate = g;
      r.seconds = MPI_Wtime() - t0;
      r.bytes = bytes_touched(g, nl_);
      r.comm_bytes = comm_bytes_ - c0;
      r.remapped = remapped;
      st.timings.push_back(r);
    }
  }

  const double t_restore = MPI_Wtime();
  for (int t = 0; t < n_; ++t)
    if (lq_[t] != t) swap_positions(t, pos_[t]);
  const double t_end = MPI_Wtime();

  double local_total = t_end - t_start, local_restore = t_end - t_restore;
  MPI_Allreduce(&local_total, &st.total_seconds, 1, MPI_DOUBLE, MPI_MAX, comm_);
  MPI_Allreduce(&local_restore, &st.restore_seconds, 1, MPI_DOUBLE, MPI_MAX, comm_);
  st.comm_bytes = comm_bytes_;
  st.exchanges = exchanges_;
  return st;
}

double DistState::norm2() const {
  const double mine = local_.norm2();
  double all = 0.0;
  MPI_Allreduce(&mine, &all, 1, MPI_DOUBLE, MPI_SUM, comm_);
  return all;
}

void DistState::dump_binary(const std::string& path) const {
  MPI_File f;
  if (MPI_File_open(comm_, path.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &f) != MPI_SUCCESS)
    throw std::runtime_error("cannot open " + path);
  MPI_File_set_size(f, 0);
  const index total = local_.size();
  const MPI_Offset base = MPI_Offset(rank_) * MPI_Offset(total * sizeof(cplx));
  bool ok = true;
  for (index s = 0; s < total; s += kMaxMsgAmps) {
    const index c = std::min(kMaxMsgAmps, total - s);
    MPI_Status status;
    if (MPI_File_write_at(f, base + MPI_Offset(s * sizeof(cplx)), local_.data() + s, static_cast<int>(2 * c),
                          MPI_DOUBLE, &status) != MPI_SUCCESS)
      ok = false;
  }
  MPI_File_close(&f);
  int all_ok = ok, all = 0;
  MPI_Allreduce(&all_ok, &all, 1, MPI_INT, MPI_LAND, comm_);
  if (!all) throw std::runtime_error("short write to " + path);
}

void write_timings_csv(std::ostream& out, const Stats& s) {
  out << "index,gate,q0,q1,theta,seconds,bytes,gbps,comm_bytes,remapped\n";
  out << std::setprecision(9);
  for (std::size_t i = 0; i < s.timings.size(); ++i) {
    const auto& t = s.timings[i];
    out << i << ',' << gate_name(t.gate.kind) << ',' << t.gate.q0 << ',' << t.gate.q1 << ',' << t.gate.theta
        << ',' << t.seconds << ',' << t.bytes << ',' << (t.seconds > 0 ? t.bytes / t.seconds * 1e-9 : 0.0)
        << ',' << t.comm_bytes << ',' << t.remapped << '\n';
  }
}

RunResult to_run_result(const Stats& s) {
  RunResult r;
  r.total_seconds = s.total_seconds;
  r.timings.reserve(s.timings.size());
  for (const auto& t : s.timings) r.timings.push_back({t.gate, t.seconds, t.bytes});
  return r;
}

}

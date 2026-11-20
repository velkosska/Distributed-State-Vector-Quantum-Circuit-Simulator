#pragma once
#include <mpi.h>

#include <string>
#include <vector>

#include "qsim/circuit.hpp"
#include "qsim/simulator.hpp"
#include "qsim/state.hpp"

namespace qsim::dist {

struct Options {
  bool remap = true;
  bool time_each_gate = true;
  index chunk_amps = index{1} << 22;
};

struct GateRecord {
  Gate gate;
  double seconds = 0.0;
  double bytes = 0.0;
  double comm_bytes = 0.0;
  int remapped = 0;
};

struct Stats {
  double total_seconds = 0.0;
  double restore_seconds = 0.0;
  long long comm_gates = 0;
  long long remaps = 0;
  long long exchanges = 0;
  double comm_bytes = 0.0;
  std::vector<GateRecord> timings;
};

class DistState {
 public:
  DistState(int num_qubits, MPI_Comm comm, const Options& opt = {});

  int num_qubits() const { return n_; }
  int num_local() const { return nl_; }
  int num_ranks() const { return size_; }
  int rank() const { return rank_; }
  index chunk_amps() const { return chunk_; }

  Stats run(const std::vector<Gate>& gates);

  double norm2() const;
  void dump_binary(const std::string& path) const;
  const StateVector& local() const { return local_; }

 private:
  void apply_physical(const Gate& g, int a, int b);
  void swap_data(int a, int b);
  void swap_positions(int a, int b);
  void exchange_dense_1q(int gpos, const Mat2& m);
  void exchange_controlled_local(int cpos, int tpos, const Mat2& m);
  void exchange_swap_local_global(int lpos, int gpos);
  void exchange_swap_global_global(int a, int b);
  void scale_all(cplx f);

  bool is_comm(const Gate& g, int a, int b) const;
  int needs_comm_qubit(const Gate& g) const;
  int pick_victim(const std::vector<Gate>& gates, std::size_t i, int protect_logical) const;

  bool bit(int gpos) const { return (rank_ >> (gpos - nl_)) & 1; }
  int partner(int gpos) const { return rank_ ^ (1 << (gpos - nl_)); }
  void sendrecv(const cplx* send, cplx* recv, index count, int peer);

  MPI_Comm comm_;
  int rank_ = 0, size_ = 1, p_ = 0, n_ = 0, nl_ = 0;
  Options opt_;
  index chunk_;
  StateVector local_;
  std::vector<int> pos_;
  std::vector<int> lq_;
  std::vector<cplx> sbuf_, rbuf_;
  double comm_bytes_ = 0.0;
  long long exchanges_ = 0;
};

void write_timings_csv(std::ostream& out, const Stats& s);
RunResult to_run_result(const Stats& s);

}

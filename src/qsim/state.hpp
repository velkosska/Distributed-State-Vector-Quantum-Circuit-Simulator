#pragma once
// State vector of 2^n complex<double> amplitudes.
// Qubit q is bit q of the amplitude index (little-endian, same as Qiskit).
#include <cstddef>
#include <memory>
#include <string>

#include "qsim/types.hpp"

namespace qsim {

class StateVector {
 public:
  explicit StateVector(int num_qubits);  // allocates and sets |0...0>

  int num_qubits() const { return nq_; }
  index size() const { return pow2(nq_); }
  std::size_t bytes() const { return size() * sizeof(cplx); }
  cplx* data() { return amp_.get(); }
  const cplx* data() const { return amp_.get(); }
  cplx& operator[](index i) { return amp_[i]; }
  const cplx& operator[](index i) const { return amp_[i]; }

  // Set to |0...0>. The loop that first touches the memory lives here so an
  // OpenMP build can do first-touch placement with the same static schedule
  // as the gate kernels.
  void set_zero_state();
  double norm2() const;
  void dump_binary(const std::string& path) const;  // raw complex<double>, index order

 private:
  struct Free { void operator()(cplx* p) const; };
  int nq_;
  std::unique_ptr<cplx[], Free> amp_;
};

}  // namespace qsim

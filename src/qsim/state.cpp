#include "qsim/state.hpp"

#include <cstdio>
#include <cstdlib>
#include <new>
#include <stdexcept>

#include "qsim/state_loop.hpp"

namespace qsim {

void StateVector::Free::operator()(cplx* p) const { std::free(p); }

StateVector::StateVector(int num_qubits) : nq_(num_qubits) {
  if (num_qubits < 1 || num_qubits > 40) throw std::invalid_argument("num_qubits must be in [1, 40]");
  constexpr std::size_t align = detail::state_alignment;
  std::size_t nbytes = (bytes() + align - 1) / align * align;
  // Raw allocation: no constructor runs, so pages are first touched in set_zero_state().
  void* p = std::aligned_alloc(align, nbytes);
  if (!p) throw std::bad_alloc();
  amp_.reset(static_cast<cplx*>(p));
  set_zero_state();
}

void StateVector::set_zero_state() {
  cplx* a = amp_.get();
  detail::for_each_state_index<0>(size(), [&](index i) {
    // Construct in place so the thread writing each tile also faults its pages.
    new (a + i) cplx(0.0, 0.0);
  });
  a[0] = cplx(1.0, 0.0);
}

double StateVector::norm2() const {
  double s = 0.0;
  const cplx* a = amp_.get();
  for (index i = 0; i < size(); ++i) s += std::norm(a[i]);
  return s;
}

void StateVector::dump_binary(const std::string& path) const {
  std::FILE* f = std::fopen(path.c_str(), "wb");
  if (!f) throw std::runtime_error("cannot open " + path);
  const std::size_t n = std::fwrite(amp_.get(), sizeof(cplx), size(), f);
  std::fclose(f);
  if (n != size()) throw std::runtime_error("short write to " + path);
}

}  // namespace qsim

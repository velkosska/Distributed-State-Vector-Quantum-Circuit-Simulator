#pragma once
// Basic types shared by every backend (serial, OpenMP, MPI, CUDA).
#include <array>
#include <complex>
#include <cstdint>

namespace qsim {

using real  = double;
using cplx  = std::complex<real>;
using index = std::uint64_t;   // amplitude index; supports up to 63 qubits

// Row-major 2x2 matrix: {m00, m01, m10, m11}.
// Acting on (a0, a1) where a0 = <..0_q..|psi>, a1 = <..1_q..|psi>:
//   a0' = m00*a0 + m01*a1
//   a1' = m10*a0 + m11*a1
using Mat2 = std::array<cplx, 4>;

inline constexpr index pow2(int k) { return index{1} << k; }

// Insert a 0 bit at position q of i (bits >= q are shifted up by one).
// Enumerating k = 0 .. 2^(n-1)-1 and taking i0 = insert_zero_bit(k, q)
// visits every amplitude pair (i0, i0 | 2^q) exactly once.
inline constexpr index insert_zero_bit(index i, int q) {
  const index low = i & (pow2(q) - 1);
  return ((i >> q) << (q + 1)) | low;
}

// Insert two 0 bits at positions lo < hi (positions refer to the final index).
inline constexpr index insert_two_zero_bits(index i, int lo, int hi) {
  return insert_zero_bit(insert_zero_bit(i, lo), hi);
}

}  // namespace qsim

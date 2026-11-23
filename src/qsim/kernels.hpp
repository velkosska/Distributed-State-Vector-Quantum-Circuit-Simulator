#pragma once
// Gate kernels: the hot loops of the simulator.
//
// Every kernel works on a raw amplitude buffer `amp` of length 2^nq and does
// not know about circuits, files or timers. This is deliberate so that
//   * OpenMP can share the static partition used for state first touch,
//   * MPI can call the same kernels on the rank-local slice for gates on
//     local qubits, and use apply_1q_pairs() for gates on global qubits after
//     exchanging a slice with the partner rank,
//   * CUDA can reuse the index arithmetic (insert_zero_bit) 1:1 in a kernel
//     where the thread id plays the role of the loop counter k.
// See docs/kernel_api.md.
#include "qsim/state_loop.hpp"

namespace qsim::kern {

// Dense single-qubit gate on target q.
inline void apply_1q(cplx* amp, int nq, int q, const Mat2& m) {
  const index half = pow2(nq - 1);
  const index stride = pow2(q);
  const cplx m00 = m[0], m01 = m[1], m10 = m[2], m11 = m[3];
  detail::for_each_state_index<1>(half, [&](index k) {
    const index i0 = insert_zero_bit(k, q);
    const index i1 = i0 | stride;
    const cplx a0 = amp[i0], a1 = amp[i1];
    amp[i0] = m00 * a0 + m01 * a1;
    amp[i1] = m10 * a0 + m11 * a1;
  });
}

// Diagonal single-qubit gate diag(d0, d1) on q (Z, S, T, P, RZ, ...).
// If d0 == 1 only the |1> half is touched.
inline void apply_1q_diag(cplx* amp, int nq, int q, cplx d0, cplx d1) {
  const index half = pow2(nq - 1);
  const index stride = pow2(q);
  if (d0 == cplx(1.0, 0.0)) {
    detail::for_each_state_index<1>(half, [&](index k) {
      amp[insert_zero_bit(k, q) | stride] *= d1;
    });
  } else {
    detail::for_each_state_index<1>(half, [&](index k) {
      const index i0 = insert_zero_bit(k, q);
      amp[i0] *= d0;
      amp[i0 | stride] *= d1;
    });
  }
}

// Controlled single-qubit gate: apply m to target t where control c is 1.
inline void apply_controlled_1q(cplx* amp, int nq, int c, int t, const Mat2& m) {
  const index quarter = pow2(nq - 2);
  const int lo = c < t ? c : t, hi = c < t ? t : c;
  const index cbit = pow2(c), tbit = pow2(t);
  const cplx m00 = m[0], m01 = m[1], m10 = m[2], m11 = m[3];
  detail::for_each_state_index<2>(quarter, [&](index k) {
    const index i0 = insert_two_zero_bits(k, lo, hi) | cbit;
    const index i1 = i0 | tbit;
    const cplx a0 = amp[i0], a1 = amp[i1];
    amp[i0] = m00 * a0 + m01 * a1;
    amp[i1] = m10 * a0 + m11 * a1;
  });
}

// Multiply amplitudes with both q0 and q1 set by `phase` (CZ: -1, CP: e^{i theta}).
inline void apply_controlled_phase(cplx* amp, int nq, int q0, int q1, cplx phase) {
  const index quarter = pow2(nq - 2);
  const int lo = q0 < q1 ? q0 : q1, hi = q0 < q1 ? q1 : q0;
  const index both = pow2(q0) | pow2(q1);
  detail::for_each_state_index<2>(quarter, [&](index k) {
    amp[insert_two_zero_bits(k, lo, hi) | both] *= phase;
  });
}

// SWAP: exchange amplitudes |..0_a..1_b..> <-> |..1_a..0_b..>.
inline void apply_swap(cplx* amp, int nq, int a, int b) {
  const index quarter = pow2(nq - 2);
  const int lo = a < b ? a : b, hi = a < b ? b : a;
  const index abit = pow2(a), bbit = pow2(b);
  detail::for_each_state_index<2>(quarter, [&](index k) {
    const index base = insert_two_zero_bits(k, lo, hi);
    const index i = base | abit, j = base | bbit;
    const cplx tmp = amp[i];
    amp[i] = amp[j];
    amp[j] = tmp;
  });
}

// Split-buffer form of apply_1q for the distributed backend: lo[k] holds the
// amplitude with the target bit 0 and hi[k] the matching one with bit 1.
// Used after an MPI exchange when the target is a global (rank) qubit.
inline void apply_1q_pairs(cplx* lo, cplx* hi, index count, const Mat2& m) {
  const cplx m00 = m[0], m01 = m[1], m10 = m[2], m11 = m[3];
#if defined(QSIM_USE_OPENMP)
#pragma omp parallel for schedule(static)
#endif
  for (index k = 0; k < count; ++k) {
    const cplx a0 = lo[k], a1 = hi[k];
    lo[k] = m00 * a0 + m01 * a1;
    hi[k] = m10 * a0 + m11 * a1;
  }
}

}  // namespace qsim::kern

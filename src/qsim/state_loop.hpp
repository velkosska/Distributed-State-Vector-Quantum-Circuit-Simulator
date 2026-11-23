#pragma once

#include "qsim/types.hpp"

#if defined(QSIM_USE_OPENMP)
#include <algorithm>
#include <omp.h>
#endif

namespace qsim::detail {

// Use the same 4 KiB units for state allocation, first touch and gate work.
inline constexpr index state_alignment = 4096;

// The power-of-two count omits FixedBits amplitude bits. The resulting
// tile count is identical for initialization and every gate on the state.
template<int FixedBits, class Function>
inline void for_each_state_index(index count, const Function& apply) {
#if defined(QSIM_USE_OPENMP)
  constexpr index indices_per_tile = (state_alignment / sizeof(cplx)) >> FixedBits;
  const index tiles = count / indices_per_tile;
  // Keep states below 64 KiB on the direct loop and give each worker
  // at least four tiles when a team is worthwhile.
  const int workers = tiles >= 16 ? static_cast<int>(std::min<index>(omp_get_max_threads(), tiles / 4)) : 1;
  if (workers > 1) {
#pragma omp parallel num_threads(workers)
    {
      const index team = omp_get_num_threads();
      const index worker = omp_get_thread_num();
      const index begin = (tiles * worker / team) * indices_per_tile;
      const index end = (tiles * (worker + 1) / team) * indices_per_tile;
      for (index k = begin; k < end; ++k) apply(k);
    }
    return;
  }
#endif
  for (index k = 0; k < count; ++k) apply(k);
}

}  // namespace qsim::detail

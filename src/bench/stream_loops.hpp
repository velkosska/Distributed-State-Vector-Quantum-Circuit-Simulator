#pragma once

namespace stream {

inline void initialize(double* a, double* b, double* c, long n) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (long i = 0; i < n; ++i) {
    a[i] = 1.0;
    b[i] = 2.0;
    c[i] = 0.0;
  }
}

inline void triad(double* a, const double* b, const double* c, long n) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (long i = 0; i < n; ++i) a[i] = b[i] + 3.0 * c[i];
}

}  // namespace stream

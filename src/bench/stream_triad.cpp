// stream_triad: minimal STREAM-style bandwidth probe (copy and triad) used as
// the memory roof in the roofline plot. Arrays are first touched in parallel
// with the same static schedule as the kernels, so pages land on the NUMA node
// of the thread that later uses them.
//
//   stream_triad [MiB_per_array=2048] [reps=10]
// Prints one line:
//   STREAM threads=.. array_MiB=.. copy_GBs=.. triad_GBs=..   (best of reps)
// Bytes counted like STREAM: copy 2 x 8 B, triad 3 x 8 B per element
// (write-allocate traffic is not counted, same as the qsim GB/s model).
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <memory>

#include "stream_loops.hpp"
#ifdef _OPENMP
#include <omp.h>
#endif

int main(int argc, char** argv) {
  const double mib = argc > 1 ? std::atof(argv[1]) : 2048.0;
  const int reps = argc > 2 ? std::atoi(argv[2]) : 10;
  const long n = static_cast<long>(mib * 1024 * 1024 / sizeof(double));
  std::unique_ptr<double[]> a(new double[n]), b(new double[n]), c(new double[n]);
  stream::initialize(a.get(), b.get(), c.get(), n);

  using clk = std::chrono::steady_clock;
  double best_copy = 1e30, best_triad = 1e30;
  for (int r = 0; r < reps; ++r) {
    auto t0 = clk::now();
#pragma omp parallel for schedule(static)
    for (long i = 0; i < n; ++i) c[i] = a[i];
    auto t1 = clk::now();
    stream::triad(a.get(), b.get(), c.get(), n);
    auto t2 = clk::now();
    best_copy = std::min(best_copy, std::chrono::duration<double>(t1 - t0).count());
    best_triad = std::min(best_triad, std::chrono::duration<double>(t2 - t1).count());
  }
  int threads = 1;
#ifdef _OPENMP
  threads = omp_get_max_threads();
#endif
  volatile double sink = a[n / 2];  // keep the loops alive
  (void)sink;
  std::printf("STREAM threads=%d array_MiB=%.0f copy_GBs=%.3f triad_GBs=%.3f\n", threads, mib,
              2.0 * 8 * n / best_copy * 1e-9, 3.0 * 8 * n / best_triad * 1e-9);
  return 0;
}

// CUDA gate kernels and driver. Each kernel is a 1:1 port of the loop in
// src/qsim/kernels.hpp: the global thread id plays the role of the pair index k.
#include "qsim/cuda/cuda_sim.hpp"

#include <cuda_runtime.h>
#include <thrust/complex.h>

#include <sstream>
#include <stdexcept>
#include <string>

namespace qsim::cuda {
namespace {

using dcplx = thrust::complex<double>;
static_assert(sizeof(dcplx) == sizeof(cplx), "thrust::complex<double> must match std::complex<double>");

void check(cudaError_t e, const char* what) {
  if (e != cudaSuccess) throw std::runtime_error(std::string("CUDA error in ") + what + ": " + cudaGetErrorString(e));
}
#define QSIM_CUDA_CHECK(call) check((call), #call)

inline dcplx to_d(cplx c) { return dcplx(c.real(), c.imag()); }

constexpr int kBlock = 256;
inline unsigned grid_for(index n) { return static_cast<unsigned>((n + kBlock - 1) / kBlock); }

__device__ inline index thread_k() { return static_cast<index>(blockIdx.x) * blockDim.x + threadIdx.x; }

// ---------------------------------------------------------------- kernels

// Dense single-qubit gate on q (kern::apply_1q).
__global__ void k_apply_1q(dcplx* amp, index half, int q, dcplx m00, dcplx m01, dcplx m10, dcplx m11) {
  const index k = thread_k();
  if (k >= half) return;
  const index i0 = insert_zero_bit(k, q);
  const index i1 = i0 | pow2(q);
  const dcplx a0 = amp[i0], a1 = amp[i1];
  amp[i0] = m00 * a0 + m01 * a1;
  amp[i1] = m10 * a0 + m11 * a1;
}

// Diagonal gate with d0 == 1: only the |1> half is touched (kern::apply_1q_diag).
__global__ void k_apply_diag_hi(dcplx* amp, index half, int q, dcplx d1) {
  const index k = thread_k();
  if (k >= half) return;
  amp[insert_zero_bit(k, q) | pow2(q)] *= d1;
}

// General diagonal gate diag(d0, d1) on q.
__global__ void k_apply_diag(dcplx* amp, index half, int q, dcplx d0, dcplx d1) {
  const index k = thread_k();
  if (k >= half) return;
  const index i0 = insert_zero_bit(k, q);
  amp[i0] *= d0;
  amp[i0 | pow2(q)] *= d1;
}

// Controlled single-qubit gate (kern::apply_controlled_1q).
__global__ void k_apply_controlled_1q(dcplx* amp, index quarter, int lo, int hi, index cbit, index tbit,
                                      dcplx m00, dcplx m01, dcplx m10, dcplx m11) {
  const index k = thread_k();
  if (k >= quarter) return;
  const index i0 = insert_two_zero_bits(k, lo, hi) | cbit;
  const index i1 = i0 | tbit;
  const dcplx a0 = amp[i0], a1 = amp[i1];
  amp[i0] = m00 * a0 + m01 * a1;
  amp[i1] = m10 * a0 + m11 * a1;
}

// Phase on amplitudes with both qubits set (kern::apply_controlled_phase).
__global__ void k_apply_controlled_phase(dcplx* amp, index quarter, int lo, int hi, index both, dcplx phase) {
  const index k = thread_k();
  if (k >= quarter) return;
  amp[insert_two_zero_bits(k, lo, hi) | both] *= phase;
}

// SWAP (kern::apply_swap).
__global__ void k_apply_swap(dcplx* amp, index quarter, int lo, int hi, index abit, index bbit) {
  const index k = thread_k();
  if (k >= quarter) return;
  const index base = insert_two_zero_bits(k, lo, hi);
  const index i = base | abit, j = base | bbit;
  const dcplx tmp = amp[i];
  amp[i] = amp[j];
  amp[j] = tmp;
}

// ---------------------------------------------------------------- dispatch

// Mirrors qsim::apply_gate() in simulator.cpp. Matrices are built on the host
// with matrix_1q() and passed to the kernel by value.
void launch_gate(dcplx* amp, int nq, const Gate& g) {
  using K = GateKind;
  const index half = pow2(nq - 1);
  switch (g.kind) {
    case K::H: case K::X: case K::Y: case K::RX: case K::RY: case K::U: {
      const Mat2 m = matrix_1q(g);
      k_apply_1q<<<grid_for(half), kBlock>>>(amp, half, g.q0, to_d(m[0]), to_d(m[1]), to_d(m[2]), to_d(m[3]));
      return;
    }
    case K::Z: case K::S: case K::SDG: case K::T: case K::TDG: case K::P: case K::RZ: {
      const Mat2 m = matrix_1q(g);
      if (m[0] == cplx(1.0, 0.0))
        k_apply_diag_hi<<<grid_for(half), kBlock>>>(amp, half, g.q0, to_d(m[3]));
      else
        k_apply_diag<<<grid_for(half), kBlock>>>(amp, half, g.q0, to_d(m[0]), to_d(m[3]));
      return;
    }
    default: break;
  }

  // Two-qubit gates.
  const index quarter = pow2(nq - 2);
  const int lo = g.q0 < g.q1 ? g.q0 : g.q1, hi = g.q0 < g.q1 ? g.q1 : g.q0;
  switch (g.kind) {
    case K::CX: case K::CY: {
      const Mat2 m = matrix_1q(g);
      k_apply_controlled_1q<<<grid_for(quarter), kBlock>>>(amp, quarter, lo, hi, pow2(g.q0), pow2(g.q1),
                                                           to_d(m[0]), to_d(m[1]), to_d(m[2]), to_d(m[3]));
      return;
    }
    case K::CZ:
      k_apply_controlled_phase<<<grid_for(quarter), kBlock>>>(amp, quarter, lo, hi, pow2(g.q0) | pow2(g.q1),
                                                              dcplx(-1.0, 0.0));
      return;
    case K::CP:
      k_apply_controlled_phase<<<grid_for(quarter), kBlock>>>(amp, quarter, lo, hi, pow2(g.q0) | pow2(g.q1),
                                                              to_d(std::polar(1.0, g.theta)));
      return;
    case K::SWAP:
      k_apply_swap<<<grid_for(quarter), kBlock>>>(amp, quarter, lo, hi, pow2(g.q0), pow2(g.q1));
      return;
    default: break;
  }
  throw std::logic_error("cuda launch_gate: unhandled gate");
}

// ---------------------------------------------------------------- RAII helpers

struct DeviceBuffer {
  dcplx* p = nullptr;
  explicit DeviceBuffer(std::size_t bytes) { QSIM_CUDA_CHECK(cudaMalloc(&p, bytes)); }
  ~DeviceBuffer() { if (p) cudaFree(p); }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

struct Event {
  cudaEvent_t e{};
  Event() { QSIM_CUDA_CHECK(cudaEventCreate(&e)); }
  ~Event() { cudaEventDestroy(e); }
  Event(const Event&) = delete;
  Event& operator=(const Event&) = delete;
};

// Seconds between two recorded events (waits for `stop`).
double elapsed(const Event& start, const Event& stop) {
  QSIM_CUDA_CHECK(cudaEventSynchronize(stop.e));
  float ms = 0.0f;
  QSIM_CUDA_CHECK(cudaEventElapsedTime(&ms, start.e, stop.e));
  return ms * 1e-3;
}

}  // namespace

CudaRunResult run(StateVector& s, const std::vector<Gate>& gates, const RunOptions& opt) {
  CudaRunResult r;
  const int nq = s.num_qubits();
  const std::size_t bytes = s.bytes();

  int dev = 0;
  QSIM_CUDA_CHECK(cudaGetDevice(&dev));
  cudaDeviceProp prop{};
  QSIM_CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
  r.device_name = prop.name;

  std::size_t free_b = 0, total_b = 0;
  QSIM_CUDA_CHECK(cudaMemGetInfo(&free_b, &total_b));
  if (bytes > free_b) {
    std::ostringstream msg;
    msg << nq << " qubits need " << bytes / double(1 << 30) << " GiB on the GPU, but only "
        << free_b / double(1 << 30) << " GiB are free on " << prop.name;
    throw std::runtime_error(msg.str());
  }

  DeviceBuffer d(bytes);
  Event start, stop, g0, g1;

  // Host -> device (the state lives on the GPU from here on).
  QSIM_CUDA_CHECK(cudaEventRecord(start.e));
  QSIM_CUDA_CHECK(cudaMemcpy(d.p, s.data(), bytes, cudaMemcpyHostToDevice));
  QSIM_CUDA_CHECK(cudaEventRecord(stop.e));
  r.h2d_seconds = elapsed(start, stop);

  // Gates: one kernel launch each.
  if (opt.time_each_gate) r.run.timings.reserve(gates.size());
  QSIM_CUDA_CHECK(cudaEventRecord(start.e));
  for (const auto& g : gates) {
    if (opt.time_each_gate) {
      QSIM_CUDA_CHECK(cudaEventRecord(g0.e));
      launch_gate(d.p, nq, g);
      QSIM_CUDA_CHECK(cudaGetLastError());
      QSIM_CUDA_CHECK(cudaEventRecord(g1.e));
      r.run.timings.push_back({g, elapsed(g0, g1), bytes_touched(g, nq)});
    } else {
      launch_gate(d.p, nq, g);
      QSIM_CUDA_CHECK(cudaGetLastError());
    }
  }
  QSIM_CUDA_CHECK(cudaEventRecord(stop.e));
  r.run.total_seconds = elapsed(start, stop);

  // Device -> host.
  QSIM_CUDA_CHECK(cudaEventRecord(start.e));
  QSIM_CUDA_CHECK(cudaMemcpy(s.data(), d.p, bytes, cudaMemcpyDeviceToHost));
  QSIM_CUDA_CHECK(cudaEventRecord(stop.e));
  r.d2h_seconds = elapsed(start, stop);

  return r;
}

}  // namespace qsim::cuda

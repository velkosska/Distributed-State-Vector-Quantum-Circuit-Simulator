// Small self-contained checks of the gate kernels (no test framework needed).
#include <cmath>
#include <cstdio>
#include <sstream>

#include "qsim/circuit.hpp"
#include "qsim/simulator.hpp"
#include "qsim/state.hpp"

using namespace qsim;

static int failures = 0;
#define CHECK(cond)                                                    \
  do {                                                                 \
    if (!(cond)) {                                                     \
      std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);      \
      ++failures;                                                      \
    }                                                                  \
  } while (0)

static bool near(cplx a, cplx b, double tol = 1e-12) { return std::abs(a - b) < tol; }

static StateVector run_text(const char* text) {
  std::istringstream in(text);
  Circuit c = read_circuit(in);
  StateVector s(c.num_qubits);
  run(s, c.gates, {false});
  return s;
}

int main() {
  const double r = 1.0 / std::sqrt(2.0);

  {  // Bell state on qubits 0,1 of 3: (|000> + |011>)/sqrt2
    auto s = run_text("qubits 3\nh 0\ncx 0 1\n");
    CHECK(near(s[0], r));
    CHECK(near(s[3], r));
    CHECK(near(s[1], 0) && near(s[2], 0));
  }
  {  // X on qubit 2 -> index 4 (little-endian)
    auto s = run_text("qubits 3\nx 2\n");
    CHECK(near(s[4], 1));
  }
  {  // SWAP moves |001> to |100>
    auto s = run_text("qubits 3\nx 0\nswap 0 2\n");
    CHECK(near(s[4], 1));
  }
  {  // CZ and CP(pi) agree; phase only on |11>
    auto a = run_text("qubits 2\nh 0\nh 1\ncz 0 1\n");
    auto b = run_text("qubits 2\nh 0\nh 1\ncp 1 0 3.141592653589793\n");
    for (index i = 0; i < 4; ++i) CHECK(near(a[i], b[i]));
    CHECK(near(a[3], -0.5));
  }
  {  // HH = I, S S = Z, T T = S, RX(pi) = -iX
    auto s = run_text("qubits 1\nh 0\nh 0\n");
    CHECK(near(s[0], 1));
    auto z = run_text("qubits 1\nh 0\ns 0\ns 0\nh 0\n");  // HZH = X
    CHECK(near(z[1], 1));
    auto t = run_text("qubits 1\nx 0\nt 0\nt 0\nsdg 0\n");
    CHECK(near(t[1], 1));
    auto rx = run_text("qubits 1\nrx 0 3.141592653589793\n");
    CHECK(near(rx[1], cplx(0, -1)));
  }
  {  // generic U equal to Y
    auto u = run_text("qubits 2\nu 1 0 0 0 -1 0 1 0 0\n");
    auto y = run_text("qubits 2\ny 1\n");
    for (index i = 0; i < 4; ++i) CHECK(near(u[i], y[i]));
  }
  {  // controlled gate with control above target
    auto s = run_text("qubits 3\nx 2\ncx 2 0\n");
    CHECK(near(s[5], 1));
  }

  if (failures == 0) std::printf("all kernel tests passed\n");
  return failures == 0 ? 0 : 1;
}

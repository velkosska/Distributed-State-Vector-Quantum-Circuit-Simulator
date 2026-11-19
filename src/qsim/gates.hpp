#pragma once
// Gate description used by the circuit reader, the fusion pass and every backend.
#include <string>
#include <vector>

#include "qsim/types.hpp"

namespace qsim {

enum class GateKind {
  // single-qubit, fixed
  H, X, Y, Z, S, SDG, T, TDG,
  // single-qubit, parametrised
  P, RX, RY, RZ,
  // generic single-qubit unitary (also the output of 1q fusion)
  U,
  // two-qubit
  CX, CY, CZ, CP, SWAP,
};

struct Gate {
  GateKind kind;
  int q0 = 0;          // target for 1q gates; control for CX/CY/CP/CZ; first qubit of SWAP
  int q1 = -1;         // target for controlled gates; second qubit of SWAP; -1 for 1q gates
  double theta = 0.0;  // angle for P, RX, RY, RZ, CP
  Mat2 m{};            // matrix for U (filled by the reader or by fusion)
};

int num_qubits(GateKind k);         // 1 or 2
int num_params(GateKind k);         // 0 or 1 (U carries its matrix instead)
bool is_diagonal(GateKind k);       // Z S SDG T TDG P RZ CZ CP
const char* gate_name(GateKind k);  // lower-case mnemonic used in circuit files
bool parse_gate_name(const std::string& s, GateKind& out);

// 2x2 matrix of a single-qubit gate (or of the target action of CX/CY/CP).
Mat2 matrix_1q(const Gate& g);

// Bytes read+written by one application of g on 2^nq amplitudes. Used to
// report effective bandwidth; it follows what the kernels actually touch.
double bytes_touched(const Gate& g, int nq);

}  // namespace qsim

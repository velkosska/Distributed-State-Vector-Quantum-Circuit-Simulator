#pragma once
// Gate-fusion pass (Optimisation 1). Runs on the gate list before simulation.
//
// FusionMode::None returns the gates unchanged.
//
// FusionMode::SingleQubit ("--fusion 1q") merges every run of single-qubit
// gates that act on the same qubit into one gate, so the state is swept once
// per run instead of once per gate. Gates on other qubits do not interrupt a
// run (they commute with it). A run ends when a two-qubit gate that does not
// commute with it touches its qubit:
//   * a run of diagonal gates (Z S Sdg T Tdg P RZ) commutes with CZ and CP on
//     either qubit and with the control of CX / CY, so it is carried past them;
//   * a SWAP moves the pending runs of its two qubits onto each other;
//   * anything else flushes the run just before the two-qubit gate.
// A flushed run becomes
//   * the original gate, if the run has one gate;
//   * nothing, if the product is the identity up to a global phase;
//   * P(phi) if the product is diagonal (diag(d0, d1) = d0 * P(arg d1/d0)),
//     so diagonal runs stay cheap (half the amplitudes) and need no
//     communication on a global qubit in the MPI backend;
//   * GateKind::U with the 2x2 product otherwise.
// Global phases dropped this way are collected and multiplied into one dense
// single-qubit gate of the output (or, if there is none, emitted as
// RZ(-2a) P(2a) on qubit 0), so the final state matches the unfused circuit
// exactly, not only up to a global phase.
//
// The output contains ordinary Gates of existing kinds, so every backend
// (serial, OpenMP, MPI, CUDA) runs it without changes.
#include <string>
#include <vector>

#include "qsim/gates.hpp"

namespace qsim {

enum class FusionMode { None, SingleQubit };

bool parse_fusion_mode(const std::string& s, FusionMode& out);
std::vector<Gate> fuse(const std::vector<Gate>& gates, int num_qubits, FusionMode mode);

}  // namespace qsim

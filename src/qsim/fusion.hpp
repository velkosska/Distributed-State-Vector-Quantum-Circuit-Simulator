#pragma once
// Gate-fusion pass (Optimisation 1). Runs on the gate list before simulation.
//
// Status: hook only. FusionMode::None returns the gates unchanged. The
// intended first implementation is FusionMode::SingleQubit: merge runs of
// consecutive 1q gates on the same qubit into one GateKind::U (product of
// their 2x2 matrices), which cuts the number of passes over the state.
#include <string>
#include <vector>

#include "qsim/gates.hpp"

namespace qsim {

enum class FusionMode { None, SingleQubit };

bool parse_fusion_mode(const std::string& s, FusionMode& out);
std::vector<Gate> fuse(const std::vector<Gate>& gates, int num_qubits, FusionMode mode);

}  // namespace qsim

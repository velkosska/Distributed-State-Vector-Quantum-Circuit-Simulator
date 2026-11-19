#include "qsim/fusion.hpp"

#include <stdexcept>

namespace qsim {

bool parse_fusion_mode(const std::string& s, FusionMode& out) {
  if (s == "none") { out = FusionMode::None; return true; }
  if (s == "1q") { out = FusionMode::SingleQubit; return true; }
  return false;
}

std::vector<Gate> fuse(const std::vector<Gate>& gates, int /*num_qubits*/, FusionMode mode) {
  switch (mode) {
    case FusionMode::None:
      return gates;
    case FusionMode::SingleQubit:
      // TODO(optimisation 1): implement; see header comment.
      throw std::runtime_error("fusion mode '1q' is not implemented yet");
  }
  return gates;
}

}  // namespace qsim

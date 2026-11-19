#pragma once
// Circuit container and text reader. Format: docs/circuit_format.md
#include <iosfwd>
#include <string>
#include <vector>

#include "qsim/gates.hpp"

namespace qsim {

struct Circuit {
  int num_qubits = 0;
  std::vector<Gate> gates;
};

Circuit read_circuit(std::istream& in, const std::string& source_name = "<stream>");
Circuit read_circuit_file(const std::string& path);
void write_circuit(std::ostream& out, const Circuit& c);

}  // namespace qsim

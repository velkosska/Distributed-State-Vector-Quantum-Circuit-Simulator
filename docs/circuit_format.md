# Circuit file format

Plain text, one statement per line. `#` starts a comment. Blank lines are ignored.

```
# 3-qubit GHZ state
qubits 3
h 0
cx 0 1
cx 1 2
rz 2 0.785398163397448
```

The first statement must be `qubits N` (1 ≤ N ≤ 40). Each following line is
`name qubit(s) [angle] [matrix]`. Qubit `q` is bit `q` of the amplitude index
(little-endian, the same convention as Qiskit), so `x 2` on `|000>` gives index 4.

| Gate | Syntax | Matrix / action |
|---|---|---|
| `h x y z` | `h q` | standard Pauli / Hadamard |
| `s sdg t tdg` | `t q` | diag(1, i), diag(1, -i), diag(1, e^{iπ/4}), diag(1, e^{-iπ/4}) |
| `p` | `p q θ` | diag(1, e^{iθ}) |
| `rx ry rz` | `rx q θ` | e^{-iθX/2}, e^{-iθY/2}, e^{-iθZ/2} (Qiskit convention) |
| `u` | `u q re00 im00 re01 im01 re10 im10 re11 im11` | any 2×2 matrix, row-major |
| `cx` (`cnot`) `cy` | `cx control target` | controlled X / Y |
| `cz` | `cz a b` | −1 on \|11⟩ (symmetric) |
| `cp` | `cp a b θ` | e^{iθ} on \|11⟩ (symmetric) |
| `swap` | `swap a b` | exchange the two qubits |

Angles are in radians. Write them with full precision (the generators use
`repr(float)`, 17 significant digits) so validation can reach 1e-10.

Generate benchmark circuits with `src/tools/gen_circuits.py`; read and convert
them in Python with `src/tools/qsim_circuit.py` (`Circuit.read(path).to_qiskit()`).

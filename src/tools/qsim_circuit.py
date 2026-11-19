"""Shared helpers for the qsim text circuit format (see docs/circuit_format.md).

A circuit is kept as (num_qubits, [ (name, qubits_tuple, params_tuple), ... ]).
"""
from __future__ import annotations

ONE_Q = {"h", "x", "y", "z", "s", "sdg", "t", "tdg"}
ONE_Q_PARAM = {"p", "rx", "ry", "rz"}
TWO_Q = {"cx", "cy", "cz", "swap"}
TWO_Q_PARAM = {"cp"}


class Circuit:
    def __init__(self, num_qubits: int):
        self.num_qubits = num_qubits
        self.ops: list[tuple[str, tuple[int, ...], tuple[float, ...]]] = []

    # builder helpers -----------------------------------------------------
    def add(self, name: str, *qubits: int, params: tuple[float, ...] = ()):
        for q in qubits:
            assert 0 <= q < self.num_qubits, (name, qubits)
        self.ops.append((name, tuple(qubits), tuple(params)))
        return self

    def __getattr__(self, name):  # c.h(0), c.cx(0, 1), c.rz(2, theta=0.3)
        if name in ONE_Q | TWO_Q:
            return lambda *q: self.add(name, *q)
        if name in ONE_Q_PARAM | TWO_Q_PARAM:
            return lambda *q, theta: self.add(name, *q, params=(theta,))
        raise AttributeError(name)

    # text I/O ------------------------------------------------------------
    def to_text(self, header: str = "") -> str:
        lines = [f"# {l}" for l in header.splitlines()] if header else []
        lines.append(f"qubits {self.num_qubits}")
        for name, qs, ps in self.ops:
            lines.append(" ".join([name, *map(str, qs), *(repr(float(p)) for p in ps)]))
        return "\n".join(lines) + "\n"

    def write(self, path: str, header: str = ""):
        with open(path, "w") as f:
            f.write(self.to_text(header))

    @staticmethod
    def read(path: str) -> "Circuit":
        c = None
        with open(path) as f:
            for raw in f:
                tok = raw.split("#", 1)[0].split()
                if not tok:
                    continue
                if tok[0] == "qubits":
                    c = Circuit(int(tok[1]))
                    continue
                name = "cx" if tok[0] == "cnot" else tok[0]
                nq = 2 if name in TWO_Q | TWO_Q_PARAM else 1
                qs = tuple(int(t) for t in tok[1:1 + nq])
                ps = tuple(float(t) for t in tok[1 + nq:])
                c.ops.append((name, qs, ps))
        return c

    # Qiskit conversion (Qiskit uses the same little-endian qubit order) ----
    def to_qiskit(self):
        import numpy as np
        from qiskit import QuantumCircuit
        from qiskit.circuit.library import UnitaryGate

        qc = QuantumCircuit(self.num_qubits)
        for name, qs, ps in self.ops:
            if name == "u":
                m = np.array([complex(ps[0], ps[1]), complex(ps[2], ps[3]),
                              complex(ps[4], ps[5]), complex(ps[6], ps[7])]).reshape(2, 2)
                qc.append(UnitaryGate(m, check_input=False), [qs[0]])
            elif name in ONE_Q_PARAM | TWO_Q_PARAM:
                getattr(qc, name)(ps[0], *qs)
            else:
                getattr(qc, name)(*qs)
        return qc

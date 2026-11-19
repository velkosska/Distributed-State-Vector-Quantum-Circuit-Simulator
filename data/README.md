# data/

Circuit files in the qsim text format (see `docs/circuit_format.md`).

Benchmark circuits are **not committed**: they are regenerated deterministically
from `src/tools/gen_circuits.py` with fixed seeds (the Slurm scripts do this at
job start). Only the tiny hand-checkable files in `examples/` are tracked.

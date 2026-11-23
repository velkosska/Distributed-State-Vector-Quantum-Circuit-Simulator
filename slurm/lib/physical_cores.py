"""Print one allowed Linux CPU ID per physical core for taskset."""

import os
from pathlib import Path

cores = {}
for cpu in sorted(os.sched_getaffinity(0)):
    siblings = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list")
    cores.setdefault(siblings.read_text().strip(), cpu)
print(",".join(map(str, cores.values())))

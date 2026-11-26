"""Print the swimmer regression references of tests/test_ph_dmc_swimmer.py.

Usage (from the repository root, as the test suite runs, 2 threads):
    OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python benchmarks/ph_swimmer_references.py

Since 2026-11-10 the swimmers run dm_control's own physics (2 ms Euler) and integrate
their model with the semi_implicit map at that step; the short runs that pin this
convention, REFERENCE_SWIMMER_ARMS, come from this script (first run in Vista job
1135351). It runs each key of REFERENCE_SWIMMER_ARMS through tiny_arm_summary (the
test's own function, at the test's SWIMMER_BUDGET) and prints a Python literal to paste
over the dict, unchanged, whenever the swimmer convention changes on purpose.
"""

from __future__ import annotations

import pprint
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import torch  # noqa: E402

from tests.test_ph_dmc_swimmer import (  # noqa: E402
    REFERENCE_SWIMMER_ARMS,
    SWIMMER_BUDGET,
    tiny_arm_summary,
)


def main() -> None:
    torch.set_num_threads(2)
    references = {}
    for key in REFERENCE_SWIMMER_ARMS:
        env_id, arm = key.split("/")
        references[key] = tiny_arm_summary(env_id, arm, SWIMMER_BUDGET)
        print(f"# {key} done", file=sys.stderr, flush=True)
    print("REFERENCE_SWIMMER_ARMS = " + pprint.pformat(references, width=100, sort_dicts=True))


if __name__ == "__main__":
    main()

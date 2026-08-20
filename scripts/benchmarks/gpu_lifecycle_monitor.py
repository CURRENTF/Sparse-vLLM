# SPDX-License-Identifier: Apache-2.0
"""Compatibility entrypoint for the canonical coarse GPU monitor."""

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmark.efficiency.hardware_monitor import main


if __name__ == "__main__":
    main()

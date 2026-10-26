"""One documented command per test tier.

    python scripts/run_tests.py unit          # fast; no dataset, weights or network
    python scripts/run_tests.py integration   # module boundaries; fixtures only
    python scripts/run_tests.py model_smoke   # one real pretrained model path (from Phase 4)
    python scripts/run_tests.py slow          # full RWF-2000 runs (from Phase 1 onwards)
    python scripts/run_tests.py fast          # unit + integration (run this on every change)

pytest exits with code 5 when a tier contains no tests yet. For the tiers that are
still empty (model_smoke, slow) that is reported as "nothing to run yet" instead of
as a failure.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TIERS = {
    "unit": ["-m", "unit"],
    "integration": ["-m", "integration"],
    "model_smoke": ["-m", "model_smoke"],
    "slow": ["-m", "slow"],
    "fast": ["-m", "unit or integration"],
}
EXIT_NO_TESTS_COLLECTED = 5


def main(argv: list[str]) -> int:
    if len(argv) < 1 or argv[0] not in TIERS:
        print(__doc__)
        return 64
    command = [sys.executable, "-m", "pytest", *TIERS[argv[0]], *argv[1:]]
    code = subprocess.call(command, cwd=ROOT)
    if code == EXIT_NO_TESTS_COLLECTED and argv[0] in ("model_smoke", "slow"):
        print(f"No '{argv[0]}' tests exist yet - they are added in later phases.")
        return 0
    return code


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

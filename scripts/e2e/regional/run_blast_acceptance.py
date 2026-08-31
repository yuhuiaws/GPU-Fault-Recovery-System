from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.e2e.regional.blast_acceptance_base import CASE_IDS  # noqa: E402
from scripts.e2e.regional.blast_acceptance_cases_2 import (  # noqa: E402
    BlastCasesTwo as Runner,
)
from scripts.e2e.regional.regional_case_contract import (  # noqa: E402
    predecessor_path,
)
from scripts.e2e.regional.regional_live_fixture import (  # noqa: E402
    predecessor_evidence,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one read-only regional BLAST acceptance audit."
    )
    parser.add_argument("--case", choices=CASE_IDS, required=True)
    parser.add_argument(
        "--site",
        type=Path,
        default=(
            Path(os.environ["GPU_FAULT_SITE_FILE"])
            if os.getenv("GPU_FAULT_SITE_FILE")
            else None
        ),
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=(
            Path(os.environ["GPU_FAULT_ACCEPTANCE_RUN_DIR"])
            if os.getenv("GPU_FAULT_ACCEPTANCE_RUN_DIR")
            else None
        ),
    )
    parser.add_argument(
        "--e2e-dir",
        type=Path,
        default=os.getenv("GPU_FAULT_E2E001_EVIDENCE_DIR", "."),
    )
    parser.add_argument(
        "--trusted-cpu-baseline",
        type=Path,
        default=os.getenv("GPU_FAULT_TRUSTED_CPU_BASELINE", "."),
    )
    parser.add_argument("--predecessor-evidence", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.site is None or args.run_dir is None:
        raise SystemExit("--site and --run-dir are required")
    required_paths = [args.site]
    if args.case == "GF-REGIONAL-BLAST-001":
        required_paths.extend((args.e2e_dir, args.trusted_cpu_baseline))
    for path in required_paths:
        if not path.exists():
            raise SystemExit(f"required path does not exist: {path}")
    predecessor_id, path = predecessor_path(
        args.run_dir,
        args.case,
        args.predecessor_evidence,
    )
    predecessor = (
        predecessor_evidence(path, predecessor_id)
        if predecessor_id is not None and path is not None
        else {"valid": True, "case_id": None, "verdict": "NOT_REQUIRED"}
    )
    runner = Runner(
        site_path=args.site,
        run_dir=args.run_dir,
        case_id=args.case,
        e2e_dir=args.e2e_dir,
        trusted_cpu_baseline=args.trusted_cpu_baseline,
        predecessor=predecessor,
    )
    return int(runner.run())


if __name__ == "__main__":
    raise SystemExit(main())

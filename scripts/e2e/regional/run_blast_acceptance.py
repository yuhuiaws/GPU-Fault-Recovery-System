from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.e2e.regional.blast_acceptance_base import (  # noqa: E402
    CASE_IDS,
    E2E001_CASE_ID,
    blast001_input_errors,
    default_e2e_dir,
    default_trusted_cpu_baseline,
)
from scripts.e2e.regional.blast_acceptance_cases_2 import (  # noqa: E402
    BlastCasesTwo as Runner,
)
from scripts.e2e.regional.regional_case_contract import (  # noqa: E402
    predecessor_path,
)
from scripts.e2e.regional.regional_live_fixture import (  # noqa: E402
    predecessor_evidence,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
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
    # Both default to this run's E2E-001 case directory once --run-dir is
    # known (see `resolve_blast001_inputs`); the old default of `.` passed the
    # exists() check on any working directory and then crashed reading
    # `./execution-card.json`.
    parser.add_argument(
        "--e2e-dir",
        type=Path,
        default=(
            Path(os.environ["GPU_FAULT_E2E001_EVIDENCE_DIR"])
            if os.getenv("GPU_FAULT_E2E001_EVIDENCE_DIR")
            else None
        ),
        help=f"E2E-001 case directory; defaults to <run-dir>/cases/{E2E001_CASE_ID}",
    )
    parser.add_argument(
        "--trusted-cpu-baseline",
        type=Path,
        default=(
            Path(os.environ["GPU_FAULT_TRUSTED_CPU_BASELINE"])
            if os.getenv("GPU_FAULT_TRUSTED_CPU_BASELINE")
            else None
        ),
        help=(
            "CPU node snapshot taken before the fault window; defaults to "
            "E2E-001's cpu-nodes-before.json"
        ),
    )
    parser.add_argument("--predecessor-evidence", default="")
    parser.add_argument(
        "--preflight-reuse-seconds",
        type=int,
        default=None,
        help="reuse a blast-preflight.json younger than this (default 30 minutes)",
    )
    return parser.parse_args(argv)


def resolve_blast001_inputs(args: argparse.Namespace) -> tuple[Path, Path]:
    e2e_dir = args.e2e_dir or default_e2e_dir(args.run_dir)
    baseline = args.trusted_cpu_baseline or default_trusted_cpu_baseline(e2e_dir)
    return e2e_dir, baseline


def main() -> int:
    args = parse_args()
    if args.site is None or args.run_dir is None:
        raise SystemExit("--site and --run-dir are required")
    if not args.site.is_file():
        raise SystemExit(f"site file does not exist: {args.site}")
    e2e_dir, trusted_cpu_baseline = resolve_blast001_inputs(args)
    if args.case == "GF-REGIONAL-BLAST-001":
        errors = blast001_input_errors(e2e_dir, trusted_cpu_baseline)
        if errors:
            raise SystemExit("BLAST-001 inputs are incomplete: " + "; ".join(errors))
    predecessor_id, path = predecessor_path(
        args.run_dir,
        args.case,
        args.predecessor_evidence,
    )
    runner_arguments = {
        "site_path": args.site,
        "run_dir": args.run_dir,
        "case_id": args.case,
        "e2e_dir": e2e_dir,
        "trusted_cpu_baseline": trusted_cpu_baseline,
    }
    if args.preflight_reuse_seconds is not None:
        runner_arguments["preflight_reuse_seconds"] = args.preflight_reuse_seconds
    # The predecessor's PASS must have been earned against the release and
    # cluster this audit reads, so the identity is resolved before the runner
    # judges the predecessor. A multi-cluster site has no single cluster_id;
    # the release binding still applies.
    runner = Runner(predecessor={"valid": True}, **runner_arguments)
    identity = runner.evidence_identity()
    predecessor = (
        predecessor_evidence(path, predecessor_id, **identity)
        if predecessor_id is not None and path is not None
        else {"valid": True, "case_id": None, "verdict": "NOT_REQUIRED"}
    )
    runner.predecessor = predecessor
    return int(runner.run())


if __name__ == "__main__":
    raise SystemExit(main())

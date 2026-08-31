from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.e2e.regional.capacity_acceptance_base import CapError  # noqa: E402
from scripts.e2e.regional.capacity_acceptance_cases import (  # noqa: E402
    CapacityAcceptanceCases,
)
from scripts.e2e.regional.live_driver_guard import (  # noqa: E402
    add_live_arguments,
    authorize_execution,
    build_plan,
)
from scripts.e2e.regional.regional_case_contract import (  # noqa: E402
    predecessor_path,
)
from scripts.e2e.regional.regional_live_fixture import (  # noqa: E402
    predecessor_evidence,
)

CASE_IDS = tuple(f"GF-REGIONAL-CAP-{number:03d}" for number in range(1, 5))
CapHarness = CapacityAcceptanceCases


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one isolated regional capacity acceptance case."
    )
    add_live_arguments(parser, confirmation="CASE_SPECIFIC_CONFIRMATION")
    parser.add_argument("--case", choices=CASE_IDS, required=True)
    parser.add_argument("--site", type=Path, required=True)
    parser.add_argument("--predecessor-evidence", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    os.umask(0o077)
    confirmation = args.case.removeprefix("GF-REGIONAL-").replace("-", "") + "_EXECUTE"
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
    environment = {
        "GPU_FAULT_SITE_FILE": str(args.site.resolve()),
        "GPU_FAULT_CAPACITY_CASE": args.case,
    }
    if not args.execute:
        plan = build_plan(
            run_dir=args.run_dir,
            case_id=args.case,
            attempt=args.attempt,
            confirmation=confirmation,
            environment=environment,
            details={
                "risk": "live-non-destructive",
                "predecessor": predecessor,
                "mutation": (
                    "create disposable capacity probe Deployments, Services, "
                    "Secrets and one isolated PostgreSQL database"
                ),
                "stop_conditions": [
                    "formal predecessor evidence is not PASS",
                    "the selected CPU EKS or release identity drifts",
                    "a probe cannot create or drop its isolated database",
                    "the production baseline changes",
                ],
                "rollback": {
                    "runner_finally_deletes_probe_resources": True,
                    "runner_finally_drops_isolated_databases": True,
                    "production_registry_is_not_modified": True,
                },
            },
        )
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0 if predecessor.get("valid", False) else 1
    if args.confirm != confirmation:
        raise CapError(f"confirmation must be exactly {confirmation}")
    authorize_execution(
        args,
        case_id=args.case,
        confirmation=confirmation,
        environment=environment,
    )
    harness = CapHarness(
        site_path=args.site,
        run_dir=args.run_dir,
        case_id=args.case,
        predecessor=predecessor,
    )
    return harness.run()


if __name__ == "__main__":
    raise SystemExit(main())

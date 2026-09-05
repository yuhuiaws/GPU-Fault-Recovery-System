from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.e2e.regional.acceptance_runner_common import (  # noqa: E402
    write_json_atomic,
)
from scripts.e2e.regional.boot_acceptance_common import (  # noqa: E402
    BootAcceptanceError,
    SiteFixture,
    utc_now,
)
from scripts.e2e.regional.boot_acceptance_lifecycle import (  # noqa: E402
    run_boot016,
    run_boot017,
    run_boot018,
)
from scripts.e2e.regional.boot_acceptance_runtime import (  # noqa: E402
    run_boot011,
    run_boot012,
    run_boot013,
    run_boot014,
    run_boot015,
)
from scripts.e2e.regional.live_driver_guard import (  # noqa: E402
    add_live_arguments,
    authorize_execution,
    build_plan,
    install_site_profile,
)
from scripts.e2e.regional.regional_case_contract import (  # noqa: E402
    case_evidence_path,
    predecessor_path,
)
from scripts.e2e.regional.regional_live_fixture import (  # noqa: E402
    predecessor_evidence,
)

CASE_IDS = tuple(f"GF-REGIONAL-BOOT-{number:03d}" for number in range(11, 19))


def case_plan(case_id: str, arguments: argparse.Namespace) -> dict[str, Any]:
    mutations = {
        "GF-REGIONAL-BOOT-011": "read isolated greenfield Kubernetes and IAM state",
        "GF-REGIONAL-BOOT-012": "run read-only TLS/readiness/STS negative probes",
        "GF-REGIONAL-BOOT-013": "run local SES client validation with zero delivery",
        "GF-REGIONAL-BOOT-014": "read dispatcher configuration and outbox state",
        "GF-REGIONAL-BOOT-015": (
            "insert one old unclaimable remote command and delete it in finally"
        ),
        "GF-REGIONAL-BOOT-016": (
            "deploy and uninstall one isolated site through gpu-fault-admin"
        ),
        "GF-REGIONAL-BOOT-017": "reuse BOOT-016 live site and run manual-order gates",
        "GF-REGIONAL-BOOT-018": (
            "build two temporary releases and run read-only live verify"
        ),
    }
    return {
        "risk": "case-defined",
        "case_id": case_id,
        "mutation": mutations[case_id],
        "target": {
            "site": str(arguments.site.resolve()) if arguments.site else None,
            "cluster_id": arguments.cluster_id or None,
            "bootstrap_state_dir": (
                str(arguments.bootstrap_state_dir.resolve())
                if arguments.bootstrap_state_dir
                else None
            ),
            "cpu_cluster_arn": arguments.cpu_cluster_arn or None,
            "gpu_cluster_count": len(arguments.gpu_cluster_arn),
        },
        "stop_conditions": [
            "formal predecessor evidence is not PASS",
            "site, Region, context or cluster identity drifts",
            "any Secret value would need to be copied into evidence",
            "a readiness, cleanup or residual assertion fails",
            "provider node replacement is requested or observed",
        ],
        "rollback": {
            "BOOT-015 deletes the injected remote command in finally": True,
            "BOOT-016 failure cleanup uses gpu-fault-admin uninstall": True,
            "BOOT-018 removes the isolated site unless explicitly retained": True,
            "temporary build directories are removed": True,
        },
    }


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        description="Run one guarded regional BOOT-011..018 acceptance case."
    )
    add_live_arguments(value, confirmation="CASE_SPECIFIC_CONFIRMATION")
    value.add_argument("--case", choices=CASE_IDS, required=True)
    value.add_argument("--site", type=Path)
    value.add_argument("--production-site", type=Path)
    value.add_argument("--cluster-id", default="")
    value.add_argument("--predecessor-evidence", default="")
    value.add_argument("--bootstrap-state-dir", type=Path)
    value.add_argument("--cpu-cluster-arn", default="")
    value.add_argument("--gpu-cluster-arn", action="append", default=[])
    value.add_argument("--admin-email", default="")
    value.add_argument("--retain-bootstrap-site", action="store_true")
    return value


def validate_arguments(arguments: argparse.Namespace) -> None:
    site_cases = {
        "GF-REGIONAL-BOOT-011",
        "GF-REGIONAL-BOOT-012",
        "GF-REGIONAL-BOOT-013",
        "GF-REGIONAL-BOOT-014",
        "GF-REGIONAL-BOOT-015",
    }
    if arguments.case in site_cases and arguments.site is None:
        raise BootAcceptanceError(f"{arguments.case} requires --site")
    if arguments.case == "GF-REGIONAL-BOOT-011":
        if arguments.production_site is None:
            raise BootAcceptanceError("BOOT-011 requires --production-site")
    if (
        arguments.case
        in {
            "GF-REGIONAL-BOOT-016",
            "GF-REGIONAL-BOOT-017",
            "GF-REGIONAL-BOOT-018",
        }
        and arguments.bootstrap_state_dir is None
    ):
        raise BootAcceptanceError(f"{arguments.case} requires --bootstrap-state-dir")
    if arguments.case == "GF-REGIONAL-BOOT-016" and not all(
        (
            arguments.cpu_cluster_arn,
            arguments.gpu_cluster_arn,
            arguments.admin_email,
        )
    ):
        raise BootAcceptanceError("BOOT-016 requires CPU/GPU ARNs and --admin-email")


def main() -> int:
    install_site_profile()
    arguments = parser().parse_args()
    os.umask(0o077)
    validate_arguments(arguments)
    confirmation = (
        arguments.case.removeprefix("GF-REGIONAL-").replace("-", "") + "_EXECUTE"
    )
    predecessor_id, path = predecessor_path(
        arguments.run_dir,
        arguments.case,
        arguments.predecessor_evidence,
    )
    predecessor = (
        predecessor_evidence(path, predecessor_id)
        if predecessor_id is not None and path is not None
        else {"valid": True, "case_id": None, "verdict": "NOT_REQUIRED"}
    )
    environment = {
        "GPU_FAULT_BOOT_CASE": arguments.case,
        "GPU_FAULT_SITE_FILE": str(arguments.site.resolve()) if arguments.site else "",
        "GPU_FAULT_TARGET_CLUSTER_ID": arguments.cluster_id,
        "GPU_FAULT_BOOTSTRAP_STATE_DIR": (
            str(arguments.bootstrap_state_dir.resolve())
            if arguments.bootstrap_state_dir
            else ""
        ),
    }
    if not arguments.execute:
        plan = build_plan(
            run_dir=arguments.run_dir,
            case_id=arguments.case,
            attempt=arguments.attempt,
            confirmation=confirmation,
            environment=environment,
            details={
                **case_plan(arguments.case, arguments),
                "predecessor": predecessor,
            },
        )
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0 if predecessor.get("valid", False) else 1
    if arguments.confirm != confirmation:
        raise BootAcceptanceError(f"confirmation must be exactly {confirmation}")
    authorize_execution(
        arguments,
        case_id=arguments.case,
        confirmation=confirmation,
        environment=environment,
    )
    if not predecessor.get("valid", False):
        raise BootAcceptanceError("formal predecessor evidence is not PASS")
    case_dir = arguments.run_dir / "cases" / arguments.case
    case_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    started_at = utc_now()
    try:
        if arguments.case == "GF-REGIONAL-BOOT-016":
            outcome = run_boot016(arguments, case_dir)
        elif arguments.case == "GF-REGIONAL-BOOT-017":
            outcome = run_boot017(arguments, predecessor, case_dir)
        elif arguments.case == "GF-REGIONAL-BOOT-018":
            outcome = run_boot018(arguments, case_dir)
        else:
            fixture = SiteFixture(arguments.site, arguments.cluster_id)
            handlers: dict[str, Callable[[], dict[str, Any]]] = {
                "GF-REGIONAL-BOOT-011": lambda: run_boot011(
                    fixture,
                    production_site=arguments.production_site,
                ),
                "GF-REGIONAL-BOOT-012": lambda: run_boot012(fixture),
                "GF-REGIONAL-BOOT-013": lambda: run_boot013(fixture),
                "GF-REGIONAL-BOOT-014": lambda: run_boot014(fixture),
                "GF-REGIONAL-BOOT-015": lambda: run_boot015(
                    fixture,
                    case_dir=case_dir,
                    attempt=arguments.attempt,
                ),
            }
            outcome = handlers[arguments.case]()
    except Exception as exc:
        outcome = {
            "verdict": "FAIL",
            "error": f"{type(exc).__name__}: {exc}",
            "limitations": [
                "The case stopped at the first failed assertion; later checks "
                "were not treated as executed."
            ],
        }
    result = {
        "schema_version": 2,
        "report_type": "fault-acceptance",
        "case_id": arguments.case,
        "verdict": outcome.get("verdict", "FAIL"),
        "started_at": started_at,
        "executed_at": utc_now(),
        "predecessor": predecessor,
        **{key: value for key, value in outcome.items() if key != "verdict"},
    }
    write_json_atomic(case_evidence_path(arguments.run_dir, arguments.case), result)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

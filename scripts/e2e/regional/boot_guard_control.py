"""Shared authorization and canonical evidence for the BOOT shell fixture."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from scripts.e2e.regional.acceptance_runner_common import (  # noqa: E402
    utc_now,
    write_json_atomic,
)
from scripts.e2e.regional.boot_acceptance_common import run  # noqa: E402
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
from scripts.e2e.regional.regional_live_fixture import predecessor_evidence  # noqa: E402
from scripts.e2e.regional.regional_pod_inventory import ready_pod_records  # noqa: E402

CONFIRMATION = "BOOT_GUARD_EXECUTE"
CASE_NUMBERS = (1, 2, 3, 4, 5, 7, 8, 9, 10)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    add_live_arguments(value, confirmation=CONFIRMATION)
    return value


def case_id(number: int) -> str:
    if number not in CASE_NUMBERS:
        raise RuntimeError("unknown or retired BOOT guard case")
    return f"GF-REGIONAL-BOOT-{number:03d}"


def healthy_api_pods(
    deployment: dict[str, Any], inventory: dict[str, Any]
) -> list[dict[str, Any]]:
    replicas = deployment.get("spec", {}).get("replicas")
    records = ready_pod_records(inventory)
    if (
        type(replicas) is not int
        or replicas != 3
        or len(inventory["items"]) != replicas
        or len(records) != replicas
        or len({item["uid"] for item in records}) != replicas
        or len({item["name"] for item in records}) != replicas
        or len({item["node"] for item in records}) != replicas
        or any(not item["node"] for item in records)
    ):
        raise RuntimeError(
            "BOOT guard requires three complete Ready API replicas on distinct nodes"
        )
    return records


def target_identity(environment: dict[str, str]) -> dict[str, Any]:
    prefix = [
        "kubectl",
        "--kubeconfig",
        environment["CPU_KUBECONFIG"],
        "-n",
        environment["NAMESPACE"],
    ]
    metadata = json.loads(
        run(
            [
                *prefix,
                "get",
                "configmap",
                "gpu-fault-release-metadata",
                "-o",
                "json",
            ]
        ).stdout
    )
    deployment = json.loads(
        run(
            [
                *prefix,
                "get",
                "deployment",
                "gpu-fault-api-ha",
                "-o",
                "json",
            ]
        ).stdout
    )
    inventory = json.loads(
        run(
            [
                *prefix,
                "get",
                "pod",
                "-l",
                "app=gpu-fault-api-ha",
                "-o",
                "json",
            ]
        ).stdout
    )
    pods = healthy_api_pods(deployment, inventory)
    release_id = metadata.get("data", {}).get("release-id")
    uid = deployment.get("metadata", {}).get("uid")
    generation = deployment.get("metadata", {}).get("generation")
    if (
        not isinstance(release_id, str)
        or not release_id
        or not isinstance(uid, str)
        or not uid
        or type(generation) is not int
        or generation < 1
    ):
        raise RuntimeError("BOOT guard CPU target identity is incomplete")
    return {
        "release_id": release_id,
        "cpu_deployment_uid": uid,
        "generation": generation,
        "cpu_pods": pods,
    }


def predecessor(run_dir: Path, identifier: str, release_id: str) -> dict[str, Any]:
    previous, path = predecessor_path(run_dir, identifier, "")
    if previous is None or path is None:
        return {"valid": True, "verdict": "NOT_REQUIRED"}
    return predecessor_evidence(path, previous, release_id=release_id)


def main() -> int:
    install_site_profile()
    arguments = parser().parse_args()
    environment = {
        name: os.environ.get(name, "").strip()
        for name in (
            "CPU_KUBECONFIG",
            "NAMESPACE",
            "AWS_REGION",
            "CPU_HYPERPOD_CLUSTER",
            "BOOT_GUARD_START_CASE",
            "GUARD_PROBE_BASE",
        )
    }
    if not all(environment.values()):
        raise RuntimeError("BOOT guard scope environment is incomplete")
    if arguments.run_dir.resolve() != Path(os.environ["RUN_DIR"]).resolve():
        raise RuntimeError("--run-dir must match the shell RUN_DIR")
    start = int(environment["BOOT_GUARD_START_CASE"])
    if start not in {1, 7, 8}:
        raise RuntimeError("BOOT guard start must be 1, 7, or 8")
    selected = [case_id(number) for number in CASE_NUMBERS if number >= start]
    identity = target_identity(environment)
    previous = predecessor(arguments.run_dir, selected[0], identity["release_id"])
    details = {
        "cases": selected,
        "identity": identity,
        "predecessor": previous,
        "mutation": "disposable CPU startup probe, Secrets and isolated database only",
        "stop_conditions": [
            "predecessor is not PASS",
            "CPU release or Deployment identity changed",
            "database or credential isolation is unproven",
            "maintenance window ended",
            "cleanup or Pod/readiness evidence is incomplete",
        ],
        "rollback": "delete probe, verify Pod/Service absence, then drop only its database",
    }
    if not arguments.execute:
        plan = build_plan(
            run_dir=arguments.run_dir,
            case_id=selected[0],
            attempt=arguments.attempt,
            confirmation=CONFIRMATION,
            details=details,
            arguments=arguments,
            preflight_passed=previous.get("valid") is True,
            environment=environment,
        )
        print(json.dumps(plan, sort_keys=True))
        return 0 if previous.get("valid") is True else 1
    authorize_execution(
        arguments,
        case_id=selected[0],
        confirmation=CONFIRMATION,
        environment=environment,
        details=details,
    )
    if previous.get("valid") is not True:
        raise RuntimeError("BOOT guard predecessor is not PASS")
    record = os.environ.get("BOOT_GUARD_RECORD_VERDICT", "")
    if record:
        identifier = os.environ.get("BOOT_GUARD_RECORD_CASE", "")
        if identifier not in selected or record not in {"PASS", "FAIL"}:
            raise RuntimeError("invalid BOOT guard evidence request")
        prior = predecessor(arguments.run_dir, identifier, identity["release_id"])
        document = {
            "schema_version": 2,
            "report_type": "fault-acceptance",
            "case_id": identifier,
            "attempt": arguments.attempt,
            "verdict": record if prior.get("valid") is True else "FAIL",
            "executed_at": utc_now(),
            "predecessor": prior,
            "cluster_id": prior.get("evidence_cluster_id"),
            **identity,
            "text_evidence": str(arguments.run_dir / "cases" / f"{identifier}.txt"),
        }
        write_json_atomic(case_evidence_path(arguments.run_dir, identifier), document)
        if prior.get("valid") is not True:
            raise RuntimeError("formal BOOT guard predecessor is not PASS")
    print("EXECUTION_AUTHORIZED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

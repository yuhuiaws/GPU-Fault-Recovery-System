#!/usr/bin/env python3
"""BOOT-030/031: read-only deployed remote-command/workflow state-table proof."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.e2e.regional.acceptance_runner_common import utc_now, write_json_atomic  # noqa: E402
from scripts.e2e.regional.acceptance_supervision import bind_command_supervision  # noqa: E402
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
    RegionalLiveFixture,
    component_python,
    install_abort_signals,
    predecessor_evidence,
    run_case_main,
    settings_from_arguments,
)
from scripts.e2e.regional.regional_commands import RegionalFixtureError  # noqa: E402
from gpu_fault.schema_migrations import LATEST_POSTGRES_SCHEMA_VERSION  # noqa: E402

CASES = {"GF-REGIONAL-BOOT-030": "remote_command", "GF-REGIONAL-BOOT-031": "workflow"}
CONFIRMATION = "VERIFY_DEPLOYED_STATE_TABLE_READ_ONLY"
PROBE = Path(__file__).with_name("probes") / "state_table_snapshot.py"
CPU_ROLES = (
    "gpu-fault-api-ha",
    "gpu-fault-control-worker",
    "gpu-fault-telemetry-spool-worker",
)
MODES = ("legacy", "dual", "dedicated")
STATE_FIELDS = (
    "kind",
    "mode",
    "revision",
    "legacy_rows",
    "dedicated_rows",
    "backfill_complete",
    "legacy_purged",
    "verification_applicable",
    "verification_performed",
    "verified",
    "missing_rows",
    "mismatched_rows",
    "extra_rows",
    "invalid_records",
    "noncanonical_records",
)


def evidence_identity(regional: RegionalLiveFixture) -> dict[str, str]:
    value = regional.evidence_identity()
    fields = ("release_id", "cluster_id")
    if not isinstance(value, dict) or any(
        not isinstance(value.get(key), str) or not value[key].strip() for key in fields
    ):
        raise RegionalFixtureError("release/cluster identity is unavailable")
    return {key: value[key] for key in fields}


def cpu_population(regional: RegionalLiveFixture) -> dict[str, Any]:
    population: dict[str, Any] = {}
    pod_names: set[str] = set()
    pod_uids: set[str] = set()
    deployment_uids: set[str] = set()
    for app in CPU_ROLES:
        value = json.loads(
            regional.kubectl("cpu", "get", "deployment", app, "-o", "json")
        )
        metadata, spec, status = (
            value["metadata"],
            value["spec"],
            value.get("status", {}),
        )
        replicas, generation = spec.get("replicas"), metadata.get("generation")
        if (
            type(replicas) is not int
            or replicas < 0
            or type(generation) is not int
            or generation < 1
            or not isinstance(metadata.get("uid"), str)
            or not metadata["uid"]
            or metadata.get("name") != app
            or metadata["uid"] in deployment_uids
            or metadata.get("deletionTimestamp")
        ):
            raise RegionalFixtureError(
                "CPU Deployment identity or replica count is invalid"
            )
        deployment_uids.add(metadata["uid"])
        observed = regional.ready_pods("cpu", app)
        if not isinstance(observed, list) or any(
            not isinstance(pod, dict)
            or any(
                not isinstance(pod.get(key), str) or not pod[key]
                for key in ("name", "uid")
            )
            for pod in observed
        ):
            raise RegionalFixtureError("CPU Pod identity is missing or malformed")
        pods = sorted(
            ({"name": pod["name"], "uid": pod["uid"]} for pod in observed),
            key=lambda pod: pod["name"],
        )
        for pod in pods:
            if pod["name"] in pod_names or pod["uid"] in pod_uids:
                raise RegionalFixtureError(
                    "CPU Pod identity is duplicated across the population"
                )
            pod_names.add(pod["name"])
            pod_uids.add(pod["uid"])
        if replicas == 0:
            if (
                app != CPU_ROLES[-1]
                or pods
                or any(
                    type(status.get(key, 0)) is not int or status.get(key, 0) != 0
                    for key in (
                        "replicas",
                        "readyReplicas",
                        "updatedReplicas",
                        "availableReplicas",
                    )
                )
            ):
                raise RegionalFixtureError(
                    "required CPU role is disabled or has a residual population"
                )
        elif (
            type(status.get("observedGeneration")) is not int
            or status["observedGeneration"] < generation
            or any(
                type(status.get(key)) is not int or status[key] != replicas
                for key in (
                    "replicas",
                    "readyReplicas",
                    "updatedReplicas",
                    "availableReplicas",
                )
            )
            or len(pods) != replicas
            or len({(pod["name"], pod["uid"]) for pod in pods}) != replicas
        ):
            raise RegionalFixtureError(
                "CPU role does not have its complete Ready population"
            )
        population[app] = {
            "uid": metadata["uid"],
            "generation": generation,
            "replicas": replicas,
            "pods": pods,
        }
    return population


def state_errors(
    report: dict[str, Any], kind: str, expected_mode: str, *, verify: bool
) -> list[str]:
    errors = []
    if kind not in CASES.values() or expected_mode not in MODES:
        errors.append("unsupported state-table acceptance target")
    if report.get("verdict", "PASS") != "PASS":
        errors.append("state-table probe reported failure")
    if (
        type(report.get("schema_version")) is not int
        or report["schema_version"] != LATEST_POSTGRES_SCHEMA_VERSION
    ):
        errors.append("schema version differs from this release")
    if any(
        report.get(key) is not True for key in ("schema_valid", "read_only", "writer")
    ):
        errors.append("schema/read-only writer proof is incomplete")
    digest = report.get("database_sha256")
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(c not in "0123456789abcdef" for c in digest)
    ):
        errors.append("database identity is missing")
    state = report.get("state")
    if not isinstance(state, dict):
        return [*errors, "migration state is missing"]
    if state.get("kind") != kind or state.get("mode") != expected_mode:
        errors.append("state table or migration mode differs from the approved target")
    for key in ("revision", "legacy_rows", "dedicated_rows"):
        if type(state.get(key)) is not int or state[key] < 0:
            errors.append(f"{key} is not a nonnegative integer")
    if any(
        type(state.get(key)) is not bool
        for key in ("backfill_complete", "legacy_purged")
    ):
        errors.append("migration completion flags are malformed")
    if state.get("verification_applicable") is not (expected_mode == "dual"):
        errors.append("verification applicability differs from the migration mode")
    if state.get("verification_performed") is not (verify and expected_mode == "dual"):
        errors.append("verification execution differs from the requested scope")
    if expected_mode == "dual" and verify:
        if (
            state.get("verification_performed") is not True
            or state.get("verified") is not True
        ):
            errors.append("dual-write equality was not verified")
        if state.get("backfill_complete") is not True:
            errors.append("dual-write backfill is incomplete")
        for key in (
            "missing_rows",
            "mismatched_rows",
            "extra_rows",
            "invalid_records",
            "noncanonical_records",
        ):
            if type(state.get(key)) is not int or state[key] != 0:
                errors.append(f"dual-write verification failed at {key}")
    if expected_mode == "dedicated" and state.get("backfill_complete") is not True:
        errors.append("dedicated mode lacks completed backfill")
    if state.get("legacy_purged") is True and (
        state.get("legacy_rows") != 0 or expected_mode != "dedicated"
    ):
        errors.append(
            "legacy purge marker conflicts with retained rows or migration mode"
        )
    return errors


def audit(
    regional: RegionalLiveFixture,
    kind: str,
    expected_mode: str,
    *,
    verify: bool,
    deadline: datetime,
    planned: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if kind not in CASES.values() or expected_mode not in MODES:
        raise RegionalFixtureError("unsupported state-table acceptance target")
    if deadline.tzinfo is None:
        raise RegionalFixtureError("state-table audit deadline needs a timezone")
    identity = evidence_identity(regional)
    before = cpu_population(regional)
    if planned is not None and (
        planned.get("identity") != identity
        or planned.get("population") != before
        or planned.get("kind") != kind
        or planned.get("expected_mode") != expected_mode
    ):
        raise RegionalFixtureError(
            "approved state-table target changed before the audit"
        )
    snapshots: list[dict[str, Any]] = []
    source = PROBE.read_text(encoding="utf-8")
    for app, deployment in before.items():
        for pod in deployment["pods"]:
            if (deadline - datetime.now(timezone.utc)).total_seconds() < 45:
                raise RegionalFixtureError(
                    "state-table audit deadline has insufficient remaining time"
                )
            full = verify and expected_mode == "dual" and not snapshots
            output = regional.kubectl(
                "cpu",
                "exec",
                "-i",
                pod["name"],
                "--",
                component_python("cpu"),
                "-",
                kind,
                "verify" if full else "metadata",
                input_text=source,
                timeout=40,
            )
            value = json.loads(output.strip().splitlines()[-1])
            if not isinstance(value, dict):
                raise RegionalFixtureError("state-table probe returned a non-object")
            failures = state_errors(value, kind, expected_mode, verify=full)
            if failures:
                raise RegionalFixtureError("; ".join(failures))
            snapshot = {
                key: value[key]
                for key in (
                    "schema_version",
                    "schema_valid",
                    "read_only",
                    "writer",
                    "database_sha256",
                )
            }
            snapshot["state"] = {
                key: value["state"][key]
                for key in STATE_FIELDS
                if key in value["state"]
            }
            snapshots.append(
                {**snapshot, "app": app, "pod": pod["name"], "pod_uid": pod["uid"]}
            )
    if not snapshots:
        raise RegionalFixtureError("no CPU state-table proof was collected")
    bindings = {
        (
            item["database_sha256"],
            item["schema_version"],
            item["state"]["mode"],
            item["state"]["revision"],
        )
        for item in snapshots
    }
    if (
        len(bindings) != 1
        or cpu_population(regional) != before
        or evidence_identity(regional) != identity
    ):
        raise RegionalFixtureError(
            "deployment, database or migration identity changed during the audit"
        )
    if datetime.now(timezone.utc) >= deadline:
        raise RegionalFixtureError("state-table audit exceeded the approved deadline")
    return {
        "verdict": "PASS",
        **identity,
        "kind": kind,
        "expected_mode": expected_mode,
        "snapshots": snapshots,
        "mutation_performed": False,
        "cleanup": {
            "required": False,
            "reason": "read-only SQL and Kubernetes reads only",
        },
        "limitations": [
            "This is a point-in-time deployed-state proof, not an online migration or failover drill.",
            "No mode switch, backfill, legacy purge, HOT-rate measurement or schema DDL is performed.",
            "Dual mode compares full reconstructed records once; other CPU Pods verify the same database/mode.",
        ],
    }


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    add_live_arguments(value, confirmation=CONFIRMATION)
    value.add_argument("--case", choices=tuple(CASES), required=True)
    value.add_argument("--expected-mode", choices=MODES, required=True)
    for name in (
        "cpu-kubeconfig",
        "gpu-kubeconfig",
        "gpu-context",
        "cluster-id",
        "region",
        "predecessor-evidence",
    ):
        value.add_argument(f"--{name}", default="")
    value.add_argument("--namespace", default="gpu-fault-system")
    return value


def preflight(
    regional: RegionalLiveFixture, arguments: argparse.Namespace
) -> dict[str, Any]:
    details: dict[str, Any] = {
        "identity": {},
        "predecessor": {"valid": False},
        "population": {},
        "kind": CASES[arguments.case],
        "expected_mode": arguments.expected_mode,
        "risk": "read-only-signal-replay",
        "mutation": "none",
        "errors": [],
    }
    errors = details["errors"]
    try:
        identity = evidence_identity(regional)
        details["identity"] = identity
        previous_id, path = predecessor_path(
            arguments.run_dir, arguments.case, arguments.predecessor_evidence
        )
        predecessor = (
            predecessor_evidence(path, previous_id, **identity)
            if previous_id is not None and path is not None
            else {"valid": True, "verdict": "NOT_REQUIRED"}
        )
        details["predecessor"] = predecessor
        if predecessor.get("valid") is not True:
            errors.append("formal predecessor is not PASS")
    except Exception as exc:
        errors.append(
            f"release/predecessor could not be verified ({type(exc).__name__})"
        )
    try:
        details["population"] = cpu_population(regional)
    except Exception as exc:
        errors.append(f"CPU inventory could not be verified ({type(exc).__name__})")
    return details


def execute(
    regional: RegionalLiveFixture,
    arguments: argparse.Namespace,
    deadline: datetime,
    details: dict[str, Any],
) -> dict[str, Any]:
    plan = json.loads(
        (arguments.run_dir / "cases" / arguments.case / "plan.json").read_text()
    )
    if details["errors"] or plan["details"]["identity"] != details["identity"]:
        raise RegionalFixtureError("predecessor or release identity changed")
    outcome = audit(
        regional,
        CASES[arguments.case],
        arguments.expected_mode,
        verify=True,
        deadline=deadline,
        planned=plan["details"],
    )
    return {**outcome, "predecessor": details["predecessor"]}


def main() -> int:
    install_site_profile()
    arguments = parser().parse_args()
    os.umask(0o077)
    path = (
        case_evidence_path(arguments.run_dir, arguments.case)
        if arguments.execute
        else arguments.run_dir / "cases" / arguments.case / "plan.json"
    )
    # Invalidate an earlier proof before any operation that can fail, including reads.
    result: dict[str, Any] = {
        "case_id": arguments.case,
        "attempt": arguments.attempt,
        "verdict": "NOT_RUN",
        "status": "RUNNING",
        "mutation_performed": False,
        "started_at": utc_now(),
    }
    if not arguments.execute:
        result["preflight_passed"] = False
    write_json_atomic(path, result)
    try:
        bind_command_supervision(arguments.run_dir)
        install_abort_signals()
        regional = RegionalLiveFixture(settings_from_arguments(arguments))
        environment = regional.settings.environment()
        deadline = (
            authorize_execution(
                arguments,
                case_id=arguments.case,
                confirmation=CONFIRMATION,
                environment=environment,
            )
            if arguments.execute
            else None
        )
        details = preflight(regional, arguments)
        if not arguments.execute:
            result = build_plan(
                run_dir=arguments.run_dir,
                case_id=arguments.case,
                attempt=arguments.attempt,
                confirmation=CONFIRMATION,
                environment=environment,
                arguments=arguments,
                preflight_passed=not details["errors"],
                details=details,
            )
        else:
            assert deadline is not None
            result.update(details["identity"])
            result["predecessor"] = details["predecessor"]
            result.update(execute(regional, arguments, deadline, details))
    except BaseException as exc:
        result["verdict"] = "FAIL"
        result["error_type"] = type(exc).__name__
        result["reason"] = "read-only state-table verification failed"
        if not isinstance(exc, Exception):
            raise
    finally:
        passed = (
            result.get("verdict") == "PASS"
            if arguments.execute
            else result.get("preflight_passed") is True
        )
        result["status"] = "COMPLETED" if passed else "FAILED"
        result["completed_at"] = utc_now()
        write_json_atomic(path, result)
    print(json.dumps(result, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(run_case_main(main))

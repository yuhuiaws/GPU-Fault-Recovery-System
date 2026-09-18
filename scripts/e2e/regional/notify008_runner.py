"""Approved-plan orchestration for an isolated, explicitly SIMULATED provider."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from scripts.e2e.regional.acceptance_runner_common import utc_now, write_json_atomic
from scripts.e2e.regional.acceptance_scope import current_acceptance_scope
from scripts.e2e.regional.live_driver_guard import CaseRunner, add_live_arguments
from scripts.e2e.regional.notify008_bundle import source_bundle
from scripts.e2e.regional.notify008_fixture import CLEANUP_SECONDS, CpuAPI, Sandbox
from scripts.e2e.regional.notify008_target import (
    Settings,
    predecessor,
    sandbox_preflight,
    source_target,
)
from scripts.e2e.regional.probes.notify008_protocol import (
    CASE_ID,
    CONFIRMATION,
    SCOPE,
    ProbeError,
    Target,
    digest,
    report_errors,
)
from scripts.e2e.regional.regional_case_contract import case_evidence_path


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        description="Isolated PostgreSQL/runtime notification ambiguity with a SIMULATED provider."
    )
    add_live_arguments(value, confirmation=CONFIRMATION)
    for name in (
        "cpu-kubeconfig",
        "cpu-context",
        "cluster-id",
        "region",
        "postgres-image",
    ):
        value.add_argument(f"--{name}", required=True)
    value.add_argument("--namespace", default="gpu-fault-system")
    return value


def configure(arguments: argparse.Namespace) -> Settings:
    path = Path(arguments.cpu_kubeconfig).expanduser().resolve()
    values = (
        arguments.cpu_context,
        arguments.namespace,
        arguments.cluster_id,
        arguments.region,
        arguments.postgres_image,
    )
    if not path.is_file() or any(
        not isinstance(value, str) or not value for value in values
    ):
        raise ProbeError(
            "notification scenario CPU connection or identity is incomplete"
        )
    return Settings(path, *values)


def read_only_preflight(settings: Settings, case_dir: Path) -> dict[str, Any]:
    try:
        api = CpuAPI(settings.cpu_kubeconfig, settings.cpu_context)
        target = source_target(api, settings, f"notify008-{uuid4().hex[:16]}")
        prior = predecessor(case_dir.parents[1], target)
        version = json.loads(api.call("get", "--raw", "/version"))
        if int(str(version["minor"]).rstrip("+")) < 30:
            raise ProbeError("NOTIFY008 requires stable Pod scheduling gates")
        sandbox_preflight(api, target)
        return {
            "errors": [],
            "target": asdict(target),
            "predecessor": prior,
            "bundle_sha256": digest(source_bundle()),
        }
    except Exception as exc:
        return {"errors": [f"preflight refused: {type(exc).__name__}"]}


def plan_details(settings: Settings, preflight: dict[str, Any]) -> dict[str, Any]:
    return {
        **preflight,
        "risk": "live-non-destructive",
        "provider": "SIMULATED",
        "validation_scope": SCOPE,
        "mutation": "create one owned CPU namespace, namespace-owned priority-zero non-default Never PriorityClass, gated/inert Job and private PostgreSQL; kill only probe-owned runtime children",
        "stop_conditions": [
            "canonical predecessor, release, source Pod, CPU node, resource UID or image drift",
            "unknown admitted Pod fields, credentials, service-account injection or host access",
            "deadline, ambiguous mutation acknowledgement, missing independent acceptance or cleanup proof",
        ],
        "cleanup": "stop both owned container PID1 barriers, observe termination, UID-delete namespace and owned PriorityClass, independently verify both absent",
        "cleanup_seconds": CLEANUP_SECONDS,
        "limitations": [
            "SIMULATED provider, not external SNS/SES mail",
            "isolated PostgreSQL, not production Aurora data or failover",
            "provider acceptance before commit can be delivered again; no exactly-once guarantee",
        ],
    }


def begin_execution(run_dir: Path, attempt: int) -> Path:
    path = case_evidence_path(run_dir, CASE_ID)
    if path.exists():
        path.replace(path.with_name(f"{CASE_ID}.before-{attempt}-{uuid4().hex}.json"))
    write_json_atomic(
        path,
        {
            "schema_version": 2,
            "report_type": "fault-acceptance",
            "case_id": CASE_ID,
            "attempt": attempt,
            "verdict": "FAIL",
            "status": "RUNNING",
            "formal_sequence_satisfied": False,
            "errors": ["authorized execution has not completed"],
        },
    )
    return path


def execute_case(
    settings: Settings, run_dir: Path, attempt: int, deadline: datetime
) -> int:
    # The shared caller has authorized this attempt; no stale PASS may outlive it.
    path = begin_execution(run_dir, attempt)
    case_dir = path.parent
    target: Target | None = None
    sandbox: Sandbox | None = None
    prior: dict[str, Any] = {}
    scope: dict[str, Any] = {"formal_sequence_satisfied": False}
    started: str | None = None
    outcome: dict[str, Any] = {"verdict": "FAIL", "errors": []}
    interrupted: BaseException | None = None
    try:
        started = utc_now()
        scope = current_acceptance_scope().result_fields()
        plan = json.loads((case_dir / "plan.json").read_text(encoding="utf-8"))
        details = plan["details"]
        target = Target(**details["target"])
        if details.get("bundle_sha256") != digest(source_bundle()):
            raise ProbeError("approved portable probe bundle changed")
        if (
            deadline - datetime.now(UTC)
        ).total_seconds() < target.seconds + CLEANUP_SECONDS:
            raise ProbeError(
                "maintenance window cannot contain execution and cleanup budgets"
            )
        api = CpuAPI(settings.cpu_kubeconfig, settings.cpu_context)

        def check_target() -> None:
            source_target(api, settings, target.run_id, expected=target)

        check_target()
        prior = predecessor(run_dir, target)
        sandbox = Sandbox(api, target, case_dir, deadline, check_target)
        sandbox.create()
        sandbox.admit()
        inspected = sandbox.execute_probe("inspect")
        if (
            inspected.get("armed") is not False
            or inspected.get("stop_requested") is not False
        ):
            raise ProbeError("newly admitted sandbox is not inert")
        sandbox.arm()
        prepared = sandbox.execute_probe("prepare", timeout=90)
        if prepared != {
            "postgres_major": 16,
            "database_is_unix_socket": True,
            "production_credentials_loaded": False,
        }:
            raise ProbeError("private PostgreSQL preparation identity differs")
        probe = sandbox.execute_probe("run", timeout=target.seconds)
        errors = report_errors(probe, run_id=target.run_id)
        if errors or probe.get("pod_uid") != sandbox.record["pod"]["uid"]:
            raise ProbeError(
                "isolated process/provider/Store report did not satisfy the contract"
            )
        check_target()
        sandbox.check_priority_class()
        outcome = {"verdict": "PASS", "errors": [], "probe": probe}
    except BaseException as exc:
        outcome["verdict"] = "FAIL"
        outcome["errors"].append(f"execution refused: {type(exc).__name__}")
        if not isinstance(exc, Exception):
            interrupted = exc
    finally:
        cleanup = {
            "namespace_absent": sandbox is None,
            "priorityclass_absent": sandbox is None,
            "process_termination_proven": sandbox is None,
            "errors": [],
        }
        if sandbox is not None:
            try:
                cleanup = sandbox.cleanup()
            except BaseException as exc:
                cleanup = {
                    "namespace_absent": False,
                    "priorityclass_absent": False,
                    "process_termination_proven": False,
                    "errors": [f"cleanup failed: {type(exc).__name__}"],
                }
                if interrupted is None and not isinstance(exc, Exception):
                    interrupted = exc
        if (
            cleanup["errors"]
            or cleanup["namespace_absent"] is not True
            or cleanup.get("priorityclass_absent") is not True
            or cleanup["process_termination_proven"] is not True
        ):
            outcome["verdict"] = "FAIL"
            outcome["errors"].append(
                "owned cleanup or process termination was not proven"
            )
        result = {
            "schema_version": 2,
            "report_type": "fault-acceptance",
            "case_id": CASE_ID,
            "attempt": attempt,
            "release_id": target.release_id if target is not None else None,
            "cluster_id": settings.cluster_id,
            "started_at": started,
            "executed_at": utc_now(),
            **scope,
            "validation_scope": SCOPE,
            "provider": "SIMULATED",
            "predecessor": prior,
            "runtime_binding": asdict(target) if target is not None else None,
            **outcome,
            "status": "COMPLETED" if outcome["verdict"] == "PASS" else "FAILED",
            "cleanup": cleanup,
        }
        write_json_atomic(path, result)
    if interrupted is not None:
        raise interrupted
    return 0 if outcome["verdict"] == "PASS" else 1


CASE = CaseRunner(
    case_id=CASE_ID,
    confirmation=CONFIRMATION,
    parser=parser,
    configure=configure,
    read_only_preflight=read_only_preflight,
    plan_details=plan_details,
    execute_case=execute_case,
)

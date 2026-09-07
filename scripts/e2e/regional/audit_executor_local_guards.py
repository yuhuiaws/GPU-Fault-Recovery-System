"""Executor-local guard fixture for GF-REGIONAL-ISO-002 and GF-REGIONAL-CMD-011.

Both cases prove a refusal the cluster executor makes *before* it talks to any
adapter: a command whose cluster is not the executor's own, and a command whose
fencing token disagrees with the workflow/incident it carries. Neither command is
ever written to a store or a queue -- the executor object is built bare and fed
the malformed command directly -- so the fixture has no side effects and needs
no cluster.

Given ``--run-dir`` it also writes ``cases/<id>/<id>.json`` per case so the two
cases carry evidence like the rest of the CMD group; stdout keeps the same
two-key JSON it always printed.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from gpu_fault.cluster_executor import ClusterActionExecutor
from gpu_fault.models import (
    FaultIncident,
    WorkflowOperation,
    WorkflowRequest,
    WorkflowStatus,
    WorkflowStepSpec,
)
from gpu_fault.regional import RemoteActionCommand

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ISO_002 = "GF-REGIONAL-ISO-002"
CMD_011 = "GF-REGIONAL-CMD-011"
CASE_IDS = (ISO_002, CMD_011)


def _executor(cluster_id: str) -> ClusterActionExecutor:
    executor = object.__new__(ClusterActionExecutor)
    executor.client = SimpleNamespace(cluster_id=cluster_id)
    executor.allowed_namespaces = set()
    executor.fleet_registry = None
    executor.adapters = []
    executor.executor_id = "regional-local-guard-audit"
    executor.unexpected_failures = 0
    return executor


def _command(
    *,
    cluster_id: str,
    command_fencing_token: int,
    workflow_fencing_token: int,
    incident_fencing_token: int,
    suffix: str,
) -> RemoteActionCommand:
    incident = FaultIncident(
        incident_id=f"audit-{suffix}-incident",
        event_id=f"audit-{suffix}-event",
        event_type="REGIONAL_LOCAL_GUARD_AUDIT",
        cluster_id=cluster_id,
        node_ids=["audit-nonexistent-node"],
        policy_version="audit-v1",
        policy_source="audit",
        fencing_token=incident_fencing_token,
    )
    step = WorkflowStepSpec(
        operation=WorkflowOperation.TRIGGER_HEALTH_SNAPSHOT,
        execution_owner="gpu-fault-node-agent",
        node_ids=["audit-nonexistent-node"],
    )
    workflow = WorkflowRequest(
        request_id=f"audit-{suffix}-workflow",
        incident_id=incident.incident_id,
        status=WorkflowStatus.RUNNING,
        fencing_token=workflow_fencing_token,
        official_steps=[step],
    )
    return RemoteActionCommand(
        command_id=f"audit-{suffix}-command",
        cluster_id=cluster_id,
        workflow_request_id=workflow.request_id,
        incident_id=incident.incident_id,
        step_index=0,
        fencing_token=command_fencing_token,
        idempotency_key=f"audit/{suffix}",
        lease_token=f"audit-{suffix}-lease-token",
        step=step,
        workflow=workflow,
        incident=incident,
    )


def evaluate(
    cluster_id: str, other_cluster_id: str
) -> tuple[dict[str, dict[str, Any]], dict[str, list[str]]]:
    """Run both guards; return the observed outcomes and per-case failures."""

    executor = _executor(cluster_id)
    cross_cluster = executor._execute(
        _command(
            cluster_id=other_cluster_id,
            command_fencing_token=1,
            workflow_fencing_token=1,
            incident_fencing_token=1,
            suffix="iso002",
        )
    )
    stale_fencing = executor._execute(
        _command(
            cluster_id=cluster_id,
            command_fencing_token=1,
            workflow_fencing_token=1,
            incident_fencing_token=2,
            suffix="cmd011",
        )
    )
    payload = {
        ISO_002: {
            "status": cross_cluster.status.value,
            "status_source": cross_cluster.status_source,
            "error": cross_cluster.error,
        },
        CMD_011: {
            "status": stale_fencing.status.value,
            "status_source": stale_fencing.status_source,
            "error": stale_fencing.error,
        },
    }
    errors: dict[str, list[str]] = {ISO_002: [], CMD_011: []}
    if cross_cluster.status.value != "FAILED":
        errors[ISO_002].append(f"status is {cross_cluster.status.value}, not FAILED")
    if cross_cluster.status_source != "executor-rejected":
        errors[ISO_002].append(f"status_source is {cross_cluster.status_source!r}")
    if "command cluster does not match executor cluster" not in (
        cross_cluster.error or ""
    ):
        errors[ISO_002].append(
            f"error does not name the cluster mismatch: {cross_cluster.error!r}"
        )
    if stale_fencing.status.value != "FAILED":
        errors[CMD_011].append(f"status is {stale_fencing.status.value}, not FAILED")
    if stale_fencing.status_source != "executor-rejected":
        errors[CMD_011].append(f"status_source is {stale_fencing.status_source!r}")
    if "command fencing token does not match workflow/incident" not in (
        stale_fencing.error or ""
    ):
        errors[CMD_011].append(
            f"error does not name the fencing mismatch: {stale_fencing.error!r}"
        )
    if executor.unexpected_failures != 0:
        for case_id in CASE_IDS:
            errors[case_id].append(
                "the guard counted an unexpected failure "
                f"({executor.unexpected_failures}); a refusal is not an internal error"
            )
    return payload, errors


def write_evidence(
    run_dir: Path,
    payload: dict[str, dict[str, Any]],
    errors: dict[str, list[str]],
    *,
    cluster_id: str,
    release_id: str = "",
) -> list[Path]:
    """One ``cases/<id>/<id>.json`` per case, through the shared scoped writer."""

    from scripts.e2e.regional.acceptance_runner_common import write_json_atomic
    from scripts.e2e.regional.regional_case_contract import case_evidence_path

    written = []
    for case_id in CASE_IDS:
        document: dict[str, Any] = {
            "schema_version": 1,
            "report_type": "fault-acceptance",
            "case_id": case_id,
            "verdict": "PASS" if not errors[case_id] else "FAIL",
            "cluster_id": cluster_id,
            "executed_at": datetime.now(timezone.utc).isoformat(),
            "fixture": "executor-local guard; no store, queue or cluster touched",
            "observed": payload[case_id],
            "errors": list(errors[case_id]),
        }
        if release_id:
            document["release_id"] = release_id
        path = case_evidence_path(run_dir, case_id)
        write_json_atomic(path, document)
        written.append(path)
    return written


def run(cluster_id: str, other_cluster_id: str) -> dict[str, dict[str, Any]]:
    payload, errors = evaluate(cluster_id, other_cluster_id)
    failures = [
        f"{case_id}: {error}" for case_id in CASE_IDS for error in errors[case_id]
    ]
    if failures:
        raise AssertionError("; ".join(failures))
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--cluster-id", default="cluster-a")
    parser.add_argument("--other-cluster-id", default="cluster-b")
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="write cases/<id>/<id>.json for ISO-002 and CMD-011 under this directory",
    )
    parser.add_argument(
        "--release-id",
        default="",
        help="release the evidence is bound to (from the release state ConfigMap)",
    )
    arguments = parser.parse_args(argv)
    payload, errors = evaluate(arguments.cluster_id, arguments.other_cluster_id)
    if arguments.run_dir is not None:
        write_evidence(
            arguments.run_dir,
            payload,
            errors,
            cluster_id=arguments.cluster_id,
            release_id=arguments.release_id,
        )
    print(json.dumps(payload, indent=2, sort_keys=True))
    failures = [
        f"{case_id}: {error}" for case_id in CASE_IDS for error in errors[case_id]
    ]
    for failure in failures:
        print(failure, file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

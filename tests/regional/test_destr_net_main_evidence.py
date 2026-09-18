"""Single-read predecessor reuse and bounded remediated retry regressions."""

from __future__ import annotations

import hashlib
import io
import json
import sys
from contextlib import redirect_stdout
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

import gpu_fault.cluster_executor as executor_module
from gpu_fault.execution.models import WorkflowStepOutcome
from gpu_fault.models import WorkflowExecutionRequest
from scripts.e2e.regional import run_destr010_fabric_manager_restart as fabric
from scripts.e2e.regional import run_destr012_managed_recovery_guard as managed
from scripts.e2e.regional.regional_live_fixture import predecessor_evidence
from tests.execution.test_cluster_executor_lease_and_report import remote_command


def predecessor_document() -> dict[str, Any]:
    return {
        "case_id": managed.PREDECESSOR_CASE_ID,
        "verdict": "PASS",
        "status": "COMPLETED",
        "release_id": "release-a",
        "cluster_id": "cluster-a",
        "workflow_request_id": "workflow-9",
        "workflow_official_steps": [
            {"operation": operation, "execution_owner": managed.MANAGED_WORKLOAD_OWNER}
            for operation in managed.GROUP_A_OWNER_OPERATIONS
        ],
    }


def binding() -> dict[str, Any]:
    return {
        "evidence_valid": True,
        "expected_release_id": "release-a",
        "expected_cluster_id": "cluster-a",
    }


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("case_id", "GF-REGIONAL-DESTR-010"),
        ("verdict", "FAIL"),
        ("status", "RUNNING"),
        ("release_id", "release-b"),
        ("cluster_id", "cluster-b"),
    ],
)
def test_cached_preflight_success_does_not_authorize_changed_evidence(
    tmp_path: Path, field: str, value: str
) -> None:
    path = tmp_path / "predecessor.json"
    document = predecessor_document()
    document[field] = value
    path.write_text(json.dumps(document), encoding="utf-8")
    result = managed.group_a_from_evidence(path, binding())
    assert result["verdict"] == "FAIL", result
    assert "not a PASS bound" in result["error"]


def test_predecessor_steps_and_sha_come_from_the_same_single_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "predecessor.json"
    original = json.dumps(predecessor_document())
    path.write_text(original, encoding="utf-8")
    reads: list[Path] = []
    read_evidence = managed.read_predecessor_evidence

    def read_then_replace(
        target: Path, *args: Any, **kwargs: Any
    ) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        value, facts = read_evidence(target, *args, **kwargs)
        reads.append(target)
        changed = {
            **predecessor_document(),
            "verdict": "FAIL",
            "workflow_official_steps": [{"operation": "RESTART_NODE"}],
        }
        target.write_text(json.dumps(changed), encoding="utf-8")
        return value, facts

    monkeypatch.setattr(managed, "read_predecessor_evidence", read_then_replace)
    result = managed.group_a_from_evidence(path, binding())
    assert reads == [path], "validated facts must not authorize a second file read"
    assert result["verdict"] == "PASS", result
    assert (
        result["workflow_official_steps"]
        == predecessor_document()["workflow_official_steps"]
    )
    assert (
        result["predecessor_evidence_sha256"]
        == hashlib.sha256(original.encode()).hexdigest()
    )
    assert managed.group_a_from_evidence(path, binding())["verdict"] == "FAIL"


@pytest.mark.parametrize("valid", [False, 1, "true", None])
def test_group_a_requires_a_literal_true_evidence_fact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, valid: Any
) -> None:
    monkeypatch.setattr(
        managed,
        "read_predecessor_evidence",
        lambda *a, **k: (predecessor_document(), {"evidence_valid": valid}),
    )
    result = managed.group_a_from_evidence(tmp_path / "evidence.json", binding())
    assert result["verdict"] == "FAIL", result


def test_selective_sequence_waiver_still_allows_only_bound_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts.e2e.regional.acceptance_scope import (
        EXECUTION_SCOPE_ENV,
        SELECTION_REFERENCE_ENV,
    )

    monkeypatch.setenv(EXECUTION_SCOPE_ENV, "selective")
    monkeypatch.setenv(SELECTION_REFERENCE_ENV, "CHG-MAIN-INTEGRATION")
    path = tmp_path / "predecessor.json"
    path.write_text(
        json.dumps({**predecessor_document(), "execution_scope": "selective"})
    )
    facts = predecessor_evidence(
        path,
        managed.PREDECESSOR_CASE_ID,
        release_id="release-a",
        cluster_id="cluster-a",
    )
    assert facts["verdict"] == "SKIPPED_BY_OPERATOR"
    assert facts["evidence_valid"] is True
    result = managed.group_a_from_evidence(path, facts)
    assert result["verdict"] == "PASS", result
    assert result["predecessor_evidence_sha256"] == facts["evidence_sha256"]
    changed = {**predecessor_document(), "workflow_request_id": "another"}
    path.write_text(json.dumps(changed))
    result = managed.group_a_from_evidence(path, facts)
    assert result["verdict"] == "FAIL"
    assert "changed since preflight" in result["error"]


def test_duplicate_owner_rows_do_not_overwrite_a_bad_owner() -> None:
    steps = predecessor_document()["workflow_official_steps"]
    steps.insert(0, {"operation": "STOP_WORKLOADS", "execution_owner": "provider"})
    assert managed.group_a_owner_errors(steps), "dict last-wins is not uniqueness proof"
    assert managed.group_a_owner_errors(cast(Any, [None])), (
        "malformed steps cannot establish the predecessor's execution owner"
    )


def test_duplicate_window_wait_cannot_extend_the_approved_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = datetime(2026, 9, 14, 12, tzinfo=timezone.utc)
    sleeps: list[float] = []
    monkeypatch.setattr(managed.time, "sleep", sleeps.append)
    with pytest.raises(managed.RegionalFixtureError, match="maintenance window"):
        managed.wait_out_duplicate_window(
            first,
            now=first + timedelta(seconds=10),
            maintenance_window_end=first + timedelta(seconds=40),
        )
    assert sleeps == []


def test_fabric_replay_builds_its_request_through_the_dispatch_layer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    command = remote_command("fabric-replay")
    request = WorkflowExecutionRequest(expected_fencing_token=command.fencing_token)
    calls: list[tuple[str, Any]] = []

    def build(current: Any) -> WorkflowExecutionRequest:
        calls.append(("dispatch", current))
        return request

    def execute(context: Any) -> WorkflowStepOutcome:
        calls.append(("adapter", context))
        return WorkflowStepOutcome.succeeded(
            operation_id="replay", details={"cached": True}
        )

    executor = SimpleNamespace(
        dispatch=SimpleNamespace(execution_request=build),
        adapters=[SimpleNamespace(owner="gpu-fault-node-agent", execute=execute)],
    )
    monkeypatch.setattr(executor_module, "executor_from_environment", lambda: executor)
    monkeypatch.setattr(sys, "argv", ["probe", command.model_dump_json()])
    output = io.StringIO()
    with redirect_stdout(output):
        exec(fabric.REPLAY_SCRIPT, {})
    assert [name for name, _ in calls] == ["dispatch", "adapter"]
    assert calls[0][1] == command
    context = calls[1][1]
    assert context.request is request
    assert context.idempotency_key == command.idempotency_key
    assert json.loads(output.getvalue())["details"]["cached"] is True

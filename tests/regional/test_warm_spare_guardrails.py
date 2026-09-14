"""The DESTR-005/006/007 guard audit: one failure, one case; one pytest, three verdicts.

The review found one ``try`` around every live probe, so an executor Pod that
would not exec failed all three cases with the same message and left the probes
after it unrun; three interpreter start-ups for five focused tests; a negative
CloudTrail read counted as proof inside the delivery lag; and the negative
cluster described twice.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import audit_warm_spare_guardrails as warm_spare

MANAGED_GUARD = {
    "status": "FAILED",
    "error": (
        "healthy warm-spare replacement cannot be delegated "
        "to managed/provider node recovery"
    ),
}
EXECUTOR_GUARDS = {
    "coordinator_guard": {
        "status": "FAILED",
        "error": (
            "healthy warm-spare replacement is required but the "
            "spare coordinator is disabled"
        ),
    },
    "startup_guard_error": (
        "regional HyperPod spare failover requires "
        "GPU_FAULT_CLUSTER_EXECUTOR_REMOTE_STATE=true"
    ),
}


def _gpu_node(name: str) -> dict[str, Any]:
    return {
        "name": name,
        "uid": f"uid-{name}",
        "ready": "True",
        "gpu_allocatable": 8,
        "unschedulable": False,
        "taints": [],
        "ownership_annotations": {
            key: None for key in warm_spare.OWNERSHIP_ANNOTATIONS
        },
    }


def _healthy_env() -> list[dict[str, Any]]:
    return [
        {
            "pod": f"exec-{index}",
            "spare_failover": "true",
            "remote_state": "true",
            "allow_replace": "false",
        }
        for index in range(2)
    ]


def _patch_healthy(monkeypatch: pytest.MonkeyPatch) -> None:
    baseline = [_gpu_node("node-a")]
    monkeypatch.setattr(warm_spare, "node_snapshot", lambda: list(baseline))
    monkeypatch.setattr(warm_spare, "run_pytest", lambda _dir, _nodeids: True)
    monkeypatch.setattr(
        warm_spare,
        "cluster_recovery",
        lambda name: {
            "cluster_name": name,
            "status": "InService",
            "node_recovery": "None",
        },
    )
    monkeypatch.setattr(warm_spare, "executor_env", _healthy_env)
    monkeypatch.setattr(warm_spare, "replace_events", lambda _start, _end: [])
    monkeypatch.setattr(
        warm_spare, "deployed_managed_owner_probe", lambda: MANAGED_GUARD
    )
    monkeypatch.setattr(
        warm_spare, "deployed_executor_guard_probes", lambda: EXECUTOR_GUARDS
    )


def _result(run_dir: Path, case_id: str) -> dict[str, Any]:
    document = json.loads(
        (run_dir / "cases" / case_id / f"{case_id}.json").read_text(encoding="utf-8")
    )
    assert isinstance(document, dict), document
    return document


def test_one_probe_failure_fails_only_the_case_that_depends_on_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_healthy(monkeypatch)

    def broken() -> dict[str, Any]:
        raise RuntimeError("executor exec refused")

    monkeypatch.setattr(warm_spare, "deployed_executor_guard_probes", broken)

    exit_code = warm_spare.run_audit(
        tmp_path, ["GF-REGIONAL-DESTR-005", "GF-REGIONAL-DESTR-007"]
    )

    five = _result(tmp_path, "GF-REGIONAL-DESTR-005")
    seven = _result(tmp_path, "GF-REGIONAL-DESTR-007")
    assert exit_code == 1
    assert five["verdict"] == "PASS", five["errors"]
    assert seven["verdict"] == "FAIL"
    assert "RuntimeError: executor exec refused" in seven["errors"]
    assert "deployed executor guard probes did not run" in seven["errors"]
    # DESTR-005's own probe ran and is recorded; DESTR-007's is not its business.
    assert five["deployed_managed_owner_probe"] == MANAGED_GUARD
    assert "deployed_executor_guard_probes" not in five
    assert "deployed_managed_owner_probe" not in seven


def test_a_shared_read_failure_is_charged_to_every_case(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_healthy(monkeypatch)

    def unavailable(_name: str) -> dict[str, Any]:
        raise RuntimeError("describe-cluster unavailable")

    monkeypatch.setattr(warm_spare, "cluster_recovery", unavailable)

    warm_spare.run_audit(tmp_path, ["GF-REGIONAL-DESTR-005", "GF-REGIONAL-DESTR-007"])

    for case_id in ("GF-REGIONAL-DESTR-005", "GF-REGIONAL-DESTR-007"):
        errors = _result(tmp_path, case_id)["errors"]
        assert "RuntimeError: describe-cluster unavailable" in errors, errors
        # The probes after the failing read still ran.
        assert "did not run" not in " ".join(errors), errors


def test_the_focused_tests_run_in_one_process_and_are_attributed_per_case(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_healthy(monkeypatch)
    calls: list[list[str]] = []
    definitions = warm_spare.case_definitions()
    failing = definitions["GF-REGIONAL-DESTR-007"][0]

    def one_run(log_dir: Path, nodeids: list[str]) -> bool:
        calls.append(list(nodeids))
        (log_dir / "pytest.log").write_text(
            f"FAILED {failing} - AssertionError: boom\n1 failed, 3 passed\n"
        )
        return False

    monkeypatch.setattr(warm_spare, "run_pytest", one_run)

    warm_spare.run_audit(tmp_path, list(warm_spare.CASE_IDS[::2]))

    assert len(calls) == 1, calls
    assert calls[0] == [
        *definitions["GF-REGIONAL-DESTR-005"],
        *definitions["GF-REGIONAL-DESTR-007"],
    ]
    five = _result(tmp_path, "GF-REGIONAL-DESTR-005")
    seven = _result(tmp_path, "GF-REGIONAL-DESTR-007")
    assert "focused pytest failed" not in five["errors"], five["errors"]
    assert "focused pytest failed" in seven["errors"], seven["errors"]
    assert five["focused_pytest_log"] == str(tmp_path / "cases" / "pytest.log")


def test_failed_nodeids_read_pytests_short_summary_only() -> None:
    # A usage error is not a test failure the log can attribute to a case.
    assert warm_spare.failed_nodeids("ERROR: usage: pytest [options]\n") == set()
    assert warm_spare.failed_nodeids(
        "FAILED tests/a.py::test_x - AssertionError\nERROR tests/b.py::test_y\n"
        "PASSED tests/c.py::test_z\n"
    ) == {"tests/a.py::test_x", "tests/b.py::test_y"}


def test_run_focused_pytest_attribution_shapes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    definitions = {"a": ["t.py::x"], "b": ["t.py::y", "t.py::z"]}

    monkeypatch.setattr(warm_spare, "run_pytest", lambda _dir, _ids: True)
    assert warm_spare.run_focused_pytest(tmp_path, definitions) == {
        "a": True,
        "b": True,
    }

    def failing_unattributed(log_dir: Path, _ids: list[str]) -> bool:
        (log_dir / "pytest.log").write_text("collection error\n")
        return False

    monkeypatch.setattr(warm_spare, "run_pytest", failing_unattributed)
    assert warm_spare.run_focused_pytest(tmp_path, definitions) == {
        "a": False,
        "b": False,
    }

    def failing_b(log_dir: Path, _ids: list[str]) -> bool:
        (log_dir / "pytest.log").write_text("FAILED t.py::z - boom\n")
        return False

    monkeypatch.setattr(warm_spare, "run_pytest", failing_b)
    assert warm_spare.run_focused_pytest(tmp_path, definitions) == {
        "a": True,
        "b": False,
    }
    assert warm_spare.run_focused_pytest(tmp_path, {}) == {}


def test_a_negative_cloudtrail_read_is_provisional_inside_the_delivery_lag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_healthy(monkeypatch)

    warm_spare.run_audit(tmp_path, ["GF-REGIONAL-DESTR-005"])

    result = _result(tmp_path, "GF-REGIONAL-DESTR-005")
    # The window just ended, so "no replace event" is not yet proof.
    assert result["replace_events"] == []
    assert result["replace_events_provisional"] is True
    assert result["verdict"] == "PASS"
    now = datetime(2026, 9, 7, 12, 0, tzinfo=timezone.utc)
    assert warm_spare.cloudtrail_provisional(now - timedelta(minutes=5), now=now), (
        "a window ended five minutes ago is still inside CloudTrail's lag"
    )
    assert not warm_spare.cloudtrail_provisional(
        now - timedelta(minutes=16), now=now
    ), "a window ended sixteen minutes ago has been fully delivered"


def test_the_negative_cluster_is_read_from_the_recorded_snapshot_not_described_twice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_healthy(monkeypatch)
    described: list[str] = []

    def describe(name: str) -> dict[str, Any]:
        described.append(name)
        return {"cluster_name": name, "status": "InService", "node_recovery": "None"}

    monkeypatch.setattr(warm_spare, "cluster_recovery", describe)
    monkeypatch.setattr(warm_spare, "MANAGED_GPU_CLUSTER", "managed")
    monkeypatch.setattr(warm_spare, "AUTOMATIC_NEGATIVE_CLUSTER", "negative")
    snapshot = {
        "payloads": {
            "cluster_name": "negative",
            "describe_cluster": {
                "ClusterName": "negative",
                "ClusterStatus": "InService",
                "NodeRecovery": "Automatic",
            },
        },
        "recorded_at": "2026-09-07T12:00:00Z",
        "payload_digest": "d" * 64,
        "recorded_by": "test",
    }
    monkeypatch.setattr(warm_spare, "record_provider_snapshot", lambda _name: snapshot)
    monkeypatch.setattr(
        warm_spare,
        "deployed_automatic_recovery_probe",
        lambda _snapshot: {
            "observed_node_recovery": "Automatic",
            "warm_spare_guard": {
                "status": "FAILED",
                "error": warm_spare.AUTOMATIC_GUARD_ERROR,
            },
            "control_without_warm_spare_strategy": {
                "status": "FAILED",
                "error": "HyperPod automatic node recovery is enabled",
            },
        },
    )

    exit_code = warm_spare.run_audit(tmp_path, ["GF-REGIONAL-DESTR-006"])

    result = _result(tmp_path, "GF-REGIONAL-DESTR-006")
    assert exit_code == 0, result["errors"]
    assert described == ["managed"], described
    assert result["automatic_negative_cluster"] == {
        "cluster_name": "negative",
        "status": "InService",
        "node_recovery": "Automatic",
    }
    assert warm_spare.automatic_cluster_recovery({}) == {
        "cluster_name": None,
        "status": None,
        "node_recovery": None,
    }


def test_the_isolated_node_reader_carries_the_probes_past_the_isolation_fence() -> None:
    """The deployed step adapter refuses any HyperPod mutation until it has read
    the target Node and seen it isolated for the incident; the probes' read-only
    Kubernetes stand-in must satisfy exactly that read, and nothing more, or the
    guards the cases exist to prove are never reached."""
    from datetime import datetime, timezone
    from types import SimpleNamespace

    from gpu_fault.adapters.hyperpod.lifecycle import HyperPodLifecycleStepAdapter
    from gpu_fault.execution import WorkflowStepContext
    from gpu_fault.models import (
        FaultIncident,
        IncidentState,
        WorkflowExecutionRequest,
        WorkflowOperation,
        WorkflowRequest,
        WorkflowStatus,
        WorkflowStepSpec,
    )

    namespace: dict[str, Any] = {}
    exec(warm_spare.ISOLATED_NODE_READER, namespace)  # noqa: S102 - the probe text is the artifact
    kubernetes, reader = namespace["isolated_kubernetes_adapter"]("incident-probe", 1)
    now = datetime.now(timezone.utc)
    step = WorkflowStepSpec(
        operation=WorkflowOperation.REPLACE_NODE,
        execution_owner="gpu-fault-hyperpod-adapter",
        node_ids=["audit-node"],
        parameters={"replacement_strategy": "HEALTHY_WARM_SPARE_ONLY"},
    )
    incident = FaultIncident(
        incident_id="incident-probe",
        event_id="event-probe",
        event_type="REGIONAL_ACCEPTANCE",
        cluster_id="cluster",
        node_ids=["audit-node"],
        policy_version="probe/v1",
        policy_source="ACCEPTANCE",
        state=IncidentState.ACTION_PENDING,
        workflow_request_id="workflow-probe",
        fencing_token=1,
        created_at=now,
        updated_at=now,
    )
    workflow = WorkflowRequest(
        request_id="workflow-probe",
        incident_id="incident-probe",
        status=WorkflowStatus.RUNNING,
        official_action="REPLACE_NODE",
        fencing_token=1,
        official_steps=[step],
        completed_operations=[WorkflowOperation.MARK_UNSCHEDULABLE],
        created_at=now,
        updated_at=now,
    )
    context = WorkflowStepContext(
        workflow=workflow,
        incident=incident,
        step=step,
        step_index=0,
        request=WorkflowExecutionRequest(
            expected_fencing_token=1,
            confirm_cluster_name="cluster",
            isolation_verified_nodes=["audit-node"],
        ),
        idempotency_key="workflow-probe/0/REPLACE_NODE",
    )

    def preflight(*_args: Any, **_kwargs: Any) -> SimpleNamespace:
        return SimpleNamespace(
            safe_to_submit=True, gate_failures=[], node_recovery="None"
        )

    without = HyperPodLifecycleStepAdapter(object())
    without.dispatcher.preflight = preflight
    refused = without.execute(context)
    assert "observe node isolation" in str(refused.error), refused

    carried = HyperPodLifecycleStepAdapter(object(), kubernetes_adapter=kubernetes)
    carried.dispatcher.preflight = preflight
    outcome = carried.execute(context)
    assert outcome.error == (
        "healthy warm-spare replacement is required but the "
        "spare coordinator is disabled"
    ), outcome
    assert reader.reads == ["audit-node"]
    assert not hasattr(kubernetes.core, "patch_node"), (
        "the stand-in must stay read-only"
    )

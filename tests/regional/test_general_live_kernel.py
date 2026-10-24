"""GF-GENERAL-LIVE-016 severe kernel log: category-module verdict tests.

Driven over fakes (no systemd, journalctl or control plane): a ``FakeProbe``
stands in for the kernel-log replay host probe and a scripted ``FakeRegional``
answers the node-log activity reads, so each variant's finding -> incident ->
cordon/quarantine -> notification chain is graded on realistic documents. The
negative variants prove the trusted-source and current-boot gates.
"""

from __future__ import annotations

from pathlib import Path

from scripts.e2e.regional import general_live_kernel as mod
from scripts.e2e.regional.general_live_registry import coverage_rows
from tests.regional._general_live_fakes import (
    FakeProbe,
    FakeRegional,
    make_context,
    make_settings,
    store_script,
)

NODE = "ip-10-0-43-5.us-west-2.compute.internal"


def _spec(variant: str) -> object:
    return next(item for item in mod.VARIANTS if item.variant == variant)


def _host_script() -> dict[str, object]:
    return {
        "install-records": {"records": 1},
        "open-window": {"after": {"ActiveState": "active"}, "proof_scope": "live-replay"},
        "arm": {"armed": True},
        "status": {"injected": {"emission_counts": {"0": 2}}},
        "close-window": {"clean": True},
    }


def _log_incident(
    *,
    incident_id: str = "inc-log-1",
    event_id: str = "log-gpu-eks-cluster-node-abc",
    policy_source: str = "SITE_KERNEL_LOG",
    action: str = "QUARANTINE",
    state: str = "QUARANTINED",
    workflow: str = "wf-log-1",
) -> dict[str, object]:
    return {
        "incident_id": incident_id,
        "event_id": event_id,
        "event_type": "NODE_HEALTH",
        "policy_source": policy_source,
        "effective_action": action,
        "state": state,
        "created_at": "2026-10-24T00:03:00+00:00",
        "workflow_request_id": workflow,
        "node_ids": [NODE],
    }


def _log_workflow(
    operations: list[str],
    *,
    incident_id: str = "inc-log-1",
    request_id: str = "wf-log-1",
) -> dict[str, object]:
    return {
        "request_id": request_id,
        "incident_id": incident_id,
        "status": "SUCCEEDED",
        "created_at": "2026-10-24T00:03:00+00:00",
        "official_steps": [{"operation": op, "parameters": {}} for op in operations],
        "step_executions": [],
    }


def _notification(incident_id: str = "inc-log-1") -> dict[str, object]:
    return {
        "notification_id": "ntf-1",
        "incident_id": incident_id,
        "deduplication_key": f"kernel-log/{incident_id}",
        "subject": "severe kernel log",
        "status": "SENT",
    }


def _activity(
    *,
    incidents: list[dict[str, object]],
    workflows: list[dict[str, object]],
    notifications: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    return {
        "incidents": incidents,
        "workflows": workflows,
        "markers": [],
        "notifications": notifications or [],
        "batches": [],
        "metric_latest": [],
        "collector_statuses": [],
    }


def _ctx(variant: str, *, regional: FakeRegional, tmp_path: Path):
    spec = _spec(variant)
    fake = FakeProbe(_host_script())
    ctx = make_context(
        spec,
        tmp_path=tmp_path,
        regional=regional,
        probes={f"kernel-replay-journal@{NODE}": fake},
        settings=make_settings(spec, node=NODE),
        window_minutes=60,
    )
    return ctx, fake


def _isolated_node(isolated: bool) -> dict[str, object]:
    return {
        "name": NODE,
        "ready": "True",
        "unschedulable": isolated,
        "taints": [{"key": "gpu-fault.io/quarantine"}] if isolated else [],
        "ownership_annotations": {"gpu-fault.io/incident-id": "inc-log-1"} if isolated else {},
    }


def test_rdma_fatal_quarantines_and_notifies(tmp_path: Path) -> None:
    activity = _activity(
        incidents=[_log_incident()],
        workflows=[_log_workflow(["FREEZE_EVIDENCE", "MARK_UNSCHEDULABLE", "QUARANTINE"])],
        notifications=[_notification()],
    )
    regional = FakeRegional(
        cpu_python=store_script({"activity": activity}),
        node_snapshot=_isolated_node(True),
    )
    ctx, _ = _ctx("rdma_fatal", regional=regional, tmp_path=tmp_path)
    result = mod.rdma_fatal(ctx)
    assert result["verdict"] == "PASS", result["errors"]
    assert result["proof_scope"] == "live-replay", result


def test_quarantine_fails_when_workflow_plans_hardware(tmp_path: Path) -> None:
    activity = _activity(
        incidents=[_log_incident()],
        workflows=[
            _log_workflow(["FREEZE_EVIDENCE", "MARK_UNSCHEDULABLE", "QUARANTINE", "RESET_GPU"])
        ],
        notifications=[_notification()],
    )
    regional = FakeRegional(
        cpu_python=store_script({"activity": activity}),
        node_snapshot=_isolated_node(True),
    )
    ctx, _ = _ctx("uncorrectable_mce", regional=regional, tmp_path=tmp_path)
    result = mod.uncorrectable_mce(ctx)
    assert result["verdict"] == "FAIL", result
    assert any("RESET_GPU" in error for error in result["errors"]), result["errors"]


def test_inconclusive_when_nothing_was_ingested(tmp_path: Path) -> None:
    from scripts.e2e.regional.general_live_fixture import VariantInconclusive

    regional = FakeRegional(
        cpu_python=store_script({"activity": _activity(incidents=[], workflows=[])}),
        node_snapshot=_isolated_node(False),
    )
    spec = _spec("rdma_fatal")
    fake = FakeProbe({**_host_script(), "status": {"injected": {"emission_counts": {}}}})
    ctx = make_context(
        spec,
        tmp_path=tmp_path,
        regional=regional,
        probes={f"kernel-replay-journal@{NODE}": fake},
        settings=make_settings(spec, node=NODE),
        window_minutes=60,
    )
    try:
        mod.rdma_fatal(ctx)
    except VariantInconclusive as exc:
        assert "no reviewed records" in str(exc), str(exc)
    else:
        raise AssertionError("an empty injection oracle must be INCONCLUSIVE")


def test_corrected_mce_is_diagnostic_not_isolation(tmp_path: Path) -> None:
    activity = _activity(
        incidents=[
            _log_incident(
                policy_source="SITE_APPLICATION_LOG", action="RUN_DIAGNOSTICS", state="ACTION_PENDING"
            )
        ],
        workflows=[_log_workflow(["FREEZE_EVIDENCE", "VALIDATE_HOST"])],
    )
    regional = FakeRegional(
        cpu_python=store_script({"activity": activity}),
        node_snapshot=_isolated_node(False),
    )
    ctx, _ = _ctx("corrected_mce", regional=regional, tmp_path=tmp_path)
    result = mod.corrected_mce(ctx)
    assert result["verdict"] == "PASS", result["errors"]


def test_user_text_produces_no_finding(tmp_path: Path) -> None:
    regional = FakeRegional(
        cpu_python=store_script({"activity": _activity(incidents=[], workflows=[])}),
        node_snapshot=_isolated_node(False),
    )
    ctx, _ = _ctx("user_text", regional=regional, tmp_path=tmp_path)
    result = mod.user_text(ctx)
    assert result["verdict"] == "PASS", result["errors"]


def test_user_text_fails_if_it_isolates(tmp_path: Path) -> None:
    activity = _activity(
        incidents=[_log_incident()],
        workflows=[_log_workflow(["FREEZE_EVIDENCE", "QUARANTINE"])],
    )
    regional = FakeRegional(
        cpu_python=store_script({"activity": activity}),
        node_snapshot=_isolated_node(True),
    )
    ctx, _ = _ctx("user_text", regional=regional, tmp_path=tmp_path)
    result = mod.user_text(ctx)
    assert result["verdict"] == "FAIL", result


def test_unknown_attempt_isolates_without_stop(tmp_path: Path) -> None:
    activity = _activity(
        incidents=[_log_incident()],
        workflows=[_log_workflow(["FREEZE_EVIDENCE", "MARK_UNSCHEDULABLE", "QUARANTINE"])],
    )
    regional = FakeRegional(
        cpu_python=store_script({"activity": activity}),
        node_snapshot=_isolated_node(True),
    )
    ctx, _ = _ctx("unknown_attempt", regional=regional, tmp_path=tmp_path)
    result = mod.unknown_attempt(ctx)
    assert result["verdict"] == "PASS", result["errors"]


def test_unknown_attempt_fails_if_it_stops_a_workload(tmp_path: Path) -> None:
    activity = _activity(
        incidents=[_log_incident()],
        workflows=[_log_workflow(["MARK_UNSCHEDULABLE", "STOP_WORKLOADS", "QUARANTINE"])],
    )
    regional = FakeRegional(
        cpu_python=store_script({"activity": activity}),
        node_snapshot=_isolated_node(True),
    )
    ctx, _ = _ctx("unknown_attempt", regional=regional, tmp_path=tmp_path)
    result = mod.unknown_attempt(ctx)
    assert result["verdict"] == "FAIL", result
    assert any("stopped a workload" in error for error in result["errors"]), result["errors"]


def test_cursor_pagination_requires_every_record(tmp_path: Path) -> None:
    incidents = [
        _log_incident(incident_id=f"inc-log-{i}", event_id=f"log-c-{i}", workflow=f"wf-{i}")
        for i in range(3)
    ]
    workflows = [
        _log_workflow(
            ["MARK_UNSCHEDULABLE", "QUARANTINE"], incident_id=f"inc-log-{i}", request_id=f"wf-{i}"
        )
        for i in range(3)
    ]
    regional = FakeRegional(
        cpu_python=store_script({"activity": _activity(incidents=incidents, workflows=workflows)}),
        node_snapshot=_isolated_node(True),
    )
    ctx, _ = _ctx("cursor_pagination", regional=regional, tmp_path=tmp_path)
    result = mod.cursor_pagination(ctx)
    assert result["verdict"] == "PASS", result["errors"]
    assert len(result["incidents"]) == 3, result


def test_variants_match_the_coverage_row() -> None:
    expected = set(coverage_rows()["GF-GENERAL-LIVE-016"])
    assert {item.variant for item in mod.VARIANTS} == expected, "016 registers every coverage variant"
    assert all(item.proof_mode == "live-replay" for item in mod.VARIANTS), "016 is live-replay"

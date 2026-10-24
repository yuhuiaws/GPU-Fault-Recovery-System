"""GF-GENERAL-LIVE-017 application error logs: category-module verdict tests."""

from __future__ import annotations

from pathlib import Path

from scripts.e2e.regional import general_live_applogs as mod
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
        "open-window": {"after": {"ActiveState": "active"}},
        "arm": {"armed": True},
        "status": {"injected": {"emission_counts": {"0": 2}}},
        "close-window": {"clean": True},
    }


def _incident(
    action: str, *, incident_id: str = "inc-log-1", workflow: str = "wf-1"
) -> dict[str, object]:
    return {
        "incident_id": incident_id,
        "event_id": "log-gpu-eks-cluster-node-abc",
        "event_type": "NODE_HEALTH",
        "policy_source": "SITE_APPLICATION_LOG",
        "effective_action": action,
        "state": "ACTION_PENDING",
        "created_at": "2026-10-24T00:03:00+00:00",
        "workflow_request_id": workflow,
        "node_ids": [NODE],
    }


def _workflow(
    ops: list[str], *, incident_id: str = "inc-log-1", request_id: str = "wf-1"
) -> dict[str, object]:
    return {
        "request_id": request_id,
        "incident_id": incident_id,
        "status": "SUCCEEDED",
        "created_at": "2026-10-24T00:03:00+00:00",
        "official_steps": [{"operation": op, "parameters": {}} for op in ops],
        "step_executions": [],
    }


def _activity(
    incidents: list[dict[str, object]], workflows: list[dict[str, object]]
) -> dict[str, object]:
    return {
        "incidents": incidents,
        "workflows": workflows,
        "markers": [],
        "notifications": [],
        "batches": [],
        "metric_latest": [],
        "collector_statuses": [],
    }


def _node(isolated: bool) -> dict[str, object]:
    return {
        "name": NODE,
        "ready": "True",
        "unschedulable": isolated,
        "taints": [],
        "ownership_annotations": {},
    }


def _ctx(variant: str, *, activity: dict[str, object], isolated: bool, tmp_path: Path):
    spec = _spec(variant)
    fake = FakeProbe(_host_script())
    regional = FakeRegional(
        cpu_python=store_script({"activity": activity}), node_snapshot=_node(isolated)
    )
    return make_context(
        spec,
        tmp_path=tmp_path,
        regional=regional,
        probes={f"kernel-replay-journal@{NODE}": fake},
        settings=make_settings(spec, node=NODE),
        window_minutes=60,
    )


def test_cgroup_oom_is_a_diagnostic(tmp_path: Path) -> None:
    activity = _activity(
        [_incident("RUN_DIAGNOSTICS")],
        [_workflow(["FREEZE_EVIDENCE", "VALIDATE_HOST"])],
    )
    ctx = _ctx("cgroup_oom", activity=activity, isolated=False, tmp_path=tmp_path)
    result = mod.cgroup_oom(ctx)
    assert result["verdict"] == "PASS", result["errors"]


def test_cgroup_oom_fails_if_it_stops_a_workload(tmp_path: Path) -> None:
    activity = _activity(
        [_incident("RUN_DIAGNOSTICS")],
        [_workflow(["FREEZE_EVIDENCE", "STOP_WORKLOADS"])],
    )
    ctx = _ctx("cgroup_oom", activity=activity, isolated=False, tmp_path=tmp_path)
    result = mod.cgroup_oom(ctx)
    assert result["verdict"] == "FAIL", result
    assert any("stopped a workload" in error for error in result["errors"]), result[
        "errors"
    ]


def test_nccl_error_is_evidence_only(tmp_path: Path) -> None:
    activity = _activity([_incident("NO_ACTION")], [_workflow(["FREEZE_EVIDENCE"])])
    ctx = _ctx("nccl_error", activity=activity, isolated=False, tmp_path=tmp_path)
    result = mod.nccl_error(ctx)
    assert result["verdict"] == "PASS", result["errors"]


def test_nccl_error_fails_on_a_diagnostic_workflow(tmp_path: Path) -> None:
    activity = _activity(
        [_incident("NO_ACTION")],
        [_workflow(["FREEZE_EVIDENCE", "COLLECT_DIAGNOSTIC_BUNDLE"])],
    )
    ctx = _ctx("nccl_error", activity=activity, isolated=False, tmp_path=tmp_path)
    result = mod.nccl_error(ctx)
    assert result["verdict"] == "FAIL", result
    assert any("diagnostic workflow" in error for error in result["errors"]), result[
        "errors"
    ]


def test_distributed_error_fails_if_it_isolates(tmp_path: Path) -> None:
    activity = _activity([_incident("QUARANTINE")], [_workflow(["QUARANTINE"])])
    ctx = _ctx("distributed_error", activity=activity, isolated=True, tmp_path=tmp_path)
    result = mod.distributed_error(ctx)
    assert result["verdict"] == "FAIL", result


def test_variants_match_the_coverage_row() -> None:
    expected = set(coverage_rows()["GF-GENERAL-LIVE-017"])
    assert {item.variant for item in mod.VARIANTS} == expected, (
        "017 registers every coverage variant"
    )
    implemented = {item.variant for item in mod.VARIANTS if item.execute is not None}
    assert implemented == {"cgroup_oom", "nccl_error", "distributed_error"}, implemented

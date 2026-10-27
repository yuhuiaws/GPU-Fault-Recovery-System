"""GF-GENERAL-LIVE-024 ``wrong_scope_source``: the repair chain settles under the
error GPU's terminal hold.

Live 2026-10-27 14:33-14:41Z (HyperPod, R13g, harness int/r13c-live28, the
one-read plan) the variant FAILED "repair_settled: the row_remap_pending
incident ... is QUARANTINED, not RECOVERED by its own chain": every workflow
had ended, no rung was planned (the hardware stage was empty), the node did
not reboot (boot id cc70db9b unchanged) and the repair incident was still the
node's isolation owner when the validated restore released it. That is the
product's design, not a fault: the error GPU's CRITICAL finding plans a
QUARANTINE that holds the node for good (``TERMINAL_QUARANTINE_NODES``), and
the repair GPU's RESET_GPU plan on the same node inherits the hold
(``workflow_quarantine.inherit_terminal_quarantine``: a QUARANTINE inserted,
RESTORE_SCHEDULING superseded), so with every executed step SUCCEEDED its
incident settles QUARANTINED. 019's reset variants have no sibling hold and
settle RECOVERED (live PASS on both clusters).

Here the product's own records for the minting batch -- ingested through the
real ``GpuMetricsService`` and node-health path -- are pinned, and then fed
to the variant: the designed QUARANTINED is a PASS, while a failed chain, a
QUARANTINED without the hold, a hold with no error-GPU source, or a chain that
never reset is the stage's error; the node is left clean either way.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from gpu_fault.app import ApplicationContext
from gpu_fault.app.ingest.gpu_findings import gpu_node_health_findings
from gpu_fault.app.ingest.node_health import NodeHealthIngestionService
from gpu_fault.execution.terminal_state import terminal_incident_state
from gpu_fault.gpu_metrics import GpuMetricBatch, GpuMetricSample, GpuMetricSource
from gpu_fault.models import WorkflowStatus, WorkloadState
from gpu_fault.workflow_quarantine import (
    TERMINAL_QUARANTINE_NODES,
    terminal_quarantine_nodes,
)
from scripts.e2e.regional import general_live_gpu_composite as comp
from tests.regional._general_live_fakes import FakeRegional, make_context, make_settings
from tests.regional._general_live_gpu_fakes import (
    audit_records,
    dcgm_probe,
    gpu_cpu_python,
)

G0 = "GPU-00000000-0000-4000-8000-000000000000"
G1 = "GPU-11111111-1111-4111-8111-111111111111"
NODE = FakeRegional().node_snapshot("")["name"]
START = datetime(2026, 10, 27, 14, 36, 5, tzinfo=timezone.utc)
SPEC = {spec.variant: spec for spec in comp.VARIANTS}["wrong_scope_source"]


def _sample(name: str, value: float, gpu: str, index: str) -> GpuMetricSample:
    return GpuMetricSample(
        metric_name=f"DCGM_{name.upper()}",
        canonical_name=name,
        value=value,
        gpu_index=index,
        gpu_uuid=gpu,
        pci_bdf=f"00000000:0{int(index) + 1}:00.0",
        labels={"modelName": "NVIDIA H200"},
    )


def _batch(tick: int, dbe: float, pending: float) -> GpuMetricBatch:
    return GpuMetricBatch(
        batch_id=f"b{tick}",
        cluster_id="gpu-eks-cluster",
        node_id=NODE,
        observed_at=START + timedelta(seconds=15 * tick),
        source=GpuMetricSource.DCGM_EXPORTER,
        samples=[
            _sample("gpu_temperature_c", 40.0, G0, "0"),
            _sample("gpu_temperature_c", 41.0, G1, "1"),
            _sample("ecc_dbe_volatile_total", dbe, G0, "0"),
            _sample("row_remap_pending", pending, G1, "1"),
        ],
        runtime_profile_version="simulated-v1",
    )


def _product_records(*, dbe: float = 1.0) -> dict[str, tuple[Any, Any]]:
    """The product's incidents and workflows for the minting batch, by metric:
    the batch through ``GpuMetricsService.ingest``, its new findings through
    ``gpu_node_health_findings`` and the node-health ingest, as the telemetry
    ingest does (the idle test node: workload IDLE)."""

    ctx = ApplicationContext()
    ctx.gpu_metrics.ingest(_batch(0, 0.0, 0.0))
    minted = ctx.gpu_metrics.ingest(_batch(1, dbe, 1.0))
    findings = [
        item.model_copy(update={"workload_state": WorkloadState.IDLE})
        for item in gpu_node_health_findings(ctx, minted.new_findings)
    ]
    NodeHealthIngestionService(ctx).ingest("b1", findings)
    records = {}
    for finding in findings:
        incident = ctx.store.get_incident_by_event(finding.event_id)
        records[str(finding.metric_name)] = (
            incident,
            ctx.store.get_workflow(incident.workflow_request_id),
        )
    return records


def _live_indexes(workflow: Any) -> list[int]:
    return [
        index
        for index in range(len(workflow.official_steps))
        if index not in workflow.superseded_step_indexes
    ]


def _settle(
    incident: Any, workflow: Any, *, failed: str | None = None
) -> tuple[dict[str, Any], dict[str, Any]]:
    """The records as the activity probe reads them once the chain has run:
    every live step SUCCEEDED (or the chain stopped FAILED at ``failed``), the
    incident in the state ``terminal_incident_state`` gives the record."""

    executions = []
    for index in _live_indexes(workflow):
        operation = workflow.official_steps[index].operation.value
        status = "FAILED" if operation == failed else "SUCCEEDED"
        executions.append(
            {
                "step_index": index,
                "operation": operation,
                "status": status,
                "phase": "official",
                "updated_at": (START + timedelta(seconds=10 * index)).isoformat(),
            }
        )
        if status == "FAILED":
            break
    status = WorkflowStatus.FAILED if failed else WorkflowStatus.SUCCEEDED
    completed = [
        item["step_index"] for item in executions if item["status"] == "SUCCEEDED"
    ]
    # What the executor keeps of a finished step: its index and its operation.
    done = workflow.model_copy(
        update={
            "completed_step_indexes": completed,
            "completed_operations": [
                workflow.official_steps[index].operation for index in completed
            ],
        }
    )
    state = terminal_incident_state(done, status)
    record = workflow.model_dump(mode="json")
    record.update(status=status.value, step_executions=executions)
    return incident.model_dump(mode="json") | {"state": state.value}, record


def _activity(incidents: list, workflows: list) -> dict[str, Any]:
    return {
        "incidents": incidents,
        "workflows": workflows,
        "batches": [{"observed_at": "t"}],
        "markers": [],
        "commands": [],
        "notifications": [],
        "metric_latest": [],
        "collector_statuses": [],
        "agent": None,
    }


def _run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, incidents: list, workflows: list
) -> dict[str, Any]:
    restored: list[list[str]] = []
    monkeypatch.setattr(
        comp,
        "_validated_restore",
        lambda ctx, items: restored.append([str(i["incident_id"]) for i in items])
        or [],
    )
    regional = FakeRegional(
        cpu_python=gpu_cpu_python(
            activity=_activity(incidents, workflows), gpu_state={"active_findings": []}
        )
    )
    settings = make_settings(SPEC, node=NODE)
    ctx = make_context(
        SPEC, tmp_path=tmp_path, regional=regional, probes={}, settings=settings
    )
    probe = dcgm_probe(audit=audit_records({f"{G0}/ecc_dbe_volatile_total": 1.0}))
    ctx.probes[f"dcgm@{ctx.settings.node}"] = probe
    out = comp.wrong_scope_source(ctx)
    out["_restored"] = restored
    out["_probe"] = probe
    out["_ctx"] = ctx
    return out


def _live_shape(**settle: Any) -> tuple[list, list]:
    records = _product_records()
    error = _settle(*records["ecc_dbe_volatile_total"])
    repair = _settle(*records["row_remap_pending"], **settle)
    return [error[0], repair[0]], [error[1], repair[1]]


# --------------------------------------------------------------------------- #
# The product's records for the minting batch
# --------------------------------------------------------------------------- #
def test_the_repair_chain_inherits_the_error_gpus_terminal_hold() -> None:
    """One batch, two GPUs: the error GPU's CRITICAL DBE plans a QUARANTINE
    that holds the node for good; the repair GPU's RESET_GPU plan inherits it,
    so its incident settles QUARANTINED with every executed step SUCCEEDED."""

    records = _product_records()
    assert set(records) == {"ecc_dbe_volatile_total", "row_remap_pending"}
    error_incident, error = records["ecc_dbe_volatile_total"]
    repair_incident, repair = records["row_remap_pending"]
    assert error_incident.incident_id != repair_incident.incident_id
    assert [step.operation.value for step in error.official_steps] == [
        "FREEZE_EVIDENCE",
        "MARK_UNSCHEDULABLE",
        "QUARANTINE",
    ]
    assert terminal_quarantine_nodes(error) == frozenset({NODE})
    planned = [step.operation.value for step in repair.official_steps]
    live = [planned[index] for index in _live_indexes(repair)]
    assert "RESET_GPU" in live and "VALIDATE_GPU" in live, live
    assert "QUARANTINE" in live, "the inherited hold is an inserted QUARANTINE"
    assert "RESTORE_SCHEDULING" in planned and "RESTORE_SCHEDULING" not in live, (
        "the chain's readmission is superseded under the hold"
    )
    assert terminal_quarantine_nodes(repair) == frozenset({NODE})
    incident, record = _settle(repair_incident, repair)
    assert record["status"] == "SUCCEEDED"
    assert incident["state"] == "QUARANTINED", (
        "a sibling's recovery never undoes the node's persistent hold"
    )
    hold = [
        step["parameters"].get(TERMINAL_QUARANTINE_NODES)
        for step in record["official_steps"]
        if step["operation"] == "QUARANTINE"
    ]
    assert hold == [[NODE]], hold


def test_the_pending_flag_alone_settles_recovered_by_its_own_chain() -> None:
    """019's shape: no sibling hold, the chain readmits the node itself."""

    records = _product_records(dbe=0.0)
    assert set(records) == {"row_remap_pending"}
    incident, workflow = records["row_remap_pending"]
    assert terminal_quarantine_nodes(workflow) == frozenset()
    live = [workflow.official_steps[i].operation.value for i in _live_indexes(workflow)]
    assert live[-1] == "RESTORE_SCHEDULING", live
    assert _settle(incident, workflow)[0]["state"] == "RECOVERED"


# --------------------------------------------------------------------------- #
# The variant over those records
# --------------------------------------------------------------------------- #
def test_wrong_scope_source_passes_on_the_live_record_shape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The 2026-10-27 record: the repair chain SUCCEEDED, its incident
    QUARANTINED under the error GPU's hold -- PASS; the validated restore
    releases both incidents and the record is kept."""

    incidents, workflows = _live_shape()
    out = _run(tmp_path, monkeypatch, incidents, workflows)
    assert out["verdict"] == "PASS", out["errors"]
    assert out["stages"]["repair_settled"] == []
    repair_id = incidents[1]["incident_id"]
    assert out["repair_incidents"] == [repair_id]
    assert out["repair_incidents_restored"] == [repair_id]
    assert out["_restored"] == [[incidents[0]["incident_id"], repair_id]], (
        "both incidents are released: the node ends clean"
    )
    record = json.loads((out["_ctx"].case_dir / "repair-record.json").read_text())
    assert {item["incident_id"] for item in record["incidents"]} == {
        incidents[0]["incident_id"],
        repair_id,
    }
    assert any("terminal hold" in note for note in out["notes"]), out["notes"]
    calls = [call[0] for call in out["_probe"].calls]
    assert "advance" not in calls and calls.count("close-run") == 1, calls


def test_a_repair_chain_that_failed_is_the_stage_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A post-reset VALIDATE_GPU that failed (the held-plan shape of
    2026-10-26) leaves the incident QUARANTINED too -- the chain's own failure,
    named; the node is still released."""

    incidents, workflows = _live_shape(failed="VALIDATE_GPU")
    out = _run(tmp_path, monkeypatch, incidents, workflows)
    assert out["verdict"] == "FAIL"
    assert out["stages"]["repair_settled"] == [
        f"the row_remap_pending chain on {G1} ended FAILED (FAILED: VALIDATE_GPU)"
    ]
    assert out["repair_incidents_restored"] == [incidents[1]["incident_id"]]


def test_quarantined_without_the_hold_is_the_stage_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    incidents, workflows = _live_shape()
    for step in workflows[1]["official_steps"]:
        step["parameters"].pop(TERMINAL_QUARANTINE_NODES, None)
    out = _run(tmp_path, monkeypatch, incidents, workflows)
    assert out["verdict"] == "FAIL"
    (error,) = out["stages"]["repair_settled"]
    assert "QUARANTINED, not RECOVERED by its own chain nor held for the error GPU" in (
        error
    )
    assert f"its chain carries no terminal hold on {NODE}" in error


def test_a_hold_with_no_error_gpu_source_is_the_stage_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    incidents, workflows = _live_shape()
    for step in workflows[0]["official_steps"]:
        step["parameters"].pop(TERMINAL_QUARANTINE_NODES, None)
    out = _run(tmp_path, monkeypatch, incidents, workflows)
    assert out["verdict"] == "FAIL"
    (error,) = out["stages"]["repair_settled"]
    assert f"no error-GPU incident holds {NODE} under a terminal quarantine" in error


def test_a_chain_that_never_reset_is_the_stage_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    incidents, workflows = _live_shape()
    workflows[1]["step_executions"] = [
        item
        for item in workflows[1]["step_executions"]
        if item["operation"] != "RESET_GPU"
    ]
    out = _run(tmp_path, monkeypatch, incidents, workflows)
    assert out["verdict"] == "FAIL"
    assert out["stages"]["repair_settled"] == [
        f"the row_remap_pending chain on {G1} never executed RESET_GPU"
    ]


def test_a_repair_chain_still_running_is_not_restored_over(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    incidents, workflows = _live_shape()
    workflows[1] = dict(workflows[1], status="RUNNING")
    incidents[1] = dict(incidents[1], state="ACTION_PENDING")
    out = _run(tmp_path, monkeypatch, incidents, workflows)
    assert out["verdict"] == "FAIL"
    assert any(
        "did not settle through its own chain" in item
        for item in out["stages"]["repair_settled"]
    ), out["stages"]
    assert out["repair_incidents_restored"] == []
    assert out["_restored"] == [[incidents[0]["incident_id"]]]


def test_the_hold_helpers_read_the_product_records() -> None:
    incidents, workflows = _live_shape()
    activity = _activity(incidents, workflows)
    assert comp.terminal_hold_nodes(workflows[1]) == {NODE}
    assert (
        comp.hold_errors(activity, workflows[1], node=NODE, hold_sources=[incidents[0]])
        == []
    )
    assert comp.hold_errors(activity, workflows[1], node=NODE, hold_sources=[]) == [
        f"no error-GPU incident holds {NODE} under a terminal quarantine"
    ]
    assert [
        item["incident_id"] for item in comp.isolated_repair(activity, [incidents[1]])
    ] == [incidents[1]["incident_id"]]
    record = comp.record_of(activity, [incidents[1]])
    assert [item["request_id"] for item in record["workflows"]] == [
        workflows[1]["request_id"]
    ]

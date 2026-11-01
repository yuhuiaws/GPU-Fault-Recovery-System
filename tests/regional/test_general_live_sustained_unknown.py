"""GF-GENERAL-LIVE-025 ``sustained_unknown`` judges the chain it waited for.

The variant paused the exporter, waited for the field-completeness incident and
then threw that read away (``_ = detection``): its stages were the observation
UNKNOWN, no hardware step in whatever workflows there were, and a fresh success
after the resume -- all true of a product that minted nothing at all. Its
expected text is "a bounded snapshot + observation validation": the incident,
TRIGGER_HEALTH_SNAPSHOT then VALIDATE_HOST, nothing hardware, and a validation
that ran and has not passed while the only GPU_METRICS source is paused.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from gpu_fault.app.ingest.gpu_findings import gpu_node_health_findings
from gpu_fault.gpu_collection_errors import collection_error_finding
from gpu_fault.gpu_metric_models import GpuFindingState
from gpu_fault.gpu_metrics import GpuMetricBatch, GpuMetricSource, GpuMetricsThresholds
from scripts.e2e.regional import general_live_collectors as col
from tests.regional._general_live_fakes import (
    FakeProbe,
    FakeRegional,
    incident,
    make_context,
    make_settings,
    workflow,
)

NODE = "ip-10-0-43-5.us-west-2.compute.internal"
SPEC = next(item for item in col.VARIANTS if item.variant == "sustained_unknown")
BOUNDED = ["FREEZE_EVIDENCE", "TRIGGER_HEALTH_SNAPSHOT", "VALIDATE_HOST"]
#: What VALIDATE_HOST records while the exporter is paused
#: (``health_evidence_validation.pending``).
WAITING = {
    "step_index": 2,
    "operation": "VALIDATE_HOST",
    "status": "WAITING",
    "details": {
        "host_diagnostic_failure": True,
        "health_evidence_pending": "collector state is UNKNOWN",
        "missing_observations": {f"{NODE}/GPU_METRICS": "COLLECTION_ERROR"},
    },
    "started_at": "2026-10-24T00:04:00+00:00",
    "updated_at": "2026-10-24T00:04:00+00:00",
}


def completeness_event() -> str:
    """The event id the product mints for a failed scrape's completeness
    finding (``collection_error_finding`` -> ``gpu_node_health_findings``):
    the streak's booked finding is NO_ACTION (evidence only, no incident); an
    error batch past the sustain window asks for RUN_DIAGNOSTICS under the
    streak's first finding id, and that one opens the incident."""

    batch = GpuMetricBatch(
        batch_id=f"dcgm-{NODE}-1761363470000000",
        cluster_id="gpu-eks-cluster",
        node_id=NODE,
        observed_at=datetime(2026, 10, 25, 3, 0, tzinfo=timezone.utc),
        source=GpuMetricSource.DCGM_EXPORTER,
        samples=[],
        collection_errors=["DCGM exporter scrape failed: timed out"],
    )
    thresholds = GpuMetricsThresholds()
    booked = collection_error_finding(batch, None, thresholds)
    assert booked is not None, "a scrape with collection errors books a finding"
    later = batch.model_copy(
        update={
            "batch_id": f"{batch.batch_id}-sustained",
            "observed_at": batch.observed_at
            + timedelta(seconds=thresholds.collection_error_sustain_seconds),
        }
    )
    streak = GpuFindingState(observed_at=batch.observed_at, finding=booked[1])
    sustained = collection_error_finding(later, streak, thresholds)
    assert sustained is not None and sustained[2], "the streak outlasts its window"
    return gpu_node_health_findings(None, [sustained[1]])[0].event_id


def paused_doc(
    *,
    minted: bool = True,
    operations: list[str] | None = None,
    executions: list[dict[str, Any]] | None = None,
    status: str = "RUNNING",
) -> dict[str, Any]:
    """The node's activity while the exporter is paused."""

    incidents, workflows = [], []
    if minted:
        incidents = [
            {
                **incident("inc-dcgm", "dcgm_field_completeness", workflow="wf-dcgm"),
                "event_id": completeness_event(),
            }
        ]
        workflows = [
            workflow(
                "wf-dcgm",
                "inc-dcgm",
                BOUNDED if operations is None else operations,
                status=status,
                executions=[WAITING] if executions is None else executions,
            )
        ]
    return {
        "incidents": incidents,
        "workflows": workflows,
        "batches": [],
        "metric_latest": [observation("UNKNOWN")],
        "markers": [],
        "commands": [],
        "health_signals": {},
        "notifications": [],
        "collector_statuses": [],
    }


def observation(state: str) -> dict[str, Any]:
    return {
        "name": "collector_observation_valid",
        "device": "GPU_METRICS",
        "value": 1 if state == "HEALTHY" else 0,
        "labels": {
            "state": state,
            "reason": "FRESH_SUCCESS" if state == "HEALTHY" else "COLLECTION_ERROR",
        },
        "observed_at": "t",
    }


RESUMED = {
    "incidents": [],
    "workflows": [],
    "batches": [],
    "metric_latest": [observation("HEALTHY")],
    "markers": [],
    "commands": [],
    "health_signals": {},
    "notifications": [],
    "collector_statuses": [
        {
            "collector": "GPU_METRICS",
            "last_success_at": "2026-10-24T01:00:00+00:00",
            "last_error_at": None,
            "errors": [],
        }
    ],
    "agent": {"lifecycle_state": "ACTIVE"},
}


def run(
    tmp_path: Path, paused: dict[str, Any] | list[dict[str, Any]]
) -> dict[str, Any]:
    """The variant over ``paused`` reads (a list is served in order, the last
    repeating) until the node probe resumed the exporter, ``RESUMED`` after."""

    node = FakeProbe(
        {
            "survey": {
                "collector_env": {"GPU_FAULT_HOST_INTERVAL_SECONDS": "15"},
                "exporter_processes": [{"pid": 100, "start_ticks": "5"}],
                "residue": {"units": [], "paused_units": {}, "paused_processes": {}},
            },
            "exporter-pause": {"pid": 100, "stopped": True, "deadman": "d"},
            "exporter-resume": {"processes": [{"pid": 100, "resumed": True}]},
        }
    )
    reads = paused if isinstance(paused, list) else [paused]
    served = {"count": 0}

    def responder(script: str, *arguments: str) -> dict[str, Any]:
        if any(call and call[0] == "exporter-resume" for call in node.calls):
            return dict(RESUMED)
        index = min(served["count"], len(reads) - 1)
        served["count"] += 1
        return dict(reads[index])

    ctx = make_context(
        SPEC,
        tmp_path=tmp_path,
        regional=FakeRegional(cpu_python=responder),
        probes={f"node@{NODE}": node},
        settings=make_settings(SPEC),
    )
    return col.sustained_unknown(ctx)


def test_the_bounded_chain_with_a_waiting_validation_passes(tmp_path: Path) -> None:
    result = run(tmp_path, paused_doc())
    assert result["verdict"] == "PASS", result["errors"]
    assert set(result["stages"]) == {
        "detection",
        "validation",
        "unknown",
        "hardware",
        "bounded",
        "recovery",
    }, result["stages"]
    after = result["validation"]["after_sustain"]
    assert after["status"] == "WAITING", after
    assert after["health_evidence_pending"] == "collector state is UNKNOWN", after


def test_a_product_that_minted_nothing_fails(tmp_path: Path) -> None:
    """The old pass: UNKNOWN, no workflow to carry a hardware step, a fresh
    success after the resume -- and no incident at all."""

    result = run(tmp_path, paused_doc(minted=False))
    assert result["verdict"] == "FAIL", result
    assert result["stages"]["detection"] == [
        "no dcgm_field_completeness finding reached an incident"
    ], result["stages"]
    assert (
        result["stages"]["unknown"]
        == result["stages"]["bounded"]
        == result["stages"]["recovery"]
        == []
    ), "every stage the old verdict had reads clean"


def test_a_chain_without_the_snapshot_fails(tmp_path: Path) -> None:
    result = run(tmp_path, paused_doc(operations=["FREEZE_EVIDENCE", "VALIDATE_HOST"]))
    assert result["verdict"] == "FAIL", result
    assert any(
        "lack ['TRIGGER_HEALTH_SNAPSHOT', 'VALIDATE_HOST']" in error
        for error in result["stages"]["detection"]
    ), result["stages"]


def test_a_validation_that_never_ran_while_paused_fails(tmp_path: Path) -> None:
    result = run(tmp_path, paused_doc(executions=[]))
    assert result["verdict"] == "FAIL", result
    assert result["stages"]["validation"] == [
        "the bounded chain never ran its VALIDATE_HOST while the exporter was paused "
        "(workflow RUNNING, executed [])"
    ], result["stages"]


def test_a_validation_that_passed_while_paused_fails(tmp_path: Path) -> None:
    succeeded = {**WAITING, "status": "SUCCEEDED", "details": {}}
    result = run(tmp_path, paused_doc(executions=[succeeded], status="SUCCEEDED"))
    assert result["verdict"] == "FAIL", result
    assert result["stages"]["detection"] == [
        "VALIDATE_HOST SUCCEEDED while the exporter was still paused: a stale "
        "success passed the observation validation"
    ], result["stages"]


def test_detection_waits_for_the_validation_to_be_reached(tmp_path: Path) -> None:
    """The chain is read once VALIDATE_HOST has a record: an incident whose
    chain is still at its snapshot is read again, still paused."""

    reads = [paused_doc(minted=False)] * 2 + [paused_doc(executions=[])] * 2
    result = run(tmp_path, [*reads, paused_doc()])
    assert result["verdict"] == "PASS", result["errors"]
    assert result["validation"]["at_detection"]["status"] == "WAITING", result[
        "validation"
    ]

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any

import pytest

from gpu_fault.app import create_app
from gpu_fault.app.builtin_metric_contributors import (
    completion_state_metric_lines,
    control_loop_metric_lines,
    policy_metric_lines,
    postgres_pool_metric_lines,
    workflow_stall_metric_lines,
)
from gpu_fault.app.metrics import render_prometheus_metrics
from gpu_fault.app.metrics_sections import render_runtime_metrics_two
from gpu_fault.app.process_metrics import parse_lines
from gpu_fault.execution import ProductionExecutorConfig
from gpu_fault.models import WorkflowOperation, WorkflowStatus, WorkflowStepStatus
from gpu_fault.processor import ProcessorCoordinator
from tests._builders import build_context, workflow_request, workflow_step_execution
from tests.regional._cov95_runtime_support import offline_runtime as offline_runtime

LABEL = 'unit"\\scope\nline'
ESCAPED = 'unit\\"\\\\scope\\nline'


def test_renderer_preserves_populated_queue_and_phase_snapshots_with_safe_labels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = build_context()
    app = create_app(context)
    runtime = app.state.runtime
    processor = ProcessorCoordinator(
        context.store,
        owner_id="unit",
        internal_token="synthetic-unit-replay",
        active_consumers=False,
    )
    snapshot = processor.metrics_snapshot()
    snapshot.update(
        {
            "retry_rescheduled_by_path": {LABEL: 2},
            "lane_holder_by_path": {LABEL: {"count": 2, "sum": 1.25, "max": 0.75}},
            "in_flight_by_phase": {LABEL: {"count": 3, "oldest_seconds": 1.5}},
            "stale_superseded_by_path": {LABEL: 4},
            "consumer": {"cycles": 9},
        }
    )
    snapshot["claim"].update(
        {
            "rounds_by_stream": {LABEL: 5},
            "rows_by_stream": {LABEL: 6},
            "empty_by_stream": {LABEL: 7},
        }
    )
    snapshot["spool"]["rows_by_path"] = {LABEL: 8}
    runtime.processor = SimpleNamespace(metrics_snapshot=lambda: snapshot)
    runtime.processor_admission_rejections_by_path[LABEL] = 2
    runtime.processor_queue_bypass_paths = runtime.processor_queue_bypass_paths | {
        LABEL
    }
    runtime.processor_queue_bypasses_by_path[LABEL] = 3
    runtime.telemetry_spool_admitted_by_path[LABEL] = 4
    runtime.processor_replay_tracker.start(
        "unit-request", owner_id="unit", path="/unit", lane_epoch=1, phase=LABEL
    )
    monkeypatch.setattr(
        runtime.processor_admission_batcher,
        "scope_queue_snapshot",
        lambda: ([(LABEL, 1.25)], [(LABEL, 3, 4.5)]),
    )
    try:
        lines = render_prometheus_metrics(runtime)
    finally:
        runtime.processor_replay_tracker.finish("unit-request")
        for name in (
            "store_io",
            "decode_io",
            "fault_store_io",
            "evidence_store_io",
            "fault_decode_io",
            "telemetry_spool_store_io",
        ):
            getattr(runtime, name).close()
    expected = [
        f'gpu_fault_processor_retry_rescheduled_by_path_total{{path="{ESCAPED}"}} 2',
        f'gpu_fault_processor_lane_holder_seconds_sum{{path="{ESCAPED}"}} 1.250000',
        f'gpu_fault_processor_in_flight_by_phase{{phase="{ESCAPED}"}} 3',
        f'gpu_fault_processor_replay_inbound_by_phase{{phase="{ESCAPED}"}} 1',
        f'gpu_fault_processor_claim_rounds_by_stream_total{{stream="{ESCAPED}"}} 5',
        f'gpu_fault_telemetry_spool_claim_rows_by_path_total{{path="{ESCAPED}"}} 8',
        f'gpu_fault_processor_admission_batch_scope_in_flight_seconds{{cluster_id="{ESCAPED}"}} 1.250000',
        f'gpu_fault_processor_admission_batch_scope_pending{{cluster_id="{ESCAPED}"}} 3',
        f'gpu_fault_processor_admission_batch_scope_wait_seconds{{cluster_id="{ESCAPED}"}} 4.500000',
        f'gpu_fault_processor_stale_superseded_by_path_total{{path="{ESCAPED}"}} 4',
        f'gpu_fault_processor_admission_rejections_by_path_total{{path="{ESCAPED}"}} 2',
        f'gpu_fault_processor_queue_bypass_total{{path="{ESCAPED}"}} 3',
        f'gpu_fault_telemetry_spool_admitted_by_path_total{{path="{ESCAPED}"}} 4',
        "gpu_fault_processor_consumer_cycles_total 9",
    ]
    assert set(expected) <= set(lines), set(expected) - set(lines)
    assert all("\n" not in line for line in lines), "labels must not inject new series"
    assert not any(
        line.startswith("gpu_fault_processor_consumer_running ") for line in lines
    ), "a partial snapshot cannot invent missing consumer state"


def test_legacy_store_without_spool_statistics_has_a_defined_empty_view() -> None:
    lines = []
    result = render_runtime_metrics_two(
        lines, SimpleNamespace(store=object()), False, False, set(), {}
    )
    assert result == {
        "depth": 0,
        "leased": 0,
        "oldest_age_seconds": 0.0,
        "by_cluster": {},
        "payload_bytes": 0,
        "leased_bytes": 0,
    }
    assert lines == ["gpu_fault_processor_queue_bypass_enabled 0"]


@pytest.mark.parametrize("shape", ["missing", "invalid-values"])
def test_partial_periodic_snapshots_do_not_invent_success_timestamps(
    shape: str,
) -> None:
    context = build_context()
    snapshot: dict[str, Any] = {
        "last_cycle_timestamp_seconds": None,
        "lease_error_last_seen_timestamp_seconds": None,
        "periodic_job_errors_total": {},
    }
    if shape == "missing":
        snapshot["job_last_run_timestamp_seconds"] = None
        snapshot["job_error_last_seen_timestamp_seconds"] = []
    else:
        snapshot["job_last_run_timestamp_seconds"] = {"identity": None, LABEL: 12.5}
        snapshot["job_error_last_seen_timestamp_seconds"] = {
            "identity": "unknown",
            LABEL: 14.5,
        }
    context.periodic_runner = SimpleNamespace(metrics_snapshot=lambda: snapshot)
    lines = control_loop_metric_lines(SimpleNamespace(context=context))
    assert "gpu_fault_periodic_last_cycle_timestamp_seconds 0.000" in lines
    assert not any('{periodic_job="identity"}' in line for line in lines), lines
    if shape == "invalid-values":
        assert (
            f'gpu_fault_periodic_job_last_run_timestamp_seconds{{periodic_job="{ESCAPED}"}} 12.500'
            in lines
        )
        assert (
            f'gpu_fault_periodic_job_error_last_seen_timestamp_seconds{{periodic_job="{ESCAPED}"}} 14.500'
            in lines
        )


def test_optional_metric_producers_do_not_fabricate_unavailable_fields() -> None:
    context = build_context()
    context.gpu_metrics = SimpleNamespace()
    context.policy = SimpleNamespace(unknown_product_counts=lambda: {LABEL: 2})
    runtime = SimpleNamespace(context=context)
    assert (
        f'gpu_fault_policy_unknown_product_total{{product="{ESCAPED}"}} 2'
        in policy_metric_lines(runtime)
    )
    assert (
        'gpu_fault_gpu_findings_without_incident_total{reason="suppressed_by_composite"} 0'
        in completion_state_metric_lines(runtime)
    )
    context.postgres_pool_capacity = SimpleNamespace(
        pool_max=8,
        demand_by_consumer={},
        oversubscription_ratio=0,
        unpooled_connections=0,
    )
    lines = postgres_pool_metric_lines(runtime)
    assert "gpu_fault_postgres_pool_max_size 8" in lines
    assert not any(
        line.startswith("gpu_fault_postgres_pool_headroom_connections ")
        for line in lines
    ), "legacy estimates cannot claim a measured headroom value"


def test_workflow_stall_metrics_ignore_completed_steps_in_an_open_workflow() -> None:
    now = datetime.now(timezone.utc)
    workflow = workflow_request(
        "unit-workflow",
        "unit-incident",
        status=WorkflowStatus.RUNNING,
        execution_deadline=now - timedelta(seconds=3),
        step_executions=[
            workflow_step_execution(0, WorkflowOperation.FREEZE_EVIDENCE),
            workflow_step_execution(
                1,
                WorkflowOperation.VERIFY_NO_GPU_CLIENTS,
                WorkflowStepStatus.WAITING,
                started_at=now - timedelta(seconds=10),
            ),
        ],
    )
    lines = workflow_stall_metric_lines(
        [workflow],
        now,
        {WorkflowStatus.SUCCEEDED, WorkflowStatus.FAILED},
        ProductionExecutorConfig(
            enabled=False, executor_id="unit", allowed_operations=set()
        ),
    )
    assert not any('operation="FREEZE_EVIDENCE"' in line for line in lines), lines
    (waiting,) = parse_lines(lines).samples["gpu_fault_workflow_step_waiting_seconds"]
    assert waiting.labels == (("operation", "VERIFY_NO_GPU_CLIENTS"),)
    assert float(waiting.value) == 10.0

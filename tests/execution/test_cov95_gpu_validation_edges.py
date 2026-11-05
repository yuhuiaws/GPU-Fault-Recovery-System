"""``gpu_fault.adapters.gpu_validation`` edges the validation suites do not reach.

The inventory gate on requirements it cannot bound (an instance type the
inventory table does not know, a count that is not a number, metrics of the
other family), a GPU metric missing on a step that names no GPUs, a Fabric
Manager heartbeat too old to judge, the quiesce-taint lineage over records
that are missing or loop, and the small readers the adapter builds on. The
adapter runs over the neighbouring ``ValidationHarness`` feed
(``tests/execution/_cov95_runtime_validation.py``) or a store-shaped fake.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any

import pytest

from gpu_fault.adapters import quarantine_taint_value
from gpu_fault.adapters.gpu_validation import (
    GpuValidationAdapter,
    current_interface_readings,
    finding_unjudged_since,
    newest_gpu_sample_times,
    quiesce_taint_pending,
    telemetry_readiness_after,
)
from gpu_fault.collector_requirements import CollectorServiceState
from gpu_fault.execution import WorkflowExecutionRequest, WorkflowStepContext
from gpu_fault.gpu_fabric_manager import FABRIC_MANAGER_STATUS
from gpu_fault.gpu_metrics import GpuMetricSample
from gpu_fault.health_evidence_validation import cooldown_key
from gpu_fault.health_recovery_contract import COOLDOWN_SECONDS, HEALTH_CONTRACT
from gpu_fault.models import (
    IncidentState,
    WorkflowOperation,
    WorkflowStatus,
    WorkflowStepStatus,
)
from gpu_fault.store import InMemoryStore
from gpu_fault.store.shared.errors import NotFoundError
from gpu_fault.telemetry import CollectorKind, CollectorStatus
from tests._builders import (
    build_store,
    fault_incident,
    workflow_request,
    workflow_step,
    workflow_step_execution,
)
from tests.execution._cov95_runtime_validation import ValidationHarness

OP = WorkflowOperation
GPU = OP.VALIDATE_GPU
GPU_PLUGIN = OP.RESTART_GPU_DEVICE_PLUGIN
NOW = datetime(2026, 11, 5, 9, 0, tzinfo=timezone.utc)


def test_without_a_clean_batch_time_every_link_reading_is_current() -> None:
    stale = SimpleNamespace(
        name="network_link_up", observed_at=NOW - timedelta(hours=1)
    )
    load = SimpleNamespace(name="load1_per_cpu", observed_at=NOW - timedelta(hours=1))
    assert current_interface_readings([stale, load], None) == [stale, load], (
        "with no clean HOST_TELEMETRY batch to date them, no reading is dropped"
    )
    assert current_interface_readings([stale, load], NOW) == [load], (
        "a link row older than the newest clean batch is out of scope"
    )


def test_a_step_waiting_cap_that_is_not_positive_is_refused() -> None:
    with pytest.raises(ValueError, match="step waiting cap must be positive"):
        GpuValidationAdapter(SimpleNamespace(store=None), step_waiting_cap=timedelta(0))


def _inventory(requirement: dict[str, Any], reading: float | None = 8):
    h = ValidationHarness(GPU, action=GPU_PLUGIN, requirements={"node-a": requirement})
    if reading is not None:
        h.feed.host_metric("node-a", "gpu_inventory_active_count", reading)
    return h.execute()


def test_an_unknown_instance_type_leaves_the_configured_count_as_it_is() -> None:
    requirement = {
        "active_metric": "gpu_inventory_active_count",
        "expected_count": 9,
        "node_instance_type": "x9.unlisted",
    }
    outcome = _inventory(requirement)
    assert outcome.status is WorkflowStepStatus.FAILED, outcome
    assert outcome.details["node_failures"] == {
        "node-a": ["gpu_inventory_active_count=8,expected=9"]
    }, "no physical count to cap against: the configured count is validated"
    assert "inventory_requirement_clamped" not in outcome.details, outcome.details

    capped = _inventory({**requirement, "node_instance_type": "p5en.48xlarge"})
    assert capped.status is WorkflowStepStatus.SUCCEEDED, capped
    assert capped.details["inventory_requirement_clamped"] == {
        "node-a": {
            "gpu_inventory_active_count": {
                "configured": 9,
                "physical": 8,
                "instance_type": "p5en.48xlarge",
            }
        }
    }, capped.details


def test_a_count_that_is_not_a_number_is_kept_and_never_matches() -> None:
    outcome = _inventory(
        {
            "active_metric": "gpu_inventory_active_count",
            "expected_count": "eight",
            "node_instance_type": "p5en.48xlarge",
        }
    )
    assert outcome.status is WorkflowStepStatus.FAILED, outcome
    assert outcome.details["node_failures"] == {
        "node-a": ["gpu_inventory_active_count=8,expected=eight"]
    }, "an unreadable requirement fails closed, it is not capped into a pass"


def test_a_requirement_of_the_other_family_asks_nothing_of_a_gpu_validation() -> None:
    outcome = _inventory({"metrics": {"efa_inventory_active_count": 16}}, reading=None)
    assert outcome.status is WorkflowStepStatus.SUCCEEDED, outcome
    assert "inventory_evidence" not in outcome.details, outcome.details


def test_a_missing_metric_on_a_step_without_named_gpus_waits_on_the_node() -> None:
    h = ValidationHarness(GPU)
    h.context = replace(
        h.context, step=h.context.step.model_copy(update={"gpu_uuids": []})
    )
    h.feed.gpu["node-a"] = [
        item
        for item in h.feed.gpu["node-a"]
        if item.sample.canonical_name != "gpu_temperature_c"
    ]

    outcome = h.execute()

    assert outcome.status is WorkflowStepStatus.WAITING, outcome
    assert outcome.details["node_pending"]["node-a"] == {
        "missing_recent_metrics": ["gpu_temperature_c"]
    }, "no per-GPU breakdown when the step names no GPUs"


def _legacy_cooldown(findings: list[Any]) -> tuple[ValidationHarness, InMemoryStore]:
    """A VALIDATE_GPU persisted with the old THERMAL_COOLDOWN contract (new
    plans carry ENVIRONMENT_REVIEW; in-flight plans are not rewritten), over a
    feed whose health-signal states live in an in-memory store."""

    h = ValidationHarness(GPU)
    signals = InMemoryStore()
    h.feed.get_health_signal_state = signals.get_health_signal_state
    h.feed.claim_health_signal_transition = signals.claim_health_signal_transition
    h.feed.findings = lambda cluster_id, node_id: list(findings)
    contract = {"mode": "THERMAL_COOLDOWN", "metric": "gpu_temperature_c"}
    step = h.context.step.model_copy(
        update={"parameters": {**h.context.step.parameters, HEALTH_CONTRACT: contract}}
    )
    workflow = h.context.workflow.model_copy(
        update={"created_at": h.action_at - timedelta(minutes=1)}
    )
    h.context = replace(h.context, step=step, workflow=workflow)
    return h, signals


def test_a_legacy_cooldown_plan_waits_until_cool_readings_span_the_cooldown() -> None:
    h, signals = _legacy_cooldown([])

    outcome = h.execute()

    assert outcome.status is WorkflowStepStatus.WAITING, outcome
    assert outcome.details["health_evidence_pending"] == "thermal cooldown not complete"
    assert outcome.details["node_pending"] == {
        "node-a": f"healthy observations must span {COOLDOWN_SECONDS}s"
    }, outcome.details
    state = signals.get_health_signal_state(cooldown_key(h.context, "node-a"))
    assert state is not None and state.active is True, (
        "the first cool reading starts the cooldown run"
    )


def test_a_hot_finding_under_a_legacy_cooldown_plan_restarts_the_cooldown() -> None:
    hot = SimpleNamespace(
        finding_id="finding-hot",
        gpu_uuid="GPU-a",
        canonical_name="gpu_temperature_c",
        severity=SimpleNamespace(value="CRITICAL"),
        threshold_source=None,
    )
    findings: list[Any] = []
    h, signals = _legacy_cooldown(findings)
    key = cooldown_key(h.context, "node-a")
    assert h.execute().status is WorkflowStepStatus.WAITING
    started = signals.get_health_signal_state(key)
    assert started is not None and started.active is True, "the cooldown run started"

    findings.append(hot)
    outcome = h.execute()

    assert outcome.status is WorkflowStepStatus.WAITING, outcome
    assert outcome.details["node_pending"]["node-a"] == {
        "thermal_cooling": ["finding-hot"]
    }, outcome.details
    reset = signals.get_health_signal_state(key)
    assert reset is not None and reset.active is False, (
        "a GPU that is hot again resets the cooldown run instead of failing the node"
    )


def _fabric_context(*, seen_seconds_ago: float):
    """A diagnostic VALIDATE_GPU (no step before it) of an H200 node whose
    fabric trained, with a Fabric Manager unit the Agent last reported failed."""

    now = datetime.now(timezone.utc)
    sampled_at = now - timedelta(seconds=5)
    seen_at = now - timedelta(seconds=seen_seconds_ago)

    class FabricStore:
        def list_collector_statuses(self, cluster_id, node_id):
            return [
                CollectorStatus(
                    cluster_id=cluster_id,
                    node_id=node_id,
                    collector=CollectorKind.GPU_METRICS,
                    observed_at=sampled_at,
                    ingested_at=sampled_at,
                    last_success_at=sampled_at,
                )
            ]

        def list_telemetry_metrics_latest(self, cluster_id, node_id):
            return []

        def get_agent(self, cluster_id, node_id):
            return SimpleNamespace(
                collector_services={
                    "nvidia-fabricmanager": CollectorServiceState.model_validate(
                        {"active": "failed", "enabled": "enabled"}
                    )
                },
                last_seen_at=seen_at,
            )

        def get_workflow(self, request_id):
            raise NotFoundError(request_id)

    class FabricMetrics:
        store = FabricStore()

        def latest(self, cluster_id, node_id):
            return [
                SimpleNamespace(
                    observed_at=sampled_at,
                    sample=GpuMetricSample(
                        metric_name=name,
                        canonical_name=name,
                        value=value,
                        gpu_uuid="GPU-a",
                        labels={"modelName": "NVIDIA H200"},
                    ),
                )
                for name, value in (
                    ("gpu_temperature_c", 35.0),
                    (FABRIC_MANAGER_STATUS, 3),
                )
            ]

        def findings(self, cluster_id, node_id):
            return []

    step = workflow_step(
        GPU, "gpu-fault-validation-adapter", node_ids=["node-a"], gpu_uuids=["GPU-a"]
    )
    workflow = workflow_request(
        "workflow-diagnostic",
        "incident-diagnostic",
        status=WorkflowStatus.RUNNING,
        official_steps=[step],
    )
    incident = fault_incident(
        "incident-diagnostic", "event-diagnostic", node_ids=["node-a"]
    )
    adapter = GpuValidationAdapter(FabricMetrics(), store=FabricMetrics.store)
    return adapter, WorkflowStepContext(
        workflow=workflow,
        incident=incident,
        step=step,
        step_index=0,
        request=WorkflowExecutionRequest(expected_fencing_token=3),
        idempotency_key="validation/fabric-heartbeat",
    )


def test_a_heartbeat_older_than_the_sample_age_does_not_judge_the_unit() -> None:
    adapter, context = _fabric_context(seen_seconds_ago=600)
    outcome = adapter.execute(context)
    assert outcome.status is WorkflowStepStatus.SUCCEEDED, outcome.details

    adapter, context = _fabric_context(seen_seconds_ago=10)
    fresh = adapter.execute(context)
    assert fresh.status is WorkflowStepStatus.FAILED, fresh.details
    assert fresh.details["node_failures"]["node-a"] == ["fabric_manager_unit_failed"], (
        "a recent heartbeat that reports the unit failed is judged"
    )


NODE = "ip-10-0-41-164"


def test_the_quiesce_lineage_skips_unreadable_records_and_loops() -> None:
    """The reboot successor names its reset predecessor, which names the
    successor back; the successor's incident points at a workflow no store
    holds, and the predecessor's incident is gone: the taint is still found."""

    store = build_store()
    successor_id = "workflow-reboot-after-wf-missing"
    predecessor = workflow_request(
        "wf-reset",
        "inc-archived",
        status=WorkflowStatus.FAILED,
        predecessor_workflow_id=successor_id,
        official_steps=[
            workflow_step(OP.QUIESCE_GPU_SERVICES, node_ids=[NODE]),
            workflow_step(OP.RESET_GPU, node_ids=[NODE]),
        ],
        step_executions=[
            workflow_step_execution(
                0, OP.QUIESCE_GPU_SERVICES, details={"quiesce_taint_nodes": [NODE]}
            ),
            workflow_step_execution(1, OP.RESET_GPU, WorkflowStepStatus.FAILED),
        ],
    )
    store.save_workflow(predecessor)
    step = workflow_step(GPU, "gpu-fault-validation-adapter", node_ids=[NODE])
    successor = workflow_request(
        successor_id,
        "inc-reboot-after-wf-missing",
        status=WorkflowStatus.RUNNING,
        predecessor_workflow_id="wf-reset",
        official_steps=[step],
    )
    incident = fault_incident(
        "inc-reboot-after-wf-missing",
        "reboot-after-wf-missing",
        state=IncidentState.ACTION_PENDING,
        node_ids=[NODE],
    )
    context = WorkflowStepContext(
        workflow=successor,
        incident=incident,
        step=step,
        step_index=0,
        request=WorkflowExecutionRequest(expected_fencing_token=3),
        idempotency_key="validation/lineage",
    )

    pending = quiesce_taint_pending(store, context, NODE)

    assert pending["quiesce_taint_holders"] == {
        "wf-reset": quarantine_taint_value("inc-archived")
    }, pending
    assert (
        "the node still carries the gpu-fault quiesce taint (NoExecute) of wf-reset"
        in (pending["reason"])
    )


def test_a_finding_without_a_comparable_time_keeps_the_fixed_grace() -> None:
    grace_end = NOW
    assert finding_unjudged_since(SimpleNamespace(observed_at=NOW), grace_end) is True
    naive = SimpleNamespace(observed_at=NOW.replace(tzinfo=None))
    assert finding_unjudged_since(naive, grace_end) is False, (
        "a naive time cannot be compared with the grace end"
    )
    assert finding_unjudged_since(SimpleNamespace(), grace_end) is False


def test_the_newest_sample_times_skip_rows_without_a_sample_or_a_time() -> None:
    sample = GpuMetricSample(
        metric_name="gpu_temperature_c",
        canonical_name="gpu_temperature_c",
        value=35,
        gpu_uuid="GPU-a",
    )
    later = NOW + timedelta(seconds=30)
    rows = [
        SimpleNamespace(sample=None, observed_at=later + timedelta(hours=1)),
        SimpleNamespace(sample=sample, observed_at=None),
        SimpleNamespace(sample=sample, observed_at=later),
        SimpleNamespace(sample=sample, observed_at=NOW),
    ]
    assert newest_gpu_sample_times(rows) == {("GPU-a", "gpu_temperature_c"): later}


def test_only_a_validation_step_has_a_telemetry_readiness_barrier() -> None:
    restart = workflow_step(OP.RESTART_NODE, node_ids=[NODE])
    workflow = workflow_request(
        "workflow-restart",
        "incident-restart",
        official_steps=[workflow_step(OP.RESTART_GPU_DEVICE_PLUGIN), restart],
        step_executions=[
            workflow_step_execution(0, OP.RESTART_GPU_DEVICE_PLUGIN, updated_at=NOW)
        ],
    )
    context = WorkflowStepContext(
        workflow=workflow,
        incident=fault_incident("incident-restart", "event-restart"),
        step=restart,
        step_index=1,
        request=WorkflowExecutionRequest(expected_fencing_token=3),
        idempotency_key="restart/readiness",
    )
    assert telemetry_readiness_after(context) is None, (
        "a node action does not wait for telemetry readiness"
    )

"""A task NIC/port scope that cannot be determined is an explicit UNKNOWN verdict.

Live (HyperPod, GF-GENERAL-LIVE-011.active_to_zero): neither node could bind its
rank processes to EFA ports, and the hung DAG's VALIDATE_FABRIC waited on
``TASK_NIC_PORT_SCOPE_UNAVAILABLE`` until the 600 s per-step cap. The recheck
survey that fills the scope had already run; nothing would ever fill it.
"""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from gpu_fault.adapters.gpu_validation import GpuValidationAdapter
from gpu_fault.execution import WorkflowStepContext
from gpu_fault.fabric_scope import (
    reported_evidence,
    scope_evidence,
    unknown_task_fabric,
    validation_evidence,
)
from gpu_fault.models import (
    WorkflowExecutionRequest,
    WorkflowOperation,
    WorkflowStepStatus,
)
from gpu_fault.telemetry import CollectorKind, CollectorStatus
from tests._builders import build_store, copy_model
from tests.execution._support import workflow_state

FABRIC = WorkflowOperation.VALIDATE_FABRIC
SUPPORT = WorkflowOperation.ESCALATE_SUPPORT
HUNG = {
    "health_recovery_contract": {
        "mode": "HUNG_DIAGNOSTIC",
        "detected_at": "2026-10-24T13:50:17+00:00",
    },
    "require_task_fabric_scope": True,
    "training_fabric_by_node": {},
}


class Store:
    def __init__(self, now):
        self.now = now

    def list_collector_statuses(self, cluster_id, node_id):
        return [
            CollectorStatus(
                cluster_id=cluster_id,
                node_id=node_id,
                collector=collector,
                observed_at=self.now,
                ingested_at=self.now,
                last_success_at=self.now,
            )
            for collector in (CollectorKind.GPU_METRICS, CollectorKind.HOST_TELEMETRY)
        ]

    def list_telemetry_metrics_latest(self, cluster_id, node_id):
        # Only the management link, healthy.
        return [
            SimpleNamespace(
                observed_at=self.now, name="network_link_up", value=1, device="eth0"
            )
        ]


class Metrics:
    def __init__(self, now):
        self.store = Store(now)
        self.now = now

    def latest(self, cluster_id, node_id):
        return [
            SimpleNamespace(
                observed_at=self.now, sample=SimpleNamespace(canonical_name=name)
            )
            for name in (
                "nvlink_crc_aggregate_error_total",
                "nvlink_recovery_aggregate_error_total",
                "nvlink_replay_aggregate_error_total",
            )
        ]

    def findings(self, cluster_id, node_id):
        return []


def validate(handoff: bool):
    now = datetime.now(timezone.utc)
    store = build_store()
    operations = [FABRIC, SUPPORT] if handoff else [FABRIC]
    incident, workflow = workflow_state(store, operations)
    steps = [
        copy_model(
            step,
            execution_owner="gpu-fault-validation-adapter",
            parameters=dict(HUNG),
            depends_on_step_indexes=[0] if index else [],
        )
        for index, step in enumerate(workflow.official_steps)
    ]
    workflow = copy_model(workflow, official_steps=steps, dag_enabled=True)
    metrics = Metrics(now)
    return GpuValidationAdapter(metrics, store=metrics.store).execute(
        WorkflowStepContext(
            workflow=workflow,
            incident=incident,
            step=steps[0],
            step_index=0,
            request=WorkflowExecutionRequest(expected_fencing_token=3),
            idempotency_key="validation/hung-fabric",
        )
    )


def test_unbound_scope_resolves_unknown_before_a_support_handoff():
    outcome = validate(handoff=True)
    assert outcome.status is WorkflowStepStatus.SUCCEEDED, outcome
    assert outcome.details["fabric_status"] == "UNKNOWN", outcome
    assert outcome.details["validated_nodes"] == [], outcome
    assert outcome.details["unvalidated_nodes"] == ["node-a"], outcome
    assert outcome.details["task_fabric_unknown_nodes"] == {
        "node-a": {
            "fabric_status": "UNKNOWN",
            "reason": "TASK_NIC_PORT_SCOPE_UNAVAILABLE",
            "scope_basis": "NOT_REPORTED",
        }
    }, outcome


def test_without_a_handoff_an_unknown_scope_never_completes_the_step():
    """A SUCCEEDED diagnostic with no hand-off would end RECOVERED: keep waiting."""

    outcome = validate(handoff=False)
    assert outcome.status is WorkflowStepStatus.WAITING, outcome
    assert outcome.details["node_pending"]["node-a"]["reason"] == (
        "TASK_NIC_PORT_SCOPE_UNAVAILABLE"
    ), outcome
    assert "task_fabric_unknown_nodes" not in outcome.details, outcome


def test_partial_scope_names_the_validated_and_the_unknown_nodes():
    unknown = {"node-b": {"fabric_status": "UNKNOWN"}}
    evidence = validation_evidence(FABRIC, {}, unknown, ["node-a", "node-b"])
    assert evidence["fabric_evidence_scope"] == "PARTIAL_TASK_SCOPE", evidence
    assert evidence["validated_nodes"] == ["node-a"], evidence
    assert evidence["nccl_recovery_proven"] is False, evidence
    assert "fabric_status" not in validation_evidence(FABRIC, {}), (
        "a fully bound validation keeps its established evidence"
    )


@pytest.mark.parametrize(
    "binding,reason",
    [
        (None, "TASK_NIC_PORT_SCOPE_UNAVAILABLE"),
        ({"basis": "UNKNOWN"}, "TASK_NIC_PORT_SCOPE_UNAVAILABLE"),
        (
            {
                "basis": "TASK_DEVICE_BINDING",
                "network_interfaces": [],
                "rdma_ports": [],
            },
            "INVALID_TASK_DEVICE_BINDING",
        ),
        (
            {
                "basis": "TASK_DEVICE_BINDING",
                "network_interfaces": [],
                "rdma_ports": ["e/1"],
            },
            None,
        ),
    ],
)
def test_each_node_gets_its_own_scope_verdict(binding, reason):
    step = SimpleNamespace(
        operation=FABRIC,
        node_ids=["node-a"],
        parameters={
            **HUNG,
            "training_fabric_by_node": {} if binding is None else {"node-a": binding},
        },
    )
    verdicts = unknown_task_fabric(step)
    assert (verdicts.get("node-a") or {}).get("reason") == reason, verdicts


def test_an_ordinary_fabric_validation_has_no_task_scope_verdict():
    step = SimpleNamespace(operation=FABRIC, node_ids=["node-a"], parameters={})
    assert unknown_task_fabric(step) == {}


def test_survey_evidence_is_bounded_to_codes_and_counts():
    evidence = scope_evidence(
        {
            "basis": "UNKNOWN",
            "reason": "TASK_RDMA_COVERAGE_INCOMPLETE",
            "source": "LIVE_PROCESS_RDMA_QP",
            "expected_pids": [100, 101],
            "covered_pids": [100],
            "rdma_ports": ["rdmap16s27/1"],
        }
    )
    assert evidence == {
        "scope_basis": "UNKNOWN",
        "scope_reason": "TASK_RDMA_COVERAGE_INCOMPLETE",
        "scope_source": "LIVE_PROCESS_RDMA_QP",
        "expected_process_count": 2,
        "covered_process_count": 1,
    }, evidence
    assert reported_evidence(evidence) == evidence
    assert scope_evidence({"reason": "x" * 500, "basis": ["UNKNOWN"]}) == {
        "scope_basis": "NOT_REPORTED"
    }
    assert reported_evidence(
        {"scope_reason": "<script>", "covered_process_count": -1}
    ) == {"scope_basis": "NOT_REPORTED"}
    assert reported_evidence("garbage") == {"scope_basis": "NOT_REPORTED"}

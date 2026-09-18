from __future__ import annotations

from dataclasses import replace

import pytest

from gpu_fault.notifications import registry
from gpu_fault.notifications.dcgm_diagnostic import DcgmDiagnosticEmailBuilder
from gpu_fault.notifications.diagnostic_inconclusive import (
    DiagnosticInconclusiveEmailBuilder,
)
from gpu_fault.notifications.hardware_escalation import HardwareEscalationEmailBuilder
from gpu_fault.notifications.nvlink74_mechanical import Nvlink74MechanicalEmailBuilder
from gpu_fault.notifications.restart_guard import RestartGuardEmailBuilder


def action_context():
    return {
        "cluster_id": "cluster-local",
        "incident_id": "incident-local",
        "workflow_id": "workflow-local",
        "event_id": "event-local",
        "event_type": "SXID",
        "policy_source": "unit-policy",
        "official_action": None,
        "reasons": ["unit-reason"],
        "operation_id": "operation-local",
        "workload_ids": ["training/job/train"],
    }


@pytest.mark.parametrize("defect", ["missing-kind", "missing-method"])
def test_registry_rejects_incomplete_or_unbuildable_definitions(monkeypatch, defect):
    kind = registry.NotificationKind.DCGM_DIAGNOSTIC
    if defect == "missing-kind":
        monkeypatch.delitem(registry.NOTIFICATION_REGISTRY, kind)
        message = "not exhaustive"
    else:
        monkeypatch.setitem(
            registry.NOTIFICATION_REGISTRY,
            kind,
            replace(registry.NOTIFICATION_REGISTRY[kind], method="absent"),
        )
        message = "builder has no method"
    with pytest.raises(RuntimeError, match=message):
        registry.validate_notification_registry()


@pytest.mark.parametrize(
    "sxid,partition,expected_sxid,expected_partition",
    [
        (
            {"b": [5, 6], "a": [4], "ignored": "invalid"},
            {"b": "partition-b", "a": "partition-a", "ignored": 1},
            "a=4; b=5,6",
            "a=partition-a; b=partition-b",
        ),
        ({"ignored": "invalid"}, {"ignored": 1}, "UNKNOWN", "UNKNOWN"),
        (None, None, "UNKNOWN", "UNKNOWN"),
    ],
)
def test_fabric_reset_rendering_preserves_per_node_identity_and_unknowns(
    sxid, partition, expected_sxid, expected_partition
):
    result = RestartGuardEmailBuilder().build_fabric_reset_completed(
        **action_context(),
        node_results={"node-local": {}},
        sxid=sxid,
        fabric_partition=partition,
    )
    assert (
        result.deduplication_key
        == "cluster-local/incident-local/fabric-reset/operation-local"
    )
    assert expected_sxid in result.body_text
    assert expected_partition in result.body_text
    assert "unit-reason" in result.body_text
    assert result.support_case_draft == ""


def test_gpu_reset_rendering_handles_legacy_results_without_inventing_gpu_ids():
    result = RestartGuardEmailBuilder().build_gpu_reset_completed(
        **action_context(),
        node_ids=["node-dict", "node-string"],
        gpu_uuids=["GPU-local"],
        node_results={
            "node-dict": {"reset_gpu_uuids": "invalid"},
            "node-string": "NODE_LOST",
        },
    )
    node_lines = [
        line
        for line in result.body_text.splitlines()
        if line.startswith("- Node") and "node-dict" in line
    ]
    assert len(node_lines) == 1
    assert "GPU-local" in node_lines[0]
    assert "NODE_LOST" in result.body_text
    assert result.category == "ACTION_COMPLETED"
    assert result.priority == 10


def test_count_approval_only_renders_complete_workloads_with_explicit_context():
    result = RestartGuardEmailBuilder().build_gpu_count_change(
        cluster_id="cluster-local",
        incident_id="incident-local",
        job_id="training",
        attempt_id="attempt",
        workload_ids=["invalid", "/job/unnamed", "training/job/train"],
        source_gpu_count=8,
        target_gpu_count=4,
        approval_annotation="8-to-4",
    )
    commands = [
        line.strip() for line in result.body_text.splitlines() if "kubectl" in line
    ]
    assert len(commands) == 1
    assert '--context "${GPU_FAULT_KUBE_CONTEXT:' in commands[0]
    assert "-n training annotate job train" in commands[0]
    assert "8-to-4" in commands[0]
    assert "invalid" not in commands[0]


def test_budget_exhaustion_keeps_the_attempt_and_budget_visible():
    result = RestartGuardEmailBuilder().build_budget_exhausted(
        cluster_id="cluster-local",
        incident_id="incident-local",
        job_id="training",
        attempt_id="attempt",
        restart_count=3,
        restart_budget=3,
    )
    assert (
        result.deduplication_key == "cluster-local/training/restart-budget-exhausted/3"
    )
    assert "attempt" in result.body_text
    assert "3" in result.subject


def test_dcgm_ignores_untyped_entries_and_keeps_highest_outcome_and_unique_evidence():
    result = DcgmDiagnosticEmailBuilder().build(
        cluster_id="cluster-local",
        incident_id="incident-local",
        workflow_id="workflow-local",
        event_id="event-local",
        operation_id="operation-local",
        workload_ids=[],
        control_plane_action="DRAIN_AND_QUARANTINE",
        node_results={
            "a": {
                "diagnostic_outcome": "PASS",
                "diagnostic_findings": [
                    None,
                    {"test_name": "unit-check", "status": "PASS"},
                ],
                "recommended_actions": [None, {"action_code": "UNIT_GUIDANCE"}],
                "evidence_ref": "s3://local/evidence",
            },
            "b": {
                "diagnostic_outcome": "FAIL",
                "diagnostic_findings": [],
                "evidence_ref": "s3://local/evidence",
            },
        },
    )
    assert "FAIL" in result.subject
    assert "unit-check" in result.body_text
    assert "UNIT_GUIDANCE" in result.body_text
    assert "DRAIN_AND_QUARANTINE" in result.body_text
    assert result.evidence_refs == ["s3://local/evidence"]


def test_non_nvlink_mechanical_task_never_claims_a_vendor_case_was_created():
    result = Nvlink74MechanicalEmailBuilder().build(
        cluster_id="cluster-local",
        incident_id="incident-local",
        workflow_id="workflow-local",
        node_ids=["node-local"],
        link_id=None,
        pci_bdf=None,
        occurrence_counts={},
        annotation="approval",
        annotation_value="pending",
        xid=119,
    )
    assert "XID 119" in result.subject
    assert "/gpu-mechanical/" in result.deduplication_key
    assert "Operator task only" in result.support_case_draft
    assert "not created" in result.support_case_draft


def test_diagnostic_and_hardware_escalation_keep_observed_operation_context():
    diagnostic = DiagnosticInconclusiveEmailBuilder().build(
        cluster_id="cluster-local",
        incident_id="incident-local",
        workflow_id="workflow-local",
        event_id="event-local",
        node_ids=["node-local"],
        operations=["RUN_DCGM_DIAGNOSTIC"],
        failed_operation="RUN_DCGM_DIAGNOSTIC",
        error="unit timeout",
        policy_source="unit-policy",
        official_action=None,
        reasons=["unit-reason"],
    )
    assert diagnostic.category == "ADVISORY"
    assert "unit timeout" in diagnostic.body_text
    assert "RUN_DCGM_DIAGNOSTIC" in diagnostic.body_text
    assert diagnostic.support_case_draft == ""
    escalated = HardwareEscalationEmailBuilder().build(
        cluster_id="cluster-local",
        incident_id="incident-local",
        workflow_id="workflow-local",
        event_id="event-local",
        event_type="XID",
        node_ids=["node-local"],
        workload_ids=[],
        reasons=["unit-reason"],
        failed_operations=["RUN_DCGM_DIAGNOSTIC"],
        policy_source="unit-policy",
        official_action=None,
        ticket_id="internal-local-draft",
    )
    assert "internal-local-draft" in escalated.support_case_draft
    assert "RUN_DCGM_DIAGNOSTIC" in escalated.support_case_draft
    assert "unit-reason" in escalated.support_case_draft

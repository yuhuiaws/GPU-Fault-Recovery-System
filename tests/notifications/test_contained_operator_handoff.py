from __future__ import annotations

from gpu_fault.notifications import ContainedOperatorHandoffEmailBuilder
from gpu_fault.notifications.contained_operator_handoff import (
    CONTAINED_OPERATOR_HANDOFF_CATEGORY,
    CONTAINED_OPERATOR_HANDOFF_TEMPLATE_VERSION,
)
from gpu_fault.notifications.registry import NOTIFICATION_REGISTRY, NotificationKind


def _build(**overrides):
    values = dict(
        cluster_id="gpu-eks-cluster",
        incident_id="inc-row-remap",
        workflow_id="workflow-row-remap",
        event_id="gpu-dcgm-node-a-GPU-a-row_remap_failure",
        node_ids=["ip-10-0-43-5"],
        containment_operations=[
            "FREEZE_EVIDENCE",
            "MARK_UNSCHEDULABLE",
            "QUARANTINE",
            "COLLECT_DIAGNOSTIC_BUNDLE",
        ],
        waiting_operations=["VALIDATE_GPU", "ESCALATE_SUPPORT", "RUN_FIELD_DIAGNOSTIC"],
        missing_owner_capabilities=["memoryDiagnostics"],
        blocked_reasons=["no executable owner for memoryDiagnostics"],
        policy_source="NVIDIA_GPU_MEMORY_ERROR_MANAGEMENT",
        official_action="RUN_FIELD_DIAGNOSTIC_FOR_RMA",
        runtime_profile_version="regional-eks-7a2b",
        reasons=[
            "row remap failure; field diagnosis required before any RMA conclusion"
        ],
    )
    values.update(overrides)
    return ContainedOperatorHandoffEmailBuilder().build(**values)


def test_the_notice_states_the_containment_the_missing_owner_and_both_exits():
    notification = _build()

    assert (
        notification.category
        == CONTAINED_OPERATOR_HANDOFF_CATEGORY
        == "OPERATOR_REVIEW"
    )
    assert notification.deduplication_key == (
        "gpu-eks-cluster/inc-row-remap/contained-operator-handoff/workflow-row-remap"
    )
    assert notification.subject == (
        "[GPU FAULT][contained, operator action required] gpu-eks-cluster: ip-10-0-43-5"
    )
    body = notification.body_text
    assert "Steps run: FREEZE_EVIDENCE -> MARK_UNSCHEDULABLE -> QUARANTINE" in body
    assert "cordoned and carry this incident's gpu-fault.io/quarantined" in body
    assert "no workload was stopped, no GPU reset" in body
    assert "  - memoryDiagnostics" in body
    assert (
        "Steps that did not run: VALIDATE_GPU -> ESCALATE_SUPPORT -> RUN_FIELD_DIAGNOSTIC"
        in body
    )
    assert "  - no executable owner for memoryDiagnostics" in body
    assert "--incident-id\n  inc-row-remap --disposition restore --plan" in body
    assert "--close-quarantined" in body
    assert "Runtime Profile: regional-eks-7a2b" in body
    assert "row remap failure; field diagnosis required" in body
    assert body.rstrip().endswith(
        f"Template: {CONTAINED_OPERATOR_HANDOFF_TEMPLATE_VERSION}"
    ), body[-120:]
    assert notification.support_case_draft == ""
    assert notification.evidence_refs == []


def test_the_notice_is_one_per_workflow_and_fills_unknowns():
    first = _build()
    second = _build(reasons=["a later reason"])
    assert first.deduplication_key == second.deduplication_key

    sparse = _build(
        node_ids=[],
        containment_operations=[],
        waiting_operations=[],
        missing_owner_capabilities=[],
        blocked_reasons=[],
        official_action=None,
        runtime_profile_version=None,
        reasons=[],
    )
    assert "gpu-eks-cluster: UNKNOWN" in sparse.subject
    assert "Steps run: NONE" in sparse.body_text
    assert "Official action: NONE" in sparse.body_text
    assert "Runtime Profile: UNKNOWN" in sparse.body_text


def test_the_kind_is_registered_with_its_builder():
    definition = NOTIFICATION_REGISTRY[NotificationKind.CONTAINED_OPERATOR_HANDOFF]
    assert definition.builder_type is ContainedOperatorHandoffEmailBuilder
    assert definition.method == "build"

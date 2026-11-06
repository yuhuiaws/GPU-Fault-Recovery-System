from __future__ import annotations

from gpu_fault.notifications import QuarantinedNodeNoticeEmailBuilder
from gpu_fault.notifications.quarantined_node_notice import (
    QUARANTINED_NODE_NOTICE_CATEGORY,
    QUARANTINED_NODE_NOTICE_TEMPLATE_VERSION,
)
from gpu_fault.notifications.registry import NOTIFICATION_REGISTRY, NotificationKind

SUSPENDED = (
    "workload training/pytorchjob/train stays suspended (stopped by workflow "
    "workflow-bmc): the restore does not restart it; its owner must restart it"
)


def _build(**overrides):
    values = dict(
        cluster_id="gpu-eks-cluster",
        incident_id="inc-bmc",
        workflow_id="workflow-bmc",
        event_id="host-telemetry-1-bmc_critical_sensor-node",
        node_ids=["ip-10-0-43-5"],
        steps_run=["FREEZE_EVIDENCE", "MARK_UNSCHEDULABLE", "QUARANTINE"],
        suspended_workloads=[],
        policy_source="SITE_NODE_HEALTH",
        recommended_action="QUARANTINE",
        runtime_profile_version="regional-eks-7a2b",
        reasons=["BMC reports 2 critical sensors: CPU1 Temp=96.000 degrees C [cr]"],
    )
    values.update(overrides)
    return QuarantinedNodeNoticeEmailBuilder().build(**values)


def test_the_notice_names_the_node_the_finding_the_hold_and_the_exits():
    notification = _build()

    assert (
        notification.category == QUARANTINED_NODE_NOTICE_CATEGORY == "OPERATOR_REVIEW"
    )
    assert QUARANTINED_NODE_NOTICE_TEMPLATE_VERSION == "quarantined-node-notice-zh-v1"
    assert notification.deduplication_key == (
        "gpu-eks-cluster/inc-bmc/quarantined-node-notice"
    )
    assert notification.subject == (
        "[GPU FAULT][quarantined, operator action required] "
        "gpu-eks-cluster: ip-10-0-43-5"
    )
    body = notification.body_text
    assert body.startswith("GPU node quarantined; the repair and the exit need an"), (
        body[:120]
    )
    assert "- Nodes: ip-10-0-43-5" in body
    assert "  - BMC reports 2 critical sensors: CPU1 Temp=96.000 degrees C [cr]" in (
        body
    )
    assert "Steps run: FREEZE_EVIDENCE -> MARK_UNSCHEDULABLE -> QUARANTINE" in body
    assert "cordoned and carry this incident's gpu-fault.io/quarantined" in body
    assert "This workflow opened no\n  support case." in body
    assert "restore does not restart them either; their owner must):\n  - NONE" in (
        body
    )
    assert "workflow SUCCEEDED, incident QUARANTINED" in body
    assert "--incident-id\n  inc-bmc --disposition restore --plan" in body
    assert "--close-quarantined" in body
    assert "Recommended action: QUARANTINE" in body
    assert "Runtime Profile: regional-eks-7a2b" in body
    assert body.rstrip().endswith(
        f"Template: {QUARANTINED_NODE_NOTICE_TEMPLATE_VERSION}"
    ), body[-120:]
    assert notification.support_case_draft == ""
    assert notification.evidence_refs == []


def test_the_notice_is_one_per_incident_and_fills_unknowns():
    first = _build()
    assert (
        first.deduplication_key
        == _build(workflow_id="workflow-rerun").deduplication_key
    )
    assert first.deduplication_key == _build(reasons=["later"]).deduplication_key

    stopped = _build(suspended_workloads=[SUSPENDED])
    assert f"  - {SUSPENDED}" in stopped.body_text

    sparse = _build(
        node_ids=[],
        steps_run=[],
        recommended_action=None,
        runtime_profile_version=None,
        reasons=[],
    )
    assert sparse.subject.endswith("gpu-eks-cluster: UNKNOWN"), sparse.subject
    assert "Steps run: NONE" in sparse.body_text
    assert "Recommended action: NONE" in sparse.body_text
    assert "Runtime Profile: UNKNOWN" in sparse.body_text
    assert "2. Finding\n  - UNKNOWN" in sparse.body_text


def test_the_kind_is_registered_with_its_builder():
    definition = NOTIFICATION_REGISTRY[NotificationKind.QUARANTINED_NODE_NOTICE]
    assert definition.builder_type is QuarantinedNodeNoticeEmailBuilder
    assert definition.method == "build"

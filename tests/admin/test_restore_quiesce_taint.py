"""``--disposition restore`` strips an orphaned quiesce taint of its incident.

Live (gpu-eks-cluster, XID 46 on ip-10-0-41-164): the node ended with the
reset chain's ``gpu-fault.io/quiesce`` NoExecute taint, the support successor's
``gpu-fault.io/quarantined`` taint and the successor's isolation annotation.
``restore`` of the reset's incident judged the node "isolated by another
incident" and skipped it; nothing in the product removed the taint and the
operator did it by hand. The per-node decision now treats a quiesce taint whose
value is the incident's own as isolation by that incident: the node is
restored -- the quiesce taint only, the other incident's cordon stays -- and,
because that restore readmits nothing, it is not validated first (the exporter
the validation would wait on is exactly what the taint evicts).
"""

from __future__ import annotations

from datetime import datetime, timezone

from gpu_fault.adapters.common import (
    ANNOTATION_INCIDENT,
    QUARANTINE_TAINT,
    QUIESCE_TAINT,
    quarantine_taint_value,
)
from gpu_fault.admin import remediation_plans
from gpu_fault.orchestration.validated_restore import validated_restore_steps
from tests.admin.conftest import TEST_OPERATOR_ARN

NOW = datetime(2026, 10, 5, 20, 40, tzinfo=timezone.utc)
RESET = "inc-xid46-reset"
SUPPORT = "inc-support-after-workflow-c1aef7fb"
NODE = "ip-10-0-41-164"


def _node(
    *,
    annotation: str | None,
    quarantine: str | None,
    quiesce: str | None,
    unschedulable: bool = True,
) -> dict:
    taints = []
    if quarantine is not None:
        taints.append(
            {"key": QUARANTINE_TAINT, "value": quarantine, "effect": "NoSchedule"}
        )
    if quiesce is not None:
        taints.append({"key": QUIESCE_TAINT, "value": quiesce, "effect": "NoExecute"})
    return {
        "metadata": {
            "name": NODE,
            "annotations": {ANNOTATION_INCIDENT: annotation} if annotation else {},
        },
        "spec": {"unschedulable": unschedulable, "taints": taints},
    }


def _inspection(state: str = "QUARANTINED") -> dict:
    return {
        "incident": {
            "incident_id": RESET,
            "event_id": "event-xid46",
            "event_type": "XID",
            "cluster_id": "gpu-eks-cluster",
            "node_ids": [NODE],
            "gpu_uuids": ["GPU-a"],
            "policy_version": "610",
            "policy_source": "NVIDIA",
            "fencing_token": 1,
            "workflow_request_id": "wf-reset",
            "state": state,
            "reasons": ["XID 46"],
        },
        "workflow": {
            "request_id": "wf-reset",
            "incident_id": RESET,
            "fencing_token": 1,
            "status": "FAILED",
            "official_steps": [
                {"operation": "MARK_UNSCHEDULABLE", "node_ids": [NODE]},
                {"operation": "QUIESCE_GPU_SERVICES", "node_ids": [NODE]},
                {"operation": "RESET_GPU", "node_ids": [NODE]},
            ],
        },
        "open_workflows": [],
        "blocked_workflows": [],
        "active_observations": [],
        "existing_markers": [],
        "acknowledgement_timeout_seconds": 86400,
    }


def test_a_quiesce_taint_under_another_incidents_cordon_is_this_incidents_to_lift() -> (
    None
):
    node = _node(
        annotation=SUPPORT,
        quarantine=quarantine_taint_value(SUPPORT),
        quiesce=quarantine_taint_value(RESET),
    )

    decision = remediation_plans.node_restore_decision(NODE, node, incident_id=RESET)

    assert decision["decision"] == remediation_plans.DECISION_RESTORE
    assert decision["quiesce_taint_only"] is True
    assert decision["quiesce_taint_value"] == quarantine_taint_value(RESET)
    assert decision["incident_annotation"] == SUPPORT
    assert "strips only that taint" in decision["reason"]
    assert "leaves the owner's isolation in place" in decision["reason"]


def test_another_incidents_quiesce_taint_changes_nothing() -> None:
    node = _node(
        annotation=SUPPORT,
        quarantine=quarantine_taint_value(SUPPORT),
        quiesce=quarantine_taint_value("inc-elsewhere"),
    )

    decision = remediation_plans.node_restore_decision(NODE, node, incident_id=RESET)

    assert decision["decision"] == remediation_plans.DECISION_SKIP
    assert decision["quiesce_taint_only"] is False
    assert "is isolated by incident" in decision["reason"]


def test_an_unannotated_node_with_only_this_incidents_quiesce_taint_is_restored() -> (
    None
):
    node = _node(
        annotation=None, quarantine=None, quiesce=quarantine_taint_value(RESET)
    )

    decision = remediation_plans.node_restore_decision(NODE, node, incident_id=RESET)

    assert decision["decision"] == remediation_plans.DECISION_RESTORE
    assert decision["quiesce_taint_only"] is True
    assert "no isolation annotation" in decision["reason"]


def test_the_incidents_own_cordon_with_its_quiesce_taint_is_a_full_restore() -> None:
    node = _node(
        annotation=RESET,
        quarantine=quarantine_taint_value(RESET),
        quiesce=quarantine_taint_value(RESET),
    )

    decision = remediation_plans.node_restore_decision(NODE, node, incident_id=RESET)

    assert decision["decision"] == remediation_plans.DECISION_RESTORE
    assert decision["quiesce_taint_only"] is False, (
        "the node is readmitted by this restore, so it is validated first"
    )


def test_the_quiesce_only_restore_plan_releases_without_validating() -> None:
    nodes = {
        NODE: _node(
            annotation=SUPPORT,
            quarantine=quarantine_taint_value(SUPPORT),
            quiesce=quarantine_taint_value(RESET),
        )
    }

    plan = remediation_plans.build_restore_plan(
        _inspection(),
        incident_id=RESET,
        operator=TEST_OPERATOR_ARN,
        reference="CHG-Q1",
        live_nodes=nodes,
        now=NOW,
    )

    assert plan.restore_node_ids == (NODE,)
    assert [step["operation"] for step in plan.steps] == ["RESTORE_SCHEDULING"], (
        "VALIDATE_GPU would wait on the exporter the taint evicts, and a node that "
        "stays cordoned by another incident has nothing to prove here"
    )
    assert plan.steps[0]["node_ids"] == [NODE]
    assert plan.steps[0]["execution_owner"] == "gpu-fault-kubernetes-adapter"
    assert plan.node_decisions[0]["quiesce_taint_only"] is True
    assert plan.warnings == ()


def test_release_only_nodes_join_the_restore_step_of_a_validated_restore() -> None:
    steps = validated_restore_steps(
        ["node-a", "node-b"], ["GPU-1"], release_only_node_ids=["node-b"]
    )

    operations = [step.operation.value for step in steps]
    assert operations == [
        "VALIDATE_GPU",
        "VALIDATE_HOST",
        "VALIDATE_FABRIC",
        "RESTORE_SCHEDULING",
    ]
    assert steps[0].node_ids == ["node-a"], "only the readmitted node is validated"
    assert steps[0].gpu_uuids == ["GPU-1"]
    assert steps[-1].node_ids == ["node-a", "node-b"]
    assert (
        validated_restore_steps(
            ["node-b"], ["GPU-1"], release_only_node_ids=["node-b"]
        )[0].gpu_uuids
        == []
    ), "a release-only restore names no GPUs"

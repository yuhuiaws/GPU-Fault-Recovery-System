"""``submit-remediation --disposition restore`` exits a RECOVERED incident that
still isolates a node.

Live on gpu-eks-cluster (GF-REGIONAL-DESTR-001, eks-managed): the RESET_GPU
chain cordoned and tainted the node (MARK_UNSCHEDULABLE), then FAILED when
VERIFY_NO_GPU_CLIENTS ran out its window; a queued RESTART_GPU_DEVICE_PLUGIN
successor of the same incident SUCCEEDED and took the incident to RECOVERED.
The failed chain's isolation never lifted (no RESTORE_SCHEDULING, no
QUARANTINED verdict on the incident) and every ``restore`` was refused "is
RECOVERED, not QUARANTINED or ESCALATED" -- `config spare --declare`, the
token rotation's node wave and every case on the node refused in turn.

The restore now accepts a RECOVERED incident under the ESCALATED rule: every
workflow ended, a node still carrying this incident's taint value or isolation
annotation, the same incident and fencing token; the in-Pod script binds the
inspected state and needs no change.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from gpu_fault.adapters.common import (
    ANNOTATION_FENCING,
    ANNOTATION_INCIDENT,
    ANNOTATION_PREVIOUS_UNSCHEDULABLE,
    quarantine_taint_value,
)
from gpu_fault.admin import remediation_plans
from gpu_fault.admin.bootstrap_common import BootstrapError
from tests.admin.conftest import TEST_OPERATOR_ARN
from tests.admin.test_admin_submit_remediation import (
    NOW,
    QUARANTINED,
    RESTORE_STEPS,
    RestoreHarness,
    _quarantined_inspection,
    _quarantined_nodes,
)

RECOVERED = QUARANTINED
RESET_CHAIN = "workflow-7e52bf99-reset"
PLUGIN_SUCCESSOR = "workflow-8053283e-device-plugin"
PLUGIN_STEPS = (
    "FREEZE_EVIDENCE",
    "RESTART_GPU_DEVICE_PLUGIN",
    "TRIGGER_HEALTH_SNAPSHOT",
    "VALIDATE_GPU",
)


def _recovered_inspection(status: str = "SUCCEEDED", **overrides: object) -> dict:
    """The live record: RECOVERED, pointing at the SUCCEEDED plugin successor."""

    value = _quarantined_inspection()
    value["incident"].update(
        {
            "incident_id": RECOVERED,
            "state": "RECOVERED",
            "workflow_request_id": PLUGIN_SUCCESSOR,
            "fencing_token": 1,
            "reasons": ["node-a: gpu_kubernetes_allocatable_mismatch"],
        }
    )
    value["workflow"] = {
        "request_id": PLUGIN_SUCCESSOR,
        "incident_id": RECOVERED,
        "fencing_token": 1,
        "status": status,
        "predecessor_workflow_id": RESET_CHAIN,
        "official_steps": [
            {"operation": operation, "node_ids": ["node-a"]}
            for operation in PLUGIN_STEPS
        ],
        "completed_operations": list(PLUGIN_STEPS),
    }
    value["blocked_workflows"] = []
    value.update(overrides)
    return value


def _isolated_nodes() -> dict:
    """The node as the failed reset left it: cordon, taint and annotations."""

    return _quarantined_nodes(
        taint_value=quarantine_taint_value(RECOVERED),
        annotations={
            ANNOTATION_INCIDENT: RECOVERED,
            ANNOTATION_FENCING: "1",
            ANNOTATION_PREVIOUS_UNSCHEDULABLE: "false",
        },
    )


def _plan(inspection: dict, nodes: dict | None = None) -> remediation_plans.RestorePlan:
    return remediation_plans.build_restore_plan(
        inspection,
        incident_id=RECOVERED,
        operator=TEST_OPERATOR_ARN,
        reference="CHG-R1",
        live_nodes=_isolated_nodes() if nodes is None else nodes,
        now=NOW,
    )


@pytest.mark.parametrize("status", sorted(remediation_plans.ENDED_WORKFLOW_STATUSES))
def test_restore_accepts_a_recovered_incident_whose_node_is_still_isolated(
    status: str,
) -> None:
    plan = _plan(_recovered_inspection(status))

    assert plan.state == "RECOVERED"
    assert plan.incident_id == RECOVERED and plan.fencing_token == 1, (
        "the restore runs under the same incident and the fencing token the "
        "node's annotation carries"
    )
    assert plan.workflow_request_id == PLUGIN_SUCCESSOR
    assert plan.existing_restore_workflow_id is None
    assert [step["operation"] for step in plan.steps] == RESTORE_STEPS
    assert plan.restore_node_ids == ("node-a",)
    (decision,) = plan.node_decisions
    assert decision["decision"] == "restore"
    assert decision["quarantine_taint_value"] == quarantine_taint_value(RECOVERED)
    assert plan.warnings == ()


def test_restore_of_a_recovered_incident_waits_for_blocked_records() -> None:
    with pytest.raises(BootstrapError) as refused:
        _plan(_recovered_inspection("BLOCKED"))
    message = str(refused.value)
    assert f"RECOVERED with BLOCKED workflow {PLUGIN_SUCCESSOR}" in message, message
    assert "restore of a recovered incident needs every workflow ended" in message


def test_restore_of_a_recovered_incident_still_refuses_an_open_workflow() -> None:
    with pytest.raises(
        BootstrapError, match=f"open workflow {PLUGIN_SUCCESSOR} \\(RUNNING\\)"
    ):
        _plan(_recovered_inspection("RUNNING"))


def test_a_recovered_incident_with_no_isolation_left_is_closed_not_restored() -> None:
    clean = _quarantined_nodes(taint_value=None)
    clean["node-a"]["spec"]["unschedulable"] = False

    with pytest.raises(BootstrapError) as refused:
        _plan(_recovered_inspection(), nodes=clean)
    assert str(refused.value).startswith(
        f"no node of incident {RECOVERED} is still isolated"
    ), str(refused.value)


def test_the_recovered_restore_is_submitted_under_the_inspected_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = RestoreHarness(tmp_path, monkeypatch)
    harness.inspection = _recovered_inspection()
    harness.nodes = _isolated_nodes()

    plan_only = harness.submit(plan_only=True, reference="CHG-R1")
    assert plan_only.plan is not None and plan_only.plan.state == "RECOVERED"
    assert harness.submissions == [], "--plan writes nothing"

    result = harness.submit(reference="CHG-R1")

    (submission,) = harness.submissions
    assert submission["plan"]["state"] == "RECOVERED"
    assert submission["plan"]["restore_node_ids"] == ["node-a"]
    assert submission["plan"]["next_operations"] == RESTORE_STEPS
    assert result.plan is not None and result.plan.fencing_token == 1
    assert "is now ACTION_PENDING" in result.message

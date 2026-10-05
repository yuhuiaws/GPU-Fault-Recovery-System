"""``submit-remediation --disposition restore`` exits an exhausted escalation.

Live on HyperPod (GF-REGIONAL-COLLECT-016): a RESET_GPU drill escalated to a
``workflow-reboot-after-<wf>`` successor that isolated the node and then ended
FAILED. The incident was ESCALATED, every workflow terminal, and the node Ready
on its installer boot id yet still cordoned, tainted
``gpu-fault.io/quarantined=incident-<digest>`` and annotated with the incident
and fencing token. Every operator exit refused: ``restore`` ("not
QUARANTINED"), ``confirm-node-action`` (no BLOCKED workflow), the hardware
dispositions (not waiting on CHECK_MECHANICALS), and ``workflow-reconcile
--close-incident`` would have closed the record while orphaning the isolation.

The restore now accepts an ESCALATED incident whose workflows have all ended,
compiling the same validated restore under the same incident and fencing
token; a BLOCKED record still goes through ``confirm-node-action`` /
``workflow-reconcile`` first, an open workflow is still refused, and the
in-Pod script (which runs on the deployed image) needs no change.
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import pytest

from gpu_fault.adapters.common import (
    ANNOTATION_FENCING,
    ANNOTATION_INCIDENT,
    ANNOTATION_PREVIOUS_UNSCHEDULABLE,
    quarantine_taint_value,
)
from gpu_fault.admin import remediation_plans
from gpu_fault.admin import submit_remediation as module
from gpu_fault.admin.bootstrap_common import BootstrapError
from gpu_fault.admin.submit_remediation import submit_restore as real_submit_restore
from gpu_fault.execution.terminal_state import terminal_incident_state
from gpu_fault.models import IncidentState, WorkflowOperation, WorkflowStatus
from gpu_fault.orchestration.validated_restore import build_validated_restore_workflow
from tests.admin.conftest import TEST_OPERATOR_ARN
from tests.admin.test_admin_submit_remediation import (
    NOW,
    QUARANTINED,
    RESTORE_STEPS,
    RestoreHarness,
    _quarantined_inspection,
    _quarantined_nodes,
)
from tests.admin.test_submit_remediation_mixed_site import EKS_CLUSTER, _mixed_site

ESCALATED = QUARANTINED
SUCCESSOR = "workflow-reboot-after-wf-reset-9"
LADDER_STEPS = (
    "FREEZE_EVIDENCE",
    "MARK_UNSCHEDULABLE",
    "QUARANTINE",
    "RESTART_NODE",
    "VALIDATE_GPU",
    "VALIDATE_HOST",
    "VALIDATE_FABRIC",
    "RESTORE_SCHEDULING",
)


def _escalated_inspection(status: str = "FAILED", **overrides: object) -> dict:
    """The live record: ESCALATED, pointing at a terminal reboot successor."""

    value = _quarantined_inspection()
    value["incident"].update(
        {
            "incident_id": ESCALATED,
            "state": "ESCALATED",
            "workflow_request_id": SUCCESSOR,
            "reasons": ["reset remediation failed; failed operation: RESET_GPU"],
        }
    )
    value["workflow"] = {
        "request_id": SUCCESSOR,
        "incident_id": ESCALATED,
        "fencing_token": 3,
        "status": status,
        "official_steps": [
            {"operation": operation, "node_ids": ["node-a"]}
            for operation in LADDER_STEPS
        ],
        "completed_operations": ["FREEZE_EVIDENCE"],
        "terminal_failure_reason": "MARK_UNSCHEDULABLE failed on one node",
    }
    value["blocked_workflows"] = []
    value.update(overrides)
    return value


def _isolated_nodes() -> dict:
    """The node as the live successor left it: cordon, taint and annotations."""

    return _quarantined_nodes(
        taint_value=quarantine_taint_value(ESCALATED),
        annotations={
            ANNOTATION_INCIDENT: ESCALATED,
            ANNOTATION_FENCING: "3",
            ANNOTATION_PREVIOUS_UNSCHEDULABLE: "false",
        },
    )


def _plan(inspection: dict, nodes: dict | None = None) -> remediation_plans.RestorePlan:
    return remediation_plans.build_restore_plan(
        inspection,
        incident_id=ESCALATED,
        operator=TEST_OPERATOR_ARN,
        reference="CHG-E1",
        live_nodes=_isolated_nodes() if nodes is None else nodes,
        now=NOW,
    )


@pytest.mark.parametrize("status", sorted(remediation_plans.ENDED_WORKFLOW_STATUSES))
def test_restore_accepts_an_escalated_incident_once_every_workflow_has_ended(
    status: str,
) -> None:
    plan = _plan(_escalated_inspection(status))

    assert plan.state == "ESCALATED"
    assert plan.incident_id == ESCALATED and plan.fencing_token == 3, (
        "the restore runs under the same incident and the fencing token the "
        "node's annotation carries"
    )
    assert plan.workflow_request_id == SUCCESSOR
    assert plan.existing_restore_workflow_id is None
    assert [step["operation"] for step in plan.steps] == RESTORE_STEPS
    assert plan.restore_node_ids == ("node-a",)
    (decision,) = plan.node_decisions
    assert decision["decision"] == "restore"
    assert decision["quarantine_taint_value"] == quarantine_taint_value(ESCALATED)
    assert plan.warnings == ()
    assert plan.as_dict()["state"] == "ESCALATED"


def test_restore_of_an_escalated_incident_waits_for_blocked_records() -> None:
    parked = _escalated_inspection("BLOCKED")
    with pytest.raises(BootstrapError) as refused:
        _plan(parked)
    message = str(refused.value)
    assert f"BLOCKED workflow {SUCCESSOR}" in message, message
    assert "confirm-node-action" in message and "workflow-reconcile" in message
    assert "FAILED/SUCCEEDED/SUPERSEDED" in message

    sibling = _escalated_inspection(
        "FAILED",
        blocked_workflows=[
            {
                "request_id": "wf-dag-branch",
                "status": "BLOCKED",
                "incident_id": ESCALATED,
            }
        ],
    )
    with pytest.raises(BootstrapError, match="BLOCKED workflow wf-dag-branch"):
        _plan(sibling)


def test_restore_of_an_escalated_incident_still_refuses_an_open_workflow() -> None:
    with pytest.raises(
        BootstrapError, match=f"open workflow {SUCCESSOR} \\(RUNNING\\)"
    ):
        _plan(_escalated_inspection("RUNNING"))

    with pytest.raises(BootstrapError, match="wf-replace \\(SAFETY_PENDING\\)"):
        _plan(
            _escalated_inspection(
                "FAILED",
                open_workflows=[
                    {"request_id": "wf-replace", "status": "SAFETY_PENDING"}
                ],
            )
        )


def test_restore_refuses_the_planning_states_by_name() -> None:
    inspection = _escalated_inspection("RUNNING")
    inspection["incident"]["state"] = "ACTION_PENDING"

    with pytest.raises(
        BootstrapError,
        match="is ACTION_PENDING, not QUARANTINED, ESCALATED or RECOVERED",
    ):
        _plan(inspection)


def test_an_escalated_incident_whose_isolation_is_gone_is_closed_not_restored() -> None:
    clean = _quarantined_nodes(taint_value=None)
    clean["node-a"]["spec"]["unschedulable"] = False

    with pytest.raises(BootstrapError) as refused:
        _plan(_escalated_inspection("FAILED"), nodes=clean)
    message = str(refused.value)
    assert message.startswith(f"no node of incident {ESCALATED} is still isolated"), (
        message
    )
    assert f"--close-incident {ESCALATED}" in message, (
        "with nothing left to release, closing the record is the right exit"
    )


def test_the_escalated_restore_is_submitted_under_the_inspected_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = RestoreHarness(tmp_path, monkeypatch)
    harness.inspection = _escalated_inspection("FAILED")
    harness.nodes = _isolated_nodes()

    plan_only = harness.submit(plan_only=True, reference="CHG-E1")
    assert plan_only.plan is not None and plan_only.plan.state == "ESCALATED"
    assert harness.submissions == [], "--plan writes nothing"

    result = harness.submit(reference="CHG-E1")

    (submission,) = harness.submissions
    assert submission["plan"]["state"] == "ESCALATED"
    assert submission["plan"]["restore_node_ids"] == ["node-a"]
    assert submission["plan"]["next_operations"] == RESTORE_STEPS
    assert result.plan is not None and result.plan.fencing_token == 3
    assert "is now ACTION_PENDING" in result.message
    assert [path.name.startswith("restore-") for path in harness.evidence()] == [True]


def test_the_escalated_restore_payload_binds_state_and_the_clusters_profile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Mixed site: the eks-managed cluster's restore carries ``eks-v1`` and the
    ESCALATED state it inspected, so the in-Pod drift check compares the
    record against what the operator saw."""

    harness = RestoreHarness(tmp_path, monkeypatch)
    _mixed_site(harness.site)
    payloads: list[dict] = []

    def command(site, payload, *, script):
        payloads.append(json.loads(json.dumps(payload)))
        return {"no_op": False, "workflow": {"request_id": "wf-r"}, "incident": {}}

    monkeypatch.setattr(module, "run_control_plane_script", command)
    for cluster_id, version in ((EKS_CLUSTER, "eks-v1"), ("gpu-a", "hyperpod-v1")):
        inspection = _escalated_inspection("SUPERSEDED")
        inspection["incident"]["cluster_id"] = cluster_id
        plan = _plan(inspection)
        real_submit_restore(harness.site, plan, operator="ops@example", reference="CHG")
        payload = payloads[-1]
        assert payload["mode"] == "restore"
        assert payload["expected"] == {
            "fencing_token": 3,
            "workflow_request_id": SUCCESSOR,
            "state": "ESCALATED",
            "node_ids": ["node-a"],
        }
        assert payload["node_ids"] == ["node-a"]
        assert payload["runtime_profile_version"] == version, cluster_id


def test_the_in_pod_restore_mode_accepts_the_escalated_state_it_inspected(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The deployed script re-checks drift and open workflows, not the state:
    an ESCALATED incident with a FAILED workflow gets its restore pair."""

    from gpu_fault.app import ApplicationContext
    from tests._builders import build_store
    from tests.orchestration._incident_closure_support import _escalated_reset

    store = build_store()
    incident, failed = _escalated_reset(store, incident_id="inc-ladder")
    assert incident.state is IncidentState.ESCALATED
    assert failed.status is WorkflowStatus.FAILED
    context = ApplicationContext(store=store)
    woken: list[bool] = []
    monkeypatch.setattr(context.dispatcher, "wake", lambda: woken.append(True))
    monkeypatch.setattr(
        ApplicationContext, "from_environment", classmethod(lambda cls: context)
    )
    monkeypatch.setattr(
        sys,
        "stdin",
        io.StringIO(
            json.dumps(
                {
                    "mode": "restore",
                    "incident_id": "inc-ladder",
                    "expected": {
                        "fencing_token": incident.fencing_token,
                        "workflow_request_id": failed.request_id,
                        "state": "ESCALATED",
                        "node_ids": ["node-a"],
                    },
                    "node_ids": ["node-a"],
                    "operator": "ops@example",
                    "reference": "CHG-E1",
                    "runtime_profile_version": "hyperpod-v1",
                }
            )
        ),
    )

    exec(compile(module.REMEDIATION_SCRIPT, "<submit-remediation>", "exec"), {})
    result = json.loads(capsys.readouterr().out)

    assert result["no_op"] is False
    restore = store.get_workflow(result["workflow"]["request_id"])
    assert restore.request_id.startswith("workflow-validated-restore-"), (
        restore.request_id
    )
    assert restore.status is WorkflowStatus.PENDING
    assert restore.fencing_token == incident.fencing_token
    assert [step.operation.value for step in restore.official_steps] == RESTORE_STEPS
    stored = store.get_incident("inc-ladder")
    assert stored.state is IncidentState.ACTION_PENDING
    assert stored.workflow_request_id == restore.request_id
    assert stored.reasons[-1] == "operator restore requested by ops@example (CHG-E1)"
    assert woken == [True]


def test_a_succeeded_restore_of_an_escalated_incident_ends_recovered() -> None:
    """The executor's terminal rule needs no special case: a validated restore
    that released scheduling leaves no isolation, so the incident that was
    ESCALATED ends RECOVERED (and its markers retire with it)."""

    from tests._builders import build_store
    from tests.orchestration._incident_closure_support import _escalated_reset

    incident, _ = _escalated_reset(build_store(), incident_id="inc-ladder")
    pending, restore = build_validated_restore_workflow(
        incident, operator="ops@example", reference="CHG-E1", now=NOW
    )
    assert pending.state is IncidentState.ACTION_PENDING
    finished = restore.model_copy(
        update={
            "completed_step_indexes": list(range(len(restore.official_steps))),
            "completed_operations": [step.operation for step in restore.official_steps],
        }
    )
    assert WorkflowOperation.RESTORE_SCHEDULING in finished.completed_operations

    assert (
        terminal_incident_state(finished, WorkflowStatus.SUCCEEDED)
        is IncidentState.RECOVERED
    )
    assert (
        terminal_incident_state(finished, WorkflowStatus.FAILED)
        is IncidentState.ESCALATED
    ), "a failed validation hands the node back to the operator, isolation intact"

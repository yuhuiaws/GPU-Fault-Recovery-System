"""``workflow-reconcile`` for a parked record whose node action an operator confirmed.

Live (gpu-eks-cluster, XID 46): the RESET_GPU record parked BLOCKED with an
unknown outcome, the node rebooted and its GPU validated by a later restore,
and ``confirm-node-action`` recorded the operator's confirmation -- yet the
record could end only with the release it blocked: the deployed planner knew
no such shape, the deployed incident close refused the open record, and
``gpu-fault-admin config`` re-rolls the deployed release from its own source
snapshot, so its gate was the deployed probe. The shape is planned by the
CLI-shipped script, proven on node evidence by the CLI and applied through a
bridge with primitives every image has.
"""

from __future__ import annotations

import io
import json
import runpy
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from gpu_fault.admin import workflow_reconcile as reconcile
from gpu_fault.admin import workflow_reconcile_operator_confirmed as confirmed
from gpu_fault.admin.bootstrap_common import BootstrapError
from gpu_fault.workflow_reconcile import NODE_RESTORE_PROOF_REASON, OPERATOR_CONFIRMED
from tests.admin.conftest import TEST_OPERATOR_ARN

NODE = "ip-10-0-41-164.us-west-2.compute.internal"
RECORD = "workflow-ec95fa59"
INCIDENT = "inc-xid-46"
RESTORE_ATTEMPT = f"{RECORD}-restore-attempt"
PLAN_ID = "plan-reset-xid-46"
REFERENCE = "CHG-2026-1006-02"
CONFIRMING_ACTOR = "arn:aws:sts::123456789012:assumed-role/Admin/ops"
CONFIRMATION = {
    "actor": CONFIRMING_ACTOR,
    "reference": "CHG-2026-1006-01",
    "confirmed_at": "2026-10-06T01:30:49+00:00",
    "node_id": NODE,
    "operation": "RESET_GPU",
    "settled_by": "reboot_since_action",
    "previous_boot_id": "05aac614-57e4-4b3e-910a-4d4785a63e57",
    "agent": {"boot_id": "4300ad45-7b50-4a2c-9dad-6fbe0b7bc21c"},
}
BOUND_CONFIRMATIONS = [
    {
        "step_index": 1,
        "operation": "RESET_GPU",
        "phase": "official",
        "node_id": NODE,
        "actor": CONFIRMING_ACTOR,
        "reference": "CHG-2026-1006-01",
        "confirmed_at": "2026-10-06T01:30:49+00:00",
        "settled_by": "reboot_since_action",
    }
]
PROBE = Path(__file__).resolve().parents[2] / (
    "deploy/control-plane/regional/probes/workflow_safety.py"
)


# ------------------------------------------------------------------ fixtures


def _control_plane(now: datetime) -> Any:
    """An in-memory control plane with the live record and its neighbours."""

    from gpu_fault.models import (
        BlockedKind,
        IncidentState,
        PlanStep,
        RecoveryAction,
        RecoveryPlan,
        WorkflowOperation,
        WorkflowStatus,
        WorkflowStepStatus,
    )
    from tests._builders import (
        build_store,
        fault_incident,
        workflow_request,
        workflow_step,
        workflow_step_execution,
    )

    store = build_store()
    quiesce = WorkflowOperation.QUIESCE_GPU_SERVICES
    reset = WorkflowOperation.RESET_GPU
    restore = WorkflowOperation.RESTORE_GPU_SERVICES

    def parked(
        request_id: str,
        incident_id: str,
        *,
        reset_details: dict[str, Any],
        incident_state: IncidentState = IncidentState.ESCALATED,
        node_ids: list[str] | None = None,
        **values: Any,
    ) -> None:
        nodes = node_ids or [NODE]
        fields: dict[str, Any] = {
            "status": WorkflowStatus.BLOCKED,
            "blocked_kind": BlockedKind.NEEDS_OPERATOR,
            "fencing_token": 2,
            "execution_epoch": 4,
            "official_steps": [
                workflow_step(quiesce, node_ids=nodes),
                workflow_step(reset, node_ids=nodes),
                workflow_step(restore, node_ids=nodes),
            ],
            "completed_step_indexes": [0],
            "completed_operations": [quiesce],
            "step_executions": [
                workflow_step_execution(0, quiesce, phase="official"),
                workflow_step_execution(
                    1,
                    reset,
                    WorkflowStepStatus.FAILED,
                    phase="official",
                    details=reset_details,
                ),
                workflow_step_execution(
                    2,
                    restore,
                    WorkflowStepStatus.FAILED,
                    phase="official",
                    details={"restore_gpu_services_withheld": True},
                ),
            ],
            "remediation_budget_claims": ["claim-reset"],
            "created_at": now - timedelta(hours=3),
            "updated_at": now - timedelta(hours=1),
            **values,
        }
        store.save_workflow(workflow_request(request_id, incident_id, **fields))
        store.save_workflow(
            workflow_request(
                f"{request_id}-restore-attempt",
                incident_id,
                status=WorkflowStatus.FAILED,
                fencing_token=2,
                official_steps=[workflow_step(restore, node_ids=nodes)],
            )
        )
        store.save_incident(
            fault_incident(
                incident_id,
                f"event-{incident_id}",
                cluster_id="gpu-eks-cluster",
                node_ids=nodes,
                state=incident_state,
                workflow_request_id=f"{request_id}-restore-attempt",
                fencing_token=2,
            )
        )

    confirmed_details = {
        "outcome_unknown": False,
        "manual_confirmation_required": False,
        "node_action_state": "OPERATOR_CONFIRMED",
        "operator_confirmed": CONFIRMATION,
    }
    parked(RECORD, INCIDENT, reset_details=confirmed_details, source_plan_id=PLAN_ID)
    store.save_plan(
        RecoveryPlan(
            plan_id=PLAN_ID,
            incident_id=INCIDENT,
            attempt_id="attempt-1",
            trigger="XID 46",
            runtime_profile_version="eks-v1",
            steps=[PlanStep(action=RecoveryAction.RESET_GPU, execution_owner="agent")],
            workflow_request_id=RECORD,
        )
    )
    parked(
        "workflow-unconfirmed",
        "inc-unconfirmed",
        reset_details={"outcome_unknown": True, "manual_confirmation_required": True},
    )
    parked(
        "workflow-pending-incident",
        "inc-pending",
        reset_details=confirmed_details,
        incident_state=IncidentState.ACTION_PENDING,
    )
    parked(
        "workflow-owned",
        "inc-owned",
        reset_details=confirmed_details,
        execution_owner_id="executor-1",
    )
    parked(
        "workflow-other-node",
        "inc-other-node",
        reset_details={
            **confirmed_details,
            "operator_confirmed": {**CONFIRMATION, "node_id": "node-elsewhere"},
        },
    )
    return store


def _exec_script(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    store: Any,
    payload: dict[str, Any],
) -> dict[str, Any]:
    from gpu_fault.app import ApplicationContext

    context = ApplicationContext(store=store)
    monkeypatch.setattr(
        ApplicationContext, "from_environment", classmethod(lambda cls: context)
    )
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(payload)))
    exec(compile(reconcile.RECONCILE_SCRIPT, "<workflow-reconcile>", "exec"), {})
    return json.loads(capsys.readouterr().out)


def _plan_item(plan: dict[str, Any], request_id: str) -> dict[str, Any]:
    [item] = [item for item in plan["items"] if item["request_id"] == request_id]
    return item


def _site(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        release_config={
            "site_name": "staging",
            "aws_region": "us-west-2",
            "cpu_eks_arn": "arn:aws:eks:us-west-2:123456789012:cluster/cpu-control",
            "clusters": [{"cluster_id": "gpu-eks-cluster", "context": "gpu-eks"}],
        },
        environment={"KUBECONFIG": str(tmp_path / "gpu.kubeconfig")},
        source_sha256="a" * 64,
    )


def _confirmed_item(**overrides: object) -> dict[str, object]:
    """The Pod's operator-confirmed item for the live record."""

    item: dict[str, object] = {
        "request_id": RECORD,
        "incident_id": INCIDENT,
        "cluster_id": "gpu-eks-cluster",
        "node_ids": [NODE],
        "incident_state": "ESCALATED",
        "blocked_kind": "NEEDS_OPERATOR",
        "fencing_token": 2,
        "execution_epoch": 4,
        "workflow_updated_at": "2026-10-06T00:30:00+00:00",
        "source_plan_id": PLAN_ID,
        "source_plan_status": "PENDING",
        "open_remote_commands": [],
        "confirmations": [dict(entry) for entry in BOUND_CONFIRMATIONS],
        "terminalization": OPERATOR_CONFIRMED,
        "eligible": False,
        "reasons": [NODE_RESTORE_PROOF_REASON],
    }
    item.update(overrides)
    return item


def _runtime_plan(*items: dict[str, object]) -> dict[str, object]:
    return {
        "schema_version": 1,
        "mode": "workflow-reconcile-plan",
        "evaluated_at": "2026-10-06T02:00:00+00:00",
        "plan_sha256": "b" * 64,
        "discovery": None,
        "never_dispatched_discovery": None,
        "items": list(items) or [_confirmed_item()],
    }


def _kubectl(nodes: list[dict[str, Any]]) -> Any:
    """``run_command`` double: the GPU cluster's node list."""

    def run(command: list[str], **_options: object) -> SimpleNamespace:
        assert command[-4:] == ["get", "nodes", "-o", "json"], command
        return SimpleNamespace(
            returncode=0, stdout=json.dumps({"items": nodes}), stderr=""
        )

    return run


def _node(
    *, unschedulable: bool = False, taints: list[dict[str, str]] | None = None
) -> dict[str, Any]:
    return {
        "metadata": {"name": NODE, "annotations": {}},
        "spec": {"unschedulable": unschedulable, "taints": taints or []},
    }


def _pod(plans: list[dict[str, object]], calls: list[dict[str, object]]) -> Any:
    queue = list(plans)

    def run(_site: object, payload: dict[str, object], **_kwargs: object) -> Any:
        calls.append(payload)
        if payload["mode"] == "plan":
            return queue.pop(0) if queue else _runtime_plan()
        return {
            "mode": "workflow-reconcile-apply",
            "applied_workflow_ids": [
                str(item["request_id"]) for item in payload.get("items") or []
            ],
            "failed_workflow_ids": [],
            "failures": {},
            "resolved_plan_ids": [PLAN_ID],
            "archive_eligible_incident_ids": [INCIDENT],
            "records_deleted": 0,
        }

    return run


# ------------------------------------------------------------- the Pod plan


def test_the_pod_plan_takes_the_operator_confirmed_verdict(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A BLOCKED record the deployed planner refused (no verified restore
    successor, a node-mutating operation completed) takes the operator-confirmed
    verdict when it carries a confirmation -- with the node proof as its one
    missing reason and the confirmations bound in the item; a record without a
    confirmation keeps the deployed verdict; every other gate names itself."""

    store = _control_plane(datetime.now(timezone.utc))

    explicit = _exec_script(
        monkeypatch,
        capsys,
        store,
        {
            "mode": "plan",
            "workflow_ids": [
                RECORD,
                "workflow-unconfirmed",
                "workflow-pending-incident",
                "workflow-owned",
                "workflow-other-node",
                "workflow-missing",
            ],
        },
    )

    item = _plan_item(explicit, RECORD)
    assert item["terminalization"] == OPERATOR_CONFIRMED
    assert item["eligible"] is False
    assert item["reasons"] == [NODE_RESTORE_PROOF_REASON]
    assert item["confirmations"] == BOUND_CONFIRMATIONS
    assert item["source_plan_id"] == PLAN_ID
    assert item["source_plan_status"] == "PENDING"
    assert item["blocked_kind"] == "NEEDS_OPERATOR"
    assert item["node_ids"] == [NODE]
    assert item["incident_state"] == "ESCALATED"
    assert item["open_remote_commands"] == []
    assert item["fencing_token"] == 2 and item["execution_epoch"] == 4

    unconfirmed = _plan_item(explicit, "workflow-unconfirmed")
    assert unconfirmed["terminalization"] != OPERATOR_CONFIRMED
    assert "workflow has no verified restore successor" in unconfirmed["reasons"]
    assert "confirmations" not in unconfirmed, "the deployed verdict stands"

    pending = _plan_item(explicit, "workflow-pending-incident")
    assert pending["terminalization"] == OPERATOR_CONFIRMED
    assert pending["reasons"] == [
        "incident is ACTION_PENDING, still waiting on a workflow",
        NODE_RESTORE_PROOF_REASON,
    ]
    assert (
        "workflow still has an execution owner"
        in (_plan_item(explicit, "workflow-owned")["reasons"])
    )
    assert (
        "confirmation names a node outside the incident: node-elsewhere"
        in _plan_item(explicit, "workflow-other-node")["reasons"]
    )
    assert _plan_item(explicit, "workflow-missing")["reasons"] == [
        "workflow does not exist"
    ]
    assert len(explicit["plan_sha256"]) == 64

    discovered = _exec_script(
        monkeypatch, capsys, store, {"mode": "plan", "workflow_ids": []}
    )
    assert _plan_item(discovered, RECORD)["terminalization"] == OPERATOR_CONFIRMED, (
        "the deployed BLOCKED discovery already lists the record; it takes the verdict"
    )
    assert discovered["discovery"]["selected"] == 5


def test_the_fallback_rule_and_planner_match_the_module(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Against an image without the shape the script plans it itself; the two
    copies must say the same thing, record for record."""

    import gpu_fault.workflow_reconcile as pod_module

    store = _control_plane(datetime.now(timezone.utc))
    payload = {"mode": "plan", "workflow_ids": []}

    with_module = _exec_script(monkeypatch, capsys, store, payload)
    monkeypatch.delattr(pod_module, "operator_confirmed_plan_items")
    monkeypatch.delattr(pod_module, "operator_confirmed_node_actions")
    monkeypatch.delattr(pod_module, "operator_confirmed_reconciliation_reasons")
    fallback = _exec_script(monkeypatch, capsys, store, payload)

    assert fallback["items"] == with_module["items"]
    assert fallback["plan_sha256"] == with_module["plan_sha256"]
    assert [
        item["request_id"]
        for item in with_module["items"]
        if item["terminalization"] == OPERATOR_CONFIRMED
    ] == [
        RECORD,
        "workflow-other-node",
        "workflow-owned",
        "workflow-pending-incident",
    ], "every confirmed record takes the verdict, eligible or not"


# ----------------------------------------------------- the admin-side evidence


def test_the_admin_plan_promotes_the_record_on_a_clean_node(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(reconcile, "_run_reconcile", _pod([], []))
    monkeypatch.setattr(reconcile, "run_command", _kubectl([_node()]))

    plan = reconcile.run_workflow_reconcile(_site(tmp_path), tmp_path, dry_run=True)

    item = _plan_item(plan, RECORD)
    assert item["eligible"] is True, item
    assert item["reasons"] == []
    assert item["terminalization"] == OPERATOR_CONFIRMED
    assert item["scheduling_evidence"]["restored"] is True
    [node] = item["scheduling_evidence"]["nodes"]
    assert node["node_id"] == NODE and node["exists"] is True
    assert plan["dry_run"] is True


@pytest.mark.parametrize(
    "node",
    [
        _node(unschedulable=True),
        _node(taints=[{"key": reconcile.QUARANTINE_TAINT, "value": INCIDENT}]),
        _node(taints=[{"key": reconcile.QUIESCE_TAINT, "value": "held"}]),
    ],
    ids=["cordoned", "quarantine-taint", "quiesce-taint"],
)
def test_a_node_still_isolated_keeps_the_record_ineligible(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, node: dict[str, Any]
) -> None:
    monkeypatch.setattr(reconcile, "_run_reconcile", _pod([], []))
    monkeypatch.setattr(reconcile, "run_command", _kubectl([node]))

    plan = reconcile.run_workflow_reconcile(_site(tmp_path), tmp_path, dry_run=True)

    item = _plan_item(plan, RECORD)
    assert item["eligible"] is False
    assert item["reasons"] == [
        NODE_RESTORE_PROOF_REASON,
        "GPU node scheduling state has not been restored",
    ]
    with pytest.raises(BootstrapError, match="refuses ineligible records"):
        reconcile.run_workflow_reconcile(
            _site(tmp_path), tmp_path, workflow_ids=(RECORD,), reference=REFERENCE
        )


def test_a_departed_node_is_judged_by_the_provider_like_a_never_changed_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The record's node may have been replaced since the confirmation; the
    departed-node proof the never-changed shape takes applies to it too."""

    assert OPERATOR_CONFIRMED in reconcile.DEPARTED_NODE_TERMINALIZATIONS
    seen: list[tuple[str, str]] = []

    def departed(
        _site: object, cluster_id: str, node_id: str, *_args: object, **_kw: object
    ) -> dict[str, Any]:
        seen.append((cluster_id, node_id))
        return {
            "node_id": node_id,
            "exists": False,
            "provider": "eks-managed",
            "absent_from_provider": True,
            "restored": True,
            "blockers": [],
        }

    monkeypatch.setattr(reconcile, "_run_reconcile", _pod([], []))
    monkeypatch.setattr(reconcile, "run_command", _kubectl([]))
    monkeypatch.setattr(reconcile, "_departed_node_evidence", departed)
    monkeypatch.setattr(
        reconcile.workflow_reconcile_departed_nodes,
        "RecordedInstanceIds",
        lambda *_args, **_kw: SimpleNamespace(get=lambda *_a: "i-00000000000000001"),
    )

    plan = reconcile.run_workflow_reconcile(_site(tmp_path), tmp_path, dry_run=True)

    assert seen == [("gpu-eks-cluster", NODE)]
    item = _plan_item(plan, RECORD)
    assert item["eligible"] is True, item
    [node] = item["scheduling_evidence"]["nodes"]
    assert node["absent_from_provider"] is True


@pytest.mark.parametrize(
    "changes",
    [
        {"reasons": [NODE_RESTORE_PROOF_REASON, "incident is missing"]},
        {"reasons": []},
        {"terminalization": "never-changed"},
        {"eligible": True},
        {"scheduling_evidence": {"restored": True, "nodes": []}},
        {"scheduling_evidence": {"restored": False, "nodes": [{"restored": True}]}},
        {"scheduling_evidence": {"restored": True, "nodes": [{"restored": False}]}},
    ],
)
def test_only_the_exact_confirmed_shape_is_promoted(changes: dict[str, object]) -> None:
    item = _confirmed_item(
        scheduling_evidence={"restored": True, "nodes": [{"restored": True}]}
    )
    before = dict(item)
    item.update(changes)
    expected_eligible = bool(item.get("eligible"))

    confirmed.promote_operator_confirmed(item)

    assert bool(item["eligible"]) is expected_eligible, (before, changes, item)

    exact = _confirmed_item(
        scheduling_evidence={"restored": True, "nodes": [{"restored": True}]}
    )
    confirmed.promote_operator_confirmed(exact)
    assert exact["eligible"] is True and exact["reasons"] == []


# ------------------------------------------------------------------ the apply


def test_the_apply_goes_through_the_operator_confirmed_bridge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(reconcile, "_run_reconcile", _pod([], calls))
    monkeypatch.setattr(reconcile, "run_command", _kubectl([_node()]))

    result = reconcile.run_workflow_reconcile(
        _site(tmp_path), tmp_path, workflow_ids=(RECORD,), reference=REFERENCE
    )

    assert [call["mode"] for call in calls] == ["plan", "plan", confirmed.BRIDGE_MODE]
    bridge = calls[-1]
    [entry] = bridge["items"]
    assert entry["request_id"] == RECORD
    assert entry["fencing_token"] == 2 and entry["execution_epoch"] == 4
    assert entry["confirmations"] == BOUND_CONFIRMATIONS
    assert entry["node_evidence"] == [
        {
            "node_id": NODE,
            "exists": True,
            "restored": True,
            "unschedulable": False,
            "quarantine_taint": False,
            "isolation_annotation_count": 0,
            "provider": None,
            "absent_from_provider": None,
        }
    ]
    assert bridge["reference"] == REFERENCE
    assert bridge["actor"] == TEST_OPERATOR_ARN
    assert bridge["admin_plan_sha256"] == result["plan_sha256"]
    assert result["applied_workflow_ids"] == [RECORD]
    assert result["resolved_plan_ids"] == [PLAN_ID]
    archive = tmp_path / reconcile.HISTORY_PATH / str(result["plan_sha256"])
    plan = json.loads((archive / "plan.json").read_text(encoding="utf-8"))
    assert plan["items"][0]["confirmations"] == BOUND_CONFIRMATIONS, (
        "the archive keeps the attestations the approval bound"
    )


def _entry(store: Any, request_id: str, **overrides: Any) -> dict[str, Any]:
    workflow = store.get_workflow(request_id)
    values: dict[str, Any] = {
        "request_id": request_id,
        "fencing_token": workflow.fencing_token,
        "execution_epoch": workflow.execution_epoch,
        "confirmations": [dict(entry) for entry in BOUND_CONFIRMATIONS],
        "node_evidence": [
            {
                "node_id": NODE,
                "exists": True,
                "restored": True,
                "unschedulable": False,
                "quarantine_taint": False,
                "isolation_annotation_count": 0,
                "provider": None,
                "absent_from_provider": None,
            }
        ],
    }
    values.update(overrides)
    return values


def test_the_bridge_ends_the_record_and_leaves_the_incident_to_the_operator(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """``apply-operator-confirmed`` against an in-memory control plane: the rule
    re-derived minus the proof the evidence answers, the compare-and-set on
    fencing token, epoch and the confirmations themselves, the evidence bound to
    the incident's node set, ``amend_workflow`` with the audit event, the source
    plan's reconciliation reference, the incident's audit line with its state
    kept -- and afterwards the deployed close accepts the ESCALATED incident on
    authority and the release gate sees no record at all."""

    from gpu_fault.app import ApplicationContext
    from gpu_fault.models import IncidentState, WorkflowEventKind, WorkflowStatus
    from gpu_fault.orchestration.incident_closure import IncidentClosureService

    now = datetime.now(timezone.utc)
    store = _control_plane(now)

    result = _exec_script(
        monkeypatch,
        capsys,
        store,
        {
            "mode": confirmed.BRIDGE_MODE,
            "items": [
                _entry(store, RECORD),
                _entry(store, "workflow-unconfirmed"),
                _entry(store, "workflow-pending-incident"),
                _entry(store, "workflow-owned"),
                _entry(store, "workflow-other-node"),
            ],
            "reference": REFERENCE,
            "actor": TEST_OPERATOR_ARN,
            "admin_plan_sha256": "d" * 64,
        },
    )

    assert result["applied_workflow_ids"] == [RECORD]
    assert result["failed_workflow_ids"] == [
        "workflow-other-node",
        "workflow-owned",
        "workflow-pending-incident",
        "workflow-unconfirmed",
    ]
    assert (
        "no node action of the record carries an operator confirmation"
        in (result["failures"]["workflow-unconfirmed"])
    )
    assert (
        "incident is ACTION_PENDING" in result["failures"]["workflow-pending-incident"]
    )
    assert "execution owner" in result["failures"]["workflow-owned"]
    assert "outside the incident" in result["failures"]["workflow-other-node"]
    assert result["resolved_plan_ids"] == [PLAN_ID]
    assert result["archive_eligible_incident_ids"] == [INCIDENT]
    assert result["records_deleted"] == 0

    closed = store.get_workflow(RECORD)
    assert closed.status is WorkflowStatus.SUPERSEDED
    assert closed.preempted_by_workflow_id is None
    assert closed.remediation_budget_claims == []
    assert "whose node action the operator confirmed settled" in (
        closed.preemption_reason or ""
    )
    assert "(step 1 RESET_GPU (reboot_since_action))" in (
        closed.preemption_reason or ""
    )
    event = closed.events[-1]
    assert event.kind is WorkflowEventKind.OPERATOR_RECONCILED
    assert event.actor == TEST_OPERATOR_ARN
    assert event.details["terminalization"] == OPERATOR_CONFIRMED
    assert event.details["previous_status"] == "BLOCKED"
    assert event.details["admin_plan_sha256"] == "d" * 64
    assert event.details["operator_confirmations"] == BOUND_CONFIRMATIONS
    assert event.details["node_evidence"][0]["node_id"] == NODE
    assert event.details["expected_fencing_token"] == 2

    plan = store.get_plan(PLAN_ID)
    assert plan.reconciliation_reference == REFERENCE
    assert plan.reconciled_at is not None
    assert plan.resolved_by_restore_workflow_id is None

    incident = store.get_incident(INCIDENT)
    assert incident.state is IncidentState.ESCALATED, "the state is the operator's"
    assert incident.reasons[-1].startswith(
        f"operator reconciliation {REFERENCE}: closed"
    ), incident.reasons
    assert incident.workflow_request_id == RESTORE_ATTEMPT, (
        "the pointer is not rewritten"
    )
    preview = IncidentClosureService(store).preview(INCIDENT)
    assert preview["closable"] is True, preview
    assert preview["open_workflow_id"] is None

    for request_id in (
        "workflow-unconfirmed",
        "workflow-pending-incident",
        "workflow-owned",
        "workflow-other-node",
    ):
        assert store.get_workflow(request_id).status is WorkflowStatus.BLOCKED

    monkeypatch.setattr(
        ApplicationContext,
        "from_environment",
        staticmethod(lambda: SimpleNamespace(store=store)),
    )
    runpy.run_path(str(PROBE))
    report = json.loads(capsys.readouterr().out)
    assert RECORD not in report["blockers"], report
    assert RECORD not in report["operator_confirmed_parked"], (
        "terminal: not even set aside"
    )


def test_the_bridge_refuses_a_record_that_moved_under_the_approval(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from gpu_fault.models import WorkflowStatus

    now = datetime.now(timezone.utc)
    store = _control_plane(now)
    rewritten = [{**BOUND_CONFIRMATIONS[0], "reference": "CHG-forged"}]

    def apply(entry: dict[str, Any]) -> dict[str, Any]:
        result = _exec_script(
            monkeypatch,
            capsys,
            store,
            {
                "mode": confirmed.BRIDGE_MODE,
                "items": [entry],
                "reference": REFERENCE,
                "actor": TEST_OPERATOR_ARN,
                "admin_plan_sha256": "d" * 64,
            },
        )
        assert result["applied_workflow_ids"] == []
        assert store.get_workflow(RECORD).status is WorkflowStatus.BLOCKED
        return dict(result["failures"])

    assert (
        "fencing token changed" in apply(_entry(store, RECORD, fencing_token=3))[RECORD]
    )
    assert (
        "execution epoch changed"
        in (apply(_entry(store, RECORD, execution_epoch=5))[RECORD])
    )
    assert (
        "operator confirmations changed since the plan"
        in (apply(_entry(store, RECORD, confirmations=rewritten))[RECORD])
    )
    unrestored = _entry(store, RECORD)
    unrestored["node_evidence"][0]["restored"] = False
    assert "does not show every node restored" in apply(unrestored)[RECORD]
    elsewhere = _entry(store, RECORD)
    elsewhere["node_evidence"][0]["node_id"] = "node-elsewhere"
    assert "incident node set changed" in apply(elsewhere)[RECORD]

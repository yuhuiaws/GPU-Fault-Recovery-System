"""``workflow-reconcile`` and the PENDING record of a node HyperPod reclaimed.

On 2026-10-01 a spot GPU node was terminated while the control plane ingested
its last telemetry; two workflows for its incidents were created the same
second and stayed PENDING -- no step execution, no remote command, no event --
because the node's Agent was gone and nothing ever dispatched them. The release
preflight refused every deploy on them, ``uninstall`` counted them as live, and
``workflow-reconcile`` discovered BLOCKED records only.

The admin side now takes the Pod's never-dispatched items (planned beside the
BLOCKED backlog, ineligible for the one proof the Pod cannot read), proves the
node absent from Kubernetes *and* from HyperPod, records which sources were
consulted and when, and applies through ``apply-never-dispatched`` with
primitives every image has. A node still in Kubernetes, still listed by
HyperPod, a record younger than the dispatch guard and any other Pod-side
reason keep the record ineligible; evidence that moves between plan and apply
refuses the apply by name.
"""

from __future__ import annotations

import io
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from gpu_fault.admin import workflow_reconcile as reconcile
from gpu_fault.admin import workflow_reconcile_never_dispatched as never_dispatched
from gpu_fault.admin.bootstrap_common import BootstrapError
from gpu_fault.workflow_resolution import DEPARTED_NODE_PROOF_REASON
from tests.admin.conftest import TEST_OPERATOR_ARN

NODE = "hyperpod-i-00000000000000001"
INSTANCE = "i-00000000000000001"
OTHER_INSTANCE = "i-00000000000000002"
PENDING_ID = "workflow-pending-xid46"
BLOCKED_ID = "workflow-blocked-never-changed"
INCIDENT_ID = "incident-xid-46"
CREATED_AT = "2026-10-01T06:00:00+00:00"
REFERENCE = "CHG-2026-1001-01"


def _site(tmp_path: Path, *, hyperpod: bool = True) -> SimpleNamespace:
    cluster: dict[str, object] = {"cluster_id": "gpu-a", "context": "gpu-a-context"}
    if hyperpod:
        cluster["hyperpod_cluster_name"] = "gpu-a-hyperpod"
        cluster["region"] = "us-west-2"
    return SimpleNamespace(
        release_config={
            "site_name": "staging",
            "aws_region": "us-west-2",
            "cpu_eks_arn": "arn:aws:eks:us-west-2:123456789012:cluster/cpu-control",
            "clusters": [cluster],
        },
        environment={"KUBECONFIG": str(tmp_path / "gpu.kubeconfig")},
        source_sha256="a" * 64,
    )


def _pending_item(**overrides: object) -> dict[str, object]:
    """The Pod's never-dispatched item for the stuck record."""

    item: dict[str, object] = {
        "request_id": PENDING_ID,
        "incident_id": INCIDENT_ID,
        "cluster_id": "gpu-a",
        "node_ids": [NODE],
        "incident_state": "ACTION_PENDING",
        "fencing_token": 2,
        "execution_epoch": 0,
        "workflow_created_at": CREATED_AT,
        "workflow_updated_at": CREATED_AT,
        "step_execution_count": 0,
        "remote_command_count": 0,
        "terminalization": never_dispatched.NEVER_DISPATCHED,
        "eligible": False,
        "reasons": [DEPARTED_NODE_PROOF_REASON],
    }
    item.update(overrides)
    return item


def _blocked_item(**overrides: object) -> dict[str, object]:
    """A never-changed BLOCKED item on a node that is present and clean."""

    item: dict[str, object] = {
        "request_id": BLOCKED_ID,
        "incident_id": "incident-blocked",
        "cluster_id": "gpu-a",
        "node_ids": ["node-present"],
        "incident_state": "ESCALATED",
        "blocked_kind": "NEEDS_OPERATOR",
        "fencing_token": 3,
        "execution_epoch": 1,
        "workflow_updated_at": CREATED_AT,
        "successor_workflow_id": None,
        "source_plan_id": "plan-a",
        "source_plan_status": "FAILED",
        "open_remote_commands": [],
        "waiting_step_indexes": [],
        "never_changed_a_node": True,
        "completed_containment_operations": [],
        "terminalization": "never-changed",
        "eligible": True,
        "reasons": [],
    }
    item.update(overrides)
    return item


def _runtime_plan(*items: dict[str, object]) -> dict[str, object]:
    return {
        "schema_version": 1,
        "mode": "workflow-reconcile-plan",
        "evaluated_at": "2026-10-01T07:00:00+00:00",
        "plan_sha256": "b" * 64,
        "discovery": {
            "scanned": 1,
            "selected": 1,
            "remaining": 0,
            "scan_truncated": False,
        },
        "never_dispatched_discovery": {
            "scanned": 2,
            "selected": 1,
            "remaining": 0,
            "scan_truncated": False,
        },
        "items": list(items) or [_pending_item()],
    }


class Commands:
    """``run_command`` double: the GPU node list and the HyperPod inventory."""

    def __init__(
        self,
        *,
        instances: tuple[str, ...] = (OTHER_INSTANCE,),
        kubernetes_nodes: tuple[str, ...] = ("node-present",),
        aws_error: str | None = None,
    ) -> None:
        self.instances = instances
        self.kubernetes_nodes = kubernetes_nodes
        self.aws_error = aws_error
        self.calls: list[list[str]] = []

    def __call__(self, command: list[str], **_options: object) -> SimpleNamespace:
        self.calls.append(list(command))
        if command[0] == "kubectl":
            assert command[-4:] == ["get", "nodes", "-o", "json"], command
            return SimpleNamespace(
                returncode=0,
                stdout=json.dumps(
                    {
                        "items": [
                            {
                                "metadata": {"name": name, "annotations": {}},
                                "spec": {"unschedulable": False, "taints": []},
                            }
                            for name in self.kubernetes_nodes
                        ]
                    }
                ),
                stderr="",
            )
        assert command[:5] == [
            "aws",
            "sagemaker",
            "list-cluster-nodes",
            "--cluster-name",
            "gpu-a-hyperpod",
        ], command
        if self.aws_error is not None:
            return SimpleNamespace(returncode=254, stdout="", stderr=self.aws_error)
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps(
                {
                    "ClusterNodeSummaries": [
                        {"InstanceId": instance, "NodeLogicalId": f"n-{index}"}
                        for index, instance in enumerate(self.instances, start=1)
                    ]
                }
            ),
            stderr="",
        )

    @property
    def aws_calls(self) -> list[list[str]]:
        return [call for call in self.calls if call[0] == "aws"]


def _pod(plans: list[dict[str, object]], calls: list[dict[str, object]]):
    queue = list(plans)

    def run(_site: object, payload: dict[str, object], **_kwargs: object):
        calls.append(payload)
        if payload["mode"] == "plan":
            return queue.pop(0) if queue else _runtime_plan()
        return {
            "mode": "workflow-reconcile-apply",
            "applied_workflow_ids": [
                str(item["request_id"]) for item in payload.get("items") or []
            ]
            or list(payload.get("workflow_ids") or []),
            "failed_workflow_ids": [],
            "failures": {},
            "resolved_plan_ids": [],
            "archive_eligible_incident_ids": [INCIDENT_ID],
            "records_deleted": 0,
        }

    return run


def _dry_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    commands: Commands,
    *,
    plans: list[dict[str, object]] | None = None,
    **selectors: Any,
) -> dict[str, Any]:
    monkeypatch.setattr(reconcile, "_run_reconcile", _pod(plans or [], []))
    monkeypatch.setattr(reconcile, "run_command", commands)
    return reconcile.run_workflow_reconcile(
        _site(tmp_path), tmp_path, dry_run=True, **selectors
    )


def _node(plan: dict[str, Any], request_id: str = PENDING_ID) -> dict[str, Any]:
    [item] = [item for item in plan["items"] if item["request_id"] == request_id]
    [node] = item["scheduling_evidence"]["nodes"]
    return node


def _item(plan: dict[str, Any], request_id: str = PENDING_ID) -> dict[str, Any]:
    [item] = [item for item in plan["items"] if item["request_id"] == request_id]
    return item


# ------------------------------------------------------------- the discovery


def test_a_never_dispatched_record_of_a_departed_node_is_selected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    commands = Commands()

    plan = _dry_run(tmp_path, monkeypatch, commands)

    item, node = _item(plan), _node(plan)
    assert item["eligible"] is True, item["reasons"]
    assert item["reasons"] == []
    assert item["terminalization"] == never_dispatched.NEVER_DISPATCHED
    assert node["absent_from_kubernetes"] is True
    assert node["absent_from_provider"] is True
    assert node["instance_id"] == INSTANCE
    assert node["restored"] is True and node["blockers"] == []
    assert node["sources"] == {
        "kubernetes": {"consulted": True, "present": False},
        "hyperpod": {
            "consulted": True,
            "cluster_name": "gpu-a-hyperpod",
            "region": "us-west-2",
            "listed": False,
        },
    }
    read_at = item["evidence_read_at"]
    assert sorted(read_at) == ["hyperpod", "kubernetes"]
    for value in read_at.values():
        assert datetime.fromisoformat(value).tzinfo is not None, value
    assert plan["never_dispatched_discovery"]["selected"] == 1
    assert len(commands.aws_calls) == 1, "one provider read per cluster per plan"


def test_a_record_whose_node_is_still_in_kubernetes_is_not_selected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Present -- Ready or NotReady, cordoned or clean -- is not departed."""

    commands = Commands(kubernetes_nodes=(NODE,))

    plan = _dry_run(tmp_path, monkeypatch, commands)

    item, node = _item(plan), _node(plan)
    assert item["eligible"] is False
    assert item["reasons"] == [
        DEPARTED_NODE_PROOF_REASON,
        "GPU node scheduling state has not been restored",
    ]
    assert node["absent_from_kubernetes"] is False
    assert node["blockers"] == [never_dispatched.KUBERNETES_PRESENT_BLOCKER]
    assert node["sources"] == {"kubernetes": {"consulted": True, "present": True}}
    assert commands.aws_calls == [], "a node Kubernetes has is never looked up"


def test_a_node_hyperpod_still_lists_is_not_departed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    commands = Commands(instances=(INSTANCE, OTHER_INSTANCE))

    plan = _dry_run(tmp_path, monkeypatch, commands)

    item, node = _item(plan), _node(plan)
    assert item["eligible"] is False
    assert node["absent_from_kubernetes"] is True
    assert node["absent_from_provider"] is False
    assert node["sources"]["hyperpod"]["listed"] is True
    assert node["blockers"] == [
        f"node is missing from Kubernetes but HyperPod still lists instance {INSTANCE}"
    ]


@pytest.mark.parametrize(
    "commands", [Commands(aws_error="An error occurred (AccessDeniedException)")]
)
def test_a_failed_provider_lookup_keeps_the_record_ineligible(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, commands: Commands
) -> None:
    plan = _dry_run(tmp_path, monkeypatch, commands)

    item, node = _item(plan), _node(plan)
    assert item["eligible"] is False
    assert node["absent_from_provider"] is False
    assert node["sources"]["hyperpod"]["consulted"] is False
    assert node["blockers"][1].startswith("provider membership unknown"), node


def test_a_record_younger_than_the_guard_is_not_selected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The Pod lists the guard beside the proof reason; departed-node evidence
    answers the proof only, so the record stays as the Pod judged it."""

    young = _pending_item(
        reasons=[
            "workflow is younger than the 10-minute dispatch guard",
            DEPARTED_NODE_PROOF_REASON,
        ]
    )

    plan = _dry_run(tmp_path, monkeypatch, Commands(), plans=[_runtime_plan(young)])

    item = _item(plan)
    assert item["eligible"] is False
    assert item["reasons"] == young["reasons"], "the evidence answers one reason only"
    assert _node(plan)["absent_from_provider"] is True, "the node did depart"


def test_the_blocked_path_is_unchanged_beside_the_new_shape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A never-changed BLOCKED item on a present, clean node is still eligible;
    the same present node would keep a never-dispatched item out."""

    plan = _dry_run(
        tmp_path,
        monkeypatch,
        Commands(),
        plans=[_runtime_plan(_blocked_item(), _pending_item())],
    )

    blocked, pending = _item(plan, BLOCKED_ID), _item(plan)
    assert blocked["eligible"] is True, blocked["reasons"]
    assert blocked["terminalization"] == "never-changed"
    assert "evidence_read_at" not in blocked, "read times belong to the new shape"
    assert "sources" not in _node(plan, BLOCKED_ID)
    assert pending["eligible"] is True, pending["reasons"]
    assert plan["discovery"] == {
        "scanned": 1,
        "selected": 1,
        "remaining": 0,
        "scan_truncated": False,
    }


def test_the_selectors_reach_the_pod_planner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(reconcile, "_run_reconcile", _pod([], calls))
    monkeypatch.setattr(reconcile, "run_command", Commands())

    reconcile.run_workflow_reconcile(
        _site(tmp_path), tmp_path, incident_ids=(INCIDENT_ID,), dry_run=True
    )
    reconcile.run_workflow_reconcile(
        _site(tmp_path), tmp_path, workflow_ids=(PENDING_ID,), dry_run=True
    )

    assert calls[0] == {
        "mode": "plan",
        "workflow_ids": [],
        "incident_ids": [INCIDENT_ID],
    }
    assert calls[1] == {"mode": "plan", "workflow_ids": [PENDING_ID]}


# ------------------------------------------------------------------ the digest


def test_the_read_times_are_outside_the_digest_and_the_verdicts_inside(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = _dry_run(tmp_path, monkeypatch, Commands())
    second = _dry_run(tmp_path, monkeypatch, Commands())
    listed = _dry_run(tmp_path, monkeypatch, Commands(instances=(INSTANCE,)))

    assert first["plan_sha256"] == second["plan_sha256"], (
        "a re-plan a moment later is not drift"
    )
    assert first["plan_sha256"] != listed["plan_sha256"], (
        "the provider's verdict binds the approval"
    )


def test_evidence_that_changes_between_plan_and_apply_refuses_the_apply(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The instance reappears at HyperPod between the plan and the re-plan."""

    calls: list[dict[str, object]] = []
    monkeypatch.setattr(reconcile, "_run_reconcile", _pod([], calls))
    answers = iter([Commands(), Commands(instances=(INSTANCE,))])
    current: list[Commands] = []

    def run_command(command: list[str], **options: object) -> SimpleNamespace:
        if command[0] == "kubectl" or not current:
            if command[0] == "kubectl":
                current.append(next(answers))
        return current[-1](command, **options)

    monkeypatch.setattr(reconcile, "run_command", run_command)

    with pytest.raises(BootstrapError, match="plan changed before apply") as refused:
        reconcile.run_workflow_reconcile(
            _site(tmp_path), tmp_path, workflow_ids=(PENDING_ID,), reference=REFERENCE
        )

    message = str(refused.value)
    assert f"{PENDING_ID}: eligible True -> False" in message
    assert f"{PENDING_ID}: scheduling_evidence" in message
    assert [call["mode"] for call in calls] == ["plan", "plan"], "nothing was applied"
    assert not (tmp_path / reconcile.HISTORY_PATH).exists(), (
        "a refusal archives nothing"
    )


# ------------------------------------------------------------------ the bridge


def test_the_apply_goes_through_the_never_dispatched_bridge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(reconcile, "_run_reconcile", _pod([], calls))
    monkeypatch.setattr(reconcile, "run_command", Commands())

    result = reconcile.run_workflow_reconcile(
        _site(tmp_path), tmp_path, workflow_ids=(PENDING_ID,), reference=REFERENCE
    )

    assert [call["mode"] for call in calls] == [
        "plan",
        "plan",
        "apply-never-dispatched",
    ]
    bridge = calls[-1]
    [entry] = bridge["items"]
    assert entry["request_id"] == PENDING_ID
    assert entry["fencing_token"] == 2 and entry["execution_epoch"] == 0
    assert entry["workflow_created_at"] == CREATED_AT
    [evidence] = entry["departed_node_evidence"]
    assert evidence["node_id"] == NODE
    assert evidence["instance_id"] == INSTANCE
    assert evidence["absent_from_kubernetes"] is True
    assert evidence["absent_from_provider"] is True
    assert evidence["sources"]["hyperpod"]["cluster_name"] == "gpu-a-hyperpod"
    assert sorted(evidence["read_at"]) == ["hyperpod", "kubernetes"]
    assert bridge["reference"] == REFERENCE
    assert bridge["actor"] == TEST_OPERATOR_ARN
    assert bridge["admin_plan_sha256"] == result["plan_sha256"]
    assert result["applied_workflow_ids"] == [PENDING_ID]
    assert result["records_deleted"] == 0
    archive = tmp_path / reconcile.HISTORY_PATH / str(result["plan_sha256"])
    plan = json.loads((archive / "plan.json").read_text(encoding="utf-8"))
    assert (
        plan["items"][0]["scheduling_evidence"]["nodes"][0]["sources"]["hyperpod"][
            "listed"
        ]
        is False
    )
    assert plan["items"][0]["evidence_read_at"], "the archive keeps the read times"


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


def _control_plane(now: datetime) -> Any:
    """An in-memory control plane with the stuck record and its neighbours."""

    from gpu_fault.models import (
        BlockedKind,
        IncidentState,
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

    plugin = WorkflowOperation.RESTART_GPU_DEVICE_PLUGIN
    store = build_store()

    def record(
        request_id: str,
        incident_id: str,
        *,
        age: timedelta = timedelta(hours=1),
        node_ids: list[str] | None = None,
        **values: Any,
    ) -> None:
        fields: dict[str, Any] = {
            "status": WorkflowStatus.PENDING,
            "fencing_token": 2,
            "execution_epoch": 0,
            "official_steps": [workflow_step(plugin, node_ids=node_ids or [NODE])],
            "created_at": now - age,
            "updated_at": now - age,
            **values,
        }
        store.save_workflow(workflow_request(request_id, incident_id, **fields))
        store.save_incident(
            fault_incident(
                incident_id,
                f"event-{incident_id}",
                cluster_id="gpu-a",
                node_ids=node_ids or [NODE],
                state=IncidentState.ACTION_PENDING,
                workflow_request_id=request_id,
                fencing_token=2,
            )
        )

    record(PENDING_ID, INCIDENT_ID)
    record("workflow-young", "incident-young", age=timedelta(minutes=3))
    record(
        "workflow-ran",
        "incident-ran",
        step_executions=[workflow_step_execution(0, plugin, WorkflowStepStatus.FAILED)],
    )
    record(
        "workflow-two-nodes",
        "incident-two-nodes",
        node_ids=[NODE, "hyperpod-i-00000000000000002"],
    )
    record("workflow-listed", "incident-listed")
    record(
        "workflow-blocked",
        "incident-blocked",
        status=WorkflowStatus.BLOCKED,
        blocked_kind=BlockedKind.NEEDS_OPERATOR,
    )
    return store


def _evidence(
    *node_ids: str, absent_from_provider: bool = True
) -> list[dict[str, Any]]:
    return [
        {
            "node_id": node_id,
            "instance_id": node_id.removeprefix("hyperpod-"),
            "provider": "hyperpod",
            "absent_from_kubernetes": True,
            "absent_from_provider": absent_from_provider,
            "sources": {
                "kubernetes": {"consulted": True, "present": False},
                "hyperpod": {
                    "consulted": True,
                    "cluster_name": "gpu-a-hyperpod",
                    "region": "us-west-2",
                    "listed": not absent_from_provider,
                },
            },
            "read_at": {
                "kubernetes": "2026-10-01T07:00:00+00:00",
                "hyperpod": "2026-10-01T07:00:01+00:00",
            },
        }
        for node_id in node_ids
    ]


def test_the_bridge_closes_the_record_and_hands_the_incident_to_the_operator(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """``apply-never-dispatched`` against an in-memory control plane: the rule
    re-derived minus the proof the evidence answers, the compare-and-set on
    fencing token, epoch and ``created_at``, the evidence bound to the
    incident's node set, ``amend_workflow`` with the audit event carrying the
    evidence, the incident ESCALATED with the same audit line, and every other
    item reported, not written."""

    from gpu_fault.models import IncidentState, WorkflowEventKind, WorkflowStatus
    from gpu_fault.orchestration.incident_closure import IncidentClosureService

    now = datetime.now(timezone.utc)
    store = _control_plane(now)

    def entry(request_id: str, **overrides: Any) -> dict[str, Any]:
        values: dict[str, Any] = {
            "request_id": request_id,
            "fencing_token": 2,
            "execution_epoch": 0,
            "workflow_created_at": store.get_workflow(
                request_id
            ).created_at.isoformat(),
            "departed_node_evidence": _evidence(NODE),
        }
        values.update(overrides)
        return values

    result = _exec_script(
        monkeypatch,
        capsys,
        store,
        {
            "mode": "apply-never-dispatched",
            "items": [
                entry(PENDING_ID),
                entry("workflow-young"),
                entry("workflow-ran"),
                entry("workflow-blocked"),
                entry("workflow-two-nodes"),
                entry(
                    "workflow-listed",
                    departed_node_evidence=_evidence(NODE, absent_from_provider=False),
                ),
            ],
            "reference": REFERENCE,
            "actor": TEST_OPERATOR_ARN,
            "admin_plan_sha256": "d" * 64,
        },
    )

    assert result["applied_workflow_ids"] == [PENDING_ID]
    assert result["failed_workflow_ids"] == [
        "workflow-blocked",
        "workflow-listed",
        "workflow-ran",
        "workflow-two-nodes",
        "workflow-young",
    ]
    assert (
        "younger than the 10-minute dispatch guard"
        in result["failures"]["workflow-young"]
    )
    assert "has step executions" in result["failures"]["workflow-ran"]
    assert "status is BLOCKED, not PENDING" in result["failures"]["workflow-blocked"]
    assert "incident node set changed" in result["failures"]["workflow-two-nodes"]
    assert "does not prove every node gone" in result["failures"]["workflow-listed"]
    assert result["archive_eligible_incident_ids"] == [INCIDENT_ID]
    assert result["records_deleted"] == 0

    closed = store.get_workflow(PENDING_ID)
    assert closed.status is WorkflowStatus.SUPERSEDED
    assert closed.preempted_by_workflow_id is None
    assert "never dispatched, after its node left Kubernetes and HyperPod" in (
        closed.preemption_reason or ""
    )
    event = closed.events[-1]
    assert event.kind is WorkflowEventKind.OPERATOR_RECONCILED
    assert event.actor == TEST_OPERATOR_ARN
    assert event.details["terminalization"] == never_dispatched.NEVER_DISPATCHED
    assert event.details["previous_status"] == "PENDING"
    assert event.details["admin_plan_sha256"] == "d" * 64
    assert event.details["departed_node_evidence"] == [
        {
            "node_id": NODE,
            "instance_id": INSTANCE,
            "provider": "hyperpod",
            "absent_from_kubernetes": True,
            "absent_from_provider": True,
            "hyperpod_cluster_name": "gpu-a-hyperpod",
            "region": "us-west-2",
            "kubernetes_read_at": "2026-10-01T07:00:00+00:00",
            "hyperpod_read_at": "2026-10-01T07:00:01+00:00",
        }
    ], "the sources, their verdicts and their read times are on the event"
    assert event.details["expected_fencing_token"] == 2

    incident = store.get_incident(INCIDENT_ID)
    assert incident.state is IncidentState.ESCALATED, "handed to the operator"
    assert incident.reasons[-1].startswith(
        f"operator reconciliation {REFERENCE}: closed"
    ), incident.reasons
    assert incident.workflow_request_id == PENDING_ID, "the pointer is not rewritten"
    preview = IncidentClosureService(store).preview(INCIDENT_ID)
    assert preview["closable"] is True, preview
    assert preview["open_workflow_id"] is None

    for request_id in (
        "workflow-young",
        "workflow-ran",
        "workflow-two-nodes",
        "workflow-listed",
    ):
        assert store.get_workflow(request_id).status is WorkflowStatus.PENDING
        assert store.get_incident(
            f"incident-{request_id.removeprefix('workflow-')}"
        ).state is (IncidentState.ACTION_PENDING)
    assert store.get_workflow("workflow-blocked").status is WorkflowStatus.BLOCKED


def test_the_bridge_refuses_a_record_that_moved_under_the_approval(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from gpu_fault.models import WorkflowStatus

    store = _control_plane(datetime.now(timezone.utc))
    created_at = store.get_workflow(PENDING_ID).created_at.isoformat()

    result = _exec_script(
        monkeypatch,
        capsys,
        store,
        {
            "mode": "apply-never-dispatched",
            "items": [
                {
                    "request_id": PENDING_ID,
                    "fencing_token": 3,
                    "execution_epoch": 0,
                    "workflow_created_at": created_at,
                    "departed_node_evidence": _evidence(NODE),
                }
            ],
            "reference": REFERENCE,
            "actor": TEST_OPERATOR_ARN,
            "admin_plan_sha256": "d" * 64,
        },
    )

    assert result["applied_workflow_ids"] == []
    assert (
        "fencing token changed: expected 3, found 2" in result["failures"][PENDING_ID]
    )
    assert store.get_workflow(PENDING_ID).status is WorkflowStatus.PENDING


# --------------------------------------------- the planner shipped to an old image


def test_the_pod_plan_merges_the_never_dispatched_shape_in(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Discovery lists it beside the BLOCKED backlog; an explicit PENDING id the
    deployed planner refused for not being BLOCKED takes the new verdict."""

    store = _control_plane(datetime.now(timezone.utc))

    discovered = _exec_script(
        monkeypatch, capsys, store, {"mode": "plan", "workflow_ids": []}
    )
    explicit = _exec_script(
        monkeypatch,
        capsys,
        store,
        {"mode": "plan", "workflow_ids": [PENDING_ID, "workflow-blocked"]},
    )

    assert [item["request_id"] for item in discovered["items"]] == [
        "workflow-blocked",
        "workflow-listed",
        PENDING_ID,
        "workflow-two-nodes",
    ], "BLOCKED first, then the never-dispatched records"
    assert discovered["never_dispatched_discovery"] == {
        "scanned": 5,
        "selected": 3,
        "remaining": 0,
        "scan_truncated": False,
    }
    by_id = {item["request_id"]: item for item in explicit["items"]}
    assert by_id[PENDING_ID]["terminalization"] == never_dispatched.NEVER_DISPATCHED
    assert by_id[PENDING_ID]["reasons"] == [DEPARTED_NODE_PROOF_REASON]
    assert by_id["workflow-blocked"]["terminalization"] == "never-changed"
    assert len(explicit["plan_sha256"]) == 64


def test_the_fallback_planner_matches_the_module(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Against an image without ``never_dispatched_plan_items`` the script
    plans the shape itself; the two copies must say the same thing."""

    import gpu_fault.workflow_reconcile as pod_module
    import gpu_fault.workflow_resolution as resolution_module

    store = _control_plane(datetime.now(timezone.utc))
    payload = {"mode": "plan", "workflow_ids": []}

    with_module = _exec_script(monkeypatch, capsys, store, payload)
    monkeypatch.delattr(pod_module, "never_dispatched_plan_items")
    monkeypatch.delattr(resolution_module, "never_dispatched_reconciliation_reasons")
    fallback = _exec_script(monkeypatch, capsys, store, payload)

    assert fallback["items"] == with_module["items"]
    assert (
        fallback["never_dispatched_discovery"]
        == with_module["never_dispatched_discovery"]
    )
    assert fallback["plan_sha256"] == with_module["plan_sha256"]


@pytest.mark.parametrize(
    "changes",
    [
        {"reasons": [DEPARTED_NODE_PROOF_REASON, "incident is missing"]},
        {"reasons": []},
        {"terminalization": "never-changed"},
        {"eligible": True},
        {"scheduling_evidence": {"restored": True, "nodes": []}},
        {
            "scheduling_evidence": {
                "restored": True,
                "nodes": [
                    {"absent_from_kubernetes": True, "absent_from_provider": False}
                ],
            }
        },
        {
            "scheduling_evidence": {
                "restored": False,
                "nodes": [
                    {"absent_from_kubernetes": True, "absent_from_provider": True}
                ],
            }
        },
    ],
)
def test_only_the_exact_departed_shape_is_promoted(changes: dict[str, object]) -> None:
    item = _pending_item(
        scheduling_evidence={
            "restored": True,
            "nodes": [{"absent_from_kubernetes": True, "absent_from_provider": True}],
        }
    )
    item.update(changes)
    before = json.loads(json.dumps(item))

    never_dispatched.promote_never_dispatched(item)

    assert item == before

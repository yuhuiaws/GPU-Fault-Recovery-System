"""``workflow-reconcile`` and the node HyperPod replaced.

A ``never-changed`` record whose node is gone from Kubernetes used to fail
closed on "node is missing" for ever: the node it would need to show restored
no longer exists, ``confirm-node-action`` needs a Ready node, and on
2026-09-30 the operator ended up hand-editing the database. The admin side now
asks the provider: the node name ``hyperpod-<instance-id>`` is looked up in
``aws sagemaker list-cluster-nodes``; an instance HyperPod no longer lists
carries no cordon, taint or annotation, so there is nothing to restore. Every
other answer -- still listed, not HyperPod-managed, lookup failed, a name that
is no instance -- keeps failing closed and says why. ``verified-restore`` items
keep the Kubernetes-only rule.

The same record's plan was replaced in place, so the deployed planner also
refused it for the source plan it no longer had; that shape is promoted to
``never-changed-non-plan`` and applied through the ``apply-never-changed``
bridge, with primitives every image has.
"""

from __future__ import annotations

import io
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from gpu_fault.admin import workflow_reconcile as reconcile
from gpu_fault.admin.bootstrap_common import BootstrapError
from tests.admin.conftest import TEST_OPERATOR_ARN

NODE = "hyperpod-i-00000000000000001"
INSTANCE = "i-00000000000000001"
REQUEST_ID = "workflow-b2cf18a64127092442f07935"


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


def _item(**overrides: object) -> dict[str, object]:
    """The runtime plan item for the reaped record, as the checkout plans it."""

    item: dict[str, object] = {
        "request_id": REQUEST_ID,
        "incident_id": "incident-efa-mismatch",
        "cluster_id": "gpu-a",
        "node_ids": [NODE],
        "incident_state": "ESCALATED",
        "blocked_kind": "NEEDS_OPERATOR",
        "fencing_token": 2,
        "execution_epoch": 1,
        "workflow_updated_at": "2026-09-30T06:00:00+00:00",
        "successor_workflow_id": None,
        "source_plan_id": None,
        "source_plan_status": None,
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
        "evaluated_at": "2026-09-30T07:00:00+00:00",
        "plan_sha256": "b" * 64,
        "items": list(items) or [_item()],
    }


class Commands:
    """``run_command`` double: an empty GPU node list and a HyperPod inventory.

    ``instances`` are spread over two pages so the pagination is exercised on
    every read; ``aws_error`` makes the provider call fail.
    """

    def __init__(
        self,
        *,
        instances: tuple[str, ...] = (),
        aws_error: str | None = None,
        kubernetes_nodes: tuple[str, ...] = (),
    ) -> None:
        self.instances = instances
        self.aws_error = aws_error
        self.kubernetes_nodes = kubernetes_nodes
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
        assert command[5:9] == ["--region", "us-west-2", "--output", "json"], command
        if self.aws_error is not None:
            return SimpleNamespace(returncode=254, stdout="", stderr=self.aws_error)
        first_page = "--next-token" not in command
        page = (
            [{"InstanceId": "i-00000000000000002", "NodeLogicalId": "n-1"}]
            if first_page
            else [
                {"InstanceId": instance, "NodeLogicalId": f"n-{index}"}
                for index, instance in enumerate(self.instances, start=2)
            ]
        )
        body: dict[str, object] = {"ClusterNodeSummaries": page}
        if first_page:
            body["NextToken"] = "page-2"
        else:
            assert command[command.index("--next-token") + 1] == "page-2"
        return SimpleNamespace(returncode=0, stdout=json.dumps(body), stderr="")

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
            "archive_eligible_incident_ids": ["incident-efa-mismatch"],
            "records_deleted": 0,
        }

    return run


def _dry_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    commands: Commands,
    *,
    site: SimpleNamespace | None = None,
    plans: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    monkeypatch.setattr(reconcile, "_run_reconcile", _pod(plans or [], []))
    monkeypatch.setattr(reconcile, "run_command", commands)
    return reconcile.run_workflow_reconcile(
        site or _site(tmp_path), tmp_path, workflow_ids=(REQUEST_ID,), dry_run=True
    )


def _evidence(plan: dict[str, object]) -> tuple[dict[str, object], dict[str, object]]:
    [item] = plan["items"]  # type: ignore[misc]
    evidence = item["scheduling_evidence"]
    [node] = evidence["nodes"]
    return item, node


# --------------------------------------------------------------- the four verdicts


def test_a_node_hyperpod_no_longer_lists_is_restored_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    commands = Commands(instances=("i-00000000000000002",))

    plan = _dry_run(tmp_path, monkeypatch, commands)

    item, node = _evidence(plan)
    assert node == {
        "node_id": NODE,
        "exists": False,
        "provider": "hyperpod",
        "instance_id": INSTANCE,
        "absent_from_provider": True,
        "restored": True,
        "blockers": [],
    }
    assert item["scheduling_evidence"]["restored"] is True
    assert item["eligible"] is True, item["reasons"]
    assert len(commands.aws_calls) == 2, "both inventory pages were read"


def test_a_node_hyperpod_still_lists_keeps_the_record_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    commands = Commands(instances=(INSTANCE,))

    plan = _dry_run(tmp_path, monkeypatch, commands)

    item, node = _evidence(plan)
    assert node["absent_from_provider"] is False
    assert node["restored"] is False
    assert node["blockers"] == [
        f"node is missing from Kubernetes but HyperPod still lists instance {INSTANCE}"
    ]
    assert item["eligible"] is False
    assert item["reasons"] == ["GPU node scheduling state has not been restored"]


def test_a_cluster_that_is_not_hyperpod_managed_fails_closed_without_aws(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    commands = Commands()

    plan = _dry_run(
        tmp_path, monkeypatch, commands, site=_site(tmp_path, hyperpod=False)
    )

    item, node = _evidence(plan)
    assert node["exists"] is False and node["restored"] is False
    assert node["blockers"] == [
        "node is missing",
        "provider membership unknown: GPU cluster gpu-a is not HyperPod-managed "
        "in the site",
    ]
    assert item["eligible"] is False
    assert commands.aws_calls == [], "no cluster name, no provider call"


def test_a_failed_provider_lookup_fails_closed_and_names_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    commands = Commands(aws_error="An error occurred (AccessDeniedException)")

    plan = _dry_run(tmp_path, monkeypatch, commands)

    item, node = _evidence(plan)
    assert node["restored"] is False
    assert node["blockers"][0] == "node is missing"
    assert node["blockers"][1].startswith(
        "provider membership unknown: cannot list HyperPod nodes of gpu-a"
    ), node["blockers"]
    assert item["eligible"] is False


# ------------------------------------------------------------------ the boundaries


def test_a_verified_restore_item_keeps_the_kubernetes_only_rule(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    commands = Commands(instances=("i-00000000000000002",))

    plan = _dry_run(
        tmp_path,
        monkeypatch,
        commands,
        plans=[
            _runtime_plan(
                _item(
                    terminalization="verified-restore",
                    successor_workflow_id="workflow-restored",
                    never_changed_a_node=False,
                )
            )
        ],
    )

    item, node = _evidence(plan)
    assert node["blockers"] == ["node is missing"]
    assert item["eligible"] is False
    assert commands.aws_calls == [], "a restore successor is proven on the node"


def test_a_node_name_without_an_instance_id_is_never_looked_up(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    commands = Commands(instances=("i-00000000000000002",))

    plan = _dry_run(
        tmp_path,
        monkeypatch,
        commands,
        plans=[_runtime_plan(_item(node_ids=["node-a"]))],
    )

    _item_value, node = _evidence(plan)
    assert node["blockers"] == [
        "node is missing",
        "provider membership unknown: node name carries no HyperPod instance id",
    ]
    assert commands.aws_calls == []


def test_a_node_still_in_kubernetes_is_judged_there_not_at_the_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    commands = Commands(instances=(INSTANCE,), kubernetes_nodes=(NODE,))

    plan = _dry_run(tmp_path, monkeypatch, commands)

    item, node = _evidence(plan)
    assert node["exists"] is True and node["restored"] is True
    assert "provider" not in node
    assert item["eligible"] is True
    assert commands.aws_calls == []


def test_the_provider_evidence_is_bound_by_the_plan_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    absent = _dry_run(
        tmp_path, monkeypatch, Commands(instances=("i-00000000000000002",))
    )
    present = _dry_run(tmp_path, monkeypatch, Commands(instances=(INSTANCE,)))

    assert absent["plan_sha256"] != present["plan_sha256"], (
        "an approval of the departed-node verdict must not apply to a node that "
        "reappeared at the provider"
    )


def test_one_provider_read_per_cluster_per_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    commands = Commands(instances=("i-00000000000000002",))
    second = _item(
        request_id="workflow-second", node_ids=["hyperpod-i-0000000000000000"]
    )
    plans = [_runtime_plan(_item(), second)]
    monkeypatch.setattr(reconcile, "_run_reconcile", _pod(plans, []))
    monkeypatch.setattr(reconcile, "run_command", commands)

    plan = reconcile.run_workflow_reconcile(
        _site(tmp_path),
        tmp_path,
        workflow_ids=(REQUEST_ID, "workflow-second"),
        dry_run=True,
    )

    assert all(item["eligible"] for item in plan["items"]), plan["items"]
    assert len(commands.aws_calls) == 2, "two pages, read once for both items"


def test_hyperpod_instance_ids_refuses_a_repeating_page_token(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def run(command: list[str], **_options: object) -> SimpleNamespace:
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps({"ClusterNodeSummaries": [], "NextToken": "same"}),
            stderr="",
        )

    monkeypatch.setattr(reconcile, "run_command", run)

    with pytest.raises(BootstrapError, match="repeats its page token"):
        reconcile.hyperpod_instance_ids(_site(tmp_path), "gpu-a")


# ------------------------------------------------- never-changed-non-plan bridge


def _deployed_item(**overrides: object) -> dict[str, object]:
    """What a deployed planner with the old gate says about the record."""

    values: dict[str, object] = {
        "eligible": False,
        "reasons": ["workflow has no source recovery plan"],
    }
    values.update(overrides)
    return _item(**values)


def test_the_deployed_planners_refusal_is_promoted_and_bridged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        reconcile,
        "_run_reconcile",
        _pod([_runtime_plan(_deployed_item()), _runtime_plan(_deployed_item())], calls),
    )
    monkeypatch.setattr(
        reconcile, "run_command", Commands(instances=("i-00000000000000002",))
    )

    result = reconcile.run_workflow_reconcile(
        _site(tmp_path),
        tmp_path,
        workflow_ids=(REQUEST_ID,),
        reference="CHG-2026-0930-01",
    )

    assert [call["mode"] for call in calls] == ["plan", "plan", "apply-never-changed"]
    bridge = calls[-1]
    assert bridge["items"] == [
        {"request_id": REQUEST_ID, "fencing_token": 2, "execution_epoch": 1}
    ], "no successor to name on this bridge"
    assert bridge["reference"] == "CHG-2026-0930-01"
    assert bridge["actor"] == TEST_OPERATOR_ARN
    assert bridge["admin_plan_sha256"] == result["plan_sha256"]
    assert result["applied_workflow_ids"] == [REQUEST_ID]
    assert result["resolved_plan_ids"] == []
    assert result["records_deleted"] == 0
    archive = tmp_path / reconcile.HISTORY_PATH / str(result["plan_sha256"])
    plan = json.loads((archive / "plan.json").read_text(encoding="utf-8"))
    assert plan["items"][0]["terminalization"] == reconcile.NEVER_CHANGED_NON_PLAN
    assert plan["items"][0]["scheduling_evidence"]["nodes"][0]["absent_from_provider"]


@pytest.mark.parametrize(
    "changes",
    [
        {"reasons": ["workflow has no source recovery plan", "incident is missing"]},
        {"terminalization": "verified-restore"},
        {"successor_workflow_id": "workflow-restored"},
        {"source_plan_id": "plan-a"},
        {"eligible": True, "reasons": []},
    ],
)
def test_only_the_exact_never_changed_non_plan_shape_is_promoted(
    changes: dict[str, object],
) -> None:
    item = _deployed_item(**changes)
    before = dict(item)

    reconcile.promote_non_plan_never_changed(item)

    assert item == before


def test_the_never_changed_bridge_writes_with_deployed_primitives(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """``apply-never-changed`` against an in-memory control plane: the
    resolver's eligibility minus the plan gate, the compare-and-set on fencing
    token and epoch, ``amend_workflow`` with the audit event, no plan and no
    incident write; anything else is reported, not written."""

    from gpu_fault.app import ApplicationContext
    from gpu_fault.models import (
        BlockedKind,
        IncidentState,
        WorkflowEventCode,
        WorkflowEventKind,
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
    now = datetime.now(timezone.utc)
    store = build_store()

    def parked(request_id: str, incident_id: str, **values: object) -> None:
        store.save_workflow(
            workflow_request(
                request_id,
                incident_id,
                status=WorkflowStatus.BLOCKED,
                blocked_kind=BlockedKind.NEEDS_OPERATOR,
                fencing_token=2,
                execution_epoch=1,
                official_steps=[workflow_step(plugin, node_ids=[NODE])],
                step_executions=[
                    workflow_step_execution(
                        0,
                        plugin,
                        WorkflowStepStatus.FAILED,
                        details={"outcome_unknown": True},
                    )
                ],
                updated_at=now - timedelta(hours=1),
                **values,
            )
        )
        store.save_incident(
            fault_incident(
                incident_id,
                f"event-{incident_id}",
                node_ids=[NODE],
                state=IncidentState.ESCALATED,
                workflow_request_id=request_id,
                fencing_token=2,
            )
        )

    parked(REQUEST_ID, "incident-efa-mismatch")
    parked("workflow-changed", "incident-changed", completed_operations=[plugin])
    parked("workflow-moved", "incident-moved")
    parked("workflow-planned", "incident-planned", source_plan_id="plan-x")
    incident_before = store.get_incident("incident-efa-mismatch")
    context = ApplicationContext(store=store)
    monkeypatch.setattr(
        ApplicationContext, "from_environment", classmethod(lambda cls: context)
    )
    payload = {
        "mode": "apply-never-changed",
        "items": [
            {"request_id": REQUEST_ID, "fencing_token": 2, "execution_epoch": 1},
            {
                "request_id": "workflow-changed",
                "fencing_token": 2,
                "execution_epoch": 1,
            },
            {"request_id": "workflow-moved", "fencing_token": 3, "execution_epoch": 1},
            {
                "request_id": "workflow-planned",
                "fencing_token": 2,
                "execution_epoch": 1,
            },
        ],
        "reference": "CHG-2026-0930-01",
        "actor": TEST_OPERATOR_ARN,
        "admin_plan_sha256": "d" * 64,
    }
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(payload)))

    exec(compile(reconcile.RECONCILE_SCRIPT, "<workflow-reconcile>", "exec"), {})
    result = json.loads(capsys.readouterr().out)

    assert result["applied_workflow_ids"] == [REQUEST_ID]
    assert result["failed_workflow_ids"] == [
        "workflow-changed",
        "workflow-moved",
        "workflow-planned",
    ]
    assert "node-mutating operation" in result["failures"]["workflow-changed"]
    assert "fencing token changed: expected 3" in result["failures"]["workflow-moved"]
    assert "has a source recovery plan" in result["failures"]["workflow-planned"]
    assert result["resolved_plan_ids"] == []
    assert result["archive_eligible_incident_ids"] == ["incident-efa-mismatch"]
    assert result["records_deleted"] == 0
    closed = store.get_workflow(REQUEST_ID)
    assert closed.status is WorkflowStatus.SUPERSEDED
    assert closed.preempted_by_workflow_id is None
    assert "completed no node-mutating operation, with its incident ESCALATED" in (
        closed.preemption_reason or ""
    )
    event = closed.events[-1]
    assert event.kind is WorkflowEventKind.OPERATOR_RECONCILED
    assert event.code == WorkflowEventCode.OPERATOR_RECONCILED.value
    assert event.actor == TEST_OPERATOR_ARN
    assert event.details["terminalization"] == reconcile.NEVER_CHANGED_NON_PLAN
    assert event.details["expected_fencing_token"] == 2
    assert event.details["admin_plan_sha256"] == "d" * 64
    assert event.details["previous_status"] == "BLOCKED"
    assert store.get_incident("incident-efa-mismatch") == incident_before, (
        "the never-changed bridge never writes the incident"
    )
    for request_id in ("workflow-changed", "workflow-moved", "workflow-planned"):
        assert store.get_workflow(request_id).status is WorkflowStatus.BLOCKED

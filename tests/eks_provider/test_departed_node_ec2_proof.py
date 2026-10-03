"""Departed-node proof on an ``eks-managed`` cluster: EC2 terminated or not found.

Mirrors ``tests/admin/test_admin_workflow_reconcile_departed_node.py`` for a
site whose GPU cluster carries ``provider: eks-managed``: the node name is a
private DNS name with no instance id in it, so the fleet's recorded
``node_instance_id`` is asked of ``aws ec2 describe-instances``. Terminated or
unknown instances prove the departure; anything else, a missing recorded id,
or a failed lookup keeps the record blocked and says why. The admin entry
points (``workflow-reconcile --dry-run`` and ``--retire-departed-agents``) are
driven end to end with the Pod script and ``aws``/``kubectl`` doubled.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from gpu_fault.admin import workflow_reconcile as reconcile
from gpu_fault.admin import workflow_reconcile_departed_agents as departed
from gpu_fault.admin import workflow_reconcile_departed_nodes as departed_nodes
from gpu_fault.admin import workflow_reconcile_never_dispatched as never_dispatched
from gpu_fault.admin.bootstrap_common import BootstrapError
from gpu_fault.workflow_resolution import DEPARTED_NODE_PROOF_REASON

NODE = "ip-10-0-33-205.us-west-2.compute.internal"
INSTANCE = "i-0a000001"
LIVE_NODE = "ip-10-0-40-17.us-west-2.compute.internal"
LIVE_INSTANCE = "i-0a000002"
UID_NODE = "ip-10-0-50-1.us-west-2.compute.internal"
PENDING_ID = "workflow-pending-xid46"
MISSING = {
    "node_id": NODE,
    "exists": False,
    "restored": False,
    "blockers": ["node is missing"],
}


def _site(tmp_path: Path, **cluster_overrides: object) -> SimpleNamespace:
    cluster: dict[str, object] = {
        "cluster_id": "gpu-a",
        "context": "gpu-a-context",
        "provider": "eks-managed",
        "hyperpod_cluster_name": "",
        "region": "us-west-2",
    }
    cluster.update(cluster_overrides)
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


def _target(tmp_path: Path, **overrides: object) -> dict[str, Any]:
    return _site(tmp_path, **overrides).release_config["clusters"][0]


class Commands:
    """``run_command`` double: the GPU node list and ``ec2 describe-instances``."""

    def __init__(
        self,
        *,
        states: dict[str, str] | None = None,
        aws_error: str | None = None,
        kubernetes_nodes: tuple[str, ...] = (LIVE_NODE,),
    ) -> None:
        self.states = {LIVE_INSTANCE: "running", **(states or {})}
        self.aws_error = aws_error
        self.kubernetes_nodes = kubernetes_nodes
        self.calls: list[list[str]] = []

    def __call__(self, command: list[str], **_options: object) -> SimpleNamespace:
        self.calls.append(list(command))
        if command[0] == "kubectl":
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
            "ec2",
            "describe-instances",
            "--region",
            "us-west-2",
        ], command
        assert "sagemaker" not in command, "an eks-managed cluster never asks SageMaker"
        if self.aws_error is not None:
            return SimpleNamespace(returncode=254, stdout="", stderr=self.aws_error)
        values = command[command.index("--filters") + 1].removeprefix(
            "Name=instance-id,Values="
        )
        rows = [
            [instance_id, self.states[instance_id]]
            for instance_id in values.split(",")
            if instance_id in self.states
        ]
        return SimpleNamespace(returncode=0, stdout=json.dumps(rows), stderr="")

    @property
    def aws_calls(self) -> list[list[str]]:
        return [call for call in self.calls if call[0] == "aws"]


def _pod(instance_ids: dict[str, str | None], plan: dict[str, Any] | None = None):
    """``run_control_plane_script`` double: the runtime plan and Agent instance ids."""

    calls: list[dict[str, Any]] = []

    def run(_site: object, payload: dict[str, Any], **kwargs: object) -> dict[str, Any]:
        calls.append(payload)
        if payload["mode"] == "plan":
            assert plan is not None, "no runtime plan was prepared for this test"
            return plan
        assert payload["mode"] == "agent-instance-ids", payload
        assert "store.get_agent" in str(kwargs.get("script"))
        return {
            "mode": "agent-instance-ids",
            "instance_ids": {
                node: instance_ids[node]
                for node in payload["node_ids"]
                if node in instance_ids
            },
        }

    run.calls = calls  # type: ignore[attr-defined]
    return run


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
            "scanned": 1,
            "selected": 1,
            "remaining": 0,
            "scan_truncated": False,
        },
        "items": list(items),
    }


def _pending_item(**overrides: object) -> dict[str, object]:
    item: dict[str, object] = {
        "request_id": PENDING_ID,
        "incident_id": "incident-xid-46",
        "cluster_id": "gpu-a",
        "node_ids": [NODE],
        "incident_state": "ACTION_PENDING",
        "fencing_token": 2,
        "execution_epoch": 0,
        "workflow_created_at": "2026-09-30T06:00:00+00:00",
        "workflow_updated_at": "2026-09-30T06:00:00+00:00",
        "step_execution_count": 0,
        "remote_command_count": 0,
        "terminalization": never_dispatched.NEVER_DISPATCHED,
        "eligible": False,
        "reasons": [DEPARTED_NODE_PROOF_REASON],
    }
    item.update(overrides)
    return item


# ------------------------------------------------------------- the EC2 read


def test_ec2_instance_states_reads_by_filter_and_omits_unknown_ids() -> None:
    commands = Commands(states={INSTANCE: "terminated"})

    states = departed_nodes.ec2_instance_states(
        "us-west-2", [INSTANCE, "i-0000dead"], run_command=commands
    )

    assert states == {INSTANCE: "terminated"}
    [call] = commands.aws_calls
    assert call[call.index("--filters") + 1] == (
        f"Name=instance-id,Values=i-0000dead,{INSTANCE}"
    )
    assert call[call.index("--query") + 1] == (
        "Reservations[].Instances[].[InstanceId,State.Name]"
    )
    with pytest.raises(BootstrapError, match="cannot describe EC2 instances"):
        departed_nodes.ec2_instance_states(
            "us-west-2", [INSTANCE], run_command=Commands(aws_error="AccessDenied")
        )
    assert (
        departed_nodes.ec2_instance_states("us-west-2", [], run_command=commands) == {}
    )


# --------------------------------------------------------- the node verdict


@pytest.mark.parametrize(
    "state,restored",
    [("terminated", True), ("not-found", True), ("running", False), ("stopped", False)],
)
def test_departed_node_evidence_restores_only_a_terminated_or_unknown_instance(
    tmp_path: Path, state: str, restored: bool
) -> None:
    commands = Commands(states={} if state == "not-found" else {INSTANCE: state})
    read_at: dict[str, str] = {}

    evidence = departed_nodes.departed_node_evidence(
        _target(tmp_path),
        "gpu-a",
        NODE,
        instance_id=INSTANCE,
        aws_region="us-west-2",
        missing=MISSING,
        states={},
        read_at=read_at,
        run_command=commands,
    )

    assert evidence["node_id"] == NODE
    assert evidence["exists"] is False
    assert evidence["provider"] == "eks-managed"
    assert evidence["instance_id"] == INSTANCE
    assert evidence["ec2_state"] == state
    assert evidence["restored"] is restored
    assert evidence["absent_from_provider"] is restored
    if restored:
        assert evidence["blockers"] == []
    else:
        assert evidence["blockers"] == [
            f"node is missing from Kubernetes but EC2 instance {INSTANCE} is {state}"
        ]
    assert set(read_at) == {"ec2"}
    assert len(commands.aws_calls) == 1


@pytest.mark.parametrize("hint", [None, "", "a2c4e6f8-node-uid"])
def test_departed_node_evidence_without_a_recorded_instance_id_stays_blocked(
    tmp_path: Path, hint: str | None
) -> None:
    commands = Commands()

    evidence = departed_nodes.departed_node_evidence(
        _target(tmp_path),
        "gpu-a",
        NODE,
        instance_id=hint,
        aws_region="us-west-2",
        missing=MISSING,
        states={},
        read_at=None,
        run_command=commands,
    )

    assert evidence["restored"] is False
    assert evidence["provider"] == "eks-managed"
    assert evidence["blockers"][0] == "node is missing"
    assert "the fleet recorded no EC2 instance id" in evidence["blockers"][1]
    assert commands.aws_calls == [], "a Node UID is never sent to EC2 as an instance id"


def test_departed_node_evidence_reports_a_failed_lookup_once_per_cluster(
    tmp_path: Path,
) -> None:
    commands = Commands(aws_error="UnauthorizedOperation")
    states: dict[str, dict[str, str]] = {}
    verdicts = [
        departed_nodes.departed_node_evidence(
            _target(tmp_path),
            "gpu-a",
            node,
            instance_id=instance,
            aws_region="us-west-2",
            missing={**MISSING, "node_id": node},
            states=states,
            read_at=None,
            run_command=commands,
        )
        for node, instance in ((NODE, INSTANCE), (LIVE_NODE, LIVE_INSTANCE))
    ]

    for evidence in verdicts:
        assert evidence["restored"] is False
        assert evidence["blockers"][1].startswith(
            "provider membership unknown: cannot describe EC2"
        ), evidence
    assert len(commands.aws_calls) == 1, "one failed read is enough for the plan"


def test_departed_node_evidence_needs_a_region(tmp_path: Path) -> None:
    commands = Commands()

    evidence = departed_nodes.departed_node_evidence(
        _target(tmp_path, region=""),
        "gpu-a",
        NODE,
        instance_id=INSTANCE,
        aws_region="",
        missing=MISSING,
        states={},
        read_at=None,
        run_command=commands,
    )

    assert evidence["restored"] is False
    assert "no AWS region in the site" in evidence["blockers"][1]
    assert commands.aws_calls == []


def test_hyperpod_clusters_keep_the_list_cluster_nodes_proof(tmp_path: Path) -> None:
    calls: list[list[str]] = []

    def run_command(command: list[str], **_options: object) -> SimpleNamespace:
        calls.append(list(command))
        assert command[:3] == ["aws", "sagemaker", "list-cluster-nodes"], command
        return SimpleNamespace(
            returncode=0, stdout=json.dumps({"ClusterNodeSummaries": []}), stderr=""
        )

    target = _target(
        tmp_path, provider="hyperpod", hyperpod_cluster_name="gpu-a-hyperpod"
    )
    read_at: dict[str, str] = {}

    evidence = departed_nodes.hyperpod_departed_node_evidence(
        _site(tmp_path),
        target,
        "gpu-a",
        "hyperpod-i-0a000001",
        missing={**MISSING, "node_id": "hyperpod-i-0a000001"},
        provider_inventories={},
        read_at=read_at,
        run_command=run_command,
    )

    assert evidence["provider"] == "hyperpod"
    assert evidence["restored"] is True
    assert set(read_at) == {"hyperpod"}
    assert len(calls) == 1
    assert departed_nodes.cluster_provider_of(target) == "hyperpod"
    assert departed_nodes.cluster_provider_of({}) == "hyperpod"
    assert departed_nodes.cluster_provider_of(_target(tmp_path)) == "eks-managed"


def test_never_dispatched_evidence_names_ec2_as_the_provider_source(
    tmp_path: Path,
) -> None:
    departed_evidence = {
        "node_id": NODE,
        "exists": False,
        "provider": "eks-managed",
        "instance_id": INSTANCE,
        "ec2_state": "terminated",
        "absent_from_provider": True,
        "restored": True,
        "blockers": [],
    }

    evidence = never_dispatched.node_evidence(
        NODE,
        departed=departed_evidence,
        target=_target(tmp_path),
        aws_region="us-west-2",
    )

    assert evidence["absent_from_kubernetes"] is True
    assert evidence["absent_from_provider"] is True
    assert evidence["sources"] == {
        "kubernetes": {"consulted": True, "present": False},
        "ec2": {
            "consulted": True,
            "cluster_name": "gpu-a",
            "region": "us-west-2",
            "listed": False,
            "state": "terminated",
        },
    }
    assert "hyperpod" not in evidence["sources"]


# ----------------------------------------------- the recorded instance ids


def test_recorded_instance_ids_come_from_the_agent_records(tmp_path: Path) -> None:
    run = _pod({NODE: INSTANCE, LIVE_NODE: "node-uid-not-an-instance"})
    items = [
        {"cluster_id": "gpu-a", "node_ids": [NODE, LIVE_NODE]},
        {"cluster_id": "gpu-a", "node_ids": [NODE]},
    ]

    recorded = departed_nodes.RecordedInstanceIds(
        _site(tmp_path), items, cluster_target=reconcile.cluster_target, run_script=run
    )

    assert recorded.get("gpu-a", NODE) == INSTANCE
    assert recorded.get("gpu-a", LIVE_NODE) is None, "a Node UID is not an instance id"
    assert recorded.get("gpu-a", "unknown-node") is None
    assert len(run.calls) == 1, "one Pod read per cluster"
    assert run.calls[0]["node_ids"] == sorted([NODE, LIVE_NODE])

    hyperpod_site = _site(tmp_path, provider="hyperpod", hyperpod_cluster_name="hp")
    untouched = departed_nodes.RecordedInstanceIds(
        hyperpod_site, items, cluster_target=reconcile.cluster_target, run_script=run
    )
    assert untouched.get("gpu-a", NODE) is None
    assert len(run.calls) == 1, "HyperPod clusters never ask for agent instance ids"


def test_recorded_instance_ids_treat_a_failed_pod_read_as_unknown(
    tmp_path: Path,
) -> None:
    def failing(
        _site: object, _payload: dict[str, Any], **_kwargs: object
    ) -> dict[str, Any]:
        raise BootstrapError("no Running CPU ingress Pod")

    recorded = departed_nodes.RecordedInstanceIds(
        _site(tmp_path),
        [{"cluster_id": "gpu-a", "node_ids": [NODE]}],
        cluster_target=reconcile.cluster_target,
        run_script=failing,
    )

    assert recorded.get("gpu-a", NODE) is None
    with pytest.raises(BootstrapError, match="invalid"):
        departed_nodes.recorded_instance_ids(
            _site(tmp_path),
            "gpu-a",
            [NODE],
            run_script=lambda *_a, **_k: {"instance_ids": "nope"},
        )


# ------------------------------------------------- the administrator paths


def test_workflow_reconcile_proves_a_never_dispatched_node_gone_through_ec2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    commands = Commands(states={INSTANCE: "terminated"})
    pod = _pod({NODE: INSTANCE}, _runtime_plan(_pending_item()))
    monkeypatch.setattr(reconcile, "_run_reconcile", pod)
    monkeypatch.setattr(reconcile, "run_command", commands)

    plan = reconcile.run_workflow_reconcile(
        _site(tmp_path), tmp_path, workflow_ids=(PENDING_ID,), dry_run=True
    )

    [item] = plan["items"]
    assert item["eligible"] is True, item["reasons"]
    [node] = item["scheduling_evidence"]["nodes"]
    assert node["provider"] == "eks-managed"
    assert node["instance_id"] == INSTANCE
    assert node["absent_from_kubernetes"] is True
    assert node["absent_from_provider"] is True
    assert node["sources"]["ec2"] == {
        "consulted": True,
        "cluster_name": "gpu-a",
        "region": "us-west-2",
        "listed": False,
        "state": "terminated",
    }
    assert sorted(item["evidence_read_at"]) == ["ec2", "kubernetes"]
    assert [call["mode"] for call in pod.calls] == ["plan", "agent-instance-ids"]
    assert len(commands.aws_calls) == 1
    [entry] = never_dispatched.bridge_entry(item)["departed_node_evidence"]
    assert entry["provider"] == "eks-managed"
    assert entry["instance_id"] == INSTANCE
    assert entry["sources"]["ec2"]["cluster_name"] == "gpu-a"


def test_workflow_reconcile_keeps_a_still_running_instance_blocked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    commands = Commands(states={INSTANCE: "running"})
    monkeypatch.setattr(
        reconcile,
        "_run_reconcile",
        _pod({NODE: INSTANCE}, _runtime_plan(_pending_item())),
    )
    monkeypatch.setattr(reconcile, "run_command", commands)

    plan = reconcile.run_workflow_reconcile(
        _site(tmp_path), tmp_path, workflow_ids=(PENDING_ID,), dry_run=True
    )

    [item] = plan["items"]
    assert item["eligible"] is False
    [node] = item["scheduling_evidence"]["nodes"]
    assert node["absent_from_provider"] is False
    assert node["sources"]["ec2"]["listed"] is True
    assert node["sources"]["ec2"]["state"] == "running"
    assert DEPARTED_NODE_PROOF_REASON in item["reasons"], (
        "an unproven departure keeps the Pod's own verdict and reasons"
    )
    assert item["scheduling_evidence"]["blockers"] == [
        f"{NODE}: node is missing from Kubernetes but EC2 instance {INSTANCE} is running"
    ]


def test_workflow_reconcile_without_an_agent_record_cannot_prove_departure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    commands = Commands(states={INSTANCE: "terminated"})
    monkeypatch.setattr(
        reconcile, "_run_reconcile", _pod({}, _runtime_plan(_pending_item()))
    )
    monkeypatch.setattr(reconcile, "run_command", commands)

    plan = reconcile.run_workflow_reconcile(
        _site(tmp_path), tmp_path, workflow_ids=(PENDING_ID,), dry_run=True
    )

    [item] = plan["items"]
    assert item["eligible"] is False
    assert DEPARTED_NODE_PROOF_REASON in item["reasons"], item["reasons"]
    blockers = item["scheduling_evidence"]["blockers"]
    assert len(blockers) == 2, blockers
    assert "the fleet recorded no EC2 instance id" in blockers[1]
    assert commands.aws_calls == [], "nothing to ask EC2 about"


def test_retire_departed_agents_plans_with_the_agent_record_instance_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    agents = [
        {
            "cluster_id": "gpu-a",
            "node_id": node_id,
            "lifecycle_state": "ACTIVE",
            "generation": 4,
            "node_instance_id": instance_id,
            "agent_incarnation_id": f"inc-{node_id}",
            "transition_id": None,
            "first_seen_at": "2026-09-30T10:00:00+00:00",
            "last_seen_at": "2026-10-01T05:00:00+00:00",
            "lease_expires_at": "2026-10-01T05:01:00+00:00",
            "agent_version": "2026.9.10",
            "runtime_profile_version": "eks-v1",
        }
        for node_id, instance_id in (
            (NODE, INSTANCE),
            (LIVE_NODE, LIVE_INSTANCE),
            (UID_NODE, "a2c4e6f8-node-uid"),
        )
    ]
    pod_calls: list[dict[str, Any]] = []

    def run_pod_script(_site: object, payload: dict[str, Any]) -> dict[str, Any]:
        pod_calls.append(payload)
        assert payload["mode"] == "list-agents"
        return {
            "mode": "agent-inventory",
            "evaluated_at": "2026-10-01T07:00:00+00:00",
            "agents": agents,
        }

    commands = Commands(states={INSTANCE: "terminated"})
    monkeypatch.setattr(departed, "run_pod_script", run_pod_script)
    monkeypatch.setattr(reconcile, "run_command", commands)

    plan = departed.run_retire_departed_agents(_site(tmp_path), tmp_path, dry_run=True)

    by_node = {item["node_id"]: item for item in plan["items"]}
    gone = by_node[NODE]
    assert gone["eligible"] is True, gone["reasons"]
    assert gone["node_evidence"]["provider"] == "eks-managed"
    assert gone["node_evidence"]["instance_id"] == INSTANCE
    assert gone["node_evidence"]["sources"]["ec2"]["state"] == "terminated"
    assert "ec2" in gone["evidence_read_at"]
    live = by_node[LIVE_NODE]
    assert live["eligible"] is False
    assert live["reasons"] == [f"{LIVE_NODE}: {departed.KUBERNETES_PRESENT_REASON}"]
    uid_only = by_node[UID_NODE]
    assert uid_only["eligible"] is False
    assert any(
        "recorded no EC2 instance id" in reason for reason in uid_only["reasons"]
    ), uid_only["reasons"]
    assert len(commands.aws_calls) == 1, "one describe-instances per departed instance"
    assert [call["mode"] for call in pod_calls] == ["list-agents"]

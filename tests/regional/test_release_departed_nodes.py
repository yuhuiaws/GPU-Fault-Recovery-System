"""A node the provider reclaimed mid-release stops blocking the release gates.

Live 2026-10-06: a HyperPod spot node left between two fleet waves. The
rollback snapshot still named four nodes, so every resumed rollback ended in
``rollback Agent heartbeat mismatch``; the never-dispatched PENDING workflow
the product had opened for the node blocked wave 2 and the automatic rollback.
These cases drive that exact shape -- four recorded, one departed, three
converged -- through the rollback verification and the wave gate, and pin the
proof fail-closed: a NotReady node, a still-listed instance, a running EC2
instance, a live Agent lease and a failed lookup all keep the node expected.
"""

from __future__ import annotations

import io
import json
import runpy
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from gpu_fault.app import ApplicationContext
from gpu_fault_release import regional_release_departed_nodes as departed
from gpu_fault_release import regional_release_fleet_rollout as fleet
from gpu_fault_release import regional_release_validation as validation
from gpu_fault_release.regional_release_config import ClusterTarget, ReleaseError
from tests.regional._cov95_release_support import ResourceRelease, previous_snapshot

PROBE = Path(__file__).resolve().parents[2] / (
    "deploy/control-plane/regional/probes/agent_lifecycle_facts.py"
)
# Three nodes that stayed and the one HyperPod reclaimed (synthetic ids).
LIVE = (
    "hyperpod-i-000000000000a1",
    "hyperpod-i-000000000000a2",
    "hyperpod-i-000000000000a3",
)
GONE = "hyperpod-i-0000000000000000"
GONE_INSTANCE = "i-0000000000000000"
EKS_INSTANCE = "i-00000000000000001"
PAST = "2026-10-06T23:20:00+00:00"
REVOKED = {
    "lifecycle_state": "REVOKED",
    "lease_expires_at": PAST,
    "last_seen_at": PAST,
    "lease_live": False,
    "node_instance_id": None,
}


def node_item(
    release: ResourceRelease, name: str, *, labels: dict[str, str], ready: bool = True
) -> dict[str, Any]:
    return {
        "metadata": {
            "name": name,
            "uid": f"{name}-uid",
            "labels": labels,
            "annotations": {
                "gpu-fault.io/installer-state": "Succeeded",
                "gpu-fault.io/installer-node-uid": f"{name}-uid",
                "gpu-fault.io/installer-artifact-sha256": release.node_wheel_sha,
                "gpu-fault.io/installer-config-digest": release.config.agent_config_digest,
                "gpu-fault.io/installer-bundle-sha256": release.bundle_sha,
                "gpu-fault.io/installer-template-sha256": release.node_template_sha,
            },
        },
        "status": {
            "conditions": [{"type": "Ready", "status": "True" if ready else "False"}]
        },
    }


def aws_handler(
    *, listed: tuple[str, ...] = (), ec2: dict[str, str] | None = None
) -> Any:
    def handle(arguments: list[str], _kwargs: dict[str, Any]) -> str:
        if arguments[:3] == ["aws", "sagemaker", "list-cluster-nodes"]:
            return json.dumps(
                {"ClusterNodeSummaries": [{"InstanceId": item} for item in listed]}
            )
        if arguments[:3] == ["aws", "ec2", "describe-instances"]:
            return json.dumps([[key, value] for key, value in (ec2 or {}).items()])
        raise AssertionError(f"unexpected command: {arguments[:4]}")

    return handle


def lifecycle_probe(
    monkeypatch: pytest.MonkeyPatch, agents: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    """Answer the ``agent_lifecycle_facts`` exec and record what was asked."""

    requests: list[dict[str, Any]] = []

    def exec_probe(_release: Any, **kwargs: Any) -> str:
        requests.append(json.loads(kwargs["input_text"]))
        return json.dumps({"read_at": PAST, "agents": agents})

    monkeypatch.setattr(departed, "exec_cpu_ingress_probe", exec_probe)
    return requests


def hyperpod_fixture(
    monkeypatch: pytest.MonkeyPatch,
    *,
    present: tuple[str, ...] = LIVE,
    recorded: tuple[str, ...] = (*LIVE, GONE),
    listed: tuple[str, ...] = tuple(name.removeprefix("hyperpod-") for name in LIVE),
    agents: dict[str, dict[str, Any]] | None = None,
    not_ready: tuple[str, ...] = (),
) -> tuple[ResourceRelease, dict[str, Any], ClusterTarget, list[dict[str, Any]]]:
    release = ResourceRelease(("gpu-a",))
    target = release.config.clusters[0]
    previous = previous_snapshot(release)
    previous["agent_identities"]["gpu-a"]["node_ids"] = list(recorded)
    labels = {"sagemaker.amazonaws.com/cluster-name": target.hyperpod_cluster_name}
    release.documents[("gpu-a", "nodes", "")] = {
        "items": [
            node_item(release, name, labels=labels, ready=name not in not_ready)
            for name in present
        ]
    }
    release.runner.handler = aws_handler(listed=listed)
    requests = lifecycle_probe(
        monkeypatch, {GONE: REVOKED} if agents is None else agents
    )
    return release, previous, target, requests


def test_rollback_verification_drops_a_provably_departed_node(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    release, previous, target, requests = hyperpod_fixture(monkeypatch)

    validation.validate_agent_rollback_target(release, previous, target)

    check = release.heartbeat_checks[-1]
    assert (check["node_count"], check["node_names"]) == (3, tuple(sorted(LIVE)))
    assert requests == [{"cluster_id": "gpu-a", "node_ids": [GONE]}]
    inventory_reads = [
        arguments
        for arguments, _kwargs in release.runner.calls
        if arguments[:3] == ["aws", "sagemaker", "list-cluster-nodes"]
    ]
    assert len(inventory_reads) == 1, release.runner.calls
    assert inventory_reads[0][3:8] == [
        "--cluster-name",
        target.hyperpod_cluster_name,
        "--region",
        target.region,
        "--output",
    ]
    entry = release.state["departed_nodes"]["gpu-a"]
    assert (entry["release_id"], entry["phase"]) == (
        release.release_id,
        "rollback-verifying",
    )
    (row,) = entry["nodes"]
    assert (row["node_id"], row["provider"], row["instance_id"]) == (
        GONE,
        "hyperpod",
        GONE_INSTANCE,
    )
    assert row["kubernetes"]["present"] is False
    assert (
        row["provider_inventory"]["listed"],
        row["provider_inventory"]["cluster_name"],
    ) == (False, target.hyperpod_cluster_name)
    assert row["agent"]["lifecycle_state"] == "REVOKED"
    assert row["retire_hint"] is None
    assert departed.recorded_departed_nodes(
        {**release.state, "release_id": release.release_id}, "gpu-a"
    ) == (GONE,)


def test_an_expired_active_record_is_departed_and_names_the_retire_verb(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    silent = {**REVOKED, "lifecycle_state": "ACTIVE"}
    release, previous, target, _requests = hyperpod_fixture(
        monkeypatch, agents={GONE: silent}
    )

    validation.validate_agent_rollback_target(release, previous, target)

    assert release.heartbeat_checks[-1]["node_count"] == 3
    (row,) = release.state["departed_nodes"]["gpu-a"]["nodes"]
    assert row["agent"]["lifecycle_state"] == "ACTIVE"
    assert f"--retire-departed-agents --node {GONE}" in row["retire_hint"]


@pytest.mark.parametrize(
    "fault,problem",
    [
        ("live-lease", "ACTIVE with a live lease"),
        ("listed", "still lists instance"),
        ("lookup-failed", "provider membership unknown: cannot list HyperPod nodes"),
        ("no-instance-id", "carries no HyperPod instance id"),
        ("no-record", "still lists instance"),
    ],
)
def test_an_unproven_departure_refuses_and_names_the_node(
    monkeypatch: pytest.MonkeyPatch, fault: str, problem: str
) -> None:
    gone = "node-x" if fault == "no-instance-id" else GONE
    agents: dict[str, dict[str, Any]] = {gone: REVOKED}
    listed = tuple(name.removeprefix("hyperpod-") for name in LIVE)
    if fault == "live-lease":
        agents = {gone: {**REVOKED, "lifecycle_state": "ACTIVE", "lease_live": True}}
    elif fault in {"listed", "no-record"}:
        listed = (*listed, GONE_INSTANCE)
        if fault == "no-record":
            agents = {}
    release, previous, target, _requests = hyperpod_fixture(
        monkeypatch, recorded=(*LIVE, gone), listed=listed, agents=agents
    )
    if fault == "lookup-failed":

        def refuse(_arguments: list[str], _kwargs: dict[str, Any]) -> str:
            raise ReleaseError("aws sagemaker failed")

        release.runner.handler = refuse

    with pytest.raises(ReleaseError, match=problem) as failure:
        validation.validate_agent_rollback_target(release, previous, target)

    assert str(failure.value).startswith(
        "gpu-a rollback Agent node set names nodes missing from Kubernetes that are "
        f"not provably departed: {gone}: "
    ), str(failure.value)
    assert release.heartbeat_checks == [], "the refusal comes before any heartbeat read"
    assert "departed_nodes" not in release.state


def test_a_not_ready_node_is_still_expected(monkeypatch: pytest.MonkeyPatch) -> None:
    release, previous, target, requests = hyperpod_fixture(
        monkeypatch, present=(*LIVE, GONE), not_ready=(GONE,)
    )
    release.heartbeat_ready = False

    with pytest.raises(ReleaseError, match="rollback Agent heartbeat mismatch"):
        validation.validate_agent_rollback_target(release, previous, target)

    assert release.heartbeat_checks[-1]["node_count"] == 4
    assert requests == [], "a Node that exists is never asked about in the Pod"
    assert release.runner.calls == [], "nor looked up at the provider"


def test_a_snapshot_whose_every_node_departed_verifies_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    release, previous, target, _requests = hyperpod_fixture(
        monkeypatch, present=(), recorded=(GONE,), listed=()
    )

    with pytest.raises(ReleaseError, match="every recorded node departed"):
        validation.validate_agent_rollback_target(release, previous, target)

    assert release.heartbeat_checks == []


def eks_fixture(
    monkeypatch: pytest.MonkeyPatch,
    *,
    ec2: dict[str, str],
    record: dict[str, Any] | None,
) -> tuple[ResourceRelease, dict[str, Any], ClusterTarget]:
    release = ResourceRelease(("gpu-e",))
    base = release.config.clusters[0]
    target = ClusterTarget(
        cluster_id=base.cluster_id,
        context=base.context,
        executor_irsa_role_arn=base.executor_irsa_role_arn,
        region=base.region,
        hyperpod_cluster_name="",
        eks_cluster_arn=base.eks_cluster_arn,
        provider="eks-managed",
        node_selector="gpu-fault.io/gpu-node=true",
    )
    release.config.clusters = (target,)
    previous = previous_snapshot(release)
    gone = "ip-10-0-0-9.ec2.internal"
    live = ("ip-10-0-0-7.ec2.internal", "ip-10-0-0-8.ec2.internal")
    previous["agent_identities"]["gpu-e"]["node_ids"] = [*live, gone]
    labels = {"gpu-fault.io/gpu-node": "true"}
    release.documents[("gpu-e", "nodes", "")] = {
        "items": [node_item(release, name, labels=labels) for name in live]
    }
    release.runner.handler = aws_handler(ec2=ec2)
    lifecycle_probe(monkeypatch, {} if record is None else {gone: record})
    return release, previous, target


@pytest.mark.parametrize("state", ["terminated", "not-found"])
def test_an_eks_node_departs_when_its_recorded_instance_is_gone(
    monkeypatch: pytest.MonkeyPatch, state: str
) -> None:
    release, previous, target = eks_fixture(
        monkeypatch,
        ec2={EKS_INSTANCE: "terminated"} if state == "terminated" else {},
        record={**REVOKED, "node_instance_id": EKS_INSTANCE},
    )

    validation.validate_agent_rollback_target(release, previous, target)

    assert release.heartbeat_checks[-1]["node_count"] == 2
    (row,) = release.state["departed_nodes"]["gpu-e"]["nodes"]
    assert (row["provider"], row["instance_id"]) == ("eks-managed", EKS_INSTANCE)
    assert (
        row["provider_inventory"]["source"],
        row["provider_inventory"]["state"],
    ) == ("ec2", state)
    describe = [
        arguments
        for arguments, _kwargs in release.runner.calls
        if arguments[:3] == ["aws", "ec2", "describe-instances"]
    ]
    assert len(describe) == 1 and "--instance-ids" not in describe[0], describe


@pytest.mark.parametrize(
    "ec2,record,problem",
    [
        (
            {EKS_INSTANCE: "running"},
            {**REVOKED, "node_instance_id": EKS_INSTANCE},
            f"EC2 instance {EKS_INSTANCE} is running",
        ),
        ({}, REVOKED, "recorded no EC2 instance id"),
        ({}, None, "recorded no EC2 instance id"),
    ],
)
def test_an_eks_node_without_a_terminated_instance_stays_expected(
    monkeypatch: pytest.MonkeyPatch,
    ec2: dict[str, str],
    record: dict[str, Any] | None,
    problem: str,
) -> None:
    release, previous, target = eks_fixture(monkeypatch, ec2=ec2, record=record)

    with pytest.raises(ReleaseError, match=problem):
        validation.validate_agent_rollback_target(release, previous, target)

    assert release.heartbeat_checks == []


def wave_snapshot(entries: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "open_remote": {},
        "destructive_workflow_count": len(entries),
        "destructive_workflows": [str(item.get("request_id")) for item in entries],
        "never_dispatched_destructive_workflows": entries,
        "agent_blocker_count": 0,
        "agent_blockers": [],
        "minimum_lease_remaining_seconds": 60,
    }


def test_wave_gate_sets_aside_a_never_dispatched_record_of_a_departed_node(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    release, _previous, target, requests = hyperpod_fixture(monkeypatch)
    snapshot = wave_snapshot(
        [
            {
                "request_id": "workflow-00000001",
                "incident_id": "incident-1",
                "cluster_id": "gpu-a",
                "node_ids": [GONE],
                "created_at": PAST,
            }
        ]
    )
    monkeypatch.setattr(
        fleet, "exec_cpu_ingress_probe", lambda *_args, **_kwargs: json.dumps(snapshot)
    )

    decided = fleet.ensure_rollout_wave_safe(
        release, target, wave=(LIVE[0],), node_names=LIVE, timeout_seconds=1
    )

    assert decided == snapshot
    assert requests == [{"cluster_id": "gpu-a", "node_ids": [GONE]}]
    entry = release.state["departed_nodes"]["gpu-a"]
    assert (entry["phase"], [row["node_id"] for row in entry["nodes"]]) == (
        "fleet-wave",
        [GONE],
    )


@pytest.mark.parametrize(
    "fault,problem",
    [
        ("listed", "still lists instance"),
        ("no-nodes", "the incident names no node"),
        ("other-cluster", "is not in this release"),
        ("present", "node is still in Kubernetes"),
    ],
)
def test_wave_gate_keeps_blocking_on_an_unproven_departure(
    monkeypatch: pytest.MonkeyPatch, fault: str, problem: str
) -> None:
    listed = tuple(name.removeprefix("hyperpod-") for name in LIVE)
    release, _previous, target, _requests = hyperpod_fixture(
        monkeypatch,
        present=(*LIVE, GONE) if fault == "present" else LIVE,
        listed=(*listed, GONE_INSTANCE) if fault == "listed" else listed,
    )
    entry = {
        "request_id": "workflow-00000001",
        "incident_id": "incident-1",
        "cluster_id": "gpu-b" if fault == "other-cluster" else "gpu-a",
        "node_ids": [] if fault == "no-nodes" else [GONE],
        "created_at": PAST,
    }
    snapshot = wave_snapshot([entry])
    snapshot["destructive_workflows"].append("workflow-00000002")
    snapshot["destructive_workflow_count"] = 2
    monkeypatch.setattr(
        fleet, "exec_cpu_ingress_probe", lambda *_args, **_kwargs: json.dumps(snapshot)
    )

    with pytest.raises(ReleaseError, match="active destructive workflows") as failure:
        fleet.ensure_rollout_wave_safe(
            release, target, wave=(LIVE[0],), node_names=LIVE, timeout_seconds=1
        )

    blocked = json.loads(str(failure.value).split(": ", 1)[1])
    assert blocked["count"] == 2
    assert blocked["request_ids"] == ["workflow-00000001", "workflow-00000002"]
    assert "departed_node_workflows" not in blocked
    assert blocked["departure_unproven"][0].startswith("workflow-00000001: "), blocked
    assert problem in blocked["departure_unproven"][0]


def test_wave_gate_reports_the_records_it_set_aside_beside_the_rest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    release, _previous, target, _requests = hyperpod_fixture(monkeypatch)
    snapshot = wave_snapshot(
        [{"request_id": "workflow-00000001", "cluster_id": "gpu-a", "node_ids": [GONE]}]
    )
    snapshot["destructive_workflows"].append("workflow-00000002")
    snapshot["destructive_workflow_count"] = 2
    monkeypatch.setattr(
        fleet, "exec_cpu_ingress_probe", lambda *_args, **_kwargs: json.dumps(snapshot)
    )

    with pytest.raises(ReleaseError, match="active destructive workflows") as failure:
        fleet.ensure_rollout_wave_safe(
            release, target, wave=(LIVE[0],), node_names=LIVE, timeout_seconds=1
        )

    blocked = json.loads(str(failure.value).split(": ", 1)[1])
    assert (blocked["count"], blocked["request_ids"]) == (1, ["workflow-00000002"])
    assert blocked["departed_node_workflows"] == ["workflow-00000001"]


def test_recorded_proofs_belong_to_one_release() -> None:
    release = ResourceRelease(("gpu-a",))
    target = release.config.clusters[0]
    release.state["departed_nodes"] = {
        "gpu-a": {"release_id": "older", "nodes": [{"node_id": "stale"}]}
    }
    departed.record_departed_nodes(
        release, target, [{"node_id": GONE, "agent": None}], phase="resume"
    )
    departed.record_departed_nodes(
        release, target, [{"node_id": GONE, "agent": REVOKED}], phase="resume"
    )

    entry = release.state["departed_nodes"]["gpu-a"]
    assert [row["node_id"] for row in entry["nodes"]] == [GONE], (
        "an older release's rows are dropped and a node is recorded once"
    )
    assert entry["nodes"][0]["agent"] == REVOKED, "the latest proof wins"
    state = {**release.state, "release_id": release.release_id}
    assert departed.departed_nodes_summary(state) == {
        "departed_nodes": {"gpu-a": entry}
    }
    assert departed.recorded_departed_nodes(state, "gpu-a") == (GONE,)
    assert departed.departed_nodes_summary({**state, "release_id": "next"}) == {}
    assert (
        departed.recorded_departed_nodes({**state, "release_id": "next"}, "gpu-a") == ()
    )
    departed.record_departed_nodes(release, target, [], phase="resume")
    assert release.state["departed_nodes"]["gpu-a"] == entry, (
        "an empty proof records nothing and erases nothing"
    )


def test_agent_lifecycle_facts_probe_reports_every_state(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    now = datetime.now(timezone.utc)
    records = [
        SimpleNamespace(
            node_id=GONE,
            lifecycle_state=SimpleNamespace(value="REVOKED"),
            lease_expires_at=now - timedelta(hours=1),
            last_seen_at=now - timedelta(hours=1),
            node_instance_id=None,
        ),
        SimpleNamespace(
            node_id=LIVE[0],
            lifecycle_state="ACTIVE",
            lease_expires_at=now + timedelta(minutes=1),
            last_seen_at=now,
            node_instance_id=EKS_INSTANCE,
        ),
        SimpleNamespace(
            node_id="unasked",
            lifecycle_state="ACTIVE",
            lease_expires_at=None,
            last_seen_at=now,
        ),
    ]
    asked: list[str] = []

    def list_agents(cluster_id: str) -> list[Any]:
        asked.append(cluster_id)
        return records

    monkeypatch.setattr(
        ApplicationContext,
        "from_environment",
        staticmethod(
            lambda: SimpleNamespace(store=SimpleNamespace(list_agents=list_agents))
        ),
    )
    monkeypatch.setattr(
        "sys.stdin",
        io.StringIO(
            json.dumps({"cluster_id": "gpu-a", "node_ids": [GONE, LIVE[0], "absent"]})
        ),
    )

    runpy.run_path(str(PROBE))

    report = json.loads(capsys.readouterr().out)
    assert asked == ["gpu-a"]
    assert set(report["agents"]) == {GONE, LIVE[0]}
    assert (
        report["agents"][GONE]["lifecycle_state"],
        report["agents"][GONE]["lease_live"],
    ) == ("REVOKED", False)
    assert (
        report["agents"][LIVE[0]]["lifecycle_state"],
        report["agents"][LIVE[0]]["lease_live"],
        report["agents"][LIVE[0]]["node_instance_id"],
    ) == ("ACTIVE", True, EKS_INSTANCE)
    datetime.fromisoformat(report["read_at"])

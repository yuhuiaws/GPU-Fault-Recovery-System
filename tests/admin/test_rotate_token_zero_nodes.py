"""``rotate-token`` of a GPU cluster without nodes (``rotate_token_zero_nodes``).

The node roll stopped on the empty node list ("has no GPU nodes for fleet
rollout") after the registry overlap, the connection Secret and the data plane
had moved to the new token, and the open journal refused every deploy of the
site. The roll is now skipped and recorded on the release engine's zero-node
proof; a cluster with nodes, or whose node list cannot be read, rolls -- and
refuses -- as before.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from gpu_fault.admin import rotate_token as module
from gpu_fault.admin import rotate_token_zero_nodes as zero_nodes
from gpu_fault_release import regional_release_token_rotation_safety as safety
from gpu_fault_release.regional_release_config import ClusterTarget, ReleaseError
from tests.admin.test_admin_rotate_token import Harness
from tests.regional._parked_cluster_support import OWNERSHIP, cpu_nodegroup, nodegroup

EKS_ARN = "arn:aws:eks:us-west-2:123456789012:cluster/gpu-eks-cluster"
EKS = ClusterTarget(
    cluster_id="gpu-eks-cluster",
    context="gpu-eks-context",
    executor_irsa_role_arn="arn:aws:iam::123456789012:role/executor",
    region="us-west-2",
    hyperpod_cluster_name="",
    eks_cluster_arn=EKS_ARN,
    provider="eks-managed",
    node_selector=OWNERSHIP,
)


def instance_groups(target: int = 0, current: int = 0) -> dict[str, Any]:
    """``describe-cluster`` of a HyperPod cluster with one GPU instance group."""

    return {
        "InstanceGroups": [
            {
                "InstanceGroupName": "gpu",
                "InstanceType": "ml.p5en.48xlarge",
                "Status": "InService",
                "TargetCount": target,
                "CurrentCount": current,
                "KubernetesConfig": {"CurrentLabels": {}},
            }
        ]
    }


class Cluster:
    """The reads a node roll makes of one GPU cluster and its provider."""

    def __init__(
        self,
        *,
        nodes: list[dict[str, Any]] | Exception | None = None,
        provider: dict[str, Any] | None = None,
        nodegroups: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        self.nodes = [] if nodes is None else nodes
        self.provider = instance_groups() if provider is None else provider
        self.nodegroups = (
            {"gpu-group": nodegroup("gpu-group"), "cpu-group": cpu_nodegroup()}
            if nodegroups is None
            else nodegroups
        )
        self.commands: list[list[str]] = []

    def get_json(self, arguments: list[str]) -> dict[str, Any]:
        self.commands.append(list(arguments))
        if "nodes" in arguments:
            if isinstance(self.nodes, Exception):
                raise self.nodes
            return {"apiVersion": "v1", "kind": "List", "items": self.nodes}
        raise AssertionError(f"unexpected read {arguments}")

    def run(self, arguments: list[str], **_options: Any) -> str:
        self.commands.append(list(arguments))
        if arguments[:3] == ["aws", "sagemaker", "describe-cluster"]:
            return json.dumps(self.provider)
        if arguments[:3] == ["aws", "eks", "list-nodegroups"]:
            return json.dumps({"nodegroups": sorted(self.nodegroups)})
        if arguments[:3] == ["aws", "eks", "describe-nodegroup"]:
            name = arguments[arguments.index("--nodegroup-name") + 1]
            return json.dumps(self.nodegroups[name])
        raise AssertionError(f"unexpected command {arguments}")

    def probe_output(
        self, arguments: list[str], *, timeout_seconds: float
    ) -> tuple[int, str, str]:
        self.commands.append(list(arguments))
        if "nodepools.karpenter.sh" in arguments:
            return 1, "", 'error: the server doesn\'t have a resource type "nodepools"'
        raise AssertionError(f"unexpected probe {arguments}")

    def release(self) -> SimpleNamespace:
        runner = SimpleNamespace(
            run=self.run, probe_output=self.probe_output, dry_run=False
        )
        return SimpleNamespace(
            runner=runner,
            **{
                "_get_json": self.get_json,
                "_gpu": lambda target, *args: [
                    "kubectl",
                    "--context",
                    target.context,
                    *args,
                ],
            },
        )

    def mutations(self) -> list[list[str]]:
        return [
            command
            for command in self.commands
            if {"annotate", "patch", "create", "delete"} & set(command)
        ]


def hyperpod_target() -> ClusterTarget:
    return ClusterTarget(
        cluster_id="gpu-a",
        context="gpu-a-context",
        executor_irsa_role_arn="arn:aws:iam::123456789012:role/executor",
        region="us-west-2",
        hyperpod_cluster_name="hp-a",
        eks_cluster_arn="arn:aws:eks:us-west-2:123456789012:cluster/gpu-a",
    )


@pytest.fixture
def no_node_read(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """The roll's own node read; it must not be asked of a cluster without nodes."""

    asked: list[str] = []

    def target_node_names(_release: Any, target: Any) -> tuple[str, ...]:
        asked.append(target.cluster_id)
        raise ReleaseError(f"{target.cluster_id} has no GPU nodes for fleet rollout")

    monkeypatch.setattr(module, "target_node_names", target_node_names)
    return asked


@pytest.mark.parametrize(
    ("target", "cluster", "state"),
    [
        (EKS, Cluster(), "parked"),
        (
            EKS,
            Cluster(nodegroups={"gpu-group": nodegroup("gpu-group", desired=2)}),
            "unavailable",
        ),
        (hyperpod_target(), Cluster(), "parked"),
        (hyperpod_target(), Cluster(provider=instance_groups(4, 0)), "unavailable"),
    ],
    ids=["eks-parked", "eks-unavailable", "hyperpod-parked", "hyperpod-unavailable"],
)
def test_a_cluster_without_nodes_skips_its_node_roll_and_records_it(
    no_node_read: list[str],
    capsys: pytest.CaptureFixture[str],
    target: ClusterTarget,
    cluster: Cluster,
    state: str,
) -> None:
    progress: dict[str, Any] = {"completed_nodes": []}
    saves: list[dict[str, Any]] = []

    result = module.roll_node_tokens(
        cluster.release(),
        target,
        rotation_id="r1",
        progress=progress,
        record=lambda: saves.append(json.loads(json.dumps(progress))),
    )

    entry = progress[zero_nodes.ZERO_NODES_KEY]
    assert entry["state"] == state, entry
    assert entry["skipped"] == ["node-waves", "agent-wait"], entry
    assert entry["departed_unrolled"] == [], "a fresh roll had no node to leave"
    assert result == {
        "node_count": 0,
        "waves": [],
        "reinstalled_nodes": [],
        zero_nodes.ZERO_NODES_KEY: entry,
    }, result
    assert saves and saves[-1][zero_nodes.ZERO_NODES_KEY] == entry, (
        "the skip is journaled before the step completes"
    )
    assert no_node_read == [], "the roll's node read is not asked"
    assert cluster.mutations() == [], "no wave ConfigMap, annotation or Agent wait"
    narration = capsys.readouterr().err
    assert "zero-node-gpu-cluster" in narration and "step=rotate-token" in narration, (
        narration
    )
    assert f"state={state}" in narration, narration


def test_a_cluster_with_nodes_rolls_as_before(monkeypatch: pytest.MonkeyPatch) -> None:
    """A selected node: the roll reads its node set as always (here refused by
    the recorded first-wave inventory, which still guards a resume)."""

    cluster = Cluster(nodes=[{"metadata": {"name": "node-1"}}])
    monkeypatch.setattr(
        module, "target_node_names", lambda _release, _target: ("node-1",)
    )
    progress = {"completed_nodes": [], "node_names": ["node-1", "node-2"]}

    with pytest.raises(module.BootstrapError, match="inventory changed"):
        module.roll_node_tokens(
            cluster.release(),
            EKS,
            rotation_id="r1",
            progress=progress,
            record=lambda: None,
        )
    assert zero_nodes.ZERO_NODES_KEY not in progress, "nothing recorded"


def test_an_unreadable_node_list_keeps_the_refusal(no_node_read: list[str]) -> None:
    cluster = Cluster(nodes=RuntimeError("the API server is unreachable"))
    progress: dict[str, Any] = {"completed_nodes": []}

    with pytest.raises(ReleaseError, match="has no GPU nodes for fleet rollout"):
        module.roll_node_tokens(
            cluster.release(),
            EKS,
            rotation_id="r1",
            progress=progress,
            record=lambda: None,
        )
    assert no_node_read == ["gpu-eks-cluster"], "the old path ran and refused"
    assert zero_nodes.ZERO_NODES_KEY not in progress, "nothing recorded"


def test_nodes_that_left_mid_rotation_are_named_and_not_rolled(
    no_node_read: list[str],
) -> None:
    """A forward pass recorded two nodes and rolled one; both left before the
    resume, and a rollback's moved node left too: nothing is left to roll."""

    forward = {"completed_nodes": ["node-1"], "node_names": ["node-1", "node-2"]}
    result = module.roll_node_tokens(
        Cluster().release(), EKS, rotation_id="r1", progress=forward, record=dict
    )
    assert forward[zero_nodes.ZERO_NODES_KEY]["departed_unrolled"] == ["node-2"]
    assert result["reinstalled_nodes"] == ["node-1"], result

    rollback = {"completed_nodes": [], "forward_nodes": ["node-1"]}
    module.roll_node_tokens(
        Cluster().release(),
        EKS,
        rotation_id="r1",
        progress=rollback,
        record=dict,
        only_nodes=frozenset({"node-1"}),
    )
    assert rollback[zero_nodes.ZERO_NODES_KEY]["departed_unrolled"] == ["node-1"]


def test_a_rotation_of_a_parked_cluster_completes_and_blocks_no_deploy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, no_node_read: list[str]
) -> None:
    """The real rotation with its real node roll over a parked HyperPod
    cluster: every step runs, the journal and the summary carry the zero-node
    record, and the deploy preflight's rotation check then passes."""

    roll = module.roll_node_tokens
    harness = Harness(tmp_path, monkeypatch)
    monkeypatch.setattr(module, "roll_node_tokens", roll)
    cluster = Cluster()
    recorder = harness.release.runner.run

    def run(arguments: list[str], **options: Any) -> str:
        if arguments[:1] == ["aws"]:
            return cluster.run(list(arguments))
        return recorder(arguments, **options)

    harness.release.runner.run = run
    harness.release.runner.probe_output = cluster.probe_output
    setattr(harness.release, "_get_json", cluster.get_json)

    summary = harness.rotate()

    assert summary["status"] == module.STATUS_COMPLETED, summary
    assert summary["reinstalled_nodes"] == [], summary
    assert summary[zero_nodes.ZERO_NODES_KEY]["state"] == "parked", summary
    state = harness.state
    evidence = state["steps"][module.STEP_NODES_ROLLED]["evidence"]
    assert evidence[zero_nodes.ZERO_NODES_KEY]["state"] == "parked", evidence
    assert state["node_rollout"][zero_nodes.ZERO_NODES_KEY]["groups"][0]["name"] == (
        "gpu"
    ), state["node_rollout"]
    kinds = [call[0] for call in harness.calls]
    assert kinds == [
        "publish",
        "connection_secret",
        "restart_data_plane",
        "acceptance",
        "write_registry",
        "restart_control_plane",
        "publish",
    ], kinds
    assert not [call for call in harness.release.runner.calls if "annotate" in call], (
        "no node was marked for reinstall"
    )
    assert no_node_read == [], "the roll's node read is not asked"
    release = SimpleNamespace(
        config=SimpleNamespace(admin_state_dir=tmp_path, clusters=[hyperpod_target()])
    )
    assert safety.token_rotation_snapshot(release)["clusters"] == {
        "gpu-a": module.STATUS_COMPLETED
    }, "a completed rotation of a parked cluster blocks no deploy"

"""When a GPU cluster without nodes is parked on purpose, and when it is not.

Zero selected nodes alone never proves a park: the provider must declare the
capacity zero (an EKS managed node group at ``desiredSize`` 0, a HyperPod
instance group at ``TargetCount`` 0) and nothing else -- a Karpenter NodePool
-- may still launch a node the selector selects. Every other shape is an
outage, a replacement in progress or a misconfiguration, and stays a failure
that now says why it is not parked.
"""

from __future__ import annotations

from typing import Any

import pytest

from gpu_fault_release import regional_parked_cluster as PARKED
from gpu_fault_release.regional_release_diff import (
    ReleaseComponent,
    ReleaseExecutionPlan,
)
from tests.regional._parked_cluster_support import (
    CPU_GROUP,
    GROUP,
    LIVE_NODEPOOLS,
    OWNERSHIP,
    SITE,
    cpu_nodegroup,
    nodegroup,
    nodepool,
)

LEGACY_SELECTOR = "nvidia.com/gpu.present=true"


def _eks(
    described: dict[str, dict[str, Any]],
    *,
    selector: str = OWNERSHIP,
    nodepools: Any = LIVE_NODEPOOLS,
) -> dict[str, Any]:
    def read() -> Any:
        if isinstance(nodepools, Exception):
            raise nodepools
        return nodepools

    return PARKED.eks_parked(described, selector=selector, nodepools=read)


def test_the_live_shape_is_parked_and_names_its_evidence() -> None:
    """Live: the only GPU group at min 0 / max 2 / desired 0, a CPU group
    beside it, three unlabelled Karpenter NodePools."""

    proof = _eks({GROUP: nodegroup(), CPU_GROUP: cpu_nodegroup()})

    assert proof["parked"] is True, proof
    assert proof["provider"] == "eks-managed"
    assert proof["groups"] == [
        {"name": GROUP, "status": "ACTIVE", "desired": 0, "min": 0, "max": 2}
    ], "only the group the selector covers is evidence; the CPU group is not"
    assert proof["proven_by"] == [
        "eks:DescribeNodegroup status and scalingConfig",
        "Karpenter NodePools (none)",
    ]


@pytest.mark.parametrize(
    "described,reason",
    [
        (
            {GROUP: nodegroup(desired=2)},
            f"node group {GROUP} wants 2 node(s) (desiredSize 2): its nodes are "
            "lost or still joining",
        ),
        (
            {GROUP: nodegroup(status="UPDATING")},
            f"node group {GROUP} is UPDATING (being created, updated or replaced)",
        ),
        (
            {GROUP: nodegroup(labelled=False), CPU_GROUP: cpu_nodegroup()},
            "the node selector covers no managed node group",
        ),
        (
            {GROUP: nodegroup(), "gpu-second": nodegroup("gpu-second", desired=1)},
            "node group gpu-second wants 1 node(s)",
        ),
    ],
    ids=["outage", "being-replaced", "wrong-selector", "mixed-groups"],
)
def test_zero_nodes_without_a_declared_zero_capacity_is_not_parked(
    described: dict[str, dict[str, Any]], reason: str
) -> None:
    proof = _eks(described)

    assert proof["parked"] is False, proof
    assert reason in proof["reason"], proof["reason"]


def test_a_mixed_cluster_names_only_the_group_that_wants_nodes() -> None:
    proof = _eks({GROUP: nodegroup(), "gpu-second": nodegroup("gpu-second", desired=1)})

    assert GROUP not in proof["reason"], "the group at zero is not the problem"
    assert [item["name"] for item in proof["groups"]] == [GROUP, "gpu-second"]


@pytest.mark.parametrize(
    "pool,supplies",
    [
        (nodepool("labelled", labels={"gpu-fault.io/control-plane": SITE}), True),
        (
            nodepool(
                "required-in",
                requirements=[
                    {
                        "key": "gpu-fault.io/control-plane",
                        "operator": "In",
                        "values": [SITE],
                    }
                ],
            ),
            True,
        ),
        (
            nodepool(
                "required-exists",
                requirements=[
                    {"key": "gpu-fault.io/control-plane", "operator": "Exists"}
                ],
            ),
            True,
        ),
        (
            nodepool(
                "another-site",
                requirements=[
                    {
                        "key": "gpu-fault.io/control-plane",
                        "operator": "In",
                        "values": ["another-site"],
                    }
                ],
            ),
            False,
        ),
        (nodepool("another-label", labels={"gpu-fault.io/control-plane": "x"}), False),
        (nodepool("unlabelled"), False),
    ],
    ids=[
        "template-label",
        "requirement-in",
        "requirement-exists",
        "requirement-other-site",
        "label-other-site",
        "unlabelled",
    ],
)
def test_a_nodepool_blocks_the_park_only_if_its_nodes_could_be_selected(
    pool: dict[str, Any], supplies: bool
) -> None:
    """Template labels and requirements both put labels on a Karpenter node."""

    proof = _eks({GROUP: nodegroup()}, nodepools={"items": [pool]})

    assert proof["parked"] is not supplies, proof
    if supplies:
        assert proof["nodepools"] == [pool["metadata"]["name"]]
        assert "can still launch nodes the node selector selects" in proof["reason"]


def test_a_legacy_runtime_label_selector_is_blocked_by_any_nodepool() -> None:
    """``nvidia.com/gpu.present`` is put on a node at runtime; no template
    says which NodePool's nodes get it."""

    proof = _eks(
        {GROUP: nodegroup()},
        selector=LEGACY_SELECTOR,
        nodepools={"items": [nodepool("unlabelled")]},
    )
    assert proof["parked"] is False, proof
    assert proof["nodepools"] == ["unlabelled"]

    without = _eks({GROUP: nodegroup()}, selector=LEGACY_SELECTOR, nodepools=None)
    assert without["parked"] is True, "no Karpenter CRD: the groups decide"
    assert without["proven_by"][-1] == "Karpenter NodePools (absent)"


def test_an_unreadable_nodepool_list_proves_nothing() -> None:
    proof = _eks({GROUP: nodegroup()}, nodepools=RuntimeError("Forbidden"))

    assert proof["parked"] is False, proof
    assert proof["reason"] == "the Karpenter NodePools could not be read: Forbidden"


# --- HyperPod --------------------------------------------------------------------

HP_CLUSTER = "hp-gpu-a"
HP_SELECTOR = f"{OWNERSHIP},sagemaker.amazonaws.com/cluster-name={HP_CLUSTER}"


def _instance_group(
    name: str, *, target: int, current: int, labelled: bool = True
) -> dict[str, Any]:
    labels = {"gpu-fault.io/control-plane": SITE} if labelled else {}
    return {
        "InstanceGroupName": name,
        "InstanceType": "ml.p5en.48xlarge" if labelled else "ml.g6.12xlarge",
        "CurrentCount": current,
        "TargetCount": target,
        "Status": "InService",
        "KubernetesConfig": {"CurrentLabels": labels},
    }


#: Live: the site's spot group at 4/4 beside an unmanaged group at 0/0.
LIVE_INSTANCE_GROUPS = [
    _instance_group("gpu-spot-group", target=4, current=4),
    _instance_group("unmanaged-g6-group", target=0, current=0, labelled=False),
]


def test_a_hyperpod_group_at_zero_beside_a_populated_one_does_not_park_the_cluster() -> (
    None
):
    """The cluster's own group wants four instances; the unmanaged group at
    0/0 is outside the selector and proves nothing."""

    proof = PARKED.hyperpod_parked(
        LIVE_INSTANCE_GROUPS, cluster_name=HP_CLUSTER, selector=HP_SELECTOR
    )

    assert proof["parked"] is False, proof
    assert proof["groups"] == [
        {"name": "gpu-spot-group", "status": "InService", "target": 4, "current": 4}
    ]
    assert "instance group gpu-spot-group wants 4 instance(s)" in proof["reason"]


def test_a_hyperpod_cluster_whose_covered_groups_are_all_at_zero_is_parked() -> None:
    groups = [
        _instance_group("gpu-spot-group", target=0, current=0),
        LIVE_INSTANCE_GROUPS[1],
    ]

    proof = PARKED.hyperpod_parked(
        groups, cluster_name=HP_CLUSTER, selector=HP_SELECTOR
    )

    assert proof["parked"] is True, proof
    assert proof["proven_by"] == ["sagemaker:DescribeCluster InstanceGroups"]


@pytest.mark.parametrize(
    "group,reason",
    [
        (
            {
                **_instance_group("gpu-spot-group", target=0, current=0),
                "Status": "Updating",
            },
            "instance group gpu-spot-group is Updating",
        ),
        (
            _instance_group("gpu-spot-group", target=0, current=2),
            "still has 2 instance(s) at TargetCount 0",
        ),
        (
            {
                key: value
                for key, value in _instance_group(
                    "gpu-spot-group", target=0, current=0
                ).items()
                if key != "TargetCount"
            },
            "reports no TargetCount/CurrentCount",
        ),
    ],
    ids=["updating", "scale-in-in-progress", "no-target-count"],
)
def test_a_hyperpod_group_not_proven_at_zero_is_not_parked(
    group: dict[str, Any], reason: str
) -> None:
    proof = PARKED.hyperpod_parked(
        [group], cluster_name=HP_CLUSTER, selector=HP_SELECTOR
    )

    assert proof["parked"] is False, proof
    assert reason in proof["reason"], proof["reason"]


def test_the_legacy_cluster_name_selector_covers_every_instance_group() -> None:
    """A cluster joined before the ownership convention selects every group
    of its HyperPod cluster: the unmanaged group at 0/0 alone cannot park it."""

    proof = PARKED.hyperpod_parked(
        LIVE_INSTANCE_GROUPS,
        cluster_name=HP_CLUSTER,
        selector=f"sagemaker.amazonaws.com/cluster-name={HP_CLUSTER}",
    )

    assert proof["parked"] is False, proof
    assert {item["name"] for item in proof["groups"]} == {
        "gpu-spot-group",
        "unmanaged-g6-group",
    }


def test_the_hyperpod_context_query_reads_the_counts_and_the_status() -> None:
    """One ``describe-cluster`` serves the context check, the advisory and the
    proof: no new call, no new permission."""

    for field in (
        "TargetCount:TargetCount",
        "CurrentCount:CurrentCount",
        "Status:Status",
    ):
        assert field in PARKED.HYPERPOD_CONTEXT_QUERY, field


# --- the messages ---------------------------------------------------------------


def test_a_failure_says_why_the_cluster_is_not_parked() -> None:
    not_parked = {"parked": False, "reason": "node group g wants 2 node(s)"}

    assert (
        PARKED.zero_nodes_failure("c has no Kubernetes nodes", not_parked)
        == "c has no Kubernetes nodes; not parked: node group g wants 2 node(s)"
    )
    assert PARKED.zero_nodes_failure("c has no Kubernetes nodes", None) == (
        "c has no Kubernetes nodes"
    ), "a cluster never asked keeps the message it always had"


def _checks(proof: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "name": "regional_contexts",
            "status": "PASS",
            "details": {"providers": {"gpu-eks-cluster": {"parked": proof}}},
        }
    ]


def test_a_data_plane_release_is_refused_while_a_cluster_is_parked() -> None:
    proof = _eks({GROUP: nodegroup()})
    plan = ReleaseExecutionPlan(
        nodes=(
            ReleaseComponent.CPU_STAGE,
            ReleaseComponent.EXECUTOR,
            ReleaseComponent.AGENT,
        )
    )

    refusal = PARKED.release_refusal(_checks(proof), plan)

    assert refusal is not None and refusal["status"] == "FAIL", refusal
    assert refusal["name"] == "parked_gpu_clusters"
    summary = refusal["summary"]
    assert summary.startswith(
        "GPU cluster(s) parked at zero nodes: gpu-eks-cluster;"
    ), summary
    assert "(executor, agent)" in summary, "names the GPU components it changes"
    assert "gpu-fault-admin remove-cluster" in summary, "names both exits"
    assert "let its nodes join" in summary
    assert summary.endswith("Nothing was changed."), summary
    assert refusal["details"]["parked"] == {"gpu-eks-cluster": proof}


@pytest.mark.parametrize(
    "nodes",
    [
        (
            ReleaseComponent.REGISTRY,
            ReleaseComponent.CPU_STAGE,
            ReleaseComponent.CPU_FINALIZE,
        ),
        (ReleaseComponent.OBSERVABILITY, ReleaseComponent.CPU_FINALIZE),
    ],
    ids=["control-plane-only", "observability"],
)
def test_a_control_plane_release_proceeds_over_a_parked_cluster(
    nodes: tuple[ReleaseComponent, ...],
) -> None:
    proof = _eks({GROUP: nodegroup()})

    assert (
        PARKED.release_refusal(_checks(proof), ReleaseExecutionPlan(nodes=nodes))
        is None
    )


def test_an_unproven_cluster_is_not_reported_parked() -> None:
    proof = _eks({GROUP: nodegroup(desired=2)})
    plan = ReleaseExecutionPlan(nodes=(ReleaseComponent.EXECUTOR,))

    assert PARKED.release_refusal(_checks(proof), plan) is None, (
        "a cluster that is not parked fails its own checks instead"
    )
    assert PARKED.parked_cluster_ids(_checks(proof)) == []


def test_the_status_header_names_the_parked_cluster_and_its_groups() -> None:
    health = {"checks": _checks(_eks({GROUP: nodegroup()}))}

    (line,) = PARKED.status_lines(health)

    assert line.startswith(
        "parked GPU cluster gpu-eks-cluster (eks-managed): no node selected; every "
        "managed node group the node selector covers is ACTIVE at desiredSize 0"
    ), line
    assert f"({GROUP}: desiredSize 0, min 0, max 2)" in line
    assert "not a failure" in line and "refused until then" in line

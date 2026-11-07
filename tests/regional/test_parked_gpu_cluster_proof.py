"""When a GPU cluster without nodes is parked on purpose, and when it is not.

Zero selected nodes alone never proves a park: the provider must declare the
capacity zero (an EKS managed node group at ``desiredSize`` 0, a HyperPod
instance group at ``TargetCount`` 0) and nothing else -- a Karpenter NodePool
-- may still launch a node the selector selects. Every other shape is an
outage, a replacement in progress or a misconfiguration: the cluster is
*unavailable*, and every message about it says why it is not parked.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
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
    assert [
        {key: item[key] for key in ("name", "status", "desired", "min", "max")}
        for item in proof["groups"]
    ] == [{"name": GROUP, "status": "ACTIVE", "desired": 0, "min": 0, "max": 2}], (
        "only the group the selector covers is evidence; the CPU group is not"
    )
    (group,) = proof["groups"]
    assert group["instance_types"] == ["p5en.48xlarge"], "what its nodes are born as"
    assert group["labels"]["gpu-fault.io/dcgm-hostengine"] == "host", (
        "the group's declared labels are kept for the template's render node"
    )
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
    assert [
        {key: item[key] for key in ("name", "status", "target", "current")}
        for item in proof["groups"]
    ] == [{"name": "gpu-spot-group", "status": "InService", "target": 4, "current": 4}]
    assert proof["groups"][0]["instance_types"] == ["ml.p5en.48xlarge"]
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


AGENT_WAVES = ReleaseExecutionPlan(
    nodes=(
        ReleaseComponent.CPU_STAGE,
        ReleaseComponent.EXECUTOR,
        ReleaseComponent.RECONCILER,
        ReleaseComponent.AGENT,
    )
)


@pytest.mark.parametrize(
    "plan",
    [
        AGENT_WAVES,
        ReleaseExecutionPlan(nodes=(ReleaseComponent.EXECUTOR,)),
        ReleaseExecutionPlan(
            nodes=(ReleaseComponent.REGISTRY, ReleaseComponent.CPU_FINALIZE)
        ),
    ],
    ids=["agent-waves", "executor", "control-plane-only"],
)
def test_no_release_kind_is_refused_for_zero_nodes_alone(
    plan: ReleaseExecutionPlan,
) -> None:
    """Zero GPU nodes affect only their own cluster: a parked cluster never
    refuses a release, beside a populated cluster or with every cluster at
    zero -- an Agent-wave release's rollback then stages the previous
    release's own Agent pins, there being no live Agent to take them from."""

    proof = _eks({GROUP: nodegroup()})

    assert PARKED.release_refusal(_checks(proof), plan) is None


def test_an_unproven_cluster_is_reported_unavailable_with_its_reason() -> None:
    proof = _eks({GROUP: nodegroup(desired=2)})

    (entry,) = PARKED.zero_node_clusters(_checks(proof)).values()

    assert entry["state"] == "unavailable", entry
    assert entry["reason"] == proof["reason"]
    assert "not_gating_findings" not in entry, "a check that filed none"
    assert PARKED.release_refusal(_checks(proof), AGENT_WAVES) is None, (
        "an outage does not refuse a release, beside a populated cluster or not"
    )


@pytest.mark.parametrize(
    "plan,refused",
    [
        (ReleaseExecutionPlan(nodes=(ReleaseComponent.RECONCILER,)), True),
        (AGENT_WAVES, True),
        (ReleaseExecutionPlan(nodes=(ReleaseComponent.EXECUTOR,)), False),
        (
            ReleaseExecutionPlan(
                nodes=(ReleaseComponent.REGISTRY, ReleaseComponent.CPU_FINALIZE)
            ),
            False,
        ),
    ],
    ids=["reconciler", "agent-waves", "executor-only", "control-plane-only"],
)
@pytest.mark.parametrize("desired", [0, 2], ids=["parked", "unavailable"])
def test_a_reconciler_roll_needs_a_declared_gpu_instance_type(
    plan: ReleaseExecutionPlan, refused: bool, desired: int
) -> None:
    """The template of a cluster without nodes is rendered from a declared
    node: with no GPU instance type the installer knows, a release that rolls
    the Reconciler could neither roll it nor be told so before it started."""

    group = {**nodegroup(desired=desired), "instanceTypes": ["x9.huge"]}
    proof = _eks({GROUP: group})

    refusal = PARKED.release_refusal(_checks(proof), plan)

    assert (refusal is not None) is refused, refusal
    if refusal is not None:
        assert refusal["summary"] == (
            "GPU cluster(s) gpu-eks-cluster (declared: x9.huge) have zero nodes "
            "and no node group declaring a GPU instance type the installer "
            "knows: a release that rolls the Reconciler renders their installer "
            "template from a declared node, and there is none. Scale a group up "
            "and let a node join, or declare a supported instance type on the "
            "group, then deploy again. Nothing was changed."
        )


def test_a_selector_that_covers_no_group_declares_nothing_to_render() -> None:
    proof = _eks({GROUP: nodegroup(labelled=False), CPU_GROUP: cpu_nodegroup()})

    refusal = PARKED.release_refusal(_checks(proof), AGENT_WAVES)

    assert refusal is not None and "(declared: none)" in refusal["summary"], refusal


def test_reports_name_each_parked_cluster_with_its_state_and_groups() -> None:
    proof = _eks({GROUP: nodegroup()})

    (entry,) = PARKED.zero_node_clusters(_checks(proof)).values()

    assert entry["state"] == "parked"
    assert entry["provider"] == "eks-managed"
    assert entry["reason"] == proof["reason"]
    assert entry["groups"] == proof["groups"]


def test_the_status_header_names_the_parked_cluster_and_its_groups() -> None:
    health = {"checks": _checks(_eks({GROUP: nodegroup()}))}

    (line,) = PARKED.status_lines(health)

    assert line.startswith(
        "parked GPU cluster gpu-eks-cluster (eks-managed): no node selected; every "
        "managed node group the node selector covers is ACTIVE at desiredSize 0"
    ), line
    assert f"({GROUP}: desiredSize 0, min 0, max 2)" in line
    assert "not a failure" in line, line
    assert "join with the current release by themselves" in line, line


def test_the_status_header_names_an_unavailable_cluster_and_why() -> None:
    proof = _eks({GROUP: nodegroup(desired=2)})

    (line,) = PARKED.status_lines({"checks": _checks(proof)})

    assert line == (
        f"GPU cluster gpu-eks-cluster unavailable: no node selected; node group "
        f"{GROUP} wants 2 node(s) (desiredSize 2): its nodes are lost or still "
        "joining; the deploy skips its node work"
    )


def test_a_check_that_filed_findings_reads_warn_once() -> None:
    proof = _eks({GROUP: nodegroup(desired=2)})
    checks = _checks(proof)
    checks[0]["details"]["providers"]["gpu-eks-cluster"]["zero_node_findings"] = [
        "gpu-eks-cluster has no Kubernetes nodes; not parked: r"
    ]
    checks.append({"name": "aurora", "status": "FAIL", "summary": "down"})

    fields = PARKED.report_fields(checks)
    again = PARKED.report_fields(fields["checks"])

    contexts, aurora = fields["checks"]
    assert contexts["status"] == "WARN", contexts
    assert contexts["summary"].endswith(
        "; not gating (GPU cluster without nodes, unavailable): gpu-eks-cluster "
        "has no Kubernetes nodes; not parked: r"
    ), contexts["summary"]
    assert aurora == checks[1], "another check is left as it is"
    assert again["checks"] == fields["checks"], "marked once"
    entry = fields["zero_node_clusters"]["gpu-eks-cluster"]
    assert entry["not_gating_findings"] == [
        {
            "check": "regional_contexts",
            "finding": "gpu-eks-cluster has no Kubernetes nodes; not parked: r",
        }
    ]


def test_only_status_gates_the_findings() -> None:
    from types import SimpleNamespace

    release = SimpleNamespace()
    proof = _eks({GROUP: nodegroup(desired=2)})
    section: dict[str, Any] = {}

    PARKED.zero_node_finding(
        release,
        section,
        "c has no Kubernetes nodes",
        proof,
        check="regional_contexts",
        cluster_id="c",
        details={"x": 1},
    )
    assert section["zero_node_findings"] == [
        PARKED.zero_nodes_failure("c has no Kubernetes nodes", proof)
    ]
    with PARKED.gating_findings(release), pytest.raises(PARKED.CheckFailure) as failed:
        PARKED.zero_node_finding(
            release,
            {},
            "c has no Kubernetes nodes",
            proof,
            check="regional_contexts",
            cluster_id="c",
            details={"x": 1},
        )
    assert failed.value.details == {"x": 1}
    assert PARKED.gating(release) is False, "restored after the status"


# --- the render node and the snapshot entry ------------------------------------


def test_the_render_node_carries_the_declared_group() -> None:
    from types import SimpleNamespace

    proof = _eks({GROUP: nodegroup()})
    target = SimpleNamespace(
        cluster_id="gpu-eks-cluster", provider="eks-managed", hyperpod_cluster_name=""
    )

    node = PARKED.render_node(proof, target)

    labels = node["metadata"]["labels"]
    assert labels["node.kubernetes.io/instance-type"] == "p5en.48xlarge"
    assert labels["eks.amazonaws.com/nodegroup"] == GROUP
    assert labels["gpu-fault.io/control-plane"] == SITE
    assert labels["gpu-fault.io/dcgm-hostengine"] == "host"
    assert node["status"]["addresses"] == [{"type": "InternalIP", "address": "0.0.0.0"}]
    environment = PARKED.reconciler_environment(proof, target)
    assert environment["GPU_FAULT_ZERO_NODE_CLUSTER"] == "true"
    assert json.loads(environment["GPU_FAULT_ZERO_NODE_RENDER_NODE"]) == node


def test_a_group_of_an_unknown_gpu_type_cannot_render_a_template() -> None:
    from types import SimpleNamespace

    group = {**nodegroup(), "instanceTypes": ["x9.huge"]}
    proof = _eks({GROUP: group})
    target = SimpleNamespace(
        cluster_id="gpu-eks-cluster", provider="eks-managed", hyperpod_cluster_name=""
    )

    with pytest.raises(PARKED.ReleaseError, match="declare no GPU instance type"):
        PARKED.render_node(proof, target)


def test_a_pinned_template_needs_no_render_node() -> None:
    """A rollback restores the previous template: nothing is rendered, so a
    cluster that declares no known GPU instance type rolls back all the same."""

    from types import SimpleNamespace

    proof = _eks({GROUP: {**nodegroup(desired=2), "instanceTypes": ["x9.huge"]}})
    target = SimpleNamespace(
        cluster_id="gpu-eks-cluster", provider="eks-managed", hyperpod_cluster_name=""
    )

    assert PARKED.reconciler_environment(
        proof, target, {"GPU_FAULT_INSTALLER_TEMPLATE_CONFIG_MAP": "previous"}
    ) == {"GPU_FAULT_ZERO_NODE_CLUSTER": "true", "GPU_FAULT_ZERO_NODE_RENDER_NODE": ""}
    with pytest.raises(PARKED.ReleaseError, match="declare no GPU instance type"):
        PARKED.reconciler_environment(proof, target, {})


def test_the_snapshot_entry_has_no_identity_and_no_node() -> None:
    proof = _eks({GROUP: nodegroup()})

    entry = PARKED.zero_node_identity(proof)

    assert entry["node_ids"] == []
    assert PARKED.is_zero_node_identity(entry), entry
    assert not PARKED.is_zero_node_identity({"node_ids": ["n"], "zero_nodes": {}}), (
        "an entry with a node is a live identity"
    )
    assert PARKED.live_identities({"a": {"node_ids": ["n"]}, "b": entry}) == {
        "a": {"node_ids": ["n"]}
    }


@pytest.mark.parametrize(
    "answer",
    [{}, {"kind": "List"}, {"items": None}, {"items": {}}],
    ids=["empty-document", "no-items", "null-items", "items-not-a-list"],
)
def test_a_node_answer_without_an_items_list_proves_nothing(answer: Any) -> None:
    """Only a list the API returns empty is a cluster without nodes; an answer
    without an ``items`` list leaves the cluster to the node path, which
    refuses an empty node set."""

    reads: list[list[str]] = []
    release = SimpleNamespace(
        **{
            "_gpu": lambda _target, *arguments: ["kubectl", *arguments],
            "_get_json": lambda arguments: reads.append(arguments) or answer,
        }
    )
    target = SimpleNamespace(
        cluster_id="gpu-eks-cluster", provider="eks-managed", node_selector=OWNERSHIP
    )

    assert PARKED.without_nodes(release, target) is None, answer
    assert reads == [["kubectl", "get", "nodes", "-l", OWNERSHIP]], (
        "only the node list is read; no provider document"
    )

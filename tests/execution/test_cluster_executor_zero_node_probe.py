"""The executor's zero-node check (``cluster_executor.zero_node_probe``).

Every minute the executor counts the nodes its selector selects; only when the
API lists none does it read the provider -- HyperPod ``DescribeCluster``, EKS
``ListNodegroups``/``DescribeNodegroup``, the Karpenter NodePools -- and judge
it with the deploy host's rule. The verdict is three gauges and a timestamp
(``GpuFaultGpuClusterUnavailable`` pages on one of them); a node list the API
does not answer changes nothing, and nothing here ever raises into the loop.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any

import pytest

from gpu_fault import zero_node_capacity as capacity
from gpu_fault.cluster_executor.metrics import ClusterExecutorMetrics
from gpu_fault.cluster_executor.zero_node_probe import (
    KARPENTER_VERSIONS,
    ZeroNodeCapacityProbe,
)
from gpu_fault.cluster_scope import ClusterProvider
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

HYPERPOD_SELECTOR = "sagemaker.amazonaws.com/cluster-name=hp-a"


class ApiError(Exception):
    """A Kubernetes ``ApiException`` as far as the probe reads one."""

    def __init__(self, status: int) -> None:
        super().__init__(f"({status}) Reason: refused")
        self.status = status


class Core:
    def __init__(self, nodes: list[str] | Exception) -> None:
        self.nodes = nodes
        self.selectors: list[str] = []

    def list_node(self, *, label_selector: str) -> Any:
        self.selectors.append(label_selector)
        if isinstance(self.nodes, Exception):
            raise self.nodes
        return SimpleNamespace(
            items=[
                SimpleNamespace(metadata=SimpleNamespace(name=n)) for n in self.nodes
            ]
        )


class Custom:
    """``served``: version -> NodePool list; ``None`` is a cluster without the CRD."""

    def __init__(self, served: dict[str, Any] | Exception | None) -> None:
        self.served = served
        self.calls: list[tuple[str, str, str]] = []

    def list_cluster_custom_object(self, group: str, version: str, plural: str) -> Any:
        self.calls.append((group, version, plural))
        if isinstance(self.served, Exception):
            raise self.served
        if self.served is None or version not in self.served:
            raise ApiError(404)
        return self.served[version]


class Eks:
    def __init__(
        self, nodegroups: dict[str, Any] | Exception, *, page: int = 1
    ) -> None:
        self.nodegroups = nodegroups
        self.page = page
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def list_nodegroups(self, **arguments: Any) -> dict[str, Any]:
        self.calls.append(("list_nodegroups", arguments))
        if isinstance(self.nodegroups, Exception):
            raise self.nodegroups
        names = sorted(self.nodegroups)
        start = int(arguments.get("nextToken") or 0)
        answer: dict[str, Any] = {"nodegroups": names[start : start + self.page]}
        if start + self.page < len(names):
            answer["nextToken"] = str(start + self.page)
        return answer

    def describe_nodegroup(self, **arguments: Any) -> dict[str, Any]:
        self.calls.append(("describe_nodegroup", arguments))
        assert isinstance(self.nodegroups, dict), "a failing list is never described"
        return {"nodegroup": self.nodegroups[arguments["nodegroupName"]]}


class SageMaker:
    def __init__(self, groups: list[dict[str, Any]] | Exception) -> None:
        self.groups = groups
        self.calls: list[dict[str, Any]] = []

    def describe_cluster(self, **arguments: Any) -> dict[str, Any]:
        self.calls.append(arguments)
        if isinstance(self.groups, Exception):
            raise self.groups
        return {"ClusterName": "hp-a", "InstanceGroups": self.groups}


def instance_group(
    *, target: int = 0, current: int = 0, status: str = "InService"
) -> dict[str, Any]:
    return {
        "InstanceGroupName": "gpu",
        "InstanceType": "ml.p5en.48xlarge",
        "Status": status,
        "TargetCount": target,
        "CurrentCount": current,
        "ThreadsPerCore": 1,
        "KubernetesConfig": {"CurrentLabels": {}},
    }


def eks_probe(
    nodes: list[str] | Exception = (),  # type: ignore[assignment]
    *,
    nodegroups: dict[str, Any] | Exception | None = None,
    nodepools: dict[str, Any] | Exception | None = None,
    page: int = 1,
) -> ZeroNodeCapacityProbe:
    return ZeroNodeCapacityProbe(
        Core(list(nodes) if not isinstance(nodes, Exception) else nodes),
        provider=ClusterProvider.EKS_MANAGED,
        node_selector=OWNERSHIP,
        eks_cluster="gpu-eks-cluster",
        custom=Custom(nodepools),
        eks_client=Eks(
            {GROUP: nodegroup(), CPU_GROUP: cpu_nodegroup()}
            if nodegroups is None
            else nodegroups,
            page=page,
        ),
        sagemaker_client=SageMaker(AssertionError("no SageMaker on EKS")),
    )


def hyperpod_probe(
    nodes: list[str] | Exception = (),  # type: ignore[assignment]
    *,
    groups: list[dict[str, Any]] | Exception | None = None,
) -> ZeroNodeCapacityProbe:
    return ZeroNodeCapacityProbe(
        Core(list(nodes) if not isinstance(nodes, Exception) else nodes),
        provider=ClusterProvider.HYPERPOD,
        node_selector=HYPERPOD_SELECTOR,
        hyperpod_cluster="hp-a",
        custom=Custom(None),
        sagemaker_client=SageMaker([instance_group()] if groups is None else groups),
        eks_client=Eks(AssertionError("no EKS node groups on HyperPod")),
    )


def gauges(metrics: ClusterExecutorMetrics) -> dict[str, float]:
    names = (
        "selected_nodes",
        "zero_node_parked",
        "zero_node_unavailable",
        "last_zero_node_check_timestamp",
    )
    values = {}
    for line in metrics.family.render().splitlines():
        if line.startswith("#"):
            continue
        name, value = line.split()
        short = name.removeprefix(metrics.family.prefix)
        if short in names:
            values[short] = float(value)
    return values


def test_a_cluster_with_nodes_reads_no_provider() -> None:
    probe = eks_probe(["node-a", "node-b"])
    metrics = ClusterExecutorMetrics()

    assert probe.run(metrics) is None
    values = gauges(metrics)
    assert values["selected_nodes"] == 2, values
    assert values["zero_node_parked"] == values["zero_node_unavailable"] == 0
    assert values["last_zero_node_check_timestamp"] > 0, "the verdict is stamped"
    assert probe.eks.calls == [] and probe.custom.calls == [], "nothing else is read"
    assert probe.core.selectors == [OWNERSHIP], "the selector the executor holds"


@pytest.mark.parametrize(
    "nodepools",
    [None, {"v1": LIVE_NODEPOOLS}, {"v1beta1": {"items": []}}],
    ids=["no-karpenter-crd", "live-nodepools", "v1beta1-only"],
)
def test_an_eks_cluster_parked_at_zero_reads_parked(nodepools: Any) -> None:
    probe = eks_probe(nodepools=nodepools)
    metrics = ClusterExecutorMetrics()

    proof = probe.run(metrics)

    assert proof is not None and proof["parked"] is True, proof
    assert [group["name"] for group in proof["groups"]] == [GROUP], proof
    values = gauges(metrics)
    assert (values["zero_node_parked"], values["zero_node_unavailable"]) == (1, 0)
    assert values["selected_nodes"] == 0
    asked = [version for _group, version, _plural in probe.custom.calls]
    expected = (
        list(KARPENTER_VERSIONS) if nodepools != {"v1": LIVE_NODEPOOLS} else ["v1"]
    )
    assert asked == expected, asked


def test_every_page_of_node_groups_is_read_and_each_group_described() -> None:
    probe = eks_probe(page=1)

    probe.run(ClusterExecutorMetrics())

    pages = [
        arguments for name, arguments in probe.eks.calls if name == "list_nodegroups"
    ]
    assert pages == [
        {"clusterName": "gpu-eks-cluster"},
        {"clusterName": "gpu-eks-cluster", "nextToken": "1"},
    ], pages
    described = sorted(
        arguments["nodegroupName"]
        for name, arguments in probe.eks.calls
        if name == "describe_nodegroup"
    )
    assert described == sorted([GROUP, CPU_GROUP]), described


SUPPLYING = {
    "v1": {
        "items": [
            nodepool(
                "gpu-owned",
                labels={"gpu-fault.io/control-plane": SITE, "node-type": "gpu"},
            )
        ]
    }
}


@pytest.mark.parametrize(
    ("nodegroups", "nodepools", "reason"),
    [
        (
            {GROUP: nodegroup(desired=2)},
            None,
            "wants 2 node(s) (desiredSize 2): its nodes are lost or still joining",
        ),
        (
            {GROUP: nodegroup(status="UPDATING")},
            None,
            "is UPDATING (being created, updated or replaced)",
        ),
        ({CPU_GROUP: cpu_nodegroup()}, None, "covers no managed node group"),
        ({GROUP: nodegroup()}, SUPPLYING, "gpu-owned can still launch nodes"),
        (
            {GROUP: nodegroup()},
            ApiError(403),
            "the Karpenter NodePools could not be read: (403)",
        ),
        (
            RuntimeError("AccessDeniedException: eks:ListNodegroups"),
            None,
            capacity.UNREADABLE_REASON + "AccessDeniedException: eks:ListNodegroups",
        ),
    ],
    ids=[
        "group-wants-nodes",
        "group-updating",
        "selector-covers-no-group",
        "nodepool-supplies",
        "nodepools-forbidden",
        "nodegroups-denied",
    ],
)
def test_an_eks_cluster_that_is_not_parked_reads_unavailable_and_why(
    nodegroups: Any, nodepools: Any, reason: str
) -> None:
    probe = eks_probe(nodegroups=nodegroups, nodepools=nodepools)
    metrics = ClusterExecutorMetrics()

    proof = probe.run(metrics)

    assert proof is not None and proof["parked"] is False, proof
    assert reason in proof["reason"], proof["reason"]
    values = gauges(metrics)
    assert (values["zero_node_parked"], values["zero_node_unavailable"]) == (0, 1)


@pytest.mark.parametrize(
    ("groups", "state", "reason"),
    [
        ([instance_group()], capacity.PARKED, "InService at TargetCount 0"),
        (
            [instance_group(target=4)],
            capacity.UNAVAILABLE,
            "wants 4 instance(s) (TargetCount 4, CurrentCount 0)",
        ),
        (
            [instance_group(status="Updating")],
            capacity.UNAVAILABLE,
            "instance group gpu is Updating",
        ),
        (
            RuntimeError("ThrottlingException"),
            capacity.UNAVAILABLE,
            capacity.UNREADABLE_REASON + "ThrottlingException",
        ),
    ],
    ids=["parked", "target-wants-instances", "updating", "describe-fails"],
)
def test_a_hyperpod_cluster_without_nodes_is_judged_from_describe_cluster(
    groups: Any, state: str, reason: str
) -> None:
    probe = hyperpod_probe(groups=groups)
    metrics = ClusterExecutorMetrics()

    proof = probe.run(metrics)

    assert proof is not None and capacity.state_of(proof) == state, proof
    assert reason in proof["reason"], proof["reason"]
    assert probe.sagemaker.calls == [{"ClusterName": "hp-a"}], probe.sagemaker.calls
    values = gauges(metrics)
    assert values["zero_node_parked"] == (state == capacity.PARKED)
    assert values["zero_node_unavailable"] == (state == capacity.UNAVAILABLE)


def test_an_unreadable_node_list_reaches_no_verdict_and_never_raises(
    caplog: pytest.LogCaptureFixture,
) -> None:
    now = [0.0]
    probe = eks_probe(nodegroups={GROUP: nodegroup(desired=2)})
    probe.clock = lambda: now[0]
    metrics = ClusterExecutorMetrics()
    probe.observe(metrics)
    before = gauges(metrics)
    probe.core.nodes = ApiError(503)
    now[0] = probe.interval_seconds + 1

    with caplog.at_level(logging.WARNING, logger="gpu_fault.cluster_executor"):
        probe.observe(metrics)

    assert gauges(metrics) == before, "the gauges and the timestamp keep their values"
    assert probe.failures_total == 1
    assert "could not read the selected nodes" in caplog.text, caplog.text
    assert "ApiError" in caplog.text and "Reason" not in caplog.text, (
        "only the exception class is logged"
    )


def test_a_node_list_answer_without_items_is_no_verdict() -> None:
    probe = eks_probe()
    probe.core.list_node = lambda **_k: SimpleNamespace(items=None)  # type: ignore[method-assign]

    with pytest.raises(TypeError, match="no items list"):
        probe.run(ClusterExecutorMetrics())


def test_the_check_runs_once_per_interval() -> None:
    now = [0.0]
    probe = eks_probe(["node-a"])
    probe.clock = lambda: now[0]
    metrics = ClusterExecutorMetrics()

    probe.observe(metrics)
    probe.observe(metrics)
    now[0] = probe.interval_seconds + 1
    probe.observe(metrics)

    assert probe.core.selectors == [OWNERSHIP, OWNERSHIP], "two checks, not three"


def test_only_a_change_of_state_is_said(caplog: pytest.LogCaptureFixture) -> None:
    probe = eks_probe(nodegroups={GROUP: nodegroup(desired=2)})
    metrics = ClusterExecutorMetrics()

    with caplog.at_level(logging.WARNING, logger="gpu_fault.cluster_executor"):
        probe.run(metrics)
        probe.run(metrics)
        probe.core.nodes = ["node-a"]
        probe.run(metrics)

    lines = [record.getMessage() for record in caplog.records]
    assert len(lines) == 2, lines
    assert lines[0].startswith("GPU cluster without nodes (unavailable)"), lines
    assert lines[1].startswith("GPU cluster has nodes again: 1 node(s)"), lines


@pytest.mark.parametrize(
    ("provider", "keywords", "message"),
    [
        (ClusterProvider.EKS_MANAGED, {}, "EKS cluster name"),
        (ClusterProvider.HYPERPOD, {}, "HyperPod cluster name"),
        (
            ClusterProvider.HYPERPOD,
            {"hyperpod_cluster": "hp-a", "node_selector": " "},
            "node_selector",
        ),
    ],
)
def test_a_probe_without_its_cluster_names_is_refused_at_construction(
    provider: ClusterProvider, keywords: dict[str, str], message: str
) -> None:
    arguments: dict[str, Any] = {"node_selector": OWNERSHIP, **keywords}
    with pytest.raises(ValueError, match=message):
        ZeroNodeCapacityProbe(Core([]), provider=provider, **arguments)

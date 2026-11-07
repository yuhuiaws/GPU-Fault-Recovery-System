"""The executor's verdict on its cluster without nodes: the gauges a page reads.

``status`` names a GPU cluster whose selector selects no node -- parked at
zero by its provider, or unavailable while its capacity is lost or still
joining -- but nothing paged for an unavailable one: the collector-silence,
stale-Agent and topology alerts count ACTIVE Agents only, and the departed-node
cleanup retires the Agents of the nodes that left. The executor runs on the
cluster's other nodes and already lists its selected nodes; every
``ZERO_NODE_PROBE_INTERVAL_SECONDS`` it reads that list and -- only when the
API returns it empty -- the provider documents the deploy host's proof reads,
with the executor's own grants, and judges them with the same rule
(``gpu_fault.zero_node_capacity``). Four series follow:

* ``gpu_fault_cluster_executor_selected_nodes`` -- the nodes the selector selects;
* ``gpu_fault_cluster_executor_zero_node_parked`` -- 1 when none is selected and
  the provider declares that capacity zero;
* ``gpu_fault_cluster_executor_zero_node_unavailable`` -- 1 when none is
  selected and the provider does not (a group that wants nodes, one not
  ``ACTIVE``/``InService``, a selector that covers no group, a NodePool that
  could supply nodes, a provider read that failed);
* ``gpu_fault_cluster_executor_last_zero_node_check_timestamp`` -- when the last
  verdict was reached.

A node list the API does not answer reaches no verdict: the gauges keep their
values and the timestamp does not move. A failure is one warning and the next
cycle's retry, never an exception into the claim loop; a state change is one
line, a steady state is silent.

The reads, each a grant of the executor's own:

* HyperPod: ``sagemaker:DescribeCluster`` on the cluster's own ARN;
* eks-managed: ``eks:ListNodegroups`` on the cluster's ARN and
  ``eks:DescribeNodegroup`` on its node groups;
* both: get/list on ``nodepools.karpenter.sh`` in the executor ClusterRole --
  a cluster without the CRD answers 404 and has no NodePool.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Mapping
from typing import Any, Protocol

from gpu_fault import zero_node_capacity as capacity
from gpu_fault.cluster_scope import ClusterProvider

# The pre-split module's name: operators filter on ``gpu_fault.cluster_executor``.
LOGGER = logging.getLogger("gpu_fault.cluster_executor")

#: How often the selected nodes are counted; the provider is read only when
#: there are none. Two reads a minute per replica at most on a HyperPod
#: cluster, a node-group list and one describe per group on an EKS one.
ZERO_NODE_PROBE_INTERVAL_SECONDS = 60.0
KARPENTER_GROUP = "karpenter.sh"
#: The served versions asked in turn: ``v1`` since Karpenter 1.0, ``v1beta1``
#: before. 404 on every one is a cluster without the CRD.
KARPENTER_VERSIONS = ("v1", "v1beta1")
KARPENTER_PLURAL = "nodepools"
#: The ``list_nodegroups`` pages read before the answer is declared broken.
NODEGROUP_PAGE_LIMIT = 50


class ZeroNodeGauges(Protocol):
    """Where the verdict goes (``ClusterExecutorMetrics``)."""

    def zero_node_state(
        self, *, selected: int, parked: bool, unavailable: bool
    ) -> None: ...


def _value(item: Any, name: str, default: Any = None) -> Any:
    if isinstance(item, Mapping):
        return item.get(name, default)
    return getattr(item, name, default)


def _api_status(error: Exception) -> int | None:
    status = getattr(error, "status", None)
    return status if isinstance(status, int) else None


class ZeroNodeCapacityProbe:
    """One verdict on the cluster's selected nodes per interval."""

    def __init__(
        self,
        core: Any,
        *,
        provider: ClusterProvider,
        node_selector: str,
        hyperpod_cluster: str = "",
        eks_cluster: str = "",
        region_name: str | None = None,
        custom: Any | None = None,
        sagemaker_client: Any | None = None,
        eks_client: Any | None = None,
        interval_seconds: float = ZERO_NODE_PROBE_INTERVAL_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not node_selector.strip():
            raise ValueError("node_selector must not be empty")
        if provider is ClusterProvider.EKS_MANAGED and not eks_cluster.strip():
            raise ValueError("an eks-managed cluster needs its EKS cluster name")
        if provider is ClusterProvider.HYPERPOD and not hyperpod_cluster.strip():
            raise ValueError("a HyperPod cluster needs its HyperPod cluster name")
        if interval_seconds <= 0:
            raise ValueError("interval_seconds must be positive")
        self.core = core
        self.custom = custom
        self.provider = provider
        self.node_selector = node_selector.strip()
        self.hyperpod_cluster = hyperpod_cluster.strip()
        self.eks_cluster = eks_cluster.strip()
        self.region_name = region_name
        self.interval_seconds = interval_seconds
        self.clock = clock
        self._sagemaker = sagemaker_client
        self._eks = eks_client
        self._next_due: float | None = None
        self._last_state: str | None = None
        self.failures_total = 0

    @staticmethod
    def _create_client(service: str, region_name: str | None) -> Any:
        import boto3

        return boto3.client(service, region_name=region_name)

    @property
    def sagemaker(self) -> Any:
        # Built on first use: a cluster that never loses its nodes never needs it.
        if self._sagemaker is None:
            self._sagemaker = self._create_client("sagemaker", self.region_name)
        return self._sagemaker

    @property
    def eks(self) -> Any:
        if self._eks is None:
            self._eks = self._create_client("eks", self.region_name)
        return self._eks

    def due(self) -> bool:
        return self._next_due is None or self.clock() >= self._next_due

    # --- the reads ---------------------------------------------------------

    def selected_nodes(self) -> int:
        """How many Nodes the API lists for the selector; raises when unread.

        Taken as the API answers, as the deploy host takes it
        (``regional_parked_cluster._selected_nodes``): only a list the API
        itself returns empty is a cluster without nodes.
        """

        items = _value(self.core.list_node(label_selector=self.node_selector), "items")
        if not isinstance(items, list):
            raise TypeError("the node list answer carries no items list")
        return len(items)

    def nodepools(self) -> Mapping[str, Any] | None:
        """The cluster's Karpenter NodePools; ``None`` without the CRD."""

        if self.custom is None:
            raise RuntimeError("no Kubernetes custom-objects client to read NodePools")
        for version in KARPENTER_VERSIONS:
            try:
                answer = self.custom.list_cluster_custom_object(
                    KARPENTER_GROUP, version, KARPENTER_PLURAL
                )
            except Exception as exc:
                if _api_status(exc) == 404:
                    continue
                raise
            return answer if isinstance(answer, Mapping) else {}
        return None

    def nodegroups(self) -> dict[str, Mapping[str, Any]]:
        """Every managed node group of the EKS cluster, described."""

        names: list[str] = []
        token: str | None = None
        for _page in range(NODEGROUP_PAGE_LIMIT):
            arguments: dict[str, Any] = {"clusterName": self.eks_cluster}
            if token:
                arguments["nextToken"] = token
            answer = self.eks.list_nodegroups(**arguments)
            names.extend(str(name) for name in answer.get("nodegroups") or () if name)
            next_token = answer.get("nextToken")
            if not next_token:
                return {
                    name: self.eks.describe_nodegroup(
                        clusterName=self.eks_cluster, nodegroupName=name
                    )["nodegroup"]
                    for name in sorted(set(names))
                }
            if next_token == token:
                raise RuntimeError("list_nodegroups repeats its page token")
            token = str(next_token)
        raise RuntimeError(f"list_nodegroups exceeds {NODEGROUP_PAGE_LIMIT} pages")

    def instance_groups(self) -> list[Any]:
        """The HyperPod cluster's instance groups (``DescribeCluster``)."""

        answer = self.sagemaker.describe_cluster(ClusterName=self.hyperpod_cluster)
        return list(answer.get("InstanceGroups") or ())

    def proof(self) -> dict[str, Any]:
        """The provider's verdict on the cluster, read now; never raises."""

        return capacity.capacity_proof(
            self.provider.value,
            selector=self.node_selector,
            cluster_name=self.hyperpod_cluster,
            nodegroups=self.nodegroups,
            instance_groups=self.instance_groups,
            nodepools=self.nodepools,
        )

    # --- one cycle ---------------------------------------------------------

    def run(self, gauges: ZeroNodeGauges) -> dict[str, Any] | None:
        """Count the selected nodes, judge the provider when there are none,
        and set the gauges; returns the proof (``None`` with nodes)."""

        self._next_due = self.clock() + self.interval_seconds
        try:
            selected = self.selected_nodes()
        except Exception:
            self.failures_total += 1
            raise
        proof = self.proof() if selected == 0 else None
        state = "nodes" if proof is None else capacity.state_of(proof)
        gauges.zero_node_state(
            selected=selected,
            parked=state == capacity.PARKED,
            unavailable=state == capacity.UNAVAILABLE,
        )
        if state != self._last_state:
            self._said(state, selected, proof)
        self._last_state = state
        return proof

    def _said(self, state: str, selected: int, proof: Mapping[str, Any] | None) -> None:
        if proof is None:
            if self._last_state is not None:
                LOGGER.warning(
                    "GPU cluster has nodes again: %d node(s) selected by %s",
                    selected,
                    self.node_selector,
                )
            return
        LOGGER.warning(
            "GPU cluster without nodes (%s): no node selected by %s; %s",
            state,
            self.node_selector,
            proof.get("reason"),
        )

    def observe(self, gauges: ZeroNodeGauges) -> None:
        """``run`` when due; never raises into the claim loop."""

        if not self.due():
            return
        try:
            self.run(gauges)
        except Exception as exc:  # noqa: BLE001 - housekeeping must not stop claims
            LOGGER.warning(
                "zero-node check could not read the selected nodes (failures %d): "
                "%s; the gauges keep their last values",
                self.failures_total,
                type(exc).__name__,
            )

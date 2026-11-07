"""Bootstrap tasks of a GPU cluster without nodes, judged by the release engine's proof.

The release engine skips the node work of a GPU cluster whose recorded node
selector selects no Kubernetes node (``gpu_fault_release.regional_parked_cluster``:
*parked* when its provider declares the capacity zero, *unavailable* otherwise):
zero GPU nodes never stop a deploy of the site. The bootstrap that runs before
the release reads the GPU clusters too, and live it did stop one: the
``node_keys:<cluster>`` task of an eks-managed cluster scaled to zero ran the
node-key helper, which refuses an empty node inventory, and the deploy failed
before any roll.

``zero_node_proof`` asks exactly what the engine asks -- the node list under the
cluster's recorded selector and, only when the API returns it empty, the
provider documents (``describe-nodegroup`` and the Karpenter NodePools, or
HyperPod ``describe-cluster``) -- through ``regional_parked_cluster.without_nodes``
itself, with the admin runner and the GPU kubeconfig, in the run that acts on
it. The reads are read-only, so a probe runner answers them. A node list that
cannot be read proves nothing: the task then runs exactly as before and fails
closed on what it finds.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from gpu_fault.admin.bootstrap_common import (
    BootstrapError,
    ClusterIdentity,
    CommandRunner,
)
from gpu_fault_release import regional_parked_cluster as parked_cluster

#: The ``node_keys:<cluster>`` narration of a cluster without nodes.
NODE_KEYS_STEP = "node_keys"
NODE_KEYS_RECORDED = "no node, no key to provision"


class _Reads:
    """What ``regional_parked_cluster`` reads of a release, answered by the admin
    runner against one GPU cluster (``runner`` is the release's runner too)."""

    def __init__(
        self, runner: CommandRunner, *, gpu_kubeconfig: Path, context: str
    ) -> None:
        self._runner = runner
        self._kubectl = ["kubectl", "--kubeconfig", str(gpu_kubeconfig)]
        if context:
            self._kubectl += ["--context", context]
        self.runner = self

    def run(
        self, arguments: Sequence[str], *, sensitive: bool = False, **_unused: Any
    ) -> str:
        return self._runner.run(list(arguments), capture=True, sensitive=sensitive)

    def probe_output(
        self, arguments: Sequence[str], *, timeout_seconds: float
    ) -> tuple[int, str, str]:
        try:
            output = self._runner.run(list(arguments), timeout_seconds=timeout_seconds)
        except BootstrapError as error:
            return 1, "", str(error)
        return 0, output, ""

    def _gpu(self, _target: object, *arguments: str) -> list[str]:
        return [*self._kubectl, *arguments]

    def _get_json(self, arguments: Sequence[str]) -> dict[str, Any]:
        raw = self._runner.run([*arguments, "-o", "json"])
        value = json.loads(raw) if raw else {}
        return value if isinstance(value, dict) else {}


def _target(cluster: ClusterIdentity, cluster_id: str) -> SimpleNamespace:
    """``cluster`` in the shape of a release target (``ClusterTarget``)."""

    return SimpleNamespace(
        cluster_id=cluster_id,
        provider=cluster.provider,
        node_selector=cluster.node_selector,
        hyperpod_cluster_name=cluster.hyperpod_name,
        eks_cluster_arn=cluster.eks_arn,
        region=cluster.region,
    )


def zero_node_proof(
    runner: CommandRunner,
    *,
    gpu_kubeconfig: Path,
    cluster: ClusterIdentity,
    cluster_id: str,
) -> dict[str, Any] | None:
    """The engine's proof of ``cluster`` -- parked or unavailable -- when its
    selector selects no node, read now; ``None`` when it selects one or its node
    list cannot be read."""

    return parked_cluster.without_nodes(
        _Reads(runner, gpu_kubeconfig=gpu_kubeconfig, context=cluster.context),
        _target(cluster, cluster_id),
    )


def zero_node_keys(
    runner: CommandRunner,
    *,
    gpu_kubeconfig: Path,
    cluster: ClusterIdentity,
    cluster_id: str,
) -> dict[str, Any] | None:
    """The ``node_keys:<cluster>`` result of a cluster without nodes, after
    saying so; ``None`` for every other cluster, whose task runs as before.

    Nothing is provisioned, deleted or rotated: the GPU and CPU node-key
    Secrets keep every key they hold (the helper would rewrite the GPU Secret to
    the keys of the nodes it lists -- none). A node that joins later is keyed by
    the executor's node-key sync (``gpu_fault.cluster_executor.node_key_sync``),
    which patches the missing names into the existing Secret, and installed by
    the steady-state Reconciler the release rolled; the next deploy that finds
    nodes provisions as always.
    """

    proof = zero_node_proof(
        runner, gpu_kubeconfig=gpu_kubeconfig, cluster=cluster, cluster_id=cluster_id
    )
    if proof is None:
        return None
    parked_cluster.narrate(
        cluster_id,
        state=parked_cluster.state_of(proof),
        step=NODE_KEYS_STEP,
        recorded=NODE_KEYS_RECORDED,
        reason=str(proof.get("reason")),
    )
    return {
        "cluster_id": cluster_id,
        parked_cluster.ZERO_NODES_IDENTITY_KEY: parked_cluster.summary(proof),
    }

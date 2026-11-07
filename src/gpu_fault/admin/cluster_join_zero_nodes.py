"""``join-cluster`` of a GPU cluster without nodes.

A join used to need a node: its node inventory refused an empty list ("target
GPU node inventory has missing or conflicting identities") after it had already
written the kubeconfig alias, the namespace and the cluster token, and the
automatic rollback undid them. A cluster whose recorded selector selects no node
-- parked at zero by its provider, or unavailable while its capacity is lost or
still joining (``gpu_fault_release.regional_parked_cluster``) -- now joins with
its node work skipped and recorded, as the release engine joins it
(``join_cluster``'s node runtime is zero-node aware):

* the node inventory is the release engine's proof (``bootstrap_zero_nodes``):
  no node, the cluster's ``zero_nodes`` record (state, provider, reason, groups)
  in ``LOCAL_INPUTS_READY`` and ``NODE_NAMES_VERIFIED``, and one
  ``zero-node-gpu-cluster ... step=join`` line;
* the node keys: no key is provisioned -- there is no node -- and the GPU
  ``gpu-fault-node-action-keys`` Secret the Executor mounts and its node-key sync
  fills (it never creates the Secret) is created empty, never through the key
  helper; the CPU Secret is not touched;
* the collector readiness: nothing will ever report from a cluster without
  nodes, so a join that still has none waits for nothing (``zero_node_not_awaited``).

Two refusals stay, made before anything is created on the cluster: a cluster
whose node groups declare no GPU instance type the installer knows (its
installer template is rendered from a declared node, ``render_node``, and there
is none), and a cluster registered for node-key custody (custody binds the node
set it authorizes, and there is none). A node list that cannot be read proves
nothing: the inventory refuses as before.

A node that joins while the join runs is not deferred (only an upgrade defers
newcomers): the join's node roll installs it in a wave, and its key comes from
the Executor's node-key sync within its interval. A node that joins afterwards is
installed by the steady-state Reconciler with the current release.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

from gpu_fault.admin.bootstrap_common import (
    BootstrapError,
    ClusterIdentity,
    CommandRunner,
    ReadOnlyProbeRunner,
)
from gpu_fault.admin.bootstrap_zero_nodes import zero_node_proof
from gpu_fault.admin.diagnostics import diagnostic_text
from gpu_fault.admin.node_key_proof import read_node_key_proof
from gpu_fault_release import regional_parked_cluster as parked_cluster
from gpu_fault_release.regional_release_config import ReleaseError

if TYPE_CHECKING:
    from gpu_fault.admin.cluster_join import JoinClusterRequest, JoinExecution
    from gpu_fault.admin.site import RenderedSite

#: The journal key of a join without nodes.
ZERO_NODES_KEY = "zero_nodes"
NODE_KEYS_SECRET = "gpu-fault-node-action-keys"
STEP = "join"
RECORDED = "no node, nothing to key or install"


def recorded(evidence: Mapping[str, Any]) -> bool:
    """Whether join evidence records a cluster without nodes."""

    return isinstance(evidence.get(ZERO_NODES_KEY), Mapping)


def _proof(
    runner: CommandRunner, target: ClusterIdentity, cluster_id: str, kubeconfig: Path
) -> dict[str, Any] | None:
    return zero_node_proof(
        ReadOnlyProbeRunner(runner),
        gpu_kubeconfig=kubeconfig,
        cluster=target,
        cluster_id=cluster_id,
    )


def _refuse_unjoinable(
    site: RenderedSite,
    proof: Mapping[str, Any],
    target: ClusterIdentity,
    cluster_id: str,
) -> None:
    from gpu_fault.admin.custody_zero_nodes import custody_registered

    if custody_registered(site, target.eks_arn):
        raise BootstrapError(
            f"join-cluster refuses {cluster_id}: it has no node, and node-key "
            "custody is registered for it -- custody binds the node set it "
            "authorizes, and there is none; scale a covered group up, let a node "
            "join and rerun join-cluster"
        )
    try:
        parked_cluster.render_node(
            proof,
            SimpleNamespace(
                cluster_id=cluster_id,
                provider=target.provider,
                hyperpod_cluster_name=target.hyperpod_name,
            ),
        )
    except ReleaseError as exc:
        raise BootstrapError(f"join-cluster refuses {cluster_id}: {exc}") from None


def refuse_unjoinable(
    request: JoinClusterRequest,
    runner: CommandRunner,
    target: ClusterIdentity,
    cluster_id: str,
    kubeconfig: Path,
) -> None:
    """Refuse, before anything is created on it, a cluster without nodes that
    cannot be joined that way; read-only."""

    proof = _proof(runner, target, cluster_id, kubeconfig)
    if proof is not None:
        _refuse_unjoinable(request.site, proof, target, cluster_id)


def node_inventory(
    request: JoinClusterRequest,
    runner: CommandRunner,
    target: ClusterIdentity,
    cluster_id: str,
    kubeconfig: Path,
) -> tuple[list[str], dict[str, str], dict[str, Any] | None]:
    """The joining cluster's nodes, their UIDs and -- for a cluster without
    nodes -- its zero-node record instead of a node list."""

    from gpu_fault.admin import cluster_join as join

    proof = _proof(runner, target, cluster_id, kubeconfig)
    if proof is None:
        identities: dict[str, str] = {}
        nodes = join._list_nodes(
            runner, kubeconfig=kubeconfig, target=target, identities=identities
        )
        return nodes, identities, None
    _refuse_unjoinable(request.site, proof, target, cluster_id)
    parked_cluster.narrate(
        cluster_id,
        state=parked_cluster.state_of(proof),
        step=STEP,
        recorded=RECORDED,
        reason=str(proof.get("reason")),
    )
    return [], {}, parked_cluster.summary(proof)


def empty_node_key_secret(
    runner: CommandRunner,
    site: RenderedSite,
    target: ClusterIdentity,
    cluster_id: str,
    kubeconfig: Path,
    verified: Mapping[str, Any],
    started: Callable[[], None],
) -> dict[str, Any]:
    """The node-key prerequisite of a join without nodes: no key, and the GPU
    Secret the Executor mounts created empty (an existing one is kept as it is)."""

    namespace = str(site.release_config["namespace"])
    kubectl = ["kubectl", "--kubeconfig", str(kubeconfig), "--context", target.context]
    started()
    present = read_node_key_proof(runner, kubectl, namespace)
    created = present is None
    if created:
        document = {
            "apiVersion": "v1",
            "kind": "Secret",
            "metadata": {"name": NODE_KEYS_SECRET, "namespace": namespace},
            "type": "Opaque",
            "data": {},
        }
        runner.run(
            [*kubectl, "-n", namespace, "create", "-f", "-"],
            input_text=json.dumps(document),
            mutate=True,
        )
        present = read_node_key_proof(runner, kubectl, namespace)
    if present is None:
        raise BootstrapError(
            f"{namespace}/{NODE_KEYS_SECRET} creation was not observed"
        )
    return {
        "cluster_id": cluster_id,
        ZERO_NODES_KEY: dict(verified[ZERO_NODES_KEY]),
        "gpu_node_key_secret": {
            "uid": present.uid,
            "keys": len(present.digests),
            "created": created,
        },
    }


class _JoinCommands(CommandRunner):
    """Read-only commands through the join's own ``run_command``."""

    def __init__(self, site: RenderedSite) -> None:
        super().__init__()
        self._site = site

    def run(self, arguments: Any, **options: Any) -> str:
        from gpu_fault.admin import cluster_join as join
        from gpu_fault.admin.site import effective_environment

        result = join.run_command(
            list(arguments),
            environment=effective_environment(self._site),
            timeout_seconds=options.get("timeout_seconds") or 60,
        )
        if result.returncode:
            raise BootstrapError(
                f"{arguments[0]} exited {result.returncode}: "
                + diagnostic_text(str(result.stderr or ""))
            )
        return str(result.stdout or "")


def collector_readiness(
    execution: JoinExecution, wait: Callable[[Any, str], dict[str, Any]]
) -> dict[str, Any]:
    """The ``COLLECTORS_READY`` evidence: ``wait`` for the joined cluster's
    collectors, except for a cluster that joined without nodes and still has
    none, from which nothing will ever report."""

    if recorded(execution.local):
        proof = _proof(
            _JoinCommands(execution.candidate),
            execution.target,
            execution.cluster_id,
            Path(str(execution.local["gpu_kubeconfig"])),
        )
        if proof is not None:
            parked_cluster.narrate(
                execution.cluster_id,
                state=parked_cluster.state_of(proof),
                step="collector-readiness",
                recorded="no node, no collector to wait for",
                reason=str(proof.get("reason")),
            )
            return {
                "nodes": 0,
                "ready": None,
                parked_cluster.NOT_AWAITED_KEY: True,
                ZERO_NODES_KEY: parked_cluster.summary(proof),
            }
    readiness = wait(execution.candidate, execution.cluster_id)
    return {
        "nodes": len(readiness.get("nodes") or []),
        "ready": readiness.get("ready"),
    }

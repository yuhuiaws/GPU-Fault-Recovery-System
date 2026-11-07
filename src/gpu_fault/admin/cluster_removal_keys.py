"""Digest-only ownership evidence for removing a cluster's node keys.

The GPU Secret may also hold keys of nodes that departed since the last deploy;
each is accepted only when its node is proven departed, and recorded
(``cluster_removal_departed_keys``).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from gpu_fault.admin import cluster_join_nodes
from gpu_fault.admin import cluster_removal_departed_keys as departed_keys
from gpu_fault.admin.bootstrap_common import BootstrapError, CommandRunner
from gpu_fault.admin.cluster_node_key_digests import node_key_digests
from gpu_fault.admin.node_key_proof import read_node_key_proof
from gpu_fault.admin.site import RenderedSite


def validate_key_binding(binding: object, nodes: Sequence[str]) -> dict[str, Any]:
    if not isinstance(binding, dict) or any(
        not isinstance(binding.get(key), str) or not binding[key]
        for key in ("cpu_secret_uid", "gpu_secret_uid")
    ):
        raise BootstrapError("removal has no saved node-key Secret UID proof")
    node_key_digests(
        binding.get("expected_key_sha256"),
        nodes,
        refusal="removal has no complete node-key digest proof",
    )
    departed_keys.recorded_departed_keys(binding, nodes)
    return binding


def verify_node_key_ownership(
    site: RenderedSite,
    cluster_id: str,
    nodes: Sequence[str],
    *,
    runner: CommandRunner,
    gpu_kubectl: Sequence[str],
    saved: dict[str, Any] | None = None,
    namespace_present: bool = True,
    allow_missing: bool = False,
    removed: bool = False,
    target: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """The node-key ownership binding of ``nodes``; with ``target`` (the site's
    cluster entry), extra GPU keys of nodes proven departed are recorded."""

    if not nodes:
        return {}
    names = set(nodes)
    claims = cluster_join_nodes.read_node_claims(site, runner)
    if any(
        names.intersection(owned)
        for owner, owned in claims.nodes_by_cluster.items()
        if owner != cluster_id
    ) or any(
        node in names and owner != cluster_id
        for node, owner in claims.agent_owners.items()
    ):
        raise BootstrapError("node-key NodeNames are owned by another cluster")
    namespace = str(site.release_config["namespace"])
    cpu = read_node_key_proof(
        runner,
        ["kubectl", "--kubeconfig", str(site.release_config["cpu_kubeconfig"])],
        namespace,
    )
    if cpu is None:
        raise BootstrapError("CPU node-key Secret ownership is unavailable")
    gpu = (
        read_node_key_proof(runner, gpu_kubectl, namespace)
        if namespace_present
        else None
    )
    if namespace_present and (
        gpu is None
        or gpu.rotation_pending
        or not names <= set(gpu.digests)
        or (target is None and set(gpu.digests) != names)
    ):
        raise BootstrapError("GPU node-key ownership is incomplete or rotating")
    if saved is None:
        if gpu is None:
            raise BootstrapError("removal has no saved node-key ownership proof")
        extras = {node: gpu.digests[node] for node in set(gpu.digests) - names}
        saved = {
            "cpu_secret_uid": cpu.uid,
            "gpu_secret_uid": gpu.uid,
            "expected_key_sha256": {node: gpu.digests[node] for node in sorted(names)},
        }
        if extras and target is not None:
            saved[departed_keys.DEPARTED_KEYS] = departed_keys.departed_node_keys(
                site,
                target,
                cluster_id,
                extras,
                cpu.digests,
                runner=runner,
                gpu_kubectl=gpu_kubectl,
            )
    binding = validate_key_binding(saved, nodes)
    expected = binding["expected_key_sha256"]
    recorded = departed_keys.recorded_departed_keys(binding, nodes)
    if cpu.uid != binding["cpu_secret_uid"] or (
        gpu is not None
        and (
            gpu.uid != binding["gpu_secret_uid"]
            or not departed_keys.gpu_digests_hold(gpu.digests, expected, recorded)
        )
    ):
        raise BootstrapError("node-key Secret incarnation or ownership changed")
    if removed:
        if names.intersection(cpu.digests):
            raise BootstrapError("removed node-key entries reappeared on the CPU")
    elif any(
        cpu.digests.get(node) != expected[node]
        for node in nodes
        if not allow_missing or node in cpu.digests
    ):
        raise BootstrapError("CPU node-key ownership changed during removal")
    return binding

"""Digest-only ownership evidence for removing a cluster's node keys."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from gpu_fault.admin import cluster_join_nodes
from gpu_fault.admin.bootstrap_common import BootstrapError, CommandRunner
from gpu_fault.admin.node_key_proof import read_node_key_proof
from gpu_fault.admin.site import RenderedSite


def validate_key_binding(binding: object, nodes: Sequence[str]) -> dict[str, Any]:
    if not isinstance(binding, dict) or any(
        not isinstance(binding.get(key), str) or not binding[key]
        for key in ("cpu_secret_uid", "gpu_secret_uid")
    ):
        raise BootstrapError("removal has no saved node-key Secret UID proof")
    digests = binding.get("expected_key_sha256")
    if (
        not isinstance(digests, dict)
        or set(digests) != set(nodes)
        or any(
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for value in digests.values()
        )
    ):
        raise BootstrapError("removal has no complete node-key digest proof")
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
) -> dict[str, Any]:
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
        gpu is None or gpu.rotation_pending or set(gpu.digests) != names
    ):
        raise BootstrapError("GPU node-key ownership is incomplete or rotating")
    if saved is None:
        if gpu is None:
            raise BootstrapError("removal has no saved node-key ownership proof")
        saved = {
            "cpu_secret_uid": cpu.uid,
            "gpu_secret_uid": gpu.uid,
            "expected_key_sha256": dict(gpu.digests),
        }
    binding = validate_key_binding(saved, nodes)
    expected = binding["expected_key_sha256"]
    if cpu.uid != binding["cpu_secret_uid"] or (
        gpu is not None
        and (gpu.uid != binding["gpu_secret_uid"] or gpu.digests != expected)
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

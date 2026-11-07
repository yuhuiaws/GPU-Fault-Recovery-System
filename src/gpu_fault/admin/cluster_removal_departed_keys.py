"""The keys of departed nodes a GPU cluster's node-key Secret still holds.

The Executor's node-key sync only ever adds keys
(``gpu_fault.cluster_executor.node_key_sync``: "nothing is deleted"), so the
GPU ``gpu-fault-node-action-keys`` Secret of a cluster that lost a node since
its last deploy -- a spot reclaim, a scale-in, a provider replacement -- still
holds that node's key until the next deploy's key helper rewrites it.
``remove-cluster`` bound its node-key ownership to the GPU Secret holding
exactly the selected nodes' keys and refused anything else as "GPU node-key
ownership is incomplete or rotating": such a cluster could not be removed.

An extra key is now accepted when its node is proven departed, on the evidence
``workflow-reconcile`` and the control plane's departed-node cleanup judge with
(``gpu_fault.departed_nodes``, through ``workflow_reconcile_departed_nodes``):

* the node is absent from an unfiltered node list of the cluster -- a node
  that is still there under any label is no departed node;
* its provider no longer has it: on HyperPod the instance its name carries is
  not in ``list-cluster-nodes``; on an eks-managed cluster EC2 reports the
  instance the fleet recorded for it (the Agent record's ``node_instance_id``,
  read in a CPU ingress Pod) ``terminated`` or not at all.

Anything less is no proof and the removal refuses, naming each node and its
blocker: a node still in Kubernetes, a Kubernetes, HyperPod, EC2 or control
plane read that fails or answers ambiguously, a name without a HyperPod
instance id, no recorded instance id, an instance the provider still has.
Every read goes through the removal's own command runner, and nothing here
writes.

A proven key is recorded in ``DISCOVERED.node_key_ownership.departed_node_keys``
-- its digest, the digest of the CPU entry of the same name (``None`` when there
is none) and the evidence -- and one ``remove-cluster-departed-node-keys`` line
names how many were kept. A resume re-checks the recorded digests and proves
nothing anew. What the removal deletes does not change: the CPU entries removed
are still exactly the selected nodes'; a departed node's CPU entry is recorded
and kept, never deleted, and its GPU key goes with the namespace.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any

from gpu_fault.admin import workflow_reconcile_departed_nodes as departed
from gpu_fault.admin.bootstrap_common import BootstrapError, CommandRunner
from gpu_fault.admin.cluster_join_nodes import NODE_NAMES_TEMPLATE
from gpu_fault.admin.cluster_node_key_digests import NAMED_NODES, is_node_key_digest
from gpu_fault.admin.site import RenderedSite, effective_environment
from gpu_fault_release.regional_deployment_inventory import CPU_INGRESS_DEPLOYMENT
from gpu_fault.departed_nodes import provider_lookup_blocker
from gpu_fault_release.regional_release_narration import narrate_step

#: The ``node_key_ownership`` key of the departed nodes' keys.
DEPARTED_KEYS = "departed_node_keys"
NARRATION = "remove-cluster-departed-node-keys"
STILL_IN_KUBERNETES = (
    "node is still in Kubernetes outside the cluster's node selector; its key "
    "is no departed node's"
)


def _commands(
    runner: CommandRunner, environment: Mapping[str, str]
) -> Callable[..., Any]:
    """``run_command`` for the departed-node helpers, over the removal's runner."""

    def run(arguments: Sequence[str], *, timeout_seconds: float = 120) -> Any:
        try:
            stdout = runner.run(
                list(arguments), env=environment, timeout_seconds=timeout_seconds
            )
        except BootstrapError as exc:
            return SimpleNamespace(returncode=1, stdout="", stderr=str(exc))
        return SimpleNamespace(returncode=0, stdout=stdout, stderr="")

    return run


def _scripts(
    runner: CommandRunner, environment: Mapping[str, str]
) -> Callable[..., Any]:
    """``run_script`` for the recorded instance ids: one read in a Running CPU
    ingress Pod, over the removal's runner."""

    def run(site: RenderedSite, payload: dict[str, Any], *, script: str) -> Any:
        prefix = [
            "kubectl",
            "--kubeconfig",
            str(site.release_config["cpu_kubeconfig"]),
            "-n",
            str(site.release_config["namespace"]),
        ]
        pod = runner.run(
            [
                *prefix,
                "get",
                "pod",
                "-l",
                f"app={CPU_INGRESS_DEPLOYMENT}",
                "--field-selector=status.phase=Running",
                "-o",
                "jsonpath={.items[0].metadata.name}",
            ],
            env=environment,
            timeout_seconds=30,
        ).strip()
        if not pod:
            raise BootstrapError("no Running CPU ingress Pod to read the fleet from")
        output = runner.run(
            [*prefix, "exec", "-i", pod, "--", "python3", "-c", script],
            input_text=json.dumps(payload, separators=(",", ":")),
            env=environment,
            sensitive=True,
            timeout_seconds=120,
        )
        try:
            value = json.loads(output)
        except ValueError:
            raise BootstrapError("the fleet's answer is not valid JSON") from None
        if not isinstance(value, dict):
            raise BootstrapError("the fleet's answer is not an object")
        return value

    return run


def _kubernetes_node_names(
    runner: CommandRunner, gpu_kubectl: Sequence[str], environment: Mapping[str, str]
) -> frozenset[str]:
    """Every Node of the cluster, whatever its labels."""

    output = runner.run(
        [*gpu_kubectl, "get", "nodes", "-o", "go-template=" + NODE_NAMES_TEMPLATE],
        env=environment,
        timeout_seconds=60,
    )
    try:
        names = json.loads(output)
    except ValueError:
        raise BootstrapError("the node list is not valid JSON") from None
    if not isinstance(names, list) or any(
        not isinstance(name, str) or not name for name in names
    ):
        raise BootstrapError("the node list has ambiguous node names")
    return frozenset(names)


def _provider_evidence(
    site: RenderedSite,
    target: Mapping[str, Any],
    cluster_id: str,
    nodes: Sequence[str],
    *,
    runner: CommandRunner,
    environment: Mapping[str, str],
    read_at: dict[str, str],
) -> dict[str, dict[str, Any]]:
    """The provider evidence of each node Kubernetes no longer has."""

    if not nodes:
        return {}
    run_command = _commands(runner, environment)
    missing: dict[str, Any] = {"exists": False, "restored": False, "blockers": []}
    if departed.cluster_provider_of(target) != departed.EKS_PROVIDER:
        inventories: dict[str, Any] = {}
        return {
            node: departed.hyperpod_departed_node_evidence(
                site,
                target,
                cluster_id,
                node,
                missing={**missing, "node_id": node},
                provider_inventories=inventories,
                read_at=read_at,
                run_command=run_command,
            )
            for node in nodes
        }
    try:
        recorded = departed.recorded_instance_ids(
            site, cluster_id, nodes, run_script=_scripts(runner, environment)
        )
    except BootstrapError as exc:
        blocker = provider_lookup_blocker(
            f"cannot read the instance ids the fleet recorded: {exc}"
        )
        return {node: {"node_id": node, "blockers": [blocker]} for node in nodes}
    states: dict[str, dict[str, str]] = {}
    return {
        node: departed.departed_node_evidence(
            target,
            cluster_id,
            node,
            instance_id=recorded.get(node),
            aws_region=site.release_config.get("aws_region"),
            missing={**missing, "node_id": node},
            states=states,
            read_at=read_at,
            run_command=run_command,
        )
        for node in nodes
    }


def _refusal(blockers: Mapping[str, Sequence[str]]) -> BootstrapError:
    named = sorted(blockers)
    shown = "; ".join(
        f"{node}: {', '.join(blockers[node])}" for node in named[:NAMED_NODES]
    )
    hidden = len(named) - NAMED_NODES
    return BootstrapError(
        "the GPU node-key Secret holds keys of nodes not proven departed: "
        + shown
        + (f"; and {hidden} more" if hidden > 0 else "")
    )


def departed_node_keys(
    site: RenderedSite,
    target: Mapping[str, Any],
    cluster_id: str,
    extras: Mapping[str, str],
    cpu_digests: Mapping[str, str],
    *,
    runner: CommandRunner,
    gpu_kubectl: Sequence[str],
) -> dict[str, dict[str, Any]]:
    """The record of the GPU keys ``extras`` (node -> digest) whose nodes are
    proven departed, said in one line; refuses unless every one is."""

    environment = effective_environment(site)
    read_at = {"kubernetes": datetime.now(timezone.utc).isoformat()}
    try:
        present = _kubernetes_node_names(runner, gpu_kubectl, environment)
    except BootstrapError as exc:
        reason = f"the cluster's node list cannot be read: {exc}"
        raise _refusal({node: [reason] for node in extras}) from None
    blockers = {node: [STILL_IN_KUBERNETES] for node in extras if node in present}
    absent = sorted(node for node in extras if node not in present)
    evidence = _provider_evidence(
        site,
        target,
        cluster_id,
        absent,
        runner=runner,
        environment=environment,
        read_at=read_at,
    )
    for node, proof in evidence.items():
        if proof.get("absent_from_provider") is not True or proof.get("blockers"):
            blockers[node] = list(proof.get("blockers") or ["no provider proof"])
    if blockers:
        raise _refusal(blockers)
    record = {
        node: {
            "sha256": extras[node],
            "cpu_sha256": cpu_digests.get(node),
            "evidence": {
                "absent_from_kubernetes": True,
                **{
                    key: evidence[node][key]
                    for key in ("provider", "instance_id", "ec2_state")
                    if key in evidence[node]
                },
                "absent_from_provider": True,
                "read_at": dict(read_at),
            },
        }
        for node in absent
    }
    narrate_step(
        NARRATION,
        cluster=cluster_id,
        kept=len(record),
        cpu_entries_kept=sum(1 for item in record.values() if item["cpu_sha256"]),
        nodes=",".join(absent[:NAMED_NODES]),
        **({"more": len(absent) - NAMED_NODES} if len(absent) > NAMED_NODES else {}),
    )
    return record


def recorded_departed_keys(
    binding: Mapping[str, Any], nodes: Sequence[str]
) -> dict[str, str]:
    """The departed keys a saved ownership binding records (none for one that
    predates them), or a refusal when the record is malformed."""

    value = binding.get(DEPARTED_KEYS)
    if value is None:
        return {}
    if (
        not isinstance(value, dict)
        or not value
        or set(value) & set(nodes)
        or any(
            not isinstance(item, dict) or not is_node_key_digest(item.get("sha256"))
            for item in value.values()
        )
    ):
        raise BootstrapError(
            "removal has no complete node-key digest proof: the departed node "
            "keys it recorded are malformed"
        )
    return {str(node): str(item["sha256"]) for node, item in value.items()}


def gpu_digests_hold(
    digests: Mapping[str, str],
    expected: Mapping[str, str],
    recorded: Mapping[str, str],
) -> bool:
    """Whether the GPU Secret still holds the selected nodes' keys as bound and
    every other key it holds is a recorded departed node's, unchanged (a
    departed key that has since gone is no change)."""

    return all(
        digests.get(node) == digest for node, digest in expected.items()
    ) and all(
        recorded.get(node) == digest
        for node, digest in digests.items()
        if node not in expected
    )

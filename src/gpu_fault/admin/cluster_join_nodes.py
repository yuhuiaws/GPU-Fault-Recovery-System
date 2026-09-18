from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextvars import copy_context
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence

from gpu_fault.admin.bootstrap_common import (
    BootstrapError,
    ClusterIdentity,
    CommandRunner,
)
from gpu_fault.admin.cluster_join_state import complete_step, step_done
from gpu_fault.admin.deploy_limits import DEPLOY_CONCURRENCY
from gpu_fault.admin.node_key_proof import read_node_key_proof
from gpu_fault.admin.site import RenderedSite, effective_environment
from gpu_fault.fleet import derive_node_action_secret
from gpu_fault_release.regional_deployment_inventory import CPU_INGRESS_DEPLOYMENT

if TYPE_CHECKING:
    from gpu_fault.admin.cluster_join import JoinClusterRequest, JoinInputs

EXPECTED_NODES_ENV = "GPU_FAULT_NODE_KEY_EXPECTED_NODES_JSON"
NODE_NAMES_TEMPLATE = (
    '{{printf "["}}{{$sep := ""}}{{range .items}}'
    '{{printf "%s%q" $sep .metadata.name}}{{$sep = ","}}{{end}}{{printf "]"}}'
)
KEY_NAMES_TEMPLATE = (
    '{{printf "["}}{{$sep := ""}}{{range $key, $value := .data}}'
    '{{printf "%s%q" $sep $key}}{{$sep = ","}}{{end}}{{printf "]"}}'
)
AGENT_OWNERS_SCRIPT = r"""
import json
import os
import urllib.request

request = urllib.request.Request(
    "http://127.0.0.1:8080/v1/fleet/agents",
    headers={"X-GPU-Fault-Execution-Token": os.environ["GPU_FAULT_EXECUTION_TOKEN"]},
)
with urllib.request.urlopen(request, timeout=30) as response:
    agents = json.load(response)
if not isinstance(agents, list):
    raise ValueError("Agent inventory is unavailable")
owners = {}
for agent in agents:
    node = agent["node_id"]
    cluster = agent["cluster_id"]
    if not isinstance(node, str) or not node or not isinstance(cluster, str) or not cluster:
        raise ValueError("Agent ownership is incomplete")
    if node in owners and owners[node] != cluster:
        raise ValueError("Agent ownership is ambiguous")
    owners[node] = cluster
print(json.dumps(owners, separators=(",", ":")))
"""


@dataclass(frozen=True)
class NodeClaims:
    nodes_by_cluster: dict[str, frozenset[str]] = field(default_factory=dict)
    agent_owners: dict[str, str] = field(default_factory=dict)
    cpu_key_names: frozenset[str] = frozenset()
    observed_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


def read_target_node_inventory(
    runner: CommandRunner, kubeconfig: Path, target: ClusterIdentity
) -> dict[str, str]:
    output = runner.run(
        [
            "kubectl",
            "--kubeconfig",
            str(kubeconfig),
            "--context",
            target.context,
            "get",
            "nodes",
            "-l",
            "sagemaker.amazonaws.com/cluster-name=" + target.hyperpod_name,
            "-o",
            "json",
        ],
        timeout_seconds=60,
    )
    try:
        document = json.loads(output)
        nodes = document["items"]
        if not isinstance(nodes, list) or not nodes:
            raise ValueError
        inventory: dict[str, str] = {}
        for item in nodes:
            metadata = item["metadata"]
            name, uid = metadata["name"], metadata["uid"]
            if (
                item.get("kind") != "Node"
                or not isinstance(name, str)
                or not name
                or name in inventory
                or not isinstance(uid, str)
                or not uid
                or (metadata.get("labels") or {}).get(
                    "sagemaker.amazonaws.com/cluster-name"
                )
                != target.hyperpod_name
            ):
                raise ValueError
            inventory[name] = uid
    except (AttributeError, KeyError, TypeError, ValueError):
        raise BootstrapError(
            "target GPU node inventory has missing or conflicting identities"
        ) from None
    return inventory


def _names(output: str, description: str) -> frozenset[str]:
    try:
        values = json.loads(output)
    except ValueError:
        raise BootstrapError(f"{description} is not valid JSON") from None
    if (
        not isinstance(values, list)
        or any(not isinstance(item, str) or not item for item in values)
        or len(values) != len(set(values))
    ):
        raise BootstrapError(f"{description} has ambiguous node names")
    return frozenset(values)


def _cluster_nodes(
    site: RenderedSite, runner: CommandRunner, cluster: dict[str, Any]
) -> frozenset[str]:
    environment = effective_environment(site)
    arguments = ["kubectl"]
    kubeconfig = site.release_config.get("gpu_kubeconfig") or environment.get(
        "KUBECONFIG"
    )
    if kubeconfig:
        arguments.extend(["--kubeconfig", str(kubeconfig)])
    return _names(
        runner.run(
            [
                *arguments,
                "--context",
                str(cluster["context"]),
                "get",
                "nodes",
                "-l",
                "sagemaker.amazonaws.com/cluster-name="
                + str(cluster["hyperpod_cluster_name"]),
                "-o",
                "go-template=" + NODE_NAMES_TEMPLATE,
            ],
            env=environment,
            timeout_seconds=60,
        ),
        "managed cluster node inventory",
    )


def _cpu_keys(site: RenderedSite, runner: CommandRunner) -> frozenset[str]:
    output = runner.run(
        [
            "kubectl",
            "--kubeconfig",
            str(site.release_config["cpu_kubeconfig"]),
            "-n",
            str(site.release_config["namespace"]),
            "get",
            "secret",
            "gpu-fault-node-action-keys",
            "--ignore-not-found",
            "-o",
            "go-template=" + KEY_NAMES_TEMPLATE,
        ],
        env=effective_environment(site),
        sensitive=True,
        timeout_seconds=60,
    )
    return _names(output, "CPU node-key inventory") if output.strip() else frozenset()


def _agent_owners(site: RenderedSite, runner: CommandRunner) -> dict[str, str]:
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
        env=effective_environment(site),
        timeout_seconds=30,
    ).strip()
    if not pod:
        raise BootstrapError("cannot inspect global Agent node ownership")
    output = runner.run(
        [*prefix, "exec", pod, "--", "python3", "-c", AGENT_OWNERS_SCRIPT],
        env=effective_environment(site),
        sensitive=True,
        timeout_seconds=60,
    )
    try:
        owners = json.loads(output)
    except ValueError:
        raise BootstrapError("global Agent ownership is not valid JSON") from None
    if not isinstance(owners, dict) or any(
        not isinstance(node, str)
        or not node
        or not isinstance(cluster, str)
        or not cluster
        for node, cluster in owners.items()
    ):
        raise BootstrapError("global Agent ownership is incomplete")
    return dict(owners)


def read_node_claims(site: RenderedSite, runner: CommandRunner) -> NodeClaims:
    clusters = list(site.release_config["clusters"])
    nodes: dict[str, frozenset[str]] = {}
    with ThreadPoolExecutor(
        max_workers=min(DEPLOY_CONCURRENCY.candidate_clusters, len(clusters) + 2)
    ) as executor:
        keys = executor.submit(copy_context().run, _cpu_keys, site, runner)
        agents = executor.submit(copy_context().run, _agent_owners, site, runner)
        futures = {
            executor.submit(
                copy_context().run, _cluster_nodes, site, runner, cluster
            ): str(cluster["cluster_id"])
            for cluster in clusters
        }
        for future in as_completed(futures):
            nodes[futures[future]] = future.result()
        return NodeClaims(nodes, agents.result(), keys.result())


def assert_batch_node_names_unique(inputs: Sequence[JoinInputs]) -> None:
    owners: dict[str, str] = {}
    for item in inputs:
        for node in item.local["nodes"]:
            if node in owners and owners[node] != item.cluster_id:
                raise BootstrapError(
                    "batch join contains globally conflicting NodeNames"
                )
            owners[node] = item.cluster_id


def verify_join_node_names(
    request: JoinClusterRequest,
    inputs: JoinInputs,
    *,
    claims: NodeClaims,
    state_path: Path,
    state: dict[str, Any],
    runner: CommandRunner,
) -> None:
    raw_nodes = inputs.local.get("nodes")
    if (
        not isinstance(raw_nodes, list)
        or not raw_nodes
        or any(not isinstance(node, str) or not node for node in raw_nodes)
        or len(raw_nodes) != len(set(raw_nodes))
    ):
        raise BootstrapError("join node inventory is empty or ambiguous")
    nodes = set(raw_nodes)
    node_uids = inputs.local.get("node_uids")
    if (
        not isinstance(node_uids, dict)
        or set(node_uids) != nodes
        or any(not isinstance(uid, str) or not uid for uid in node_uids.values())
    ):
        raise BootstrapError("join node inventory has no complete UID binding")
    foreign = {
        node
        for cluster, names in claims.nodes_by_cluster.items()
        if cluster != inputs.cluster_id
        for node in names & nodes
    }
    foreign.update(
        node
        for node in nodes
        if node in claims.agent_owners
        and claims.agent_owners[node] != inputs.cluster_id
    )
    previous = (state.get("evidence") or {}).get("NODE_NAMES_VERIFIED") or {}
    scope = {
        "site_id": request.site.release_config["site_name"],
        "cpu_eks_arn": request.site.release_config["cpu_eks_arn"],
        "cluster_id": inputs.cluster_id,
        "eks_arn": inputs.target.eks_arn,
        "hyperpod_arn": inputs.target.hyperpod_arn,
        "nodes": sorted(nodes),
        "node_uids": dict(node_uids),
    }
    already_claimed = step_done(state, "NODE_KEYS_STARTED") and all(
        previous.get(key) == value for key, value in scope.items()
    )
    if not already_claimed:
        foreign.update(nodes & claims.cpu_key_names)
    if foreign:
        raise BootstrapError(
            "join NodeNames conflict with existing cluster, Agent, or CPU key ownership"
        )
    master = (
        Path(str(inputs.local["fleet_master_file"])).read_text(encoding="utf-8").strip()
    )
    digests = {
        node: hashlib.sha256(
            derive_node_action_secret(master, inputs.cluster_id, node).encode()
        ).hexdigest()
        for node in sorted(nodes)
    }
    gpu_keys = read_node_key_proof(
        runner,
        [
            "kubectl",
            "--kubeconfig",
            str(inputs.local["gpu_kubeconfig"]),
            "--context",
            inputs.target.context,
        ],
        str(request.site.release_config["namespace"]),
    )
    if already_claimed:
        expected = previous.get("expected_key_sha256")
        if (
            not isinstance(expected, dict)
            or set(expected) != nodes
            or any(
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
                for value in expected.values()
            )
        ):
            raise BootstrapError("join retry has no complete node-key digest proof")
        digests = dict(expected)
        if gpu_keys is not None and any(
            node in gpu_keys.digests and gpu_keys.digests[node] != digests[node]
            for node in nodes
        ):
            raise BootstrapError("GPU node keys changed since the join proof")
    elif gpu_keys is not None:
        digests.update(
            {node: gpu_keys.digests[node] for node in nodes & gpu_keys.digests.keys()}
        )
    complete_step(
        state_path,
        state,
        "NODE_NAMES_VERIFIED",
        {**scope, "observed_at": claims.observed_at, "expected_key_sha256": digests},
    )


class BoundNodeKeyRunner(CommandRunner):
    def __init__(self, delegate: CommandRunner, nodes: Mapping[str, str]) -> None:
        super().__init__()
        self._delegate = delegate
        self._nodes = dict(sorted(nodes.items()))

    def run(
        self,
        arguments: Sequence[str],
        *,
        input_text: str | None = None,
        env: Mapping[str, str] | None = None,
        cwd: Path | None = None,
        capture: bool = True,
        sensitive: bool = False,
        mutate: bool = False,
        timeout_seconds: float | None = None,
    ) -> str:
        environment = dict(env or {})
        environment[EXPECTED_NODES_ENV] = json.dumps(self._nodes, separators=(",", ":"))
        environment["GPU_FAULT_ROTATE_NODE_ACTION_KEY"] = ""
        environment["GPU_FAULT_NODE_ACTION_KEYS_SECRET"] = "gpu-fault-node-action-keys"
        environment["GPU_FAULT_CONTROL_PLANE_CONTEXT"] = ""
        return self._delegate.run(
            arguments,
            input_text=input_text,
            env=environment,
            cwd=cwd,
            capture=capture,
            sensitive=sensitive,
            mutate=mutate,
            timeout_seconds=timeout_seconds,
        )

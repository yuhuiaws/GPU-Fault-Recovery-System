"""The distributed-training topology a GPU cluster can actually run.

The checked-in PyTorchJob fixtures are written for three p5en-class nodes: a
Master and two Workers, eight GPUs and sixteen EFA devices per Pod, so the
``all_reduce`` of ``rank + 1`` over 24 ranks is 300.0 and the observation must
name ranks 0..2. ``GF-REGIONAL-WORKLOAD-001/002`` and ``GF-REGIONAL-E2E-001``
hard-coded that shape and refused anything smaller ("at least three
schedulable GPU nodes are required"), which left every case BLOCKED on a
two-node plain-EKS cluster that trains perfectly well on two nodes (live
2026-10-05: 16 ranks, NCCL over EFA, SUCCESS on both).

Three was never a requirement of the cases: the Master/Worker split needs
two Pods on two nodes to prove multi-node NCCL, and every check -- the pod
spread, the ``all_reduce`` sum, the rank set, ``expected_critical_ranks`` --
is a function of the node count and the GPUs per node. This module derives
that function from the cluster: the Ready, schedulable, isolation-taint-free
GPU nodes (the provider's node-group taint is tolerated by the projection),
at most the fixtures' three, at least two, with the GPU count every chosen
node allocates. A three-node HyperPod cluster derives exactly the native
topology and the fixtures are used verbatim; a smaller cluster gets a copy of
the manifest with the Worker replicas, ``--nproc-per-node``, GPU and EFA
requests rewritten, and the topology travels into the case evidence.
"""

from __future__ import annotations

import hashlib
import re
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from scripts.e2e.regional.regional_commands import RegionalFixtureError

#: The fixtures' native shape: three Pods, eight GPUs and sixteen EFA devices each.
NATIVE_NODES = 3
NATIVE_GPUS_PER_NODE = 8
NATIVE_EFA_PER_NODE = 16
#: Multi-node NCCL needs a Master and at least one Worker.
MINIMUM_NODES = 2
GPU_RESOURCE = "nvidia.com/gpu"
EFA_RESOURCE = "vpc.amazonaws.com/efa"
NPROC_PATTERN = re.compile(r"--nproc-per-node=\d+")


class TrainingTopologyError(RegionalFixtureError):
    """The cluster cannot host the fixtures' distributed training at all.

    A fixture refusal: the runners' preflights record it by name like any
    other ``RegionalFixtureError`` instead of dying on an unexpected exception.
    """


@dataclass(frozen=True)
class TrainingTopology:
    node_count: int
    gpus_per_node: int
    candidate_nodes: tuple[str, ...]
    efa_per_node: int = NATIVE_EFA_PER_NODE

    @property
    def world_size(self) -> int:
        return self.node_count * self.gpus_per_node

    @property
    def ranks(self) -> set[int]:
        """The torchrun node ranks the observation must name: one per Pod."""

        return set(range(self.node_count))

    @property
    def is_native(self) -> bool:
        return (
            self.node_count == NATIVE_NODES
            and self.gpus_per_node == NATIVE_GPUS_PER_NODE
            and self.efa_per_node == NATIVE_EFA_PER_NODE
        )

    def all_reduce_marker(self) -> str:
        """The ``all_reduce=`` text every rank prints: sum of ``rank + 1``."""

        return all_reduce_marker(self.world_size)

    def evidence(self) -> dict[str, Any]:
        return {
            "node_count": self.node_count,
            "gpus_per_node": self.gpus_per_node,
            "efa_per_node": self.efa_per_node,
            "world_size": self.world_size,
            "ranks": sorted(self.ranks),
            "candidate_nodes": list(self.candidate_nodes),
            "native": self.is_native,
        }


NATIVE_TOPOLOGY = TrainingTopology(NATIVE_NODES, NATIVE_GPUS_PER_NODE, ())


def schedulable_training_nodes(
    nodes: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    """Ready, schedulable GPU nodes with no isolating taint, by name."""

    return sorted(
        (
            item
            for item in nodes
            if item.get("ready") == "True"
            and not item.get("unschedulable")
            and not item.get("taints")
        ),
        key=lambda item: str(item.get("name")),
    )


def _allocatable_gpus(node: Mapping[str, Any]) -> int | None:
    value = node.get("gpu_allocatable")
    if value is None:
        return None
    try:
        count = int(str(value))
    except (TypeError, ValueError):
        return None
    return count if count > 0 else None


def derive_training_topology(
    nodes: Sequence[Mapping[str, Any]],
    *,
    maximum_nodes: int = NATIVE_NODES,
    minimum_nodes: int = MINIMUM_NODES,
    schedulable_only: bool = True,
) -> TrainingTopology:
    """The topology the cluster's schedulable GPU nodes can run, or a refusal.

    The node count is the smaller of the fixtures' three and the schedulable
    count; fewer than two is a named refusal. The GPUs per Pod are what every
    chosen node allocates when the projection reports it, else the fixtures'
    eight; nodes that disagree are refused rather than averaged, because a
    Pod's GPU request is one number. ``schedulable_only=False`` takes ``nodes``
    as already filtered (a preflight's candidate list).
    """

    candidates = (
        schedulable_training_nodes(nodes)
        if schedulable_only
        else sorted(nodes, key=lambda item: str(item.get("name")))
    )
    if len(candidates) < minimum_nodes:
        names = ", ".join(str(item.get("name")) for item in candidates) or "none"
        raise TrainingTopologyError(
            f"at least {minimum_nodes} schedulable GPU nodes are required for "
            f"distributed training; found {len(candidates)} ({names})"
        )
    chosen = candidates[:maximum_nodes]
    counts = {_allocatable_gpus(item) for item in chosen}
    counts.discard(None)
    if len(counts) > 1:
        raise TrainingTopologyError(
            "the schedulable GPU nodes allocate different GPU counts: "
            + ", ".join(
                f"{item.get('name')}={_allocatable_gpus(item)}" for item in chosen
            )
        )
    gpus_per_node = int(next(iter(counts))) if counts else NATIVE_GPUS_PER_NODE
    efa_per_node = (
        NATIVE_EFA_PER_NODE * gpus_per_node // NATIVE_GPUS_PER_NODE
        if gpus_per_node != NATIVE_GPUS_PER_NODE
        else NATIVE_EFA_PER_NODE
    )
    return TrainingTopology(
        node_count=len(chosen),
        gpus_per_node=gpus_per_node,
        candidate_nodes=tuple(str(item.get("name")) for item in chosen),
        efa_per_node=efa_per_node,
    )


def preflight_topology(
    candidates: Sequence[Mapping[str, Any]], errors: list[str]
) -> TrainingTopology | None:
    """The topology an already-filtered candidate list can run, or ``None``
    with the refusal appended to ``errors`` (a preflight's reading)."""

    try:
        return derive_training_topology(candidates, schedulable_only=False)
    except TrainingTopologyError as exc:
        errors.append(str(exc))
        return None


def topology_limitations(topology: TrainingTopology, *, what: str) -> list[str]:
    """What a run on a non-native topology cannot prove; empty when native.

    The restart and recovery contracts are proven for the derived world size,
    and "the other nodes were untouched" is observed on the peers that exist:
    one on a two-node cluster, not the two a three-node job leaves beside the
    target.
    """

    if topology.is_native:
        return []
    return [
        f"{what} ran on the derived {topology.node_count}-node/"
        f"{topology.world_size}-rank topology, not the native {NATIVE_NODES}-node/"
        f"{NATIVE_TOPOLOGY.world_size}-rank job: the restart and recovery "
        "contracts are proven for that world size only, and 'other nodes "
        f"untouched' is observed on {topology.node_count - 1} peer node(s) "
        f"rather than {NATIVE_NODES - 1}."
    ]


def _rewrite_container(container: dict[str, Any], topology: TrainingTopology) -> None:
    command = container.get("command")
    if isinstance(command, list):
        container["command"] = [
            NPROC_PATTERN.sub(f"--nproc-per-node={topology.gpus_per_node}", str(part))
            if isinstance(part, str)
            else part
            for part in command
        ]
    args = container.get("args")
    if isinstance(args, list):
        container["args"] = [
            NPROC_PATTERN.sub(f"--nproc-per-node={topology.gpus_per_node}", str(part))
            if isinstance(part, str)
            else part
            for part in args
        ]
    resources = container.get("resources")
    if not isinstance(resources, dict):
        return
    for field in ("requests", "limits"):
        values = resources.get(field)
        if not isinstance(values, dict):
            continue
        if GPU_RESOURCE in values:
            values[GPU_RESOURCE] = str(topology.gpus_per_node)
        if EFA_RESOURCE in values:
            if topology.efa_per_node > 0:
                values[EFA_RESOURCE] = str(topology.efa_per_node)
            else:
                del values[EFA_RESOURCE]


def render_topology_manifest(
    source: Path, topology: TrainingTopology, destination: Path | None = None
) -> Path:
    """``source`` for the native topology, else a rewritten copy for ``topology``.

    Only PyTorchJob manifests are rewritten: the Worker replicas become
    ``node_count - 1`` (the Master is the other Pod) and every container's
    ``--nproc-per-node``, GPU and EFA requests follow the GPUs per node. The
    copy lives beside the case evidence (``destination``) or in a private
    temporary directory, and a HyperPod cluster never sees a copy at all.
    """

    if topology.is_native:
        return source
    document = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or str(document.get("kind")) != "PyTorchJob":
        raise TrainingTopologyError(
            f"{source.name} is not a PyTorchJob; the topology cannot be rewritten"
        )
    replicas = (document.get("spec") or {}).get("pytorchReplicaSpecs")
    if not isinstance(replicas, dict) or "Master" not in replicas:
        raise TrainingTopologyError(f"{source.name} has no Master replica spec")
    if topology.node_count > 1:
        worker = replicas.get("Worker")
        if not isinstance(worker, dict):
            raise TrainingTopologyError(f"{source.name} has no Worker replica spec")
        worker["replicas"] = topology.node_count - 1
    else:
        replicas.pop("Worker", None)
    for replica in replicas.values():
        if not isinstance(replica, dict):
            continue
        pod_spec = (replica.get("template") or {}).get("spec") or {}
        for container in pod_spec.get("containers") or []:
            if isinstance(container, dict):
                _rewrite_container(container, topology)
    if destination is None:
        destination = (
            Path(tempfile.mkdtemp(prefix="gpu-fault-training-topology-")) / source.name
        )
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    destination.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    destination.chmod(0o600)
    return destination


def all_reduce_marker(world_size: int) -> str:
    """The ``all_reduce=`` text every rank prints for ``world_size`` ranks."""

    return f"all_reduce={world_size * (world_size + 1) / 2}"


def manifest_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

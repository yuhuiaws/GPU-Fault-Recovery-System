"""Runtime container ids: how a node finds the cgroups of a managed job.

The hung-diagnostic chain enumerates a job's processes by cgroup
(``workload_cgroup_paths_by_node``) and attaches to nothing outside it. The
watcher knew a container's cgroup only from a ``gpu-fault.io/cgroup-path`` Pod
annotation that nothing sets -- the path names the runtime's container id,
which exists only once the container runs -- so every live managed job planned
``{node: []}``: each COLLECT_HUNG_SUMMARY answered ``SCOPE_UNKNOWN`` and
COLLECT_HUNG_TRIAGE never sampled a rank (HyperPod, GF-GENERAL-LIVE-011, every
variant, 2026-10-24/25).

What the watcher does read for every running container is the id the kubelet
reports in ``status.containerStatuses[].containerID`` (``containerd://<id>``).
The control plane sends those ids per node (:data:`WORKLOAD_CONTAINER_IDS`)
for the containers that name no cgroup themselves, and the Node Agent finds
each one's cgroup in its own ``/proc/<pid>/cgroup``, where every CRI runtime
names a container's cgroup after the same id: ``cri-containerd-<id>.scope``
(containerd, systemd driver), ``crio-<id>.scope``, ``docker-<id>.scope``, or the
bare ``<id>`` (cgroupfs driver). CRI-O's monitor runs beside the container in
``crio-conmon-<id>.scope`` and is not part of it.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from typing import Any

#: Hung-diagnostic step parameter, per node: the runtime ids of the attempt's
#: running containers on that node that carry no cgroup path of their own.
WORKLOAD_CONTAINER_IDS = "workload_container_ids_by_node"
#: At most this many ids are planned or resolved for one node.
MAX_NODE_CONTAINER_IDS = 64

_RUNTIME_ID = re.compile(r"[0-9a-f]{64}")
_CONTAINER_SEGMENT = re.compile(
    r"(?:(?:cri-containerd|crio|docker)-)?(?P<id>[0-9a-f]{64})(?:\.scope)?"
)


def runtime_container_id(value: object) -> str | None:
    """The 64-hex runtime id in a ``containerID`` (``<runtime>://<id>``).

    Anything else -- a short or mixed-case id, no id at all -- is ``None``: a
    value no node could match proves nothing about a process.
    """

    if not isinstance(value, str):
        return None
    identifier = value.rpartition("://")[2]
    return identifier if _RUNTIME_ID.fullmatch(identifier) else None


def segment_container_id(segment: str) -> str | None:
    """The runtime id one cgroup path segment names as its container."""

    match = _CONTAINER_SEGMENT.fullmatch(segment)
    return match.group("id") if match is not None else None


def container_ids_by_node(
    containers: Iterable[Any], node_ids: Sequence[str]
) -> dict[str, list[str]]:
    """Per node, the runtime ids of the running containers naming no cgroup.

    ``containers`` are an attempt's ``ContainerObservation`` rows. A container
    that carries an explicit ``cgroup_path`` is scoped by it already; an id no
    node could match (:func:`runtime_container_id`) is left out.
    """

    result: dict[str, set[str]] = {node: set() for node in node_ids}
    for container in containers:
        identifier = runtime_container_id(container.container_id)
        if (
            container.node_id in result
            and not container.terminated
            and not container.cgroup_path
            and identifier is not None
        ):
            result[container.node_id].add(identifier)
    return {
        node: sorted(values)[:MAX_NODE_CONTAINER_IDS] for node, values in result.items()
    }

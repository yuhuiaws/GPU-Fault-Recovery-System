"""This node's cgroups of a managed job, resolved from its runtime container ids.

Why the control plane sends ids instead of paths: :mod:`gpu_fault.
workload_container_scope`. The ids come from the Kubernetes API, the paths from
this host's own ``/proc``: a process is in scope only when the cgroup its
runtime placed it in is named after one of the attempt's containers. The
resolved paths join the explicit ones in ``workload_cgroup_paths_by_node``, so
every later identity check (survey, triage, FR series) reads one scope.
"""

from __future__ import annotations

import errno
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from gpu_fault.node_agent.operations.hung_summary_io import (
    MAX_SCAN_PROCESSES,
    read_bounded,
)
from gpu_fault.node_agent.protocol import NodeActionCommand
from gpu_fault.workload_container_scope import (
    MAX_NODE_CONTAINER_IDS,
    WORKLOAD_CONTAINER_IDS,
    runtime_container_id,
    segment_container_id,
)

WORKLOAD_CGROUP_PATHS = "workload_cgroup_paths_by_node"
#: The evidence names at most this many unresolved ids (all are counted).
REPORTED_UNRESOLVED_IDS = 8


def container_cgroups(
    proc_root: Path, container_ids: set[str]
) -> tuple[dict[str, set[str]], bool]:
    """The cgroup directory of each id in ``container_ids`` with a process here.

    One process of a container carries all of the container's paths (one per
    cgroup v1 hierarchy), so the scan stops once every id is seen. The flag is
    ``False`` when the scan was cut short (process bound, unreadable entries)
    before every id was seen: an id missing then may exist all the same.
    """

    found: dict[str, set[str]] = {}
    complete = True
    scanned = 0
    try:
        for directory in proc_root.iterdir():
            if len(found) == len(container_ids):
                break
            if not directory.name.isdigit():
                continue
            scanned += 1
            if scanned > MAX_SCAN_PROCESSES:
                complete = False
                break
            try:
                text = read_bounded(directory / "cgroup")
            except OSError as exc:
                # A process that exited meanwhile is normal; an unreadable
                # membership is not proof that it is someone else's.
                complete = complete and exc.errno in {errno.ENOENT, errno.ESRCH}
                continue
            except ValueError:
                complete = False
                continue
            for line in text.splitlines():
                parts = line.split(":", 2)
                path = parts[2].strip().rstrip("/") if len(parts) == 3 else ""
                if not path.startswith("/"):
                    continue
                segments = path.split("/")
                for index, segment in enumerate(segments):
                    identifier = segment_container_id(segment)
                    if identifier is not None and identifier in container_ids:
                        # The container's own cgroup; a process in a cgroup
                        # the container made below it is still inside.
                        found.setdefault(identifier, set()).add(
                            "/".join(segments[: index + 1])
                        )
                        break
    except OSError:
        complete = False
    return found, complete or len(found) == len(container_ids)


def scope_without_container_ids(explicit: Any) -> dict[str, Any]:
    """The evidence of a scope no runtime container id contributed to."""

    if explicit:
        return {"basis": "EXPLICIT_CGROUP_PATHS"}
    return {"basis": "UNAVAILABLE", "reason": "NO_CGROUP_PATH_OR_CONTAINER_ID"}


def resolve_container_ids(
    proc_root: Path, identifiers: Sequence[str], explicit: Any = ()
) -> tuple[list[str], dict[str, Any]]:
    """The cgroups of ``identifiers`` on this node, and how far they resolved.

    ``identifiers`` are valid runtime ids, unique and sorted. An id with no
    process here resolves to nothing; the evidence counts it (and names up to
    :data:`REPORTED_UNRESOLVED_IDS`) so a missing container never reads as an
    empty one. ``explicit`` are the command's own cgroup paths, only counted.
    """

    found, complete = container_cgroups(proc_root, set(identifiers))
    resolved = sorted({path for paths in found.values() for path in paths})
    unresolved = [identifier for identifier in identifiers if identifier not in found]
    evidence: dict[str, Any] = {
        "basis": "RUNTIME_CONTAINER_ID",
        "container_count": len(identifiers),
        "resolved_container_count": len(found),
        "unresolved_container_count": len(unresolved),
        "unresolved_container_ids": unresolved[:REPORTED_UNRESOLVED_IDS],
        "cgroup_path_count": len(resolved),
        "complete": not unresolved,
    }
    if unresolved:
        evidence["reason"] = (
            "CONTAINER_CGROUP_NOT_FOUND" if complete else "PROCESS_SCAN_INCOMPLETE"
        )
    if explicit:
        evidence["explicit_cgroup_path_count"] = len(explicit)
    return resolved, evidence


def resolve_workload_scope(
    proc_root: Path, command: NodeActionCommand
) -> tuple[NodeActionCommand, dict[str, Any]]:
    """``command`` with this node's container cgroups added to its scope.

    The explicit paths stay first and as they came (the callers' own checks
    still refuse a malformed one); the evidence says where the scope came from
    and which containers it could not find. Without a usable id the command is
    returned unchanged.
    """

    parameters = command.parameters
    mapping = parameters.get(WORKLOAD_CGROUP_PATHS)
    explicit = mapping.get(command.node_id) if isinstance(mapping, dict) else None
    explicit = [] if explicit is None else explicit
    by_node = parameters.get(WORKLOAD_CONTAINER_IDS)
    requested = by_node.get(command.node_id) if isinstance(by_node, dict) else None
    identifiers = sorted(
        {
            identifier
            for value in (requested if isinstance(requested, list) else [])[
                :MAX_NODE_CONTAINER_IDS
            ]
            if (identifier := runtime_container_id(value)) is not None
        }
    )
    if not identifiers:
        return command, scope_without_container_ids(explicit)
    resolved, evidence = resolve_container_ids(proc_root, identifiers, explicit)
    if not resolved or not isinstance(explicit, list):
        return command, evidence
    scope = list(dict.fromkeys([*explicit, *resolved]))
    updated = command.model_copy(
        update={
            "parameters": {
                **parameters,
                WORKLOAD_CGROUP_PATHS: {
                    **(mapping if isinstance(mapping, dict) else {}),
                    command.node_id: scope,
                },
            }
        }
    )
    return updated, evidence


def quiesce_workload_scope(
    proc_root: Path, command: NodeActionCommand, explicit: set[str]
) -> tuple[set[str], dict[str, Any]]:
    """The QUIESCE device sweep's workload cgroups here, and their evidence.

    The sweep TERM/KILLs a holder of the target GPU that runs in one of these
    cgroups, so the attempt's runtime ids are read more strictly than the
    read-only hung chain reads them: a value this Agent cannot read in full
    refuses the quiesce before anything stops, as a malformed explicit path
    does (``_quiesce_scope``). An id with no process here adds nothing -- after
    STOP_WORKLOADS that is the normal case, the container has nothing left to
    sweep -- and a scan cut short adds only the containers it saw. A holder
    left outside the scope stays alive, and VERIFY_NO_GPU_CLIENTS, the
    authority on survivors, refuses the reset on it. Nothing outside the
    attempt's own containers enters the scope.
    """

    by_node = command.parameters.get(WORKLOAD_CONTAINER_IDS)
    if by_node is None:
        return explicit, scope_without_container_ids(explicit)
    if not isinstance(by_node, dict):
        raise RuntimeError(
            f"{WORKLOAD_CONTAINER_IDS} must map node ids to lists of runtime "
            "container ids"
        )
    requested = by_node.get(command.node_id)
    requested = [] if requested is None else requested
    if not isinstance(requested, list) or len(requested) > MAX_NODE_CONTAINER_IDS:
        raise RuntimeError(
            f"{WORKLOAD_CONTAINER_IDS} of {command.node_id} must be a list of at "
            f"most {MAX_NODE_CONTAINER_IDS} runtime container ids"
        )
    identifiers: set[str] = set()
    for value in requested:
        identifier = runtime_container_id(value)
        if identifier is None:
            raise RuntimeError(
                f"{WORKLOAD_CONTAINER_IDS} of {command.node_id} holds a value "
                "that is not a 64-hex runtime container id"
            )
        identifiers.add(identifier)
    if not identifiers:
        return explicit, scope_without_container_ids(explicit)
    resolved, evidence = resolve_container_ids(proc_root, sorted(identifiers), explicit)
    return explicit | set(resolved), evidence


def target_workload_scope(
    proc_root: Path, cgroup_paths: Sequence[str], container_ids: Sequence[str]
) -> tuple[list[str], dict[str, Any]]:
    """A passive-monitor target's scope on this node, and its evidence.

    The target's explicit cgroups first, then the cgroups of its containers
    (``CommunicationTarget.container_ids``, validated runtime ids). Read-only:
    the sampler reads ``/proc`` and attaches to nothing.
    """

    explicit = list(cgroup_paths)
    if not container_ids:
        return explicit, scope_without_container_ids(explicit)
    resolved, evidence = resolve_container_ids(
        proc_root, sorted(set(container_ids)), explicit
    )
    return list(dict.fromkeys([*explicit, *resolved])), evidence

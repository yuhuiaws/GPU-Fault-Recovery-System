"""Where a scope's GPUs are, so a step built for one node names only its GPUs.

A single-node VALIDATE_GPU answers for every GPU it names
(``adapters/gpu_validation.py``): its node must report fresh telemetry for
each one, and VALIDATE_GPU / VALIDATE_FABRIC count only the findings on the
named GPUs. A GPU of another node named on such a step can never report there
-- the step waits out its per-step cap and fails into an escalation of a node
that may be healthy -- and a step naming only another node's GPUs ignores its
own node's findings. The escalation successor narrows the source's nodes to
the failed ones; when the source workflow had not recorded which GPU sits on
which node, it handed the narrowed successor every GPU of the source. Since
batch 146 an attempt incident names every rank's GPUs on every node, so a
survivor's or the dead rank's GPUs landed on a node that has none (batch 149).

Placement, strongest first: what the scope's nodes report themselves -- the
GPU inventory snapshot and the latest per-GPU telemetry (a GPU that fell off
the bus keeps its last sample there and stays placed on its node) -- and the
attempt's container observations (the node and the GPUs the kubelet
assigned); then the source workflow's own per-node lists (``gpu_uuids_by_node``
on a step, a single-node step's GPUs) for the GPUs the store says nothing
about; last, a single-node source incident places its own GPUs on its node.
The validation rule itself is not weakened: a step still needs fresh evidence
for every GPU it names.

Decision for a GPU nothing places (``escalation_gpu_scope``): when every node
of the source scope is one of the successor's nodes, the GPU can only be on
one of them and stays in the successor's scope (fail-closed, as before). When
the source spans other nodes it may sit on a node that did not fail: it is
named on none of the successor's steps (it stays on the source incident), and
each failed node's VALIDATE_GPU / VALIDATE_HOST / VALIDATE_FABRIC names the
node's full known GPU inventory instead -- every GPU the node reports must
report fresh, and a node nothing is known about validates node-wide (fresh
node telemetry, every finding on the node). Such a GPU was never reported by
any node of the scope, so the failed node gives up no evidence it could have
produced; naming it would only make a healthy node wait out the cap.
"""

from __future__ import annotations

import logging
from collections.abc import Collection, Iterable, Mapping, Sequence
from typing import Any, NamedTuple

from gpu_fault.models import (
    FaultIncident,
    WorkflowOperation,
    WorkflowRequest,
    WorkflowStepSpec,
)

LOGGER = logging.getLogger(__name__)

# The steps whose GPU list is a per-GPU evidence requirement on their nodes.
# VALIDATE_HOST reads no GPU; it carries the same scope so the plan reads
# consistently.
VALIDATION_OPERATIONS = frozenset(
    {
        WorkflowOperation.VALIDATE_GPU,
        WorkflowOperation.VALIDATE_HOST,
        WorkflowOperation.VALIDATE_FABRIC,
    }
)


class EscalationGpuScope(NamedTuple):
    """The GPU scope of an escalation successor over its failed nodes."""

    #: The successor incident's GPUs: the source GPUs placed on a failed node.
    gpu_uuids: list[str]
    #: Per failed node, for node-action steps: only when every source GPU is
    #: placed and every failed node holds some (else empty).
    gpu_uuids_by_node: dict[str, list[str]]
    #: Per failed node, the GPUs its VALIDATE_* steps name.
    validation_gpu_uuids_by_node: dict[str, list[str]]


def _dedup(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(values))


def workflow_gpu_placement(workflow: WorkflowRequest) -> dict[str, list[str]]:
    """The per-node GPUs the workflow recorded itself.

    An explicit ``gpu_uuids_by_node`` entry for a node of its step wins over
    the GPUs of the node's single-node steps (a job DAG branch carries its
    node-local GPUs directly). A multi-node step's GPU list says nothing about
    which of its nodes holds which GPU.
    """

    explicit: dict[str, list[str]] = {}
    derived: dict[str, list[str]] = {}
    for step in workflow.official_steps:
        mapping = step.parameters.get("gpu_uuids_by_node", {})
        if isinstance(mapping, dict):
            for node_id, values in mapping.items():
                if (
                    node_id in step.node_ids
                    and isinstance(values, list)
                    and values
                    and all(isinstance(value, str) and value for value in values)
                ):
                    explicit.setdefault(node_id, []).extend(values)
        if len(step.node_ids) == 1 and step.gpu_uuids:
            derived.setdefault(step.node_ids[0], []).extend(step.gpu_uuids)
    return {
        node_id: _dedup(explicit.get(node_id) or derived[node_id])
        for node_id in explicit.keys() | derived.keys()
    }


def _read(what: str, subject: str, store: Any, method: str, *args: Any) -> Any:
    try:
        return getattr(store, method)(*args)
    except Exception:  # noqa: BLE001 - an unknown placement is not an error here
        LOGGER.warning("GPU placement: %s unavailable for %s", what, subject)
        return None


def store_gpu_placement(
    store: Any,
    cluster_id: str,
    node_ids: Iterable[str],
    *,
    attempt_id: str | None = None,
) -> dict[str, set[str]]:
    """The GPUs each node is known to carry, read from the store.

    The node's GPU inventory snapshot (what it enumerates now), the GPUs its
    latest per-GPU telemetry names (one that stopped reporting included) and,
    for ``attempt_id``, the GPUs the attempt's container observations put on a
    node. An unreadable source contributes nothing.
    """

    placement: dict[str, set[str]] = {}
    for node_id in dict.fromkeys(node_ids):
        known = placement.setdefault(node_id, set())
        snapshot = _read(
            "the GPU inventory snapshot",
            node_id,
            store,
            "get_gpu_inventory_snapshot",
            cluster_id,
            node_id,
        )
        known.update(device.gpu_uuid for device in getattr(snapshot, "devices", ()))
        latest = _read(
            "the latest GPU telemetry",
            node_id,
            store,
            "list_gpu_metrics_latest",
            cluster_id,
            node_id,
        )
        known.update(
            str(gpu_uuid)
            for item in latest or ()
            if (gpu_uuid := getattr(item.sample, "gpu_uuid", None))
        )
    if attempt_id:
        observations = _read(
            "the container observations",
            attempt_id,
            store,
            "list_attempt_observations",
            cluster_id,
        )
        for observation in observations or ():
            if observation.attempt_id != attempt_id:
                continue
            for container in observation.containers:
                if container.node_id and container.gpu_uuids:
                    placement.setdefault(container.node_id, set()).update(
                        container.gpu_uuids
                    )
    return {node_id: known for node_id, known in placement.items() if known}


def escalation_gpu_scope(
    workflow: WorkflowRequest,
    source: FaultIncident,
    failed_nodes: Sequence[str],
    store: Any = None,
    *,
    lineage_node_ids: Iterable[str] = (),
) -> EscalationGpuScope:
    """Resolve the failed nodes' GPUs without assigning a sibling's to them.

    What a node reports itself (``store_gpu_placement``, read when ``store`` is
    given) places a GPU first: a step that narrowed its nodes but kept the
    union of GPUs (a RESTORE_SCHEDULING trimmed to the readmitted node, a
    compensation restore) is not proof that a sibling's GPU sits here. The
    workflow's own lists place the GPUs the store says nothing about, and a
    single-node source incident places the rest of its own GPUs on its node.
    A GPU placed only on nodes that did not fail is not the successor's; a GPU
    nothing places follows the decision in the module docstring. Without store
    evidence the workflow's lists decide alone, as before. ``lineage_node_ids``
    are the nodes of the escalations ``source`` itself descends from: a
    successor an earlier release built carries its source's whole GPU list on
    single-node steps, and those nodes' reports say where those GPUs are.
    """

    recorded = workflow_gpu_placement(workflow)
    all_gpus = _dedup(
        [
            *source.gpu_uuids,
            *(gpu for step in workflow.official_steps for gpu in step.gpu_uuids),
            *(gpu for node_id in sorted(recorded) for gpu in recorded[node_id]),
        ]
    )
    scope_nodes = {
        *source.node_ids,
        *(node_id for step in workflow.official_steps for node_id in step.node_ids),
        *lineage_node_ids,
    }
    stored = (
        {}
        if store is None
        else store_gpu_placement(
            store, source.cluster_id, sorted(scope_nodes), attempt_id=source.attempt_id
        )
    )
    scope_nodes.update(stored)
    in_scope = set(all_gpus)
    owners: dict[str, set[str]] = {}
    for node_id, values in stored.items():
        for gpu in values & in_scope:
            owners.setdefault(gpu, set()).add(node_id)
    reported = set(owners)
    for node_id, gpus in recorded.items():
        for gpu in set(gpus) - reported:
            owners.setdefault(gpu, set()).add(node_id)
    if len(source.node_ids) == 1:
        for gpu in source.gpu_uuids:
            owners.setdefault(gpu, {source.node_ids[0]})
    shares = {
        node_id: _dedup(
            gpu
            for gpu in [*recorded.get(node_id, ()), *all_gpus]
            if node_id in owners.get(gpu, ())
        )
        for node_id in failed_nodes
    }
    unplaced = [gpu for gpu in all_gpus if not owners.get(gpu)]
    if unplaced and scope_nodes <= set(failed_nodes):
        # What nothing places can only be on the failed nodes: keep all of it.
        return EscalationGpuScope(
            all_gpus,
            {},
            {node: _dedup([*shares[node], *unplaced]) for node in failed_nodes},
        )
    complete = not unplaced and all(shares.values())
    # The node's full known inventory: its share and every GPU it reports.
    validation = {
        node_id: (
            shares[node_id]
            if shares[node_id] and not unplaced
            else _dedup([*shares[node_id], *sorted(stored.get(node_id, ()))])
        )
        for node_id in failed_nodes
    }
    return EscalationGpuScope(
        _dedup(gpu for node_id in failed_nodes for gpu in shares[node_id]),
        dict(shares) if complete else {},
        validation,
    )


def scope_validation_steps(
    steps: Sequence[WorkflowStepSpec],
    validation_gpu_uuids_by_node: Mapping[str, Collection[str]],
) -> list[WorkflowStepSpec]:
    """Give each VALIDATE_* step the GPUs its own nodes answer for."""

    scoped = []
    for step in steps:
        if step.operation in VALIDATION_OPERATIONS and all(
            node_id in validation_gpu_uuids_by_node for node_id in step.node_ids
        ):
            step = step.model_copy(
                update={
                    "gpu_uuids": _dedup(
                        gpu
                        for node_id in step.node_ids
                        for gpu in validation_gpu_uuids_by_node[node_id]
                    )
                }
            )
        scoped.append(step)
    return scoped

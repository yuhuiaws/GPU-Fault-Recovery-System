"""The inventory verdict a validation reads from an erroring host collector.

The validation adapter (``adapters/gpu_validation.py``) judges a node only on
collectors whose newest batch is clean: a collection error names a reading the
collector could not take, so the rows such a batch lacks would read as clean
and the step waits for a clean one. The inventory gate is the exception this
module carries: the inventory contributors withhold the active count when they
have no reading (an unreadable sysfs tree, a GPU query that listed nothing --
no row, no verdict), so a count they did report after the recovery action is
the node's own, whatever another contributor of the same tick could not read.
Such a batch can fail the node; it never passes it.

Live 2026-10-27 (GF-GENERAL-LIVE-013 ``post_repair_failure``, both clusters):
the trap re-unbound the EFA function 4 s after REMEDIATE_EFA_DRIVER's rebind,
the host collector TRIGGER_HEALTH_SNAPSHOT restarted delivered
``efa_inventory_active_count=15`` within seconds and kept delivering it, but
every batch also carried ``_rdma``'s "incomplete EFA traffic coverage: 15/16
devices", so HOST_TELEMETRY stayed COLLECTION_ERROR, the adapter answered
``missing_or_stale_collectors`` before a row was read, VALIDATE_FABRIC waited
out the 600 s cap, and the REBOOT_NODE the plan asked for was withheld as
missing evidence (``validation_evidence_missing``) -- the designed answer to a
node nothing was learned about, given to a node whose failure the collector
had reported for ten minutes.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from gpu_fault.execution import WorkflowStepContext
from gpu_fault.models import WorkflowOperation
from gpu_fault.observation_health import reported_since
from gpu_fault.telemetry import CollectorKind


def collector_statuses(
    store: Any, cluster_id: str, node_id: str
) -> dict[CollectorKind, Any]:
    """The node's collector statuses by kind."""

    return {
        item.collector: item
        for item in store.list_collector_statuses(cluster_id, node_id)
    }


def inventory_contributor_error(error: str) -> bool:
    """Whether a host collection error was raised by an inventory contributor.

    The host collector prefixes every collection error with the contributor
    that raised it (``collectors/host/collector.py`` ``collect_once``:
    ``_gpu_inventory``, ``_efa_inventory``, ``_rdma``, ...). One from an
    inventory contributor means the count of that batch was not read in full
    (a GPU query that exited non-zero after listing the GPUs that still answer
    ships the partial count with its error), so the inventory gate reads no
    verdict from it -- unreadable is UNKNOWN; an error of any other
    contributor leaves the counts the inventory contributors did report
    standing.
    """

    return error.split(":", 1)[0].strip().endswith("_inventory")


def inventory_failure_despite_collection_error(
    adapter: Any,
    context: WorkflowStepContext,
    node_id: str,
    operation: WorkflowOperation,
    requirements: dict[str, Any],
    recovery_operations: frozenset[WorkflowOperation],
    completed_recovery: bool,
    now: datetime,
    readiness_after: datetime | None,
    statuses: dict[CollectorKind, Any],
    failures: dict[str, list[str]],
    clamped: dict[str, Any],
    evidence: dict[str, Any],
    collector_errors: dict[str, list[str]],
) -> bool:
    """Fail ``node_id`` on an inventory row its host collector delivered since
    the action, although a collector the step waits for is stale or erroring.

    ``adapter`` is the validation adapter: its sample ages, its store and its
    inventory gate (``_validate_inventory``) judge the rows exactly as on the
    clean path -- the requirement, the instance-type bound and the observation
    window (``inventory_observation_window``: a row newer than the last
    recovery step) are the same; only the collector check in front of them is
    answered by ``reported_since`` (the newest batch of either kind, within
    the post-action age and after the action) instead of by a clean status.
    A failing row goes to ``failures`` / ``clamped`` / ``evidence`` as the
    gate's own verdict and the batch's collection errors to
    ``collector_errors`` (``details.host_collector_errors``); the batch of an
    inventory contributor's own error, a row older than the action or a row
    that meets the requirement is no verdict (``False``): the caller keeps the
    node pending on its collectors as before.
    """

    max_age = (
        adapter.post_action_max_sample_age
        if readiness_after is not None
        else adapter.max_sample_age
    )
    host = statuses.get(CollectorKind.HOST_TELEMETRY)
    if host is None or not reported_since(
        host, now, max_age.total_seconds(), after=readiness_after
    ):
        return False
    if any(inventory_contributor_error(error) for error in host.errors):
        return False
    all_latest = adapter.store.list_telemetry_metrics_latest(
        context.incident.cluster_id, node_id
    )
    latest = [
        item
        for item in all_latest
        if now - item.observed_at <= max_age
        and (readiness_after is None or item.observed_at > readiness_after)
    ]
    observed: dict[str, list[str]] = {}
    observed_clamped: dict[str, Any] = {}
    observed_evidence: dict[str, Any] = {}
    waiting = adapter._validate_inventory(
        context,
        node_id,
        operation,
        requirements,
        recovery_operations,
        completed_recovery,
        all_latest,
        latest,
        observed,
        observed_clamped,
        observed_evidence,
    )
    if waiting is not None or not observed.get(node_id):
        return False
    failures.setdefault(node_id, []).extend(observed[node_id])
    clamped.update(observed_clamped)
    evidence.update(observed_evidence)
    if host.errors:
        collector_errors[node_id] = list(host.errors)
    return True

"""The node-level finding a collector's failed scrape becomes.

A collector reports a scrape it could not perform as a sample-less batch with
``collection_errors`` (F5). The ingest books that as one WARNING
``dcgm_field_completeness`` finding per node, keyed here, and clears it on the
node's next clean batch.

Booked is not actionable. The finding carries NO_ACTION, and so opens no
incident, until the node's streak of error batches -- consecutive, no clean
batch in between -- has lasted ``collection_error_sustain_seconds``, measured
on batch ``observed_at`` from the streak's first error batch; from then on it
asks for RUN_DIAGNOSTICS until a clean batch ends the streak. Live 2026-10-17
(gpu-eks-cluster): a release re-rendered the dcgm-exporter DaemonSet, the one
scrape that fell inside the Pod restart failed, the incident opened at once,
its VALIDATE_GPU ran 3 s before the next clean batch cleared the finding and
refused, and the DRAIN successor quarantined a healthy node.

The streak rides in the booked finding, so it survives across batches on every
store backend without a schema change: the finding keeps the ``observed_at``
and the ``finding_id`` of the streak's first error batch, the state's
``observed_at`` is the latest batch's. Crossing the window changes the action,
which the store reports as an activation -- the thermal consecutive-breach
model -- under one event id per streak.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from gpu_fault.gpu_metric_models import (
    SITE_METRIC_POLICY_VERSION,
    GpuFindingState,
    GpuHealthFinding,
    GpuHealthSeverity,
)
from gpu_fault.models import RecoveryAction

if TYPE_CHECKING:
    from gpu_fault.gpu_metrics import GpuMetricBatch, GpuMetricsThresholds


def collection_error_key(batch: GpuMetricBatch) -> tuple[str, str, str, str]:
    return (
        batch.cluster_id,
        batch.node_id,
        "node",
        "dcgm_field_completeness",
    )


def collection_error_actionable(finding: GpuHealthFinding) -> bool:
    """Whether a booked collection-error finding has outlasted its window."""

    return finding.automatic_action != RecoveryAction.NO_ACTION.value


def collection_error_finding(
    batch: GpuMetricBatch,
    previous: GpuFindingState | None,
    thresholds: GpuMetricsThresholds,
) -> tuple[tuple[str, str, str, str], GpuHealthFinding, bool] | None:
    """The finding to book for ``batch``, its key, and whether it may open
    an incident (it has outlasted the sustain window); None for a clean one."""

    if not batch.collection_errors:
        return None
    key = collection_error_key(batch)
    sustain_seconds = thresholds.collection_error_sustain_seconds
    # An active finding is the streak this batch continues: a clean batch
    # would have cleared it. A finding booked before the window existed asks
    # for RUN_DIAGNOSTICS already and so stays escalated.
    streak = previous.finding if previous is not None else None
    started_at = streak.observed_at if streak is not None else batch.observed_at
    lasted = (batch.observed_at - started_at).total_seconds()
    sustained = (
        streak is not None and collection_error_actionable(streak)
    ) or lasted >= sustain_seconds
    errors = "; ".join(batch.collection_errors)
    finding = GpuHealthFinding(
        finding_id=(
            streak.finding_id if streak is not None else f"{batch.batch_id}-dcgm-fields"
        ),
        cluster_id=batch.cluster_id,
        node_id=batch.node_id,
        observed_at=started_at,
        severity=GpuHealthSeverity.WARNING,
        reason=(
            f"{errors}; failing for {lasted:.0f}s, "
            + (
                f"past the {sustain_seconds}s sustain window"
                if sustained
                else f"no action before the {sustain_seconds}s sustain window"
            )
        ),
        canonical_name="dcgm_field_completeness",
        value=float(len(batch.collection_errors)),
        evidence_ref=batch.evidence_ref,
        automatic_action=(
            RecoveryAction.RUN_DIAGNOSTICS if sustained else RecoveryAction.NO_ACTION
        ).value,
        policy_source="SITE_DCGM_FIELD_COMPLETENESS",
        policy_version=SITE_METRIC_POLICY_VERSION,
        runtime_profile_version=batch.runtime_profile_version,
        workload_state=batch.workload_state,
        affected_workload_ids=batch.affected_workload_ids,
    )
    return key, finding, sustained

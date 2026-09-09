"""The node-level finding a collector's failed scrape becomes.

A collector reports a scrape it could not perform as a sample-less batch with
``collection_errors`` (F5). The ingest books that as one WARNING
``dcgm_field_completeness`` finding per node, keyed here, and clears it on the
node's next clean batch.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from gpu_fault.gpu_metric_models import (
    SITE_METRIC_POLICY_VERSION,
    GpuHealthFinding,
    GpuHealthSeverity,
)
from gpu_fault.models import RecoveryAction

if TYPE_CHECKING:
    from gpu_fault.gpu_metrics import GpuMetricBatch


def collection_error_key(batch: GpuMetricBatch) -> tuple[str, str, str, str]:
    return (
        batch.cluster_id,
        batch.node_id,
        "node",
        "dcgm_field_completeness",
    )


def collection_error_finding(
    batch: GpuMetricBatch,
) -> tuple[tuple[str, str, str, str], GpuHealthFinding] | None:
    if not batch.collection_errors:
        return None
    key = collection_error_key(batch)
    finding = GpuHealthFinding(
        finding_id=f"{batch.batch_id}-dcgm-fields",
        cluster_id=batch.cluster_id,
        node_id=batch.node_id,
        observed_at=batch.observed_at,
        severity=GpuHealthSeverity.WARNING,
        reason="; ".join(batch.collection_errors),
        canonical_name="dcgm_field_completeness",
        value=float(len(batch.collection_errors)),
        evidence_ref=batch.evidence_ref,
        automatic_action=RecoveryAction.RUN_DIAGNOSTICS.value,
        policy_source="SITE_DCGM_FIELD_COMPLETENESS",
        policy_version=SITE_METRIC_POLICY_VERSION,
        runtime_profile_version=batch.runtime_profile_version,
        workload_state=batch.workload_state,
        affected_workload_ids=batch.affected_workload_ids,
    )
    return key, finding

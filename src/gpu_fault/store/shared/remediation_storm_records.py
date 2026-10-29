"""The remediation storm breaker in the stores (design §9.5).

The breaker of a cluster is one ``remediation_storm_breaker`` record keyed by
its cluster id (``gpu_fault.remediation_storm``). A claim reads and writes it
inside its own transaction -- after the budget scopes admitted the claim, so
a refused claim never counts -- and an operator's release or abort replaces
it by compare-and-set under the same ``remediation_storm/<cluster>`` lock
key, so the two cannot interleave on any backend.
"""

from __future__ import annotations

from datetime import datetime
from typing import cast

from gpu_fault.models import (
    WorkflowEventCode,
    WorkflowEventKind,
    WorkflowRequest,
    record_workflow_event,
)
from gpu_fault.remediation_storm import (
    STORM_RECORD_KIND,
    RemediationStormBreaker,
    StormCandidate,
    storm_admission,
    storm_scope,
)
from gpu_fault.store.shared.errors import (
    RemediationBudgetError,
    RemediationStormHold,
    StaleWriteError,
)
from gpu_fault.store.shared.primitives import (
    GetOptionalRecord,
    ListRecords,
    PutRecord,
    StateTransaction,
)
from gpu_fault.store.shared.remediation_budgets import blocked_by_remediation_budget


def storm_lock_key(cluster_id: str) -> str:
    """The lock a claim's admission and an operator write both take."""

    return f"remediation_storm/{cluster_id}"


def judge_storm(
    record: RemediationStormBreaker | None,
    candidate: StormCandidate,
    *,
    now: datetime,
) -> tuple[RemediationStormBreaker, RemediationStormHold | None]:
    """The record to write for this first admission, and the hold to raise
    after writing it (``None``: admitted)."""

    judged, refusal = storm_admission(
        record, candidate, now=now, cluster_nodes=candidate.cluster_nodes
    )
    if refusal is None:
        return judged, None
    tripped = record is None or not record.open
    return judged, RemediationStormHold(
        refusal,
        scope=storm_scope(candidate.cluster_id),
        record=judged,
        tripped=tripped,
    )


def refused_claim(
    workflow: WorkflowRequest, exc: RemediationBudgetError, *, now: datetime
) -> WorkflowRequest:
    """``blocked_by_remediation_budget`` for any refusal; the claim that trips
    a breaker also carries the trip on its record (one HOLD event)."""

    blocked = blocked_by_remediation_budget(
        workflow, str(exc), scope=exc.scope, now=now
    )
    if not isinstance(exc, RemediationStormHold) or not exc.tripped:
        return blocked
    record = cast(RemediationStormBreaker, exc.record)
    code = WorkflowEventCode.REMEDIATION_STORM_TRIPPED.value
    return record_workflow_event(
        blocked,
        WorkflowEventKind.HOLD,
        code=code,
        reason=str(exc),
        actor="remediation-storm-breaker",
        status=workflow.status.value,
        details={
            "reason": code,
            "remediation_workflow_id": None,
            "cluster_id": record.cluster_id,
            "trip_id": record.trip_id,
            "node_ids": list(record.trip_node_ids),
            "domains": record.trip_domains,
            "threshold": record.trip_threshold,
            "cluster_nodes": record.trip_cluster_nodes,
        },
        at=now,
    )


class SharedRemediationStormMixin:
    # Attributes supplied by the composed concrete implementation.
    _get_optional: GetOptionalRecord
    _list: ListRecords
    _put: PutRecord
    _state_transaction: StateTransaction

    def get_remediation_storm_breaker(
        self, cluster_id: str
    ) -> RemediationStormBreaker | None:
        return cast(
            RemediationStormBreaker | None,
            self._get_optional(STORM_RECORD_KIND, cluster_id),
        )

    def list_remediation_storm_breakers(self) -> list[RemediationStormBreaker]:
        records = cast(list[RemediationStormBreaker], self._list(STORM_RECORD_KIND))
        return sorted(records, key=lambda record: record.cluster_id)

    def _judge_storm(self, candidate: StormCandidate, now: datetime) -> None:
        """``judge_storm`` inside the claim's transaction (whose caller holds
        the ``storm_lock_key`` lock): write the record, then raise the hold."""

        record, held = judge_storm(
            self.get_remediation_storm_breaker(candidate.cluster_id),
            candidate,
            now=now,
        )
        self._put(STORM_RECORD_KIND, candidate.cluster_id, record)
        if held is not None:
            raise held

    def replace_remediation_storm_breaker(
        self,
        record: RemediationStormBreaker,
        *,
        expected: RemediationStormBreaker | None,
    ) -> RemediationStormBreaker:
        """Write ``record`` if the stored one still equals ``expected``."""

        with self._state_transaction(storm_lock_key(record.cluster_id)):
            current = cast(
                RemediationStormBreaker | None,
                self._get_optional(STORM_RECORD_KIND, record.cluster_id),
            )
            if current != expected:
                raise StaleWriteError(
                    f"remediation storm breaker of {record.cluster_id} changed"
                )
            self._put(STORM_RECORD_KIND, record.cluster_id, record)
            return record

"""The remediation storm breakers in the in-memory store (design §9.5)."""

from __future__ import annotations

from threading import RLock

from datetime import datetime

from gpu_fault.remediation_storm import RemediationStormBreaker, StormCandidate
from gpu_fault.store.shared.errors import StaleWriteError
from gpu_fault.store.shared.remediation_storm_records import judge_storm


def admit_storm_in(
    breakers: dict[str, RemediationStormBreaker],
    candidate: StormCandidate,
    now: datetime,
) -> None:
    """``judge_storm`` on the memory store's dict, under the caller's lock:
    the record is written, then the hold (if any) raised."""

    record, held = judge_storm(breakers.get(candidate.cluster_id), candidate, now=now)
    breakers[candidate.cluster_id] = record
    if held is not None:
        raise held


class MemoryRemediationStormMixin:
    # Attributes supplied by the composed concrete implementation.
    _lock: RLock
    _remediation_storm_breakers: dict[str, RemediationStormBreaker]

    def get_remediation_storm_breaker(
        self, cluster_id: str
    ) -> RemediationStormBreaker | None:
        with self._lock:
            return self._remediation_storm_breakers.get(cluster_id)

    def list_remediation_storm_breakers(self) -> list[RemediationStormBreaker]:
        with self._lock:
            records = list(self._remediation_storm_breakers.values())
        return sorted(records, key=lambda record: record.cluster_id)

    def replace_remediation_storm_breaker(
        self,
        record: RemediationStormBreaker,
        *,
        expected: RemediationStormBreaker | None,
    ) -> RemediationStormBreaker:
        """Write ``record`` if the stored one still equals ``expected``."""

        with self._lock:
            if self._remediation_storm_breakers.get(record.cluster_id) != expected:
                raise StaleWriteError(
                    f"remediation storm breaker of {record.cluster_id} changed"
                )
            self._remediation_storm_breakers[record.cluster_id] = record
            return record

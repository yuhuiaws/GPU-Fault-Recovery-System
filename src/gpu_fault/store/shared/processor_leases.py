"""Processor leadership and periodic-task leases shared by the key/value
stores: one row each, renewed or taken over inside one ``_state_transaction``."""

from __future__ import annotations

from datetime import datetime, timedelta

from gpu_fault.processor import (
    PeriodicTaskLease,
    ProcessorLeadership,
)
from gpu_fault.store.shared.primitives import (
    GetOptionalRecord,
    PutRecord,
    StateTransaction,
)


class SharedProcessorLeaseMixin:
    # Attributes supplied by the composed concrete implementation.
    _get_optional: GetOptionalRecord
    _put: PutRecord
    _state_transaction: StateTransaction

    def acquire_processor_leadership(
        self,
        owner_id: str,
        *,
        now: datetime,
        lease_duration: timedelta,
    ):
        with self._state_transaction("processor_leadership/regional"):
            current = self._get_optional("processor_leadership", "regional")
            if (
                current is not None
                and current.owner_id != owner_id
                and current.lease_expires_at > now
            ):
                return current
            epoch = (
                current.epoch
                if current is not None and current.owner_id == owner_id
                else (current.epoch + 1 if current is not None else 1)
            )
            leadership = ProcessorLeadership(
                owner_id=owner_id,
                epoch=epoch,
                lease_expires_at=now + lease_duration,
                updated_at=now,
            )
            self._put(
                "processor_leadership",
                "regional",
                leadership,
            )
            return leadership

    def get_processor_leadership(self):
        return self._get_optional("processor_leadership", "regional")

    def acquire_periodic_task_lease(
        self,
        task_key: str,
        owner_id: str,
        *,
        now: datetime,
        lease_duration: timedelta,
    ):
        with self._state_transaction(f"periodic_task_lease/{task_key}"):
            current = self._get_optional("periodic_task_lease", task_key)
            if (
                current is not None
                and current.owner_id != owner_id
                and current.lease_expires_at > now
            ):
                return current
            epoch = (
                current.epoch
                if current is not None
                and current.owner_id == owner_id
                and current.lease_expires_at > now
                else (current.epoch + 1 if current is not None else 1)
            )
            lease = PeriodicTaskLease(
                task_key=task_key,
                owner_id=owner_id,
                epoch=epoch,
                lease_expires_at=now + lease_duration,
                updated_at=now,
            )
            self._put("periodic_task_lease", task_key, lease)
            return lease

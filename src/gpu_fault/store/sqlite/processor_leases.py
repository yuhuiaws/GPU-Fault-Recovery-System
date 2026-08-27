from __future__ import annotations

from typing import Any, Callable

from datetime import datetime, timedelta, timezone

from gpu_fault.processor import (
    PeriodicTaskLease,
    ProcessorLeadership,
    ProcessorRequestStatus,
)


class SqliteProcessorLeaseMixin:
    # Attributes supplied by the composed concrete implementation.
    get_processor_request: Callable[..., Any]

    _delete: Callable[..., Any]
    _get_optional: Callable[..., Any]
    _list: Callable[..., Any]
    _put: Callable[..., Any]
    _state_transaction: Callable[..., Any]

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

    def validate_processor_lane(
        self,
        ordering_key: str,
        owner_id: str,
        epoch: int,
        lease_token: str,
    ) -> bool:
        lane = self._get_optional("processor_lane", ordering_key)
        return bool(
            lane is not None
            and lane.owner_id == owner_id
            and lane.epoch == epoch
            and lane.lease_token == lease_token
            and lane.lease_expires_at > datetime.now(timezone.utc)
        )

    def renew_active_processor_request(
        self,
        request_id: str,
        owner_id: str,
        lane_epoch: int,
        lease_token: str,
        *,
        lease_duration: timedelta,
    ) -> bool:
        with self._state_transaction(f"processor_request/{request_id}"):
            current = self.get_processor_request(request_id)
            key = current.ordering_key()
            lane = self._get_optional("processor_lane", key)
            now = datetime.now(timezone.utc)
            if (
                lane is None
                or lane.owner_id != owner_id
                or lane.epoch != lane_epoch
                or lane.lease_token != lease_token
                or lane.lease_expires_at <= now
                or current.lease_owner != owner_id
                or current.leader_epoch != lane_epoch
                or current.lease_token != lease_token
            ):
                return False
            expires_at = now + lease_duration
            self._put(
                "processor_lane",
                key,
                lane.model_copy(
                    update={
                        "lease_expires_at": expires_at,
                        "updated_at": now,
                    }
                ),
            )
            self._put(
                "processor_request",
                request_id,
                current.model_copy(
                    update={
                        "lease_expires_at": expires_at,
                        "updated_at": now,
                    }
                ),
            )
            return True

    def release_active_processor_request(
        self,
        request_id: str,
        owner_id: str,
        lane_epoch: int,
        lease_token: str,
        *,
        not_before: datetime | None = None,
        retry_count: int | None = None,
    ) -> None:
        with self._state_transaction(f"processor_request/{request_id}"):
            current = self.get_processor_request(request_id)
            key = current.ordering_key()
            lane = self._get_optional("processor_lane", key)
            if (
                lane is None
                or lane.owner_id != owner_id
                or lane.epoch != lane_epoch
                or lane.lease_token != lease_token
                or current.lease_owner != owner_id
                or current.leader_epoch != lane_epoch
                or current.lease_token != lease_token
            ):
                return
            # See the Postgres store: completion does not clear the
            # fencing fields, so this is the only thing keeping a
            # COMPLETED request from being reopened as PENDING.
            if current.status != ProcessorRequestStatus.LEASED:
                return
            now = datetime.now(timezone.utc)
            self._put(
                "processor_lane",
                key,
                lane.model_copy(
                    update={
                        "lease_expires_at": now,
                        "updated_at": now,
                    }
                ),
            )
            self._put(
                "processor_request",
                request_id,
                current.model_copy(
                    update={
                        "status": ProcessorRequestStatus.PENDING,
                        "lease_owner": None,
                        "leader_epoch": None,
                        "lease_token": None,
                        "lease_expires_at": None,
                        "not_before": not_before,
                        "retry_count": (
                            current.retry_count if retry_count is None else retry_count
                        ),
                        "updated_at": now,
                    }
                ),
            )

    def complete_active_processor_request(
        self,
        request_id: str,
        owner_id: str,
        lane_epoch: int,
        lease_token: str,
        *,
        response_status: int,
        response_content_type: str | None,
        response_body_base64: str,
        path: str | None = None,
    ):
        # See InMemoryStore: accepted for call compatibility with the
        # Postgres store, which routes on it.
        _ = path
        with self._state_transaction(f"processor_request/{request_id}"):
            current = self.get_processor_request(request_id)
            key = current.ordering_key()
            lane = self._get_optional("processor_lane", key)
            now = datetime.now(timezone.utc)
            if (
                lane is None
                or lane.owner_id != owner_id
                or lane.epoch != lane_epoch
                or lane.lease_token != lease_token
                or lane.lease_expires_at <= now
                or current.lease_owner != owner_id
                or current.leader_epoch != lane_epoch
                or current.lease_token != lease_token
            ):
                raise ValueError("stale processor lane fencing token")
            completed = current.model_copy(
                update={
                    "status": ProcessorRequestStatus.COMPLETED,
                    "response_status": response_status,
                    "response_content_type": response_content_type,
                    "response_body_base64": response_body_base64,
                    "lease_expires_at": None,
                    "updated_at": now,
                }
            )
            self._put("processor_request", request_id, completed)
            self._put(
                "processor_lane",
                key,
                lane.model_copy(
                    update={
                        "lease_expires_at": now,
                        "updated_at": now,
                    }
                ),
            )
            return completed

    def release_processor_request(
        self,
        request_id: str,
        owner_id: str,
        leader_epoch: int,
        lease_token: str,
        *,
        not_before: datetime | None = None,
        retry_count: int | None = None,
    ) -> None:
        with self._state_transaction(f"processor_request/{request_id}"):
            current = self.get_processor_request(request_id)
            if (
                current.lease_owner != owner_id
                or current.leader_epoch != leader_epoch
                or current.lease_token != lease_token
            ):
                return
            self._put(
                "processor_request",
                request_id,
                current.model_copy(
                    update={
                        "status": ProcessorRequestStatus.PENDING,
                        "lease_owner": None,
                        "leader_epoch": None,
                        "lease_token": None,
                        "lease_expires_at": None,
                        "not_before": not_before,
                        "retry_count": (
                            current.retry_count if retry_count is None else retry_count
                        ),
                        "updated_at": datetime.now(timezone.utc),
                    }
                ),
            )

    def complete_processor_request(
        self,
        request_id: str,
        owner_id: str,
        leader_epoch: int,
        lease_token: str,
        *,
        response_status: int,
        response_content_type: str | None,
        response_body_base64: str,
    ):
        with self._state_transaction(f"processor_request/{request_id}"):
            leadership = self.get_processor_leadership()
            current = self.get_processor_request(request_id)
            now = datetime.now(timezone.utc)
            if (
                leadership is None
                or leadership.owner_id != owner_id
                or leadership.epoch != leader_epoch
                or leadership.lease_expires_at <= now
                or current.lease_owner != owner_id
                or current.leader_epoch != leader_epoch
                or current.lease_token != lease_token
                or current.lease_expires_at is None
                or current.lease_expires_at <= now
            ):
                raise ValueError("stale processor fencing token")
            completed = current.model_copy(
                update={
                    "status": ProcessorRequestStatus.COMPLETED,
                    "response_status": response_status,
                    "response_content_type": response_content_type,
                    "response_body_base64": response_body_base64,
                    "lease_expires_at": None,
                    "updated_at": now,
                }
            )
            self._put("processor_request", request_id, completed)
            return completed

    def cleanup_processor_lanes(
        self,
        *,
        older_than: datetime,
        limit: int,
    ) -> int:
        with self._state_transaction("processor_lane/cleanup"):
            busy = {
                item.ordering_key()
                for item in self._list("processor_request")
                if item.status
                in {
                    ProcessorRequestStatus.PENDING,
                    ProcessorRequestStatus.LEASED,
                }
            }
            victims = [
                lane.ordering_key
                for lane in sorted(
                    self._list("processor_lane"),
                    key=lambda item: (
                        item.lease_expires_at,
                        item.ordering_key,
                    ),
                )
                if lane.lease_expires_at <= older_than and lane.ordering_key not in busy
            ][:limit]
            for key in victims:
                self._delete("processor_lane", key)
            return len(victims)

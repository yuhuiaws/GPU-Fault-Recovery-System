from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Callable

from gpu_fault.processor import (
    ProcessorRequestStatus,
)
from gpu_fault.store.shared.cleanup_log import log_cleanup
from gpu_fault.store.shared.errors import StaleFencingTokenError


class SqliteProcessorLeaseMixin:
    # Attributes supplied by the composed concrete implementation.
    get_processor_leadership: Callable[..., Any]
    get_processor_request: Callable[..., Any]

    _delete: Callable[..., Any]
    _get_optional: Callable[..., Any]
    _list: Callable[..., Any]
    _put: Callable[..., Any]
    _state_transaction: Callable[..., Any]

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
                raise StaleFencingTokenError("stale processor lane fencing token")
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
            # A COMPLETED row keeps its fencing fields; only LEASED goes back.
            if current.status != ProcessorRequestStatus.LEASED:
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
                raise StaleFencingTokenError("stale processor fencing token")
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

    def reclaim_expired_processor_leases(
        self,
        *,
        now: datetime,
        limit: int,
    ) -> int:
        """Hand LEASED rows whose lease lapsed back to PENDING (F-D5)."""

        with self._state_transaction("processor_request/reclaim-expired"):
            expired = sorted(
                (
                    item
                    for item in self._list("processor_request")
                    if item.status is ProcessorRequestStatus.LEASED
                    and item.lease_expires_at is not None
                    and item.lease_expires_at <= now
                ),
                key=lambda item: (item.lease_expires_at, item.request_id),
            )[:limit]
            for item in expired:
                self._put(
                    "processor_request",
                    item.request_id,
                    item.model_copy(
                        update={
                            "status": ProcessorRequestStatus.PENDING,
                            "lease_owner": None,
                            "leader_epoch": None,
                            "lease_token": None,
                            "lease_expires_at": None,
                            "not_before": None,
                            "retry_count": item.retry_count + 1,
                            "updated_at": max(now, item.updated_at),
                        }
                    ),
                )
            return len(expired)

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
            return log_cleanup("processor_lane", victims)

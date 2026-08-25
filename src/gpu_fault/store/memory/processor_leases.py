from __future__ import annotations

from typing import Any, Callable

from contextlib import contextmanager
from datetime import datetime, timedelta, timezone

from gpu_fault.processor import (
    PeriodicTaskLease,
    ProcessorLeadership,
    ProcessorRequestStatus,
)


class MemoryProcessorLeaseMixin:
    # Attributes supplied by the composed concrete implementation.
    _periodic_task_leases: Any
    _processor_lanes: Any
    _processor_requests: Any
    get_processor_request: Callable[..., Any]

    _lock: Any

    def complete_active_processor_requests_batch(self, completions):
        return [
            self.complete_active_processor_request(**completion)
            for completion in completions
        ]

    @contextmanager
    def processor_batch_transaction(self):
        yield

    def acquire_processor_leadership(
        self,
        owner_id: str,
        *,
        now: datetime,
        lease_duration: timedelta,
    ):
        with self._lock:
            current = self._processor_leadership
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
            self._processor_leadership = leadership
            return leadership

    def get_processor_leadership(self):
        with self._lock:
            return self._processor_leadership

    def acquire_periodic_task_lease(
        self,
        task_key: str,
        owner_id: str,
        *,
        now: datetime,
        lease_duration: timedelta,
    ):
        with self._lock:
            current = self._periodic_task_leases.get(task_key)
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
            self._periodic_task_leases[task_key] = lease
            return lease

    def validate_processor_lane(
        self,
        ordering_key: str,
        owner_id: str,
        epoch: int,
        lease_token: str,
    ) -> bool:
        with self._lock:
            lane = self._processor_lanes.get(ordering_key)
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
        with self._lock:
            current = self.get_processor_request(request_id)
            lane_key = current.ordering_key()
            lane = self._processor_lanes.get(lane_key)
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
            self._processor_lanes[lane_key] = lane.model_copy(
                update={
                    "lease_expires_at": expires_at,
                    "updated_at": now,
                }
            )
            self._processor_requests[request_id] = current.model_copy(
                update={
                    "lease_expires_at": expires_at,
                    "updated_at": now,
                }
            )
            return True

    def release_active_processor_request(
        self,
        request_id: str,
        owner_id: str,
        lane_epoch: int,
        lease_token: str,
    ) -> None:
        with self._lock:
            current = self.get_processor_request(request_id)
            lane_key = current.ordering_key()
            lane = self._processor_lanes.get(lane_key)
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
            # Completing leaves the fencing fields in place, so a
            # COMPLETED request passes the check above; releasing it
            # would reopen it as PENDING and run its side effects a
            # second time. Same guard as the Postgres store, so the
            # three backends answer the same for the same input.
            if current.status != ProcessorRequestStatus.LEASED:
                return
            now = datetime.now(timezone.utc)
            self._processor_lanes[lane_key] = lane.model_copy(
                update={
                    "lease_expires_at": now,
                    "updated_at": now,
                }
            )
            self._processor_requests[request_id] = current.model_copy(
                update={
                    "status": ProcessorRequestStatus.PENDING,
                    "lease_owner": None,
                    "leader_epoch": None,
                    "lease_token": None,
                    "lease_expires_at": None,
                    "updated_at": now,
                }
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
        # ``path`` is what the Postgres store uses to route observation
        # completions into its group commit; here it is accepted and
        # ignored so both stores take the same call.
        _ = path
        with self._lock:
            current = self.get_processor_request(request_id)
            lane_key = current.ordering_key()
            lane = self._processor_lanes.get(lane_key)
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
            self._processor_requests[request_id] = completed
            self._processor_lanes[lane_key] = lane.model_copy(
                update={
                    "lease_expires_at": now,
                    "updated_at": now,
                }
            )
            return completed

    def release_processor_request(
        self,
        request_id: str,
        owner_id: str,
        leader_epoch: int,
        lease_token: str,
    ) -> None:
        with self._lock:
            current = self.get_processor_request(request_id)
            if (
                current.lease_owner != owner_id
                or current.leader_epoch != leader_epoch
                or current.lease_token != lease_token
            ):
                return
            self._processor_requests[request_id] = current.model_copy(
                update={
                    "status": ProcessorRequestStatus.PENDING,
                    "lease_owner": None,
                    "leader_epoch": None,
                    "lease_token": None,
                    "lease_expires_at": None,
                    "updated_at": datetime.now(timezone.utc),
                }
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
        with self._lock:
            current = self.get_processor_request(request_id)
            leadership = self._processor_leadership
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
            self._processor_requests[request_id] = completed
            return completed

    def cleanup_processor_lanes(
        self,
        *,
        older_than: datetime,
        limit: int,
    ) -> int:
        """Drop lane rows whose lease expired and that hold no work.

        Lane keys are unbounded in cardinality (one per attempt, incident,
        workflow and recovery plan), and the claim path joins the lane
        table, so retiring idle lanes keeps that join bounded. Deleting a
        lane resets its epoch to 1 on the next claim, which is safe because
        every fencing predicate also matches ``lease_token``.
        """

        with self._lock:
            busy = {
                item.ordering_key()
                for item in self._processor_requests.values()
                if item.status
                in {
                    ProcessorRequestStatus.PENDING,
                    ProcessorRequestStatus.LEASED,
                }
            }
            victims = [
                key
                for key, lane in sorted(
                    self._processor_lanes.items(),
                    key=lambda entry: (
                        entry[1].lease_expires_at,
                        entry[0],
                    ),
                )
                if lane.lease_expires_at <= older_than and key not in busy
            ][:limit]
            for key in victims:
                del self._processor_lanes[key]
            return len(victims)

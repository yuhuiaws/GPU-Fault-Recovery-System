from __future__ import annotations

import base64
import hashlib
import json
import logging
from collections.abc import Iterable
from datetime import datetime, timezone
from enum import StrEnum
from uuid import uuid4

from pydantic import Field, PrivateAttr
from gpu_fault.channel_registry import (
    COLLECTOR_HEALTH_PATH,
    COLLECTOR_EVENT_PREFIX,
    WORKLOAD_OBSERVATIONS_PATH,
    ChannelLane,
    channel_for_path,
    is_fault_path,
)
from gpu_fault.models import StrictModel


LOGGER = logging.getLogger(__name__)


class ProcessorRequestStatus(StrEnum):
    PENDING = "PENDING"
    LEASED = "LEASED"
    COMPLETED = "COMPLETED"


class ProcessorLanePolicy(StrEnum):
    STRICT = "STRICT"
    REORDERABLE = "REORDERABLE"


class ProcessorLeadership(StrictModel):
    owner_id: str
    epoch: int = Field(ge=1)
    lease_expires_at: datetime
    updated_at: datetime


class PeriodicTaskLease(StrictModel):
    task_key: str
    owner_id: str
    epoch: int = Field(ge=1)
    lease_expires_at: datetime
    updated_at: datetime


class ProcessorLaneLease(StrictModel):
    ordering_key: str
    owner_id: str
    epoch: int = Field(ge=1)
    lease_token: str
    lease_expires_at: datetime
    updated_at: datetime


def processor_partition_id(key_value: str | None, partition_count: int) -> int:
    if partition_count <= 0:
        raise ValueError("processor partition count must be positive")
    key = (key_value or "__unscoped__").encode("utf-8")
    value = int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "big")
    return value % partition_count


class ProcessorRequest(StrictModel):
    request_id: str = Field(default_factory=lambda: f"processor-{uuid4()}")
    method: str
    path: str
    query: str = ""
    body_base64: str = ""
    content_type: str | None = None
    cluster_id: str | None = None
    correlation_key: str | None = None
    correlation_scope_keys: list[str] = Field(default_factory=list)
    execution_authorized: bool = False
    status: ProcessorRequestStatus = ProcessorRequestStatus.PENDING
    lease_owner: str | None = None
    leader_epoch: int | None = None
    lease_token: str | None = None
    lease_expires_at: datetime | None = None
    response_status: int | None = None
    response_content_type: str | None = None
    response_body_base64: str | None = None
    not_before: datetime | None = None
    retry_count: int = Field(default=0, ge=0)
    lane_policy: ProcessorLanePolicy = ProcessorLanePolicy.STRICT
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    _ordering_key_cache: tuple[str, str, str, str] | None = PrivateAttr(default=None)
    # Same witness discipline as the ordering key: the tier is read on
    # every admission, pool routing decision and coalescing check, and
    # deciding it parses the body.
    _queue_priority_cache: tuple[str, str, int] | None = PrivateAttr(default=None)

    @classmethod
    def from_http(
        cls,
        *,
        method: str,
        path: str,
        query: str,
        body: bytes,
        content_type: str | None,
        cluster_id: str | None,
        execution_authorized: bool = False,
        parsed_payload: dict | None = None,
    ) -> ProcessorRequest:
        correlation_key = None
        correlation_scope_keys: list[str] = []
        if parsed_payload is not None:
            payload = parsed_payload
        else:
            try:
                payload = json.loads(body) if body else {}
            except (UnicodeDecodeError, json.JSONDecodeError):
                payload = {}
        if not isinstance(payload, dict):
            payload = {}
        channel = channel_for_path(path)
        if (
            channel is not None
            and (channel.lane is ChannelLane.ATTEMPT or channel.correlated_fault)
        ) or path in {
            "/v1/attempts/failure-detected",
            "/v1/attempts/terminal",
            "/v1/gpu-events/xid",
            "/v1/gpu-events/sxid",
        }:
            job_id = payload.get("job_id")
            attempt_id = payload.get("attempt_id")
            if (
                cluster_id
                and isinstance(job_id, str)
                and job_id
                and isinstance(attempt_id, str)
                and attempt_id
            ):
                correlation_key = json.dumps(
                    [cluster_id, job_id, attempt_id],
                    separators=(",", ":"),
                )
                correlation_scope_keys.append(correlation_key)
            node_ids = set()
            node_id = payload.get("node_id")
            if isinstance(node_id, str) and node_id:
                node_ids.add(node_id)
            if path == WORKLOAD_OBSERVATIONS_PATH:
                for container in payload.get("containers") or []:
                    if not isinstance(container, dict):
                        continue
                    container_node = container.get("node_id")
                    if isinstance(container_node, str) and container_node:
                        node_ids.add(container_node)
            for scoped_node_id in sorted(node_ids):
                correlation_scope_keys.append(
                    json.dumps(
                        [cluster_id, "node", scoped_node_id],
                        separators=(",", ":"),
                    )
                )
        lane_policy = ProcessorLanePolicy.STRICT
        if (
            channel is not None
            and channel.latest_wins
            and channel.priority(payload) == 100
        ):
            lane_policy = ProcessorLanePolicy.REORDERABLE
        return cls(
            method=method,
            path=path,
            query=query,
            body_base64=base64.b64encode(body).decode("ascii"),
            content_type=content_type,
            cluster_id=cluster_id,
            correlation_key=correlation_key,
            correlation_scope_keys=correlation_scope_keys,
            execution_authorized=execution_authorized,
            lane_policy=lane_policy,
        )

    def body(self) -> bytes:
        return base64.b64decode(self.body_base64)

    def response_body(self) -> bytes:
        if self.response_body_base64 is None:
            return b""
        return base64.b64decode(self.response_body_base64)

    def available_for_claim(self, now: datetime) -> bool:
        return self.not_before is None or self.not_before <= now

    def is_strict_retry_barrier(self, now: datetime) -> bool:
        return (
            self.status is ProcessorRequestStatus.PENDING
            and self.lane_policy is ProcessorLanePolicy.STRICT
            and not self.available_for_claim(now)
        )

    def ordering_key(self) -> str:
        # ``model_copy`` carries private attributes over, so the cache is
        # validated against every input the lane derivation reads instead
        # of being trusted unconditionally.
        witness = (
            self.body_base64,
            self.path,
            self.cluster_id or "",
        )
        cached = self._ordering_key_cache
        if cached is not None and cached[:3] == witness:
            return cached[3]
        key = self._compute_ordering_key()
        self._ordering_key_cache = (*witness, key)
        return key

    def coalescable(self) -> bool:
        """Whether a pending request on the same lane may be overwritten.

        Only routine requests whose registered channel declares
        ``latest_wins`` qualify. Five channels currently opt in; the
        registry flag, rather than a lane suffix or the priority tier, is
        authoritative. Shared node lanes cannot safely imply coalescing,
        and non-routine edge-filtered samples must remain distinct because
        detector confirmation counters need every intermediate sample.
        """
        if self.queue_priority() != 100:
            return False
        channel = channel_for_path(self.path)
        return bool(channel and channel.latest_wins)

    def spool_key(self) -> str:
        """Primary key of this sample in the telemetry spool.

        The spool has no lane table and no coalescing query: its primary
        key does both jobs. A channel that supersedes its own samples keys
        on the lane alone, so an ``INSERT ... ON CONFLICT`` overwrites the
        pending payload the way ``try_enqueue_processor_request`` does with
        a ``SELECT ... FOR UPDATE`` and a second write. A channel that does
        not - edge-filtered batches, whose consecutive-breach counters need
        every sample - carries the request id, so each one gets its own
        row and nothing is dropped.

        This is not an ordering key. Rows on the same lane may be replayed
        concurrently, which is safe because every telemetry write compares
        ``observed_at`` under a row lock and keeps the newer value; the
        lane the queue gives these paths only serialises writes that
        already exclude each other.
        """

        key = self.ordering_key()
        if self.coalescable():
            return key
        return f"{key}:{self.request_id}"

    def _json_payload(self) -> dict | None:
        """The body as an object, or ``None`` when it is not one.

        Callers that only read fields can treat both the same; the tier
        cannot, because "no reasons in the body" is a routine batch while
        "no readable body" is something nobody has vouched for.
        """

        try:
            payload = json.loads(self.body())
        except (UnicodeDecodeError, json.JSONDecodeError):
            return None
        return payload if isinstance(payload, dict) else None

    def _compute_ordering_key(self) -> str:
        cluster = self.cluster_id or "__unscoped__"
        payload = self._json_payload() or {}
        attempt_id = payload.get("attempt_id")
        if (
            (
                (
                    (channel := channel_for_path(self.path)) is not None
                    and channel.lane is ChannelLane.ATTEMPT
                )
                or self.path in {"/v1/gpu-events/xid", "/v1/gpu-events/sxid"}
                or self.path.startswith("/v1/attempts/")
            )
            and isinstance(attempt_id, str)
            and attempt_id
        ):
            return f"{cluster}:attempt:{attempt_id}"
        for prefix, scope_name, reserved in (
            ("/v1/incidents/", "incident", set()),
            (
                "/v1/workflows/",
                "workflow",
                {"dispatch"},
            ),
            (
                "/v1/recovery-plans/",
                "recovery-plan",
                set(),
            ),
        ):
            if self.path.startswith(prefix):
                resource_id = self.path.removeprefix(prefix).split("/", 1)[0]
                if resource_id and resource_id not in reserved:
                    return f"{cluster}:{scope_name}:{resource_id}"
        node_id = payload.get("node_id")
        if (
            self.path.startswith(
                (
                    COLLECTOR_EVENT_PREFIX,
                    "/v1/provider-events/",
                    "/v1/gpu-events/",
                )
            )
            and isinstance(node_id, str)
            and node_id
        ):
            channel = channel_for_path(self.path)
            if channel and channel.lane is ChannelLane.GPU_INVENTORY:
                return f"{cluster}:node:{node_id}:gpu-inventory"
            if channel and channel.lane is ChannelLane.EDGE_SUMMARY:
                reasons = payload.get("edge_filter_reasons")
                collection_errors = payload.get("collection_errors")
                if reasons == ["health-summary"] and not collection_errors:
                    suffix = channel.summary_lane_suffix
                    if suffix is None:
                        return f"{cluster}:node:{node_id}"
                    if self.path == COLLECTOR_HEALTH_PATH:
                        collector = str(payload.get("collector") or "unknown").lower()
                        suffix = f"{suffix}-{collector}"
                    return f"{cluster}:node:{node_id}:{suffix}"
            return f"{cluster}:node:{node_id}"
        return cluster

    def is_routine_telemetry(self) -> bool:
        """Whether an edge-filtered batch is reporting that all is well.

        The node-side filter delivers a batch either because a periodic
        summary is due or because something in it looks wrong, and it says
        which in ``edge_filter_reasons``. A batch that carries a threshold
        breach, a sustained window, a collection error, an EFA traffic
        change or a confirmed candidate is evidence: the detectors count
        consecutive breaches, so it may not be coalesced away, and it
        should not queue behind the samples that report nothing.

        Two shapes read as routine even though they carry no summary
        reason, and both are correct: the nvidia-smi fallback batch posts
        the full sample set with no reasons at all, and a collector with
        the edge filter switched off marks everything ``filter-disabled``.
        Neither is a report about a threshold - with the filter off there
        is no signal in the reasons to read.

        Every reason has to be routine for the batch to be: one breach
        alongside a due summary is still a breach. An unknown token is
        read as evidence, which costs a batch its coalescing and nothing
        else -- the safe direction for a vocabulary that lives in a
        collector rolling on its own schedule.
        """

        channel = channel_for_path(self.path)
        if channel is None or not channel.edge_filtered:
            return False
        payload = self._json_payload()
        if payload is None:
            # A body the queue cannot read is not a body promising that
            # all is well: leave it in the middle tier, where it keeps its
            # own row and its completion record until it is rejected.
            return False
        if payload.get("collection_errors"):
            return False
        reasons = payload.get("edge_filter_reasons") or []
        if not isinstance(reasons, list):
            return False
        return channel.is_routine_payload(payload)

    def spoolable(self) -> bool:
        """Whether this request may go to the spool instead of the queue.

        Deliberately not "priority 100": the spool keeps no completion
        record and no lane, which is only acceptable for a channel whose
        next sample restates the same node state. Node logs and rank
        heartbeats are routine but are not that, and a new endpoint has to
        be named here on purpose rather than inherit the spool by being
        given a priority.
        """

        channel = channel_for_path(self.path)
        return bool(channel and channel.spoolable) and (self.queue_priority() == 100)

    def queue_priority(self) -> int:
        """Which of three service tiers this request belongs to.

        ``0`` - a confirmed fault or a control-plane action on one. Never
        dropped, never coalesced, claimed first, and the only tier with
        reserved queue depth and its own ingress threads.

        ``50`` - evidence and decisions read from it: edge-filtered
        batches that report something wrong, workload observations,
        watcher observations and the node-action paths. Not coalescable,
        and ``_stale_disposition`` refuses to retire most of it unsent.

        ``100`` - routine traffic that its own next sample supersedes:
        health summaries, inventory snapshots, rank heartbeats and
        ordinary node logs. Latest-wins where a lane exists, and the tier
        the admission batcher and the telemetry spool work on.

        The tier is stored on the queue row at admission, so a change here
        applies to newly enqueued requests and leaves in-flight rows on
        the tier they were admitted with.
        """

        witness = (self.body_base64, self.path)
        cached = self._queue_priority_cache
        if cached is not None and cached[:2] == witness:
            return cached[2]
        priority = self._compute_queue_priority()
        self._queue_priority_cache = (*witness, priority)
        return priority

    def _compute_queue_priority(self) -> int:
        if is_fault_path(self.path):
            return 0
        channel = channel_for_path(self.path)
        if channel is not None:
            return channel.priority(self._json_payload())
        return 50

    def is_correlated_fault(self) -> bool:
        return bool(self.correlation_scope_keys) and (
            self.path
            in {
                "/v1/gpu-events/xid",
                "/v1/gpu-events/sxid",
                "/v1/attempts/failure-detected",
                "/v1/attempts/terminal",
            }
            or bool(
                (channel := channel_for_path(self.path)) and channel.correlated_fault
            )
        )

    def claim_priority(
        self,
        pending_fault_scope_keys: set[str],
        routine_starvation_before: datetime | None = None,
    ) -> int:
        if (
            self.path == WORKLOAD_OBSERVATIONS_PATH
            and not pending_fault_scope_keys.isdisjoint(self.correlation_scope_keys)
        ):
            return -1
        priority = self.queue_priority()
        if (
            priority == 100
            and routine_starvation_before is not None
            and self.created_at <= routine_starvation_before
        ):
            return 49
        return priority

    def waits_for_observation(self, pending_observation_scope_keys: set[str]) -> bool:
        return (
            self.is_correlated_fault()
            and not pending_observation_scope_keys.isdisjoint(
                self.correlation_scope_keys
            )
        )


def deferred_strict_processor_lanes(
    requests: Iterable[ProcessorRequest],
    now: datetime,
) -> set[str]:
    return {
        request.ordering_key()
        for request in requests
        if request.is_strict_retry_barrier(now)
    }


def processor_request_claimable(
    request: ProcessorRequest,
    *,
    now: datetime,
    deferred_strict_lanes: set[str],
) -> bool:
    if request.status is ProcessorRequestStatus.PENDING:
        if not request.available_for_claim(now):
            return False
    elif not (
        request.status is ProcessorRequestStatus.LEASED
        and (request.lease_expires_at is None or request.lease_expires_at <= now)
    ):
        return False
    return request.ordering_key() not in deferred_strict_lanes

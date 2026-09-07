from __future__ import annotations

import hashlib
import logging
import os
from datetime import datetime, timedelta, timezone
from threading import RLock
from typing import Any

from gpu_fault.env import env_bool
from gpu_fault.models import (
    AdvisoryNotification,
    NotificationDelivery,
    NotificationDispatchReport,
    NotificationResult,
    NotificationStatus,
    WorkflowOperation,
)
from gpu_fault.notifications import (
    EfaRdmaEventEmailBuilder,
    HardwareInventoryEmailBuilder,
    HostResourceEventEmailBuilder,
    HyperPodAdvisoryEmailBuilder,
    NotApplicableEmailBuilder,
    RestartGuardEmailBuilder,
    SxidEventEmailBuilder,
    XidInvestigatoryEmailBuilder,
)
from gpu_fault.notifications.registry import (
    NotificationBuilderRegistry,
    NotificationKind,
)
from gpu_fault.notification_preview import (
    AdvisoryNotApplicableError as AdvisoryNotApplicableError,
    NotificationPreviewCallbacks,
    NotificationPreviewService,
)
from gpu_fault.ports import NotificationPort
from gpu_fault.policy import (
    FaultPolicyDecision,
    SxidEvent,
    XidEvent,
)
from gpu_fault.store import (
    NotFoundError,
    WorkflowLeaseError,
)
from gpu_fault.store.contracts import ControlPlaneStore

LOGGER = logging.getLogger(__name__)


def _onoff(value: bool) -> str:
    return "on" if value else "off"


DEFAULT_NOTIFICATION_TTL_SECONDS = 6 * 3600
CATEGORY_TTL_SECONDS = {
    # A trend sample is restated by the next cooldown bucket, so an
    # undelivered one is worth less than the mail it would cost.
    "HEALTH_TREND": 3600,
}
THROTTLE_ERROR_CODES = frozenset(
    {
        "TooManyRequestsException",
        "ThrottlingException",
        "Throttling",
        "ThrottledException",
        "RequestThrottled",
        "RequestThrottledException",
        "ProvisionedThroughputExceededException",
        "TransactionInProgressException",
        "SlowDown",
        "LimitExceededException",
    }
)


def _is_throttled(exc: BaseException) -> bool:
    """Did the provider refuse because of *our rate*, not this message?

    The distinction decides whether an attempt should be charged. SES
    answers ``TooManyRequestsException`` once the account's send rate is
    exceeded, which says nothing about the notification in hand -- and
    charging it an attempt walks an entire storm's worth of mail to its
    terminal state for the sole reason that it arrived at once.
    """

    codes = {type(exc).__name__}
    response = getattr(exc, "response", None)
    if isinstance(response, dict):
        error = response.get("Error")
        if isinstance(error, dict):
            codes.add(str(error.get("Code", "")))
    return bool(codes & THROTTLE_ERROR_CODES)


def _resolved_dispatch_scan_limit(configured: int | None) -> int:
    """How much notification history one synchronous dispatch may read.

    Nothing ever leaves that table, and nothing ever leaves the set that path
    counts as pending either -- a drill and an expired advisory are both recorded
    SKIPPED, and SKIPPED is picked up again -- so the read grew without bound and
    carried one extra result lookup per row. 2000 is far above any backlog a
    deployment running the dispatcher should hold, and a deployment that
    genuinely exceeds it wants ``GPU_FAULT_NOTIFICATION_ASYNC_DELIVERY``, which
    claims from an indexed outbox instead of scanning.
    """

    limit = (
        int(os.getenv("GPU_FAULT_NOTIFICATION_DISPATCH_SCAN_LIMIT", "2000"))
        if configured is None
        else configured
    )
    if limit < 1:
        raise ValueError("notification dispatch scan limit must be positive")
    return limit


def _dispatch_remote_node_action_completion(
    service: AdvisoryNotificationService,
    command: Any,
    existing_id: str | None,
) -> list[NotificationResult]:
    common = {
        "cluster_id": command.cluster_id,
        "incident_id": command.incident.incident_id,
        "workflow_id": command.workflow.request_id,
        "event_id": command.incident.event_id,
        "event_type": command.incident.event_type,
        "policy_source": command.incident.policy_source,
        "official_action": command.incident.official_action,
        "reasons": command.incident.reasons,
        "operation_id": command.idempotency_key,
        "node_results": command.result_details.get("node_results", {}),
        "workload_ids": command.step.workload_ids,
    }
    if command.step.operation is WorkflowOperation.RESTART_FABRIC_MANAGER:
        kind = NotificationKind.FABRIC_MANAGER_RESTARTED
        values = common
    elif command.step.operation is WorkflowOperation.RESET_GPU:
        kind = NotificationKind.GPU_RESET_COMPLETED
        values = {
            **common,
            "node_ids": command.step.node_ids,
            "gpu_uuids": command.step.gpu_uuids,
        }
    else:
        return []
    notification = service.builders.build(kind, **values)
    notification = service.store.save_notification_if_absent(notification)
    if notification.notification_id == existing_id:
        return []
    return [service.send(notification.notification_id)]


class AdvisoryNotificationService:
    """Creates and delivers customer-administrator notifications."""

    def __init__(
        self,
        store: ControlPlaneStore,
        notifier: NotificationPort,
        builder: HyperPodAdvisoryEmailBuilder | None = None,
        not_applicable_builder: NotApplicableEmailBuilder | None = None,
        xid_investigatory_builder: (XidInvestigatoryEmailBuilder | None) = None,
        sxid_event_builder: SxidEventEmailBuilder | None = None,
        efa_rdma_event_builder: EfaRdmaEventEmailBuilder | None = None,
        hardware_inventory_builder: (HardwareInventoryEmailBuilder | None) = None,
        host_resource_event_builder: (HostResourceEventEmailBuilder | None) = None,
        restart_email_builder: RestartGuardEmailBuilder | None = None,
        async_delivery: bool | None = None,
        deliver_backlog: bool | None = None,
        backlog_grace_seconds: int | None = None,
        ttl_seconds: int | None = None,
        deliver_drills: bool | None = None,
        dispatch_scan_limit: int | None = None,
    ) -> None:
        self.store = store
        self.notifier = notifier
        restart_builder = restart_email_builder or RestartGuardEmailBuilder()
        overrides = {
            NotificationKind.HYPERPOD_ADVISORY: (
                builder or HyperPodAdvisoryEmailBuilder()
            ),
            NotificationKind.NOT_APPLICABLE: (
                not_applicable_builder or NotApplicableEmailBuilder()
            ),
            NotificationKind.XID_INVESTIGATORY: (
                xid_investigatory_builder or XidInvestigatoryEmailBuilder()
            ),
            NotificationKind.SXID_EVENT: (
                sxid_event_builder or SxidEventEmailBuilder()
            ),
            NotificationKind.EFA_RDMA_EVENT: (
                efa_rdma_event_builder or EfaRdmaEventEmailBuilder()
            ),
            NotificationKind.HARDWARE_INVENTORY: (
                hardware_inventory_builder or HardwareInventoryEmailBuilder()
            ),
            NotificationKind.HOST_RESOURCE: (
                host_resource_event_builder or HostResourceEventEmailBuilder()
            ),
        }
        for kind in (
            NotificationKind.GPU_COUNT_CHANGE,
            NotificationKind.BUDGET_EXHAUSTED,
            NotificationKind.WORKLOAD_RESTARTED,
            NotificationKind.NODE_RESTARTED,
            NotificationKind.FABRIC_MANAGER_RESTARTED,
            NotificationKind.FABRIC_RESET_COMPLETED,
            NotificationKind.GPU_RESET_COMPLETED,
        ):
            overrides[kind] = restart_builder
        self.builders = NotificationBuilderRegistry(overrides)
        self.previews = NotificationPreviewService(
            self.store,
            self.builders,
            NotificationPreviewCallbacks(
                incident_lock=self._incident_lock,
                evidence_refs_for_incident=(self._evidence_refs_for_incident),
            ),
        )
        self.async_delivery = (
            env_bool("GPU_FAULT_NOTIFICATION_ASYNC_DELIVERY", False)
            if async_delivery is None
            else async_delivery
        )
        self.dispatcher_enabled = env_bool(
            "GPU_FAULT_NOTIFICATION_DISPATCHER_ENABLED", False
        )
        self.deliver_backlog = (
            env_bool("GPU_FAULT_NOTIFICATION_DELIVER_BACKLOG", False)
            if deliver_backlog is None
            else deliver_backlog
        )
        self.backlog_grace_seconds = (
            int(
                os.getenv(
                    "GPU_FAULT_NOTIFICATION_BACKLOG_GRACE_SECONDS",
                    "300",
                )
            )
            if backlog_grace_seconds is None
            else backlog_grace_seconds
        )
        if self.backlog_grace_seconds < 0:
            raise ValueError("notification backlog grace must not be negative")
        self.ttl_seconds = (
            int(
                os.getenv(
                    "GPU_FAULT_NOTIFICATION_TTL_SECONDS",
                    str(DEFAULT_NOTIFICATION_TTL_SECONDS),
                )
            )
            if ttl_seconds is None
            else ttl_seconds
        )
        if self.ttl_seconds < 0:
            raise ValueError("notification ttl must not be negative")
        # Drills are mail about a fault that did not happen. The label
        # already says so in the subject and the first line of the body,
        # which is enough for a human reading one and not enough for a
        # pressure test raising 16,384: opt in per deployment if the point
        # of the drill is to exercise the mail path itself.
        self.deliver_drills = (
            env_bool("GPU_FAULT_NOTIFICATION_DELIVER_DRILLS", False)
            if deliver_drills is None
            else deliver_drills
        )
        self.dispatch_scan_limit = _resolved_dispatch_scan_limit(dispatch_scan_limit)
        # Notification construction is idempotent in the store. A single
        # process-wide lock made unrelated faults wait behind one
        # notification that was reading evidence. Stripe by object key so
        # duplicate work for the same incident remains serialized without
        # turning the whole process into a one-request service.
        self._locks = tuple(RLock() for _ in range(64))
        LOGGER.info(
            "notification delivery: %s",
            self.describe_delivery_mode(),
        )

    def _lock_for(self, key: str) -> RLock:
        digest = hashlib.blake2b(
            key.encode("utf-8"),
            digest_size=8,
        ).digest()
        return self._locks[int.from_bytes(digest, "big") % len(self._locks)]

    def _incident_lock(self, incident_id: str) -> RLock:
        return self._lock_for(f"incident/{incident_id}")

    def _notification_lock(self, notification_id: str) -> RLock:
        return self._lock_for(f"notification/{notification_id}")

    def _evidence_refs_for_incident(
        self,
        incident_id: str,
        *explicit_refs: str | None,
    ) -> list[str]:
        return list(
            dict.fromkeys(
                item
                for item in [
                    *explicit_refs,
                    *(
                        marker.raw_evidence_ref
                        for marker in (
                            self.store.list_markers_for_incident(incident_id)
                        )
                    ),
                ]
                if item
            )
        )

    def delivers_externally(self) -> bool:
        """True only if a notification actually leaves this process.

        The same product of switches ``describe_delivery_mode`` explains,
        reduced to a yes/no so startup can refuse to run an active
        control plane with no way to reach a human. Persisting a
        notification in the store is not a channel: nobody reads the
        store when the recovery they never heard about failed.
        """

        channel = getattr(self.notifier, "config", None)
        if channel is None or not getattr(channel, "execution_enabled", False):
            return False
        return not self.async_delivery or self.dispatcher_enabled

    def describe_delivery_mode(self) -> str:
        """Spell out what the delivery switches *combine* to.

        Three independent switches -- ALLOW_EMAIL (may we call SES),
        DISPATCHER_ENABLED (does this process drain the outbox) and
        ASYNC_DELIVERY (does send() enqueue or deliver inline) -- decide
        whether an operator is actually paged. Each is individually
        sensible, but their product is not: the common failure is every
        notification landing in the store with status QUEUED and nothing
        anywhere saying why no mail arrived. Stating the resolved
        behaviour once at startup turns a database dig into a log grep.
        """

        channel = getattr(self.notifier, "config", None)
        email_enabled = bool(getattr(channel, "execution_enabled", False))
        switches = (
            f"async={_onoff(self.async_delivery)} "
            f"dispatcher={_onoff(self.dispatcher_enabled)} "
            f"email={_onoff(email_enabled)} "
            f"ttl={self._describe_ttl()} "
            f"drills={'delivered' if self.deliver_drills else 'suppressed'}"
        )
        if channel is None:
            outcome = (
                "NOT DELIVERED -- no delivery channel is configured "
                "(set GPU_FAULT_EMAIL_SENDER and "
                "GPU_FAULT_EMAIL_RECIPIENTS); notifications are "
                "persisted only"
            )
        elif not email_enabled:
            outcome = (
                "NOT DELIVERED -- notifications are persisted but SES "
                "is disabled (set GPU_FAULT_ALLOW_EMAIL=true to send)"
            )
        elif not self.async_delivery:
            outcome = "delivered inline by the requesting thread"
        elif self.dispatcher_enabled:
            outcome = "queued to the outbox and drained by this process"
        else:
            outcome = (
                "NOT DELIVERED -- notifications are queued to the "
                "outbox but this process does not drain it (set "
                "GPU_FAULT_NOTIFICATION_DISPATCHER_ENABLED=true on at "
                "least one replica, or POST "
                "/v1/advisory-notifications/dispatch)"
            )
        return f"{switches} -> {outcome}"

    def suppresses(self, notification: AdvisoryNotification) -> str | None:
        """Why this notification must not be mailed, or ``None``.

        Only drills for now. Kept separate from the shelf life because the
        two answer different questions -- "this was never real" against
        "this was real and is no longer worth mailing" -- and because a
        drill has to be stopped before it reaches the outbox, while an
        expiry can only be judged when the delivery is claimed.
        """

        if self.deliver_drills or notification.drill_id is None:
            return None
        return (
            f"drill {notification.drill_id}: not delivered because it "
            "describes a fault that did not happen (set "
            "GPU_FAULT_NOTIFICATION_DELIVER_DRILLS=true to mail drills)"
        )

    def _describe_ttl(self) -> str:
        if self.deliver_backlog:
            return "off (DELIVER_BACKLOG)"
        if self.ttl_seconds <= 0:
            return "off"
        return f"{self.ttl_seconds}s"

    def category_ttl_seconds(self, category: str) -> int:
        """Shelf life for one category in seconds; ``0`` means never expires.

        Categories age at different rates: a trend summary is superseded
        by the next cooldown bucket, while a fault advisory is the only
        record an operator has that a node was rebooted on their behalf.
        A per-category override is read from the environment on every
        call so an operator can widen the window on a live process that is
        already dropping mail.
        """

        suffix = "".join(
            character if character.isalnum() else "_" for character in category.upper()
        )
        override = os.getenv(f"GPU_FAULT_NOTIFICATION_TTL_SECONDS_{suffix}")
        if override is not None:
            value = int(override)
            if value < 0:
                raise ValueError(
                    f"notification ttl for {category} must not be negative"
                )
            return value
        if self.ttl_seconds <= 0:
            return 0
        return min(
            self.ttl_seconds,
            CATEGORY_TTL_SECONDS.get(category, self.ttl_seconds),
        )

    def delivery_deadline(
        self,
        notification: AdvisoryNotification,
        delivery=None,
    ) -> datetime | None:
        """The point after which sending this does more harm than good.

        The outbox has no upper bound on how long a notification can sit
        in it: the dispatcher being off, or SES being denied, queues rows
        that all become deliverable the instant the blockage clears. The
        watermark only covers the very first time a deployment enables the
        dispatcher, so every later gap drains as a flood of mail about
        nodes that were recovered hours ago -- which reads to the operator
        and to their mail provider as an attack rather than as a page.

        Measured from the event, not from the last attempt, so a retry
        chain cannot extend the window indefinitely. ``not_before`` moves
        it forward because a deliberately deferred notification must not
        expire before it is due.
        """

        if self.deliver_backlog:
            # An operator who asked for history has already accepted the
            # volume that comes with it.
            return None
        ttl = self.category_ttl_seconds(notification.category)
        if ttl <= 0:
            return None
        anchor = notification.created_at
        for candidate in (
            notification.not_before,
            getattr(delivery, "requeued_at", None),
        ):
            if candidate is not None and candidate > anchor:
                anchor = candidate
        return anchor + timedelta(seconds=ttl)

    def dispatch_remote_completion(self, command) -> list[NotificationResult]:
        from gpu_fault.regional import RemoteCommandStatus

        if command.status is not RemoteCommandStatus.SUCCEEDED:
            return []
        results = []
        existing_id = command.result_details.get("notification_id")
        if isinstance(existing_id, str):
            try:
                self.store.get_notification(existing_id)
            except NotFoundError:
                pass
            else:
                results.append(self.send(existing_id))
        if command.step.operation is WorkflowOperation.RESTART_WORKLOAD:
            values = command.result_details.get("notification_context", {})
            required = {
                "job_id",
                "source_attempt_id",
                "restart_attempt_id",
                "source_gpu_count",
                "target_gpu_count",
                "restart_count",
                "restart_budget",
            }
            if isinstance(values, dict) and required <= values.keys():
                notification = self.builders.build(
                    NotificationKind.WORKLOAD_RESTARTED,
                    cluster_id=command.cluster_id,
                    incident_id=command.incident.incident_id,
                    workflow_id=command.workflow.request_id,
                    operation_id=command.idempotency_key,
                    job_id=str(values["job_id"]),
                    source_attempt_id=str(values["source_attempt_id"]),
                    restart_attempt_id=str(values["restart_attempt_id"]),
                    workload_ids=command.step.workload_ids,
                    source_gpu_count=int(values["source_gpu_count"]),
                    target_gpu_count=int(values["target_gpu_count"]),
                    restart_count=int(values["restart_count"]),
                    restart_budget=int(values["restart_budget"]),
                )
                notification = self.store.save_notification_if_absent(notification)
                if notification.notification_id != existing_id:
                    results.append(self.send(notification.notification_id))
        results.extend(
            _dispatch_remote_node_action_completion(self, command, existing_id)
        )
        return results

    def preview_not_applicable(
        self,
        incident_id: str,
        event: XidEvent,
        decision: FaultPolicyDecision,
    ) -> AdvisoryNotification:
        return self.previews.preview_not_applicable(incident_id, event, decision)

    def preview_xid_investigatory(
        self,
        incident_id: str,
        event: XidEvent,
        decision: FaultPolicyDecision,
    ) -> AdvisoryNotification:
        return self.previews.preview_xid_investigatory(incident_id, event, decision)

    def preview_sxid_event(
        self,
        incident_id: str,
        event: SxidEvent,
        decision: FaultPolicyDecision,
    ) -> AdvisoryNotification:
        return self.previews.preview_sxid_event(incident_id, event, decision)

    def preview_efa_rdma_event(self, incident_id: str, finding) -> AdvisoryNotification:
        return self.previews.preview_efa_rdma_event(incident_id, finding)

    def preview_hardware_inventory_event(
        self, incident_id: str, finding
    ) -> AdvisoryNotification:
        return self.previews.preview_hardware_inventory_event(incident_id, finding)

    def preview_host_resource_event(
        self, incident_id: str, finding
    ) -> AdvisoryNotification:
        return self.previews.preview_host_resource_event(incident_id, finding)

    def preview(self, incident_id: str) -> AdvisoryNotification:
        return self.previews.preview(incident_id)

    def try_preview(self, incident_id: str) -> AdvisoryNotification | None:
        return self.previews.try_preview(incident_id)

    def send(self, notification_id: str) -> NotificationResult:
        with self._notification_lock(notification_id):
            notification = self.store.get_notification(notification_id)
            existing = self.store.get_notification_result(notification_id)
            if existing is not None and existing.status is NotificationStatus.SENT:
                return existing.model_copy(
                    update={"status": NotificationStatus.DUPLICATE}
                )
            suppressed = self.suppresses(notification)
            if suppressed is not None:
                # Recorded, and deliberately not enqueued. Saving a
                # notification already creates its delivery row, so the
                # outbox retires the drill once on its own; what this
                # avoids is enqueueing it *again*, which restarts the
                # shelf life and revives a row the dispatcher has already
                # retired -- for a pressure test, thousands of them ahead
                # of live traffic in the claim order.
                result = NotificationResult(
                    notification_id=notification.notification_id,
                    status=NotificationStatus.SKIPPED,
                    reason=suppressed,
                )
                self.store.save_notification_result(result)
                return result
            if self.async_delivery:
                self.store.enqueue_notification_delivery(notification.notification_id)
                return NotificationResult(
                    notification_id=notification.notification_id,
                    status=NotificationStatus.QUEUED,
                    reason="queued for asynchronous delivery",
                )
            try:
                result = self.notifier.send(notification)
            except Exception as exc:
                result = NotificationResult(
                    notification_id=notification.notification_id,
                    status=NotificationStatus.FAILED,
                    reason=f"{type(exc).__name__}: {exc}",
                )
            self.store.save_notification_result(result)
            return result

    def dispatch_outbox(
        self,
        owner_id: str,
        *,
        limit: int = 25,
        lease_seconds: int = 120,
        max_attempts: int = 8,
        retry_base_seconds: int = 15,
        retry_max_seconds: int = 900,
    ) -> NotificationDispatchReport:
        if not 1 <= limit <= 100:
            raise ValueError("dispatch limit must be between 1 and 100")
        if lease_seconds < 30:
            raise ValueError("notification lease must be at least 30 seconds")
        now = datetime.now(timezone.utc)
        suppressed = self._establish_watermark(owner_id, now=now)
        deliveries = self.store.claim_notification_deliveries(
            owner_id,
            now=now,
            lease_duration=timedelta(seconds=lease_seconds),
            limit=limit,
        )
        results = []
        expired = 0
        throttled = 0
        suppressed_drills = 0
        oldest_expired: float = 0.0
        for index, delivery in enumerate(deliveries):
            notification = self.store.get_notification(delivery.notification_id)
            # ``send`` keeps drills out of the outbox, so reaching one here
            # means it was enqueued before the switch was set -- or by a
            # caller that enqueued directly. Retiring it costs one write
            # against re-claiming it every cycle until it expires.
            suppression = self.suppresses(notification)
            if suppression is not None:
                self._retire_delivery(
                    notification,
                    delivery,
                    owner_id=owner_id,
                    reason=suppression,
                )
                suppressed_drills += 1
                continue
            deadline = self.delivery_deadline(notification, delivery)
            if deadline is not None and deadline <= now:
                age = (now - notification.created_at).total_seconds()
                self._expire_delivery(
                    notification,
                    delivery,
                    owner_id=owner_id,
                    deadline=deadline,
                    age_seconds=age,
                )
                expired += 1
                oldest_expired = max(oldest_expired, age)
                continue
            recorded = self.store.get_notification_result(notification.notification_id)
            if recorded is not None and recorded.status is NotificationStatus.SENT:
                # The notification id is the idempotency key and the
                # recorded result is the shared truth: an inline ``send``
                # on a sibling, or a dispatcher whose lease lapsed after
                # its mail went out, may have delivered this since the
                # claim. The claim query skips SENT rows, but only as of
                # the claim (P0-38B). Retire the delivery without calling
                # the provider again.
                result = recorded.model_copy(
                    update={
                        "reason": (
                            recorded.reason
                            or "already delivered by another path; not sent again"
                        )
                    }
                )
                self._record_delivery_outcome(
                    notification,
                    delivery,
                    owner_id=owner_id,
                    result=result,
                    now=datetime.now(timezone.utc),
                    retry_at=None,
                    terminal=False,
                )
                results.append(result)
                continue
            try:
                result = self.notifier.send(notification)
            except Exception as exc:
                if _is_throttled(exc):
                    # The provider is rate limiting this process, not
                    # objecting to this message. Every further send in
                    # this batch would deepen the throttle and charge an
                    # attempt to a notification that did nothing wrong --
                    # which is how a storm ends up mailed three times and
                    # then declared undeliverable. Hand the rest of the
                    # batch back untouched and let the next poll retry one.
                    remaining = deliveries[index:]
                    throttled = len(remaining)
                    self._release_throttled(
                        remaining,
                        owner_id=owner_id,
                        retry_seconds=retry_base_seconds,
                    )
                    LOGGER.warning(
                        "notification delivery throttled by the provider "
                        "(%s); returned %d claimed notification(s) to the "
                        "outbox without charging an attempt and stopped "
                        "this cycle -- lower "
                        "GPU_FAULT_NOTIFICATION_BATCH_SIZE or the number "
                        "of dispatcher replicas if this repeats",
                        exc,
                        throttled,
                    )
                    break
                result = NotificationResult(
                    notification_id=notification.notification_id,
                    status=NotificationStatus.FAILED,
                    reason=f"{type(exc).__name__}: {exc}",
                )
            if (
                result.status is NotificationStatus.DUPLICATE
                and result.provider_message_id
            ):
                result = result.model_copy(
                    update={
                        "status": NotificationStatus.SENT,
                        "reason": (
                            result.reason
                            or "provider already accepted this notification"
                        ),
                    }
                )
            completed_at = datetime.now(timezone.utc)
            attempts = delivery.attempts + 1
            terminal = (
                result.status is not NotificationStatus.SENT
                and attempts >= max_attempts
            )
            delay = min(
                retry_max_seconds,
                retry_base_seconds * (2 ** max(0, attempts - 1)),
            )
            self._record_delivery_outcome(
                notification,
                delivery,
                owner_id=owner_id,
                result=result,
                now=completed_at,
                retry_at=completed_at + timedelta(seconds=delay),
                terminal=terminal,
            )
            results.append(result)
        if expired:
            # One line per cycle, not per notification: a flood of drops
            # logged individually is the same noise problem in the log
            # that the drops exist to keep out of the mailbox.
            LOGGER.warning(
                "retired %d notification(s) unsent past their shelf life "
                "(oldest %.0fs old, ttl %s); the outbox is draining "
                "slower than it fills or was blocked -- the notifications "
                "themselves are still readable at "
                "/v1/advisory-notifications",
                expired,
                oldest_expired,
                self._describe_ttl(),
            )
        return NotificationDispatchReport(
            attempted=len(results),
            sent=sum(item.status is NotificationStatus.SENT for item in results),
            skipped=sum(item.status is NotificationStatus.SKIPPED for item in results),
            failed=sum(item.status is NotificationStatus.FAILED for item in results),
            results=results,
            suppressed_backlog=suppressed,
            expired=expired,
            throttled=throttled,
            suppressed_drills=suppressed_drills,
        )

    def _record_delivery_outcome(
        self,
        notification: AdvisoryNotification,
        delivery: NotificationDelivery,
        *,
        owner_id: str,
        result: NotificationResult,
        now: datetime,
        retry_at: datetime | None,
        terminal: bool,
    ) -> None:
        try:
            self.store.complete_notification_delivery(
                notification.notification_id,
                owner_id=owner_id,
                lease_epoch=delivery.lease_epoch,
                result=result,
                now=now,
                retry_at=retry_at,
                terminal=terminal,
            )
        except WorkflowLeaseError:
            # The mail is already out. Losing the record of that
            # because the lease ran out while the provider was slow is
            # what turns one notification into an unbounded stream of
            # identical mail: the row stays claimable, the next
            # replica sends it again, and its bookkeeping fails the
            # same way. Persist the outcome even though the lease is
            # gone -- the claim query skips anything already SENT.
            self.store.save_notification_result(result)
            LOGGER.warning(
                "notification %s was %s but its delivery lease had "
                "already expired; recorded the outcome anyway to stop "
                "it being sent again (raise "
                "GPU_FAULT_NOTIFICATION_LEASE_SECONDS above the time "
                "a full batch takes)",
                notification.notification_id,
                result.status.value,
            )

    def _release_throttled(
        self,
        deliveries,
        *,
        owner_id: str,
        retry_seconds: int,
    ) -> None:
        now = datetime.now(timezone.utc)
        retry_at = now + timedelta(seconds=retry_seconds)
        for delivery in deliveries:
            self.store.release_notification_delivery(
                delivery.notification_id,
                owner_id=owner_id,
                lease_epoch=delivery.lease_epoch,
                now=now,
                retry_at=retry_at,
            )

    def _expire_delivery(
        self,
        notification: AdvisoryNotification,
        delivery,
        *,
        owner_id: str,
        deadline: datetime,
        age_seconds: float,
    ) -> None:
        self._retire_delivery(
            notification,
            delivery,
            owner_id=owner_id,
            reason=(
                f"expired: {age_seconds:.0f}s old, past its "
                f"{self.category_ttl_seconds(notification.category)}s "
                f"{notification.category} shelf life "
                f"(deadline {deadline.isoformat()})"
            ),
        )

    def _retire_delivery(
        self,
        notification: AdvisoryNotification,
        delivery,
        *,
        owner_id: str,
        reason: str,
    ) -> None:
        """Retire a claimed delivery without paying for a send.

        Terminal on purpose: a retired notification that went back to
        PENDING would be re-claimed every cycle forever, and the row it
        occupies is ahead of live traffic in the claim order.
        """

        result = NotificationResult(
            notification_id=notification.notification_id,
            status=NotificationStatus.SKIPPED,
            reason=reason,
        )
        try:
            self.store.complete_notification_delivery(
                notification.notification_id,
                owner_id=owner_id,
                lease_epoch=delivery.lease_epoch,
                result=result,
                now=datetime.now(timezone.utc),
                terminal=True,
            )
        except WorkflowLeaseError:
            # Nothing was sent, so the only thing worth keeping is the
            # verdict; whoever holds the lease now will reach the same one.
            self.store.save_notification_result(result)

    def _establish_watermark(self, owner_id: str, *, now: datetime) -> int:
        """Draw the line between history and live traffic, once.

        Enabling the dispatcher on a deployment that has been queueing
        notifications for days would otherwise mail the whole archive in
        the first few poll cycles. Only the first caller establishes the
        watermark, so replicas starting later still deliver everything
        raised after the feature went live.
        """
        establisher = getattr(
            self.store,
            "establish_notification_watermark",
            None,
        )
        if establisher is None:
            return 0
        # The first poll cycle lands seconds after the process starts, and
        # a sibling replica may already be raising incidents, so anything
        # recent is live traffic rather than backlog.
        cutoff = now - timedelta(seconds=self.backlog_grace_seconds)
        watermark = establisher(
            established_at=cutoff,
            established_by=owner_id,
            suppress_backlog=not self.deliver_backlog,
        )
        return (
            watermark.suppressed
            if watermark.established_by == owner_id
            and watermark.established_at == cutoff
            else 0
        )

    def dispatch_pending(self, limit: int = 25) -> NotificationDispatchReport:
        if not 1 <= limit <= 100:
            raise ValueError("dispatch limit must be between 1 and 100")
        # Newest-first, then re-ordered oldest-first below, rather than reading
        # the oldest ``dispatch_scan_limit`` rows directly: expired advisories and
        # drills stay pending forever and accumulate at the old end, so an
        # oldest-first budget would eventually be spent entirely on entries that
        # can never be sent. See ``NotificationDispatchReport.scan_truncated``.
        scanned = self.store.list_notifications(
            limit=self.dispatch_scan_limit + 1,
            newest_first=True,
        )
        scan_truncated = len(scanned) > self.dispatch_scan_limit
        window = sorted(
            scanned[: self.dispatch_scan_limit],
            key=lambda item: (item.created_at, item.notification_id),
        )
        now = datetime.now(timezone.utc)
        pending = [
            item
            for item in window
            if (
                (result := self.store.get_notification_result(item.notification_id))
                is None
                or result.status
                in {
                    NotificationStatus.SKIPPED,
                    NotificationStatus.FAILED,
                }
            )
        ]
        # This is a bulk re-send over the whole history, and an expired
        # notification is recorded as SKIPPED -- so without the deadline
        # here, one call to this endpoint mails back everything the
        # dispatcher just decided was too old to mail. Re-sending a
        # specific notification on purpose still goes through ``send``,
        # which does not apply a deadline.
        # Drills are recorded SKIPPED, and SKIPPED is what this path picks
        # up, so they have to be filtered rather than left to ``send`` --
        # otherwise every call rewrites the same verdict for every drill in
        # the history and spends the limit on them.
        live = [item for item in pending if self.suppresses(item) is None]
        suppressed_drills = len(pending) - len(live)
        candidates = [
            item
            for item in live
            if (deadline := self.delivery_deadline(item)) is None or deadline > now
        ]
        expired = len(live) - len(candidates)
        candidates = candidates[:limit]
        results = [self.send(item.notification_id) for item in candidates]
        return NotificationDispatchReport(
            expired=expired,
            suppressed_drills=suppressed_drills,
            scan_truncated=scan_truncated,
            attempted=len(results),
            sent=sum(item.status is NotificationStatus.SENT for item in results),
            skipped=sum(item.status is NotificationStatus.SKIPPED for item in results),
            failed=sum(item.status is NotificationStatus.FAILED for item in results),
            results=results,
        )

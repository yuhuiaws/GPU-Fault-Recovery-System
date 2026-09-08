"""Retention for the kinds that had none: markers, notifications, completion
records and registry heartbeat rows.

Control-plane review 2026-09-08, F-8 / G-9 (and F-5 for the registry rows).
``grep "DELETE FROM gpu_fault_objects"`` never touched ``marker``,
``notification``, ``notification_delivery``, ``notification_result``,
``decision``, ``event`` or ``regional_registry_member``; every provider event scanned the whole marker
kind and every scrape aggregated the whole notification kind. Each sweep here
takes only settled rows past retention and keeps anything an incident that
still exists references -- the archiver bundles those with the incident (F-I1).
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from threading import Event
from types import SimpleNamespace

import pytest

from gpu_fault.app.periodic_services import PeriodicServiceConfig, PeriodicServiceRunner
from gpu_fault.models import (
    AdvisoryNotification,
    AllocationEntry,
    CompletionDecision,
    DecisionStatus,
    Environment,
    IncidentState,
    MarkerScope,
    NodeMarker,
    NotificationDeliveryStatus,
    NotificationResult,
    NotificationStatus,
    RankExitStatus,
    RecoveryAction,
    Severity,
    TerminalEvent,
    TerminalStatus,
)
from gpu_fault.regional import RegionalRegistryMember
from gpu_fault.store import NotFoundError, SqliteStore
from gpu_fault.telemetry import CollectorKind
from tests._builders import build_store, fault_incident
from tests.store._postgres_processor_claim_support import (
    POSTGRES_URL,
    _truncate,
    postgres_store_instance,
)

# Far enough ahead of the wall clock that rows the store stamps with
# ``datetime.now()`` (delivery ``available_at``) are always claimable at NOW.
NOW = datetime(2030, 1, 1, 12, 0, tzinfo=timezone.utc)
OLD = NOW - timedelta(days=40)
CUTOFF = NOW - timedelta(days=30)
LIMIT = 100


@pytest.fixture(params=["memory", "sqlite", "postgres"])
def store(request, tmp_path):
    if request.param == "memory":
        yield build_store()
        return
    if request.param == "sqlite":
        instance = SqliteStore(str(tmp_path / "unbounded-kinds.db"))
        try:
            yield instance
        finally:
            instance.close()
        return
    if not POSTGRES_URL:
        pytest.skip("GPU_FAULT_TEST_POSTGRES_URL is required")
    yield from postgres_store_instance()
    _truncate()


@pytest.fixture(params=["memory", "sqlite", "postgres"])
def durable_store(request, tmp_path):
    """All three backends implement the registry-member sweep."""

    if request.param == "memory":
        yield build_store()
        return
    if request.param == "sqlite":
        instance = SqliteStore(str(tmp_path / "members.db"))
        try:
            yield instance
        finally:
            instance.close()
        return
    if not POSTGRES_URL:
        pytest.skip("GPU_FAULT_TEST_POSTGRES_URL is required")
    yield from postgres_store_instance()
    _truncate()


# ---------------------------------------------------------------- markers


def _marker(marker_id: str, *, active: bool, observed_at: datetime, incident_id: str):
    return NodeMarker(
        marker_id=marker_id,
        source="test",
        cluster_id="cluster-a",
        trusted=True,
        active=active,
        incident_id=incident_id,
        observed_at=observed_at,
        expires_at=observed_at + timedelta(minutes=30),
        scope=MarkerScope(node_ids=["node-a"]),
        severity=Severity.CRITICAL,
        recommended_action=RecoveryAction.RESET_GPU,
        action_owner="test",
        mapping_version="v1",
    )


def test_inactive_orphan_markers_past_retention_are_removed(store) -> None:
    store.save_incident(fault_incident("inc-live", "event-live"))
    store.add_marker(
        _marker("m-old-orphan", active=False, observed_at=OLD, incident_id="inc-gone")
    )
    store.add_marker(
        _marker("m-old-active", active=True, observed_at=OLD, incident_id="inc-gone")
    )
    store.add_marker(
        _marker(
            "m-old-live-incident", active=False, observed_at=OLD, incident_id="inc-live"
        )
    )
    store.add_marker(
        _marker("m-fresh", active=False, observed_at=NOW, incident_id="inc-gone")
    )

    removed = store.cleanup_inactive_markers(older_than=CUTOFF, limit=LIMIT)

    assert removed == 1
    remaining = {
        marker.marker_id for marker in store.list_markers_for_incident("inc-gone")
    }
    assert remaining == {"m-old-active", "m-fresh"}
    assert [m.marker_id for m in store.list_markers_for_incident("inc-live")] == [
        "m-old-live-incident"
    ]


def test_marker_sweep_is_bounded_and_oldest_first(store) -> None:
    for index in range(3):
        store.add_marker(
            _marker(
                f"m-{index}",
                active=False,
                observed_at=OLD + timedelta(hours=index),
                incident_id="inc-gone",
            )
        )

    assert store.cleanup_inactive_markers(older_than=CUTOFF, limit=2) == 2
    assert [m.marker_id for m in store.list_markers_for_incident("inc-gone")] == ["m-2"]


# ---------------------------------------------------------- notifications


def _notification(store, suffix: str, *, created_at: datetime, incident_id: str):
    return store.save_notification_if_absent(
        AdvisoryNotification(
            notification_id=f"notification-{suffix}",
            deduplication_key=f"dedup-{suffix}",
            cluster_name="cluster-a",
            incident_id=incident_id,
            subject="subject",
            body_text="body",
            support_case_draft="",
            created_at=created_at,
        )
    )


def _settle(store, verdicts: dict[str, NotificationStatus]) -> None:
    """One claim of everything queued; the named deliveries are completed
    terminally with their verdict, the rest are released back (RETRY)."""

    claimed = store.claim_notification_deliveries(
        "pod-a", now=NOW, lease_duration=timedelta(seconds=60), limit=100
    )
    for delivery in claimed:
        status = verdicts.get(delivery.notification_id)
        if status is None:
            store.release_notification_delivery(
                delivery.notification_id,
                owner_id="pod-a",
                lease_epoch=delivery.lease_epoch,
                now=NOW,
                retry_at=NOW,
            )
            continue
        store.complete_notification_delivery(
            delivery.notification_id,
            owner_id="pod-a",
            lease_epoch=delivery.lease_epoch,
            result=NotificationResult(
                notification_id=delivery.notification_id, status=status
            ),
            now=NOW,
            terminal=True,
        )


def test_settled_orphan_notifications_are_removed_with_delivery_result_and_dedup(
    store,
) -> None:
    store.save_incident(fault_incident("inc-live", "event-live"))
    sent = _notification(store, "sent", created_at=OLD, incident_id="inc-gone")
    dead = _notification(store, "dead", created_at=OLD, incident_id="inc-gone")
    pending = _notification(store, "pending", created_at=OLD, incident_id="inc-gone")
    owned = _notification(store, "owned", created_at=OLD, incident_id="inc-live")
    fresh = _notification(store, "fresh", created_at=NOW, incident_id="inc-gone")
    _settle(
        store,
        {
            sent.notification_id: NotificationStatus.SENT,
            dead.notification_id: NotificationStatus.FAILED,
        },
    )
    assert store.get_notification_delivery(pending.notification_id).status is (
        NotificationDeliveryStatus.RETRY
    )
    assert store.get_notification_delivery(dead.notification_id).status is (
        NotificationDeliveryStatus.DEAD
    )

    removed = store.cleanup_terminal_notifications(older_than=CUTOFF, limit=LIMIT)

    assert removed == 2
    for gone in (sent, dead):
        with pytest.raises(NotFoundError):
            store.get_notification(gone.notification_id)
        assert store.get_notification_delivery(gone.notification_id) is None
        assert store.get_notification_result(gone.notification_id) is None
    for kept in (pending, owned, fresh):
        assert store.get_notification(kept.notification_id).notification_id == (
            kept.notification_id
        )
    # The dedup link went with the notification: the same key creates a new
    # notification instead of resolving to a row that no longer exists.
    again = _notification(store, "sent", created_at=NOW, incident_id="inc-gone")
    assert again.notification_id == "notification-sent"
    assert again.created_at == NOW
    # ... while a kept notification's key still deduplicates.
    same = _notification(store, "pending", created_at=NOW, incident_id="inc-gone")
    assert same.created_at == OLD


# ----------------------------------------------------- completion records


def _event(attempt_id: str, *, ended_at: datetime) -> TerminalEvent:
    return TerminalEvent(
        cluster_id="cluster-a",
        environment=Environment.KUBERNETES,
        job_id="train-123",
        attempt_id=attempt_id,
        terminal_status=TerminalStatus.FAILED,
        ended_at=ended_at,
        rank_exit_status=[
            RankExitStatus(rank=0, exit_code=1, node_id="node-a", finished_at=ended_at)
        ],
        allocation=[
            AllocationEntry(
                node_id="node-a", instance_id="i-a", rank=0, gpu_uuids=["GPU-a"]
            )
        ],
        runtime_profile_version="profile-v1",
    )


def _decision(
    store,
    attempt_id: str,
    *,
    ended_at: datetime,
    status: DecisionStatus = DecisionStatus.NO_ACTION,
    plan_incident_id: str | None = None,
) -> CompletionDecision:
    event = _event(attempt_id, ended_at=ended_at)
    store.save_event_if_absent(event)
    plan_id = None
    if plan_incident_id is not None:
        from gpu_fault.models import PlanStep, RecoveryPlan

        plan_id = f"plan-{attempt_id}"
        store.save_plan(
            RecoveryPlan(
                plan_id=plan_id,
                incident_id=plan_incident_id,
                attempt_id=attempt_id,
                trigger="test",
                runtime_profile_version="profile-v1",
                steps=[
                    PlanStep(
                        action=RecoveryAction.ESCALATE_OPERATOR,
                        execution_owner="operator",
                    )
                ],
            )
        )
    decision = CompletionDecision(
        cluster_id="cluster-a",
        attempt_id=attempt_id,
        event_key=event.event_key,
        status=status,
        reason="test",
        recovery_plan_id=plan_id,
    )
    store.save_decision(decision)
    return decision


def test_settled_completion_records_past_retention_are_removed_together(store) -> None:
    store.save_incident(fault_incident("inc-live", "event-live"))
    old = _decision(store, "a-old", ended_at=OLD)
    owned = _decision(
        store,
        "a-owned",
        ended_at=OLD,
        status=DecisionStatus.PLAN_CREATED,
        plan_incident_id="inc-live",
    )
    orphan_plan = _decision(
        store,
        "a-orphan-plan",
        ended_at=OLD,
        status=DecisionStatus.PLAN_CREATED,
        plan_incident_id="inc-gone",
    )
    fresh = _decision(store, "a-fresh", ended_at=NOW)

    removed = store.cleanup_completion_records(older_than=CUTOFF, limit=LIMIT)

    assert removed == 2
    for gone in (old, orphan_plan):
        assert store.get_decision_by_event(gone.event_key) is None
        with pytest.raises(NotFoundError):
            store.get_event_by_attempt("cluster-a", gone.attempt_id)
    with pytest.raises(NotFoundError):
        store.get_plan(orphan_plan.recovery_plan_id)
    for kept in (owned, fresh):
        assert store.get_decision_by_event(kept.event_key) is not None
        assert store.get_event_by_attempt("cluster-a", kept.attempt_id).attempt_id == (
            kept.attempt_id
        )
    assert store.get_plan(owned.recovery_plan_id).incident_id == "inc-live"


def test_a_decision_whose_event_an_incident_still_names_is_kept(store) -> None:
    decision = _decision(store, "a-named", ended_at=OLD)
    store.save_incident(
        fault_incident(
            "inc-from-event", decision.event_key, state=IncidentState.RECOVERED
        )
    )

    assert store.cleanup_completion_records(older_than=CUTOFF, limit=LIMIT) == 0
    assert store.get_decision_by_event(decision.event_key) is not None


# -------------------------------------------------------- registry members


def _member(member_id: str, *, last_seen_at: datetime) -> RegionalRegistryMember:
    return RegionalRegistryMember(
        member_id=member_id,
        service_role="worker",
        release_id="release-1",
        generation=1,
        content_sha256="a" * 64,
        ready=True,
        started_at=last_seen_at - timedelta(hours=1),
        last_seen_at=last_seen_at,
    )


def test_registry_members_not_seen_for_a_day_are_removed(durable_store) -> None:
    store = durable_store
    store.save_regional_registry_member(
        _member("pod-old:1", last_seen_at=NOW - timedelta(days=2))
    )
    store.save_regional_registry_member(
        _member("pod-old:2", last_seen_at=NOW - timedelta(hours=25))
    )
    store.save_regional_registry_member(
        _member("pod-live:1", last_seen_at=NOW - timedelta(seconds=5))
    )

    removed = store.cleanup_stale_regional_registry_members(
        older_than=NOW - timedelta(hours=24), limit=LIMIT
    )

    assert removed == 2
    assert [m.member_id for m in store.list_regional_registry_members()] == [
        "pod-live:1"
    ]


# ------------------------------------------------------------ runner wiring


class _RecordingStore:
    def __init__(self) -> None:
        self.calls: dict[str, dict] = {}

    def __getattr__(self, name: str):
        if not name.startswith("cleanup_"):
            raise AttributeError(name)

        def record(**kwargs):
            self.calls[name] = kwargs
            return 0

        return record


def _config(**overrides) -> PeriodicServiceConfig:
    base = PeriodicServiceConfig(
        training_interval=15.0,
        spare_interval=30.0,
        identity_interval=20.0,
        cleanup_interval=60.0,
        completed_retention=600.0,
        cleanup_batch_size=10,
        cleanup_budget_seconds=0.2,
        batch_retention=86400.0,
        terminal_retention=2592000.0,
        latest_retention=2592000.0,
        finding_retention=2592000.0,
        observation_max_age=604800.0,
        remote_retention=86400.0,
        remote_claim_deadline=900.0,
        lane_retention=3600.0,
        deployment_retention=604800.0,
        archive_interval=600.0,
        silence_interval=60.0,
        silent_after={kind: 300.0 for kind in CollectorKind},
        silent_alert_interval=3600.0,
    )
    return replace(base, **overrides)


def _runner(store, *, regional: bool, **config) -> PeriodicServiceRunner:
    return PeriodicServiceRunner(
        context=SimpleNamespace(store=store, regional_mode=regional),
        processor=SimpleNamespace(
            is_healthy=lambda: True,
            active_consumers=False,
            is_leader=lambda: True,
            owner_id="pod-a:1",
        ),
        stop=Event(),
        identity_registries=[],
        ingest_node_health_findings=lambda *a, **k: None,
        notify_silent_collectors=lambda *a, **k: None,
        config=_config(**config),
    )


def test_the_cleanup_round_runs_the_new_sweeps_with_their_retention() -> None:
    store = _RecordingStore()
    runner = _runner(store, regional=True)
    before = datetime.now(timezone.utc)

    runner._cleanup_round()

    assert {
        "cleanup_inactive_markers",
        "cleanup_terminal_notifications",
        "cleanup_completion_records",
        "cleanup_stale_regional_registry_members",
    } <= set(store.calls)
    thirty_days = before - timedelta(seconds=2592000)
    for name in (
        "cleanup_inactive_markers",
        "cleanup_terminal_notifications",
        "cleanup_completion_records",
    ):
        assert abs((store.calls[name]["older_than"] - thirty_days).total_seconds()) < 5
        assert store.calls[name]["limit"] == 10
    members = store.calls["cleanup_stale_regional_registry_members"]
    assert (
        abs((members["older_than"] - (before - timedelta(days=1))).total_seconds()) < 5
    )


def test_a_non_positive_retention_switches_a_sweep_off() -> None:
    store = _RecordingStore()
    runner = _runner(
        store,
        regional=False,
        marker_retention=0.0,
        notification_retention=0.0,
        completion_record_retention=0.0,
    )

    runner._cleanup_round()

    assert not {
        "cleanup_inactive_markers",
        "cleanup_terminal_notifications",
        "cleanup_completion_records",
        "cleanup_stale_regional_registry_members",
    } & set(store.calls)


def test_retention_and_archive_batch_are_read_from_the_environment(monkeypatch) -> None:
    monkeypatch.setenv("GPU_FAULT_CONTROL_RECORD_ARCHIVE_BATCH_SIZE", "50")
    monkeypatch.setenv("GPU_FAULT_MARKER_RETENTION_SECONDS", "100")
    monkeypatch.setenv("GPU_FAULT_NOTIFICATION_RETENTION_SECONDS", "200")
    monkeypatch.setenv("GPU_FAULT_COMPLETION_RECORD_RETENTION_SECONDS", "300")
    monkeypatch.setenv("GPU_FAULT_REGISTRY_MEMBER_RETENTION_SECONDS", "400")
    monkeypatch.delenv(
        "GPU_FAULT_CONTROL_RECORD_ARCHIVE_INTERVAL_SECONDS", raising=False
    )

    config = PeriodicServiceConfig.from_environment()

    assert config.archive_batch_size == 50
    assert config.archive_interval == 600.0
    assert (
        config.marker_retention,
        config.notification_retention,
        config.completion_record_retention,
        config.registry_member_retention,
    ) == (100.0, 200.0, 300.0, 400.0)
    monkeypatch.setenv("GPU_FAULT_CONTROL_RECORD_ARCHIVE_BATCH_SIZE", "0")
    with pytest.raises(ValueError, match="ARCHIVE_BATCH_SIZE"):
        PeriodicServiceConfig.from_environment()

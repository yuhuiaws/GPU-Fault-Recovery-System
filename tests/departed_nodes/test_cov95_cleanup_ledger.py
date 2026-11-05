"""``departed_node_cleanup``: the ledger line, its notice and its retention.

A departure's ledger line is written when its verdict changes and paged once
per verdict: a node cleaned twice (a late incident of a departed node) gets
no second notice; a refusal that comes back to an earlier set of reasons is
not paged again; a notice that cannot be sent never undoes the cleanup. The
administrator's answers live as long as the line they answer, or the
retention when no line came. The cycles run on the in-memory store.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import timedelta

import pytest

from gpu_fault.departed_node_cleanup import departed_node_cleanup_metric_lines
from gpu_fault.departed_node_cleanup_actions import INCIDENT_NOT_CLOSABLE
from gpu_fault.departed_nodes import (
    DepartedNodeCleanupStatus,
    DepartedNodeIdentityAcceptance,
    DepartedNodeSpareRelease,
)
from gpu_fault.models import IncidentState
from tests._builders import build_store
from tests.departed_nodes._cov95_departure_edges import (
    ScriptedClosure,
    cleanup,
    hyperpod_departed,
    incident,
)
from tests.departed_nodes._support import (
    HP_CLUSTER,
    HP_INSTANCE,
    HP_NAME,
    HP_NODE,
    HP_PRESENT,
    ago,
)

LOGGER_NAME = "gpu_fault.departed_node_cleanup"
CLEANED = DepartedNodeCleanupStatus.CLEANED
REFUSED = DepartedNodeCleanupStatus.REFUSED


@dataclass
class FailingNotifications:
    """An alert sender that is down (SNS throttling, a revoked topic policy)."""

    attempts: list[str] = field(default_factory=list)

    def send(self, notification_id: str) -> None:
        self.attempts.append(notification_id)
        raise RuntimeError("SNS publish throttled")


def _release(
    cluster_id: str, node_id: str, *, age: timedelta
) -> DepartedNodeSpareRelease:
    return DepartedNodeSpareRelease(
        cluster_id=cluster_id,
        node_id=node_id,
        instance_id="i-0a0000ff",
        change_reference="CHG-spare",
        actor="ops@example",
        released_at=ago(seconds=age.total_seconds()),
        declaration="released",
    )


def _acceptance(
    cluster_id: str, node_id: str, *, age: timedelta
) -> DepartedNodeIdentityAcceptance:
    return DepartedNodeIdentityAcceptance(
        cluster_id=cluster_id,
        node_id=node_id,
        change_reference="CHG-accept",
        actor="ops@example",
        reason="never a node",
        accepted_at=ago(seconds=age.total_seconds()),
    )


def test_a_node_back_in_kubernetes_forgets_its_line_and_the_spare_release() -> None:
    store = build_store()
    world_ = hyperpod_departed(store)
    world_.provider.hyperpod[HP_NAME].add(HP_INSTANCE)
    [pending] = world_.cleanup.run_once()
    assert pending.status is DepartedNodeCleanupStatus.PENDING, pending.blockers
    store.save_departed_node_spare_release(
        _release(HP_CLUSTER, HP_NODE, age=timedelta(minutes=1))
    )

    world_.reports_since(HP_CLUSTER, [HP_PRESENT, HP_NODE], minutes=1)

    assert world_.cleanup.run_once() == [], "a node in Kubernetes is no departure"
    assert store.list_departed_node_cleanups(HP_CLUSTER) == []
    assert store.list_departed_node_spare_releases(HP_CLUSTER) == [], (
        "the administrator's answer goes with the line it answered"
    )


def test_a_late_incident_of_a_cleaned_node_is_ended_without_a_second_notice(
    caplog: pytest.LogCaptureFixture,
) -> None:
    store = build_store()
    world_ = hyperpod_departed(store)
    first = incident(store, "first")
    [cleaned] = world_.cleanup.run_once()
    assert cleaned.status is CLEANED, cleaned.blockers
    late = incident(store, "late")
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)

    [again] = world_.cleanup.run_once()

    assert again.status is CLEANED, again.blockers
    assert again.reference == cleaned.reference, "the same departure"
    assert again.actions == [
        f"agent-retired:{HP_NODE}",
        f"incident-closed:{first}",
        f"incident-closed:{late}",
    ], again.actions
    assert store.get_incident(late).state is IncidentState.RECOVERED
    assert len(world_.notifications.sent) == 1, "one cleaned notice per departure"
    assert any(
        record.getMessage()
        == (
            f"departed-node cleanup: {HP_CLUSTER}/{HP_NODE} CLEANED actions="
            f"agent-retired:{HP_NODE},incident-closed:{first},"
            f"incident-closed:{late} blockers=-"
        )
        for record in caplog.records
    ), [record.getMessage() for record in caplog.records]


def test_a_cleanup_without_a_notification_sender_still_writes_its_line() -> None:
    store = build_store()
    world_ = hyperpod_departed(store)
    incident(store, "quiet")

    [record] = cleanup(world_).run_once()

    assert record.status is CLEANED, record.blockers
    assert store.list_departed_node_cleanups(HP_CLUSTER) == [record]
    assert store.list_notifications() == [], "nothing is queued without a sender"


def test_a_refusal_that_returns_to_earlier_reasons_is_not_paged_again() -> None:
    store = build_store()
    world_ = hyperpod_departed(store)
    held = incident(store, "held")
    racing = incident(store, "racing")
    closure = ScriptedClosure(
        store,
        {
            held: ["refuse:incident incident-held is DETECTED, not ESCALATED"] * 3,
            racing: ["unchanged", "stale", "close"],
        },
    )
    cycle = cleanup(world_, closure=closure, notifications=world_.notifications)

    verdicts = [cycle.run_once()[0] for _ in range(3)]

    assert [item.status for item in verdicts] == [REFUSED] * 3
    assert [item.blocker_classes for item in verdicts] == [
        [INCIDENT_NOT_CLOSABLE],
        [INCIDENT_NOT_CLOSABLE, "record_moved"],
        [INCIDENT_NOT_CLOSABLE],
    ]
    assert verdicts[-1].actions == [
        f"agent-retired:{HP_NODE}",
        f"incident-closed:{racing}",
    ]
    assert len(world_.notifications.sent) == 2, (
        "the third verdict's reasons were paged by the first"
    )
    assert len(store.list_notifications()) == 2


def test_a_notice_that_cannot_be_sent_never_undoes_the_cleanup(
    caplog: pytest.LogCaptureFixture,
) -> None:
    store = build_store()
    world_ = hyperpod_departed(store)
    closed = incident(store, "unpaged")
    sender = FailingNotifications()
    caplog.set_level(logging.ERROR, logger=LOGGER_NAME)

    [record] = cleanup(world_, notifications=sender).run_once()

    assert record.status is CLEANED, record.blockers
    assert store.get_incident(closed).state is IncidentState.RECOVERED
    assert store.list_departed_node_cleanups(HP_CLUSTER) == [record]
    [queued] = store.list_notifications()
    assert sender.attempts == [queued.notification_id]
    [failure] = [
        item
        for item in caplog.records
        if item.getMessage()
        == f"departed-node cleanup notification for {HP_CLUSTER}/{HP_NODE} failed"
    ]
    assert failure.exc_info is not None, "the send failure is logged with its cause"


def test_answers_without_a_line_live_for_the_retention_only() -> None:
    store = build_store()
    world_ = hyperpod_departed(store)
    keep = "hyperpod-i-0a0000bb"
    for node_id, cluster_id, age in (
        ("hyperpod-i-0a0000aa", HP_CLUSTER, timedelta(days=8)),
        (keep, HP_CLUSTER, timedelta(hours=1)),
        ("hyperpod-i-0a0000cc", "gpu-removed", timedelta(hours=1)),
    ):
        store.save_departed_node_spare_release(_release(cluster_id, node_id, age=age))
        store.save_departed_node_identity_acceptance(
            _acceptance(cluster_id, node_id, age=age)
        )

    world_.cleanup.run_once()

    assert [item.node_id for item in store.list_departed_node_spare_releases()] == [
        keep
    ], "an old answer, or one for a cluster that left, is dropped"
    assert [
        item.node_id for item in store.list_departed_node_identity_acceptances()
    ] == [keep]


def test_the_metrics_render_the_counters_alone_without_a_ledger() -> None:
    store = build_store()
    world_ = hyperpod_departed(store)
    world_.provider.hyperpod[HP_NAME].add(HP_INSTANCE)
    [pending] = world_.cleanup.run_once()

    counters = departed_node_cleanup_metric_lines(world_.cleanup)
    with_ledger = departed_node_cleanup_metric_lines(
        world_.cleanup, ledger=store.list_departed_node_cleanups
    )

    assert counters == world_.cleanup.metric_lines()
    assert "gpu_fault_departed_node_cleanup_cycles_total 1" in counters
    assert with_ledger[: len(counters)] == counters
    assert (
        "gpu_fault_departed_node_cleanup_open_nodes"
        f'{{cluster_id="{HP_CLUSTER}",status="{pending.status.value}"}} 1'
    ) in with_ledger, with_ledger
    assert not [line for line in counters if "open_nodes{" in line], (
        "the ledger's gauges need the ledger"
    )

"""GF-GENERAL-LIVE-005 ``shared_failure_domain``: the correlator's edges.

``test_shared_storage_failure_domain.py`` drives the correlator through host
ingestion. These cases call its three entry points -- ``observe_finding`` (the
look inside the activation's transaction), ``observe_batch`` (the clear and
the safety net) and ``observe_committed`` (the look after the commit) --
directly on hand-written records: peers known only through another export's
record, stale and silent members, a closed episode rebuilt from the same
per-node incidents, a record-only fold that names the node by its marker, the
handling a member line names, and what a write does when a sibling replica
wrote the same record first.
"""

from __future__ import annotations

import logging
from datetime import timedelta
from typing import Any

import pytest

from gpu_fault.models import FaultIncident, IncidentState, RecoveryAction
from gpu_fault.orchestration.families.incident_labels import (
    NODE_HEALTH_GROUP_EVENT_TYPE,
)
from gpu_fault.shared_storage_failure_domain import (
    SHARED_FILESYSTEM_METRIC,
    SHARED_STORAGE_FAILURE_DOMAIN_WINDOW_SECONDS,
    SharedStorageFailureDomainCorrelator,
)
from gpu_fault.store import InMemoryStore, NotFoundError
from gpu_fault.store.shared.health_signals import health_signal_key
from tests.host_health._cov95_shared_storage import (
    CLUSTER,
    EXPORT,
    MOUNT,
    OTHER_EXPORT,
    T0,
    EarlierListingStore,
    SiblingReplicaStore,
    activation_event_id,
    batch,
    domain_record,
    domain_records,
    finding,
    node_incident,
    reading,
    routed_marker,
)

LOGGER_NAME = "gpu_fault.shared_storage_failure_domain"
WINDOW = timedelta(seconds=SHARED_STORAGE_FAILURE_DOMAIN_WINDOW_SECONDS)


def correlator(store: Any) -> tuple[SharedStorageFailureDomainCorrelator, list[str]]:
    sent: list[str] = []
    return SharedStorageFailureDomainCorrelator(store, sent.append), sent


def two_activations(store: InMemoryStore) -> tuple[FaultIncident, FaultIncident]:
    """node-a's and node-b's own diagnostic-only records, both still open."""

    first = node_incident("incident-a", "node-a")
    second = node_incident("incident-b", "node-b")
    store.save_incident(first)
    store.save_incident(second)
    return first, second


@pytest.mark.parametrize("window", [0, -15.0])
def test_the_correlation_window_must_be_positive(window) -> None:
    with pytest.raises(ValueError, match="window must be positive"):
        SharedStorageFailureDomainCorrelator(
            InMemoryStore(), lambda _id: None, window_seconds=window
        )


def test_a_finding_of_another_metric_is_not_correlated() -> None:
    store = InMemoryStore()
    _a, second = two_activations(store)
    reading(store, "node-a", 1, T0)
    look, sent = correlator(store)

    result = look.observe_finding(
        finding("node-b", T0, metric="memory_used_percent"), second
    )

    assert result == (None, None)
    assert domain_records(store) == [] and sent == [], "nothing is recorded"


def test_a_reading_that_is_neither_available_nor_failing_settles_nothing() -> None:
    store = InMemoryStore()
    domain = domain_record("domain-1", ["node-a", "node-b"])
    store.save_incident(domain)
    reading(store, "node-a", float("nan"), T0)
    reading(store, "node-b", 0, T0)
    look, sent = correlator(store)

    assert look.observe_batch(batch("node-a", float("nan"), T0), []) == []
    assert store.get_incident("domain-1") == domain, "NaN is no evidence either way"
    assert sent == []


def test_the_look_after_the_commit_skips_an_activation_holding_no_incident() -> None:
    """node-a's event was never linked and no record holds its activation --
    not even a closed one under a label no node-health finding carries -- so
    a failing peer does not make a record of it."""

    store = InMemoryStore()
    store.save_incident(node_incident("incident-b", "node-b"))
    store.save_incident(
        node_incident(
            "incident-xid",
            "node-a",
            batch_id="node-a-0",
            event_type="XID",
            state=IncidentState.RECOVERED,
            effective_action=RecoveryAction.RESET_GPU,
        )
    )
    reading(store, "node-b", 1, T0)
    look, sent = correlator(store)

    assert look.observe_committed(batch("node-a", 1, T0), [finding("node-a", T0)]) == []
    assert domain_records(store) == [] and sent == [], "no member without a record"


def test_a_peer_known_only_through_another_exports_record_joins_without_one() -> None:
    """node-b belongs to the record of another export and holds no activation
    of this one; its latest row says this export is down too, so it is a
    member -- with no incident of its own to point back from."""

    store = InMemoryStore()
    other = domain_record("domain-other", ["node-b", "node-c"], export=OTHER_EXPORT)
    store.save_incident(other)
    own = node_incident("incident-a", "node-a")
    store.save_incident(own)
    reading(store, "node-b", 1, T0 - timedelta(seconds=5), device="/mnt/second")
    look, sent = correlator(store)

    domain, notification_id = look.observe_finding(finding("node-a", T0), own)

    assert domain is not None and notification_id is not None, "a record was formed"
    assert domain.node_ids == ["node-a", "node-b"]
    assert domain.policy_reference == f"shared-filesystem:{EXPORT}"
    assert any(
        reason.startswith("member node-b: incident none open (")
        and "on /mnt/second" in reason
        for reason in domain.reasons
    ), domain.reasons
    assert store.get_incident("domain-other") == other, "the other export's record"
    back = store.get_incident("incident-a").reasons[-1]
    assert back.startswith(f"common failure domain {domain.incident_id}:"), back
    assert back.endswith("the diagnostic-only handling of this node is unchanged"), back
    [mail] = [store.get_notification(item) for item in sent]
    assert "  node-b: no open incident (mount /mnt/second)\n" in mail.body_text
    assert "Node handling: unchanged and diagnostic-only" in mail.body_text
    assert mail.evidence_refs == ["incident-a"]


def test_a_peer_whose_failure_is_older_than_the_window_is_not_a_peer() -> None:
    store = InMemoryStore()
    own, _b = two_activations(store)
    reading(store, "node-b", 1, T0 - WINDOW - timedelta(seconds=1))
    look, sent = correlator(store)

    assert look.observe_finding(finding("node-a", T0), own) == (None, None)
    assert domain_records(store) == [] and sent == [], "a stale failure is silence"


def _safety_net_case(*, active_since: Any) -> tuple[InMemoryStore, Any]:
    store = InMemoryStore()
    two_activations(store)
    reading(store, "node-b", 1, T0)
    if active_since is not None:
        store.claim_health_signal_transition(
            health_signal_key(CLUSTER, "node-a", SHARED_FILESYSTEM_METRIC, MOUNT),
            True,
            active_since,
        )
    return store, correlator(store)


def test_the_safety_net_acts_only_for_an_active_signal() -> None:
    store, (look, sent) = _safety_net_case(active_since=None)

    assert look.observe_batch(batch("node-a", 1, T0, received_at=T0), []) == []
    assert domain_records(store) == [] and sent == [], "no active signal, no look"


@pytest.mark.parametrize(
    ("after", "records"),
    [(WINDOW - timedelta(seconds=1), 1), (WINDOW + timedelta(seconds=1), 0)],
    ids=["inside-the-window", "after-the-window"],
)
def test_the_safety_net_ends_with_the_activations_window(after, records) -> None:
    store, (look, sent) = _safety_net_case(active_since=T0)

    notifications = look.observe_batch(
        batch("node-a", 1, T0 + timedelta(seconds=30), received_at=T0 + after), []
    )

    assert len(domain_records(store)) == records
    assert notifications == sent and len(sent) == records


def test_the_safety_net_leaves_a_node_already_in_the_record_alone() -> None:
    store, (look, sent) = _safety_net_case(active_since=T0)
    domain = domain_record("domain-1", ["node-a", "node-b"])
    store.save_incident(domain)

    assert (
        look.observe_batch(
            batch("node-a", 1, T0 + timedelta(seconds=30), received_at=T0), []
        )
        == []
    )
    assert store.get_incident("domain-1") == domain and sent == [], "no rewrite"


def test_an_available_reading_settles_only_the_record_of_its_export(caplog) -> None:
    store = InMemoryStore()
    mine = domain_record("domain-export", ["node-a", "node-b"])
    other = domain_record("domain-other", ["node-a", "node-c"], export=OTHER_EXPORT)
    own = node_incident("incident-a", "node-a")
    for incident in (mine, other, own):
        store.save_incident(incident)
    reading(store, "node-b", 0, T0)
    reading(store, "node-a", 0, T0 + timedelta(seconds=10))
    look, _sent = correlator(store)

    with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
        look.observe_batch(batch("node-a", 0, T0 + timedelta(seconds=10)), [])

    cleared = store.get_incident("domain-export")
    assert cleared.state is IncidentState.RECOVERED, cleared
    assert cleared.reasons[-1] == (
        f"cleared: every member node reports shared filesystem {EXPORT} available "
        "again (node-a, node-b)"
    )
    assert store.get_incident("domain-other") == other
    assert store.get_incident("incident-a") == own
    assert "shared-storage failure domain domain-export -> RECOVERED" in caplog.text


def test_a_member_with_no_reading_at_all_hands_the_record_to_the_operator() -> None:
    store = InMemoryStore()
    store.save_incident(domain_record("domain-1", ["node-a", "node-b"]))
    reading(store, "node-a", 0, T0)
    look, _sent = correlator(store)

    look.observe_batch(batch("node-a", 0, T0), [])

    escalated = store.get_incident("domain-1")
    assert escalated.state is IncidentState.ESCALATED, escalated
    assert escalated.reasons[-1] == (
        f"node-a reports shared filesystem {EXPORT} available again; node-b has no "
        "reading of the export: silence is not recovery, the operator owns the close"
    )


def test_an_escalated_record_is_not_rewritten_while_a_member_stays_silent() -> None:
    store = InMemoryStore()
    escalated = domain_record(
        "domain-1", ["node-a", "node-b"], state=IncidentState.ESCALATED
    )
    store.save_incident(escalated)
    reading(store, "node-a", 0, T0)
    look, _sent = correlator(store)

    look.observe_batch(batch("node-a", 0, T0), [])

    assert store.get_incident("domain-1") == escalated, "nothing new to say"


def test_a_second_outage_on_the_same_open_incidents_is_a_new_record() -> None:
    store = InMemoryStore()
    _a, second = two_activations(store)
    look, sent = correlator(store)
    reading(store, "node-a", 1, T0)
    reading(store, "node-b", 1, T0 + timedelta(seconds=15))
    first, _ = look.observe_finding(
        finding("node-b", T0 + timedelta(seconds=15)), second
    )
    reading(store, "node-a", 0, T0 + timedelta(seconds=60))
    reading(store, "node-b", 0, T0 + timedelta(seconds=75))
    look.observe_batch(batch("node-b", 0, T0 + timedelta(seconds=75)), [])
    assert store.get_incident(first.incident_id).state is IncidentState.RECOVERED

    reading(store, "node-a", 1, T0 + timedelta(seconds=120))
    reading(store, "node-b", 1, T0 + timedelta(seconds=135))
    again, notification_id = look.observe_finding(
        finding("node-b", T0 + timedelta(seconds=135), batch_id="node-b-2"), second
    )

    assert again.incident_id != first.incident_id, "the closed episode is not reused"
    assert again.state is IncidentState.DETECTED and again.node_ids == [
        "node-a",
        "node-b",
    ]
    assert store.get_incident(first.incident_id).state is IncidentState.RECOVERED
    assert notification_id is not None and len(sent) == 2, "one mail per record"
    reasons = store.get_incident("incident-b").reasons
    for record in (first, again):
        assert any(
            item.startswith(f"common failure domain {record.incident_id}:")
            for item in reasons
        ), f"incident-b must point back at {record.incident_id}"


def test_a_record_a_concurrent_look_created_is_reused_without_a_second_mail() -> None:
    store = InMemoryStore()
    _a, second = two_activations(store)
    reading(store, "node-a", 1, T0)
    reading(store, "node-b", 1, T0 + timedelta(seconds=3))
    look, sent = correlator(store)
    created, _ = look.observe_finding(
        finding("node-b", T0 + timedelta(seconds=3)), second
    )
    late, late_sent = correlator(EarlierListingStore(store))

    domain, notification_id = late.observe_finding(
        finding("node-b", T0 + timedelta(seconds=3)), second
    )

    assert domain is not None and domain.incident_id == created.incident_id, domain
    assert notification_id is None and late_sent == [], "the record mailed once"
    assert len(sent) == 1 and domain_records(store) == [created]


def test_a_join_a_sibling_replica_wrote_first_is_not_written_twice() -> None:
    store = InMemoryStore()
    store.save_incident(domain_record("domain-1", ["node-a", "node-b"]))
    joining = node_incident("incident-c", "node-c")
    store.save_incident(joining)
    view = SiblingReplicaStore(store)

    def sibling_join(current: FaultIncident) -> FaultIncident:
        return current.model_copy(
            update={
                "node_ids": ["node-a", "node-b", "node-c"],
                "reasons": [*current.reasons, "node-c joined (sibling replica)"],
            }
        )

    view.before_save("domain-1", sibling_join)
    look, sent = correlator(view)

    domain, notification_id = look.observe_finding(finding("node-c", T0), joining)

    assert notification_id is None and sent == []
    assert view.sibling_writes == 1
    stored = store.get_incident("domain-1")
    assert stored.node_ids == ["node-a", "node-b", "node-c"]
    assert [item for item in stored.reasons if "joined" in item] == [
        "node-c joined (sibling replica)"
    ], "the look re-read the record and found node-c already a member"
    assert domain is not None and domain.incident_id == "domain-1"
    joined = store.get_incident("incident-c").reasons[-1]
    assert joined.startswith("common failure domain domain-1:"), joined


def _closable(store: InMemoryStore) -> SiblingReplicaStore:
    store.save_incident(domain_record("domain-1", ["node-a", "node-b"]))
    reading(store, "node-a", 0, T0)
    reading(store, "node-b", 0, T0)
    return SiblingReplicaStore(store)


def test_a_close_another_replica_already_wrote_is_left_to_it(caplog) -> None:
    store = InMemoryStore()
    view = _closable(store)
    view.before_save(
        "domain-1",
        lambda current: current.model_copy(
            update={
                "state": IncidentState.RECOVERED,
                "reasons": [*current.reasons, "cleared by a sibling replica"],
            }
        ),
    )
    look, _sent = correlator(view)

    with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
        look.observe_batch(batch("node-a", 0, T0), [])

    stored = store.get_incident("domain-1")
    assert stored.state is IncidentState.RECOVERED
    assert stored.reasons[-1] == "cleared by a sibling replica"
    assert "-> RECOVERED" not in caplog.text, "this look wrote and logged nothing"


def test_a_record_that_keeps_moving_is_left_as_is(caplog) -> None:
    store = InMemoryStore()
    view = _closable(store)
    view.before_save(
        "domain-1",
        lambda current: current.model_copy(
            update={"reasons": [*current.reasons, "touched by a sibling replica"]}
        ),
        times=3,
    )
    look, _sent = correlator(view)

    with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
        look.observe_batch(batch("node-a", 0, T0), [])

    assert view.sibling_writes == 3, "every attempt lost to a sibling write"
    stored = store.get_incident("domain-1")
    assert stored.state is IncidentState.DETECTED, "left as is"
    assert (
        "incident domain-1 kept moving while the shared-storage failure domain was "
        "recorded on it; left as is"
    ) in caplog.text


def test_a_record_gone_after_a_lost_write_is_not_resurrected(caplog) -> None:
    store = InMemoryStore()
    view = _closable(store)
    view.before_save(
        "domain-1",
        lambda current: current.model_copy(
            update={"reasons": [*current.reasons, "archived by retention"]}
        ),
        then_vanish=True,
    )
    look, _sent = correlator(view)

    with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
        look.observe_batch(batch("node-a", 0, T0), [])

    assert store.get_incident("domain-1").reasons[-1] == "archived by retention"
    with pytest.raises(NotFoundError):
        view.get_incident("domain-1")
    assert "kept moving" not in caplog.text and "-> RECOVERED" not in caplog.text


def test_a_member_whose_incident_is_not_stored_joins_without_a_back_reference() -> None:
    store = InMemoryStore()
    store.save_incident(domain_record("domain-1", ["node-a", "node-b"]))
    unsaved = node_incident("incident-c", "node-c")
    look, sent = correlator(store)

    domain, _ = look.observe_finding(finding("node-c", T0), unsaved)

    assert domain is not None and domain.node_ids == ["node-a", "node-b", "node-c"]
    member = domain.reasons[-1]
    assert member.startswith("member node-c: incident incident-c ("), member
    assert "that incident's handling" not in member, member
    with pytest.raises(NotFoundError):
        store.get_incident("incident-c")
    assert sent == [], "a join mails nothing"


@pytest.mark.parametrize(
    ("effective", "official", "handling", "member_note", "mail_note"),
    [
        (None, None, "", "its own", "its own handling"),
        (
            RecoveryAction.NO_ACTION,
            "REBOOT_NODE",
            "REBOOT_NODE ",
            "REBOOT_NODE",
            "REBOOT_NODE",
        ),
    ],
    ids=["no-action-at-all", "official-action-only"],
)
def test_a_member_on_a_record_without_an_effective_action_keeps_its_own_handling(
    effective, official, handling, member_note, mail_note
) -> None:
    store = InMemoryStore()
    peer = node_incident(
        "incident-a", "node-a", effective_action=effective, official_action=official
    )
    own = node_incident("incident-b", "node-b")
    store.save_incident(peer)
    store.save_incident(own)
    reading(store, "node-a", 1, T0)
    look, sent = correlator(store)

    domain, _ = look.observe_finding(finding("node-b", T0 + timedelta(seconds=15)), own)

    assert domain is not None, "node-a's fresh failure is a peer"
    [line] = [item for item in domain.reasons if item.startswith("member node-a:")]
    assert line.endswith(f"; that incident's handling: {member_note})"), line
    back = store.get_incident("incident-a").reasons[-1]
    assert back.endswith(
        f"this incident's {handling}handling of this node is unchanged"
    ), back
    [mail] = [store.get_notification(item) for item in sent]
    assert f"  node-a: incident-a (mount {MOUNT}; {mail_note})\n" in mail.body_text
    assert "a node listed with an action keeps that incident's own recovery" in (
        mail.body_text
    )


def test_a_record_only_fold_names_its_node_by_the_marker() -> None:
    """node-b's activation was folded into an attempt's recovery record that
    names another node; only its marker, re-pointed at that record, says so.
    The look after the commit finds the record through it."""

    store = InMemoryStore()
    folded = node_incident(
        "incident-attempt",
        "node-x",
        event_id="event-xid-13-attempt-a",
        event_type=NODE_HEALTH_GROUP_EVENT_TYPE,
        state=IncidentState.ACTION_PENDING,
        effective_action=RecoveryAction.RESTART_WORKLOAD,
        workflow_request_id="workflow-attempt",
    )
    store.save_incident(folded)
    event_b = activation_event_id("node-b-1")
    store.add_marker(routed_marker(event_b, folded.incident_id, "node-b", T0))
    store.save_incident(node_incident("incident-a", "node-a"))
    reading(store, "node-a", 1, T0)
    reading(store, "node-b", 1, T0 + timedelta(seconds=15))
    look, sent = correlator(store)

    notifications = look.observe_committed(
        batch("node-b", 1, T0 + timedelta(seconds=15)),
        [finding("node-b", T0 + timedelta(seconds=15))],
    )

    assert notifications == sent and len(sent) == 1, notifications
    [domain] = domain_records(store)
    assert domain.node_ids == ["node-a", "node-b"]
    assert any(
        item.startswith("member node-b: incident incident-attempt (")
        and item.endswith("; that incident's handling: RESTART_WORKLOAD)")
        for item in domain.reasons
    ), domain.reasons
    back = store.get_incident("incident-attempt").reasons[-1]
    assert back.endswith(
        "this incident's RESTART_WORKLOAD handling of this node is unchanged"
    ), back


def test_a_peer_is_found_only_through_a_record_holding_its_activation() -> None:
    """node-b's own record was relabelled by grouping and has no marker: its
    event id still names the activation. node-c fails too, but its only open
    record is an XID one without a workflow, which holds no activation."""

    store = InMemoryStore()
    relabelled = node_incident(
        "incident-b-group",
        "node-b",
        event_type=NODE_HEALTH_GROUP_EVENT_TYPE,
        workflow_request_id="workflow-b",
    )
    unrelated = node_incident(
        "incident-xid-c",
        "node-c",
        event_id="event-xid-79-node-c",
        event_type="XID",
        state=IncidentState.ACTION_PENDING,
        effective_action=RecoveryAction.REBOOT_NODE,
    )
    own = node_incident("incident-a", "node-a")
    for incident in (relabelled, unrelated, own):
        store.save_incident(incident)
    reading(store, "node-b", 1, T0)
    reading(store, "node-c", 1, T0)
    look, _sent = correlator(store)

    domain, _ = look.observe_finding(finding("node-a", T0 + timedelta(seconds=10)), own)

    assert domain is not None and domain.node_ids == ["node-a", "node-b"], domain
    assert any(
        item.startswith("member node-b: incident incident-b-group (")
        for item in domain.reasons
    ), domain.reasons
    assert store.get_incident("incident-xid-c") == unrelated

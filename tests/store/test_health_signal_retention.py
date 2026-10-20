"""Health-signal state rows are retained only while they can still matter.

Batch 145. ``health_signal_state`` and ``efa_traffic_state`` rows were never
removed: every attempt that read low left its LOW_CPU/LOW_GPU windows on each
node it ran on, training-health signals stayed per attempt and rank, and the
rows of nodes that left the cluster stayed for good. The sweep
(``store/shared/signal_retention.py``) removes a row only once nothing wrote it
for the grace period and its attempt ended, its node provably departed, or --
for anything else -- nothing wrote it for the retention period. The PostgreSQL
half, including a concurrent claim, is
``test_postgres_health_signal_retention.py``.
"""

from __future__ import annotations

import sqlite3
from datetime import timedelta
from functools import partial
from types import SimpleNamespace

import pytest

from gpu_fault.efa_traffic_state import (
    efa_traffic_state_key,
    legacy_efa_traffic_state_key,
)
from gpu_fault.host_health import NodeHealthPolicy
from gpu_fault.models import (
    EfaTrafficSignal,
    HealthSignalState,
    RecoveryAction,
    Severity,
)
from gpu_fault.store import SqliteStore
from gpu_fault.store.shared.health_signals import (
    ATTEMPT_SCOPED_RULE_IDS,
    finding_health_signal_key,
    signal_key_attempt,
    training_signal_key,
)
from gpu_fault.store.shared.signal_retention import (
    ENDED,
    LIVE,
    SIGNAL_RETENTION_PAGES_PER_CALL,
    UNKNOWN,
    RetentionEvidence,
    SignalRow,
    retention_reason,
)
from gpu_fault.training_health import training_health_signal_key
from gpu_fault.watcher import WorkloadPhase
from tests._builders import attempt_observation, build_store, node_health_finding
from tests.store._health_signal_retention_cases import (
    ANCIENT,
    CLUSTER,
    FRESH,
    GPU_GATE,
    GRACE,
    LINK,
    MOUNT,
    NOW,
    OLD,
    RETENTION,
    claim,
    depart,
    efa,
    efa_rows,
    health_rows,
    latch,
    observe,
    seed_efa,
    seed_site,
    sweep,
    watch,
    window,
)


@pytest.fixture(params=["memory", "sqlite"])
def store(request, tmp_path):
    if request.param == "memory":
        yield build_store()
        return
    instance = SqliteStore(str(tmp_path / "health-signal-retention.db"))
    try:
        yield instance
    finally:
        instance.close()


# ------------------------------------------------------------- the key shapes


def test_attempt_scoped_keys_are_recognised_by_their_shape() -> None:
    assert signal_key_attempt(window(GPU_GATE, "job-a", "attempt-1")) == (
        "job-a",
        "attempt-1",
    )
    assert signal_key_attempt(training_signal_key(CLUSTER, "attempt-1", 7, "hang")) == (
        None,
        "attempt-1",
    )
    # The unscoped gate, a node rule, a mount-point device and an RDMA port:
    # nothing to do with an attempt, whatever the segment count.
    for key in (
        GPU_GATE,
        LINK,
        MOUNT,
        f"{CLUSTER}/node-a/rdma_errors_delta/mlx5_0/1",
        f"{CLUSTER}/node-a/memory_available_percent/node/LOW_MEMORY_AVAILABLE",
    ):
        assert signal_key_attempt(key) is None, key


def test_the_attempt_scoped_rules_are_the_policys_active_workload_rules() -> None:
    gated = {
        rule[0] for rule in NodeHealthPolicy.SUSTAINED_METRIC_RULES.values() if rule[6]
    }
    assert gated == ATTEMPT_SCOPED_RULE_IDS


def test_the_keys_the_deliverers_latch_parse_back_to_their_attempt() -> None:
    low = node_health_finding(
        "finding-low",
        "event-low",
        observed_at=NOW,
        category="GPU",
        severity=Severity.WARNING,
        reason="low",
        recommended_action=RecoveryAction.RUN_DIAGNOSTICS,
        metric_name="host_gpu_utilization_percent",
        value=0.0,
        device="GPU-1",
        job_id="job-a",
        attempt_id="attempt-1",
        diagnostic_parameters={"signal": "LOW_GPU_UTILIZATION"},
    )
    training = node_health_finding(
        "finding-hang",
        "event-hang",
        observed_at=NOW,
        category="TRAINING",
        severity=Severity.CRITICAL,
        reason="hang",
        recommended_action=RecoveryAction.RUN_DIAGNOSTICS,
        metric_name="training_hang",
        evidence_ref="training-progress://attempt-1/rank-3",
    )

    assert signal_key_attempt(finding_health_signal_key(low)) == ("job-a", "attempt-1")
    training_key = training_health_signal_key(training)
    assert training_key == training_signal_key(CLUSTER, "attempt-1", 3, "hang")
    assert signal_key_attempt(training_key or "") == (None, "attempt-1")


# ------------------------------------------------------------ the eligibility


def test_one_site_sweeps_to_exactly_the_rows_that_no_longer_matter(store) -> None:
    kept, removed = seed_site(store)
    efa_kept, efa_removed = seed_efa(store)

    counts = sweep(store)

    assert health_rows(store, kept) == set(kept), kept
    assert health_rows(store, removed) == set(), removed
    assert efa_rows(store, efa_kept) == efa_kept
    assert efa_rows(store, efa_removed) == set()
    assert counts == {
        "health_signal_state": len(removed),
        "efa_traffic_state": len(efa_removed),
    }
    # Nothing else is due: a second pass is a no-op.
    assert sweep(store) == {"health_signal_state": 0, "efa_traffic_state": 0}


def test_a_live_attempts_window_stays_however_long_it_was_not_written(store) -> None:
    watch(store)
    observe(store, "job-a", "attempt-1", WorkloadPhase.RUNNING)
    claim(store, window(GPU_GATE, "job-a", "attempt-1"), ANCIENT)
    latch(store, window(GPU_GATE, "job-a", "attempt-1"), ANCIENT)

    assert sweep(store)["health_signal_state"] == 0
    state = store.get_health_signal_state(window(GPU_GATE, "job-a", "attempt-1"))
    assert state is not None and state.notified is True, (
        "the latch of a live attempt still suppresses its next notification"
    )


def test_an_ended_attempts_window_goes_after_the_grace_and_not_before(store) -> None:
    watch(store)
    observe(store, "job-a", "attempt-1", WorkloadPhase.FAILED)
    claim(store, window(GPU_GATE, "job-a", "attempt-1"), FRESH)

    assert sweep(store)["health_signal_state"] == 0, "inside the grace"

    claim(store, window(GPU_GATE, "job-a", "attempt-2"), OLD)
    observe(store, "job-a", "attempt-2", WorkloadPhase.STOPPED)
    assert sweep(store)["health_signal_state"] == 1
    assert health_rows(store, [window(GPU_GATE, "job-a", "attempt-1")]) == {
        window(GPU_GATE, "job-a", "attempt-1")
    }


def test_silence_ends_an_attempt_only_while_a_watcher_vouches_for_the_cluster(
    store,
) -> None:
    observe(store, "job-a", "attempt-1", WorkloadPhase.RUNNING, at=OLD)
    claim(store, window(GPU_GATE, "job-a", "attempt-1"), OLD)
    claim(store, window(GPU_GATE, "job-a", "attempt-gone"), ANCIENT)

    # No watcher: a RUNNING observation two days old proves nothing, so only
    # the row nothing wrote for the retention period goes.
    assert sweep(store)["health_signal_state"] == 1
    assert health_rows(
        store,
        [
            window(GPU_GATE, "job-a", "attempt-1"),
            window(GPU_GATE, "job-a", "attempt-gone"),
        ],
    ) == {window(GPU_GATE, "job-a", "attempt-1")}

    # A watcher that completed a full pass a minute ago and did not see the
    # attempt: it has ended.
    watch(store, at=NOW - timedelta(minutes=1))
    assert sweep(store)["health_signal_state"] == 1
    assert health_rows(store, [window(GPU_GATE, "job-a", "attempt-1")]) == set()


def test_an_attempt_id_taken_over_by_another_job_ends_the_old_pair(store) -> None:
    watch(store)
    observe(store, "job-b", "attempt-1", WorkloadPhase.RUNNING)
    claim(store, window(GPU_GATE, "job-a", "attempt-1"), OLD)
    claim(store, window(GPU_GATE, "job-b", "attempt-1"), OLD)

    assert sweep(store)["health_signal_state"] == 1
    assert health_rows(
        store,
        [
            window(GPU_GATE, "job-a", "attempt-1"),
            window(GPU_GATE, "job-b", "attempt-1"),
        ],
    ) == {window(GPU_GATE, "job-b", "attempt-1")}


def test_rows_of_a_proven_departed_node_go_once_past_the_grace(store) -> None:
    node_b = f"{CLUSTER}/node-b/network_link_down/eth0"
    claim(store, node_b, OLD)
    latch(store, node_b, OLD)
    depart(store, "node-b", confirmed_at=FRESH)
    rejoined = f"{CLUSTER}/node-r/network_link_down/eth0"
    # The ledger line was last confirmed three days ago; a row written after
    # that belongs to a node that came back under the same name.
    depart(store, "node-r", confirmed_at=NOW - timedelta(days=3))
    claim(store, rejoined, OLD)
    unproven = f"{CLUSTER}/node-p/network_link_down/eth0"
    depart(store, "node-p", confirmed_at=FRESH, proven=False)
    claim(store, unproven, OLD)
    recent = f"{CLUSTER}/node-b/xid_errors/GPU-1"
    claim(store, recent, FRESH)

    assert sweep(store)["health_signal_state"] == 1
    assert health_rows(store, [node_b, rejoined, unproven, recent]) == {
        rejoined,
        unproven,
        recent,
    }


def test_the_unscoped_gate_of_a_live_node_stays_until_the_retention(store) -> None:
    claim(store, GPU_GATE, OLD, active=True)
    claim(store, GPU_GATE, OLD + timedelta(seconds=30), active=False)

    assert sweep(store)["health_signal_state"] == 0
    claim(store, LINK, ANCIENT)
    assert sweep(store)["health_signal_state"] == 1
    assert health_rows(store, [GPU_GATE, LINK]) == {GPU_GATE}


def test_a_window_written_after_it_was_judged_is_kept(store, monkeypatch) -> None:
    """The delete applies only to the row exactly as judged."""

    watch(store)
    observe(store, "job-a", "attempt-1", WorkloadPhase.SUCCEEDED)
    key = window(GPU_GATE, "job-a", "attempt-1")
    claim(store, key, OLD)
    read_page = getattr(store, "_signal_state_page")

    def page_then_claim(kind, **arguments):
        page = read_page(kind, **arguments)
        if kind == "health_signal_state":
            # A replica claims the key between the page read and the delete.
            claim(store, key, NOW - timedelta(seconds=5), active=False)
        return page

    monkeypatch.setattr(store, "_signal_state_page", page_then_claim)

    assert sweep(store)["health_signal_state"] == 0
    state = store.get_health_signal_state(key)
    assert state is not None and state.active is False


def test_a_deleted_window_comes_back_as_a_new_episode(store) -> None:
    """No resurrection: the next claim starts from its own sample."""

    watch(store)
    observe(store, "job-a", "attempt-1", WorkloadPhase.SUCCEEDED)
    key = window(GPU_GATE, "job-a", "attempt-1")
    claim(store, key, OLD)
    latch(store, key, OLD)
    assert sweep(store)["health_signal_state"] == 1

    assert claim(store, key, NOW) is True, "a fresh window owes its notification"
    state = store.get_health_signal_state(key)
    assert state is not None
    assert (state.active_since, state.notified) == (NOW, False)


def test_a_superseded_efa_row_goes_so_a_rollback_cannot_resume_its_streak(
    store,
) -> None:
    """A per-attempt EFA row a release before batch 145 left (an older writer
    re-created it mid-roll): once past the grace it goes, and the older
    release then starts the attempt at WARMUP instead of resuming the stale
    zero streak into HUNG_SUSPECTED."""

    legacy = legacy_efa_traffic_state_key(
        efa_traffic_state_key(CLUSTER, "node-a"), "job-a", "attempt-1"
    )
    for at, bps in ((OLD, 1000.0), (OLD + timedelta(seconds=130), 1000.0)):
        efa(store, "node-a", at, state_key=legacy, bps=bps)
    zero, _ = efa(
        store, "node-a", OLD + timedelta(seconds=150), state_key=legacy, bps=0
    )
    assert zero.zero_since == OLD + timedelta(seconds=150)

    assert sweep(store)["efa_traffic_state"] == 1

    state, emitted = efa(store, "node-a", NOW, state_key=legacy, bps=0)
    assert (state.signal, emitted) == (EfaTrafficSignal.WARMUP, False)


# ----------------------------------------------------------------- the bounds


def test_the_sweep_is_bounded_per_call_and_resumes_where_it_stopped(store) -> None:
    watch(store)
    for index in range(7):
        observe(store, "job-a", f"attempt-{index}", WorkloadPhase.SUCCEEDED)
        claim(store, window(GPU_GATE, "job-a", f"attempt-{index}"), OLD)
        claim(store, f"{CLUSTER}/node-{index}/network_link_down/eth0", OLD)

    removed = [sweep(store, limit=3)["health_signal_state"] for _ in range(4)]

    assert removed == [3, 3, 1, 0]
    assert health_rows(
        store, [f"{CLUSTER}/node-{index}/network_link_down/eth0" for index in range(7)]
    ) == {f"{CLUSTER}/node-{index}/network_link_down/eth0" for index in range(7)}


def test_one_call_scans_a_bounded_number_of_pages(store) -> None:
    """A key space of rows that all stay is walked over several calls."""

    pages = SIGNAL_RETENTION_PAGES_PER_CALL
    for index in range(pages * 2 + 1):
        claim(store, f"{CLUSTER}/node-{index:03d}/network_link_down/eth0", OLD)
    ended = f"{CLUSTER}/node-999/network_link_down/eth0"
    claim(store, ended, ANCIENT)

    # One key per page: the first call reaches only the first ``pages`` keys,
    # the third reaches the last one.
    assert sweep(store, limit=1)["health_signal_state"] == 0
    assert sweep(store, limit=1)["health_signal_state"] == 0
    assert sweep(store, limit=1)["health_signal_state"] == 1
    assert health_rows(store, [ended]) == set()


@pytest.mark.parametrize(
    ("grace", "retention", "limit"),
    [
        (timedelta(0), RETENTION, 10),
        (RETENTION + timedelta(seconds=1), RETENTION, 10),
        (GRACE, RETENTION, 0),
    ],
)
def test_the_sweep_refuses_windows_that_could_remove_open_ones(
    store, grace, retention, limit
) -> None:
    with pytest.raises(ValueError, match="health-signal retention"):
        store.cleanup_health_signal_states(
            now=NOW, grace=grace, retention=retention, limit=limit
        )


def test_a_naive_clock_is_read_as_utc(store) -> None:
    """Rows a node stamped without an offset are judged, not crashed on."""

    naive = OLD.replace(tzinfo=None)
    store.claim_health_signal_transitions([(LINK, True, naive, 0.0)])
    assert sweep(store)["health_signal_state"] == 0
    store.claim_health_signal_transitions(
        [(f"{CLUSTER}/node-a/old/node", True, ANCIENT.replace(tzinfo=None), 0.0)]
    )
    assert sweep(store)["health_signal_state"] == 1


# ------------------------------------------------------- the rule, by itself


def _row(key: str, at) -> SignalRow:
    state = HealthSignalState(signal_key=key, active=True, observed_at=at, clock_at=at)
    return SignalRow(key, state, state)


def test_a_row_written_inside_the_grace_is_never_a_candidate() -> None:
    """Whatever a backend's page returned: the rule re-judges the decoded row."""

    evidence = RetentionEvidence(SimpleNamespace(), now=NOW, grace=GRACE)
    reason = partial(
        retention_reason,
        "health_signal_state",
        evidence=evidence,
        now=NOW,
        grace=GRACE,
        retention=RETENTION,
    )

    assert reason(_row(LINK, FRESH)) is None
    assert reason(_row(LINK, ANCIENT)) == "not_updated"


def test_a_store_without_attempt_or_node_evidence_keeps_attempt_rows() -> None:
    """A narrow store (no observation, heartbeat or ledger reader) proves no
    attempt ended and no node departed: only the retention applies."""

    evidence = RetentionEvidence(SimpleNamespace(), now=NOW, grace=GRACE)
    reason = partial(
        retention_reason,
        "health_signal_state",
        evidence=evidence,
        now=NOW,
        grace=GRACE,
        retention=RETENTION,
    )

    assert evidence.attempt_status(CLUSTER, "job-a", "attempt-1") == UNKNOWN
    assert evidence.departed_at(CLUSTER, "node-a") is None
    assert reason(_row(window(GPU_GATE, "job-a", "attempt-1"), OLD)) is None
    assert reason(_row(window(GPU_GATE, "job-a", "attempt-1"), ANCIENT)) == (
        "not_updated"
    )


def test_observations_without_a_heartbeat_reader_cannot_end_an_attempt() -> None:
    observations = [
        attempt_observation("job-a", "attempt-old", OLD),
        attempt_observation("job-a", "attempt-live", FRESH),
    ]
    store = SimpleNamespace(list_attempt_observations=lambda cluster_id: observations)
    evidence = RetentionEvidence(store, now=NOW, grace=GRACE)

    assert evidence.attempt_status(CLUSTER, "job-a", "attempt-live") == LIVE
    assert evidence.attempt_status(CLUSTER, None, "attempt-live") == LIVE
    assert evidence.attempt_status(CLUSTER, "job-a", "attempt-old") == UNKNOWN
    assert evidence.attempt_status(CLUSTER, "job-b", "attempt-live") == ENDED


# ------------------------------------------------------------- the failures


def test_a_row_this_release_cannot_decode_is_kept_and_walked_past(tmp_path) -> None:
    """A row a newer release wrote (before a rollback) fails this release's
    model: it is neither judged nor deleted, and the rows after it still are."""

    path = tmp_path / "undecodable.db"
    store = SqliteStore(str(path))
    try:
        claim(store, LINK, ANCIENT)
        connection = sqlite3.connect(str(path))
        with connection:
            connection.execute(
                "INSERT INTO objects(kind, key, payload) VALUES (?, ?, ?)",
                (
                    "health_signal_state",
                    f"{CLUSTER}/node-0/newer/node",
                    '{"signal_key": "x", "active": true, "from_a_newer": 1}',
                ),
            )
        connection.close()

        assert sweep(store, limit=1)["health_signal_state"] == 1
        assert health_rows(store, [LINK]) == set()
    finally:
        store.close()


def test_one_kind_failing_does_not_starve_the_other(store, monkeypatch) -> None:
    claim(store, LINK, ANCIENT)
    stale_efa, _ = efa(store, "node-d", ANCIENT)
    read_page = getattr(store, "_signal_state_page")

    def failing_health_page(kind, **arguments):
        if kind == "health_signal_state":
            raise RuntimeError("health_signal_state page failed")
        return read_page(kind, **arguments)

    monkeypatch.setattr(store, "_signal_state_page", failing_health_page)

    with pytest.raises(RuntimeError, match="health_signal_state page failed"):
        sweep(store)
    assert efa_rows(store, [stale_efa.state_key]) == set()
    assert health_rows(store, [LINK]) == {LINK}


def test_a_failing_page_does_not_send_the_next_call_back_to_the_start(
    store, monkeypatch
) -> None:
    claim(store, f"{CLUSTER}/node-0/network_link_down/eth0", OLD)  # stays
    for index in (1, 2):
        claim(store, f"{CLUSTER}/node-{index}/network_link_down/eth0", ANCIENT)
    read_page = getattr(store, "_signal_state_page")
    pages: list[str] = []

    def second_page_fails(kind, **arguments):
        if kind == "health_signal_state":
            pages.append(arguments["after_key"])
            if len(pages) == 2:
                raise RuntimeError("second page failed")
        return read_page(kind, **arguments)

    monkeypatch.setattr(store, "_signal_state_page", second_page_fails)

    with pytest.raises(RuntimeError, match="second page failed"):
        sweep(store, limit=1)
    assert sweep(store, limit=1)["health_signal_state"] == 1

    # The first call read one page (node-0, which stays) and failed on the
    # next; the second call resumed after node-0, not at the start.
    first = f"{CLUSTER}/node-0/network_link_down/eth0"
    assert pages == ["", first, first]
    assert health_rows(store, [first, f"{CLUSTER}/node-1/network_link_down/eth0"]) == {
        first
    }

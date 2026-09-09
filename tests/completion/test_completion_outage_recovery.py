"""A control-plane outage longer than the terminal retention must lose nothing.

Final review C1 (2026-09-08). Three rules used to compose into a silent loss:
``replay`` quarantined a record after ``max_replay_attempts`` even when every
failure was transient (20 passes, about ten minutes of outage); the watcher
core pruned a terminal attempt ``terminal_retention_seconds`` after it ended,
on any sibling's observation; and the controller then evicted the attempt --
specs, cached terminal, sent keys -- once its Pods were gone. The live path
that was meant to own quarantined keys had nothing left to post, ``replay``
skipped the record for ever, and ``replay(include_quarantined=True)`` had no
caller. After a one-hour outage the terminal and its failure-detected event
sat in the ConfigMap, quarantined, for ever.

The rules now: a retryable failure never quarantines by count, only a
non-retryable status does, and a retry disposition older than a day expires
(counted); an attempt whose events are still buffered is not evicted; and
``gpu-fault-completion-watcher --replay-quarantined`` is the operator lever.
"""

from __future__ import annotations

import json
from datetime import timedelta

import pytest

from gpu_fault import completion_controller
from gpu_fault.collectors import CollectorError
from gpu_fault.completion_controller import KubernetesCompletionController
from gpu_fault.completion_outbox import KubernetesCompletionOutbox
from tests.completion.test_completion_controller import (
    Clock,
    FakeCoreApi,
    FakeSink,
    pod,
)

TERMINAL = "/v1/attempts/terminal"
FAILURE = "/v1/attempts/failure-detected"
CRITICAL = {TERMINAL, FAILURE}
FAILED = "failed-a"
SIBLING = "sibling-b"


class OutageSink(FakeSink):
    """Refuses every critical event while ``error`` is set; accepts otherwise.

    ``accepted`` keeps only the POSTs that returned, which is what "delivered"
    means to the control plane; ``posts`` (inherited) keeps every attempt.
    """

    def __init__(self, error: Exception | None) -> None:
        super().__init__()
        self.error = error
        self.accepted: list[tuple[str, dict]] = []

    def post(self, path, payload):
        self.posts.append((path, payload))
        if path in CRITICAL and self.error is not None:
            raise self.error
        self.accepted.append((path, payload))
        return {"accepted": True}


def _records(core: FakeCoreApi) -> dict[str, dict]:
    return {
        item["key"]: item for item in json.loads(core.config_map_data["events.json"])
    }


def _fleet() -> list[dict]:
    """One attempt that fails on its single rank, one sibling that keeps running."""

    return [pod(0, exit_code=1, attempt_id=FAILED), pod(1, attempt_id=SIBLING)]


def _watcher(core: FakeCoreApi, sink: OutageSink, clock: Clock):
    outbox = KubernetesCompletionOutbox(core, sink)
    return (
        KubernetesCompletionController(
            core,
            outbox,
            cluster_id="hp-cluster",
            now=clock,
            terminal_retention_seconds=3600,
        ),
        outbox,
    )


def _outage_then_gc(subject, core: FakeCoreApi, clock: Clock, *, passes: int) -> None:
    """Run ``passes`` reconcile passes, then GC the failed attempt's Pods."""

    for _ in range(passes):
        clock.value += timedelta(seconds=30)
        subject.run_once()
    core.pods = [pod(1, attempt_id=SIBLING)]
    # Past ``terminal_retention_seconds``: the sibling's next observation
    # prunes the failed attempt from the watcher core.
    clock.value += timedelta(seconds=3700)
    subject.run_once()


def test_a_terminal_buffered_through_a_long_outage_is_delivered_on_recovery() -> None:
    """The reviewer's reproduction: two attempts, Pods GC'd, +3700 s, recovery."""

    clock = Clock()
    core = FakeCoreApi(_fleet())
    sink = OutageSink(OSError("control plane unavailable"))
    subject, outbox = _watcher(core, sink, clock)

    subject.run_once()
    assert set(_records(core)) == {
        f"hp-cluster/{FAILED}/failure-detected",
        f"hp-cluster/{FAILED}/terminal",
    }, f"both critical events must be written ahead: {_records(core)}"
    # 25 passes is more than the 20 replay attempts that used to quarantine.
    _outage_then_gc(subject, core, clock, passes=25)
    assert outbox.quarantined_depth() == 0, (
        "a transient outage must never quarantine a record by attempt count: "
        f"{_records(core)}"
    )

    sink.error = None
    clock.value += timedelta(seconds=30)
    subject.run_once()

    # A set, not a list: the replay at the top of the pass and the live path
    # later in the same pass may both deliver (the pre-existing, deferred
    # same-pass double post); what this case pins is that nothing is lost.
    delivered = {path for path, payload in sink.accepted if path in CRITICAL}
    assert delivered == {FAILURE, TERMINAL}, (
        "the control plane came back, so the buffered failure-detected event "
        f"and terminal must both be delivered; accepted={sorted(delivered)}"
    )
    assert outbox.depth() == 0, (
        f"a delivered record must leave the write-ahead log: {_records(core)}"
    )


def test_a_transient_outage_never_quarantines_and_the_attempt_stays_held() -> None:
    """While the WAL holds an attempt's events, the controller keeps the attempt."""

    clock = Clock()
    core = FakeCoreApi(_fleet())
    sink = OutageSink(OSError("control plane unavailable"))
    subject, _outbox = _watcher(core, sink, clock)

    subject.run_once()
    _outage_then_gc(subject, core, clock, passes=25)

    assert subject.evicted_attempts_total == 0, (
        "an attempt whose critical events are still buffered must not be "
        "evicted -- its live path is the only one left for a quarantined key"
    )


def test_a_rejected_attempt_is_evicted_only_after_its_record_is_delivered() -> None:
    """Non-retryable status: quarantined, held, then evicted once delivered."""

    clock = Clock()
    core = FakeCoreApi(_fleet())
    sink = OutageSink(
        CollectorError("collector event rejected (422): unknown", status_code=422)
    )
    subject, outbox = _watcher(core, sink, clock)

    subject.run_once()
    _outage_then_gc(subject, core, clock, passes=2)

    assert outbox.quarantined_depth() == 2, (
        f"a 4xx verdict must still quarantine the records: {_records(core)}"
    )
    assert subject.evicted_attempts_total == 0, (
        "a quarantined record is delivered only by the live path, so the "
        "attempt that owns it must not be evicted while it is buffered"
    )

    # The operator fixed the cause; the live path delivers, then eviction runs.
    sink.error = None
    clock.value += timedelta(seconds=30)
    subject.run_once()
    clock.value += timedelta(seconds=30)
    subject.run_once()

    delivered = sorted(path for path, payload in sink.accepted if path in CRITICAL)
    assert delivered == [FAILURE, TERMINAL], (
        f"the held live path must deliver both events: accepted={delivered}"
    )
    assert outbox.depth() == 0, f"the WAL must be clean: {_records(core)}"
    assert subject.evicted_attempts_total == 1, (
        "once the WAL no longer names the attempt, retention eviction must "
        f"proceed; evicted={subject.evicted_attempts_total}"
    )


def _expiring_watcher(core: FakeCoreApi, sink: OutageSink, clock: Clock):
    """A watcher whose outbox ages records on the test clock, not the wall."""

    outbox = KubernetesCompletionOutbox(core, sink, now=lambda: clock.value.timestamp())
    controller = KubernetesCompletionController(
        core,
        outbox,
        cluster_id="hp-cluster",
        now=clock,
        terminal_retention_seconds=3600,
    )
    return controller, outbox


def test_an_expired_record_is_removed_and_the_held_live_path_re_buffers_it() -> None:
    """R5 (a): a day-old retry disposition leaves the WAL; the hold does not.

    While the watcher still holds the attempt its live path re-posts the event
    from the cache every pass, so the slot the expired record gave up is taken
    again -- by a *live* record of the same event, which is what the slot is
    for. The expiry is counted and the old disposition is gone.
    """

    clock = Clock()
    core = FakeCoreApi(_fleet())
    sink = OutageSink(OSError("control plane unavailable"))
    subject, outbox = _expiring_watcher(core, sink, clock)

    subject.run_once()
    _outage_then_gc(subject, core, clock, passes=2)
    assert outbox.depth() == 2, _records(core)
    expired_at = clock.value + timedelta(hours=24)

    clock.value = expired_at
    subject.run_once()  # the replay at the top of the pass expires both records

    assert outbox.expired_total == 2, (
        f"both day-old records must expire: expired_total={outbox.expired_total}"
    )
    assert outbox.quarantined_depth() == 0, (
        f"an expired record is removed, never kept quarantined: {_records(core)}"
    )
    records = _records(core)
    assert set(records) == {
        f"hp-cluster/{FAILED}/failure-detected",
        f"hp-cluster/{FAILED}/terminal",
    }, f"the held live path re-buffers the still-owed events: {records}"
    for record in records.values():
        assert record["buffered_at"] >= expired_at.timestamp() and (
            record["quarantined"] is False
        ), f"the re-buffered record must be a fresh live disposition: {record}"
    assert subject.evicted_attempts_total == 0, (
        "the WAL names the attempt again, so it stays held"
    )


def test_an_expired_record_leaves_a_restarted_watcher_nothing_to_hold() -> None:
    """After a restart nothing re-posts the event: the record expires and goes.

    Before R5 it stayed quarantined for ever, holding one of the 256 slots and
    the attempt id with it; only the one-shot could have delivered it. Now the
    ERROR names key and digest, the record is removed and the log is clean.
    """

    clock = Clock()
    core = FakeCoreApi(_fleet())
    sink = OutageSink(OSError("control plane unavailable"))
    first, _first_outbox = _expiring_watcher(core, sink, clock)
    first.run_once()
    _outage_then_gc(first, core, clock, passes=2)
    assert len(_records(core)) == 2, _records(core)

    # The watcher restarts with the sibling still running and the failed
    # attempt's Pods long gone: nothing in the new process re-posts them.
    restarted, outbox = _expiring_watcher(core, sink, clock)
    restarted.run_once()
    assert outbox.buffered_attempt_ids == frozenset({FAILED}), (
        f"the restored process must see the WAL still names the attempt: "
        f"{outbox.buffered_attempt_ids}"
    )
    critical_before = [path for path, _ in sink.posts if path in CRITICAL]

    clock.value += timedelta(hours=24)
    restarted.run_once()

    assert outbox.expired_total == 2, (
        f"both day-old records must expire: expired_total={outbox.expired_total}"
    )
    assert _records(core) == {}, f"expired records must be removed: {_records(core)}"
    assert outbox.buffered_attempt_ids == frozenset(), (
        f"nothing holds the attempt any more: {outbox.buffered_attempt_ids}"
    )
    assert (outbox.last_depth, outbox.last_quarantined_depth) == (0, 0)
    replayed = [path for path, _ in sink.posts if path in CRITICAL]
    assert len(replayed) == len(critical_before) + 2, (
        "the expiring pass replays each record once more before giving up; "
        f"nothing else may re-post them after the restart: {replayed}"
    )


def _quarantined_record(core: FakeCoreApi, clock: Clock) -> dict:
    """Leave one quarantined terminal in ``core``'s write-ahead ConfigMap."""

    rejecting = OutageSink(
        CollectorError("collector event rejected (422): unknown", status_code=422)
    )
    # Stamped on the test clock so a later test can age the record past 24 h.
    outbox = KubernetesCompletionOutbox(
        core, rejecting, now=lambda: clock.value.timestamp()
    )
    payload = {"cluster_id": "hp-cluster", "attempt_id": FAILED}
    with pytest.raises(CollectorError):
        outbox.post(TERMINAL, payload)
    outbox.replay()
    assert outbox.quarantined_depth() == 1, _records(core)
    return payload


def test_the_replay_quarantined_mode_delivers_once_and_exits(monkeypatch) -> None:
    """``--replay-quarantined`` is the one-shot lever the docs promise."""

    clock = Clock()
    core = FakeCoreApi([])
    payload = _quarantined_record(core, clock)
    accepting = OutageSink(None)
    controller = KubernetesCompletionController(
        core,
        KubernetesCompletionOutbox(core, accepting),
        cluster_id="hp-cluster",
        now=clock,
    )
    monkeypatch.setattr(
        controller,
        "run",
        lambda **_kwargs: pytest.fail("the one-shot mode must not start the loop"),
    )
    monkeypatch.setattr(
        completion_controller, "controller_from_environment", lambda: controller
    )
    monkeypatch.setattr(
        completion_controller, "validate_gpu_fault_environment", lambda **_kwargs: None
    )

    with pytest.raises(SystemExit) as exit_info:
        completion_controller.main(["--replay-quarantined"])

    assert exit_info.value.code == 0, (
        f"a pass that left nothing quarantined exits 0, got {exit_info.value.code}"
    )
    assert accepting.accepted == [(TERMINAL, payload)], (
        f"the quarantined terminal must be re-sent once: {accepting.posts}"
    )
    assert json.loads(core.config_map_data["events.json"]) == [], (
        f"a delivered record must be cleared: {core.config_map_data['events.json']}"
    )


def test_the_replay_quarantined_mode_exits_nonzero_when_records_remain(
    monkeypatch,
) -> None:
    clock = Clock()
    core = FakeCoreApi([])
    _quarantined_record(core, clock)
    still_rejecting = OutageSink(
        CollectorError("collector event rejected (422): unknown", status_code=422)
    )
    controller = KubernetesCompletionController(
        core,
        KubernetesCompletionOutbox(core, still_rejecting),
        cluster_id="hp-cluster",
        now=clock,
    )
    monkeypatch.setattr(
        completion_controller, "controller_from_environment", lambda: controller
    )
    monkeypatch.setattr(
        completion_controller, "validate_gpu_fault_environment", lambda **_kwargs: None
    )

    with pytest.raises(SystemExit) as exit_info:
        completion_controller.main(["--replay-quarantined"])

    assert exit_info.value.code == 1, (
        "a record the control plane still rejects stays quarantined and the "
        f"one-shot must say so in its exit status, got {exit_info.value.code}"
    )


def test_the_one_shot_keeps_a_rejected_record_through_a_transient_failure(
    monkeypatch,
) -> None:
    """A day-old ``rejected`` record must not expire under the one-shot.

    Expiry is the bound on a *retry* disposition. A rejected record is the
    control plane's verdict and the operator's evidence, and it is usually
    older than a day by the time the one-shot runs; a 5xx or a socket error
    during that run is a transient failure of the run, not a reason to
    delete the record, count it as expired and exit 0 as if nothing were
    left.
    """

    clock = Clock()
    core = FakeCoreApi([])
    _quarantined_record(core, clock)
    clock.value += timedelta(hours=25)
    down = OutageSink(OSError("control plane unavailable"))
    outbox = KubernetesCompletionOutbox(core, down, now=lambda: clock.value.timestamp())
    controller = KubernetesCompletionController(
        core, outbox, cluster_id="hp-cluster", now=clock
    )
    monkeypatch.setattr(
        completion_controller, "controller_from_environment", lambda: controller
    )
    monkeypatch.setattr(
        completion_controller, "validate_gpu_fault_environment", lambda **_kwargs: None
    )

    with pytest.raises(SystemExit) as exit_info:
        completion_controller.main(["--replay-quarantined"])

    record = _records(core).get(f"hp-cluster/{FAILED}/terminal")
    assert record is not None and record["quarantined"] is True, (
        f"a transient failure must leave the rejected record in place: {_records(core)}"
    )
    assert record["quarantine_reason"] == "rejected", record
    assert outbox.expired_total == 0, (
        f"a rejected record never expires: expired_total={outbox.expired_total}"
    )
    assert exit_info.value.code == 1, (
        "the record is still quarantined, so the one-shot must exit 1, got "
        f"{exit_info.value.code}"
    )


def test_without_the_flag_main_runs_the_loop(monkeypatch) -> None:
    started: list[str] = []
    controller = KubernetesCompletionController(
        FakeCoreApi([]), FakeSink(), cluster_id="hp-cluster"
    )
    monkeypatch.setattr(controller, "run", lambda **_kwargs: started.append("run"))
    monkeypatch.setattr(
        completion_controller, "controller_from_environment", lambda: controller
    )
    monkeypatch.setattr(
        completion_controller, "validate_gpu_fault_environment", lambda **_kwargs: None
    )

    completion_controller.main([])

    assert started == ["run"], f"the default mode is the watch loop: {started}"

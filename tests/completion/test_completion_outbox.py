from __future__ import annotations

import json
import logging
from types import SimpleNamespace

import pytest

from gpu_fault.completion_outbox import (
    CompletionOutboxFull,
    KubernetesCompletionOutbox,
    replay_completion_outbox,
)

OUTBOX_NAME = "gpu-fault-completion-watcher-outbox"
ACTIVE_NAME = f"{OUTBOX_NAME}-active"


class ConfigMapCore:
    """Fake CoreV1 ConfigMap API holding each object on its own.

    The watcher keeps the write-ahead log in ``<name>`` and routine attempt
    state in ``<name>-active`` (F6), each with its own resourceVersion. ``data``
    is the union of the two -- the two objects never share a key -- and
    ``version`` their sum, so a test can keep asserting on whichever document it
    cares about without knowing which object carries it.
    """

    def __init__(self) -> None:
        self.objects = {
            OUTBOX_NAME: {"events.json": "[]"},
            ACTIVE_NAME: {"active-attempts.json": "{}"},
        }
        self.versions = {name: 1 for name in self.objects}
        self.reads: list[str] = []

    @property
    def data(self):
        return {**self.objects[ACTIVE_NAME], **self.objects[OUTBOX_NAME]}

    @property
    def version(self):
        return sum(self.versions.values())

    def read_namespaced_config_map(self, name, namespace):
        assert name in self.objects, f"unexpected ConfigMap read: {name}"
        assert namespace == "gpu-fault-system"
        self.reads.append(name)
        return SimpleNamespace(
            metadata=SimpleNamespace(resource_version=str(self.versions[name])),
            data=dict(self.objects[name]),
        )

    def replace_namespaced_config_map(self, name, namespace, body):
        assert name in self.objects, f"unexpected ConfigMap write: {name}"
        assert body["metadata"]["resourceVersion"] == str(self.versions[name])
        self.versions[name] += 1
        self.objects[name] = dict(body["data"])

    def events(self, name):
        return len(json.loads(self.objects[name].get("events.json", "[]")))


class RecordingSink:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.posts = []

    def post(self, path, payload):
        self.posts.append((path, payload))
        if self.fail:
            raise OSError("control plane unavailable")
        return {"accepted": True}


def payload(attempt_id: str = "attempt-a"):
    return {"cluster_id": "cluster-a", "attempt_id": attempt_id}


def test_critical_completion_event_is_written_before_delivery() -> None:
    core = ConfigMapCore()
    sink = RecordingSink(fail=True)
    outbox = KubernetesCompletionOutbox(core, sink)

    with pytest.raises(OSError, match="unavailable"):
        outbox.post("/v1/attempts/terminal", payload())

    records = json.loads(core.data["events.json"])
    assert records[0]["key"] == "cluster-a/attempt-a/terminal"
    assert outbox.depth() == 1


def test_completion_outbox_replays_without_a_new_live_event() -> None:
    core = ConfigMapCore()
    failed = RecordingSink(fail=True)
    outbox = KubernetesCompletionOutbox(core, failed)
    with pytest.raises(OSError):
        outbox.post("/v1/attempts/failure-detected", payload())

    recovered = RecordingSink()
    replay = KubernetesCompletionOutbox(core, recovered)

    assert replay.replay() == 1
    assert recovered.posts == [("/v1/attempts/failure-detected", payload())]
    assert replay.depth() == 0


def test_successful_routine_observation_does_not_write_the_outbox() -> None:
    core = ConfigMapCore()
    sink = RecordingSink()
    outbox = KubernetesCompletionOutbox(core, sink)

    outbox.post("/v1/workload-observations", payload())

    assert sink.posts == [("/v1/workload-observations", payload())]
    assert outbox.depth() == 0


def test_failed_routine_observation_is_durable_latest_wins() -> None:
    core = ConfigMapCore()
    failed = RecordingSink(fail=True)
    outbox = KubernetesCompletionOutbox(core, failed)
    first = {**payload(), "observed_at": "2026-09-03T06:00:00Z"}
    latest = {**payload(), "observed_at": "2026-09-03T06:00:03Z"}

    with pytest.raises(OSError, match="unavailable"):
        outbox.post("/v1/workload-observations", first)
    with pytest.raises(OSError, match="unavailable"):
        outbox.post("/v1/workload-observations", latest)

    records = json.loads(core.data["events.json"])
    assert len(records) == 1
    assert records[0]["payload"] == latest
    recovered = RecordingSink()
    replay = KubernetesCompletionOutbox(core, recovered)
    assert replay.replay() == 1
    assert recovered.posts == [("/v1/workload-observations", latest)]
    assert replay.depth() == 0


def test_active_attempt_state_ignores_timestamp_only_refreshes() -> None:
    core = ConfigMapCore()
    outbox = KubernetesCompletionOutbox(core, RecordingSink())
    first = {
        **payload(),
        "workload_phase": "RUNNING",
        "observed_at": "2026-09-03T06:00:00Z",
        "containers": [{"node_id": "node-a", "terminated": False}],
    }
    refreshed = {**first, "observed_at": "2026-09-03T06:00:03Z"}

    assert outbox.save_attempt_observation(first) is True
    version = core.version
    assert outbox.save_attempt_observation(refreshed) is False
    assert core.version == version
    restored = KubernetesCompletionOutbox(
        core, RecordingSink()
    ).load_attempt_observations()
    assert restored == [first]
    outbox.remove_attempt_observation(first)
    assert json.loads(core.data["active-attempts.json"]) == {}


def test_completion_outbox_fails_closed_instead_of_dropping_oldest() -> None:
    core = ConfigMapCore()
    outbox = KubernetesCompletionOutbox(core, RecordingSink(fail=True), max_records=1)
    with pytest.raises(OSError):
        outbox.post("/v1/attempts/terminal", payload("attempt-a"))

    with pytest.raises(CompletionOutboxFull):
        outbox.post("/v1/attempts/terminal", payload("attempt-b"))

    records = json.loads(core.data["events.json"])
    assert [item["key"] for item in records] == ["cluster-a/attempt-a/terminal"]


class FakeApiException(Exception):
    """Stands in for ``kubernetes.client.rest.ApiException``."""

    def __init__(self, status: int) -> None:
        super().__init__(f"({status})\nReason: fake")
        self.status = status


class BrokenWriteCore(ConfigMapCore):
    """A ConfigMap whose ``replace`` always fails the way the API server does.

    ``409`` exercises the CAS retry (three passes, then the raise); ``500``
    fails on the first pass. Either way the write-ahead copy cannot be made.
    """

    def __init__(self, status: int) -> None:
        super().__init__()
        self.status = status
        self.replace_calls = 0

    def replace_namespaced_config_map(self, name, namespace, body):
        self.replace_calls += 1
        raise FakeApiException(self.status)


def snapshot(index: int, *, tail_bytes: int = 260_000) -> dict:
    line = f"2026-09-08T06:00:0{index}Z " + ("y" * 109) + "\n"
    tail = (line * (tail_bytes // len(line) + 1))[:tail_bytes]
    return {
        "record_id": f"workload-log/record-{index}",
        "node_id": f"node-{index}",
        "pod_uid": f"pod-{index}",
        "container_name": "trainer",
        "sha256": f"{index:064d}",
        "tail": tail,
        "tail_bytes": len(tail.encode()),
        "truncated": True,
        "archive_truncated": False,
        "s3_uri": f"s3://bucket/logs/{index}.log.gz",
    }


def failure_payload(*, ranks: int = 4, attempt_id: str = "attempt-a") -> dict:
    return {
        **payload(attempt_id),
        "reason": "rank 0 exited 1",
        "workload_log_snapshots": [snapshot(index) for index in range(ranks)],
    }


def test_failure_event_with_large_log_tails_is_delivered_in_full() -> None:
    """F1: a 4-rank failure event is ~1 MB, well over ``max_bytes``.

    The write-ahead copy keeps the pointers and at most 8 KB of tail so the
    ConfigMap write stays inside its bound; the live POST still carries every
    full tail, because the control plane is what archives them.
    """

    core = ConfigMapCore()
    sink = RecordingSink()
    outbox = KubernetesCompletionOutbox(core, sink)
    event = failure_payload()

    outbox.post("/v1/attempts/failure-detected", event)

    assert len(sink.posts) == 1, f"expected one live POST, got {sink.posts!r}"
    posted = sink.posts[0][1]["workload_log_snapshots"]
    assert [len(item["tail"].encode()) for item in posted] == [260_000] * 4, (
        "the live POST must carry the full tails, not the pointer-sized copy"
    )
    assert outbox.append_failures_total == 0, (
        "the pointer-sized record must fit, so nothing may be counted as a "
        f"write-ahead failure (got {outbox.append_failures_total})"
    )


def test_buffered_failure_record_keeps_pointers_and_drops_the_tail() -> None:
    core = ConfigMapCore()
    sink = RecordingSink(fail=True)
    outbox = KubernetesCompletionOutbox(core, sink)
    event = failure_payload()

    with pytest.raises(OSError, match="unavailable"):
        outbox.post("/v1/attempts/failure-detected", event)

    records = json.loads(core.data["events.json"])
    assert len(records) == 1, f"expected one buffered record, got {len(records)}"
    buffered = records[0]["payload"]["workload_log_snapshots"]
    assert len(buffered) == 4, f"every snapshot must be kept: {len(buffered)}"
    for index, item in enumerate(buffered):
        assert len(item["tail"].encode()) <= 8192, (
            f"snapshot {index} tail is {len(item['tail'].encode())} bytes, "
            "the write-ahead cap is 8192 per snapshot"
        )
        assert item["record_id"] == f"workload-log/record-{index}", (
            f"snapshot {index} lost its record_id pointer"
        )
        assert item["s3_uri"] == f"s3://bucket/logs/{index}.log.gz", (
            f"snapshot {index} lost its s3_uri pointer"
        )
        assert item["sha256"] == f"{index:064d}", (
            f"snapshot {index} lost its archive digest"
        )
        assert item["truncated"] is True, f"snapshot {index} lost its truncated flag"
        assert item["buffered_tail_truncated"] is True, (
            f"snapshot {index} must record that the WAL copy was trimmed"
        )
    document_bytes = len(core.data["events.json"].encode())
    assert document_bytes < 900_000, (
        f"the buffered document must fit the outbox bound, it is {document_bytes}"
    )
    assert event["workload_log_snapshots"][0]["tail_bytes"] == 260_000, (
        "the caller's event must not be mutated by the write-ahead copy"
    )


def test_pointer_sized_tail_is_cut_on_a_character_boundary() -> None:
    core = ConfigMapCore()
    outbox = KubernetesCompletionOutbox(core, RecordingSink(fail=True))
    event = {
        **payload(),
        "workload_log_snapshots": [{"record_id": "r", "tail": "é" * 9000}],
    }

    with pytest.raises(OSError):
        outbox.post("/v1/attempts/failure-detected", event)

    record = json.loads(core.data["events.json"])[0]
    tail = record["payload"]["workload_log_snapshots"][0]["tail"]
    assert len(tail.encode()) <= 8192, (
        f"the cut must be measured in encoded bytes, got {len(tail.encode())}"
    )
    assert "�" not in tail, f"the cut must not split a character: {tail[:8]!r}"


@pytest.mark.parametrize("status", [409, 500])
def test_outbox_append_failure_does_not_veto_the_live_post(status: int) -> None:
    """F1(b): the write-ahead copy is durability, not permission to deliver."""

    core = BrokenWriteCore(status)
    sink = RecordingSink()
    outbox = KubernetesCompletionOutbox(core, sink)

    result = outbox.post("/v1/attempts/failure-detected", payload())

    assert result == {"accepted": True}, f"the live POST result must win: {result!r}"
    assert sink.posts == [("/v1/attempts/failure-detected", payload())], (
        f"the live POST must still happen when the WAL write fails: {sink.posts!r}"
    )
    assert outbox.append_failures_total == 1, (
        f"the WAL failure must be counted, got {outbox.append_failures_total}"
    )


def test_append_failure_is_logged_once_per_key(
    caplog: pytest.LogCaptureFixture,
) -> None:
    core = BrokenWriteCore(500)
    outbox = KubernetesCompletionOutbox(core, RecordingSink())

    with caplog.at_level(logging.ERROR, logger="gpu_fault.completion_outbox"):
        outbox.post("/v1/attempts/failure-detected", payload())
        outbox.post("/v1/attempts/failure-detected", payload())
        outbox.post("/v1/attempts/terminal", payload())

    errors = [record for record in caplog.records if record.levelno == logging.ERROR]
    assert len(errors) == 2, (
        f"one ERROR per key, not per pass: {[r.getMessage() for r in errors]}"
    )
    assert outbox.append_failures_total == 3, (
        f"every failure must be counted, got {outbox.append_failures_total}"
    )


def test_append_failure_with_a_dead_sink_still_fails_closed() -> None:
    """Nothing was buffered and nothing was delivered: the caller must know.

    The write-ahead error is the one that is raised (an unwritable outbox is
    the actionable cause); the delivery error is chained onto it so the
    traceback still names both.
    """

    core = BrokenWriteCore(500)
    outbox = KubernetesCompletionOutbox(core, RecordingSink(fail=True))

    with pytest.raises(FakeApiException) as raised:
        outbox.post("/v1/attempts/terminal", payload())

    assert isinstance(raised.value.__cause__, OSError), (
        f"the live delivery error must be chained, got {raised.value.__cause__!r}"
    )
    assert outbox.append_failures_total == 1, (
        f"the WAL failure must be counted, got {outbox.append_failures_total}"
    )


def test_already_buffered_record_is_left_to_replay() -> None:
    """F8: the live path and ``replay`` both retried the same record."""

    core = ConfigMapCore()
    sink = RecordingSink(fail=True)
    outbox = KubernetesCompletionOutbox(core, sink)

    with pytest.raises(OSError):
        outbox.post("/v1/attempts/failure-detected", payload())
    assert len(sink.posts) == 1, f"the first pass posts live: {sink.posts!r}"

    result = outbox.post("/v1/attempts/failure-detected", payload())

    assert len(sink.posts) == 1, (
        f"a buffered record must not be posted live again: {sink.posts!r}"
    )
    assert result == {"deferred_to_replay": True}, (
        f"the caller must be told replay owns delivery, got {result!r}"
    )
    assert outbox.depth() == 1, "the record must still be buffered for replay"
    sink.fail = False
    assert outbox.replay() == 1, "replay owns the delivery of the buffered record"
    assert outbox.depth() == 0, "a replayed record must be removed"


def test_a_buffered_record_does_not_block_another_key() -> None:
    core = ConfigMapCore()
    sink = RecordingSink(fail=True)
    outbox = KubernetesCompletionOutbox(core, sink)
    with pytest.raises(OSError):
        outbox.post("/v1/attempts/failure-detected", payload("attempt-a"))
    sink.fail = False

    outbox.post("/v1/attempts/failure-detected", payload("attempt-b"))
    outbox.post("/v1/attempts/terminal", payload("attempt-a"))

    assert [path for path, _ in sink.posts[1:]] == [
        "/v1/attempts/failure-detected",
        "/v1/attempts/terminal",
    ], f"a new key must always reach the sink: {sink.posts!r}"
    keys = [item["key"] for item in json.loads(core.data["events.json"])]
    assert keys == ["cluster-a/attempt-a/failure-detected"], (
        f"only the undelivered record may stay buffered: {keys!r}"
    )


class BrokenRemovalCore(ConfigMapCore):
    """Appends fine; deleting a delivered record always fails.

    Models the window in which the ConfigMap became unwritable (RBAC revoked,
    quota exceeded, API server outage) *between* the write-ahead append and
    the removal that follows a successful live POST.
    """

    def replace_namespaced_config_map(self, name, namespace, body):
        after = len(json.loads(body["data"].get("events.json", "[]")))
        if after < self.events(name):
            raise FakeApiException(500)
        return super().replace_namespaced_config_map(name, namespace, body)


def test_a_quarantined_record_does_not_deadlock_the_live_post() -> None:
    """A record ``replay`` has given up on must not silence the live path.

    ``replay`` quarantines after ``max_replay_attempts`` (production default
    20, one attempt per 30 s pass) even for a purely transient error, and then
    skips the record for ever -- no production caller passes
    ``include_quarantined``. If ``post`` still answered ``deferred_to_replay``
    for that key, the event would become permanently undeliverable: the exact
    F1 outcome (workload suspended, no incident) the deferral was added
    around.
    """

    core = ConfigMapCore()
    sink = RecordingSink(fail=True)
    outbox = KubernetesCompletionOutbox(core, sink, max_replay_attempts=3)
    with pytest.raises(OSError):
        outbox.post("/v1/attempts/failure-detected", payload())
    for _ in range(3):
        assert outbox.replay() == 0, "the control plane is still down"
    assert outbox.quarantined_depth() == 1, (
        "three failed attempts with max_replay_attempts=3 must quarantine the "
        f"record, depth={outbox.depth()}"
    )

    sink.fail = False
    assert outbox.replay() == 0, (
        "a quarantined record is no longer replay's to deliver, so the live "
        f"path has to own it: {sink.posts!r}"
    )
    result = outbox.post("/v1/attempts/failure-detected", payload())

    assert result == {"accepted": True}, (
        f"the live POST must happen once replay has given up, got {result!r}"
    )
    assert sink.posts[-1] == ("/v1/attempts/failure-detected", payload()), (
        f"the recovered sink must receive the event: {sink.posts!r}"
    )
    assert outbox.depth() == 0, (
        "a live delivery must clear the quarantined record, "
        f"{core.data['events.json']!r} is left"
    )


def test_a_delivered_event_is_not_undone_by_a_failed_removal() -> None:
    """The bookkeeping write after a successful POST must not mask success.

    ``_remove`` raising made ``post`` raise, and every caller reads that as
    "the control plane never accepted this": the watcher then suspends the
    workload 30 s later over an event the control plane already has.
    """

    core = BrokenRemovalCore()
    sink = RecordingSink()
    outbox = KubernetesCompletionOutbox(core, sink)

    result = outbox.post("/v1/attempts/failure-detected", payload())

    assert result == {"accepted": True}, (
        f"the sink's response must be returned, got {result!r}"
    )
    assert outbox.append_failures_total == 1, (
        "the failed removal must be counted like any other outbox write "
        f"failure, got {outbox.append_failures_total}"
    )
    assert outbox.depth() == 1, (
        "the record necessarily stays buffered until a later pass can clear it"
    )


class OneBadRemovalCore(ConfigMapCore):
    """The first removal fails; the ConfigMap is writable again after it."""

    def __init__(self) -> None:
        super().__init__()
        self.removal_failures = 0

    def replace_namespaced_config_map(self, name, namespace, body):
        after = len(json.loads(body["data"].get("events.json", "[]")))
        if after < self.events(name) and self.removal_failures == 0:
            self.removal_failures += 1
            raise FakeApiException(500)
        return super().replace_namespaced_config_map(name, namespace, body)


class UnwritableCore(ConfigMapCore):
    """Reads and writes can be turned off independently, after set-up."""

    def __init__(self) -> None:
        super().__init__()
        self.readable = True
        self.writable = True

    def read_namespaced_config_map(self, name, namespace):
        if not self.readable:
            raise FakeApiException(500)
        return super().read_namespaced_config_map(name, namespace)

    def replace_namespaced_config_map(self, name, namespace, body):
        if not self.writable:
            raise FakeApiException(500)
        return super().replace_namespaced_config_map(name, namespace, body)


def test_a_delivered_record_that_cannot_be_cleared_is_reclaimed_by_replay() -> None:
    """A delivered record must never be left quarantined.

    ``replay`` skips quarantined records for ever and the live path is done
    with the key (the controller has recorded it as sent), so a quarantined
    record whose post-delivery removal failed would be an orphan: buffered
    for ever and stuck in ``..._quarantined_depth``. Flagging it delivered
    hands it back to ``replay``, which drops it without re-sending it.
    """

    core = OneBadRemovalCore()
    sink = RecordingSink(fail=True)
    outbox = KubernetesCompletionOutbox(core, sink, max_replay_attempts=1)
    with pytest.raises(OSError):
        outbox.post("/v1/attempts/failure-detected", payload())
    assert outbox.replay() == 0, "the control plane is still down"
    assert outbox.quarantined_depth() == 1, "one failed attempt must quarantine it"

    sink.fail = False
    result = outbox.post("/v1/attempts/failure-detected", payload())

    assert result == {"accepted": True}, f"the live POST must win, got {result!r}"
    assert core.removal_failures == 1, "the test must exercise a failed removal"
    assert outbox.quarantined_depth() == 0, (
        "a delivered record may not stay quarantined -- nothing would ever "
        f"clear it: {core.data['events.json']!r}"
    )
    posted = len(sink.posts)

    assert outbox.replay() == 0, (
        "an already delivered record must not be re-sent by replay, "
        f"{sink.posts[posted:]!r} was"
    )
    assert len(sink.posts) == posted, f"no new POST is allowed: {sink.posts!r}"
    assert outbox.depth() == 0, (
        f"replay must drop the delivered record: {core.data['events.json']!r}"
    )


def test_depth_gauges_report_the_snapshot_a_failed_replay_pass_read() -> None:
    """The gauges must not read 0 while records are piling up.

    A pass that reads the ConfigMap and then fails to write it back is
    exactly when an operator needs the depth, so the numbers are published
    from whatever snapshot the pass did read. If even the read failed there is
    nothing to publish and the last known values stay.
    """

    core = UnwritableCore()
    outbox = KubernetesCompletionOutbox(core, RecordingSink(fail=True))
    with pytest.raises(OSError):
        outbox.post("/v1/attempts/failure-detected", payload())
    assert outbox.last_depth == 0, "no replay pass has published a gauge yet"

    core.writable = False
    with pytest.raises(FakeApiException):
        outbox.replay()

    assert outbox.last_depth == 1, (
        "the pass read one buffered record before it failed; the gauge must "
        f"say so, got {outbox.last_depth}"
    )
    core.readable = False
    with pytest.raises(FakeApiException):
        outbox.replay()

    assert outbox.last_depth == 1, (
        "a pass that read nothing must leave the last known depth alone, got "
        f"{outbox.last_depth}"
    )


def test_replay_skips_the_get_when_the_outbox_is_known_empty() -> None:
    """F6: an empty outbox must not cost a ConfigMap GET per reconcile.

    The watcher replays at the top of every pass and every debounced flush, so
    an idle cluster spent one whole-document read per pass on a document it had
    just emptied itself. Depth is tracked in-process instead: unknown (a fresh
    process, or any failed write) always reads, a known zero does not.
    """

    core = ConfigMapCore()
    sink = RecordingSink()
    outbox = KubernetesCompletionOutbox(core, sink)

    assert outbox.replay() == 0, "nothing is buffered yet"
    assert core.reads == [OUTBOX_NAME], (
        f"a fresh process does not know its depth and must read: {core.reads}"
    )

    core.reads.clear()
    assert outbox.replay() == 0, "still nothing buffered"
    assert core.reads == [], (
        f"a known-empty outbox must not be read again: {core.reads}"
    )

    outbox.post("/v1/attempts/terminal", payload())
    core.reads.clear()
    assert outbox.replay() == 0, "the live POST cleared the record it buffered"
    assert core.reads == [], (
        f"an append and its delivery leave the depth known-empty: {core.reads}"
    )

    core.reads.clear()
    assert outbox.replay(include_quarantined=True) == 0, "nothing is buffered"
    assert core.reads == [], (
        "no record was ever quarantined, so include_quarantined may skip too: "
        f"{core.reads}"
    )


def test_a_failed_write_makes_the_depth_unknown_again() -> None:
    """A write that raised leaves a document nobody knows the contents of.

    The one thing the in-process depth may never do is keep a stale zero over a
    record that may have been written: the record would sit in the ConfigMap
    with no pass ever reading it again.
    """

    core = UnwritableCore()
    sink = RecordingSink()
    outbox = KubernetesCompletionOutbox(core, sink)
    assert outbox.replay() == 0, "nothing is buffered yet"

    core.writable = False
    sink.fail = True
    with pytest.raises(FakeApiException):
        outbox.post("/v1/workload-observations", payload())
    core.reads.clear()
    core.writable = True

    assert outbox.replay() == 0, "the failed write buffered nothing"
    assert core.reads == [OUTBOX_NAME], (
        f"a failed write must force the next replay to read: {core.reads}"
    )


def test_legacy_attempt_state_migrates_to_the_active_config_map() -> None:
    """The upgrade path: attempt state written by the previous release.

    The record moves to ``<name>-active`` before it is dropped from ``<name>``,
    so a crash between the two writes duplicates it rather than losing it, and
    the next start migrates the leftover again.
    """

    core = ConfigMapCore()
    legacy = {
        "cluster_id": "cluster-a",
        "attempt_id": "attempt-legacy",
        "workload_phase": "RUNNING",
    }
    core.objects[OUTBOX_NAME]["active-attempts.json"] = json.dumps(
        {"cluster-a/attempt-legacy": legacy}
    )
    outbox = KubernetesCompletionOutbox(core, RecordingSink())

    assert outbox.load_attempt_observations() == [legacy], (
        "a record written by the previous release must still be restored"
    )
    assert json.loads(core.objects[ACTIVE_NAME]["active-attempts.json"]) == {
        "cluster-a/attempt-legacy": legacy
    }, f"the record must move to the active object: {core.objects[ACTIVE_NAME]}"
    assert "active-attempts.json" not in core.objects[OUTBOX_NAME], (
        f"the WAL object must not keep attempt state: {core.objects[OUTBOX_NAME]}"
    )
    assert outbox.load_attempt_observations() == [legacy], (
        "the migration must be idempotent"
    )


class UnavailableActiveState(ConfigMapCore):
    """A ``<name>-active`` object the API server refuses.

    ``status`` is what the Kubernetes client puts on ``ApiException``, and the
    outbox reads nothing else: 404 is the upgrade window before the manifest
    that adds the object, 403 the window before the Role that names it.
    """

    def __init__(self, status: int = 404) -> None:
        super().__init__()
        self.status = status
        #: Cleared by a test that applies the manifest mid-run.
        self.refusing = True

    def _refuse(self, name):
        if name != ACTIVE_NAME or not self.refusing:
            return
        error = RuntimeError(f"configmaps {name} is refused")
        error.status = self.status
        raise error

    def read_namespaced_config_map(self, name, namespace):
        if name == ACTIVE_NAME and self.refusing:
            # Recorded before it is refused: a GET the API server rejects still
            # costs the round trip this test is counting.
            self.reads.append(name)
            self._refuse(name)
        return super().read_namespaced_config_map(name, namespace)

    def replace_namespaced_config_map(self, name, namespace, body):
        self._refuse(name)
        return super().replace_namespaced_config_map(name, namespace, body)


def test_a_missing_active_state_is_probed_once_per_window() -> None:
    """A degraded object must not cost a GET per attempt per pass (I1).

    ``save_attempt_observation`` reads before it writes, so a watcher watching
    50 attempts spent 50 refused GETs per reconcile pass -- for ever, behind a
    single ERROR line that scrolled away hours earlier.
    """

    core = UnavailableActiveState()
    clock = [1_000.0]
    outbox = KubernetesCompletionOutbox(
        core, RecordingSink(), monotonic=lambda: clock[0]
    )

    assert outbox.load_attempt_observations() == [], (
        "a missing active state must degrade, not raise"
    )
    assert outbox.active_state_unavailable == 1, (
        "the degraded state must be visible on /metrics, got "
        f"{outbox.active_state_unavailable}"
    )
    core.reads.clear()
    for index in range(50):
        outbox.save_attempt_observation(
            {"cluster_id": "cluster-a", "attempt_id": f"attempt-{index}"}
        )
    assert [name for name in core.reads if name == ACTIVE_NAME] == [], (
        f"a degraded object must not be read again inside the window: {core.reads}"
    )

    clock[0] += 61.0
    outbox.save_attempt_observation({"cluster_id": "cluster-a", "attempt_id": "later"})
    assert [name for name in core.reads if name == ACTIVE_NAME] == [ACTIVE_NAME], (
        f"one probe is owed once the window elapses: {core.reads}"
    )


def test_a_forbidden_active_state_degrades_and_another_error_still_raises() -> None:
    """403 is the same upgrade window as 404, and must not CrashLoop (M2).

    ``load_attempt_observations`` runs from the controller's constructor, which
    catches only ``ValueError``, so a Role that does not yet name the new object
    used to kill a watcher whose write-ahead log was perfectly healthy.
    """

    outbox = KubernetesCompletionOutbox(UnavailableActiveState(403), RecordingSink())
    assert outbox.load_attempt_observations() == [], (
        "a forbidden active state must degrade like a missing one"
    )
    assert outbox.active_state_unavailable == 1, (
        f"the gauge must report the outage, got {outbox.active_state_unavailable}"
    )

    fatal = KubernetesCompletionOutbox(UnavailableActiveState(500), RecordingSink())
    with pytest.raises(RuntimeError, match="refused"):
        fatal.load_attempt_observations()


class FailingDropCore(ConfigMapCore):
    """Refuses the write that drops migrated state from the WAL object."""

    def __init__(self) -> None:
        super().__init__()
        self.refuse_drop = True

    def replace_namespaced_config_map(self, name, namespace, body):
        if (
            name == OUTBOX_NAME
            and self.refuse_drop
            and "active-attempts.json" not in body["data"]
        ):
            error = RuntimeError("the WAL object could not be rewritten")
            error.status = 500
            raise error
        return super().replace_namespaced_config_map(name, namespace, body)


def test_a_legacy_drop_that_failed_is_retried_before_it_can_resurrect() -> None:
    """A half-done migration must not resurrect a finished attempt (I2).

    ``adopt`` succeeded and ``drop`` failed without a crash, so the process kept
    running with the record in both objects. The attempt then finished and was
    removed from ``<name>-active`` -- and the next restart read the leftover in
    the WAL object, restored a finished attempt as RUNNING, held it for a whole
    missing-attempt grace and terminalized it a second time.
    """

    core = FailingDropCore()
    legacy = {
        "cluster_id": "cluster-a",
        "attempt_id": "attempt-legacy",
        "workload_phase": "RUNNING",
    }
    core.objects[OUTBOX_NAME]["active-attempts.json"] = json.dumps(
        {"cluster-a/attempt-legacy": legacy}
    )
    outbox = KubernetesCompletionOutbox(core, RecordingSink())

    assert outbox.load_attempt_observations() == [legacy], (
        "the record must still be restored while the drop is owed"
    )
    assert "active-attempts.json" in core.objects[OUTBOX_NAME], (
        "this test needs the drop to have failed"
    )

    core.refuse_drop = False
    outbox.remove_attempt_observation(legacy)
    assert "active-attempts.json" not in core.objects[OUTBOX_NAME], (
        f"the owed drop must be retried: {core.objects[OUTBOX_NAME]}"
    )

    restarted = KubernetesCompletionOutbox(core, RecordingSink())
    assert restarted.load_attempt_observations() == [], (
        "a finished attempt must not come back from the WAL object: "
        f"{restarted.load_attempt_observations()}"
    )


def test_a_migration_keeps_the_newer_observation_not_the_higher_string() -> None:
    """The merge compares instants; ISO-8601 text does not sort as time.

    Pydantic renders a whole second as ``...T10:00:00Z`` and every other
    instant as ``...T10:00:00.500000Z``, and ``Z`` sorts after ``.``. So the
    string comparison the adopt used ranked the *earlier* whole-second record
    above the fresher one it was supposed to lose to, and a repeated migration
    replaced a live record with a stale copy of itself.
    """

    core = ConfigMapCore()
    key = "cluster-a/attempt-a"
    stale = {
        "cluster_id": "cluster-a",
        "attempt_id": "attempt-a",
        "workload_phase": "RUNNING",
        "observed_at": "2026-09-08T10:00:00Z",
    }
    fresher = {
        **stale,
        "workload_phase": "STOPPED",
        "observed_at": "2026-09-08T10:00:00.500000Z",
    }
    core.objects[OUTBOX_NAME]["active-attempts.json"] = json.dumps({key: stale})
    core.objects[ACTIVE_NAME]["active-attempts.json"] = json.dumps({key: fresher})
    outbox = KubernetesCompletionOutbox(core, RecordingSink())

    assert outbox.load_attempt_observations() == [fresher], (
        "the half-second-later record must survive the migration, got "
        f"{outbox.load_attempt_observations()}"
    )


def test_a_migration_survives_a_timestamp_without_a_timezone(caplog) -> None:
    """One naive timestamp must not wedge the migration for the process life.

    ``datetime`` refuses to compare a naive instant with an aware one, and the
    comparison happens inside the migration's own try/except: a single record
    without a zone (an older release, a hand-edited ConfigMap) made every
    migration attempt raise, so the legacy key was never dropped and every load
    and every owed probe printed the same traceback again. Naive means UTC here,
    which is the only zone anything in this system writes.
    """

    core = ConfigMapCore()
    key = "cluster-a/attempt-a"
    naive = {
        "cluster_id": "cluster-a",
        "attempt_id": "attempt-a",
        "workload_phase": "STOPPED",
        "observed_at": "2026-09-08T10:00:01",
    }
    aware = {
        **naive,
        "workload_phase": "RUNNING",
        "observed_at": "2026-09-08T10:00:00Z",
    }
    core.objects[OUTBOX_NAME]["active-attempts.json"] = json.dumps({key: naive})
    core.objects[ACTIVE_NAME]["active-attempts.json"] = json.dumps({key: aware})
    outbox = KubernetesCompletionOutbox(core, RecordingSink())

    with caplog.at_level(logging.ERROR, logger="gpu_fault.completion_outbox"):
        restored = outbox.load_attempt_observations()

    assert restored == [naive], (
        f"the later of the two instants must win whatever its zone: {restored}"
    )
    assert "active-attempts.json" not in core.objects[OUTBOX_NAME], (
        "the migration must complete, not raise inside its own try/except: "
        f"{core.objects[OUTBOX_NAME]}"
    )
    assert caplog.records == [], (
        "nothing about a missing zone is worth a traceback: "
        f"{[record.getMessage() for record in caplog.records]}"
    )


def test_a_recovered_active_state_is_probed_from_the_reconcile_top() -> None:
    """The gauge has to come back down on a cluster that saves nothing (I1).

    ``usable()`` is only reached from a load/save/remove, and an idle cluster
    makes none of them: the operator applied the manifest and the watcher went
    on reporting the outage for ever. The probe is spent where every pass
    already goes, and at most once per retry window.
    """

    core = UnavailableActiveState()
    clock = [1_000.0]
    logger = logging.getLogger("test.probe")
    outbox = KubernetesCompletionOutbox(
        core, RecordingSink(), monotonic=lambda: clock[0]
    )

    assert outbox.load_attempt_observations() == [], (
        "a missing active state must degrade, not raise"
    )
    core.refusing = False
    core.reads.clear()
    replay_completion_outbox(outbox, logger)
    assert outbox.active_state_unavailable == 1, (
        "the retry window must be respected even once the object is there"
    )
    probes = [name for name in core.reads if name == ACTIVE_NAME]
    assert probes == [], f"no probe is owed inside the window: {core.reads}"

    clock[0] += 61.0
    replay_completion_outbox(outbox, logger)

    assert outbox.active_state_unavailable == 0, (
        "an applied object must clear the gauge on the next pass, got "
        f"{outbox.active_state_unavailable}"
    )
    probes = [name for name in core.reads if name == ACTIVE_NAME]
    assert probes == [ACTIVE_NAME], (
        f"the recovery must cost exactly one GET: {core.reads}"
    )
    replay_completion_outbox(outbox, logger)
    probes = [name for name in core.reads if name == ACTIVE_NAME]
    assert probes == [ACTIVE_NAME], (
        f"a healthy object must not be probed at all: {core.reads}"
    )


def test_a_degraded_active_state_still_restores_the_legacy_copy() -> None:
    """An upgrade restart keeps its memory while ``<name>-active`` refuses.

    The records of the previous release are in the WAL object, unmigrated and
    perfectly readable; returning nothing threw away exactly the restart memory
    that decides whether a fault on an attempt whose Pods are gone reads ACTIVE
    or IDLE. They are read, not dropped: the migration still owns moving them.
    """

    core = UnavailableActiveState(403)
    legacy = {
        "cluster_id": "cluster-a",
        "attempt_id": "attempt-legacy",
        "workload_phase": "RUNNING",
        "observed_at": "2026-09-08T10:00:00Z",
    }
    core.objects[OUTBOX_NAME]["active-attempts.json"] = json.dumps(
        {"cluster-a/attempt-legacy": legacy}
    )
    outbox = KubernetesCompletionOutbox(core, RecordingSink())

    restored = outbox.load_attempt_observations()

    assert restored == [legacy], (
        f"the legacy copy must be restored while the new object refuses: {restored}"
    )
    assert "active-attempts.json" in core.objects[OUTBOX_NAME], (
        "the legacy copy must stay where the migration can still find it: "
        f"{core.objects[OUTBOX_NAME]}"
    )


def test_a_recovered_active_state_does_not_rewrite_unchanged_records() -> None:
    """A recovery must not cost one whole-document PUT per attempt.

    ``_attempt_digests`` is empty after a restart, after a migration and after
    every recovery, and the mutation always returned a new dict, so the
    unchanged-mutation shortcut in ``_mutate_data`` never fired: the first pass
    after a recovery rewrote the whole attempt-state document once per attempt
    -- 125 attempts of ~830 KB is ~100 MB of writes in one pass, against an
    object nothing in it had changed.

    "Unchanged" has to mean what ``_attempt_digest`` means by it. ``observed_at``
    is a spec field the watcher stamps on every single pass, so a record is never
    byte-identical to the one already stored and a comparison of the dicts fires
    exactly never on a live cluster -- which is why the saves below move
    ``observed_at`` and nothing else.
    """

    core = UnavailableActiveState()
    clock = [1_000.0]
    records = [
        {
            "cluster_id": "cluster-a",
            "attempt_id": f"attempt-{index}",
            "workload_phase": "RUNNING",
            "observed_at": "2026-09-08T10:00:00Z",
        }
        for index in range(5)
    ]
    core.objects[ACTIVE_NAME]["active-attempts.json"] = json.dumps(
        {f"cluster-a/{record['attempt_id']}": record for record in records}
    )
    outbox = KubernetesCompletionOutbox(
        core, RecordingSink(), monotonic=lambda: clock[0]
    )
    assert outbox.load_attempt_observations() == [], (
        "this test needs the object to have refused first"
    )

    core.refusing = False
    clock[0] += 61.0
    version = core.versions[ACTIVE_NAME]
    for record in records:
        # The next pass of a live watcher: same state, one minute later.
        outbox.save_attempt_observation(
            {**record, "observed_at": "2026-09-08T10:01:00Z"}
        )

    assert core.versions[ACTIVE_NAME] == version, (
        "a record the object already holds must not be written again, got "
        f"{core.versions[ACTIVE_NAME] - version} whole-document writes"
    )
    assert outbox.active_state_unavailable == 0, (
        "the first successful write must clear the gauge, got "
        f"{outbox.active_state_unavailable}"
    )
    changed = {
        **records[0],
        "workload_phase": "STOPPED",
        "observed_at": "2026-09-08T10:01:00Z",
    }
    outbox.save_attempt_observation(changed)
    assert core.versions[ACTIVE_NAME] == version + 1, (
        "a record that did change must still be written once, got "
        f"{core.versions[ACTIVE_NAME] - version} writes"
    )
    stored = json.loads(core.objects[ACTIVE_NAME]["active-attempts.json"])
    assert stored["cluster-a/attempt-0"]["workload_phase"] == "STOPPED", (
        f"the write that did happen must be the changed record: {stored}"
    )

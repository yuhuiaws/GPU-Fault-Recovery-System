from __future__ import annotations

import json
import logging
from types import SimpleNamespace

import pytest

from gpu_fault.completion_outbox import CompletionOutboxFull, KubernetesCompletionOutbox


class ConfigMapCore:
    def __init__(self) -> None:
        self.version = 1
        self.data = {"active-attempts.json": "{}", "events.json": "[]"}

    def read_namespaced_config_map(self, name, namespace):
        assert name == "gpu-fault-completion-watcher-outbox"
        assert namespace == "gpu-fault-system"
        return SimpleNamespace(
            metadata=SimpleNamespace(resource_version=str(self.version)),
            data=dict(self.data),
        )

    def replace_namespaced_config_map(self, name, namespace, body):
        assert body["metadata"]["resourceVersion"] == str(self.version)
        self.version += 1
        self.data = dict(body["data"])


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
        before = len(json.loads(self.data["events.json"]))
        after = len(json.loads(body["data"]["events.json"]))
        if after < before:
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
        "the record necessarily stays buffered; replay re-sends it and the "
        "control plane deduplicates by event_key"
    )

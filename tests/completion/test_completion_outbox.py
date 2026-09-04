from __future__ import annotations

import json
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

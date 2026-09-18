"""Reset domains and missing process coverage at the real publication boundary."""

from __future__ import annotations

import json
import os
from pathlib import Path
from threading import Event

import pytest

from gpu_fault.app import process_metrics as metrics
from gpu_fault.app.metric_aggregation import DEGRADED_METRIC, PROCESSES_METRIC

COUNTER = "gpu_fault_workflow_lifetime_exceeded_total"
GAUGE = "gpu_fault_processor_in_flight"
HISTOGRAM = "gpu_fault_processor_request_processing_seconds"


def rendered(counter: int, slot: int, *, in_flight: int = 0) -> metrics.Rendered:
    result = metrics.parse_lines(
        [
            f"# TYPE {COUNTER} counter",
            f"{COUNTER} {counter}",
            f"# TYPE {GAUGE} gauge",
            f"{GAUGE} {in_flight}",
        ]
    )
    result.slot = slot
    return result


def samples(lines: list[str], family: str) -> list[tuple[dict[str, str], float]]:
    return [
        (dict(sample.labels), float(sample.value))
        for sample in metrics.parse_lines(lines).samples[family]
    ]


@pytest.fixture
def slots(monkeypatch: pytest.MonkeyPatch):
    registry = metrics.SlotRegistry()
    monkeypatch.setattr(metrics, "SLOTS", registry)
    monkeypatch.setattr(metrics.time, "time", lambda: 1000.0)
    monkeypatch.setattr(metrics.os, "kill", lambda *_args: None)
    yield registry
    registry.release()


@pytest.mark.parametrize("first,second", [(7, 11), (0, 11), (0, 0), (0, 21)])
def test_child_counters_keep_independent_bounded_reset_domains(
    first: int, second: int
) -> None:
    lines = metrics.aggregate(
        rendered(first, 0, in_flight=3), [rendered(second, 1, in_flight=4)]
    )
    assert samples(lines, COUNTER) == [
        ({"process": "0"}, first),
        ({"process": "1"}, second),
    ]
    assert samples(lines, GAUGE) == [({}, 7)]


def test_histogram_buckets_counts_and_sums_keep_the_same_slot() -> None:
    peers = []
    for slot, count, elapsed in ((0, 2, 0.5), (1, 3, 1.5)):
        peer = metrics.parse_lines(
            [
                f"# TYPE {HISTOGRAM} histogram",
                f'{HISTOGRAM}_bucket{{le="1"}} {count}',
                f'{HISTOGRAM}_bucket{{le="+Inf"}} {count}',
                f"{HISTOGRAM}_count {count}",
                f"{HISTOGRAM}_sum {elapsed}",
            ]
        )
        peer.slot = slot
        peers.append(peer)
    lines = metrics.aggregate(peers[0], peers[1:])
    actual = metrics.parse_lines(lines).samples[HISTOGRAM]
    assert len(actual) == 8
    for slot, count, elapsed in (("0", 2, 0.5), ("1", 3, 1.5)):
        selected = [
            sample for sample in actual if dict(sample.labels)["process"] == slot
        ]
        assert {sample.name for sample in selected} == {
            f"{HISTOGRAM}_bucket",
            f"{HISTOGRAM}_count",
            f"{HISTOGRAM}_sum",
        }
        assert [float(s.value) for s in selected] == [count, count, count, elapsed]


@pytest.mark.parametrize(
    "stamp", [None, True, "1000", float("nan"), float("inf"), -1, 939.999, 1000.001]
)
def test_untrusted_publication_time_degrades_instead_of_reporting_a_healthy_peer(
    tmp_path: Path, slots: metrics.SlotRegistry, stamp: object
) -> None:
    peer_pid = os.getpid() + 100000
    metrics.publish(tmp_path, rendered(11, 1), pid=peer_pid, slot=1)
    path = tmp_path / f"{peer_pid}.json"
    payload = json.loads(path.read_text())
    payload["published_at"] = stamp
    path.write_text(json.dumps(payload))

    lines = metrics.pod_coherent_lines([f"{COUNTER} 7"], directory=tmp_path)

    assert samples(lines, COUNTER) == [({"process": "0"}, 7)]
    assert f"{DEGRADED_METRIC} 1" in lines
    assert f"{PROCESSES_METRIC} 1" in lines
    assert path.exists(), "unknown live evidence is not deleted"


@pytest.mark.parametrize("stamp", [940.0, 999.0, 1000.0])
def test_fresh_publication_and_inclusive_age_boundary_are_accepted(
    tmp_path: Path, slots: metrics.SlotRegistry, stamp: float
) -> None:
    peer_pid = os.getpid() + 100000
    metrics.publish(tmp_path, rendered(11, 1), pid=peer_pid, slot=1)
    path = tmp_path / f"{peer_pid}.json"
    payload = json.loads(path.read_text())
    payload["published_at"] = stamp
    path.write_text(json.dumps(payload))
    lines = metrics.pod_coherent_lines([f"{COUNTER} 7"], directory=tmp_path)
    assert samples(lines, COUNTER) == [({"process": "0"}, 7), ({"process": "1"}, 11)]
    assert f"{DEGRADED_METRIC} 0" in lines
    assert f"{PROCESSES_METRIC} 2" in lines


@pytest.mark.parametrize("contents", ["{", "[]", '{"format": 999}'])
def test_unreadable_live_peer_is_explicitly_incomplete(
    tmp_path: Path, slots: metrics.SlotRegistry, contents: str
) -> None:
    path = tmp_path / f"{os.getpid() + 100000}.json"
    path.write_text(contents)
    lines = metrics.pod_coherent_lines([f"{COUNTER} 7"], directory=tmp_path)
    assert f"{DEGRADED_METRIC} 1" in lines
    assert f"{PROCESSES_METRIC} 1" in lines


@pytest.mark.parametrize("slot", [None, True, -1, 0, 16, "1"])
def test_bad_or_duplicate_slot_never_creates_another_counter_reset_domain(
    tmp_path: Path, slots: metrics.SlotRegistry, slot: object
) -> None:
    peer_pid = os.getpid() + 100000
    metrics.publish(tmp_path, rendered(11, 1), pid=peer_pid, slot=1)
    path = tmp_path / f"{peer_pid}.json"
    payload = json.loads(path.read_text())
    payload["slot"] = slot
    path.write_text(json.dumps(payload))
    lines = metrics.pod_coherent_lines([f"{COUNTER} 7"], directory=tmp_path)
    assert samples(lines, COUNTER) == [({"process": "0"}, 7)]
    assert f"{DEGRADED_METRIC} 1" in lines


def test_held_slot_without_publication_is_unknown_and_recovers(
    tmp_path: Path, slots: metrics.SlotRegistry
) -> None:
    assert slots.slot(tmp_path) == 0
    peer = metrics.SlotRegistry()
    try:
        assert peer.slot(tmp_path) == 1
        assert metrics.active_slots(tmp_path) == {0, 1}
        missing = metrics.pod_coherent_lines([f"{COUNTER} 7"], directory=tmp_path)
        assert f"{DEGRADED_METRIC} 1" in missing
        metrics.publish(tmp_path, rendered(11, 1), pid=os.getpid() + 100000, slot=1)
        recovered = metrics.pod_coherent_lines([f"{COUNTER} 7"], directory=tmp_path)
        assert f"{DEGRADED_METRIC} 0" in recovered
        assert f"{PROCESSES_METRIC} 2" in recovered
    finally:
        peer.release()
    assert metrics.active_slots(tmp_path) == {0}


def test_dead_peer_is_removed_without_claiming_missing_live_coverage(
    tmp_path: Path, slots: metrics.SlotRegistry, monkeypatch: pytest.MonkeyPatch
) -> None:
    peer_pid = os.getpid() + 100000
    metrics.publish(tmp_path, rendered(11, 1), pid=peer_pid, slot=1)

    def dead(_pid: int, _signal: int) -> None:
        raise ProcessLookupError

    monkeypatch.setattr(metrics.os, "kill", dead)
    lines = metrics.pod_coherent_lines([f"{COUNTER} 7"], directory=tmp_path)
    assert f"{DEGRADED_METRIC} 0" in lines
    assert not (tmp_path / f"{peer_pid}.json").exists(), (
        "a dead PID's publication must be removed"
    )


def test_slot_exhaustion_degrades_without_publishing_an_unowned_slot(
    tmp_path: Path, slots: metrics.SlotRegistry, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(metrics, "MAX_SLOTS", 1)
    peer = metrics.SlotRegistry()
    try:
        assert peer.slot(tmp_path) == 0
        lines = metrics.pod_coherent_lines([f"{COUNTER} 7"], directory=tmp_path)
        assert f"{DEGRADED_METRIC} 1" in lines
        assert not (tmp_path / f"{os.getpid()}.json").exists(), (
            "an exhausted allocator must not publish under an unowned slot"
        )
    finally:
        peer.release()


@pytest.mark.parametrize(
    "defect",
    [
        "header-type",
        "value",
        "label-type",
        "duplicate-label",
        "name-type",
        "pid",
        "boolean-format",
        "float-format",
    ],
)
def test_malformed_publication_degrades_even_when_its_clock_is_fresh(
    tmp_path: Path, slots: metrics.SlotRegistry, defect: str
) -> None:
    peer_pid = os.getpid() + 100000
    metrics.publish(tmp_path, rendered(11, 1), pid=peer_pid, slot=1)
    path = tmp_path / f"{peer_pid}.json"
    payload = json.loads(path.read_text())
    if defect == "header-type":
        payload["families"][COUNTER] = [42, {}]
    elif defect == "value":
        payload["samples"][0][2] = "not-a-number"
    elif defect == "label-type":
        payload["samples"][0][1] = [["label", 42]]
    elif defect == "duplicate-label":
        payload["samples"][0][1] = [["label", "a"], ["label", "b"]]
    elif defect == "name-type":
        payload["samples"][0][0] = []
    elif defect == "pid":
        payload["pid"] = str(peer_pid)
    else:
        payload["format"] = True if defect == "boolean-format" else 1.0
    path.write_text(json.dumps(payload))
    lines = metrics.pod_coherent_lines([f"{COUNTER} 7"], directory=tmp_path)
    assert f"{DEGRADED_METRIC} 1" in lines
    assert f"{PROCESSES_METRIC} 1" in lines


def test_publication_completed_during_a_read_is_not_mistaken_for_future_evidence(
    tmp_path: Path, slots: metrics.SlotRegistry, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = [1000.0]
    monkeypatch.setattr(metrics.time, "time", lambda: clock[0])
    peer_pid = os.getpid() + 100000
    metrics.publish(tmp_path, rendered(11, 1), pid=peer_pid, slot=1)
    path = tmp_path / f"{peer_pid}.json"
    payload = json.loads(path.read_text())
    payload["published_at"] = 1001.0
    path.write_text(json.dumps(payload))
    read = Path.read_text

    def completed(target: Path, *args, **kwargs):
        result = read(target, *args, **kwargs)
        if target == path:
            clock[0] = 1001.0
        return result

    monkeypatch.setattr(Path, "read_text", completed)
    lines = metrics.pod_coherent_lines([f"{COUNTER} 7"], directory=tmp_path)
    assert f"{DEGRADED_METRIC} 0" in lines
    assert f"{PROCESSES_METRIC} 2" in lines


def test_missing_directory_has_no_sibling_publications(tmp_path: Path) -> None:
    assert metrics.live_siblings(tmp_path / "absent") == []


def test_exhausted_publisher_never_renders_or_publishes_an_unowned_slot(
    tmp_path: Path,
    slots: metrics.SlotRegistry,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setattr(metrics, "MAX_SLOTS", 1)
    holder = metrics.SlotRegistry()
    stop = Event()
    stop.set()

    def forbidden():
        raise AssertionError("rendering without a slot is not a publication")

    try:
        assert holder.slot(tmp_path) == 0
        metrics.publish_forever(forbidden, stop, directory=tmp_path)
        assert "no free process metric slot" in caplog.text
        assert not (tmp_path / f"{os.getpid()}.json").exists(), (
            "the publisher must not write a snapshot without owning a slot"
        )
    finally:
        holder.release()

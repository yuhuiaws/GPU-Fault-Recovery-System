from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

from gpu_fault.app import process_metrics as metrics
from gpu_fault.app.metric_aggregation import DEGRADED_METRIC, PROCESSES_METRIC
from tests.app_services._cov95_runtime_workers import Stop
from tests.regional._cov95_runtime_support import offline_runtime as offline_runtime

SUM = "gpu_fault_processor_in_flight"
MAX = "gpu_fault_workflow_dispatch_last_cycle_timestamp_seconds"
PER_PROCESS = "gpu_fault_processor_notification_shard"


@pytest.fixture
def slots(
    monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> metrics.SlotRegistry:
    registry = metrics.SlotRegistry()
    monkeypatch.setattr(metrics, "SLOTS", registry)
    request.addfinalizer(registry.release)
    return registry


@pytest.mark.parametrize("labels", [None, {"bad": "value"}, [["name"]], [42]])
def test_malformed_sibling_sample_does_not_break_the_scrape_or_drop_valid_siblings(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    slots: metrics.SlotRegistry,
    labels: Any,
) -> None:
    sibling = os.getpid() + 100000
    metrics.publish(
        tmp_path, metrics.parse_lines([f"{SUM} 7", f"{MAX} 13"]), pid=sibling, slot=1
    )
    path = tmp_path / f"{sibling}.json"
    payload = json.loads(path.read_text())
    payload["samples"][0][1] = labels
    path.write_text(json.dumps(payload), encoding="ascii")
    monkeypatch.setattr(metrics.os, "kill", lambda pid, signal: None)
    result = metrics.pod_coherent_lines([f"{SUM} 5"], directory=tmp_path)
    assert f"{SUM} 5" in result
    assert f"{MAX} 13" in result
    assert f"{DEGRADED_METRIC} 1" in result
    assert json.loads(path.read_text()) == payload


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        ("+Inf", "2", "+Inf"),
        ("-Inf", "2", "-Inf"),
        ("NaN", "2", "NaN"),
        ("1e2", "1", "101.0"),
        ("0.25", "0.500", "0.750"),
        ("2", "invalid", "2"),
    ],
)
def test_numeric_aggregation_preserves_nonfinite_decimal_and_fallback_semantics(
    left: str, right: str, expected: str
) -> None:
    local = metrics.parse_lines([f"{SUM} {left}"])
    peer = metrics.parse_lines([f"{SUM} {right}"])
    assert metrics.aggregate(local, [peer]) == [f"{SUM} {expected}"]


def test_parser_handles_spacing_headers_and_per_process_label_replacement() -> None:
    family = "unit_runtime_gauge"
    rendered = metrics.parse_lines(
        [
            "\n",
            "# ordinary comment",
            f"# HELP {family} first",
            f"# HELP {family} second",
            f"# TYPE {family} gauge",
            f'{family}{{ , role="worker", }} 2',
        ]
    )
    assert rendered.family_type(family) == "gauge"
    assert rendered.family_type("missing") is None
    assert rendered.samples[family][0].line() == f'{family}{{role="worker"}} 2'
    assert rendered.families[family][0] == f"# HELP {family} first"
    with pytest.raises(ValueError, match="unquoted"):
        metrics.parse_lines(["unit_runtime_bad{role=worker} 1"])
    local = metrics.parse_lines([f'{PER_PROCESS}{{process="old",role="worker"}} 1'])
    peer = metrics.parse_lines([f"{PER_PROCESS} 2"])
    peer.slot, peer.pid = 4, 123
    assert metrics.aggregate(local, [peer]) == [
        f'{PER_PROCESS}{{role="worker",process="0"}} 1',
        f'{PER_PROCESS}{{process="4"}} 2',
    ]


def test_unregistered_plugin_metrics_have_documented_fallbacks(
    caplog: pytest.LogCaptureFixture,
) -> None:
    local = metrics.parse_lines(
        [
            "# TYPE unit_runtime_custom summary",
            "unit_runtime_custom_count 2",
            "unit_runtime_undeclared_total 3",
            "unit_runtime_untyped 5",
        ]
    )
    peer = metrics.parse_lines(
        [
            "# TYPE unit_runtime_custom summary",
            "unit_runtime_custom_count 4",
            "unit_runtime_undeclared_total 6",
            "unit_runtime_untyped 2",
        ]
    )
    local.slot, peer.slot = 0, 1
    result = metrics.aggregate(local, [peer])
    assert 'unit_runtime_custom_count{process="0"} 2' in result
    assert 'unit_runtime_custom_count{process="1"} 4' in result
    assert 'unit_runtime_undeclared_total{process="0"} 3' in result
    assert 'unit_runtime_undeclared_total{process="1"} 6' in result
    assert "unit_runtime_untyped 5" in result
    assert "has no aggregation strategy" in caplog.text
    warnings = caplog.text.count("has no aggregation strategy")
    assert metrics.aggregate(local, [peer]) == result
    assert caplog.text.count("has no aggregation strategy") == warnings


def test_publish_failure_keeps_previous_snapshot_and_removes_temporary_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = tmp_path / "123.json"
    metrics.publish(tmp_path, metrics.parse_lines([f"{SUM} 2"]), pid=123)
    before = path.read_bytes()

    def failed(*args: Any) -> None:
        raise OSError("synthetic replace failure")

    monkeypatch.setattr(metrics.os, "replace", failed)
    with pytest.raises(OSError, match="replace failure"):
        metrics.publish(tmp_path, metrics.parse_lines([f"{SUM} 99"]), pid=123)
    assert path.read_bytes() == before
    assert list(tmp_path.glob("*.tmp")) == []


@pytest.mark.parametrize(
    "defect",
    [
        "permission",
        "dead",
        "bad-json",
        "bad-format",
        "bad-families",
        "bad-samples",
        "bad-root",
        "bad-no-families",
        "bad-no-samples",
    ],
)
def test_sibling_reading_handles_liveness_and_malformed_snapshot_surfaces(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, defect: str
) -> None:
    metrics.publish(tmp_path, metrics.parse_lines([f"{SUM} 2"]), pid=123, slot=1)
    path = tmp_path / "123.json"
    payload = json.loads(path.read_text())
    if defect == "bad-json":
        path.write_text("{", encoding="ascii")
    elif defect == "bad-format":
        payload["format"] = 99
    elif defect == "bad-families":
        payload["families"] = {SUM: ["not-two-fields"], "unknown": "not-a-list"}
    elif defect == "bad-samples":
        payload["samples"] = ["not-a-sample", [1, 2]]
    elif defect == "bad-root":
        payload = []
    elif defect == "bad-no-families":
        payload["families"] = None
    elif defect == "bad-no-samples":
        payload["samples"] = None
    if defect.startswith("bad-") and defect != "bad-json":
        path.write_text(json.dumps(payload), encoding="ascii")

    def alive(pid: int, signal: int) -> None:
        assert signal == 0
        if defect == "permission":
            raise PermissionError("synthetic process owner")
        if defect == "dead":
            raise ProcessLookupError("synthetic exited process")

    monkeypatch.setattr(metrics.os, "kill", alive)
    result = metrics.live_siblings(tmp_path, pid=456)
    assert len(result) == (0 if defect == "dead" else 1)
    if defect.startswith("bad-"):
        assert result[0].degraded, "malformed live evidence must mark unknown coverage"
    if defect == "bad-no-families":
        assert result[0].samples[SUM][0].value == "2"
        assert result[0].families[SUM] == (None, None)
    if defect == "bad-no-samples":
        assert result[0].samples[SUM] == []
    assert path.exists() is (defect != "dead")


def test_slot_exhaustion_and_directory_change_do_not_reuse_an_owned_lock(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(metrics, "MAX_SLOTS", 1)
    first, second = metrics.SlotRegistry(), metrics.SlotRegistry()
    try:
        assert first.slot(tmp_path / "one") == 0
        assert second.slot(tmp_path / "one") is None
        assert first.slot(tmp_path / "two") == 0
        assert second.slot(tmp_path / "one") == 0
    finally:
        first.release()
        second.release()


def test_publisher_recovers_after_render_failure_and_stops_after_a_complete_write(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    slots: metrics.SlotRegistry,
    caplog: pytest.LogCaptureFixture,
) -> None:
    calls = []

    def render() -> list[str]:
        calls.append("render")
        if len(calls) == 1:
            raise RuntimeError("synthetic render failure")
        return [f"{SUM} 3"]

    stop = Stop(1)
    metrics.publish_forever(render, stop, directory=tmp_path, interval=0.25)
    assert calls == ["render", "render"]
    assert stop.waits == [0.25, 0.25]
    assert "could not be published" in caplog.text
    document = json.loads((tmp_path / f"{os.getpid()}.json").read_text())
    assert document["samples"] == [[SUM, [], "3"]]


def test_failed_publication_falls_back_to_the_local_view_with_degraded_flag(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, slots: metrics.SlotRegistry
) -> None:
    def failed(*args: Any, **kwargs: Any) -> None:
        raise OSError("synthetic shared filesystem error")

    monkeypatch.setattr(metrics, "publish", failed)
    lines = metrics.pod_coherent_lines([f"{SUM} 7"], directory=tmp_path)
    assert f"{SUM} 7" in lines
    assert f"{PROCESSES_METRIC} 1" in lines
    assert f"{DEGRADED_METRIC} 1" in lines


def test_declared_gauge_without_any_samples_does_not_invent_zero() -> None:
    family = "gpu_fault_processor_queue_depth"
    header = f"# TYPE {family} gauge"
    local = metrics.Rendered(families={family: (None, header)})
    assert metrics.aggregate(local, []) == [header]
    peer = metrics.parse_lines([f"{family} 9"])
    assert metrics.aggregate(local, [peer]) == [header, f"{family} 9"]

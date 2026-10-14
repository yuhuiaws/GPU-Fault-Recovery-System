"""Each step records its Aurora writer's start capacity and fails its
measurement on a cluster nobody connects to.

Earlier formal rounds named an orphan Aurora cluster in
``GPU_FAULT_AURORA_CLUSTER_ID`` (0.5 ACU, zero connections), and every step's
aurora.json described it without notice. A step now records the writer it
resolved at its start -- ``ServerlessDatabaseCapacity`` and ``CPUUtilization``
at the step start, at the window start and their maximum over the window --
and a named cluster without a single connection during the window fails the
step's measurement (exit 4, ``status.json`` ``measurement-failed``) while the
product readings and verdict stay as they were.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from scripts.perf import regional_action_capacity_suite as action
from scripts.perf import regional_capacity_database as database
from scripts.perf import regional_capacity_placement as placement_module
from scripts.perf import regional_capacity_suite as capacity
from tests.regional._perf_caller_support import CapacityWire, action_args, capacity_args

# Minute-aligned epochs: the step resolves its writer in the first minute, the
# load runs from 1_000_350 to 1_000_470.
STEP_START = 1_000_050.0
WINDOW = (1_000_350.0, 1_000_470.0)
PLACEMENT = {
    "observed_at": datetime.fromtimestamp(STEP_START, timezone.utc).isoformat(),
    "aurora": {"cluster_id": "aurora-a", "writer_instance_id": "aurora-a-1"},
}


def series(*points: tuple[float, float, float]) -> list[dict[str, Any]]:
    return [
        {
            "timestamp": datetime.fromtimestamp(at, timezone.utc).isoformat(),
            "average": average,
            "maximum": maximum,
        }
        for at, average, maximum in points
    ]


METRICS = {
    "ServerlessDatabaseCapacity": series(
        (1_000_020, 0.5, 0.5),
        (1_000_080, 0.5, 0.5),
        (1_000_320, 8.0, 16.0),
        (1_000_380, 40.0, 64.0),
        (1_000_440, 64.0, 96.0),
        (1_000_560, 128.0, 128.0),
    ),
    "CPUUtilization": series(
        (1_000_020, 3.0, 4.0), (1_000_320, 20.0, 35.0), (1_000_440, 60.0, 85.0)
    ),
    "DatabaseConnections": series((1_000_320, 40.0, 52.0), (1_000_440, 44.0, 60.0)),
}


def measure(metrics: dict[str, Any]) -> dict[str, Any]:
    return database.aurora_measurement(
        metrics, placement=PLACEMENT, window_start=WINDOW[0], window_end=WINDOW[1]
    )


def test_the_writer_readings_cover_the_step_start_the_window_start_and_its_peak() -> (
    None
):
    measurement = measure(METRICS)

    assert measurement["serverless_capacity_acu"] == {
        "at_step_start": 0.5,
        "at_window_start": 8.0,
        "max_during_window": 96.0,
    }, "the minute after the window is not part of its peak"
    assert measurement["cpu_utilization_percent"] == {
        "at_step_start": 3.0,
        "at_window_start": 20.0,
        "max_during_window": 85.0,
    }
    assert measurement["database_connections_max_during_window"] == 60.0
    assert (measurement["status"], measurement["reason"]) == ("PASS", None)
    assert (measurement["cluster_id"], measurement["writer_instance_id"]) == (
        "aurora-a",
        "aurora-a-1",
    )
    assert measurement["step_started_at_epoch"] == STEP_START


def test_a_cluster_without_a_connection_in_the_window_fails_the_measurement() -> None:
    idle = {
        **METRICS,
        # Busy before the step and after it: only the window counts.
        "DatabaseConnections": series(
            (1_000_020, 9.0, 9.0), (1_000_320, 0.0, 0.0), (1_000_560, 7.0, 7.0)
        ),
    }

    measurement = measure(idle)

    assert measurement["status"] == "FAIL"
    assert measurement["database_connections_max_during_window"] == 0.0
    assert "Aurora cluster aurora-a had zero DatabaseConnections" in str(
        measurement["reason"]
    )


@pytest.mark.parametrize(
    "connections",
    [None, {"error": "AccessDenied"}, series((1_000_560, 7.0, 7.0))],
    ids=["missing", "error", "outside-window"],
)
def test_no_connection_datapoint_in_the_window_leaves_it_unverified(
    connections: object,
) -> None:
    metrics = {**METRICS, "DatabaseConnections": connections}

    measurement = measure(metrics)

    assert measurement["status"] == "UNVERIFIED", measurement
    assert measurement["database_connections_max_during_window"] is None


def test_the_summary_maxima_keep_the_load_window_range_they_always_read() -> None:
    fields = database.aurora_step_fields(
        METRICS, placement=PLACEMENT, window_start=WINDOW[0], window_end=WINDOW[1]
    )

    assert fields["aurora_capacity_max"] == 128.0, (
        "from 120 s before the window to the end of the read, as before"
    )
    assert fields["aurora_cpu_max"] == 85.0
    assert fields["aurora_measurement"] == measure(METRICS)
    assert (
        database.aurora_step_fields(
            {}, placement=PLACEMENT, window_start=WINDOW[0], window_end=WINDOW[1]
        )["aurora_capacity_max"]
        is None
    ), "no datapoint is no maximum"


def test_a_failed_measurement_never_hides_the_steps_own_failure() -> None:
    failed = {"aurora_measurement": {"status": "FAIL"}}

    assert database.measured_exit(1, failed) == 1
    assert database.measured_exit(3, failed) == 3, "the budget limit keeps its code"
    assert database.measured_exit(0, failed) == database.MEASUREMENT_FAILED_EXIT == 4
    assert database.measured_exit(0, {"aurora_measurement": {"status": "PASS"}}) == 0
    assert database.measured_exit(0, {}) == 0
    assert database.step_status(0, "job failed") == {"status": "ok"}
    assert database.step_status(1, "job failed") == {
        "status": "aborted",
        "reason": "job failed",
    }
    assert database.step_status(4, "job failed", "zero connections") == {
        "status": "measurement-failed",
        "reason": "zero connections",
    }
    assert (
        database.step_status(4, "verdict failed")["reason"]
        == database.MEASUREMENT_FAILED_REASON
    )


def idle_cluster(reads: list[tuple[float, float]]) -> Any:
    """CloudWatch for a named cluster that serves nobody: 0.5 ACU, 0 links."""

    def window(start: float, end: float, *, writer: dict[str, Any]) -> dict:
        assert writer["writer_instance_id"] == "aurora-a-1", writer
        reads.append((start, end))
        minutes = range(int(start // 60) * 60, int(end) + 60, 60)
        return {
            "ServerlessDatabaseCapacity": series(*((at, 0.5, 0.5) for at in minutes)),
            "CPUUtilization": series(*((at, 1.0, 2.0) for at in minutes)),
            "DatabaseConnections": series(*((at, 0.0, 0.0) for at in minutes)),
        }

    return window


def test_the_burst_suite_reads_from_its_start_and_fails_the_measurement_not_the_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    wire = CapacityWire(tmp_path, monkeypatch)
    resolved_at = time.time()
    start = {**PLACEMENT, "observed_at": datetime.now(timezone.utc).isoformat()}
    monkeypatch.setattr(capacity, "start_placement", lambda *_args: start)
    reads: list[tuple[float, float]] = []
    monkeypatch.setattr(capacity, "aurora_window", idle_cluster(reads))

    assert capacity.main(capacity_args("all", tmp_path, "--lead-seconds", "0")) == 4

    summary = json.loads((tmp_path / "summary.json").read_text())
    status = json.loads((tmp_path / "status.json").read_text())
    assert summary["job_status"] == "Complete", "the product readings stay"
    assert summary["aurora_measurement"]["status"] == "FAIL"
    assert (
        summary["aurora_measurement"]["serverless_capacity_acu"]["at_step_start"] == 0.5
    ), "the capacity the step started on"
    assert status["status"] == "measurement-failed"
    assert "zero DatabaseConnections" in status["reason"]
    ((read_from, _read_to),) = reads
    assert read_from <= resolved_at + 1, "aurora.json reads from the step start"
    assert wire.teardowns, "the run still tears down"


def test_a_window_that_ends_before_it_starts_reads_its_first_minute() -> None:
    later = {
        **METRICS,
        "DatabaseConnections": series((1_000_320, 0.0, 0.0), (1_000_380, 5.0, 5.0)),
    }

    measurement = database.aurora_measurement(
        later, placement=PLACEMENT, window_start=1_000_350.0, window_end=1_000_330.0
    )

    assert measurement["database_connections_max_during_window"] == 0.0, (
        "only the minute holding the window start"
    )


def test_a_step_read_starts_at_most_the_lookback_before_its_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reads: list[tuple[float, float]] = []
    monkeypatch.setattr(capacity, "aurora_window", idle_cluster(reads))
    window_start = STEP_START + 2 * database.STEP_LOOKBACK_SECONDS

    evidence = capacity.aurora_step(
        window_start, window_start + 60, placement=PLACEMENT
    )

    assert reads == [
        (window_start - database.STEP_LOOKBACK_SECONDS, window_start + 60)
    ], "CloudWatch refuses a read of more than a day of minutes"
    assert (
        evidence["aurora_measurement"]["serverless_capacity_acu"]["at_step_start"]
        is None
    ), "a step start outside the read has no reading"


@pytest.mark.parametrize("named", [True, False], ids=["named", "unset"])
def test_the_action_suite_records_the_aurora_step_when_the_round_names_aurora(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, named: bool
) -> None:
    CapacityWire(tmp_path, monkeypatch)
    snapshot = {
        **PLACEMENT,
        "observed_at": datetime.now(timezone.utc).isoformat(),
        "control_plane": {"recorded": True},
    }
    monkeypatch.setattr(placement_module, "run_placement", lambda _id: dict(snapshot))
    monkeypatch.setattr(capacity, "aurora_window", idle_cluster([]))
    if named:
        monkeypatch.setenv("GPU_FAULT_AURORA_CLUSTER_ID", "aurora-a")

    result = action.main(action_args(tmp_path))

    summary = json.loads((tmp_path / "summary.json").read_text())
    status = json.loads((tmp_path / "status.json").read_text())
    if named:
        assert result == database.MEASUREMENT_FAILED_EXIT
        assert summary["aurora_measurement"]["status"] == "FAIL"
        assert status["status"] == "measurement-failed"
    else:
        assert result == 0
        assert summary["aurora_measurement"] == {
            "status": "UNVERIFIED",
            "reason": "GPU_FAULT_AURORA_CLUSTER_ID is not set",
        }, "a round that names no cluster measures none and says so"
        assert status["status"] == "ok"

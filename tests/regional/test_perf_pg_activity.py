"""The burst suite samples pg_stat_activity itself, through the load and drain.

A formal record must carry the sampler summary of the first two storm minutes
(active statement samples, the ``LWLock:BufferContent`` share, peak concurrent
admission INSERTs, the autovacuum share); the sampler ran outside the
repository and beside the integrated rounds only. The program itself runs
against PostgreSQL in ``tests/store/test_postgres_regional_protocol.py``.
"""

from __future__ import annotations

import json
import subprocess
import threading
import time
from collections.abc import Iterator
from typing import Any

import pytest

from scripts.perf import regional_capacity_pg_activity as activity
from scripts.perf import regional_capacity_registry as registry
from scripts.perf import regional_capacity_timeseries as timeseries


def sample(t: float, **fields: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "t": t,
        "active": sum((fields.get("waits") or {}).values()),
        "waits": {},
        "admission_inserts": 0,
        "autovacuum": 0,
        "statements": {},
    }
    return {**row, **fields}


SAMPLES = [
    sample(
        100.0,
        waits={"LWLock:BufferContent": 30, "CPU": 10},
        admission_inserts=28,
        statements={"INSERT INTO gpu_fault_processor_queue ( request_id": 28},
    ),
    {"t": 100.5, "error": "OperationalError"},
    sample(101.0, waits={"LWLock:BufferContent": 10, "IO:DataFileRead": 10}),
    sample(250.0, waits={"CPU": 4}, autovacuum=1, statements={"SELECT N": 4}),
]


def test_the_storm_minutes_and_the_whole_window_are_reduced_alike() -> None:
    summary = activity.activity_summary(SAMPLES, load_start=99.0)

    storm = summary["first_two_minutes"]
    assert summary["errors"] == 1
    assert storm["samples"] == 2, "the load's first 120 s only"
    assert storm["active_statement_samples"] == 60
    assert storm["buffer_content_share"] == pytest.approx(40 / 60, abs=0.001)
    assert storm["peak_admission_inserts"] == 28
    assert storm["peak_active"] == 40
    assert storm["mean_active"] == 30.0
    assert storm["samples_with_autovacuum"] == 0
    window = summary["window"]
    assert window["samples"] == 3
    assert window["autovacuum_sample_share"] == pytest.approx(1 / 3, abs=0.001)
    assert window["top_statements"] == {
        "INSERT INTO gpu_fault_processor_queue ( request_id": 28,
        "SELECT N": 4,
    }
    assert window["wait_shares"]["LWLock:BufferContent"] == pytest.approx(
        40 / 64, abs=0.001
    )
    assert activity.activity_summary([], load_start=None) == {
        "errors": 0,
        "window": activity.activity_summary([], load_start=0.0)["window"],
        "first_two_minutes": None,
    }, "no load start, no storm reading"


class FakeSession:
    """A remote sampler: samples until its stdin (the stop channel) closes."""

    def __init__(self, values: list[dict[str, Any]]) -> None:
        self.closed = threading.Event()
        self.written = b""
        self.stdin = self
        self.stdout = self._lines(values)

    def write(self, data: bytes) -> int:
        self.written += data
        return len(data)

    def flush(self) -> None:
        return None

    def close(self) -> None:
        self.closed.set()

    def kill(self) -> None:
        return None

    def _lines(self, values: list[dict[str, Any]]) -> Iterator[bytes]:
        for item in values:
            yield (json.dumps(item) + "\n").encode()
        self.closed.wait(timeout=30)


def test_one_api_pod_samples_the_database_until_the_run_stops_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(registry, "CONTROL_KUBECONFIG", "/tmp/cpu.kubeconfig")
    monkeypatch.setattr(registry, "CONTROL_NAMESPACE", "gpu-fault-system")
    calls: list[list[str]] = []
    sessions: list[FakeSession] = []

    def popen(argv: list[str], **kwargs: Any) -> FakeSession:
        assert kwargs["stdin"] is subprocess.PIPE, kwargs
        calls.append(argv)
        sessions.append(FakeSession(SAMPLES))
        return sessions[-1]

    sampler = activity.PgActivity(
        ["gpu-fault-control-worker-0", "gpu-fault-api-ha-1", "gpu-fault-api-ha-2"],
        popen=popen,
    ).start()
    deadline = time.monotonic() + 10
    while (
        sampler.session is not None
        and len(sampler.session.samples) < len(SAMPLES)
        and time.monotonic() < deadline
    ):
        time.sleep(0.01)
    sampler.stop()

    ((argv),) = calls
    assert argv[argv.index("-i") + 1] == "gpu-fault-api-ha-1", "one api-ha Pod"
    assert {"SAMPLE_INTERVAL=0.5", "SAMPLE_LIMIT=7200", "SAMPLE_TOP=5"} <= set(argv)
    assert argv[-3:] == ["python3", "-c", timeseries.BOOTSTRAP]
    (session,) = sessions
    assert session.written == timeseries.session_input(activity.program())
    assert session.closed.is_set(), "stop() closes the stop channel"
    program = activity.program()
    assert "def store_dsn" in program and "{STORE_DSN_SNIPPET}" not in program, (
        "the program reads the store's own DSN"
    )
    compile(program, "pg_activity.py", "exec")
    document = sampler.document()
    assert document["pod"] == "gpu-fault-api-ha-1"
    assert len(document["samples"]) == len(SAMPLES)
    assert document["error"] is None
    summary = sampler.summary(99.0)
    assert summary["first_two_minutes"]["peak_admission_inserts"] == 28


def test_without_an_api_pod_the_record_says_why_it_sampled_nothing() -> None:
    def popen(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("no session without an api-ha Pod")

    sampler = activity.PgActivity(["gpu-fault-control-worker-0"], popen=popen).start()
    sampler.stop()

    assert sampler.document()["error"] == "no api-ha Pod"
    assert sampler.summary(1.0)["window"]["samples"] == 0

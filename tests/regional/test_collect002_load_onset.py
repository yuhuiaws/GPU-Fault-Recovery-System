"""COLLECT-002 measures from the load's observed onset, not from its exec.

Live (HyperPod, H200, host DCGM 4.7.0, exporter ``-c 15000``, interval 15 s,
three confirmation samples) the case failed a healthy node: 63.4 s from the
load-start receipt against the 61.6-s staged bound. Every product stage was
nominal -- ticks 15.088 s apart, the confirming sample two ticks after the
first anomalous one, 0.087 s to the Store -- and the whole excess sat between
the receipt and the first anomalous sample: 33.1 s against the 31 s that NVML,
the DCGM watch and one collector tick allow. That stage holds
``dcgmproftester``'s own start-up (exec, an embedded DCGM over every GPU, a
forked CUDA worker with its context, PTX module and cuBLAS), which no source
bounds. The throttle probe now reads the loaded GPU after the receipt until a
read reaches the collector's correlation thresholds, and the latency runs from
the last read still below them; the pipeline bound is unchanged.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from gpu_fault.gpu_metrics import GpuMetricsThresholds
from scripts.e2e.regional import collect002_latency_bound as bound
from scripts.e2e.regional import run_collector_acceptance as runner
from scripts.e2e.regional.probes import collector_node_probe as probe
from tests.regional._collector_power_receipt import candidate_record, load_receipt

#: The failed live attempt, from its evidence: the opening and quiet GPU-chain
#: deliveries, the request, the load-start receipt, the confirming sample and its
#: ingest (payload, then the stored record).
OPENING = datetime(2026, 10, 21, 5, 5, 0, 797849, tzinfo=timezone.utc)
QUIET = datetime(2026, 10, 21, 5, 10, 32, 186380, tzinfo=timezone.utc)
REQUESTED = datetime(2026, 10, 21, 5, 10, 46, 826969, tzinfo=timezone.utc)
RECEIPT = datetime(2026, 10, 21, 5, 10, 59, 397867, tzinfo=timezone.utc)
CONFIRMED = datetime(2026, 10, 21, 5, 12, 2, 714089, tzinfo=timezone.utc)
INGESTED = datetime(2026, 10, 21, 5, 12, 2, 800666, tzinfo=timezone.utc)
STORED = datetime(2026, 10, 21, 5, 12, 2, 868331, tzinfo=timezone.utc)
TICK = (CONFIRMED - QUIET) / 6
#: The tick before the first anomalous one read a DCGM sample at most one
#: 15-s watch old that did not carry the load, so NVML crossed the thresholds
#: no earlier than 3.05 s after the receipt: the earliest onset the evidence
#: allows, read 0.4 s apart.
IDLE_READ = RECEIPT + timedelta(seconds=2.65)
LOADED_READ = RECEIPT + timedelta(seconds=3.05)
LIVE_COMPONENTS = (
    "nvml_average 1s, dcgm_watch 15s, collector_phase 15s, confirmation 30s, "
    "collector_tick_overrun 0.528s, delivery 0.087s"
)
EXPORTER_POD = {
    "metadata": {"name": "gpu-fault-dcgm-exporter-dlm2s"},
    "status": {"phase": "Running"},
    "spec": {
        "containers": [
            {
                "name": "exporter",
                "args": ["-a", "127.0.0.1:9400", "-c", "15000"],
                "env": [
                    {"name": "DCGM_REMOTE_HOSTENGINE_INFO", "value": "127.0.0.1:5555"}
                ],
            }
        ]
    },
}


def onset_reads(
    idle: datetime | None = IDLE_READ, loaded: datetime | None = LOADED_READ
) -> dict[str, Any]:
    return {
        "idle_read_at": None if idle is None else idle.isoformat(),
        "loaded_read_at": None if loaded is None else loaded.isoformat(),
    }


def run_live_attempt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    reads: dict[str, Any] | None,
    ticks_late: int = 0,
    delivery_seconds: float | None = None,
) -> dict[str, Any]:
    """``run_collect002`` on the live attempt's stamps, with this probe's reads."""

    confirmed = CONFIRMED + ticks_late * TICK
    ingested = confirmed + (
        INGESTED - CONFIRMED
        if delivery_seconds is None
        else timedelta(seconds=delivery_seconds)
    )
    record = candidate_record(confirmed)
    record["payload"]["ingested_at"] = ingested.isoformat()
    record["ingested_at"] = (ingested + (STORED - INGESTED)).isoformat()
    for sample in record["payload"]["samples"]:
        sample["value"] = {"power_usage_w": 207.8}.get(
            sample["canonical_name"], sample["value"]
        )
    current = [REQUESTED]
    monkeypatch.setattr(
        runner,
        "datetime",
        SimpleNamespace(
            now=lambda tz=None: current[0], fromisoformat=datetime.fromisoformat
        ),
    )
    monkeypatch.setattr(runner, "seconds_until_next_summary", lambda *a, **k: 0)
    monkeypatch.setattr(runner, "require_action_time", lambda *a: None)
    monkeypatch.setattr(runner, "recent_evidence", lambda *a, **k: [record])
    stamps = iter([OPENING, QUIET, confirmed])

    def stamp(_fixture: object) -> str:
        value = next(stamps)
        if value == confirmed:
            current[0] = confirmed + timedelta(seconds=29.9)
        return value.isoformat()

    monkeypatch.setattr(runner, "gpu_metrics_stamp", stamp)

    def execute(command: str, *arguments: str, **kwargs: object) -> dict[str, Any]:
        if command == "throttle-gpu":
            receipt = load_receipt(arguments[arguments.index("--run-id") + 1], RECEIPT)
            current[0] = LOADED_READ + timedelta(seconds=0.2)
            return {
                **receipt,
                **({} if reads is None else {"load_onset": reads}),
                "mutation_started": True,
                "timer_armed": True,
                "load_unit_active": True,
            }
        current[0] += timedelta(seconds=10)
        return {
            "restored": True,
            "load_stopped": True,
            "timer_disarmed": True,
            "cleanup_verified": True,
        }

    fixture = SimpleNamespace(
        node="node-a",
        regional=SimpleNamespace(
            settings=SimpleNamespace(cluster_id="cluster-a"),
            kubectl=lambda *a, **k: json.dumps({"items": [EXPORTER_POD]}),
        ),
        execute=execute,
        snapshot=lambda: {
            "collector_env": {
                "GPU_FAULT_METRICS_INTERVAL_SECONDS": "15",
                "GPU_FAULT_DCGM_HEALTH_SUMMARY_SECONDS": "300",
                "GPU_FAULT_DCGM_EDGE_CONFIRMATION_SAMPLES": "3",
            },
            "gpu_power": [
                {"index": 0, "power_limit_w": 700.0, "power_default_limit_w": 700.0}
            ],
        },
    )
    return runner.run_collect002(fixture, tmp_path, 1)


def test_the_live_attempt_passes_once_the_latency_runs_from_the_observed_onset(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    result = run_live_attempt(monkeypatch, tmp_path, reads=onset_reads())

    assert result["verdict"] == "PASS", result["errors"]
    breakdown = result["allowed_latency_breakdown"]
    assert breakdown["components"] == {
        "nvml_average_seconds": 1.0,
        "dcgm_watch_seconds": 15.0,
        "collector_phase_seconds": 15.0,
        "confirmation_seconds": 30.0,
        "collector_tick_overrun_seconds": 0.528,
        "delivery_seconds": 0.087,
        "load_onset_excess_seconds": 0.0,
    }
    assert result["allowed_latency_seconds"] == 61.615, "the pipeline bound is kept"
    assert result["delivery_latency_seconds"] == pytest.approx(63.402799 - 2.65)
    assert breakdown["load_onset"] == {
        "source": "load-onset",
        "measured_from": IDLE_READ.isoformat(),
        "window_seconds": 0.4,
        "excluded_seconds": 2.65,
        "reads": onset_reads(),
    }
    assert breakdown["stages_without_allowance"] == [], "the start-up is excluded"
    assert result["load_started_at"] == RECEIPT.isoformat()
    assert result["preparation_seconds"] == pytest.approx(12.570898)


def test_without_onset_reads_the_live_attempt_still_fails_from_the_receipt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An older probe's receipt: nothing pins the onset, the start-up stays
    unpriced and the live verdict is reproduced, naming its reference."""

    result = run_live_attempt(monkeypatch, tmp_path, reads=None)

    assert result["verdict"] == "FAIL"
    assert result["errors"] == [
        "the anomaly took 63.4s to deliver, more than the 61.6s bound "
        f"({LIVE_COMPONENTS}), measured from the receipt"
    ]
    breakdown = result["allowed_latency_breakdown"]
    assert breakdown["stages_without_allowance"] == [bound.WITHOUT_ALLOWANCE]
    assert breakdown["load_onset"]["source"] == "load-start-receipt"
    assert breakdown["load_onset"]["measured_from"] == RECEIPT.isoformat()


def test_a_first_anomalous_sample_two_ticks_late_still_fails_from_the_onset(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    result = run_live_attempt(monkeypatch, tmp_path, reads=onset_reads(), ticks_late=2)

    assert result["verdict"] == "FAIL"
    assert result["allowed_latency_seconds"] == pytest.approx(61.791)
    assert result["delivery_latency_seconds"] == pytest.approx(90.93, abs=0.01)
    assert len(result["errors"]) == 1, result["errors"]
    assert result["errors"][0].startswith("the anomaly took 90.9s to deliver"), result[
        "errors"
    ]
    assert result["errors"][0].endswith("measured from the load onset read"), (
        "the reference is named"
    )


def test_a_delivery_beyond_one_interval_still_fails_from_the_onset(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    result = run_live_attempt(
        monkeypatch, tmp_path, reads=onset_reads(), delivery_seconds=16.0
    )

    assert result["verdict"] == "FAIL"
    assert result["errors"] == [
        "the confirming batch took 16.0s from its sample to the Store, more than "
        "one 15s collector interval: a slow or retried delivery, not sampling"
    ]


def test_an_onset_read_gap_wider_than_nvml_s_second_adds_only_its_excess() -> None:
    record = candidate_record(CONFIRMED)
    record["payload"]["ingested_at"] = INGESTED.isoformat()
    delivery = {
        "record_ids": [record["record_id"]],
        "delivered_at": INGESTED.isoformat(),
    }
    exporter = bound.exporter_interval([EXPORTER_POD])
    onset = bound.load_onset(
        {"load_onset": onset_reads(loaded=IDLE_READ + timedelta(seconds=2.5))}, RECEIPT
    )

    result = bound.allowed_latency(
        15, 3, exporter, QUIET.isoformat(), [record], delivery, onset
    )

    assert onset["window_seconds"] == 2.5
    assert result["components"]["load_onset_excess_seconds"] == 1.5
    assert result["total_seconds"] == pytest.approx(61.615 + 1.5)
    latency = bound.measured_latency(INGESTED.isoformat(), result)
    assert latency == pytest.approx(63.402799 - 2.65)
    assert bound.latency_errors(latency, result, interval=15) == []


@pytest.mark.parametrize(
    ("reads", "detail"),
    [
        (None, "the throttle receipt carries no onset reads"),
        ("idle", "the throttle receipt carries no onset reads"),
        (
            {"idle_read_at": IDLE_READ.isoformat(), "loaded_read_at": None},
            "no read reached the thresholds: the probe's wait ended first",
        ),
        (
            {"idle_read_at": None, "loaded_read_at": None, "error": "ProbeError: x"},
            "no read reached the thresholds: ProbeError: x",
        ),
        (onset_reads(idle=RECEIPT - timedelta(seconds=1)), "out of order"),
        (onset_reads(idle=LOADED_READ), "out of order"),
        (onset_reads(loaded=LOADED_READ.replace(tzinfo=None)), "malformed"),
        ({"idle_read_at": "soon", "loaded_read_at": LOADED_READ.isoformat()}, "order"),
    ],
)
def test_reads_that_do_not_pin_the_onset_leave_the_receipt_as_the_reference(
    reads: object, detail: str
) -> None:
    onset = bound.load_onset({} if reads is None else {"load_onset": reads}, RECEIPT)

    assert onset["source"] == "load-start-receipt"
    assert onset["measured_from"] == RECEIPT.isoformat()
    assert detail in onset["detail"]
    assert onset["excluded_seconds"] == 0.0


def test_a_first_read_already_at_the_thresholds_measures_from_the_receipt() -> None:
    onset = bound.load_onset({"load_onset": onset_reads(idle=None)}, RECEIPT)

    assert onset["source"] == "load-onset"
    assert onset["measured_from"] == RECEIPT.isoformat()
    assert onset["window_seconds"] == 3.05, "the whole start-up is the first stage"
    assert onset["excluded_seconds"] == 0.0


class Reads:
    """``gpu_power_state`` answers on a fake clock that each read advances."""

    def __init__(
        self, readings: list[tuple[float, float] | Exception], *, seconds: float = 0.2
    ) -> None:
        self.readings = readings
        self.seconds = seconds
        self.now = 100.0
        self.epoch = RECEIPT
        self.calls: list[bool] = []

    def state(self, *, strict: bool = False) -> list[dict[str, Any]]:
        self.calls.append(strict)
        self.now += self.seconds
        reading = self.readings[min(len(self.calls), len(self.readings)) - 1]
        if isinstance(reading, Exception):
            raise reading
        utilization, draw = reading
        return [
            {"index": 1, "power_draw_w": 75.0, "power_limit_w": 200.0},
            {
                "index": 0,
                "utilization_percent": utilization,
                "power_draw_w": draw,
                "power_limit_w": 200.0,
            },
        ]

    def utc(self, tz: object = None) -> datetime:
        return self.epoch + timedelta(seconds=self.now - 100.0)

    def sleep(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture
def reads(monkeypatch: pytest.MonkeyPatch) -> Reads:
    fake = Reads([])
    monkeypatch.setattr(probe, "gpu_power_state", fake.state)
    monkeypatch.setattr(
        probe, "time", SimpleNamespace(monotonic=lambda: fake.now, sleep=fake.sleep)
    )
    monkeypatch.setattr(
        probe,
        "datetime",
        SimpleNamespace(now=fake.utc, fromisoformat=datetime.fromisoformat),
    )
    return fake


def stamp_of(fake: Reads, seconds: float) -> str:
    return (fake.epoch + timedelta(seconds=seconds)).isoformat()


def test_the_probe_reads_until_the_loaded_gpu_reaches_both_thresholds(
    reads: Reads,
) -> None:
    thresholds = GpuMetricsThresholds()
    utilization = thresholds.power_correlation_min_utilization_percent
    at_cap = thresholds.power_limit_ratio * 200.0
    reads.readings = [(0.0, 75.0), (99.0, at_cap - 0.5), (utilization, at_cap)]

    onset = probe.power_load_onset({"gpu_index": 0})

    assert reads.calls == [True, True, True], "strict reads of the power query"
    # Reads take 0.2 s and 0.1 s pass between them: the third read starts at
    # 0.6 s and ends at 0.8 s; it is the first at the collector's defaults.
    assert onset == {
        "idle_read_at": stamp_of(reads, 0.3),
        "loaded_read_at": stamp_of(reads, 0.8),
    }


def test_a_load_that_never_reaches_the_thresholds_ends_the_reads_at_the_wait(
    reads: Reads,
) -> None:
    reads.readings = [(79.0, 200.0)]

    onset = probe.power_load_onset({"gpu_index": 0})

    assert onset["loaded_read_at"] is None
    assert onset["idle_read_at"] is not None
    assert reads.now - 100.0 == pytest.approx(
        probe.POWER_LOAD_START_WAIT_SECONDS, abs=0.31
    ), "bounded by the receipt's wait"


def test_a_failed_read_is_named_and_never_raised(reads: Reads) -> None:
    reads.readings = [(0.0, 75.0), probe.ProbeError("GPU power query failed")]

    onset = probe.power_load_onset({"gpu_index": 0})

    assert onset == {
        "idle_read_at": stamp_of(reads, 0.0),
        "loaded_read_at": None,
        "error": "ProbeError: GPU power query failed",
    }
    reads.readings, reads.calls = [(0.0, 75.0)], []
    missing = probe.power_load_onset({"gpu_index": 7})
    assert missing["error"].startswith("KeyError"), missing

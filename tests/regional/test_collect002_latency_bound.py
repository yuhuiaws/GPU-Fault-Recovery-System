"""COLLECT-002's latency bound: one term per stage of the sampling chain.

Live (plain EKS, the first run with the node's host DCGM): a healthy node
delivered the power anomaly 61.3 s after the load start against the old 60-s
bound, ``interval x (confirmation + 1)``; the same node and release passed at
46.6 s on the next attempt. The edge filter needs ``confirmation`` samples, the
first of which is the anomaly's own, so the old formula's spare period was
silently the DCGM watch period, and nothing was left for NVML's 1-s average,
the collector loop's own work or the delivery. These tests pin the bound that
names every stage (``scripts/e2e/regional/collect002_latency_bound.py``).
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from gpu_fault.collectors.gpu.dcgm import DcgmMetricsCollector
from gpu_fault.collectors.models import CollectorContext
from gpu_fault.dcgm_exporter_cadence import DCGM_EXPORTER_COLLECT_INTERVAL_MS
from scripts.e2e.regional import collect002_latency_bound as bound

#: The live attempt that failed: the quiet delivery, the confirming sample and
#: its ingest, all from the run's record (host DCGM mode, ``-c 15000``).
QUIET = "2026-10-17T19:16:30.870000+00:00"
CONFIRMED = datetime(2026, 10, 17, 19, 17, 47, 460000, tzinfo=timezone.utc)
INGESTED = CONFIRMED + timedelta(seconds=0.98)
LOAD_START = datetime(2026, 10, 17, 19, 16, 47, 120000, tzinfo=timezone.utc)
RECORD_ID = "gpu-metrics/dcgm-node-a-1"


def exporter_pod(
    *,
    args: list[str] | None = None,
    env: dict[str, str] | None = None,
    name: str = "gpu-fault-dcgm-exporter-a",
    phase: str = "Running",
) -> dict[str, Any]:
    return {
        "metadata": {"name": name},
        "status": {"phase": phase},
        "spec": {
            "containers": [
                {
                    "name": "exporter",
                    "args": ["-a", "127.0.0.1:9400", "-c", "15000"]
                    if args is None
                    else args,
                    "env": [
                        {"name": key, "value": value}
                        for key, value in (
                            {"DCGM_REMOTE_HOSTENGINE_INFO": "127.0.0.1:5555"}
                            if env is None
                            else env
                        ).items()
                    ],
                }
            ]
        },
    }


def live_bound(exporter: dict[str, Any]) -> dict[str, Any]:
    record = {"record_id": RECORD_ID, "payload": {"observed_at": CONFIRMED.isoformat()}}
    delivery = {"record_ids": [RECORD_ID], "delivered_at": INGESTED.isoformat()}
    return bound.allowed_latency(15, 3, exporter, QUIET, [record], delivery)


def test_the_bound_names_one_term_per_stage_and_passes_the_live_attempt() -> None:
    result = live_bound(bound.exporter_interval([exporter_pod()]))

    assert result["components"] == {
        "nvml_average_seconds": 1.0,
        "dcgm_watch_seconds": 15.0,
        "collector_phase_seconds": 15.0,
        "confirmation_seconds": 30.0,
        "collector_tick_overrun_seconds": 1.59,
        "delivery_seconds": 0.98,
    }
    assert result["total_seconds"] == 63.57
    assert result["collector"]["window"] == {
        "quiet_stamp": QUIET,
        "confirming_sample_at": CONFIRMED.isoformat(),
        "seconds": 76.59,
        "ticks": 5,
        "overrun_seconds": 1.59,
    }
    assert result["exporter"]["source"] == "pod-args"
    assert result["exporter"]["dcgm_hostengine"] == "host"
    latency = (INGESTED - LOAD_START).total_seconds()
    assert latency > 15 * (3 + 1), "the old bound failed this healthy node"
    assert bound.latency_errors(latency, result, interval=15) == []
    assert result["stages_without_allowance"] == [bound.WITHOUT_ALLOWANCE]


def test_embedded_and_host_dcgm_share_one_bound() -> None:
    """Both modes watch at the exporter's period and both read the cache:
    the host engine only ever samples at the shortest of its watchers'
    periods, so the mode is recorded, never priced."""

    host = live_bound(bound.exporter_interval([exporter_pod()]))
    embedded = live_bound(bound.exporter_interval([exporter_pod(env={})]))

    assert embedded["exporter"]["dcgm_hostengine"] == "embedded"
    assert host["exporter"]["dcgm_hostengine"] == "host"
    assert embedded["components"] == host["components"]
    assert embedded["total_seconds"] == host["total_seconds"]
    flagged = bound.exporter_interval(
        [exporter_pod(args=["-r", "10.0.0.1:5555", "-c", "15000"], env={})]
    )
    assert flagged["dcgm_hostengine"] == "host"


@pytest.mark.parametrize(
    ("args", "env", "milliseconds", "source"),
    [
        (["-c", "15000"], {}, 15000, "pod-args"),
        (["-c=20000"], {}, 20000, "pod-args"),
        (["--collect-interval", "25000"], {}, 25000, "pod-args"),
        (["--collect-interval=5000"], {}, 5000, "pod-args"),
        (["-collect-interval", "7000"], {}, 7000, "pod-args"),
        (["-c", "10000", "-c", "12000"], {}, 12000, "pod-args"),
        ([], {"DCGM_EXPORTER_INTERVAL": "18000"}, 18000, "pod-env"),
        (["-c", "16000"], {"DCGM_EXPORTER_INTERVAL": "18000"}, 16000, "pod-args"),
        (["-a", "127.0.0.1:9400"], {}, 30000, "exporter-default"),
        (["--", "-c", "1000"], {}, 30000, "exporter-default"),
        (["-c", "15s"], {}, 30000, "pod-args-unresolved"),
        ([], {"DCGM_EXPORTER_INTERVAL": ""}, 30000, "pod-env-unresolved"),
    ],
)
def test_the_collect_interval_is_read_the_way_the_exporter_reads_it(
    args: list[str], env: dict[str, str], milliseconds: int, source: str
) -> None:
    """Flag (one or two dashes, ``=`` or a separate value, last one wins),
    then ``DCGM_EXPORTER_INTERVAL``, then the exporter's own 30-s default --
    urfave/cli's order. A value the exporter could not parse either is not
    guessed: the documented default applies and the source says so."""

    result = bound.exporter_interval([exporter_pod(args=args, env=env)])

    assert result["collect_interval_ms"] == milliseconds
    assert result["source"] == source


def test_a_missing_collect_interval_widens_only_the_dcgm_watch_term() -> None:
    measured = live_bound(bound.exporter_interval([exporter_pod()]))
    missing = live_bound(
        bound.exporter_interval([exporter_pod(args=["-a", "127.0.0.1:9400"])])
    )

    assert missing["components"]["dcgm_watch_seconds"] == 30.0
    assert missing["total_seconds"] == measured["total_seconds"] + 15
    assert {
        key: value
        for key, value in missing["components"].items()
        if key != "dcgm_watch_seconds"
    } == {
        key: value
        for key, value in measured["components"].items()
        if key != "dcgm_watch_seconds"
    }


class Regional:
    def __init__(self, answer: object) -> None:
        self.answer = answer
        self.calls: list[tuple[tuple[str, ...], dict[str, object]]] = []

    def kubectl(self, *arguments: str, **options: object) -> str:
        self.calls.append((arguments, options))
        if isinstance(self.answer, Exception):
            raise self.answer
        return json.dumps(self.answer)


class Fixture:
    def __init__(self, answer: object) -> None:
        self.regional = Regional(answer)
        self.node = "node-a"


def test_the_exporter_is_read_from_the_pods_on_the_target_node() -> None:
    fixture = Fixture(
        {
            "items": [
                exporter_pod(name="old", args=["-c", "30000"]),
                exporter_pod(name="new"),
                exporter_pod(name="pending", args=["-c", "60000"], phase="Pending"),
            ]
        }
    )

    result = bound.read_exporter(fixture)

    assert fixture.regional.calls == [
        (
            (
                "gpu",
                "get",
                "pod",
                "-l",
                "app=gpu-fault-dcgm-exporter",
                "--field-selector",
                "spec.nodeName=node-a",
                "-o",
                "json",
            ),
            {"timeout": 60},
        )
    ]
    assert result["pods"] == ["new", "old"], "only running Pods serve the scrape"
    assert result["collect_interval_ms"] == 30000, "a roll in flight takes the longer"
    assert result["rendered_collect_interval_ms"] == DCGM_EXPORTER_COLLECT_INTERVAL_MS


@pytest.mark.parametrize(
    ("answer", "source"),
    [
        (RuntimeError("forbidden: pods is forbidden"), "unreadable"),
        ({"kind": "Status"}, "unreadable"),
        ({"items": []}, "no-exporter-pod"),
        ({"items": [exporter_pod(phase="Pending")]}, "no-exporter-pod"),
    ],
)
def test_an_unreadable_or_absent_exporter_takes_the_documented_default(
    answer: object, source: str
) -> None:
    result = bound.read_exporter(Fixture(answer))

    assert result["collect_interval_ms"] == bound.EXPORTER_DEFAULT_INTERVAL_MS
    assert result["source"] == source
    assert result["detail"]
    assert bound.EXPORTER_DEFAULT_INTERVAL_MS >= DCGM_EXPORTER_COLLECT_INTERVAL_MS, (
        "the assumption may over-allow the product's exporter, never under-allow"
    )


def test_the_collector_window_counts_ticks_and_measures_their_work() -> None:
    quiet = datetime(2026, 10, 1, tzinfo=timezone.utc)

    window = bound.collector_window(
        quiet.isoformat(), quiet + timedelta(seconds=61), 15
    )
    assert (window["ticks"], window["overrun_seconds"]) == (4, 1.0)
    exact = bound.collector_window(quiet.isoformat(), quiet + timedelta(seconds=45), 15)
    assert (exact["ticks"], exact["overrun_seconds"]) == (3, 0.0)
    # A loop that lost a whole interval reads as one tick more: the allowance
    # shrinks (the case fails on the stalled collector), it never widens.
    stalled = bound.collector_window(
        quiet.isoformat(), quiet + timedelta(seconds=76), 15
    )
    assert (stalled["ticks"], stalled["overrun_seconds"]) == (5, 1.0)
    assert bound.collector_window(quiet.isoformat(), None, 15)["overrun_seconds"] == 0


def test_a_slow_delivery_is_named_apart_from_sampling() -> None:
    record = {"record_id": RECORD_ID, "payload": {"observed_at": CONFIRMED.isoformat()}}
    late = (CONFIRMED + timedelta(seconds=17)).isoformat()
    result = bound.allowed_latency(
        15,
        3,
        bound.exporter_interval([exporter_pod()]),
        QUIET,
        [record],
        {"record_ids": [RECORD_ID], "delivered_at": late},
    )

    errors = bound.latency_errors(60.0, result, interval=15)

    assert len(errors) == 1
    assert "took 17.0s from its sample to the Store" in errors[0]
    assert "slow or retried delivery, not sampling" in errors[0]


def test_a_late_anomaly_names_every_component_of_the_bound() -> None:
    result = live_bound(bound.exporter_interval([exporter_pod()]))

    errors = bound.latency_errors(78.0, result, interval=15)

    assert errors == [
        "the anomaly took 78.0s to deliver, more than the 63.6s bound "
        "(nvml_average 1s, dcgm_watch 15s, collector_phase 15s, confirmation 30s, "
        "collector_tick_overrun 1.59s, delivery 0.98s), measured from the receipt"
    ]
    assert bound.latency_errors(-1.0, result, interval=15), "a negative latency fails"


def test_no_delivery_leaves_only_the_configured_terms() -> None:
    result = bound.allowed_latency(
        15,
        3,
        bound.exporter_interval([exporter_pod()]),
        QUIET,
        [],
        {"record_ids": [], "delivered_at": None},
    )

    assert result["total_seconds"] == 61.0
    assert result["components"]["collector_tick_overrun_seconds"] == 0
    assert result["components"]["delivery_seconds"] == 0
    assert result["collector"]["window"]["ticks"] is None


def test_the_scrape_tick_the_bound_prices_carries_no_cadence_jitter(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """The bound's collector terms are whole intervals plus the measured work:
    true only while the metrics loop sleeps exactly its interval. The cadence
    jitter moves the health summary, the inventory and the start-up; if it
    ever reaches this sleep, the collector terms need its bound."""

    monkeypatch.setenv("GPU_FAULT_COLLECTOR_CADENCE_JITTER_RATIO", "0.5")
    request = tmp_path / "gpu.request"
    request.write_text("trigger\n", encoding="ascii")

    class Sink:
        def post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
            return {"accepted": True}

    collector = DcgmMetricsCollector(
        Sink(),
        CollectorContext(cluster_id="cluster-a"),
        node_id="node-a",
        interval_seconds=15,
        force_snapshot_path=str(request),
    )
    sleeps: list[float] = []

    def sleep(seconds: float) -> None:
        sleeps.append(seconds)
        if len(sleeps) == 4:
            raise KeyboardInterrupt

    monkeypatch.setattr(collector, "collect_once", lambda: None)
    monkeypatch.setattr("gpu_fault.collectors.gpu.dcgm.time.sleep", sleep)
    with pytest.raises(KeyboardInterrupt):
        collector.run()

    assert sleeps == [15, 15, 15, 15]

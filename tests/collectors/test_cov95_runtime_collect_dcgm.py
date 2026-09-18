"""DCGM scrape lifecycle and counter edges over fake transports."""

from __future__ import annotations

import sys
from datetime import timedelta
from types import SimpleNamespace

import pytest

from gpu_fault.channel_registry import GPU_METRICS_PATH
from gpu_fault.collectors.gpu import dcgm
from gpu_fault.collectors.sinks import CollectorError
from tests.collectors import _cov95_runtime_collect as support
from tests.collectors._cov95_runtime_collect_gpu import gpu_runner

isolated_runtime = support.isolated_runtime


def collector(sink=None, **kwargs):
    return dcgm.DcgmMetricsCollector(
        sink if sink is not None else support.RecordingSink(),
        support.collector_context(),
        node_id="node-a",
        runner=gpu_runner,
        **kwargs,
    )


@pytest.mark.parametrize(
    "options",
    [
        {"exporter_interval_seconds": 0},
        {"history_max_points": 0},
        {"failure_backoff_threshold": 0},
    ],
)
def test_invalid_dcgm_cadence_is_rejected_before_probes(options):
    with pytest.raises(ValueError, match="positive"):
        collector(**options)


def test_missing_prometheus_parser_is_an_explicit_collection_failure(monkeypatch):
    reader = collector()
    monkeypatch.setitem(sys.modules, "prometheus_client.parser", None)
    with pytest.raises(CollectorError, match="DCGM Prometheus parsing"):
        reader.collect_text('DCGM_FI_DEV_GPU_TEMP{UUID="GPU-a"} 20\n')


def test_counter_reset_or_nonforward_clock_does_not_confirm_a_throttle():
    reader = collector(edge_confirmation_samples=1)
    texts = [
        'DCGM_FI_DEV_POWER_VIOLATION{UUID="GPU-a"} 1000000000\n',
        'DCGM_FI_DEV_POWER_VIOLATION{UUID="GPU-a"} 2000000000\n',
        'DCGM_FI_DEV_POWER_VIOLATION{UUID="GPU-a"} 1000\n',
    ]
    batches = [
        reader.collect_text(text, observed_at=when)
        for text, when in zip(
            texts,
            [support.NOW, support.NOW, support.NOW + timedelta(seconds=1)],
            strict=True,
        )
    ]
    assert all(
        "candidate-confirmed" not in batch.edge_filter_reasons for batch in batches
    ), "clock ties and counter resets are not observed throttling intervals"


def test_changed_xid_is_delivered_even_without_threshold_confirmation():
    sink = support.RecordingSink()
    reader = collector(sink, edge_confirmation_samples=3)
    reader.collect_text(
        'DCGM_FI_DEV_XID_ERRORS{UUID="GPU-a"} 0\n', observed_at=support.NOW
    )
    changed = reader.collect_text(
        'DCGM_FI_DEV_XID_ERRORS{UUID="GPU-a"} 13\n',
        observed_at=support.NOW + timedelta(seconds=1),
    )
    assert "xid-changed" in changed.edge_filter_reasons
    assert len(sink.requests) == 2


@pytest.mark.parametrize("failed_report", [False, True])
@pytest.mark.parametrize("startup_grace_seconds", [None, 0])
def test_run_retains_force_request_on_failure_and_consumes_it_after_success(
    monkeypatch, tmp_path, failed_report, startup_grace_seconds, caplog
):
    clock = support.Clock()
    marker = tmp_path / "gpu.request"
    marker.write_text("requested")
    sink = support.RecordingSink(
        CollectorError("fake error report rejected") if failed_report else None
    )
    reader = collector(
        sink,
        force_snapshot_path=str(marker),
        now=clock.now,
        startup_grace_seconds=startup_grace_seconds,
    )
    scrapes = []
    waits = []

    def scrape(url, **kwargs):
        scrapes.append(url)
        if len(scrapes) == 1:
            raise OSError("fake exporter unavailable")
        return support.Response(body=b'DCGM_FI_DEV_GPU_TEMP{UUID="GPU-a"} 20\n')

    def sleep(seconds):
        waits.append(seconds)
        if len(waits) == 1:
            assert marker.exists(), "failed scrape must retain the requested snapshot"
            sink.error = None
            clock.sleep(seconds)
        else:
            raise support.StopLoop

    monkeypatch.setattr(dcgm, "urlopen", scrape)
    monkeypatch.setattr(dcgm, "time", SimpleNamespace(sleep=sleep))
    with pytest.raises(support.StopLoop):
        reader.run()
    assert not marker.exists(), "successful retry must consume its force request"
    assert len(scrapes) == 2, "the failed scrape must be retried once"
    assert waits == [15, 15], "one failed scrape must not increase the retry interval"
    scrape_errors = [
        payload
        for path, payload in sink.requests
        if path == GPU_METRICS_PATH and not payload["samples"]
    ]
    if startup_grace_seconds is None:
        assert scrape_errors == [], (
            "default startup grace must not report the initial exporter outage"
        )
        assert "inside the 300s startup grace" in caplog.text, (
            "the suppressed startup failure must remain observable"
        )
    else:
        assert len(scrape_errors) == 1, (
            "outside startup grace the failed scrape must report"
        )
        assert scrape_errors[0]["edge_filter_reasons"] == ["collection-error"], (
            "exporter unavailability is a collection failure, not hardware evidence"
        )
        assert len(scrape_errors[0]["collection_errors"]) == 1 and (
            "cannot scrape DCGM exporter" in scrape_errors[0]["collection_errors"][0]
        ), "the error-only batch must identify the failed exporter scrape"
    assert ("DCGM collection error report was not delivered" in caplog.text) is (
        failed_report and startup_grace_seconds == 0
    ), "delivery failure is logged only when an error report was actually attempted"
    snapshots = [
        payload
        for path, payload in sink.requests
        if path == GPU_METRICS_PATH and payload["samples"]
    ]
    assert len(snapshots) == 1, (
        "force-request consumption must follow delivery of real parsed samples"
    )
    assert [
        (sample["gpu_uuid"], sample["value"])
        for sample in snapshots[0]["samples"]
        if sample["canonical_name"] == "gpu_temperature_c"
    ] == [("GPU-a", 20.0)], "the retried scrape must deliver the observed temperature"
    assert len(snapshots[0]["collection_errors"]) == 1 and (
        "missing required fields" in snapshots[0]["collection_errors"][0]
    ), "a partial scrape must retain missing-field diagnostics alongside its samples"


def test_startup_spread_precedes_a_real_scrape_when_no_force_request_exists(
    monkeypatch, tmp_path
):
    clock = support.Clock()
    waits = []
    sink = support.RecordingSink()
    reader = collector(
        sink, force_snapshot_path=str(tmp_path / "absent.request"), now=clock.now
    )

    def sleep(seconds):
        waits.append(seconds)
        clock.sleep(seconds)
        if len(waits) == 2:
            raise support.StopLoop

    monkeypatch.setattr(dcgm, "time", SimpleNamespace(sleep=sleep))
    monkeypatch.setattr(
        dcgm,
        "urlopen",
        lambda *args, **kwargs: support.Response(
            body=b'DCGGM_UNKNOWN 2\nDCGM_FI_DEV_GPU_TEMP{UUID="GPU-a"} 20\n'
        ),
    )
    with pytest.raises(support.StopLoop):
        reader.run()
    assert 0 < waits[0] <= reader.startup_spread_seconds
    assert waits[1] == reader.interval_seconds
    assert sink.requests[-1][1]["samples"], (
        "spread must be followed by actual parsed metrics"
    )

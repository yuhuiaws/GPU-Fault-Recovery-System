from __future__ import annotations

import json
import sys

from scripts.perf import control_plane_capacity_probe as probe
from scripts.perf.control_plane_capacity_probe import percentile


def test_capacity_probe_percentile_uses_nearest_rank() -> None:
    values = [0.4, 0.1, 0.3, 0.2]

    assert percentile(values, 0.50) == 0.2
    assert percentile(values, 0.95) == 0.4
    assert percentile([], 0.95) is None


def test_capacity_probe_counts_timeouts_in_latency_distribution(
    monkeypatch, capsys
) -> None:
    outcomes = iter([(None, 10.0, "TimeoutError")] * 9 + [(200, 0.1, None)])
    monkeypatch.setattr(probe, "request_once", lambda *_args, **_kwargs: next(outcomes))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "capacity-probe",
            "--url",
            "https://control.example",
            "--requests",
            "10",
            "--concurrency",
            "1",
            "--max-error-rate",
            "1",
            "--max-p95-seconds",
            "5",
        ],
    )

    result = probe.main()
    payload = json.loads(capsys.readouterr().out)

    assert result == 1
    assert payload["transport_errors"] == 9
    assert payload["latency_seconds"]["p95"] == 10.0
    assert payload["latency_seconds"]["max"] == 10.0

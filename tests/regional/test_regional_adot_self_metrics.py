"""The ADOT self-metrics verify reads AMP, never the loopback ``:8889`` reader.

The collector's self-metrics reader is bound to 127.0.0.1 on purpose, so the
API-server proxy the check first used was refused on every live verify
(deploy #25, 2026-09-09); the check now asks AMP for the self-scrape job's
series, which also proves scrape, keep list and remote_write end to end.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from gpu_fault_release import regional_admin_checks as CHECKS
from gpu_fault_release import regional_adot_self_metrics as ADOT

ROOT = Path(__file__).resolve().parents[2]


class _AdotRelease:
    config = SimpleNamespace(
        namespace="gpu-fault-system",
        aws_region="us-west-2",
        health=SimpleNamespace(amp_workspace_id="ws-1"),
    )

    def __init__(self) -> None:
        self.kubectl_calls: list[tuple[str, ...]] = []

    def _cpu(self, *args):
        self.kubectl_calls.append(args)
        return args

    @staticmethod
    def _get_json(_command):
        return {
            "items": [
                {
                    "metadata": {"name": "gpu-fault-adot-1"},
                    "status": {"phase": "Running"},
                }
            ]
        }


def _amp_stub(names: set[str], *, up: float | None = 1.0):
    queries: list[str] = []

    def query(_release, expression: str):
        queries.append(expression)
        if expression.startswith("max(up"):
            return [{"metric": {}, "value": [0, str(up)]}] if up is not None else []
        return [{"metric": {"__name__": name}, "value": [0, "1"]} for name in names]

    return query, queries


def test_the_check_reads_the_self_scrape_series_from_amp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    required = ADOT.adot_self_metric_names()
    assert "otelcol_process_memory_rss" in required, "the memory alert reads it"
    # A collector that never lost a batch has never published the failure
    # counter; the check must not demand it.
    assert "otelcol_exporter_send_failed_metric_points" in ADOT.LAZY_ADOT_SELF_SERIES

    query, queries = _amp_stub(set(required) - ADOT.LAZY_ADOT_SELF_SERIES)
    monkeypatch.setattr(ADOT, "amp_instant_query", query)
    release = _AdotRelease()
    value = CHECKS.check_adot_self_metrics(release)
    assert value.status == "PASS"
    assert "gpu-fault-adot-1" in value.summary
    assert "gpu-fault-adot-self" in value.summary
    assert all('job="gpu-fault-adot-self"' in expression for expression in queries), (
        queries
    )
    assert any("otelcol_.+" in expression for expression in queries), queries


def test_a_renamed_series_fails_the_check_by_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    renamed = {
        name.replace("otelcol_process_memory_rss", "otelcol_process_memory_rss_bytes")
        for name in ADOT.adot_self_metric_names()
    }
    query, _queries = _amp_stub(renamed)
    monkeypatch.setattr(ADOT, "amp_instant_query", query)
    with pytest.raises(ADOT.ReleaseError, match="otelcol_process_memory_rss"):
        ADOT.adot_self_metrics_report(_AdotRelease())


@pytest.mark.parametrize("up", [0.0, None])
def test_a_missing_or_down_self_scrape_fails_the_check(
    monkeypatch: pytest.MonkeyPatch, up: float | None
) -> None:
    query, _queries = _amp_stub(set(ADOT.adot_self_metric_names()), up=up)
    monkeypatch.setattr(ADOT, "amp_instant_query", query)
    with pytest.raises(ADOT.ReleaseError, match="not scraping itself"):
        ADOT.adot_self_metrics_report(_AdotRelease())


def test_the_check_never_reaches_the_loopback_reader_through_the_proxy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The manifest binds the self-metrics reader to 127.0.0.1 on purpose; a
    check that reaches it through the API-server proxy can never pass, so the
    only kubectl the check may issue is the Pod listing."""
    query, _queries = _amp_stub(set(ADOT.adot_self_metric_names()))
    monkeypatch.setattr(ADOT, "amp_instant_query", query)
    release = _AdotRelease()
    ADOT.adot_self_metrics_report(release)
    assert release.kubectl_calls, "the check still confirms a Running collector"
    assert all("--raw" not in call for call in release.kubectl_calls), (
        release.kubectl_calls
    )
    manifest = (
        ROOT / "deploy" / "observability" / "adot-control-plane.yaml"
    ).read_text("utf-8")
    assert "host: 127.0.0.1" in manifest, "the loopback binding is part of the design"

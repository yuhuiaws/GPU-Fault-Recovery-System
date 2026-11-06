"""A per-cluster spool cap refusal names its cluster on ``/metrics``.

``gpu_fault_telemetry_spool_rejected_total`` carries only ``scope``, so a
refusal by the per-cluster depth cap could not be attributed to the cluster
that hit it: the perf plan's "no spool rejection for the other clusters"
verdict had to come from the load generators, and the
``GpuFaultTelemetrySpoolAdmissionRejected`` runbook's "check whether one
cluster is flooding" had nothing to read. The processor queue answers the same
question with a twin family beside its zero-initialised scope family; the spool
now has the same pair, and the scope family keeps its series identity for the
alert's ``increase()``.
"""

from __future__ import annotations

import asyncio
import re
from datetime import datetime, timezone
from pathlib import Path

import pytest
import yaml

from gpu_fault.app import ApplicationContext, create_app, process_metrics
from gpu_fault.app.metric_aggregation import STRATEGIES, Strategy
from tests._builders import asgi_client, build_store
from tests.metrics._assertions import assert_sample

ROOT = Path(__file__).resolve().parents[2]
TOKEN = "telemetry-spool-rejection-token-" + "x" * 32
GPU_METRICS_PATH = "/v1/collector-events/gpu-metrics"
SCOPE_FAMILY = "gpu_fault_telemetry_spool_rejected_total"
CLUSTER_FAMILY = "gpu_fault_telemetry_spool_rejected_by_cluster_total"


def _payload(cluster_id: str, node_id: str) -> dict:
    return {
        "batch_id": f"{cluster_id}-{node_id}",
        "cluster_id": cluster_id,
        "node_id": node_id,
        "observed_at": datetime.now(timezone.utc).isoformat(),
        "edge_filter_reasons": ["health-summary"],
        "samples": [],
    }


def _post_then_scrape(
    monkeypatch: pytest.MonkeyPatch,
    *,
    max_depth: int,
    max_cluster_depth: int,
    senders: list[tuple[str | None, str]],
) -> tuple[list[tuple[int, str | None]], str]:
    """POST one routine gpu-metrics sample per (cluster, node), then scrape."""

    monkeypatch.setenv("GPU_FAULT_PROCESSOR_MODE", "active-active")
    monkeypatch.setenv("POD_UID", "pod-spool-rejections-by-cluster")
    monkeypatch.setenv("GPU_FAULT_TELEMETRY_SPOOL", "true")
    monkeypatch.setenv("GPU_FAULT_TELEMETRY_SPOOL_MAX_DEPTH", str(max_depth))
    monkeypatch.setenv(
        "GPU_FAULT_TELEMETRY_SPOOL_MAX_CLUSTER_DEPTH", str(max_cluster_depth)
    )
    monkeypatch.delenv("GPU_FAULT_PROCESSOR_SNAPSHOT_BYPASS", raising=False)
    app = create_app(ApplicationContext(store=build_store(), execution_token=TOKEN))

    async def scenario() -> tuple[list[tuple[int, str | None]], str]:
        answers: list[tuple[int, str | None]] = []
        async with asgi_client(app) as client:
            for cluster_id, node_id in senders:
                headers = (
                    {} if cluster_id is None else {"X-GPU-Fault-Cluster-ID": cluster_id}
                )
                response = await client.post(
                    GPU_METRICS_PATH,
                    json=_payload(cluster_id or "unscoped", node_id),
                    headers=headers,
                )
                answers.append((response.status_code, response.json().get("scope")))
            metrics = await client.get("/metrics")
            assert metrics.status_code == 200, metrics.text
        await app.state.telemetry_spool_batcher.close()
        return answers, metrics.text

    return asyncio.run(scenario())


def _cluster_samples(text: str) -> list[process_metrics.Sample]:
    rendered = process_metrics.parse_lines(text.splitlines())
    return list(rendered.samples.get(CLUSTER_FAMILY, ()))


def test_a_cluster_cap_refusal_is_counted_for_its_cluster(monkeypatch) -> None:
    answers, text = _post_then_scrape(
        monkeypatch,
        max_depth=100,
        max_cluster_depth=1,
        senders=[
            ("cluster-a", "node-1"),
            ("cluster-a", "node-2"),
            ("cluster-b", "node-1"),
        ],
    )

    assert answers == [(202, None), (429, "cluster"), (202, None)], (
        "the second cluster-a sample must hit cluster-a's own cap and only that"
    )
    assert_sample(
        text, CLUSTER_FAMILY, 1, cluster_id="cluster-a", scope="cluster", process="0"
    )
    assert [dict(sample.labels)["cluster_id"] for sample in _cluster_samples(text)] == [
        "cluster-a"
    ], "a cluster that was never refused must have no by-cluster series"
    # The scope family keeps counting the refusal, and keeps both scopes'
    # zero-initialised series for GpuFaultTelemetrySpoolAdmissionRejected.
    assert_sample(text, SCOPE_FAMILY, 1, scope="cluster", process="0")
    assert_sample(text, SCOPE_FAMILY, 0, scope="global", process="0")
    rendered = process_metrics.parse_lines(text.splitlines())
    assert rendered.family_type(CLUSTER_FAMILY) == "counter", rendered.families.get(
        CLUSTER_FAMILY
    )


def test_a_global_cap_refusal_adds_no_cluster_series(monkeypatch) -> None:
    answers, text = _post_then_scrape(
        monkeypatch,
        max_depth=1,
        max_cluster_depth=100,
        senders=[("cluster-a", "node-1"), ("cluster-b", "node-1")],
    )

    assert answers == [(202, None), (429, "global")], answers
    assert_sample(text, SCOPE_FAMILY, 1, scope="global", process="0")
    assert_sample(text, SCOPE_FAMILY, 0, scope="cluster", process="0")
    assert _cluster_samples(text) == [], (
        "a refusal by the region-wide cap belongs to no cluster"
    )
    rendered = process_metrics.parse_lines(text.splitlines())
    assert rendered.family_type(CLUSTER_FAMILY) == "counter", (
        "the family is declared even before any cluster was refused"
    )


def test_an_unscoped_refusal_is_labelled_like_the_spool_cluster_depth(
    monkeypatch,
) -> None:
    """A request without a cluster header is held to the cap of the spool's
    ``__unscoped__`` scope, the label ``gpu_fault_telemetry_spool_cluster_depth``
    gives those rows, so the two families join on ``cluster_id``."""

    answers, text = _post_then_scrape(
        monkeypatch,
        max_depth=100,
        max_cluster_depth=1,
        senders=[(None, "node-1"), (None, "node-2")],
    )

    assert answers == [(202, None), (429, "cluster")], answers
    assert_sample(
        text, CLUSTER_FAMILY, 1, cluster_id="__unscoped__", scope="cluster", process="0"
    )
    assert_sample(
        text, "gpu_fault_telemetry_spool_cluster_depth", 1, cluster_id="__unscoped__"
    )


def control_plane_keep_regex() -> str:
    documents = yaml.safe_load_all(
        (ROOT / "deploy/observability/adot-control-plane.yaml").read_text(
            encoding="utf-8"
        )
    )
    config = next(item for item in documents if item and item["kind"] == "ConfigMap")
    jobs = yaml.safe_load(config["data"]["collector.yaml"])["receivers"]["prometheus"][
        "config"
    ]["scrape_configs"]
    job = next(item for item in jobs if item["job_name"] == "gpu-fault-control-plane")
    return next(
        item["regex"]
        for item in job["metric_relabel_configs"]
        if item["action"] == "keep" and item["source_labels"] == ["__name__"]
    )


def test_both_rejection_families_reach_amp_as_per_process_counters(monkeypatch) -> None:
    """The ADOT keep list is the only way into AMP: the scope family the alert
    reads and the per-cluster twin the runbook and the capacity dashboard read
    both pass it (``gpu_fault_telemetry_spool_.+``), and both keep a process
    slot, so one restarted process cannot reset its siblings' counts."""

    _, text = _post_then_scrape(
        monkeypatch,
        max_depth=100,
        max_cluster_depth=1,
        senders=[("cluster-a", "node-1"), ("cluster-a", "node-2")],
    )
    rendered = process_metrics.parse_lines(text.splitlines())
    names = {SCOPE_FAMILY, CLUSTER_FAMILY} | {
        sample.name
        for family in (SCOPE_FAMILY, CLUSTER_FAMILY)
        for sample in rendered.samples.get(family, ())
    }
    keep = control_plane_keep_regex()

    dropped = sorted(name for name in names if not re.fullmatch(keep, name))
    assert dropped == [], f"the ADOT keep filter drops: {dropped}"
    assert _cluster_samples(text), "the scenario rendered no by-cluster sample"
    for family in (SCOPE_FAMILY, CLUSTER_FAMILY):
        assert STRATEGIES.get(family) is Strategy.PER_PROCESS, (
            f"{family} is a counter: its reset domain is the process"
        )

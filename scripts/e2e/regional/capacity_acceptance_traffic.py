"""Bounded traffic and metric sampling for the isolated CAP-001 probe."""

from __future__ import annotations

import math
import threading
import time
from datetime import datetime, timezone
from typing import Any

import httpx

from scripts.e2e.regional.capacity_acceptance_base import (
    CapError,
    CapHarnessBase,
    Probe,
    utc_now,
)


def sparse_cluster_queue_depth(
    values: list[tuple[str, dict[str, str], float]], cluster_id: str
) -> float:
    total = CapHarnessBase.metric_value(values, "gpu_fault_processor_queue_depth")
    by_cluster: dict[str, float] = {}
    for metric, labels, value in values:
        if metric != "gpu_fault_processor_cluster_queue_depth":
            continue
        cluster = labels.get("cluster_id")
        if (
            not cluster
            or cluster in by_cluster
            or not math.isfinite(value)
            or value < 0
        ):
            raise CapError("per-cluster queue metric is invalid or duplicated")
        by_cluster[cluster] = value
    # PostgreSQL deliberately omits empty clusters. A complete matching global
    # census, not mere absence of a series, is the proof that omission means zero.
    if sum(by_cluster.values()) != total:
        raise CapError("sparse cluster queue census does not match total depth")
    return by_cluster.get(cluster_id, 0.0)


def cap001_monitor(
    harness: CapHarnessBase,
    probe: Probe,
    stop: threading.Event,
    maxima: dict[str, float],
    samples: list[dict[str, Any]],
    errors: list[str],
) -> None:
    while not stop.wait(1):
        try:
            values = harness.metrics(probe.url)
            sample: dict[str, Any] = {
                "observed_at": utc_now(),
                "queue_depth": harness.metric_value(
                    values, "gpu_fault_processor_queue_depth"
                ),
                **{
                    f"{name}_depth": sparse_cluster_queue_depth(
                        values, f"cap-cluster-{index:03d}"
                    )
                    for name, index in (("a", 0), ("b", 1))
                },
            }
            for name, index in (("a", 0), ("b", 1)):
                rejections = [
                    value
                    for metric, labels, value in values
                    if metric
                    == "gpu_fault_processor_admission_rejections_by_cluster_total"
                    and labels.get("cluster_id") == f"cap-cluster-{index:03d}"
                ]
                if any(not math.isfinite(value) or value < 0 for value in rejections):
                    raise CapError("admission rejection metric is invalid")
                sample[f"{name}_rejections"] = sum(rejections)
            samples.append(sample)
            for key, value in sample.items():
                if isinstance(value, (int, float)):
                    maxima[key] = max(maxima.get(key, 0.0), float(value))
        except Exception as exc:
            errors.append(type(exc).__name__)


def cap001_send(
    harness: CapHarnessBase,
    client: httpx.Client,
    phase: str,
    cluster_index: int,
    sequence: int,
    scheduled: float,
) -> dict[str, Any]:
    delay = scheduled - time.perf_counter()
    if delay > 0:
        time.sleep(delay)
    observed = datetime.now(timezone.utc)
    begin = time.perf_counter()
    payload = {
        "batch_id": f"cap001-{phase}-c{cluster_index:03d}-{sequence:05d}",
        "cluster_id": f"cap-cluster-{cluster_index:03d}",
        "node_id": f"node-{phase}-c{cluster_index:03d}-{sequence:05d}",
        "observed_at": observed.isoformat(),
        "samples": [{"name": "network_link_up", "value": 1, "device": "eth0"}],
        "runtime_profile_version": "hyperpod-v1",
        "edge_filter_reasons": ["capacity-test"],
    }
    headers = harness.cluster_headers(
        f"cap-cluster-{cluster_index:03d}", harness.tokens[cluster_index]
    )
    transport_retries = 0
    status: int | str = "not-sent"
    retry_after = None
    for attempt in range(3):
        try:
            response = client.post(
                "/v1/collector-events/host-telemetry", headers=headers, json=payload
            )
            status = response.status_code
            retry_after = response.headers.get("Retry-After")
            break
        except httpx.TransportError as exc:
            transport_retries += 1
            status = f"transport-error:{type(exc).__name__}"
            if attempt < 2:
                probe = getattr(harness, "active_probe", None)
                if isinstance(exc, httpx.ConnectError) and probe is not None:
                    # Nothing is listening on the local port: the kubectl
                    # forward died. Re-establish it (or fail the case with the
                    # Pod's diagnostics if the Pod itself restarted) before
                    # the retry; a live forward makes this a no-op.
                    harness.ensure_probe_transport(
                        probe, reason=f"{phase} send: {exc!r}"
                    )
                time.sleep(0.05)
    return {
        "phase": phase,
        "cluster": cluster_index,
        "status": status,
        "retry_after": retry_after,
        "latency_ms": (time.perf_counter() - begin) * 1000,
        "transport_retries": transport_retries,
    }

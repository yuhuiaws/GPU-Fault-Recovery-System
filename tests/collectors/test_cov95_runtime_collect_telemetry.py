"""Telemetry binding, freshness and policy inputs without node actions."""

from __future__ import annotations

from datetime import timedelta
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from gpu_fault.gpu_metrics import (
    GpuInventoryDevice,
    GpuInventorySnapshot,
    GpuMetricBatch,
    GpuMetricSample,
    GpuMetricSource,
    GpuMetricsService,
    GpuMetricsThresholds,
)
from gpu_fault.host_health import NodeHealthPolicy
from gpu_fault.policy import SxidClassification, SxidLinkScope
from gpu_fault.telemetry import (
    CollectorKind,
    CollectorStatus,
    NvSwitchPortTopologyService,
    TelemetryMetricLatest,
    WorkloadCoverageHeartbeat,
    WorkloadTopologyService,
    merge_collector_status,
)
from tests._builders import (
    attempt_observation,
    build_store,
    build_sxid_event,
    container_observation,
)
from tests.collectors import _cov95_runtime_collect as support

isolated_runtime = support.isolated_runtime


@pytest.mark.parametrize(
    ("selector", "active"),
    [
        ({"pod_uid": "pod-a"}, True),
        ({"pod_uid": "other"}, False),
        ({"container_id": "docker://container-a"}, True),
        ({"container_id": "other"}, False),
        ({"host_pid": 23}, True),
        ({"host_pid": 24}, False),
        ({"cgroup_path": "/job/a/child"}, True),
        ({"cgroup_path": "/job"}, True),
        ({"cgroup_path": "/another"}, False),
        ({"target_gpu_uuids": {"GPU-a"}}, True),
        ({"target_gpu_uuids": {"GPU-other"}}, False),
    ],
)
def test_workload_context_filters_the_observed_container_identity(selector, active):
    store = build_store()
    service = WorkloadTopologyService(store)
    service.observe(
        attempt_observation(
            "job-a",
            "attempt-a",
            support.NOW,
            workload_ids=["training/job-a"],
            containers=[
                container_observation(
                    "pod-a",
                    "pod-a",
                    1,
                    "node-a",
                    gpu_uuids=["GPU-a"],
                    container_id="containerd://container-a",
                    host_pid=23,
                    cgroup_path="/job/a/",
                ),
                container_observation(
                    "pod-b", "pod-b", 2, "node-a", gpu_uuids=["GPU-b"]
                ),
            ],
        )
    )
    result = service.resolve("cluster-a", "node-a", support.NOW, **selector)
    assert result.workload_state == ("ACTIVE" if active else "IDLE")
    assert result.ranks == ([1] if active else [])
    assert result.gpu_uuids == (["GPU-a"] if active else [])


def test_aware_coverage_replaces_legacy_naive_row_instead_of_poisoning_cluster():
    store = build_store()
    legacy = WorkloadCoverageHeartbeat.model_construct(
        cluster_id="legacy-runtime-coverage",
        observed_at=support.NOW.replace(tzinfo=None),
        watcher_instance="old-reader",
        watched_attempts=0,
        watched_pods=0,
    )
    assert store.save_workload_coverage_heartbeat(legacy), (
        "private legacy row must be seeded"
    )
    service = WorkloadTopologyService(store)
    current = WorkloadCoverageHeartbeat(
        cluster_id=legacy.cluster_id,
        observed_at=support.NOW,
        watcher_instance="new-reader",
    )
    assert service.observe_coverage(current), (
        "unusable old timestamp must not veto current coverage"
    )
    assert store.get_workload_coverage_heartbeat(legacy.cluster_id) == current
    assert (
        service.resolve(legacy.cluster_id, "node-a", support.NOW).workload_state
        == "IDLE"
    )


@pytest.mark.parametrize("mode", ["first", "older", "erroring", "recovered"])
def test_collector_status_merge_preserves_rejection_until_a_newer_success(mode):
    previous = CollectorStatus(
        cluster_id="cluster-a",
        node_id="node-a",
        collector=CollectorKind.GPU_METRICS,
        observed_at=support.NOW,
        ingested_at=support.NOW,
        last_error_at=support.NOW,
        errors=["rejected-event: shape drift", "old local error"],
    )
    status = previous.model_copy(
        update={
            "observed_at": support.NOW
            + timedelta(seconds=-1 if mode == "older" else 1),
            "errors": [],
            "last_error_at": None,
            "last_success_at": support.NOW + timedelta(seconds=1)
            if mode == "recovered"
            else None,
        }
    )
    merged = merge_collector_status(None if mode == "first" else previous, status)
    if mode == "older":
        assert merged is None
    elif mode == "first":
        assert merged is status
    elif mode == "erroring":
        assert merged.errors == ["rejected-event: shape drift"]
        assert merged.last_error_at == support.NOW
    else:
        assert merged.errors == []
        assert merged.last_success_at > merged.last_error_at


def sxid_event(**updates):
    return build_sxid_event(
        "sxid-a",
        support.NOW,
        11001,
        SxidClassification.FATAL,
        "NVIDIA_FABRIC_MANAGER_CATALOG",
        switch_id="nvidia-nvswitch0",
        port="018",
        product="A100",
        link_scope=SxidLinkScope.UNKNOWN,
        **updates,
    )


def topology(**updates):
    labels = {
        "switch_id": "0",
        "port": "18",
        "trusted": "true",
        "link_scope": "UNKNOWN",
        "peer_type": "GPU",
        "fabric_partition": "fabric-a",
        "gpu_uuid": "GPU-a",
    }
    labels.update(updates.pop("labels", {}))
    return TelemetryMetricLatest(
        **{
            "cluster_id": "cluster-a",
            "node_id": "node-a",
            "observed_at": support.NOW,
            "name": "nvswitch_port_topology",
            "value": 1,
            "device": "0/18",
            "labels": labels,
            **updates,
        }
    )


@pytest.mark.parametrize(
    "peer,scope",
    [
        ("GPU", SxidLinkScope.ACCESS),
        ("NVSWITCH", SxidLinkScope.TRUNK),
        ("SWITCH", SxidLinkScope.TRUNK),
        ("UNKNOWN", SxidLinkScope.UNKNOWN),
    ],
)
def test_trusted_topology_resolves_peer_type_without_trusting_raw_fault_description(
    peer, scope
):
    store = build_store()
    store.observe_telemetry_metric(topology(labels={"peer_type": peer}))
    event = sxid_event(participating_gpu_uuids=["GPU-old"], fabric_partition="retained")
    result = NvSwitchPortTopologyService(store).resolve(event)
    assert result.link_scope is scope
    assert result.fabric_partition == "retained"
    assert result.participating_gpu_uuids == (
        ["GPU-a", "GPU-old"] if scope is SxidLinkScope.ACCESS else ["GPU-old"]
    )


@pytest.mark.parametrize(
    "updates",
    [
        {"name": "other-metric"},
        {"value": 0},
        {"observed_at": support.NOW - timedelta(seconds=301)},
        {"observed_at": support.NOW + timedelta(seconds=31)},
        {"labels": {"trusted": "false"}},
        {"labels": {"switch_id": "1"}},
        {"labels": {"port": "other"}},
    ],
)
def test_untrusted_stale_or_mismatched_topology_cannot_authorize_link_scope(updates):
    store = build_store()
    store.observe_telemetry_metric(topology(**updates))
    result = NvSwitchPortTopologyService(store).resolve(sxid_event())
    assert result.link_scope is SxidLinkScope.UNKNOWN
    assert result.link_scope_source is None


def test_explicit_link_scope_does_not_consult_ambient_topology():
    event = sxid_event().model_copy(update={"link_scope": SxidLinkScope.TRUNK})
    store = SimpleNamespace(list_telemetry_metrics_latest=support.forbidden)
    assert NvSwitchPortTopologyService(store).resolve(event) is event


@pytest.mark.parametrize("duplicate", ["gpu_index", "pci_bdf"])
def test_inventory_rejects_duplicate_physical_identity(duplicate):
    first = GpuInventoryDevice(gpu_index=0, gpu_uuid="GPU-a", pci_bdf="0000:01:00.0")
    second = GpuInventoryDevice(gpu_index=1, gpu_uuid="GPU-b", pci_bdf="0000:02:00.0")
    second = second.model_copy(update={duplicate: getattr(first, duplicate)})
    with pytest.raises(ValidationError, match="duplicate"):
        GpuInventorySnapshot(
            cluster_id="cluster-a",
            node_id="node-a",
            observed_at=support.NOW,
            source=GpuMetricSource.NVIDIA_SMI,
            source_boot_id="boot-a",
            devices=[first, second],
        )


@pytest.mark.parametrize(
    "options",
    [{"gpu_temperature_critical_c": 85}, {"memory_temperature_critical_c": 90}],
)
def test_temperature_policy_requires_critical_threshold_above_warning(options):
    with pytest.raises(ValidationError, match="must exceed warning"):
        GpuMetricsThresholds(**options)


def test_nonfinite_metrics_are_ignored_and_collection_failure_clears_after_clean_scrape():
    service = GpuMetricsService()
    batch = GpuMetricBatch(
        batch_id="bad-scrape",
        cluster_id="cluster-a",
        node_id="node-a",
        observed_at=support.NOW,
        source=GpuMetricSource.DCGM_EXPORTER,
        samples=[
            GpuMetricSample(
                metric_name="temperature",
                canonical_name="gpu_temperature_c",
                value=float("nan"),
            ),
            GpuMetricSample(
                metric_name="temperature",
                canonical_name="gpu_temperature_c",
                value=float("inf"),
            ),
        ],
        collection_errors=["fake exporter unavailable"],
    )
    first = service.ingest(batch)
    assert first.accepted_samples == 0
    assert len(first.findings) == 1
    assert first.xid_events == []
    clean = batch.model_copy(
        update={
            "batch_id": "clean-scrape",
            "observed_at": support.NOW + timedelta(seconds=1),
            "samples": [],
            "collection_errors": [],
        }
    )
    assert service.ingest(clean).findings == []
    repeated = batch.model_copy(
        update={
            "batch_id": "error-returned",
            "observed_at": support.NOW + timedelta(seconds=2),
        }
    )
    assert len(service.ingest(repeated).new_findings) == 1


@pytest.mark.parametrize(
    ("variable", "value"),
    [
        ("GPU_FAULT_EFA_TRAFFIC_BASELINE_ALPHA", "0"),
        ("GPU_FAULT_LOW_UTILIZATION_DURATION_SECONDS", "14"),
        ("GPU_FAULT_EFA_TRAFFIC_DROP_RATIO", "1"),
        ("GPU_FAULT_EFA_TRAFFIC_SPIKE_RATIO", "1"),
        ("GPU_FAULT_EFA_TRAFFIC_ZERO_HUNG_SECONDS", "1"),
        ("GPU_FAULT_EFA_TRAFFIC_PROGRESS_SUPPRESSION_MAX_SECONDS", "0"),
    ],
)
def test_host_policy_rejects_incoherent_detection_windows_before_reading_store(
    monkeypatch, variable, value
):
    monkeypatch.setenv(variable, value)
    with pytest.raises(ValueError):
        NodeHealthPolicy(SimpleNamespace())

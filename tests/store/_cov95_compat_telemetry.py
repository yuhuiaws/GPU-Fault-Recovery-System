from __future__ import annotations

from gpu_fault.gpu_metric_models import GpuHealthFinding, GpuHealthSeverity
from gpu_fault.gpu_metrics import (
    GpuInventoryDevice,
    GpuInventorySnapshot,
    GpuMetricLatest,
    GpuMetricSample,
    GpuMetricsIngestionResult,
    GpuMetricSource,
)
from gpu_fault.telemetry import (
    CollectorKind,
    CollectorMetricsSnapshotRecord,
    CollectorStatus,
)
from gpu_fault.telemetry_models import TelemetryMetricLatest
from gpu_fault.training_models import TrainingProgressHeartbeat
from tests._builders import attempt_observation, container_observation
from tests.store._cov95_compat_support import NOW, request_model

CLUSTER = "compat-cluster"
NODE = "compat-node"
GPU = "GPU-compat"


def gpu_reading(*, at=NOW, value=42.0):
    return GpuMetricLatest(
        cluster_id=CLUSTER,
        node_id=NODE,
        observed_at=at,
        source=GpuMetricSource.DCGM_EXPORTER,
        sample=GpuMetricSample(
            metric_name="DCGM_FI_DEV_GPU_TEMP",
            canonical_name="temperature",
            gpu_uuid=GPU,
            value=value,
        ),
    )


def finding(finding_id, *, at=NOW, automatic_action=None):
    return GpuHealthFinding(
        finding_id=finding_id,
        cluster_id=CLUSTER,
        node_id=NODE,
        gpu_uuid=GPU,
        observed_at=at,
        severity=GpuHealthSeverity.WARNING,
        reason="compatibility state fixture",
        canonical_name="temperature",
        value=85.0,
        automatic_action=automatic_action,
    )


def write_dataset(store):
    metric = gpu_reading()
    key = (CLUSTER, NODE, GPU, metric.sample.canonical_name)
    inventory = GpuInventorySnapshot(
        snapshot_id="compat-inventory",
        cluster_id=CLUSTER,
        node_id=NODE,
        observed_at=NOW,
        source=GpuMetricSource.NVIDIA_SMI,
        source_boot_id="compat-boot",
        devices=[GpuInventoryDevice(gpu_index=0, gpu_uuid=GPU, pci_bdf="0000:01:00.0")],
        expected_gpu_count=1,
    )
    ingestion = GpuMetricsIngestionResult(batch_id="compat-batch", accepted_samples=1)
    status = CollectorStatus(
        cluster_id=CLUSTER,
        node_id=NODE,
        collector=CollectorKind.GPU_METRICS,
        observed_at=NOW,
        ingested_at=NOW,
        last_success_at=NOW,
    )
    snapshot = CollectorMetricsSnapshotRecord(
        observed_at=NOW, lines=["compat_metric 1"], details=[{"cluster_id": CLUSTER}]
    )
    host = TelemetryMetricLatest(
        cluster_id=CLUSTER,
        node_id=NODE,
        observed_at=NOW,
        name="load",
        value=1.0,
        device="cpu",
    )
    progress = TrainingProgressHeartbeat(
        heartbeat_id="compat-heartbeat",
        cluster_id=CLUSTER,
        attempt_id="compat-attempt",
        rank=0,
        observed_at=NOW,
        node_id=NODE,
        step=3,
    )
    observation = attempt_observation(
        "compat-job",
        "compat-attempt",
        NOW,
        cluster_id=CLUSTER,
        containers=[
            container_observation("compat-pod", "compat-pod", 0, NODE, gpu_uuids=[GPU])
        ],
    )
    active_finding = finding("compat-finding")
    queued = request_model("compat-queued", cluster_id=CLUSTER, node_id=NODE)
    spooled = request_model("compat-spooled", cluster_id=CLUSTER, node_id=NODE)
    store.observe_gpu_metric(key, metric)
    store.save_gpu_inventory_snapshot(inventory)
    store.save_gpu_metrics_batch((CLUSTER, NODE, ingestion.batch_id), ingestion)
    store.update_gpu_finding(key, active_finding, NOW)
    store.save_collector_status(status)
    store.save_collector_metrics_snapshot(snapshot)
    store.observe_telemetry_metric(host)
    store.observe_training_progress(progress)
    store.save_attempt_observation(observation)
    store.enqueue_processor_request(queued)
    store.try_spool_telemetry_requests(
        [spooled], max_depth=10, max_cluster_depth=10, now=NOW
    )
    return {
        "key": key,
        "metric": metric,
        "inventory": inventory,
        "ingestion": ingestion,
        "status": status,
        "snapshot": snapshot,
        "host": host,
        "progress": progress,
        "observation": observation,
        "finding": active_finding,
        "queued": queued,
    }


def assert_persisted_dataset(store, expected):
    assert store.list_gpu_metrics_latest(CLUSTER, NODE) == [expected["metric"]]
    assert store.get_gpu_inventory_snapshot(CLUSTER, NODE) == expected["inventory"]
    assert (
        store.get_gpu_metrics_batch((CLUSTER, NODE, "compat-batch"))
        == expected["ingestion"]
    )
    assert store.get_gpu_finding_state(expected["key"]).finding == expected["finding"]
    assert store.list_collector_statuses(CLUSTER) == [expected["status"]]
    assert store.get_collector_metrics_snapshot() == expected["snapshot"]
    assert store.list_telemetry_metrics_latest(CLUSTER, NODE) == [expected["host"]]
    assert store.list_training_progress(CLUSTER, "compat-attempt") == [
        expected["progress"]
    ]
    assert store.list_attempt_observations(CLUSTER) == [expected["observation"]]
    assert (
        store.get_processor_request("compat-queued").body() == expected["queued"].body()
    )

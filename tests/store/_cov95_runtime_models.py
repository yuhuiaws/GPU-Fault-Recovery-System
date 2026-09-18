from __future__ import annotations

from datetime import datetime, timedelta, timezone

from gpu_fault.gpu_metric_models import GpuHealthFinding, GpuHealthSeverity
from gpu_fault.gpu_metrics import (
    GpuInventoryDevice,
    GpuInventorySnapshot,
    GpuMetricLatest,
    GpuMetricSample,
    GpuMetricsIngestionResult,
    GpuMetricSource,
)
from gpu_fault.models import Environment, TerminalEvent, TerminalStatus
from gpu_fault.training_models import TrainingProgressHeartbeat
from tests._builders import attempt_observation

NOW = datetime(2026, 9, 12, tzinfo=timezone.utc)
CLUSTER = "cov95-runtime"
NODE = "node-example"


def metric(value=70.0, seconds=0, *, cluster=CLUSTER, node=NODE):
    return GpuMetricLatest(
        cluster_id=cluster,
        node_id=node,
        observed_at=NOW + timedelta(seconds=seconds),
        source=GpuMetricSource.DCGM_EXPORTER,
        sample=GpuMetricSample(
            metric_name="temperature",
            canonical_name="temperature",
            gpu_uuid="GPU-example",
            value=value,
        ),
    )


def metric_key(latest):
    return (
        latest.cluster_id,
        latest.node_id,
        latest.sample.gpu_uuid,
        latest.sample.canonical_name,
    )


def ingestion(batch_id="batch-example", count=1):
    return GpuMetricsIngestionResult(batch_id=batch_id, accepted_samples=count)


def inventory(seconds=0, *, node=NODE):
    return GpuInventorySnapshot(
        snapshot_id=f"inventory-{node}-{seconds}",
        cluster_id=CLUSTER,
        node_id=node,
        observed_at=NOW + timedelta(seconds=seconds),
        source=GpuMetricSource.NVIDIA_SMI,
        source_boot_id="boot-example",
        devices=[
            GpuInventoryDevice(
                gpu_index=0, gpu_uuid="GPU-example", pci_bdf="0000:01:00.0"
            )
        ],
    )


def finding(seconds=0, *, severity=GpuHealthSeverity.WARNING, action=None, node=NODE):
    return GpuHealthFinding(
        finding_id=f"finding-{node}-{seconds}",
        cluster_id=CLUSTER,
        node_id=node,
        observed_at=NOW + timedelta(seconds=seconds),
        severity=severity,
        reason="example threshold",
        canonical_name="temperature",
        value=70.0,
        gpu_uuid="GPU-example",
        automatic_action=action,
    )


def progress(seconds=0, *, step=1, attempt="attempt-example", rank=0):
    return TrainingProgressHeartbeat(
        heartbeat_id=f"heartbeat-{attempt}-{rank}-{seconds}",
        cluster_id=CLUSTER,
        attempt_id=attempt,
        rank=rank,
        node_id=NODE,
        observed_at=NOW + timedelta(seconds=seconds),
        step=step,
    )


def observation(seconds=0, *, attempt="attempt-example", cluster=CLUSTER):
    return attempt_observation(
        "job-example", attempt, NOW + timedelta(seconds=seconds), cluster_id=cluster
    )


def terminal(seconds=2, *, attempt="attempt-example"):
    return TerminalEvent(
        cluster_id=CLUSTER,
        environment=Environment.HYPERPOD_EKS,
        job_id="job-example",
        attempt_id=attempt,
        terminal_status=TerminalStatus.SUCCEEDED,
        ended_at=NOW + timedelta(seconds=seconds),
        runtime_profile_version="simulated-v1",
    )

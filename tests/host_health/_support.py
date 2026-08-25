"""Shared fixtures and builders for split test shards."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from gpu_fault.host_health import HostMetricSample, HostTelemetryBatch, NodeHealthPolicy
from gpu_fault.models import NotificationResult, NotificationStatus
from gpu_fault.store import InMemoryStore
from gpu_fault.training_health import TrainingProgressHeartbeat
from tests._builders import (
    attempt_observation,
    build_store,
    container_observation,
    host_telemetry_batch,
)

NOW = datetime(2026, 7, 21, 12, 0, tzinfo=timezone.utc)


def telemetry(
    batch_id: str, name: str, value: float, observed_at: datetime = NOW
) -> HostTelemetryBatch:
    return host_telemetry_batch(
        batch_id,
        observed_at,
        [HostMetricSample(name=name, value=value, device="/")],
        runtime_profile_version="simulated-v1",
    )


def efa_telemetry(
    batch_id: str,
    value: float,
    observed_at: datetime,
    rank_liveness: dict[str, float] | None = None,
) -> HostTelemetryBatch:
    return host_telemetry_batch(
        batch_id,
        observed_at,
        [
            HostMetricSample(
                name="efa_traffic_bytes_per_second", value=value, unit="bytes/second"
            ),
            *(
                HostMetricSample(name=f"training_rank_{name}", value=metric)
                for name, metric in (rank_liveness or {}).items()
            ),
        ],
        workload_state="ACTIVE",
        affected_workload_ids=["training/job/job-a"],
        runtime_profile_version="simulated-v1",
    )


def observe_running_attempt(
    store: InMemoryStore, observed_at: datetime, node_ids: tuple[str, ...] = ("node-a",)
) -> None:
    store.save_attempt_observation(
        attempt_observation(
            "job-a",
            "attempt-a",
            observed_at,
            expected_critical_ranks=len(node_ids),
            workload_ids=["training/job/job-a"],
            containers=[
                container_observation(
                    f"pod-{index}",
                    f"worker-{index}",
                    index,
                    node_id,
                    gpu_uuids=[f"GPU-{index}"],
                )
                for index, node_id in enumerate(node_ids)
            ],
        )
    )


def zero_traffic_with_progress(
    monkeypatch, *, progress_at: tuple[int, ...], samples: tuple[int, ...]
) -> list[str]:
    """Feed zero EFA traffic while ranks report progress.

    ``progress_at`` are offsets in seconds where the attempt reports a
    step increase plus an in-flight checkpoint; ``samples`` are the
    offsets of the traffic observations. Returns ``(offset, signal)``
    for every emitted finding, so the test can assert *when* the
    escalation happened and not merely that it did.
    """

    monkeypatch.setenv("GPU_FAULT_EFA_TRAFFIC_MIN_ACTIVE_BPS", "100")
    monkeypatch.setenv("GPU_FAULT_EFA_TRAFFIC_STARTUP_GRACE_SECONDS", "0")
    monkeypatch.setenv("GPU_FAULT_EFA_TRAFFIC_ZERO_WARNING_SECONDS", "15")
    monkeypatch.setenv("GPU_FAULT_EFA_TRAFFIC_ZERO_HUNG_SECONDS", "30")
    store = build_store()
    policy = NodeHealthPolicy(store)
    for step, offset in enumerate(progress_at):
        store.observe_training_progress(
            TrainingProgressHeartbeat(
                cluster_id="cluster-a",
                attempt_id="attempt-a",
                rank=0,
                node_id="node-a",
                observed_at=NOW + timedelta(seconds=offset),
                step=step + 1,
                checkpoint_ref=f"s3://ckpt/{step}",
            )
        )
    signals = []
    for index, offset in enumerate(samples):
        observed_at = NOW + timedelta(seconds=offset)
        observe_running_attempt(store, observed_at)
        for finding in policy.evaluate_metrics(
            efa_telemetry(f"zero-{index}", 1_000 if index == 0 else 0, observed_at)
        ):
            signals.append((offset, finding.diagnostic_parameters["diagnostic_reason"]))
    return signals


def zero_traffic_with_liveness(
    monkeypatch,
    *,
    samples: tuple[int, ...],
    progress_until: int | None,
    max_suppression: str | None = None,
) -> list[tuple[int, str, dict]]:
    """Feed zero EFA traffic while the node reports per-rank liveness.

    ``progress_until`` is the last offset at which the node still saw a
    rank advance, exactly what the ``/proc`` probe measures; ``None``
    means the node published no liveness samples at all. Returns
    ``(offset, signal, rank_liveness)`` for each emitted finding.
    """

    monkeypatch.setenv("GPU_FAULT_EFA_TRAFFIC_MIN_ACTIVE_BPS", "100")
    monkeypatch.setenv("GPU_FAULT_EFA_TRAFFIC_STARTUP_GRACE_SECONDS", "0")
    monkeypatch.setenv("GPU_FAULT_EFA_TRAFFIC_ZERO_WARNING_SECONDS", "15")
    monkeypatch.setenv("GPU_FAULT_EFA_TRAFFIC_ZERO_HUNG_SECONDS", "30")
    if max_suppression is not None:
        monkeypatch.setenv(
            ("GPU_FAULT_EFA_TRAFFIC_PROGRESS_SUPPRESSION_MAX_SECONDS"), max_suppression
        )
    store = build_store()
    policy = NodeHealthPolicy(store)
    emitted = []
    for index, offset in enumerate(samples):
        observed_at = NOW + timedelta(seconds=offset)
        observe_running_attempt(store, observed_at)
        liveness = None
        if progress_until is not None:
            advancing = offset <= progress_until
            liveness = {
                "process_count": 8,
                "advancing_count": 8 if advancing else 0,
                "seconds_since_progress": float(
                    max(0, offset - min(offset, progress_until))
                ),
                "write_bytes_delta": 0.0,
                "cpu_ticks_delta": 1_500.0,
            }
        for finding in policy.evaluate_metrics(
            efa_telemetry(
                f"live-{index}", 1_000 if index == 0 else 0, observed_at, liveness
            )
        ):
            emitted.append(
                (
                    offset,
                    finding.diagnostic_parameters["diagnostic_reason"],
                    finding.diagnostic_parameters["rank_liveness"],
                )
            )
    return emitted


class RecordingNotifier:
    def __init__(self) -> None:
        self.notifications = []

    def send(self, notification):
        self.notifications.append(notification)
        return NotificationResult(
            notification_id=notification.notification_id,
            status=NotificationStatus.SENT,
            provider_message_id="ses-efa-rdma-1",
        )

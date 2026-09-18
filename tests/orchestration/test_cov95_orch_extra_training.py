from __future__ import annotations

from datetime import timedelta

import pytest

from gpu_fault.orchestration import IncidentOrchestrator
from gpu_fault.training_health import (
    TrainingHealthPolicy,
    TrainingHealthService,
    TrainingProgressHeartbeat,
    training_health_signal_key,
)
from tests.orchestration._cov95_orch_extra_safety import (
    orch_extra_isolation as orch_extra_isolation,
)
from tests.orchestration._cov95_orch_extra_support import (
    NOW,
    finding,
    memory_store,
    observation,
)


def heartbeat(rank=0, **updates):
    return TrainingProgressHeartbeat(
        **{
            "cluster_id": "cluster-a",
            "attempt_id": "unit-attempt",
            "rank": rank,
            "observed_at": NOW,
            "step": 100,
            "loss": 1.0,
            **updates,
        }
    )


def test_progress_enrichment_uses_the_matching_allocation_and_preserves_explicit_fields():
    store = memory_store()
    store.save_attempt_observation(observation(attempt_id="other-attempt"))
    allocated = observation()
    store.save_attempt_observation(allocated)
    service = TrainingHealthService(store)
    first = heartbeat()
    assert service.ingest(first).accepted is True
    saved = store.list_training_progress("cluster-a")[0]
    assert saved.node_id == "node-0"
    assert saved.gpu_uuids == ["GPU-0"]
    assert saved.pod_uid == "pod-0"
    assert saved.container_name == "trainer"
    explicit = heartbeat(
        observed_at=NOW + timedelta(seconds=1),
        node_id="node-0",
        gpu_uuids=["GPU-explicit"],
        pod_uid="explicit-pod",
        container_name="explicit-container",
    )
    assert service.ingest(explicit).accepted is True
    assert store.list_training_progress("cluster-a") == [explicit]


def test_wrong_node_heartbeat_cannot_write_progress_or_emit_findings():
    store = memory_store()
    store.save_attempt_observation(observation())
    service = TrainingHealthService(store)
    with pytest.raises(ValueError, match="node does not match allocation"):
        service.ingest(heartbeat(node_id="other-node", numerical_error=True))
    assert store.list_training_progress("cluster-a") == []
    assert (
        store.get_health_signal_state(
            "cluster-a/unit-attempt/rank-0/training-nonfinite-loss"
        )
        is None
    )


@pytest.mark.parametrize("age", [0, -1])
def test_old_or_equal_progress_is_rejected_without_replacing_the_current_sample(age):
    store = memory_store()
    service = TrainingHealthService(store)
    current = heartbeat()
    service.ingest(current)
    rejected = heartbeat(
        observed_at=NOW + timedelta(seconds=age), step=1, numerical_error=True
    )
    result = service.ingest(rejected)
    assert result.accepted is False
    assert result.heartbeat_id == rejected.heartbeat_id
    assert result.findings == []
    assert store.list_training_progress("cluster-a") == [current]


@pytest.mark.parametrize("loss", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_loss_latches_only_after_notification_and_rearms_after_recovery(loss):
    store = memory_store()
    service = TrainingHealthService(store)
    first = service.ingest(heartbeat(loss=loss))
    assert len(first.findings) == 1
    emitted = first.findings[0]
    assert emitted.metric_name == "training_nonfinite-loss"
    assert emitted.node_id == "UNKNOWN"
    assert emitted.runtime_profile_version is None
    assert emitted.affected_workload_ids == []
    key = training_health_signal_key(emitted)
    assert key == "cluster-a/unit-attempt/rank-0/training-nonfinite-loss"
    assert store.get_health_signal_state(key).notified is False
    service.mark_notified(first.findings)
    quiet = service.ingest(heartbeat(loss=loss, observed_at=NOW + timedelta(seconds=1)))
    assert quiet.findings == []
    service.ingest(heartbeat(observed_at=NOW + timedelta(seconds=2)))
    assert store.get_health_signal_state(key).active is False
    assert store.get_health_signal_state(key).notified is False
    again = service.ingest(heartbeat(loss=loss, observed_at=NOW + timedelta(seconds=3)))
    assert len(again.findings) == 1
    assert again.findings[0].event_id != emitted.event_id


def test_step_regression_is_owed_until_latched_and_rearms_after_progress():
    store = memory_store()
    service = TrainingHealthService(store)
    service.ingest(heartbeat(step=100))
    regressed = service.ingest(
        heartbeat(step=90, observed_at=NOW + timedelta(seconds=1))
    )
    assert [item.metric_name for item in regressed.findings] == [
        "training_step-regression"
    ]
    service.mark_notified(regressed.findings)
    assert (
        service.ingest(
            heartbeat(step=80, observed_at=NOW + timedelta(seconds=2))
        ).findings
        == []
    )
    assert (
        service.ingest(
            heartbeat(step=90, observed_at=NOW + timedelta(seconds=3))
        ).findings
        == []
    )
    renewed = service.ingest(heartbeat(step=85, observed_at=NOW + timedelta(seconds=4)))
    assert [item.metric_name for item in renewed.findings] == [
        "training_step-regression"
    ]
    service.ingest(heartbeat(step=None, observed_at=NOW + timedelta(seconds=5)))
    assert (
        store.get_health_signal_state(
            "cluster-a/unit-attempt/rank-0/training-step-regression"
        ).active
        is False
    )


@pytest.mark.parametrize("phase", ["PENDING", "FAILED", "STOPPED", "SUCCEEDED"])
def test_scan_all_does_not_invent_hangs_for_nonrunning_attempts(phase):
    store = memory_store()
    store.save_attempt_observation(observation(workload_phase=phase))
    assert (
        TrainingHealthService(store).scan_all(now=NOW + timedelta(hours=1)).findings
        == []
    )


def test_startup_grace_uses_first_observation_and_does_not_require_a_first_heartbeat():
    store = memory_store()
    store.save_attempt_observation(observation())
    service = TrainingHealthService(
        store, TrainingHealthPolicy(startup_grace_seconds=30)
    )
    assert service.scan_all(now=NOW + timedelta(seconds=30)).findings == []
    result = service.scan_all(now=NOW + timedelta(seconds=31))
    assert {item.metric_name for item in result.findings} == {"training_hang"}
    assert {item.node_id for item in result.findings} == {"node-0", "node-1"}
    assert {item.runtime_profile_version for item in result.findings} == {
        "simulated-v1"
    }
    assert store.list_training_progress("cluster-a") == []


@pytest.mark.parametrize(
    ("steps", "rates", "expected"),
    [
        ([100, 60], [None, None], {"node-1"}),
        ([None, None], [100, 1], {"node-1"}),
        ([100, 99], [0, 0], set()),
        ([100, None], [100, None], set()),
        ([None, None], [100, 34], set()),
    ],
)
def test_straggler_scan_uses_available_peer_medians_without_inventing_missing_values(
    steps, rates, expected
):
    store = memory_store()
    store.save_attempt_observation(observation())
    service = TrainingHealthService(store)
    for rank, (step, rate) in enumerate(zip(steps, rates, strict=True)):
        service.ingest(heartbeat(rank, step=step, samples_per_second=rate))
    result = service.scan("cluster-a", now=NOW + timedelta(seconds=1))
    assert {item.node_id for item in result.findings} == expected
    assert all(item.metric_name == "training_straggler" for item in result.findings), (
        "fresh peer samples should not be reported as heartbeat loss"
    )


@pytest.mark.parametrize(
    ("reference", "metric"),
    [
        (None, "training_hang"),
        ("s3://unit/evidence", "training_hang"),
        ("training-progress://unit-attempt/rank-0", "cpu_usage"),
        ("training-progress:///rank-0", "training_hang"),
        ("training-progress://unit-attempt", "training_hang"),
        ("training-progress://unit-attempt/rank-x", "training_hang"),
        ("training-progress://unit-attempt/rank--1", "training_hang"),
    ],
)
def test_unrecognized_finding_cannot_latch_a_training_signal(reference, metric):
    store = memory_store()
    service = TrainingHealthService(store)
    value = finding(evidence_ref=reference, metric_name=metric)
    assert training_health_signal_key(value) is None
    service.mark_notified([value])
    assert (
        store.get_health_signal_state("cluster-a/unit-attempt/rank-0/training-hang")
        is None
    )


def test_signal_identity_preserves_attempt_path_and_canonicalizes_rank():
    value = finding(
        evidence_ref="training-progress://unit/attempt/rank-03",
        metric_name="training_hang",
    )
    assert (
        training_health_signal_key(value)
        == "cluster-a/unit/attempt/rank-3/training-hang"
    )


def test_training_incidents_cannot_collide_between_clusters_with_the_same_attempt_id():
    store = memory_store()
    service = TrainingHealthService(store)
    orchestrator = IncidentOrchestrator(store)
    incidents = []
    for cluster in ("cluster-a", "cluster-b"):
        store.save_attempt_observation(observation(cluster_id=cluster))
        result = service.ingest(heartbeat(cluster_id=cluster, numerical_error=True))
        assert len(result.findings) == 1
        incident, workflow = orchestrator.ingest_node_health(result.findings[0])
        assert workflow is not None, (
            "a training diagnostic was not represented in Store"
        )
        assert incident.cluster_id == cluster, (
            "a same-attempt training finding reused another cluster's incident"
        )
        incidents.append(incident.incident_id)
    assert len(set(incidents)) == 2, (
        "two cluster-scoped health signals shared one incident"
    )


def test_training_policy_environment_is_used_without_starting_any_worker(monkeypatch):
    monkeypatch.setenv("GPU_FAULT_TRAINING_HEARTBEAT_TIMEOUT_SECONDS", "45")
    monkeypatch.setenv("GPU_FAULT_TRAINING_STARTUP_GRACE_SECONDS", "90")
    monkeypatch.setenv("GPU_FAULT_TRAINING_MAX_STEP_LAG", "8")
    monkeypatch.setenv("GPU_FAULT_TRAINING_MIN_THROUGHPUT_RATIO", "0.25")
    policy = TrainingHealthPolicy.from_environment()
    assert policy.model_dump() == {
        "heartbeat_timeout_seconds": 45,
        "startup_grace_seconds": 90,
        "max_step_lag": 8,
        "min_throughput_ratio": 0.25,
    }

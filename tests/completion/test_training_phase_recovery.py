from datetime import timedelta

import pytest

from gpu_fault.training_health import TrainingHealthService
from tests._builders import build_store
from tests.completion.test_training_health import NOW, heartbeat, observation


def publish(store, service, at, *, phase="TRAINING", step=10, **updates):
    store.save_attempt_observation(observation(observed_at=at))
    value = heartbeat(0, observed_at=at, step=step).model_copy(
        update={"labels": {"phase": phase}, **updates}
    )
    return service.ingest(value)


@pytest.mark.parametrize(
    "phase", ["INITIALIZING", "CHECKPOINT_LOAD", "CHECKPOINT_SAVE", "RESTORING"]
)
def test_live_phase_pause_is_bounded(phase):
    store = build_store()
    service = TrainingHealthService(store)
    publish(store, service, NOW, phase=phase)
    publish(store, service, NOW + timedelta(seconds=119), phase=phase)
    result = service.scan("cluster-a", now=NOW + timedelta(seconds=121))
    assert not any(item.node_id == "node-0" for item in result.findings), result
    later = NOW + timedelta(seconds=1801)
    publish(store, service, later, phase=phase)
    result = service.scan("cluster-a", now=later)
    assert any(
        item.metric_name == "training_hang" and item.node_id == "node-0"
        for item in result.findings
    ), result


def test_stale_heartbeat_is_reporter_silence_not_confirmed_nccl_hang():
    store = build_store()
    service = TrainingHealthService(store)
    publish(store, service, NOW, phase="CHECKPOINT_SAVE")
    result = service.scan("cluster-a", now=NOW + timedelta(seconds=121))
    assert any(
        item.metric_name == "training_reporter-silent" for item in result.findings
    ), result
    assert not any(item.metric_name == "training_hang" for item in result.findings), (
        result
    )


def test_checkpoint_restore_can_regress_step_only_inside_the_bounded_phase():
    store = build_store()
    service = TrainingHealthService(store)
    publish(store, service, NOW, step=100)
    restored = publish(
        store,
        service,
        NOW + timedelta(seconds=1),
        phase="RESTORING",
        step=50,
        checkpoint_ref="checkpoint-50",
    )
    assert restored.findings == [], restored
    regressed = publish(store, service, NOW + timedelta(seconds=2), step=40)
    assert [item.metric_name for item in regressed.findings] == [
        "training_step-regression"
    ]


def test_unknown_or_old_attempt_cannot_stop_current_training():
    store = build_store()
    service = TrainingHealthService(store)
    assert not service.ingest(heartbeat(0, numerical_error=True)).accepted, (
        "an unallocated heartbeat has no authority to stop work"
    )
    store.save_attempt_observation(observation())
    stale = service.ingest(
        heartbeat(0, observed_at=NOW + timedelta(seconds=121), numerical_error=True)
    )
    assert stale.accepted is False and stale.findings == [], stale


@pytest.mark.parametrize(
    "identity",
    [
        {"pod_uid": "other-pod"},
        {"container_name": "sidecar"},
        {"gpu_uuids": ["GPU-other"]},
    ],
    ids=["pod", "container", "gpu"],
)
def test_forged_progress_identity_has_no_stop_authority(identity):
    store = build_store()
    store.save_attempt_observation(observation())
    service = TrainingHealthService(store)
    with pytest.raises(ValueError, match="identity does not match allocation"):
        service.ingest(heartbeat(0, numerical_error=True).model_copy(update=identity))
    assert store.list_training_progress("cluster-a") == [], (
        "no invalid progress persisted"
    )

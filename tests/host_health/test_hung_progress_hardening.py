from datetime import timedelta

import pytest

from gpu_fault.host_health import NodeHealthPolicy
from gpu_fault.models import EfaTrafficSignal
from gpu_fault.training_health import TrainingHealthService, TrainingProgressHeartbeat
from tests._builders import build_store
from tests.completion.test_training_health import NOW
from tests.completion.test_training_phase_recovery import publish
from tests.host_health._support import efa_telemetry, observe_running_attempt


def test_stalled_heartbeats_do_not_rearm_delivered_hang():
    store = build_store()
    service = TrainingHealthService(store)
    publish(store, service, NOW)
    at = NOW + timedelta(seconds=121)
    publish(store, service, at)
    first = service.scan("cluster-a", now=at)
    service.mark_notified(first.findings)
    publish(store, service, at + timedelta(seconds=1))
    again = service.scan("cluster-a", now=at + timedelta(seconds=1))
    assert not any(
        item.metric_name == "training_hang" and item.node_id == "node-0"
        for item in again.findings
    ), again
    publish(store, service, at + timedelta(seconds=2), step=11)
    publish(store, service, at + timedelta(seconds=124), step=11)
    recurrence = service.scan("cluster-a", now=at + timedelta(seconds=124))
    assert any(
        item.metric_name == "training_hang" and item.node_id == "node-0"
        for item in recurrence.findings
    ), recurrence


def test_long_observation_gap_cannot_count_as_continuous_zero():
    store = build_store()
    policy = NodeHealthPolicy(store)
    for offset, rate in [(0, 2_000_000), (130, 0), (200, 0), (900, 0)]:
        at = NOW + timedelta(seconds=offset)
        observe_running_attempt(store, at)
        policy.evaluate_metrics(efa_telemetry(f"gap-{offset}", rate, at))
    state = store.get_efa_traffic_state("cluster-a/node-a")
    assert state.signal is EfaTrafficSignal.ZERO_PENDING, state
    assert state.zero_since == NOW + timedelta(seconds=900), state


@pytest.mark.parametrize("step", [None, 10, "advancing"])
@pytest.mark.parametrize(
    "phase", ["TRAINING", "INITIALIZING", "CHECKPOINT_SAVE", "CHECKPOINT_LOAD"]
)
def test_application_steps_and_writes_do_not_change_rdma_idle_clock(
    monkeypatch, step, phase
):
    monkeypatch.setenv("GPU_FAULT_EFA_TRAFFIC_MIN_ACTIVE_BPS", "100")
    monkeypatch.setenv("GPU_FAULT_EFA_TRAFFIC_STARTUP_GRACE_SECONDS", "0")
    monkeypatch.setenv("GPU_FAULT_EFA_TRAFFIC_ZERO_WARNING_SECONDS", "15")
    monkeypatch.setenv("GPU_FAULT_EFA_TRAFFIC_ZERO_HUNG_SECONDS", "30")
    store = build_store()
    policy = NodeHealthPolicy(store)
    service = TrainingHealthService(store)
    findings = []
    for offset in (0, 15, 30, 45):
        at = NOW + timedelta(seconds=offset)
        observe_running_attempt(store, at)
        service.ingest(
            TrainingProgressHeartbeat(
                cluster_id="cluster-a",
                attempt_id="attempt-a",
                rank=0,
                node_id="node-a",
                observed_at=at,
                step=offset if step == "advancing" else step,
                labels={"phase": phase},
                checkpoint_ref="checkpoint",
            )
        )
        findings += policy.evaluate_metrics(
            efa_telemetry(
                f"writes-{offset}",
                1000 if not offset else 0,
                at,
                rank_liveness={
                    "process_count": 8,
                    "advancing_count": 8,
                    "seconds_since_progress": 0,
                    "write_bytes_delta": 1000000,
                },
            )
        )
    hung = [
        item
        for item in findings
        if item.diagnostic_parameters["diagnostic_reason"]
        == "EFA_TRAFFIC_HUNG_SUSPECTED"
    ]
    assert len(hung) == 1, findings
    assert hung[0].diagnostic_parameters["rank_liveness"]["use"] == "CONTEXT_ONLY"
    assert hung[0].observed_at == NOW + timedelta(seconds=45), hung

from datetime import timedelta

import pytest

from gpu_fault.execution import WorkflowStepOutcome
from gpu_fault.models import IncidentState, WorkflowStatus
from gpu_fault.models import WorkflowOperation as Op
from gpu_fault.store import InMemoryStore, SqliteStore
from gpu_fault.training_health import TrainingHealthService
from tests._builders import build_store
from tests.completion.test_training_health import NOW
from tests.completion.test_training_phase_recovery import publish
from tests.execution._cov95_runtime_workflows import FlowHarness


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_missing_step_does_not_move_progress_or_hide_an_existing_hang(
    backend, tmp_path
):
    store = (
        InMemoryStore()
        if backend == "memory"
        else SqliteStore(str(tmp_path / "progress.db"))
    )
    try:
        service = TrainingHealthService(store)
        publish(store, service, NOW, step=10)
        later = NOW + timedelta(seconds=121)
        publish(store, service, later, step=10)
        service.mark_notified(service.scan("cluster-a", now=later).findings)
        publish(store, service, later + timedelta(seconds=1), step=None)
        state = next(
            item
            for item in store.list_training_progress_states("cluster-a")
            if item.heartbeat.rank == 0
        )
        assert state.last_progress_at == NOW, state
        result = service.scan("cluster-a", now=later + timedelta(seconds=1))
        assert any(
            item.metric_name == "training_progress-unavailable"
            for item in result.findings
        ), result
        publish(store, service, later + timedelta(seconds=2), step=10)
        state = next(
            item
            for item in store.list_training_progress_states("cluster-a")
            if item.heartbeat.rank == 0
        )
        assert state.last_progress_at == NOW, (
            "the same step returning after a gap is not progress"
        )
        assert not any(
            item.metric_name == "training_hang" and item.node_id == "node-0"
            for item in service.scan(
                "cluster-a", now=later + timedelta(seconds=2)
            ).findings
        ), "delivered hang remains latched"
        publish(store, service, later + timedelta(seconds=3), step=11)
        state = next(
            item
            for item in store.list_training_progress_states("cluster-a")
            if item.heartbeat.rank == 0
        )
        assert state.last_progress_at == later + timedelta(seconds=3)
    finally:
        if backend == "sqlite":
            store.close()


def test_new_checkpoint_phase_uses_phase_start_not_previous_training_step_age():
    store = build_store()
    service = TrainingHealthService(store)
    publish(store, service, NOW, step=10)
    at = NOW + timedelta(seconds=2000)
    publish(store, service, at, step=10, phase="CHECKPOINT_SAVE")
    result = service.scan("cluster-a", now=at)
    assert not any(
        item.node_id == "node-0" and item.metric_name == "training_hang"
        for item in result.findings
    ), result
    at += timedelta(seconds=1801)
    publish(store, service, at, step=10, phase="CHECKPOINT_SAVE")
    assert any(
        item.node_id == "node-0" and item.metric_name == "training_hang"
        for item in service.scan("cluster-a", now=at).findings
    ), "phase cannot suppress indefinitely"


def test_phase_flapping_without_real_progress_cannot_replenish_pause_budget():
    store = build_store()
    service = TrainingHealthService(store)
    publish(store, service, NOW, phase="CHECKPOINT_SAVE")
    publish(store, service, NOW + timedelta(seconds=1700), phase="TRAINING")
    later = NOW + timedelta(seconds=1801)
    publish(store, service, later, phase="CHECKPOINT_SAVE")
    assert any(
        item.node_id == "node-0" and item.metric_name == "training_hang"
        for item in service.scan("cluster-a", now=later).findings
    ), "only a new step replenishes a previously used pause budget"


@pytest.mark.parametrize("mode", ["TRAINING", "HUNG_DIAGNOSTIC"])
def test_failed_hung_evidence_is_escalated_not_recovered(mode):
    harness = FlowHarness([Op.COLLECT_HUNG_SUMMARY])
    harness.amend(
        official_steps=[
            harness.workflow.official_steps[0].model_copy(
                update={"parameters": {"health_recovery_contract": {"mode": mode}}}
            )
        ]
    )
    harness.adapter.outcomes[Op.COLLECT_HUNG_SUMMARY] = WorkflowStepOutcome.failed(
        "agent unavailable"
    )
    result = harness.execute()
    assert result.status is WorkflowStatus.FAILED, result
    assert (
        harness.store.get_incident(harness.incident.incident_id).state
        is IncidentState.ESCALATED
    )

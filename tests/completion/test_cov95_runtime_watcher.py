from __future__ import annotations

from datetime import timedelta

import pytest

from gpu_fault.models import Environment, TerminalStatus
from gpu_fault.watcher import (
    AttemptObservation,
    AttemptPhase,
    CompletionWatcher,
    ContainerObservation,
    WorkloadPhase,
)
from tests.completion._support import NOW


def observation(attempt_id, *, phase=WorkloadPhase.RUNNING, at=NOW, **values):
    return AttemptObservation(
        cluster_id="cluster-local",
        environment=Environment.HYPERPOD_EKS,
        job_id="training",
        attempt_id=attempt_id,
        workload_phase=phase,
        observed_at=at,
        expected_critical_ranks=1,
        runtime_profile_version="profile-local",
        **values,
    )


@pytest.mark.parametrize(
    "options,message",
    [
        ({"terminated": True}, "requires exit_code"),
        ({"exit_code": 0}, "running container cannot"),
        ({"finished_at": NOW}, "running container cannot"),
    ],
)
def test_container_termination_contract_rejects_contradictory_observations(
    options, message
):
    with pytest.raises(ValueError, match=message):
        ContainerObservation(
            pod_uid="uid",
            pod_name="trainer",
            container_name="trainer",
            role="worker",
            rank=0,
            **options,
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("cluster_id", "other-cluster"),
        ("environment", Environment.KUBERNETES),
        ("job_id", "other-training"),
        ("runtime_profile_version", "other-profile"),
    ],
)
@pytest.mark.parametrize("terminal", [True, False])
def test_identity_drift_cannot_reuse_active_or_terminal_attempt_state(
    field, value, terminal
):
    watcher = CompletionWatcher()
    original = observation(
        "attempt", phase=WorkloadPhase.STOPPED if terminal else WorkloadPhase.RUNNING
    )
    watcher.observe(original)
    with pytest.raises(ValueError, match=field):
        watcher.observe(original.model_copy(update={field: value}))
    stable = watcher.observe(original)
    assert stable.duplicate_terminal is terminal
    assert stable.phase is (
        AttemptPhase.TERMINAL_STOPPED if terminal else AttemptPhase.RUNNING
    )


def test_pruning_expires_then_trims_oldest_terminals_without_losing_active_attempts():
    watcher = CompletionWatcher(terminal_retention_seconds=60, max_attempts=200)
    for index in range(20):
        watcher.observe(
            observation(
                f"terminal-{index:03d}",
                phase=WorkloadPhase.STOPPED,
                at=NOW if index < 2 else NOW + timedelta(seconds=50),
            )
        )
    for index in range(90):
        watcher.observe(
            observation(f"active-{index:03d}", at=NOW + timedelta(seconds=50))
        )
    watcher.max_attempts = 100
    watcher.observe(observation("active-000", at=NOW + timedelta(seconds=61)))
    assert watcher.pruned_attempts_total == 10
    assert watcher.take_pruned_attempt_ids() == {
        f"terminal-{index:03d}" for index in range(10)
    }
    assert watcher.take_pruned_attempt_ids() == set()
    retained = watcher.observe(
        observation(
            "terminal-010", phase=WorkloadPhase.STOPPED, at=NOW + timedelta(seconds=61)
        )
    )
    assert retained.duplicate_terminal is True
    for index in range(90):
        assert (
            watcher.observe(
                observation(f"active-{index:03d}", at=NOW + timedelta(seconds=61))
            ).phase
            is AttemptPhase.RUNNING
        )
    assert watcher.pruned_attempts_total == 10


def test_no_active_attempt_is_evicted_when_terminals_cannot_satisfy_the_budget():
    watcher = CompletionWatcher(max_attempts=100)
    for index in range(105):
        watcher.observe(observation(f"active-{index}"))
    assert watcher.pruned_attempts_total == 0
    assert watcher.take_pruned_attempt_ids() == set()
    for index in range(105):
        with pytest.raises(ValueError, match="job_id"):
            watcher.observe(
                observation(f"active-{index}").model_copy(update={"job_id": "drift"})
            )


def test_current_terminal_survives_retention_while_siblings_are_pruned():
    watcher = CompletionWatcher(terminal_retention_seconds=60)
    held = observation("held", phase=WorkloadPhase.STOPPED)
    other = observation("other", phase=WorkloadPhase.STOPPED)
    watcher.observe(held)
    watcher.observe(other)
    result = watcher.observe(
        held.model_copy(update={"observed_at": NOW + timedelta(seconds=61)})
    )
    assert result.duplicate_terminal is True
    assert watcher.take_pruned_attempt_ids() == {"other"}


def test_owned_stop_is_terminal_without_automatic_recovery_takeover() -> None:
    watcher = CompletionWatcher()
    ended = ContainerObservation(
        pod_uid="uid",
        pod_name="trainer",
        container_name="trainer",
        role="worker",
        rank=0,
        node_id="node-local",
        gpu_uuids=["GPU-local"],
        terminated=True,
        exit_code=137,
        finished_at=NOW,
    )
    stopped = observation(
        "attempt",
        phase=WorkloadPhase.FAILED,
        containers=[ended],
        termination_initiator_incident_id="incident-owned-stop",
    )
    result = watcher.observe(stopped)
    assert result.terminal_event.terminal_status is TerminalStatus.STOPPED
    assert result.failure_detected is None
    assert result.commands == []
    resumed = stopped.model_copy(
        update={
            "workload_phase": WorkloadPhase.RUNNING,
            "termination_initiator_incident_id": None,
            "containers": [
                ended.model_copy(
                    update={"terminated": False, "exit_code": None, "finished_at": None}
                )
            ],
        }
    )
    replay = watcher.observe(resumed)
    assert replay.duplicate_terminal is True
    assert replay.terminal_event == result.terminal_event
    assert replay.failure_detected is None
    assert replay.commands == []

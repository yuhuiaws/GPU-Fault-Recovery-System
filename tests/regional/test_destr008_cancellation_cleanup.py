from __future__ import annotations

import json
from typing import Any

import pytest

from gpu_fault.models import WorkflowStatus
from gpu_fault.regional import RemoteCommandResult
from gpu_fault.remote_command_models import RemoteCommandStatus
from scripts.e2e.regional.probes import destr008_cancellation_probe as probe
from scripts.e2e.regional.probes import destr008_cancellation_protocol as wire
from tests.regional.test_destr008_cancellation_bridge import arguments, main_bridge
from tests.regional.test_destr008_cancellation_bridge import (
    restore_logging as restore_logging,
)
from tests.regional.test_destr008_cancellation_probe import (
    NOW,
    acknowledge,
    armed_submission,
    claim,
    command,
    parent_write,
    seed,
    setup,
)


def cleanup(port, store, *, attempt_id="cleanup-a", seconds=30, **kwargs):
    return probe.Watchdog(
        port,
        store,
        sleep=lambda _: None,
        cleanup_only=True,
        cleanup_seconds=seconds,
        cleanup_attempt_id=attempt_id,
        **kwargs,
    )


def fail_closed(port, *, now):
    return probe.persist_failure(port, code="PARENT_LOST", now=now, monitoring=False)


def test_cleanup_gets_fresh_quiescence_but_original_case_stays_failed() -> None:
    bound, _, port, store, watchdog, _, _ = armed_submission(
        status=WorkflowStatus.FAILED
    )
    failed = fail_closed(port, now=bound.deadline_at)
    original_control = port.read().control
    later = bound.deadline_at + 1000
    assert watchdog.tick(later) == failed, (
        "normal restart must not reset terminal failure"
    )
    observer = cleanup(port, store)
    first = observer.tick(later)
    assert first.state == "REVOKED" and first.case_failed
    assert first.failure == failed.failure and first.sequence > failed.sequence
    assert first.cleanup.started_at == later and first.cleanup.deadline_at == later + 30
    assert first.quiet_since == later
    done = observer.tick(later + 5)
    assert done.state == "QUIESCENT" and done.case_failed
    assert done.failure == failed.failure
    assert done.commands_active == done.workflows_active == 0 and done.source_complete
    assert done.revocation == failed.revocation and not done.fence_release_authorized
    assert port.read().control == original_control
    assert port.read().envelope.plan.deadline_at == bound.deadline_at


def test_cleanup_timeout_does_not_reset_or_extend_its_window() -> None:
    bound, _, port, store, _, incident, workflow = armed_submission(
        status=WorkflowStatus.FAILED
    )
    remote = command(store, incident, workflow, status=RemoteCommandStatus.LEASED)
    failed = fail_closed(port, now=bound.deadline_at)
    now = bound.deadline_at + 1000
    observer = cleanup(port, store, seconds=10)
    assert observer.tick(now).commands_active == 1
    expired = observer.tick(now + 10)
    assert expired.state == "FAILED" and expired.error_code == "CLEANUP_TIMEOUT"
    assert not expired.monitoring and expired.failure == failed.failure
    assert expired.cleanup.deadline_at == now + 10
    assert (
        store.get_remote_command(remote.command_id).status is RemoteCommandStatus.LEASED
    )
    retried = cleanup(port, store, seconds=10).tick(now + 20)
    assert retried.error_code == "CLEANUP_TIMEOUT" and not retried.monitoring
    assert retried.cleanup == expired.cleanup and retried.failure == failed.failure
    altered = cleanup(port, store, seconds=30).tick(now + 20)
    assert altered.error_code == "CLEANUP_BINDING"
    assert altered.cleanup == expired.cleanup, "same attempt cannot extend its budget"


def test_new_cleanup_attempt_proves_real_completion_after_an_expired_attempt() -> None:
    bound, _, port, store, _, incident, workflow = armed_submission(
        status=WorkflowStatus.FAILED
    )
    remote = command(store, incident, workflow, status=RemoteCommandStatus.LEASED)
    original = fail_closed(port, now=bound.deadline_at)
    now = bound.deadline_at + 1000
    first = cleanup(port, store, seconds=10)
    first.tick(now)
    assert first.tick(now + 10).error_code == "CLEANUP_TIMEOUT"
    completed = store.complete_remote_command(
        bound.cluster_id,
        remote.command_id,
        RemoteCommandResult(
            lease_token="local-test-lease",
            status=RemoteCommandStatus.FAILED,
            error="local physical execution ended",
        ),
    )
    assert completed.lease_owner is None and completed.lease_expires_at is None
    second = cleanup(port, store, attempt_id="cleanup-b")
    resumed = second.tick(now + 100)
    assert resumed.state == "REVOKED" and resumed.quiet_since == now + 100
    done = second.tick(now + 105)
    assert done.state == "QUIESCENT" and done.cleanup.attempt_id == "cleanup-b"
    assert done.case_failed and done.failure == original.failure
    assert done.sequence > resumed.sequence and not done.fence_release_authorized


def test_unknown_submitting_source_remains_unresolved_during_cleanup() -> None:
    bound, _, port, store, watchdog = setup()
    watchdog.tick(NOW)
    claim(bound, port)
    original = fail_closed(port, now=bound.deadline_at)
    now = bound.deadline_at + 1000
    observer = cleanup(port, store, seconds=10)
    unknown = observer.tick(now)
    assert unknown.state == "FAILED" and unknown.error_code == "SOURCE_UNRESOLVED"
    assert unknown.commands_active is unknown.workflows_active is None
    assert unknown.producer.state == "SUBMITTING" and not unknown.source_complete
    assert unknown.case_failed and unknown.failure == original.failure
    final = observer.tick(now + 10)
    assert final.error_code == "CLEANUP_TIMEOUT" and not final.monitoring
    assert final.producer_revoked and final.failure == original.failure
    with pytest.raises(wire.ProbeError, match="SUBMISSION_REVOKED"):
        wire.claim_submission(bound, port.read().control, claim_id="late", now=now + 10)


def test_source_bound_late_ack_is_accepted_only_by_explicit_cleanup_resume() -> None:
    bound, _, port, store, watchdog = setup()
    watchdog.tick(NOW)
    claim(bound, port)
    seed(store, bound, status=WorkflowStatus.FAILED)
    original = fail_closed(port, now=bound.deadline_at)
    later = bound.deadline_at + 1000
    acknowledge(bound, port, now=later)
    with pytest.raises(wire.ProbeError, match="TERMINAL_CONTROL_CHANGED"):
        wire.validate_receipt(
            bound, port.read().control, original, uid=port.uid, now=later
        )
    observer = cleanup(port, store)
    resumed = observer.tick(later)
    assert resumed.state == "REVOKED" and resumed.source_complete
    done = observer.tick(later + 5)
    assert (
        done.state == "QUIESCENT"
        and done.failure == original.failure
        and done.case_failed
    )


def test_cleanup_only_seals_not_started_before_store_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bound, _, port, store, watchdog = setup()
    watchdog.tick(NOW)
    read = store.list_job_recovery_workflow_incidents

    def scoped(*args: Any, **kwargs: Any):
        state = port.read()
        assert state.control.revocation is not None
        assert state.status.cleanup is not None and state.status.case_failed
        return read(*args, **kwargs)

    monkeypatch.setattr(store, "list_job_recovery_workflow_incidents", scoped)
    observer = cleanup(port, store)
    first = observer.tick(NOW + 1)
    assert first.failure.code == "CLEANUP_ONLY"
    assert observer.tick(NOW + 6).state == "QUIESCENT"
    with pytest.raises(wire.ProbeError, match="SUBMISSION_REVOKED"):
        wire.claim_submission(bound, port.read().control, claim_id="late", now=NOW + 7)


def test_cleanup_budget_is_checked_between_store_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bound, _, port, store, _, _, workflow = armed_submission(
        status=WorkflowStatus.FAILED
    )
    fail_closed(port, now=bound.deadline_at)
    clock = [0.0]
    query = store.list_job_recovery_workflow_incidents

    def slow(*args: Any, **kwargs: Any):
        rows = query(*args, **kwargs)
        clock[0] = 11
        return rows

    monkeypatch.setattr(store, "list_job_recovery_workflow_incidents", slow)
    observer = cleanup(port, store, seconds=10, monotonic=lambda: clock[0])
    failed = observer.tick(bound.deadline_at + 1000)
    assert failed.error_code == "CLEANUP_TIMEOUT" and not failed.monitoring
    assert store.get_workflow(workflow.request_id).workload_withdrawn_at is None


def test_slow_store_read_does_not_count_toward_unobserved_quiet_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bound, _, port, store, _, _, _ = armed_submission(status=WorkflowStatus.FAILED)
    fail_closed(port, now=bound.deadline_at)
    clock = [0.0]
    query = store.list_job_recovery_workflow_incidents
    slow_once = True

    def slow(*args: Any, **kwargs: Any):
        nonlocal slow_once
        rows = query(*args, **kwargs)
        if slow_once:
            slow_once = False
            clock[0] += 4
        return rows

    monkeypatch.setattr(store, "list_job_recovery_workflow_incidents", slow)
    now = bound.deadline_at + 1000
    observer = cleanup(port, store, monotonic=lambda: clock[0])
    first = observer.tick(now)
    assert first.quiet_since == first.observed_at == now + 4
    assert observer.tick(now + 5).state == "REVOKED"
    assert observer.tick(now + 9).state == "QUIESCENT"


def test_original_negative_verdict_failure_does_not_prevent_safe_cleanup() -> None:
    bound, _, port, store, watchdog, _, _ = armed_submission(
        status=WorkflowStatus.SUCCEEDED
    )
    parent_write(port, wire.request_close(bound, port.read().control, now=NOW + 3))
    failed = watchdog.tick(NOW + 3)
    assert failed.error_code == "NOT_NEGATIVE_TERMINAL"
    observer = cleanup(port, store)
    observer.tick(bound.deadline_at + 100)
    done = observer.tick(bound.deadline_at + 105)
    assert done.state == "QUIESCENT" and done.case_failed
    assert (
        done.failure == failed.failure and done.failure.code == "NOT_NEGATIVE_TERMINAL"
    )


@pytest.mark.parametrize(
    "options",
    [
        {"cleanup_only": True},
        {"cleanup_only": True, "cleanup_attempt_id": "a", "cleanup_seconds": 4},
        {"cleanup_only": True, "cleanup_attempt_id": "a", "cleanup_seconds": 181},
        {"cleanup_only": True, "cleanup_attempt_id": "a", "cleanup_seconds": True},
        {"cleanup_only": True, "cleanup_attempt_id": "../wrong"},
        {"cleanup_attempt_id": "a"},
    ],
)
def test_invalid_cleanup_request_has_no_store_authority(
    options: dict[str, Any],
) -> None:
    _, _, port, store, _ = setup()
    with pytest.raises(wire.ProbeError, match="CLEANUP_ARGUMENTS"):
        probe.Watchdog(port, store, **options)


@pytest.mark.usefixtures("restore_logging")
def test_main_cleanup_flags_start_a_fresh_bound_observation(
    monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    bound, api, store, _ = main_bridge(monkeypatch)
    port = probe.KubernetesControlMap(
        api,
        namespace="cpu",
        name="watchdog",
        uid="cm-uid",
        plan_sha256=wire.digest(bound),
        probe_sha256=bound.probe_sha256,
    )
    original = fail_closed(port, now=NOW)
    now = bound.deadline_at + 1000
    monkeypatch.setattr(probe.time, "time", lambda: float(now))
    run = probe.run
    times = iter([now, now + 5])
    monkeypatch.setattr(
        probe,
        "run",
        lambda port, store, **kwargs: run(
            port, store, clock=lambda: next(times), sleep=lambda _: None, **kwargs
        ),
    )
    result = probe.main(
        [
            *arguments(bound),
            "--cleanup-only",
            "--cleanup-seconds",
            "30",
            "--cleanup-attempt-id",
            "cleanup-main",
        ]
    )
    assert result == 0 and store.closed
    done = port.read().status
    assert done.state == "QUIESCENT" and done.case_failed
    assert (
        done.failure == original.failure and done.cleanup.attempt_id == "cleanup-main"
    )
    assert "QUIESCENT" in capsys.readouterr().out


def test_cleanup_parameters_without_explicit_mode_are_rejected(capsys) -> None:
    bound, _, _, _, _ = setup()
    assert probe.main([*arguments(bound), "--cleanup-attempt-id", "a"]) == 1
    assert json.loads(capsys.readouterr().out)["error_code"] == "CLEANUP_ARGUMENTS"

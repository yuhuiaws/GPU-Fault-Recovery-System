"""Completion path: one store transaction per event, no process lock, and a
watchdog for ``PENDING_TRIAGE``.

FINAL-建议汇总 F-G2 (3)(4)(5): the event row, the plan (with its incident and
workflow) and the decision are written inside one ``completion_transaction``
keyed by the event; the only remote call (quick-triage submission) happens
after that transaction committed; a decision stuck in ``PENDING_TRIAGE`` is
expired into a conservative plan by ``reconcile_pending_triage``.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from typing import Iterator

import pytest

from gpu_fault.app import ApplicationContext
from gpu_fault.models import (
    CompletionDecision,
    DecisionStatus,
    DiagnosticRequest,
    RecoveryAction,
    RecoveryPlan,
    TerminalEvent,
    TriageFinding,
    TriageOutcome,
    TriageReport,
)
from gpu_fault.service import CompletionService
from gpu_fault.store import InMemoryStore
from tests._builders import build_context, copy_model


class RecordingStore(InMemoryStore):
    """Remembers which completion transaction each write happened under."""

    def __init__(self) -> None:
        super().__init__()
        self.open_keys: list[str] = []
        self.writes: list[tuple[str, str | None]] = []

    def _current_key(self) -> str | None:
        return self.open_keys[-1] if self.open_keys else None

    @contextmanager
    def _recorded(self, event_key: str) -> Iterator[None]:
        self.open_keys.append(event_key)
        try:
            with self._lock:
                yield
        finally:
            self.open_keys.pop()

    def completion_transaction(self, event_key: str):
        return self._recorded(event_key)

    def save_event_if_absent(self, event: TerminalEvent) -> bool:
        self.writes.append(("event", self._current_key()))
        return super().save_event_if_absent(event)

    def save_plan(self, plan: RecoveryPlan) -> None:
        self.writes.append(("plan", self._current_key()))
        super().save_plan(plan)

    def save_decision(self, decision: CompletionDecision) -> None:
        self.writes.append(("decision", self._current_key()))
        super().save_decision(decision)

    def save_diagnostic(self, request: DiagnosticRequest) -> None:
        self.writes.append(("diagnostic", self._current_key()))
        super().save_diagnostic(request)


class ObservingDiagnostics:
    """Records what the store looked like at the moment of submission."""

    def __init__(self, store: RecordingStore) -> None:
        self.store = store
        self.submitted: list[DiagnosticRequest] = []
        self.open_keys_at_submit: list[list[str]] = []
        self.decision_at_submit: list[CompletionDecision | None] = []

    def submit(self, request: DiagnosticRequest) -> str:
        self.submitted.append(request)
        self.open_keys_at_submit.append(list(self.store.open_keys))
        self.decision_at_submit.append(
            self.store.get_decision_by_event(
                f"{request.cluster_id}/{request.attempt_id}/TrainingAttemptTerminal"
            )
        )
        return request.request_id


class FailingDiagnostics:
    def submit(self, request: DiagnosticRequest) -> str:
        raise ConnectionError("dcgm endpoint unreachable")


@pytest.fixture
def recording_store() -> RecordingStore:
    return RecordingStore()


@pytest.fixture
def recording_context(recording_store: RecordingStore) -> ApplicationContext:
    return build_context(store=recording_store)


def _no_allocation(event: TerminalEvent) -> TerminalEvent:
    return copy_model(event, allocation=[])


def test_completion_service_holds_no_process_wide_lock(
    context: ApplicationContext,
) -> None:
    """Active-active replicas share nothing in-process; the store serializes."""

    assert not hasattr(context.completion, "_lock"), (
        "CompletionService still carries an in-process RLock"
    )


def test_terminal_writes_all_happen_inside_the_event_transaction(
    recording_context: ApplicationContext,
    recording_store: RecordingStore,
    failed_event: TerminalEvent,
) -> None:
    """Event row, plan and decision are one unit keyed by the event."""

    decision = recording_context.completion.handle_terminal(
        _no_allocation(failed_event)
    )

    assert decision.status is DecisionStatus.PLAN_CREATED
    kinds = {kind for kind, _key in recording_store.writes}
    assert kinds == {"event", "plan", "decision"}, recording_store.writes
    assert all(
        key == failed_event.event_key for _kind, key in recording_store.writes
    ), f"a completion write escaped the event transaction: {recording_store.writes}"


def test_triage_submission_happens_after_the_decision_is_committed(
    recording_store: RecordingStore, failed_event: TerminalEvent
) -> None:
    """The remote call no longer sits between the two writes (P0-48B)."""

    diagnostics = ObservingDiagnostics(recording_store)
    build_context(store=recording_store)  # seeds the simulated profile
    service = CompletionService(recording_store, diagnostics)

    decision = service.handle_terminal(failed_event)

    assert decision.status is DecisionStatus.PENDING_TRIAGE
    assert len(diagnostics.submitted) == 1, "quick triage was not submitted"
    assert diagnostics.open_keys_at_submit == [[]], (
        "the diagnostic was submitted while the completion transaction was open"
    )
    persisted = diagnostics.decision_at_submit[0]
    assert persisted is not None, "the decision was not committed before submission"
    assert persisted.status is DecisionStatus.PENDING_TRIAGE
    assert persisted.diagnostic_request_id == diagnostics.submitted[0].request_id
    assert ("diagnostic", failed_event.event_key) in recording_store.writes, (
        "the diagnostic request was not persisted inside the transaction"
    )


def test_a_failing_submission_leaves_a_pending_triage_decision_for_the_watchdog(
    recording_store: RecordingStore, failed_event: TerminalEvent
) -> None:
    build_context(store=recording_store)  # seeds the simulated profile
    service = CompletionService(recording_store, FailingDiagnostics())

    decision = service.handle_terminal(failed_event)

    assert decision.status is DecisionStatus.PENDING_TRIAGE
    persisted = recording_store.get_decision_by_event(failed_event.event_key)
    assert persisted is not None, "the decision was lost with the failed submission"
    assert persisted.status is DecisionStatus.PENDING_TRIAGE


def test_stale_pending_triage_is_expired_into_a_conservative_plan(
    context: ApplicationContext, failed_event: TerminalEvent
) -> None:
    """P1-48E / P1-62F: ``PENDING_TRIAGE`` gets a watchdog."""

    service: CompletionService = context.completion
    pending = service.handle_terminal(failed_event)
    assert pending.status is DecisionStatus.PENDING_TRIAGE
    now = datetime.now(timezone.utc)

    fresh = service.reconcile_pending_triage(now=now)
    assert fresh == [], "a decision inside the deadline was expired"

    expired = service.reconcile_pending_triage(
        now=now + service.pending_triage_deadline + timedelta(minutes=1)
    )

    assert [item.event_key for item in expired] == [failed_event.event_key]
    decision = expired[0]
    assert decision.status is DecisionStatus.PLAN_CREATED
    assert decision.recovery_plan_id, "the expired decision has no plan"
    assert decision.diagnostic_request_id == pending.diagnostic_request_id
    stored = context.store.get_decision_by_event(failed_event.event_key)
    assert stored == decision
    plan = context.store.get_plan(decision.recovery_plan_id)
    actions = [step.action for step in plan.steps]
    assert RecoveryAction.ESCALATE_OPERATOR in actions, actions
    assert RecoveryAction.RESTART_WORKLOAD not in actions, (
        "a timed-out triage must not restart the workload automatically"
    )
    assert RecoveryAction.QUARANTINE not in actions, (
        "a control-plane timeout is not evidence against the nodes"
    )
    assert plan.trigger == "quick-triage:TIMEOUT"

    again = service.reconcile_pending_triage(
        now=now + service.pending_triage_deadline + timedelta(minutes=2)
    )
    assert again == [], "an already expired decision was expired twice"


def test_a_late_triage_report_after_expiry_is_a_duplicate_not_a_second_plan(
    context: ApplicationContext, failed_event: TerminalEvent
) -> None:
    service: CompletionService = context.completion
    pending = service.handle_terminal(failed_event)
    now = datetime.now(timezone.utc)
    expired = service.reconcile_pending_triage(
        now=now + service.pending_triage_deadline + timedelta(minutes=1)
    )[0]
    assert pending.diagnostic_request_id is not None

    late = service.handle_triage(
        TriageReport(
            request_id=pending.diagnostic_request_id,
            cluster_id=failed_event.cluster_id,
            attempt_id=failed_event.attempt_id,
            findings=[
                TriageFinding(
                    node_id="node-a",
                    outcome=TriageOutcome.FAIL,
                    proposed_action=RecoveryAction.RESET_GPU,
                )
            ],
            completed_at=now,
        )
    )

    assert late.duplicate is True
    assert late.recovery_plan_id == expired.recovery_plan_id


def test_pending_triage_deadline_is_a_constructor_setting(
    context: ApplicationContext,
) -> None:
    service = CompletionService(
        context.store, context.diagnostics, pending_triage_deadline=timedelta(minutes=3)
    )
    assert service.pending_triage_deadline == timedelta(minutes=3)
    with pytest.raises(ValueError):
        CompletionService(
            context.store, context.diagnostics, pending_triage_deadline=timedelta(0)
        )


class PollingDiagnostics:
    """A quick-triage adapter whose ``result`` poll records the store's open
    completion transactions at the moment it is called."""

    def __init__(self, store: RecordingStore) -> None:
        self.store = store
        self.polled: list[str] = []
        self.open_keys_at_result: list[list[str]] = []

    def submit(self, request: DiagnosticRequest) -> str:
        return request.request_id

    def result(self, request_id: str) -> TriageReport | None:
        self.polled.append(request_id)
        self.open_keys_at_result.append(list(self.store.open_keys))
        return None


def test_the_watchdog_polls_the_adapter_outside_the_completion_transaction(
    recording_store: RecordingStore, failed_event: TerminalEvent
) -> None:
    """Store review 2026-09-07, item D: the adapter's ``result`` is a remote
    call, so it must not run under the ``completion/<key>`` advisory lock and
    the pooled connection the transaction pins."""

    diagnostics = PollingDiagnostics(recording_store)
    build_context(store=recording_store)  # seeds the simulated profile
    service = CompletionService(recording_store, diagnostics)
    pending = service.handle_terminal(failed_event)
    assert pending.status is DecisionStatus.PENDING_TRIAGE
    # The submit path probes a synchronous adapter once, after its commit.
    assert diagnostics.polled == [pending.diagnostic_request_id]

    expired = service.reconcile_pending_triage(
        now=datetime.now(timezone.utc)
        + service.pending_triage_deadline
        + timedelta(minutes=1)
    )

    assert [item.event_key for item in expired] == [failed_event.event_key]
    assert diagnostics.polled == [pending.diagnostic_request_id] * 2, (
        "the watchdog must poll the adapter exactly once per stale decision"
    )
    assert diagnostics.open_keys_at_result == [[], []], (
        "the diagnostics adapter was polled while a completion transaction was open"
    )
    assert ("decision", failed_event.event_key) in recording_store.writes, (
        "the expiry decision was written outside the completion transaction"
    )


# --------------------------------------------------------------------------
# Control-plane review 2026-09-08, F-7: the watchdog's age predicate and its
# failure accounting.


def test_a_pending_triage_decision_without_a_diagnostic_row_is_not_expired_early(
    context: ApplicationContext, failed_event: TerminalEvent
) -> None:
    """``older_than`` treated "no diagnostic row" as "infinitely old": a decision
    whose diagnostic write was lost was expired into a conservative plan within
    the first scan. The event's ``ended_at`` is the fallback age, and a decision
    with neither is never a candidate."""

    service: CompletionService = context.completion
    context.store.save_event_if_absent(failed_event)
    context.store.save_decision(
        CompletionDecision(
            cluster_id=failed_event.cluster_id,
            attempt_id=failed_event.attempt_id,
            event_key=failed_event.event_key,
            status=DecisionStatus.PENDING_TRIAGE,
            reason="triage submitted",
            diagnostic_request_id="diag-never-written",
        )
    )
    just_after = failed_event.ended_at + timedelta(minutes=1)

    assert service.reconcile_pending_triage(now=just_after) == [], (
        "a decision with no diagnostic row was expired inside the deadline"
    )
    stored = context.store.get_decision_by_event(failed_event.event_key)
    assert stored is not None and stored.status is DecisionStatus.PENDING_TRIAGE

    expired = service.reconcile_pending_triage(
        now=failed_event.ended_at
        + service.pending_triage_deadline
        + timedelta(minutes=1)
    )
    assert [item.event_key for item in expired] == [failed_event.event_key]
    assert expired[0].status is DecisionStatus.PLAN_CREATED


def test_watchdog_failures_are_counted_by_reason_and_exported(
    context: ApplicationContext, failed_event: TerminalEvent
) -> None:
    """A decision the watchdog cannot expire -- here its event row is gone --
    used to log a traceback every scan and count nowhere."""

    from types import SimpleNamespace

    from gpu_fault.app.periodic_services import PeriodicServiceRunner

    service: CompletionService = context.completion
    old = failed_event.ended_at - timedelta(hours=2)
    context.store.save_diagnostic(
        DiagnosticRequest(
            request_id="diag-orphan",
            cluster_id=failed_event.cluster_id,
            attempt_id=failed_event.attempt_id,
            node_ids=["node-a"],
            checks=["dcgm"],
            created_at=old,
        )
    )
    context.store.save_decision(
        CompletionDecision(
            cluster_id=failed_event.cluster_id,
            attempt_id=failed_event.attempt_id,
            event_key=failed_event.event_key,
            status=DecisionStatus.PENDING_TRIAGE,
            reason="triage submitted",
            diagnostic_request_id="diag-orphan",
        )
    )

    assert service.reconcile_pending_triage(now=failed_event.ended_at) == []

    assert service.pending_triage_reconcile_failures_total == {"NotFoundError": 1}
    assert service.pending_triage_reconcile_failure_last_seen_timestamp_seconds > 0

    runner = PeriodicServiceRunner.__new__(PeriodicServiceRunner)
    runner.__init__(
        context=context,
        processor=SimpleNamespace(
            is_healthy=lambda: True,
            active_consumers=False,
            is_leader=lambda: True,
            owner_id="pod-a:1",
        ),
        stop=__import__("threading").Event(),
        identity_registries=[],
        ingest_node_health_findings=lambda *a, **k: None,
        notify_silent_collectors=lambda *a, **k: None,
    )
    # The runner exports the counter; rendering it as
    # gpu_fault_completion_pending_triage_reconcile_failures_total{reason}
    # is Agent 5's line in builtin_metric_contributors (HANDOFF-agent4).
    snapshot = runner.metrics_snapshot()
    assert snapshot["completion_pending_triage_reconcile_failures_total"] == {
        "NotFoundError": 1
    }
    assert (
        snapshot[
            "completion_pending_triage_reconcile_failure_last_seen_timestamp_seconds"
        ]
        > 0
    )

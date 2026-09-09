"""State the review made the code keep is visible on /metrics (F-L1, batch 3).

The dispatcher learned to count preemption-pending rows, the age of the
oldest PENDING workflow and generations waiting on an operator; the store
learned to count decisions by status, events without a decision and
incidents by state; the GPU metrics service counts CRITICAL findings it
deliberately closed without an incident. None of it had a line on the
endpoint, so none of it could carry an alert.
"""

from __future__ import annotations

import asyncio

from gpu_fault.app import ApplicationContext, create_app
from gpu_fault.models import CompletionDecision, DecisionStatus, IncidentState
from tests._builders import asgi_client, fault_incident


def _scrape(app) -> str:
    async def scenario():
        async with asgi_client(app) as client:
            return (await client.get("/metrics")).text

    return asyncio.run(scenario())


def _context(monkeypatch, suffix: str) -> ApplicationContext:
    monkeypatch.setenv("GPU_FAULT_PROCESSOR_MODE", "active-active")
    monkeypatch.setenv("POD_UID", f"pod-metrics-{suffix}")
    return ApplicationContext(execution_token="processor-metrics-token-" + suffix * 32)


def test_dispatcher_pending_state_reaches_the_metrics_endpoint(monkeypatch) -> None:
    context = _context(monkeypatch, "a")
    app = create_app(context)
    context.dispatcher.preemption_pending_seen_total = 31
    context.dispatcher.pending_age_seconds_max = 12.5
    context.dispatcher.pending_age_warnings_total = 32
    context.dispatcher.retired_generation_awaiting_operator = 33
    context.dispatcher.filtered_total["batch_limit"] = 4
    context.dispatcher.filtered_total["preemption_pending"] = 5

    text = _scrape(app)

    for line in (
        "gpu_fault_workflow_dispatch_preemption_pending_seen_total 31",
        "gpu_fault_workflow_pending_age_seconds_max 12.5",
        "gpu_fault_workflow_pending_age_warnings_total 32",
        "gpu_fault_workflow_retired_generation_awaiting_operator 33",
        'gpu_fault_workflow_dispatch_filtered_total{reason="batch_limit"} 4',
        'gpu_fault_workflow_dispatch_filtered_total{reason="preemption_pending"} 5',
    ):
        assert line in text, line


def test_dispatch_filter_reasons_are_zero_filled_before_the_first_cycle(
    monkeypatch,
) -> None:
    context = _context(monkeypatch, "b")
    app = create_app(context)

    text = _scrape(app)

    for line in (
        'gpu_fault_workflow_dispatch_filtered_total{reason="batch_limit"} 0',
        'gpu_fault_workflow_dispatch_filtered_total{reason="preemption_pending"} 0',
    ):
        assert line in text, line


def test_a_filter_reason_outside_the_known_set_is_still_exported(monkeypatch) -> None:
    """The dispatcher owns the totals (§69); a reason it counted that has no
    zero-filled series of its own still reaches the endpoint."""

    context = _context(monkeypatch, "c")
    app = create_app(context)
    context.dispatcher.filtered_total["predecessor"] = 2

    text = _scrape(app)

    assert 'gpu_fault_workflow_dispatch_filtered_total{reason="predecessor"} 2' in text


def test_completion_decision_and_incident_state_gauges_are_zero_filled(
    monkeypatch,
) -> None:
    context = _context(monkeypatch, "d")
    app = create_app(context)

    text = _scrape(app)

    for status in DecisionStatus:
        line = f'gpu_fault_completion_decisions{{status="{status.value}"}} 0'
        assert line in text, line
    assert "gpu_fault_completion_events_without_decision 0" in text
    for state in IncidentState:
        line = f'gpu_fault_incidents_by_state{{state="{state.value}"}} 0'
        assert line in text, line


def test_completion_decision_and_incident_state_gauges_count_store_rows(
    monkeypatch,
) -> None:
    context = _context(monkeypatch, "e")
    app = create_app(context)
    context.store.save_decision(
        CompletionDecision(
            cluster_id="cluster-a",
            attempt_id="attempt-1",
            event_key="cluster-a/attempt-1/TrainingAttemptTerminal",
            status=DecisionStatus.NO_ACTION,
            reason="no fault evidence",
        )
    )
    context.store.save_incident(
        fault_incident("incident-1", "event-1", state=IncidentState.ESCALATED)
    )
    context.store.save_incident(
        fault_incident("incident-2", "event-2", state=IncidentState.ESCALATED)
    )

    text = _scrape(app)

    for line in (
        'gpu_fault_completion_decisions{status="NO_ACTION"} 1',
        'gpu_fault_completion_decisions{status="PLAN_CREATED"} 0',
        'gpu_fault_incidents_by_state{state="ESCALATED"} 2',
        'gpu_fault_incidents_by_state{state="DETECTED"} 0',
    ):
        assert line in text, line


def test_gpu_findings_closed_without_an_incident_reach_the_metrics_endpoint(
    monkeypatch,
) -> None:
    context = _context(monkeypatch, "f")
    app = create_app(context)

    before = _scrape(app)
    context.gpu_metrics.findings_without_incident["suppressed_by_composite"] += 7
    after = _scrape(app)

    zero = 'gpu_fault_gpu_findings_without_incident_total{reason="suppressed_by_composite"} 0'
    seven = 'gpu_fault_gpu_findings_without_incident_total{reason="suppressed_by_composite"} 7'
    assert zero in before, "the known reason must be present at zero"
    assert seven in after, after

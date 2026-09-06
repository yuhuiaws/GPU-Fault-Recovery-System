"""The PENDING_TRIAGE watchdog deadline is operator configuration, at both
places the application builds its ``CompletionService`` (F-G2 (4)).

The service already refused a non-positive deadline; the application never
passed one, so the 15-minute default was the only value a deployment could
have.
"""

from __future__ import annotations

from datetime import timedelta

import pytest

from gpu_fault.app import ApplicationContext

ENV = "GPU_FAULT_PENDING_TRIAGE_DEADLINE_SECONDS"


def test_the_simulated_context_reads_the_deadline_from_the_environment(monkeypatch):
    monkeypatch.setenv(ENV, "120")

    context = ApplicationContext()

    assert context.completion.pending_triage_deadline == timedelta(seconds=120)


def test_the_deadline_defaults_to_fifteen_minutes(monkeypatch):
    monkeypatch.delenv(ENV, raising=False)

    context = ApplicationContext()

    assert context.completion.pending_triage_deadline == timedelta(seconds=900)


@pytest.mark.parametrize("value", ["0", "-5"])
def test_a_non_positive_deadline_is_refused_by_name(monkeypatch, value):
    monkeypatch.setenv(ENV, value)

    with pytest.raises(ValueError, match=ENV):
        ApplicationContext()


def test_the_production_context_passes_the_same_deadline(monkeypatch, tmp_path):
    monkeypatch.setenv("GPU_FAULT_EXECUTOR_MODE", "active")
    monkeypatch.setenv("GPU_FAULT_ALLOW_SINGLE_CLUSTER", "true")
    monkeypatch.setenv("GPU_FAULT_ACKNOWLEDGE_NO_ALERT_CHANNEL", "true")
    monkeypatch.setenv("GPU_FAULT_STORE_URL", f"sqlite:///{tmp_path / 'active.db'}")
    monkeypatch.setenv("GPU_FAULT_EXECUTION_TOKEN", "t" * 32)
    monkeypatch.setenv("GPU_FAULT_ALLOWED_OPERATIONS", "FREEZE_EVIDENCE")
    monkeypatch.setenv(ENV, "300")

    context = ApplicationContext.from_environment()

    assert context.completion.pending_triage_deadline == timedelta(seconds=300)

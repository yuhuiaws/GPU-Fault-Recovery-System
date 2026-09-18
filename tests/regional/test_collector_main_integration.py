"""Offline coverage of the collector's integrated terminal and budget contracts."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest

from scripts.e2e.regional import collector_acceptance_fixture as collector
from scripts.e2e.regional import run_collect016_training_recovery as runner


@pytest.mark.parametrize("successor", ["SUCCEEDED", "FAILED", "BLOCKED", "SUPERSEDED"])
def test_superseded_predecessor_is_settled_with_its_successor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, successor: str
) -> None:
    fixture = object.__new__(collector.CollectorAcceptanceFixture)
    light = {"workflows": [{"status": "SUPERSEDED"}, {"status": successor}]}
    full = {**light, "evidence": [{"record_id": "r1"}], "evidence_scanned": True}
    read = Mock(side_effect=[light, full])
    sleep = Mock(side_effect=AssertionError("settled workflows must not sleep"))
    monkeypatch.setattr(fixture, "store_snapshot", read)
    monkeypatch.setattr(collector.time, "sleep", sleep)

    assert (
        fixture.wait_marker("marker-a", case_dir=tmp_path, terminal_workflow=True)
        == full
    )
    assert [call.kwargs["scan_evidence"] for call in read.call_args_list] == [
        False,
        True,
    ]


def test_full_evidence_read_rechecks_terminal_status_after_superseded_light_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = object.__new__(collector.CollectorAcceptanceFixture)
    states = [
        {"workflows": [{"status": "SUPERSEDED"}]},
        {
            "workflows": [{"status": "SUPERSEDED"}, {"status": "RUNNING"}],
            "evidence": [{"record_id": "r1"}],
        },
        {"workflows": [{"status": "SUPERSEDED"}, {"status": "SUCCEEDED"}]},
        {
            "workflows": [{"status": "SUPERSEDED"}, {"status": "SUCCEEDED"}],
            "evidence": [{"record_id": "r1"}],
        },
    ]
    read = Mock(side_effect=states)
    sleep = Mock()
    monkeypatch.setattr(fixture, "store_snapshot", read)
    monkeypatch.setattr(collector.time, "sleep", sleep)

    assert (
        fixture.wait_marker("marker-a", case_dir=tmp_path, terminal_workflow=True)
        == states[-1]
    )
    sleep.assert_called_once_with(collector.STORE_POLL_SECONDS)
    assert [call.kwargs["scan_evidence"] for call in read.call_args_list] == [
        False,
        True,
        False,
        True,
    ]


@pytest.fixture
def budget_states() -> SimpleNamespace:
    return SimpleNamespace(
        a={
            "incident": {"workload_identity_source": "SOLE_ACTIVE_ATTEMPT_ON_NODE"},
            "workflow": {
                "official_steps": [
                    {"operation": "RESTART_WORKLOAD", "parameters": {"job_id": "job-a"}}
                ],
                "blocked_reasons": [],
            },
        },
        b={
            "workflow": {
                "status": "FAILED",
                "completed_operations": ["FREEZE_EVIDENCE", "STOP_WORKLOADS"],
                "step_executions": [
                    {
                        "operation": "RESTART_WORKLOAD",
                        "status": "FAILED",
                        "details": {"reason": "RESTART_BUDGET_EXHAUSTED"},
                    }
                ],
            },
            "commands": [
                {
                    "command_id": "stop-command",
                    "status": "SUCCEEDED",
                    "step": {"operation": "STOP_WORKLOADS"},
                }
            ],
            "restart_budget": {"restart_count": 1, "budget": 1},
        },
    )


def test_budget_withholds_restart_while_allowing_completed_stop_and_evidence(
    budget_states: SimpleNamespace,
) -> None:
    assert (
        runner.restart_budget_section_errors(
            budget_states.a, budget_states.b, job_id="job-a"
        )
        == []
    )


@pytest.mark.parametrize("status", ["PENDING", "LEASED", "SUCCEEDED", "FAILED"])
def test_any_dispatched_restart_command_rejects_the_budget_verdict(
    budget_states: SimpleNamespace, status: str
) -> None:
    budget_states.b["commands"].append(
        {"status": status, "step": {"operation": "RESTART_WORKLOAD"}}
    )
    errors = runner.restart_budget_section_errors(
        budget_states.a, budget_states.b, job_id="job-a"
    )
    assert any("RESTART_WORKLOAD remote command" in error for error in errors), errors


def test_completed_restart_without_a_remote_command_still_rejects_the_verdict(
    budget_states: SimpleNamespace,
) -> None:
    budget_states.b["workflow"]["completed_operations"].append("RESTART_WORKLOAD")
    errors = runner.restart_budget_section_errors(
        budget_states.a, budget_states.b, job_id="job-a"
    )
    assert "B: budget rejection completed RESTART_WORKLOAD" in errors, errors


@pytest.mark.parametrize(
    "report",
    [
        None,
        False,
        {},
        "unknown",
        [None],
        [{}],
        [{"step": None}],
        [{"step": {}}],
        [{"step": {"operation": "UNKNOWN"}}],
        [{"step": {"operation": []}}],
    ],
)
def test_unknown_command_reports_cannot_prove_the_restart_was_withheld(
    budget_states: SimpleNamespace, report: Any
) -> None:
    budget_states.b["commands"] = report
    errors = runner.restart_budget_section_errors(
        budget_states.a, budget_states.b, job_id="job-a"
    )
    assert "B: remote command report is missing or malformed" in errors, errors


def test_missing_command_report_cannot_prove_the_restart_was_withheld(
    budget_states: SimpleNamespace,
) -> None:
    budget_states.b.pop("commands")
    errors = runner.restart_budget_section_errors(
        budget_states.a, budget_states.b, job_id="job-a"
    )
    assert "B: remote command report is missing or malformed" in errors, errors


@pytest.mark.parametrize(
    "completed", [None, False, {}, "STOP_WORKLOADS", [[]], ["UNKNOWN"]]
)
def test_malformed_completed_operation_reports_are_rejected(
    budget_states: SimpleNamespace, completed: Any
) -> None:
    budget_states.b["workflow"]["completed_operations"] = completed
    errors = runner.restart_budget_section_errors(
        budget_states.a, budget_states.b, job_id="job-a"
    )
    assert "B: completed workflow operations are malformed" in errors, errors

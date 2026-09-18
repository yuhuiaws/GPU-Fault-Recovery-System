"""Reject inconsistent observations through complete NET command runners."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import pytest

from scripts.e2e.regional import run_net002_command_recovery as net002
from scripts.e2e.regional import run_net003_result_retry as net003
from scripts.e2e.regional import run_net006_lease_loss_withheld_result as net006
from tests.regional import _cov95_net_commands as command_support
from tests.regional._cov95_collect_net import no_external_effects  # noqa: F401
from tests.regional._cov95_net_commands import run_command

command_runner = command_support.command_runner


def failed_report(harness: Any) -> dict[str, Any]:
    assert run_command(harness) == 1
    path = (
        harness.root
        / "cases"
        / harness.module.CASE_ID
        / f"{harness.module.CASE_ID}.json"
    )
    report = json.loads(path.read_text())
    assert report["verdict"] == "FAIL"
    assert report.get("errors") or report.get("error"), (
        "rejected evidence needs a reason"
    )
    assert any(call[0] == "cleanup" for call in harness.calls), (
        "evidence mismatch cannot bypass owned cleanup"
    )
    return report


@pytest.mark.parametrize("command_runner", [net002], indirect=True, ids=["net002"])
@pytest.mark.parametrize(
    "change",
    [
        ("state", "claimed_total", 3),
        ("state", "reported_failures", 2),
        ("state", "unexpected_failures", 1),
        ("state", "lease_renewal_failures", 0),
        ("ready", "block_rollback_seconds", 90),
        ("ready", "lease_seconds", 30),
        ("ready", "result_submission_gate", False),
        ("ready", "action_requires_network_block", False),
        ("expired", "status", "PENDING"),
        ("expired", "lease_expires_at", "changed"),
    ],
)
def test_net002_rejects_counter_timer_gate_and_lease_mismatches(
    command_runner: Any, change: tuple[str, str, Any]
) -> None:
    section, key, value = change
    getattr(command_runner, section)[key] = value
    failed_report(command_runner)


@pytest.mark.parametrize("command_runner", [net002], indirect=True, ids=["net002"])
@pytest.mark.parametrize(
    "problem", ["physical-count", "ledger-keys", "cached", "log", "overrun"]
)
def test_net002_requires_one_physical_action_and_bounded_stale_result_window(
    command_runner: Any, problem: str
) -> None:
    harness = command_runner
    if problem == "physical-count":
        harness.states["/state/ledger.json"]["physical_count"] = 2
    elif problem == "ledger-keys":
        harness.states["/state/ledger.json"]["keys"] = []
    elif problem == "cached":
        harness.final["result_details"]["cached"] = False
    elif problem == "log":
        harness.logs = ""
    else:
        harness.clock.on_sleep = lambda seconds: setattr(
            harness.clock, "now", harness.clock.now + 150
        )
    failed_report(harness)


@pytest.mark.parametrize("command_runner", [net003], indirect=True, ids=["net003"])
@pytest.mark.parametrize(
    "change",
    [
        ("state", "reported_failures", 1),
        ("state", "unexpected_failures", 1),
        ("state", "lease_renewal_failures", 1),
        ("ready", "drop_rollback_seconds", 9),
        ("ready", "lease_seconds", 30),
        ("ready", "result_connection_reset", False),
        ("ready", "response_loss_mode", "reset-before-response"),
        ("ready", "result_retry_owner", "test-client"),
        ("ready", "http_timeout_seconds", 5),
    ],
)
def test_net003_rejects_changed_transport_contract_and_counters(
    command_runner: Any, change: tuple[str, str, Any]
) -> None:
    section, key, value = change
    getattr(command_runner, section)[key] = value
    failed_report(command_runner)


@pytest.mark.parametrize("command_runner", [net003], indirect=True, ids=["net003"])
@pytest.mark.parametrize(
    "problem",
    [
        "ledger-count",
        "ledger-keys",
        "drop-reset",
        "unforwarded",
        "no-response",
        "no-interruption",
        "rollback",
        "missing-update",
        "missing-replay-time",
        "wrong-replay-status",
        "notification-identity",
        "baseline-notification",
        "baseline-delivery",
        "baseline-result",
        "baseline-dedup",
        "final-notification",
        "final-delivery",
        "final-result",
        "final-status",
        "final-dedup",
    ],
)
def test_net003_result_and_notification_identity_mismatches_fail_the_real_case(
    command_runner: Any, problem: str
) -> None:
    harness = command_runner
    if problem == "ledger-count":
        harness.states["/state/ledger.json"]["physical_count"] = 2
    elif problem == "ledger-keys":
        harness.states["/state/ledger.json"]["keys"] = []
    elif problem in {"drop-reset", "unforwarded", "no-response"}:
        key = {
            "drop-reset": "connection_reset",
            "unforwarded": "request_forwarded",
            "no-response": "upstream_response_bytes",
        }[problem]
        harness.states["/state/drop-observed.json"][key] = False
    elif problem == "no-interruption":
        harness.states["/state/result-interrupted.json"]["first_post_succeeded"] = True
    elif problem == "rollback":
        harness.rollback = True
    elif problem == "missing-update":
        harness.final["updated_at"] = "not-a-time"
    elif problem == "missing-replay-time":
        harness.states["/state/result-replays.json"]["replay_sent_at_epoch"] = None
    elif problem == "wrong-replay-status":
        harness.states["/state/result-replays.json"]["responses"][0]["status"] = (
            "FAILED"
        )
    elif problem == "notification-identity":
        harness.final["result_details"]["notification_id"] = "foreign"
    elif problem == "baseline-result":
        harness.notifications["objects"]["notification_result"] = {"count": 1}
    elif problem == "final-result":
        harness.final_notification_result["count"] = 2
    elif problem == "final-status":
        harness.final_notification_result["status"] = "SENT"
    else:
        prefix, kind = problem.split("-", 1)
        document = (
            harness.notifications
            if prefix == "baseline"
            else harness.final_notifications
        )
        if kind == "dedup":
            document["dedup_link_count"] = 2
        else:
            document["objects"][
                "notification_delivery" if kind == "delivery" else "notification"
            ]["count"] = 2
    failed_report(harness)


@pytest.mark.parametrize("command_runner", [net006], indirect=True, ids=["net006"])
@pytest.mark.parametrize(
    "problem",
    [
        "rollback",
        "overrun",
        "missing-gate",
        "cached",
        "final-failed",
        "executor-id",
        "log",
        "ledger-keys",
    ],
)
def test_net006_withheld_result_requires_bound_timing_and_identity(
    command_runner: Any, problem: str
) -> None:
    harness = command_runner
    if problem == "rollback":
        harness.rollback = True
    elif problem == "overrun":
        harness.clock.on_sleep = lambda seconds: setattr(
            harness.clock, "now", harness.clock.now + 200
        )
    elif problem == "missing-gate":
        harness.states["/state/action-gate-observed.json"]["observed_at_epoch"] = None
    elif problem == "cached":
        harness.states["/state/action-returned.json"]["cached"] = True
    elif problem == "final-failed":
        harness.final["status"] = "FAILED"
    elif problem == "executor-id":
        harness.states["/state/claim-state.json"]["executor_id"] = "foreign"
    elif problem == "ledger-keys":
        harness.states["/state/ledger.json"]["keys"] = []
    else:
        harness.logs = ""
    failed_report(harness)


@pytest.mark.parametrize(
    "value", [None, "invalid", datetime(2026, 9, 1), "2026-09-01T00:00:00Z"]
)
def test_net006_timestamp_parser_normalizes_only_parseable_observations(
    value: Any,
) -> None:
    observed = net006.verdicts.parse_time(value)
    if value is None or value == "invalid":
        assert observed is None
    else:
        assert observed == datetime(2026, 9, 1, tzinfo=timezone.utc)

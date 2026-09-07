"""Contract tests for GF-REGIONAL-NOTIFY-007 (delivery states).

The drill probe is also exercised for real against an in-memory store with a
recording notifier, so the JSON the runner judges is the JSON the shipped
``dispatch_outbox`` produces; the live runner only swaps in the SES notifier.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pytest

from gpu_fault.models import (
    AdvisoryNotification,
    NotificationResult,
    NotificationStatus,
)
from scripts.e2e.regional import notify007_verdicts as verdicts
from scripts.e2e.regional import run_notify007_delivery_states as notify007
from scripts.e2e.regional.probes import notify007_delivery_drill as drill

ROOT = Path(__file__).resolve().parents[2]


def _text(errors: list[str]) -> str:
    return "\n".join(errors)


class RecordingNotifier:
    def __init__(self) -> None:
        self.sent: list[str] = []

    def send(self, notification: AdvisoryNotification) -> NotificationResult:
        self.sent.append(notification.notification_id)
        return NotificationResult(
            notification_id=notification.notification_id,
            status=NotificationStatus.SENT,
            provider_message_id=f"ses-{len(self.sent)}",
        )


def test_the_drill_reproduces_retry_then_sent_and_dead_on_the_shipped_outbox(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        drill, "notification_notifier_from_environment", RecordingNotifier
    )
    monkeypatch.setattr(time, "sleep", lambda seconds: None)
    monkeypatch.setattr(drill, "RETRY_BASE_SECONDS", 0)

    retry = drill.run_retry_phase("cluster-a", "d1")
    dead = drill.run_dead_phase("cluster-a", "d1")

    assert verdicts.retry_phase_errors(retry) == [], retry
    assert verdicts.dead_phase_errors(dead) == [], dead


def _retry(**overrides: Any) -> dict[str, Any]:
    value = {
        "notifier_attempts": 2,
        "last_cycle_timestamp_seconds": 1.0,
        "after_first_cycle": {
            "report": {"attempted": 1, "sent": 0, "failed": 1},
            "stats": {
                "by_status": {
                    "PENDING": 0,
                    "LEASED": 0,
                    "RETRY": 1,
                    "SENT": 0,
                    "DEAD": 0,
                }
            },
            "result": None,
            "dead_lettered_total": 0,
        },
        "after_second_cycle": {
            "report": {"attempted": 1, "sent": 1, "failed": 0},
            "stats": {
                "by_status": {
                    "PENDING": 0,
                    "LEASED": 0,
                    "RETRY": 0,
                    "SENT": 1,
                    "DEAD": 0,
                }
            },
            "result": {"status": "SENT", "provider_message_id_present": True},
            "dead_lettered_total": 0,
        },
    }
    value.update(overrides)
    return value


def test_the_retry_phase_contract_passes_when_retry_is_not_failed() -> None:
    assert verdicts.retry_phase_errors(_retry()) == []


def test_the_retry_phase_contract_rejects_a_failed_result_on_a_retryable_failure() -> (
    None
):
    first = dict(_retry()["after_first_cycle"])
    first["result"] = {"status": "FAILED", "provider_message_id_present": False}
    assert "result row was written for a retryable failure" in _text(
        verdicts.retry_phase_errors(_retry(after_first_cycle=first))
    )
    first = dict(_retry()["after_first_cycle"])
    first["stats"] = {"by_status": {"RETRY": 0, "DEAD": 1, "SENT": 0}}
    errors = verdicts.retry_phase_errors(_retry(after_first_cycle=first))
    assert "expected RETRY=1" in _text(errors) and "counted as DEAD or SENT" in _text(
        errors
    ), errors
    second = dict(_retry()["after_second_cycle"])
    second["result"] = {"status": "SENT", "provider_message_id_present": False}
    assert "did not end SENT with a provider id" in _text(
        verdicts.retry_phase_errors(_retry(after_second_cycle=second))
    )
    assert "expected 2" in _text(
        verdicts.retry_phase_errors(_retry(notifier_attempts=3))
    )


def test_the_dead_phase_contract_wants_dead_terminal_failed_and_one_dead_letter() -> (
    None
):
    dead = {
        "notifier_attempts": 1,
        "stats": {"by_status": {"DEAD": 1, "RETRY": 0}},
        "result": {"status": "FAILED"},
        "dead_lettered_total": 1,
    }
    assert verdicts.dead_phase_errors(dead) == []
    assert "no terminal FAILED result" in _text(
        verdicts.dead_phase_errors({**dead, "result": None})
    )
    assert "expected DEAD=1" in _text(
        verdicts.dead_phase_errors({**dead, "stats": {"by_status": {"RETRY": 1}}})
    )
    assert "dead_lettered_total is 0" in _text(
        verdicts.dead_phase_errors({**dead, "dead_lettered_total": 0})
    )


def _metrics(failed: int, dead: int, *, families: bool = True) -> str:
    lines = [f'gpu_fault_notification_total{{status="FAILED"}} {failed}']
    if families:
        lines.extend(
            f'gpu_fault_notification_delivery_total{{status="{status}"}} {dead if status == "DEAD" else 0}'
            for status in ("PENDING", "LEASED", "RETRY", "SENT", "DEAD")
        )
        lines.extend(
            [
                "gpu_fault_notification_oldest_pending_age_seconds 0",
                "gpu_fault_notification_dead_lettered_total 0",
                "gpu_fault_notification_dispatch_last_cycle_timestamp_seconds 1.0",
            ]
        )
    return "\n".join(lines)


def test_the_exported_families_and_production_untouched_contracts() -> None:
    assert verdicts.exported_family_errors([_metrics(0, 0)]) == []
    errors = verdicts.exported_family_errors([_metrics(0, 0, families=False)])
    assert "gpu_fault_notification_delivery_total is not exported" in _text(errors), (
        errors
    )
    partial = _metrics(0, 0).replace(
        'gpu_fault_notification_delivery_total{status="DEAD"} 0\n', ""
    )
    assert "no status='DEAD' series" in _text(
        verdicts.exported_family_errors([partial])
    )
    assert (
        verdicts.production_untouched_errors([_metrics(2, 1)], [_metrics(2, 1)]) == []
    )
    assert "status=FAILED" in _text(
        verdicts.production_untouched_errors([_metrics(2, 1)], [_metrics(3, 1)])
    )
    assert "status=DEAD" in _text(
        verdicts.production_untouched_errors([_metrics(2, 1)], [_metrics(2, 2)])
    )


def test_the_shipped_alert_rule_reads_terminal_failed_and_has_its_runbook() -> None:
    rules = (ROOT / "deploy/observability/amp-rules.yaml").read_text(encoding="utf-8")
    runbook = (ROOT / "docs/管理员日常运维.md").read_text(encoding="utf-8")
    assert verdicts.alert_rule_errors(rules, runbook) == []
    retry_rule = rules.replace(
        'gpu_fault_notification_total{status="FAILED"}',
        'gpu_fault_notification_delivery_total{status="RETRY"}',
    )
    errors = verdicts.alert_rule_errors(retry_rule, runbook)
    assert "does not read gpu_fault_notification_total" in _text(errors), errors
    assert "no runbook anchor" in _text(verdicts.alert_rule_errors(rules, "nothing"))


def test_the_runner_is_plan_by_default_with_the_documented_flags(
    tmp_path: Path,
) -> None:
    parser = notify007.parser()
    arguments = parser.parse_args(["--run-dir", str(tmp_path)])
    assert arguments.execute is False and arguments.confirm == ""
    for flag in ("--plan", "--execute", "--confirm", "--maintenance-window-end"):
        assert flag in parser.format_help(), flag
    with pytest.raises(SystemExit):
        parser.parse_args(["--run-dir", str(tmp_path), "--plan", "--execute"])
    assert notify007.CASE_ID == "GF-REGIONAL-NOTIFY-007"
    assert notify007.CONFIRMATION == "NOTIFY007_EXECUTE"
    assert verdicts.PREDECESSOR_CASE_ID == "GF-REGIONAL-NOTIFY-006"
    assert (
        ROOT / "scripts/e2e/regional/run_notify007_delivery_states.py"
    ).stat().st_mode & 0o777 == 0o775

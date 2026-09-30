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
import yaml

from gpu_fault.models import (
    AdvisoryNotification,
    NotificationResult,
    NotificationStatus,
)
from scripts.e2e.regional import notify007_verdicts as verdicts
from scripts.e2e.regional import run_notify007_delivery_states as notify007
from scripts.e2e.regional.probes import notify007_delivery_drill as drill

ROOT = Path(__file__).resolve().parents[2]
TERMINAL_METRIC = "gpu_fault_notification_terminal_failure_last_seen_timestamp_seconds"
LATEST_FAILURE = (
    f"max by (control_plane_cluster, region) (max_over_time({TERMINAL_METRIC}[15m]))"
)
RUNBOOK_TEXT = f"### {verdicts.FAILING_ALERT}\n\n### {verdicts.UNDELIVERED_ALERT}\n"


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
                "gpu_fault_notification_terminal_failure_last_seen_timestamp_seconds 0",
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
    runbook = (ROOT / "docs/en/administrator-operations.md").read_text(encoding="utf-8")
    assert verdicts.alert_rule_errors(rules, runbook) == []
    assert "no runbook anchor" in _text(verdicts.alert_rule_errors(rules, "nothing"))


@pytest.fixture
def alert_document() -> dict[str, Any]:
    return {
        "groups": [
            {
                "name": "notification-contract",
                "rules": [
                    {
                        "alert": verdicts.FAILING_ALERT,
                        "expr": (
                            f"(time() - {LATEST_FAILURE}) < 900 "
                            f"and {LATEST_FAILURE} > 0"
                        ),
                        "for": "5m",
                        "labels": {"severity": "critical"},
                        "annotations": {
                            "runbook_url": "docs/en/administrator-operations.md"
                            "#gpufaultnotificationdeliveryfailing"
                        },
                    },
                    {
                        "alert": verdicts.UNDELIVERED_ALERT,
                        "expr": (
                            "max by (control_plane_cluster, region) "
                            "(gpu_fault_notification_oldest_pending_age_seconds) > 1800"
                        ),
                        "for": "10m",
                        "labels": {"severity": "critical"},
                        "annotations": {
                            "runbook_url": "docs/en/administrator-operations.md"
                            "#gpufaultnotificationundeliveredtoolong"
                        },
                    },
                ],
            }
        ]
    }


def _failing_rule(document: dict[str, Any]) -> dict[str, Any]:
    return next(
        rule
        for group in document["groups"]
        for rule in group["rules"]
        if rule.get("alert") == verdicts.FAILING_ALERT
    )


def _alert_errors(document: dict[str, Any], runbook: str = RUNBOOK_TEXT) -> list[str]:
    return verdicts.alert_rule_errors(yaml.safe_dump(document), runbook)


@pytest.mark.parametrize("flow_style", [False, True])
def test_alert_contract_accepts_structured_yaml_and_promql_whitespace(
    alert_document: dict[str, Any], flow_style: bool
) -> None:
    rule = _failing_rule(alert_document)
    rule["expr"] = rule["expr"].replace("(", "( \n").replace(")", "\t)")
    text = yaml.safe_dump(alert_document, default_flow_style=flow_style)
    assert verdicts.alert_rule_errors(text, RUNBOOK_TEXT) == []


@pytest.mark.parametrize(
    "metric",
    [
        'gpu_fault_notification_total{status="FAILED"}',
        'gpu_fault_notification_delivery_total{status="RETRY"}',
        "gpu_fault_notification_dead_lettered_total",
        TERMINAL_METRIC + "_unrelated",
    ],
    ids=["retained-failed", "retry-population", "dead-counter", "wrong-name"],
)
def test_alert_contract_rejects_nonterminal_event_metrics(
    alert_document: dict[str, Any], metric: str
) -> None:
    rule = _failing_rule(alert_document)
    assert rule["expr"].count(TERMINAL_METRIC) == 2
    rule["expr"] = rule["expr"].replace(TERMINAL_METRIC, metric)
    assert "terminal failure event-time contract" in _text(
        _alert_errors(alert_document)
    )


@pytest.mark.parametrize(
    "old,new",
    [
        ("[15m]", "[5m]"),
        ("[15m]", "[30m]"),
        ("[15m]", ""),
        ("max_over_time", "last_over_time"),
        ("max_over_time", "increase"),
        ("max_over_time", "delta"),
        ("< 900", "< 1800"),
        ("< 900", "<= 900"),
        ("< 900", "> 900"),
        ("< 900", "< bool 900"),
        ("> 0", ">= 0"),
        ("> 0", "> bool 0"),
        ("> 0", "> 1"),
        ("and", "or"),
        ("control_plane_cluster, region", "region"),
        ("control_plane_cluster, region", "control_plane_cluster, region, pod"),
        ("max by", "sum by"),
        ("time()", "timestamp(vector(1))"),
    ],
    ids=[
        "short-window",
        "long-window",
        "no-window",
        "no-restart-memory",
        "counter-increase",
        "population-delta",
        "long-age",
        "inclusive-age",
        "stale-age",
        "bool-age",
        "zero-stamp",
        "bool-stamp",
        "wrong-stamp-threshold",
        "or",
        "missing-cluster",
        "replica-fanout",
        "sum",
        "scrape-time",
    ],
)
def test_alert_contract_rejects_changed_event_window_or_logic(
    alert_document: dict[str, Any], old: str, new: str
) -> None:
    rule = _failing_rule(alert_document)
    assert old in rule["expr"], "negative control must change the parsed expression"
    rule["expr"] = rule["expr"].replace(old, new, 1)
    assert "terminal failure event-time contract" in _text(
        _alert_errors(alert_document)
    )


@pytest.mark.parametrize("wrapper", ["comment", "string", "always-true", "no-guard"])
def test_alert_contract_cannot_pass_from_an_expression_mention(
    alert_document: dict[str, Any], wrapper: str
) -> None:
    rule = _failing_rule(alert_document)
    expr = rule["expr"]
    if wrapper == "comment":
        rule["expr"] = f"vector(1) # {expr}"
    elif wrapper == "string":
        rule["expr"] = f'label_replace(vector(1), "note", "{expr}", "__name__", ".*")'
    elif wrapper == "always-true":
        rule["expr"] = f"({expr}) or vector(1)"
    else:
        rule["expr"] = expr.partition(" and ")[0]
    assert "terminal failure event-time contract" in _text(
        _alert_errors(alert_document)
    )


@pytest.mark.parametrize(
    "field,value,error",
    [
        ("expr", None, "event-time contract"),
        ("expr", True, "event-time contract"),
        ("expr", 1, "event-time contract"),
        ("expr", [], "event-time contract"),
        ("expr", {"text": LATEST_FAILURE}, "event-time contract"),
        ("labels", {"severity": "warning"}, "severity must be critical"),
        ("labels", {"severity": True}, "severity must be critical"),
        ("labels", {}, "severity must be critical"),
        ("labels", None, "severity must be critical"),
        ("labels", ["critical"], "severity must be critical"),
        ("for", "1m", "hold duration must be 5m"),
        ("for", "15m", "hold duration must be 5m"),
        ("for", None, "hold duration must be 5m"),
        ("for", 300, "hold duration must be 5m"),
        ("for", True, "hold duration must be 5m"),
        ("annotations", {}, "no runbook anchor"),
        ("annotations", None, "no runbook anchor"),
        ("annotations", ["runbook_url"], "no runbook anchor"),
    ],
)
def test_alert_contract_rejects_wrong_typed_fields_severity_and_hold(
    alert_document: dict[str, Any], field: str, value: Any, error: str
) -> None:
    _failing_rule(alert_document)[field] = value
    assert error in _text(_alert_errors(alert_document))


@pytest.mark.parametrize(
    "text",
    [
        "",
        "null",
        "groups: [",
        "groups: []\n---\ngroups: []",
        "[]",
        "true",
        "rules",
        "groups: null",
        "groups: {}",
        "groups: [null]",
        "groups: [{rules: false}]",
        "groups: [{rules: [null]}]",
        "groups: [{rules: [unstructured]}]",
    ],
)
def test_alert_contract_rejects_malformed_yaml_structure(text: str) -> None:
    errors = verdicts.alert_rule_errors(text, RUNBOOK_TEXT)
    assert errors, "malformed YAML must produce a failing rule verdict"
    assert verdicts.case_verdict({"alert_rule": errors}) == "FAIL"


@pytest.mark.parametrize("name", [verdicts.FAILING_ALERT, verdicts.UNDELIVERED_ALERT])
@pytest.mark.parametrize("duplicate", [False, True])
def test_alert_contract_requires_one_definition_per_alert(
    alert_document: dict[str, Any], name: str, duplicate: bool
) -> None:
    rules = alert_document["groups"][0]["rules"]
    target = next(rule for rule in rules if rule["alert"] == name)
    if duplicate:
        alert_document["groups"].append(
            {"name": "duplicate", "rules": [{**target, "expr": "vector(1)"}]}
        )
    else:
        rules.remove(target)
    expected = "defined more than once" if duplicate else "not defined"
    assert f"alert {name} is {expected}" in _alert_errors(alert_document)


@pytest.mark.parametrize("name", [verdicts.FAILING_ALERT, verdicts.UNDELIVERED_ALERT])
@pytest.mark.parametrize("defect", ["absent", "fragment", "other-alert", "document"])
def test_alert_contract_requires_the_correct_runbook_url(
    alert_document: dict[str, Any], name: str, defect: str
) -> None:
    target = next(
        rule for rule in alert_document["groups"][0]["rules"] if rule["alert"] == name
    )
    annotations = target["annotations"]
    if defect == "absent":
        annotations.pop("runbook_url")
    elif defect == "fragment":
        annotations["runbook_url"] += "-missing"
    elif defect == "other-alert":
        other = next(
            rule
            for rule in alert_document["groups"][0]["rules"]
            if rule["alert"] != name
        )
        annotations["runbook_url"] = other["annotations"]["runbook_url"]
    else:
        annotations["runbook_url"] = f"docs/missing.md#{name.lower()}"
    assert _alert_errors(alert_document) == [
        f"alert {name} has no runbook anchor in the operations manual"
    ]


@pytest.mark.parametrize("name", [verdicts.FAILING_ALERT, verdicts.UNDELIVERED_ALERT])
@pytest.mark.parametrize(
    "template",
    [
        "{name}",
        "See `{name}`.",
        "### {name}Extra",
        "```markdown\n### {name}\n```",
        "~~~markdown\n### {name}\n~~~",
        "````markdown\n```\n### {name}\n````",
        "<!--\n### {name}\n-->",
    ],
)
def test_alert_contract_requires_real_headings_not_runbook_mentions(
    alert_document: dict[str, Any], name: str, template: str
) -> None:
    runbook = RUNBOOK_TEXT.replace(f"### {name}\n", template.format(name=name) + "\n")
    assert _alert_errors(alert_document, runbook) == [
        f"alert {name} has no runbook anchor in the operations manual"
    ]


def test_alert_headings_after_closed_code_and_comment_blocks_are_valid(
    alert_document: dict[str, Any],
) -> None:
    runbook = (
        "````markdown\n```\n### Example\n````\n"
        "~~~text\n```not-a-closing-fence\n~~~\n<!-- ignored -->\n" + RUNBOOK_TEXT
    )
    assert _alert_errors(alert_document, runbook) == []


def test_terminal_event_rule_does_not_replace_the_production_census_proof() -> None:
    assert verdicts.RESULT_METRIC == "gpu_fault_notification_total"
    before = _metrics(2, 1).replace(f"{TERMINAL_METRIC} 0", f"{TERMINAL_METRIC} 100")
    after = _metrics(3, 1).replace(f"{TERMINAL_METRIC} 0", f"{TERMINAL_METRIC} 100")
    assert "status=FAILED" in _text(
        verdicts.production_untouched_errors([before], [after])
    )
    timestamp_only = before.replace(
        'gpu_fault_notification_total{status="FAILED"} 2', ""
    )
    assert "missing or changed gpu_fault_notification_total" in _text(
        verdicts.production_untouched_errors([timestamp_only], [timestamp_only])
    )


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
    assert verdicts.PREDECESSOR_CASE_ID == "GF-REGIONAL-NOTIFY-005"
    assert (
        ROOT / "scripts/e2e/regional/run_notify007_delivery_states.py"
    ).stat().st_mode & 0o777 == 0o775

"""Pure verdict functions and constants of GF-REGIONAL-NOTIFY-007.

The case proves ARCH-E1 on a real regional deployment: a retryable delivery
failure is RETRY on the outbox row and nothing else -- no FAILED result row,
so ``GpuFaultNotificationDeliveryFailing`` (terminal FAILED semantics) stays
quiet -- and the row is SENT on the next cycle; a delivery that exhausts its
attempts is DEAD with a terminal FAILED result and one count on
``dead_lettered_total``. The drill runs in an isolated in-memory store inside
a control-worker Pod; the deployment's ``/metrics`` is read before and after
to prove the E1 families are exported and that no production FAILED result
appeared.

Every function here judges the drill's JSON, ``/metrics`` text and the alert
rule file; none touches a cluster.
"""

from __future__ import annotations

from typing import Any

import yaml  # type: ignore[import-untyped,unused-ignore]

from scripts.e2e.regional.collector_window_fixture import (
    metric_max,
    parse_metric_samples,
)

CASE_ID = "GF-REGIONAL-NOTIFY-007"
CONFIRMATION = "NOTIFY007_EXECUTE"
PREDECESSOR_CASE_ID = "GF-REGIONAL-NOTIFY-006"
DELIVERY_METRIC = "gpu_fault_notification_delivery_total"
OLDEST_PENDING_METRIC = "gpu_fault_notification_oldest_pending_age_seconds"
DEAD_LETTERED_METRIC = "gpu_fault_notification_dead_lettered_total"
DISPATCH_CYCLE_METRIC = "gpu_fault_notification_dispatch_last_cycle_timestamp_seconds"
RESULT_METRIC = "gpu_fault_notification_total"
FAILING_ALERT = "GpuFaultNotificationDeliveryFailing"
UNDELIVERED_ALERT = "GpuFaultNotificationUndeliveredTooLong"
REQUIRED_FAMILIES = (
    DELIVERY_METRIC,
    OLDEST_PENDING_METRIC,
    DEAD_LETTERED_METRIC,
    DISPATCH_CYCLE_METRIC,
)


def retry_phase_errors(retry: dict[str, Any]) -> list[str]:
    """First cycle: RETRY and no result; second cycle: SENT with a provider id."""

    errors: list[str] = []
    first = retry.get("after_first_cycle") or {}
    stats = (first.get("stats") or {}).get("by_status") or {}
    if int(stats.get("RETRY") or 0) != 1:
        errors.append(
            f"after the transient failure by_status is {stats}, expected RETRY=1"
        )
    if int(stats.get("DEAD") or 0) or int(stats.get("SENT") or 0):
        errors.append(f"a transient failure was counted as DEAD or SENT: {stats}")
    if first.get("result") is not None:
        errors.append(
            f"a result row was written for a retryable failure: {first.get('result')}"
        )
    if int(first.get("dead_lettered_total") or 0):
        errors.append("dead_lettered_total counted a retryable failure")
    if int((first.get("report") or {}).get("failed") or 0) != 1:
        errors.append("the first dispatch cycle did not report the failed attempt")
    second = retry.get("after_second_cycle") or {}
    stats = (second.get("stats") or {}).get("by_status") or {}
    if int(stats.get("SENT") or 0) != 1 or int(stats.get("RETRY") or 0):
        errors.append(f"after the retry by_status is {stats}, expected SENT=1 RETRY=0")
    result = second.get("result") or {}
    if result.get("status") != "SENT" or not result.get("provider_message_id_present"):
        errors.append(
            f"the retried delivery did not end SENT with a provider id: {result}"
        )
    if int(second.get("dead_lettered_total") or 0):
        errors.append("dead_lettered_total moved on a delivery that was sent")
    if int(retry.get("notifier_attempts") or 0) != 2:
        errors.append(
            f"the provider was called {retry.get('notifier_attempts')} times, expected 2"
        )
    if float(retry.get("last_cycle_timestamp_seconds") or 0) <= 0:
        errors.append("the dispatcher never stamped last_cycle_timestamp_seconds")
    return errors


def dead_phase_errors(dead: dict[str, Any]) -> list[str]:
    """One exhausted attempt: DEAD row, terminal FAILED result, one dead letter."""

    errors: list[str] = []
    stats = (dead.get("stats") or {}).get("by_status") or {}
    if int(stats.get("DEAD") or 0) != 1:
        errors.append(
            f"after exhausting attempts by_status is {stats}, expected DEAD=1"
        )
    result = dead.get("result") or {}
    if result.get("status") != "FAILED":
        errors.append(f"the dead delivery has no terminal FAILED result: {result}")
    if int(dead.get("dead_lettered_total") or 0) != 1:
        errors.append(
            f"dead_lettered_total is {dead.get('dead_lettered_total')}, expected 1"
        )
    if int(dead.get("notifier_attempts") or 0) != 1:
        errors.append(
            f"the provider was called {dead.get('notifier_attempts')} times, expected 1"
        )
    return errors


def exported_family_errors(texts: list[str]) -> list[str]:
    """The E1 families are on the deployment's /metrics, with every status label."""

    errors: list[str] = []
    for family in REQUIRED_FAMILIES:
        if not any(parse_metric_samples(text, family) for text in texts):
            errors.append(f"{family} is not exported by any control-plane replica")
    statuses = {
        sample["labels"].get("status")
        for text in texts
        for sample in parse_metric_samples(text, DELIVERY_METRIC)
    }
    for status in ("PENDING", "LEASED", "RETRY", "SENT", "DEAD"):
        if statuses and status not in statuses:
            errors.append(f"{DELIVERY_METRIC} has no status={status!r} series")
    return errors


def production_untouched_errors(before: list[str], after: list[str]) -> list[str]:
    """The drill wrote no production row: FAILED results and DEAD rows did not move."""

    errors: list[str] = []
    for family, where in (
        (RESULT_METRIC, {"status": "FAILED"}),
        (DELIVERY_METRIC, {"status": "DEAD"}),
    ):
        earlier = metric_max(before, family, where=where)
        later = metric_max(after, family, where=where)
        if earlier is not None and later is not None and later > earlier:
            errors.append(
                f"{family}{{status={where['status']}}} rose from {earlier} to {later} "
                "during an isolated drill"
            )
    return errors


def alert_rule_errors(rules_text: str, runbook_text: str) -> list[str]:
    """The failing alert reads terminal FAILED results and its runbook anchor exists."""

    errors: list[str] = []
    document = yaml.safe_load(rules_text) or {}
    rules = {
        str(rule.get("alert")): rule
        for group in document.get("groups") or []
        for rule in group.get("rules") or []
        if isinstance(rule, dict) and rule.get("alert")
    }
    for name in (FAILING_ALERT, UNDELIVERED_ALERT):
        rule = rules.get(name)
        if rule is None:
            errors.append(f"alert {name} is not defined")
            continue
        anchor = str((rule.get("annotations") or {}).get("runbook_url") or "")
        slug = anchor.rsplit("#", 1)[-1]
        if not slug or slug not in runbook_text.lower().replace(" ", ""):
            errors.append(
                f"alert {name} has no runbook anchor in the operations manual"
            )
    failing = rules.get(FAILING_ALERT) or {}
    expr = str(failing.get("expr") or "")
    if 'status="FAILED"' not in expr or RESULT_METRIC not in expr:
        errors.append(f"{FAILING_ALERT} does not read {RESULT_METRIC}{{status=FAILED}}")
    if "RETRY" in expr:
        errors.append(f"{FAILING_ALERT} still reads the RETRY state")
    return errors


def case_verdict(stages: dict[str, list[str]]) -> str:
    return "PASS" if not any(stages.values()) else "FAIL"

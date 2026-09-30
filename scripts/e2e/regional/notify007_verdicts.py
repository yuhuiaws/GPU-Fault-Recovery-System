"""Pure verdict functions and constants of GF-REGIONAL-NOTIFY-007.

The case proves ARCH-E1 on a real regional deployment: a retryable delivery
failure is RETRY on the outbox row and nothing else -- no FAILED result row,
so ``GpuFaultNotificationDeliveryFailing`` (terminal failure event time) stays
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

import math
import re
from collections.abc import Mapping, Sequence
from typing import Any

import yaml  # type: ignore[import-untyped,unused-ignore]

from scripts.e2e.regional.collector_window_fixture import (
    parse_metric_samples,
)

CASE_ID = "GF-REGIONAL-NOTIFY-007"
CONFIRMATION = "NOTIFY007_EXECUTE"
PREDECESSOR_CASE_ID = "GF-REGIONAL-NOTIFY-005"
DELIVERY_METRIC = "gpu_fault_notification_delivery_total"
OLDEST_PENDING_METRIC = "gpu_fault_notification_oldest_pending_age_seconds"
DEAD_LETTERED_METRIC = "gpu_fault_notification_dead_lettered_total"
DISPATCH_CYCLE_METRIC = "gpu_fault_notification_dispatch_last_cycle_timestamp_seconds"
RESULT_METRIC = "gpu_fault_notification_total"
TERMINAL_FAILURE_METRIC = (
    "gpu_fault_notification_terminal_failure_last_seen_timestamp_seconds"
)
FAILING_ALERT = "GpuFaultNotificationDeliveryFailing"
UNDELIVERED_ALERT = "GpuFaultNotificationUndeliveredTooLong"
RUNBOOK_DOCUMENT = "docs/en/administrator-operations.md"
REQUIRED_FAMILIES = (
    DEAD_LETTERED_METRIC,
    DISPATCH_CYCLE_METRIC,
    TERMINAL_FAILURE_METRIC,
)
WORKER_FAMILIES = (DELIVERY_METRIC, OLDEST_PENDING_METRIC, RESULT_METRIC)
CPU_ROLES = frozenset({"worker", "ingress", "spool-worker"})


def only_delivery_state(stats: dict[str, Any], expected: str) -> bool:
    return (
        stats.get(expected) == 1
        and all(type(value) is int and value >= 0 for value in stats.values())
        and sum(stats.values()) == 1
    )


def retry_phase_errors(retry: dict[str, Any]) -> list[str]:
    """First cycle: RETRY and no result; second cycle: SENT with a provider id."""

    errors: list[str] = []
    first = retry.get("after_first_cycle") or {}
    stats = (first.get("stats") or {}).get("by_status") or {}
    if not only_delivery_state(stats, "RETRY"):
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
    if not only_delivery_state(stats, "SENT"):
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
    if not only_delivery_state(stats, "DEAD"):
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


def _replicas(texts: Mapping[str, str] | Sequence[str]) -> dict[str, str]:
    if isinstance(texts, Mapping):
        return dict(texts)
    return {str(index): text for index, text in enumerate(texts)}


def exported_family_errors(
    texts: Mapping[str, str] | Sequence[str],
    *,
    roles: Mapping[str, str] | None = None,
) -> list[str]:
    """Require local metrics on every replica and fleet censuses on workers."""
    errors: list[str] = []
    replicas = _replicas(texts)
    if not replicas:
        return ["no control-plane metrics were observed"]
    if roles is not None and (
        roles.keys() != replicas.keys() or not set(roles.values()) <= CPU_ROLES
    ):
        return ["control-plane metric role inventory is missing or invalid"]
    for pod, text in replicas.items():
        worker = roles is None or roles[pod] == "worker"
        families = (*REQUIRED_FAMILIES, *(WORKER_FAMILIES if worker else ()))
        for family in families:
            samples = parse_metric_samples(text, family)
            if not samples:
                errors.append(f"{family} is not exported by replica {pod}")
            elif any(
                not math.isfinite(float(item["value"])) or item["value"] < 0
                for item in samples
            ):
                errors.append(f"{pod}: {family} has an invalid value")
        if not worker:
            continue
        statuses = {
            sample["labels"].get("status")
            for sample in parse_metric_samples(text, DELIVERY_METRIC)
        }
        for status in ("PENDING", "LEASED", "RETRY", "SENT", "DEAD"):
            if statuses and status not in statuses:
                errors.append(
                    f"{pod}: {DELIVERY_METRIC} has no status={status!r} series"
                )
    return errors


def production_untouched_errors(
    before: Mapping[str, str] | Sequence[str],
    after: Mapping[str, str] | Sequence[str],
    *,
    roles: Mapping[str, str] | None = None,
) -> list[str]:
    """Missing observations or replica changes cannot prove no production writes."""
    earlier_replicas, later_replicas = _replicas(before), _replicas(after)
    errors: list[str] = []
    if not earlier_replicas or earlier_replicas.keys() != later_replicas.keys():
        errors.append("control-plane replica identities changed or were not observed")
    if roles is not None and (
        roles.keys() != earlier_replicas.keys() or not set(roles.values()) <= CPU_ROLES
    ):
        return [*errors, "control-plane metric role inventory is missing or invalid"]
    for pod in earlier_replicas.keys() & later_replicas.keys():
        families: list[tuple[str, str | None]] = [(TERMINAL_FAILURE_METRIC, None)]
        if roles is None or roles[pod] == "worker":
            families.extend(((RESULT_METRIC, "FAILED"), (DELIVERY_METRIC, "DEAD")))
        for family, status in families:
            readings = []
            for text in (earlier_replicas[pod], later_replicas[pod]):
                samples = [
                    item
                    for item in parse_metric_samples(text, family)
                    if status is None or item["labels"].get("status") == status
                ]
                readings.append(
                    {
                        tuple(sorted(item["labels"].items())): float(item["value"])
                        for item in samples
                    }
                )
            first, second = readings
            if not first or first.keys() != second.keys():
                errors.append(
                    f"{pod}: missing or changed {family}{{status={status}}} series"
                )
                continue
            for labels, value in first.items():
                current = second[labels]
                if (
                    not math.isfinite(value)
                    or not math.isfinite(current)
                    or min(value, current) < 0
                ):
                    errors.append(f"{pod}: invalid {family}{{status={status}}}")
                elif current > value:
                    errors.append(
                        f"{pod}: {family}{{status={status}}} rose from {value} to {current}"
                    )
    return errors


def _alert_rules(rules_text: str) -> list[dict[str, Any]]:
    document = yaml.safe_load(rules_text)
    if document is None:
        document = {}
    if not isinstance(document, dict) or not isinstance(
        document.get("groups", []), list
    ):
        raise ValueError("alert rule document must contain a groups list")
    rules: list[dict[str, Any]] = []
    for group in document.get("groups", []):
        if not isinstance(group, dict) or not isinstance(group.get("rules"), list):
            raise ValueError("alert rule group must contain a rules list")
        for rule in group["rules"]:
            if not isinstance(rule, dict):
                raise ValueError("alert rule must be a mapping")
            rules.append(rule)
    return rules


def _runbook_headings(text: str) -> set[str]:
    """Only canonical alert headings outside comments and code blocks are anchors."""
    headings: set[str] = set()
    fence: str | None = None
    for line in re.sub(r"<!--.*?(?:-->|$)", "", text, flags=re.DOTALL).splitlines():
        marker = re.match(r"^ {0,3}(`{3,}|~{3,})", line)
        if marker is not None:
            if fence is None:
                fence = marker[1]
            elif marker[1].startswith(fence) and not line[marker.end() :].strip():
                fence = None
            continue
        heading = re.fullmatch(r"### (Gpu[A-Za-z0-9]+)[ \t]*", line)
        if fence is None and heading is not None:
            headings.add(heading[1])
    return headings


def alert_rule_errors(rules_text: str, runbook_text: str) -> list[str]:
    """Require the reviewed terminal-event rule and both canonical runbook anchors."""
    try:
        entries = _alert_rules(rules_text)
    except yaml.YAMLError:
        return ["invalid alert rule YAML"]
    except ValueError as exc:
        return [str(exc)]

    errors: list[str] = []
    rules: dict[str, dict[str, Any]] = {}
    headings = _runbook_headings(runbook_text)
    for name in (FAILING_ALERT, UNDELIVERED_ALERT):
        matches = [rule for rule in entries if rule.get("alert") == name]
        if not matches:
            errors.append(f"alert {name} is not defined")
            continue
        if len(matches) != 1:
            errors.append(f"alert {name} is defined more than once")
            continue
        rule = rules[name] = matches[0]
        annotations = rule.get("annotations")
        if (
            not isinstance(annotations, dict)
            or annotations.get("runbook_url") != f"{RUNBOOK_DOCUMENT}#{name.lower()}"
            or name not in headings
        ):
            errors.append(
                f"alert {name} has no runbook anchor in the operations manual"
            )
    failing = rules.get(FAILING_ALERT)
    if failing is None:
        return errors
    latest = (
        r"max\s+by\s*\(\s*control_plane_cluster\s*,\s*region\s*\)\s*"
        r"\(\s*max_over_time\s*\(\s*"
        + re.escape(TERMINAL_FAILURE_METRIC)
        + r"\s*\[\s*15m\s*\]\s*\)\s*\)"
    )
    expr = failing.get("expr")
    # Match the whole reviewed expression; metric mentions cannot prove its logic.
    if (
        not isinstance(expr, str)
        or re.fullmatch(
            rf"\s*\(\s*time\s*\(\s*\)\s*-\s*{latest}\s*\)\s*<\s*900"
            rf"\s+and\s+{latest}\s*>\s*0\s*",
            expr,
        )
        is None
    ):
        errors.append(
            f"{FAILING_ALERT} does not match the terminal failure event-time "
            f"contract: max_over_time({TERMINAL_FAILURE_METRIC}[15m]), "
            "max by (control_plane_cluster, region), age < 900 and timestamp > 0"
        )
    labels = failing.get("labels")
    if not isinstance(labels, dict) or labels.get("severity") != "critical":
        errors.append(f"{FAILING_ALERT} severity must be critical")
    if failing.get("for") != "5m":
        errors.append(f"{FAILING_ALERT} hold duration must be 5m")
    return errors


def case_verdict(stages: dict[str, list[str]]) -> str:
    return "PASS" if not any(stages.values()) else "FAIL"

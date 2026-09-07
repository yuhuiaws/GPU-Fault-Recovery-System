"""Pure verdict functions and constants of GF-REGIONAL-PREEMPT-037.

The case proves ARCH-E3 on a real regional deployment: the workflow dispatcher
stamps ``gpu_fault_workflow_dispatch_last_cycle_timestamp_seconds`` at the top
of every cycle, so a control plane whose dispatcher is not running can no
longer look healthy through frozen progress gauges. With the dispatcher
switched off on every control-worker replica for one bounded window, the
stamp ages (or reads 0 on replicas that never ticked), the
``GpuFaultWorkflowDispatcherStalled`` expression becomes true and stays true
for its ``for`` duration, the periodic runner's own stamp stays fresh (one
loop stalled, not the process), and closing the window makes the stamp fresh
again within a poll interval.

The alert is judged by evaluating its own expression against the replicas'
``/metrics`` text on the runner's clock, with the rule's threshold and ``for``
read from the shipped rule file. Every function here judges documents the
runner wrote and touches no cluster.
"""

from __future__ import annotations

from typing import Any

import yaml  # type: ignore[import-untyped,unused-ignore]

from scripts.e2e.regional.collector_window_fixture import metric_max

CASE_ID = "GF-REGIONAL-PREEMPT-037"
CONFIRMATION = "PREEMPT037_EXECUTE"
PREDECESSOR_CASE_ID = "GF-REGIONAL-PREEMPT-036"
DEPLOYMENT = "gpu-fault-control-worker"
CONTAINER = "control-worker"
VARIABLE = "GPU_FAULT_ENABLE_WORKFLOW_DISPATCHER"
WINDOW_VALUE = "false"
DISPATCH_METRIC = "gpu_fault_workflow_dispatch_last_cycle_timestamp_seconds"
PERIODIC_METRIC = "gpu_fault_periodic_last_cycle_timestamp_seconds"
STALLED_ALERT = "GpuFaultWorkflowDispatcherStalled"
PERIODIC_ALERT = "GpuFaultPeriodicRunnerStalled"
DEFAULT_THRESHOLD_SECONDS = 300
DEFAULT_FOR_SECONDS = 300
POLL_SECONDS = 30
# Nothing may be waiting on the dispatcher while it is off.
QUIESCENT_WORKFLOW_STATUSES = frozenset(
    {"SUCCEEDED", "FAILED", "BLOCKED", "SUPERSEDED"}
)


def _duration_seconds(value: str) -> int:
    text = str(value).strip()
    units = {"s": 1, "m": 60, "h": 3600}
    if text and text[-1] in units:
        return int(float(text[:-1]) * units[text[-1]])
    return int(float(text))


def alert_rule(rules_text: str, name: str) -> dict[str, Any] | None:
    document = yaml.safe_load(rules_text) or {}
    for group in document.get("groups") or []:
        for rule in group.get("rules") or []:
            if isinstance(rule, dict) and rule.get("alert") == name:
                return rule
    return None


def stall_rule_parameters(rules_text: str) -> dict[str, Any]:
    """Threshold, ``for`` and runbook anchor of the stall alert, from the rule file."""

    rule = alert_rule(rules_text, STALLED_ALERT)
    if rule is None:
        raise ValueError(f"{STALLED_ALERT} is not defined in the rule file")
    expr = str(rule.get("expr") or "")
    threshold = DEFAULT_THRESHOLD_SECONDS
    tail = expr.rsplit(">", 1)
    if len(tail) == 2 and tail[1].strip().isdigit():
        threshold = int(tail[1].strip())
    return {
        "expr": expr,
        "metric": DISPATCH_METRIC,
        "threshold_seconds": threshold,
        "for_seconds": _duration_seconds(
            str(rule.get("for") or f"{DEFAULT_FOR_SECONDS}s")
        ),
        "runbook_url": str((rule.get("annotations") or {}).get("runbook_url") or ""),
        "severity": str((rule.get("labels") or {}).get("severity") or ""),
    }


def rule_errors(parameters: dict[str, Any], runbook_text: str) -> list[str]:
    errors: list[str] = []
    if DISPATCH_METRIC not in parameters.get("expr", ""):
        errors.append(f"{STALLED_ALERT} does not read {DISPATCH_METRIC}")
    slug = str(parameters.get("runbook_url") or "").rsplit("#", 1)[-1]
    if not slug or slug not in runbook_text.lower().replace(" ", ""):
        errors.append(f"{STALLED_ALERT} has no runbook anchor in the operations manual")
    if parameters.get("severity") != "critical":
        errors.append(f"{STALLED_ALERT} severity is {parameters.get('severity')!r}")
    return errors


def stalled(texts: list[str], *, now: float, threshold_seconds: int) -> bool:
    """The alert expression: ``time() - max(stamp) > threshold``."""

    latest = metric_max(texts, DISPATCH_METRIC)
    if latest is None:
        return True
    return now - latest > threshold_seconds


def periodic_alive(texts: list[str], *, now: float, threshold_seconds: int) -> bool:
    latest = metric_max(texts, PERIODIC_METRIC)
    return latest is not None and now - latest <= threshold_seconds


def quiescence_errors(workflows: list[dict[str, Any]]) -> list[str]:
    """No workflow may be PENDING/RUNNING/WAITING while the dispatcher is off."""

    active = [
        f"{item.get('request_id')}={item.get('status')}"
        for item in workflows
        if str(item.get("status")) not in QUIESCENT_WORKFLOW_STATUSES
    ]
    if active:
        return [f"workflows are still in flight: {active}"]
    return []


def window_errors(record: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if record.get("variable") != VARIABLE or record.get("value") != WINDOW_VALUE:
        errors.append(f"the window did not set {VARIABLE}={WINDOW_VALUE}")
    if not record.get("replicas"):
        errors.append("no ready replica reported the window value")
    for replica in record.get("replicas") or []:
        if (replica.get("values") or {}).get(VARIABLE) != WINDOW_VALUE:
            errors.append(
                f"replica {replica.get('pod')} does not read the window value"
            )
    return errors


def stall_timeline_errors(
    timeline: list[dict[str, Any]], *, for_seconds: int
) -> list[str]:
    """The expression was true continuously for at least the rule's ``for``."""

    if not timeline:
        return ["no stall samples were recorded"]
    true_samples = [item for item in timeline if item.get("stalled")]
    if not true_samples:
        return ["the stall expression never became true while the dispatcher was off"]
    if any(
        not item.get("stalled") for item in timeline[timeline.index(true_samples[0]) :]
    ):
        return ["the stall expression flickered false after becoming true"]
    held = float(timeline[-1]["observed_epoch"]) - float(
        true_samples[0]["observed_epoch"]
    )
    if held < for_seconds:
        return [
            f"the stall expression held for {held:.0f}s, below the rule's for={for_seconds}s"
        ]
    if any(not item.get("periodic_alive") for item in timeline):
        return ["the periodic runner's stamp aged too: the whole process stalled"]
    return []


def recovery_errors(
    samples: list[dict[str, Any]], *, threshold_seconds: int
) -> list[str]:
    fresh = [item for item in samples if not item.get("stalled")]
    if not fresh:
        return [
            "the dispatch stamp did not become fresh within "
            f"{threshold_seconds}s of closing the window"
        ]
    return []


def restore_errors(record: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if record.get("restored_state") != record.get("baseline"):
        errors.append(
            "the control-worker env was not restored to the recorded baseline"
        )
    for replica in record.get("replicas_after_close") or []:
        expected = (record.get("baseline") or {}).get("value")
        if (replica.get("values") or {}).get(VARIABLE) != expected:
            errors.append(
                f"replica {replica.get('pod')} does not read the baseline value"
            )
    return errors


def case_verdict(stages: dict[str, list[str]]) -> str:
    return "PASS" if not any(stages.values()) else "FAIL"

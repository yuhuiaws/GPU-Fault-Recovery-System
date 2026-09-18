"""Read the source completion plan and its bound replacement-attempt evidence."""

from __future__ import annotations

import time
from datetime import datetime
from typing import Any

from gpu_fault.admin.deadlines import deadline_scope
from scripts.e2e.regional.collector_action_guard import (
    finite_seconds,
    require_action_time,
)
from scripts.e2e.regional.regional_commands import RegionalFixtureError
from scripts.e2e.regional.regional_live_fixture import RegionalLiveFixture

PASSIVE_PROBE = r"""
import json
import sys
from datetime import datetime, timezone
from gpu_fault.app import ApplicationContext
from gpu_fault.store import NotFoundError

cluster_id, job_id, source_attempt, new_attempt = sys.argv[1:]
store = ApplicationContext.from_environment().store

def read(function, *args):
    try:
        return function(*args)
    except NotFoundError:
        return None

event = read(store.get_event_by_attempt, cluster_id, source_attempt)
if event is not None and (
    event.cluster_id != cluster_id or event.job_id != job_id
    or event.attempt_id != source_attempt
):
    raise ValueError("completion source event identity differs")
decision = read(store.get_decision_by_attempt, cluster_id, source_attempt)
if decision is not None and (
    decision.cluster_id != cluster_id or decision.attempt_id != source_attempt
):
    raise ValueError("completion decision identity differs")
plan = (
    read(store.get_plan, decision.recovery_plan_id)
    if decision is not None and decision.recovery_plan_id else None
)
workflow = (
    read(store.get_workflow, plan.workflow_request_id)
    if plan is not None and plan.workflow_request_id else None
)
incident = read(store.get_incident, plan.incident_id) if plan is not None else None
if incident is not None and (
    incident.cluster_id != cluster_id or incident.job_id != job_id
    or incident.attempt_id != source_attempt
):
    raise ValueError("completion incident identity differs")
predecessor = (
    read(store.get_workflow, workflow.predecessor_workflow_id)
    if workflow is not None and workflow.predecessor_workflow_id else None
)
predecessor_incident = (
    read(store.get_incident, predecessor.incident_id)
    if predecessor is not None else None
)
if predecessor_incident is not None and (
    predecessor_incident.cluster_id != cluster_id
    or predecessor_incident.job_id != job_id
):
    raise ValueError("completion predecessor identity differs")
values = {
    "completion_event": event, "completion_decision": decision,
    "recovery_plan": plan, "recovery_workflow": workflow,
    "recovery_incident": incident, "predecessor_workflow": predecessor,
    "predecessor_incident": predecessor_incident,
    "restart_budget": read(store.get_restart_budget, cluster_id, job_id),
}
result = {
    name: value.model_dump(mode="json") if value is not None else None
    for name, value in values.items()
}
result["commands"] = [
    item.model_dump(mode="json", exclude={"lease_token"})
    for item in (
        store.list_remote_commands(workflow_request_ids=[workflow.request_id])
        if workflow is not None else []
    )
]
if event is not None:
    result["completion_event"]["event_key"] = event.event_key
result["observations"] = [
    observation.model_dump(mode="json")
    for observation in store.list_attempt_observations(cluster_id)
    if observation.job_id == job_id and observation.attempt_id == new_attempt
]
result["captured_at"] = datetime.now(timezone.utc).isoformat()
print(json.dumps(result, sort_keys=True))
"""


def completion_chain_errors(
    state: dict[str, Any], *, cluster_id: str, job_id: str, attempt_id: str
) -> list[str]:
    event = state.get("completion_event") or {}
    decision = state.get("completion_decision") or {}
    plan = state.get("recovery_plan") or {}
    workflow = state.get("recovery_workflow") or {}
    incident = state.get("recovery_incident") or {}
    errors = []
    if not cluster_id or not job_id or not attempt_id:
        errors.append("passive recovery requires an explicit cluster/job/attempt")
    if (
        event.get("cluster_id") != cluster_id
        or event.get("job_id") != job_id
        or event.get("attempt_id") != attempt_id
        or event.get("terminal_status") not in {"FAILED", "TIMED_OUT"}
        or not event.get("allocation")
        or not event.get("event_key")
        or decision.get("event_key") != event.get("event_key")
        or decision.get("cluster_id") != cluster_id
        or decision.get("attempt_id") != attempt_id
        or decision.get("status") != "PLAN_CREATED"
    ):
        errors.append("passive recovery has no bound source terminal decision")
    if (
        not plan.get("plan_id")
        or decision.get("recovery_plan_id") != plan.get("plan_id")
        or plan.get("attempt_id") != attempt_id
        or plan.get("trigger") != "no-hardware-evidence:RESTART"
        or [item.get("action") for item in plan.get("steps") or []]
        != ["RESTART_WORKLOAD"]
        or plan.get("status") != "SUCCEEDED"
        or not workflow.get("request_id")
        or plan.get("workflow_request_id") != workflow.get("request_id")
        or workflow.get("source_plan_id") != plan.get("plan_id")
        or workflow.get("status") != "SUCCEEDED"
        or [item.get("operation") for item in workflow.get("official_steps") or []]
        != ["RESTART_WORKLOAD"]
        or len(
            [
                item
                for item in workflow.get("step_executions") or []
                if item.get("operation") == "RESTART_WORKLOAD"
                and item.get("status") == "SUCCEEDED"
                and type(item.get("step_index")) is int
                and item.get("step_index") == 0
                and item.get("phase") in (None, "official")
            ]
        )
        != 1
        or not incident.get("incident_id")
        or workflow.get("incident_id") != incident.get("incident_id")
        or plan.get("incident_id") != incident.get("incident_id")
        or incident.get("cluster_id") != cluster_id
        or incident.get("job_id") != job_id
        or incident.get("attempt_id") != attempt_id
    ):
        errors.append("passive restart lacks its completed current recovery plan")
    required_incident = plan.get("restart_after_incident_id")
    predecessor_id = workflow.get("predecessor_workflow_id")
    if required_incident or predecessor_id:
        predecessor = state.get("predecessor_workflow") or {}
        predecessor_incident = state.get("predecessor_incident") or {}
        if (
            not predecessor_id
            or predecessor.get("request_id") != predecessor_id
            or predecessor.get("status") != "SUCCEEDED"
            or not predecessor_incident.get("incident_id")
            or predecessor.get("incident_id") != predecessor_incident.get("incident_id")
            or predecessor_incident.get("cluster_id") != cluster_id
            or predecessor_incident.get("job_id") != job_id
            or predecessor_incident.get("state") != "RECOVERED"
            or (
                required_incident is not None
                and required_incident != predecessor_incident.get("incident_id")
            )
        ):
            errors.append("passive restart lacks successful bound containment")
    return errors


def wait_passive_restart(
    regional: RegionalLiveFixture,
    *,
    job_id: str,
    attempt_id: str,
    timeout_seconds: float = 180,
) -> dict[str, Any]:
    """Read the completed source recovery before adopting any replacement Pods."""

    if any(
        not isinstance(value, str) or not value.strip()
        for value in (regional.settings.cluster_id, job_id, attempt_id)
    ):
        raise RegionalFixtureError(
            "passive restart requires an explicit cluster/job/attempt"
        )
    timeout = finite_seconds(timeout_seconds)
    end = time.monotonic() + timeout
    last_errors: list[str] = []
    with deadline_scope("COLLECT-021 passive restart proof", timeout) as deadline:
        while time.monotonic() < end:
            require_action_time()
            state = regional.cpu_python(
                PASSIVE_PROBE,
                regional.settings.cluster_id,
                job_id,
                attempt_id,
                "",
            )
            deadline.remaining()
            last_errors = completion_chain_errors(
                state,
                cluster_id=regional.settings.cluster_id,
                job_id=job_id,
                attempt_id=attempt_id,
            )
            require_action_time()
            if not last_errors and time.monotonic() < end:
                return state
            time.sleep(min(5, max(0, end - time.monotonic())))
    raise RegionalFixtureError(
        "passive restart evidence did not converge: " + "; ".join(last_errors)
    )


def _fresh_observation(observation: dict[str, Any], captured_at: Any) -> bool:
    try:
        observed = datetime.fromisoformat(
            str(observation.get("observed_at") or "").replace("Z", "+00:00")
        )
        captured = datetime.fromisoformat(str(captured_at or "").replace("Z", "+00:00"))
        if observed.tzinfo is None or captured.tzinfo is None:
            return False
        return -5 <= (captured - observed).total_seconds() <= 120
    except (ValueError, TypeError):
        return False


def wait_passive_completion(
    regional: RegionalLiveFixture,
    *,
    job_id: str,
    attempt_id: str,
    restarted: dict[str, Any],
    timeout_seconds: float = 180,
) -> dict[str, Any]:
    pods = restarted.get("pods") or []
    attempts = {item.get("attempt_id") for item in pods}
    uids = {item.get("uid") for item in pods}
    if (
        len(attempts) != 1
        or not all(attempts)
        or attempt_id in attempts
        or not pods
        or len(uids) != len(pods)
        or not all(uids)
    ):
        raise RegionalFixtureError("replacement Pods do not identify one new attempt")
    new_attempt = str(next(iter(attempts)))
    timeout = finite_seconds(timeout_seconds)
    end = time.monotonic() + timeout
    last_errors: list[str] = []
    with deadline_scope("COLLECT-021 passive completion proof", timeout) as deadline:
        while time.monotonic() < end:
            require_action_time()
            state = regional.cpu_python(
                PASSIVE_PROBE,
                regional.settings.cluster_id,
                job_id,
                attempt_id,
                new_attempt,
            )
            deadline.remaining()
            last_errors = completion_chain_errors(
                state,
                cluster_id=regional.settings.cluster_id,
                job_id=job_id,
                attempt_id=attempt_id,
            )
            observations = state.get("observations") or []
            matched = [
                item
                for item in observations
                if item.get("cluster_id") == regional.settings.cluster_id
                and item.get("job_id") == job_id
                and item.get("attempt_id") == new_attempt
                and item.get("workload_phase") == "RUNNING"
                and _fresh_observation(item, state.get("captured_at"))
                and {entry.get("pod_uid") for entry in item.get("containers") or []}
                == uids
            ]
            if len(matched) != 1:
                last_errors.append("replacement attempt Observation has not converged")
            restart_count = (state.get("restart_budget") or {}).get("restart_count")
            if type(restart_count) is not int or restart_count != 1:
                last_errors.append("passive restart budget has not converged")
            require_action_time()
            if not last_errors and time.monotonic() < end:
                return state
            time.sleep(min(5, max(0, end - time.monotonic())))
    raise RegionalFixtureError(
        "passive completion evidence did not converge: " + "; ".join(last_errors)
    )

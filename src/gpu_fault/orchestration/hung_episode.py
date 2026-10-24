"""Rearm an evidence round only from persisted, same-attempt traffic evidence."""

from datetime import datetime
from typing import Any

from gpu_fault.models import WorkflowOperation as Op


def diagnostic_only_hung(workflow: Any) -> bool:
    operations = {step.operation for step in workflow.official_steps}
    return Op.COLLECT_HUNG_TRIAGE in operations and operations <= {
        Op.FREEZE_EVIDENCE,
        Op.COLLECT_HUNG_SUMMARY,
        Op.COLLECT_HUNG_TRIAGE,
        Op.COLLECT_DIAGNOSTIC_BUNDLE,
        Op.VALIDATE_FABRIC,
        Op.ESCALATE_SUPPORT,
    }


def new_hung_episode(store: Any, finding: Any, workflow: Any) -> bool:
    if not diagnostic_only_hung(workflow):
        return False
    reported = finding.diagnostic_parameters.get("efa_stall_episode")
    if not isinstance(reported, dict):
        return False
    state = store.get_efa_traffic_state(
        store.efa_traffic_state_key(finding.cluster_id, finding.node_id)
    )
    if (
        state is None
        or state.job_id != finding.job_id
        or state.attempt_id != finding.attempt_id
        or state.stall_episode is None
        or not state.stall_episode.hung_reported
        or state.stall_episode.episode_id != reported.get("episode_id")
    ):
        return False
    prior = [
        step.parameters.get("health_recovery_contract", {}).get("detected_at")
        for step in workflow.official_steps
    ]
    try:
        detected = max(datetime.fromisoformat(value) for value in prior if value)
    except (ValueError, TypeError):
        detected = workflow.created_at
    witnesses = [state]
    reporters = {
        step.parameters.get("reporting_node_id")
        for step in workflow.official_steps
        if step.parameters.get("reporting_node_id")
    }
    for node in sorted(reporters)[:1]:
        if node != finding.node_id:
            witnesses.append(
                store.get_efa_traffic_state(
                    store.efa_traffic_state_key(finding.cluster_id, node)
                )
            )
    return any(
        row is not None
        and row.job_id == finding.job_id
        and row.attempt_id == finding.attempt_id
        and row.last_stable_traffic_at is not None
        and detected
        < row.last_stable_traffic_at
        < state.stall_episode.started_at
        <= finding.observed_at
        for row in witnesses
    )

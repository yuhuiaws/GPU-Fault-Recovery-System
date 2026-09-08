from __future__ import annotations

import hashlib
from typing import Any

from gpu_fault.dcgm_diagnostic_analysis import dcgm_failures_are_configuration_only


ANNOTATION_INCIDENT = "gpu-fault.io/incident-id"

ANNOTATION_FENCING = "gpu-fault.io/fencing-token"

ANNOTATION_WORKFLOW = "gpu-fault.io/workflow-id"

ANNOTATION_OPERATION = "gpu-fault.io/operation-id"

ANNOTATION_EXECUTION_EPOCH = "gpu-fault.io/execution-epoch"

ANNOTATION_STEP_INDEX = "gpu-fault.io/workflow-step-index"

ANNOTATION_PREVIOUS_UNSCHEDULABLE = "gpu-fault.io/previous-unschedulable"

ANNOTATION_TERMINATION_INCIDENT = "gpu-fault.io/termination-initiator-incident-id"

ANNOTATION_TRAINING_CONTAINER = "gpu-fault.io/training-container"

ANNOTATION_RESTART_BUDGET = "gpu-fault.io/restart-budget"

ANNOTATION_RESTART_COUNT = "gpu-fault.io/restart-count"

ANNOTATION_TARGET_GPU_COUNT = "gpu-fault.io/target-gpu-count"

ANNOTATION_APPROVE_GPU_COUNT_CHANGE = "gpu-fault.io/approve-gpu-count-change"

ANNOTATION_MECHANICAL_INSPECTION_COMPLETE = (
    "gpu-fault.io/mechanical-inspection-complete"
)

ANNOTATION_EFA_PLUGIN_RESTART_OPERATION = "gpu-fault.io/efa-plugin-restart-operation"

ANNOTATION_EFA_PLUGIN_RESTART_POD_UID = "gpu-fault.io/efa-plugin-restart-pod-uid"

ANNOTATION_EFA_PLUGIN_RESTART_STARTED_AT = "gpu-fault.io/efa-plugin-restart-started-at"

ANNOTATION_GPU_PLUGIN_RESTART_OPERATION = "gpu-fault.io/gpu-plugin-restart-operation"

ANNOTATION_GPU_PLUGIN_RESTART_POD_UID = "gpu-fault.io/gpu-plugin-restart-pod-uid"

ANNOTATION_GPU_PLUGIN_RESTART_STARTED_AT = "gpu-fault.io/gpu-plugin-restart-started-at"

LABEL_ATTEMPT_ID = "gpu-fault.io/attempt-id"

LABEL_JOB_ID = "gpu-fault.io/job-id"

QUARANTINE_TAINT = "gpu-fault.io/quarantined"


class NodeIsolationRejected(ValueError):
    """A safety refusal, not an adapter defect."""


class NodeActionPending(RuntimeError):
    def __init__(self, command_id: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(f"node action {command_id} is pending")
        self.command_id = command_id
        self.details = details or {}


NODE_ACTION_ACCEPTED_NODES_KEY = "node_action_accepted_nodes"


def node_action_accepted_nodes(details: Any) -> set[str]:
    """The step's nodes whose mutation provably began, from its own details.

    Written by the node-action layer only where the mutation cannot be called
    back: the send whose acceptance the agent confirmed (PENDING in its ledger)
    and the nodes this step already finished. The regional executor's
    destructive fleet preflight reads it, so this is the writer's and the
    reader's single definition of the key.

    Anything else is "not accepted": a record from before the key existed, a
    step-level marker with no node list, or a malformed value. The fence then
    runs, which is the direction that cannot let a fleet rollout and a node
    mutation overlap.
    """

    if not isinstance(details, dict):
        return set()
    raw = details.get(NODE_ACTION_ACCEPTED_NODES_KEY)
    if not isinstance(raw, list):
        return set()
    return {item for item in raw if isinstance(item, str) and item}


def quarantine_taint_value(incident_id: str) -> str:
    digest = hashlib.sha256(incident_id.encode("utf-8")).hexdigest()[:24]
    return f"incident-{digest}"


def dcgm_result_is_configuration_only(value: Any) -> bool:
    """True when a node's DCGM failures are configuration severity.

    Trusts the node agent's own grading when it reports one and
    otherwise re-derives it from the findings, so a node agent that
    reports severities but predates the outcome grading still cannot
    drain a healthy node. An agent old enough to report no severity at
    all stays fail-closed: it drains, exactly as it does today.
    """
    if not isinstance(value, dict):
        return False
    graded = value.get("configuration_only_failures")
    if isinstance(graded, bool):
        return graded
    if value.get("parse_error"):
        return False
    findings = value.get("diagnostic_findings")
    if not isinstance(findings, list):
        return False
    return dcgm_failures_are_configuration_only(
        [item for item in findings if isinstance(item, dict)]
    )

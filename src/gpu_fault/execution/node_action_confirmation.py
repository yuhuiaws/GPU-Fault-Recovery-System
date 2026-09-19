"""Operator confirmation of a node action the executor could not observe.

A ``RESTART_NODE`` that reaches its waiting cap without a receipt fails with
``outcome_unknown`` and ``manual_confirmation_required``; ``terminal_state``
then parks the workflow ``BLOCKED / NEEDS_OPERATOR`` so nothing automatic
treats the node as free. That is the intended fail-closed shape -- and until
this module it was a dead end: ``has_unresolved_node_action`` fenced the
validated restore, the incident close and the reconcile alike, while no verb
let the operator record what the fleet registry and Kubernetes already
showed. GF-REGIONAL-DESTR-014 (unknown-reboot) produced exactly that: the node
rebooted while its agent was disabled, came back Ready on a new boot with the
agent active, and one parked record kept both of the incident's nodes out of
fault handling indefinitely.

``confirm_node_action`` is the verdict both sides of ``gpu-fault-admin
submit-remediation --disposition confirm-node-action`` compute: the admin CLI
for ``--plan`` and the refusal it prints, the CPU ingress Pod again, on a fresh
read, before it writes. It is pure over plain values so the two cannot drift:

* the workflow and incident records;
* ``KubernetesNodeEvidence`` -- what ``kubectl get node`` shows (read by the
  admin CLI through the site's GPU kubeconfig; the Pod has none);
* ``AgentEvidence`` -- the fleet registry's ``AgentRecord`` for the node;
* the workflow's remote commands, reduced by ``remote_command_evidence`` to
  the keys the verdict reads.

For ``RESTART_NODE`` the outcome is proven by a boot transition: the node is
Ready, the Node Agent is ACTIVE with a live lease and was seen after the
action started, kubelet and the agent report the *same* boot id, and that boot
id differs from the one recorded before the reboot (``agent_baselines`` on
the step or its remote command -- the same baseline the HyperPod lifecycle
adapter's automatic confirmation reads -- the STOP ownership authorization, or
a single-node incident's ``source_boot_id``). For every other node-mutating
operation the only proof the control plane holds is the Node Agent's own
terminal answer relayed on the remote command (``node_results``); without one
the ledger on the node is the sole record and the verb refuses. Refusals name
the exact fact that is missing, so the operator knows what to establish, not
what to type.

The confirmation is recorded verbatim on the step (``operator_confirmed``),
the step's uncertainty flags are answered, the step itself stays FAILED, and
an ``OPERATOR_RECONCILED`` / ``NODE_ACTION_CONFIRMED`` event lands in the same
write. Ending the parked record is still the reconcile's or the incident
close's job; this module only answers the question they were waiting on.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from gpu_fault.execution.node_action_uncertainty import (
    OPERATOR_CONFIRMED_KEY,
    latest_node_action_executions,
    operator_confirmation,
    unresolved_node_action,
)
from gpu_fault.fleet import AgentLifecycleState, AgentRecord
from gpu_fault.models import (
    FaultIncident,
    WorkflowEventCode,
    WorkflowEventKind,
    WorkflowOperation,
    WorkflowRequest,
    WorkflowStatus,
    WorkflowStepExecution,
    WorkflowStepSpec,
    WorkflowStepStatus,
    bounded_event_details,
    record_workflow_event,
)
from gpu_fault.remote_command_models import BATCHED_RESULTS_KEY

CONFIRMATION_VERSION = 1
# The uncertainty flags a confirmation answers; their prior values are kept
# under ``superseded_flags`` so the record still shows what the executor saw.
UNCERTAINTY_FLAGS = (
    "outcome_unknown",
    "manual_confirmation_required",
    "node_action_interrupted",
    "node_action_response_unknown",
    "ownership_permit_delivery_unknown",
)
OPEN_REMOTE_STATUSES = frozenset({"PENDING", "LEASED", "WAITING"})
TERMINAL_NODE_RESULT_STATUSES = frozenset({"SUCCEEDED", "FAILED"})
BOOT_TRANSITION_OPERATIONS = frozenset({WorkflowOperation.RESTART_NODE})
AGENT_BASELINES_KEY = "agent_baselines"
STOP_REBOOT_AUTHORIZATION_KEY = "stop_reboot_authorization_v1"
NODE_RESULTS_KEY = "node_results"
# ``result_details`` keys the verdict reads; ``remote_command_evidence`` keeps
# exactly these so the admin payload carries evidence, not command bodies.
EVIDENCE_RESULT_KEYS = (
    AGENT_BASELINES_KEY,
    NODE_RESULTS_KEY,
    STOP_REBOOT_AUTHORIZATION_KEY,
)


def _text(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _timestamp(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else None


@dataclass(frozen=True)
class KubernetesNodeEvidence:
    """What ``kubectl get node <id> -o json`` shows about the node right now."""

    node_id: str
    exists: bool = True
    uid: str | None = None
    boot_id: str | None = None
    ready: bool = False
    ready_since: str | None = None
    unschedulable: bool = False

    @classmethod
    def from_node(
        cls, node_id: str, node: Mapping[str, Any] | None
    ) -> "KubernetesNodeEvidence":
        if node is None:
            return cls(node_id=node_id, exists=False)
        metadata = _mapping(node.get("metadata"))
        status = _mapping(node.get("status"))
        spec = _mapping(node.get("spec"))
        info = _mapping(status.get("nodeInfo"))
        ready = False
        ready_since: str | None = None
        for condition in status.get("conditions") or []:
            if isinstance(condition, Mapping) and condition.get("type") == "Ready":
                ready = condition.get("status") == "True"
                ready_since = _text(condition.get("lastTransitionTime"))
        return cls(
            node_id=node_id,
            exists=True,
            uid=_text(metadata.get("uid")),
            boot_id=_text(info.get("bootID")),
            ready=ready,
            ready_since=ready_since,
            unschedulable=bool(spec.get("unschedulable", False)),
        )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "KubernetesNodeEvidence":
        return cls(
            node_id=str(value["node_id"]),
            exists=bool(value.get("exists", True)),
            uid=_text(value.get("uid")),
            boot_id=_text(value.get("boot_id")),
            ready=bool(value.get("ready", False)),
            ready_since=_text(value.get("ready_since")),
            unschedulable=bool(value.get("unschedulable", False)),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "exists": self.exists,
            "uid": self.uid,
            "boot_id": self.boot_id,
            "ready": self.ready,
            "ready_since": self.ready_since,
            "unschedulable": self.unschedulable,
        }


@dataclass(frozen=True)
class AgentEvidence:
    """The fleet registry's ``AgentRecord`` for the node, or its absence."""

    node_id: str
    present: bool = False
    boot_id: str | None = None
    agent_incarnation_id: str | None = None
    lifecycle_state: str | None = None
    generation: int | None = None
    last_seen_at: datetime | None = None
    lease_expires_at: datetime | None = None
    retired_incarnation_ids: tuple[str, ...] = ()

    @classmethod
    def from_record(cls, node_id: str, record: AgentRecord | None) -> "AgentEvidence":
        if record is None:
            return cls(node_id=node_id)
        return cls(
            node_id=node_id,
            present=True,
            boot_id=record.boot_id,
            agent_incarnation_id=record.agent_incarnation_id,
            lifecycle_state=record.lifecycle_state.value,
            generation=record.generation,
            last_seen_at=record.last_seen_at,
            lease_expires_at=record.lease_expires_at,
            retired_incarnation_ids=tuple(record.retired_incarnation_ids),
        )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "AgentEvidence":
        if value is None:
            raise ValueError("agent evidence requires a node id")
        node_id = str(value["node_id"])
        if not value.get("present", True):
            return cls(node_id=node_id)
        generation = value.get("generation")
        return cls(
            node_id=node_id,
            present=True,
            boot_id=_text(value.get("boot_id")),
            agent_incarnation_id=_text(value.get("agent_incarnation_id")),
            lifecycle_state=_text(value.get("lifecycle_state")),
            generation=generation if isinstance(generation, int) else None,
            last_seen_at=_timestamp(value.get("last_seen_at")),
            lease_expires_at=_timestamp(value.get("lease_expires_at")),
            retired_incarnation_ids=tuple(
                str(item) for item in value.get("retired_incarnation_ids") or []
            ),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "present": self.present,
            "boot_id": self.boot_id,
            "agent_incarnation_id": self.agent_incarnation_id,
            "lifecycle_state": self.lifecycle_state,
            "generation": self.generation,
            "last_seen_at": (
                self.last_seen_at.isoformat() if self.last_seen_at else None
            ),
            "lease_expires_at": (
                self.lease_expires_at.isoformat() if self.lease_expires_at else None
            ),
            "retired_incarnation_ids": list(self.retired_incarnation_ids),
        }


@dataclass(frozen=True)
class NodeActionConfirmationVerdict:
    """What the evidence supports: confirmations to write, or the refusals."""

    node_id: str
    refusals: tuple[str, ...] = ()
    # One entry per unresolved node action on the node, paired with the
    # execution record it answers (same order).
    confirmations: tuple[dict[str, Any], ...] = ()
    executions: tuple[WorkflowStepExecution, ...] = ()
    # Confirmations already on the record for this node (an idempotent rerun).
    already_confirmed: tuple[dict[str, Any], ...] = ()
    warnings: tuple[str, ...] = field(default_factory=tuple)


def remote_command_evidence(command: Mapping[str, Any]) -> dict[str, Any]:
    """Reduce a dumped ``RemoteActionCommand`` to what the verdict reads.

    The command row carries the whole workflow and incident and the executor's
    full result payload; the admin payload only needs the ids, the status and
    the three evidence keys (per command and per batched entry).
    """

    raw_details = command.get("result_details")
    details = raw_details if isinstance(raw_details, Mapping) else {}
    kept: dict[str, Any] = {
        key: details[key] for key in EVIDENCE_RESULT_KEYS if key in details
    }
    batched = details.get(BATCHED_RESULTS_KEY)
    if isinstance(batched, Mapping):
        entries: dict[str, Any] = {}
        for index, entry in batched.items():
            if not isinstance(entry, Mapping):
                continue
            entry_details = entry.get("details")
            entries[str(index)] = {
                "status": entry.get("status"),
                "details": (
                    {
                        key: entry_details[key]
                        for key in EVIDENCE_RESULT_KEYS
                        if key in entry_details
                    }
                    if isinstance(entry_details, Mapping)
                    else {}
                ),
            }
        kept[BATCHED_RESULTS_KEY] = entries
    batched_steps = command.get("batched_steps")
    indexes = (
        [
            int(item["step_index"])
            for item in batched_steps
            if isinstance(item, Mapping) and isinstance(item.get("step_index"), int)
        ]
        if isinstance(batched_steps, list)
        else list(command.get("batched_step_indexes") or [])
    )
    return {
        "command_id": command.get("command_id"),
        "step_index": command.get("step_index"),
        "batched_step_indexes": indexes,
        "status": command.get("status"),
        "status_source": command.get("status_source"),
        "fencing_token": command.get("fencing_token"),
        "result_details": kept,
    }


def _status_text(value: Any) -> str:
    return str(getattr(value, "value", value))


def _covers(command: Mapping[str, Any], index: int) -> bool:
    if command.get("step_index") == index:
        return True
    return index in [int(item) for item in command.get("batched_step_indexes") or []]


def _step_for(
    workflow: WorkflowRequest, execution: WorkflowStepExecution
) -> WorkflowStepSpec | None:
    candidates: list[list[WorkflowStepSpec]] = []
    if execution.phase in (None, "official"):
        candidates.append(workflow.official_steps)
    if execution.phase in (None, "safety"):
        candidates.append(workflow.safety_steps)
    for steps in candidates:
        if (
            0 <= execution.step_index < len(steps)
            and steps[execution.step_index].operation is execution.operation
        ):
            return steps[execution.step_index]
    return None


def describe_node_action(
    execution: WorkflowStepExecution, step: WorkflowStepSpec | None
) -> str:
    nodes = ", ".join(step.node_ids) if step is not None else "unknown nodes"
    return f"step {execution.step_index} {execution.operation.value} on {nodes}"


def node_action_executions(
    workflow: WorkflowRequest,
) -> list[tuple[WorkflowStepExecution, WorkflowStepSpec | None]]:
    """The newest record of every node-mutating step, paired with its step spec."""

    return [
        (execution, _step_for(workflow, execution))
        for execution in latest_node_action_executions(workflow)
    ]


def unresolved_node_actions_on(
    workflow: WorkflowRequest, node_id: str
) -> tuple[list[str], list[str]]:
    """``(on this node, elsewhere)``: the unresolved node actions, described.

    What the admin CLI uses to pick the parked record a node belongs to, and
    what a refusal lists when the named node has nothing unresolved.
    """

    on_node: list[str] = []
    elsewhere: list[str] = []
    for execution, step in node_action_executions(workflow):
        if not unresolved_node_action(execution):
            continue
        target = on_node if step is not None and node_id in step.node_ids else elsewhere
        target.append(describe_node_action(execution, step))
    return on_node, elsewhere


def confirmed_node_actions_on(
    workflow: WorkflowRequest, node_id: str
) -> list[dict[str, Any]]:
    """The operator confirmations already recorded for ``node_id``."""

    return [
        confirmation
        for execution, step in node_action_executions(workflow)
        if step is not None
        and node_id in step.node_ids
        and (confirmation := operator_confirmation(execution.details)) is not None
    ]


def _record_refusals(
    workflow: WorkflowRequest,
    incident: FaultIncident,
    node_id: str,
    remote_commands: Sequence[Mapping[str, Any]],
    now: datetime,
) -> list[str]:
    reasons: list[str] = []
    if node_id not in incident.node_ids:
        reasons.append(
            f"node {node_id} is not named by incident {incident.incident_id} "
            f"(nodes: {', '.join(sorted(incident.node_ids)) or 'none'})"
        )
    if workflow.incident_id != incident.incident_id:
        reasons.append(
            f"workflow {workflow.request_id} belongs to incident "
            f"{workflow.incident_id}, not {incident.incident_id}"
        )
    if workflow.status is not WorkflowStatus.BLOCKED:
        reasons.append(
            f"workflow {workflow.request_id} is {workflow.status.value}, not BLOCKED; "
            "only a parked record takes an operator confirmation"
        )
    if workflow.fencing_token != incident.fencing_token:
        reasons.append(
            f"stale generation: workflow fencing token {workflow.fencing_token} "
            f"differs from incident {incident.fencing_token}"
        )
    if workflow.execution_owner_id is not None:
        reasons.append("workflow still has an execution owner")
    if (
        workflow.execution_lease_expires_at is not None
        and workflow.execution_lease_expires_at > now
    ):
        reasons.append("workflow execution lease has not expired")
    waiting = sorted(
        {
            f"step {item.step_index} {item.operation.value}"
            for item in workflow.step_executions
            if item.status is WorkflowStepStatus.WAITING
        }
    )
    if waiting:
        reasons.append(
            "workflow has a WAITING step the executor may still act on: "
            + ", ".join(waiting)
        )
    open_commands = sorted(
        f"{item.get('command_id')} ({_status_text(item.get('status'))})"
        for item in remote_commands
        if _status_text(item.get("status")) in OPEN_REMOTE_STATUSES
    )
    if open_commands:
        reasons.append(
            "workflow has an open remote command " + ", ".join(open_commands)
        )
    return reasons


def _baseline_from(
    container: Mapping[str, Any] | None, node_id: str
) -> tuple[str | None, str | None]:
    """``(boot_id, agent_incarnation_id)`` from an ``agent_baselines`` mapping."""

    if not isinstance(container, Mapping):
        return None, None
    baseline = container.get(node_id)
    if not isinstance(baseline, Mapping):
        return None, None
    return _text(baseline.get("boot_id")), _text(baseline.get("agent_incarnation_id"))


def _authorization_boot(
    container: Mapping[str, Any] | None, node_id: str
) -> str | None:
    if not isinstance(container, Mapping):
        return None
    nodes = container.get("nodes")
    if not isinstance(nodes, Mapping) or not isinstance(nodes.get(node_id), Mapping):
        return None
    return _text(nodes[node_id].get("boot_id"))


def _batched_details(
    command: Mapping[str, Any], index: int
) -> Mapping[str, Any] | None:
    details = command.get("result_details")
    if not isinstance(details, Mapping):
        return None
    batched = details.get(BATCHED_RESULTS_KEY)
    if not isinstance(batched, Mapping) or not isinstance(
        batched.get(str(index)), Mapping
    ):
        return None
    entry_details = batched[str(index)].get("details")
    return entry_details if isinstance(entry_details, Mapping) else None


def previous_boot_id(
    execution: WorkflowStepExecution,
    incident: FaultIncident,
    node_id: str,
    agent: AgentEvidence,
    remote_commands: Sequence[Mapping[str, Any]],
) -> tuple[str, str, str | None] | None:
    """``(boot_id, source, baseline incarnation)`` recorded before the action.

    Sources, in order of how directly they describe the node before *this*
    step: the step's own ``agent_baselines`` (the HyperPod lifecycle adapter
    snapshots the agent record when it submits the reboot), the same key on
    the remote command that carried the step (per command, then per batched
    entry -- the CPU-side record keeps pointers, the command keeps the
    payload), the STOP ownership authorization the kubernetes adapter captured
    for the reboot, and finally the incident's ``source_boot_id`` -- only for
    a single-node incident, or when the agent has retired that boot as an
    incarnation, the same rule the automatic confirmation applies.
    """

    details = execution.details
    boot, incarnation = _baseline_from(details.get(AGENT_BASELINES_KEY), node_id)
    if boot:
        return boot, f"step.{AGENT_BASELINES_KEY}", incarnation
    for command in remote_commands:
        if not _covers(command, execution.step_index):
            continue
        command_id = command.get("command_id")
        result = command.get("result_details")
        result = result if isinstance(result, Mapping) else {}
        boot, incarnation = _baseline_from(result.get(AGENT_BASELINES_KEY), node_id)
        if boot:
            return (
                boot,
                f"remote_command.{command_id}.{AGENT_BASELINES_KEY}",
                incarnation,
            )
        entry = _batched_details(command, execution.step_index)
        boot, incarnation = _baseline_from(
            entry.get(AGENT_BASELINES_KEY) if entry is not None else None, node_id
        )
        if boot:
            return (
                boot,
                f"remote_command.{command_id}.{BATCHED_RESULTS_KEY}."
                f"{execution.step_index}.{AGENT_BASELINES_KEY}",
                incarnation,
            )
    boot = _authorization_boot(details.get(STOP_REBOOT_AUTHORIZATION_KEY), node_id)
    if boot:
        return boot, f"step.{STOP_REBOOT_AUTHORIZATION_KEY}", None
    for command in remote_commands:
        if not _covers(command, execution.step_index):
            continue
        result = command.get("result_details")
        result = result if isinstance(result, Mapping) else {}
        entry = _batched_details(command, execution.step_index)
        for container, label in (
            (result, f"remote_command.{command.get('command_id')}"),
            (
                entry,
                f"remote_command.{command.get('command_id')}.{BATCHED_RESULTS_KEY}."
                f"{execution.step_index}",
            ),
        ):
            boot = _authorization_boot(
                container.get(STOP_REBOOT_AUTHORIZATION_KEY)
                if container is not None
                else None,
                node_id,
            )
            if boot:
                return boot, f"{label}.{STOP_REBOOT_AUTHORIZATION_KEY}", None
    source = incident.source_boot_id
    if source and (
        list(incident.node_ids) == [node_id] or source in agent.retired_incarnation_ids
    ):
        return source, "incident.source_boot_id", None
    return None


def _terminal_node_result(
    execution: WorkflowStepExecution,
    node_id: str,
    remote_commands: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any] | None, str | None]:
    """The Node Agent's terminal answer for ``node_id`` on this step, if relayed."""

    containers: list[tuple[Mapping[str, Any] | None, str]] = [
        (execution.details, "step")
    ]
    for command in remote_commands:
        if not _covers(command, execution.step_index):
            continue
        result = command.get("result_details")
        containers.append(
            (
                result if isinstance(result, Mapping) else None,
                f"remote_command.{command.get('command_id')}",
            )
        )
        containers.append(
            (
                _batched_details(command, execution.step_index),
                f"remote_command.{command.get('command_id')}.{BATCHED_RESULTS_KEY}."
                f"{execution.step_index}",
            )
        )
    for container, label in containers:
        if container is None:
            continue
        results = container.get(NODE_RESULTS_KEY)
        if not isinstance(results, Mapping) or not isinstance(
            results.get(node_id), Mapping
        ):
            continue
        result = dict(results[node_id])
        if _status_text(result.get("status")) in TERMINAL_NODE_RESULT_STATUSES:
            return result, f"{label}.{NODE_RESULTS_KEY}"
    return None, None


def _common_evidence_refusals(
    execution: WorkflowStepExecution,
    node_id: str,
    node: KubernetesNodeEvidence,
    agent: AgentEvidence,
    now: datetime,
) -> list[str]:
    reasons: list[str] = []
    operation = execution.operation.value
    if not node.exists:
        reasons.append(f"node {node_id} is not in the cluster")
    else:
        if not node.ready:
            reasons.append(f"node {node_id} is not Ready")
        if not node.uid:
            reasons.append(f"cannot judge node {node_id}: node identity is incomplete")
    if not agent.present:
        reasons.append(f"node {node_id} has no Node Agent record in the fleet registry")
        return reasons
    if agent.lifecycle_state != AgentLifecycleState.ACTIVE.value:
        reasons.append(
            f"node {node_id} Node Agent lifecycle state is {agent.lifecycle_state}, "
            "not ACTIVE"
        )
    if agent.lease_expires_at is None or agent.lease_expires_at <= now:
        reasons.append(
            f"node {node_id} Node Agent lease expired"
            + (
                f" at {agent.lease_expires_at.isoformat()}"
                if agent.lease_expires_at
                else ""
            )
            + "; a fresh heartbeat is required"
        )
    if agent.last_seen_at is None or agent.last_seen_at < execution.started_at:
        reasons.append(
            f"node {node_id} Node Agent has not been seen since {operation} started "
            f"at {execution.started_at.isoformat()}"
        )
    return reasons


def _boot_transition_refusals(
    execution: WorkflowStepExecution,
    incident: FaultIncident,
    node_id: str,
    node: KubernetesNodeEvidence,
    agent: AgentEvidence,
    remote_commands: Sequence[Mapping[str, Any]],
) -> tuple[list[str], tuple[str, str, str | None] | None]:
    reasons: list[str] = []
    operation = execution.operation.value
    if not agent.boot_id:
        reasons.append(f"node {node_id} Node Agent record carries no boot id")
    if node.exists and not node.boot_id:
        reasons.append(f"node {node_id} kubelet reports no boot id")
    if node.boot_id and agent.boot_id and node.boot_id != agent.boot_id:
        reasons.append(
            f"node {node_id} kubelet reports boot {node.boot_id} but the Node Agent "
            f"reports {agent.boot_id}; the two must agree on the running boot"
        )
    previous = previous_boot_id(execution, incident, node_id, agent, remote_commands)
    if previous is None:
        reasons.append(
            f"no pre-reboot boot id is recorded for node {node_id}: neither "
            f"{AGENT_BASELINES_KEY} on {operation} step {execution.step_index} or its "
            f"remote commands, nor {STOP_REBOOT_AUTHORIZATION_KEY}, nor a usable "
            "incident source_boot_id; the reboot cannot be proven from the record"
        )
        return reasons, None
    boot, _source, baseline_incarnation = previous
    for observed, label in ((node.boot_id, "kubelet"), (agent.boot_id, "Node Agent")):
        if observed and observed == boot:
            reasons.append(
                f"node {node_id} still runs boot {boot} recorded before {operation} "
                f"({label}); the reboot did not happen"
            )
    if (
        baseline_incarnation
        and agent.agent_incarnation_id
        and agent.agent_incarnation_id == baseline_incarnation
    ):
        reasons.append(
            f"node {node_id} reports the same Node Agent incarnation "
            f"{baseline_incarnation} as before {operation}"
        )
    return reasons, previous


def successor_validation(
    successor: WorkflowRequest | None,
    workflow: WorkflowRequest,
    incident: FaultIncident,
    node_id: str,
) -> dict[str, Any] | None:
    """The later validated restore that validated ``node_id`` and put it back.

    The strongest evidence the control plane holds about a node action whose
    own answer was lost: the product itself validated the GPU and the host
    (``VALIDATE_GPU`` complete) and restored scheduling, under the same
    incident and generation, and the incident ended RECOVERED pointing at that
    restore. HA-004's RESET_GPU was refused by the agent before touching the
    GPU, the executor read the refusal as outcome-unknown, and the successor
    then validated that very GPU (2026-09-18).
    """

    if successor is None or successor.request_id == workflow.request_id:
        return None
    completed = set(successor.completed_operations)
    validating_steps = [
        step
        for step in successor.official_steps
        if step.operation is WorkflowOperation.VALIDATE_GPU and node_id in step.node_ids
    ]
    if (
        successor.incident_id != incident.incident_id
        or successor.status is not WorkflowStatus.SUCCEEDED
        or successor.fencing_token != workflow.fencing_token
        or incident.fencing_token != workflow.fencing_token
        or incident.state.value != "RECOVERED"
        or incident.workflow_request_id != successor.request_id
        or WorkflowOperation.VALIDATE_GPU not in completed
        or WorkflowOperation.RESTORE_SCHEDULING not in completed
        or not validating_steps
    ):
        return None
    return {
        "request_id": successor.request_id,
        "status": successor.status.value,
        "completed_operations": [item.value for item in successor.completed_operations],
        "validated_gpu_uuids": sorted(
            {gpu for step in validating_steps for gpu in step.gpu_uuids}
        ),
        "updated_at": successor.updated_at.isoformat(),
    }


def _confirmation(
    execution: WorkflowStepExecution,
    step: WorkflowStepSpec | None,
    node_id: str,
    node: KubernetesNodeEvidence,
    agent: AgentEvidence,
    *,
    previous: tuple[str, str, str | None] | None,
    terminal: tuple[dict[str, Any] | None, str | None],
    validated_by: dict[str, Any] | None = None,
    actor: str,
    reference: str,
    now: datetime,
) -> dict[str, Any]:
    return {
        "version": CONFIRMATION_VERSION,
        "actor": actor,
        "reference": reference,
        "confirmed_at": now.isoformat(),
        "node_id": node_id,
        "operation": execution.operation.value,
        "step_index": execution.step_index,
        "phase": execution.phase,
        "step_node_ids": list(step.node_ids) if step is not None else [],
        "adapter_operation_id": execution.adapter_operation_id,
        "remote_command_id": _text(execution.details.get("remote_command_id")),
        "observed_boot_id": node.boot_id or agent.boot_id,
        "previous_boot_id": previous[0] if previous else None,
        "previous_boot_id_source": previous[1] if previous else None,
        "agent_generation": agent.generation,
        "agent_incarnation_id": agent.agent_incarnation_id,
        "agent_last_seen_at": (
            agent.last_seen_at.isoformat() if agent.last_seen_at else None
        ),
        "node_uid": node.uid,
        "node_ready_since": node.ready_since,
        "kubernetes_node": node.as_dict(),
        "agent": agent.as_dict(),
        "terminal_node_result": terminal[0],
        "terminal_node_result_source": terminal[1],
        "validated_by_workflow": validated_by,
        "superseded_flags": {
            key: execution.details[key]
            for key in UNCERTAINTY_FLAGS
            if key in execution.details
        },
    }


def confirm_node_action(
    *,
    workflow: WorkflowRequest,
    incident: FaultIncident,
    node_id: str,
    node: KubernetesNodeEvidence,
    agent: AgentEvidence,
    remote_commands: Sequence[Mapping[str, Any]],
    actor: str,
    reference: str,
    now: datetime,
    successor: WorkflowRequest | None = None,
) -> NodeActionConfirmationVerdict:
    """Judge the evidence for every unresolved node action on ``node_id``.

    Returns the confirmations to write, or the refusals -- never both. A node
    whose actions are all confirmed already yields ``already_confirmed`` and
    nothing to write. A node the record never left unresolved is a refusal
    that lists what *is* unresolved, so a typo in the node id cannot confirm
    the wrong thing. ``successor`` is the incident's current workflow when it
    is a later validated restore (``successor_validation``): for a node action
    other than a reboot it is accepted as evidence beside, or instead of, the
    agent's terminal answer on the remote command.
    """

    refusals = _record_refusals(workflow, incident, node_id, remote_commands, now)
    targets: list[tuple[WorkflowStepExecution, WorkflowStepSpec | None]] = []
    elsewhere: list[str] = []
    for execution, step in node_action_executions(workflow):
        if not unresolved_node_action(execution):
            continue
        if step is not None and node_id in step.node_ids:
            targets.append((execution, step))
        else:
            elsewhere.append(describe_node_action(execution, step))
    already = confirmed_node_actions_on(workflow, node_id)
    if not targets and not already:
        refusals.append(
            f"workflow {workflow.request_id} has no unresolved node action on node "
            f"{node_id}"
            + (
                "; unresolved: " + "; ".join(elsewhere)
                if elsewhere
                else "; nothing is unresolved"
            )
        )
    if refusals:
        return NodeActionConfirmationVerdict(node_id=node_id, refusals=tuple(refusals))
    confirmations: list[dict[str, Any]] = []
    executions: list[WorkflowStepExecution] = []
    for execution, step in targets:
        reasons = _common_evidence_refusals(execution, node_id, node, agent, now)
        previous: tuple[str, str, str | None] | None = None
        terminal: tuple[dict[str, Any] | None, str | None] = (None, None)
        validated_by: dict[str, Any] | None = None
        if execution.operation in BOOT_TRANSITION_OPERATIONS:
            more, previous = _boot_transition_refusals(
                execution, incident, node_id, node, agent, remote_commands
            )
            reasons.extend(more)
        else:
            terminal = _terminal_node_result(execution, node_id, remote_commands)
            validated_by = successor_validation(successor, workflow, incident, node_id)
            if terminal[0] is None and validated_by is None:
                reasons.append(
                    f"no terminal Node Agent result for {execution.operation.value} on "
                    f"node {node_id} is recorded on step {execution.step_index} or its "
                    "remote commands, and no later validated restore of this incident "
                    "validated the node; the agent ledger on the node is the only "
                    "record and this disposition cannot read it"
                )
        if reasons:
            refusals.extend(reasons)
            continue
        confirmations.append(
            _confirmation(
                execution,
                step,
                node_id,
                node,
                agent,
                previous=previous,
                terminal=terminal,
                validated_by=validated_by,
                actor=actor,
                reference=reference,
                now=now,
            )
        )
        executions.append(execution)
    if refusals:
        return NodeActionConfirmationVerdict(node_id=node_id, refusals=tuple(refusals))
    return NodeActionConfirmationVerdict(
        node_id=node_id,
        confirmations=tuple(confirmations),
        executions=tuple(executions),
        already_confirmed=tuple(already),
    )


def confirmed_execution(
    execution: WorkflowStepExecution, confirmation: Mapping[str, Any], *, now: datetime
) -> WorkflowStepExecution:
    """``execution`` with its uncertainty answered and the confirmation on it."""

    details = dict(execution.details)
    for key in UNCERTAINTY_FLAGS:
        if key in details:
            details[key] = False
    if details.get("node_action_state") == "PENDING":
        details["node_action_state"] = "OPERATOR_CONFIRMED"
    details[OPERATOR_CONFIRMED_KEY] = dict(confirmation)
    return execution.model_copy(update={"details": details, "updated_at": now})


def apply_node_action_confirmation(
    workflow: WorkflowRequest,
    verdict: NodeActionConfirmationVerdict,
    *,
    now: datetime,
) -> WorkflowRequest:
    """``workflow`` with every confirmation of ``verdict`` recorded and audited.

    Each confirmed execution replaces its record in place (the identity rule
    of ``has_unresolved_node_action``: the newest record answers), and one
    ``OPERATOR_RECONCILED`` / ``NODE_ACTION_CONFIRMED`` event per confirmation
    is appended so the write and its attribution land together; actor and
    reference are the ones the verdict bound. The status is untouched: the
    record stays BLOCKED until the reconcile or the incident close ends it.
    """

    if verdict.refusals:
        raise ValueError(
            "node action confirmation refused: " + "; ".join(verdict.refusals)
        )
    executions = list(workflow.step_executions)
    updated = workflow
    for execution, confirmation in zip(
        verdict.executions, verdict.confirmations, strict=True
    ):
        actor = str(confirmation["actor"])
        reference = str(confirmation["reference"])
        positions = [
            position
            for position, item in enumerate(executions)
            if item.phase == execution.phase
            and item.step_index == execution.step_index
            and item.operation is execution.operation
        ]
        if not positions:
            raise ValueError(
                f"step {execution.step_index} {execution.operation.value} is no longer "
                "on the workflow record"
            )
        executions[positions[-1]] = confirmed_execution(
            execution, confirmation, now=now
        )
        updated = record_workflow_event(
            updated,
            WorkflowEventKind.OPERATOR_RECONCILED,
            code=WorkflowEventCode.NODE_ACTION_CONFIRMED.value,
            actor=actor,
            step_index=execution.step_index,
            operation=execution.operation,
            phase=execution.phase,
            status=workflow.status.value,
            reason=(
                f"operator confirmed {execution.operation.value} outcome on "
                f"{verdict.node_id} ({reference})"
            ),
            details=bounded_event_details(
                {
                    key: value
                    for key, value in confirmation.items()
                    if key not in {"kubernetes_node", "agent", "superseded_flags"}
                }
            ),
            at=now,
        )
    return updated.model_copy(update={"step_executions": executions})

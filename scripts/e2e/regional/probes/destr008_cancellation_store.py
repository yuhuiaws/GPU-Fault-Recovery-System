"""Typed, bounded Store withdrawal and drain proof; no physical execution."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Protocol, TypeVar

from pydantic import BaseModel, ValidationError

from gpu_fault.models import (
    FaultIncident,
    PlanStatus,
    RecoveryPlan,
    WorkflowEvent,
    WorkflowEventCode,
    WorkflowEventKind,
    WorkflowRequest,
    WorkflowStatus,
    WorkflowStepSpec,
    WorkflowStepStatus,
)
from gpu_fault.orchestration.escalation import ESCALATION_NAMES, unknown_outcome_failure
from gpu_fault.regional import RemoteActionCommand
from gpu_fault.remote_command_models import RemoteCommandStatus
from gpu_fault.store.shared.errors import NotFoundError

if TYPE_CHECKING or __package__:
    from .destr008_cancellation_protocol import (
        MAX_OWNED_RECORDS,
        SOURCE,
        Control,
        Plan,
        ProbeError,
        Receipt,
        Root,
        digest,
        encode,
    )
else:
    from destr008_cancellation_protocol import (
        MAX_OWNED_RECORDS,
        SOURCE,
        Control,
        Plan,
        ProbeError,
        Receipt,
        Root,
        digest,
        encode,
    )

MAX_RECORD_BYTES = 262144
WITHDRAWAL_REASON = "DESTR008 producer revoked; withdraw this acceptance recovery"
TERMINAL_WORKFLOWS = {
    WorkflowStatus.FAILED,
    WorkflowStatus.SUPERSEDED,
    WorkflowStatus.SUCCEEDED,
    WorkflowStatus.BLOCKED,
}
TERMINAL_COMMANDS = {RemoteCommandStatus.SUCCEEDED, RemoteCommandStatus.FAILED}


class CancellationStore(Protocol):
    def get_incident_by_event(self, event_id: str) -> FaultIncident | None: ...

    def get_incident(self, incident_id: str) -> FaultIncident: ...

    def get_workflow(self, request_id: str) -> WorkflowRequest: ...

    def get_plan(self, plan_id: str) -> RecoveryPlan: ...

    def list_job_recovery_workflow_incidents(
        self,
        cluster_id: str,
        job_id: str,
        attempt_id: str,
        *,
        limit: int,
        include_terminal: bool,
    ) -> list[tuple[FaultIncident, WorkflowRequest]]: ...

    def list_remote_commands(
        self, *, workflow_request_ids: Iterable[str]
    ) -> list[RemoteActionCommand]: ...

    def has_incomplete_processor_requests_for_scopes(
        self, cluster_id: str, scope_keys: set[str]
    ) -> bool: ...

    def amend_workflow(
        self, request_id: str, updates: Mapping[str, object], *, event: WorkflowEvent
    ) -> WorkflowRequest: ...

    def cancel_remote_commands_for_workflow(
        self, workflow_request_id: str, *, reason: str
    ) -> dict[str, int]: ...

    def close(self) -> None: ...


Model = TypeVar("Model", bound=BaseModel)


def checked(model: type[Model], value: Model) -> Model:
    if not isinstance(value, model):
        raise ProbeError("STORE_SHAPE")
    try:
        if len(encode(value)) > MAX_RECORD_BYTES:
            raise ProbeError("RECORD_SIZE")
        return model.model_validate(
            value.model_dump(mode="python", warnings="error"), strict=True
        )
    except (ValidationError, ValueError, TypeError, RecursionError):
        raise ProbeError("STORE_SHAPE") from None


def bounded(items: list[Any], maximum: int = MAX_OWNED_RECORDS) -> None:
    if type(items) is not list or len(items) > maximum:
        raise ProbeError("INVENTORY_SIZE")


def bind_incident(plan: Plan, incident: FaultIncident) -> None:
    bind_created_at(plan, incident.created_at)
    if (
        incident.cluster_id != plan.cluster_id
        or incident.job_id != plan.job_id
        or incident.attempt_id != plan.attempt_id
        or set(incident.node_ids) != {plan.fault_node}
        or len(incident.node_ids) != 1
        or incident.policy_source != "SITE_SYNTHETIC_REPLACEMENT_TEST"
    ):
        raise ProbeError("INCIDENT_SCOPE")


def bind_created_at(plan: Plan, created_at: datetime) -> None:
    if created_at.tzinfo is None or created_at.timestamp() < plan.created_at:
        raise ProbeError("RECORD_SOURCE")


def bind_parameters(plan: Plan, value: Any, *, depth: int = 0) -> None:
    if depth > 8:
        raise ProbeError("PARAMETER_SIZE")
    if isinstance(value, dict):
        if len(value) > 256:
            raise ProbeError("PARAMETER_SIZE")
        identities = {
            "cluster_id": plan.cluster_id,
            "job_id": plan.job_id,
            "attempt_id": plan.attempt_id,
            "source_attempt_id": plan.attempt_id,
            "fault_node": plan.fault_node,
            "spare_node": plan.spare_node,
            "runtime_profile_version": plan.runtime_profile_version,
        }
        for key, item in value.items():
            if key in identities and item != identities[key]:
                raise ProbeError("PARAMETER_SCOPE")
            if key in {"node_id", "target_node_id"} and item not in {
                plan.fault_node,
                plan.spare_node,
            }:
                raise ProbeError("PARAMETER_SCOPE")
            if key in {"node_ids", "target_node_ids", "branch_node_ids"} and (
                not isinstance(item, list)
                or not all(
                    isinstance(node, str) and node in {plan.fault_node, plan.spare_node}
                    for node in item
                )
            ):
                raise ProbeError("PARAMETER_SCOPE")
            if key in {"workload_ids", "affected_workload_ids"} and (
                not isinstance(item, list)
                or not all(
                    isinstance(work, str) and work in plan.workload_ids for work in item
                )
            ):
                raise ProbeError("PARAMETER_SCOPE")
            bind_parameters(plan, item, depth=depth + 1)
    elif isinstance(value, list):
        if len(value) > 256:
            raise ProbeError("PARAMETER_SIZE")
        for item in value:
            bind_parameters(plan, item, depth=depth + 1)


def bind_step(plan: Plan, step: WorkflowStepSpec) -> None:
    if (
        not set(step.node_ids).issubset({plan.fault_node, plan.spare_node})
        or not set(step.branch_node_ids).issubset({plan.fault_node, plan.spare_node})
        or not set(step.workload_ids).issubset(plan.workload_ids)
    ):
        raise ProbeError("STEP_SCOPE")
    bind_parameters(plan, step.parameters)


def bind_workflow(
    plan: Plan, incident: FaultIncident, workflow: WorkflowRequest
) -> None:
    bind_incident(plan, incident)
    bind_created_at(plan, workflow.created_at)
    if (
        workflow.incident_id != incident.incident_id
        or workflow.runtime_profile_version != plan.runtime_profile_version
        or workflow.fencing_token > incident.fencing_token
    ):
        raise ProbeError("WORKFLOW_SCOPE")
    for step in [*workflow.official_steps, *workflow.safety_steps]:
        bind_step(plan, step)
    for execution in workflow.step_executions:
        bind_parameters(plan, execution.details)
        if execution.details.get("restart_attempt_id") is not None:
            raise ProbeError("UNEXPECTED_RESTART")
    if (
        workflow.workload_withdrawn_at is not None
        and not workflow.workload_withdrawn_reason
    ):
        raise ProbeError("WITHDRAWAL_SHAPE")


def bind_command(
    plan: Plan,
    command: RemoteActionCommand,
    pairs: dict[str, tuple[FaultIncident, WorkflowRequest]],
) -> None:
    bind_created_at(plan, command.created_at)
    pair = pairs.get(command.workflow_request_id)
    if pair is None:
        raise ProbeError("COMMAND_SCOPE")
    incident, workflow = pair
    if (
        command.cluster_id != plan.cluster_id
        or command.incident_id != incident.incident_id
        or command.incident.incident_id != incident.incident_id
        or command.incident.event_id != incident.event_id
        or command.workflow.request_id != workflow.request_id
        or command.fencing_token != command.workflow.fencing_token
        or command.fencing_token > workflow.fencing_token
    ):
        raise ProbeError("COMMAND_SCOPE")
    lease = (command.lease_owner, command.lease_token, command.lease_expires_at)
    if (
        command.status in {RemoteCommandStatus.PENDING, RemoteCommandStatus.WAITING}
        and any(value is not None for value in lease)
    ) or (
        command.status is RemoteCommandStatus.LEASED
        and any(value is None for value in lease)
    ):
        raise ProbeError("COMMAND_LEASE_SHAPE")
    bind_workflow(plan, command.incident, command.workflow)
    bind_step(plan, command.step)
    for step in command.batched_steps:
        bind_step(plan, step.step)
    authorization = command.restart_authorization
    if authorization is not None and (
        authorization.cluster_id != plan.cluster_id
        or authorization.job_id != plan.job_id
        or authorization.source_attempt_id != plan.attempt_id
    ):
        raise ProbeError("COMMAND_SCOPE")
    bind_parameters(plan, command.result_details)


def outcome_unknown(value: Any) -> bool:
    if isinstance(value, dict):
        return (
            unknown_outcome_failure(value)
            or value.get("stale_fence_swept") is True
            or value.get("post_cancellation_status") == "WAITING"
            or value.get("post_stale_fence_status") == "WAITING"
            or any(outcome_unknown(item) for item in value.values())
        )
    return isinstance(value, list) and any(outcome_unknown(item) for item in value)


def command_drained(command: RemoteActionCommand) -> bool:
    return (
        command.status in TERMINAL_COMMANDS
        and command.lease_owner is None
        and command.lease_token is None
        and command.lease_expires_at is None
        and not outcome_unknown(command.result_details)
    )


def workflow_drained(workflow: WorkflowRequest) -> bool:
    return (
        workflow.status in TERMINAL_WORKFLOWS
        and workflow.workload_withdrawn_at is not None
        and workflow.execution_owner_id is None
        and workflow.execution_lease_expires_at is None
        and not any(
            item.status is WorkflowStepStatus.WAITING or outcome_unknown(item.details)
            for item in workflow.step_executions
        )
    )


@dataclass(frozen=True)
class Inventory:
    root: Root | None
    pairs: dict[str, tuple[FaultIncident, WorkflowRequest]]
    commands: dict[str, RemoteActionCommand]
    source_complete: bool
    pending_creation: bool

    def proof(self) -> dict[str, Any]:
        root_unknown = self.root is None and not self.source_complete
        states = {
            "workflows": {
                key: {
                    "incident_id": incident.incident_id,
                    "event_id": incident.event_id,
                    "status": item.status.value,
                    "merge_revision": item.merge_revision,
                    "execution_epoch": item.execution_epoch,
                    "fencing_token": item.fencing_token,
                    "updated_at": item.updated_at.isoformat(),
                    "drained": workflow_drained(item),
                    "failure_handled": item.failure_handled_at is not None,
                }
                for key, (incident, item) in self.pairs.items()
            },
            "commands": {
                key: {
                    "status": item.status.value,
                    "updated_at": item.updated_at.isoformat(),
                    "drained": command_drained(item),
                    "cancellation_requested": item.cancellation_requested_at
                    is not None,
                }
                for key, item in self.commands.items()
            },
            "source_complete": self.source_complete,
            "pending_creation": self.pending_creation,
        }
        return {
            "root": self.root,
            "workflow_ids": sorted(self.pairs),
            "command_ids": sorted(self.commands),
            "commands_active": (
                None
                if root_unknown
                else sum(not command_drained(item) for item in self.commands.values())
            ),
            "workflows_active": (
                None
                if root_unknown
                else sum(not workflow_drained(item) for _, item in self.pairs.values())
            ),
            "source_complete": self.source_complete,
            "pending_creation": self.pending_creation or root_unknown,
            "inventory_sha256": digest(states),
        }


class RelatedHistory:
    """A bounded cache populated only by the scoped read and exact record IDs."""

    def __init__(
        self, store: CancellationStore, plan: Plan, check: Callable[[], None]
    ) -> None:
        self.store = store
        self.plan = plan
        self.check = check
        self.records: dict[str, tuple[FaultIncident, WorkflowRequest]] = {}

    def add(
        self, incident: FaultIncident, workflow: WorkflowRequest
    ) -> tuple[FaultIncident, WorkflowRequest]:
        incident = checked(FaultIncident, incident)
        workflow = checked(WorkflowRequest, workflow)
        bind_workflow(self.plan, incident, workflow)
        existing = self.records.get(workflow.request_id)
        if existing is not None and existing != (incident, workflow):
            raise ProbeError("INVENTORY_CHANGED")
        self.records[workflow.request_id] = incident, workflow
        if len(self.records) > MAX_OWNED_RECORDS:
            raise ProbeError("INVENTORY_SIZE")
        return incident, workflow

    def load(
        self, key: str, *, refresh: bool = False
    ) -> tuple[FaultIncident, WorkflowRequest]:
        self.check()
        cached = self.records.get(key)
        if cached is not None and not refresh:
            return cached
        workflow = checked(WorkflowRequest, self.store.get_workflow(key))
        if workflow.request_id != key:
            raise ProbeError("WORKFLOW_SCOPE")
        self.check()
        incident = checked(FaultIncident, self.store.get_incident(workflow.incident_id))
        return self.add(incident, workflow)

    def ancestry(self, key: str) -> list[str]:
        chain: list[str] = []
        current: str | None = key
        while current is not None:
            if current in chain:
                raise ProbeError("DESCENDANT_SOURCE")
            chain.append(current)
            _, workflow = self.load(current)
            current = workflow.predecessor_workflow_id
        return chain


def inventory(
    store: CancellationStore,
    plan: Plan,
    control: Control,
    previous: Receipt | None,
    *,
    checkpoint: Callable[[], None] | None = None,
) -> Inventory:
    check = checkpoint or (lambda: None)
    check()
    related = store.list_job_recovery_workflow_incidents(
        plan.cluster_id,
        plan.job_id,
        plan.attempt_id,
        limit=MAX_OWNED_RECORDS + 1,
        include_terminal=True,
    )
    bounded(related)
    history = RelatedHistory(store, plan, check)
    for incident, workflow in related:
        if workflow.request_id in history.records:
            raise ProbeError("DUPLICATE_WORKFLOW")
        history.add(incident, workflow)
    listed_ids = set(history.records)
    check()
    candidate = store.get_incident_by_event(plan.event_id)
    scopes = {
        encode([plan.cluster_id, plan.job_id, plan.attempt_id]),
        encode([plan.cluster_id, "node", plan.fault_node]),
        encode([plan.cluster_id, "node", plan.spare_node]),
    }
    check()
    pending = store.has_incomplete_processor_requests_for_scopes(
        plan.cluster_id, scopes
    )
    if type(pending) is not bool:
        raise ProbeError("STORE_SHAPE")
    if candidate is None:
        if related or (previous is not None and previous.root is not None):
            raise ProbeError("ROOT_MISSING")
        return Inventory(
            root=None,
            pairs={},
            commands={},
            source_complete=control.producer.state == "NOT_STARTED",
            pending_creation=pending,
        )
    root_incident = checked(FaultIncident, candidate)
    bind_incident(plan, root_incident)
    if root_incident.event_id != plan.event_id:
        raise ProbeError("EVENT_SOURCE")
    if control.producer.state == "NOT_STARTED":
        raise ProbeError("UNCLAIMED_EVENT")
    if root_incident.workflow_request_id is None:
        raise ProbeError("ROOT_MISSING")
    current_incident, _ = history.load(root_incident.workflow_request_id, refresh=True)
    if current_incident != root_incident:
        raise ProbeError("INVENTORY_CHANGED")
    ack = control.producer.ack
    if ack is not None:
        if ack.workflow_request_id not in listed_ids:
            raise ProbeError("INVENTORY_CHANGED")
        root_id = ack.workflow_request_id
    elif previous is not None and previous.root is not None:
        root_id = previous.root.workflow_request_id
    else:
        chain = history.ancestry(root_incident.workflow_request_id)
        roots = [
            item.request_id
            for _, item in history.records.values()
            if item.incident_id == root_incident.incident_id
            and item.predecessor_workflow_id is None
        ]
        if len(roots) != 1 or roots[0] != chain[-1]:
            raise ProbeError("ROOT_BINDING")
        root_id = roots[0]
    root = Root(incident_id=root_incident.incident_id, workflow_request_id=root_id)
    if (ack is not None and ack.incident_id != root.incident_id) or (
        previous is not None and previous.root is not None and previous.root != root
    ):
        raise ProbeError("ROOT_BINDING")
    pairs: dict[str, tuple[FaultIncident, WorkflowRequest]] = {}
    if previous is not None:
        for key in previous.workflow_ids:
            history.ancestry(key)
    queue = [root_id]
    while queue:
        key = queue.pop()
        if key in pairs:
            continue
        incident, workflow = history.load(key)
        if key == root_id and (
            incident.incident_id != root.incident_id
            or workflow.predecessor_workflow_id is not None
        ):
            raise ProbeError("ROOT_BINDING")
        pairs[key] = incident, workflow
        for pointer in (
            incident.workflow_request_id,
            workflow.preempted_by_workflow_id,
            workflow.preemption_pending_by_workflow_id,
        ):
            if pointer is not None:
                history.ancestry(pointer)
        for name in ESCALATION_NAMES.values():
            event_id = f"{name}-after-{key}"
            check()
            escalated = store.get_incident_by_event(event_id)
            if escalated is None:
                continue
            escalated = checked(FaultIncident, escalated)
            if (
                escalated.event_id != event_id
                or escalated.incident_id != f"inc-{event_id}"
            ):
                raise ProbeError("DESCENDANT_SOURCE")
            escalation_id = f"workflow-{event_id}"
            child_incident, _ = history.load(escalation_id)
            if child_incident != escalated:
                raise ProbeError("INVENTORY_CHANGED")
            if escalated.workflow_request_id is None:
                raise ProbeError("DESCENDANT_UNRESOLVED")
            history.ancestry(escalated.workflow_request_id)
            queue.append(escalation_id)
        queue.extend(
            item.request_id
            for _, item in history.records.values()
            if item.predecessor_workflow_id in pairs and item.request_id not in pairs
        )
    owned_incidents = {incident.incident_id for incident, _ in pairs.values()}
    for incident, _ in pairs.values():
        if incident.workflow_request_id not in pairs:
            raise ProbeError("DESCENDANT_UNRESOLVED")
    for key, (incident, _) in history.records.items():
        if key not in pairs:
            if incident.incident_id in owned_incidents:
                raise ProbeError("DESCENDANT_UNRESOLVED")
            raise ProbeError("UNRELATED_JOB_RECOVERY")
    for _, workflow in pairs.values():
        for successor in (
            workflow.preempted_by_workflow_id,
            workflow.preemption_pending_by_workflow_id,
        ):
            if successor is not None and successor not in pairs:
                raise ProbeError("DESCENDANT_UNRESOLVED")
        pending |= (
            workflow.status is WorkflowStatus.FAILED
            and workflow.failure_handled_at is None
        )
        if workflow.source_plan_id is not None:
            check()
            recovery = checked(RecoveryPlan, store.get_plan(workflow.source_plan_id))
            if (
                recovery.plan_id != workflow.source_plan_id
                or recovery.incident_id != workflow.incident_id
                or recovery.attempt_id != plan.attempt_id
                or recovery.workflow_request_id != workflow.request_id
                or recovery.runtime_profile_version != plan.runtime_profile_version
            ):
                raise ProbeError("RECOVERY_PLAN_SCOPE")
            pending |= recovery.status in {PlanStatus.PENDING, PlanStatus.RUNNING}
    check()
    commands_list = store.list_remote_commands(workflow_request_ids=sorted(pairs))
    bounded(commands_list)
    commands: dict[str, RemoteActionCommand] = {}
    for item in commands_list:
        command = checked(RemoteActionCommand, item)
        bind_command(plan, command, pairs)
        if command.command_id in commands:
            raise ProbeError("DUPLICATE_COMMAND")
        commands[command.command_id] = command
    if previous is not None and (
        not set(previous.workflow_ids).issubset(pairs)
        or not set(previous.command_ids).issubset(commands)
    ):
        raise ProbeError("RECORD_MISSING")
    return Inventory(root, pairs, commands, ack is not None, pending)


def withdraw(
    store: CancellationStore,
    plan: Plan,
    control: Control,
    snapshot: Inventory,
    *,
    now: int,
    checkpoint: Callable[[], None] | None = None,
) -> None:
    if control.revocation is None:
        raise ProbeError("REVOCATION_REQUIRED")
    check = checkpoint or (lambda: None)
    for key, (bound_incident, _) in snapshot.pairs.items():
        check()
        current = checked(WorkflowRequest, store.get_workflow(key))
        check()
        incident = checked(FaultIncident, store.get_incident(current.incident_id))
        bind_workflow(plan, incident, current)
        if (
            current.request_id != key
            or incident.incident_id != bound_incident.incident_id
            or incident.event_id != bound_incident.event_id
        ):
            raise ProbeError("WORKFLOW_SCOPE")
        if current.workload_withdrawn_at is None:
            stamp = datetime.fromtimestamp(now, timezone.utc)
            check()
            amended = store.amend_workflow(
                key,
                {
                    "workload_withdrawn_at": stamp,
                    "workload_withdrawn_reason": WITHDRAWAL_REASON,
                },
                event=WorkflowEvent(
                    kind=WorkflowEventKind.HOLD,
                    code=WorkflowEventCode.WORKLOAD_WITHDRAWN.value,
                    actor=SOURCE,
                    at=stamp,
                    reason=WITHDRAWAL_REASON,
                    details={"plan_sha256": digest(plan), "event_id": plan.event_id},
                ),
            )
            amended = checked(WorkflowRequest, amended)
            bind_workflow(plan, incident, amended)
            if amended.request_id != key or amended.workload_withdrawn_at is None:
                raise ProbeError("WITHDRAWAL_NOT_ACKNOWLEDGED")
        check()
        fresh = store.list_remote_commands(workflow_request_ids=[key])
        bounded(fresh)
        for item in fresh:
            bind_command(
                plan, checked(RemoteActionCommand, item), {key: (incident, current)}
            )
        check()
        result = store.cancel_remote_commands_for_workflow(
            key, reason=WITHDRAWAL_REASON
        )
        if (
            type(result) is not dict
            or set(result) != {"cancelled", "cancellation_requested"}
            or any(
                type(value) is not int or not 0 <= value <= MAX_OWNED_RECORDS
                for value in result.values()
            )
        ):
            raise ProbeError("CANCELLATION_SHAPE")


def read_inventory(
    store: CancellationStore,
    plan: Plan,
    control: Control,
    previous: Receipt | None,
    *,
    checkpoint: Callable[[], None] | None = None,
) -> Inventory:
    try:
        return inventory(store, plan, control, previous, checkpoint=checkpoint)
    except NotFoundError:
        raise ProbeError("RECORD_MISSING") from None

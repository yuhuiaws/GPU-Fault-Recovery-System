"""Departures whose records end in the less travelled ways.

``tests/departed_nodes/_support.py`` builds the live departure shapes. These
helpers add what the edge cases need on top of it: a HyperPod departure with
no records of its own (each test adds the incident it is about), DESTR-020's
alias whose departure rests on the administrator's acceptance, a parked
record, a closure service scripted per call, and store views standing in for
what another writer does between two reads of the cleanup.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any

from gpu_fault.departed_node_cleanup import (
    DepartedNodeCleanup,
    DepartedNodeCleanupConfig,
)
from gpu_fault.departed_nodes import DepartedNodeIdentityAcceptance
from gpu_fault.execution.node_action_uncertainty import OPERATOR_CONFIRMED_KEY
from gpu_fault.models import (
    BlockedKind,
    IncidentState,
    WorkflowOperation,
    WorkflowStatus,
    WorkflowStepStatus,
)
from gpu_fault.orchestration.incident_closure import IncidentNotClosable
from gpu_fault.store.shared.errors import StaleWriteError
from tests._builders import (
    fault_incident,
    workflow_request,
    workflow_step,
    workflow_step_execution,
)
from tests.departed_nodes._support import (
    EKS_CLUSTER,
    EKS_PRESENT,
    HP_CLUSTER,
    HP_NAME,
    HP_NODE,
    HP_PRESENT,
    World,
    agent,
    ago,
    registration,
    world,
)

ALIAS = "acceptance-alias-0a0b0c0d0e0f"
OP = WorkflowOperation


def hyperpod_departed(store: Any) -> World:
    """HyperPod reclaimed ``HP_NODE``: proven departed, no incident yet."""

    result = world(store)
    store.save_regional_cluster(registration(HP_CLUSTER))
    store.save_agent(
        agent(HP_CLUSTER, HP_NODE, instance_id="node-uid-1", last_seen=ago(minutes=20))
    )
    result.provider.hyperpod[HP_NAME] = {"i-0a000002"}
    result.reports_since(HP_CLUSTER, [HP_PRESENT], minutes=12)
    return result


def accepted_alias(store: Any) -> World:
    """DESTR-020's alias, a node no report ever listed and the provider cannot
    be asked about, whose departure the administrator accepted."""

    result = world(store)
    store.save_regional_cluster(registration(EKS_CLUSTER, provider="eks-managed"))
    result.reports_since(EKS_CLUSTER, [EKS_PRESENT], minutes=15)
    store.save_departed_node_identity_acceptance(
        DepartedNodeIdentityAcceptance(
            cluster_id=EKS_CLUSTER,
            node_id=ALIAS,
            change_reference="CHG-1",
            actor="ops@example",
            reason="DESTR-020 alias; never a node",
            accepted_at=ago(seconds=5),
            kubernetes={"open": True, "never_reported": True},
        )
    )
    return result


def cleanup(
    world_: World, *, store: Any = None, closure: Any = None, notifications: Any = None
) -> DepartedNodeCleanup:
    """The world's cleanup with some of its collaborators swapped."""

    return DepartedNodeCleanup(
        world_.store if store is None else store,
        registry=world_.registry,
        incident_closure=world_.closure if closure is None else closure,
        provider=world_.provider,
        notifications=notifications,
        config=DepartedNodeCleanupConfig(),
        waiting_ttl=timedelta(hours=2),
    )


def incident(
    store: Any,
    suffix: str,
    *,
    cluster_id: str = HP_CLUSTER,
    node_id: str = HP_NODE,
    state: IncidentState = IncidentState.ESCALATED,
    workflow_request_id: str | None = None,
) -> str:
    incident_id = f"incident-{suffix}"
    store.save_incident(
        fault_incident(
            incident_id,
            f"event-{suffix}",
            cluster_id=cluster_id,
            node_ids=[node_id],
            state=state,
            workflow_request_id=workflow_request_id,
            fencing_token=1,
        )
    )
    return incident_id


def parked(
    store: Any,
    suffix: str,
    *,
    cluster_id: str = HP_CLUSTER,
    node_id: str = HP_NODE,
    status: WorkflowStatus = WorkflowStatus.BLOCKED,
    confirmed: bool = False,
) -> tuple[str, str]:
    """An ESCALATED incident whose workflow stopped on a RESET_GPU nobody saw
    answered; ``confirmed`` records an operator's confirmation of it."""

    request_id = f"workflow-{suffix}"
    created = ago(hours=2)
    details: dict[str, Any] = {
        "outcome_unknown": True,
        "manual_confirmation_required": True,
        "node_action_response_unknown": True,
    }
    if confirmed:
        details = {
            **{key: False for key in details},
            OPERATOR_CONFIRMED_KEY: {
                "actor": "ops@example",
                "reference": "CHG-confirm",
                "confirmed_at": ago(minutes=5).isoformat(),
                "node_id": node_id,
                "operation": OP.RESET_GPU.value,
            },
        }
    steps = [
        workflow_step(OP.MARK_UNSCHEDULABLE, node_ids=[node_id]),
        workflow_step(OP.RESET_GPU, node_ids=[node_id]),
        workflow_step(OP.RESTORE_SCHEDULING, node_ids=[node_id]),
    ]
    store.save_workflow(
        workflow_request(
            request_id,
            f"incident-{suffix}",
            status=status,
            blocked_kind=(
                BlockedKind.NEEDS_OPERATOR if status is WorkflowStatus.BLOCKED else None
            ),
            fencing_token=1,
            execution_epoch=2,
            official_steps=steps,
            completed_step_indexes=[0],
            completed_operations=[OP.MARK_UNSCHEDULABLE],
            step_executions=[
                workflow_step_execution(
                    0, OP.MARK_UNSCHEDULABLE, phase="official", started_at=created
                ),
                workflow_step_execution(
                    1,
                    OP.RESET_GPU,
                    WorkflowStepStatus.FAILED,
                    phase="official",
                    details=details,
                    started_at=created,
                    updated_at=created + timedelta(minutes=30),
                ),
            ],
            created_at=created,
            updated_at=ago(minutes=30),
        )
    )
    incident_id = incident(
        store,
        suffix,
        cluster_id=cluster_id,
        node_id=node_id,
        workflow_request_id=request_id,
    )
    return request_id, incident_id


class ScriptedClosure:
    """``IncidentClosureService`` answering each close from a per-incident
    script: ``refuse:<text>``, ``stale``, ``unchanged`` or ``close``."""

    def __init__(self, store: Any, script: dict[str, list[str]]) -> None:
        self.store = store
        self.script = script
        self.calls: list[tuple[str, str, str]] = []

    def close_incident(
        self,
        incident_id: str,
        *,
        reason: str,
        operator: str,
        reference: str | None = None,
        evidence: Any = None,
    ) -> tuple[Any, bool]:
        self.calls.append((incident_id, operator, reason))
        answer = self.script[incident_id].pop(0)
        current = self.store.get_incident(incident_id)
        if answer.startswith("refuse:"):
            raise IncidentNotClosable(answer.removeprefix("refuse:"))
        if answer == "stale":
            raise StaleWriteError(f"incident/{incident_id} changed since it was read")
        return current, answer == "close"


class StoreView:
    """The cleanup's store; subclasses change what one read or write sees."""

    def __init__(self, store: Any) -> None:
        self.store = store

    def __getattr__(self, name: str) -> Any:
        return getattr(self.store, name)


class OperatorClosesMidCycle(StoreView):
    """An operator closes ``incident_id`` while the cleanup reads its workflows."""

    def __init__(self, store: Any, incident_id: str) -> None:
        super().__init__(store)
        self.incident_id = incident_id

    def list_active_workflow_incidents(self, *args: Any, **kwargs: Any) -> Any:
        current = self.store.get_incident(self.incident_id)
        if current.state is not IncidentState.RECOVERED:
            self.store.save_incident(
                current.model_copy(
                    update={
                        "state": IncidentState.RECOVERED,
                        "reasons": [*current.reasons, "closed by an operator"],
                    }
                ),
                expected=current,
            )
        return self.store.list_active_workflow_incidents(*args, **kwargs)


class ConcurrentWorkflowWrite(StoreView):
    """Another writer lands on ``request_id`` before the cleanup's write does."""

    def __init__(self, store: Any, request_id: str) -> None:
        super().__init__(store)
        self.request_id = request_id
        self.refused_writes = 0

    def amend_workflow(self, request_id: str, *args: Any, **kwargs: Any) -> Any:
        if request_id == self.request_id:
            self.refused_writes += 1
            raise StaleWriteError(f"workflow/{request_id} changed since it was read")
        return self.store.amend_workflow(request_id, *args, **kwargs)


class ReplannedBeforeApply(StoreView):
    """``request_id`` is re-planned (its fencing token moves) between the
    cleanup's reconcile plan and the apply's own rebuild of it: the second read
    of the record's remote commands is where the apply's rebuild begins."""

    def __init__(self, store: Any, request_id: str) -> None:
        super().__init__(store)
        self.request_id = request_id
        self.command_reads = 0

    def list_remote_commands(self, *args: Any, **kwargs: Any) -> Any:
        if self.request_id in (kwargs.get("workflow_request_ids") or ()):
            self.command_reads += 1
            if self.command_reads == 2:
                current = self.store.get_workflow(self.request_id)
                self.store.save_workflow(
                    current.model_copy(
                        update={"fencing_token": current.fencing_token + 1}
                    ),
                    expected=current,
                )
        return self.store.list_remote_commands(*args, **kwargs)

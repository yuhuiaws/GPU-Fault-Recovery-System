"""Single-process live Executor probe; node keys never leave this GPU Pod."""

from __future__ import annotations

import json
import os
import re
import select
import secrets
import sys
import time
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Literal, TextIO

from gpu_fault.adapters import KubernetesWorkflowAdapter, NodeActionWorkflowAdapter
from gpu_fault.adapters.kubernetes.stop_ownership import (
    STOP_RECEIPT_KEY,
    KubernetesStopOwnershipValidator,
    StopOwnershipReceipt,
    stop_ownership_scope,
)
from gpu_fault.adapters.node_action.lease_guard import active_lease_guard
from gpu_fault.cluster_executor.bootstrap import executor_from_environment
from gpu_fault.execution import WorkflowStepContext, WorkflowStepOutcome
from gpu_fault.models import (
    FaultIncident,
    WorkflowExecutionRequest,
    WorkflowOperation,
    WorkflowRequest,
    WorkflowStepExecution,
    WorkflowStepSpec,
    WorkflowStepStatus,
)
from gpu_fault.node_agent.late_ownership import current_ownership_challenge
from gpu_fault.node_agent.protocol import NodeActionExecutionState
from gpu_fault.orchestration.escalation import unknown_outcome_failure

from scripts.e2e.regional.late_ownership_barrier import BoundaryDenied, process_identity
from scripts.e2e.regional.late_ownership_contract import (
    AcceptanceScope,
    DecisionReceipt,
    MutationReceipt,
    Participant,
    QuiescenceReceipt,
    RecheckPermit,
    StopReceipt,
    WitnessStart,
    WorkloadIdentity,
)

MAX_MESSAGE_BYTES = 65536
CASE_OPERATIONS = frozenset(
    {
        WorkflowOperation.STOP_WORKLOADS,
        WorkflowOperation.MARK_UNSCHEDULABLE,
        WorkflowOperation.QUIESCE_GPU_SERVICES,
        WorkflowOperation.VERIFY_NO_GPU_CLIENTS,
        WorkflowOperation.RESET_GPU,
        WorkflowOperation.RESTORE_GPU_SERVICES,
        WorkflowOperation.RESTORE_SCHEDULING,
    }
)


def readonly_calibration_completed(outcome: WorkflowStepOutcome, *, node: str) -> bool:
    """Calibrate authenticated execution, not the pre-STOP absence of clients.

    VERIFY remains a read-only query and still refuses active CUDA clients.
    That exact refusal is a valid calibration; it is never recorded as a
    successful workflow barrier. The independent witness additionally requires
    a completed NVIDIA query before it releases the real STOP rendezvous.
    """
    if unknown_outcome_failure(outcome.details):
        return False
    if outcome.status is WorkflowStepStatus.SUCCEEDED:
        return True
    if outcome.status is not WorkflowStepStatus.FAILED:
        return False
    return bool(
        (outcome.details or {}).get("gpu_client_quiesce_attempt") == 1
        and re.fullmatch(
            re.escape(f"node agent {node}: GPU compute clients are still active: ")
            + r"GPU-[A-Za-z0-9-]+:[1-9][0-9]*"
            + r"(?:, GPU-[A-Za-z0-9-]+:[1-9][0-9]*)*",
            outcome.error or "",
        )
    )


class UidBoundCore:
    def __init__(self, core: Any, scope: AcceptanceScope, serialize: Any) -> None:
        self.core = core
        self.scope = scope
        self.serialize = serialize

    def __getattr__(self, name: str) -> Any:
        return getattr(self.core, name)

    def read_node(self, name: str, **kwargs: Any) -> Any:
        expected = next((node for node in self.scope.nodes if node.name == name), None)
        if expected is None:
            raise BoundaryDenied("node read escaped the approved target set")
        value = self.core.read_node(name, **kwargs)
        document = self.serialize(value)
        if (document.get("metadata") or {}).get("uid") != expected.uid or (
            (document.get("status") or {}).get("nodeInfo") or {}
        ).get("bootID") != expected.boot_id:
            raise BoundaryDenied("node UID or boot changed during physical acceptance")
        return value

    def patch_node(self, name: str, body: Any, **kwargs: Any) -> Any:
        current = self.serialize(self.read_node(name))
        if not isinstance(body, dict):
            raise BoundaryDenied("node mutation has no structured UID precondition")
        patched = deepcopy(body)
        metadata = patched.setdefault("metadata", {})
        metadata["uid"] = current["metadata"]["uid"]
        metadata.setdefault("resourceVersion", current["metadata"]["resourceVersion"])
        return self.core.patch_node(name, patched, **kwargs)


class ExecutorProbe:
    def __init__(
        self,
        scope: AcceptanceScope,
        workflow: WorkflowRequest,
        incident: FaultIncident,
        executor: Any,
        *,
        incoming: TextIO = sys.stdin,
        outgoing: TextIO = sys.stdout,
    ) -> None:
        self.scope = scope
        self.workflow = workflow
        self.incident = incident
        self.executor = executor
        self.incoming = incoming
        self.outgoing = outgoing
        self.producer = process_identity(os.getpid())
        self.sequence = 0
        self.cleanup_deadline: float | None = None
        self.revoked = False
        self.stopped: StopReceipt | None = None
        self.mutation: MutationReceipt | None = None
        self.permit: RecheckPermit | None = None
        self.starts: tuple[WitnessStart, WitnessStart] | None = None
        self.node_commands: dict[str, str] = {}
        self.node_operations: dict[str, WorkflowOperation] = {}
        self.outcomes: list[WorkflowStepOutcome] = []
        kube = [
            item
            for item in executor.adapters
            if isinstance(item, KubernetesWorkflowAdapter)
        ]
        nodes = [
            item
            for item in executor.adapters
            if isinstance(item, NodeActionWorkflowAdapter)
        ]
        if len(kube) != 1 or len(nodes) != 1:
            raise BoundaryDenied(
                "live probe requires the installed Kubernetes and Node Action adapters"
            )
        self.kube = kube[0]
        self.node = nodes[0]
        self.validator: KubernetesStopOwnershipValidator = (
            executor.stop_ownership_validator
        )
        if (
            executor.client.cluster_id != scope.cluster_id
            or incident.cluster_id != scope.cluster_id
            or scope.workload.namespace not in executor.allowed_namespaces
            or workflow.request_id != scope.workflow_id
            or incident.incident_id != scope.incident_id
            or workflow.incident_id != incident.incident_id
            or workflow.fencing_token != scope.fencing_token
            or incident.fencing_token != scope.fencing_token
            or workflow.execution_epoch != scope.execution_epoch
            or workflow.runtime_profile_version != scope.runtime_profile
            or any(
                step.operation not in CASE_OPERATIONS
                for step in workflow.official_steps
            )
            or set(node for step in workflow.official_steps for node in step.node_ids)
            != {node.name for node in scope.nodes}
        ):
            raise BoundaryDenied("live probe inputs are outside the approved scope")
        core = UidBoundCore(self.kube.core, scope, self.kube._serialize_workload)
        self.kube.core = core
        self.validator.core = core
        self.validator.before_recheck = self.at_physical_boundary

    def emit(self, kind: str, payload: dict[str, Any]) -> None:
        self.sequence += 1
        message = json.dumps(
            {
                "kind": kind,
                "scope_sha256": self.scope.digest(),
                "sequence": self.sequence,
                "payload": payload,
            },
            separators=(",", ":"),
            allow_nan=False,
        )
        if len(message.encode()) > MAX_MESSAGE_BYTES:
            raise BoundaryDenied("live probe response exceeds its bound")
        self.outgoing.write(message + "\n")
        self.outgoing.flush()

    def receive(self, kind: str, *, cleanup: bool = False) -> dict[str, Any]:
        remaining = (
            self.scope.maintenance_end - datetime.now(timezone.utc)
        ).total_seconds()
        if cleanup:
            if self.cleanup_deadline is None:
                self.cleanup_deadline = time.monotonic() + 180
            remaining = self.cleanup_deadline - time.monotonic()
        if remaining <= 0:
            raise BoundaryDenied("live probe maintenance window ended")
        if not select.select([self.incoming], [], [], remaining)[0]:
            raise BoundaryDenied("live probe controller stopped responding")
        line = self.incoming.readline(MAX_MESSAGE_BYTES + 1)
        if (
            not line
            or len(line.encode()) > MAX_MESSAGE_BYTES
            or not line.endswith("\n")
        ):
            self.revoked = True
            raise BoundaryDenied("live probe controller was lost")
        message = json.loads(line)
        if (
            not isinstance(message, dict)
            or set(message) != {"kind", "scope_sha256", "payload"}
            or message["kind"] != kind
            or message["scope_sha256"] != self.scope.digest()
            or not isinstance(message["payload"], dict)
        ):
            self.revoked = True
            raise BoundaryDenied("live probe control message is stale or out of order")
        return message["payload"]

    def lease_reason(self) -> str | None:
        if (
            self.revoked
            or datetime.now(timezone.utc) >= self.scope.maintenance_end
            or select.select([self.incoming], [], [], 0)[0]
        ):
            self.revoked = True
            return "owned acceptance controller is no longer authorized"
        nonce = secrets.token_hex(32)
        expected = {
            "workflow_id": self.scope.workflow_id,
            "fencing_token": self.scope.fencing_token,
            "execution_epoch": self.scope.execution_epoch,
            "nonce": nonce,
        }
        try:
            self.emit("holder-check", expected)
            checked = self.receive("holder-check-result")
            if (
                any(checked.get(key) != value for key, value in expected.items())
                or checked.get("holder_valid") is not True
                or checked.get("lifetime_deadline_at")
                != self.scope.maintenance_end.isoformat()
                or datetime.now(timezone.utc) >= self.scope.maintenance_end
            ):
                raise BoundaryDenied("owned CPU holder check is stale")
        except Exception:
            self.revoked = True
            return "owned acceptance workflow lease could not be verified"
        return None

    def context(self, index: int) -> WorkflowStepContext:
        step = self.workflow.official_steps[index]
        return WorkflowStepContext(
            workflow=self.workflow,
            incident=self.incident,
            step=step,
            step_index=index,
            request=WorkflowExecutionRequest(
                expected_fencing_token=self.scope.fencing_token
            ),
            idempotency_key=f"{self.workflow.request_id}/{index}/{step.operation.value}",
        )

    def run_step(self, index: int, *, cleanup: bool = False) -> WorkflowStepOutcome:
        context = self.context(index)
        if cleanup and context.step.operation not in {
            WorkflowOperation.RESTORE_GPU_SERVICES,
            WorkflowOperation.RESTORE_SCHEDULING,
        }:
            raise BoundaryDenied("cleanup cannot authorize a new hardware action")
        adapter = self.kube if self.kube.supports(context.step) else self.node
        deadline = (
            self.cleanup_deadline or time.monotonic()
            if cleanup
            else time.monotonic()
            + max(
                0,
                (
                    self.scope.maintenance_end - datetime.now(timezone.utc)
                ).total_seconds(),
            )
        )
        while time.monotonic() < deadline:
            if not cleanup:
                self.scope.check_window(datetime.now(timezone.utc))
                if self.lease_reason() is not None:
                    raise BoundaryDenied("owned acceptance workflow lease was lost")
            lease = active_lease_guard.set(
                (lambda: None) if cleanup else self.lease_reason
            )
            try:
                with stop_ownership_scope(self.validator):
                    outcome = adapter.execute(context)
            finally:
                active_lease_guard.reset(lease)
            details = outcome.details or {}
            command_id = details.get("node_action_command_id")
            if isinstance(command_id, str):
                self.node_commands[command_id] = context.step.node_ids[0]
                self.node_operations[command_id] = context.step.operation
            for node_id, result in (details.get("node_results") or {}).items():
                command_id = result.get("node_action_command_id")
                if isinstance(command_id, str):
                    self.node_commands[command_id] = str(node_id)
                    self.node_operations[command_id] = context.step.operation
            execution = WorkflowStepExecution(
                step_index=index,
                operation=context.step.operation,
                status=outcome.status,
                phase="official",
                adapter_operation_id=outcome.adapter_operation_id,
                details=details,
                error=outcome.error,
            )
            completed = self.workflow.completed_step_indexes
            self.workflow = self.workflow.model_copy(
                update={
                    "step_executions": [*self.workflow.step_executions, execution],
                    "completed_step_indexes": (
                        sorted({*completed, index})
                        if outcome.status is WorkflowStepStatus.SUCCEEDED
                        else completed
                    ),
                    "updated_at": datetime.now(timezone.utc),
                }
            )
            self.outcomes.append(outcome)
            if outcome.status is not WorkflowStepStatus.WAITING:
                return outcome
            context = self.context(index)
            # This wait services cancellation, not the ownership race. The
            # mutation is released only by a native Agent challenge below.
            if select.select(
                [self.incoming], [], [], min(0.25, max(0, deadline - time.monotonic()))
            )[0]:
                self.revoked = True
                raise BoundaryDenied("controller interrupted a pending node action")
        raise BoundaryDenied("live step did not finish in its bounded window")

    def current_workload(self) -> WorkloadIdentity:
        expected = self.scope.workload
        current = self.kube._read_workload(
            expected.namespace, "pytorchjob", expected.name
        )
        metadata = self.kube._serialize_workload(current).get("metadata") or {}
        owners = metadata.get("ownerReferences") or []
        if len(owners) > 1:
            raise BoundaryDenied("live workload ownership is ambiguous")
        owner_uid = (
            str(owners[0].get("uid") or "")
            if owners
            else str(metadata.get("uid") or "")
        )
        return WorkloadIdentity(
            namespace=str(metadata.get("namespace") or ""),
            name=str(metadata.get("name") or ""),
            uid=str(metadata.get("uid") or ""),
            owner_uid=owner_uid,
            attempt_id=str(
                (metadata.get("labels") or {}).get("gpu-fault.io/attempt-id") or ""
            ),
        )

    def at_physical_boundary(
        self, context: WorkflowStepContext, receipt: StopOwnershipReceipt
    ) -> None:
        challenge = current_ownership_challenge()
        if (
            challenge is None
            or challenge.boundary != "AGENT_PRE_SPAWN"
            or context.step.operation is not WorkflowOperation.RESET_GPU
        ):
            return
        if self.permit is not None:
            if self.scope.scenario != "unchanged-owner":
                raise BoundaryDenied(
                    "a late callback followed the refused physical boundary"
                )
            return
        if self.starts is None or self.stopped is not None:
            raise BoundaryDenied("physical callback has no unique armed session")
        self.node_commands[challenge.command_id] = challenge.node_id
        self.node_operations[challenge.command_id] = context.step.operation
        if self.current_workload() != self.scope.workload:
            raise BoundaryDenied("ownership changed before the controlled boundary")
        pods = self.kube._serialize_workload(
            self.kube.core.list_namespaced_pod(self.scope.workload.namespace)
        ).get("items")
        if not isinstance(pods, list) or any(
            (pod.get("metadata") or {}).get("uid")
            in {item.pod_uid for item in self.scope.participants}
            for pod in pods
        ):
            raise BoundaryDenied(
                "source Pod absence was not observed at the physical boundary"
            )
        self.stopped = StopReceipt(
            scope_sha256=self.scope.digest(),
            producer=self.producer,
            executor_uid=self.scope.executor_uid,
            boundary_id=challenge.nonce,
            stop_command_id=receipt.stop_idempotency_key,
            queued_command_id=challenge.command_id,
            agent_generation=challenge.agent_generation,
            agent_boundary="AGENT_PRE_SPAWN",
            sequence=self.sequence + 1,
            workload=self.scope.workload,
            participants=self.scope.participants,
            absent_pod_uids=tuple(item.pod_uid for item in self.scope.participants),
            empty_client_node_uids=tuple(item.uid for item in self.scope.nodes),
            witness_start_sha256=(self.starts[0].digest(), self.starts[1].digest()),
            hardware_submitted=False,
            gate_closed=True,
        )
        self.emit("stop", self.stopped.model_dump(mode="json"))
        mutation = self.receive("observe-mutation")
        current = self.current_workload()
        participants: tuple[Participant, ...] = self.scope.participants
        sibling_uid = None
        sibling_node = None
        if self.scope.scenario == "late-sibling":
            sibling_uid = str(mutation.get("sibling_uid") or "")
            sibling_node = self.scope.nodes[1].uid
            if not sibling_uid or mutation.get("physical_client_verified") is not True:
                raise BoundaryDenied(
                    "late sibling has no independent physical client receipt"
                )
            response = self.kube._serialize_workload(
                self.kube.core.list_namespaced_pod(self.scope.workload.namespace)
            )
            matches = [
                pod
                for pod in response.get("items", [])
                if (pod.get("metadata") or {}).get("uid") == sibling_uid
            ]
            if (
                len(matches) != 1
                or (matches[0].get("spec") or {}).get("nodeName")
                != self.scope.nodes[1].name
                or (matches[0].get("status") or {}).get("phase") != "Running"
                or not any(
                    owner.get("uid") == self.scope.workload.uid
                    and owner.get("controller") is True
                    for owner in (matches[0].get("metadata") or {}).get(
                        "ownerReferences", []
                    )
                )
            ):
                raise BoundaryDenied(
                    "late sibling UID/owner/placement was not observed"
                )
            participants = (
                *participants,
                Participant(
                    pod_uid=sibling_uid,
                    owner_uid=self.scope.workload.owner_uid,
                    node_uid=sibling_node,
                ),
            )
        metadata = self.kube._serialize_workload(
            self.kube._read_workload(current.namespace, "pytorchjob", current.name)
        )["metadata"]
        self.mutation = MutationReceipt(
            scope_sha256=self.scope.digest(),
            producer=self.producer,
            executor_uid=self.scope.executor_uid,
            stop_sha256=self.stopped.digest(),
            sequence=self.sequence + 1,
            workload=current,
            participants=participants,
            resource_version=str(metadata.get("resourceVersion") or ""),
            live_sibling_pod_uid=sibling_uid,
            live_sibling_node_uid=sibling_node,
            sibling_gpu_client_observed=sibling_uid is not None,
        )
        self.emit("mutation", self.mutation.model_dump(mode="json"))
        permit = RecheckPermit.model_validate_json(json.dumps(self.receive("recheck")))
        if (
            permit.scope_sha256 != self.scope.digest()
            or permit.stop_sha256 != self.stopped.digest()
            or permit.boundary_id != challenge.nonce
            or permit.mutation_sha256 != self.mutation.digest()
        ):
            raise BoundaryDenied("physical callback received a stale release")
        self.permit = permit

    def drain_actions(self) -> None:
        deadline = self.cleanup_deadline or time.monotonic()
        pending = dict(self.node_commands)
        while pending and time.monotonic() < deadline:
            for command_id, node_id in list(pending.items()):
                self.kube.core.read_node(node_id)
                value = self.node.read_action_result(
                    self.scope.cluster_id, node_id, command_id
                )
                if value is None:
                    raise BoundaryDenied("accepted Node Action has no terminal receipt")
                if value.command_id != command_id:
                    raise BoundaryDenied(
                        "Node Action receipt belongs to another command"
                    )
                if value.state is not NodeActionExecutionState.PENDING:
                    if (
                        value.result is None
                        or value.result.command_id != command_id
                        or value.result.operation != self.node_operations[command_id]
                        or value.state.value != value.result.status.value
                    ):
                        raise BoundaryDenied(
                            "Node Action terminal receipt is incomplete"
                        )
                    pending.pop(command_id)
            if pending and select.select([self.incoming], [], [], 0.25)[0]:
                raise BoundaryDenied("controller interrupted physical action drainage")
        if pending:
            raise BoundaryDenied("Node Actions did not reach terminal quiescence")

    def run(self) -> None:
        if not all(
            self.node.read_ownership_capability(self.scope.cluster_id, node.name)
            for node in self.scope.nodes
        ):
            raise BoundaryDenied("an Agent lacks final ownership enforcement")
        self.emit("ready", {"producer": self.producer.model_dump(mode="json")})
        self.receive("calibrate")
        calibrations = []
        for node in self.scope.nodes:
            context = WorkflowStepContext(
                workflow=self.workflow,
                incident=self.incident,
                step=WorkflowStepSpec(
                    operation=WorkflowOperation.VERIFY_NO_GPU_CLIENTS,
                    execution_owner=self.node.owner,
                    node_ids=[node.name],
                    parameters={
                        "compute_clients_only": True,
                        "spare_health_check": True,
                    },
                ),
                step_index=0,
                request=WorkflowExecutionRequest(
                    expected_fencing_token=self.scope.fencing_token
                ),
                idempotency_key=f"{self.workflow.request_id}/calibration/{node.name}",
            )
            for _ in range(100):
                self.scope.check_window(datetime.now(timezone.utc))
                if self.lease_reason() is not None:
                    raise BoundaryDenied("calibration workflow lease was lost")
                guard = active_lease_guard.set(self.lease_reason)
                try:
                    outcome = self.node.execute(context)
                finally:
                    active_lease_guard.reset(guard)
                if outcome.status is not WorkflowStepStatus.WAITING:
                    if not readonly_calibration_completed(outcome, node=node.name):
                        raise BoundaryDenied("physical calibration failed")
                    calibrations.append(
                        {
                            "node": node.name,
                            "operation": context.step.operation.value,
                            "status": outcome.status.value,
                            "compute_clients_present": (
                                outcome.status is WorkflowStepStatus.FAILED
                            ),
                            "workflow_barrier_satisfied": False,
                        }
                    )
                    break
                if select.select([self.incoming], [], [], 0.25)[0]:
                    raise BoundaryDenied("controller interrupted physical calibration")
            else:
                raise BoundaryDenied("physical calibration did not finish")
        self.emit(
            "calibrated",
            {
                "nodes": [node.name for node in self.scope.nodes],
                "queries": calibrations,
            },
        )
        payload = self.receive("begin")
        starts = payload.get("witness_starts")
        if not isinstance(starts, list) or len(starts) != 2:
            raise BoundaryDenied("both physical witnesses are required")
        self.starts = (
            WitnessStart.model_validate_json(json.dumps(starts[0])),
            WitnessStart.model_validate_json(json.dumps(starts[1])),
        )
        outcomes = []
        for index, step in enumerate(self.workflow.official_steps):
            if step.operation in {
                WorkflowOperation.RESTORE_GPU_SERVICES,
                WorkflowOperation.RESTORE_SCHEDULING,
            }:
                continue
            if self.scope.scenario != "unchanged-owner" and step.node_ids == [
                self.scope.nodes[1].name
            ]:
                continue
            outcome = self.run_step(index)
            outcomes.append(outcome)
            if (
                step.operation is WorkflowOperation.STOP_WORKLOADS
                and outcome.status is WorkflowStepStatus.SUCCEEDED
            ):
                receipt = StopOwnershipReceipt.model_validate_json(
                    json.dumps((outcome.details or {}).get(STOP_RECEIPT_KEY))
                )
                self.emit(
                    "contained", {"stop_idempotency_key": receipt.stop_idempotency_key}
                )
                if self.receive("continue-boundary"):
                    raise BoundaryDenied(
                        "containment acknowledgement has unexpected data"
                    )
            if outcome.status is WorkflowStepStatus.FAILED:
                break
        if self.stopped is None or self.mutation is None or self.permit is None:
            raise BoundaryDenied(
                "the installed Agent did not reach its physical ownership boundary"
            )
        last = outcomes[-1]
        code: Literal["ALLOWED", "STOP_OWNERSHIP_DRIFT", "STOP_PARTICIPANTS_CHANGED"]
        reason = (last.details or {}).get("reason")
        if last.status is WorkflowStepStatus.SUCCEEDED:
            code = "ALLOWED"
        elif reason == "STOP_OWNERSHIP_DRIFT":
            code = "STOP_OWNERSHIP_DRIFT"
        elif reason == "STOP_PARTICIPANTS_CHANGED":
            code = "STOP_PARTICIPANTS_CHANGED"
        else:
            raise BoundaryDenied(
                "the candidate failed without the expected ownership refusal"
            )
        decision = DecisionReceipt(
            scope_sha256=self.scope.digest(),
            producer=self.producer,
            executor_uid=self.scope.executor_uid,
            permit_sha256=self.permit.digest(),
            sequence=self.sequence + 1,
            decision=code,
            checked_node_uids=tuple(node.uid for node in self.scope.nodes),
            hardware_submitted=code == "ALLOWED",
            restart_submitted=False,
            product_guard="kubernetes-stop-ownership/v1",
        )
        self.emit("decision", decision.model_dump(mode="json"))
        self.receive("quiesce", cleanup=True)
        self.revoked = True
        self.validator.before_recheck = None
        self.drain_actions()
        for index, step in enumerate(self.workflow.official_steps):
            if step.operation is not WorkflowOperation.RESTORE_GPU_SERVICES:
                continue
            if self.scope.scenario != "unchanged-owner" and step.node_ids == [
                self.scope.nodes[1].name
            ]:
                continue
            if (
                self.run_step(index, cleanup=True).status
                is not WorkflowStepStatus.SUCCEEDED
            ):
                raise BoundaryDenied("owned node compensation failed")
        self.drain_actions()
        self.emit("services-restored", {"node_commands_terminal": True})
        self.receive("restore-scheduling", cleanup=True)
        for index, step in enumerate(self.workflow.official_steps):
            if step.operation is not WorkflowOperation.RESTORE_SCHEDULING:
                continue
            if self.scope.scenario != "unchanged-owner" and step.node_ids == [
                self.scope.nodes[1].name
            ]:
                continue
            if (
                self.run_step(index, cleanup=True).status
                is not WorkflowStepStatus.SUCCEEDED
            ):
                raise BoundaryDenied("owned scheduling restoration failed")
        self.emit(
            "actions-drained", {"workflow": self.workflow.model_dump(mode="json")}
        )
        terminal = self.receive("confirm-terminal", cleanup=True)
        if (
            terminal.get("workflow_id") != self.scope.workflow_id
            or terminal.get("fencing_token") != self.scope.fencing_token
            or terminal.get("execution_epoch") != self.scope.execution_epoch
            or terminal.get("status") not in {"SUCCEEDED", "SUPERSEDED"}
        ):
            raise BoundaryDenied("owned control-plane workflow is not terminal")
        quiet = QuiescenceReceipt(
            scope_sha256=self.scope.digest(),
            producer=self.producer,
            executor_uid=self.scope.executor_uid,
            decision_sha256=decision.digest(),
            sequence=self.sequence + 1,
            open_commands=0,
            pending_callbacks=0,
            workflow_terminal=True,
            gate_revoked=True,
        )
        self.emit("quiescence", quiet.model_dump(mode="json"))
        self.receive("revoke", cleanup=True)
        self.emit("revoked", {"gate_revoked": True})
        self.receive("finish", cleanup=True)
        self.emit(
            "finished",
            {
                "workflow": self.workflow.model_dump(mode="json"),
                "gate_revoked": True,
            },
        )


def main() -> int:
    try:
        line = sys.stdin.readline(MAX_MESSAGE_BYTES + 1)
        if len(line.encode("utf-8")) > MAX_MESSAGE_BYTES:
            raise BoundaryDenied("live probe input exceeds its byte bound")
        request = json.loads(line)
        if request.get("inspect_only") is True:
            executor = executor_from_environment()
            if request["cluster_id"] != executor.client.cluster_id:
                raise BoundaryDenied("probe inspection crossed the cluster boundary")
            nodes = request["nodes"]
            if not isinstance(nodes, list) or len(nodes) != 2 or len(set(nodes)) != 2:
                raise BoundaryDenied("probe inspection requires two explicit nodes")
            adapters = [
                item
                for item in executor.adapters
                if isinstance(item, NodeActionWorkflowAdapter)
            ]
            if len(adapters) != 1:
                raise BoundaryDenied("installed Node Action adapter is unavailable")
            print(
                json.dumps(
                    {
                        "protocol_ready": all(
                            adapters[0].read_ownership_capability(
                                executor.client.cluster_id, node
                            )
                            for node in nodes
                        ),
                        "cluster_id": executor.client.cluster_id,
                        "nodes": nodes,
                    }
                ),
                flush=True,
            )
            return 0
        scope = AcceptanceScope.model_validate_json(json.dumps(request["scope"]))
        scope.check_window(datetime.now(timezone.utc))
        executor = executor_from_environment()
        probe = ExecutorProbe(
            scope,
            WorkflowRequest.model_validate(request["workflow"]),
            FaultIncident.model_validate(request["incident"]),
            executor,
        )
        probe.run()
        return 0
    except BaseException as exc:
        print(json.dumps({"error_kind": type(exc).__name__}), flush=True)
        return 1

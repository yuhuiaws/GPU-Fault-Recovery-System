"""Bind expected test-workload replacements to completed product commands."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, cast

from scripts.e2e.regional.regional_commands import RegionalFixtureError


class WorkloadOwnershipError(RegionalFixtureError):
    """Identity rejection, not a transient Kubernetes read failure."""


def require(value: bool, reason: str) -> None:
    if not value:
        raise WorkloadOwnershipError(f"managed workload restart custody: {reason}")


def mapping(value: Any, name: str) -> Mapping[str, Any]:
    require(isinstance(value, dict), f"{name} is missing or malformed")
    return cast(Mapping[str, Any], value)


def records(value: Any, name: str) -> list[dict[str, Any]]:
    require(
        isinstance(value, list) and all(isinstance(item, dict) for item in value),
        f"{name} is missing or malformed",
    )
    return cast(list[dict[str, Any]], value)


@dataclass(frozen=True)
class RestartCustody:
    cluster_id: str
    namespace: str
    job_id: str
    source_attempt_id: str
    target_attempt_id: str
    source_workload_id: str
    source_uid: str
    target_workload_ids: frozenset[str]
    workflow_id: str
    incident_id: str
    operation_id: str
    execution_epoch: int
    fencing_token: int
    step_index: int
    restart_count: int
    restart_budget: int
    #: Controllers whose restart binding was already proven. The product's own
    #: later mutations of the same controller (a second STOP_WORKLOADS in
    #: COLLECT-016 segment B) legitimately re-annotate it, so a controller is
    #: held to the binding once, on adoption, and identified by UID afterwards.
    verified: set[tuple[str, str]] = field(
        default_factory=set, compare=False, hash=False, repr=False
    )

    @classmethod
    def from_state(
        cls,
        state: dict[str, Any],
        *,
        cluster_id: str,
        namespace: str,
        job_id: str,
        source_attempt_id: str,
        source_workload_id: str,
        source_uid: str,
        gpu_count: int,
        restart_budget: int,
    ) -> RestartCustody:
        require(bool(source_uid), "original controller UID is unproven")
        workflow = mapping(state.get("workflow"), "workflow")
        incident = mapping(state.get("incident"), "incident")
        workflow_id, incident_id = (
            workflow.get("request_id"),
            incident.get("incident_id"),
        )
        require(
            isinstance(workflow_id, str)
            and bool(workflow_id)
            and isinstance(incident_id, str)
            and bool(incident_id)
            and workflow.get("status") == "SUCCEEDED"
            and workflow.get("incident_id") == incident_id
            and incident.get("cluster_id") == cluster_id
            and incident.get("job_id") == job_id,
            "successful workflow or incident identity differs",
        )
        executions = records(workflow.get("step_executions"), "step executions")
        restarts = [
            item
            for item in executions
            if item.get("operation") == "RESTART_WORKLOAD"
            and item.get("status") == "SUCCEEDED"
            and source_workload_id in (item.get("details") or {}).get("workloads", [])
        ]
        require(len(restarts) == 1, "restart execution is missing or ambiguous")
        execution = restarts[0]
        index, phase = execution.get("step_index"), execution.get("phase")
        require(
            type(index) is int and index >= 0 and phase in {"official", "safety"},
            "restart phase or index is invalid",
        )
        index = cast(int, index)
        steps = records(workflow.get(f"{phase}_steps"), "current plan")
        require(index < len(steps), "restart index is outside the current plan")
        step = steps[index]
        scope = {
            "cluster_id": cluster_id,
            "job_id": job_id,
            "source_attempt_id": source_attempt_id,
            "source_gpu_count": gpu_count,
            "restart_budget": restart_budget,
        }
        require(
            step.get("operation") == "RESTART_WORKLOAD"
            and source_workload_id in step.get("workload_ids", [])
            and all(
                (step.get("parameters") or {}).get(key) == value
                and type((step.get("parameters") or {}).get(key)) is type(value)
                for key, value in scope.items()
            ),
            "current restart plan scope differs",
        )
        operation = execution.get("adapter_operation_id")
        require(
            isinstance(operation, str) and operation.startswith("remote/"),
            "restart has no remote execution receipt",
        )
        operation = cast(str, operation)
        commands = [
            item
            for item in records(state.get("commands"), "remote commands")
            if item.get("command_id") == operation.removeprefix("remote/")
        ]
        require(len(commands) == 1, "remote restart command is missing or ambiguous")
        command = commands[0]
        command_workflow = mapping(command.get("workflow"), "command workflow")
        command_step = mapping(command.get("step"), "command step")
        operation_id = command.get("idempotency_key")
        epoch, fencing = (
            command_workflow.get("execution_epoch"),
            command.get("fencing_token"),
        )
        require(
            command.get("status") == "SUCCEEDED"
            and command.get("cluster_id") == cluster_id
            and command.get("incident_id") == incident_id
            and command.get("workflow_request_id") == workflow_id
            and command_workflow.get("request_id") == workflow_id
            and command.get("step_index") == index
            and command_step.get("operation") == "RESTART_WORKLOAD"
            and source_workload_id in command_step.get("workload_ids", [])
            and all(
                (command_step.get("parameters") or {}).get(key) == value
                and type((command_step.get("parameters") or {}).get(key)) is type(value)
                for key, value in scope.items()
            )
            and isinstance(operation_id, str)
            and bool(operation_id)
            and type(epoch) is int
            and epoch >= 0
            and type(fencing) is int
            and fencing >= 0
            and fencing == workflow.get("fencing_token"),
            "completed remote restart identity or scope differs",
        )
        details = mapping(execution.get("details"), "restart details")
        result = mapping(command.get("result_details"), "remote restart result")
        target = details.get("restart_attempt_id")
        require(
            isinstance(target, str)
            and bool(target)
            and target != source_attempt_id
            and result.get("restart_attempt_id") == target,
            "target attempt is missing or contradictory",
        )
        notification = mapping(details.get("notification_context"), "restart context")
        count = notification.get("restart_count")
        authorization = mapping(
            command.get("restart_authorization"), "restart authorization"
        )
        require(
            all(
                notification.get(key) == value
                and type(notification.get(key)) is type(value)
                for key, value in scope.items()
                if key != "cluster_id"
            )
            and notification.get("restart_attempt_id") == target
            and notification.get("target_gpu_count") == gpu_count
            and type(count) is int
            and 0 < count <= restart_budget
            and all(
                authorization.get(key) == value
                and type(authorization.get(key)) is type(value)
                for key, value in scope.items()
            )
            and authorization.get("reservation_id") == operation_id
            and authorization.get("restart_count") == count
            and result.get("notification_context") == dict(notification),
            "restart budget or result context differs",
        )
        target_ids = details.get("restarted_workload_ids", [source_workload_id])
        require(
            isinstance(target_ids, list)
            and len(target_ids) == 1
            and all(
                isinstance(item, str)
                and item.startswith(namespace + "/")
                and len(item.split("/")) == 3
                and item.split("/")[1] == source_workload_id.split("/")[1]
                and bool(item.split("/")[2])
                for item in target_ids
            )
            and len(set(target_ids)) == len(target_ids)
            and result.get("restarted_workload_ids", [source_workload_id])
            == target_ids,
            "replacement controller result is invalid or contradictory",
        )
        require(
            result.get("workloads") == details.get("workloads")
            and result.get("suspended") is False
            and details.get("suspended") is False,
            "remote restart completion differs from the workflow result",
        )
        # A stop receipt, when present, must agree with the fixture's original UID.
        for item in executions:
            receipt = (item.get("details") or {}).get("stop_ownership_receipt_v1")
            if not isinstance(receipt, dict):
                continue
            for source in records(receipt.get("workloads"), "stopped workloads"):
                if source.get("workload_id") == source_workload_id:
                    require(
                        source.get("uid") == source_uid
                        and source.get("attempt_id") == source_attempt_id,
                        "STOP receipt names another source controller",
                    )
        return cls(
            cluster_id=cluster_id,
            namespace=namespace,
            job_id=job_id,
            source_attempt_id=source_attempt_id,
            target_attempt_id=cast(str, target),
            source_workload_id=source_workload_id,
            source_uid=source_uid,
            target_workload_ids=frozenset(target_ids),
            workflow_id=cast(str, workflow_id),
            incident_id=cast(str, incident_id),
            operation_id=cast(str, operation_id),
            execution_epoch=cast(int, epoch),
            fencing_token=cast(int, fencing),
            step_index=index,
            restart_count=cast(int, count),
            restart_budget=restart_budget,
        )

    def validate(
        self,
        document: dict[str, Any],
        *,
        owner: str,
        known_uids: dict[tuple[str, str], str],
        read: Callable[[str, str], dict[str, Any] | None],
    ) -> tuple[str, str, str]:
        current = document
        visited: set[tuple[str, str]] = set()
        for _ in range(8):
            metadata = mapping(current.get("metadata"), "resource metadata")
            raw_kind, name, uid = (
                current.get("kind"),
                metadata.get("name"),
                metadata.get("uid"),
            )
            require(
                isinstance(raw_kind, str)
                and raw_kind in {"Pod", "Job", "PyTorchJob", "JobSet"}
                and isinstance(name, str)
                and bool(name)
                and isinstance(uid, str)
                and bool(uid)
                and metadata.get("namespace") == self.namespace,
                "resource identity differs",
            )
            kind, name, uid = (
                cast(str, raw_kind).lower(),
                cast(str, name),
                cast(str, uid),
            )
            labels = mapping(metadata.get("labels"), "resource labels")
            require(
                labels.get("gpu-fault.io/acceptance-owner") == owner
                and labels.get("gpu-fault.io/job-id") == self.job_id
                and labels.get("gpu-fault.io/attempt-id") == self.target_attempt_id,
                "replacement labels differ",
            )
            identity = (kind, name)
            require(identity not in visited, "controller chain cycles")
            visited.add(identity)
            workload_id = f"{self.namespace}/{kind}/{name}"
            if workload_id in self.target_workload_ids:
                expected_uid = (
                    self.source_uid
                    if workload_id == self.source_workload_id
                    else known_uids.get(identity, uid)
                )
                require(uid == expected_uid, "replacement controller UID changed")
                if identity in self.verified:
                    return kind, name, uid
                annotations = mapping(
                    metadata.get("annotations"), "controller annotations"
                )
                expected = {
                    "gpu-fault.io/workflow-id": self.workflow_id,
                    "gpu-fault.io/incident-id": self.incident_id,
                    "gpu-fault.io/operation-id": self.operation_id,
                    "gpu-fault.io/execution-epoch": str(self.execution_epoch),
                    "gpu-fault.io/fencing-token": str(self.fencing_token),
                    "gpu-fault.io/workflow-step-index": str(self.step_index),
                    "gpu-fault.io/restart-count": str(self.restart_count),
                    "gpu-fault.io/restart-budget": str(self.restart_budget),
                }
                mismatched = sorted(
                    key
                    for key, value in expected.items()
                    if annotations.get(key) != value
                )
                require(
                    not mismatched,
                    "controller does not bind the completed restart: "
                    + ", ".join(
                        f"{key}={annotations.get(key)!r} (expected {expected[key]!r})"
                        for key in mismatched
                    ),
                )
                self.verified.add(identity)
                return kind, name, uid
            owners = records(metadata.get("ownerReferences"), "controller owners")
            require(
                len(owners) == 1 and owners[0].get("controller") is True,
                "replacement has no unique controller",
            )
            parent = owners[0]
            require(
                parent.get("kind") in {"Job", "PyTorchJob", "JobSet"}
                and isinstance(parent.get("name"), str)
                and isinstance(parent.get("uid"), str),
                "unsupported controller reference",
            )
            parent_document = read(parent["kind"].lower(), parent["name"])
            require(
                parent_document is not None
                and parent_document.get("apiVersion") == parent.get("apiVersion")
                and (parent_document.get("metadata") or {}).get("uid") == parent["uid"],
                "replacement controller reference changed",
            )
            current = cast(dict[str, Any], parent_document)
        raise WorkloadOwnershipError(
            "managed workload controller chain exceeds its bound"
        )

"""Completed product restart receipts for stateful fixture tests."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from scripts.e2e.regional.managed_workload_fixture import ManagedWorkloadFixture


def restart_state(
    fixture: ManagedWorkloadFixture, *, retry_parent_name: str | None = None
) -> dict[str, Any]:
    source_id = (
        f"{fixture.regional.settings.namespace}/{fixture.resource}/{fixture.name}"
    )
    parameters = {
        "cluster_id": fixture.regional.settings.cluster_id,
        "job_id": fixture.settings.job_id,
        "source_attempt_id": fixture.settings.attempt_id,
        "source_gpu_count": fixture.settings.expected_gpu_count,
        "restart_budget": fixture.settings.restart_budget,
    }
    step = {
        "operation": "RESTART_WORKLOAD",
        "parameters": parameters,
        "workload_ids": [source_id],
    }
    operation = "workflow-restart/0/RESTART_WORKLOAD"
    notification = {
        key: value for key, value in parameters.items() if key != "cluster_id"
    }
    notification.update(
        restart_attempt_id=fixture.settings.attempt_id + "-r-test",
        target_gpu_count=fixture.settings.expected_gpu_count,
        restart_count=1,
    )
    details: dict[str, Any] = {
        "notification_context": notification,
        "restart_attempt_id": notification["restart_attempt_id"],
        "workloads": [source_id],
        "suspended": False,
    }
    if retry_parent_name:
        details["restarted_workload_ids"] = [
            f"{fixture.regional.settings.namespace}/{fixture.resource}/{retry_parent_name}"
        ]
    return {
        "incident": {
            "incident_id": "incident-restart",
            "cluster_id": parameters["cluster_id"],
            "job_id": fixture.settings.job_id,
        },
        "workflow": {
            "request_id": "workflow-restart",
            "incident_id": "incident-restart",
            "status": "SUCCEEDED",
            "fencing_token": 4,
            "official_steps": [deepcopy(step)],
            "step_executions": [
                {
                    "phase": "official",
                    "step_index": 0,
                    "operation": "RESTART_WORKLOAD",
                    "status": "SUCCEEDED",
                    "adapter_operation_id": "remote/command-restart",
                    "details": deepcopy(details),
                },
                {
                    "operation": "STOP_WORKLOADS",
                    "status": "SUCCEEDED",
                    "details": {
                        "stop_ownership_receipt_v1": {
                            "workloads": [
                                {
                                    "workload_id": source_id,
                                    "uid": fixture.owned_uids[
                                        (fixture.resource, fixture.name)
                                    ],
                                    "attempt_id": fixture.settings.attempt_id,
                                }
                            ]
                        }
                    },
                },
            ],
        },
        "commands": [
            {
                "command_id": "command-restart",
                "workflow_request_id": "workflow-restart",
                "incident_id": "incident-restart",
                "cluster_id": parameters["cluster_id"],
                "status": "SUCCEEDED",
                "step_index": 0,
                "idempotency_key": operation,
                "fencing_token": 4,
                "workflow": {"request_id": "workflow-restart", "execution_epoch": 2},
                "step": deepcopy(step),
                "restart_authorization": {
                    **parameters,
                    "reservation_id": operation,
                    "restart_count": 1,
                },
                "result_details": deepcopy(details),
            }
        ],
    }


def apply_restart_metadata(document: dict[str, Any], state: dict[str, Any]) -> None:
    command = state["commands"][0]
    document["metadata"]["labels"]["gpu-fault.io/attempt-id"] = command[
        "result_details"
    ]["restart_attempt_id"]
    document["metadata"].setdefault("annotations", {}).update(
        {
            "gpu-fault.io/workflow-id": command["workflow_request_id"],
            "gpu-fault.io/incident-id": command["incident_id"],
            "gpu-fault.io/operation-id": command["idempotency_key"],
            "gpu-fault.io/execution-epoch": str(command["workflow"]["execution_epoch"]),
            "gpu-fault.io/fencing-token": str(command["fencing_token"]),
            "gpu-fault.io/workflow-step-index": str(command["step_index"]),
            "gpu-fault.io/restart-count": "1",
            "gpu-fault.io/restart-budget": str(
                command["restart_authorization"]["restart_budget"]
            ),
        }
    )

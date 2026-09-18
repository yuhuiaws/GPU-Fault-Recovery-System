"""Local receipts for the sequential COLLECT-014 runner contract."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta
from typing import Any

from gpu_fault.node_agent.ledger import canonical_digest
from tests.regional._collector_reset_support import reset_documents


class SequenceResetAudit:
    def __init__(self, *, node: str = "hyperpod-node") -> None:
        self.node = node
        self.current, _after, _state = reset_documents(
            "RESET_ALL_GPUS_NVSWITCHES", node=node
        )
        self.current["quiesce_states"] = []
        self.state: dict[str, Any] = {}

    def start_variant(self, sxid: int) -> None:
        _before, after, state = reset_documents(
            "RESET_ALL_GPUS_NVSWITCHES", node=self.node
        )
        previous = datetime.fromisoformat(self.current["captured_at"])
        workflow = state["workflow"]
        incident = state["incident"]
        workflow["request_id"] = f"workflow-{sxid}"
        workflow["incident_id"] = incident["incident_id"] = f"incident-{sxid}"
        workflow["official_steps"][0]["parameters"] = {"sxid": sxid}
        workflow["completed_operations"] = [
            "RESET_ALL_GPUS_NVSWITCHES",
            "RESTORE_GPU_SERVICES",
        ]
        row = after["ledger"][0]
        row.update(
            command_id=f"node-command-{sxid}",
            workflow_request_id=workflow["request_id"],
            incident_id=incident["incident_id"],
            parameters_digest=canonical_digest({"sxid": sxid}),
            started_at=(previous + timedelta(seconds=1)).isoformat(),
            completed_at=(previous + timedelta(seconds=2)).isoformat(),
        )
        row["result"]["command_id"] = row["command_id"]
        self.current = {
            **self.current,
            "captured_at": (previous + timedelta(seconds=3)).isoformat(),
            "ledger": [*self.current["ledger"], row],
        }
        event_id = f"sxid-{sxid}"
        self.state = {
            "workflows": [workflow],
            "incidents": [incident],
            "fabric_events": [
                {
                    "event_id": event_id,
                    "sxid": sxid,
                    "classification": "FATAL" if sxid == 10003 else "NON_FATAL",
                    "classification_source": "NVIDIA_FABRIC_MANAGER_CATALOG",
                }
            ],
            "decisions": [
                {
                    "event_id": event_id,
                    "event_type": "SXID",
                    "official_action": "RESET_ALL_GPUS_AND_NVSWITCHES",
                    "disposition": "EXECUTABLE",
                    "action": None,
                    "workflow_request_id": workflow["request_id"],
                    "incident_id": incident["incident_id"],
                }
            ],
            "commands": [],
        }

    def execute(self, command: str, *args: str, **kwargs: Any) -> dict[str, Any]:
        assert command == "reset-audit"
        return deepcopy(self.current)


def restored_node() -> dict[str, Any]:
    return {
        "uid": "node-uid",
        "ready": "True",
        "ownership_annotations": {},
        "unschedulable": False,
        "taints": [],
    }

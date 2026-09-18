from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta, timezone
from typing import Any

from gpu_fault.node_agent.ledger import canonical_digest


def reset_documents(
    operation: str = "RESET_GPU", *, node: str = "node-a"
) -> tuple[dict[str, Any], ...]:
    now = datetime.now(timezone.utc)
    inventory = [
        {"uuid": "GPU-a", "pci_bdf": "0000:59:00.0"},
        {"uuid": "GPU-b", "pci_bdf": "0000:5a:00.0"},
    ]
    targets = ["GPU-a"] if operation == "RESET_GPU" else ["GPU-a", "GPU-b"]
    before = {
        "boot_id": "boot-1",
        "captured_at": now.isoformat(),
        "gpu_inventory": inventory,
        "ledger": [],
    }
    details = {
        "reset_gpu_uuids": targets,
        "reset_attempts": 1,
        "verified_no_gpu_clients": True,
        "reset_scope": "ALL_LOCAL_GPUS_AND_NVSWITCHES",
        "inventory_verified_before": True,
        "inventory_verified_after": True,
    }
    row = {
        "command_id": "node-command",
        "attempt": 1,
        "state": "SUCCEEDED",
        "operation": operation,
        "workflow_request_id": "workflow-a",
        "incident_id": "incident-a",
        "fencing_token": 2,
        "gpu_uuids": targets,
        "parameters_digest": canonical_digest({}),
        "signature_digest_present": True,
        "started_at": (now + timedelta(seconds=1)).isoformat(),
        "completed_at": (now + timedelta(seconds=2)).isoformat(),
        "result": {
            "command_id": "node-command",
            "attempt": 1,
            "operation": operation,
            "status": "SUCCEEDED",
            "details": details,
        },
    }
    after = {
        **deepcopy(before),
        "captured_at": (now + timedelta(seconds=3)).isoformat(),
        "ledger": [row],
    }
    state = {
        "event": {"gpu_uuid": "GPU-a"},
        "incident": {"incident_id": "incident-a", "node_ids": [node]},
        "workflow": {
            "request_id": "workflow-a",
            "incident_id": "incident-a",
            "fencing_token": 2,
            "status": "SUCCEEDED",
            "official_steps": [
                {"operation": operation, "node_ids": [node], "gpu_uuids": targets}
            ],
            "step_executions": [{"operation": operation, "status": "SUCCEEDED"}],
        },
    }
    return before, after, state


class ResetAudit:
    def __init__(
        self, operation: str = "RESET_GPU", *, node: str = "hyperpod-node"
    ) -> None:
        self.before, self.after, self.state = reset_documents(operation, node=node)
        self.reads = 0
        self.node = node

    def execute(self, command: str, *args: str, **kwargs: Any) -> dict[str, Any]:
        assert command == "reset-audit"
        self.reads += 1
        return deepcopy(self.before if self.reads == 1 else self.after)

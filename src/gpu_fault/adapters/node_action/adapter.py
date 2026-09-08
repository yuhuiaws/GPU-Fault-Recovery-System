from __future__ import annotations

import json
import os
from datetime import timedelta
from typing import Any

from gpu_fault.adapters.node_action.barriers import NodeActionBarrierMixin
from gpu_fault.adapters.node_action.notifications import NodeActionNotificationMixin
from gpu_fault.adapters.node_action.step_execution import NodeActionExecutionService
from gpu_fault.adapters.node_action.transport import NodeActionTransportMixin
from gpu_fault.execution import (
    WorkflowStepContext,
    WorkflowStepOutcome,
)
from gpu_fault.fleet import (
    NODE_ACTION_KEY_VERSION_DERIVED,
    NODE_ACTION_KEY_VERSION_SHARED,
    BarrierCoordinator,
    FleetRegistry,
)
from gpu_fault.models import (
    WorkflowStepSpec,
)
from gpu_fault.node_action_keys import node_action_secrets_from_environment
from gpu_fault.notifications import (
    DcgmDiagnosticEmailBuilder,
    RestartGuardEmailBuilder,
)
from gpu_fault.operation_registry import (
    OperationAdapter,
    operations_for_adapter,
)


class NodeActionWorkflowAdapter(
    NodeActionNotificationMixin,
    NodeActionBarrierMixin,
    NodeActionTransportMixin,
):
    """Sends signed, fenced GPU actions to an explicitly mapped node."""

    OPERATIONS = operations_for_adapter(OperationAdapter.NODE_ACTION)

    def __init__(
        self,
        endpoints: dict[str, str],
        secret: str,
        *,
        node_secrets: dict[str, str] | None = None,
        node_action_key_version: int = (NODE_ACTION_KEY_VERSION_SHARED),
        owner: str = "gpu-fault-node-agent",
        sender=None,
        command_ttl: timedelta = timedelta(minutes=2),
        registry: FleetRegistry | None = None,
        barriers: BarrierCoordinator | None = None,
        store: Any | None = None,
        alert_sender=None,
        verify_max_attempts: int = 60,
        maintenance_window_seconds: int = 420,
        submit_timeout_seconds: int = 10,
        poll_timeout_seconds: int = 5,
        max_parallel_node_actions: int = 16,
        node_action_retry_limit: int = 3,
    ) -> None:
        node_secrets = dict(node_secrets or {})
        if secret and len(secret) < 32:
            raise ValueError("node action shared secret must be at least 32 characters")
        if any(
            not node_id or len(node_secret) < 32
            for node_id, node_secret in node_secrets.items()
        ):
            raise ValueError(
                "node action node secrets must map node IDs to "
                "secrets of at least 32 characters"
            )
        if not secret and not node_secrets:
            raise ValueError("node action shared secret or node secrets are required")
        if node_action_key_version not in {
            NODE_ACTION_KEY_VERSION_SHARED,
            NODE_ACTION_KEY_VERSION_DERIVED,
        }:
            raise ValueError("node action key version must be 1 or 2")
        if not endpoints and registry is None:
            raise ValueError("node action endpoints or a fleet registry are required")
        self.endpoints = endpoints
        self.secret = secret
        self.node_secrets = node_secrets
        self.node_action_key_version = node_action_key_version
        self.owner = owner
        self.sender = sender
        self.command_ttl = command_ttl
        self.registry = registry
        self.barriers = barriers
        self.store = store
        self.alert_sender = alert_sender
        self.restart_email_builder = RestartGuardEmailBuilder()
        self.dcgm_email_builder = DcgmDiagnosticEmailBuilder()
        if verify_max_attempts < 1:
            raise ValueError("verify_max_attempts must be positive")
        self.verify_max_attempts = verify_max_attempts
        if not 30 <= maintenance_window_seconds <= 3600:
            raise ValueError(
                "agent maintenance window must be between 30 and 3600 seconds"
            )
        self.maintenance_window = timedelta(seconds=maintenance_window_seconds)
        if not 1 <= submit_timeout_seconds <= 60:
            raise ValueError(
                "node action submit timeout must be between 1 and 60 seconds"
            )
        if not 1 <= poll_timeout_seconds <= 60:
            raise ValueError(
                "node action poll timeout must be between 1 and 60 seconds"
            )
        self.submit_timeout_seconds = submit_timeout_seconds
        self.poll_timeout_seconds = poll_timeout_seconds
        if not 1 <= max_parallel_node_actions <= 64:
            raise ValueError("node action parallelism must be between 1 and 64")
        self.max_parallel_node_actions = max_parallel_node_actions
        # Re-submits of one command after the agent stored a retryable
        # failure. Zero would make every retryable failure terminal on the
        # first poll; a large value only delays the step bound.
        if not 0 <= node_action_retry_limit <= 20:
            raise ValueError("node action retry limit must be between 0 and 20")
        self.node_action_retry_limit = node_action_retry_limit

    @classmethod
    def from_environment(
        cls,
        *,
        registry: FleetRegistry | None = None,
        barriers: BarrierCoordinator | None = None,
        store: Any | None = None,
        alert_sender=None,
    ) -> NodeActionWorkflowAdapter:
        try:
            endpoints = json.loads(os.getenv("GPU_FAULT_NODE_AGENT_ENDPOINTS", "{}"))
        except json.JSONDecodeError as exc:
            raise ValueError("GPU_FAULT_NODE_AGENT_ENDPOINTS must be JSON") from exc
        if not isinstance(endpoints, dict) or not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in endpoints.items()
        ):
            raise ValueError("GPU_FAULT_NODE_AGENT_ENDPOINTS must map node IDs to URLs")
        node_secrets = node_action_secrets_from_environment()
        return cls(
            endpoints,
            os.getenv("GPU_FAULT_NODE_ACTION_SECRET", ""),
            node_secrets=node_secrets,
            node_action_key_version=int(
                os.getenv("GPU_FAULT_NODE_ACTION_KEY_VERSION", "1")
            ),
            owner=os.getenv(
                "GPU_FAULT_NODE_ACTION_OWNER",
                "gpu-fault-node-agent",
            ),
            registry=registry,
            barriers=barriers,
            store=store,
            alert_sender=alert_sender,
            verify_max_attempts=int(
                os.getenv("GPU_FAULT_GPU_CLIENT_VERIFY_MAX_ATTEMPTS", "60")
            ),
            maintenance_window_seconds=int(
                os.getenv(
                    "GPU_FAULT_AGENT_MAINTENANCE_WINDOW_SECONDS",
                    "420",
                )
            ),
            submit_timeout_seconds=int(
                os.getenv(
                    "GPU_FAULT_NODE_ACTION_SUBMIT_TIMEOUT_SECONDS",
                    "10",
                )
            ),
            poll_timeout_seconds=int(
                os.getenv(
                    "GPU_FAULT_NODE_ACTION_POLL_TIMEOUT_SECONDS",
                    "5",
                )
            ),
            max_parallel_node_actions=int(
                os.getenv("GPU_FAULT_NODE_ACTION_MAX_PARALLEL", "16")
            ),
            node_action_retry_limit=int(
                os.getenv("GPU_FAULT_NODE_ACTION_RETRY_LIMIT", "3")
            ),
        )

    def supports(self, step: WorkflowStepSpec) -> bool:
        return step.execution_owner == self.owner and step.operation in self.OPERATIONS

    def execute(self, context: WorkflowStepContext) -> WorkflowStepOutcome:
        return NodeActionExecutionService(self).execute(context)

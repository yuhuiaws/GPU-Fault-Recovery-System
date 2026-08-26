from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Callable
from urllib import error as urllib_error
from urllib import parse as urllib_parse
from urllib import request as urllib_request

from gpu_fault.adapters.common import NodeActionPending
from gpu_fault.execution import (
    WorkflowStepContext,
    WorkflowStepOutcome,
)
from gpu_fault.fleet import (
    NODE_ACTION_KEY_VERSION_SHARED,
)
from gpu_fault.models import (
    WorkflowOperation,
)
from gpu_fault.node_action_keys import (
    resolve_node_action_secret,
)
from gpu_fault.node_agent import (
    NodeActionCommand,
    NodeActionExecutionState,
    NodeActionResult,
    NodeActionSubmission,
    SignedNodeAction,
    sign_node_action,
    sign_result_query,
)
from gpu_fault.transport.http_client import urlopen


class NodeActionTransportMixin:
    # Attributes supplied by the composed concrete implementation.
    endpoints: Any

    _maintenance_generations: Callable[..., Any]
    command_ttl: Any
    node_action_key_version: Any
    node_secrets: Any
    poll_timeout_seconds: Any
    registry: Any
    secret: Any
    sender: Callable[..., Any]
    submit_timeout_seconds: Any

    def knows_node(self, cluster_id: str, node_id: str) -> bool:
        """Whether this adapter can address node_id at all.

        Used to pick which provider alias to target. With a fleet
        registry the registry is the sole authority: falling back to the
        static map would reintroduce the stale-address bug this method
        exists to avoid. Without a registry the map is all there is.
        Not a readiness check -- a quiesced or draining agent is still
        addressable, and callers resolve liveness via _endpoint().
        """
        if self.registry is not None:
            try:
                self.registry.store.get_agent(cluster_id, node_id)
                return True
            except (KeyError, ValueError, TypeError):
                return False
        return node_id in self.endpoints

    def _endpoint(
        self,
        cluster_id: str,
        node_id: str,
        expected_generation: int | None,
        *,
        maintenance: bool = False,
    ) -> str | None:
        if self.registry is not None:
            if maintenance and expected_generation is not None:
                return self.registry.maintenance_endpoint(
                    cluster_id, node_id, expected_generation
                )
            endpoint, generation = self.registry.endpoint(cluster_id, node_id)
            if expected_generation is not None and generation != expected_generation:
                raise ValueError(
                    f"agent generation changed from "
                    f"{expected_generation} to {generation}"
                )
            return endpoint
        return self.endpoints.get(node_id)

    def _send_action(
        self,
        context: WorkflowStepContext,
        node_id: str,
        operation: WorkflowOperation,
        gpu_uuids: list[str],
        *,
        command_suffix: str,
        agent_generation: int | None = None,
    ) -> NodeActionResult | WorkflowStepOutcome:
        try:
            maintenance = self._maintenance_generations(context) is not None
            endpoint = self._endpoint(
                context.incident.cluster_id,
                node_id,
                agent_generation,
                maintenance=maintenance,
            )
            action_secret = self._secret_for_node(context.incident.cluster_id, node_id)
        except ValueError as exc:
            return WorkflowStepOutcome.failed(str(exc))
        if not endpoint:
            return WorkflowStepOutcome.failed(f"no node action endpoint for {node_id}")
        now = datetime.now(timezone.utc)
        command = NodeActionCommand(
            command_id=(
                f"{context.idempotency_key}/{command_suffix}"
                + (f"/agent-{agent_generation}" if agent_generation is not None else "")
            ),
            workflow_request_id=context.workflow.request_id,
            incident_id=context.incident.incident_id,
            fencing_token=context.workflow.fencing_token,
            operation=operation,
            node_id=node_id,
            agent_generation=agent_generation,
            gpu_uuids=gpu_uuids,
            parameters=context.step.parameters,
            issued_at=now,
            expires_at=now + self.command_ttl,
        )
        envelope = SignedNodeAction(
            command=command,
            signature=sign_node_action(command, action_secret),
        )
        try:
            if self.sender is not None:
                return self.sender(endpoint, envelope)
            return self._send(endpoint, envelope, secret=action_secret)
        except NodeActionPending as exc:
            return WorkflowStepOutcome.waiting(
                operation_id=context.idempotency_key,
                details={
                    "node_action_command_id": exc.command_id,
                    "node_action_state": "PENDING",
                    **exc.details,
                },
            )
        except urllib_error.HTTPError as exc:
            raw_detail = exc.read().decode(errors="replace")
            structured: dict[str, Any] = {}
            try:
                parsed = json.loads(raw_detail)
                detail = parsed.get("detail", parsed)
                if isinstance(detail, dict):
                    structured = detail
            except json.JSONDecodeError:
                pass
            return WorkflowStepOutcome.failed(
                f"node agent {node_id} rejected request: "
                f"HTTP {exc.code}: "
                + str(structured.get("message") or raw_detail or exc.reason),
                details={
                    "node_action_error_code": structured.get("code", "HTTP_REJECTION"),
                    "node_action_retryable": bool(structured.get("retryable", False)),
                    "node_action_requires_new_command": bool(
                        structured.get("requires_new_command", False)
                    ),
                    "http_status": exc.code,
                },
            )
        except (
            TimeoutError,
            urllib_error.URLError,
            OSError,
        ) as exc:
            return WorkflowStepOutcome.waiting(
                operation_id=context.idempotency_key,
                details={
                    "node_action_command_id": command.command_id,
                    "node_action_state": "TRANSPORT_RETRY",
                    "transport_error": (f"{type(exc).__name__}: {exc}"),
                },
            )
        except Exception as exc:
            return WorkflowStepOutcome.failed(
                f"node agent {node_id} request failed: {type(exc).__name__}: {exc}"
            )

    def _secret_for_node(self, cluster_id: str, node_id: str) -> str:
        key_version = self.node_action_key_version
        if self.registry is not None:
            try:
                record = self.registry.store.get_agent(cluster_id, node_id)
            except Exception as exc:
                raise ValueError(
                    f"agent key metadata is unavailable for {node_id}"
                ) from exc
            key_version = getattr(
                record,
                "node_action_key_version",
                NODE_ACTION_KEY_VERSION_SHARED,
            )
            policy = getattr(self.registry, "policy", None)
            required_key_version = getattr(
                policy,
                "required_node_action_key_version",
                None,
            )
            if required_key_version is not None and key_version != required_key_version:
                raise ValueError(f"agent key version for {node_id} is not accepted")
        return resolve_node_action_secret(
            self.secret,
            self.node_secrets,
            cluster_id,
            node_id,
            key_version,
        )

    def _send(
        self,
        endpoint: str,
        envelope: SignedNodeAction,
        *,
        secret: str,
    ) -> NodeActionResult:
        command_id = envelope.command.command_id
        # The result carries the same operational detail as the command,
        # so the read is authenticated with the same shared secret. An
        # agent that has not been rolled yet ignores the two extra query
        # parameters, so the control plane can start signing first.
        issued_at = datetime.now(timezone.utc).isoformat()
        result_url = (
            endpoint.rstrip("/")
            + "/v1/node-actions/result?"
            + urllib_parse.urlencode(
                {
                    "command_id": command_id,
                    "issued_at": issued_at,
                    "signature": sign_result_query(command_id, issued_at, secret),
                }
            )
        )
        poll = urllib_request.Request(
            result_url,
            method="GET",
        )
        try:
            with urlopen(poll, timeout=self.poll_timeout_seconds) as response:
                state = NodeActionSubmission.model_validate_json(response.read())
        except urllib_error.HTTPError as exc:
            if exc.code != 404:
                raise
            state = None
        if state is None:
            request = urllib_request.Request(
                endpoint.rstrip("/") + "/v1/node-actions/submit",
                data=envelope.model_dump_json().encode(),
                headers={
                    "Content-Type": "application/json",
                    "Idempotency-Key": command_id,
                },
                method="POST",
            )
            with urlopen(request, timeout=self.submit_timeout_seconds) as response:
                state = NodeActionSubmission.model_validate_json(response.read())
        if state.state is NodeActionExecutionState.PENDING:
            raise NodeActionPending(
                command_id,
                {
                    "node_action_endpoint": endpoint,
                },
            )
        if state.result is None:
            raise RuntimeError("node action completed without a result")
        return state.result

from __future__ import annotations

import json
import ssl
from datetime import datetime, timezone
from hashlib import sha256
from typing import Any, Callable
from urllib import error as urllib_error
from urllib import parse as urllib_parse
from urllib import request as urllib_request

from gpu_fault.adapters.common import NodeActionPending
from gpu_fault.adapters.node_action.lease_guard import lease_hold_reason
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
    NodeActionStatus,
    NodeActionSubmission,
    SignedNodeAction,
    sign_node_action,
    sign_result_query,
)
from gpu_fault.transport.http_client import urlopen

# Agent answers that mean "ask again later with the same command": the agent
# itself is unwell (5xx), or it asked for a pause (408/429). A 5xx after a
# successful operation -- the ledger write failing behind a finished reset --
# is the case that must not become a FAILED step.
TRANSIENT_HTTP_STATUSES = frozenset({408, 429})


class NodeActionTransportMixin:
    # Attributes supplied by the composed concrete implementation.
    endpoints: Any

    _maintenance_generations: Callable[..., Any]
    command_ttl: Any
    node_action_key_version: Any
    node_action_retry_limit: int
    node_secrets: Any
    poll_timeout_seconds: Any
    registry: Any
    secret: Any
    sender: Callable[..., Any]
    submit_timeout_seconds: Any

    @property
    def _ssl_context_cache(self) -> dict[tuple[str, str], ssl.SSLContext]:
        """This adapter's pinned TLS contexts, created on first use.

        Held per instance and reached through ``getattr``/``setattr`` because
        the mixin owns no ``__init__`` (every concrete adapter would have to
        remember to call it) and a class-level dict would be shared by every
        adapter in the process, including ones built for another cluster.
        """

        cache: dict[tuple[str, str], ssl.SSLContext] | None = getattr(
            self, "_node_action_ssl_contexts", None
        )
        if cache is None:
            cache = {}
            setattr(self, "_node_action_ssl_contexts", cache)
        return cache

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
        hold_reason = lease_hold_reason()
        if hold_reason is not None:
            # The executor no longer holds (or was told to give up) its lease
            # on this command. Nothing new may start; whatever the agent is
            # already doing stays in its ledger for the next lease holder.
            return WorkflowStepOutcome.waiting(
                operation_id=context.idempotency_key,
                details={
                    "node_action_state": "LEASE_LOST",
                    "node_action_not_started": True,
                    "waiting_node": node_id,
                    "reason": hold_reason,
                },
            )
        try:
            maintenance = self._maintenance_generations(context) is not None
            endpoint = self._endpoint(
                context.incident.cluster_id,
                node_id,
                agent_generation,
                maintenance=maintenance,
            )
            # One read for both the key version and the pinned certificate:
            # they cannot disagree between two reads of the same record, and
            # the second read was a control-plane round trip per node per poll.
            record = (
                self._agent_record(context.incident.cluster_id, node_id)
                if self.registry is not None
                else None
            )
            action_secret = self._secret_for_node(
                context.incident.cluster_id, node_id, record=record
            )
        except ValueError as exc:
            return WorkflowStepOutcome.failed(str(exc))
        if not endpoint:
            return WorkflowStepOutcome.failed(f"no node action endpoint for {node_id}")
        endpoint = str(endpoint)
        try:
            ssl_context = self._ssl_context(
                context.incident.cluster_id,
                node_id,
                endpoint,
                record=record,
            )
        except ValueError as exc:
            return WorkflowStepOutcome.failed(str(exc))
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
            return self._send(
                endpoint,
                envelope,
                secret=action_secret,
                ssl_context=ssl_context,
            )
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
            return self._classify_http_error(context, node_id, command, exc)
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

    @staticmethod
    def _classify_http_error(
        context: WorkflowStepContext,
        node_id: str,
        command: NodeActionCommand,
        exc: urllib_error.HTTPError,
    ) -> WorkflowStepOutcome:
        raw_detail = exc.read().decode(errors="replace")
        structured: dict[str, Any] = {}
        try:
            parsed = json.loads(raw_detail)
            detail = parsed.get("detail", parsed)
            if isinstance(detail, dict):
                structured = detail
        except json.JSONDecodeError:
            pass
        message = str(structured.get("message") or raw_detail or exc.reason)
        code = str(structured.get("code", "HTTP_REJECTION"))
        retryable = bool(structured.get("retryable", False))
        requires_new_command = bool(structured.get("requires_new_command", False))
        common = {
            "node_action_command_id": command.command_id,
            "node_action_error_code": code,
            "node_action_retryable": retryable,
            "node_action_requires_new_command": requires_new_command,
            "http_status": exc.code,
        }
        if requires_new_command:
            # STALE_AGENT_GENERATION, COMMAND_EXPIRED, STALE_FENCING_TOKEN: the
            # agent will never accept this envelope again, but the step is not
            # lost -- the next dispatch builds a fresh envelope (new issued_at
            # and signature), and the control plane decides whether the
            # workflow still holds the fence. Failing here turned a superseded
            # command into a FAILED GPU.
            return WorkflowStepOutcome.waiting(
                operation_id=context.idempotency_key,
                details={
                    **common,
                    "node_action_state": "NEW_COMMAND_REQUIRED",
                    "waiting_node": node_id,
                    "reason": f"node agent {node_id} requires a new command: "
                    f"HTTP {exc.code} {code}: {message}",
                },
            )
        if exc.code >= 500 or exc.code in TRANSIENT_HTTP_STATUSES or retryable:
            return WorkflowStepOutcome.waiting(
                operation_id=context.idempotency_key,
                details={
                    **common,
                    "node_action_state": "TRANSPORT_RETRY",
                    "node_action_transport_retry": True,
                    "waiting_node": node_id,
                    "reason": f"node agent {node_id} answered HTTP {exc.code} "
                    f"{code}: {message}",
                },
            )
        return WorkflowStepOutcome.failed(
            f"node agent {node_id} rejected request: HTTP {exc.code}: {message}",
            details=common,
        )

    def _agent_record(self, cluster_id: str, node_id: str) -> Any:
        """Read one agent record for one send.

        The key version and the pinned TLS certificate both live on this
        record, and both are needed for every send. Read separately they cost
        two control-plane round trips per node per cycle -- and a node action
        that answers WAITING is re-dispatched every poll for as long as the
        agent works, so this is the executor's steadiest avoidable load.

        Fetch failures become ``ValueError``: the caller turns that into a
        failed step, which is the fail-closed answer for "we cannot tell which
        key or certificate this agent expects".
        """

        try:
            return self.registry.store.get_agent(cluster_id, node_id)
        except Exception as exc:
            raise ValueError(
                f"agent key metadata is unavailable for {node_id}"
            ) from exc

    def _secret_for_node(
        self,
        cluster_id: str,
        node_id: str,
        *,
        record: Any = None,
    ) -> str:
        key_version = self.node_action_key_version
        if self.registry is not None:
            if record is None:
                record = self._agent_record(cluster_id, node_id)
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
        ssl_context: ssl.SSLContext | None = None,
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
            with urlopen(
                poll,
                timeout=self.poll_timeout_seconds,
                ssl_context=ssl_context,
            ) as response:
                state = NodeActionSubmission.model_validate_json(response.read())
        except urllib_error.HTTPError as exc:
            if exc.code != 404:
                raise
            state = None
        pending_details: dict[str, Any] = {"node_action_endpoint": endpoint}
        if state is not None and self._should_resubmit(state):
            # The agent's ledger holds a retryable failure and will run the
            # command again as attempt + 1 when the same envelope is
            # re-submitted; it never retries on its own. Polling alone left
            # the step spinning on that row until its bound.
            failed = state.result
            if failed is not None:
                pending_details.update(
                    {
                        "node_action_resubmitted": True,
                        "node_action_attempt": failed.attempt + 1,
                        "node_action_last_error": failed.error,
                    }
                )
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
            with urlopen(
                request,
                timeout=self.submit_timeout_seconds,
                ssl_context=ssl_context,
            ) as response:
                state = NodeActionSubmission.model_validate_json(response.read())
        if state.state is NodeActionExecutionState.PENDING:
            raise NodeActionPending(command_id, pending_details)
        if state.result is None:
            raise RuntimeError("node action completed without a result")
        return self._retry_exhausted(state.result) or state.result

    def _should_resubmit(self, state: NodeActionSubmission) -> bool:
        result = state.result
        return (
            state.state is NodeActionExecutionState.FAILED
            and result is not None
            and result.status is NodeActionStatus.FAILED
            and result.retryable
            and result.attempt <= int(self.node_action_retry_limit)
        )

    def _retry_exhausted(self, result: NodeActionResult) -> NodeActionResult | None:
        """A retryable failure past the re-submit bound becomes terminal.

        Without this the step folds every retryable row into WAITING and the
        only exit is the step bound; with it the operator sees how many times
        the agent tried and what the last attempt said.
        """

        limit = int(self.node_action_retry_limit)
        if not (
            result.status is NodeActionStatus.FAILED
            and result.retryable
            and result.attempt > limit
        ):
            return None
        return result.model_copy(
            update={
                "retryable": False,
                "error": (
                    f"node action failed after {result.attempt} attempts "
                    f"(re-submit limit {limit}): "
                    f"{result.error or 'unknown error'}"
                ),
                "details": {
                    **result.details,
                    "node_action_retry_exhausted": True,
                    "node_action_attempts": result.attempt,
                    "node_action_last_error": result.error,
                },
            }
        )

    def _ssl_context(
        self,
        cluster_id: str,
        node_id: str,
        endpoint: str,
        *,
        record: Any = None,
    ) -> ssl.SSLContext | None:
        """The pinned TLS context for one node, reused across sends.

        ``transport/http_client.py`` keys its connection pool on
        ``id(ssl_context)``, so a context built per send means a full TLS
        handshake per send and a pooled connection nothing can ever match
        again. Cached per ``(node_id, sha256(certificate))``: keyed by node
        alone the cache would keep trusting a retired certificate for the life
        of the process, which no restart-free path could recover from.

        Two threads racing the same cold key build two contexts and one wins;
        that costs one extra handshake and nothing else, so the cache stays
        lock-free.
        """

        if not endpoint.startswith("https://"):
            return None
        cache = self._ssl_context_cache
        if self.registry is None:
            key = ("", "system-trust")
            context = cache.get(key)
            if context is None:
                context = ssl.create_default_context()
                cache[key] = context
            return context
        if record is None:
            record = self._agent_record(cluster_id, node_id)
        certificate = getattr(record, "tls_certificate_pem", None)
        if not certificate:
            raise ValueError(f"agent TLS certificate is unavailable for {node_id}")
        key = (node_id, sha256(certificate.encode()).hexdigest())
        context = cache.get(key)
        if context is None:
            context = ssl.create_default_context(cadata=certificate)
            context.check_hostname = False
            # One live certificate per node: dropping the node's other entries
            # keeps the cache bounded by fleet size across rotations.
            for stale in [entry for entry in cache if entry[0] == node_id]:
                cache.pop(stale, None)
            cache[key] = context
        return context

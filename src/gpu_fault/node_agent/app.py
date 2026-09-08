from __future__ import annotations

import ipaddress
import logging
import os
import socket
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from threading import Event, RLock, Thread
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from gpu_fault.env import env_bool
from gpu_fault.env_validation import validate_gpu_fault_environment
from gpu_fault.logging_setup import configure_logging
from gpu_fault.node_agent.config import executor_from_environment
from gpu_fault.node_agent.executor import NodeActionExecutor
from gpu_fault.node_agent.heartbeat import (
    AgentHeartbeatReporter,
    heartbeat_reporter_from_environment,
)
from gpu_fault.node_agent.protocol import (
    RESULT_QUERY_MAX_SKEW_SECONDS,
    NodeActionCommand,
    NodeActionExecutionState,
    NodeActionResult,
    NodeActionStatus,
    NodeActionSubmission,
    SignedNodeAction,
    verify_result_query,
)

LOGGER = logging.getLogger(__name__)


def _validate_unsigned_result_query_migration() -> None:
    release_id = os.getenv("GPU_FAULT_RELEASE_ID", "").strip()
    migration_release = os.getenv(
        "GPU_FAULT_NODE_ACTION_RESULT_SIGNATURE_MIGRATION_RELEASE_ID",
        "",
    ).strip()
    raw_expiry = os.getenv(
        "GPU_FAULT_NODE_ACTION_RESULT_SIGNATURE_MIGRATION_EXPIRES_AT",
        "",
    ).strip()
    if not release_id or migration_release != release_id:
        raise RuntimeError(
            "unsigned result-query migration must be bound to the "
            "current GPU_FAULT_RELEASE_ID"
        )
    try:
        expires_at = datetime.fromisoformat(raw_expiry.replace("Z", "+00:00"))
    except ValueError as exc:
        raise RuntimeError(
            "unsigned result-query migration requires an ISO-8601 expiry"
        ) from exc
    if expires_at.tzinfo is None:
        raise RuntimeError("unsigned result-query migration expiry must include UTC")
    now = datetime.now(timezone.utc)
    expires_at = expires_at.astimezone(timezone.utc)
    if expires_at <= now:
        raise RuntimeError("unsigned result-query migration has expired")
    if expires_at > now + timedelta(days=14):
        raise RuntimeError("unsigned result-query migration cannot exceed 14 days")


def _result_query_signature_required() -> bool:
    required = env_bool("GPU_FAULT_NODE_ACTION_RESULT_SIGNATURE_REQUIRED", True)
    if not required:
        _validate_unsigned_result_query_migration()
    return required


def _node_action_rejection(
    error: ValueError,
) -> tuple[int, dict[str, Any]]:
    message = str(error)
    normalized = message.lower()
    if "invalid node action signature" in normalized:
        status, code, retryable, new_command = (
            401,
            "INVALID_SIGNATURE",
            False,
            False,
        )
    elif "targets a different node" in normalized:
        status, code, retryable, new_command = (
            422,
            "TARGET_NODE_MISMATCH",
            False,
            False,
        )
    elif "generation is not known yet" in normalized:
        # The agent has not completed a heartbeat since it started; the
        # command may be perfectly valid, so it is retryable as-is.
        status, code, retryable, new_command = (
            409,
            "AGENT_GENERATION_UNKNOWN",
            True,
            False,
        )
    elif "different agent generation" in normalized:
        status, code, retryable, new_command = (
            409,
            "STALE_AGENT_GENERATION",
            True,
            True,
        )
    elif "is not allowed" in normalized:
        status, code, retryable, new_command = (
            403,
            "OPERATION_NOT_ALLOWED",
            False,
            False,
        )
    elif "issued_at is in the future" in normalized:
        status, code, retryable, new_command = (
            422,
            "INVALID_ISSUED_AT",
            False,
            True,
        )
    elif "command has expired" in normalized:
        status, code, retryable, new_command = (
            410,
            "COMMAND_EXPIRED",
            True,
            True,
        )
    elif "ttl exceeds" in normalized:
        status, code, retryable, new_command = (
            422,
            "INVALID_TTL",
            False,
            True,
        )
    elif "stale node action fencing token" in normalized:
        status, code, retryable, new_command = (
            409,
            "STALE_FENCING_TOKEN",
            True,
            True,
        )
    elif "command_id reused" in normalized:
        # The id already names a different command, so retrying this body can
        # only collide again; the control plane has to mint a new command_id.
        status, code, retryable, new_command = (
            409,
            "COMMAND_ID_REUSED",
            False,
            True,
        )
    else:
        status, code, retryable, new_command = (
            409,
            "ACTION_CONFLICT",
            False,
            False,
        )
    return (
        status,
        {
            "code": code,
            "message": message,
            "retryable": retryable,
            "requires_new_command": new_command,
        },
    )


def _reconcile_quiesce_after_boot(agent: NodeActionExecutor) -> None:
    """Undo a quiesce that a reboot interrupted before serving any command.

    The fail-safe timer is transient and dies with the boot; the state file does
    not. Left alone it blocks the incident's next quiesce and would let a
    RESET_GPU pass ``assert_quiesced`` on a boot that never quiesced.
    """

    manager = getattr(agent, "quiesce_manager", None)
    reconcile = getattr(manager, "reconcile_after_boot", None)
    if reconcile is None:
        return
    try:
        report = reconcile()
    except Exception:  # noqa: BLE001 - startup must still serve; the file stays
        LOGGER.exception("quiesce state reconcile after boot failed")
        return
    if report["restored"] or report["failed"]:
        LOGGER.warning(
            "quiesce state reconciled after boot %s: restored=%s failed=%s kept=%s",
            report["boot_id"],
            [item.get("incident_id") for item in report["restored"]],
            [item.get("incident_id") for item in report["failed"]],
            [item.get("incident_id") for item in report["kept"]],
        )


def _reconciled_agent(executor: NodeActionExecutor | None) -> NodeActionExecutor:
    agent = executor or executor_from_environment()
    _reconcile_quiesce_after_boot(agent)
    return agent


def create_node_agent_app(
    executor: NodeActionExecutor | None = None,
    heartbeat_reporter: AgentHeartbeatReporter | None = None,
) -> FastAPI:
    agent = _reconciled_agent(executor)
    reporter = (
        heartbeat_reporter
        if heartbeat_reporter is not None
        else heartbeat_reporter_from_environment(agent)
    )
    action_pool = ThreadPoolExecutor(
        max_workers=int(os.getenv("GPU_FAULT_NODE_ACTION_WORKERS", "4")),
        thread_name_prefix="gpu-fault-node-action",
    )
    action_futures: dict[str, Any] = {}
    action_commands: dict[str, NodeActionCommand] = {}
    action_lock = RLock()
    require_signed_result_query = _result_query_signature_required()
    result_query_max_skew_seconds = int(
        os.getenv(
            "GPU_FAULT_NODE_ACTION_RESULT_SIGNATURE_SKEW_SECONDS",
            str(RESULT_QUERY_MAX_SKEW_SECONDS),
        )
    )
    if not require_signed_result_query:
        LOGGER.warning(
            "node action result queries are unauthenticated: anything "
            "that can reach this port can read every recovery action's "
            "result; unset "
            "GPU_FAULT_NODE_ACTION_RESULT_SIGNATURE_REQUIRED once the "
            "control plane signs its polls"
        )

    def finished_result(
        command_id: str,
        command: NodeActionCommand,
        future: Any,
    ) -> NodeActionResult | None:
        """The outcome of a finished pooled execute, or None if it was rejected.

        A ``ValueError`` out of ``execute`` is a rejection -- expired while
        queued, stale fencing token, command_id reused -- not an outcome. The
        command_id is deterministic, so persisting it as a permanent FAILED row
        answered every later, freshly signed envelope with that stale row. It
        is dropped instead: the next poll 404s and the transport resubmits.
        """

        try:
            completed: NodeActionResult = future.result()
            return completed
        except ValueError as exc:
            LOGGER.warning(
                "node action rejected inside the action pool command_id=%s "
                "operation=%s reason=%s",
                command_id,
                command.operation.value,
                exc,
            )
            return None
        except Exception as exc:
            # Not the action failing -- ``execute`` reports that as a FAILED
            # result -- but the pool wrapper around it, most often the marker
            # write itself. The answer stays retryable: the ledger now refuses
            # to re-run an attempt that was already dispatched, so a resubmit
            # can only read the recorded outcome, never repeat the action.
            row = agent.ledger.latest_row(command_id)
            if row is None:
                attempt = 1
            elif row[2] is None:
                # The marker of this attempt is already on disk; reusing its
                # number keeps one row per attempt instead of writing a second
                # row for the attempt that is being reported.
                attempt = row[1]
            else:
                attempt = row[1] + 1
            result = NodeActionResult(
                command_id=command_id,
                operation=command.operation,
                status=NodeActionStatus.FAILED,
                error=f"{type(exc).__name__}: {exc}",
                retryable=True,
                attempt=attempt,
            )
            try:
                agent.ledger.save(result)
            except Exception:
                LOGGER.exception(
                    "failed to persist asynchronous node action failure command_id=%s",
                    command_id,
                )
            return result

    def drop_future(command_id: str, future: Any = None) -> None:
        """Forget one command's future, unless a newer attempt replaced it.

        A future is done a moment before its callback runs, so a resubmit can
        register the next attempt in between; popping by command_id alone would
        then throw away the running attempt and answer the poll from the
        previous attempt's row.
        """

        with action_lock:
            if future is not None and action_futures.get(command_id) is not future:
                return
            action_futures.pop(command_id, None)
            action_commands.pop(command_id, None)

    def submission_state(
        command_id: str,
    ) -> NodeActionSubmission | None:
        # The future is read before the ledger: a resubmitted retryable failure
        # runs as a new attempt while the previous attempt's row is still the
        # newest one on disk, and answering with that row would look like the
        # resubmit had been refused.
        with action_lock:
            future = action_futures.get(command_id)
            command = action_commands.get(command_id)
        if future is not None and command is not None:
            if not future.done():
                return NodeActionSubmission(
                    command_id=command_id,
                    state=NodeActionExecutionState.PENDING,
                )
            result = finished_result(command_id, command, future)
            drop_future(command_id, future)
            if result is not None:
                return NodeActionSubmission(
                    command_id=command_id,
                    state=NodeActionExecutionState(result.status.value),
                    result=result,
                )
        result = agent.ledger.get(command_id)
        if result is None:
            return None
        drop_future(command_id, future)
        return NodeActionSubmission(
            command_id=command_id,
            state=NodeActionExecutionState(result.status.value),
            result=result,
        )

    def awaiting_retry(state: NodeActionSubmission) -> bool:
        """A retryable failure is not an answer; the resubmit runs attempt+1."""

        result = state.result
        return (
            result is not None
            and result.status is NodeActionStatus.FAILED
            and result.retryable
        )

    def finalize_action(
        command_id: str,
        command: NodeActionCommand,
        future: Any,
    ) -> None:
        try:
            finished_result(command_id, command, future)
        finally:
            drop_future(command_id, future)

    @asynccontextmanager
    async def lifespan(_):
        stop = Event()
        worker = None
        if reporter is not None:
            worker = Thread(
                target=reporter.run,
                args=(stop,),
                name="gpu-fault-agent-heartbeat",
                daemon=True,
            )
            worker.start()
        try:
            yield
        finally:
            stop.set()
            if worker is not None:
                worker.join(timeout=reporter.interval_seconds + 2)
            action_pool.shutdown(wait=False, cancel_futures=False)

    app = FastAPI(
        title="GPU Fault Node Action Agent",
        lifespan=lifespan,
    )

    @app.get("/healthz")
    async def healthz() -> JSONResponse:
        """Local health: 503 only when the ledger cannot be written.

        Heartbeat staleness and the command counters are informational --
        the control plane already fences a stale agent -- but they give an
        operator on the node a view without reaching the control plane.
        """

        payload: dict[str, Any] = {"status": "ok"}
        status_code = 200
        ledger_state: dict[str, Any] = {"writable": True}
        probe = getattr(agent.ledger, "probe_writable", None)
        if callable(probe):
            try:
                probe()
            except Exception as exc:  # noqa: BLE001 - any failure is "not writable"
                ledger_state = {
                    "writable": False,
                    "error": type(exc).__name__,
                    "reason": str(exc),
                }
                payload["status"] = "degraded"
                status_code = 503
        payload["ledger"] = ledger_state
        heartbeat_health = getattr(reporter, "health_snapshot", None)
        payload["heartbeat"] = (
            heartbeat_health() if callable(heartbeat_health) else {"configured": False}
        )
        counters = getattr(agent, "counters_snapshot", None)
        payload["counters"] = counters() if callable(counters) else {}
        return JSONResponse(status_code=status_code, content=payload)

    @app.post(
        "/v1/node-actions/submit",
        response_model=NodeActionSubmission,
    )
    async def submit_action(
        envelope: SignedNodeAction,
    ) -> NodeActionSubmission:
        try:
            command = agent.validate_submission(envelope)
        except ValueError as exc:
            status_code, detail = _node_action_rejection(exc)
            raise HTTPException(status_code=status_code, detail=detail) from exc
        existing = submission_state(command.command_id)
        if existing is not None and not awaiting_retry(existing):
            return existing
        with action_lock:
            future = action_futures.get(command.command_id)
            if future is None:
                action_commands[command.command_id] = command
                future = action_pool.submit(agent.execute, envelope)
                action_futures[command.command_id] = future
                future.add_done_callback(
                    lambda completed,
                    command_id=command.command_id,
                    submitted=command: finalize_action(command_id, submitted, completed)
                )
        pending = NodeActionSubmission(
            command_id=command.command_id,
            state=NodeActionExecutionState.PENDING,
        )
        if existing is not None:
            # A retryable failure was just handed back to the pool as the next
            # attempt; the row it left behind is not the answer to this submit.
            return pending
        return submission_state(command.command_id) or pending

    @app.get(
        "/v1/node-actions/result",
        response_model=NodeActionSubmission,
    )
    async def action_result(
        command_id: str,
        issued_at: str | None = None,
        signature: str | None = None,
    ) -> NodeActionSubmission:
        # Set to false only to poll a fleet whose control plane has not
        # been rolled yet: the signing side ships in the same wheel, so
        # roll the control plane first, then the node bundle, and this
        # never has to be touched. An unsigned query is answered either
        # way when it is false, which is the whole point -- it is a
        # migration escape hatch, not a mode.
        if require_signed_result_query or signature is not None:
            try:
                verify_result_query(
                    command_id,
                    issued_at,
                    signature,
                    agent.secret,
                    max_skew_seconds=result_query_max_skew_seconds,
                )
            except ValueError as exc:
                raise HTTPException(
                    status_code=401,
                    detail={
                        "code": "INVALID_SIGNATURE",
                        "message": str(exc),
                        "retryable": False,
                        "requires_new_command": False,
                    },
                ) from exc
        state = submission_state(command_id)
        if state is None:
            raise HTTPException(
                status_code=404,
                detail="node action command is unknown",
            )
        return state

    return app


def _default_node_agent_host() -> str:
    for name in (socket.getfqdn(), socket.gethostname()):
        try:
            addresses = socket.getaddrinfo(name, None, type=socket.SOCK_STREAM)
        except socket.gaierror:
            continue
        for _family, _kind, _proto, _canon, sockaddr in addresses:
            address = sockaddr[0]
            try:
                parsed = ipaddress.ip_address(address)
            except ValueError:
                continue
            if (
                parsed.is_private
                and not parsed.is_loopback
                and not parsed.is_unspecified
                and not parsed.is_link_local
            ):
                return address
    return "127.0.0.1"


def run() -> None:
    """Serve the node action API.

    TLS is the default requirement. A deployment that cannot provision
    node certificates must opt into cleartext explicitly; the HMAC still
    authenticates commands, but it does not hide node IDs, GPU UUIDs,
    process names or diagnostic paths from anything on the network path.
    """
    import ssl

    import uvicorn

    # uvicorn configures only its own loggers; without a root handler every
    # INFO line the executor writes for a command is discarded before journald.
    configure_logging()
    validate_gpu_fault_environment(process_name="gpu-fault-node-agent")
    host = os.getenv("GPU_FAULT_NODE_AGENT_HOST", _default_node_agent_host())
    port = int(os.getenv("GPU_FAULT_NODE_AGENT_PORT", "9099"))
    certificate = os.getenv("GPU_FAULT_NODE_AGENT_TLS_CERT", "").strip()
    private_key = os.getenv("GPU_FAULT_NODE_AGENT_TLS_KEY", "").strip()
    client_ca = os.getenv("GPU_FAULT_NODE_AGENT_TLS_CLIENT_CA", "").strip()
    allow_plaintext = env_bool("GPU_FAULT_NODE_AGENT_ALLOW_PLAINTEXT", False)
    if bool(certificate) != bool(private_key):
        raise ValueError(
            "node agent TLS needs both "
            "GPU_FAULT_NODE_AGENT_TLS_CERT and "
            "GPU_FAULT_NODE_AGENT_TLS_KEY"
        )
    if client_ca and not certificate:
        raise ValueError(
            "node agent client certificate verification requires "
            "GPU_FAULT_NODE_AGENT_TLS_CERT"
        )
    options: dict[str, Any] = {}
    if certificate:
        options["ssl_certfile"] = certificate
        options["ssl_keyfile"] = private_key
        key_password = os.getenv("GPU_FAULT_NODE_AGENT_TLS_KEY_PASSWORD", "")
        if key_password:
            options["ssl_keyfile_password"] = key_password
        if client_ca:
            options["ssl_ca_certs"] = client_ca
            options["ssl_cert_reqs"] = ssl.CERT_REQUIRED
        LOGGER.info(
            "node agent serving HTTPS on %s:%s (client certificate %s)",
            host,
            port,
            "required" if client_ca else "not requested",
        )
    else:
        if not allow_plaintext:
            raise RuntimeError(
                "node agent TLS is required; configure "
                "GPU_FAULT_NODE_AGENT_TLS_CERT and "
                "GPU_FAULT_NODE_AGENT_TLS_KEY, or explicitly set "
                "GPU_FAULT_NODE_AGENT_ALLOW_PLAINTEXT=true for a "
                "network-isolated deployment"
            )
        LOGGER.warning(
            "node agent serving plain HTTP on %s:%s: signed commands "
            "and results travel in cleartext; set "
            "GPU_FAULT_NODE_AGENT_TLS_CERT and "
            "GPU_FAULT_NODE_AGENT_TLS_KEY to serve HTTPS, and "
            "advertise the https:// endpoint",
            host,
            port,
        )
    uvicorn.run(
        create_node_agent_app(),
        host=host,
        port=port,
        **options,
    )

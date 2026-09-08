from __future__ import annotations

import hmac
import logging
import os
import sqlite3
import subprocess
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Event, RLock
from typing import Any
from typing import Callable


from gpu_fault.fleet import (
    NODE_ACTION_KEY_VERSION_DERIVED,
    NODE_ACTION_KEY_VERSION_SHARED,
)
from gpu_fault.models import WorkflowOperation
from gpu_fault.node_agent.ledger import NodeActionLedger, canonical_digest
from gpu_fault.node_agent.protocol import (
    NodeActionCommand,
    NodeActionResult,
    NodeActionStatus,
    SignedNodeAction,
    sign_node_action,
)
from gpu_fault.node_agent.quiesce import GpuServiceQuiesceManager
from gpu_fault.node_agent.operations import (
    ClientOperationsMixin,
    DiagnosticOperationsMixin,
    EfaOperationsMixin,
    FlightRecorderOperationsMixin,
    HungProcessOperationsMixin,
    HungTriageOperationsMixin,
    RemediationOperationsMixin,
    ResetOperationsMixin,
)
from gpu_fault.node_agent.operations.registry import (
    OPERATION_HANDLERS,
    operation_handler_name,
    validate_operation_handlers,
)

LOGGER = logging.getLogger(__name__)

COUNTER_NAMES = ("accepted", "completed", "failed", "rejected")

# How hard the executor tries to write a finished result before it gives up and
# closes the attempt as INTERRUPTED. The handler has already run at this point,
# so giving up quietly is the one thing that must not happen.
LEDGER_SAVE_RETRIES = 3
LEDGER_SAVE_RETRY_SECONDS = 0.2
UNPERSISTED_RESULT_ERROR = "result could not be persisted"


def _command_log_fields(command: NodeActionCommand, attempt: int) -> str:
    """Structured key=value context for one command's journald lines.

    Identifiers only: no parameters, GPU UUID lists, signatures or secrets.
    """

    return (
        f"command_id={command.command_id} "
        f"incident_id={command.incident_id} "
        f"workflow_request_id={command.workflow_request_id} "
        f"operation={command.operation.value} "
        f"node_id={command.node_id} "
        f"fencing_token={command.fencing_token} "
        f"agent_generation={command.agent_generation} "
        f"gpu_count={len(command.gpu_uuids)} "
        f"attempt={attempt}"
    )


class NodeActionExecutor(
    DiagnosticOperationsMixin,
    RemediationOperationsMixin,
    HungTriageOperationsMixin,
    HungProcessOperationsMixin,
    FlightRecorderOperationsMixin,
    EfaOperationsMixin,
    ClientOperationsMixin,
    ResetOperationsMixin,
):
    OPERATIONS = frozenset(OPERATION_HANDLERS)

    def __init__(
        self,
        *,
        secret: str,
        node_action_key_version: int = (NODE_ACTION_KEY_VERSION_SHARED),
        node_ids: set[str],
        allowed_operations: set[WorkflowOperation],
        reset_enabled: bool,
        single_gpu_reset_supported: bool = True,
        fabric_reset_enabled: bool = False,
        fabric_manager_restart_enabled: bool = False,
        service_quiesce_enabled: bool = False,
        diagnostic_output_dir: str = ("/var/lib/gpu-fault/diagnostics"),
        diagnostic_s3_uri: str | None = None,
        diagnostic_retention_seconds: int = 604800,
        diagnostic_max_archives: int = 20,
        infiniband_root: str = "/sys/class/infiniband",
        efa_driver_remediation_enabled: bool = False,
        efa_pci_devices_root: str = "/sys/bus/pci/devices",
        efa_driver_bind_path: str = "/sys/bus/pci/drivers/efa/bind",
        efa_driver_module: str = "efa",
        proc_root: str = "/proc",
        health_snapshot_request_dir: str = ("/var/lib/gpu-fault/health-snapshot"),
        expand_python_cgroup_processes: bool = True,
        python_stack_tool: str = ("/opt/gpu-fault/venv/bin/py-spy"),
        field_diagnostic_enabled: bool = False,
        field_diagnostic_command: tuple[str, ...] = (),
        field_diagnostic_sha256: str | None = None,
        memory_field_diagnostic_command: tuple[str, ...] = (),
        memory_field_diagnostic_sha256: str | None = None,
        field_diagnostic_timeout_seconds: int = 1800,
        driver_remediation_enabled: bool = False,
        driver_remediation_command: tuple[str, ...] = (),
        driver_remediation_sha256: str | None = None,
        target_driver_branch: int | None = None,
        firmware_update_enabled: bool = False,
        firmware_update_command: tuple[str, ...] = (),
        firmware_update_sha256: str | None = None,
        target_firmware_version: str | None = None,
        firmware_verify_command: tuple[str, ...] = (),
        firmware_verify_sha256: str | None = None,
        quiesce_manager: GpuServiceQuiesceManager | None = None,
        ledger: NodeActionLedger,
        runner: Callable[..., subprocess.CompletedProcess] = (subprocess.run),
        device_client_finder: (
            Callable[[set[str]], list[dict[str, str]]] | None
        ) = None,
        gpu_device_path_finder: (Callable[[], dict[str, str]] | None) = None,
        device_client_samples: int = 3,
        device_client_sample_interval_seconds: float = 2.0,
        inflight_wait_timeout_seconds: int = 2100,
        agent_generation: int | None = None,
        now: Callable[[], datetime] = (lambda: datetime.now(timezone.utc)),
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if len(secret) < 32:
            raise ValueError("node action shared secret must be at least 32 characters")
        if node_action_key_version not in {
            NODE_ACTION_KEY_VERSION_SHARED,
            NODE_ACTION_KEY_VERSION_DERIVED,
        }:
            raise ValueError("node action key version must be 1 or 2")
        if not 1 <= device_client_samples <= 10:
            raise ValueError("device client samples must be between 1 and 10")
        if not 0 <= device_client_sample_interval_seconds <= 10:
            raise ValueError(
                "device client sample interval must be between 0 and 10 seconds"
            )
        if not 10 <= inflight_wait_timeout_seconds <= 7200:
            raise ValueError(
                "in-flight wait timeout must be between 10 and 7200 seconds"
            )
        unsupported = allowed_operations - self.OPERATIONS
        if unsupported:
            raise ValueError(
                "unsupported node action operations: "
                + ", ".join(sorted(item.value for item in unsupported))
            )
        self.secret = secret
        self.node_action_key_version = node_action_key_version
        self.node_ids = node_ids
        self.allowed_operations = allowed_operations
        self.reset_enabled = reset_enabled
        self.single_gpu_reset_supported = single_gpu_reset_supported
        self.fabric_reset_enabled = fabric_reset_enabled
        self.fabric_manager_restart_enabled = fabric_manager_restart_enabled
        self.service_quiesce_enabled = service_quiesce_enabled
        self.diagnostic_output_dir = Path(diagnostic_output_dir)
        self.diagnostic_s3_uri = (
            diagnostic_s3_uri.rstrip("/") if diagnostic_s3_uri else None
        )
        if diagnostic_retention_seconds < 3600:
            raise ValueError("diagnostic retention must be at least one hour")
        if diagnostic_max_archives < 1:
            raise ValueError("diagnostic max archives must be positive")
        self.diagnostic_retention_seconds = diagnostic_retention_seconds
        self.diagnostic_max_archives = diagnostic_max_archives
        self.infiniband_root = Path(infiniband_root)
        self.efa_driver_remediation_enabled = efa_driver_remediation_enabled
        self.efa_pci_devices_root = Path(efa_pci_devices_root)
        self.efa_driver_bind_path = Path(efa_driver_bind_path)
        self.efa_driver_module = efa_driver_module
        self.proc_root = Path(proc_root)
        self.health_snapshot_request_dir = Path(health_snapshot_request_dir)
        self.expand_python_cgroup_processes = expand_python_cgroup_processes
        if python_stack_tool and not os.path.isabs(python_stack_tool):
            raise ValueError("Python stack tool path must be absolute")
        self.python_stack_tool = python_stack_tool
        self.field_diagnostic_enabled = field_diagnostic_enabled
        self.field_diagnostic_command = field_diagnostic_command
        self.field_diagnostic_sha256 = field_diagnostic_sha256
        self.memory_field_diagnostic_command = memory_field_diagnostic_command
        self.memory_field_diagnostic_sha256 = memory_field_diagnostic_sha256
        if not 60 <= field_diagnostic_timeout_seconds <= 7200:
            raise ValueError(
                "Field Diagnostic timeout must be between 60 and 7200 seconds"
            )
        self.field_diagnostic_timeout_seconds = field_diagnostic_timeout_seconds
        self.driver_remediation_enabled = driver_remediation_enabled
        self.driver_remediation_command = driver_remediation_command
        self.driver_remediation_sha256 = driver_remediation_sha256
        self.target_driver_branch = target_driver_branch
        self.firmware_update_enabled = firmware_update_enabled
        self.firmware_update_command = firmware_update_command
        self.firmware_update_sha256 = firmware_update_sha256
        self.target_firmware_version = target_firmware_version
        self.firmware_verify_command = firmware_verify_command
        self.firmware_verify_sha256 = firmware_verify_sha256
        self.ledger = ledger
        self._inflight_lock = RLock()
        self._inflight: dict[str, Event] = {}
        self._counter_lock = RLock()
        self._counters: dict[str, int] = dict.fromkeys(COUNTER_NAMES, 0)
        self.runner = runner
        self.device_client_finder = device_client_finder or self._device_clients
        self.gpu_device_path_finder = gpu_device_path_finder or self._gpu_device_paths
        self.device_client_samples = device_client_samples
        self.device_client_sample_interval_seconds = (
            device_client_sample_interval_seconds
        )
        self.inflight_wait_timeout_seconds = inflight_wait_timeout_seconds
        self.agent_generation = agent_generation
        self.now = now
        self.sleep = sleep
        self.quiesce_manager = quiesce_manager
        if service_quiesce_enabled and quiesce_manager is None:
            raise ValueError("service quiesce requires a quiesce manager")
        if field_diagnostic_enabled:
            self._validate_field_diagnostic_config(
                field_diagnostic_command,
                field_diagnostic_sha256,
            )
            if memory_field_diagnostic_command:
                self._validate_field_diagnostic_config(
                    memory_field_diagnostic_command,
                    memory_field_diagnostic_sha256,
                )
        if driver_remediation_enabled:
            self._validate_remediation_config(
                "driver",
                driver_remediation_command,
                driver_remediation_sha256,
                str(target_driver_branch or ""),
            )
        if firmware_update_enabled:
            self._validate_remediation_config(
                "firmware",
                firmware_update_command,
                firmware_update_sha256,
                target_firmware_version or "",
            )
            if not firmware_verify_command:
                raise ValueError("firmware update requires a verification command")
            self._validate_remediation_config(
                "firmware verification",
                firmware_verify_command,
                firmware_verify_sha256,
                target_firmware_version or "",
            )

    def set_agent_generation(self, generation: int) -> None:
        self.agent_generation = generation

    def _count(self, name: str) -> None:
        with self._counter_lock:
            self._counters[name] += 1

    def counters_snapshot(self) -> dict[str, int]:
        """Commands accepted / completed / failed / rejected since start."""

        with self._counter_lock:
            return dict(self._counters)

    def validate_submission(self, envelope: SignedNodeAction) -> NodeActionCommand:
        try:
            return self._validate_submission(envelope)
        except ValueError as exc:
            self._count("rejected")
            LOGGER.warning(
                "node action rejected %s reason=%s",
                _command_log_fields(envelope.command, 0),
                exc,
            )
            raise

    def _validate_submission(self, envelope: SignedNodeAction) -> NodeActionCommand:
        command = envelope.command
        expected = sign_node_action(command, self.secret)
        if not hmac.compare_digest(expected, envelope.signature):
            raise ValueError("invalid node action signature")
        if command.node_id not in self.node_ids:
            raise ValueError("node action targets a different node")
        if command.agent_generation is not None:
            if self.agent_generation is None:
                # No heartbeat has succeeded yet, so this agent does not know
                # its own generation. The command may well be addressed to it;
                # answer "not yet" (retryable, same command) rather than
                # "wrong agent" (which asks the control plane for a new one).
                raise ValueError("node action agent generation is not known yet")
            if command.agent_generation != self.agent_generation:
                raise ValueError("node action targets a different agent generation")
        if command.operation not in self.allowed_operations:
            raise ValueError(f"operation {command.operation.value} is not allowed")
        now = self.now()
        if command.issued_at > now + timedelta(seconds=30):
            raise ValueError("node action issued_at is in the future")
        if command.expires_at <= now:
            raise ValueError("node action command has expired")
        if command.expires_at - command.issued_at > timedelta(minutes=5):
            raise ValueError("node action TTL exceeds five minutes")
        self._reject_command_id_reuse(command)
        return command

    def _reject_command_id_reuse(self, command: NodeActionCommand) -> None:
        """One command_id is one command; a different body is not a replay.

        Replay protection compared the command_id alone, so a validly signed
        command with the same id and different targets was answered with the
        first command's result -- a SUCCEEDED reset reported for GPUs that
        were never reset. The ledger has kept the operation, the GPU UUIDs and
        a digest of the parameters per attempt all along; this compares them.
        Columns a row does not carry (rows written before this schema, or by
        ``save`` alone) are not evidence of a mismatch.
        """

        history = self.ledger.attempt_history(command.command_id)
        if not history:
            return
        row = history[-1]
        recorded_uuids = row.get("gpu_uuids")
        recorded_digest = row.get("parameters_digest")
        recorded_operation = row.get("operation")
        reused = (
            (
                recorded_operation is not None
                and recorded_operation != command.operation.value
            )
            or (
                recorded_uuids is not None
                and sorted(recorded_uuids) != sorted(command.gpu_uuids)
            )
            or (
                recorded_digest is not None
                and recorded_digest != canonical_digest(command.parameters)
            )
        )
        if reused:
            raise ValueError("node action command_id reused for a different command")

    def _replay_or_attempt(
        self, command: NodeActionCommand
    ) -> tuple[NodeActionResult | None, int]:
        """Answer a replay from the ledger, or say which attempt number is next.

        An IN_PROGRESS row is not "no result": the handler for that attempt has
        already been dispatched. ``ledger.get`` hid such a row behind ``None``,
        so a resubmit took attempt 1 again and ran the destructive handler a
        second time. With no in-flight Event for the command there is no thread
        left to write that result either, so the attempt is closed as
        INTERRUPTED and returned -- manual confirmation, never a re-run.
        """

        row = self.ledger.latest_row(command.command_id)
        if row is None:
            return None, 1
        state, attempt, result = row
        if result is None:
            with self._inflight_lock:
                if command.command_id in self._inflight:
                    # A live thread owns this attempt; the caller waits on it.
                    return None, attempt
                result = self._close_dispatched_attempt(command, attempt)
            LOGGER.error(
                "node action attempt was closed without a persisted result %s state=%s",
                _command_log_fields(command, attempt),
                state,
            )
        if not result.retryable:
            return result, attempt
        return None, result.attempt + 1

    def _close_dispatched_attempt(
        self, command: NodeActionCommand, attempt: int
    ) -> NodeActionResult:
        """Close a dispatched attempt that has no result, without ever raising.

        Callers reach this from the submit path, where an exception escapes into
        the pool wrapper -- and that wrapper used to answer with a *retryable*
        failure, which asks for attempt+1 and re-runs the destructive handler.
        A ledger that refuses both the UPDATE and the read still has to fail
        closed, so the last resort is an unwritten INTERRUPTED answer: manual
        confirmation, never a repeat.
        """

        closed: NodeActionResult | None = None
        try:
            closed = self.ledger.mark_interrupted(
                command.command_id, attempt, UNPERSISTED_RESULT_ERROR
            )
            if closed is None:
                # The row was rewritten while we were looking at it.
                closed = self.ledger.get(command.command_id)
        except Exception:  # noqa: BLE001 - a broken ledger must not re-run actions
            LOGGER.exception(
                "node action attempt could not be closed in the ledger %s",
                _command_log_fields(command, attempt),
            )
        if closed is not None:
            return closed
        return NodeActionResult(
            command_id=command.command_id,
            operation=command.operation,
            status=NodeActionStatus.INTERRUPTED,
            error=UNPERSISTED_RESULT_ERROR,
            retryable=False,
            attempt=attempt,
        )

    def execute(self, envelope: SignedNodeAction) -> NodeActionResult:
        command = self.validate_submission(envelope)
        replayed, attempt = self._replay_or_attempt(command)
        if replayed is not None:
            return replayed
        if not self.ledger.accept_fencing(command.incident_id, command.fencing_token):
            raise ValueError("stale node action fencing token")
        with self._inflight_lock:
            inflight = self._inflight.get(command.command_id)
            if inflight is None:
                replayed, attempt = self._replay_or_attempt(command)
                if replayed is not None:
                    return replayed
                inflight = Event()
                self._inflight[command.command_id] = inflight
                owns_execution = True
            else:
                owns_execution = False
        if not owns_execution:
            if not inflight.wait(timeout=self.inflight_wait_timeout_seconds):
                raise RuntimeError("timed out waiting for in-flight node action")
            existing = self.ledger.get(command.command_id)
            if existing is None:
                raise RuntimeError(
                    "in-flight node action completed without a ledger result"
                )
            return existing

        fields = _command_log_fields(command, attempt)
        self._count("accepted")
        LOGGER.info("node action accepted %s", fields)
        try:
            self.ledger.mark_in_progress(command, attempt, signature=envelope.signature)
        except Exception:
            # Nothing has been dispatched, but the Event is already registered.
            # Leaving it unset turns every later submit for this command_id into
            # a waiter that blocks for inflight_wait_timeout_seconds, so release
            # it before the failure travels back to the caller.
            with self._inflight_lock:
                self._inflight.pop(command.command_id, None)
                inflight.set()
            raise
        LOGGER.info("node action started %s", fields)
        started = time.monotonic()
        exit_code: int | None = None
        try:
            handler = getattr(self, operation_handler_name(command.operation))
            details = handler(command)
            result = NodeActionResult(
                command_id=command.command_id,
                operation=command.operation,
                status=NodeActionStatus.SUCCEEDED,
                details=details,
                attempt=attempt,
            )
            self._count("completed")
            LOGGER.info(
                "node action completed %s status=SUCCEEDED duration_ms=%d",
                fields,
                int((time.monotonic() - started) * 1000),
            )
        except Exception as exc:
            returncode = getattr(exc, "returncode", None)
            exit_code = returncode if isinstance(returncode, int) else None
            result = NodeActionResult(
                command_id=command.command_id,
                operation=command.operation,
                status=NodeActionStatus.FAILED,
                error=f"{type(exc).__name__}: {exc}",
                retryable=self._retryable_action_error(exc),
                attempt=attempt,
            )
            self._count("failed")
            LOGGER.info(
                "node action failed %s status=FAILED error_class=%s "
                "retryable=%s exit_code=%s duration_ms=%d",
                fields,
                type(exc).__name__,
                result.retryable,
                exit_code,
                int((time.monotonic() - started) * 1000),
            )
        try:
            result = self._persist_result(result, exit_code=exit_code, fields=fields)
        finally:
            with self._inflight_lock:
                self._inflight.pop(command.command_id, None)
                inflight.set()
        return result

    def _persist_result(
        self,
        result: NodeActionResult,
        *,
        exit_code: int | None,
        fields: str,
    ) -> NodeActionResult:
        """Write one finished attempt, and close it if the write cannot happen.

        The handler has run by now, so raising here would leave the IN_PROGRESS
        marker on disk for a resubmit to overwrite -- and the destructive action
        would run again. A full or locked ledger is often momentary, so the
        write is retried; if it still fails the attempt is closed as INTERRUPTED
        (non-retryable, manual confirmation) and that marker is what the caller
        and the poll see.
        """

        failure: Exception | None = None
        for remaining in range(LEDGER_SAVE_RETRIES, -1, -1):
            try:
                self.ledger.save(result, exit_code=exit_code)
                return result
            except Exception as exc:  # noqa: BLE001 - any write failure is the same
                failure = exc
                if remaining:
                    self.sleep(LEDGER_SAVE_RETRY_SECONDS)
        error = f"{UNPERSISTED_RESULT_ERROR}: {type(failure).__name__}: {failure}"
        LOGGER.error(
            "node action result could not be written to the ledger %s "
            "status=%s error_class=%s",
            fields,
            result.status.value,
            type(failure).__name__,
        )
        unpersisted = result.model_copy(
            update={
                "status": NodeActionStatus.INTERRUPTED,
                "error": error,
                "retryable": False,
            }
        )
        try:
            self.ledger.mark_interrupted(result.command_id, result.attempt, error)
        except Exception:
            # Even the marker could not be written, so nothing on disk shows
            # this attempt finished and the poll answers 404. The IN_PROGRESS
            # row is what protects the node: a resubmit closes it as INTERRUPTED
            # instead of running the action again.
            LOGGER.exception(
                "node action interrupted marker could not be written %s", fields
            )
        return unpersisted

    def _execute_quiesce(self, command: NodeActionCommand) -> dict[str, Any]:
        if not self.reset_enabled:
            raise RuntimeError("GPU reset is disabled by node configuration")
        if not self.service_quiesce_enabled:
            raise RuntimeError("GPU service quiesce is disabled by node configuration")
        quiesce_manager = self.quiesce_manager
        if quiesce_manager is None:
            raise RuntimeError("GPU service quiesce manager is unavailable")
        target_device_paths, workload_cgroup_paths = self._quiesce_scope(command)
        return quiesce_manager.quiesce(
            incident_id=command.incident_id,
            workflow_request_id=command.workflow_request_id,
            target_device_paths=target_device_paths,
            workload_cgroup_paths=workload_cgroup_paths,
        )

    def _execute_verify_no_clients(self, command: NodeActionCommand) -> dict[str, Any]:
        return self._verify_no_clients(
            command.gpu_uuids,
            include_device_clients=not bool(
                command.parameters.get("compute_clients_only", False)
            ),
        )

    def _execute_health_snapshot(self, _command: NodeActionCommand) -> dict[str, Any]:
        return self._trigger_health_snapshot()

    def _execute_reset_gpu(self, command: NodeActionCommand) -> dict[str, Any]:
        if self.service_quiesce_enabled:
            quiesce_manager = self.quiesce_manager
            if quiesce_manager is None:
                raise RuntimeError("GPU service quiesce manager is unavailable")
            quiesce_manager.assert_quiesced(incident_id=command.incident_id)
        return self._reset_gpu(command.gpu_uuids)

    def _execute_reset_all(self, command: NodeActionCommand) -> dict[str, Any]:
        if not self.service_quiesce_enabled:
            raise RuntimeError("full fabric reset requires GPU service quiesce")
        quiesce_manager = self.quiesce_manager
        if quiesce_manager is None:
            raise RuntimeError("GPU service quiesce manager is unavailable")
        quiesce_manager.assert_quiesced(incident_id=command.incident_id)
        return self._reset_all_gpus_nvswitches(command.gpu_uuids)

    def _execute_restore(self, command: NodeActionCommand) -> dict[str, Any]:
        if not self.service_quiesce_enabled:
            raise RuntimeError("GPU service restore is disabled by node configuration")
        quiesce_manager = self.quiesce_manager
        if quiesce_manager is None:
            raise RuntimeError("GPU service quiesce manager is unavailable")
        return quiesce_manager.restore(incident_id=command.incident_id)

    @staticmethod
    def _retryable_action_error(error: Exception) -> bool:
        return isinstance(
            error,
            (
                OSError,
                TimeoutError,
                subprocess.TimeoutExpired,
                sqlite3.OperationalError,
            ),
        )


validate_operation_handlers(NodeActionExecutor)

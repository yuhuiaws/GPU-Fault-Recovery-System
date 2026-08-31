from __future__ import annotations

import hmac
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
from gpu_fault.node_agent.ledger import NodeActionLedger
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

    def validate_submission(self, envelope: SignedNodeAction) -> NodeActionCommand:
        command = envelope.command
        expected = sign_node_action(command, self.secret)
        if not hmac.compare_digest(expected, envelope.signature):
            raise ValueError("invalid node action signature")
        if command.node_id not in self.node_ids:
            raise ValueError("node action targets a different node")
        if (
            command.agent_generation is not None
            and command.agent_generation != self.agent_generation
        ):
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
        if self.ledger.get(command.command_id) is not None:
            return command
        return command

    def execute(self, envelope: SignedNodeAction) -> NodeActionResult:
        command = self.validate_submission(envelope)
        existing = self.ledger.get(command.command_id)
        if existing is not None:
            if not existing.retryable:
                return existing
            attempt = existing.attempt + 1
        else:
            attempt = 1
        if not self.ledger.accept_fencing(command.incident_id, command.fencing_token):
            raise ValueError("stale node action fencing token")
        with self._inflight_lock:
            inflight = self._inflight.get(command.command_id)
            if inflight is None:
                completed = self.ledger.get(command.command_id)
                if completed is not None and not completed.retryable:
                    return completed
                if completed is not None:
                    attempt = completed.attempt + 1
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

        self.ledger.mark_in_progress(command, attempt)
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
        except Exception as exc:
            result = NodeActionResult(
                command_id=command.command_id,
                operation=command.operation,
                status=NodeActionStatus.FAILED,
                error=f"{type(exc).__name__}: {exc}",
                retryable=self._retryable_action_error(exc),
                attempt=attempt,
            )
        try:
            self.ledger.save(result)
        finally:
            with self._inflight_lock:
                self._inflight.pop(command.command_id, None)
                inflight.set()
        return result

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

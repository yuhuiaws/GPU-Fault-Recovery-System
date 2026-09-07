from __future__ import annotations

import hashlib
import json
import os
import shlex
import socket
import tempfile
from pathlib import Path

from gpu_fault.env import env_bool
from gpu_fault.models import WorkflowOperation
from gpu_fault.node_agent.common import (
    DEFAULT_DEVICE_SWEEP_PROCESSES,
    DEFAULT_QUIESCE_CONTAINERS,
    DEFAULT_QUIESCE_PROCESSES,
    DEFAULT_QUIESCE_SERVICES,
)

from gpu_fault.node_agent.executor import NodeActionExecutor
from gpu_fault.node_agent.ledger import NodeActionLedger
from gpu_fault.node_agent.quiesce import GpuServiceQuiesceManager
from gpu_fault.operation_registry import HOST_PROC_ROOT_OPERATIONS


def agent_config_payload(
    executor: NodeActionExecutor,
    runtime_profile_version: str,
) -> dict[str, object]:
    """Build the config-digest payload for one executor.

    The control-plane pin
    (``GPU_FAULT_REQUIRED_AGENT_CONFIG_DIGEST``) has to equal the
    digest an agent actually reports, or the fleet consistency gate
    fails every node-owned step. Deployment tooling must therefore
    derive the pin by calling this function -- never by re-listing the
    fields in a second place, which is how a 3-service quiesce literal
    in ``deploy.sh`` came to be compared against the agent's real
    6-service value.
    """
    payload: dict[str, object] = {
        "node_action_key_version": (executor.node_action_key_version),
        "allowed_operations": sorted(
            item.value for item in executor.allowed_operations
        ),
        "reset_enabled": executor.reset_enabled,
        "single_gpu_reset_supported": (executor.single_gpu_reset_supported),
        "fabric_reset_enabled": executor.fabric_reset_enabled,
        "service_quiesce_enabled": (executor.service_quiesce_enabled),
        "fabric_manager_restart_enabled": (executor.fabric_manager_restart_enabled),
        "diagnostic_s3_uri": executor.diagnostic_s3_uri,
        "field_diagnostic_enabled": (executor.field_diagnostic_enabled),
        "field_diagnostic_sha256": (executor.field_diagnostic_sha256),
        "field_diagnostic_command": list(executor.field_diagnostic_command),
        "field_diagnostic_timeout_seconds": (executor.field_diagnostic_timeout_seconds),
        "memory_field_diagnostic_enabled": bool(
            executor.memory_field_diagnostic_command
        ),
        "memory_field_diagnostic_sha256": (executor.memory_field_diagnostic_sha256),
        "memory_field_diagnostic_command": list(
            executor.memory_field_diagnostic_command
        ),
        "driver_remediation_enabled": (executor.driver_remediation_enabled),
        "efa_driver_remediation_enabled": (executor.efa_driver_remediation_enabled),
        "driver_remediation_sha256": (executor.driver_remediation_sha256),
        "driver_remediation_command": list(executor.driver_remediation_command),
        "target_driver_branch": executor.target_driver_branch,
        "firmware_update_enabled": executor.firmware_update_enabled,
        "firmware_update_sha256": executor.firmware_update_sha256,
        "firmware_update_command": list(executor.firmware_update_command),
        "firmware_verify_sha256": executor.firmware_verify_sha256,
        "firmware_verify_command": list(executor.firmware_verify_command),
        "target_firmware_version": (executor.target_firmware_version),
        "runtime_profile_version": runtime_profile_version,
        "diagnostic_retention_seconds": (executor.diagnostic_retention_seconds),
        "diagnostic_max_archives": (executor.diagnostic_max_archives),
        "proc_root": str(executor.proc_root),
        "health_snapshot_request_dir": str(executor.health_snapshot_request_dir),
        "python_stack_tool": executor.python_stack_tool,
        "device_client_samples": executor.device_client_samples,
        "device_client_sample_interval_seconds": (
            executor.device_client_sample_interval_seconds
        ),
        "inflight_wait_timeout_seconds": (executor.inflight_wait_timeout_seconds),
        "ledger_retention_seconds": (executor.ledger.retention_seconds),
        "ledger_max_results": executor.ledger.max_results,
    }
    if executor.quiesce_manager is not None:
        payload["quiesce"] = {
            "services": executor.quiesce_manager.services,
            "processes": executor.quiesce_manager.processes,
            "device_sweep_processes": sorted(
                executor.quiesce_manager.device_sweep_processes
            ),
            "proc_root": str(executor.quiesce_manager.proc_root),
            "failsafe_seconds": (executor.quiesce_manager.failsafe_seconds),
            "retry_seconds": executor.quiesce_manager.retry_seconds,
            "settle_seconds": executor.quiesce_manager.settle_seconds,
            "restore_settle_seconds": (executor.quiesce_manager.restore_settle_seconds),
            "container_stop_timeout_seconds": (
                executor.quiesce_manager.container_stop_timeout_seconds
            ),
            "container_restore_timeout_seconds": (
                executor.quiesce_manager.container_restore_timeout_seconds
            ),
            "device_sweep_timeout_seconds": (
                executor.quiesce_manager.device_sweep_timeout_seconds
            ),
        }
    return payload


def agent_config_digest(
    executor: NodeActionExecutor,
    runtime_profile_version: str,
) -> str:
    """Digest of :func:`agent_config_payload`, as heartbeats report it."""
    return hashlib.sha256(
        json.dumps(
            agent_config_payload(executor, runtime_profile_version),
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def validate_host_proc_root(proc_root: str) -> str:
    root = Path(proc_root)
    try:
        pid_one_name = (root / "1" / "comm").read_text(encoding="utf-8").strip()
    except (OSError, UnicodeError) as exc:
        raise ValueError(
            f"GPU_FAULT_PROC_ROOT does not expose host pid 1: {exc}"
        ) from exc
    if not pid_one_name:
        raise ValueError("GPU_FAULT_PROC_ROOT host pid 1 comm is empty")
    try:
        configured_namespace = os.stat(root / "1" / "ns" / "pid")
        local_namespace = os.stat("/proc/1/ns/pid")
    except OSError as exc:
        raise ValueError(
            "GPU_FAULT_PROC_ROOT must expose the host pid namespace"
        ) from exc
    same_namespace = (
        configured_namespace.st_dev == local_namespace.st_dev
        and configured_namespace.st_ino == local_namespace.st_ino
    )
    if same_namespace and pid_one_name not in {"systemd", "init"}:
        raise ValueError(
            "GPU_FAULT_PROC_ROOT points at the node-agent container "
            f"pid namespace (pid 1 is {pid_one_name!r})"
        )
    return pid_one_name


def executor_from_environment(
    *, validate_runtime_paths: bool = True
) -> NodeActionExecutor:
    hostname = socket.gethostname()
    fqdn = socket.getfqdn()
    node_ids = {
        value
        for value in {
            hostname,
            fqdn,
            os.getenv("NODE_NAME"),
        }
        if value
    }
    operations = {
        WorkflowOperation(value.strip())
        for value in os.getenv(
            "GPU_FAULT_NODE_ALLOWED_OPERATIONS",
            "VERIFY_NO_GPU_CLIENTS",
        ).split(",")
        if value.strip()
    }
    proc_root = os.getenv("GPU_FAULT_PROC_ROOT", "/proc")
    if validate_runtime_paths and operations.intersection(HOST_PROC_ROOT_OPERATIONS):
        validate_host_proc_root(proc_root)
    quiesce_enabled, quiesce_manager = _quiesce_from_environment(proc_root, operations)
    return _executor_from_settings(
        node_ids, operations, proc_root, quiesce_enabled, quiesce_manager
    )


def _quiesce_from_environment(proc_root, operations):
    quiesce_enabled = env_bool("GPU_FAULT_NODE_ALLOW_SERVICE_QUIESCE", False)
    state_dir = os.getenv(
        "GPU_FAULT_QUIESCE_STATE_DIR",
        "/var/lib/gpu-fault/quiesce",
    )
    quiesce_manager = (
        GpuServiceQuiesceManager(
            state_dir=state_dir,
            services=_csv_tuple(
                os.getenv(
                    "GPU_FAULT_QUIESCE_SERVICES",
                    ",".join(DEFAULT_QUIESCE_SERVICES),
                )
            ),
            processes=_csv_tuple(
                os.getenv(
                    "GPU_FAULT_QUIESCE_PROCESSES",
                    ",".join(DEFAULT_QUIESCE_PROCESSES),
                )
            ),
            containers=_csv_tuple(
                os.getenv(
                    "GPU_FAULT_QUIESCE_CONTAINERS",
                    ",".join(DEFAULT_QUIESCE_CONTAINERS),
                )
            ),
            failsafe_seconds=int(
                os.getenv("GPU_FAULT_QUIESCE_FAILSAFE_SECONDS", "420")
            ),
            retry_seconds=int(os.getenv("GPU_FAULT_QUIESCE_RETRY_SECONDS", "60")),
            settle_seconds=float(os.getenv("GPU_FAULT_QUIESCE_SETTLE_SECONDS", "2")),
            restore_settle_seconds=float(
                os.getenv("GPU_FAULT_RESTORE_SETTLE_SECONDS", "30")
            ),
            container_stop_timeout_seconds=int(
                os.getenv("GPU_FAULT_CONTAINER_STOP_TIMEOUT_SECONDS", "30")
            ),
            container_restore_timeout_seconds=int(
                os.getenv(
                    "GPU_FAULT_CONTAINER_RESTORE_TIMEOUT_SECONDS",
                    "180",
                )
            ),
            device_sweep_timeout_seconds=int(
                os.getenv("GPU_FAULT_DEVICE_SWEEP_TIMEOUT_SECONDS", "20")
            ),
            device_sweep_processes=_csv_tuple(
                os.getenv(
                    "GPU_FAULT_DEVICE_SWEEP_PROCESSES",
                    ",".join(DEFAULT_DEVICE_SWEEP_PROCESSES),
                )
            ),
            proc_root=proc_root,
            restore_command=os.getenv(
                "GPU_FAULT_QUIESCE_RESTORE_COMMAND",
                "/opt/gpu-fault/venv/bin/gpu-fault-restore-gpu-services",
            ),
        )
        if quiesce_enabled
        else None
    )
    if (
        quiesce_manager is not None
        and operations.intersection(
            {
                WorkflowOperation.REMEDIATE_DRIVER,
                WorkflowOperation.UPDATE_SOFTWARE_FIRMWARE,
            }
        )
        and quiesce_manager.failsafe_seconds
        < int(
            os.getenv(
                "GPU_FAULT_LONG_MUTATION_MIN_FAILSAFE_SECONDS",
                "2100",
            )
        )
    ):
        raise ValueError(
            "driver/firmware remediation requires a quiesce "
            "fail-safe window of at least 2100 seconds"
        )
    return quiesce_enabled, quiesce_manager


def _executor_from_settings(
    node_ids, operations, proc_root, quiesce_enabled, quiesce_manager
):
    return NodeActionExecutor(
        secret=os.getenv("GPU_FAULT_NODE_ACTION_SECRET", ""),
        node_action_key_version=int(
            os.getenv("GPU_FAULT_NODE_ACTION_KEY_VERSION", "1")
        ),
        node_ids=node_ids,
        allowed_operations=operations,
        reset_enabled=env_bool("GPU_FAULT_NODE_ALLOW_GPU_RESET", False),
        single_gpu_reset_supported=env_bool(
            "GPU_FAULT_NODE_SINGLE_GPU_RESET_SUPPORTED", True
        ),
        fabric_reset_enabled=env_bool("GPU_FAULT_NODE_ALLOW_FABRIC_RESET", False),
        fabric_manager_restart_enabled=env_bool(
            "GPU_FAULT_NODE_ALLOW_FABRIC_MANAGER_RESTART", False
        ),
        service_quiesce_enabled=quiesce_enabled,
        diagnostic_output_dir=os.getenv(
            "GPU_FAULT_DIAGNOSTIC_OUTPUT_DIR",
            "/var/lib/gpu-fault/diagnostics",
        ),
        diagnostic_s3_uri=(os.getenv("GPU_FAULT_DIAGNOSTIC_S3_URI") or None),
        diagnostic_retention_seconds=int(
            os.getenv(
                "GPU_FAULT_DIAGNOSTIC_RETENTION_SECONDS",
                "604800",
            )
        ),
        diagnostic_max_archives=int(
            os.getenv("GPU_FAULT_DIAGNOSTIC_MAX_ARCHIVES", "20")
        ),
        infiniband_root=os.getenv(
            "GPU_FAULT_INFINIBAND_ROOT",
            "/sys/class/infiniband",
        ),
        efa_driver_remediation_enabled=env_bool(
            "GPU_FAULT_NODE_ALLOW_EFA_DRIVER_REMEDIATION", False
        ),
        efa_pci_devices_root=os.getenv(
            "GPU_FAULT_PCI_DEVICES_ROOT",
            "/sys/bus/pci/devices",
        ),
        efa_driver_bind_path=os.getenv(
            "GPU_FAULT_EFA_DRIVER_BIND_PATH",
            "/sys/bus/pci/drivers/efa/bind",
        ),
        efa_driver_module=os.getenv("GPU_FAULT_EFA_DRIVER_MODULE", "efa"),
        proc_root=proc_root,
        health_snapshot_request_dir=os.getenv(
            "GPU_FAULT_HEALTH_SNAPSHOT_REQUEST_DIR",
            "/var/lib/gpu-fault/health-snapshot",
        ),
        python_stack_tool=os.getenv(
            "GPU_FAULT_PYTHON_STACK_TOOL",
            "/opt/gpu-fault/venv/bin/py-spy",
        ),
        field_diagnostic_enabled=env_bool(
            "GPU_FAULT_NODE_ALLOW_FIELD_DIAGNOSTIC", False
        ),
        field_diagnostic_command=tuple(
            shlex.split(os.getenv("GPU_FAULT_FIELD_DIAGNOSTIC_COMMAND", ""))
        ),
        field_diagnostic_sha256=(
            os.getenv("GPU_FAULT_FIELD_DIAGNOSTIC_SHA256") or None
        ),
        memory_field_diagnostic_command=tuple(
            shlex.split(
                os.getenv(
                    "GPU_FAULT_MEMORY_FIELD_DIAGNOSTIC_COMMAND",
                    "",
                )
            )
        ),
        memory_field_diagnostic_sha256=(
            os.getenv("GPU_FAULT_MEMORY_FIELD_DIAGNOSTIC_SHA256") or None
        ),
        field_diagnostic_timeout_seconds=int(
            os.getenv(
                "GPU_FAULT_FIELD_DIAGNOSTIC_TIMEOUT_SECONDS",
                "1800",
            )
        ),
        driver_remediation_enabled=env_bool(
            "GPU_FAULT_NODE_ALLOW_DRIVER_REMEDIATION", False
        ),
        driver_remediation_command=tuple(
            shlex.split(os.getenv("GPU_FAULT_DRIVER_REMEDIATION_COMMAND", ""))
        ),
        driver_remediation_sha256=(
            os.getenv("GPU_FAULT_DRIVER_REMEDIATION_SHA256") or None
        ),
        target_driver_branch=(
            int(os.environ["GPU_FAULT_TARGET_DRIVER_BRANCH"])
            if os.getenv("GPU_FAULT_TARGET_DRIVER_BRANCH")
            else None
        ),
        firmware_update_enabled=env_bool("GPU_FAULT_NODE_ALLOW_FIRMWARE_UPDATE", False),
        firmware_update_command=tuple(
            shlex.split(os.getenv("GPU_FAULT_FIRMWARE_UPDATE_COMMAND", ""))
        ),
        firmware_update_sha256=(os.getenv("GPU_FAULT_FIRMWARE_UPDATE_SHA256") or None),
        target_firmware_version=(
            os.getenv("GPU_FAULT_TARGET_FIRMWARE_VERSION") or None
        ),
        firmware_verify_command=tuple(
            shlex.split(os.getenv("GPU_FAULT_FIRMWARE_VERIFY_COMMAND", ""))
        ),
        firmware_verify_sha256=(os.getenv("GPU_FAULT_FIRMWARE_VERIFY_SHA256") or None),
        quiesce_manager=quiesce_manager,
        device_client_samples=int(os.getenv("GPU_FAULT_DEVICE_CLIENT_SAMPLES", "3")),
        device_client_sample_interval_seconds=float(
            os.getenv(
                "GPU_FAULT_DEVICE_CLIENT_SAMPLE_INTERVAL_SECONDS",
                "2",
            )
        ),
        inflight_wait_timeout_seconds=int(
            os.getenv(
                "GPU_FAULT_NODE_INFLIGHT_WAIT_TIMEOUT_SECONDS",
                "2100",
            )
        ),
        ledger=_ledger_from_environment(),
    )


def _ledger_from_environment() -> NodeActionLedger:
    return NodeActionLedger(
        os.getenv(
            "GPU_FAULT_NODE_ACTION_DB",
            "/var/lib/gpu-fault/node-actions.db",
        ),
        retention_seconds=int(
            os.getenv("GPU_FAULT_NODE_ACTION_RETENTION_SECONDS", "604800")
        ),
        max_results=int(os.getenv("GPU_FAULT_NODE_ACTION_MAX_RESULTS", "10000")),
    )


def _csv_tuple(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def print_config_digest() -> None:
    """Print the config digest for the current environment.

    Deployment tooling calls this to derive
    ``GPU_FAULT_REQUIRED_AGENT_CONFIG_DIGEST`` from the same code the
    agent runs, instead of re-deriving it from a hand-maintained copy
    of the payload.

    The shared secret and the action ledger are required to build an
    executor but contribute nothing to the digest, so stand-ins are
    used: a caller computing a pin has no reason to hold the node
    secret or to create a ledger next to a running agent's.
    """
    names = (
        "GPU_FAULT_NODE_ACTION_SECRET",
        "GPU_FAULT_NODE_ACTION_DB",
        "GPU_FAULT_QUIESCE_STATE_DIR",
    )
    previous = {name: os.environ.get(name) for name in names}
    try:
        with tempfile.TemporaryDirectory() as scratch:
            os.environ.setdefault("GPU_FAULT_NODE_ACTION_SECRET", "0" * 32)
            os.environ["GPU_FAULT_NODE_ACTION_DB"] = str(
                Path(scratch) / "config-digest-ledger.db"
            )
            os.environ["GPU_FAULT_QUIESCE_STATE_DIR"] = str(Path(scratch) / "quiesce")
            executor = executor_from_environment(validate_runtime_paths=False)
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
    runtime_profile = os.getenv("GPU_FAULT_NODE_RUNTIME_PROFILE_VERSION", "").strip()
    if not runtime_profile:
        raise SystemExit("GPU_FAULT_NODE_RUNTIME_PROFILE_VERSION is required")
    print(agent_config_digest(executor, runtime_profile))

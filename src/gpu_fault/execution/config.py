from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Mapping
from uuid import uuid4

from gpu_fault.models import (
    WorkflowOperation,
)
from gpu_fault.execution.models import WorkflowExecutionError


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProductionExecutorConfig:
    enabled: bool
    executor_id: str
    allowed_operations: frozenset[WorkflowOperation]
    lease_duration_seconds: int = 180
    workflow_execution_timeout_seconds: int = 3600
    workflow_preemption_enabled: bool = True

    @classmethod
    def from_environment(cls) -> ProductionExecutorConfig:
        return cls.from_mapping(os.environ)

    @classmethod
    def from_mapping(cls, values: Mapping[str, str]) -> ProductionExecutorConfig:
        raw_operations = [
            value.strip()
            for value in values.get("GPU_FAULT_ALLOWED_OPERATIONS", "").split(",")
            if value.strip()
        ]
        try:
            allowed = frozenset(WorkflowOperation(value) for value in raw_operations)
        except ValueError as exc:
            raise WorkflowExecutionError(
                f"invalid GPU_FAULT_ALLOWED_OPERATIONS: {exc}"
            ) from exc
        executor_id = values.get("GPU_FAULT_EXECUTOR_ID", "").strip()
        if not executor_id:
            identity = values.get("POD_UID") or values.get("HOSTNAME") or "unknown-host"
            executor_id = f"gpu-fault-control-plane/{identity}/{uuid4()}"
        lease_duration = int(
            values.get("GPU_FAULT_WORKFLOW_LEASE_DURATION_SECONDS", "180")
        )
        if lease_duration < 30:
            raise WorkflowExecutionError(
                "workflow lease duration must be at least 30 seconds"
            )
        return cls(
            enabled=(
                values.get("GPU_FAULT_EXECUTOR_MODE", "simulation").strip().lower()
                == "active"
            ),
            executor_id=executor_id,
            allowed_operations=allowed,
            lease_duration_seconds=lease_duration,
            workflow_execution_timeout_seconds=int(
                values.get(
                    "GPU_FAULT_WORKFLOW_EXECUTION_TIMEOUT_SECONDS",
                    "3600",
                )
            ),
            workflow_preemption_enabled=(
                values.get("GPU_FAULT_ENABLE_WORKFLOW_PREEMPTION", "true")
                .strip()
                .lower()
                == "true"
            ),
        )


@dataclass(frozen=True)
class WorkflowDispatcherConfig:
    enabled: bool
    poll_interval_seconds: float = 5.0
    batch_size: int = 100
    max_workers: int = 8
    confirm_cluster_name: str | None = None

    @classmethod
    def from_environment(cls, executor_enabled: bool) -> WorkflowDispatcherConfig:
        raw_enabled = os.getenv("GPU_FAULT_ENABLE_WORKFLOW_DISPATCHER", "true")
        return cls(
            enabled=(executor_enabled and raw_enabled.strip().lower() == "true"),
            poll_interval_seconds=float(
                os.getenv("GPU_FAULT_WORKFLOW_POLL_INTERVAL_SECONDS", "5")
            ),
            batch_size=int(os.getenv("GPU_FAULT_WORKFLOW_BATCH_SIZE", "100")),
            max_workers=int(os.getenv("GPU_FAULT_WORKFLOW_DISPATCHER_WORKERS", "8")),
            confirm_cluster_name=(os.getenv("GPU_FAULT_HYPERPOD_CLUSTER") or None),
        )

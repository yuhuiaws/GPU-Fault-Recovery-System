from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Mapping
from uuid import uuid4

from gpu_fault.models import (
    WorkflowOperation,
)
from gpu_fault.execution.models import WorkflowExecutionError
from gpu_fault.execution.remediation_budget import RemediationBudgetPolicy


LOGGER = logging.getLogger(__name__)

# The operations whose waiting is owned by ``HyperPodManagedRecoveryObserver``:
# the control plane submits nothing and watches the provider replace or reboot
# the node, so the wait is as long as that provider takes.
MANAGED_RECOVERY_OPERATIONS = (
    WorkflowOperation.REPLACE_NODE,
    WorkflowOperation.RESTART_NODE,
)


def managed_recovery_timeout_seconds(values: Mapping[str, str]) -> int:
    """The window a delegated node recovery is allowed to stay unconfirmed.

    The key and the default are written out rather than held in constants because
    ``scripts/generate-env-reference.py`` reads them statically; behind a name it
    loses both the variable's type and its documented default.
    """

    return int(
        values.get("GPU_FAULT_HYPERPOD_MANAGED_RECOVERY_TIMEOUT_SECONDS", "1800")
    )


def managed_recovery_step_overrides(
    values: Mapping[str, str],
) -> dict[WorkflowOperation, int]:
    """Per-operation waiting ceilings, derived rather than configured twice.

    The delegated node operations take theirs from the managed recovery window
    itself, so the two can never drift apart. Configuring the cap separately is
    what produced the defect this exists to prevent: a 1800s cap in front of a
    2700s window silently replaced the observer's escalation notification with a
    generic step failure fifteen minutes early.
    """

    window = managed_recovery_timeout_seconds(values)
    return {operation: window for operation in MANAGED_RECOVERY_OPERATIONS}


@dataclass(frozen=True)
class ProductionExecutorConfig:
    enabled: bool
    executor_id: str
    allowed_operations: frozenset[WorkflowOperation]
    lease_duration_seconds: int = 180
    workflow_execution_timeout_seconds: int = 1800
    # A single step's own bound, held independently of whatever retry counter
    # that step's adapter keeps. Every per-operation counter
    # (``verify_max_attempts``, ``gpu_reset_commit_attempt``) is read back out
    # of the step execution record, so a counter that cannot advance -- a
    # borrowed ``step_index``, an outcome that is never persisted -- silently
    # removes the only bound the step had. Before this existed the next bound
    # below it was ``workflow_execution_timeout_seconds``, an hour away, with
    # nothing reported in between.
    #
    # The default is sized for the steps that make up almost every workflow --
    # quiesce, verify, reset, restore, restart -- which finish in seconds to a
    # few minutes. The handful of operations that legitimately wait far longer
    # get an override below rather than dragging this ceiling up for everything.
    step_waiting_timeout_seconds: int = 600
    step_waiting_warning_seconds: int = 300
    # Per-operation ceilings for the steps whose adapter owns a longer window of
    # its own. A cap below the adapter's window is worse than no cap: it fails
    # the step before the operation was designed to give up, on the generic
    # failure path, so the adapter's own timeout handling -- for managed
    # recovery, an escalation notification -- never runs.
    # Defaulted rather than empty, because a config built directly -- the
    # simulation context, a test fixture -- would otherwise cap a delegated
    # replacement at the default and lose the observer's escalation, which is the
    # exact defect the derivation above exists to prevent.
    step_waiting_timeout_overrides: Mapping[WorkflowOperation, int] = field(
        default_factory=lambda: managed_recovery_step_overrides({})
    )
    workflow_preemption_enabled: bool = True
    remediation_budget: RemediationBudgetPolicy = RemediationBudgetPolicy()

    def step_waiting_limit(self, operation: WorkflowOperation) -> int:
        """The waiting ceiling that applies to one operation."""

        return self.step_waiting_timeout_overrides.get(
            operation,
            self.step_waiting_timeout_seconds,
        )

    def step_waiting_warning_limit(self, operation: WorkflowOperation) -> int:
        """When one operation's wait stops being normal.

        Derived from the operation's own ceiling rather than shared, because a
        flat threshold in front of a raised ceiling warns about the normal path:
        a delegated node replacement legitimately waits fifteen to twenty-five
        minutes, so the default five-minute warning would fire on every healthy
        replacement and mean nothing by the time one actually hung.

        What the configured pair defines is the lead time -- how long before the
        cap an operator should hear about it -- and that is what carries over to
        an operation with a longer ceiling.
        """

        lead = self.step_waiting_timeout_seconds - self.step_waiting_warning_seconds
        return max(1, self.step_waiting_limit(operation) - lead)

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
        workflow_timeout = int(
            values.get(
                "GPU_FAULT_WORKFLOW_EXECUTION_TIMEOUT_SECONDS",
                "1800",
            )
        )
        step_timeout = int(values.get("GPU_FAULT_WORKFLOW_STEP_TIMEOUT_SECONDS", "600"))
        step_warning = int(values.get("GPU_FAULT_WORKFLOW_STEP_WARNING_SECONDS", "300"))
        if step_timeout <= 0:
            raise WorkflowExecutionError("workflow step timeout must be positive")
        if step_warning <= 0:
            raise WorkflowExecutionError(
                "workflow step warning threshold must be positive"
            )
        if step_warning > step_timeout:
            raise WorkflowExecutionError(
                "workflow step warning threshold must not exceed the step timeout"
            )
        overrides = managed_recovery_step_overrides(values)
        for operation, limit in sorted(overrides.items()):
            if limit < step_timeout:
                raise WorkflowExecutionError(
                    f"step timeout override for {operation.value} ({limit}s) is "
                    f"below the default step timeout ({step_timeout}s); an "
                    "override exists to raise a ceiling for a slower operation"
                )
        # Only the default cap. An override belongs to an operation whose adapter
        # owns the wait, and that adapter clamps its own deadline inside the
        # workflow deadline -- so such a step is bounded either way, and warning
        # about it would fire on the shipped defaults, where the managed recovery
        # window and the workflow budget are deliberately the same length.
        #
        # Not fatal, because the workflow budget is the setting an operator is
        # most likely to lower for a drill. But a default cap at or above the
        # workflow budget can never fire, and the steps it covers have no adapter
        # window behind them, which puts the deployment back in the unbounded
        # state this setting exists to end.
        if step_timeout >= workflow_timeout:
            LOGGER.warning(
                "the default step timeout (%ss) is not below the workflow "
                "execution timeout (%ss), so no step will ever be capped before "
                "the workflow deadline",
                step_timeout,
                workflow_timeout,
            )
        return cls(
            enabled=(
                values.get("GPU_FAULT_EXECUTOR_MODE", "simulation").strip().lower()
                == "active"
            ),
            executor_id=executor_id,
            allowed_operations=allowed,
            lease_duration_seconds=lease_duration,
            workflow_execution_timeout_seconds=workflow_timeout,
            step_waiting_timeout_seconds=step_timeout,
            step_waiting_warning_seconds=step_warning,
            step_waiting_timeout_overrides=overrides,
            workflow_preemption_enabled=(
                values.get("GPU_FAULT_ENABLE_WORKFLOW_PREEMPTION", "true")
                .strip()
                .lower()
                == "true"
            ),
            remediation_budget=RemediationBudgetPolicy.from_mapping(values),
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

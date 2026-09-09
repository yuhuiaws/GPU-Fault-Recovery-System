from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Iterable, Mapping
from uuid import uuid4

from gpu_fault.env import env_bool
from gpu_fault.execution.invariants import InvariantMode
from gpu_fault.execution.models import WorkflowExecutionError
from gpu_fault.execution.remediation_budget import RemediationBudgetPolicy
from gpu_fault.models import (
    WorkflowOperation,
)

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


# The operations whose node action is an install the agent runs for up to
# 1800 s (``node_agent/operations/remediation.py``, the driver and firmware
# subprocess timeout) inside a unit systemd allows 1900 s to stop
# (``deploy/systemd/gpu-fault-node-agent.service``, TimeoutStopSec). The same
# three the registry names ``generation_stable_command_id`` -- a restart under
# one of them replays from the ledger rather than reinstalling -- which is why
# the wait is long by design and ``tests/test_registry_contracts.py`` pins the
# two sets to each other.
NODE_INSTALL_OPERATIONS = (
    WorkflowOperation.REMEDIATE_DRIVER,
    WorkflowOperation.UPDATE_SOFTWARE_FIRMWARE,
    WorkflowOperation.REMEDIATE_EFA_DRIVER,
)
DEFAULT_NODE_INSTALL_STEP_TIMEOUT_SECONDS = 1900


def node_install_step_timeout_seconds(values: Mapping[str, str]) -> int:
    """How long a driver, firmware or EFA-driver install step may stay WAITING.

    The step's clock runs from its first record and a WAITING poll does not
    move it (``step_bounds.record_attempt``), so this has to clear the agent's
    own 1800 s install timeout plus its verification, or the control plane
    fails a running install on the generic "past the per-step cap" path before
    the agent's verdict can land. 1900 s matches the unit's ``TimeoutStopSec``.
    Written out for ``scripts/generate-env-reference.py``.

    The workflow execution budget (``GPU_FAULT_WORKFLOW_EXECUTION_TIMEOUT_
    SECONDS``, 1800 s from the first claim) would still cut a full-length
    install at ``1800 - delta``, so ``claim_deadlines`` floors an install
    workflow's execution deadline at this ceiling plus
    ``node_install_containment_seconds`` (F2); the lifetime still caps it.
    """

    value = int(
        values.get(
            "GPU_FAULT_WORKFLOW_INSTALL_STEP_TIMEOUT_SECONDS",
            "1900",
        )
    )
    if value <= 0:
        raise WorkflowExecutionError("workflow install step timeout must be positive")
    return value


def node_install_step_overrides(
    values: Mapping[str, str],
) -> dict[WorkflowOperation, int]:
    """One ceiling for the three installs, derived rather than configured thrice."""

    ceiling = node_install_step_timeout_seconds(values)
    return {operation: ceiling for operation in NODE_INSTALL_OPERATIONS}


DEFAULT_NODE_INSTALL_CONTAINMENT_SECONDS = 600


def node_install_containment_seconds(values: Mapping[str, str]) -> int:
    """The allowance a workflow holding an install gets for the steps around it.

    ``claim_deadlines`` floors such a workflow's execution deadline at the claim
    plus the install ceiling plus this (F2). Before the install run a quiesce
    and VERIFY_NO_GPU_CLIENTS, whose total wait at the shipped defaults is
    60 attempts x 5 s = 300 s, so one default per-step cap (600 s) holds both
    with room. Without the floor the 1800 s workflow budget cut a full-length
    install at ``1800 - delta`` and the raised step ceiling never fired. ``0``
    is legal -- a compression drill wants the floor to equal the ceiling -- so
    only a negative value is refused. Written out for
    ``scripts/generate-env-reference.py``.
    """

    value = int(values.get("GPU_FAULT_WORKFLOW_INSTALL_CONTAINMENT_SECONDS", "600"))
    if value < 0:
        raise WorkflowExecutionError(
            "workflow install containment allowance must not be negative"
        )
    return value


def _adapter_owned_step_overrides() -> dict[WorkflowOperation, int]:
    """The overrides a directly built config must not lose (see the field)."""

    return {**managed_recovery_step_overrides({}), **node_install_step_overrides({})}


# Steps whose whole purpose is to wait for a human to act and confirm
# (a mechanical inspection). Their waiting ceiling and the deadlines of the
# workflow that contains them follow the operator's clock, not the machine's.
OPERATOR_ACKNOWLEDGEMENT_OPERATIONS = frozenset({WorkflowOperation.CHECK_MECHANICALS})
DEFAULT_OPERATOR_ACKNOWLEDGEMENT_TIMEOUT_SECONDS = 86400


def operator_acknowledgement_timeout_seconds(values: Mapping[str, str]) -> int:
    """How long a step may wait for an operator's acknowledgement.

    One working day by default: long enough for a mechanical check, finite so
    the workflow cannot hang for ever. Still a gate -- it can be tightened by
    ``GPU_FAULT_OPERATOR_ACKNOWLEDGEMENT_TIMEOUT_SECONDS`` -- just one whose
    threshold matches the operation's meaning instead of the ten minutes a
    reset or a restore is allowed.
    """

    value = int(
        values.get("GPU_FAULT_OPERATOR_ACKNOWLEDGEMENT_TIMEOUT_SECONDS", "86400")
    )
    if value <= 0:
        raise WorkflowExecutionError(
            "operator acknowledgement timeout must be positive"
        )
    return value


def node_busy_wait_seconds(values: Mapping[str, str]) -> float:
    """Rule A's one window: how long a job waits on a node another remediation
    is repairing before it is stopped instead of restarted.

    Read here for both readers -- the dispatcher, which holds a job workflow
    that has not started, and the executor, which caps the ``after_incident``
    restart premise once its step is running -- so the two cannot drift apart.
    The key and the default are written out for ``generate-env-reference.py``.
    """

    value = float(values.get("GPU_FAULT_JOB_WORKFLOW_NODE_BUSY_WAIT_SECONDS", "240"))
    if value <= 0:
        raise WorkflowExecutionError("job workflow node-busy wait must be positive")
    return value


def default_step_waiting_overrides(
    values: Mapping[str, str],
) -> dict[WorkflowOperation, int]:
    """Every per-operation waiting ceiling the environment implies."""

    acknowledgement = operator_acknowledgement_timeout_seconds(values)
    return {
        **managed_recovery_step_overrides(values),
        **node_install_step_overrides(values),
        **{
            operation: acknowledgement
            for operation in OPERATOR_ACKNOWLEDGEMENT_OPERATIONS
        },
    }


def workflow_invariant_mode(values: Mapping[str, str]) -> InvariantMode:
    """How strictly the executor checks workflow invariants after each write."""

    raw = values.get("GPU_FAULT_WORKFLOW_INVARIANT_CHECKS", "log").strip().lower()
    try:
        return InvariantMode(raw)
    except ValueError as exc:
        raise WorkflowExecutionError(
            "GPU_FAULT_WORKFLOW_INVARIANT_CHECKS must be one of off, log, raise"
        ) from exc


@dataclass(frozen=True)
class ProductionExecutorConfig:
    enabled: bool
    executor_id: str
    allowed_operations: frozenset[WorkflowOperation]
    lease_duration_seconds: int = 180
    workflow_execution_timeout_seconds: int = 1800
    # A workflow that contains an operator-acknowledgement step (see
    # OPERATOR_ACKNOWLEDGEMENT_OPERATIONS) gets at least this long before its
    # execution deadline and its F-N1 lifetime, from the claim that admitted
    # the step; the step itself waits at most this long.
    operator_acknowledgement_timeout_seconds: int = (
        DEFAULT_OPERATOR_ACKNOWLEDGEMENT_TIMEOUT_SECONDS
    )
    # Review item 6: state-machine invariants are checked after every leased
    # write; ``log`` reports, ``raise`` fails the write (tests, staging).
    workflow_invariant_mode: InvariantMode = InvariantMode.LOG
    # F-N1: hard lifetime of a remediation, by kind. A job workflow (one that
    # stops and restarts a training job, usually multi-node) and a single-node
    # workflow without a job each get one budget for their whole escalation
    # chain; at the deadline the workflow fails and goes to an operator.
    job_workflow_lifetime_seconds: int = 3600
    node_workflow_lifetime_seconds: int = 3600
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
    # exact defect the derivation above exists to prevent; the same default
    # would fail a 20-minute driver install at 600 s while the agent still runs.
    step_waiting_timeout_overrides: Mapping[WorkflowOperation, int] = field(
        default_factory=_adapter_owned_step_overrides
    )
    # F2: the execution deadline of a workflow that contains one of
    # NODE_INSTALL_OPERATIONS is floored at the claim plus the install ceiling
    # plus this allowance for the containment steps around the install (see
    # ``install_execution_floor_seconds``).
    node_install_containment_seconds: int = DEFAULT_NODE_INSTALL_CONTAINMENT_SECONDS
    # Rule A: the window a job waits on a node under another remediation. The
    # same variable as ``WorkflowDispatcherConfig.node_busy_wait_seconds``;
    # here it caps the ``after_incident`` restart premise (a WAITING outcome
    # with ``reason=NODE_UNDER_REMEDIATION``) in ``bounded_waiting_outcome``,
    # since the dispatcher's hold only covers a workflow that has not started.
    # It is the cap only while the premise names no remediation, or one that
    # is terminal or unreadable: while the named remediation is still open the
    # running restart is bounded by that remediation's ``lifetime_deadline_at``
    # instead (``step_bounds._premise_hold_limit``), because a GPU reset chain
    # (420 s maintenance window + VALIDATE_GPU + RESTORE_SCHEDULING) or a
    # reboot chain (HyperPod, up to 2700 s) legitimately outlasts 240 s and
    # cutting the wait there failed the restart just before the repair landed.
    node_busy_wait_seconds: float = 240.0
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

    def install_execution_floor_seconds(self) -> int:
        """The least execution budget a workflow holding a node install gets (F2).

        The install ceiling (one knob for the three installs, so the largest is
        the value) plus the containment allowance. ``claim_deadlines`` floors
        the execution deadline here on the claim that stamps it, still capped
        by the lifetime; ``validate_timing_relationships`` refuses a node
        lifetime that holds the ceiling but not this floor.
        """

        ceiling = max(
            self.step_waiting_limit(operation) for operation in NODE_INSTALL_OPERATIONS
        )
        return ceiling + self.node_install_containment_seconds

    def execution_budget_seconds(
        self, operations: Iterable[WorkflowOperation]
    ) -> float:
        """The budget ``claim_deadlines`` stamps for a workflow of ``operations``.

        The plain execution timeout, lifted to the operator-acknowledgement
        window when the workflow waits on an inspection and to the install
        floor when it holds a node install. ``step_bounds.step_elapsed_since``
        subtracts this from the stamped deadline to recover the window start;
        subtracting the plain timeout from a floored deadline put that start in
        the future and a step's measured wait at 0 (before the clamp, -84599).
        """

        present = set(operations)
        budget = float(self.workflow_execution_timeout_seconds)
        if not present.isdisjoint(OPERATOR_ACKNOWLEDGEMENT_OPERATIONS):
            budget = max(budget, float(self.operator_acknowledgement_timeout_seconds))
        if not present.isdisjoint(NODE_INSTALL_OPERATIONS):
            budget = max(budget, float(self.install_execution_floor_seconds()))
        return budget

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
        job_lifetime = int(
            values.get("GPU_FAULT_JOB_WORKFLOW_MAX_LIFETIME_SECONDS", "3600")
        )
        node_lifetime = int(
            values.get("GPU_FAULT_NODE_WORKFLOW_MAX_LIFETIME_SECONDS", "3600")
        )
        if job_lifetime <= 0 or node_lifetime <= 0:
            raise RuntimeError("workflow lifetimes must be positive")
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
        overrides = default_step_waiting_overrides(values)
        acknowledgement = operator_acknowledgement_timeout_seconds(values)
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
            operator_acknowledgement_timeout_seconds=acknowledgement,
            job_workflow_lifetime_seconds=job_lifetime,
            node_workflow_lifetime_seconds=node_lifetime,
            step_waiting_timeout_seconds=step_timeout,
            step_waiting_warning_seconds=step_warning,
            step_waiting_timeout_overrides=overrides,
            node_install_containment_seconds=node_install_containment_seconds(values),
            node_busy_wait_seconds=node_busy_wait_seconds(values),
            workflow_preemption_enabled=env_bool(
                "GPU_FAULT_ENABLE_WORKFLOW_PREEMPTION", True, environ=values
            ),
            remediation_budget=RemediationBudgetPolicy.from_mapping(values),
            workflow_invariant_mode=workflow_invariant_mode(values),
        )


@dataclass(frozen=True)
class WorkflowDispatcherConfig:
    enabled: bool
    poll_interval_seconds: float = 5.0
    batch_size: int = 100
    max_workers: int = 8
    confirm_cluster_name: str | None = None
    # Seconds one process holds the fleet-wide dispatch lease; 0 disables the
    # lease and every process scans (the pre-F-A1 behaviour, kept for direct
    # construction in tests). Production enables it from the environment.
    dispatch_lease_seconds: float = 0.0
    # How many times the failure handler may raise for one FAILED workflow
    # before the record is stamped handled and counted as abandoned (F-A6).
    failure_handling_max_attempts: int = 5
    # Backoff written into ``not_before`` after an unrecognised internal error,
    # so the record stays executable without being retried every tick.
    internal_error_backoff_seconds: float = 60.0
    # F-N1 §8 / rule A: how long a not-yet-started job workflow waits for
    # another remediation on one of its nodes before it gives up by stopping
    # the job. One variable with ``ProductionExecutorConfig``'s field of the
    # same name; ``validate_timing_relationships`` refuses a pair that differs.
    node_busy_wait_seconds: float = 240.0
    # F-C7: how long one dispatch cycle waits for its batch before it stops
    # queuing more work and scans again. Rows not started by then stay
    # PENDING for the next cycle; running ones finish. 0 waits for the batch.
    cycle_deadline_seconds: float = 0.0
    # F-A5: the PENDING watchdog observes and never terminalizes. When the
    # oldest dispatchable PENDING / SAFETY_PENDING row the scan saw has been
    # eligible for longer than this, the tick logs a warning naming it and
    # counts ``pending_age_warnings_total``; ``pending_age_seconds_max`` is
    # the gauge behind it. 0 disables the warning, never the gauge.
    pending_age_warning_seconds: float = 900.0

    @classmethod
    def from_environment(cls, executor_enabled: bool) -> WorkflowDispatcherConfig:
        return cls.from_mapping(os.environ, executor_enabled)

    @classmethod
    def from_mapping(
        cls, values: Mapping[str, str], executor_enabled: bool
    ) -> WorkflowDispatcherConfig:
        dispatcher_enabled = env_bool(
            "GPU_FAULT_ENABLE_WORKFLOW_DISPATCHER", True, environ=values
        )
        poll_interval_seconds = float(
            values.get("GPU_FAULT_WORKFLOW_POLL_INTERVAL_SECONDS", "5")
        )
        return cls(
            enabled=(executor_enabled and dispatcher_enabled),
            poll_interval_seconds=poll_interval_seconds,
            batch_size=int(values.get("GPU_FAULT_WORKFLOW_BATCH_SIZE", "100")),
            max_workers=int(values.get("GPU_FAULT_WORKFLOW_DISPATCHER_WORKERS", "8")),
            confirm_cluster_name=(values.get("GPU_FAULT_HYPERPOD_CLUSTER") or None),
            dispatch_lease_seconds=float(
                values.get(
                    "GPU_FAULT_WORKFLOW_DISPATCH_LEASE_SECONDS",
                    str(max(15.0, poll_interval_seconds * 3)),
                )
            ),
            failure_handling_max_attempts=int(
                values.get("GPU_FAULT_WORKFLOW_FAILURE_HANDLING_MAX_ATTEMPTS", "5")
            ),
            internal_error_backoff_seconds=float(
                values.get("GPU_FAULT_WORKFLOW_INTERNAL_ERROR_BACKOFF_SECONDS", "60")
            ),
            node_busy_wait_seconds=node_busy_wait_seconds(values),
            cycle_deadline_seconds=float(
                values.get("GPU_FAULT_WORKFLOW_DISPATCH_CYCLE_SECONDS", "0")
            ),
            pending_age_warning_seconds=float(
                values.get("GPU_FAULT_WORKFLOW_PENDING_AGE_WARNING_SECONDS", "900")
            ),
        )


class TimingConfigurationError(WorkflowExecutionError):
    """Two or more timing knobs are ordered so one can never do its job."""


def _seconds(value: float) -> str:
    return f"{int(value) if float(value).is_integer() else value}s"


def validate_timing_relationships(
    executor: ProductionExecutorConfig,
    dispatcher: WorkflowDispatcherConfig,
    *,
    verify_max_attempts: int | None,
    branch_max_rungs: int,
) -> list[str]:
    """One sentence per timing relationship the two configs break (review item 3).

    A sentence prefixed ``warning:`` is not a violation: it names a relationship
    that could not be checked from what the caller knew. The rules, each
    derived from the code that consumes the pair:

    * every step waiting ceiling <= the node workflow lifetime.
      ``step_bounds.bounded_waiting_outcome`` (step_bounds.py:148) fails a
      step at ``step_waiting_limit``, but ``step_bounds.deadline_outcome``
      (step_bounds.py:66-71) fails it at ``lifetime_deadline_at`` first, so a
      ceiling above the lifetime never fires and the adapter's own timeout
      handling -- for managed recovery, the escalation notification -- is
      skipped. ``OPERATOR_ACKNOWLEDGEMENT_OPERATIONS`` are exempt because
      ``claim_deadlines`` (restart_budget_preflight.py:124-130) floors the
      lifetime at that ceiling.
    * the managed recovery window plus one reboot allowance <= the job
      lifetime. A job workflow's branch that fails its reset reboots the node
      and, on a further failure, delegates its replacement
      (branch_escalation.py:105-124); both waits are the managed window
      (``managed_recovery_step_overrides``), stamped once per workflow by
      ``claim_deadlines`` (restart_budget_preflight.py:109-117).
    * the workflow execution timeout <= both lifetimes. ``claim_deadlines``
      returns ``min(execution, lifetime)`` (restart_budget_preflight.py:131),
      so a longer timeout is silently truncated and the configured value lies.
    * the install workflow execution floor <= the node workflow lifetime, once
      the lifetime holds the install ceiling. ``claim_deadlines`` floors an
      install workflow's execution deadline at the ceiling plus the containment
      allowance (``install_execution_floor_seconds``) and then caps it by the
      lifetime, so a lifetime between the two truncates every driver, firmware
      or EFA-driver install workflow silently. A lifetime below the ceiling is
      already the first rule's sentence and is not reported twice.
    * the lease duration < the execution timeout. The lease is renewed once
      per step (executor.py:207-212, ``_lease_duration`` executor.py:1661);
      a lease at or above the timeout means an executor that died mid-step
      holds the record past the point the deadline should have failed it.
    * the executor's node-busy wait == the dispatcher's. They are one knob
      (``GPU_FAULT_JOB_WORKFLOW_NODE_BUSY_WAIT_SECONDS``, rule A): the
      dispatcher holds a job workflow that has not started
      (dispatcher.py:467-478) and the executor caps the running restart
      premise (``step_bounds.bounded_waiting_outcome``) for the same window
      whenever the premise's remediation is not an open workflow; while it
      is open the executor defers to that remediation's lifetime instead
      (``step_bounds._premise_hold_limit``), which is never shorter than the
      window, so the equality here still means one deadline for one wait.
      Two values would give one job two different deadlines for one wait.
    * the node-busy wait < the VERIFY_NO_GPU_CLIENTS total wait. A job
      workflow whose node is under another remediation gives up at
      ``created_at + node_busy_wait_seconds`` by stopping its job
      (dispatcher.py:459-468); the other remediation's verify retries once
      per dispatch (barriers.py:36-52, step_execution.py:239) and each
      dispatch is one poll interval apart (dispatcher.py:1333), so its total
      wait is ``verify_max_attempts x poll_interval_seconds``. The verify
      started earlier, so a busy wait that is not strictly shorter can never
      stop the job before the verify gives up.
    * the job lifetime >= ``branch_max_rungs`` x the managed window. Each rung
      the escalator takes (branch_escalation.py:107) is one more delegated
      recovery inside the same lifetime.
    * the dispatch lease >= 3 x the poll interval. ``from_mapping`` defaults it
      to ``max(15, 3 x poll)`` so one missed tick does not lose the fleet-wide
      lease (dispatcher.py:145, 280-288); an explicit override below that
      re-opens the two-dispatcher window. 0 disables the lease and is not
      compared.
    * the cycle deadline (when > 0) >= the poll interval, or the cycle stops
      queuing work before the next scan would have run (dispatcher.py:176-180).
    """

    violations: list[str] = []
    node_lifetime = executor.node_workflow_lifetime_seconds
    job_lifetime = executor.job_workflow_lifetime_seconds
    execution = executor.workflow_execution_timeout_seconds

    ceilings: list[tuple[str, int]] = [
        ("default", executor.step_waiting_timeout_seconds)
    ]
    ceilings.extend(
        (operation.value, limit)
        for operation, limit in sorted(
            executor.step_waiting_timeout_overrides.items(),
            key=lambda item: item[0].value,
        )
        if operation not in OPERATOR_ACKNOWLEDGEMENT_OPERATIONS
    )
    over_node = [
        f"{name} {_seconds(limit)}" for name, limit in ceilings if limit > node_lifetime
    ]
    if over_node:
        violations.append(
            "step waiting ceilings exceed the node workflow lifetime "
            f"({_seconds(node_lifetime)}): {', '.join(over_node)}, so the lifetime "
            "fails those steps before their own cap and skips the adapter's "
            "timeout handling"
        )

    managed_window = executor.step_waiting_limit(WorkflowOperation.REPLACE_NODE)
    reboot_allowance = executor.step_waiting_limit(WorkflowOperation.RESTART_NODE)
    if managed_window + reboot_allowance > job_lifetime:
        violations.append(
            f"the managed recovery window ({_seconds(managed_window)}) plus one "
            f"reboot allowance ({_seconds(reboot_allowance)}) exceeds the job "
            f"workflow lifetime ({_seconds(job_lifetime)}), so a branch that "
            "reboots and then delegates a replacement is failed by the lifetime"
        )

    exceeded = [
        f"the {kind} workflow lifetime ({_seconds(lifetime)})"
        for kind, lifetime in (("job", job_lifetime), ("node", node_lifetime))
        if execution > lifetime
    ]
    if exceeded:
        violations.append(
            f"the workflow execution timeout ({_seconds(execution)}) exceeds "
            + " and ".join(exceeded)
            + ", so claim_deadlines silently truncates it"
        )

    install_ceiling = max(
        executor.step_waiting_limit(operation) for operation in NODE_INSTALL_OPERATIONS
    )
    install_floor = executor.install_execution_floor_seconds()
    if install_ceiling <= node_lifetime < install_floor:
        violations.append(
            f"the node workflow lifetime ({_seconds(node_lifetime)}, "
            "GPU_FAULT_NODE_WORKFLOW_MAX_LIFETIME_SECONDS) holds the install step "
            f"ceiling ({_seconds(install_ceiling)}, "
            "GPU_FAULT_WORKFLOW_INSTALL_STEP_TIMEOUT_SECONDS) but not the install "
            f"workflow execution floor ({_seconds(install_floor)}: the ceiling plus "
            f"{_seconds(executor.node_install_containment_seconds)} "
            "GPU_FAULT_WORKFLOW_INSTALL_CONTAINMENT_SECONDS), so claim_deadlines "
            "truncates every driver, firmware or EFA-driver install workflow to "
            "the lifetime and the floor never applies"
        )

    if executor.lease_duration_seconds >= execution:
        violations.append(
            f"the workflow lease duration ({_seconds(executor.lease_duration_seconds)}) "
            f"is not below the workflow execution timeout ({_seconds(execution)}), "
            "so a dead executor holds the record past its deadline"
        )

    busy = dispatcher.node_busy_wait_seconds
    if executor.node_busy_wait_seconds != busy:
        violations.append(
            "the executor's job workflow node-busy wait "
            f"({_seconds(executor.node_busy_wait_seconds)}) differs from the "
            f"dispatcher's ({_seconds(busy)}); both must read "
            "GPU_FAULT_JOB_WORKFLOW_NODE_BUSY_WAIT_SECONDS, or a job waiting "
            "on a node under repair gets two deadlines for one wait"
        )
    poll = dispatcher.poll_interval_seconds
    if verify_max_attempts is None:
        violations.append(
            "warning: the VERIFY_NO_GPU_CLIENTS total wait is unknown "
            "(verify_max_attempts not given), so the job workflow node-busy wait "
            f"({_seconds(busy)}) cannot be checked against it"
        )
    else:
        verify_total = verify_max_attempts * poll
        if busy >= verify_total:
            violations.append(
                f"the job workflow node-busy wait ({_seconds(busy)}) is not below "
                f"the VERIFY_NO_GPU_CLIENTS total wait ({verify_max_attempts} "
                f"attempts x {_seconds(poll)} poll = {_seconds(verify_total)}), so "
                "a waiting job workflow cannot stop its job before the node "
                "remediation's verify gives up"
            )

    rungs_total = branch_max_rungs * managed_window
    if rungs_total > job_lifetime:
        violations.append(
            f"the job workflow lifetime ({_seconds(job_lifetime)}) cannot hold "
            f"{branch_max_rungs} escalation rungs of the managed recovery window "
            f"({_seconds(managed_window)} each, {_seconds(rungs_total)} in all): "
            f"the escalated branch can take up to {branch_max_rungs} delegated "
            "recoveries"
        )

    lease = dispatcher.dispatch_lease_seconds
    if lease > 0 and lease < 3 * poll:
        violations.append(
            f"the dispatch lease ({_seconds(lease)}) is below three poll intervals "
            f"({_seconds(3 * poll)}), so one missed tick hands the fleet-wide "
            "scan to a second dispatcher"
        )

    cycle = dispatcher.cycle_deadline_seconds
    if cycle > 0 and cycle < poll:
        violations.append(
            f"the dispatch cycle deadline ({_seconds(cycle)}) is below the poll "
            f"interval ({_seconds(poll)}), so a cycle stops queuing work before "
            "the next scan is due"
        )
    return violations


def validate_timing_or_raise(
    executor: ProductionExecutorConfig,
    dispatcher: WorkflowDispatcherConfig,
    *,
    verify_max_attempts: int | None,
    branch_max_rungs: int,
) -> list[str]:
    """Fail closed on every violation; return the ``warning:`` strings to log."""

    reported = validate_timing_relationships(
        executor,
        dispatcher,
        verify_max_attempts=verify_max_attempts,
        branch_max_rungs=branch_max_rungs,
    )
    warnings = [line for line in reported if line.startswith("warning: ")]
    violations = [line for line in reported if not line.startswith("warning: ")]
    if violations:
        raise TimingConfigurationError(
            "timing configuration is inconsistent: " + "; ".join(violations)
        )
    return warnings


def validate_timing_from_environment(values: Mapping[str, str]) -> list[str]:
    """Build both configs from ``values`` and validate them together.

    Reads the two knobs owned elsewhere exactly as their owners do:
    ``GPU_FAULT_GPU_CLIENT_VERIFY_MAX_ATTEMPTS`` (node-action adapter) and
    ``GPU_FAULT_BRANCH_ESCALATION_MAX_RUNGS`` (``app.context._branch_escalator``).
    Raises ``TimingConfigurationError`` on a violation; returns the warnings.
    """

    executor = ProductionExecutorConfig.from_mapping(values)
    dispatcher = WorkflowDispatcherConfig.from_mapping(values, executor.enabled)
    verify_max_attempts = int(
        values.get("GPU_FAULT_GPU_CLIENT_VERIFY_MAX_ATTEMPTS", "60")
    )
    branch_max_rungs = int(values.get("GPU_FAULT_BRANCH_ESCALATION_MAX_RUNGS", "2"))
    return validate_timing_or_raise(
        executor,
        dispatcher,
        verify_max_attempts=verify_max_attempts,
        branch_max_rungs=branch_max_rungs,
    )

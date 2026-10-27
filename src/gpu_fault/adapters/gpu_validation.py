from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timedelta, timezone
from typing import Any

from gpu_fault.adapters.common import quarantine_taint_value
from gpu_fault.adapters.pending_repair_rejudge import (
    awaited_findings_text,
    pending_repair_reset,
    pending_repair_wait,
)
from gpu_fault.cluster_scope import FABRIC_MANAGER_UNIT
from gpu_fault.dcgm_fields import missing_fabric_metric_groups
from gpu_fault.execution import (
    WorkflowStepContext,
    WorkflowStepOutcome,
)
from gpu_fault.execution.config import DEFAULT_STEP_WAITING_TIMEOUT_SECONDS
from gpu_fault.execution.hung_handoff import support_handoff_follows
from gpu_fault.fabric_scope import (
    requires_task_scope,
    unknown_task_fabric,
    validate_task_fabric,
    validation_evidence,
)
from gpu_fault.gpu_fabric_manager import fabric_validation
from gpu_fault.gpu_fabric_window import fabric_state_loss
from gpu_fault.gpu_instance_inventory import gpu_instance_inventory
from gpu_fault.gpu_metrics import GpuMetricSource, GpuMetricsService
from gpu_fault.gpu_observation import metric_device_key
from gpu_fault.gpu_temperature_policy import TLIMIT_READING
from gpu_fault.health_evidence_validation import (
    reset_cooldown,
    reset_unproven_cooldown,
    special_health_validation,
    thermal_cooldown,
    validation_success,
)
from gpu_fault.health_recovery_contract import (
    HEALTH_CONTRACT,
    THERMAL_NAMES,
    prevents_hardware_escalation,
)
from gpu_fault.host_diagnostics import (
    HOST_DIAGNOSTIC_PROBE,
    apply_diagnostic_probe,
    diagnostic_failure_details,
    validate_host_metrics,
)
from gpu_fault.models import (
    WorkflowOperation,
    WorkflowStepSpec,
    WorkflowStepStatus,
)
from gpu_fault.adapters.inventory_verdict import (
    collector_statuses,
    inventory_failure_despite_collection_error,
)
from gpu_fault.observation_health import observation_reason
from gpu_fault.operation_registry import (
    OperationAdapter,
    operations_for_adapter,
)
from gpu_fault.orchestration.escalation import escalation_origin
from gpu_fault.orchestration.validated_restore import is_validated_restore_workflow
from gpu_fault.store.shared.errors import NotFoundError
from gpu_fault.telemetry import (
    CollectorKind,
)
from gpu_fault.workflow_quarantine import quiesce_taint_nodes_held


#: The per-interface link readings. The host collector writes one for every
#: interface in its scope -- each interface on its required list, present or
#: missing (``network_link_down=1``), and every device-backed netdev -- into
#: every batch it delivers (``collectors/host/network.py``).
INTERFACE_LINK_METRICS = frozenset({"network_link_down", "network_link_up"})

#: The actions whose completion the GPU (EFA) inventory gate waits for, and
#: after which the ACTIVE count must be re-read.
GPU_INVENTORY_RECOVERY_OPERATIONS = frozenset(
    {
        WorkflowOperation.RESTART_NODE,
        WorkflowOperation.REPLACE_NODE,
        WorkflowOperation.RESTART_GPU_DEVICE_PLUGIN,
    }
)
EFA_INVENTORY_RECOVERY_OPERATIONS = frozenset(
    {
        WorkflowOperation.REMEDIATE_EFA_DRIVER,
        WorkflowOperation.RESTART_EFA_DEVICE_PLUGIN,
        WorkflowOperation.RESTART_NODE,
        WorkflowOperation.REPLACE_NODE,
    }
)
#: The actions after which GPU and host telemetry starts over: a validation
#: behind one of them reads only samples newer than its completion.
TELEMETRY_READINESS_OPERATIONS = frozenset(
    {
        WorkflowOperation.RESET_GPU,
        WorkflowOperation.RECOVER_DRAIN_P2P,
        WorkflowOperation.RESET_ALL_GPUS_NVSWITCHES,
        WorkflowOperation.RESTART_FABRIC_MANAGER,
        WorkflowOperation.REMEDIATE_DRIVER,
        WorkflowOperation.UPDATE_SOFTWARE_FIRMWARE,
        WorkflowOperation.REMEDIATE_EFA_DRIVER,
        WorkflowOperation.RESTART_EFA_DEVICE_PLUGIN,
        WorkflowOperation.RESTART_GPU_DEVICE_PLUGIN,
        WorkflowOperation.RESTART_NODE,
        WorkflowOperation.REPLACE_NODE,
    }
)
#: The WARNING families VALIDATE_GPU grants the warning grace and, past it,
#: b202's wait for newer GPU metrics: thermal (the temperature grace),
#: correctable memory and power (the transient grace). The pending-repair
#: flags are judged against the reset instead (``pending_repair_rejudge``).
THERMAL_WARNING_NAMES = frozenset(
    {
        "gpu_temperature_c",
        TLIMIT_READING,
        "memory_temperature_c",
        "thermal_violation_total_us",
        "clock_throttle_reasons",
        "composite:THERMAL_STRESS",
    }
)
MEMORY_WARNING_NAMES = frozenset(
    {
        "ecc_sbe_volatile_total",
        "ecc_sbe_aggregate_total",
        "retired_pages_sbe_total",
        "row_remap_correctable_total",
        "composite:CORRECTABLE_MEMORY_DEGRADATION",
    }
)
POWER_WARNING_NAMES = frozenset(
    {"power_violation_total_us", "composite:POWER_LIMIT_THROTTLING"}
)


def current_interface_readings(
    latest: list[Any], newest_batch_at: datetime | None
) -> list[Any]:
    """Drop the link readings the node's newest complete host batch no longer has.

    A latest-value row outlives the interface it describes: once an interface
    leaves the collector's scope -- taken off its required-interface list, or
    a test interface deleted and its collector window closed -- its last
    reading stays inside the validation window, and a ``network_link_down=1``
    (``network_link_up=0``) failed VALIDATE_HOST (VALIDATE_FABRIC) on an
    interface the node no longer reports (live 2026-10-25,
    GF-GENERAL-LIVE-008 interface_missing: the validated restore ended
    FAILED). ``newest_batch_at`` must be the HOST_TELEMETRY status's last
    success with no collection error since -- what the caller's collector
    check already demands -- so that batch read every interface in scope and
    a link row older than it is out of scope. Every other row is kept: not
    every contributor reports every device in every batch.
    """

    if newest_batch_at is None:
        return list(latest)
    return [
        item
        for item in latest
        if item.name not in INTERFACE_LINK_METRICS
        or item.observed_at >= newest_batch_at
    ]


def latest_inventory_requirement(row: Any, active_name: str) -> dict[str, Any]:
    """The inventory requirement of a reboot that no inventory finding planned.

    It validates against the node's latest ``*_inventory_expected_count`` row,
    which is the collector's *configured* count. The row's own
    ``node_instance_type`` label travels with it so that
    ``_bound_by_instance_inventory`` caps a count above the type's physical one
    exactly as a plan-frozen requirement is capped (GF-GENERAL-LIVE-012
    ``expectation_wrong``: ``expected=9`` on an 8-GPU node failed every
    reboot's validation and escalated the typo into REPLACE_NODE). A row
    without the label is validated as configured, as before.
    """

    requirement: dict[str, Any] = {"metrics": {active_name: row.value}}
    instance_type = (getattr(row, "labels", None) or {}).get("node_instance_type")
    if instance_type:
        requirement["node_instance_type"] = instance_type
    return requirement


class GpuValidationAdapter:
    OPERATIONS = operations_for_adapter(OperationAdapter.GPU_VALIDATION)

    def __init__(
        self,
        metrics: GpuMetricsService,
        *,
        store=None,
        owner: str = "gpu-fault-validation-adapter",
        max_sample_age: timedelta = timedelta(minutes=2),
        post_action_max_sample_age: timedelta = timedelta(minutes=6),
        require_rdma: bool = False,
        temperature_warning_grace: timedelta = timedelta(minutes=2),
        transient_warning_grace: timedelta = timedelta(minutes=2),
        restore_collector_grace: timedelta = timedelta(seconds=120),
        step_waiting_cap: timedelta = timedelta(
            seconds=DEFAULT_STEP_WAITING_TIMEOUT_SECONDS
        ),
    ) -> None:
        self.metrics = metrics
        self.store = store or metrics.store
        self.owner = owner
        self.max_sample_age = max_sample_age
        if post_action_max_sample_age < max_sample_age:
            raise ValueError("post-action sample age cannot be less than normal age")
        self.post_action_max_sample_age = post_action_max_sample_age
        self.require_rdma = require_rdma
        if temperature_warning_grace < timedelta(seconds=15):
            raise ValueError("temperature warning grace must be at least 15 seconds")
        self.temperature_warning_grace = temperature_warning_grace
        if transient_warning_grace < timedelta(seconds=15):
            raise ValueError("transient warning grace must be at least 15 seconds")
        self.transient_warning_grace = transient_warning_grace
        if restore_collector_grace < timedelta(seconds=15):
            raise ValueError("restore collector grace must be at least 15 seconds")
        # How long a validated restore waits for the node's collectors before
        # it fails naming them: an operator-driven restore of a node whose
        # DaemonSets are not running (live: the exporter evicted, 604 s of
        # silence under the 600 s cap) learns nothing from the cap.
        self.restore_collector_grace = restore_collector_grace
        if step_waiting_cap <= timedelta(0):
            raise ValueError("step waiting cap must be positive")
        # The executor's per-step cap for VALIDATE_GPU
        # (``step_bounds.bounded_waiting_outcome``), named in the WAITING
        # reason of a wait for newer GPU metrics so the record says when the
        # wait can end at the latest. The executor enforces it; this is the
        # same number, not a second bound.
        self.step_waiting_cap = step_waiting_cap

    def supports(self, step: WorkflowStepSpec) -> bool:
        return step.execution_owner == self.owner and step.operation in self.OPERATIONS

    def execute(self, context: WorkflowStepContext) -> WorkflowStepOutcome:
        now = datetime.now(timezone.utc)
        if (special := special_health_validation(context, self.store, now)) is not None:
            return special
        # DCGM omits XID_ERRORS on healthy GPUs on some driver/DCGM
        # combinations. Fresh temperature proves that GPU telemetry has
        # resumed; an XID series, when present, is still evaluated below.
        required = {"gpu_temperature_c"}
        pending: dict[str, dict[str, list[str]]] = {}
        failures: dict[str, list[str]] = {}
        node_findings: dict[str, list[dict[str, Any]]] = {}
        clamped: dict[str, dict[str, Any]] = {}
        evidence: dict[str, dict[str, Any]] = {}
        operation = context.step.operation
        probe = context.step.parameters.get(HOST_DIAGNOSTIC_PROBE)
        raw_inventory_requirements = context.step.parameters.get(
            "inventory_requirements_by_node", {}
        )
        inventory_requirements = (
            raw_inventory_requirements
            if isinstance(raw_inventory_requirements, dict)
            else {}
        )
        inventory_recovery_operations = (
            EFA_INVENTORY_RECOVERY_OPERATIONS
            if operation is WorkflowOperation.VALIDATE_FABRIC
            else GPU_INVENTORY_RECOVERY_OPERATIONS
        )
        completed_node_recovery = any(
            execution.step_index < context.step_index
            and execution.status is WorkflowStepStatus.SUCCEEDED
            and execution.operation
            in {
                WorkflowOperation.RESTART_NODE,
                WorkflowOperation.REPLACE_NODE,
            }
            for execution in context.workflow.step_executions
        )
        readiness_after = telemetry_readiness_after(context)
        required_collectors = required_collectors_for(
            operation,
            host_telemetry=bool(
                inventory_requirements or completed_node_recovery or probe
            ),
        )
        if operation is WorkflowOperation.VALIDATE_FABRIC:
            required = set()
        # Per node, the host collector's collection errors at a verdict the
        # inventory gate read from its erroring batch (``host_collector_errors``).
        collector_errors: dict[str, list[str]] = {}
        for node_id in context.step.node_ids:
            statuses = collector_statuses(
                self.store, context.incident.cluster_id, node_id
            )
            recent = self._recent_host_metrics(
                context,
                node_id,
                now,
                readiness_after,
                required_collectors,
                statuses,
            )
            if isinstance(recent, dict):
                if inventory_failure_despite_collection_error(
                    self,
                    context,
                    node_id,
                    operation,
                    inventory_requirements,
                    inventory_recovery_operations,
                    completed_node_recovery,
                    now,
                    readiness_after,
                    statuses,
                    failures,
                    clamped,
                    evidence,
                    collector_errors,
                ):
                    continue
                dark = self._restore_collectors_dark(context, node_id, recent, now)
                if dark is not None:
                    return dark
                pending[node_id] = {
                    **recent,
                    **quiesce_taint_pending(self.store, context, node_id),
                }
                continue
            all_host_latest, host_latest = recent
            if apply_diagnostic_probe(probe, host_latest, node_id, pending, failures):
                continue
            inventory_pending = self._validate_inventory(
                context,
                node_id,
                operation,
                inventory_requirements,
                inventory_recovery_operations,
                completed_node_recovery,
                all_host_latest,
                host_latest,
                failures,
                clamped,
                evidence,
            )
            if inventory_pending is not None:
                pending[node_id] = inventory_pending
                continue
            if operation is WorkflowOperation.VALIDATE_HOST:
                host_pending = self._validate_host_metrics(
                    host_latest, node_id, failures
                )
                if host_pending is not None:
                    pending[node_id] = host_pending
                continue
            metric_pending = self._validate_gpu_metric_presence(
                context, node_id, now, readiness_after, required
            )
            if metric_pending is not None:
                pending[node_id] = {
                    **metric_pending,
                    **quiesce_taint_pending(self.store, context, node_id),
                }
                continue
            if operation is WorkflowOperation.VALIDATE_FABRIC:
                fabric_pending = self._validate_fabric_metrics(
                    host_latest, node_id, failures, context
                )
                if fabric_pending is not None:
                    pending[node_id] = fabric_pending
                    continue
            if operation is WorkflowOperation.VALIDATE_GPU:
                nvlink_pending = self._validate_nvlink_fabric(
                    context, node_id, now, readiness_after, failures
                )
                if nvlink_pending is not None:
                    pending[node_id] = nvlink_pending
                    continue
            finding_pending = self._validate_active_findings(
                context, node_id, operation, now, failures, node_findings
            )
            if finding_pending is not None:
                pending[node_id] = finding_pending
        diagnostic_details = diagnostic_failure_details(probe, pending, failures)
        diagnostic_details.update(unknown_task_fabric_details(context))
        reset_unproven_cooldown(self.store, context, now, pending, failures)
        if prevents_hardware_escalation(context.step.parameters):
            diagnostic_details["host_diagnostic_failure"] = True
        erroring: dict[str, Any] = {}
        if collector_errors:
            erroring["host_collector_errors"] = collector_errors
        if pending:
            return WorkflowStepOutcome.waiting(
                details={
                    "pending_nodes": sorted(pending),
                    "node_pending": pending,
                    "failed_nodes_observed": sorted(failures),
                    "node_failures_observed": failures,
                    **diagnostic_details,
                    **erroring,
                }
            )
        if failures:
            failures = {
                node_id: sorted(set(reasons)) for node_id, reasons in failures.items()
            }
            failed_nodes = sorted(failures)
            return WorkflowStepOutcome.failed(
                "validation failed on nodes: "
                + "; ".join(
                    node_failure_text(node_id, failures[node_id], node_findings)
                    for node_id in failed_nodes
                ),
                details={
                    "failed_nodes": failed_nodes,
                    "node_failures": failures,
                    "validation": operation.value,
                    **diagnostic_details,
                    **erroring,
                    **({"node_findings": node_findings} if node_findings else {}),
                    **inventory_details(clamped, evidence),
                },
            )
        cooling = thermal_cooldown(context, self.metrics, now, self._node_gpu_targets)
        if cooling is not None:
            return cooling
        return validation_success(
            context, **success_evidence(context, clamped, evidence)
        )

    def _restore_collectors_dark(
        self,
        context: WorkflowStepContext,
        node_id: str,
        pending: dict[str, Any],
        now: datetime,
    ) -> WorkflowStepOutcome | None:
        """Fail a validated restore whose collectors stayed silent past the grace.

        Only the operator-driven validated restore (``workflow-validated-restore-``):
        the chain's own post-action validation keeps the step cap, because the
        exporter it waits for is the one its RESTORE just let back. The wait is
        measured from the step's first attempt (``step_executions``).
        """

        stale = pending.get("missing_or_stale_collectors")
        if not stale or not is_validated_restore_workflow(context.workflow.request_id):
            return None
        started = step_started_at(context)
        if started is None or now - started < self.restore_collector_grace:
            return None
        waited = int((now - started).total_seconds())
        collectors = ", ".join(str(item) for item in stale)
        reason = (
            f"the gpu-fault collectors on {node_id} ({collectors}) have not "
            f"reported for {waited}s: the node's DaemonSets are not running "
            "(check the Pods on the node, the gpu-fault.io/quiesce taint and the "
            "quiesced host services), then rerun the restore"
        )
        return WorkflowStepOutcome.failed(
            f"validation failed on nodes: {node_id}: {reason}",
            details={
                "failed_nodes": [node_id],
                "node_failures": {node_id: [f"collectors_not_reporting: {collectors}"]},
                "validation": context.step.operation.value,
                "collectors_not_reporting": {node_id: list(stale)},
                "restore_collector_grace_seconds": int(
                    self.restore_collector_grace.total_seconds()
                ),
                **quiesce_taint_pending(self.store, context, node_id),
            },
        )

    def _recent_host_metrics(
        self,
        context,
        node_id,
        now,
        readiness_after,
        required_collectors,
        statuses=None,
    ):
        max_age = (
            self.post_action_max_sample_age
            if readiness_after is not None
            else self.max_sample_age
        )
        if statuses is None:
            statuses = collector_statuses(
                self.store, context.incident.cluster_id, node_id
            )
        stale = sorted(
            collector.value
            for collector in required_collectors
            if observation_reason(
                statuses.get(collector),
                now,
                max_age.total_seconds(),
                after=readiness_after,
            )
            is not None
        )
        if stale:
            return {"missing_or_stale_collectors": stale}
        all_latest = self.store.list_telemetry_metrics_latest(
            context.incident.cluster_id, node_id
        )
        latest = [
            item
            for item in all_latest
            if now - item.observed_at <= max_age
            and (readiness_after is None or item.observed_at > readiness_after)
        ]
        host = statuses.get(CollectorKind.HOST_TELEMETRY)
        if CollectorKind.HOST_TELEMETRY in required_collectors and host is not None:
            # Checked above: fresh, and no collection error since its success.
            latest = current_interface_readings(latest, host.last_success_at)
        return all_latest, latest

    def _validate_inventory(
        self,
        context: WorkflowStepContext,
        node_id: str,
        operation: WorkflowOperation,
        requirements: dict[str, Any],
        recovery_operations: frozenset[WorkflowOperation],
        completed_recovery: bool,
        all_latest: list[Any],
        latest: list[Any],
        failures: dict[str, list[str]],
        clamped: dict[str, Any] | None = None,
        evidence: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Judge the node's ACTIVE inventory against the plan's requirement on
        a reading from the window ``inventory_observation_window`` opens; the
        reading each verdict rests on goes into ``evidence``
        (``details.inventory_evidence``)."""

        requirement = requirements.get(node_id)
        if (
            not isinstance(requirement, dict)
            and completed_recovery
            and operation
            in {
                WorkflowOperation.VALIDATE_GPU,
                WorkflowOperation.VALIDATE_FABRIC,
            }
        ):
            expected_name, active_name = (
                (
                    "gpu_inventory_expected_count",
                    "gpu_inventory_active_count",
                )
                if operation is WorkflowOperation.VALIDATE_GPU
                else (
                    "efa_inventory_expected_count",
                    "efa_inventory_active_count",
                )
            )
            expected = [item for item in all_latest if item.name == expected_name]
            if expected:
                requirement = latest_inventory_requirement(expected[-1], active_name)
        if not isinstance(requirement, dict):
            return None
        required = self._required_inventory_metrics(requirement, operation)
        required, bounded = self._bound_by_instance_inventory(requirement, required)
        if bounded and clamped is not None:
            clamped[node_id] = bounded
        if not required:
            return None
        window = inventory_observation_window(context, recovery_operations)
        if window is None:
            return {
                "missing_inventory_action_completion": [
                    item.value
                    for item in sorted(
                        recovery_operations,
                        key=lambda value: value.value,
                    )
                ]
            }
        observed_after, pending_key = window
        for metric, expected_count in required.items():
            samples = [
                item
                for item in latest
                if item.name == metric and item.observed_at > observed_after
            ]
            if not samples:
                return {
                    pending_key: [metric],
                    "observed_after": [observed_after.isoformat()],
                }
            actual = samples[-1].value
            if evidence is not None:
                evidence.setdefault(node_id, {})[metric] = {
                    "value": actual,
                    "expected": expected_count,
                    "observed_at": samples[-1].observed_at.isoformat(),
                }
            if actual != expected_count:
                failures.setdefault(node_id, []).append(
                    f"{metric}={actual:g},expected={expected_count}"
                )
        return None

    @staticmethod
    def _required_inventory_metrics(requirement, operation):
        raw = requirement.get("metrics")
        if isinstance(raw, dict):
            prefix = (
                "gpu_inventory_"
                if operation is WorkflowOperation.VALIDATE_GPU
                else "efa_inventory_"
            )
            return {
                str(name): value
                for name, value in raw.items()
                if str(name).startswith(prefix)
            }
        metric = requirement.get("active_metric")
        return {str(metric): requirement.get("expected_count")} if metric else {}

    @staticmethod
    def _bound_by_instance_inventory(
        requirement: dict[str, Any],
        required: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
        """Cap each expected count at what the node's instance type carries.

        The requirement is the finding's configured expected count, frozen into
        the plan. A configured count above the instance type's physical GPU
        (or EFA device) count is a configuration error that no hardware action
        can satisfy: validated literally it fails a healthy node for ever and
        escalates a config mistake into REPLACE_NODE and a support ticket
        (live: ``expected=9`` on an 8-GPU p5en, GF-REGIONAL-COLLECT-004). The
        physical count is the ceiling; a configured count at or below it is
        validated as configured, so a real loss (7 of 8) still fails. Returns
        ``(bounded required metrics, {metric: {configured, physical,
        instance_type}} for the ones that were capped)``.
        """

        instance_type = requirement.get("node_instance_type")
        if not isinstance(instance_type, str) or not instance_type:
            return required, {}
        try:
            gpu_count, efa_count = gpu_instance_inventory(instance_type)
        except ValueError:
            return required, {}
        bounded: dict[str, Any] = {}
        capped: dict[str, dict[str, Any]] = {}
        for metric, expected in required.items():
            ceiling = gpu_count if metric.startswith("gpu_inventory_") else efa_count
            try:
                configured = int(expected)
            except (TypeError, ValueError):
                bounded[metric] = expected
                continue
            if configured > ceiling:
                capped[metric] = {
                    "configured": configured,
                    "physical": ceiling,
                    "instance_type": instance_type,
                }
                configured = ceiling
            bounded[metric] = configured
        return bounded, capped

    @staticmethod
    def _validate_host_metrics(latest, node_id, failures):
        return validate_host_metrics(latest, node_id, failures)

    def _validate_gpu_metric_presence(
        self,
        context,
        node_id,
        now,
        readiness_after,
        required,
    ):
        latest = self.metrics.latest(context.incident.cluster_id, node_id)
        max_age = (
            self.post_action_max_sample_age
            if readiness_after is not None
            else self.max_sample_age
        )
        sources = {
            source
            for item in latest
            if (source := getattr(item, "source", None)) is not None
            if now - item.observed_at <= max_age
        }
        required = set(required)
        if (
            context.step.operation is WorkflowOperation.VALIDATE_GPU
            and sources
            and sources <= {GpuMetricSource.NVIDIA_SMI}
        ):
            required.update(
                {
                    "ecc_dbe_volatile_total",
                    "row_remap_failure",
                    "row_remap_pending",
                }
            )
        fresh = [
            item
            for item in latest
            if now - item.observed_at <= max_age
            and (readiness_after is None or item.observed_at > readiness_after)
        ]
        names = {item.sample.canonical_name for item in fresh}
        missing = sorted(required - names)
        if context.step.operation is WorkflowOperation.VALIDATE_FABRIC:
            missing.extend(missing_fabric_metric_groups(names))
        # The node-wide union above lets a sibling GPU answer for one that
        # fell off the bus after RESET_GPU. When the step names its GPUs,
        # every one of them must have reported for itself -- on the node that
        # owns it: a multi-node step carries the union of its nodes' GPUs
        # (a merge widens validation across the faulted nodes), and a node
        # can never report a sibling's GPU (live 2026-10-05: that wait only
        # ended at the per-step cap and escalated a healthy node to a reboot).
        by_gpu: dict[str, list[str]] = {}
        if required and context.step.gpu_uuids:
            fresh_by_gpu: dict[str, set[str]] = {}
            for item in fresh:
                gpu_uuid = getattr(item.sample, "gpu_uuid", None)
                if gpu_uuid:
                    fresh_by_gpu.setdefault(str(gpu_uuid), set()).add(
                        item.sample.canonical_name
                    )
            for gpu_uuid in self._node_gpu_targets(context, node_id, latest):
                gpu_missing = sorted(required - fresh_by_gpu.get(gpu_uuid, set()))
                if gpu_missing:
                    by_gpu[gpu_uuid] = gpu_missing
        if not missing and not by_gpu:
            return None
        pending: dict[str, Any] = {}
        if missing:
            pending["missing_recent_metrics"] = missing
        if by_gpu:
            pending["missing_recent_metrics_by_gpu"] = by_gpu
        return pending

    @staticmethod
    def _node_gpu_targets(
        context: WorkflowStepContext, node_id: str, latest: Sequence[Any]
    ) -> list[str]:
        """The step's GPUs this node answers for.

        A single-node step answers for every GPU it names (unchanged). A
        multi-node step answers per node: the node's entry in
        ``gpu_uuids_by_node`` when the step carries one, else the GPUs the
        node has ever reported -- a GPU that fell off the bus is still in its
        latest samples and stays required; a sibling's GPU never appears and
        is not. A node owning none of them validates node-wide.
        """

        targets = list(dict.fromkeys(context.step.gpu_uuids))
        if len(context.step.node_ids) <= 1:
            return targets
        mapping = context.step.parameters.get("gpu_uuids_by_node")
        owned: set[str]
        if isinstance(mapping, dict) and isinstance(mapping.get(node_id), list):
            owned = {str(value) for value in mapping[node_id]}
        else:
            owned = {
                str(gpu_uuid)
                for item in latest
                if (gpu_uuid := getattr(item.sample, "gpu_uuid", None))
            }
        return [gpu_uuid for gpu_uuid in targets if gpu_uuid in owned]

    def _validate_fabric_metrics(self, latest, node_id, failures, context=None):
        parameters = context.step.parameters if context is not None else {}
        if requires_task_scope(parameters):
            if context is not None and node_id in resolved_unknown_task_fabric(context):
                # Resolved: the step reports this node UNKNOWN, it does not
                # wait for a scope nothing will fill in.
                return None
            bindings = parameters.get("training_fabric_by_node")
            scope = bindings.get(node_id) if isinstance(bindings, dict) else None
            pending, failed = validate_task_fabric(latest, scope)
            if failed:
                failures.setdefault(node_id, []).extend(failed)
            return pending
        link_states = [item.value for item in latest if item.name == "network_link_up"]
        if not link_states:
            return {"missing_recent_fabric_metrics": ["network_link_up"]}
        rdma = [
            item
            for item in latest
            if item.name in {"rdma_link_down", "rdma_errors_delta"}
        ]
        if self.require_rdma and not rdma:
            return {"missing_recent_fabric_metrics": ["rdma_link_down"]}
        failed = [
            item.name
            for item in latest
            if item.name
            in {
                "rdma_link_down",
                "rdma_errors_delta",
                "efa_rnr_errors_delta",
                "efa_retry_errors_delta",
                "efa_cq_errors_delta",
                "network_errors_delta",
                "network_drops_delta",
            }
            and item.value > 0
        ]
        if not all(value == 1 for value in link_states):
            failed.append("network_link_up")
        if failed:
            failures.setdefault(node_id, []).extend(sorted(set(failed)))
        return None

    def _validate_nvlink_fabric(
        self,
        context: WorkflowStepContext,
        node_id: str,
        now: datetime,
        readiness_after: datetime | None,
        failures: dict[str, list[str]],
    ) -> dict[str, Any] | None:
        """Every GPU of a graded NVSwitch node (Hopper SXM, HGX B200/B300)
        joined the NVLink fabric, the node's Fabric Manager unit runs (on NVL5
        it runs NVLSM too; its state follows Fabric Manager's PID), and this
        workflow did not lose the fabric state
        (``gpu_fabric_window.fabric_state_loss``).

        Live (round 9, plain EKS, GF-REGIONAL-COLLECT-015): the reboot's boot
        left Fabric Manager failed and every GPU "in progress", and VALIDATE_GPU,
        VALIDATE_HOST and VALIDATE_FABRIC all passed, so scheduling was restored
        on a node where CUDA could not start. Read from the fresh samples of the
        same window as the presence check. Nothing reported is not a failure: an
        exporter without field 170, a product the fabric state is not graded
        for (``gpu_fabric_manager.nvswitch_fabric_product``), an Agent that does
        not report the unit, a unit systemd does not enable.
        """

        max_age = (
            self.post_action_max_sample_age
            if readiness_after is not None
            else self.max_sample_age
        )
        fresh = [
            (item.sample, item.observed_at)
            for item in self.metrics.latest(context.incident.cluster_id, node_id)
            if now - item.observed_at <= max_age
            and (readiness_after is None or item.observed_at > readiness_after)
        ]
        graded, training, failure = fabric_validation(fresh)
        if not graded:
            return None
        lost = fabric_state_loss(context.workflow, node_id)
        if failure is not None or lost is not None:
            failures.setdefault(node_id, []).extend(
                item for item in (failure, lost) if item is not None
            )
            return None
        pending: dict[str, Any] = (
            {"nvlink_fabric_training": training} if training else {}
        )
        unit = self._fabric_manager_unit(context, node_id, now)
        if unit in {"activating", "reloading", "not_reported_since_action"}:
            pending["fabric_manager_unit"] = unit
        elif unit is not None:
            failures.setdefault(node_id, []).append(f"fabric_manager_unit_{unit}")
        return pending or None

    def _fabric_manager_unit(
        self, context: WorkflowStepContext, node_id: str, now: datetime
    ) -> str | None:
        """The Fabric Manager unit's state when it is not simply running.

        Read off the Agent record, which every heartbeat (30 s) refreshes. It
        must postdate the workflow's last completed step, not just the reset: a
        heartbeat sent between the reset and the RESTORE that started Fabric
        Manager again says "inactive" about a unit that is running now. With no
        step before this one, a record older than the sample age says nothing.
        """

        try:
            agent = self.store.get_agent(context.incident.cluster_id, node_id)
        except (AttributeError, KeyError, NotFoundError):
            return None
        state = agent.collector_services.get(FABRIC_MANAGER_UNIT)
        if state is None or state.enabled.value not in {
            "enabled",
            "enabled-runtime",
            "static",
        }:
            return None
        acted = [
            execution.updated_at
            for execution in context.workflow.step_executions
            if execution.step_index < context.step_index
            and execution.status is WorkflowStepStatus.SUCCEEDED
        ]
        if acted and agent.last_seen_at <= max(acted):
            return "not_reported_since_action"
        if not acted and now - agent.last_seen_at > self.max_sample_age:
            return None
        return None if state.active.value == "active" else state.active.value

    def _validate_active_findings(
        self,
        context: WorkflowStepContext,
        node_id: str,
        operation: WorkflowOperation,
        now: datetime,
        failures: dict[str, list[str]],
        node_findings: dict[str, list[dict[str, Any]]] | None = None,
    ) -> dict[str, list[str]] | None:
        findings = self.metrics.findings(context.incident.cluster_id, node_id)
        if (
            operation
            in {
                WorkflowOperation.VALIDATE_GPU,
                WorkflowOperation.VALIDATE_FABRIC,
            }
            and context.step.gpu_uuids
        ):
            targets = set(context.step.gpu_uuids)
            findings = [
                finding
                for finding in findings
                if finding.gpu_uuid is None or finding.gpu_uuid in targets
            ]
        if not findings:
            return None
        contract = context.step.parameters.get(HEALTH_CONTRACT, {})
        if contract.get("mode") == "THERMAL_COOLDOWN" and all(
            item.canonical_name in THERMAL_NAMES for item in findings
        ):
            reset_cooldown(self.store, context, node_id, now)
            return {"thermal_cooling": [item.finding_id for item in findings]}
        warning_grace = None
        if _warning_only(findings, THERMAL_WARNING_NAMES):
            warning_grace = self.temperature_warning_grace
        elif _warning_only(findings, MEMORY_WARNING_NAMES) or _warning_only(
            findings, POWER_WARNING_NAMES
        ):
            warning_grace = self.transient_warning_grace
        if operation is WorkflowOperation.VALIDATE_GPU and warning_grace is not None:
            grace_end = context.workflow.created_at + warning_grace
            if now < grace_end:
                return {
                    "transient_gpu_warning_cooldown": [
                        finding.finding_id for finding in findings
                    ]
                }
            if all(finding_unjudged_since(item, grace_end) for item in findings):
                return awaiting_newer_gpu_metrics(
                    self.metrics,
                    self.step_waiting_cap,
                    context,
                    node_id,
                    findings,
                    grace_end,
                    now,
                )
        if operation is WorkflowOperation.VALIDATE_GPU and (
            waiting := awaiting_reset_rejudgement(
                self.metrics, self.step_waiting_cap, context, node_id, findings, now
            )
        ):
            return waiting
        failures.setdefault(node_id, []).append("active_gpu_health_findings")
        if node_findings is not None:
            node_findings[node_id] = [finding_summary(item) for item in findings]
        return None


def resolved_unknown_task_fabric(
    context: WorkflowStepContext,
) -> dict[str, dict[str, Any]]:
    """The nodes whose task NIC/port scope cannot be determined, as the
    explicit UNKNOWN verdicts this step resolves to (``unknown_task_fabric``).

    Only when a support hand-off of this workflow runs after the step: the hung
    DAG's ESCALATE_SUPPORT takes the verdict to a human and ends the incident
    ESCALATED. Without one an UNKNOWN must not complete the step -- a SUCCEEDED
    diagnostic would close the incident RECOVERED -- so the node keeps waiting
    as before. Live (HyperPod, GF-GENERAL-LIVE-011): no node could bind its
    ranks to EFA ports, VALIDATE_FABRIC waited for a scope nothing would fill
    until the 600 s per-step cap, and the workflow ended FAILED before
    ESCALATE_SUPPORT ran.
    """

    if not support_handoff_follows(context.workflow, context.step_index):
        return {}
    return unknown_task_fabric(context.step)


def unknown_task_fabric_details(context: WorkflowStepContext) -> dict[str, Any]:
    unknown = resolved_unknown_task_fabric(context)
    return {"task_fabric_unknown_nodes": unknown} if unknown else {}


def inventory_observation_window(
    context: WorkflowStepContext,
    recovery_operations: frozenset[WorkflowOperation],
) -> tuple[datetime, str] | None:
    """Since when a reading counts for the inventory gate, with the pending key
    that names the wait for one; ``None`` while the plan has one of
    ``recovery_operations`` ahead of this step that has not completed, which
    the gate waits for.

    After a completed recovery action (the driver, plugin, reboot and
    replacement steps) the reading must postdate its completion: a count
    cached before the repair proves nothing about it. A diagnostic chain
    (``RUN_DIAGNOSTICS``: FREEZE_EVIDENCE -> VALIDATE_FABRIC for an EFA
    ``LINK_INACTIVE`` / ``EXCESS_DEVICE`` / unknown failure mode) plans no
    such action, so there is no completion to wait for: the reading must
    postdate the moment the chain reached this step -- the last completed
    step's ``updated_at``, never earlier than the workflow's creation -- which
    excludes the collector's confirming samples and admits its next batch.
    Live 2026-10-25 (GF-GENERAL-LIVE-013 ``link_inactive``): the gate waited
    for an action completion no step would produce, failed at the 600 s
    per-step cap, and the cap's failure escalated a node whose port had been
    ACTIVE again for minutes to RESTART_NODE. Positional like the rest of the
    adapter's step arithmetic: the inventory chains are sequential.
    """

    workflow = context.workflow
    completed = [
        execution
        for execution in workflow.step_executions
        if execution.step_index < context.step_index
        and execution.status is WorkflowStepStatus.SUCCEEDED
    ]
    acted: list[datetime] = [
        item.updated_at for item in completed if item.operation in recovery_operations
    ]
    if acted:
        return max(acted), "missing_post_action_inventory_metric"
    steps = (
        workflow.safety_steps
        if workflow.executes_safety_steps
        else workflow.official_steps
    )
    if any(
        step.operation in recovery_operations for step in steps[: context.step_index]
    ):
        return None
    reached: list[datetime] = [
        workflow.created_at,
        *(item.updated_at for item in completed),
    ]
    return max(reached), "missing_fresh_inventory_metric"


def inventory_details(
    clamped: dict[str, Any], evidence: dict[str, Any]
) -> dict[str, Any]:
    """The inventory gate's record on a verdict: the configured counts it
    capped and the readings it judged."""

    return {
        **({"inventory_requirement_clamped": clamped} if clamped else {}),
        **({"inventory_evidence": evidence} if evidence else {}),
    }


def success_evidence(
    context: WorkflowStepContext,
    clamped: dict[str, Any],
    evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        **validation_evidence(
            context.step.operation,
            clamped,
            resolved_unknown_task_fabric(context),
            context.step.node_ids,
        ),
        **inventory_details({}, evidence or {}),
    }


#: How many findings the failure text spells out per node; the rest are counted
#: and every one is in ``details.node_findings``.
FAILURE_TEXT_FINDINGS = 3


def quiesce_taint_pending(
    store: Any, context: WorkflowStepContext, node_id: str
) -> dict[str, Any]:
    """Why GPU telemetry may be missing: a ``gpu-fault.io/quiesce`` taint a
    workflow of this incident's lineage applied to ``node_id`` and never
    recorded removing.

    The validation adapter has no cluster client; the step records do carry
    the taint's history (``quiesce_taint_nodes`` / ``quiesce_taint_removed_nodes``
    on the QUIESCE and RESTORE executions). Live (gpu-eks-cluster, XID 46): the
    reset chain's compensation was fenced, its NoExecute taint stayed, the
    exporter was evicted, and the reboot successor's VALIDATE_GPU "stayed
    non-terminal for 602s" with ``missing_recent_metrics`` as its only word.
    The wait is the same; its reason is now named, so the operator strips the
    taint (``submit-remediation --disposition restore`` of the owner incident)
    instead of waiting out the cap. Lineage: this workflow, its
    ``predecessor_workflow_id`` chain and the ``<rung>-after-<workflow>`` origins
    of its incidents, bounded; an unreadable record simply contributes nothing.
    """

    holders: dict[str, str] = {}
    frontier: list[str] = [context.workflow.request_id]
    seen: set[str] = set()
    while frontier and len(seen) < 16:
        request_id = frontier.pop()
        if request_id in seen:
            continue
        seen.add(request_id)
        if request_id == context.workflow.request_id:
            workflow = context.workflow
        else:
            try:
                workflow = store.get_workflow(request_id)
            except Exception:  # noqa: BLE001 - lineage evidence is best effort
                continue
        if node_id in quiesce_taint_nodes_held(workflow):
            holders[workflow.request_id] = quarantine_taint_value(workflow.incident_id)
        if workflow.predecessor_workflow_id:
            frontier.append(workflow.predecessor_workflow_id)
        try:
            incident = (
                context.incident
                if workflow.incident_id == context.incident.incident_id
                else store.get_incident(workflow.incident_id)
            )
        except Exception:  # noqa: BLE001 - lineage evidence is best effort
            continue
        origin = escalation_origin(incident.event_id)
        if origin is not None:
            frontier.append(origin[1])
    if not holders:
        return {}
    return {
        "quiesce_taint_present": True,
        "quiesce_taint_holders": holders,
        "reason": (
            "the node still carries the gpu-fault quiesce taint (NoExecute) of "
            + ", ".join(sorted(holders))
            + "; the exporter cannot run until it is removed -- restore the owner "
            "incident (gpu-fault-admin submit-remediation --disposition restore)"
        ),
    }


def finding_summary(finding: Any) -> dict[str, Any]:
    """The probe/metric identity of an active finding, for the step record."""

    severity = getattr(finding, "severity", None)
    return {
        "finding_id": getattr(finding, "finding_id", None),
        "canonical_name": getattr(finding, "canonical_name", None),
        "severity": getattr(severity, "value", severity),
        "value": getattr(finding, "value", None),
        "threshold_value": getattr(finding, "threshold_value", None),
        "gpu_uuid": getattr(finding, "gpu_uuid", None),
        "pci_bdf": getattr(finding, "pci_bdf", None),
    }


def _finding_text(summary: dict[str, Any]) -> str:
    parts = [
        str(summary.get("canonical_name") or summary.get("finding_id") or "finding")
    ]
    if summary.get("severity"):
        parts.append(str(summary["severity"]))
    value, threshold = summary.get("value"), summary.get("threshold_value")
    # A T.Limit margin counts down: it breaches at or below its threshold.
    relation = "<=" if summary.get("canonical_name") == TLIMIT_READING else ">"
    if isinstance(value, (int, float)):
        parts.append(
            f"{value:g}"
            + (
                f"{relation}{threshold:g}"
                if isinstance(threshold, (int, float))
                else ""
            )
        )
    if summary.get("gpu_uuid"):
        parts.append(str(summary["gpu_uuid"]))
    return " ".join(parts)


def node_failure_text(
    node_id: str,
    reasons: Sequence[str],
    node_findings: dict[str, list[dict[str, Any]]],
) -> str:
    """``node (reason, ...)`` with the active findings named by probe/metric.

    Live 2026-10-05 (gpu-eks-cluster, ip-10-0-41-164): VALIDATE_GPU failed on a
    ``temperature_validation`` finding and the step said only ``validation
    failed on nodes: ip-10-0-41-164``; the operator had to open the finding
    store to learn which check failed. The text now carries what
    ``details.node_findings`` records: name, severity, value against threshold
    and the GPU.
    """

    items = []
    for reason in reasons:
        findings = node_findings.get(node_id) or []
        if reason != "active_gpu_health_findings" or not findings:
            items.append(reason)
            continue
        shown = [_finding_text(item) for item in findings[:FAILURE_TEXT_FINDINGS]]
        more = len(findings) - len(shown)
        items.append(
            "active_gpu_health_findings["
            + "; ".join(shown)
            + (f"; +{more} more" if more > 0 else "")
            + "]"
        )
    return f"{node_id} ({', '.join(items)})"


def _warning_only(findings, names: frozenset[str]) -> bool:
    return all(
        getattr(finding, "canonical_name", None) in names
        and getattr(getattr(finding, "severity", None), "value", None) == "WARNING"
        for finding in findings
    )


def step_started_at(context: WorkflowStepContext) -> datetime | None:
    """When this step's clock started, off its live execution record.

    ``record_attempt`` keeps the first attempt's ``started_at`` across WAITING
    polls, and ``step_bounds.bounded_waiting_outcome`` measures the per-step
    cap from it (clamped to the execution window, which can only push the cap
    later). ``None`` on the step's very first attempt, before any record.
    """

    return next(
        (
            execution.started_at
            for execution in reversed(context.workflow.step_executions)
            if execution.step_index == context.step_index
            and execution.operation is context.step.operation
        ),
        None,
    )


def finding_unjudged_since(finding: Any, grace_end: datetime) -> bool:
    """Has no GPU metrics batch newer than the finding's current state re-judged it?

    The ingest replaces a finding it re-grades (a newer ``observed_at``) and
    drops one it clears, so an active finding whose ``observed_at`` is still
    at or before the grace end is exactly the state the grace end saw: nothing
    newer has judged it -- no later batch, or a later batch whose counter reset
    the ingest refused to grade. A finding without a comparable ``observed_at``
    cannot be dated and keeps the fixed-grace verdict. Duck-typed on purpose:
    the module's ``datetime`` name is a test clock in several suites.
    """

    observed_at = getattr(finding, "observed_at", None)
    if observed_at is None:
        return False
    try:
        return bool(observed_at <= grace_end)
    except TypeError:
        return False


def finding_device_key(finding: Any) -> str:
    """The device key a finding's metric samples are kept under
    (``gpu_observation.metric_device_key``): GPU UUID, else PCI BDF, else the
    node."""

    return str(
        getattr(finding, "gpu_uuid", None)
        or getattr(finding, "pci_bdf", None)
        or "node"
    )


def newest_gpu_sample_times(latest: Sequence[Any]) -> dict[tuple[str, str], datetime]:
    """``(device key, metric) -> observed_at`` of the node's latest GPU samples."""

    newest: dict[tuple[str, str], datetime] = {}
    for item in latest:
        sample = getattr(item, "sample", None)
        observed_at = getattr(item, "observed_at", None)
        if sample is None or observed_at is None:
            continue
        key = (metric_device_key(sample), str(sample.canonical_name))
        if key not in newest or observed_at > newest[key]:
            newest[key] = observed_at
    return newest


def awaiting_newer_gpu_metrics_reason(
    entries: Sequence[dict[str, Any]], grace_end: datetime, step_cap_at: datetime
) -> str:
    """The operator-facing sentence for a wait on newer GPU metrics: which
    finding, when its current state was observed, whether a newer sample
    already arrived without re-judging it (a counter reset), that the grace
    has elapsed, and when the executor's per-step cap ends the wait."""

    return (
        "awaiting a GPU metrics batch newer than "
        + awaited_findings_text(entries)
        + f" to re-judge it; grace elapsed {grace_end.isoformat()}, "
        f"step cap {step_cap_at.isoformat()}"
    )


def awaiting_newer_gpu_metrics(
    metrics: Any,
    step_waiting_cap: timedelta,
    context: WorkflowStepContext,
    node_id: str,
    findings: Sequence[Any],
    grace_end: datetime,
    now: datetime,
) -> dict[str, Any]:
    """The wait a WARNING-only finding earns past the grace while nothing newer
    than its current state has re-judged it.

    A growth finding (``thermal_violation_total_us`` by
    ``thermal_duration_decision``, the power and correctable-memory
    counterparts) describes the interval that ends at the batch that minted or
    re-graded it. Whether the GPU is healthy *now* is decided by the next
    GPU_METRICS batch for the node: it replaces the finding (newer
    ``observed_at``, a new ``finding_id``) when the growth goes on, clears it
    when the counter is flat, and leaves it untouched when the counter reset
    (an invalid delta is not graded). The collector delivers edges only on
    confirmation and recovery, and otherwise one health summary per node on
    its deterministic stable phase (270-330 s apart), so until that batch
    lands the finding's verdict is the one the grace end already saw.

    The fixed grace failed here instead. Live, one physical sequence read two
    ways: plain EKS 2026-10-26 (workflows created 23:24:17.98Z) PASSED at
    23:24:49Z because a summary landed between the confirmed and the recovered
    reads and the recovered edge was a zero delta; HyperPod 2026-10-27 (created
    01:48:01.5Z) FAILED at the grace end 01:50:01.6Z with
    ``active_gpu_health_findings[thermal_violation_total_us WARNING
    3e+06>10000]`` on the finding re-minted by the recovered edge at 01:48:32Z
    -- the only newer batch, the 01:49:47Z summary, carried a counter reset and
    could not re-judge it. The verdict followed the node's summary phase
    (b201's sweep: 51 of 60 phases inconclusive for the same data), not the
    data. So past the grace the step keeps WAITING, names what it waits for,
    and leaves the bound to the executor's per-step cap
    (``step_bounds.bounded_waiting_outcome``, 600 s by default), which no wait
    can exceed; ``step_waiting_cap`` is that configured limit, named here so
    the record says when the wait ends at the latest. A finding whose current
    state postdates the grace end *was* re-judged by newer data and is still
    active: that fails as before.
    """

    entries, step_cap_at = awaited_gpu_metrics(
        metrics, step_waiting_cap, context, node_id, findings, now
    )
    return {
        "awaiting_newer_gpu_metrics": [entry["finding_id"] for entry in entries],
        "awaiting_newer_gpu_metrics_findings": entries,
        "grace_elapsed_at": grace_end.isoformat(),
        "step_cap_at": step_cap_at.isoformat(),
        "reason": awaiting_newer_gpu_metrics_reason(entries, grace_end, step_cap_at),
    }


def awaiting_reset_rejudgement(
    metrics: Any,
    step_waiting_cap: timedelta,
    context: WorkflowStepContext,
    node_id: str,
    findings: Sequence[Any],
    now: datetime,
) -> dict[str, Any] | None:
    """The wait pending-repair flags earn after this workflow's GPU reset or
    node reboot while no batch newer than the reset has re-judged them (b209,
    ``pending_repair_rejudge``); ``None`` when the existing verdict stands."""

    reset = pending_repair_reset(context, node_id, findings)
    if reset is None or not all(
        finding_unjudged_since(item, reset.updated_at) for item in findings
    ):
        return None
    entries, step_cap_at = awaited_gpu_metrics(
        metrics, step_waiting_cap, context, node_id, findings, now
    )
    return pending_repair_wait(entries, reset, step_cap_at)


def awaited_gpu_metrics(
    metrics: Any,
    step_waiting_cap: timedelta,
    context: WorkflowStepContext,
    node_id: str,
    findings: Sequence[Any],
    now: datetime,
) -> tuple[list[dict[str, Any]], datetime]:
    """The findings a wait for newer GPU metrics names, each with the newest
    sample of its metric when that one is newer and did not re-judge it, and
    when the executor's per-step cap ends the wait."""

    newest = newest_gpu_sample_times(
        metrics.latest(context.incident.cluster_id, node_id)
    )
    started = step_started_at(context)
    step_cap_at = (started if started is not None else now) + step_waiting_cap
    entries: list[dict[str, Any]] = []
    for finding in findings:
        latest = newest.get((finding_device_key(finding), finding.canonical_name))
        entries.append(
            {
                "finding_id": finding.finding_id,
                "canonical_name": finding.canonical_name,
                "gpu_uuid": finding.gpu_uuid,
                "observed_at": finding.observed_at.isoformat(),
                "newest_sample_observed_at": (
                    latest.isoformat()
                    if latest is not None and latest > finding.observed_at
                    else None
                ),
            }
        )
    return entries, step_cap_at


_VALIDATIONS = frozenset(
    {
        WorkflowOperation.VALIDATE_GPU,
        WorkflowOperation.VALIDATE_HOST,
        WorkflowOperation.VALIDATE_FABRIC,
    }
)


def telemetry_readiness_after(context: WorkflowStepContext) -> datetime | None:
    """The completion of the last ``TELEMETRY_READINESS_OPERATIONS`` step ahead
    of this validation, after which its telemetry starts over; ``None`` without
    one."""

    if context.step.operation not in _VALIDATIONS:
        return None
    barriers = [
        execution.updated_at
        for execution in context.workflow.step_executions
        if (
            execution.step_index < context.step_index
            and execution.status is WorkflowStepStatus.SUCCEEDED
            and execution.operation in TELEMETRY_READINESS_OPERATIONS
        )
    ]
    return max(barriers) if barriers else None


def required_collectors_for(
    operation: WorkflowOperation, *, host_telemetry: bool
) -> set[CollectorKind]:
    """The collectors a validation needs fresh before it judges a node;
    ``host_telemetry`` adds the host collector (an inventory requirement, a
    completed node recovery, a host diagnostic probe)."""

    if operation is WorkflowOperation.VALIDATE_HOST:
        required = {CollectorKind.HOST_TELEMETRY}
    elif operation is WorkflowOperation.VALIDATE_FABRIC:
        required = {CollectorKind.GPU_METRICS, CollectorKind.HOST_TELEMETRY}
    else:
        required = {CollectorKind.GPU_METRICS}
    if host_telemetry:
        required.add(CollectorKind.HOST_TELEMETRY)
    return required

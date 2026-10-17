from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timedelta, timezone
from typing import Any

from gpu_fault.cluster_scope import FABRIC_MANAGER_UNIT
from gpu_fault.dcgm_fields import missing_fabric_metric_groups
from gpu_fault.execution import (
    WorkflowStepContext,
    WorkflowStepOutcome,
)
from gpu_fault.gpu_fabric_manager import fabric_validation
from gpu_fault.gpu_instance_inventory import gpu_instance_inventory
from gpu_fault.gpu_metrics import GpuMetricSource, GpuMetricsService
from gpu_fault.models import (
    WorkflowOperation,
    WorkflowStepSpec,
    WorkflowStepStatus,
)
from gpu_fault.operation_registry import (
    OperationAdapter,
    operations_for_adapter,
)
from gpu_fault.adapters.common import quarantine_taint_value
from gpu_fault.orchestration.escalation import escalation_origin
from gpu_fault.orchestration.validated_restore import is_validated_restore_workflow
from gpu_fault.store.shared.errors import NotFoundError
from gpu_fault.telemetry import (
    CollectorKind,
)
from gpu_fault.workflow_quarantine import quiesce_taint_nodes_held


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

    def supports(self, step: WorkflowStepSpec) -> bool:
        return step.execution_owner == self.owner and step.operation in self.OPERATIONS

    def execute(self, context: WorkflowStepContext) -> WorkflowStepOutcome:
        now = datetime.now(timezone.utc)
        # DCGM omits XID_ERRORS on healthy GPUs on some driver/DCGM
        # combinations. Fresh temperature proves that GPU telemetry has
        # resumed; an XID series, when present, is still evaluated below.
        required = {"gpu_temperature_c"}
        pending: dict[str, dict[str, list[str]]] = {}
        failures: dict[str, list[str]] = {}
        node_findings: dict[str, list[dict[str, Any]]] = {}
        clamped: dict[str, dict[str, Any]] = {}
        operation = context.step.operation
        raw_inventory_requirements = context.step.parameters.get(
            "inventory_requirements_by_node", {}
        )
        inventory_requirements = (
            raw_inventory_requirements
            if isinstance(raw_inventory_requirements, dict)
            else {}
        )
        gpu_inventory_recovery_operations = {
            WorkflowOperation.RESTART_NODE,
            WorkflowOperation.REPLACE_NODE,
            WorkflowOperation.RESTART_GPU_DEVICE_PLUGIN,
        }
        efa_inventory_recovery_operations = {
            WorkflowOperation.REMEDIATE_EFA_DRIVER,
            WorkflowOperation.RESTART_EFA_DEVICE_PLUGIN,
            WorkflowOperation.RESTART_NODE,
            WorkflowOperation.REPLACE_NODE,
        }
        telemetry_readiness_operations = {
            WorkflowOperation.RESET_GPU,
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
        inventory_recovery_operations = (
            efa_inventory_recovery_operations
            if operation is WorkflowOperation.VALIDATE_FABRIC
            else gpu_inventory_recovery_operations
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
        readiness_after = None
        if operation in {
            WorkflowOperation.VALIDATE_GPU,
            WorkflowOperation.VALIDATE_HOST,
            WorkflowOperation.VALIDATE_FABRIC,
        }:
            readiness_barriers = [
                execution.updated_at
                for execution in context.workflow.step_executions
                if (
                    execution.step_index < context.step_index
                    and execution.status is WorkflowStepStatus.SUCCEEDED
                    and execution.operation in telemetry_readiness_operations
                )
            ]
            if readiness_barriers:
                readiness_after = max(readiness_barriers)
        if operation is WorkflowOperation.VALIDATE_HOST:
            required_collectors = {CollectorKind.HOST_TELEMETRY}
        elif operation is WorkflowOperation.VALIDATE_FABRIC:
            required_collectors = {
                CollectorKind.GPU_METRICS,
                CollectorKind.HOST_TELEMETRY,
            }
            required = set()
        else:
            required_collectors = {CollectorKind.GPU_METRICS}
        if inventory_requirements or completed_node_recovery:
            required_collectors.add(CollectorKind.HOST_TELEMETRY)
        for node_id in context.step.node_ids:
            recent = self._recent_host_metrics(
                context,
                node_id,
                now,
                readiness_after,
                required_collectors,
            )
            if isinstance(recent, dict):
                dark = self._restore_collectors_dark(context, node_id, recent, now)
                if dark is not None:
                    return dark
                pending[node_id] = {
                    **recent,
                    **quiesce_taint_pending(self.store, context, node_id),
                }
                continue
            all_host_latest, host_latest = recent
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
                    host_latest, node_id, failures
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
        if pending:
            return WorkflowStepOutcome.waiting(
                details={
                    "pending_nodes": sorted(pending),
                    "node_pending": pending,
                    "failed_nodes_observed": sorted(failures),
                    "node_failures_observed": failures,
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
                    **({"node_findings": node_findings} if node_findings else {}),
                    **({"inventory_requirement_clamped": clamped} if clamped else {}),
                },
            )
        return WorkflowStepOutcome.succeeded(
            operation_id=context.idempotency_key,
            details={
                "validated_nodes": context.step.node_ids,
                "validation": context.step.operation.value,
                **({"inventory_requirement_clamped": clamped} if clamped else {}),
            },
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
        started = next(
            (
                execution.started_at
                for execution in reversed(context.workflow.step_executions)
                if execution.step_index == context.step_index
                and execution.operation is context.step.operation
            ),
            None,
        )
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
    ):
        max_age = (
            self.post_action_max_sample_age
            if readiness_after is not None
            else self.max_sample_age
        )
        statuses = {
            item.collector: item
            for item in self.store.list_collector_statuses(
                context.incident.cluster_id, node_id
            )
        }
        stale = sorted(
            collector.value
            for collector in required_collectors
            if collector not in statuses
            or statuses[collector].last_success_at is None
            or now - statuses[collector].last_success_at > max_age
            or (
                readiness_after is not None
                and statuses[collector].last_success_at <= readiness_after
            )
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
        return all_latest, latest

    def _validate_inventory(
        self,
        context,
        node_id,
        operation,
        requirements,
        recovery_operations,
        completed_recovery,
        all_latest,
        latest,
        failures,
        clamped=None,
    ):
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
                requirement = {"metrics": {active_name: expected[-1].value}}
        if not isinstance(requirement, dict):
            return None
        required = self._required_inventory_metrics(requirement, operation)
        required, bounded = self._bound_by_instance_inventory(requirement, required)
        if bounded and clamped is not None:
            clamped[node_id] = bounded
        executions = [
            item
            for item in context.workflow.step_executions
            if item.step_index < context.step_index
            and item.status is WorkflowStepStatus.SUCCEEDED
            and item.operation in recovery_operations
        ]
        if required and not executions:
            return {
                "missing_inventory_action_completion": [
                    item.value
                    for item in sorted(
                        recovery_operations,
                        key=lambda value: value.value,
                    )
                ]
            }
        observed_after = (
            max(item.updated_at for item in executions) if executions else None
        )
        for metric, expected_count in required.items():
            samples = [
                item
                for item in latest
                if item.name == metric
                and observed_after is not None
                and item.observed_at > observed_after
            ]
            if not samples:
                return {
                    "missing_post_action_inventory_metric": [metric],
                    "observed_after": [observed_after.isoformat()],
                }
            actual = samples[-1].value
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
        names = {item.name for item in latest}
        missing = sorted(
            {
                "load1_per_cpu",
                "memory_used_percent",
                "filesystem_used_percent",
            }
            - names
        )
        if missing:
            return {"missing_recent_host_metrics": missing}
        thresholds = {
            "cpu_usage_percent": 98,
            "load1_per_cpu": 2,
            "memory_used_percent": 95,
            "swap_used_percent": 80,
            "filesystem_used_percent": 98,
            "shared_filesystem_unavailable": 1,
            "shared_filesystem_used_percent": 98,
            "disk_io_util_percent": 98,
            "disk_io_await_ms": 100,
            "smart_health_failed": 1,
            "network_link_down": 1,
            "bmc_critical_sensor": 1,
        }
        unhealthy = sorted(
            {
                item.name
                for item in latest
                if item.name in thresholds and item.value >= thresholds[item.name]
            }
        )
        if unhealthy:
            failures.setdefault(node_id, []).extend(unhealthy)
        return None

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

    def _validate_fabric_metrics(self, latest, node_id, failures):
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
        if not any(value >= 1 for value in link_states):
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
        """Every GPU of a Hopper NVSwitch node joined the NVLink fabric, and the
        node's Fabric Manager unit runs.

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
        if failure is not None:
            failures.setdefault(node_id, []).append(failure)
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
        thermal_names = {
            "gpu_temperature_c",
            "memory_temperature_c",
            "thermal_violation_total_us",
            "clock_throttle_reasons",
            "composite:THERMAL_STRESS",
        }
        memory_names = {
            "ecc_sbe_volatile_total",
            "ecc_sbe_aggregate_total",
            "retired_pages_sbe_total",
            "row_remap_correctable_total",
            "composite:CORRECTABLE_MEMORY_DEGRADATION",
        }
        power_names = {
            "power_violation_total_us",
            "composite:POWER_LIMIT_THROTTLING",
        }
        warning_grace = None
        if _warning_only(findings, thermal_names):
            warning_grace = self.temperature_warning_grace
        elif _warning_only(findings, memory_names) or _warning_only(
            findings, power_names
        ):
            warning_grace = self.transient_warning_grace
        if (
            operation is WorkflowOperation.VALIDATE_GPU
            and warning_grace is not None
            and now < context.workflow.created_at + warning_grace
        ):
            return {
                "transient_gpu_warning_cooldown": [
                    finding.finding_id for finding in findings
                ]
            }
        failures.setdefault(node_id, []).append("active_gpu_health_findings")
        if node_findings is not None:
            node_findings[node_id] = [finding_summary(item) for item in findings]
        return None


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
    if isinstance(value, (int, float)):
        parts.append(
            f"{value:g}"
            + (f">{threshold:g}" if isinstance(threshold, (int, float)) else "")
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


def _warning_only(findings, names: set[str]) -> bool:
    return all(
        getattr(finding, "canonical_name", None) in names
        and getattr(getattr(finding, "severity", None), "value", None) == "WARNING"
        for finding in findings
    )

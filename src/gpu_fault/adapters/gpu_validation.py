from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from gpu_fault.dcgm_fields import missing_fabric_metric_groups
from gpu_fault.execution import (
    WorkflowStepContext,
    WorkflowStepOutcome,
)
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
from gpu_fault.telemetry import (
    CollectorKind,
)


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
                pending[node_id] = recent
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
                pending[node_id] = metric_pending
                continue
            if operation is WorkflowOperation.VALIDATE_FABRIC:
                fabric_pending = self._validate_fabric_metrics(
                    host_latest, node_id, failures
                )
                if fabric_pending is not None:
                    pending[node_id] = fabric_pending
                    continue
            finding_pending = self._validate_active_findings(
                context, node_id, operation, now, failures
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
                "validation failed on nodes: " + ", ".join(failed_nodes),
                details={
                    "failed_nodes": failed_nodes,
                    "node_failures": failures,
                    "validation": operation.value,
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
        # every one of them must have reported for itself.
        by_gpu: dict[str, list[str]] = {}
        if required and context.step.gpu_uuids:
            fresh_by_gpu: dict[str, set[str]] = {}
            for item in fresh:
                gpu_uuid = getattr(item.sample, "gpu_uuid", None)
                if gpu_uuid:
                    fresh_by_gpu.setdefault(str(gpu_uuid), set()).add(
                        item.sample.canonical_name
                    )
            for gpu_uuid in dict.fromkeys(context.step.gpu_uuids):
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

    def _validate_active_findings(
        self,
        context,
        node_id,
        operation,
        now,
        failures,
    ):
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
        transient_names = {
            # Correctable memory degradation self-clears as ECC scrubs and
            # row-remap absorbs the affected cells.
            "ecc_sbe_volatile_total",
            "ecc_sbe_aggregate_total",
            "retired_pages_sbe_total",
            "row_remap_correctable_total",
            "composite:CORRECTABLE_MEMORY_DEGRADATION",
            # Power-limit throttling is a benign WARNING that self-clears once
            # the workload's power draw falls back under the enforced limit;
            # its automatic action is only RUN_DIAGNOSTICS. Without a grace
            # window here, a throttle still active at VALIDATE_GPU time fails
            # validation and escalates the diagnostic workflow into a DRAIN.
            "power_violation_total_us",
            "composite:POWER_LIMIT_THROTTLING",
        }
        warning_grace = None
        if _warning_only(findings, thermal_names):
            warning_grace = self.temperature_warning_grace
        elif _warning_only(findings, transient_names):
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
        return None


def _warning_only(findings, names: set[str]) -> bool:
    return all(
        getattr(finding, "canonical_name", None) in names
        and getattr(getattr(finding, "severity", None), "value", None) == "WARNING"
        for finding in findings
    )

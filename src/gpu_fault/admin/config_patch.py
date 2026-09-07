"""Authoring a desired AdminConfig: camelCase patches and named presets.

Everything here produces a *desired* config, so it validates in full,
including the capacity rules the perf evidence fixes (see
``capacity_evidence``). Reading a recorded or live state is
``AdminConfig.from_mapping`` in ``config`` and deliberately is not.

The 32/50 presets model the perf plan's multi-cluster topology
(性能压测验收方案 §2): N clusters x 256 nodes. At 12,800 nodes the plan's own
§13.3 run needed Min ACU pre-provisioned near 128 to stop returning 503, so a
preset carries the Aurora floor of the topology it names instead of the 8 ACU
single-cluster default.
"""

from __future__ import annotations

from dataclasses import replace

from gpu_fault.admin.capacity_evidence import aurora_min_acu_floor
from gpu_fault.admin.config import (
    AdminConfig,
    AdminConfigError,
    AuroraCapacityConfig,
    CapacityConfig,
    EvidenceConfig,
    NotificationDeliveryConfig,
    ProcessorConfig,
    RemediationCapacity,
    TelemetrySpoolCapacity,
    WorkflowConfig,
    _integer,
    _mapping,
    _number,
    default_admin_config,
)
from gpu_fault.admin.config_parser import boolean_field

PRESET_NODES_PER_CLUSTER = 256
PRESET_MAX_ACU = 128.0


def preset_admin_config(name: str) -> AdminConfig:
    normalized = name.strip().lower()
    if normalized == "default":
        return default_admin_config()
    match normalized:
        case "32-disabled":
            clusters = 32
            spool = TelemetrySpoolCapacity(enabled=False, replicas=0)
        case "32-enabled":
            clusters = 32
            spool = TelemetrySpoolCapacity(enabled=True, replicas=3)
        case "50-disabled":
            clusters = 50
            spool = TelemetrySpoolCapacity(enabled=False, replicas=0)
        case "50-enabled":
            clusters = 50
            spool = TelemetrySpoolCapacity(enabled=True, replicas=3)
        case _:
            raise AdminConfigError(
                "unknown capacity preset; expected one of default, "
                "32-disabled, 32-enabled, 50-disabled, or 50-enabled"
            )
    managed_nodes: int = clusters * PRESET_NODES_PER_CLUSTER
    min_acu: float = aurora_min_acu_floor(managed_nodes)
    config = AdminConfig(
        capacity=CapacityConfig(
            control_worker_replicas=6,
            telemetry_spool=spool,
            remediation=RemediationCapacity(
                max_active_region=clusters * 4,
                max_active_per_cluster=4,
                max_active_per_node=1,
                max_active_per_failure_domain=1,
                max_active_per_resource_class=4,
            ),
            largest_cluster_node_count=PRESET_NODES_PER_CLUSTER,
            managed_node_count=managed_nodes,
        ),
        aurora=AuroraCapacityConfig(
            min_acu=min_acu,
            max_acu=max(PRESET_MAX_ACU, min_acu),
        ),
    )
    config.validate()
    return config


def _preset_aurora(
    current: AuroraCapacityConfig,
    preset: AdminConfig,
) -> AuroraCapacityConfig:
    """Aurora after choosing a preset: raised to the preset's floor, never lowered.

    A preset names a topology, and the topology fixes the Min ACU floor. The
    change is proposed in the plan like any other and applied through the
    audited RDS path; only an unrecorded change would be a fabrication.
    """

    if current.min_acu >= preset.aurora.min_acu:
        return current
    return AuroraCapacityConfig(
        min_acu=preset.aurora.min_acu,
        max_acu=max(current.max_acu, preset.aurora.max_acu),
    )


def apply_capacity_patch(
    base: AdminConfig,
    value: object,
    *,
    path: str = "spec.capacity",
    validate: bool = True,
) -> AdminConfig:
    """Apply a camelCase capacity patch on ``base``.

    ``validate=False`` is for a caller that applies the rest of the document
    afterwards and validates the whole: a file may raise the node counts and
    the Aurora floor together, and the half-applied config is not the one to
    judge.
    """

    data = _mapping(
        value or {},
        path,
        allowed={
            "preset",
            "controlWorkerReplicas",
            "telemetrySpool",
            "remediation",
            "largestClusterNodeCount",
            "managedNodeCount",
        },
    )
    raw_preset = data.get("preset")
    if raw_preset is not None and (
        not isinstance(raw_preset, str) or not raw_preset.strip()
    ):
        raise AdminConfigError(f"{path}.preset must be a non-empty string")
    aurora = base.aurora
    if isinstance(raw_preset, str):
        preset = preset_admin_config(raw_preset)
        current_capacity = preset.capacity
        aurora = _preset_aurora(base.aurora, preset)
    else:
        current_capacity = base.capacity
    spool_data = _mapping(
        data.get("telemetrySpool") or {},
        f"{path}.telemetrySpool",
        allowed={"enabled", "replicas"},
    )
    remediation_data = _mapping(
        data.get("remediation") or {},
        f"{path}.remediation",
        allowed={
            "maxActiveRegion",
            "maxActivePerCluster",
            "maxActivePerResourceClass",
        },
    )
    current_spool = current_capacity.telemetry_spool
    current_remediation = current_capacity.remediation
    config = replace(
        base,
        aurora=aurora,
        capacity=CapacityConfig(
            control_worker_replicas=_integer(
                data.get("controlWorkerReplicas"),
                f"{path}.controlWorkerReplicas",
                default=current_capacity.control_worker_replicas,
            ),
            telemetry_spool=TelemetrySpoolCapacity(
                enabled=boolean_field(
                    spool_data.get("enabled"),
                    f"{path}.telemetrySpool.enabled",
                    default=current_spool.enabled,
                    error=AdminConfigError,
                ),
                replicas=_integer(
                    spool_data.get("replicas"),
                    f"{path}.telemetrySpool.replicas",
                    default=current_spool.replicas,
                ),
            ),
            remediation=RemediationCapacity(
                max_active_region=_integer(
                    remediation_data.get("maxActiveRegion"),
                    f"{path}.remediation.maxActiveRegion",
                    default=current_remediation.max_active_region,
                ),
                max_active_per_cluster=_integer(
                    remediation_data.get("maxActivePerCluster"),
                    f"{path}.remediation.maxActivePerCluster",
                    default=current_remediation.max_active_per_cluster,
                ),
                max_active_per_node=current_remediation.max_active_per_node,
                max_active_per_failure_domain=(
                    current_remediation.max_active_per_failure_domain
                ),
                max_active_per_resource_class=_integer(
                    remediation_data.get("maxActivePerResourceClass"),
                    f"{path}.remediation.maxActivePerResourceClass",
                    default=(current_remediation.max_active_per_resource_class),
                ),
            ),
            largest_cluster_node_count=_integer(
                data.get("largestClusterNodeCount"),
                f"{path}.largestClusterNodeCount",
                default=current_capacity.largest_cluster_node_count,
            ),
            managed_node_count=_integer(
                data.get("managedNodeCount"),
                f"{path}.managedNodeCount",
                default=current_capacity.managed_node_count,
            ),
        ),
    )
    if validate:
        config.validate()
    return config


def apply_admin_config_patch(
    base: AdminConfig,
    value: object,
    *,
    path: str = "spec",
) -> AdminConfig:
    data = _mapping(
        value or {},
        path,
        allowed={
            "capacity",
            "aurora",
            "processor",
            "workflow",
            "notificationDelivery",
            "evidence",
        },
    )
    # Not validated on its own: the file may raise the node counts and the
    # Aurora floor (or the queue depth) together, and only the complete
    # config below can be judged.
    current = (
        apply_capacity_patch(
            base,
            data.get("capacity"),
            path=f"{path}.capacity",
            validate=False,
        )
        if "capacity" in data
        else base
    )
    aurora_data = _mapping(
        data.get("aurora") or {},
        f"{path}.aurora",
        allowed={"minAcu", "maxAcu"},
    )
    processor_data = _mapping(
        data.get("processor") or {},
        f"{path}.processor",
        allowed={
            "maxQueueDepth",
            "maxClusterQueueDepth",
            "retryAfterSeconds",
            "retryBackoffSeconds",
            "retryBackoffMaxSeconds",
            "completedRetentionSeconds",
        },
    )
    workflow_data = _mapping(
        data.get("workflow") or {},
        f"{path}.workflow",
        allowed={"pollIntervalSeconds", "dispatcherWorkers"},
    )
    notification_data = _mapping(
        data.get("notificationDelivery") or {},
        f"{path}.notificationDelivery",
        allowed={
            "batchSize",
            "maxAttempts",
        },
    )
    evidence_data = _mapping(
        data.get("evidence") or {},
        f"{path}.evidence",
        allowed={"retentionHours", "maxRecordsPerNode"},
    )
    processor = current.processor
    aurora = current.aurora
    workflow = current.workflow
    notification = current.notification_delivery
    evidence = current.evidence
    max_queue_depth = _integer(
        processor_data.get("maxQueueDepth"),
        f"{path}.processor.maxQueueDepth",
        default=processor.max_queue_depth,
    )
    max_cluster_queue_depth = _integer(
        processor_data.get("maxClusterQueueDepth"),
        f"{path}.processor.maxClusterQueueDepth",
        default=processor.max_cluster_queue_depth,
    )
    retry_backoff_max_seconds = _integer(
        processor_data.get("retryBackoffMaxSeconds"),
        f"{path}.processor.retryBackoffMaxSeconds",
        default=processor.retry_backoff_max_seconds,
    )
    config = AdminConfig(
        capacity=current.capacity,
        aurora=AuroraCapacityConfig(
            min_acu=_number(
                aurora_data.get("minAcu"),
                f"{path}.aurora.minAcu",
                default=aurora.min_acu,
            ),
            max_acu=_number(
                aurora_data.get("maxAcu"),
                f"{path}.aurora.maxAcu",
                default=aurora.max_acu,
            ),
        ),
        processor=ProcessorConfig(
            max_queue_depth=max_queue_depth,
            max_cluster_queue_depth=max_cluster_queue_depth,
            retry_after_seconds=_integer(
                processor_data.get("retryAfterSeconds"),
                f"{path}.processor.retryAfterSeconds",
                default=processor.retry_after_seconds,
            ),
            retry_backoff_seconds=_integer(
                processor_data.get("retryBackoffSeconds"),
                f"{path}.processor.retryBackoffSeconds",
                default=processor.retry_backoff_seconds,
            ),
            retry_backoff_max_seconds=retry_backoff_max_seconds,
            completed_retention_seconds=_integer(
                processor_data.get("completedRetentionSeconds"),
                f"{path}.processor.completedRetentionSeconds",
                default=processor.completed_retention_seconds,
            ),
        ),
        workflow=WorkflowConfig(
            poll_interval_seconds=_number(
                workflow_data.get("pollIntervalSeconds"),
                f"{path}.workflow.pollIntervalSeconds",
                default=workflow.poll_interval_seconds,
            ),
            dispatcher_workers=_integer(
                workflow_data.get("dispatcherWorkers"),
                f"{path}.workflow.dispatcherWorkers",
                default=workflow.dispatcher_workers,
            ),
        ),
        notification_delivery=NotificationDeliveryConfig(
            batch_size=_integer(
                notification_data.get("batchSize"),
                f"{path}.notificationDelivery.batchSize",
                default=notification.batch_size,
            ),
            max_attempts=_integer(
                notification_data.get("maxAttempts"),
                f"{path}.notificationDelivery.maxAttempts",
                default=notification.max_attempts,
            ),
        ),
        evidence=EvidenceConfig(
            retention_hours=_integer(
                evidence_data.get("retentionHours"),
                f"{path}.evidence.retentionHours",
                default=evidence.retention_hours,
            ),
            max_records_per_node=_integer(
                evidence_data.get("maxRecordsPerNode"),
                f"{path}.evidence.maxRecordsPerNode",
                default=evidence.max_records_per_node,
            ),
        ),
    )
    config.validate()
    return config

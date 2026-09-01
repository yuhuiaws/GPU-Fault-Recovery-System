from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterator, Mapping


ADMIN_CONFIG_API_VERSION = "gpu-fault.aws/v1alpha1"
ADMIN_CONFIG_KIND = "AdminConfig"
ADMIN_CONFIG_ROOT = Path("admin-config")
ADMIN_CONFIG_DESIRED = ADMIN_CONFIG_ROOT / "desired.json"
ADMIN_CONFIG_PLAN = ADMIN_CONFIG_ROOT / "plan.json"
ADMIN_CONFIG_APPROVAL = ADMIN_CONFIG_ROOT / "approval.json"
ADMIN_CONFIG_HISTORY = ADMIN_CONFIG_ROOT / "history"
ADMIN_CONFIG_LOCK = ADMIN_CONFIG_ROOT / "admin-config.lock"
APPROVAL_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{2,127}$")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
ADMIN_CONFIG_ROLES = ("ingress", "worker", "spool")


class AdminConfigError(ValueError):
    pass


@dataclass(frozen=True)
class RemediationCapacity:
    max_active_region: int = 20
    max_active_per_cluster: int = 5
    max_active_per_node: int = 1
    max_active_per_failure_domain: int = 1
    max_active_per_resource_class: int = 2

    def validate(self) -> None:
        bounds = {
            "maxActiveRegion": (self.max_active_region, 1, 4096),
            "maxActivePerCluster": (self.max_active_per_cluster, 1, 128),
            "maxActivePerNode": (self.max_active_per_node, 1, 1),
            "maxActivePerFailureDomain": (
                self.max_active_per_failure_domain,
                1,
                1,
            ),
            "maxActivePerResourceClass": (
                self.max_active_per_resource_class,
                1,
                128,
            ),
        }
        for name, (value, minimum, maximum) in bounds.items():
            if not minimum <= value <= maximum:
                raise AdminConfigError(
                    f"spec.capacity.remediation.{name} must be within "
                    f"{minimum}..{maximum}"
                )
        if self.max_active_per_cluster > self.max_active_region:
            raise AdminConfigError(
                "maxActivePerCluster must not exceed maxActiveRegion"
            )
        if self.max_active_per_resource_class > self.max_active_region:
            raise AdminConfigError(
                "maxActivePerResourceClass must not exceed maxActiveRegion"
            )

    def as_dict(self) -> dict[str, int]:
        return {
            "max_active_region": self.max_active_region,
            "max_active_per_cluster": self.max_active_per_cluster,
            "max_active_per_node": self.max_active_per_node,
            "max_active_per_failure_domain": (self.max_active_per_failure_domain),
            "max_active_per_resource_class": self.max_active_per_resource_class,
        }


@dataclass(frozen=True)
class TelemetrySpoolCapacity:
    enabled: bool = False
    replicas: int = 0

    def validate(self) -> None:
        if not 0 <= self.replicas <= 32:
            raise AdminConfigError(
                "spec.capacity.telemetrySpool.replicas must be within 0..32"
            )
        if self.enabled and self.replicas < 1:
            raise AdminConfigError(
                "enabled telemetry spool requires at least one spool replica"
            )
        if not self.enabled and self.replicas != 0:
            raise AdminConfigError("disabled telemetry spool requires replicas=0")

    def as_dict(self) -> dict[str, object]:
        return {
            "enabled": self.enabled,
            "replicas": self.replicas,
        }


@dataclass(frozen=True)
class CapacityConfig:
    control_worker_replicas: int = 6
    telemetry_spool: TelemetrySpoolCapacity = TelemetrySpoolCapacity()
    remediation: RemediationCapacity = RemediationCapacity()

    def validate(self) -> None:
        if not 1 <= self.control_worker_replicas <= 64:
            raise AdminConfigError(
                "spec.capacity.controlWorkerReplicas must be within 1..64"
            )
        self.telemetry_spool.validate()
        self.remediation.validate()
        ceiling = self.postgres_connection_ceiling()
        budget = self.postgres_fleet_connection_budget()
        if ceiling > budget:
            raise AdminConfigError(
                "control-plane PostgreSQL connection ceiling "
                f"{ceiling} exceeds the validated fleet budget {budget}"
            )

    def postgres_fleet_connection_budget(self) -> int:
        return 1240 if self.telemetry_spool.enabled else 1200

    def postgres_connection_ceiling(self) -> int:
        ingress_pool = 48 if self.telemetry_spool.enabled else 40
        ingress = ingress_pool * 4 * 3
        worker_processes = self.control_worker_replicas * 4
        worker = 24 * worker_processes + worker_processes
        spool = self.telemetry_spool.replicas * (12 + 1)
        return ingress + worker + spool

    def as_dict(self) -> dict[str, object]:
        return {
            "control_worker_replicas": self.control_worker_replicas,
            "telemetry_spool": self.telemetry_spool.as_dict(),
            "remediation": self.remediation.as_dict(),
        }


@dataclass(frozen=True)
class ProcessorConfig:
    max_queue_depth: int = 65536
    max_cluster_queue_depth: int = 1024
    retry_after_seconds: int = 2
    retry_backoff_seconds: int = 1
    retry_backoff_max_seconds: int = 30
    completed_retention_seconds: int = 600

    def validate(self) -> None:
        bounds = {
            "maxQueueDepth": (self.max_queue_depth, 1024, 1_000_000),
            "maxClusterQueueDepth": (
                self.max_cluster_queue_depth,
                1,
                100_000,
            ),
            "retryAfterSeconds": (self.retry_after_seconds, 1, 60),
            "retryBackoffSeconds": (
                self.retry_backoff_seconds,
                1,
                60,
            ),
            "retryBackoffMaxSeconds": (
                self.retry_backoff_max_seconds,
                1,
                600,
            ),
            "completedRetentionSeconds": (
                self.completed_retention_seconds,
                60,
                604800,
            ),
        }
        for name, (value, minimum, maximum) in bounds.items():
            if not minimum <= value <= maximum:
                raise AdminConfigError(
                    f"spec.processor.{name} must be within {minimum}..{maximum}"
                )
        if self.max_cluster_queue_depth > self.max_queue_depth:
            raise AdminConfigError("maxClusterQueueDepth must not exceed maxQueueDepth")
        if self.retry_backoff_max_seconds < self.retry_backoff_seconds:
            raise AdminConfigError(
                "retryBackoffMaxSeconds must not be less than retryBackoffSeconds"
            )
        if self.completed_retention_seconds < max(
            300,
            self.retry_backoff_max_seconds,
        ):
            raise AdminConfigError(
                "completedRetentionSeconds must cover retryable response age"
            )

    def as_dict(self) -> dict[str, int]:
        return {
            "max_queue_depth": self.max_queue_depth,
            "max_cluster_queue_depth": self.max_cluster_queue_depth,
            "retry_after_seconds": self.retry_after_seconds,
            "retry_backoff_seconds": self.retry_backoff_seconds,
            "retry_backoff_max_seconds": self.retry_backoff_max_seconds,
            "completed_retention_seconds": self.completed_retention_seconds,
        }


@dataclass(frozen=True)
class WorkflowConfig:
    poll_interval_seconds: float = 5.0
    dispatcher_workers: int = 8

    def validate(self) -> None:
        if not 0.5 <= self.poll_interval_seconds <= 60:
            raise AdminConfigError(
                "spec.workflow.pollIntervalSeconds must be within 0.5..60"
            )
        if not 1 <= self.dispatcher_workers <= 64:
            raise AdminConfigError(
                "spec.workflow.dispatcherWorkers must be within 1..64"
            )

    def as_dict(self) -> dict[str, object]:
        return {
            "poll_interval_seconds": self.poll_interval_seconds,
            "dispatcher_workers": self.dispatcher_workers,
        }


@dataclass(frozen=True)
class NotificationDeliveryConfig:
    batch_size: int = 25
    max_attempts: int = 8

    def validate(self) -> None:
        if not 1 <= self.batch_size <= 1000:
            raise AdminConfigError(
                "spec.notificationDelivery.batchSize must be within 1..1000"
            )
        if not 1 <= self.max_attempts <= 32:
            raise AdminConfigError(
                "spec.notificationDelivery.maxAttempts must be within 1..32"
            )

    def as_dict(self) -> dict[str, object]:
        return {
            "batch_size": self.batch_size,
            "max_attempts": self.max_attempts,
        }


@dataclass(frozen=True)
class EvidenceConfig:
    retention_hours: int = 24
    max_records_per_node: int = 10000

    def validate(self) -> None:
        if not 1 <= self.retention_hours <= 8760:
            raise AdminConfigError(
                "spec.evidence.retentionHours must be within 1..8760"
            )
        if not 100 <= self.max_records_per_node <= 1_000_000:
            raise AdminConfigError(
                "spec.evidence.maxRecordsPerNode must be within 100..1000000"
            )

    def as_dict(self) -> dict[str, int]:
        return {
            "retention_hours": self.retention_hours,
            "max_records_per_node": self.max_records_per_node,
        }


@dataclass(frozen=True)
class AdminConfig:
    capacity: CapacityConfig = CapacityConfig()
    processor: ProcessorConfig = ProcessorConfig()
    workflow: WorkflowConfig = WorkflowConfig()
    notification_delivery: NotificationDeliveryConfig = NotificationDeliveryConfig()
    evidence: EvidenceConfig = EvidenceConfig()

    def validate(self) -> None:
        self.capacity.validate()
        self.processor.validate()
        self.workflow.validate()
        self.notification_delivery.validate()
        self.evidence.validate()

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "capacity": self.capacity.as_dict(),
            "processor": self.processor.as_dict(),
            "workflow": self.workflow.as_dict(),
            "notification_delivery": self.notification_delivery.as_dict(),
            "evidence": self.evidence.as_dict(),
        }

    def sha256(self) -> str:
        return canonical_sha256(self.as_dict())

    def role_payload(self, role: str) -> dict[str, object]:
        common = {
            "processor": self.processor.as_dict(),
            "workflow": self.workflow.as_dict(),
            "notification_delivery": self.notification_delivery.as_dict(),
            "evidence": self.evidence.as_dict(),
        }
        if role == "ingress":
            return {
                **common,
                "telemetry_spool_enabled": (self.capacity.telemetry_spool.enabled),
            }
        if role == "worker":
            return {
                **common,
                "control_worker_replicas": (self.capacity.control_worker_replicas),
                "remediation": self.capacity.remediation.as_dict(),
            }
        if role == "spool":
            return {
                **common,
                "telemetry_spool": self.capacity.telemetry_spool.as_dict(),
            }
        raise AdminConfigError(f"unknown control-plane role: {role}")

    def role_sha256(self) -> dict[str, str]:
        return {
            role: canonical_sha256(self.role_payload(role))
            for role in ADMIN_CONFIG_ROLES
        }

    @classmethod
    def from_mapping(cls, value: object) -> AdminConfig:
        data = _mapping(
            value,
            "admin config",
            allowed={
                "schema_version",
                "capacity",
                "processor",
                "workflow",
                "notification_delivery",
                "evidence",
            },
        )
        if data.get("schema_version", 1) != 1:
            raise AdminConfigError("admin config schema_version must be 1")
        capacity_data = _mapping(
            data.get("capacity") or {},
            "admin config capacity",
            allowed={
                "control_worker_replicas",
                "telemetry_spool",
                "remediation",
            },
        )
        spool_data = _mapping(
            capacity_data.get("telemetry_spool") or {},
            "admin config telemetry_spool",
            allowed={"enabled", "replicas"},
        )
        remediation_data = _mapping(
            capacity_data.get("remediation") or {},
            "admin config remediation",
            allowed={
                "max_active_region",
                "max_active_per_cluster",
                "max_active_per_node",
                "max_active_per_failure_domain",
                "max_active_per_resource_class",
            },
        )
        processor_data = _mapping(
            data.get("processor") or {},
            "admin config processor",
            allowed={
                "max_queue_depth",
                "max_cluster_queue_depth",
                "retry_after_seconds",
                "retry_backoff_seconds",
                "retry_backoff_max_seconds",
                "completed_retention_seconds",
            },
        )
        workflow_data = _mapping(
            data.get("workflow") or {},
            "admin config workflow",
            allowed={
                "poll_interval_seconds",
                "dispatcher_workers",
            },
        )
        notification_data = _mapping(
            data.get("notification_delivery") or {},
            "admin config notification_delivery",
            allowed={
                "batch_size",
                "max_attempts",
            },
        )
        evidence_data = _mapping(
            data.get("evidence") or {},
            "admin config evidence",
            allowed={"retention_hours", "max_records_per_node"},
        )
        config = cls(
            capacity=CapacityConfig(
                control_worker_replicas=_integer(
                    capacity_data.get("control_worker_replicas"),
                    "admin config capacity.control_worker_replicas",
                    default=6,
                ),
                telemetry_spool=TelemetrySpoolCapacity(
                    enabled=_boolean(
                        spool_data.get("enabled"),
                        "admin config telemetry_spool.enabled",
                        default=False,
                    ),
                    replicas=_integer(
                        spool_data.get("replicas"),
                        "admin config telemetry_spool.replicas",
                        default=0,
                    ),
                ),
                remediation=RemediationCapacity(
                    max_active_region=_integer(
                        remediation_data.get("max_active_region"),
                        "admin config remediation.max_active_region",
                        default=20,
                    ),
                    max_active_per_cluster=_integer(
                        remediation_data.get("max_active_per_cluster"),
                        "admin config remediation.max_active_per_cluster",
                        default=5,
                    ),
                    max_active_per_node=_integer(
                        remediation_data.get("max_active_per_node"),
                        "admin config remediation.max_active_per_node",
                        default=1,
                    ),
                    max_active_per_failure_domain=_integer(
                        remediation_data.get("max_active_per_failure_domain"),
                        ("admin config remediation.max_active_per_failure_domain"),
                        default=1,
                    ),
                    max_active_per_resource_class=_integer(
                        remediation_data.get("max_active_per_resource_class"),
                        ("admin config remediation.max_active_per_resource_class"),
                        default=2,
                    ),
                ),
            ),
            processor=ProcessorConfig(
                max_queue_depth=_integer(
                    processor_data.get("max_queue_depth"),
                    "admin config processor.max_queue_depth",
                    default=65536,
                ),
                max_cluster_queue_depth=_integer(
                    processor_data.get("max_cluster_queue_depth"),
                    "admin config processor.max_cluster_queue_depth",
                    default=1024,
                ),
                retry_after_seconds=_integer(
                    processor_data.get("retry_after_seconds"),
                    "admin config processor.retry_after_seconds",
                    default=2,
                ),
                retry_backoff_seconds=_integer(
                    processor_data.get("retry_backoff_seconds"),
                    "admin config processor.retry_backoff_seconds",
                    default=1,
                ),
                retry_backoff_max_seconds=_integer(
                    processor_data.get("retry_backoff_max_seconds"),
                    "admin config processor.retry_backoff_max_seconds",
                    default=30,
                ),
                completed_retention_seconds=_integer(
                    processor_data.get("completed_retention_seconds"),
                    "admin config processor.completed_retention_seconds",
                    default=600,
                ),
            ),
            workflow=WorkflowConfig(
                poll_interval_seconds=_number(
                    workflow_data.get("poll_interval_seconds"),
                    "admin config workflow.poll_interval_seconds",
                    default=5.0,
                ),
                dispatcher_workers=_integer(
                    workflow_data.get("dispatcher_workers"),
                    "admin config workflow.dispatcher_workers",
                    default=8,
                ),
            ),
            notification_delivery=NotificationDeliveryConfig(
                batch_size=_integer(
                    notification_data.get("batch_size"),
                    "admin config notification_delivery.batch_size",
                    default=25,
                ),
                max_attempts=_integer(
                    notification_data.get("max_attempts"),
                    "admin config notification_delivery.max_attempts",
                    default=8,
                ),
            ),
            evidence=EvidenceConfig(
                retention_hours=_integer(
                    evidence_data.get("retention_hours"),
                    "admin config evidence.retention_hours",
                    default=24,
                ),
                max_records_per_node=_integer(
                    evidence_data.get("max_records_per_node"),
                    "admin config evidence.max_records_per_node",
                    default=10000,
                ),
            ),
        )
        config.validate()
        return config


@dataclass(frozen=True)
class PreparedAdminConfigApply:
    plan: dict[str, Any]
    config: AdminConfig
    reference: str
    no_op: bool


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def _mapping(
    value: object,
    path: str,
    *,
    allowed: set[str],
) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise AdminConfigError(f"{path} must be a mapping")
    normalized = value
    unknown = sorted(set(normalized) - allowed)
    if unknown:
        raise AdminConfigError(f"{path} contains unknown fields: {', '.join(unknown)}")
    return normalized


def _integer(value: object, path: str, *, default: int) -> int:
    if value is None:
        return default
    if not isinstance(value, int) or isinstance(value, bool):
        raise AdminConfigError(f"{path} must be an integer")
    return value


def _number(value: object, path: str, *, default: float) -> float:
    if value is None:
        return default
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise AdminConfigError(f"{path} must be a number")
    return float(value)


def _boolean(value: object, path: str, *, default: bool) -> bool:
    if value is None:
        return default
    if not isinstance(value, bool):
        raise AdminConfigError(f"{path} must be a boolean")
    return value


def default_admin_config() -> AdminConfig:
    return AdminConfig()


def preset_admin_config(name: str) -> AdminConfig:
    normalized = name.strip().lower()
    if normalized == "default":
        return default_admin_config()
    match normalized:
        case "32-disabled":
            region = 128
            spool = TelemetrySpoolCapacity(enabled=False, replicas=0)
        case "32-enabled":
            region = 128
            spool = TelemetrySpoolCapacity(enabled=True, replicas=3)
        case "50-disabled":
            region = 200
            spool = TelemetrySpoolCapacity(enabled=False, replicas=0)
        case "50-enabled":
            region = 200
            spool = TelemetrySpoolCapacity(enabled=True, replicas=3)
        case _:
            raise AdminConfigError(
                "unknown capacity preset; expected one of default, "
                "32-disabled, 32-enabled, 50-disabled, or 50-enabled"
            )
    config = AdminConfig(
        capacity=CapacityConfig(
            control_worker_replicas=6,
            telemetry_spool=spool,
            remediation=RemediationCapacity(
                max_active_region=region,
                max_active_per_cluster=4,
                max_active_per_node=1,
                max_active_per_failure_domain=1,
                max_active_per_resource_class=4,
            ),
        )
    )
    config.validate()
    return config


def apply_capacity_patch(
    base: AdminConfig,
    value: object,
    *,
    path: str = "spec.capacity",
) -> AdminConfig:
    data = _mapping(
        value or {},
        path,
        allowed={
            "preset",
            "controlWorkerReplicas",
            "telemetrySpool",
            "remediation",
        },
    )
    raw_preset = data.get("preset")
    if raw_preset is not None and (
        not isinstance(raw_preset, str) or not raw_preset.strip()
    ):
        raise AdminConfigError(f"{path}.preset must be a non-empty string")
    current_capacity = (
        preset_admin_config(raw_preset).capacity
        if isinstance(raw_preset, str)
        else base.capacity
    )
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
        capacity=CapacityConfig(
            control_worker_replicas=_integer(
                data.get("controlWorkerReplicas"),
                f"{path}.controlWorkerReplicas",
                default=current_capacity.control_worker_replicas,
            ),
            telemetry_spool=TelemetrySpoolCapacity(
                enabled=_boolean(
                    spool_data.get("enabled"),
                    f"{path}.telemetrySpool.enabled",
                    default=current_spool.enabled,
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
        ),
    )
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
            "processor",
            "workflow",
            "notificationDelivery",
            "evidence",
        },
    )
    current = (
        apply_capacity_patch(
            base,
            data.get("capacity"),
            path=f"{path}.capacity",
        )
        if "capacity" in data
        else base
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


def admin_config_desired_path(state_dir: Path) -> Path:
    return state_dir.expanduser().resolve() / ADMIN_CONFIG_DESIRED


def admin_config_plan_path(state_dir: Path) -> Path:
    return state_dir.expanduser().resolve() / ADMIN_CONFIG_PLAN


def admin_config_approval_path(state_dir: Path) -> Path:
    return state_dir.expanduser().resolve() / ADMIN_CONFIG_APPROVAL


def admin_config_history_path(state_dir: Path, plan_sha256: str) -> Path:
    if not SHA256_PATTERN.fullmatch(plan_sha256):
        raise AdminConfigError("admin config plan SHA-256 is invalid")
    return state_dir.expanduser().resolve() / ADMIN_CONFIG_HISTORY / plan_sha256


def _utc_timestamp(value: datetime | None = None) -> str:
    return (value or datetime.now(UTC)).isoformat()


def _read_json(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AdminConfigError(f"{description} is invalid") from exc
    if not isinstance(value, dict):
        raise AdminConfigError(f"{description} must be a JSON object")
    return value


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.parent.chmod(0o700)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        text=True,
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


@contextmanager
def admin_config_lock(state_dir: Path) -> Iterator[None]:
    root = state_dir.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    root.chmod(0o700)
    path = root / ADMIN_CONFIG_LOCK
    path.parent.mkdir(parents=True, exist_ok=True)
    path.parent.chmod(0o700)
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        os.fchmod(descriptor, 0o600)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise AdminConfigError(
                "another admin config operation is in progress"
            ) from exc
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _desired_record(
    config: AdminConfig,
    *,
    source: str,
    updated_at: datetime | None = None,
    plan_sha256: str | None = None,
    reference: str | None = None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "schema_version": 1,
        "config": config.as_dict(),
        "config_sha256": config.sha256(),
        "role_sha256": config.role_sha256(),
        "source": source,
        "updated_at": _utc_timestamp(updated_at),
    }
    if plan_sha256 is not None:
        record["plan_sha256"] = plan_sha256
    if reference is not None:
        record["reference"] = reference
    return record


def load_desired_admin_config(state_dir: Path) -> AdminConfig:
    path = admin_config_desired_path(state_dir)
    if not path.is_file():
        return default_admin_config()
    record = _read_json(path, "desired admin config")
    if record.get("schema_version") != 1:
        raise AdminConfigError("desired admin config schema is invalid")
    config = AdminConfig.from_mapping(record.get("config"))
    if record.get("config_sha256") != config.sha256():
        raise AdminConfigError("desired admin config digest does not match its content")
    if record.get("role_sha256") != config.role_sha256():
        raise AdminConfigError(
            "desired admin config role digests do not match its content"
        )
    return config


def persist_desired_admin_config(
    state_dir: Path,
    *,
    config: AdminConfig,
    source: str,
) -> Path:
    path = admin_config_desired_path(state_dir)
    _write_json_atomic(path, _desired_record(config, source=source))
    return path


def _site_identity_sha256(value: object) -> str:
    data = _mapping(
        value,
        "admin config site_identity",
        allowed={"site_name", "aws_region", "cpu_eks_arn"},
    )
    normalized: dict[str, str] = {}
    for field in ("site_name", "aws_region", "cpu_eks_arn"):
        raw = data.get(field)
        if not isinstance(raw, str) or not raw.strip() or raw != raw.strip():
            raise AdminConfigError(
                f"admin config site_identity.{field} must be non-empty "
                "without whitespace padding"
            )
        normalized[field] = raw
    return canonical_sha256(normalized)


def _flatten(value: object, *, prefix: str = "") -> dict[str, object]:
    if not isinstance(value, Mapping):
        return {prefix: value}
    result: dict[str, object] = {}
    for key in sorted(value):
        path = f"{prefix}.{key}" if prefix else str(key)
        result.update(_flatten(value[key], prefix=path))
    return result


def _changes(
    current: AdminConfig,
    desired: AdminConfig,
) -> list[dict[str, object]]:
    before = _flatten(
        {
            key: value
            for key, value in current.as_dict().items()
            if key != "schema_version"
        }
    )
    after = _flatten(
        {
            key: value
            for key, value in desired.as_dict().items()
            if key != "schema_version"
        }
    )
    return [
        {
            "field": field,
            "before": before.get(field),
            "after": after.get(field),
        }
        for field in sorted(set(before) | set(after))
        if before.get(field) != after.get(field)
    ]


def admin_config_plan_sha256(document: dict[str, Any]) -> str:
    expected = {
        "schema_version",
        "site_identity",
        "site_identity_sha256",
        "release_identity",
        "current_config_sha256",
        "desired_config_sha256",
        "current_role_sha256",
        "desired_role_sha256",
        "current_config",
        "desired_config",
        "affected_roles",
        "changes",
        "source",
        "approval_required",
    }
    unknown = sorted(set(document) - expected - {"plan_sha256"})
    missing = sorted(expected - set(document))
    if unknown:
        raise AdminConfigError(
            "admin config plan contains unknown fields: " + ", ".join(unknown)
        )
    if missing:
        raise AdminConfigError(
            "admin config plan is missing fields: " + ", ".join(missing)
        )
    if document.get("schema_version") != 1:
        raise AdminConfigError("admin config plan schema is invalid")
    return canonical_sha256({field: document[field] for field in sorted(expected)})


def _validated_release_identity(value: object) -> dict[str, object]:
    data = _mapping(
        value,
        "admin config release_identity",
        allowed={"release_id", "manifest_sha256", "staging_only"},
    )
    release_id = data.get("release_id")
    manifest_sha256 = data.get("manifest_sha256")
    staging_only = data.get("staging_only")
    if not isinstance(release_id, str) or not release_id.strip():
        raise AdminConfigError(
            "admin config release_identity.release_id must be non-empty"
        )
    if not isinstance(manifest_sha256, str) or not SHA256_PATTERN.fullmatch(
        manifest_sha256
    ):
        raise AdminConfigError(
            "admin config release_identity.manifest_sha256 is invalid"
        )
    if not isinstance(staging_only, bool):
        raise AdminConfigError(
            "admin config release_identity.staging_only must be a boolean"
        )
    return {
        "release_id": release_id,
        "manifest_sha256": manifest_sha256,
        "staging_only": staging_only,
    }


def _validated_plan(document: dict[str, Any]) -> tuple[dict[str, Any], str]:
    identity_digest = _site_identity_sha256(document.get("site_identity"))
    if document.get("site_identity_sha256") != identity_digest:
        raise AdminConfigError(
            "admin config plan site identity digest does not match its content"
        )
    _validated_release_identity(document.get("release_identity"))
    current = AdminConfig.from_mapping(document.get("current_config"))
    desired = AdminConfig.from_mapping(document.get("desired_config"))
    if document.get("current_config_sha256") != current.sha256():
        raise AdminConfigError(
            "admin config plan current digest does not match its content"
        )
    if document.get("desired_config_sha256") != desired.sha256():
        raise AdminConfigError(
            "admin config plan desired digest does not match its content"
        )
    if document.get("current_role_sha256") != current.role_sha256():
        raise AdminConfigError("admin config plan current role digests are invalid")
    if document.get("desired_role_sha256") != desired.role_sha256():
        raise AdminConfigError("admin config plan desired role digests are invalid")
    affected = sorted(
        role
        for role in ADMIN_CONFIG_ROLES
        if current.role_sha256()[role] != desired.role_sha256()[role]
    )
    if document.get("affected_roles") != affected:
        raise AdminConfigError(
            "admin config plan affected_roles do not match its configuration"
        )
    changes = _changes(current, desired)
    if document.get("changes") != changes:
        raise AdminConfigError(
            "admin config plan changes do not match its configuration"
        )
    if document.get("approval_required") is not bool(changes):
        raise AdminConfigError(
            "admin config plan approval_required does not match its changes"
        )
    digest = admin_config_plan_sha256(document)
    if document.get("plan_sha256") != digest:
        raise AdminConfigError("admin config plan digest does not match its content")
    return document, digest


def _admin_config_plan_document(
    current: AdminConfig,
    *,
    site_identity: Mapping[str, str],
    release_identity: Mapping[str, object],
    desired: AdminConfig,
    source: str,
) -> dict[str, Any]:
    desired.validate()
    current_roles = current.role_sha256()
    desired_roles = desired.role_sha256()
    changes = _changes(current, desired)
    document: dict[str, Any] = {
        "schema_version": 1,
        "site_identity": dict(site_identity),
        "site_identity_sha256": _site_identity_sha256(dict(site_identity)),
        "release_identity": _validated_release_identity(dict(release_identity)),
        "current_config_sha256": current.sha256(),
        "desired_config_sha256": desired.sha256(),
        "current_role_sha256": current_roles,
        "desired_role_sha256": desired_roles,
        "current_config": current.as_dict(),
        "desired_config": desired.as_dict(),
        "affected_roles": sorted(
            role
            for role in ADMIN_CONFIG_ROLES
            if current_roles[role] != desired_roles[role]
        ),
        "changes": changes,
        "source": source,
        "approval_required": bool(changes),
    }
    document["plan_sha256"] = admin_config_plan_sha256(document)
    return document


def preview_admin_config_plan(
    state_dir: Path,
    *,
    site_identity: Mapping[str, str],
    release_identity: Mapping[str, object],
    desired: AdminConfig,
    source: str,
) -> dict[str, Any]:
    with admin_config_lock(state_dir):
        return _admin_config_plan_document(
            load_desired_admin_config(state_dir),
            site_identity=site_identity,
            release_identity=release_identity,
            desired=desired,
            source=source,
        )


def create_admin_config_plan(
    state_dir: Path,
    *,
    site_identity: Mapping[str, str],
    release_identity: Mapping[str, object],
    desired: AdminConfig,
    source: str,
) -> dict[str, Any]:
    with admin_config_lock(state_dir):
        document = _admin_config_plan_document(
            load_desired_admin_config(state_dir),
            site_identity=site_identity,
            release_identity=release_identity,
            desired=desired,
            source=source,
        )
        approval_path = admin_config_approval_path(state_dir)
        if approval_path.is_file():
            previous_plan = load_admin_config_plan(state_dir)
            previous_approval = _validated_approval_record(
                _read_json(approval_path, "admin config approval"),
                plan=previous_plan,
            )
            if previous_plan["plan_sha256"] != document["plan_sha256"]:
                archive = _archive_plan_and_approval(
                    state_dir,
                    plan=previous_plan,
                    approval=previous_approval,
                )
                _write_json_atomic(
                    archive / "superseded.json",
                    {
                        "schema_version": 1,
                        "status": "SUPERSEDED",
                        "plan_sha256": previous_plan["plan_sha256"],
                        "reference": previous_approval["reference"],
                        "replacement_plan_sha256": document["plan_sha256"],
                        "reason": "a new administrator config plan was generated",
                        "superseded_at": _utc_timestamp(),
                    },
                )
                approval_path.unlink()
        _write_json_atomic(admin_config_plan_path(state_dir), document)
        return document


def load_admin_config_plan(state_dir: Path) -> dict[str, Any]:
    path = admin_config_plan_path(state_dir)
    if not path.is_file():
        raise AdminConfigError(
            "no pending admin config plan; run gpu-fault-admin "
            "capacity plan or config plan first"
        )
    plan, _digest = _validated_plan(_read_json(path, "admin config plan"))
    return plan


def _archive_plan_and_approval(
    state_dir: Path,
    *,
    plan: dict[str, Any],
    approval: dict[str, Any],
) -> Path:
    archive = admin_config_history_path(
        state_dir,
        str(plan["plan_sha256"]),
    )
    for name, value in (("plan.json", plan), ("approval.json", approval)):
        path = archive / name
        if path.is_file():
            if _read_json(path, f"archived admin config {name}") != value:
                raise AdminConfigError(
                    f"archived admin config {name} differs from active state"
                )
        else:
            _write_json_atomic(path, value)
    return archive


def _validated_approval_record(
    document: dict[str, Any],
    *,
    plan: dict[str, Any],
) -> dict[str, Any]:
    expected = {
        "schema_version",
        "plan_sha256",
        "reference",
        "site_identity",
        "site_identity_sha256",
        "release_identity",
        "desired_config_sha256",
        "affected_roles",
        "approved_at",
    }
    if set(document) != expected or document.get("schema_version") != 1:
        raise AdminConfigError("admin config approval schema is invalid")
    reference = document.get("reference")
    if not isinstance(reference, str) or not APPROVAL_PATTERN.fullmatch(reference):
        raise AdminConfigError("admin config approval reference is invalid")
    if document.get("plan_sha256") != plan.get("plan_sha256"):
        raise AdminConfigError("admin config approval does not match the active plan")
    if document.get("site_identity") != plan.get("site_identity") or document.get(
        "site_identity_sha256"
    ) != plan.get("site_identity_sha256"):
        raise AdminConfigError(
            "admin config approval site identity does not match the plan"
        )
    if _validated_release_identity(
        document.get("release_identity")
    ) != _validated_release_identity(plan.get("release_identity")):
        raise AdminConfigError(
            "admin config approval release identity does not match the plan"
        )
    if document.get("desired_config_sha256") != plan.get(
        "desired_config_sha256"
    ) or document.get("affected_roles") != plan.get("affected_roles"):
        raise AdminConfigError("admin config approval target does not match the plan")
    approved_at = document.get("approved_at")
    if not isinstance(approved_at, str):
        raise AdminConfigError("admin config approval timestamp is invalid")
    try:
        timestamp = datetime.fromisoformat(approved_at.replace("Z", "+00:00"))
    except ValueError as exc:
        raise AdminConfigError("admin config approval timestamp is invalid") from exc
    if timestamp.tzinfo is None:
        raise AdminConfigError(
            "admin config approval timestamp must include a timezone"
        )
    return document


def prepare_admin_config_apply(
    state_dir: Path,
    *,
    expected_plan_sha256: str,
    reference: str,
    current_release_identity: Mapping[str, object],
    approved_at: datetime | None = None,
) -> PreparedAdminConfigApply:
    normalized_digest = expected_plan_sha256.strip()
    normalized_reference = reference.strip()
    if not SHA256_PATTERN.fullmatch(normalized_digest):
        raise AdminConfigError("reviewed admin config plan SHA-256 is invalid")
    if not APPROVAL_PATTERN.fullmatch(normalized_reference):
        raise AdminConfigError("admin config approval reference has an invalid format")
    with admin_config_lock(state_dir):
        plan = load_admin_config_plan(state_dir)
        digest = str(plan["plan_sha256"])
        if digest != normalized_digest:
            raise AdminConfigError(
                "pending admin config plan does not match the reviewed --plan-sha256"
            )
        if _validated_release_identity(
            dict(current_release_identity)
        ) != _validated_release_identity(plan["release_identity"]):
            raise AdminConfigError(
                "current signed release differs from the reviewed config plan"
            )
        desired = AdminConfig.from_mapping(plan["desired_config"])
        current = load_desired_admin_config(state_dir)
        if current.sha256() not in {
            str(plan["current_config_sha256"]),
            str(plan["desired_config_sha256"]),
        }:
            raise AdminConfigError(
                "persisted admin config changed after the plan was generated"
            )
        approval = {
            "schema_version": 1,
            "plan_sha256": digest,
            "reference": normalized_reference,
            "site_identity": dict(plan["site_identity"]),
            "site_identity_sha256": str(plan["site_identity_sha256"]),
            "release_identity": dict(plan["release_identity"]),
            "desired_config_sha256": str(plan["desired_config_sha256"]),
            "affected_roles": list(plan["affected_roles"]),
            "approved_at": _utc_timestamp(approved_at),
        }
        approval_path = admin_config_approval_path(state_dir)
        if approval_path.is_file():
            existing = _validated_approval_record(
                _read_json(
                    approval_path,
                    "admin config approval",
                ),
                plan=plan,
            )
            if existing.get("plan_sha256") != digest:
                raise AdminConfigError(
                    "a different admin config plan already has an approval"
                )
            if existing.get("reference") != normalized_reference:
                raise AdminConfigError(
                    "admin config plan already has a different approval reference"
                )
            approval = existing
        else:
            _write_json_atomic(approval_path, approval)
        _archive_plan_and_approval(
            state_dir,
            plan=plan,
            approval=approval,
        )
        no_op = not bool(plan["changes"])
        if not no_op and current.sha256() != desired.sha256():
            _write_json_atomic(
                admin_config_desired_path(state_dir),
                _desired_record(
                    desired,
                    source=f"approved-plan:{digest}",
                    plan_sha256=digest,
                    reference=normalized_reference,
                ),
            )
        return PreparedAdminConfigApply(
            plan=plan,
            config=desired,
            reference=normalized_reference,
            no_op=no_op,
        )


def complete_admin_config_apply(
    state_dir: Path,
    *,
    expected_plan_sha256: str,
    release_id: str,
    success: bool,
    error: str | None = None,
    completed_at: datetime | None = None,
) -> Path:
    with admin_config_lock(state_dir):
        plan = load_admin_config_plan(state_dir)
        digest = str(plan["plan_sha256"])
        if digest != expected_plan_sha256:
            raise AdminConfigError("active admin config plan changed during apply")
        approval = _validated_approval_record(
            _read_json(
                admin_config_approval_path(state_dir),
                "admin config approval",
            ),
            plan=plan,
        )
        archive = _archive_plan_and_approval(
            state_dir,
            plan=plan,
            approval=approval,
        )
        result = {
            "schema_version": 1,
            "status": "APPLIED" if success else "FAILED",
            "plan_sha256": digest,
            "reference": approval["reference"],
            "release_id": release_id,
            "desired_config_sha256": plan["desired_config_sha256"],
            "affected_roles": plan["affected_roles"],
            "completed_at": _utc_timestamp(completed_at),
        }
        if error:
            result["error"] = error
        result_path = archive / ("applied.json" if success else "failed.json")
        _write_json_atomic(result_path, result)
        if success:
            admin_config_plan_path(state_dir).unlink(missing_ok=True)
            admin_config_approval_path(state_dir).unlink(missing_ok=True)
        return result_path

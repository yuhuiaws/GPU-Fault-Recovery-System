from __future__ import annotations

from collections.abc import Mapping

from gpu_fault.admin.capacity_evidence import legacy_largest_cluster_node_count


class AdminConfigParseError(ValueError):
    pass


def _mapping(
    value: object,
    path: str,
    *,
    allowed: set[str],
) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise AdminConfigParseError(f"{path} must be a mapping")
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise AdminConfigParseError(
            f"{path} contains unknown fields: {', '.join(unknown)}"
        )
    return value


def _integer(value: object, path: str, *, default: int) -> int:
    if value is None:
        return default
    if not isinstance(value, int) or isinstance(value, bool):
        raise AdminConfigParseError(f"{path} must be an integer")
    return value


def _number(value: object, path: str, *, default: float) -> float:
    if value is None:
        return default
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise AdminConfigParseError(f"{path} must be a number")
    return float(value)


def boolean_field(
    value: object,
    path: str,
    *,
    default: bool,
    error: type[ValueError] = AdminConfigParseError,
) -> bool:
    """Validate an already-parsed YAML boolean; ``None`` means ``default``.

    This is the one rule for YAML switches across the admin modules. Each
    caller keeps its own exception family by passing ``error``, so the text a
    site-config reader raises is the same text an admin-config reader raises.
    """

    if value is None:
        return default
    if not isinstance(value, bool):
        raise error(f"{path} must be a boolean")
    return value


def _capacity(data: Mapping[str, object]) -> dict[str, object]:
    capacity = _mapping(
        data.get("capacity") or {},
        "admin config capacity",
        allowed={
            "control_worker_replicas",
            "telemetry_spool",
            "remediation",
            "largest_cluster_node_count",
            "managed_node_count",
        },
    )
    # A document from before the node counts existed is read as the topology
    # its own per-cluster depth was sized for (1024 -> 256 nodes), so a legacy
    # site loads valid instead of as a 512-node default its depth cannot hold.
    # A fresh document gets 512/512 from the same rule at the default depth.
    largest_cluster_node_count = _integer(
        capacity.get("largest_cluster_node_count"),
        "admin config capacity.largest_cluster_node_count",
        default=legacy_largest_cluster_node_count(
            _processor(data)["max_cluster_queue_depth"]
        ),
    )
    spool = _mapping(
        capacity.get("telemetry_spool") or {},
        "admin config telemetry_spool",
        allowed={"enabled", "replicas"},
    )
    remediation = _mapping(
        capacity.get("remediation") or {},
        "admin config remediation",
        allowed={
            "max_active_region",
            "max_active_per_cluster",
            "max_active_per_node",
            "max_active_per_failure_domain",
            "max_active_per_resource_class",
        },
    )
    return {
        "control_worker_replicas": _integer(
            capacity.get("control_worker_replicas"),
            "admin config capacity.control_worker_replicas",
            default=6,
        ),
        "telemetry_spool": {
            "enabled": boolean_field(
                spool.get("enabled"),
                "admin config telemetry_spool.enabled",
                default=False,
            ),
            "replicas": _integer(
                spool.get("replicas"),
                "admin config telemetry_spool.replicas",
                default=0,
            ),
        },
        "remediation": {
            "max_active_region": _integer(
                remediation.get("max_active_region"),
                "admin config remediation.max_active_region",
                default=20,
            ),
            "max_active_per_cluster": _integer(
                remediation.get("max_active_per_cluster"),
                "admin config remediation.max_active_per_cluster",
                default=5,
            ),
            "max_active_per_node": _integer(
                remediation.get("max_active_per_node"),
                "admin config remediation.max_active_per_node",
                default=1,
            ),
            "max_active_per_failure_domain": _integer(
                remediation.get("max_active_per_failure_domain"),
                "admin config remediation.max_active_per_failure_domain",
                default=1,
            ),
            "max_active_per_resource_class": _integer(
                remediation.get("max_active_per_resource_class"),
                "admin config remediation.max_active_per_resource_class",
                default=2,
            ),
        },
        "largest_cluster_node_count": largest_cluster_node_count,
        "managed_node_count": _integer(
            capacity.get("managed_node_count"),
            "admin config capacity.managed_node_count",
            default=largest_cluster_node_count,
        ),
    }


def _aurora(data: Mapping[str, object]) -> dict[str, float]:
    values = _mapping(
        data.get("aurora") or {},
        "admin config aurora",
        allowed={"min_acu", "max_acu"},
    )
    legacy = "aurora" not in data
    return {
        "min_acu": _number(
            values.get("min_acu"),
            "admin config aurora.min_acu",
            default=0.5 if legacy else 8.0,
        ),
        "max_acu": _number(
            values.get("max_acu"),
            "admin config aurora.max_acu",
            default=8.0 if legacy else 32.0,
        ),
    }


def _processor(data: Mapping[str, object]) -> dict[str, int]:
    values = _mapping(
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
    defaults = {
        "max_queue_depth": 65536,
        "max_cluster_queue_depth": 4096,
        "retry_after_seconds": 2,
        "retry_backoff_seconds": 1,
        "retry_backoff_max_seconds": 30,
        "completed_retention_seconds": 600,
    }
    return {
        field: _integer(
            values.get(field),
            f"admin config processor.{field}",
            default=default,
        )
        for field, default in defaults.items()
    }


def _workflow(data: Mapping[str, object]) -> dict[str, object]:
    values = _mapping(
        data.get("workflow") or {},
        "admin config workflow",
        allowed={"poll_interval_seconds", "dispatcher_workers"},
    )
    return {
        "poll_interval_seconds": _number(
            values.get("poll_interval_seconds"),
            "admin config workflow.poll_interval_seconds",
            default=5.0,
        ),
        "dispatcher_workers": _integer(
            values.get("dispatcher_workers"),
            "admin config workflow.dispatcher_workers",
            default=8,
        ),
    }


def _notification(data: Mapping[str, object]) -> dict[str, int]:
    values = _mapping(
        data.get("notification_delivery") or {},
        "admin config notification_delivery",
        allowed={"batch_size", "max_attempts"},
    )
    return {
        "batch_size": _integer(
            values.get("batch_size"),
            "admin config notification_delivery.batch_size",
            default=25,
        ),
        "max_attempts": _integer(
            values.get("max_attempts"),
            "admin config notification_delivery.max_attempts",
            default=8,
        ),
    }


def _evidence(data: Mapping[str, object]) -> dict[str, int]:
    values = _mapping(
        data.get("evidence") or {},
        "admin config evidence",
        allowed={"retention_hours", "max_records_per_node"},
    )
    return {
        "retention_hours": _integer(
            values.get("retention_hours"),
            "admin config evidence.retention_hours",
            default=24,
        ),
        "max_records_per_node": _integer(
            values.get("max_records_per_node"),
            "admin config evidence.max_records_per_node",
            default=10000,
        ),
    }


def parse_admin_config(value: object) -> dict[str, object]:
    data = _mapping(
        value,
        "admin config",
        allowed={
            "schema_version",
            "capacity",
            "aurora",
            "processor",
            "workflow",
            "notification_delivery",
            "evidence",
        },
    )
    if data.get("schema_version", 1) != 1:
        raise AdminConfigParseError("admin config schema_version must be 1")
    return {
        "capacity": _capacity(data),
        "aurora": _aurora(data),
        "processor": _processor(data),
        "workflow": _workflow(data),
        "notification_delivery": _notification(data),
        "evidence": _evidence(data),
    }

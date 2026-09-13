"""Reconstruct the administrator config a live control plane runs.

The previous-state snapshot a release writes before it mutates anything has to
record the administrator config the site ran, because a rollback restores what
that snapshot says. Everything a role ConfigMap or Deployment carries is read
from there; the fields no Kubernetes object holds come from the release state
the previous release recorded -- never from a parser default.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from gpu_fault.admin.config import (
    AdminConfig,
    AdminConfigError,
    default_admin_config,
)
from gpu_fault_release.regional_release_config import ReleaseError

__all__ = ["captured_admin_config"]


def _captured_node_counts(worker_core: dict[str, str]) -> dict[str, int]:
    """Node counts exactly as the live worker ConfigMap declares them.

    Absent keys stay absent: this config is what a rollback restores, and a
    default filled in here would be invented capacity, not captured state
    (the Aurora 0.5/8 fabrication had exactly that shape). The parser then
    reads the legacy topology from the captured depth.
    """

    captured: dict[str, int] = {}
    for name, key in (
        ("GPU_FAULT_CAPACITY_LARGEST_CLUSTER_NODE_COUNT", "largest_cluster_node_count"),
        ("GPU_FAULT_CAPACITY_MANAGED_NODE_COUNT", "managed_node_count"),
    ):
        if name in worker_core:
            captured[key] = int(worker_core[name])
    return captured


def _captured_aurora(release: Any, recorded: Any) -> dict[str, float]:
    """The Aurora window the live cluster runs, never a parser default.

    No ConfigMap carries it, and ``AdminConfig.from_mapping`` fills a missing
    block with the 0.5/8 ACU legacy default -- which a rollback then restored
    over a live 82/128 window and the next deploy reconciled the database down
    to (live 2026-09-13). A release never moves Aurora capacity: the ``config``
    command settles it before the release starts and restores its own
    ``before`` on failure. So the truthful sources are the window the previous
    release recorded and, failing that, the candidate's own desired window,
    which is what the cluster runs by the time the snapshot is taken.
    """

    if isinstance(recorded, Mapping):
        aurora = recorded.get("aurora")
        if isinstance(aurora, Mapping) and {"min_acu", "max_acu"} <= set(aurora):
            return {
                "min_acu": float(aurora["min_acu"]),
                "max_acu": float(aurora["max_acu"]),
            }
    current = getattr(getattr(release, "config", None), "admin_config", None)
    if current is None:
        raise AdminConfigError(
            "cannot capture the live Aurora window: neither the recorded release "
            "state nor the candidate configuration declares one"
        )
    return {"min_acu": current.aurora.min_acu, "max_acu": current.aurora.max_acu}


def captured_admin_config(
    release: Any,
    snapshots: dict[str, dict[str, str]],
    *,
    recorded: Mapping[str, Any] | None = None,
) -> AdminConfig:
    """Reconstruct the live administrator config for the previous-state snapshot.

    Everything a role ConfigMap or Deployment carries is read from there;
    ``recorded`` is the admin config the live release state wrote, the only
    source for the fields no ConfigMap holds (``_captured_aurora``).
    """

    try:
        defaults = default_admin_config()
        capacity_defaults = defaults.capacity
        worker = release._get_json(
            release._cpu(
                "-n",
                release.config.namespace,
                "get",
                "deployment",
                "gpu-fault-control-worker",
            )
        )
        spool = release._get_json(
            release._cpu(
                "-n",
                release.config.namespace,
                "get",
                "deployment",
                "gpu-fault-telemetry-spool-worker",
            )
        )
        ingress_telemetry = snapshots.get(
            "gpu-fault-api-ha-config-telemetry",
            {},
        )
        worker_core = snapshots.get("gpu-fault-control-worker-config-core", {})
        ingress_processor = snapshots.get(
            "gpu-fault-api-ha-config-processor",
            {},
        )
        ingress_recovery = snapshots.get(
            "gpu-fault-api-ha-config-recovery",
            {},
        )
        ingress_notification = snapshots.get(
            "gpu-fault-api-ha-config-notification",
            {},
        )
        spool_enabled = ingress_telemetry.get(
            "GPU_FAULT_TELEMETRY_SPOOL",
            str(capacity_defaults.telemetry_spool.enabled).lower(),
        )
        if spool_enabled not in {"true", "false"}:
            raise AdminConfigError("live ingress telemetry spool value is invalid")
        worker_replicas = (worker.get("spec") or {}).get("replicas")
        spool_replicas = (spool.get("spec") or {}).get("replicas")
        remediation = capacity_defaults.remediation
        processor = defaults.processor
        workflow = defaults.workflow
        notification = defaults.notification_delivery
        evidence = defaults.evidence
        return AdminConfig.from_mapping(
            {
                "schema_version": 1,
                "aurora": _captured_aurora(release, recorded),
                "capacity": {
                    "control_worker_replicas": int(
                        capacity_defaults.control_worker_replicas
                        if worker_replicas is None
                        else worker_replicas
                    ),
                    "telemetry_spool": {
                        "enabled": spool_enabled == "true",
                        "replicas": int(
                            capacity_defaults.telemetry_spool.replicas
                            if spool_replicas is None
                            else spool_replicas
                        ),
                    },
                    "remediation": {
                        "max_active_region": int(
                            worker_core.get(
                                "GPU_FAULT_REMEDIATION_MAX_ACTIVE_REGION",
                                remediation.max_active_region,
                            )
                        ),
                        "max_active_per_cluster": int(
                            worker_core.get(
                                "GPU_FAULT_REMEDIATION_MAX_ACTIVE_PER_CLUSTER",
                                remediation.max_active_per_cluster,
                            )
                        ),
                        "max_active_per_node": int(
                            worker_core.get(
                                "GPU_FAULT_REMEDIATION_MAX_ACTIVE_PER_NODE",
                                remediation.max_active_per_node,
                            )
                        ),
                        "max_active_per_failure_domain": int(
                            worker_core.get(
                                ("GPU_FAULT_REMEDIATION_MAX_ACTIVE_PER_FAILURE_DOMAIN"),
                                remediation.max_active_per_failure_domain,
                            )
                        ),
                        "max_active_per_resource_class": int(
                            worker_core.get(
                                ("GPU_FAULT_REMEDIATION_MAX_ACTIVE_PER_RESOURCE_CLASS"),
                                remediation.max_active_per_resource_class,
                            )
                        ),
                    },
                    **_captured_node_counts(worker_core),
                },
                "processor": {
                    "max_queue_depth": int(
                        ingress_processor.get(
                            "GPU_FAULT_PROCESSOR_MAX_QUEUE_DEPTH",
                            processor.max_queue_depth,
                        )
                    ),
                    "max_cluster_queue_depth": int(
                        ingress_processor.get(
                            "GPU_FAULT_PROCESSOR_MAX_CLUSTER_QUEUE_DEPTH",
                            processor.max_cluster_queue_depth,
                        )
                    ),
                    "retry_after_seconds": int(
                        ingress_processor.get(
                            "GPU_FAULT_PROCESSOR_RETRY_AFTER_SECONDS",
                            processor.retry_after_seconds,
                        )
                    ),
                    "retry_backoff_seconds": int(
                        ingress_processor.get(
                            "GPU_FAULT_PROCESSOR_RETRY_BACKOFF_SECONDS",
                            processor.retry_backoff_seconds,
                        )
                    ),
                    "retry_backoff_max_seconds": int(
                        ingress_processor.get(
                            "GPU_FAULT_PROCESSOR_RETRY_BACKOFF_MAX_SECONDS",
                            processor.retry_backoff_max_seconds,
                        )
                    ),
                    "completed_retention_seconds": int(
                        ingress_processor.get(
                            ("GPU_FAULT_PROCESSOR_COMPLETED_RETENTION_SECONDS"),
                            processor.completed_retention_seconds,
                        )
                    ),
                },
                "workflow": {
                    "poll_interval_seconds": float(
                        ingress_recovery.get(
                            "GPU_FAULT_WORKFLOW_POLL_INTERVAL_SECONDS",
                            workflow.poll_interval_seconds,
                        )
                    ),
                    "dispatcher_workers": int(
                        ingress_recovery.get(
                            "GPU_FAULT_WORKFLOW_DISPATCHER_WORKERS",
                            workflow.dispatcher_workers,
                        )
                    ),
                },
                "notification_delivery": {
                    "batch_size": int(
                        ingress_notification.get(
                            "GPU_FAULT_NOTIFICATION_BATCH_SIZE",
                            notification.batch_size,
                        )
                    ),
                    "max_attempts": int(
                        ingress_notification.get(
                            "GPU_FAULT_NOTIFICATION_MAX_ATTEMPTS",
                            notification.max_attempts,
                        )
                    ),
                },
                "evidence": {
                    "retention_hours": int(
                        ingress_processor.get(
                            "GPU_FAULT_EVIDENCE_RETENTION_HOURS",
                            evidence.retention_hours,
                        )
                    ),
                    "max_records_per_node": int(
                        ingress_processor.get(
                            "GPU_FAULT_EVIDENCE_MAX_RECORDS_PER_NODE",
                            evidence.max_records_per_node,
                        )
                    ),
                },
            }
        )
    except (AdminConfigError, KeyError, TypeError, ValueError) as exc:
        raise ReleaseError("cannot capture a valid live administrator config") from exc

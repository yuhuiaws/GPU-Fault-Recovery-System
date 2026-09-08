"""Authoring a desired AdminConfig from a camelCase YAML ``spec``.

Everything here produces a *desired* config, so it validates in full,
including the capacity rules the perf evidence fixes (see
``capacity_evidence``). Reading a recorded or live state is
``AdminConfig.from_mapping`` in ``config`` and deliberately is not.

``spec.capacity.preset`` names a whole topology. The 32/50 presets model the
perf plan's multi-cluster runs (性能压测验收方案 §2): N clusters x 256 nodes.
At 12,800 nodes the plan's own §13.3 run needed Min ACU pre-provisioned near
128 to stop returning 503, so a preset carries the Aurora floor of the
topology it names instead of the 8 ACU single-cluster default.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace

from gpu_fault.admin.capacity_defaults import DEFAULT_CONTROL_WORKER_REPLICAS
from gpu_fault.admin.capacity_evidence import aurora_min_acu_floor
from gpu_fault.admin.config import (
    AdminConfig,
    AdminConfigError,
    AuroraCapacityConfig,
    CapacityConfig,
    RemediationCapacity,
    TelemetrySpoolCapacity,
    default_admin_config,
)

PRESET_NODES_PER_CLUSTER = 256
PRESET_MAX_ACU = 128.0
PRESET_NAMES = ("default", "32-disabled", "32-enabled", "50-disabled", "50-enabled")


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
                "unknown capacity preset; expected one of " + ", ".join(PRESET_NAMES)
            )
    managed_nodes: int = clusters * PRESET_NODES_PER_CLUSTER
    min_acu: float = aurora_min_acu_floor(managed_nodes)
    config = AdminConfig(
        capacity=CapacityConfig(
            control_worker_replicas=DEFAULT_CONTROL_WORKER_REPLICAS,
            telemetry_spool=spool,
            remediation=RemediationCapacity(
                max_active_region=clusters * 4,
                max_active_per_cluster=4,
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
    change shows in the plan like any other and goes through the audited RDS
    path; only an unrecorded change would be a fabrication.
    """

    if current.min_acu >= preset.aurora.min_acu:
        return current
    return AuroraCapacityConfig(
        min_acu=preset.aurora.min_acu,
        max_acu=max(current.max_acu, preset.aurora.max_acu),
    )


def apply_patch(base: AdminConfig, value: object, *, path: str = "spec") -> AdminConfig:
    """``base`` with a camelCase ``spec`` laid over it, validated in full.

    Omitted fields keep their ``base`` value. ``capacity.preset`` is applied
    first (its capacity replaces the base capacity and its Aurora floor is
    adopted), then the explicit fields of the same document are laid on top,
    so a file can name a preset and still override one of its values.
    """

    data: Mapping[str, object] = value if isinstance(value, Mapping) else {}
    capacity = data.get("capacity")
    if isinstance(capacity, Mapping) and "preset" in capacity:
        raw_preset = capacity["preset"]
        if not isinstance(raw_preset, str) or not raw_preset.strip():
            raise AdminConfigError(f"{path}.capacity.preset must be a non-empty string")
        preset = preset_admin_config(raw_preset)
        base = replace(
            base,
            capacity=preset.capacity,
            aurora=_preset_aurora(base.aurora, preset),
        )
        data = {
            **data,
            "capacity": {
                key: item for key, item in capacity.items() if key != "preset"
            },
        }
        value = data
    config = base.patched(value, path=path)
    config.validate()
    return config

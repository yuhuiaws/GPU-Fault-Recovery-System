from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Iterable


class ReleaseChangeKind(StrEnum):
    NOOP = "NOOP"
    CONTROL_PLANE_ONLY = "CONTROL_PLANE_ONLY"
    DATA_PLANE_COMPATIBLE = "DATA_PLANE_COMPATIBLE"
    FULL = "FULL"


class ReleaseComponent(StrEnum):
    SCHEMA = "schema"
    REGISTRY = "registry"
    CPU_STAGE = "cpu-stage"
    RUNTIME_PROFILE = "runtime-profile"
    ENDPOINT = "endpoint"
    OBSERVABILITY = "observability"
    DCGM = "dcgm"
    EXECUTOR = "executor"
    WATCHER = "watcher"
    COLLECTOR = "collector"
    RECONCILER = "reconciler"
    AGENT = "agent"
    CPU_FINALIZE = "cpu-finalize"
    VERIFY = "verify"


@dataclass(frozen=True)
class ReleaseDiff:
    kind: ReleaseChangeKind
    changed: frozenset[str]

    def has(self, *names: str) -> bool:
        return bool(self.changed.intersection(names))

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "changed": sorted(self.changed),
        }


@dataclass(frozen=True)
class ReleaseExecutionPlan:
    nodes: tuple[ReleaseComponent, ...]

    def has(self, *components: ReleaseComponent) -> bool:
        return bool(set(components).intersection(self.nodes))

    def as_dict(self) -> dict[str, Any]:
        return {"nodes": [item.value for item in self.nodes]}


PLAN_ORDER = tuple(ReleaseComponent)
ADMIN_CONFIG_CHANGE_FIELDS = frozenset(
    {
        "admin_config_ingress",
        "admin_config_worker",
        "admin_config_spool",
    }
)
ADMIN_CONFIG_ROLE_FIELDS = {
    "admin_config_ingress": "ingress",
    "admin_config_worker": "worker",
    "admin_config_spool": "spool",
}
CPU_ROLE_MANIFEST_FIELDS = {
    "cpu_ingress_manifests": "ingress",
    "cpu_worker_manifests": "worker",
    "cpu_spool_manifests": "spool",
}


def control_plane_role_targets(diff: ReleaseDiff) -> tuple[str, ...]:
    scoped = diff.changed - {"release_delivery", "rendered_manifests"}
    role_fields = {
        **ADMIN_CONFIG_ROLE_FIELDS,
        **CPU_ROLE_MANIFEST_FIELDS,
    }
    if scoped and scoped.issubset(set(role_fields)):
        return tuple(
            role
            for role in ("spool", "worker", "ingress")
            if any(
                field in scoped and target == role
                for field, target in role_fields.items()
            )
        )
    return ("spool", "worker", "ingress")


#: Plan nodes with a per-GPU-cluster half, in rollout order. OBSERVABILITY is
#: global in the compensation plan (the control-plane collector and the AMP
#: rules) but also applies the data-plane collector to every cluster target, so
#: an observability-only release still visits the clusters.
GPU_CLUSTER_COMPONENTS = (
    ReleaseComponent.ENDPOINT,
    ReleaseComponent.DCGM,
    ReleaseComponent.OBSERVABILITY,
    ReleaseComponent.EXECUTOR,
    ReleaseComponent.WATCHER,
    ReleaseComponent.COLLECTOR,
    ReleaseComponent.RECONCILER,
    ReleaseComponent.AGENT,
)


def build_execution_plan(diff: ReleaseDiff) -> ReleaseExecutionPlan:
    changed = diff.changed
    selected = {ReleaseComponent.VERIFY}
    schema = bool(changed & {"database_schema", "schema_manifests"})
    profile = bool(changed & {"runtime_profile", "runtime_profile_version"})
    endpoint = bool(
        changed
        & {
            "endpoint",
            "endpoint_manifests",
            "clusters",
        }
    )
    observability = bool(
        changed
        & {
            "observability_manifests",
            "observability_rules",
            "observability_adot",
            "adot_image",
        }
    )
    dcgm = bool(changed & {"dcgm", "dcgm_manifests", "dcgm_image"})
    executor = bool(
        changed
        & {
            "executor_wheel",
            "executor_protocol",
            "executor_manifests",
            "runtime_image",
            "clusters",
        }
    )
    watcher = bool(
        changed
        & {
            "executor_wheel",
            "watcher_manifests",
            "runtime_image",
            "clusters",
        }
    )
    collector = bool(
        changed
        & {
            "executor_wheel",
            "collector_manifests",
            "runtime_image",
            "runtime_profile_version",
            "clusters",
        }
    )
    agent = bool(
        changed
        & {
            "node_runtime_wheel",
            "node_bundle",
            "node_template",
            "node_manifests",
            "node_installer_image",
            "agent_config",
            "agent_protocol",
            "runtime_profile",
            "runtime_profile_version",
        }
    )
    reconciler = bool(
        changed
        & {
            "executor_wheel",
            "node_manifests",
            "node_installer_image",
            "runtime_image",
            "runtime_profile",
            "runtime_profile_version",
            "agent_config",
            "agent_protocol",
            "node_runtime_wheel",
            "node_bundle",
            "node_template",
        }
    )
    pin_changed = bool(
        changed
        & {
            "executor_wheel",
            "node_runtime_wheel",
            "agent_protocol",
            "executor_protocol",
            "agent_config",
        }
    )
    cpu_changed = bool(
        changed
        & {
            "control_plane_wheel",
            "cpu_manifests",
            *CPU_ROLE_MANIFEST_FIELDS,
            "runtime_image",
            "notifications",
            "clusters",
            *ADMIN_CONFIG_CHANGE_FIELDS,
        }
    )
    cpu_stage = pin_changed or profile or bool(changed & {"clusters"})
    cpu_finalize = cpu_changed or cpu_stage
    scoped = changed - {"release_delivery", "rendered_manifests"}
    role_scoped_cpu_only = bool(scoped) and scoped.issubset(
        {*ADMIN_CONFIG_CHANGE_FIELDS, *CPU_ROLE_MANIFEST_FIELDS}
    )
    registry = (
        cpu_stage
        or bool(changed & {"clusters"})
        or (cpu_finalize and not role_scoped_cpu_only)
    )

    for enabled, component in (
        (schema, ReleaseComponent.SCHEMA),
        (registry, ReleaseComponent.REGISTRY),
        (cpu_stage, ReleaseComponent.CPU_STAGE),
        (profile, ReleaseComponent.RUNTIME_PROFILE),
        (endpoint, ReleaseComponent.ENDPOINT),
        (observability, ReleaseComponent.OBSERVABILITY),
        (dcgm, ReleaseComponent.DCGM),
        (executor, ReleaseComponent.EXECUTOR),
        (watcher, ReleaseComponent.WATCHER),
        (collector, ReleaseComponent.COLLECTOR),
        (reconciler, ReleaseComponent.RECONCILER),
        (agent, ReleaseComponent.AGENT),
        (cpu_finalize, ReleaseComponent.CPU_FINALIZE),
    ):
        if enabled:
            selected.add(component)
    return ReleaseExecutionPlan(
        nodes=tuple(item for item in PLAN_ORDER if item in selected)
    )


def _legacy_value(state: dict[str, Any], name: str) -> Any:
    if name == "executor_wheel_sha256":
        return state.get(name) or state.get("wheel_sha256")
    if name == "node_wheel_sha256":
        return state.get(name) or state.get("wheel_sha256")
    return state.get(name)


def _current_profile_digest(release: Any, state: dict[str, Any]) -> Any:
    current = state.get("runtime_profile_policy_sha256")
    if current:
        return current
    legacy = state.get("runtime_profile_sha256")
    same_version = (
        state.get("runtime_profile_version") == release.config.runtime_profile_version
    )
    if same_version and legacy in {
        release.runtime_profile_sha,
        release.runtime_profile_template_sha,
    }:
        return release.runtime_profile_policy_sha
    return legacy


def diff_from_changed(changed: Iterable[str]) -> ReleaseDiff:
    normalized = frozenset(changed)
    scoped = normalized - {"release_delivery", "rendered_manifests"}
    if not scoped:
        kind = ReleaseChangeKind.NOOP
    elif scoped.issubset(
        {
            "control_plane_wheel",
            "notifications",
            "cpu_manifests",
            *CPU_ROLE_MANIFEST_FIELDS,
            "observability_manifests",
            "observability_rules",
            "observability_adot",
            "adot_image",
            *ADMIN_CONFIG_CHANGE_FIELDS,
        }
    ):
        kind = ReleaseChangeKind.CONTROL_PLANE_ONLY
    elif not scoped.intersection(
        {
            "database_schema",
            "schema_manifests",
            "agent_protocol",
            "executor_protocol",
            "agent_config",
            "runtime_profile",
            "runtime_profile_version",
            "clusters",
        }
    ):
        kind = ReleaseChangeKind.DATA_PLANE_COMPATIBLE
    else:
        kind = ReleaseChangeKind.FULL
    return ReleaseDiff(kind=kind, changed=normalized)


def classify_release(release: Any, state: dict[str, Any]) -> ReleaseDiff:
    desired = {
        "control_plane_wheel": release.wheel_sha,
        "executor_wheel": release.executor_wheel_sha,
        "node_runtime_wheel": release.node_wheel_sha,
        "node_bundle": release.bundle_sha,
        "database_schema": release.config.database_schema_version,
        "agent_protocol": release.config.agent_protocol_version,
        "executor_protocol": release.config.executor_protocol_version,
        "agent_config": release.config.agent_config_digest,
        "runtime_profile": release.runtime_profile_policy_sha,
        "runtime_profile_version": release.config.runtime_profile_version,
        "endpoint": release.endpoint_digest,
        "dcgm": release.dcgm_digest,
        "notifications": release.notification_digest,
        "clusters": release.cluster_registry_digest,
        "admin_config_ingress": release.admin_config_role_digests["ingress"],
        "admin_config_worker": release.admin_config_role_digests["worker"],
        "admin_config_spool": release.admin_config_role_digests["spool"],
        "release_delivery": release.config.release_delivery_sha256,
        "cpu_manifests": release.config.delivery_component_digests.get("cpu"),
        "cpu_ingress_manifests": release.config.delivery_component_digests.get(
            "cpu_ingress"
        ),
        "cpu_worker_manifests": release.config.delivery_component_digests.get(
            "cpu_worker"
        ),
        "cpu_spool_manifests": release.config.delivery_component_digests.get(
            "cpu_spool"
        ),
        "executor_manifests": release.config.delivery_component_digests.get("executor"),
        "watcher_manifests": release.config.delivery_component_digests.get("watcher"),
        "collector_manifests": release.config.delivery_component_digests.get(
            "collector"
        ),
        "dcgm_manifests": release.config.delivery_component_digests.get("dcgm"),
        "node_manifests": release.config.delivery_component_digests.get("node"),
        "observability_manifests": (
            release.config.delivery_component_digests.get("observability")
        ),
        "observability_rules": release.observability_rules_digest,
        "observability_adot": release.observability_adot_digest,
        "schema_manifests": release.config.delivery_component_digests.get("schema"),
        "endpoint_manifests": release.config.delivery_component_digests.get("endpoint"),
        "rendered_manifests": release.rendered_manifest_digest,
        "node_template": release.node_template_sha,
        "runtime_image": release.runtime_image,
        "node_installer_image": release.node_installer_image,
        "dcgm_image": release.dcgm_exporter_image,
        "adot_image": release.adot_image,
    }
    current = {
        "control_plane_wheel": state.get("wheel_sha256"),
        "executor_wheel": _legacy_value(state, "executor_wheel_sha256"),
        "node_runtime_wheel": _legacy_value(state, "node_wheel_sha256"),
        "node_bundle": state.get("bundle_sha256"),
        "database_schema": int(state.get("database_schema_version") or 0),
        "agent_protocol": int(state.get("agent_protocol_version") or 0),
        "executor_protocol": int(state.get("executor_protocol_version") or 0),
        "agent_config": state.get("agent_config_digest"),
        "runtime_profile": _current_profile_digest(release, state),
        "runtime_profile_version": state.get("runtime_profile_version"),
        "endpoint": state.get("endpoint_digest"),
        "dcgm": state.get("dcgm_digest"),
        "notifications": state.get("notification_digest"),
        "clusters": state.get("cluster_registry_digest"),
        "admin_config_ingress": (
            (state.get("admin_config_role_sha256") or {}).get("ingress")
        ),
        "admin_config_worker": (
            (state.get("admin_config_role_sha256") or {}).get("worker")
        ),
        "admin_config_spool": (
            (state.get("admin_config_role_sha256") or {}).get("spool")
        ),
        "release_delivery": state.get("release_delivery_sha256"),
        "cpu_manifests": state.get("cpu_manifest_sha256"),
        "cpu_ingress_manifests": state.get("cpu_ingress_manifest_sha256"),
        "cpu_worker_manifests": state.get("cpu_worker_manifest_sha256"),
        "cpu_spool_manifests": state.get("cpu_spool_manifest_sha256"),
        "executor_manifests": state.get("executor_manifest_sha256"),
        "watcher_manifests": state.get("watcher_manifest_sha256"),
        "collector_manifests": state.get("collector_manifest_sha256"),
        "dcgm_manifests": state.get("dcgm_manifest_sha256"),
        "node_manifests": state.get("node_manifest_sha256"),
        "observability_manifests": state.get("observability_manifest_sha256"),
        "observability_rules": state.get("observability_rules_sha256"),
        "observability_adot": (
            state.get("observability_adot_sha256")
            or (
                release.observability_adot_digest
                if state.get("observability_manifest_sha256")
                else None
            )
        ),
        "schema_manifests": state.get("schema_manifest_sha256"),
        "endpoint_manifests": state.get("endpoint_manifest_sha256"),
        "rendered_manifests": state.get("rendered_manifest_sha256"),
        "node_template": state.get("node_template_sha256"),
        "runtime_image": state.get("runtime_image"),
        "node_installer_image": state.get("node_installer_image"),
        "dcgm_image": state.get("dcgm_image"),
        "adot_image": state.get("adot_image"),
    }
    changed = frozenset(
        name for name, value in desired.items() if current.get(name) != value
    )
    return diff_from_changed(changed)

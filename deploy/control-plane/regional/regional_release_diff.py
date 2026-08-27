from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Iterable


class ReleaseChangeKind(StrEnum):
    NOOP = "NOOP"
    CONTROL_PLANE_ONLY = "CONTROL_PLANE_ONLY"
    DATA_PLANE_COMPATIBLE = "DATA_PLANE_COMPATIBLE"
    FULL = "FULL"


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
    if not normalized:
        kind = ReleaseChangeKind.NOOP
    elif normalized.issubset({"control_plane_wheel", "notifications"}):
        kind = ReleaseChangeKind.CONTROL_PLANE_ONLY
    elif not normalized.intersection(
        {
            "database_schema",
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
        "clusters": tuple(sorted(item.cluster_id for item in release.config.clusters)),
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
        "clusters": tuple(sorted(state.get("cluster_ids") or ())),
    }
    changed = frozenset(
        name for name, value in desired.items() if current.get(name) != value
    )
    return diff_from_changed(changed)

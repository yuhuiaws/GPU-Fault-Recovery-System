"""Every CPU role that reads an administrator field has it in its digest.

A role's digest (``AdminConfig.role_sha256``) decides whether a config
release selects that role (``regional_release_diff.control_plane_role_targets``
turns the moved digests into ``GPU_FAULT_CONTROL_PLANE_ROLE_TARGETS``), and it
is the ``gpu-fault.io/role-config-sha256`` stamp that rolls the role's Pods. A
field rendered into a role whose digest does not cover it therefore never
reaches that role's processes: ``telemetrySpool.maxClusterDepth`` was in the
spool worker's digest only, while the renderer writes
``GPU_FAULT_TELEMETRY_SPOOL_MAX_CLUSTER_DEPTH`` into the api-ha ConfigMap
alone, so a release that changed only the depth re-applied the spool worker
and left the ingress admitting at the old depth.

The readers come from the renderer itself: for each field, the roles whose
rendered ConfigMaps, Deployment or PDB change are the roles that read it, and
exactly their digests must move.
"""

from __future__ import annotations

import functools
import json
import os
import subprocess
import sys
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any

import pytest

from gpu_fault.admin.config import (
    AdminConfig,
    admin_config_desired_path,
    canonical_sha256,
    default_admin_config,
    load_desired_admin_config,
)
from gpu_fault.admin.config_file import initialize_desired_admin_config
from gpu_fault.admin.config_parser import AdminConfigError, camel_case, retired_fields
from gpu_fault.admin.config_patch import PRESET_NAMES, apply_patch, preset_admin_config
from gpu_fault.admin.config_roles import ADMIN_CONFIG_ROLES, ROLE_INPUT_READERS
from gpu_fault_release.regional_release_rendering import (
    admin_config_renderer_environment,
)
from tests.regional.test_control_plane_role_split import _deployment

ROOT = Path(__file__).resolve().parents[2]
RENDERER = ROOT / "deploy/control-plane/tools/render_control_plane_role_split.py"
# The role of every object the renderer emits, by name; the apply step's
# ``manifest_role`` uses the same prefixes.
ROLE_OBJECTS = {
    "gpu-fault-api-ha": "ingress",
    "gpu-fault-control-worker": "worker",
    "gpu-fault-telemetry-spool-worker": "spool",
}
SPOOL_ON = {
    "capacity": {
        "controlWorkerReplicas": 5,
        "telemetrySpool": {"enabled": True, "replicas": 3},
    }
}
DEEP_SPOOL = {"capacity": {"telemetrySpool": {"maxClusterDepth": 4096}}}
# (field, the spec the base config lays over the defaults, the change).
FIELD_CHANGES: tuple[tuple[str, dict[str, Any] | None, dict[str, Any]], ...] = (
    (
        "capacity.controlWorkerReplicas",
        None,
        {"capacity": {"controlWorkerReplicas": 5}},
    ),
    (
        "capacity.largestClusterNodeCount",
        None,
        {"capacity": {"largestClusterNodeCount": 256}},
    ),
    ("capacity.managedNodeCount", None, {"capacity": {"managedNodeCount": 600}}),
    (
        "capacity.telemetrySpool.enabled",
        {"capacity": {"controlWorkerReplicas": 5}},
        {"capacity": {"telemetrySpool": {"enabled": True, "replicas": 3}}},
    ),
    (
        "capacity.telemetrySpool.replicas",
        SPOOL_ON,
        {"capacity": {"telemetrySpool": {"replicas": 2}}},
    ),
    ("capacity.telemetrySpool.maxClusterDepth", None, DEEP_SPOOL),
    ("capacity.telemetrySpool.maxClusterDepth", SPOOL_ON, DEEP_SPOOL),
    (
        "capacity.telemetrySpool.maxClusterDepth",
        DEEP_SPOOL,
        {"capacity": {"telemetrySpool": {"maxClusterDepth": 1024}}},
    ),
    ("aurora.minAcu", None, {"aurora": {"minAcu": 16}}),
    ("aurora.maxAcu", None, {"aurora": {"maxAcu": 64}}),
    ("processor.maxQueueDepth", None, {"processor": {"maxQueueDepth": 131072}}),
    (
        "processor.maxClusterQueueDepth",
        None,
        {"processor": {"maxClusterQueueDepth": 8192}},
    ),
    ("processor.retryAfterSeconds", None, {"processor": {"retryAfterSeconds": 3}}),
    ("processor.retryBackoffSeconds", None, {"processor": {"retryBackoffSeconds": 2}}),
    (
        "processor.retryBackoffMaxSeconds",
        None,
        {"processor": {"retryBackoffMaxSeconds": 45}},
    ),
    (
        "processor.completedRetentionSeconds",
        None,
        {"processor": {"completedRetentionSeconds": 900}},
    ),
    (
        "processor.admissionBatchDelaySeconds",
        None,
        {"processor": {"admissionBatchDelaySeconds": 0.01}},
    ),
    (
        "processor.admissionBatchDelayMaxSeconds",
        None,
        {"processor": {"admissionBatchDelayMaxSeconds": 0.1}},
    ),
    (
        "processor.admissionFlushConcurrency",
        None,
        {"processor": {"admissionFlushConcurrency": 120}},
    ),
    (
        "processor.admissionFlushConcurrencyFloor",
        None,
        {"processor": {"admissionFlushConcurrencyFloor": 48}},
    ),
    (
        "processor.faultAdmissionFlushConcurrency",
        None,
        {"processor": {"faultAdmissionFlushConcurrency": 64}},
    ),
    (
        "processor.faultAdmissionFlushConcurrencyFloor",
        None,
        {"processor": {"faultAdmissionFlushConcurrencyFloor": 24}},
    ),
    (
        "processor.evidenceAdmissionFlushConcurrency",
        None,
        {"processor": {"evidenceAdmissionFlushConcurrency": 32}},
    ),
    (
        "processor.evidenceAdmissionFlushConcurrencyFloor",
        None,
        {"processor": {"evidenceAdmissionFlushConcurrencyFloor": 6}},
    ),
    ("workflow.pollIntervalSeconds", None, {"workflow": {"pollIntervalSeconds": 10}}),
    ("workflow.dispatcherWorkers", None, {"workflow": {"dispatcherWorkers": 12}}),
    (
        "workflow.departedNodeCleanup",
        None,
        {"workflow": {"departedNodeCleanup": False}},
    ),
    (
        "notificationDelivery.batchSize",
        None,
        {"notificationDelivery": {"batchSize": 50}},
    ),
    (
        "notificationDelivery.maxAttempts",
        None,
        {"notificationDelivery": {"maxAttempts": 10}},
    ),
    ("evidence.retentionHours", None, {"evidence": {"retentionHours": 48}}),
    ("evidence.maxRecordsPerNode", None, {"evidence": {"maxRecordsPerNode": 20000}}),
    (
        "cpuControlPlane.rollingUpdate.mode",
        None,
        {"cpuControlPlane": {"rollingUpdate": {"mode": "unchanged"}}},
    ),
    (
        "cpuControlPlane.rollingUpdate.waitIntervalSeconds",
        None,
        {"cpuControlPlane": {"rollingUpdate": {"waitIntervalSeconds": 2400}}},
    ),
    (
        "cpuControlPlane.rollbackAlarms.capacity",
        None,
        {"cpuControlPlane": {"rollbackAlarms": {"capacity": "site-b-roll-capacity"}}},
    ),
    (
        "cpuControlPlane.rollbackAlarms.notRestored",
        None,
        {
            "cpuControlPlane": {
                "rollbackAlarms": {"notRestored": "site-b-roll-not-restored"}
            }
        },
    ),
)


def _role(name: str) -> str:
    for prefix, role in ROLE_OBJECTS.items():
        if name == prefix or name.startswith(prefix + "-"):
            return role
    raise AssertionError(f"the renderer emitted {name}, which belongs to no role")


@functools.lru_cache(maxsize=None)
def _render(environment: tuple[tuple[str, str], ...]) -> dict[str, dict[str, Any]]:
    """The role-split renderer's objects for one admin config, grouped by role."""

    result = subprocess.run(
        [sys.executable, str(RENDERER), "--json"],
        input=json.dumps(_deployment()),
        text=True,
        capture_output=True,
        check=True,
        env={
            "PATH": "/usr/bin:/bin",
            "PYTHONPATH": os.environ.get("PYTHONPATH", ""),
            **dict(environment),
        },
    )
    rendered: dict[str, dict[str, Any]] = {role: {} for role in ADMIN_CONFIG_ROLES}
    for item in json.loads(result.stdout)["items"]:
        name = item["metadata"]["name"]
        rendered[_role(name)][name] = item
    return rendered


def _rendered(config: AdminConfig) -> dict[str, dict[str, Any]]:
    return _render(tuple(sorted(admin_config_renderer_environment(config).items())))


def _config(spec: dict[str, Any] | None) -> AdminConfig:
    base = default_admin_config()
    return base if spec is None else apply_patch(base, spec)


def _leaves(value: object, prefix: str = "") -> set[str]:
    if not isinstance(value, dict):
        return {prefix}
    return set().union(
        *(
            _leaves(item, f"{prefix}.{key}" if prefix else key)
            for key, item in value.items()
        )
    )


def _administrator_fields(section: Any, prefix: str = "") -> set[str]:
    """Every field an administrator can set, in its YAML spelling."""

    found: set[str] = set()
    for field in fields(section):
        if field.name in retired_fields(section):
            continue
        path = (
            f"{prefix}.{camel_case(field.name)}" if prefix else camel_case(field.name)
        )
        value = getattr(section, field.name)
        found |= _administrator_fields(value, path) if is_dataclass(value) else {path}
    return found


def test_every_administrator_field_is_swept() -> None:
    swept = set().union(*(_leaves(change) for _, _, change in FIELD_CHANGES))

    assert swept == _administrator_fields(AdminConfig()), (
        "a new administrator field needs a FIELD_CHANGES entry so its readers are "
        "checked against the role digests"
    )
    assert {field for field, _, _ in FIELD_CHANGES} <= swept, (
        "every case names a field it changes"
    )


@pytest.mark.parametrize(
    ("field", "base_spec", "change"),
    FIELD_CHANGES,
    ids=[f"{field}-{index}" for index, (field, _, _) in enumerate(FIELD_CHANGES)],
)
def test_a_change_moves_exactly_the_digests_of_the_roles_that_render_it(
    field: str, base_spec: dict[str, Any] | None, change: dict[str, Any]
) -> None:
    base = _config(base_spec)
    changed = apply_patch(base, change)
    before, after = _rendered(base), _rendered(changed)

    readers = {role for role in ADMIN_CONFIG_ROLES if before[role] != after[role]}
    moved = {
        role
        for role in ADMIN_CONFIG_ROLES
        if base.role_sha256()[role] != changed.role_sha256()[role]
    }

    assert moved == readers, (
        f"{field}: the roles whose rendered config changed ({sorted(readers)}) must be "
        f"exactly the roles whose digest moved ({sorted(moved)}); a reader outside "
        "the digest keeps the old value, a digest outside the readers rolls for nothing"
    )


def test_the_spool_depth_is_digested_for_the_ingress_that_reads_it() -> None:
    base = default_admin_config()
    deep = apply_patch(base, DEEP_SPOOL)

    assert deep.role_payload("ingress")["telemetry_spool_max_cluster_depth"] == 4096, (
        "the ingress digest carries the depth its spool admission reads"
    )
    assert "max_cluster_depth" not in deep.role_payload("spool")["telemetry_spool"], (
        "the spool worker never reads the depth, so its digest does not hold it"
    )
    assert ROLE_INPUT_READERS["telemetry_spool_max_cluster_depth"] == {"ingress"}, (
        "the declaration names the ingress as the depth's only reader"
    )
    assert "telemetry_spool_max_cluster_depth" not in base.role_payload("ingress"), (
        "omitted at its default, like the record field, so live digests stay"
    )


def test_every_payload_entry_is_declared_and_read_by_a_known_role() -> None:
    deep = apply_patch(_config(SPOOL_ON), DEEP_SPOOL)
    produced = set().union(*(deep.role_payload(role) for role in ADMIN_CONFIG_ROLES))

    assert produced == set(ROLE_INPUT_READERS), (
        "role_payload and ROLE_INPUT_READERS name the same entries"
    )
    for entry, readers in ROLE_INPUT_READERS.items():
        assert readers and readers <= set(ADMIN_CONFIG_ROLES), (
            f"{entry} must be read by at least one known role, got {sorted(readers)}"
        )
    with pytest.raises(AdminConfigError, match="unknown control-plane role: api"):
        deep.role_payload("api")


def _digests_before_the_ingress_read_the_depth(config: AdminConfig) -> dict[str, str]:
    """Role digests exactly as the releases before this change computed them:
    the spool depth inside the spool worker's section, none in the ingress's."""

    capacity = config.capacity
    common = {
        "processor": config.processor.as_dict(),
        "workflow": config.workflow.as_dict(),
        "notification_delivery": config.notification_delivery.as_dict(),
        "evidence": config.evidence.as_dict(),
        **capacity.node_counts(),
    }
    payloads = {
        "ingress": {
            **common,
            "telemetry_spool_enabled": capacity.telemetry_spool.enabled,
        },
        "worker": {
            **common,
            "control_worker_replicas": capacity.control_worker_replicas,
            "remediation": capacity.remediation.as_dict(),
        },
        "spool": {**common, "telemetry_spool": capacity.telemetry_spool.as_dict()},
    }
    return {role: canonical_sha256(payload) for role, payload in payloads.items()}


@pytest.mark.parametrize("name", PRESET_NAMES)
def test_a_site_at_the_default_depth_keeps_every_role_digest(name: str) -> None:
    config = preset_admin_config(name)

    assert config.role_sha256() == _digests_before_the_ingress_read_the_depth(config), (
        f"{name}: no live site at the default depth may see a role digest move"
    )


def _write_desired(tmp_path: Path, config: AdminConfig, roles: dict[str, str]) -> Path:
    path = admin_config_desired_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "config": config.as_dict(),
                "config_sha256": config.sha256(),
                "role_sha256": roles,
                "source": "approved-plan:deep-spool",
                "plan_sha256": "c" * 64,
                "reference": "CHG-DEEP-SPOOL",
                "updated_at": "2026-10-30T00:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )
    return path


def test_a_desired_json_with_a_deep_spool_from_before_is_verified_and_migrated(
    tmp_path: Path,
) -> None:
    config = apply_patch(_config(SPOOL_ON), DEEP_SPOOL)
    earlier = _digests_before_the_ingress_read_the_depth(config)
    assert earlier != config.role_sha256(), "the depth now moves other digests"
    path = _write_desired(tmp_path, config, earlier)
    written = path.read_bytes()

    assert load_desired_admin_config(tmp_path) == config, (
        "a record the previous release wrote verifies against its own digests"
    )
    assert path.read_bytes() == written, "a read without migration rewrites nothing"

    initialize_desired_admin_config(tmp_path)
    migrated = json.loads(path.read_text(encoding="utf-8"))

    assert migrated["role_sha256"] == config.role_sha256(), (
        "the migration re-digests the record the way this release reads it"
    )
    assert migrated["config"] == config.as_dict(), "the content is kept as recorded"
    assert migrated["source"] == (
        "legacy-admin-config-migration:approved-plan:deep-spool"
    ), "the migration names itself and the source it replaced"
    assert migrated["reference"] == "CHG-DEEP-SPOOL", "the reference survives"
    assert load_desired_admin_config(tmp_path) == config, "the migrated record verifies"


def test_a_desired_json_with_foreign_role_digests_still_fails_closed(
    tmp_path: Path,
) -> None:
    config = apply_patch(default_admin_config(), DEEP_SPOOL)
    foreign = {**config.role_sha256(), "ingress": "0" * 64}
    _write_desired(tmp_path, config, foreign)

    with pytest.raises(AdminConfigError, match="role digests do not match"):
        load_desired_admin_config(tmp_path)

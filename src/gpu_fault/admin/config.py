"""The administrator configuration: one schema, its persisted state, the apply record.

The frozen dataclasses below own every default and bound. The persisted
``desired.json`` is their snake_case ``as_dict()``; the editable YAML ``spec``
is the same tree in camelCase. A field is therefore declared once, and both
readers are the same generic walk over the dataclass fields
(``config_parser.read_section``), so the two spellings cannot drift apart.

State files under ``<state-dir>/admin-config/``:

* ``desired.json`` -- the authoritative desired config, digest-checked on read.
* ``pending.json`` -- the one apply in flight or interrupted: the config before,
  the target, the site and release identity, who approved it and when. A rerun
  of ``gpu-fault-admin config`` with the same target resumes it.
* ``history/<started>-<config_sha256>/{before,after,result}.json`` -- one
  directory per apply attempt.
"""

from __future__ import annotations

import hashlib
import json
import re
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Iterator, Mapping

from gpu_fault.admin.atomic_json import write_json_atomic
from gpu_fault.admin.capacity_evidence import (
    DEFAULT_LARGEST_CLUSTER_NODE_COUNT,
    DEFAULT_MANAGED_NODE_COUNT,
    CapacityEvidenceError,
    aurora_min_acu_floor,
    fault_reserved_cluster_depth,
    fault_reserved_queue_depth,
    legacy_largest_cluster_node_count,
    validate_capacity_evidence,
)
from gpu_fault.admin.config_parser import (
    AdminConfigError,
    camel_section,
    mapping_field,
    read_section,
    snake_section,
)
from gpu_fault.admin.operation_lock import SiteOperationBusy, site_operation_lock
from gpu_fault.digests import SHA256_PATTERN

__all__ = ["AdminConfigError", "aurora_min_acu_floor"]

ADMIN_CONFIG_API_VERSION = "gpu-fault.aws/v1alpha1"
ADMIN_CONFIG_KIND = "AdminConfig"
ADMIN_CONFIG_ROOT = Path("admin-config")
ADMIN_CONFIG_DESIRED = ADMIN_CONFIG_ROOT / "desired.json"
ADMIN_CONFIG_PENDING = ADMIN_CONFIG_ROOT / "pending.json"
ADMIN_CONFIG_HISTORY = ADMIN_CONFIG_ROOT / "history"
APPROVAL_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{2,127}$")
HISTORY_ID_PATTERN = re.compile(r"^[0-9]{8}T[0-9]{6}Z-[0-9a-f]{64}$")
ADMIN_CONFIG_ROLES = ("ingress", "worker", "spool")
# Safety constants, not administrator inputs: one remediation per node and per
# failure domain at a time. Old YAML that still spells them is accepted only at
# this value; the persisted record and every renderer keep reading them.
MAX_ACTIVE_PER_NODE = 1
MAX_ACTIVE_PER_FAILURE_DOMAIN = 1
# What a site from before the Aurora fields existed really ran; see
# ``upgrade_legacy_record``.
LEGACY_AURORA_MIN_ACU = 0.5
LEGACY_AURORA_MAX_ACU = 8.0


class _Section:
    """A configuration section; ``as_dict`` is the persisted snake_case form."""

    def as_dict(self) -> dict[str, object]:
        return snake_section(self, constants=_CONSTANT_FIELDS)


@dataclass(frozen=True)
class RemediationCapacity(_Section):
    max_active_region: int = 20
    max_active_per_cluster: int = 5
    max_active_per_resource_class: int = 2

    @property
    def max_active_per_node(self) -> int:
        return MAX_ACTIVE_PER_NODE

    @property
    def max_active_per_failure_domain(self) -> int:
        return MAX_ACTIVE_PER_FAILURE_DOMAIN

    def validate(self) -> None:
        bounds = {
            "maxActiveRegion": (self.max_active_region, 1, 4096),
            "maxActivePerCluster": (self.max_active_per_cluster, 1, 128),
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


@dataclass(frozen=True)
class TelemetrySpoolCapacity(_Section):
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


@dataclass(frozen=True)
class CapacityConfig(_Section):
    control_worker_replicas: int = 6
    # The declared topology: the biggest single GPU cluster and the whole
    # managed fleet. The processor depth, the fault reserve and the Aurora
    # floor are derived from these, so they are inputs, not tuning knobs.
    largest_cluster_node_count: int = DEFAULT_LARGEST_CLUSTER_NODE_COUNT
    managed_node_count: int = DEFAULT_MANAGED_NODE_COUNT
    telemetry_spool: TelemetrySpoolCapacity = TelemetrySpoolCapacity()
    remediation: RemediationCapacity = RemediationCapacity()

    def validate(self) -> None:
        if not 1 <= self.control_worker_replicas <= 64:
            raise AdminConfigError(
                "spec.capacity.controlWorkerReplicas must be within 1..64"
            )
        if not 1 <= self.largest_cluster_node_count <= 4096:
            raise AdminConfigError(
                "spec.capacity.largestClusterNodeCount must be within 1..4096"
            )
        if not self.largest_cluster_node_count <= self.managed_node_count <= 65536:
            raise AdminConfigError(
                "spec.capacity.managedNodeCount must be within "
                f"largestClusterNodeCount ({self.largest_cluster_node_count})..65536"
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

    def node_counts(self) -> dict[str, int]:
        return {
            "largest_cluster_node_count": self.largest_cluster_node_count,
            "managed_node_count": self.managed_node_count,
        }


@dataclass(frozen=True)
class AuroraCapacityConfig(_Section):
    min_acu: float = 8.0
    max_acu: float = 32.0

    def validate(self) -> None:
        for name, value in (("minAcu", self.min_acu), ("maxAcu", self.max_acu)):
            if not 0.5 <= value <= 256:
                raise AdminConfigError(f"spec.aurora.{name} must be within 0.5..256")
            doubled = value * 2
            if abs(doubled - round(doubled)) > 1e-9:
                raise AdminConfigError(
                    f"spec.aurora.{name} must use 0.5 ACU increments"
                )
        if self.min_acu > self.max_acu:
            raise AdminConfigError("spec.aurora.minAcu must not exceed maxAcu")


@dataclass(frozen=True)
class ProcessorConfig(_Section):
    max_queue_depth: int = 65536
    # 性能压测验收方案 §13.4: one 1000-node cluster overflowed 1024 (243 HTTP
    # 429); 4096 admitted everything.
    max_cluster_queue_depth: int = 4096
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


@dataclass(frozen=True)
class WorkflowConfig(_Section):
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


@dataclass(frozen=True)
class NotificationDeliveryConfig(_Section):
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


@dataclass(frozen=True)
class EvidenceConfig(_Section):
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


@dataclass(frozen=True)
class AdminConfig(_Section):
    capacity: CapacityConfig = CapacityConfig()
    aurora: AuroraCapacityConfig = AuroraCapacityConfig()
    processor: ProcessorConfig = ProcessorConfig()
    workflow: WorkflowConfig = WorkflowConfig()
    notification_delivery: NotificationDeliveryConfig = NotificationDeliveryConfig()
    evidence: EvidenceConfig = EvidenceConfig()

    def validate(self, *, enforce_capacity_evidence: bool = True) -> None:
        """Check every field and the cross-object capacity rules.

        ``enforce_capacity_evidence=False`` is for reading a recorded or live
        state: a site that predates the perf-evidence floors really runs what
        it runs, and refusing to read that fact would leave the administrator
        unable to load the state at all to plan the fix. Every path that
        authors a desired config (file, preset, plan) keeps the default and
        refuses.
        """

        self.capacity.validate()
        self.aurora.validate()
        self.processor.validate()
        self.workflow.validate()
        self.notification_delivery.validate()
        self.evidence.validate()
        capacity = self.capacity
        processor = self.processor
        # Structural in every mode: the runtime refuses a per-cluster reserve
        # larger than the per-cluster depth, and the reserve is derived from
        # the largest cluster.
        if capacity.largest_cluster_node_count > processor.max_cluster_queue_depth:
            raise AdminConfigError(
                f"spec.processor.maxClusterQueueDepth {processor.max_cluster_queue_depth}"
                " cannot hold one fault-priority request per node of "
                "spec.capacity.largestClusterNodeCount "
                f"{capacity.largest_cluster_node_count}"
            )
        if not enforce_capacity_evidence:
            return
        try:
            validate_capacity_evidence(
                largest_cluster_node_count=capacity.largest_cluster_node_count,
                managed_node_count=capacity.managed_node_count,
                max_cluster_queue_depth=processor.max_cluster_queue_depth,
                aurora_min_acu=self.aurora.min_acu,
            )
        except CapacityEvidenceError as exc:
            raise AdminConfigError(str(exc)) from exc

    def fault_reserved_queue_depth(self) -> int:
        """GPU_FAULT_PROCESSOR_FAULT_RESERVED_QUEUE_DEPTH as the renderers ship it."""

        return fault_reserved_queue_depth(self.processor.max_queue_depth)

    def fault_reserved_cluster_depth(self) -> int:
        """GPU_FAULT_PROCESSOR_FAULT_RESERVED_CLUSTER_DEPTH as the renderers ship it.

        An eighth of the per-cluster depth, but never less than one
        fault-priority request per node of the largest cluster: a correlated
        whole-cluster fault arrives on that many lanes at once.
        """

        return fault_reserved_cluster_depth(
            self.processor.max_cluster_queue_depth,
            self.capacity.largest_cluster_node_count,
        )

    def as_dict(self) -> dict[str, object]:
        return {"schema_version": 1, **snake_section(self, constants=_CONSTANT_FIELDS)}

    def sha256(self) -> str:
        return canonical_sha256(self.as_dict())

    def role_payload(self, role: str, *, node_counts: bool = True) -> dict[str, object]:
        common: dict[str, object] = {
            "processor": self.processor.as_dict(),
            "workflow": self.workflow.as_dict(),
            "notification_delivery": self.notification_delivery.as_dict(),
            "evidence": self.evidence.as_dict(),
        }
        # The fault reserve derived from the node counts is rendered into
        # every role, so a node count change has to move every role digest.
        # ``node_counts=False`` reproduces the digests a release before the
        # node counts recorded, for verifying such a persisted state.
        if node_counts:
            common.update(self.capacity.node_counts())
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

    def role_sha256(self, *, node_counts: bool = True) -> dict[str, str]:
        return {
            role: canonical_sha256(self.role_payload(role, node_counts=node_counts))
            for role in ADMIN_CONFIG_ROLES
        }

    def patched(self, value: object, *, path: str = "spec") -> AdminConfig:
        """This config with a camelCase YAML ``spec`` laid over it, unvalidated.

        Omitted fields keep their current value. Unvalidated because one file
        may raise the topology and its Aurora floor together, and only the
        complete result can be judged; ``config_patch.apply_patch`` validates.
        """

        return read_section(
            self, value, camel=True, path=path, constants=_CONSTANT_FIELDS
        )

    @classmethod
    def from_mapping(cls, value: object) -> AdminConfig:
        """Read a persisted or captured record: a fact, so no evidence floors."""

        if not isinstance(value, Mapping):
            raise AdminConfigError("admin config must be a mapping")
        data = dict(value)
        if data.pop("schema_version", 1) != 1:
            raise AdminConfigError("admin config schema_version must be 1")
        config = read_section(
            cls(),
            upgrade_legacy_record(data),
            camel=False,
            path="admin config",
            constants=_CONSTANT_FIELDS,
        )
        config.validate(enforce_capacity_evidence=False)
        return config


_CONSTANT_FIELDS: dict[type, dict[str, int]] = {
    RemediationCapacity: {
        "max_active_per_node": MAX_ACTIVE_PER_NODE,
        "max_active_per_failure_domain": MAX_ACTIVE_PER_FAILURE_DOMAIN,
    },
}


def admin_config_spec(config: AdminConfig) -> dict[str, object]:
    """The editable YAML ``spec``: the config in camelCase, constants omitted."""

    return camel_section(config)


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def default_admin_config() -> AdminConfig:
    return AdminConfig()


# ---------------------------------------------------------------------------
# Legacy records. Every rule here reads a persisted document from before a
# field existed as the state that site really ran, digested over exactly the
# content it had; recomputing over the filled-in config would reject every
# site that upgrades.
# ---------------------------------------------------------------------------


def upgrade_legacy_record(raw: Mapping[str, object]) -> dict[str, object]:
    """Fill in the fields a persisted config predates, with what the site ran.

    * No ``aurora``: the site was created at the 0.5/8 ACU the old bootstrap
      wrote; reading it as today's 8/32 default would silently plan a change
      nobody asked for.
    * No node counts: the topology is bounded by the per-cluster depth the
      record did declare (1024 -> 256 nodes, see
      ``legacy_largest_cluster_node_count``), and the managed count defaults
      to the largest cluster, as the old parser did.

    Anything else -- an unknown field, a wrong type -- is left for the generic
    reader to reject.
    """

    data = dict(raw)
    if "aurora" not in data:
        data["aurora"] = {
            "min_acu": LEGACY_AURORA_MIN_ACU,
            "max_acu": LEGACY_AURORA_MAX_ACU,
        }
    capacity = data.get("capacity")
    if capacity is None or isinstance(capacity, Mapping):
        filled = dict(capacity or {})
        if filled.get("largest_cluster_node_count") is None:
            processor = data.get("processor")
            depth = (
                processor.get("max_cluster_queue_depth")
                if isinstance(processor, Mapping)
                else None
            )
            filled["largest_cluster_node_count"] = legacy_largest_cluster_node_count(
                depth
                if isinstance(depth, int) and not isinstance(depth, bool)
                else ProcessorConfig().max_cluster_queue_depth
            )
        if filled.get("managed_node_count") is None:
            filled["managed_node_count"] = filled["largest_cluster_node_count"]
        data["capacity"] = filled
    return data


def _missing_node_counts(raw_config: Mapping[str, object]) -> bool:
    capacity = raw_config.get("capacity")
    return not (
        isinstance(capacity, Mapping) and "largest_cluster_node_count" in capacity
    )


def _stored_config_sha256(
    raw_config: Mapping[str, object],
    config: AdminConfig,
) -> str:
    if "aurora" not in raw_config or _missing_node_counts(raw_config):
        return canonical_sha256(dict(raw_config))
    return config.sha256()


def _stored_role_sha256(
    raw_config: Mapping[str, object],
    config: AdminConfig,
) -> dict[str, str]:
    # The first release digested capacity-only role payloads with no common
    # section at all.
    if set(raw_config) == {"schema_version", "capacity"}:
        capacity = config.capacity
        payloads = {
            "ingress": {"telemetry_spool_enabled": capacity.telemetry_spool.enabled},
            "worker": {
                "control_worker_replicas": capacity.control_worker_replicas,
                "remediation": capacity.remediation.as_dict(),
            },
            "spool": capacity.telemetry_spool.as_dict(),
        }
        return {role: canonical_sha256(payload) for role, payload in payloads.items()}
    return config.role_sha256(node_counts=not _missing_node_counts(raw_config))


# ---------------------------------------------------------------------------
# Persisted state.
# ---------------------------------------------------------------------------


def admin_config_desired_path(state_dir: Path) -> Path:
    return state_dir.expanduser().resolve() / ADMIN_CONFIG_DESIRED


def admin_config_pending_path(state_dir: Path) -> Path:
    return state_dir.expanduser().resolve() / ADMIN_CONFIG_PENDING


def admin_config_history_path(state_dir: Path, history_id: str) -> Path:
    if not HISTORY_ID_PATTERN.fullmatch(history_id):
        raise AdminConfigError("admin config history id is invalid")
    return state_dir.expanduser().resolve() / ADMIN_CONFIG_HISTORY / history_id


def admin_config_history_id(started_at: datetime, config_sha256: str) -> str:
    """``<started>-<config_sha256>``: one apply attempt's history directory."""

    return f"{_compact_timestamp(started_at)}-{config_sha256}"


def _compact_timestamp(value: datetime) -> str:
    return value.astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")


def _utc_timestamp(value: datetime | None = None) -> str:
    return (value or datetime.now(UTC)).isoformat()


def _parse_timestamp(value: object, description: str) -> datetime:
    if not isinstance(value, str):
        raise AdminConfigError(f"{description} is invalid")
    try:
        timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise AdminConfigError(f"{description} is invalid") from exc
    if timestamp.tzinfo is None:
        raise AdminConfigError(f"{description} must include a timezone")
    return timestamp


def _read_json(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AdminConfigError(f"{description} is invalid") from exc
    if not isinstance(value, dict):
        raise AdminConfigError(f"{description} must be a JSON object")
    return value


@contextmanager
def admin_config_write_lock(state_dir: Path) -> Iterator[None]:
    """The site operation lock, refused in the admin-config error family.

    Membership changes, Profile approvals and config applies all take this one
    lock, so two administrators cannot mutate one site at once.
    """

    try:
        with site_operation_lock(state_dir, wait=False):
            yield
    except SiteOperationBusy as exc:
        raise AdminConfigError("another administrator mutation is in progress") from exc


def _desired_record(
    config: AdminConfig,
    *,
    source: str,
    reference: str | None = None,
    approver_identity: str | None = None,
    updated_at: datetime | None = None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "schema_version": 1,
        "config": config.as_dict(),
        "config_sha256": config.sha256(),
        "role_sha256": config.role_sha256(),
        "source": source,
        "updated_at": _utc_timestamp(updated_at),
    }
    if reference is not None:
        record["reference"] = reference
    if approver_identity is not None:
        record["approver_identity"] = approver_identity
    return record


def load_desired_admin_config(
    state_dir: Path, *, migrate_legacy: bool = False
) -> AdminConfig:
    """The persisted desired config, or the release defaults before any exists.

    ``migrate_legacy`` rewrites a record from an older release into today's
    shape after it has been verified against its own digests; the caller
    holds the site operation lock. Without it the read is side-effect free.
    """

    path = admin_config_desired_path(state_dir)
    if not path.is_file():
        return default_admin_config()
    record = _read_json(path, "desired admin config")
    if record.get("schema_version") != 1:
        raise AdminConfigError("desired admin config schema is invalid")
    raw_config = record.get("config")
    if not isinstance(raw_config, Mapping):
        raise AdminConfigError("desired admin config content must be a mapping")
    config = AdminConfig.from_mapping(raw_config)
    if record.get("config_sha256") != _stored_config_sha256(raw_config, config):
        raise AdminConfigError("desired admin config digest does not match its content")
    if record.get("role_sha256") != _stored_role_sha256(raw_config, config):
        raise AdminConfigError("desired admin config role digests do not match")
    legacy = "aurora" not in raw_config or _missing_node_counts(raw_config)
    if migrate_legacy and legacy:
        source = record.get("source")
        migration_source = (
            "legacy-capacity-migration"
            if set(raw_config) == {"schema_version", "capacity"}
            else "legacy-admin-config-migration"
        )
        if isinstance(source, str) and source:
            migration_source += f":{source}"
        migrated = _desired_record(config, source=migration_source)
        for key in ("plan_sha256", "reference", "approver_identity"):
            if isinstance(record.get(key), str):
                migrated[key] = record[key]
        write_json_atomic(path, migrated)
    return config


def persist_desired_admin_config(
    state_dir: Path,
    *,
    config: AdminConfig,
    source: str,
) -> Path:
    path = admin_config_desired_path(state_dir)
    write_json_atomic(path, _desired_record(config, source=source))
    return path


# ---------------------------------------------------------------------------
# The change plan: what an apply would do, computed locally from two configs.
# ---------------------------------------------------------------------------


def _flatten(value: object, *, prefix: str = "") -> dict[str, object]:
    if not isinstance(value, Mapping):
        return {prefix: value}
    result: dict[str, object] = {}
    for key in sorted(value):
        path = f"{prefix}.{key}" if prefix else str(key)
        result.update(_flatten(value[key], prefix=path))
    return result


def _changes(before: AdminConfig, desired: AdminConfig) -> list[dict[str, object]]:
    old = _flatten({k: v for k, v in before.as_dict().items() if k != "schema_version"})
    new = _flatten(
        {k: v for k, v in desired.as_dict().items() if k != "schema_version"}
    )
    return [
        {"field": field, "before": old.get(field), "after": new.get(field)}
        for field in sorted(set(old) | set(new))
        if old.get(field) != new.get(field)
    ]


def _change_facts(before: AdminConfig, desired: AdminConfig) -> dict[str, Any]:
    before_roles = before.role_sha256()
    desired_roles = desired.role_sha256()
    return {
        "before_config": before.as_dict(),
        "before_config_sha256": before.sha256(),
        "desired_config": desired.as_dict(),
        "desired_config_sha256": desired.sha256(),
        "affected_roles": sorted(
            role
            for role in ADMIN_CONFIG_ROLES
            if before_roles[role] != desired_roles[role]
        ),
        "aurora_changed": before.aurora != desired.aurora,
        "changes": _changes(before, desired),
    }


def admin_config_change_plan(
    before: AdminConfig,
    desired: AdminConfig,
    *,
    source: str,
) -> dict[str, Any]:
    """The roles that would roll, the fields that change and whether Aurora moves.

    Pure and local: this is what ``--dry-run`` prints and what the pending
    record carries. ``desired`` is validated in full here, since it is about
    to be authored.
    """

    desired.validate()
    return {
        "schema_version": 1,
        "source": source,
        **_change_facts(before, desired),
    }


# ---------------------------------------------------------------------------
# The pending apply and its history.
# ---------------------------------------------------------------------------

_PENDING_FIELDS = frozenset(
    {
        "schema_version",
        "history",
        "started_at",
        "source",
        "reference",
        "approver_identity",
        "site_identity",
        "release_identity",
        "before_config",
        "before_config_sha256",
        "desired_config",
        "desired_config_sha256",
        "affected_roles",
        "aurora_changed",
        "changes",
    }
)


@dataclass(frozen=True)
class AdminConfigApply:
    """An apply in flight: the pending record with its two configs decoded."""

    record: dict[str, Any]
    before: AdminConfig
    desired: AdminConfig
    resumed: bool

    @property
    def no_op(self) -> bool:
        return not self.record["changes"]

    @property
    def config_sha256(self) -> str:
        return str(self.record["desired_config_sha256"])

    @property
    def affected_roles(self) -> list[str]:
        return [str(role) for role in self.record["affected_roles"]]

    @property
    def aurora_changed(self) -> bool:
        return bool(self.record["aurora_changed"])

    @property
    def history(self) -> str:
        return str(self.record["history"])

    @property
    def reference(self) -> str:
        return str(self.record["reference"])

    @property
    def approver_identity(self) -> str:
        return str(self.record["approver_identity"])


def config_command(state_dir: Path) -> str:
    """The exact command an error message tells the administrator to run next."""

    return f"gpu-fault-admin config --state-dir {state_dir}"


def default_admin_config_reference(approver_identity: str, started_at: datetime) -> str:
    """``<approver>:<started>`` when the administrator gives no ``--reference``."""

    stamp = _compact_timestamp(started_at)
    head = re.sub(r"[^A-Za-z0-9._:/-]", "-", approver_identity.strip()).lstrip("._:/-")
    head = (head or "operator")[: 128 - len(stamp) - 1]
    return f"{head}:{stamp}"


def _validated_site_identity(value: object) -> dict[str, str]:
    data = mapping_field(
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
    return normalized


def _validated_release_identity(value: object) -> dict[str, object]:
    data = mapping_field(
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


def _validated_pending(document: dict[str, Any], state_dir: Path) -> dict[str, Any]:
    try:
        if set(document) != _PENDING_FIELDS or document.get("schema_version") != 1:
            raise AdminConfigError("schema")
        before = AdminConfig.from_mapping(document["before_config"])
        desired = AdminConfig.from_mapping(document["desired_config"])
        for key, value in _change_facts(before, desired).items():
            if document[key] != value:
                raise AdminConfigError(f"{key} does not match its configuration")
        _validated_site_identity(document["site_identity"])
        _validated_release_identity(document["release_identity"])
        history = document["history"]
        if not isinstance(history, str) or not HISTORY_ID_PATTERN.fullmatch(history):
            raise AdminConfigError("history id")
        if not history.endswith(desired.sha256()):
            raise AdminConfigError("history id does not name the target config")
        reference = document["reference"]
        if not isinstance(reference, str) or not APPROVAL_PATTERN.fullmatch(reference):
            raise AdminConfigError("reference")
        approver = document["approver_identity"]
        if not isinstance(approver, str) or not approver.strip():
            raise AdminConfigError("approver_identity")
        _parse_timestamp(document["started_at"], "started_at")
    except AdminConfigError as exc:
        raise AdminConfigError(
            f"pending admin config apply record is invalid ({exc}); confirm the "
            "live release is committed, remove "
            f"{admin_config_pending_path(state_dir)} and rerun "
            f"{config_command(state_dir)}"
        ) from exc
    return document


def load_pending_admin_config_apply(state_dir: Path) -> dict[str, Any] | None:
    path = admin_config_pending_path(state_dir)
    if not path.is_file():
        return None
    return _validated_pending(_read_json(path, "pending admin config apply"), state_dir)


def matching_pending_admin_config_apply(
    state_dir: Path,
    *,
    site_identity: Mapping[str, str],
    release_identity: Mapping[str, object],
    desired: AdminConfig,
) -> dict[str, Any] | None:
    """The pending record if it targets this site, release and config."""

    pending = load_pending_admin_config_apply(state_dir)
    if pending is None:
        return None
    matches = (
        pending["site_identity"] == _validated_site_identity(dict(site_identity))
        and pending["release_identity"]
        == _validated_release_identity(dict(release_identity))
        and pending["desired_config_sha256"] == desired.sha256()
    )
    return pending if matches else None


def _result_record(
    record: Mapping[str, Any],
    *,
    status: str,
    release_id: str | None,
    completed_at: datetime | None,
    error: str | None = None,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "schema_version": 1,
        "status": status,
        "history": record["history"],
        "config_sha256": record["desired_config_sha256"],
        "before_config_sha256": record["before_config_sha256"],
        "affected_roles": record["affected_roles"],
        "aurora_changed": record["aurora_changed"],
        "source": record["source"],
        "reference": record["reference"],
        "approver_identity": record["approver_identity"],
        "release_id": release_id,
        "started_at": record["started_at"],
        "completed_at": _utc_timestamp(completed_at),
    }
    if error:
        result["error"] = error
    if details:
        result["details"] = details
    return result


def _write_attempt(state_dir: Path, apply: AdminConfigApply) -> None:
    desired_path = admin_config_desired_path(state_dir)
    before = (
        _read_json(desired_path, "desired admin config")
        if desired_path.is_file()
        else _desired_record(apply.before, source="release-defaults")
    )
    history = admin_config_history_path(state_dir, apply.history)
    write_json_atomic(history / "before.json", before)
    write_json_atomic(admin_config_pending_path(state_dir), apply.record)
    write_json_atomic(
        desired_path,
        _desired_record(
            apply.desired,
            source=f"pending-apply:{apply.history}",
            reference=apply.reference,
            approver_identity=apply.approver_identity,
        ),
    )


def _supersede_pending(
    state_dir: Path,
    pending: Mapping[str, Any],
    *,
    completed_at: datetime,
) -> None:
    history = admin_config_history_path(state_dir, str(pending["history"]))
    if not (history / "result.json").is_file():
        write_json_atomic(
            history / "result.json",
            _result_record(
                pending,
                status="SUPERSEDED",
                release_id=None,
                completed_at=completed_at,
                error="a different administrator config apply started",
            ),
        )
    admin_config_pending_path(state_dir).unlink(missing_ok=True)


def _resume_apply(
    state_dir: Path,
    pending: dict[str, Any],
    *,
    current: AdminConfig,
    desired: AdminConfig,
    attempt: dict[str, str],
    started_at: datetime,
) -> AdminConfigApply:
    before = AdminConfig.from_mapping(pending["before_config"])
    if current not in {before, desired}:
        raise AdminConfigError(
            "desired admin config changed after the interrupted apply began; "
            f"review {admin_config_desired_path(state_dir)} against "
            f"{admin_config_pending_path(state_dir)}, then rerun "
            f"{config_command(state_dir)}"
        )
    history = admin_config_history_path(state_dir, str(pending["history"]))
    finished = (history / "result.json").is_file()
    record = pending
    if finished:
        # The previous attempt failed and was recorded; this rerun is a new
        # attempt with its own directory, approver and reference. Attempt
        # directories are named to the second, so a rerun inside the same
        # second as the recorded one is pushed forward until the name is free
        # rather than overwriting that record.
        while (
            admin_config_history_path(
                state_dir, admin_config_history_id(started_at, desired.sha256())
            )
            / "result.json"
        ).is_file():
            started_at += timedelta(seconds=1)
        record = {
            **pending,
            **attempt,
            "started_at": _utc_timestamp(started_at),
            "history": admin_config_history_id(started_at, desired.sha256()),
        }
    apply = AdminConfigApply(
        record=record, before=before, desired=desired, resumed=True
    )
    if finished or current != desired:
        _write_attempt(state_dir, apply)
    return apply


def begin_admin_config_apply(
    state_dir: Path,
    *,
    site_identity: Mapping[str, str],
    release_identity: Mapping[str, object],
    desired: AdminConfig,
    source: str,
    approver_identity: str,
    reference: str | None = None,
    started_at: datetime | None = None,
) -> AdminConfigApply:
    """Record the apply about to happen; the caller holds the site operation lock.

    When there is something to change this writes, in order, the attempt's
    ``before.json``, ``pending.json`` and ``desired.json`` as the target, so the
    release engine renders the new config. A pending record for the same site,
    release and target is resumed: a crashed attempt keeps its history
    directory, a recorded failure gets a new attempt. A pending record for a
    different target is closed as SUPERSEDED first. A no-op writes nothing.
    """

    started = started_at or datetime.now(UTC)
    approver = approver_identity.strip()
    if not approver:
        raise AdminConfigError("admin config approver identity must be non-empty")
    normalized_reference = (
        reference or default_admin_config_reference(approver, started)
    ).strip()
    if not APPROVAL_PATTERN.fullmatch(normalized_reference):
        raise AdminConfigError(
            "--reference must be 3..128 characters from letters, digits, . _ : / -"
        )
    identity = _validated_site_identity(dict(site_identity))
    release = _validated_release_identity(dict(release_identity))
    current = load_desired_admin_config(state_dir)
    attempt = {
        "reference": normalized_reference,
        "approver_identity": approver,
        "started_at": _utc_timestamp(started),
    }
    pending = load_pending_admin_config_apply(state_dir)
    if pending is not None:
        if (
            pending["site_identity"] == identity
            and pending["release_identity"] == release
            and pending["desired_config_sha256"] == desired.sha256()
        ):
            return _resume_apply(
                state_dir,
                pending,
                current=current,
                desired=desired,
                attempt=attempt,
                started_at=started,
            )
        _supersede_pending(state_dir, pending, completed_at=started)
    record = {
        **admin_config_change_plan(current, desired, source=source),
        "site_identity": identity,
        "release_identity": release,
        **attempt,
        "history": admin_config_history_id(started, desired.sha256()),
    }
    apply = AdminConfigApply(
        record=record, before=current, desired=desired, resumed=False
    )
    if not apply.no_op:
        _write_attempt(state_dir, apply)
    return apply


def complete_admin_config_apply(
    state_dir: Path,
    *,
    config_sha256: str,
    release_id: str,
    success: bool,
    error: str | None = None,
    details: dict[str, Any] | None = None,
    restore_before: bool = True,
    completed_at: datetime | None = None,
) -> Path:
    """Close the pending apply; the caller holds the site operation lock.

    Success writes the result and clears ``pending.json``. Failure keeps it for
    the resume and, unless ``restore_before=False``, puts ``desired.json`` back
    to the config before the attempt. ``restore_before=False`` is for a failure
    after the roles already run the target (Aurora did not settle in time):
    the target is then what is live, and the rerun only has to finish waiting.
    """

    pending = load_pending_admin_config_apply(state_dir)
    if pending is None or pending["desired_config_sha256"] != config_sha256:
        raise AdminConfigError(
            "the pending admin config apply changed while this one ran; rerun "
            f"{config_command(state_dir)}"
        )
    history = admin_config_history_path(state_dir, str(pending["history"]))
    desired_path = admin_config_desired_path(state_dir)
    if not success and restore_before:
        write_json_atomic(
            desired_path,
            _desired_record(
                AdminConfig.from_mapping(pending["before_config"]),
                source=f"rollback-after-failed-apply:{pending['history']}",
                reference=str(pending["reference"]),
                approver_identity=str(pending["approver_identity"]),
            ),
        )
    write_json_atomic(
        history / "after.json",
        _read_json(desired_path, "desired admin config"),
    )
    result_path = history / "result.json"
    write_json_atomic(
        result_path,
        _result_record(
            pending,
            status="APPLIED" if success else "FAILED",
            release_id=release_id,
            completed_at=completed_at,
            error=error,
            details=details,
        ),
    )
    if success:
        admin_config_pending_path(state_dir).unlink(missing_ok=True)
    return result_path

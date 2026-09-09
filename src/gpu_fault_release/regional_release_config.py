from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped,unused-ignore]

from gpu_fault.admin.config import AdminConfig
from gpu_fault.failure_domains import FAILURE_DOMAIN_LABELS
from gpu_fault_release import repository_root

ROOT = repository_root()
DEFAULT_NAMESPACE = "gpu-fault-system"
MAX_UPGRADE_PARALLEL_CLUSTERS = 8
DIGEST_IMAGE_PATTERN = re.compile(r"^.+@sha256:[0-9a-f]{64}$")
AWS_REGION_PATTERN = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)+-[0-9]+$")
RUNTIME_PROFILE_VERSION_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}$")
EMAIL_PATTERN = re.compile(r"^[^\s@,]+@[^\s@,]+\.[^\s@,]+$")
EKS_ARN_PATTERN = re.compile(
    r"^arn:[^:]+:eks:(?P<region>[^:]+):(?P<account>[^:]+):"
    r"cluster/(?P<name>[^/]+)$"
)


class ReleaseError(RuntimeError):
    pass


class ClusterLocalReleaseError(ReleaseError):
    pass


class PartialClusterRolloutError(ReleaseError):
    pass


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def required_text(value: object, field: str) -> str:
    normalized = str(value or "").strip()
    if not normalized or normalized.startswith("REPLACE_"):
        raise ReleaseError(f"{field} is required and must not be a placeholder")
    return normalized


def required_email(value: object, field: str) -> str:
    normalized = required_text(value, field)
    if not EMAIL_PATTERN.fullmatch(normalized):
        raise ReleaseError(f"{field} must be a valid email address")
    return normalized


def email_list(value: object, field: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ReleaseError(f"{field} must be a list")
    normalized = tuple(
        dict.fromkeys(required_email(item, f"{field}[]") for item in value)
    )
    if not normalized:
        raise ReleaseError(f"{field} requires at least one email address")
    return normalized


def email_subject_prefix(value: object, field: str) -> str:
    normalized = str(value or "").strip()
    if len(normalized) > 64 or "\n" in normalized or "\r" in normalized:
        raise ReleaseError(f"{field} must be a single line of at most 64 characters")
    return normalized


def validate_aws_region(value: object, field: str = "aws_region") -> str:
    region = required_text(value, field)
    if not AWS_REGION_PATTERN.fullmatch(region):
        raise ReleaseError(f"{field} is not a valid AWS Region: {region}")
    return region


def validate_runtime_profile_version(
    value: object,
    field: str = "runtime_profile.version",
) -> str:
    version = required_text(value, field)
    if not RUNTIME_PROFILE_VERSION_PATTERN.fullmatch(version):
        raise ReleaseError(f"{field} is not a valid Runtime Profile version")
    return version


def validate_eks_arn(value: object, *, field: str, expected_region: str) -> str:
    arn = required_text(value, field)
    match = EKS_ARN_PATTERN.fullmatch(arn)
    if match is None:
        raise ReleaseError(f"{field} must be a complete EKS cluster ARN")
    if match.group("region") != expected_region:
        raise ReleaseError(
            f"{field} Region {match.group('region')} does not match "
            f"aws_region {expected_region}"
        )
    return arn


def validate_regional_arn(
    value: object,
    *,
    field: str,
    service: str,
    expected_region: str,
) -> str:
    arn = required_text(value, field)
    parts = arn.split(":", 5)
    if len(parts) != 6 or parts[0] != "arn" or parts[2] != service:
        raise ReleaseError(f"{field} must be a complete {service} ARN")
    if parts[3] != expected_region:
        raise ReleaseError(
            f"{field} Region {parts[3]} does not match aws_region {expected_region}"
        )
    return arn


@dataclass(frozen=True)
class ClusterTarget:
    cluster_id: str
    context: str
    executor_irsa_role_arn: str
    region: str
    hyperpod_cluster_name: str
    eks_cluster_arn: str
    token_file: str | None = None
    ca_file: str | None = None
    control_plane_url: str | None = None
    allowed_namespaces: tuple[str, ...] = ()
    agent_endpoint_allowed_cidrs: tuple[str, ...] = ()
    fleet_master_file: str | None = None
    #: IRSA role for the per-cluster ADOT collector (data-plane review F7):
    #: ``aps:RemoteWrite`` on the site's AMP workspace, trusted by this
    #: cluster's OIDC issuer for ``gpu-fault-system/gpu-fault-adot-dataplane``.
    #: Optional: without it the release skips the collector for this cluster
    #: and says so, rather than applying a collector with no credentials.
    adot_irsa_role_arn: str | None = None

    @classmethod
    def from_mapping(
        cls,
        value: dict[str, Any],
        *,
        expected_region: str,
    ) -> ClusterTarget:
        required = (
            "cluster_id",
            "context",
            "executor_irsa_role_arn",
            "region",
            "hyperpod_cluster_name",
            "eks_cluster_arn",
        )
        missing = [name for name in required if not value.get(name)]
        if missing:
            raise ReleaseError("cluster target is missing: " + ", ".join(missing))
        region = validate_aws_region(value["region"], "cluster target region")
        if region != expected_region:
            raise ReleaseError(
                f"cluster target Region {region} does not match "
                f"aws_region {expected_region}"
            )
        eks_cluster_arn = validate_eks_arn(
            value["eks_cluster_arn"],
            field="cluster target eks_cluster_arn",
            expected_region=expected_region,
        )
        namespaces = tuple(
            sorted(
                {
                    str(item).strip()
                    for item in value.get("allowed_namespaces", [])
                    if str(item).strip()
                }
            )
        )
        endpoint_cidrs = tuple(
            sorted(
                {
                    str(item).strip()
                    for item in value.get("agent_endpoint_allowed_cidrs", [])
                    if str(item).strip()
                }
            )
        )
        return cls(
            cluster_id=required_text(value["cluster_id"], "cluster target cluster_id"),
            context=required_text(value["context"], "cluster target context"),
            executor_irsa_role_arn=required_text(
                value["executor_irsa_role_arn"],
                "cluster target executor_irsa_role_arn",
            ),
            region=region,
            hyperpod_cluster_name=required_text(
                value["hyperpod_cluster_name"],
                "cluster target hyperpod_cluster_name",
            ),
            eks_cluster_arn=eks_cluster_arn,
            token_file=value.get("token_file"),
            ca_file=value.get("ca_file"),
            control_plane_url=value.get("control_plane_url"),
            allowed_namespaces=namespaces,
            agent_endpoint_allowed_cidrs=endpoint_cidrs,
            adot_irsa_role_arn=(
                required_text(
                    value["adot_irsa_role_arn"],
                    "cluster target adot_irsa_role_arn",
                )
                if value.get("adot_irsa_role_arn")
                else None
            ),
            fleet_master_file=value.get("fleet_master_file"),
        )


@dataclass(frozen=True)
class RegionalHealthConfig:
    aurora_cluster_id: str | None = None
    amp_workspace_id: str | None = None
    amp_rule_namespace: str = "gpu-fault-control-plane-capacity"
    sns_topic_arn: str | None = None
    certificate_min_validity_days: int = 30
    remote_command_max_unclaimed_seconds: int = 300
    require_confirmed_sns_subscription: bool = True

    @classmethod
    def from_mapping(
        cls,
        value: dict[str, Any],
        *,
        expected_region: str,
    ) -> RegionalHealthConfig:
        if not value:
            return cls()
        aurora_cluster_id = (
            required_text(value.get("aurora_cluster_id"), "health.aurora_cluster_id")
            if value.get("aurora_cluster_id")
            else None
        )
        amp_workspace_id = (
            required_text(value.get("amp_workspace_id"), "health.amp_workspace_id")
            if value.get("amp_workspace_id")
            else None
        )
        amp_rule_namespace = str(
            value.get(
                "amp_rule_namespace",
                "gpu-fault-control-plane-capacity",
            )
        ).strip()
        if not amp_rule_namespace:
            raise ReleaseError("health.amp_rule_namespace must not be empty")
        sns_topic_arn = None
        if value.get("sns_topic_arn"):
            sns_topic_arn = validate_regional_arn(
                value.get("sns_topic_arn"),
                field="health.sns_topic_arn",
                service="sns",
                expected_region=expected_region,
            )
        try:
            certificate_days = int(value.get("certificate_min_validity_days", 30))
            max_unclaimed = int(value.get("remote_command_max_unclaimed_seconds", 300))
        except (TypeError, ValueError) as exc:
            raise ReleaseError(
                "health validity and remote-command thresholds must be integers"
            ) from exc
        if not 1 <= certificate_days <= 3650:
            raise ReleaseError(
                "health.certificate_min_validity_days must be within 1..3650"
            )
        if not 1 <= max_unclaimed <= 86400:
            raise ReleaseError(
                "health.remote_command_max_unclaimed_seconds must be within 1..86400"
            )
        return cls(
            aurora_cluster_id=aurora_cluster_id,
            amp_workspace_id=amp_workspace_id,
            amp_rule_namespace=amp_rule_namespace,
            sns_topic_arn=sns_topic_arn,
            certificate_min_validity_days=certificate_days,
            remote_command_max_unclaimed_seconds=max_unclaimed,
            require_confirmed_sns_subscription=bool(
                value.get("require_confirmed_sns_subscription", True)
            ),
        )


NOTIFICATION_CHANNEL_SNS = "sns"
NOTIFICATION_CHANNEL_SES = "ses"
NOTIFICATION_CHANNELS = (NOTIFICATION_CHANNEL_SNS, NOTIFICATION_CHANNEL_SES)
NOTIFICATION_ENVIRONMENT = (
    "GPU_FAULT_NOTIFICATION_CHANNEL",
    "GPU_FAULT_SNS_TOPIC_ARN",
)


@dataclass(frozen=True)
class RegionalNotificationConfig:
    """``notifications`` from the release config: ``site.yaml`` ``spec.notifications``.

    ``channel`` defaults the way the site does: a release config written before
    the key existed carries an ``email_sender`` and stays on ``ses``, so a
    rollback onto such a record keeps the channel it shipped with; anything
    else is ``sns``. On ``sns`` only ``admin_email`` is required.
    """

    allow_email: bool = False
    acknowledge_external_alert_channel: bool = True
    admin_email: str | None = None
    email_sender: str | None = None
    email_recipients: tuple[str, ...] = ()
    email_subject_prefix: str = ""
    channel: str = NOTIFICATION_CHANNEL_SNS

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> RegionalNotificationConfig:
        if not value:
            return cls()
        channel = value.get("channel")
        if channel is None:
            channel = (
                NOTIFICATION_CHANNEL_SES
                if value.get("email_sender")
                else NOTIFICATION_CHANNEL_SNS
            )
        elif not isinstance(channel, str) or channel not in NOTIFICATION_CHANNELS:
            raise ReleaseError(
                "notifications.channel must be one of "
                + ", ".join(NOTIFICATION_CHANNELS)
            )
        allow_email = bool(value.get("allow_email", True))
        acknowledge = bool(value.get("acknowledge_external_alert_channel", False))
        admin_email = (
            required_email(value.get("admin_email"), "notifications.admin_email")
            if value.get("admin_email")
            else None
        )
        email_sender = (
            required_email(value.get("email_sender"), "notifications.email_sender")
            if value.get("email_sender")
            else None
        )
        email_recipients = (
            email_list(
                value.get("email_recipients"),
                "notifications.email_recipients",
            )
            if value.get("email_recipients")
            else ((admin_email,) if admin_email is not None else ())
        )
        subject_prefix = email_subject_prefix(
            value.get("email_subject_prefix"),
            "notifications.email_subject_prefix",
        )
        if not allow_email and not acknowledge:
            raise ReleaseError(
                "notifications must allow email or acknowledge an external alert channel"
            )
        if allow_email and channel == NOTIFICATION_CHANNEL_SNS and admin_email is None:
            raise ReleaseError("SNS notifications require notifications.admin_email")
        if (
            allow_email
            and channel == NOTIFICATION_CHANNEL_SES
            and (admin_email is None or email_sender is None or not email_recipients)
        ):
            raise ReleaseError(
                "email notifications require notifications.admin_email "
                "notifications.email_sender, and at least one recipient"
            )
        return cls(
            allow_email=allow_email,
            acknowledge_external_alert_channel=acknowledge,
            admin_email=admin_email,
            email_sender=email_sender,
            email_recipients=email_recipients,
            email_subject_prefix=subject_prefix,
            channel=channel,
        )


@dataclass(frozen=True)
class RegionalDnsConfig:
    hosted_zone_id: str | None = None
    hostname: str | None = None

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> RegionalDnsConfig:
        if not value:
            return cls()
        hosted_zone_id = required_text(
            value.get("hosted_zone_id"),
            "dns.hosted_zone_id",
        )
        hostname = required_text(value.get("hostname"), "dns.hostname").rstrip(".")
        if "." not in hostname:
            raise ReleaseError("dns.hostname must be a DNS name")
        return cls(hosted_zone_id=hosted_zone_id, hostname=hostname)


@dataclass(frozen=True)
class ReleaseArtifacts:
    release_id: str
    wheel: Path
    executor_wheel: Path
    node_wheel: Path
    bundle: Path
    database_schema_version: int
    agent_protocol_version: int
    executor_protocol_version: int
    component_digests: dict[str, str]
    manifest: dict[str, Any] | None
    manifest_schema_version: int
    delivery_identity: dict[str, Any]
    delivery_sha256: str
    delivery_component_digests: dict[str, str]
    locked_images: dict[str, str]
    node_template_sha256: str


def _resolved_artifact_paths(
    values: tuple[Path, Path, Path, Path],
    *,
    base: Path,
) -> tuple[Path, Path, Path, Path]:
    resolved = tuple(value if value.is_absolute() else base / value for value in values)
    return resolved[0], resolved[1], resolved[2], resolved[3]


def _require_artifact_files(paths: tuple[Path, Path, Path, Path]) -> None:
    if not all(path.is_file() for path in paths):
        raise ReleaseError("release component wheels and bundle must exist")


def _parse_delivery_identity(
    manifest: dict[str, Any],
    components: dict[str, Any],
) -> tuple[
    dict[str, Any],
    str,
    dict[str, str],
    dict[str, str],
    str,
]:
    delivery = dict(manifest.get("delivery") or {})
    delivery_sha256 = str(delivery.get("sha256") or "")
    unsigned = dict(delivery)
    unsigned.pop("sha256", None)
    if (
        not re.fullmatch(r"[0-9a-f]{64}", delivery_sha256)
        or canonical_sha256(unsigned) != delivery_sha256
    ):
        raise ReleaseError("release delivery identity digest is invalid")
    if (
        manifest.get("deployable") is not True
        or delivery.get("runtime_prebuilt") is not True
    ):
        raise ReleaseError(
            "schema v3 release requires a prebuilt deployable runtime image"
        )
    raw_components = dict(delivery.get("components") or {})
    legacy_components = {
        "collector",
        "cpu",
        "dcgm",
        "endpoint",
        "executor",
        "node",
        "observability",
        "schema",
        "watcher",
    }
    role_components = {"cpu_ingress", "cpu_spool", "cpu_worker"}
    if set(raw_components) not in {
        frozenset(legacy_components),
        frozenset({*legacy_components, *role_components}),
    }:
        raise ReleaseError("release delivery components are incomplete")
    component_digests = {
        name: str((raw_components[name] or {}).get("sha256") or "")
        for name in sorted(raw_components)
    }
    if not role_components.issubset(component_digests):
        component_digests.update(
            {name: component_digests["cpu"] for name in role_components}
        )
    if any(
        not re.fullmatch(r"[0-9a-f]{64}", value) for value in component_digests.values()
    ):
        raise ReleaseError("release delivery component digests are invalid")
    raw_images = dict(delivery.get("images") or {})
    expected_images = {
        "runtime",
        "node_installer",
        "dcgm_exporter",
        "adot",
    }
    if set(raw_images) != expected_images:
        raise ReleaseError("release image identity is incomplete")
    locked_images = {
        name: str((raw_images[name] or {}).get("reference") or "")
        for name in sorted(raw_images)
    }
    if any(
        not DIGEST_IMAGE_PATTERN.fullmatch(value) for value in locked_images.values()
    ):
        raise ReleaseError("release images must use immutable sha256 references")
    node_template_sha256 = str(
        (components.get("node_bundle") or {}).get("template_sha256")
        or (delivery.get("node_template_inputs") or {}).get("sha256")
        or ""
    )
    if not re.fullmatch(r"[0-9a-f]{64}", node_template_sha256):
        raise ReleaseError("release node template digest is invalid")
    database = dict(manifest.get("database") or {})
    if database.get("rollback_compatible"):
        # The runtime cannot honour this promise: a Pod refuses to start unless
        # `gpu_fault_schema_version` and the migration history match its wheel
        # exactly, so an old wheel rolled back onto a new schema CrashLoops.
        # Declaring the change compatible only let the engine attempt a rollback
        # that could never come up.
        raise ReleaseError(
            "release manifest declares database.rollback_compatible: true, which "
            "the runtime cannot honour: PostgreSQL store start-up requires an "
            "exact schema version and migration history match, so a schema "
            "change is never rollback-compatible. Remove the key; deploy a "
            "schema change with --accept-schema-change (fail-forward)"
        )
    return (
        delivery,
        delivery_sha256,
        component_digests,
        locked_images,
        node_template_sha256,
    )


def parse_delivery_identity(
    manifest: dict[str, Any],
    components: dict[str, Any],
) -> tuple[
    dict[str, Any],
    str,
    dict[str, str],
    dict[str, str],
    str,
]:
    return _parse_delivery_identity(manifest, components)


def load_release_artifacts(
    release: dict[str, Any],
    *,
    config_path: Path,
) -> ReleaseArtifacts:
    release_manifest = release.get("manifest")
    manifest: dict[str, Any] | None = None
    manifest_schema_version = 1
    delivery_identity: dict[str, Any] = {}
    delivery_sha256 = ""
    delivery_component_digests: dict[str, str] = {}
    locked_images: dict[str, str] = {}
    node_template_sha256 = ""
    if release_manifest:
        manifest_path = Path(str(release_manifest))
        if not manifest_path.is_absolute():
            manifest_path = config_path.parent / manifest_path
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest_schema_version = int(manifest.get("schema_version", 1))
        components = dict(manifest.get("components") or {})
        control_component = dict(components.get("control_plane") or {})
        executor_component = dict(components.get("executor") or {})
        node_component = dict(components.get("node_runtime") or {})
        wheel, executor_wheel, node_wheel, bundle = _resolved_artifact_paths(
            (
                Path(str(control_component.get("wheel") or manifest["wheel"])),
                Path(str(executor_component.get("wheel") or manifest["wheel"])),
                Path(str(node_component.get("wheel") or manifest["wheel"])),
                Path(str(manifest["bundle"])),
            ),
            base=ROOT,
        )
        _require_artifact_files((wheel, executor_wheel, node_wheel, bundle))
        release_id = required_text(
            manifest.get("release_id") or _sha256(wheel)[:12],
            "release manifest release_id",
        )
        database_schema_version = int(manifest.get("database_schema_version", 0))
        protocols = dict(manifest.get("protocol_versions") or {})
        agent_protocol_version = int(protocols.get("agent", 3))
        executor_protocol_version = int(protocols.get("executor", 2))
        component_digests = {
            "control_plane": str(
                control_component.get("module_digest")
                or manifest.get("module_digest")
                or ""
            ),
            "executor": str(
                executor_component.get("module_digest")
                or manifest.get("module_digest")
                or ""
            ),
            "node_runtime": str(
                node_component.get("module_digest")
                or manifest.get("module_digest")
                or ""
            ),
        }
        expected_hashes = {
            "control_plane": (
                control_component.get("wheel_sha256") or manifest.get("wheel_sha256")
            ),
            "executor": (
                executor_component.get("wheel_sha256") or manifest.get("wheel_sha256")
            ),
            "node_runtime": (
                node_component.get("wheel_sha256") or manifest.get("wheel_sha256")
            ),
            "bundle": manifest.get("bundle_sha256"),
        }
        actual_hashes = {
            "control_plane": _sha256(wheel),
            "executor": _sha256(executor_wheel),
            "node_runtime": _sha256(node_wheel),
            "bundle": _sha256(bundle),
        }
        if actual_hashes != expected_hashes:
            raise ReleaseError("release manifest hashes do not match its artifacts")
        if manifest_schema_version >= 2 and any(
            len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for value in component_digests.values()
        ):
            raise ReleaseError(
                "release component module digests must be lowercase SHA-256"
            )
        if manifest_schema_version >= 3:
            (
                delivery_identity,
                delivery_sha256,
                delivery_component_digests,
                locked_images,
                node_template_sha256,
            ) = parse_delivery_identity(
                manifest,
                components,
            )
    else:
        wheel = Path(release.get("wheel", ""))
        wheel, executor_wheel, node_wheel, bundle = _resolved_artifact_paths(
            (
                wheel,
                Path(release.get("executor_wheel") or wheel),
                Path(release.get("node_wheel") or wheel),
                Path(release.get("bundle", "")),
            ),
            base=config_path.parent,
        )
        _require_artifact_files((wheel, executor_wheel, node_wheel, bundle))
        release_id = str(release.get("id") or _sha256(wheel)[:12])
        database_schema_version = int(release.get("database_schema_version", 0))
        agent_protocol_version = int(release.get("agent_protocol_version", 3))
        executor_protocol_version = int(release.get("executor_protocol_version", 2))
        component_digests = {
            "control_plane": str(release.get("control_plane_digest") or ""),
            "executor": str(release.get("executor_digest") or ""),
            "node_runtime": str(release.get("node_runtime_digest") or ""),
        }
    if database_schema_version < 0:
        raise ReleaseError("release database_schema_version must not be negative")
    if agent_protocol_version < 1 or executor_protocol_version < 1:
        raise ReleaseError("release protocol versions must be positive")
    return ReleaseArtifacts(
        release_id=release_id,
        wheel=wheel.resolve(),
        executor_wheel=executor_wheel.resolve(),
        node_wheel=node_wheel.resolve(),
        bundle=bundle.resolve(),
        database_schema_version=database_schema_version,
        agent_protocol_version=agent_protocol_version,
        executor_protocol_version=executor_protocol_version,
        component_digests=component_digests,
        manifest=manifest,
        manifest_schema_version=manifest_schema_version,
        delivery_identity=delivery_identity,
        delivery_sha256=delivery_sha256,
        delivery_component_digests=delivery_component_digests,
        locked_images=locked_images,
        node_template_sha256=node_template_sha256,
    )


RETENTION_ENVIRONMENT = (
    "GPU_FAULT_CONTROL_RECORD_RETENTION_DAYS",
    "GPU_FAULT_CONTROL_RECORD_ARCHIVE_S3_URI",
    "GPU_FAULT_CONTROL_RECORD_ARCHIVE_INTERVAL_SECONDS",
)


@dataclass(frozen=True)
class RegionalRetentionConfig:
    """``retention`` from the release config: ``site.yaml`` ``spec.retention``.

    Off unless the days are positive; the runtime default is ``0`` and the
    worker then never archives or deletes a control record. Rendered into the
    control-worker environment only when on, so a site that never declared it
    keeps exactly the environment it had.
    """

    control_record_retention_days: int = 0
    archive_s3_uri: str | None = None
    archive_interval_seconds: int | None = None

    @property
    def enabled(self) -> bool:
        return self.control_record_retention_days > 0

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> RegionalRetentionConfig:
        if not value:
            return cls()
        days = value.get("control_record_retention_days", 0)
        if isinstance(days, bool) or not isinstance(days, int) or days < 0:
            raise ReleaseError(
                "retention.control_record_retention_days must be an integer >= 0"
            )
        uri = value.get("archive_s3_uri")
        uri = required_text(uri, "retention.archive_s3_uri") if uri else None
        if uri is not None and not uri.startswith("s3://"):
            raise ReleaseError("retention.archive_s3_uri must be an s3:// URI")
        if days > 0 and uri is None:
            raise ReleaseError(
                "retention.archive_s3_uri is required when "
                "control_record_retention_days > 0"
            )
        interval = value.get("archive_interval_seconds")
        if interval is not None and (
            isinstance(interval, bool) or not isinstance(interval, int) or interval < 1
        ):
            raise ReleaseError(
                "retention.archive_interval_seconds must be a positive integer"
            )
        return cls(
            control_record_retention_days=days,
            archive_s3_uri=uri,
            archive_interval_seconds=interval,
        )

    def environment(self) -> dict[str, str]:
        """The control-worker variables; empty while retention is off."""

        if not self.enabled or self.archive_s3_uri is None:
            return {}
        values = {
            RETENTION_ENVIRONMENT[0]: str(self.control_record_retention_days),
            RETENTION_ENVIRONMENT[1]: self.archive_s3_uri,
        }
        if self.archive_interval_seconds is not None:
            values[RETENTION_ENVIRONMENT[2]] = str(self.archive_interval_seconds)
        return values


@dataclass(frozen=True)
class ReleaseConfig:
    site_name: str
    release_id: str
    aws_region: str
    cpu_kubeconfig: str
    cpu_eks_arn: str
    cpu_hyperpod_cluster_name: str
    namespace: str
    wheel: Path
    executor_wheel: Path
    node_wheel: Path
    bundle: Path
    database_schema_version: int
    agent_protocol_version: int
    executor_protocol_version: int
    component_digests: dict[str, str]
    release_manifest_schema_version: int
    release_delivery_identity: dict[str, Any]
    release_delivery_sha256: str
    delivery_component_digests: dict[str, str]
    locked_images: dict[str, str]
    node_template_sha256: str
    upgrade_max_unavailable: int
    rollback_max_unavailable: int
    upgrade_max_parallel_clusters: int
    agent_config_digest: str
    runtime_profile_source: Path
    runtime_profile_template_source: Path
    runtime_profile_version: str
    runtime_profile_registration_cluster_id: str
    clusters: tuple[ClusterTarget, ...]
    nlb: dict[str, str]
    dns: RegionalDnsConfig
    health: RegionalHealthConfig
    notifications: RegionalNotificationConfig
    admin_config: AdminConfig
    auto_rollback: bool = True
    # Node label keys that name a failure domain, finest first. Read by the
    # failure-domain ConfigMap render and by the fleet rollout's per-domain cap.
    failure_domain_labels: tuple[str, ...] = FAILURE_DOMAIN_LABELS
    retention: RegionalRetentionConfig = RegionalRetentionConfig()

    def for_rollback(
        self,
        agent_config_digest: str,
        *,
        admin_config: AdminConfig | None = None,
    ) -> ReleaseConfig:
        return replace(
            self,
            agent_config_digest=agent_config_digest,
            admin_config=admin_config or self.admin_config,
            auto_rollback=False,
        )

    def notification_environment(self) -> dict[str, str]:
        """The control-plane variables naming the alert channel, on every role.

        Each role builds the notifier and runs the fail-closed "no alert
        channel" guard at startup, so the channel and (for ``sns``) the site
        topic must reach all three; a rollback rendered without them would
        leave a worker that cannot start.
        """

        values = {NOTIFICATION_ENVIRONMENT[0]: self.notifications.channel}
        if (
            self.notifications.channel == NOTIFICATION_CHANNEL_SNS
            and self.health.sns_topic_arn
        ):
            values[NOTIFICATION_ENVIRONMENT[1]] = self.health.sns_topic_arn
        return values

    @classmethod
    def load(cls, path: Path) -> ReleaseConfig:
        value = json.loads(path.read_text(encoding="utf-8"))
        aws_region = validate_aws_region(value.get("aws_region"))
        cpu_eks_arn = validate_eks_arn(
            value.get("cpu_eks_arn"),
            field="cpu_eks_arn",
            expected_region=aws_region,
        )
        cpu_hyperpod_cluster_name = required_text(
            value.get("cpu_hyperpod_cluster_name"),
            "cpu_hyperpod_cluster_name",
        )
        release = value.get("release") or {}
        try:
            upgrade_max_unavailable = int(release.get("upgrade_max_unavailable", 1))
            rollback_max_unavailable = int(release.get("rollback_max_unavailable", 2))
            upgrade_max_parallel_clusters = int(
                release.get("upgrade_max_parallel_clusters", 1)
            )
        except (TypeError, ValueError) as exc:
            raise ReleaseError(
                "release rollout max unavailable values must be integers"
            ) from exc
        # 0 means "auto": the rollout policy uses the built-in size cap it
        # already computes from the node count, instead of a number the site
        # had to guess. A configured value still caps the cap.
        if not 0 <= upgrade_max_unavailable <= 32:
            raise ReleaseError(
                "release.upgrade_max_unavailable must be within 0..32 (0 = auto)"
            )
        if not 1 <= rollback_max_unavailable <= 4:
            raise ReleaseError("release.rollback_max_unavailable must be within 1..4")
        # Cross-cluster parallelism multiplies the blast radius of one release, so
        # it stays opt-in: the default rolls one GPU cluster at a time. Rollback
        # is deliberately not covered by this knob and stays serial.
        if not 1 <= upgrade_max_parallel_clusters <= MAX_UPGRADE_PARALLEL_CLUSTERS:
            raise ReleaseError(
                "release.upgrade_max_parallel_clusters must be within "
                f"1..{MAX_UPGRADE_PARALLEL_CLUSTERS}"
            )
        clusters = tuple(
            ClusterTarget.from_mapping(item, expected_region=aws_region)
            for item in value.get("clusters", [])
        )
        runtime_profile = value.get("runtime_profile") or {}
        runtime_profile_source = Path(
            required_text(
                runtime_profile.get("source"),
                "runtime_profile.source",
            )
        )
        if not runtime_profile_source.is_absolute():
            runtime_profile_source = path.parent / runtime_profile_source
        runtime_profile_template_source = Path(
            str(runtime_profile.get("template_source") or runtime_profile_source)
        )
        if not runtime_profile_template_source.is_absolute():
            runtime_profile_template_source = (
                path.parent / runtime_profile_template_source
            )
        runtime_profile_version = validate_runtime_profile_version(
            runtime_profile.get("version"),
        )
        runtime_profile_registration_cluster_id = required_text(
            runtime_profile.get("registration_cluster_id"),
            "runtime_profile.registration_cluster_id",
        )
        artifacts = load_release_artifacts(release, config_path=path)
        digest = str(release.get("agent_config_digest", ""))
        cpu_kubeconfig = required_text(value.get("cpu_kubeconfig"), "cpu_kubeconfig")
        nlb = dict(value.get("nlb") or {})
        if nlb:
            for field in ("public_subnets", "security_group"):
                nlb[field] = required_text(nlb.get(field), f"nlb.{field}")
            nlb["certificate_arn"] = validate_regional_arn(
                nlb.get("certificate_arn"),
                field="nlb.certificate_arn",
                service="acm",
                expected_region=aws_region,
            )
            if nlb.get("name"):
                nlb["name"] = required_text(nlb["name"], "nlb.name")
        health = RegionalHealthConfig.from_mapping(
            dict(value.get("health") or {}),
            expected_region=aws_region,
        )
        notifications = RegionalNotificationConfig.from_mapping(
            dict(value.get("notifications") or {})
        )
        raw_admin_config = dict(value.get("admin_config") or {})
        admin_config = AdminConfig.from_mapping(raw_admin_config.get("config") or {})
        if raw_admin_config:
            if raw_admin_config.get("config_sha256") != admin_config.sha256():
                raise ReleaseError(
                    "admin_config.config_sha256 does not match its content"
                )
            if raw_admin_config.get("role_sha256") != admin_config.role_sha256():
                raise ReleaseError(
                    "admin_config.role_sha256 does not match its content"
                )
        dns = RegionalDnsConfig.from_mapping(dict(value.get("dns") or {}))
        if len(digest) != 64 or any(
            character not in "0123456789abcdef" for character in digest
        ):
            raise ReleaseError("release.agent_config_digest must be lowercase SHA-256")
        ids = [item.cluster_id for item in clusters]
        if len(ids) != len(set(ids)):
            raise ReleaseError("cluster_id values must be unique")
        if not runtime_profile_source.is_file():
            raise ReleaseError("runtime_profile.source must be an existing file")
        if not runtime_profile_template_source.is_file():
            raise ReleaseError(
                "runtime_profile.template_source must be an existing file"
            )
        try:
            runtime_profile_document = yaml.safe_load(
                runtime_profile_source.read_text(encoding="utf-8")
            )
        except (OSError, yaml.YAMLError) as exc:
            raise ReleaseError(f"cannot load runtime_profile.source: {exc}") from exc
        if not isinstance(runtime_profile_document, dict):
            raise ReleaseError(
                "runtime_profile.source must contain one RuntimeProfile mapping"
            )
        required_profile_fields = {"cluster_id", "environment", "claims", "observed"}
        missing_profile_fields = sorted(
            required_profile_fields - set(runtime_profile_document)
        )
        if missing_profile_fields:
            raise ReleaseError(
                "runtime_profile.source is missing: "
                + ", ".join(missing_profile_fields)
            )
        return cls(
            site_name=str(value.get("site_name") or path.stem),
            release_id=artifacts.release_id,
            aws_region=aws_region,
            cpu_kubeconfig=cpu_kubeconfig,
            cpu_eks_arn=cpu_eks_arn,
            cpu_hyperpod_cluster_name=cpu_hyperpod_cluster_name,
            namespace=str(value.get("namespace", DEFAULT_NAMESPACE)),
            wheel=artifacts.wheel,
            executor_wheel=artifacts.executor_wheel,
            node_wheel=artifacts.node_wheel,
            bundle=artifacts.bundle,
            database_schema_version=artifacts.database_schema_version,
            agent_protocol_version=artifacts.agent_protocol_version,
            executor_protocol_version=artifacts.executor_protocol_version,
            component_digests=artifacts.component_digests,
            release_manifest_schema_version=(artifacts.manifest_schema_version),
            release_delivery_identity=artifacts.delivery_identity,
            release_delivery_sha256=artifacts.delivery_sha256,
            delivery_component_digests=(artifacts.delivery_component_digests),
            locked_images=artifacts.locked_images,
            node_template_sha256=artifacts.node_template_sha256,
            upgrade_max_unavailable=upgrade_max_unavailable,
            rollback_max_unavailable=rollback_max_unavailable,
            upgrade_max_parallel_clusters=upgrade_max_parallel_clusters,
            agent_config_digest=digest,
            runtime_profile_source=runtime_profile_source.resolve(),
            runtime_profile_template_source=(runtime_profile_template_source.resolve()),
            runtime_profile_version=runtime_profile_version,
            runtime_profile_registration_cluster_id=(
                runtime_profile_registration_cluster_id
            ),
            clusters=clusters,
            nlb=nlb,
            dns=dns,
            health=health,
            notifications=notifications,
            admin_config=admin_config,
            auto_rollback=bool(value.get("auto_rollback", True)),
            failure_domain_labels=failure_domain_labels(
                value.get("failure_domain_labels")
            ),
            retention=RegionalRetentionConfig.from_mapping(
                dict(value.get("retention") or {})
            ),
        )


def failure_domain_labels(value: object) -> tuple[str, ...]:
    """``failure_domain_labels`` from the release config, or the built-in priority."""

    if value is None:
        return FAILURE_DOMAIN_LABELS
    if not isinstance(value, list) or not value:
        raise ReleaseError("failure_domain_labels must be a non-empty list")
    labels = tuple(required_text(item, "failure_domain_labels[]") for item in value)
    if len(set(labels)) != len(labels):
        raise ReleaseError("failure_domain_labels values must be unique")
    return labels


def render_nlb_manifest(config: ReleaseConfig, text: str) -> str:
    if not config.nlb:
        raise ReleaseError("nlb config is required")
    required = ("public_subnets", "security_group", "certificate_arn")
    missing = [name for name in required if not config.nlb.get(name)]
    if missing:
        raise ReleaseError("nlb config is missing: " + ", ".join(missing))
    replacements = {
        "REPLACE_WITH_NLB_NAME": (
            config.nlb.get("name") or f"gpu-fault-regional-{config.aws_region}"
        ),
        "REPLACE_WITH_PUBLIC_SUBNETS": config.nlb["public_subnets"],
        "REPLACE_WITH_NLB_SECURITY_GROUP": config.nlb["security_group"],
        "REPLACE_WITH_TLS_CERTIFICATE_ARN": config.nlb["certificate_arn"],
        "namespace: gpu-fault-system": f"namespace: {config.namespace}",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    if "REPLACE_WITH" in text:
        raise ReleaseError("NLB manifest still contains a placeholder")
    return text

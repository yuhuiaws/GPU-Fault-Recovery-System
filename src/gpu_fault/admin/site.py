from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence, cast

import yaml  # type: ignore[import-untyped]

from gpu_fault.admin.config import (
    admin_config_desired_path,
    load_desired_admin_config,
)
from gpu_fault.digests import SHA256_PATTERN

SITE_API_VERSION = "gpu-fault.aws/v1alpha1"
SITE_KIND = "RegionalSite"
AWS_REGION_PATTERN = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)+-[0-9]+$")
IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}$")
OCI_IMAGE_PATTERN = re.compile(r"^[^\s#]+$")
EMAIL_PATTERN = re.compile(r"^[^\s@,]+@[^\s@,]+\.[^\s@,]+$")


class SiteConfigError(ValueError):
    pass


def _mapping(
    value: object,
    path: str,
    *,
    allowed: set[str],
) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise SiteConfigError(f"{path} must be a mapping")
    normalized = cast(Mapping[str, object], value)
    unknown = sorted(set(normalized) - allowed)
    if unknown:
        raise SiteConfigError(f"{path} contains unknown fields: {', '.join(unknown)}")
    return normalized


def _required_text(value: object, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SiteConfigError(f"{path} must be a non-empty string")
    normalized = value.strip()
    if normalized.startswith("REPLACE_"):
        raise SiteConfigError(f"{path} is still a placeholder")
    return normalized


def _optional_text(value: object, path: str) -> str | None:
    if value is None:
        return None
    return _required_text(value, path)


def _optional_email(value: object, path: str) -> str | None:
    normalized = _optional_text(value, path)
    if normalized is not None and not EMAIL_PATTERN.fullmatch(normalized):
        raise SiteConfigError(f"{path} must be a valid email address")
    return normalized


def _email_list(value: object, path: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise SiteConfigError(f"{path} must be a list")
    normalized = tuple(
        dict.fromkeys(
            _optional_email(item, f"{path}[]") for item in cast(Sequence[object], value)
        )
    )
    if not normalized or any(item is None for item in normalized):
        raise SiteConfigError(f"{path} requires at least one valid email address")
    return cast(tuple[str, ...], normalized)


def _subject_prefix(value: object, path: str) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        raise SiteConfigError(f"{path} must be a string")
    normalized = value.strip()
    if len(normalized) > 64 or "\n" in normalized or "\r" in normalized:
        raise SiteConfigError(f"{path} must be a single line of at most 64 characters")
    return normalized


def _boolean(value: object, path: str, *, default: bool) -> bool:
    if value is None:
        return default
    if not isinstance(value, bool):
        raise SiteConfigError(f"{path} must be a boolean")
    return value


def _integer(
    value: object,
    path: str,
    *,
    default: int,
    minimum: int,
    maximum: int,
) -> int:
    if value is None:
        return default
    if not isinstance(value, int) or isinstance(value, bool):
        raise SiteConfigError(f"{path} must be an integer")
    if not minimum <= value <= maximum:
        raise SiteConfigError(f"{path} must be within {minimum}..{maximum}")
    return value


def _text_list(
    value: object,
    path: str,
    *,
    minimum: int,
) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise SiteConfigError(f"{path} must be a list")
    normalized = tuple(
        sorted(
            {
                _required_text(item, f"{path}[]")
                for item in cast(Sequence[object], value)
            }
        )
    )
    if len(normalized) < minimum:
        raise SiteConfigError(f"{path} requires at least {minimum} value(s)")
    return normalized


@dataclass(frozen=True)
class CpuSiteConfig:
    kubeconfig: str
    eks_arn: str
    hyperpod_cluster_name: str

    @classmethod
    def from_value(cls, value: object) -> CpuSiteConfig:
        data = _mapping(
            value,
            "spec.cpu",
            allowed={"kubeconfig", "eksArn", "hyperpodClusterName"},
        )
        return cls(
            kubeconfig=_required_text(data.get("kubeconfig"), "spec.cpu.kubeconfig"),
            eks_arn=_required_text(data.get("eksArn"), "spec.cpu.eksArn"),
            hyperpod_cluster_name=_required_text(
                data.get("hyperpodClusterName"),
                "spec.cpu.hyperpodClusterName",
            ),
        )


@dataclass(frozen=True)
class ReleaseSiteConfig:
    manifest: str
    agent_config_digest: str
    upgrade_max_unavailable: int
    rollback_max_unavailable: int
    upgrade_max_parallel_clusters: int

    @classmethod
    def from_value(cls, value: object) -> ReleaseSiteConfig:
        data = _mapping(
            value,
            "spec.release",
            allowed={
                "manifest",
                "agentConfigDigest",
                "upgradeMaxUnavailable",
                "rollbackMaxUnavailable",
                "upgradeMaxParallelClusters",
            },
        )
        digest = _required_text(
            data.get("agentConfigDigest"),
            "spec.release.agentConfigDigest",
        ).lower()
        if not SHA256_PATTERN.fullmatch(digest):
            raise SiteConfigError("spec.release.agentConfigDigest must be a SHA-256")
        return cls(
            manifest=_required_text(
                data.get("manifest"),
                "spec.release.manifest",
            ),
            agent_config_digest=digest,
            # 0 is the default and means "auto": the release computes the wave
            # size from the node count (4/8/16/32) and keeps the first wave a
            # one-node canary. A hard-coded 1 defeated that cap and turned a
            # four-node cluster into four full wave round trips.
            upgrade_max_unavailable=_integer(
                data.get("upgradeMaxUnavailable"),
                "spec.release.upgradeMaxUnavailable",
                default=0,
                minimum=0,
                maximum=32,
            ),
            rollback_max_unavailable=_integer(
                data.get("rollbackMaxUnavailable"),
                "spec.release.rollbackMaxUnavailable",
                default=2,
                minimum=1,
                maximum=4,
            ),
            # Rolling several GPU clusters at once multiplies the blast radius of
            # one release, so the default keeps a shared release at one cluster at
            # a time and an operator has to opt in per site.
            upgrade_max_parallel_clusters=_integer(
                data.get("upgradeMaxParallelClusters"),
                "spec.release.upgradeMaxParallelClusters",
                default=1,
                minimum=1,
                maximum=8,
            ),
        )


@dataclass(frozen=True)
class RuntimeProfileSiteConfig:
    source: str
    template_source: str
    version: str
    registration_cluster_id: str

    @classmethod
    def from_value(cls, value: object) -> RuntimeProfileSiteConfig:
        data = _mapping(
            value,
            "spec.runtimeProfile",
            allowed={
                "source",
                "templateSource",
                "version",
                "registrationClusterId",
            },
        )
        source = _required_text(
            data.get("source"),
            "spec.runtimeProfile.source",
        )
        version = _required_text(
            data.get("version"),
            "spec.runtimeProfile.version",
        )
        if not IDENTIFIER_PATTERN.fullmatch(version):
            raise SiteConfigError(
                "spec.runtimeProfile.version contains unsupported characters"
            )
        return cls(
            source=source,
            template_source=_required_text(
                data.get("templateSource", source),
                "spec.runtimeProfile.templateSource",
            ),
            version=version,
            registration_cluster_id=_required_text(
                data.get("registrationClusterId"),
                "spec.runtimeProfile.registrationClusterId",
            ),
        )


@dataclass(frozen=True)
class NlbSiteConfig:
    name: str
    public_subnets: tuple[str, ...]
    security_group: str
    certificate_arn: str

    @classmethod
    def from_value(cls, value: object) -> NlbSiteConfig:
        data = _mapping(
            value,
            "spec.nlb",
            allowed={
                "name",
                "publicSubnets",
                "securityGroup",
                "certificateArn",
            },
        )
        return cls(
            name=_required_text(data.get("name"), "spec.nlb.name"),
            public_subnets=_text_list(
                data.get("publicSubnets"),
                "spec.nlb.publicSubnets",
                minimum=2,
            ),
            security_group=_required_text(
                data.get("securityGroup"),
                "spec.nlb.securityGroup",
            ),
            certificate_arn=_required_text(
                data.get("certificateArn"),
                "spec.nlb.certificateArn",
            ),
        )


@dataclass(frozen=True)
class DnsSiteConfig:
    hosted_zone_id: str
    hostname: str

    @classmethod
    def from_value(cls, value: object) -> DnsSiteConfig | None:
        if value is None:
            return None
        data = _mapping(
            value,
            "spec.dns",
            allowed={"hostedZoneId", "hostname"},
        )
        hostname = _required_text(
            data.get("hostname"),
            "spec.dns.hostname",
        ).rstrip(".")
        if "." not in hostname:
            raise SiteConfigError("spec.dns.hostname must be a DNS name")
        return cls(
            hosted_zone_id=_required_text(
                data.get("hostedZoneId"),
                "spec.dns.hostedZoneId",
            ),
            hostname=hostname,
        )


@dataclass(frozen=True)
class ImageSiteConfig:
    runtime: str | None = None
    node_installer: str | None = None
    dcgm_exporter: str | None = None
    adot: str | None = None

    @classmethod
    def from_value(cls, value: object) -> ImageSiteConfig:
        data = _mapping(
            value or {},
            "spec.images",
            allowed={"runtime", "nodeInstaller", "dcgmExporter", "adot"},
        )
        items = {
            "runtime": _optional_text(data.get("runtime"), "spec.images.runtime"),
            "node_installer": _optional_text(
                data.get("nodeInstaller"),
                "spec.images.nodeInstaller",
            ),
            "dcgm_exporter": _optional_text(
                data.get("dcgmExporter"),
                "spec.images.dcgmExporter",
            ),
            "adot": _optional_text(data.get("adot"), "spec.images.adot"),
        }
        for name, image in items.items():
            if image is not None and not OCI_IMAGE_PATTERN.fullmatch(image):
                raise SiteConfigError(f"spec.images.{name} contains whitespace or #")
        return cls(**items)


@dataclass(frozen=True)
class HealthSiteConfig:
    aurora_cluster_id: str
    amp_workspace_id: str
    amp_rule_namespace: str
    sns_topic_arn: str
    certificate_min_validity_days: int
    remote_command_max_unclaimed_seconds: int
    require_confirmed_sns_subscription: bool

    @classmethod
    def from_value(cls, value: object) -> HealthSiteConfig:
        data = _mapping(
            value,
            "spec.health",
            allowed={
                "auroraClusterId",
                "ampWorkspaceId",
                "ampRuleNamespace",
                "snsTopicArn",
                "certificateMinValidityDays",
                "remoteCommandMaxUnclaimedSeconds",
                "requireConfirmedSnsSubscription",
            },
        )
        return cls(
            aurora_cluster_id=_required_text(
                data.get("auroraClusterId"),
                "spec.health.auroraClusterId",
            ),
            amp_workspace_id=_required_text(
                data.get("ampWorkspaceId"),
                "spec.health.ampWorkspaceId",
            ),
            amp_rule_namespace=_required_text(
                data.get(
                    "ampRuleNamespace",
                    "gpu-fault-control-plane-capacity",
                ),
                "spec.health.ampRuleNamespace",
            ),
            sns_topic_arn=_required_text(
                data.get("snsTopicArn"),
                "spec.health.snsTopicArn",
            ),
            certificate_min_validity_days=_integer(
                data.get("certificateMinValidityDays"),
                "spec.health.certificateMinValidityDays",
                default=30,
                minimum=1,
                maximum=3650,
            ),
            remote_command_max_unclaimed_seconds=_integer(
                data.get("remoteCommandMaxUnclaimedSeconds"),
                "spec.health.remoteCommandMaxUnclaimedSeconds",
                default=300,
                minimum=1,
                maximum=86400,
            ),
            require_confirmed_sns_subscription=_boolean(
                data.get("requireConfirmedSnsSubscription"),
                "spec.health.requireConfirmedSnsSubscription",
                default=True,
            ),
        )


@dataclass(frozen=True)
class NotificationSiteConfig:
    allow_email: bool = False
    acknowledge_external_alert_channel: bool = True
    admin_email: str | None = None
    email_sender: str | None = None
    email_recipients: tuple[str, ...] = ()
    email_subject_prefix: str = ""

    @classmethod
    def from_value(cls, value: object) -> NotificationSiteConfig:
        data = _mapping(
            value or {},
            "spec.notifications",
            allowed={
                "allowEmail",
                "acknowledgeExternalAlertChannel",
                "adminEmail",
                "emailSender",
                "emailRecipients",
                "emailSubjectPrefix",
            },
        )
        allow_email = _boolean(
            data.get("allowEmail"),
            "spec.notifications.allowEmail",
            default=False,
        )
        acknowledge = _boolean(
            data.get("acknowledgeExternalAlertChannel"),
            "spec.notifications.acknowledgeExternalAlertChannel",
            default=True,
        )
        admin_email = _optional_email(
            data.get("adminEmail"),
            "spec.notifications.adminEmail",
        )
        email_sender = _optional_email(
            data.get("emailSender"),
            "spec.notifications.emailSender",
        )
        email_recipients = (
            _email_list(
                data.get("emailRecipients"),
                "spec.notifications.emailRecipients",
            )
            if data.get("emailRecipients") is not None
            else ((admin_email,) if admin_email is not None else ())
        )
        email_subject_prefix = _subject_prefix(
            data.get("emailSubjectPrefix"),
            "spec.notifications.emailSubjectPrefix",
        )
        if not allow_email and not acknowledge:
            raise SiteConfigError(
                "notifications must enable email or acknowledge an external alert channel"
            )
        if allow_email and (
            admin_email is None or email_sender is None or not email_recipients
        ):
            raise SiteConfigError(
                "email notifications require notifications.adminEmail "
                "notifications.emailSender, and at least one recipient"
            )
        return cls(
            allow_email=allow_email,
            acknowledge_external_alert_channel=acknowledge,
            admin_email=admin_email,
            email_sender=email_sender,
            email_recipients=email_recipients,
            email_subject_prefix=email_subject_prefix,
        )


@dataclass(frozen=True)
class GpuClusterSiteConfig:
    cluster_id: str
    context: str
    region: str | None
    hyperpod_cluster_name: str
    eks_cluster_arn: str
    executor_irsa_role_arn: str
    allowed_namespaces: tuple[str, ...]
    agent_endpoint_allowed_cidrs: tuple[str, ...]
    control_plane_url: str
    token_file: str
    ca_file: str
    fleet_master_file: str

    @classmethod
    def from_value(cls, value: object, index: int) -> GpuClusterSiteConfig:
        path = f"spec.clusters[{index}]"
        data = _mapping(
            value,
            path,
            allowed={
                "clusterId",
                "context",
                "region",
                "hyperpodClusterName",
                "eksClusterArn",
                "executorIrsaRoleArn",
                "allowedNamespaces",
                "agentEndpointAllowedCidrs",
                "controlPlaneUrl",
                "tokenFile",
                "caFile",
                "fleetMasterFile",
            },
        )
        cluster_id = _required_text(data.get("clusterId"), f"{path}.clusterId")
        if not IDENTIFIER_PATTERN.fullmatch(cluster_id):
            raise SiteConfigError(f"{path}.clusterId contains unsupported characters")
        url = _required_text(
            data.get("controlPlaneUrl"),
            f"{path}.controlPlaneUrl",
        ).rstrip("/")
        if not url.startswith("https://"):
            raise SiteConfigError(f"{path}.controlPlaneUrl must use https://")
        return cls(
            cluster_id=cluster_id,
            context=_required_text(data.get("context"), f"{path}.context"),
            region=_optional_text(data.get("region"), f"{path}.region"),
            hyperpod_cluster_name=_required_text(
                data.get("hyperpodClusterName"),
                f"{path}.hyperpodClusterName",
            ),
            eks_cluster_arn=_required_text(
                data.get("eksClusterArn"),
                f"{path}.eksClusterArn",
            ),
            executor_irsa_role_arn=_required_text(
                data.get("executorIrsaRoleArn"),
                f"{path}.executorIrsaRoleArn",
            ),
            allowed_namespaces=_text_list(
                data.get("allowedNamespaces"),
                f"{path}.allowedNamespaces",
                minimum=1,
            ),
            agent_endpoint_allowed_cidrs=_text_list(
                data.get("agentEndpointAllowedCidrs") or [],
                f"{path}.agentEndpointAllowedCidrs",
                minimum=0,
            ),
            control_plane_url=url,
            token_file=_required_text(
                data.get("tokenFile"),
                f"{path}.tokenFile",
            ),
            ca_file=_required_text(data.get("caFile"), f"{path}.caFile"),
            fleet_master_file=_required_text(
                data.get("fleetMasterFile"),
                f"{path}.fleetMasterFile",
            ),
        )


@dataclass(frozen=True)
class SiteSpec:
    repository_root: str
    gpu_kubeconfig: str | None
    aws_region: str
    namespace: str
    auto_rollback: bool
    cpu: CpuSiteConfig
    release: ReleaseSiteConfig
    runtime_profile: RuntimeProfileSiteConfig
    nlb: NlbSiteConfig
    dns: DnsSiteConfig | None
    images: ImageSiteConfig
    health: HealthSiteConfig
    notifications: NotificationSiteConfig
    clusters: tuple[GpuClusterSiteConfig, ...]

    @classmethod
    def from_value(cls, value: object) -> SiteSpec:
        data = _mapping(
            value,
            "spec",
            allowed={
                "repositoryRoot",
                "gpuKubeconfig",
                "awsRegion",
                "namespace",
                "autoRollback",
                "cpu",
                "release",
                "runtimeProfile",
                "nlb",
                "dns",
                "images",
                "health",
                "notifications",
                "clusters",
            },
        )
        region = _required_text(data.get("awsRegion"), "spec.awsRegion")
        if not AWS_REGION_PATTERN.fullmatch(region):
            raise SiteConfigError("spec.awsRegion is not a valid AWS Region")
        raw_clusters = data.get("clusters")
        if not isinstance(raw_clusters, Sequence) or isinstance(
            raw_clusters,
            (str, bytes),
        ):
            raise SiteConfigError("spec.clusters must be a list")
        clusters = tuple(
            GpuClusterSiteConfig.from_value(item, index)
            for index, item in enumerate(cast(Sequence[object], raw_clusters))
        )
        ids = [item.cluster_id for item in clusters]
        if len(ids) != len(set(ids)):
            raise SiteConfigError("spec.clusters clusterId values must be unique")
        runtime_profile = RuntimeProfileSiteConfig.from_value(
            data.get("runtimeProfile")
        )
        for cluster in clusters:
            if cluster.region is not None and cluster.region != region:
                raise SiteConfigError(
                    f"cluster {cluster.cluster_id} Region does not match awsRegion"
                )
        nlb = NlbSiteConfig.from_value(data.get("nlb"))
        health = HealthSiteConfig.from_value(data.get("health"))
        if f":{region}:" not in nlb.certificate_arn:
            raise SiteConfigError(
                "spec.nlb.certificateArn Region does not match awsRegion"
            )
        if f":{region}:" not in health.sns_topic_arn:
            raise SiteConfigError(
                "spec.health.snsTopicArn Region does not match awsRegion"
            )
        return cls(
            repository_root=_required_text(
                data.get("repositoryRoot"),
                "spec.repositoryRoot",
            ),
            gpu_kubeconfig=_optional_text(
                data.get("gpuKubeconfig"),
                "spec.gpuKubeconfig",
            ),
            aws_region=region,
            namespace=_required_text(
                data.get("namespace", "gpu-fault-system"),
                "spec.namespace",
            ),
            auto_rollback=_boolean(
                data.get("autoRollback"),
                "spec.autoRollback",
                default=True,
            ),
            cpu=CpuSiteConfig.from_value(data.get("cpu")),
            release=ReleaseSiteConfig.from_value(data.get("release")),
            runtime_profile=runtime_profile,
            nlb=nlb,
            dns=DnsSiteConfig.from_value(data.get("dns")),
            images=ImageSiteConfig.from_value(data.get("images")),
            health=health,
            notifications=NotificationSiteConfig.from_value(data.get("notifications")),
            clusters=clusters,
        )


@dataclass(frozen=True)
class RegionalSite:
    name: str
    spec: SiteSpec

    @classmethod
    def from_value(cls, value: object) -> RegionalSite:
        data = _mapping(
            value,
            "site",
            allowed={"apiVersion", "kind", "metadata", "spec"},
        )
        if data.get("apiVersion") != SITE_API_VERSION:
            raise SiteConfigError(f"apiVersion must be {SITE_API_VERSION}")
        if data.get("kind") != SITE_KIND:
            raise SiteConfigError(f"kind must be {SITE_KIND}")
        metadata = _mapping(
            data.get("metadata"),
            "metadata",
            allowed={"name"},
        )
        name = _required_text(metadata.get("name"), "metadata.name")
        if not IDENTIFIER_PATTERN.fullmatch(name):
            raise SiteConfigError("metadata.name contains unsupported characters")
        return cls(name=name, spec=SiteSpec.from_value(data.get("spec")))


@dataclass(frozen=True)
class RenderedSite:
    source: Path
    repository_root: Path
    release_config: dict[str, Any]
    environment: dict[str, str]
    source_sha256: str

    @property
    def audit_summary(self) -> dict[str, Any]:
        return {
            "site": self.metadata_name,
            "site_file": str(self.source),
            "site_sha256": self.source_sha256,
            "repository_root": str(self.repository_root),
            "aws_region": self.release_config["aws_region"],
            "namespace": self.release_config["namespace"],
            "cluster_ids": [
                item["cluster_id"] for item in self.release_config["clusters"]
            ],
            "runtime_profile_version": self.release_config["runtime_profile"][
                "version"
            ],
            "admin_config_sha256": self.release_config["admin_config"]["config_sha256"],
            "admin_config_role_sha256": self.release_config["admin_config"][
                "role_sha256"
            ],
            "environment_keys": sorted(self.environment),
        }

    @property
    def metadata_name(self) -> str:
        return str(self.release_config["site_name"])


def _resolve(root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (root / path)


def load_site(path: Path, *, repository_root: Path | None = None) -> RenderedSite:
    source = path.expanduser().resolve()
    try:
        raw = source.read_bytes()
        if source.stat().st_mode & 0o077:
            raise SiteConfigError(
                f"site config must not grant group/other permissions: {source}"
            )
        document = yaml.safe_load(raw)
        site = RegionalSite.from_value(document)
    except SiteConfigError:
        raise
    except (OSError, yaml.YAMLError, ValueError) as exc:
        raise SiteConfigError(f"invalid site config {source}: {exc}") from exc

    configured_root = _resolve(
        source.parent,
        site.spec.repository_root,
    ).resolve()
    root = (repository_root or configured_root).expanduser().resolve()
    rollout = root / "deploy/control-plane/regional/rollout-regional-release.sh"
    if not rollout.is_file():
        raise SiteConfigError(
            f"repository root does not contain the regional rollout entrypoint: {root}"
        )

    cpu_kubeconfig = _resolve(root, site.spec.cpu.kubeconfig).resolve()
    release_manifest = _resolve(root, site.spec.release.manifest).resolve()
    profile_source = _resolve(root, site.spec.runtime_profile.source).resolve()
    profile_template_source = _resolve(
        root,
        site.spec.runtime_profile.template_source,
    ).resolve()
    clusters = []
    for cluster in site.spec.clusters:
        clusters.append(
            {
                "cluster_id": cluster.cluster_id,
                "context": cluster.context,
                "region": cluster.region or site.spec.aws_region,
                "hyperpod_cluster_name": cluster.hyperpod_cluster_name,
                "eks_cluster_arn": cluster.eks_cluster_arn,
                "executor_irsa_role_arn": cluster.executor_irsa_role_arn,
                "allowed_namespaces": list(cluster.allowed_namespaces),
                "agent_endpoint_allowed_cidrs": list(
                    cluster.agent_endpoint_allowed_cidrs
                ),
                "control_plane_url": cluster.control_plane_url,
                "token_file": str(_resolve(root, cluster.token_file).resolve()),
                "ca_file": str(_resolve(root, cluster.ca_file).resolve()),
                "fleet_master_file": str(
                    _resolve(root, cluster.fleet_master_file).resolve()
                ),
            }
        )
    health = site.spec.health
    admin_state_dir = source.parent
    for candidate in source.parents:
        if admin_config_desired_path(candidate).is_file():
            admin_state_dir = candidate
            break
    admin_config = load_desired_admin_config(admin_state_dir)
    release_config: dict[str, Any] = {
        "site_name": site.name,
        "aws_region": site.spec.aws_region,
        "cpu_kubeconfig": str(cpu_kubeconfig),
        "cpu_eks_arn": site.spec.cpu.eks_arn,
        "cpu_hyperpod_cluster_name": site.spec.cpu.hyperpod_cluster_name,
        "namespace": site.spec.namespace,
        "auto_rollback": site.spec.auto_rollback,
        "runtime_profile": {
            "source": str(profile_source),
            "template_source": str(profile_template_source),
            "version": site.spec.runtime_profile.version,
            "registration_cluster_id": (
                site.spec.runtime_profile.registration_cluster_id
            ),
        },
        "release": {
            "manifest": str(release_manifest),
            "agent_config_digest": site.spec.release.agent_config_digest,
            "upgrade_max_unavailable": (site.spec.release.upgrade_max_unavailable),
            "rollback_max_unavailable": (site.spec.release.rollback_max_unavailable),
            "upgrade_max_parallel_clusters": (
                site.spec.release.upgrade_max_parallel_clusters
            ),
        },
        "nlb": {
            "name": site.spec.nlb.name,
            "public_subnets": ",".join(site.spec.nlb.public_subnets),
            "security_group": site.spec.nlb.security_group,
            "certificate_arn": site.spec.nlb.certificate_arn,
        },
        "dns": (
            {
                "hosted_zone_id": site.spec.dns.hosted_zone_id,
                "hostname": site.spec.dns.hostname,
            }
            if site.spec.dns is not None
            else {}
        ),
        "health": {
            "aurora_cluster_id": health.aurora_cluster_id,
            "amp_workspace_id": health.amp_workspace_id,
            "amp_rule_namespace": health.amp_rule_namespace,
            "sns_topic_arn": health.sns_topic_arn,
            "certificate_min_validity_days": (health.certificate_min_validity_days),
            "remote_command_max_unclaimed_seconds": (
                health.remote_command_max_unclaimed_seconds
            ),
            "require_confirmed_sns_subscription": (
                health.require_confirmed_sns_subscription
            ),
        },
        "notifications": {
            "allow_email": site.spec.notifications.allow_email,
            "acknowledge_external_alert_channel": (
                site.spec.notifications.acknowledge_external_alert_channel
            ),
            "admin_email": site.spec.notifications.admin_email,
            "email_sender": site.spec.notifications.email_sender,
            "email_recipients": list(site.spec.notifications.email_recipients),
            "email_subject_prefix": (site.spec.notifications.email_subject_prefix),
        },
        "admin_config": {
            "config": admin_config.as_dict(),
            "config_sha256": admin_config.sha256(),
            "role_sha256": admin_config.role_sha256(),
        },
        "clusters": clusters,
    }
    image_values = {
        "GPU_FAULT_RUNTIME_IMAGE": site.spec.images.runtime,
        "GPU_FAULT_NODE_INSTALLER_IMAGE": site.spec.images.node_installer,
        "GPU_FAULT_DCGM_EXPORTER_IMAGE": site.spec.images.dcgm_exporter,
        "GPU_FAULT_ADOT_IMAGE": site.spec.images.adot,
    }
    environment = {
        name: value for name, value in image_values.items() if value is not None
    }
    if site.spec.gpu_kubeconfig:
        environment["KUBECONFIG"] = str(
            _resolve(root, site.spec.gpu_kubeconfig).resolve()
        )
    return RenderedSite(
        source=source,
        repository_root=root,
        release_config=release_config,
        environment=environment,
        source_sha256=hashlib.sha256(raw).hexdigest(),
    )


@contextmanager
def materialized_release_config(site: RenderedSite) -> Iterator[Path]:
    with tempfile.TemporaryDirectory(prefix="gpu-fault-admin-") as directory:
        root = Path(directory)
        root.chmod(0o700)
        path = root / "regional-release.json"
        path.write_text(
            json.dumps(site.release_config, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        path.chmod(0o600)
        yield path


def effective_environment(site: RenderedSite) -> dict[str, str]:
    return {
        **os.environ,
        **site.environment,
        "PYTHONPATH": str(site.repository_root / "src"),
        "GPU_FAULT_REPO_ROOT": str(site.repository_root),
    }

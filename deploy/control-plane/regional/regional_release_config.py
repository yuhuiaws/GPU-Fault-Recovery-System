from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_NAMESPACE = "gpu-fault-system"
AWS_REGION_PATTERN = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)+-[0-9]+$")
EKS_ARN_PATTERN = re.compile(
    r"^arn:[^:]+:eks:(?P<region>[^:]+):(?P<account>[^:]+):"
    r"cluster/(?P<name>[^/]+)$"
)


class ReleaseError(RuntimeError):
    pass


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


def validate_aws_region(value: object, field: str = "aws_region") -> str:
    region = required_text(value, field)
    if not AWS_REGION_PATTERN.fullmatch(region):
        raise ReleaseError(f"{field} is not a valid AWS Region: {region}")
    return region


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
    fleet_master_file: str | None = None

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
            fleet_master_file=value.get("fleet_master_file"),
        )


@dataclass(frozen=True)
class ReleaseConfig:
    aws_region: str
    cpu_kubeconfig: str
    cpu_eks_arn: str
    cpu_hyperpod_cluster_name: str
    namespace: str
    wheel: Path
    bundle: Path
    agent_config_digest: str
    clusters: tuple[ClusterTarget, ...]
    nlb: dict[str, str]
    auto_rollback: bool = True

    def for_rollback(self, agent_config_digest: str) -> ReleaseConfig:
        return replace(
            self,
            agent_config_digest=agent_config_digest,
            auto_rollback=False,
        )

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
        clusters = tuple(
            ClusterTarget.from_mapping(item, expected_region=aws_region)
            for item in value.get("clusters", [])
        )
        if not clusters:
            raise ReleaseError("config requires at least one GPU cluster")
        release_manifest = release.get("manifest")
        manifest: dict[str, Any] | None = None
        if release_manifest:
            manifest_path = Path(str(release_manifest))
            if not manifest_path.is_absolute():
                manifest_path = path.parent / manifest_path
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            wheel = Path(str(manifest["wheel"]))
            bundle = Path(str(manifest["bundle"]))
            if not wheel.is_absolute():
                wheel = ROOT / wheel
            if not bundle.is_absolute():
                bundle = ROOT / bundle
        else:
            wheel = Path(release.get("wheel", ""))
            bundle = Path(release.get("bundle", ""))
            if not wheel.is_absolute():
                wheel = path.parent / wheel
            if not bundle.is_absolute():
                bundle = path.parent / bundle
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
        if not wheel.is_file() or not bundle.is_file():
            raise ReleaseError("release wheel and bundle must exist")
        if manifest is not None:
            if _sha256(wheel) != manifest.get("wheel_sha256") or _sha256(
                bundle
            ) != manifest.get("bundle_sha256"):
                raise ReleaseError("release manifest hashes do not match its artifacts")
        if len(digest) != 64 or any(
            character not in "0123456789abcdef" for character in digest
        ):
            raise ReleaseError("release.agent_config_digest must be lowercase SHA-256")
        ids = [item.cluster_id for item in clusters]
        if len(ids) != len(set(ids)):
            raise ReleaseError("cluster_id values must be unique")
        return cls(
            aws_region=aws_region,
            cpu_kubeconfig=cpu_kubeconfig,
            cpu_eks_arn=cpu_eks_arn,
            cpu_hyperpod_cluster_name=cpu_hyperpod_cluster_name,
            namespace=str(value.get("namespace", DEFAULT_NAMESPACE)),
            wheel=wheel.resolve(),
            bundle=bundle.resolve(),
            agent_config_digest=digest,
            clusters=clusters,
            nlb=nlb,
            auto_rollback=bool(value.get("auto_rollback", True)),
        )


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

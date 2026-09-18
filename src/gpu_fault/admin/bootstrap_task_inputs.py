"""Task input declarations shared by scheduling and checkpoint invalidation."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field


@dataclass(frozen=True)
class TaskInputContext:
    common: Mapping[str, object]
    region: str
    account_id: str
    cpu: Mapping[str, object]
    gpu_clusters: Sequence[Mapping[str, object]]
    cluster_inputs: Mapping[str, Mapping[str, object]]
    release: Mapping[str, object]
    images: Mapping[str, object]
    assets: Mapping[str, str | None]
    sources: Mapping[str, str | None]
    notifications: Mapping[str, object]
    admin_config_sha256: object
    control_plane_wheel_sha256: str | None
    dashboards: Mapping[str, str]
    grafana: Mapping[str, object]
    release_ready: bool
    rds_ca_bundle_path: str
    custody: Mapping[str, object] = field(default_factory=dict)


InputFunction = Callable[[TaskInputContext, Mapping[str, object]], object]


@dataclass(frozen=True)
class TaskInputSpec:
    name: str
    fingerprint: InputFunction | None = None
    always_revalidate_reason: str = ""
    cluster_scoped: bool = False
    requires_release: bool = False
    deadline_seconds: int = 900
    degraded_result_key: str = ""

    def __post_init__(self) -> None:
        if bool(self.fingerprint) == bool(self.always_revalidate_reason.strip()):
            raise ValueError(
                "a task needs one input identity or an explicit revalidation reason"
            )
        if self.deadline_seconds <= 0:
            raise ValueError("task deadline must be positive")

    def digest(
        self, name: str, context: TaskInputContext, cluster: Mapping[str, object]
    ) -> str | None:
        if self.fingerprint is None or (
            self.requires_release and not context.release_ready
        ):
            return None
        return hashlib.sha256(
            json.dumps(
                {
                    "common": context.common,
                    "name": name,
                    "task": self.fingerprint(context, cluster),
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()

    def result_status(self, result: object) -> str:
        if not self.degraded_result_key:
            return "succeeded"
        value = (
            result.get(self.degraded_result_key)
            if isinstance(result, Mapping)
            else None
        )
        if not isinstance(value, Mapping) or value.get("status") not in {
            "PROVISIONED",
            "DEGRADED",
            "FAILED",
            "SKIPPED",
        }:
            raise ValueError("presentation task returned an invalid result")
        if value["status"] == "FAILED":
            raise ValueError("presentation task reported FAILED")
        return "degraded" if value["status"] == "DEGRADED" else "succeeded"


def _network_inputs(
    context: TaskInputContext, _cluster: Mapping[str, object]
) -> object:
    return {
        "cpu": context.cpu,
        "gpu_clusters": context.gpu_clusters,
        "source": context.sources["src/gpu_fault/admin/bootstrap.py"],
    }


def _monitoring_inputs(
    context: TaskInputContext, _cluster: Mapping[str, object]
) -> object:
    return {
        "source": context.sources["src/gpu_fault/admin/bootstrap_services.py"],
        "cpu": context.cpu,
    }


def _pki_inputs(context: TaskInputContext, cluster: Mapping[str, object]) -> object:
    return {
        "network": _network_inputs(context, cluster),
        "dns_source": context.sources["src/gpu_fault/admin/resource_registry_dns.py"],
    }


def _node_key_inputs(
    context: TaskInputContext, cluster: Mapping[str, object]
) -> object:
    result: dict[str, object] = {
        "asset": context.assets["deploy/node/provision-node-action-keys.sh"],
        "helper": context.assets["deploy/node/provision_node_action_keys.py"],
        "proof": context.sources["src/gpu_fault/admin/node_key_proof.py"],
        "cluster": cluster,
    }
    if context.custody:
        result["custody"] = context.custody
        result["release"] = context.release
        result["custody_sources"] = {
            name: digest
            for name, digest in context.sources.items()
            if "node_key_custody" in name
            or name.endswith(
                (
                    "bootstrap_tasks.py",
                    "bootstrap_task_inputs.py",
                    "bootstrap_services.py",
                    "bootstrap_checkpoint.py",
                    "release_artifacts.py",
                )
            )
        }
    return result


TASK_INPUT_SPECS = {
    item.name: item
    for item in (
        TaskInputSpec(
            "cpu_access",
            always_revalidate_reason="CPU kubeconfig and namespace are rechecked each invocation",
        ),
        TaskInputSpec(
            "gpu_access",
            always_revalidate_reason="GPU kubeconfig and namespace are rechecked each invocation",
        ),
        TaskInputSpec(
            "release_repositories",
            lambda c, _g: {"region": c.region, "account_id": c.account_id},
        ),
        TaskInputSpec(
            "release",
            lambda c, _g: {"release": c.release},
            requires_release=True,
            deadline_seconds=7200,
        ),
        TaskInputSpec(
            "pod_identity_agent", lambda c, _g: {"eks_arn": c.cpu["eks_arn"]}
        ),
        TaskInputSpec("nlb_network", _network_inputs),
        TaskInputSpec("pki", _pki_inputs),
        TaskInputSpec(
            "aurora",
            lambda c, _g: {"admin_config_sha256": c.admin_config_sha256},
            deadline_seconds=3600,
        ),
        TaskInputSpec(
            "aurora_ready",
            lambda c, _g: {
                "eks_arn": c.cpu["eks_arn"],
                "ca_bundle_path": c.rds_ca_bundle_path,
            },
            deadline_seconds=3600,
        ),
        TaskInputSpec(
            "load_balancer_controller", lambda c, _g: {"eks_arn": c.cpu["eks_arn"]}
        ),
        TaskInputSpec(
            "control_plane_role", lambda c, _g: {"notifications": c.notifications}
        ),
        TaskInputSpec(
            "control_record_archive_bucket",
            always_revalidate_reason=(
                "archive location and bucket hardening are resolved from the current site on every invocation"
            ),
        ),
        TaskInputSpec(
            "email_notifications", lambda c, _g: {"notifications": c.notifications}
        ),
        TaskInputSpec(
            "monitoring_resources",
            lambda c, _g: {
                "admin_email": c.notifications.get("admin_email"),
                "policy_source": c.sources["src/gpu_fault/admin/monitoring_policy.py"],
                "policy_asset": c.assets[
                    "deploy/observability/amp-sns-publish-policy.json"
                ],
            },
        ),
        TaskInputSpec("monitoring_install", _monitoring_inputs),
        TaskInputSpec(
            "grafana_install",
            lambda c, _g: {
                "dashboards": c.dashboards,
                "grafana": c.grafana,
                "alert_email": c.notifications.get("admin_email"),
            },
            degraded_result_key="grafana",
        ),
        TaskInputSpec(
            "aurora_refresh",
            lambda c, _g: {
                "source": c.sources["src/gpu_fault/admin/bootstrap_services.py"],
                "cpu": c.cpu,
            },
        ),
        TaskInputSpec(
            "node_keys",
            _node_key_inputs,
            cluster_scoped=True,
        ),
        TaskInputSpec(
            "executor_role", lambda _c, g: {"cluster": g}, cluster_scoped=True
        ),
        TaskInputSpec(
            "adot_writer_role", lambda _c, g: {"cluster": g}, cluster_scoped=True
        ),
    )
}


def task_input_spec(name: str) -> TaskInputSpec:
    prefix, separator, suffix = name.partition(":")
    spec = TASK_INPUT_SPECS.get(prefix)
    if (
        spec is None
        or spec.cluster_scoped != bool(separator)
        or (separator and not suffix)
    ):
        raise ValueError(f"bootstrap task has no declared input policy: {name}")
    return spec


def task_input_fingerprints(context: TaskInputContext) -> dict[str, str]:
    result: dict[str, str] = {}
    for spec in TASK_INPUT_SPECS.values():
        scopes = context.cluster_inputs.items() if spec.cluster_scoped else (("", {}),)
        for cluster_id, cluster in scopes:
            name = f"{spec.name}:{cluster_id}" if spec.cluster_scoped else spec.name
            digest = spec.digest(name, context, cluster)
            if digest is not None:
                result[name] = digest
    return result

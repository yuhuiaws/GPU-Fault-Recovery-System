from __future__ import annotations

import copy
import json
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

from gpu_fault_release import regional_deployment_inventory as inventory
from gpu_fault_release import regional_release_state as state
from gpu_fault_release.regional_release_config import ClusterTarget

OLD_IMAGE = "registry.example/runtime@sha256:" + "a" * 64
NEW_IMAGE = "registry.example/runtime@sha256:" + "b" * 64
DCGM_IMAGE = "registry.example/dcgm@sha256:" + "c" * 64


class RecordingRunner:
    dry_run = False

    def __init__(
        self, handler: Callable[[list[str], dict[str, Any]], str] | None = None
    ) -> None:
        self.calls: list[tuple[list[str], dict[str, Any]]] = []
        self.handler = handler

    def run(self, arguments: list[str], **kwargs: Any) -> str:
        self.calls.append((list(arguments), copy.deepcopy(kwargs)))
        if self.handler is None:
            raise AssertionError(f"unconfigured fake transport: {arguments[:4]}")
        return self.handler(arguments, kwargs)


def deployment(
    name: str,
    *,
    image: str = NEW_IMAGE,
    wheel: str = "candidate-wheel",
    container: str | None = None,
    replicas: int = 1,
) -> dict[str, Any]:
    return {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": {"name": name, "namespace": "gpu-fault-system"},
        "spec": {
            "replicas": replicas,
            "template": {
                "spec": {
                    "containers": [
                        {"name": container or name, "image": image, "env": []}
                    ],
                    "volumes": [{"name": "artifact", "configMap": {"name": wheel}}],
                }
            },
        },
        "status": {"readyReplicas": replicas},
    }


def stable_sample() -> dict[str, Any]:
    return {
        "restarts": {"cpu/api/api": 0},
        "not_ready": [],
        "queue": {"depth": 0, "oldest_age_seconds": 0},
        "remote_commands": {"by_status": {}},
        "critical_alerts": {"count": 0, "alerts": []},
    }


class Clock:
    def __init__(self) -> None:
        self.elapsed = 0.0
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.elapsed

    def time(self) -> float:
        return 2_000_000_000 + self.elapsed

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.elapsed += seconds


class ResourceRelease:
    """Release collaborator backed only by public synthetic resource documents."""

    def __init__(self, cluster_ids: tuple[str, ...] = ("gpu-a",)) -> None:
        targets = tuple(
            ClusterTarget(
                cluster_id=cluster_id,
                context=cluster_id,
                executor_irsa_role_arn="arn:aws:iam::123456789012:role/example",
                region="us-east-1",
                hyperpod_cluster_name="hp-" + cluster_id,
                eks_cluster_arn=f"arn:aws:eks:us-east-1:123456789012:cluster/{cluster_id}",
            )
            for cluster_id in cluster_ids
        )
        self.config = SimpleNamespace(
            clusters=targets,
            namespace="gpu-fault-system",
            site_name="example-site",
            aws_region="us-east-1",
            cpu_kubeconfig="/dev/null",
            cpu_eks_arn="arn:aws:eks:us-east-1:123456789012:cluster/cpu",
            agent_protocol_version=3,
            executor_protocol_version=2,
            component_digests={"node_runtime": "5" * 64, "executor": "6" * 64},
            agent_config_digest="7" * 64,
            runtime_profile_version="candidate-profile",
            release_delivery_sha256="8" * 64,
            health=SimpleNamespace(amp_workspace_id=None),
        )
        self.release_id = "candidate-release"
        self.runtime_image = NEW_IMAGE
        self.executor_image = NEW_IMAGE
        self.dcgm_exporter_image = DCGM_IMAGE
        self.cluster_registry_digest = "registry-digest"
        self.node_wheel_sha = "1" * 64
        self.executor_wheel_sha = "2" * 64
        self.bundle_sha = "3" * 64
        self.node_template_sha = "4" * 64
        self.executor_wheel_cm = "candidate-wheel"
        self.node_names = ("node-a",)
        self.metadata = {
            "required-agent-artifact-sha256": self.node_wheel_sha,
            "required-agent-compatibility-digest": self.config.component_digests[
                "node_runtime"
            ],
            "required-agent-protocol-version": "3",
            "required-agent-config-digest": self.config.agent_config_digest,
            "required-regional-executor-artifact-sha256": self.executor_wheel_sha,
            "required-regional-executor-compatibility-digest": self.config.component_digests[
                "executor"
            ],
            "required-regional-executor-protocol-version": "2",
        }
        self.documents: dict[tuple[str, str, str], Any] = {
            ("cpu", "configmap", "gpu-fault-release-metadata"): {"data": self.metadata}
        }
        self.reads: list[tuple[str, str, str]] = []
        self.runner = RecordingRunner()
        self.heartbeat_ready = True
        self.heartbeat_checks: list[dict[str, Any]] = []
        self.endpoint_checks: list[str] = []
        self.aurora_drift = False
        self.observability_drift = False
        self.state: dict[str, Any] = {}
        self.live_state: dict[str, Any] | Exception = {}
        self.samples = [stable_sample()]
        self.metric_report: dict[str, Any] = {"ready": True, "errors": []}
        self.template_name = "previous-template"
        self.template_bundle = "previous-bundle"
        for target in targets:
            for name in inventory.DEPLOYMENTS:
                self.documents[(target.context, "deployment", name)] = deployment(name)
            reconciler = deployment(
                inventory.GPU_RECONCILER_DEPLOYMENT, container="reconciler"
            )
            reconciler["spec"]["template"]["spec"]["containers"][0]["env"] = [
                {"name": "GPU_FAULT_INSTALLER_BUNDLE_SHA256", "value": self.bundle_sha},
                {
                    "name": "GPU_FAULT_INSTALLER_TEMPLATE_SHA256",
                    "value": self.node_template_sha,
                },
            ]
            self.documents[
                (target.context, "deployment", inventory.GPU_RECONCILER_DEPLOYMENT)
            ] = reconciler
            self.documents[(target.context, "daemonset", "gpu-fault-dcgm-exporter")] = (
                deployment(
                    "gpu-fault-dcgm-exporter",
                    image=DCGM_IMAGE,
                    container="dcgm-exporter",
                )
            )
            self.documents[(target.context, "nodes", "")] = {
                "items": [
                    {
                        "metadata": {
                            "name": "node-a",
                            "uid": "node-a-uid",
                            "labels": {
                                "sagemaker.amazonaws.com/cluster-name": target.hyperpod_cluster_name
                            },
                            "annotations": {
                                "gpu-fault.io/installer-state": "Succeeded",
                                "gpu-fault.io/installer-node-uid": "node-a-uid",
                                "gpu-fault.io/installer-artifact-sha256": self.node_wheel_sha,
                                "gpu-fault.io/installer-config-digest": self.config.agent_config_digest,
                                "gpu-fault.io/installer-bundle-sha256": self.bundle_sha,
                                "gpu-fault.io/installer-template-sha256": self.node_template_sha,
                            },
                        }
                    }
                ]
            }

    def _cpu(self, *arguments: str) -> list[str]:
        return ["kubectl", "--kubeconfig", "/dev/null", *arguments]

    def _gpu(self, target: ClusterTarget, *arguments: str) -> list[str]:
        return ["kubectl", "--context", target.context, *arguments]

    def _get_json(self, arguments: list[str]) -> dict[str, Any]:
        plane = (
            arguments[arguments.index("--context") + 1]
            if "--context" in arguments
            else "cpu"
        )
        offset = arguments.index("get")
        kind = arguments[offset + 1]
        tail = arguments[offset + 2 :]
        name = tail[0] if tail and not tail[0].startswith("-") else ""
        key = plane, kind, name
        self.reads.append(key)
        value = self.documents[key]
        if isinstance(value, Exception):
            raise value
        return copy.deepcopy(value)

    def _config_map_data(self, name: str) -> dict[str, str]:
        return state.config_map_data(self, name)

    def _deployment_wheel(self, prefix: list[str], name: str) -> str | None:
        return state.deployment_wheel(self, prefix, name)

    def _target_node_names(self, target: ClusterTarget) -> tuple[str, ...]:
        return self.node_names

    def _agent_heartbeats_converged(self, target: ClusterTarget, **kwargs: Any) -> bool:
        self.heartbeat_checks.append({"cluster_id": target.cluster_id, **kwargs})
        return self.heartbeat_ready

    def _verify_gpu_control_plane_endpoint(self, target: ClusterTarget) -> None:
        self.endpoint_checks.append(target.cluster_id)

    def _aurora_refresh_drift(self) -> bool:
        return self.aurora_drift

    def _observability_drift(self) -> bool:
        return self.observability_drift

    def _fleet_deployment_id(self, target: ClusterTarget, **kwargs: Any) -> str:
        return "fleet-" + target.cluster_id

    def _deployment_template_name(self, target: ClusterTarget) -> str:
        return self.template_name

    def _template_bundle(self, target: ClusterTarget, template: str) -> str:
        return self.template_bundle

    def _load_state(self) -> dict[str, Any]:
        if isinstance(self.live_state, Exception):
            raise self.live_state
        return copy.deepcopy(self.live_state)

    def _stability_snapshot(self) -> dict[str, Any]:
        return copy.deepcopy(
            self.samples.pop(0) if len(self.samples) > 1 else self.samples[0]
        )

    def _store_io_rejection_series_ready(self) -> dict[str, Any]:
        return copy.deepcopy(self.metric_report)

    def _critical_amp_alerts(self) -> dict[str, Any]:
        return {"count": 0, "alerts": []}


def previous_snapshot(release: ResourceRelease) -> dict[str, Any]:
    return {
        "metadata": dict(release.metadata),
        "runtime_image": OLD_IMAGE,
        "cpu_wheel": "previous-cpu-wheel",
        "runtime_profile_version": "previous-profile",
        "clusters": {
            target.cluster_id: {
                "wheel": "previous-wheel",
                "reconciler_wheel": "previous-wheel",
                "template": "previous-template",
                "bundle": "previous-bundle",
                "dcgm_image": DCGM_IMAGE,
            }
            for target in release.config.clusters
        },
        "agent_identities": {
            target.cluster_id: {
                "node_ids": ["node-a"],
                "agent_protocol_version": 3,
                "artifact_sha256": release.node_wheel_sha,
                "compatibility_digest": release.config.component_digests[
                    "node_runtime"
                ],
                "installer_bundle_sha256": release.bundle_sha,
                "installer_template_sha256": release.node_template_sha,
                "runtime_profile_version": "previous-profile",
                "config_digest": release.config.agent_config_digest,
            }
            for target in release.config.clusters
        },
    }


def mixed_probe_result(release: ResourceRelease) -> dict[str, Any]:
    return {
        "agent_blocker_count": 0,
        "deployment": {
            "cluster_id": release.config.clusters[0].cluster_id,
            "nodes": [
                {"node_id": name, "status": "PENDING"} for name in release.node_names
            ],
            "desired_artifact_sha256": release.node_wheel_sha,
            "desired_config_digest": release.config.agent_config_digest,
            "desired_bundle_sha256": release.bundle_sha,
            "desired_template_sha256": release.node_template_sha,
        },
    }


def json_response(value: object) -> Callable[[list[str], dict[str, Any]], str]:
    return lambda _arguments, _kwargs: json.dumps(value)

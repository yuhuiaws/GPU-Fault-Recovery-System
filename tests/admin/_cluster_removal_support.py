"""Shared, in-memory removal fixtures; no test-module import dependencies."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from gpu_fault.admin import cluster_removal as removal
from gpu_fault.admin.bootstrap_common import BootstrapError
from gpu_fault.admin.cluster_removal import RemoveClusterRequest
from gpu_fault.admin.cluster_removal_state import read_removal_state
from gpu_fault.admin.resource_registry import write_installation_resource_snapshot
from gpu_fault.admin.site import load_site
from gpu_fault.installation_resources import (
    InstallationResource,
    InstallationResourceDeletePolicy,
    InstallationResourceOwnership,
    InstallationResourceSnapshot,
)
from tests.admin.test_admin_site import site_file

EKS_ARN = "arn:aws:eks:us-east-1:123456789012:cluster/gpu-a"
HP_ARN = "arn:aws:sagemaker:us-east-1:123456789012:cluster/hp-a-id"


def resource(
    key: str,
    resource_type: str,
    resource_id: str,
    *,
    policy: InstallationResourceDeletePolicy,
) -> InstallationResource:
    now = datetime.now(timezone.utc)
    return InstallationResource(
        site_id="test-site",
        resource_key=key,
        resource_type=resource_type,
        resource_id=resource_id,
        region="us-east-1",
        account_id="123456789012",
        ownership=(
            InstallationResourceOwnership.CREATED
            if policy is InstallationResourceDeletePolicy.DELETE
            else InstallationResourceOwnership.EXTERNAL
        ),
        delete_policy=policy,
        created_at=now,
        updated_at=now,
    )


def snapshot() -> InstallationResourceSnapshot:
    value = InstallationResourceSnapshot(
        site_id="test-site",
        resources=[
            resource(
                "aws/iam/executor/gpu-a/role",
                "iam_role",
                "executor",
                policy=InstallationResourceDeletePolicy.DELETE,
            ),
            resource(
                "aws/iam/executor/gpu-a/oidc-provider",
                "iam_oidc_provider",
                "arn:aws:iam::123456789012:oidc-provider/test",
                policy=InstallationResourceDeletePolicy.PRESERVE,
            ),
            resource(
                "aws/iam/adot-writer/gpu-a/role",
                "iam_role",
                "adot-writer-role",
                policy=InstallationResourceDeletePolicy.DELETE,
            ),
            resource(
                "cluster/gpu-a/eks",
                "gpu_eks",
                EKS_ARN,
                policy=InstallationResourceDeletePolicy.PRESERVE,
            ),
            resource(
                "cluster/gpu-a/hyperpod",
                "gpu_hyperpod",
                "hp-gpu-a",
                policy=InstallationResourceDeletePolicy.PRESERVE,
            ).model_copy(update={"resource_arn": HP_ARN}),
            resource(
                "aws/route53/vpc-association/gpu-a",
                "route53_vpc_association",
                "Z123:us-east-1:vpc-gpu-a",
                policy=InstallationResourceDeletePolicy.DETACH,
            ),
            resource(
                "aws/nlb",
                "nlb",
                "gpu-fault-nlb",
                policy=InstallationResourceDeletePolicy.DELETE,
            ),
        ],
    )
    return value.model_copy(update={"source_sha256": value.digest()})


class RemovalScenario:
    def __init__(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self.path = site_file(tmp_path)
        self.original = self.path.read_bytes()
        self.calls: list[str] = []
        self.failure: str | None = None
        self.namespace_uid: str | None = "gpu-namespace-1"
        self.node_uid = "node-uid-1"
        self.incarnation = "2026-01-01T00:00:00Z"
        self.lifecycle: str | None = "ACTIVE"
        self.release_identity = "b" * 64
        self.registry = snapshot()

        def network(_runner, *, region, eks_arn):
            cpu = eks_arn.endswith("/control")
            return {
                "vpc_id": "vpc-cpu" if cpu else "vpc-gpu",
                "nat_eips": [],
                "eks_arn": eks_arn,
                "eks_created_at": "cpu-created" if cpu else self.incarnation,
                "eks_endpoint": f"https://{'cpu' if cpu else 'gpu'}.example",
            }

        def identity(_request, _target, _runner, _target_network, _cpu_network):
            return {
                "eks_arn": EKS_ARN,
                "eks_created_at": self.incarnation,
                "hyperpod_arn": HP_ARN,
                "cpu_eks_arn": "arn:aws:eks:us-east-1:123456789012:cluster/control",
                "cpu_eks_created_at": "cpu-created",
                "cpu_namespace_uid": "cpu-namespace",
            }, self.namespace_uid

        def nodes(_site, _target, *, identities):
            identities["node-a"] = self.node_uid
            return ["node-a"]

        def export(site, directory):
            self.event("snapshot")
            write_installation_resource_snapshot(
                site,
                self.registry,
                path=directory / "installation-resources-before.json",
            )
            return self.registry

        def delete_namespace(*_args):
            self.event("namespace-request")
            self.namespace_uid = None

        def detach(*_args, **_kwargs):
            self.event("aws-network")
            return {"revoked_nat_eips": [], "detached_vpc_id": None}

        def keys(*_args, **_kwargs):
            self.event("keys")
            return 1

        scenario = self

        class Cleaner:
            def __init__(self, _site):
                pass

            def validate_supported(self, _resources):
                pass

            def delete(self, resource):
                scenario.event("delete:" + resource.resource_key)

        monkeypatch.setattr(removal, "ResourceCleaner", Cleaner)
        monkeypatch.setattr(
            removal,
            "verify_node_key_ownership",
            lambda _site, _cluster_id, nodes, **kwargs: kwargs.get("saved")
            or {
                "cpu_secret_uid": "cpu-keys",
                "gpu_secret_uid": "gpu-keys",
                "expected_key_sha256": dict.fromkeys(nodes, "a" * 64),
            },
        )
        monkeypatch.setattr(
            removal, "fetch_installation_resource_registry", lambda _site: self.registry
        )
        monkeypatch.setattr(
            removal,
            "membership_runtime_snapshot",
            lambda _site: {
                "registry_generation": 1,
                "registry_content_sha256": "a" * 64,
                "live_release_identity_sha256": self.release_identity,
                "registry_cluster_states": (
                    {"gpu-a": self.lifecycle} if self.lifecycle is not None else {}
                ),
            },
        )
        for name, function in (
            ("_cluster_network", network),
            ("_removal_identity", identity),
            ("_target_nodes", nodes),
            ("_export_registry", export),
            ("_run_kubernetes_cleanup", self.cleanup),
            ("_request_target_namespace_deletion", delete_namespace),
            ("_detach_network", detach),
            ("_remove_node_action_keys", keys),
        ):
            monkeypatch.setattr(removal, name, function)
        for name, event in (
            ("_run_control_plane_drain", "drain"),
            ("_clear_installer_annotations", "annotations"),
            ("_wait_target_namespace_absent", "namespace-wait"),
            ("_run_control_plane_unregister", "unregister"),
            ("sync_installation_resource_snapshot", "aurora"),
            ("_update_bootstrap_state", "bootstrap"),
            ("refresh_failure_domain_map", "failure-domains"),
            ("_sync_release_state", "release-state"),
            ("_verify_removal_parallel", "verify"),
        ):
            monkeypatch.setattr(
                removal,
                name,
                lambda *_args, _event=event, **_kwargs: self.event(_event),
            )

    def cleanup(self, _request, _target, directory):
        self.event("cleanup")
        return directory / "cleanup.json"

    def event(self, name: str) -> None:
        self.calls.append(name)
        if self.failure == name:
            raise BootstrapError(f"simulated {name} failure")
        if name == "drain":
            self.lifecycle = "DRAINING"
        elif name == "unregister":
            self.lifecycle = None

    def request(self, arn: str = EKS_ARN) -> RemoveClusterRequest:
        return RemoveClusterRequest(
            site=load_site(self.path),
            cluster_id="gpu-a",
            confirmation="REMOVE_GPU_CLUSTER",
            gpu_cluster_arn=arn,
        )

    def state(self) -> dict[str, Any]:
        return read_removal_state(self.path.parent / "remove-cluster/gpu-a/state.json")

from __future__ import annotations

import base64
import json
from dataclasses import replace
from typing import Any

import pytest

from gpu_fault_release import regional_release_preflight as preflight
from gpu_fault_release.regional_release_config import ReleaseError
from tests.regional._cov95_release_checks import TOPIC, CheckedRelease
from tests.regional._cov95_release_support import ResourceRelease


@pytest.mark.parametrize(
    "fault,problem",
    [
        ("workspace", "not configured"),
        ("workspace-id", "identity differs"),
        ("workspace-status", "must be ACTIVE"),
        ("definition-status", "unknown AMP definition status"),
        ("namespace", "namespace identity differs"),
        ("data", "definition data"),
        ("topic", "configured SNS topic"),
    ],
)
def test_monitoring_repair_preflight_rejects_unknown_or_unbound_definitions(
    monkeypatch: pytest.MonkeyPatch, fault: str, problem: str
) -> None:
    release = CheckedRelease()
    release.aws["amp", "describe-workspace"]["workspace"]["workspaceId"] = (
        release.config.health.amp_workspace_id
    )
    definition = {
        "status": {"statusCode": "ACTIVE"},
        "name": release.config.health.amp_rule_namespace,
        "data": base64.b64encode(TOPIC.encode()).decode(),
    }
    if fault == "workspace":
        release.config.health.amp_workspace_id = None
    elif fault == "workspace-id":
        release.aws["amp", "describe-workspace"]["workspace"]["workspaceId"] = "foreign"
    elif fault == "workspace-status":
        release.aws["amp", "describe-workspace"]["workspace"]["status"][
            "statusCode"
        ] = "CREATING"
    elif fault == "definition-status":
        definition["status"]["statusCode"] = "UNKNOWN"
    elif fault == "namespace":
        definition["name"] = "other"
    elif fault == "data":
        definition["data"] = "invalid base64"
    else:
        definition["data"] = base64.b64encode(b"other-topic").decode()
    monkeypatch.setattr(preflight, "describe_amp_definition", lambda *_args: definition)
    with pytest.raises(ReleaseError, match=problem):
        preflight.monitoring_repair_preflight(release)
    assert all(
        "describe-workspace" in args for args, _kwargs in release.runner.calls
    ), "failed monitoring identity must stop before downstream transport"


def test_readable_monitoring_definitions_need_no_absence_anchor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    release = CheckedRelease()
    release.aws["amp", "describe-workspace"]["workspace"]["workspaceId"] = (
        release.config.health.amp_workspace_id
    )
    monkeypatch.setattr(
        preflight,
        "describe_amp_definition",
        lambda *_args: {
            "status": {"statusCode": "ACTIVE"},
            "name": release.config.health.amp_rule_namespace,
            "data": base64.b64encode(TOPIC.encode()).decode(),
        },
    )
    report = preflight.monitoring_repair_preflight(release)
    assert report["missing"] == []
    assert report["email_subscription"]["confirmed"] == 1
    assert release.reads == []


class ContextRelease(ResourceRelease):
    def __init__(self) -> None:
        super().__init__()
        self.config.cpu_hyperpod_cluster_name = "hp-cpu"
        self.config.clusters = tuple(
            replace(target, agent_endpoint_allowed_cidrs=("192.0.2.0/24",))
            for target in self.config.clusters
        )
        self.fault = ""
        self.iam = []
        self.documents[("gpu-a", "deployments", "")] = {"items": []}
        self.runner.handler = self.respond

    def _validate_executor_iam_role(self, target: Any) -> None:
        self.iam.append(target.cluster_id)

    def respond(self, arguments: list[str], _kwargs: dict[str, Any]) -> str:
        gpu = "--context" in arguments
        target = self.config.clusters[0] if self.config.clusters else None
        if arguments[0] == "aws":
            cluster = arguments[arguments.index("--cluster-name") + 1]
            is_gpu = cluster != self.config.cpu_hyperpod_cluster_name
            arn = target.eks_cluster_arn if is_gpu else self.config.cpu_eks_arn
            return json.dumps(
                {
                    "EksClusterArn": "other"
                    if self.fault == "hyperpod" and is_gpu
                    else arn,
                    "NodeRecovery": "Automatic"
                    if self.fault == "recovery" and is_gpu
                    else "None",
                }
            )
        if "config" in arguments:
            if self.fault == "arn-format":
                return "not-an-eks-arn"
            if self.fault == "cpu-arn" and not gpu or self.fault == "gpu-arn" and gpu:
                return "arn:aws:eks:us-east-1:123456789012:cluster/foreign"
            return target.eks_cluster_arn if gpu else self.config.cpu_eks_arn
        if "nodes" in arguments:
            if self.fault == "no-nodes":
                return '{"items":[]}'
            addresses = [
                {
                    "type": "InternalIP",
                    "address": "198.51.100.1" if self.fault == "cidr" else "192.0.2.1",
                }
            ]
            if self.fault == "no-address":
                addresses = []
            return json.dumps(
                {
                    "items": [
                        {
                            "metadata": {"name": "node-a"},
                            "status": {"addresses": addresses},
                        }
                    ]
                }
            )
        if "--raw=/readyz" in arguments:
            return "ok"
        raise AssertionError(f"unconfigured context transport: {arguments[:3]}")


@pytest.mark.parametrize(
    "fault,problem",
    [
        ("arn-format", "not an EKS ARN"),
        ("cpu-arn", "cpu_eks_arn"),
        ("gpu-arn", "configured eks_cluster_arn"),
        ("hyperpod", "does not target"),
        ("recovery", "NodeRecovery=None"),
        ("empty-cidrs", "requires Agent endpoint CIDRs"),
        ("no-nodes", "has no Kubernetes nodes"),
        ("no-address", "do not cover nodes"),
        ("cidr", "do not cover nodes"),
    ],
)
def test_context_preflight_binds_eks_hyperpod_and_agent_address_scope(
    fault: str, problem: str
) -> None:
    release = ContextRelease()
    release.fault = fault
    if fault == "empty-cidrs":
        release.config.clusters = tuple(
            replace(target, agent_endpoint_allowed_cidrs=())
            for target in release.config.clusters
        )
    with pytest.raises(ReleaseError, match=problem):
        preflight.ensure_region_contexts(release)
    assert release.iam == [], "failed cluster identity must not reach IAM acceptance"
    assert all("apply" not in args for args, _kwargs in release.runner.calls), (
        "context preflight must remain read-only on failure"
    )


@pytest.mark.parametrize("clusters", [True, False])
def test_context_preflight_accepts_only_selected_scoped_cluster_reads(
    clusters: bool,
) -> None:
    release = ContextRelease()
    if not clusters:
        release.config.clusters = ()
    preflight.ensure_region_contexts(release)
    assert release.iam == (["gpu-a"] if clusters else [])
    assert all("apply" not in args for args, _kwargs in release.runner.calls), (
        "successful context preflight must not apply resources"
    )


def test_absent_retired_inventory_does_not_query_an_unrelated_deployment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    release = ResourceRelease()
    monkeypatch.setattr(preflight.inventory, "GPU_RESOURCES", [])
    preflight.preflight_retired_collectors(release, release.config.clusters[0])
    assert release.reads == []

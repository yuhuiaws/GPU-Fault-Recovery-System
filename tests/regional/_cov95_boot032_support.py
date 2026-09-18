from __future__ import annotations

import base64
import json
from datetime import datetime, timezone

import yaml

from gpu_fault.admin.site import load_site
from gpu_fault.installation_resources import (
    InstallationResource,
    InstallationResourceSnapshot,
)
from gpu_fault.installation_resources import InstallationResourceDeletePolicy as Policy
from gpu_fault.installation_resources import InstallationResourceOwnership as Ownership
from scripts.e2e.regional import boot032_contract as contract
from tests.admin.test_admin_site import site_file

FIXTURE = "abcdef012345"
ACCOUNT = "123456789012"
REGION = "us-east-1"
CA = base64.b64encode(b"fixture-public-CA").decode()


def arn(service, name, *, region=REGION):
    return f"arn:aws:{service}:{region}:{ACCOUNT}:cluster/{name}"


def make_site(root, *, target):
    root.mkdir(mode=0o700, parents=True)
    path = site_file(root)
    document = yaml.safe_load(path.read_text())
    label = f"boot032-{FIXTURE}" if target else "accepted"
    cpu, gpu = label + "-cpu", label + "-gpu-0"
    config = document["spec"]
    config["repositoryRoot"] = str(contract.ROOT)
    config["release"]["manifest"] = str(root / "repo/dist/current-release.json")
    (root / "repo/dist/current-release.json").chmod(0o644)
    config["runtimeProfile"]["source"] = str(root / "repo/config/profile.yaml")
    config["runtimeProfile"]["registrationClusterId"] = gpu
    config["cpu"].update(eksArn=arn("eks", cpu), hyperpodClusterName=cpu)
    config["clusters"][0].update(
        clusterId=gpu,
        context=gpu,
        eksClusterArn=arn("eks", gpu),
        hyperpodClusterName=gpu,
    )
    config["health"]["auroraClusterId"] = label + "-db"
    document["metadata"]["name"] = label + "-site"
    for plane, name in (("cpu", cpu), ("gpu", gpu)):
        kube = root / "secure" / f"{plane}.kubeconfig"
        kube.write_text(
            yaml.safe_dump(
                {
                    "current-context": name,
                    "contexts": [
                        {"name": name, "context": {"cluster": name, "user": "fixture"}}
                    ],
                    "clusters": [
                        {
                            "name": name,
                            "cluster": {
                                "server": f"https://{name}.example.invalid",
                                "certificate-authority-data": CA,
                            },
                        }
                    ],
                    "users": [
                        {
                            "name": "fixture",
                            "user": {"exec": {"command": "never-executed"}},
                        }
                    ],
                }
            )
        )
        kube.chmod(0o600)
        if plane == "cpu":
            config["cpu"]["kubeconfig"] = str(kube)
        else:
            config["gpuKubeconfig"] = str(kube)
    path.write_text(yaml.safe_dump(document, sort_keys=False))
    loaded = load_site(path)
    bootstrap = root / "bootstrap-state.json"
    bootstrap.write_text(
        json.dumps(
            {
                "site_id": loaded.metadata_name,
                "phase": "site-ready",
                "resources": {
                    "initial_deploy_target": {
                        "cpu": {"eks_arn": arn("eks", cpu), "hyperpod_name": cpu},
                        "gpu_clusters": [
                            {"eks_arn": arn("eks", gpu), "hyperpod_name": gpu}
                        ],
                    }
                },
            }
        )
    )
    bootstrap.chmod(0o600)
    return loaded


def resource_snapshot(site):
    cpu, gpu = contract.cluster_specs(site)
    label = site.metadata_name
    definitions = [
        ("cluster/cpu-eks", "cpu_eks", cpu["eks_name"], Policy.PRESERVE),
        ("cluster/cpu-hyperpod", "cpu_hyperpod", cpu["hyperpod_name"], Policy.PRESERVE),
        ("cluster/gpu/eks", "gpu_eks", gpu["eks_name"], Policy.PRESERVE),
        ("cluster/gpu/hyperpod", "gpu_hyperpod", gpu["hyperpod_name"], Policy.PRESERVE),
        ("aws/nlb", "nlb", label + "-nlb", Policy.DELETE),
        ("aws/helm/lbc", "helm_release", "lbc-test", Policy.DELETE),
        (
            "aws/eks/pod-identity-agent",
            "eks_addon",
            "eks-pod-identity-agent",
            Policy.PRESERVE,
        ),
        (
            "aws/aurora/cluster",
            "aurora_cluster",
            site.release_config["health"]["aurora_cluster_id"],
            Policy.DELETE,
        ),
        ("aws/aurora/writer", "aurora_instance", label + "-writer", Policy.DELETE),
        ("aws/aurora/security-group", "security_group", label + "-sg", Policy.DELETE),
    ]
    now = datetime(2026, 9, 12, tzinfo=timezone.utc)
    rows = [
        InstallationResource(
            site_id=label,
            resource_key=key,
            resource_type=kind,
            resource_id=name,
            resource_arn=arn("eks", name, region=site.release_config["aws_region"])
            if kind.endswith("_eks")
            else None,
            provider="kubernetes" if kind == "helm_release" else "aws",
            region=site.release_config["aws_region"],
            account_id=ACCOUNT,
            attributes={"cluster_name": cpu["eks_name"]} if kind == "eks_addon" else {},
            ownership=Ownership.EXTERNAL
            if policy is Policy.PRESERVE
            else Ownership.CREATED,
            delete_policy=policy,
            created_at=now,
            updated_at=now,
        )
        for key, kind, name, policy in definitions
    ]
    value = InstallationResourceSnapshot(site_id=label, resources=rows)
    return value.model_copy(update={"source_sha256": value.digest()})

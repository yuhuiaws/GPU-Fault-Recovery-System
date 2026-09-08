from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any

from gpu_fault.admin.bootstrap_common import (
    SITE_TAG_KEY,
    BootstrapError,
    ClusterIdentity,
    CommandRunner,
    assert_site_tag,
    describe_or_absent,
    safe_name,
)
from gpu_fault.admin.bootstrap_services import (
    _ensure_pod_identity_agent,
    _ensure_pod_identity_association,
    _ensure_role,
    _ensure_service_account,
    _pod_identity_trust,
)

LBC_VERSION = "v2.17.1"
LBC_CHART_VERSION = "1.17.1"
# ``LBC_VERSION`` is a git tag, and a tag is a mutable ref: whoever controls the
# upstream repo can move it to different bytes, and this deploy would hand those
# bytes straight to ``iam:CreatePolicy``. The chart is version-pinned but its
# manifest is fetched over the network too, so the only artifact we can pin by
# content from here is the IAM policy JSON. This is the SHA-256 of
# ``docs/install/iam_policy.json`` at the ``LBC_VERSION`` tag; the fetch is
# rejected unless it matches, so a moved tag or a tampered fetch fails closed
# instead of installing an unreviewed policy. Recompute with
# ``curl -fsSL <raw url> | sha256sum`` when bumping ``LBC_VERSION``.
LBC_IAM_POLICY_SHA256 = (
    "16f232c9d9f79366fe949c4550ad517a202380058a9e48d45a4e215044a20a6a"
)


def _controller_ready(runner: CommandRunner, cpu_kubeconfig: Path) -> bool:
    try:
        document = json.loads(
            runner.run(
                [
                    "kubectl",
                    "--kubeconfig",
                    str(cpu_kubeconfig),
                    "get",
                    "deployment",
                    "-A",
                    "-o",
                    "json",
                ]
            )
        )
    except BootstrapError:
        return False
    for item in document.get("items", []):
        containers = (
            item.get("spec", {})
            .get("template", {})
            .get("spec", {})
            .get("containers", [])
        )
        if not any(
            re.search(
                r"load-balancer|operator-alb",
                str(container.get("image", "")),
                re.IGNORECASE,
            )
            for container in containers
        ):
            continue
        replicas = int(item.get("spec", {}).get("replicas", 0) or 0)
        ready = int(item.get("status", {}).get("readyReplicas", 0) or 0)
        if replicas > 0 and ready == replicas:
            return True
    return False


def _lbc_release_exists(cpu_kubeconfig: Path) -> bool:
    result = subprocess.run(
        [
            "helm",
            "--kubeconfig",
            str(cpu_kubeconfig),
            "-n",
            "kube-system",
            "status",
            "aws-load-balancer-controller",
        ],
        text=True,
        capture_output=True,
    )
    if result.returncode == 0:
        return True
    if "release: not found" in result.stderr.lower():
        return False
    raise BootstrapError(
        "cannot determine aws-load-balancer-controller ownership: "
        + result.stderr.strip()
    )


def _lbc_release_matches(
    runner: CommandRunner,
    *,
    cpu: ClusterIdentity,
    cpu_kubeconfig: Path,
) -> bool:
    status = json.loads(
        runner.run(
            [
                "helm",
                "--kubeconfig",
                str(cpu_kubeconfig),
                "-n",
                "kube-system",
                "status",
                "aws-load-balancer-controller",
                "-o",
                "json",
            ]
        )
    )
    values = json.loads(
        runner.run(
            [
                "helm",
                "--kubeconfig",
                str(cpu_kubeconfig),
                "-n",
                "kube-system",
                "get",
                "values",
                "aws-load-balancer-controller",
                "-o",
                "json",
            ]
        )
    )
    chart_version = str(
        (((status.get("chart") or {}).get("metadata") or {}).get("version")) or ""
    )
    service_account = values.get("serviceAccount") or {}
    return (
        chart_version == LBC_CHART_VERSION
        and values.get("clusterName") == cpu.eks_name
        and values.get("region") == cpu.region
        and values.get("vpcId") == cpu.vpc_id
        and int(values.get("replicaCount") or 0) == 2
        and service_account.get("create") is False
        and service_account.get("name") == "aws-load-balancer-controller"
    )


def _lbc_result(
    role: dict[str, str],
    policy_arn: str,
    association: dict[str, str],
    *,
    reused: bool,
) -> dict[str, Any]:
    return {
        "reused": reused,
        "role_arn": role["role_arn"],
        "role_ownership": role["ownership"],
        "policy_arn": policy_arn,
        "policy_ownership": "CREATED",
        "association_id": association["association_id"],
        "association_ownership": association["ownership"],
        "cluster_name": association["cluster_name"],
        "namespace": association["namespace"],
        "service_account": association["service_account"],
    }


def ensure_load_balancer_controller(
    runner: CommandRunner,
    *,
    cpu: ClusterIdentity,
    cpu_kubeconfig: Path,
    state_dir: Path,
    site_id: str,
) -> dict[str, Any]:
    release_exists = _lbc_release_exists(cpu_kubeconfig)
    controller_ready = _controller_ready(runner, cpu_kubeconfig)
    release_matches = (
        _lbc_release_matches(
            runner,
            cpu=cpu,
            cpu_kubeconfig=cpu_kubeconfig,
        )
        if release_exists
        else False
    )
    if controller_ready and not release_exists:
        return {"external": True, "controller": "cluster-managed"}
    _ensure_pod_identity_agent(runner, cpu, site_id)
    policy_name = safe_name(f"gpu-fault-{site_id}-lbc-policy", maximum=128)
    policy_arn = f"arn:aws:iam::{cpu.account_id}:policy/{policy_name}"
    # `get-policy` carries the policy's tags: one read decides both existence
    # and ownership, and only NoSuchEntity reads as absent.
    policy = describe_or_absent(
        runner,
        cpu.region,
        "iam",
        "get-policy",
        "--policy-arn",
        policy_arn,
        not_found=("NoSuchEntity",),
    )
    policy_file = state_dir / "aws-load-balancer-controller-policy.json"
    if policy is not None:
        policy_tags = (policy.get("Policy") or {}).get("Tags", [])
        if not assert_site_tag(
            policy_tags,
            site_id=site_id,
            description=f"IAM policy {policy_arn}",
            allow_missing=True,
        ):
            runner.run(
                [
                    "aws",
                    "iam",
                    "tag-policy",
                    "--policy-arn",
                    policy_arn,
                    "--tags",
                    f"Key={SITE_TAG_KEY},Value={site_id}",
                ],
                mutate=True,
                capture=False,
            )
    else:
        runner.run(
            [
                "curl",
                "-fsSLo",
                str(policy_file),
                "https://raw.githubusercontent.com/kubernetes-sigs/"
                f"aws-load-balancer-controller/{LBC_VERSION}/"
                "docs/install/iam_policy.json",
            ],
            mutate=True,
            capture=False,
        )
        fetched = policy_file.read_bytes()
        digest = hashlib.sha256(fetched).hexdigest()
        if digest != LBC_IAM_POLICY_SHA256:
            raise BootstrapError(
                "aws-load-balancer-controller iam_policy.json at "
                f"{LBC_VERSION} does not match the pinned checksum "
                f"(expected {LBC_IAM_POLICY_SHA256}, fetched {digest}); "
                "the upstream tag may have moved or the download was tampered "
                "with -- refusing to create an unreviewed IAM policy"
            )
        runner.run(
            [
                "aws",
                "iam",
                "create-policy",
                "--policy-name",
                policy_name,
                "--policy-document",
                f"file://{policy_file}",
                "--tags",
                f"Key={SITE_TAG_KEY},Value={site_id}",
            ],
            mutate=True,
            capture=False,
        )
    role_name = safe_name(f"gpu-fault-{site_id}-lbc", maximum=64)
    role = _ensure_role(
        runner,
        account_id=cpu.account_id,
        role_name=role_name,
        trust=_pod_identity_trust(cpu),
        policy_name=None,
        policy=None,
        site_id=site_id,
    )
    attached = runner.aws_json(
        cpu.region,
        "iam",
        "list-attached-role-policies",
        "--role-name",
        role_name,
    ).get("AttachedPolicies", [])
    if policy_arn not in {
        str(item.get("PolicyArn") or "") for item in attached if isinstance(item, dict)
    }:
        runner.run(
            [
                "aws",
                "iam",
                "attach-role-policy",
                "--role-name",
                role_name,
                "--policy-arn",
                policy_arn,
            ],
            mutate=True,
            capture=False,
        )
    _ensure_service_account(
        runner,
        kubeconfig=cpu_kubeconfig,
        namespace="kube-system",
        name="aws-load-balancer-controller",
    )
    association = _ensure_pod_identity_association(
        runner,
        cluster=cpu,
        namespace="kube-system",
        service_account="aws-load-balancer-controller",
        role_arn=role["role_arn"],
    )
    if release_exists and controller_ready and release_matches:
        return _lbc_result(role, policy_arn, association, reused=True)
    runner.run(
        ["helm", "repo", "add", "eks", "https://aws.github.io/eks-charts"],
        mutate=True,
        capture=False,
    )
    runner.run(["helm", "repo", "update", "eks"], mutate=True, capture=False)
    runner.run(
        [
            "helm",
            "upgrade",
            "--install",
            "aws-load-balancer-controller",
            "eks/aws-load-balancer-controller",
            "--kubeconfig",
            str(cpu_kubeconfig),
            "--namespace",
            "kube-system",
            "--version",
            LBC_CHART_VERSION,
            "--set",
            f"clusterName={cpu.eks_name}",
            "--set",
            f"region={cpu.region}",
            "--set",
            f"vpcId={cpu.vpc_id}",
            "--set",
            "replicaCount=2",
            "--set",
            "serviceAccount.create=false",
            "--set",
            "serviceAccount.name=aws-load-balancer-controller",
        ],
        mutate=True,
        capture=False,
    )
    runner.run(
        [
            "kubectl",
            "--kubeconfig",
            str(cpu_kubeconfig),
            "-n",
            "kube-system",
            "rollout",
            "status",
            "deployment/aws-load-balancer-controller",
            "--timeout=10m",
        ],
        capture=False,
    )
    return _lbc_result(role, policy_arn, association, reused=False)

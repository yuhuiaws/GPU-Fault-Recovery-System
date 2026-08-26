from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping, cast
from urllib.parse import urlsplit

from gpu_fault.admin_bootstrap_common import (
    BootstrapError,
    ClusterIdentity,
    CommandRunner,
    SITE_TAG_KEY,
    assert_site_tag,
    kubectl_apply,
    safe_name,
    tag_map,
)


LBC_VERSION = "v2.17.1"
LBC_CHART_VERSION = "1.17.1"


def _ensure_service_account(
    runner: CommandRunner,
    *,
    kubeconfig: Path,
    namespace: str,
    name: str,
) -> None:
    manifest = runner.run(
        [
            "kubectl",
            "--kubeconfig",
            str(kubeconfig),
            "-n",
            namespace,
            "create",
            "serviceaccount",
            name,
            "--dry-run=client",
            "-o",
            "yaml",
        ]
    )
    kubectl_apply(runner, kubeconfig, manifest)


def _ensure_pod_identity_agent(
    runner: CommandRunner,
    cluster: ClusterIdentity,
    site_id: str,
) -> dict[str, str]:
    exists = (
        subprocess.run(
            [
                "aws",
                "eks",
                "describe-addon",
                "--region",
                cluster.region,
                "--cluster-name",
                cluster.eks_name,
                "--addon-name",
                "eks-pod-identity-agent",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ).returncode
        == 0
    )
    ownership = "CREATED"
    if exists:
        addon = runner.aws_json(
            cluster.region,
            "eks",
            "describe-addon",
            "--cluster-name",
            cluster.eks_name,
            "--addon-name",
            "eks-pod-identity-agent",
        )["addon"]
        tags = runner.aws_json(
            cluster.region,
            "eks",
            "list-tags-for-resource",
            "--resource-arn",
            str(addon["addonArn"]),
        ).get("tags", {})
        if tag_map(tags).get(SITE_TAG_KEY) != site_id:
            ownership = "EXTERNAL"
    else:
        runner.run(
            [
                "aws",
                "eks",
                "create-addon",
                "--region",
                cluster.region,
                "--cluster-name",
                cluster.eks_name,
                "--addon-name",
                "eks-pod-identity-agent",
                "--tags",
                f"{SITE_TAG_KEY}={site_id}",
            ],
            mutate=True,
            capture=False,
        )
    runner.run(
        [
            "aws",
            "eks",
            "wait",
            "addon-active",
            "--region",
            cluster.region,
            "--cluster-name",
            cluster.eks_name,
            "--addon-name",
            "eks-pod-identity-agent",
        ],
        mutate=True,
        capture=False,
    )
    return {
        "cluster_name": cluster.eks_name,
        "addon_name": "eks-pod-identity-agent",
        "ownership": ownership,
    }


def _pod_identity_trust() -> dict[str, Any]:
    return {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {"Service": "pods.eks.amazonaws.com"},
                "Action": ["sts:AssumeRole", "sts:TagSession"],
            }
        ],
    }


def _ensure_role(
    runner: CommandRunner,
    *,
    account_id: str,
    role_name: str,
    trust: dict[str, Any],
    policy_name: str | None,
    policy: dict[str, Any] | None,
    site_id: str,
) -> dict[str, str]:
    exists = (
        subprocess.run(
            ["aws", "iam", "get-role", "--role-name", role_name],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ).returncode
        == 0
    )
    trust_text = json.dumps(trust, separators=(",", ":"))
    if exists:
        role = json.loads(
            runner.run(
                [
                    "aws",
                    "iam",
                    "get-role",
                    "--role-name",
                    role_name,
                    "--output",
                    "json",
                ]
            )
        )["Role"]
        tagged = assert_site_tag(
            role.get("Tags"),
            site_id=site_id,
            description=f"IAM role {role_name}",
            allow_missing=True,
        )
        if not tagged:
            runner.run(
                [
                    "aws",
                    "iam",
                    "tag-role",
                    "--role-name",
                    role_name,
                    "--tags",
                    f"Key={SITE_TAG_KEY},Value={site_id}",
                ],
                mutate=True,
                capture=False,
            )
        runner.run(
            [
                "aws",
                "iam",
                "update-assume-role-policy",
                "--role-name",
                role_name,
                "--policy-document",
                trust_text,
            ],
            mutate=True,
            capture=False,
        )
    else:
        runner.run(
            [
                "aws",
                "iam",
                "create-role",
                "--role-name",
                role_name,
                "--assume-role-policy-document",
                trust_text,
                "--tags",
                f"Key=gpu-fault:site-id,Value={site_id}",
            ],
            mutate=True,
            capture=False,
        )
    if policy_name and policy is not None:
        runner.run(
            [
                "aws",
                "iam",
                "put-role-policy",
                "--role-name",
                role_name,
                "--policy-name",
                policy_name,
                "--policy-document",
                json.dumps(policy, separators=(",", ":")),
            ],
            mutate=True,
            capture=False,
        )
    return {
        "role_arn": f"arn:aws:iam::{account_id}:role/{role_name}",
        "ownership": "CREATED",
        "inline_policy_name": policy_name or "",
    }


def _ensure_pod_identity_association(
    runner: CommandRunner,
    *,
    cluster: ClusterIdentity,
    namespace: str,
    service_account: str,
    role_arn: str,
) -> dict[str, str]:
    associations = cast(
        list[dict[str, Any]],
        runner.aws_json(
            cluster.region,
            "eks",
            "list-pod-identity-associations",
            "--cluster-name",
            cluster.eks_name,
            "--namespace",
            namespace,
            "--service-account",
            service_account,
        ).get("associations", []),
    )
    if associations:
        association_id = str(associations[0]["associationId"])
        runner.run(
            [
                "aws",
                "eks",
                "update-pod-identity-association",
                "--region",
                cluster.region,
                "--cluster-name",
                cluster.eks_name,
                "--association-id",
                association_id,
                "--role-arn",
                role_arn,
            ],
            mutate=True,
            capture=False,
        )
    else:
        created = runner.aws_json(
            cluster.region,
            "eks",
            "create-pod-identity-association",
            "--cluster-name",
            cluster.eks_name,
            "--namespace",
            namespace,
            "--service-account",
            service_account,
            "--role-arn",
            role_arn,
            mutate=True,
        )
        association_id = str(
            (created.get("association") or {}).get("associationId")
            or f"{cluster.eks_name}:{namespace}:{service_account}"
        )
    return {
        "association_id": association_id,
        "cluster_name": cluster.eks_name,
        "namespace": namespace,
        "service_account": service_account,
        "ownership": "CREATED",
    }


def ensure_control_plane_role(
    runner: CommandRunner,
    *,
    cpu: ClusterIdentity,
    cpu_kubeconfig: Path,
    namespace: str,
    site_id: str,
    email_sender: str | None = None,
) -> dict[str, str]:
    _ensure_pod_identity_agent(runner, cpu, site_id)
    _ensure_service_account(
        runner,
        kubeconfig=cpu_kubeconfig,
        namespace=namespace,
        name="gpu-fault-control-plane",
    )
    role_name = safe_name(f"gpu-fault-{site_id}-control", maximum=64)
    statements: list[dict[str, Any]] = [
        {
            "Effect": "Allow",
            "Action": [
                "sagemaker:DescribeCluster",
                "sagemaker:ListClusterNodes",
                "sagemaker:DescribeClusterNode",
            ],
            "Resource": (f"arn:aws:sagemaker:{cpu.region}:{cpu.account_id}:cluster/*"),
        }
    ]
    if email_sender:
        statements.append(
            {
                "Sid": "AdministratorEmail",
                "Effect": "Allow",
                "Action": "ses:SendEmail",
                "Resource": (
                    f"arn:aws:ses:{cpu.region}:{cpu.account_id}:identity/{email_sender}"
                ),
                "Condition": {
                    "StringEquals": {
                        "ses:FromAddress": email_sender,
                    }
                },
            }
        )
    role = _ensure_role(
        runner,
        account_id=cpu.account_id,
        role_name=role_name,
        trust=_pod_identity_trust(),
        policy_name="GPUFaultRegionalObserve",
        policy={
            "Version": "2012-10-17",
            "Statement": statements,
        },
        site_id=site_id,
    )
    association = _ensure_pod_identity_association(
        runner,
        cluster=cpu,
        namespace=namespace,
        service_account="gpu-fault-control-plane",
        role_arn=role["role_arn"],
    )
    return {
        **role,
        "association_id": association["association_id"],
        "association_ownership": association["ownership"],
        "cluster_name": association["cluster_name"],
        "namespace": association["namespace"],
        "service_account": association["service_account"],
    }


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


def ensure_load_balancer_controller(
    runner: CommandRunner,
    *,
    cpu: ClusterIdentity,
    cpu_kubeconfig: Path,
    state_dir: Path,
    site_id: str,
) -> dict[str, Any]:
    release_exists = _lbc_release_exists(cpu_kubeconfig)
    if _controller_ready(runner, cpu_kubeconfig) and not release_exists:
        return {
            "external": True,
            "controller": "cluster-managed",
        }
    _ensure_pod_identity_agent(runner, cpu, site_id)
    policy_name = safe_name(f"gpu-fault-{site_id}-lbc-policy", maximum=128)
    policy_arn = f"arn:aws:iam::{cpu.account_id}:policy/{policy_name}"
    policy_exists = (
        subprocess.run(
            ["aws", "iam", "get-policy", "--policy-arn", policy_arn],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ).returncode
        == 0
    )
    policy_file = state_dir / "aws-load-balancer-controller-policy.json"
    if policy_exists:
        policy_tags = json.loads(
            runner.run(
                [
                    "aws",
                    "iam",
                    "list-policy-tags",
                    "--policy-arn",
                    policy_arn,
                    "--output",
                    "json",
                ]
            )
        ).get("Tags", [])
        tagged = assert_site_tag(
            policy_tags,
            site_id=site_id,
            description=f"IAM policy {policy_arn}",
            allow_missing=True,
        )
        if not tagged:
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
        trust=_pod_identity_trust(),
        policy_name=None,
        policy=None,
        site_id=site_id,
    )
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
        mutate=True,
        capture=False,
    )
    return {
        "reused": False,
        "role_arn": role["role_arn"],
        "role_ownership": "CREATED",
        "policy_arn": policy_arn,
        "policy_ownership": "CREATED",
        "association_id": association["association_id"],
        "association_ownership": association["ownership"],
        "cluster_name": association["cluster_name"],
        "namespace": association["namespace"],
        "service_account": association["service_account"],
    }


def _oidc_thumbprint(runner: CommandRunner, issuer: str) -> str:
    host = urlsplit(issuer).hostname
    if not host:
        raise BootstrapError(f"invalid EKS OIDC issuer: {issuer}")
    output = runner.run(
        [
            "openssl",
            "s_client",
            "-servername",
            host,
            "-connect",
            f"{host}:443",
            "-showcerts",
        ],
        input_text="",
    )
    certificates = re.findall(
        r"-----BEGIN CERTIFICATE-----.*?-----END CERTIFICATE-----",
        output,
        flags=re.DOTALL,
    )
    if not certificates:
        raise BootstrapError("cannot read the EKS OIDC certificate chain")
    with tempfile.NamedTemporaryFile("w", encoding="utf-8") as handle:
        handle.write(certificates[-1])
        handle.flush()
        fingerprint = runner.run(
            [
                "openssl",
                "x509",
                "-in",
                handle.name,
                "-noout",
                "-fingerprint",
                "-sha1",
            ]
        )
    return fingerprint.rsplit("=", 1)[-1].replace(":", "").lower()


def _ensure_oidc_provider(
    runner: CommandRunner,
    cluster: ClusterIdentity,
    site_id: str,
) -> tuple[str, str, str]:
    issuer = runner.aws_text(
        cluster.region,
        "eks",
        "describe-cluster",
        "--name",
        cluster.eks_name,
        "--query",
        "cluster.identity.oidc.issuer",
    )
    if not issuer.startswith("https://"):
        raise BootstrapError(f"EKS cluster {cluster.eks_name} has no OIDC issuer")
    issuer_host = issuer.removeprefix("https://")
    provider_arn = f"arn:aws:iam::{cluster.account_id}:oidc-provider/{issuer_host}"
    exists = (
        subprocess.run(
            [
                "aws",
                "iam",
                "get-open-id-connect-provider",
                "--open-id-connect-provider-arn",
                provider_arn,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ).returncode
        == 0
    )
    ownership = "EXTERNAL"
    if exists:
        tags = json.loads(
            runner.run(
                [
                    "aws",
                    "iam",
                    "list-open-id-connect-provider-tags",
                    "--open-id-connect-provider-arn",
                    provider_arn,
                    "--output",
                    "json",
                ]
            )
        ).get("Tags", [])
        if tag_map(tags).get(SITE_TAG_KEY) == site_id:
            ownership = "CREATED"
    else:
        runner.run(
            [
                "aws",
                "iam",
                "create-open-id-connect-provider",
                "--url",
                issuer,
                "--client-id-list",
                "sts.amazonaws.com",
                "--thumbprint-list",
                _oidc_thumbprint(runner, issuer),
                "--tags",
                f"Key={SITE_TAG_KEY},Value={site_id}",
            ],
            mutate=True,
            capture=False,
        )
        ownership = "CREATED"
    return provider_arn, issuer_host, ownership


def ensure_executor_role(
    runner: CommandRunner,
    *,
    cluster: ClusterIdentity,
    namespace: str,
    site_id: str,
) -> dict[str, str]:
    provider_arn, issuer, provider_ownership = _ensure_oidc_provider(
        runner,
        cluster,
        site_id,
    )
    service_account = "gpu-fault-cluster-executor"
    role_name = safe_name(
        f"gpu-fault-{site_id}-{cluster.hyperpod_name}-executor",
        maximum=64,
    )
    trust = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {"Federated": provider_arn},
                "Action": "sts:AssumeRoleWithWebIdentity",
                "Condition": {
                    "StringEquals": {
                        f"{issuer}:aud": "sts.amazonaws.com",
                        f"{issuer}:sub": (
                            f"system:serviceaccount:{namespace}:{service_account}"
                        ),
                    }
                },
            }
        ],
    }
    policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Action": [
                    "sagemaker:DescribeCluster",
                    "sagemaker:ListClusterNodes",
                    "sagemaker:DescribeClusterNode",
                    "sagemaker:BatchRebootClusterNodes",
                ],
                "Resource": cluster.hyperpod_arn,
            }
        ],
    }
    role = _ensure_role(
        runner,
        account_id=cluster.account_id,
        role_name=role_name,
        trust=trust,
        policy_name="GPUFaultRegionalExecutor",
        policy=policy,
        site_id=site_id,
    )
    return {
        **role,
        "oidc_provider_arn": provider_arn,
        "oidc_provider_ownership": provider_ownership,
        "cluster_name": cluster.eks_name,
    }


def _ensure_amp_workspace(
    runner: CommandRunner,
    *,
    cpu: ClusterIdentity,
    site_id: str,
) -> tuple[str, bool]:
    alias = safe_name(f"gpu-fault-{site_id}", maximum=100)
    workspaces = cast(
        list[dict[str, Any]],
        runner.aws_json(
            cpu.region,
            "amp",
            "list-workspaces",
            "--alias",
            alias,
        ).get("workspaces", []),
    )
    if workspaces:
        workspace_id = str(workspaces[0]["workspaceId"])
        workspace_arn = str(workspaces[0]["arn"])
        tags = runner.aws_json(
            cpu.region,
            "amp",
            "list-tags-for-resource",
            "--resource-arn",
            workspace_arn,
        ).get("tags", {})
        tagged = assert_site_tag(
            tags,
            site_id=site_id,
            description=f"AMP workspace {workspace_id}",
            allow_missing=True,
        )
        if not tagged and not runner.dry_run:
            runner.run(
                [
                    "aws",
                    "amp",
                    "tag-resource",
                    "--region",
                    cpu.region,
                    "--resource-arn",
                    workspace_arn,
                    "--tags",
                    f"{SITE_TAG_KEY}={site_id}",
                ],
                mutate=True,
                capture=False,
            )
    elif runner.dry_run:
        workspace_id = "ws-dryrun"
    else:
        workspace_id = str(
            runner.aws_json(
                cpu.region,
                "amp",
                "create-workspace",
                "--alias",
                alias,
                "--tags",
                f"gpu-fault:site-id={site_id}",
                mutate=True,
            )["workspaceId"]
        )
    if not runner.dry_run:
        deadline = time.monotonic() + 600
        while time.monotonic() < deadline:
            workspace = runner.aws_json(
                cpu.region,
                "amp",
                "describe-workspace",
                "--workspace-id",
                workspace_id,
            )["workspace"]
            status = (workspace.get("status") or {}).get("statusCode")
            if status == "ACTIVE":
                break
            if status in {"CREATION_FAILED", "DELETING"}:
                raise BootstrapError(f"AMP workspace entered {status}")
            time.sleep(5)
        else:
            raise BootstrapError("AMP workspace did not become ACTIVE")
    return workspace_id, False


def _ensure_sns_topic(
    runner: CommandRunner,
    *,
    cpu: ClusterIdentity,
    site_id: str,
) -> tuple[str, bool]:
    topic_name = safe_name(f"gpu-fault-{site_id}-alerts", maximum=256)
    expected_topic_arn = f"arn:aws:sns:{cpu.region}:{cpu.account_id}:{topic_name}"
    topic_reused = (
        subprocess.run(
            [
                "aws",
                "sns",
                "get-topic-attributes",
                "--region",
                cpu.region,
                "--topic-arn",
                expected_topic_arn,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ).returncode
        == 0
    )
    if topic_reused:
        topic_arn = expected_topic_arn
        tags = runner.aws_json(
            cpu.region,
            "sns",
            "list-tags-for-resource",
            "--resource-arn",
            topic_arn,
        ).get("Tags", [])
        tagged = assert_site_tag(
            tags,
            site_id=site_id,
            description=f"SNS topic {topic_arn}",
            allow_missing=True,
        )
        if not tagged and not runner.dry_run:
            runner.run(
                [
                    "aws",
                    "sns",
                    "tag-resource",
                    "--region",
                    cpu.region,
                    "--resource-arn",
                    topic_arn,
                    "--tags",
                    f"Key={SITE_TAG_KEY},Value={site_id}",
                ],
                mutate=True,
                capture=False,
            )
    else:
        topic_arn = runner.aws_text(
            cpu.region,
            "sns",
            "create-topic",
            "--name",
            topic_name,
            "--tags",
            f"Key=gpu-fault:site-id,Value={site_id}",
            "--query",
            "TopicArn",
            mutate=True,
        )
    return topic_arn, False


def _ensure_sqs_queue(
    runner: CommandRunner,
    *,
    cpu: ClusterIdentity,
    site_id: str,
    topic_arn: str,
) -> tuple[str, str, bool]:
    queue_name = safe_name(f"gpu-fault-{site_id}-alerts", maximum=80)
    queue_lookup = subprocess.run(
        [
            "aws",
            "sqs",
            "get-queue-url",
            "--region",
            cpu.region,
            "--queue-name",
            queue_name,
            "--query",
            "QueueUrl",
            "--output",
            "text",
        ],
        text=True,
        capture_output=True,
    )
    queue_reused = queue_lookup.returncode == 0
    if queue_reused:
        queue_url = queue_lookup.stdout.strip()
        tags = runner.aws_json(
            cpu.region,
            "sqs",
            "list-queue-tags",
            "--queue-url",
            queue_url,
        ).get("Tags", {})
        tagged = assert_site_tag(
            tags,
            site_id=site_id,
            description=f"SQS queue {queue_url}",
            allow_missing=True,
        )
        if not tagged and not runner.dry_run:
            runner.run(
                [
                    "aws",
                    "sqs",
                    "tag-queue",
                    "--region",
                    cpu.region,
                    "--queue-url",
                    queue_url,
                    "--tags",
                    f"{SITE_TAG_KEY}={site_id}",
                ],
                mutate=True,
                capture=False,
            )
    else:
        queue_url = runner.aws_text(
            cpu.region,
            "sqs",
            "create-queue",
            "--queue-name",
            queue_name,
            "--tags",
            f"gpu-fault:site-id={site_id}",
            "--query",
            "QueueUrl",
            mutate=True,
        )
    queue_arn = runner.aws_text(
        cpu.region,
        "sqs",
        "get-queue-attributes",
        "--queue-url",
        queue_url,
        "--attribute-names",
        "QueueArn",
        "--query",
        "Attributes.QueueArn",
    )
    queue_policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {"Service": "sns.amazonaws.com"},
                "Action": "sqs:SendMessage",
                "Resource": queue_arn,
                "Condition": {"ArnEquals": {"aws:SourceArn": topic_arn}},
            }
        ],
    }
    runner.run(
        [
            "aws",
            "sqs",
            "set-queue-attributes",
            "--region",
            cpu.region,
            "--queue-url",
            queue_url,
            "--attributes",
            json.dumps(
                {
                    "Policy": json.dumps(
                        queue_policy,
                        separators=(",", ":"),
                    )
                },
                separators=(",", ":"),
            ),
        ],
        mutate=True,
        capture=False,
    )
    return queue_url, queue_arn, False


def _ensure_monitoring_subscriptions(
    runner: CommandRunner,
    *,
    cpu: ClusterIdentity,
    topic_arn: str,
    queue_arn: str,
    alert_email: str | None,
) -> tuple[str, str]:
    subscriptions = cast(
        list[dict[str, Any]],
        runner.aws_json(
            cpu.region,
            "sns",
            "list-subscriptions-by-topic",
            "--topic-arn",
            topic_arn,
        ).get("Subscriptions", []),
    )
    queue_subscription = next(
        (item for item in subscriptions if item.get("Endpoint") == queue_arn),
        None,
    )
    if queue_subscription is None:
        queue_subscription_arn = runner.aws_text(
            cpu.region,
            "sns",
            "subscribe",
            "--topic-arn",
            topic_arn,
            "--protocol",
            "sqs",
            "--notification-endpoint",
            queue_arn,
            "--query",
            "SubscriptionArn",
            mutate=True,
        )
        queue_subscription_ownership = "CREATED"
    else:
        queue_subscription_arn = str(queue_subscription.get("SubscriptionArn") or "")
        queue_subscription_ownership = "CREATED"
    if alert_email and not any(
        item.get("Endpoint") == alert_email for item in subscriptions
    ):
        runner.run(
            [
                "aws",
                "sns",
                "subscribe",
                "--region",
                cpu.region,
                "--topic-arn",
                topic_arn,
                "--protocol",
                "email",
                "--notification-endpoint",
                alert_email,
            ],
            mutate=True,
            capture=False,
        )
    return queue_subscription_arn, queue_subscription_ownership


def ensure_monitoring_resources(
    runner: CommandRunner,
    *,
    cpu: ClusterIdentity,
    site_id: str,
    alert_email: str | None,
) -> dict[str, Any]:
    workspace_id, workspace_reused = _ensure_amp_workspace(
        runner,
        cpu=cpu,
        site_id=site_id,
    )
    topic_arn, topic_reused = _ensure_sns_topic(
        runner,
        cpu=cpu,
        site_id=site_id,
    )
    queue_url, queue_arn, queue_reused = _ensure_sqs_queue(
        runner,
        cpu=cpu,
        site_id=site_id,
        topic_arn=topic_arn,
    )
    (
        queue_subscription_arn,
        queue_subscription_ownership,
    ) = _ensure_monitoring_subscriptions(
        runner,
        cpu=cpu,
        topic_arn=topic_arn,
        queue_arn=queue_arn,
        alert_email=alert_email,
    )
    return {
        "workspace_id": workspace_id,
        "workspace_ownership": "REUSED" if workspace_reused else "CREATED",
        "sns_topic_arn": topic_arn,
        "sns_topic_ownership": "REUSED" if topic_reused else "CREATED",
        "sqs_queue_url": queue_url,
        "sqs_queue_arn": queue_arn,
        "sqs_queue_ownership": "REUSED" if queue_reused else "CREATED",
        "queue_subscription_arn": queue_subscription_arn,
        "queue_subscription_ownership": queue_subscription_ownership,
    }


def install_monitoring(
    runner: CommandRunner,
    *,
    repository_root: Path,
    cpu: ClusterIdentity,
    cpu_kubeconfig: Path,
    namespace: str,
    site_id: str,
    monitoring: Mapping[str, Any],
    adot_image: str,
    alert_email: str | None,
) -> dict[str, Any]:
    topic_name = monitoring["sns_topic_arn"].rsplit(":", 1)[-1]
    role_name = safe_name(f"gpu-fault-{site_id}-amp-writer", maximum=64)
    role = _ensure_role(
        runner,
        account_id=cpu.account_id,
        role_name=role_name,
        trust=_pod_identity_trust(),
        policy_name="gpu-fault-amp-remote-write",
        policy={
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Action": ["aps:RemoteWrite"],
                    "Resource": (
                        f"arn:aws:aps:{cpu.region}:{cpu.account_id}:"
                        f"workspace/{monitoring['workspace_id']}"
                    ),
                }
            ],
        },
        site_id=site_id,
    )
    environment = {
        **os.environ,
        "AWS_REGION": cpu.region,
        "CPU_EKS_CLUSTER": cpu.eks_name,
        "CPU_KUBECONFIG": str(cpu_kubeconfig),
        "AMP_WORKSPACE_ID": monitoring["workspace_id"],
        "SNS_TOPIC_NAME": topic_name,
        "IAM_ROLE_NAME": role_name,
        "NAMESPACE": namespace,
        "GPU_FAULT_ADOT_IMAGE": adot_image,
        "GPU_FAULT_REQUIRE_CONFIRMED_SNS_SUBSCRIPTION": "true",
        "GPU_FAULT_ENABLE_ADOT": "true",
        "GPU_FAULT_ENABLE_AMP": "true",
    }
    if alert_email:
        environment["GPU_FAULT_ALERT_EMAIL"] = alert_email
    output = runner.run(
        [str(repository_root / "deploy/observability/install-amp-monitoring.sh")],
        env=environment,
        cwd=repository_root,
        mutate=True,
    )
    association = _ensure_pod_identity_association(
        runner,
        cluster=cpu,
        namespace=namespace,
        service_account="gpu-fault-adot",
        role_arn=role["role_arn"],
    )
    return {
        **role,
        "association_id": association["association_id"],
        "association_ownership": association["ownership"],
        "cluster_name": association["cluster_name"],
        "namespace": association["namespace"],
        "service_account": association["service_account"],
        "installer_output": output,
    }


def _upload_wheel_configmap(
    runner: CommandRunner,
    *,
    cpu_kubeconfig: Path,
    namespace: str,
    wheel: Path,
) -> str:
    sha = hashlib.sha256(wheel.read_bytes()).hexdigest()
    name = f"gpu-fault-control-plane-wheel-0100-{sha[:12]}"
    exists = (
        subprocess.run(
            [
                "kubectl",
                "--kubeconfig",
                str(cpu_kubeconfig),
                "-n",
                namespace,
                "get",
                "configmap",
                name,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ).returncode
        == 0
    )
    if not exists:
        runner.run(
            [
                "kubectl",
                "--kubeconfig",
                str(cpu_kubeconfig),
                "-n",
                namespace,
                "create",
                "configmap",
                name,
                f"--from-file={wheel}",
            ],
            mutate=True,
            capture=False,
        )
    return name


def install_aurora_refresh(
    runner: CommandRunner,
    *,
    repository_root: Path,
    cpu: ClusterIdentity,
    cpu_kubeconfig: Path,
    namespace: str,
    site_id: str,
    release_manifest: Path,
    runtime_image: str,
    aurora: Mapping[str, Any],
) -> dict[str, str]:
    manifest = json.loads(release_manifest.read_text(encoding="utf-8"))
    wheel = Path(manifest["wheel"])
    if not wheel.is_absolute():
        wheel = repository_root / wheel
    wheel_configmap = _upload_wheel_configmap(
        runner,
        cpu_kubeconfig=cpu_kubeconfig,
        namespace=namespace,
        wheel=wheel,
    )
    source = (
        repository_root / "deploy/control-plane/regional/aurora-credential-refresh.yaml"
    ).read_text(encoding="utf-8")
    rendered = (
        source.replace("gpu-fault-control-plane-wheel-0100", wheel_configmap)
        .replace(
            "public.ecr.aws/docker/library/python:3.12-slim",
            runtime_image,
        )
        .replace(
            "REPLACE_WITH_AURORA_MASTER_SECRET_ARN",
            aurora["master_secret_arn"],
        )
        .replace("namespace: gpu-fault-system", f"namespace: {namespace}")
    )
    kubectl_apply(runner, cpu_kubeconfig, rendered)
    role_name = safe_name(f"gpu-fault-{site_id}-aurora-refresh", maximum=64)
    statements: list[dict[str, Any]] = [
        {
            "Effect": "Allow",
            "Action": ["secretsmanager:GetSecretValue"],
            "Resource": aurora["master_secret_arn"],
        }
    ]
    if aurora.get("master_secret_kms_key_arn"):
        statements.append(
            {
                "Effect": "Allow",
                "Action": ["kms:Decrypt"],
                "Resource": aurora["master_secret_kms_key_arn"],
            }
        )
    role = _ensure_role(
        runner,
        account_id=cpu.account_id,
        role_name=role_name,
        trust=_pod_identity_trust(),
        policy_name="ReadAuroraManagedMasterSecret",
        policy={"Version": "2012-10-17", "Statement": statements},
        site_id=site_id,
    )
    association = _ensure_pod_identity_association(
        runner,
        cluster=cpu,
        namespace=namespace,
        service_account="gpu-fault-aurora-credential-refresh",
        role_arn=role["role_arn"],
    )
    job = "gpu-fault-aurora-credential-refresh-verify"
    runner.run(
        [
            "kubectl",
            "--kubeconfig",
            str(cpu_kubeconfig),
            "-n",
            namespace,
            "delete",
            "job",
            job,
            "--ignore-not-found",
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
            namespace,
            "create",
            "job",
            job,
            "--from=cronjob/gpu-fault-aurora-credential-refresh",
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
            namespace,
            "wait",
            "--for=condition=complete",
            f"job/{job}",
            "--timeout=420s",
        ],
        mutate=True,
        capture=False,
    )
    return {
        **role,
        "association_id": association["association_id"],
        "association_ownership": association["ownership"],
        "cluster_name": association["cluster_name"],
        "namespace": association["namespace"],
        "service_account": association["service_account"],
        "wheel_configmap": wheel_configmap,
    }


def provision_node_action_keys(
    runner: CommandRunner,
    *,
    repository_root: Path,
    cpu_kubeconfig: Path,
    gpu_kubeconfig: Path,
    namespace: str,
    cluster: ClusterIdentity,
    cluster_id: str,
    fleet_master_file: Path,
) -> dict[str, str]:
    environment = {
        **os.environ,
        "KUBECONFIG": str(gpu_kubeconfig),
        "GPU_FAULT_KUBECTL_CONTEXT": cluster.context,
        "GPU_FAULT_NAMESPACE": namespace,
        "GPU_FAULT_CLUSTER_ID": cluster_id,
        "GPU_FAULT_HYPERPOD_CLUSTER": cluster.hyperpod_name,
        "GPU_FAULT_FLEET_MASTER_FILE": str(fleet_master_file),
        "GPU_FAULT_CONTROL_PLANE_KUBECONFIG": str(cpu_kubeconfig),
        "GPU_FAULT_CONTROL_PLANE_NAMESPACE": namespace,
    }
    runner.run(
        [str(repository_root / "deploy/node/provision-node-action-keys.sh")],
        env=environment,
        cwd=repository_root,
        mutate=True,
        sensitive=True,
        capture=False,
    )
    return {"cluster_id": cluster_id}

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
from uuid import uuid4
from weakref import WeakKeyDictionary

from gpu_fault.admin.bootstrap_common import (
    SITE_TAG_KEY,
    BootstrapError,
    BootstrapState,
    ClusterIdentity,
    CommandRunner,
    ReadOnlyProbeRunner,
    assert_site_tag,
    kubectl_apply,
    safe_name,
    tag_map,
)
from gpu_fault.admin.bootstrap_platform_probes import (
    assert_aurora_refresh_current,
    assert_monitoring_install_current,
    assert_node_action_keys_current,
)
from gpu_fault.admin.grafana import GrafanaSettings, ensure_grafana_dashboards
from gpu_fault.admin.monitoring_subscriptions import (
    ensure_monitoring_subscriptions,
)

SNS_TOPIC_GENERATION_TAG = "gpu-fault:topic-generation"
# One Pod Identity add-on result per deploy run, keyed by the runner the run was
# handed. Three call sites ensure the same add-on (`revalidate_pod_identity_agent`,
# the control-plane role and the load balancer controller) and each one used to
# pay for a describe and an `aws eks wait addon-active`; the add-on cannot change
# between two of those calls, so the first successful read is the evidence for
# the rest of the run. The key is weak so a test's fake runner takes its cache
# with it, and so nothing here keeps a finished run's runner alive.
_POD_IDENTITY_AGENTS: WeakKeyDictionary[
    Any, dict[tuple[str, str, str], dict[str, str]]
] = WeakKeyDictionary()


def _pod_identity_agent_cache(
    runner: CommandRunner,
) -> dict[tuple[str, str, str], dict[str, str]]:
    """The add-on cache belonging to this run, found through probe wrappers.

    `run_parallel` wraps the run's runner in a fresh `ReadOnlyProbeRunner` for
    every probe, so caching against the wrapper would cache nothing that the
    following ensure could reuse. Unwrapping to the runner the deploy created
    makes a healthy read-only probe count as the read for the ensure that
    follows it in the same transaction.
    """

    root: Any = runner
    while isinstance(root, ReadOnlyProbeRunner):
        root = root._delegate
    # `setdefault`, not get-then-insert: `run_parallel` calls the tasks from a
    # thread pool, and two tasks reaching a get-then-insert at the same time would
    # each install their own dict, so one of the two add-on reads would not be
    # deduplicated after all.
    cache: dict[tuple[str, str, str], dict[str, str]] = _POD_IDENTITY_AGENTS.setdefault(
        root, {}
    )
    return cache


def _ensure_service_account(
    runner: CommandRunner,
    *,
    kubeconfig: Path,
    namespace: str,
    name: str,
) -> None:
    try:
        runner.run(
            [
                "kubectl",
                "--kubeconfig",
                str(kubeconfig),
                "-n",
                namespace,
                "get",
                "serviceaccount",
                name,
                "-o",
                "name",
            ]
        )
        return
    except BootstrapError as exc:
        message = str(exc).lower()
        if "notfound" not in message and "not found" not in message:
            raise
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
    cache = _pod_identity_agent_cache(runner)
    # `site_id` is in the key because the ensure asserts the add-on's site tag: a
    # cached result must not answer for a different site's ownership check.
    cache_key = (cluster.eks_name, cluster.region, site_id)
    cached = cache.get(cache_key)
    if cached is not None:
        return dict(cached)
    # One describe answers both questions the ensure has: whether the add-on is
    # there at all, and which site owns it. A separate silent existence probe
    # asked the same thing a second time. Only a not-found error means absent;
    # anything else (AccessDenied, a throttled call) stays fail-closed, because
    # reading it as absent would send the run into `create-addon`, which fails on
    # an add-on that already exists.
    addon: dict[str, Any] | None = None
    try:
        addon = runner.aws_json(
            cluster.region,
            "eks",
            "describe-addon",
            "--cluster-name",
            cluster.eks_name,
            "--addon-name",
            "eks-pod-identity-agent",
        )["addon"]
    except BootstrapError as exc:
        if "notfound" not in str(exc).lower():
            raise
    ownership = "CREATED"
    if addon is not None:
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
        capture=False,
    )
    result = {
        "cluster_name": cluster.eks_name,
        "addon_name": "eks-pod-identity-agent",
        "ownership": ownership,
    }
    cache[cache_key] = dict(result)
    return result


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


def pod_identity_trust() -> dict[str, Any]:
    return _pod_identity_trust()


def _normalized_document(value: object) -> object:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    if isinstance(value, dict):
        return {
            str(key): _normalized_document(item) for key, item in sorted(value.items())
        }
    if isinstance(value, list):
        return [_normalized_document(item) for item in value]
    return value


def _documents_match(left: object, right: object) -> bool:
    return _normalized_document(left) == _normalized_document(right)


def _read_role(runner: CommandRunner, *, role_name: str) -> dict[str, Any] | None:
    """The role document, or None when IAM says the role does not exist."""

    try:
        raw = runner.run(
            ["aws", "iam", "get-role", "--role-name", role_name, "--output", "json"]
        )
    except BootstrapError as exc:
        if "nosuchentity" not in str(exc).lower():
            raise
        return None
    return cast(dict[str, Any], json.loads(raw)["Role"])


def _read_inline_policy(
    runner: CommandRunner, *, role_name: str, policy_name: str
) -> Any:
    """The inline policy document, or None when it is absent.

    Absent means exactly `NoSuchEntity`. On any other error the current document
    is unknown, and writing over it could replace a narrower policy with a wider
    one with nothing in the log to say so.
    """

    try:
        raw = runner.run(
            [
                "aws",
                "iam",
                "get-role-policy",
                "--role-name",
                role_name,
                "--policy-name",
                policy_name,
                "--output",
                "json",
            ]
        )
    except BootstrapError as exc:
        if "nosuchentity" in str(exc).lower():
            return None
        raise BootstrapError(
            f"cannot inspect inline policy {role_name}/{policy_name}: {exc}"
        ) from exc
    return json.loads(raw).get("PolicyDocument")


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
    trust_text = json.dumps(trust, separators=(",", ":"))
    # One read decides both whether the role exists and whether it has drifted;
    # the silent existence probe that used to run first asked IAM for the same
    # document. `NoSuchEntity` is the only answer that means absent: on any other
    # failure the current trust policy is unknown, and treating that as absent
    # would send the run into `create-role` on a role that is already there.
    role = _read_role(runner, role_name=role_name)
    if role is not None:
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
        if not _documents_match(role.get("AssumeRolePolicyDocument"), trust):
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
        existing_policy = None
        if role is not None:
            existing_policy = _read_inline_policy(
                runner, role_name=role_name, policy_name=policy_name
            )
        if not _documents_match(existing_policy, policy):
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


def probe_iam_role(
    runner: CommandRunner,
    **kwargs: Any,
) -> dict[str, str]:
    return _ensure_role(ReadOnlyProbeRunner(runner), **kwargs)


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
        if len(associations) != 1:
            raise BootstrapError(
                f"{cluster.eks_name}/{namespace}/{service_account} has "
                "multiple Pod Identity associations"
            )
        association_id = str(associations[0]["associationId"])
        association = runner.aws_json(
            cluster.region,
            "eks",
            "describe-pod-identity-association",
            "--cluster-name",
            cluster.eks_name,
            "--association-id",
            association_id,
        ).get("association", {})
        if str(association.get("roleArn") or "") != role_arn:
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


def probe_pod_identity_association(
    runner: CommandRunner,
    **kwargs: Any,
) -> dict[str, str]:
    return _ensure_pod_identity_association(ReadOnlyProbeRunner(runner), **kwargs)


def control_plane_policy_document(
    *,
    region: str,
    account_id: str,
    email_sender: str | None = None,
) -> dict[str, Any]:
    """Every AWS permission the regional control plane is allowed to hold.

    This is a value rather than a literal buried in ``ensure_control_plane_role``
    because it is the whole blast-radius argument for the CPU side: SageMaker
    read-only, no node mutation of any kind, and email only from the verified
    sender. Stated as a document it can be checked against the same validator
    the release path applies to the executor role, without an AWS account.
    """

    statements: list[dict[str, Any]] = [
        {
            "Effect": "Allow",
            "Action": [
                "sagemaker:DescribeCluster",
                "sagemaker:ListClusterNodes",
                "sagemaker:DescribeClusterNode",
            ],
            "Resource": (f"arn:aws:sagemaker:{region}:{account_id}:cluster/*"),
        }
    ]
    if email_sender:
        statements.append(
            {
                "Sid": "AdministratorEmail",
                "Effect": "Allow",
                "Action": "ses:SendEmail",
                "Resource": (
                    f"arn:aws:ses:{region}:{account_id}:identity/{email_sender}"
                ),
                "Condition": {
                    "StringEquals": {
                        "ses:FromAddress": email_sender,
                    }
                },
            }
        )
    return {"Version": "2012-10-17", "Statement": statements}


def executor_policy_document(*, hyperpod_arn: str) -> dict[str, Any]:
    """Every AWS permission one cluster's executor is allowed to hold.

    Scoped to that cluster's HyperPod ARN, with reboot as the only mutation and
    no SES: notification stays a control-plane concern so deduplication cannot
    be bypassed from the data plane.
    """

    return {
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
                "Resource": hyperpod_arn,
            }
        ],
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
    role = _ensure_role(
        runner,
        account_id=cpu.account_id,
        role_name=role_name,
        trust=_pod_identity_trust(),
        policy_name="GPUFaultRegionalObserve",
        policy=control_plane_policy_document(
            region=cpu.region,
            account_id=cpu.account_id,
            email_sender=email_sender,
        ),
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
    policy = executor_policy_document(hyperpod_arn=cluster.hyperpod_arn)
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
) -> tuple[str, bool, str]:
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
        tags_by_key = tag_map(tags)
        tagged = assert_site_tag(
            tags,
            site_id=site_id,
            description=f"SNS topic {topic_arn}",
            allow_missing=True,
        )
        generation = str(tags_by_key.get(SNS_TOPIC_GENERATION_TAG) or "")
        if generation and not re.fullmatch(r"[0-9a-f]{32}", generation):
            raise BootstrapError(f"SNS topic {topic_arn} has an invalid generation tag")
        tag_values = []
        if not tagged:
            tag_values.append(f"Key={SITE_TAG_KEY},Value={site_id}")
        if not generation:
            generation = uuid4().hex
            tag_values.append(f"Key={SNS_TOPIC_GENERATION_TAG},Value={generation}")
        if tag_values and not runner.dry_run:
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
                    *tag_values,
                ],
                mutate=True,
                capture=False,
            )
    else:
        generation = uuid4().hex
        topic_arn = runner.aws_text(
            cpu.region,
            "sns",
            "create-topic",
            "--name",
            topic_name,
            "--tags",
            f"Key=gpu-fault:site-id,Value={site_id}",
            f"Key={SNS_TOPIC_GENERATION_TAG},Value={generation}",
            "--query",
            "TopicArn",
            mutate=True,
        )
    return topic_arn, False, generation


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
    attributes = runner.aws_json(
        cpu.region,
        "sqs",
        "get-queue-attributes",
        "--queue-url",
        queue_url,
        "--attribute-names",
        "QueueArn",
        "Policy",
    ).get("Attributes", {})
    queue_arn = str(attributes.get("QueueArn") or "")
    if not queue_arn:
        raise BootstrapError(f"SQS queue {queue_url} has no QueueArn")
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
    try:
        existing_policy = json.loads(str(attributes.get("Policy") or "{}"))
    except json.JSONDecodeError:
        existing_policy = None
    if not _documents_match(existing_policy, queue_policy):
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


def probe_sqs_queue(
    runner: CommandRunner,
    **kwargs: Any,
) -> tuple[str, str, bool]:
    return _ensure_sqs_queue(ReadOnlyProbeRunner(runner), **kwargs)


def ensure_monitoring_resources(
    runner: CommandRunner,
    *,
    state: BootstrapState | None = None,
    cpu: ClusterIdentity,
    site_id: str,
    alert_email: str | None,
) -> dict[str, Any]:
    workspace_id, workspace_reused = _ensure_amp_workspace(
        runner,
        cpu=cpu,
        site_id=site_id,
    )
    topic_arn, topic_reused, topic_generation = _ensure_sns_topic(
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
        email_subscription,
    ) = ensure_monitoring_subscriptions(
        runner,
        state=state,
        cpu=cpu,
        topic_arn=topic_arn,
        topic_generation=topic_generation,
        queue_arn=queue_arn,
        alert_email=alert_email,
    )
    return {
        "workspace_id": workspace_id,
        "workspace_ownership": "REUSED" if workspace_reused else "CREATED",
        "sns_topic_arn": topic_arn,
        "sns_topic_ownership": "REUSED" if topic_reused else "CREATED",
        "sns_topic_generation": topic_generation,
        "sqs_queue_url": queue_url,
        "sqs_queue_arn": queue_arn,
        "sqs_queue_ownership": "REUSED" if queue_reused else "CREATED",
        "queue_subscription_arn": queue_subscription_arn,
        "queue_subscription_ownership": queue_subscription_ownership,
        "email_subscription_arn": (
            email_subscription.get("subscription_arn")
            if email_subscription is not None
            else None
        ),
        "email_subscription_status": (
            email_subscription.get("status") if email_subscription is not None else None
        ),
        "email_subscription_endpoint": (
            email_subscription.get("endpoint")
            if email_subscription is not None
            else None
        ),
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
    probe_only: bool = False,
    grafana: GrafanaSettings | None = None,
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
    if probe_only:
        # The installer is a mutating script, so a probe cannot run it; the probe
        # reads back what it converges instead. The IAM role and the Pod Identity
        # association around this branch prove themselves through the read-only
        # runner, which raises mutation-required on the first write they need.
        assert_monitoring_install_current(
            runner,
            cpu_kubeconfig=cpu_kubeconfig,
            namespace=namespace,
            region=cpu.region,
            adot_image=adot_image,
            workspace_id=str(monitoring["workspace_id"]),
        )
        output = ""
    else:
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
    # Dashboards ride on this task so they are checkpointed with the AMP install
    # they visualise and re-imported when the dashboard assets change; the step
    # itself decides between a soft FAILED record and an operator-input error.
    dashboards = ensure_grafana_dashboards(
        runner,
        settings=grafana,
        cpu=cpu,
        site_id=site_id,
        amp_workspace_id=str(monitoring["workspace_id"]),
        repository_root=repository_root,
        probe_only=probe_only,
    )
    return {
        **role,
        "association_id": association["association_id"],
        "association_ownership": association["ownership"],
        "cluster_name": association["cluster_name"],
        "namespace": association["namespace"],
        "service_account": association["service_account"],
        "installer_output": output,
        "grafana": dashboards,
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
    probe_only: bool = False,
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
    if probe_only:
        assert_aurora_refresh_current(
            runner,
            cpu_kubeconfig=cpu_kubeconfig,
            namespace=namespace,
            wheel_configmap=wheel_configmap,
            runtime_image=runtime_image,
            master_secret_arn=str(aurora["master_secret_arn"]),
        )
    else:
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
    if probe_only:
        # The verify Job is one-shot proof that the CronJob this release rendered
        # can read the master Secret. The probe already confirmed that exact
        # CronJob is live, so re-running the Job would only repeat a proof the
        # checkpoint still holds.
        return {
            **role,
            "association_id": association["association_id"],
            "association_ownership": association["ownership"],
            "cluster_name": association["cluster_name"],
            "namespace": association["namespace"],
            "service_account": association["service_account"],
            "wheel_configmap": wheel_configmap,
        }
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
    probe_only: bool = False,
) -> dict[str, str]:
    if probe_only:
        # The script is mutating and needs the fleet master file, so a probe
        # cannot run it. What it converges is one scoped key per current node, so
        # the probe checks exactly that, without reading any key value.
        assert_node_action_keys_current(
            runner,
            cpu_kubeconfig=cpu_kubeconfig,
            gpu_kubeconfig=gpu_kubeconfig,
            namespace=namespace,
            cluster=cluster,
        )
        return {"cluster_id": cluster_id}
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

from __future__ import annotations

import base64
import hashlib
import json
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence, cast

from gpu_fault.admin.bootstrap import (
    _ensure_kubeconfigs,
    _require_same_scope,
    discover_cluster,
)
from gpu_fault.admin.bootstrap_common import (
    BOOTSTRAP_STATE_VERSION,
    BootstrapError,
    ClusterIdentity,
    CommandRunner,
    safe_name,
    write_yaml,
)
from gpu_fault.admin.bootstrap_site import cluster_alias
from gpu_fault.admin.site import RenderedSite, load_site
from gpu_fault.release_state_snapshot import (
    ReleaseStateSnapshotError,
    hydrate_previous_snapshot,
)

WORKSPACE_PATTERN = re.compile(r"/workspaces/(?P<workspace>ws-[A-Za-z0-9-]+)")


@dataclass(frozen=True)
class LegacySiteRequest:
    cpu_cluster_arn: str
    gpu_cluster_arns: tuple[str, ...]
    repository_root: Path
    state_dir: Path


def _kubectl_json(
    *,
    kubeconfig: Path,
    arguments: Sequence[str],
    context: str | None = None,
) -> dict[str, Any]:
    command = ["kubectl", "--kubeconfig", str(kubeconfig)]
    if context:
        command.extend(["--context", context])
    command.extend(arguments)
    result = subprocess.run(command, text=True, capture_output=True)
    if result.returncode:
        raise BootstrapError(
            f"legacy discovery failed: {' '.join(arguments[:3])}: "
            f"{result.stderr.strip()}"
        )
    return cast(dict[str, Any], json.loads(result.stdout))


def _decode_secret(document: dict[str, Any], key: str) -> str:
    value = (document.get("data") or {}).get(key)
    if not value:
        raise BootstrapError(f"legacy Secret is missing {key}")
    return base64.b64decode(value).decode()


def _write_json_private(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.chmod(0o600)
    temporary.replace(path)


def _write_live_secret(path: Path, value: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.chmod(0o600)
    temporary.replace(path)


def _allowed_namespaces(value: str) -> list[str]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        parsed = value.split(",")
    if not isinstance(parsed, list):
        raise BootstrapError("legacy allowed-namespaces has an invalid format")
    result = sorted({str(item).strip() for item in parsed if str(item).strip()})
    if not result:
        raise BootstrapError("legacy allowed-namespaces is empty")
    return result


def _pod_identity(
    runner: CommandRunner,
    *,
    cluster: ClusterIdentity,
    namespace: str,
    service_account: str,
) -> dict[str, str] | None:
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
    if not associations:
        return None
    association_id = str(associations[0]["associationId"])
    association = runner.aws_json(
        cluster.region,
        "eks",
        "describe-pod-identity-association",
        "--cluster-name",
        cluster.eks_name,
        "--association-id",
        association_id,
    )["association"]
    return {
        "association_id": association_id,
        "association_ownership": "CREATED",
        "cluster_name": cluster.eks_name,
        "namespace": namespace,
        "service_account": service_account,
        "role_arn": str(association["roleArn"]),
        "ownership": "CREATED",
    }


def _executor_role(
    *,
    gpu_kubeconfig: Path,
    cluster: ClusterIdentity,
    namespace: str,
) -> str:
    account = _kubectl_json(
        kubeconfig=gpu_kubeconfig,
        context=cluster.context,
        arguments=[
            "-n",
            namespace,
            "get",
            "serviceaccount",
            "gpu-fault-cluster-executor",
            "-o",
            "json",
        ],
    )
    role_arn = (
        account.get("metadata", {})
        .get("annotations", {})
        .get("eks.amazonaws.com/role-arn")
    )
    if not role_arn:
        raise BootstrapError(
            f"{cluster.context}: executor ServiceAccount has no IAM role"
        )
    return str(role_arn)


def _aurora(
    runner: CommandRunner,
    *,
    region: str,
    master_secret_arn: str,
) -> dict[str, Any]:
    clusters = cast(
        list[dict[str, Any]],
        runner.aws_json(region, "rds", "describe-db-clusters").get(
            "DBClusters",
            [],
        ),
    )
    matches = [
        cluster
        for cluster in clusters
        if (cluster.get("MasterUserSecret") or {}).get("SecretArn") == master_secret_arn
    ]
    if len(matches) != 1:
        raise BootstrapError(
            "Aurora master Secret did not resolve to exactly one DB cluster"
        )
    cluster = matches[0]
    groups = cluster.get("VpcSecurityGroups") or []
    if len(groups) != 1:
        raise BootstrapError("legacy Aurora must have exactly one solution SG")
    return {
        "cluster_id": cluster["DBClusterIdentifier"],
        "cluster_ownership": "CREATED",
        "instance_ids": [
            item["DBInstanceIdentifier"] for item in cluster.get("DBClusterMembers", [])
        ],
        "subnet_group": cluster["DBSubnetGroup"],
        "subnet_group_ownership": "CREATED",
        "security_group": groups[0]["VpcSecurityGroupId"],
        "security_group_ownership": "CREATED",
        "master_secret_arn": master_secret_arn,
        "master_secret_kms_key_arn": str(
            (cluster.get("MasterUserSecret") or {}).get("KmsKeyId") or ""
        ),
    }


def _pki_secret(
    runner: CommandRunner,
    *,
    region: str,
    certificate_arn: str,
) -> str | None:
    secrets = cast(
        list[dict[str, Any]],
        runner.aws_json(
            region,
            "secretsmanager",
            "list-secrets",
            "--filters",
            "Key=name,Values=gpu-fault",
        ).get("SecretList", []),
    )
    matches = []
    for item in secrets:
        name = str(item.get("Name") or "")
        if "nlb-private-pki" not in name:
            continue
        payload = json.loads(
            runner.aws_text(
                region,
                "secretsmanager",
                "get-secret-value",
                "--secret-id",
                name,
                "--query",
                "SecretString",
                sensitive=True,
            )
        )
        if payload.get("certificate_arn") == certificate_arn:
            matches.append(name)
    if len(matches) > 1:
        raise BootstrapError("multiple legacy GPU fault PKI Secrets were found")
    return matches[0] if matches else None


def _monitoring(
    runner: CommandRunner,
    *,
    cpu: ClusterIdentity,
    cpu_kubeconfig: Path,
    namespace: str,
) -> dict[str, Any]:
    config = _kubectl_json(
        kubeconfig=cpu_kubeconfig,
        arguments=[
            "-n",
            namespace,
            "get",
            "configmap",
            "gpu-fault-adot",
            "-o",
            "json",
        ],
    )
    collector = str((config.get("data") or {}).get("collector.yaml") or "")
    match = WORKSPACE_PATTERN.search(collector)
    if match is None:
        raise BootstrapError("cannot discover AMP workspace from ADOT config")
    workspace_id = match.group("workspace")
    workspace = runner.aws_json(
        cpu.region,
        "amp",
        "describe-workspace",
        "--workspace-id",
        workspace_id,
    )["workspace"]
    tags = workspace.get("tags") or {}
    workspace_ownership = (
        "EXTERNAL"
        if any(str(key).startswith("aws:cloudformation:") for key in tags)
        else "CREATED"
    )
    topic_name = f"gpu-fault-control-plane-alerts-{cpu.region}"
    topic_arn = f"arn:aws:sns:{cpu.region}:{cpu.account_id}:{topic_name}"
    runner.aws_json(
        cpu.region,
        "sns",
        "get-topic-attributes",
        "--topic-arn",
        topic_arn,
    )
    return {
        "workspace_id": workspace_id,
        "workspace_ownership": workspace_ownership,
        "sns_topic_arn": topic_arn,
        "sns_topic_ownership": "CREATED",
    }


def _release_state(
    *,
    cpu_kubeconfig: Path,
    namespace: str,
) -> tuple[dict[str, Any], dict[str, str]]:
    state_config = _kubectl_json(
        kubeconfig=cpu_kubeconfig,
        arguments=[
            "-n",
            namespace,
            "get",
            "configmap",
            "gpu-fault-regional-release-state",
            "-o",
            "json",
        ],
    )
    metadata_config = _kubectl_json(
        kubeconfig=cpu_kubeconfig,
        arguments=[
            "-n",
            namespace,
            "get",
            "configmap",
            "gpu-fault-release-metadata",
            "-o",
            "json",
        ],
    )
    state = json.loads((state_config.get("data") or {})["state.json"])
    try:
        state = hydrate_previous_snapshot(
            state,
            lambda name: _kubectl_json(
                kubeconfig=cpu_kubeconfig,
                arguments=[
                    "-n",
                    namespace,
                    "get",
                    "configmap",
                    name,
                    "-o",
                    "json",
                ],
            ),
        )
    except ReleaseStateSnapshotError as exc:
        raise BootstrapError("regional release previous snapshot is invalid") from exc
    metadata = {
        str(key): str(value)
        for key, value in (metadata_config.get("data") or {}).items()
    }
    return state, metadata


def _write_live_secrets(
    *,
    cpu_kubeconfig: Path,
    gpu_kubeconfig: Path,
    gpu_clusters: Sequence[ClusterIdentity],
    namespace: str,
    secure_dir: Path,
) -> tuple[dict[str, dict[str, str]], Path]:
    active = _kubectl_json(
        kubeconfig=cpu_kubeconfig,
        arguments=[
            "-n",
            namespace,
            "get",
            "secret",
            "gpu-fault-control-plane-active",
            "-o",
            "json",
        ],
    )
    fleet_master = secure_dir / "fleet-master"
    _write_live_secret(fleet_master, _decode_secret(active, "node-action-secret"))
    values = {}
    for cluster in gpu_clusters:
        connection = _kubectl_json(
            kubeconfig=gpu_kubeconfig,
            context=cluster.context,
            arguments=[
                "-n",
                namespace,
                "get",
                "secret",
                "gpu-fault-regional-connection",
                "-o",
                "json",
            ],
        )
        cluster_id = _decode_secret(connection, "cluster-id")
        token_file = secure_dir / f"{cluster_id}.token"
        ca_file = secure_dir / f"{cluster_id}.ca.crt"
        _write_live_secret(token_file, _decode_secret(connection, "cluster-token"))
        _write_live_secret(ca_file, _decode_secret(connection, "ca.crt"))
        values[cluster.context] = {
            "cluster_id": cluster_id,
            "token_file": str(token_file),
            "ca_file": str(ca_file),
            "control_plane_url": _decode_secret(
                connection,
                "control-plane-url",
            ),
            "hyperpod_cluster_name": _decode_secret(
                connection,
                "hyperpod-cluster-name",
            ),
            "allowed_namespaces": _decode_secret(
                connection,
                "allowed-namespaces",
            ),
        }
    return values, fleet_master


def discover_legacy_site(
    request: LegacySiteRequest,
    *,
    runner: CommandRunner | None = None,
) -> RenderedSite:
    active_runner = runner or CommandRunner()
    cpu = discover_cluster(
        active_runner,
        cluster_arn=request.cpu_cluster_arn,
        role="cpu",
        context=cluster_alias(request.cpu_cluster_arn, "cpu", 0),
    )
    with ThreadPoolExecutor(max_workers=min(8, len(request.gpu_cluster_arns))) as pool:
        futures = [
            pool.submit(
                discover_cluster,
                active_runner,
                cluster_arn=value,
                role="gpu",
                context=cluster_alias(value, "gpu", index),
            )
            for index, value in enumerate(request.gpu_cluster_arns, 1)
        ]
        gpu_clusters = [future.result() for future in futures]
    _require_same_scope(cpu, gpu_clusters)
    request.state_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    request.state_dir.chmod(0o700)
    cpu_kubeconfig, gpu_kubeconfig = _ensure_kubeconfigs(
        active_runner,
        cpu=cpu,
        gpu_clusters=gpu_clusters,
        state_dir=request.state_dir,
    )
    namespace = "gpu-fault-system"
    release_state, release_metadata = _release_state(
        cpu_kubeconfig=cpu_kubeconfig,
        namespace=namespace,
    )
    secure_dir = request.state_dir / "secure"
    secure_dir.mkdir(mode=0o700, exist_ok=True)
    connections, fleet_master = _write_live_secrets(
        cpu_kubeconfig=cpu_kubeconfig,
        gpu_kubeconfig=gpu_kubeconfig,
        gpu_clusters=gpu_clusters,
        namespace=namespace,
        secure_dir=secure_dir,
    )
    nlb_service = _kubectl_json(
        kubeconfig=cpu_kubeconfig,
        arguments=[
            "-n",
            namespace,
            "get",
            "service",
            "gpu-fault-api-nlb",
            "-o",
            "json",
        ],
    )
    annotations = nlb_service.get("metadata", {}).get("annotations", {})
    annotation = "service.beta.kubernetes.io/"
    nlb_name = str(annotations[f"{annotation}aws-load-balancer-name"])
    subnets = str(annotations[f"{annotation}aws-load-balancer-subnets"]).split(",")
    nlb_security_group = str(
        annotations[f"{annotation}aws-load-balancer-security-groups"]
    )
    certificate_arn = str(annotations[f"{annotation}aws-load-balancer-ssl-cert"])
    aurora_secret = _kubectl_json(
        kubeconfig=cpu_kubeconfig,
        arguments=[
            "-n",
            namespace,
            "get",
            "secret",
            "gpu-fault-aurora",
            "-o",
            "json",
        ],
    )
    master_secret_arn = _decode_secret(
        aurora_secret,
        "master-secret-arn",
    )
    aurora = _aurora(
        active_runner,
        region=cpu.region,
        master_secret_arn=master_secret_arn,
    )
    monitoring = _monitoring(
        active_runner,
        cpu=cpu,
        cpu_kubeconfig=cpu_kubeconfig,
        namespace=namespace,
    )
    site_seed = "|".join([cpu.eks_arn, *(item.eks_arn for item in gpu_clusters)])
    site_id = safe_name(
        f"legacy-{cpu.region}-{hashlib.sha256(site_seed.encode()).hexdigest()[:8]}",
        maximum=48,
    )
    clusters = []
    executor_state = {}
    for cluster in gpu_clusters:
        connection = connections[cluster.context]
        cluster_id = connection["cluster_id"]
        role_arn = _executor_role(
            gpu_kubeconfig=gpu_kubeconfig,
            cluster=cluster,
            namespace=namespace,
        )
        clusters.append(
            {
                "clusterId": cluster_id,
                "context": cluster.context,
                "hyperpodClusterName": connection["hyperpod_cluster_name"],
                "eksClusterArn": cluster.eks_arn,
                "executorIrsaRoleArn": role_arn,
                "allowedNamespaces": _allowed_namespaces(
                    connection["allowed_namespaces"]
                ),
                "controlPlaneUrl": connection["control_plane_url"],
                "tokenFile": connection["token_file"],
                "caFile": connection["ca_file"],
                "fleetMasterFile": str(fleet_master),
            }
        )
        issuer = active_runner.aws_text(
            cluster.region,
            "eks",
            "describe-cluster",
            "--name",
            cluster.eks_name,
            "--query",
            "cluster.identity.oidc.issuer",
        ).removeprefix("https://")
        executor_state[f"executor_role:{cluster_id}"] = {
            "role_arn": role_arn,
            "ownership": "CREATED",
            "oidc_provider_arn": (
                f"arn:aws:iam::{cluster.account_id}:oidc-provider/{issuer}"
            ),
            "oidc_provider_ownership": "EXTERNAL",
            "cluster_name": cluster.eks_name,
        }
    control_role = _pod_identity(
        active_runner,
        cluster=cpu,
        namespace=namespace,
        service_account="gpu-fault-control-plane",
    )
    refresh_role = _pod_identity(
        active_runner,
        cluster=cpu,
        namespace=namespace,
        service_account="gpu-fault-aurora-credential-refresh",
    )
    adot_role = _pod_identity(
        active_runner,
        cluster=cpu,
        namespace=namespace,
        service_account="gpu-fault-adot",
    )
    document = {
        "apiVersion": "gpu-fault.aws/v1alpha1",
        "kind": "RegionalSite",
        "metadata": {"name": site_id},
        "spec": {
            "repositoryRoot": str(request.repository_root),
            "gpuKubeconfig": str(gpu_kubeconfig),
            "awsRegion": cpu.region,
            "namespace": namespace,
            "autoRollback": True,
            "cpu": {
                "kubeconfig": str(cpu_kubeconfig),
                "eksArn": cpu.eks_arn,
                "hyperpodClusterName": cpu.hyperpod_name,
            },
            "release": {
                "manifest": str(request.repository_root / "dist/current-release.json"),
                "agentConfigDigest": release_metadata["required-agent-config-digest"],
            },
            "runtimeProfile": {
                "source": str(
                    request.repository_root
                    / "config/runtime-profile.regional-hyperpod-safe.example.yaml"
                ),
                "version": release_state["runtime_profile_version"],
                "registrationClusterId": release_state[
                    "runtime_profile_registration_cluster_id"
                ],
            },
            "nlb": {
                "name": nlb_name,
                "publicSubnets": subnets,
                "securityGroup": nlb_security_group,
                "certificateArn": certificate_arn,
            },
            "health": {
                "auroraClusterId": aurora["cluster_id"],
                "ampWorkspaceId": monitoring["workspace_id"],
                "ampRuleNamespace": "gpu-fault-control-plane-capacity",
                "snsTopicArn": monitoring["sns_topic_arn"],
                "certificateMinValidityDays": 30,
                "remoteCommandMaxUnclaimedSeconds": 300,
                "requireConfirmedSnsSubscription": True,
            },
            "notifications": {
                "allowEmail": False,
                "acknowledgeExternalAlertChannel": True,
            },
            "clusters": clusters,
        },
    }
    site_file = request.state_dir / "site.yaml"
    write_yaml(site_file, document)
    nlb_description = active_runner.aws_json(
        cpu.region,
        "ec2",
        "describe-security-groups",
        "--group-ids",
        nlb_security_group,
    )["SecurityGroups"][0]
    bootstrap_resources: dict[str, Any] = {
        "nlb_network": {
            "name": nlb_name,
            "public_subnets": subnets,
            "security_group": nlb_security_group,
            "security_group_ownership": "CREATED",
            "vpc_id": nlb_description["VpcId"],
        },
        "pki": {
            "certificate_arn": certificate_arn,
            "certificate_ownership": "CREATED",
        },
        "aurora": aurora,
        "monitoring_resources": monitoring,
        "load_balancer_controller": {"external": True},
        **executor_state,
    }
    addon_exists = (
        subprocess.run(
            [
                "aws",
                "eks",
                "describe-addon",
                "--region",
                cpu.region,
                "--cluster-name",
                cpu.eks_name,
                "--addon-name",
                "eks-pod-identity-agent",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ).returncode
        == 0
    )
    if addon_exists:
        bootstrap_resources["pod_identity_agent"] = {
            "addon_name": "eks-pod-identity-agent",
            "cluster_name": cpu.eks_name,
            "ownership": "EXTERNAL",
        }
    pki_secret = _pki_secret(
        active_runner,
        region=cpu.region,
        certificate_arn=certificate_arn,
    )
    if pki_secret:
        bootstrap_resources["pki"].update(
            {
                "pki_secret_id": pki_secret,
                "pki_secret_ownership": "CREATED",
            }
        )
    if control_role:
        bootstrap_resources["control_plane_role"] = control_role
    if refresh_role:
        bootstrap_resources["aurora_refresh"] = refresh_role
    if adot_role:
        bootstrap_resources["monitoring_install"] = adot_role
    _write_json_private(
        request.state_dir / "bootstrap-state.json",
        {
            "schema_version": BOOTSTRAP_STATE_VERSION,
            "site_id": site_id,
            "phase": "legacy-discovered",
            "resources": bootstrap_resources,
            "completed_tasks": [],
        },
    )
    return load_site(site_file)

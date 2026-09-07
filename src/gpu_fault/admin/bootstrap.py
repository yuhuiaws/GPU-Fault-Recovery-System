from __future__ import annotations

import base64
import hashlib
import ipaddress
import json
import os
import re
import secrets
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping, Sequence, cast
from urllib.parse import quote

from gpu_fault.admin.aurora_capacity import (
    AURORA_ENGINE_VERSION,
    AuroraClusterSpec,
    create_db_cluster_arguments,
)
from gpu_fault.admin.bootstrap_aurora import (
    bootstrap_aurora_capacity,
    ensure_cluster_parameter_group,
    ensure_serverless_instances,
    reconcile_cluster_diagnostics,
    reconcile_existing_capacity,
)
from gpu_fault.admin.bootstrap_common import (
    SITE_TAG_KEY,
    Arn,
    BootstrapError,
    BootstrapRequest,
    BootstrapResult,
    BootstrapState,
    ClusterIdentity,
    CommandRunner,
    assert_site_tag,
    ensure_namespace as _ensure_namespace,
    kubectl_apply as _kubectl_apply,
    safe_name as _safe_name,
    tag_map,
    write_secret as _write_secret,
    write_yaml as _write_yaml,
)
from gpu_fault.admin.bootstrap_dependencies import validate_bootstrap_dependencies
from gpu_fault.admin.bootstrap_site import (
    bind_initial_deploy_target as bootstrap_gpu_scope,
    cluster_alias as _cluster_alias,
    discover_bootstrap_scope,
    discover_subnet_cidrs,
    finalize_bootstrap_site,
    hyperpod_inventory,
    site_identifier as _site_identifier,
    unique_gpu_vpcs,
)
from gpu_fault.admin.bootstrap_tasks import (
    revalidate_pod_identity_agent,
    run_foundation_tasks,
    run_platform_prerequisite_tasks,
)
from gpu_fault.admin.config import AuroraCapacityConfig
from gpu_fault.admin.notifications import NotificationRouting
from gpu_fault.admin.release_repositories import prepare_signed_release

DEFAULT_ADOT_IMAGE_AMD64 = (
    "public.ecr.aws/aws-observability/aws-otel-collector@"
    "sha256:bb72328152c72fb9662056759b275f7cc85e115db12bbb114fbea9f68dc4816c"
)
REGIONAL_PROFILE_SOURCE = "config/runtime-profile.regional-hyperpod-safe.example.yaml"


def _cluster_name_from_eks_arn(value: str) -> str:
    arn = Arn.parse(value)
    if arn.service != "eks":
        raise BootstrapError(f"expected an EKS ARN, got {value}")
    return arn.resource_name


def _find_hyperpod_for_eks(
    runner: CommandRunner,
    *,
    eks_arn: str,
    region: str,
) -> dict[str, Any]:
    matches = [
        value
        for value in hyperpod_inventory(runner, region=region)
        if (value.get("Orchestrator") or {}).get("Eks", {}).get("ClusterArn") == eks_arn
    ]
    if len(matches) != 1:
        raise BootstrapError(
            f"EKS cluster {eks_arn} must belong to exactly one HyperPod cluster; "
            f"found {len(matches)}"
        )
    return matches[0]


def discover_cluster(
    runner: CommandRunner,
    *,
    cluster_arn: str,
    role: str,
    context: str,
) -> ClusterIdentity:
    arn = Arn.parse(cluster_arn)
    if arn.service == "sagemaker":
        hyperpod = runner.aws_json(
            arn.region,
            "sagemaker",
            "describe-cluster",
            "--cluster-name",
            cluster_arn,
        )
    elif arn.service == "eks":
        hyperpod = _find_hyperpod_for_eks(
            runner,
            eks_arn=cluster_arn,
            region=arn.region,
        )
    else:
        raise BootstrapError("cluster ARN must use the eks or sagemaker service")
    hyperpod_arn = str(hyperpod.get("ClusterArn") or "")
    hyperpod_name = str(hyperpod.get("ClusterName") or "")
    eks_arn = str(
        ((hyperpod.get("Orchestrator") or {}).get("Eks") or {}).get(
            "ClusterArn",
            "",
        )
    )
    if not all((hyperpod_arn, hyperpod_name, eks_arn)):
        raise BootstrapError(
            f"HyperPod cluster {cluster_arn} does not expose an EKS orchestrator"
        )
    eks_parsed = Arn.parse(eks_arn)
    eks_name = eks_parsed.resource_name
    eks = (
        runner.aws_json(
            eks_parsed.region,
            "eks",
            "describe-cluster",
            "--name",
            eks_name,
        ).get("cluster")
        or {}
    )
    if eks.get("status") != "ACTIVE":
        raise BootstrapError(f"EKS cluster {eks_name} is not ACTIVE")
    vpc = eks.get("resourcesVpcConfig") or {}
    subnet_ids = tuple(vpc.get("subnetIds") or ())
    subnet_cidrs = discover_subnet_cidrs(
        runner,
        region=eks_parsed.region,
        subnet_ids=subnet_ids,
    )
    node_recovery = str(hyperpod.get("NodeRecovery") or "")
    if role == "gpu" and node_recovery != "None":
        raise BootstrapError(
            f"GPU HyperPod cluster {hyperpod_name} must set NodeRecovery=None"
        )
    return ClusterIdentity(
        input_arn=cluster_arn,
        role=role,
        region=eks_parsed.region,
        account_id=eks_parsed.account,
        hyperpod_arn=hyperpod_arn,
        hyperpod_name=hyperpod_name,
        eks_arn=eks_arn,
        eks_name=eks_name,
        vpc_id=str(vpc.get("vpcId") or ""),
        subnet_ids=subnet_ids,
        node_recovery=node_recovery,
        context=context,
        subnet_cidrs=subnet_cidrs,
    )


def _require_same_scope(
    cpu: ClusterIdentity,
    gpu_clusters: Sequence[ClusterIdentity],
) -> None:
    for cluster in gpu_clusters:
        if cluster.region != cpu.region or cluster.account_id != cpu.account_id:
            raise BootstrapError(
                "CPU and GPU clusters must belong to the same AWS account and Region"
            )


def _ensure_kubeconfigs(
    runner: CommandRunner,
    *,
    cpu: ClusterIdentity,
    gpu_clusters: Sequence[ClusterIdentity],
    state_dir: Path,
) -> tuple[Path, Path]:
    cpu_config = state_dir / "cpu.kubeconfig"
    gpu_config = state_dir / "gpu.kubeconfig"
    runner.run(
        [
            "aws",
            "eks",
            "update-kubeconfig",
            "--region",
            cpu.region,
            "--name",
            cpu.eks_name,
            "--kubeconfig",
            str(cpu_config),
            "--alias",
            cpu.context,
        ],
        mutate=True,
        capture=False,
    )
    for cluster in gpu_clusters:
        runner.run(
            [
                "aws",
                "eks",
                "update-kubeconfig",
                "--region",
                cluster.region,
                "--name",
                cluster.eks_name,
                "--kubeconfig",
                str(gpu_config),
                "--alias",
                cluster.context,
            ],
            mutate=True,
            capture=False,
        )
    if not runner.dry_run:
        cpu_config.chmod(0o600)
        gpu_config.chmod(0o600)
    return cpu_config, gpu_config


def _ensure_base_secrets(
    runner: CommandRunner,
    *,
    cpu_kubeconfig: Path,
    namespace: str,
    secure_dir: Path,
) -> Path:
    secret_name = "gpu-fault-control-plane-active"
    exists = (
        subprocess.run(
            [
                "kubectl",
                "--kubeconfig",
                str(cpu_kubeconfig),
                "-n",
                namespace,
                "get",
                "secret",
                secret_name,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ).returncode
        == 0
    )
    if not exists:
        values = {
            "execution-token": secrets.token_hex(32),
            "processor-replay-secret": secrets.token_hex(32),
            "node-action-secret": secrets.token_hex(32),
        }
        command = [
            "kubectl",
            "--kubeconfig",
            str(cpu_kubeconfig),
            "-n",
            namespace,
            "create",
            "secret",
            "generic",
            secret_name,
        ]
        command.extend(f"--from-literal={key}={value}" for key, value in values.items())
        runner.run(command, mutate=True, sensitive=True, capture=False)
        master = values["node-action-secret"]
    else:
        encoded = runner.run(
            [
                "kubectl",
                "--kubeconfig",
                str(cpu_kubeconfig),
                "-n",
                namespace,
                "get",
                "secret",
                secret_name,
                "-o",
                "jsonpath={.data.node-action-secret}",
            ],
            sensitive=True,
        )
        master = base64.b64decode(encoded).decode()
    master_file = secure_dir / "fleet-master"
    _write_secret(master_file, master)
    return master_file


def _discover_adot_image(
    runner: CommandRunner,
    *,
    cpu_kubeconfig: Path,
) -> str:
    configured = os.getenv("GPU_FAULT_ADOT_IMAGE", "").strip()
    if configured:
        return configured
    document = json.loads(
        runner.run(
            [
                "kubectl",
                "--kubeconfig",
                str(cpu_kubeconfig),
                "get",
                "deployment,daemonset",
                "-A",
                "-o",
                "json",
            ]
        )
    )
    candidates = []
    for item in document.get("items", []):
        containers = (
            item.get("spec", {})
            .get("template", {})
            .get("spec", {})
            .get("containers", [])
        )
        for container in containers:
            image = str(container.get("image") or "")
            if re.search(r"aws-otel-collector|adot", image, re.IGNORECASE):
                candidates.append(image)
    if not candidates:
        nodes = json.loads(
            runner.run(
                [
                    "kubectl",
                    "--kubeconfig",
                    str(cpu_kubeconfig),
                    "get",
                    "nodes",
                    "-o",
                    "json",
                ]
            )
        )
        architectures = {
            str(item.get("status", {}).get("nodeInfo", {}).get("architecture") or "")
            for item in nodes.get("items", [])
        }
        if architectures == {"amd64"}:
            return DEFAULT_ADOT_IMAGE_AMD64
        raise BootstrapError(
            "cannot discover an approved ADOT image and CPU nodes are not "
            "uniformly amd64; set GPU_FAULT_ADOT_IMAGE"
        )
    return sorted(candidates)[0]


def _describe_subnets(
    runner: CommandRunner,
    *,
    region: str,
    subnet_ids: Sequence[str],
) -> list[dict[str, Any]]:
    if not subnet_ids:
        return []
    return cast(
        list[dict[str, Any]],
        runner.aws_json(
            region,
            "ec2",
            "describe-subnets",
            "--subnet-ids",
            *subnet_ids,
        ).get("Subnets", []),
    )


def _route_tables_for_subnet(
    runner: CommandRunner,
    *,
    region: str,
    subnet_id: str,
    vpc_id: str,
) -> list[dict[str, Any]]:
    tables = cast(
        list[dict[str, Any]],
        runner.aws_json(
            region,
            "ec2",
            "describe-route-tables",
            "--filters",
            f"Name=association.subnet-id,Values={subnet_id}",
        ).get("RouteTables", []),
    )
    if tables:
        return tables
    return cast(
        list[dict[str, Any]],
        runner.aws_json(
            region,
            "ec2",
            "describe-route-tables",
            "--filters",
            f"Name=vpc-id,Values={vpc_id}",
            "Name=association.main,Values=true",
        ).get("RouteTables", []),
    )


def _public_subnet(
    runner: CommandRunner,
    *,
    region: str,
    subnet: dict[str, Any],
) -> bool:
    return any(
        route.get("DestinationCidrBlock") == "0.0.0.0/0"
        and str(route.get("GatewayId", "")).startswith("igw-")
        for table in _route_tables_for_subnet(
            runner,
            region=region,
            subnet_id=subnet["SubnetId"],
            vpc_id=subnet["VpcId"],
        )
        for route in table.get("Routes", [])
    )


def _ensure_internet_gateway(
    runner: CommandRunner,
    *,
    region: str,
    vpc_id: str,
    site_id: str,
) -> dict[str, str]:
    gateways = cast(
        list[dict[str, Any]],
        runner.aws_json(
            region,
            "ec2",
            "describe-internet-gateways",
            "--filters",
            f"Name=attachment.vpc-id,Values={vpc_id}",
        ).get("InternetGateways", []),
    )
    if gateways:
        tags = tag_map(gateways[0].get("Tags"))
        return {
            "internet_gateway_id": str(gateways[0]["InternetGatewayId"]),
            "ownership": (
                "CREATED" if tags.get(SITE_TAG_KEY) == site_id else "EXTERNAL"
            ),
            "vpc_id": vpc_id,
        }
    if runner.dry_run:
        return {
            "internet_gateway_id": f"igw-dryrun-{site_id[:8]}",
            "ownership": "CREATED",
            "vpc_id": vpc_id,
        }
    created = runner.aws_json(
        region,
        "ec2",
        "create-internet-gateway",
        "--tag-specifications",
        "ResourceType=internet-gateway,Tags="
        f"[{{Key=gpu-fault:site-id,Value={site_id}}}]",
        mutate=True,
    )
    gateway_id = str(created["InternetGateway"]["InternetGatewayId"])
    runner.run(
        [
            "aws",
            "ec2",
            "attach-internet-gateway",
            "--region",
            region,
            "--internet-gateway-id",
            gateway_id,
            "--vpc-id",
            vpc_id,
        ],
        mutate=True,
        capture=False,
    )
    return {
        "internet_gateway_id": gateway_id,
        "ownership": "CREATED",
        "vpc_id": vpc_id,
    }


def _unused_subnet_cidrs(
    vpc_cidrs: Sequence[str],
    existing_cidrs: Sequence[str],
    *,
    count: int,
) -> list[str]:
    existing = [ipaddress.ip_network(value) for value in existing_cidrs]
    selected = []
    for raw in vpc_cidrs:
        network = ipaddress.ip_network(raw)
        prefix = max(network.prefixlen, 28)
        for candidate in network.subnets(new_prefix=prefix):
            if any(candidate.overlaps(item) for item in existing):
                continue
            selected.append(str(candidate))
            existing.append(candidate)
            if len(selected) == count:
                return selected
    raise BootstrapError("CPU VPC has no free /28 CIDR blocks for public NLB subnets")


def _owned_public_subnets(
    runner: CommandRunner,
    *,
    cluster: ClusterIdentity,
    site_id: str,
    all_subnets: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, str]], dict[str, dict[str, str]]]:
    owned = [
        item
        for item in all_subnets
        if tag_map(item.get("Tags")).get(SITE_TAG_KEY) == site_id
        and tag_map(item.get("Tags")).get("gpu-fault:purpose") == "public-nlb"
        if _public_subnet(runner, region=cluster.region, subnet=item)
    ]
    resources: list[dict[str, str]] = []
    by_az: dict[str, dict[str, str]] = {}
    for subnet in owned:
        route_tables = _route_tables_for_subnet(
            runner,
            region=cluster.region,
            subnet_id=str(subnet["SubnetId"]),
            vpc_id=cluster.vpc_id,
        )
        route_table = next(
            (
                table
                for table in route_tables
                if any(
                    item.get("SubnetId") == subnet["SubnetId"]
                    for item in table.get("Associations", [])
                )
            ),
            None,
        )
        if route_table is None:
            raise BootstrapError(
                f"owned public subnet {subnet['SubnetId']} has no route table"
            )
        association = next(
            item
            for item in route_table.get("Associations", [])
            if item.get("SubnetId") == subnet["SubnetId"]
        )
        route_table_id = str(route_table["RouteTableId"])
        route_owner = tag_map(route_table.get("Tags")).get(SITE_TAG_KEY)
        if route_owner not in {None, site_id}:
            raise BootstrapError(
                f"route table {route_table_id} belongs to site {route_owner!r}"
            )
        if route_owner is None and not runner.dry_run:
            runner.run(
                [
                    "aws",
                    "ec2",
                    "create-tags",
                    "--region",
                    cluster.region,
                    "--resources",
                    route_table_id,
                    "--tags",
                    f"Key={SITE_TAG_KEY},Value={site_id}",
                    "Key=gpu-fault:purpose,Value=public-nlb",
                ],
                mutate=True,
                capture=False,
            )
        resource = {
            "subnet_id": str(subnet["SubnetId"]),
            "availability_zone": str(subnet["AvailabilityZone"]),
            "ownership": "CREATED",
            "vpc_id": cluster.vpc_id,
            "route_table_id": route_table_id,
            "route_table_association_id": str(association["RouteTableAssociationId"]),
        }
        resources.append(resource)
        by_az.setdefault(resource["availability_zone"], resource)
    return resources, by_az


def _ensure_public_subnets(
    runner: CommandRunner,
    *,
    cluster: ClusterIdentity,
    site_id: str,
) -> dict[str, Any]:
    all_subnets = cast(
        list[dict[str, Any]],
        runner.aws_json(
            cluster.region,
            "ec2",
            "describe-subnets",
            "--filters",
            f"Name=vpc-id,Values={cluster.vpc_id}",
        ).get("Subnets", []),
    )
    subnet_resources, public_by_az = _owned_public_subnets(
        runner,
        cluster=cluster,
        site_id=site_id,
        all_subnets=all_subnets,
    )

    eks_subnets = _describe_subnets(
        runner,
        region=cluster.region,
        subnet_ids=cluster.subnet_ids,
    )
    availability_zones = sorted({item["AvailabilityZone"] for item in eks_subnets})
    if len(availability_zones) < 2:
        raise BootstrapError("CPU EKS must span at least two availability zones")
    vpc = runner.aws_json(
        cluster.region,
        "ec2",
        "describe-vpcs",
        "--vpc-ids",
        cluster.vpc_id,
    )["Vpcs"][0]
    vpc_cidrs = [
        item["CidrBlock"]
        for item in vpc.get("CidrBlockAssociationSet", [])
        if (item.get("CidrBlockState") or {}).get("State") == "associated"
    ] or [vpc["CidrBlock"]]
    needed = max(0, 2 - len(public_by_az))
    cidrs = (
        _unused_subnet_cidrs(
            vpc_cidrs,
            [item["CidrBlock"] for item in all_subnets],
            count=needed,
        )
        if needed
        else []
    )
    gateway = _ensure_internet_gateway(
        runner,
        region=cluster.region,
        vpc_id=cluster.vpc_id,
        site_id=site_id,
    )
    used_azs = set(public_by_az)
    target_azs = [az for az in availability_zones if az not in used_azs][:needed]
    for index, (az, cidr) in enumerate(
        zip(target_azs, cidrs, strict=True),
        len(subnet_resources) + 1,
    ):
        if runner.dry_run:
            subnet_id = f"subnet-dryrun-public-{index}"
            route_table_id = f"rtb-dryrun-public-{index}"
            association_id = f"rtbassoc-dryrun-public-{index}"
        else:
            created = runner.aws_json(
                cluster.region,
                "ec2",
                "create-subnet",
                "--vpc-id",
                cluster.vpc_id,
                "--availability-zone",
                az,
                "--cidr-block",
                cidr,
                "--tag-specifications",
                "ResourceType=subnet,Tags="
                f"[{{Key=gpu-fault:site-id,Value={site_id}}},"
                "{Key=gpu-fault:purpose,Value=public-nlb}]",
                mutate=True,
            )
            subnet_id = str(created["Subnet"]["SubnetId"])
            runner.run(
                [
                    "aws",
                    "ec2",
                    "modify-subnet-attribute",
                    "--region",
                    cluster.region,
                    "--subnet-id",
                    subnet_id,
                    "--map-public-ip-on-launch",
                ],
                mutate=True,
                capture=False,
            )
            route_table_id = runner.aws_text(
                cluster.region,
                "ec2",
                "create-route-table",
                "--vpc-id",
                cluster.vpc_id,
                "--tag-specifications",
                "ResourceType=route-table,Tags="
                f"[{{Key=gpu-fault:site-id,Value={site_id}}},"
                "{Key=gpu-fault:purpose,Value=public-nlb}]",
                "--query",
                "RouteTable.RouteTableId",
                mutate=True,
            )
            runner.run(
                [
                    "aws",
                    "ec2",
                    "create-route",
                    "--region",
                    cluster.region,
                    "--route-table-id",
                    route_table_id,
                    "--destination-cidr-block",
                    "0.0.0.0/0",
                    "--gateway-id",
                    gateway["internet_gateway_id"],
                ],
                mutate=True,
                capture=False,
            )
            association_id = runner.aws_text(
                cluster.region,
                "ec2",
                "associate-route-table",
                "--route-table-id",
                route_table_id,
                "--subnet-id",
                subnet_id,
                "--query",
                "AssociationId",
                mutate=True,
            )
        resource = {
            "subnet_id": subnet_id,
            "availability_zone": az,
            "ownership": "CREATED",
            "vpc_id": cluster.vpc_id,
            "route_table_id": route_table_id,
            "route_table_association_id": association_id,
        }
        subnet_resources.append(resource)
        public_by_az[az] = resource
    if len(public_by_az) < 2:
        raise BootstrapError("failed to prepare two public CPU subnets")
    selected = list(public_by_az.values())[:2]
    return {
        "public_subnets": [item["subnet_id"] for item in selected],
        "subnet_resources": subnet_resources,
        "internet_gateway": gateway,
    }


def _ensure_security_group(
    runner: CommandRunner,
    *,
    cluster: ClusterIdentity,
    group_name: str,
    description: str,
    site_id: str,
) -> dict[str, str]:
    groups = cast(
        list[dict[str, Any]],
        runner.aws_json(
            cluster.region,
            "ec2",
            "describe-security-groups",
            "--filters",
            f"Name=vpc-id,Values={cluster.vpc_id}",
            f"Name=group-name,Values={group_name}",
        ).get("SecurityGroups", []),
    )
    if groups:
        assert_site_tag(
            groups[0].get("Tags"),
            site_id=site_id,
            description=f"security group {group_name}",
        )
        return {
            "group_id": str(groups[0]["GroupId"]),
            "ownership": "CREATED",
            "vpc_id": cluster.vpc_id,
        }
    if runner.dry_run:
        return {
            "group_id": (
                "sg-dryrun-" + hashlib.sha256(group_name.encode()).hexdigest()[:8]
            ),
            "ownership": "CREATED",
            "vpc_id": cluster.vpc_id,
        }
    group_id = runner.aws_text(
        cluster.region,
        "ec2",
        "create-security-group",
        "--vpc-id",
        cluster.vpc_id,
        "--group-name",
        group_name,
        "--description",
        description,
        "--query",
        "GroupId",
        mutate=True,
    )
    runner.run(
        [
            "aws",
            "ec2",
            "create-tags",
            "--region",
            cluster.region,
            "--resources",
            group_id,
            "--tags",
            f"Key=gpu-fault:site-id,Value={site_id}",
        ],
        mutate=True,
        capture=False,
    )
    return {
        "group_id": group_id,
        "ownership": "CREATED",
        "vpc_id": cluster.vpc_id,
    }


def _gpu_nat_eips(
    runner: CommandRunner,
    cluster: ClusterIdentity,
) -> list[str]:
    gateways = cast(
        list[dict[str, Any]],
        runner.aws_json(
            cluster.region,
            "ec2",
            "describe-nat-gateways",
            "--filter",
            f"Name=vpc-id,Values={cluster.vpc_id}",
            "Name=state,Values=available",
        ).get("NatGateways", []),
    )
    values = sorted(
        {
            address["PublicIp"]
            for gateway in gateways
            for address in gateway.get("NatGatewayAddresses", [])
            if address.get("PublicIp")
        }
    )
    if not values:
        raise BootstrapError(
            f"GPU cluster {cluster.hyperpod_name} has no available NAT EIP; "
            "the pre-created cluster network must provide egress"
        )
    return values


def _authorize_nlb_sources(
    runner: CommandRunner,
    *,
    region: str,
    security_group: str,
    eips: Sequence[str],
) -> None:
    for eip in eips:
        result = subprocess.run(
            [
                "aws",
                "ec2",
                "authorize-security-group-ingress",
                "--region",
                region,
                "--group-id",
                security_group,
                "--protocol",
                "tcp",
                "--port",
                "443",
                "--cidr",
                f"{eip}/32",
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode and "InvalidPermission.Duplicate" not in result.stderr:
            raise BootstrapError(result.stderr.strip())


def _ensure_nlb_network(
    runner: CommandRunner,
    *,
    cpu: ClusterIdentity,
    gpu_clusters: Sequence[ClusterIdentity],
    site_id: str,
) -> dict[str, Any]:
    public_network = _ensure_public_subnets(
        runner,
        cluster=cpu,
        site_id=site_id,
    )
    nlb_name = _safe_name(f"gpu-fault-{site_id}", maximum=32)
    security_group = _ensure_security_group(
        runner,
        cluster=cpu,
        group_name=nlb_name,
        description="GPU fault regional TLS NLB",
        site_id=site_id,
    )
    eips = sorted(
        {eip for cluster in gpu_clusters for eip in _gpu_nat_eips(runner, cluster)}
    )
    if not runner.dry_run:
        _authorize_nlb_sources(
            runner,
            region=cpu.region,
            security_group=security_group["group_id"],
            eips=eips,
        )
    return {
        "name": nlb_name,
        "ownership": "CREATED",
        "public_subnets": public_network["public_subnets"],
        "subnet_resources": public_network["subnet_resources"],
        "internet_gateway": public_network["internet_gateway"],
        "security_group": security_group["group_id"],
        "security_group_ownership": security_group["ownership"],
        "vpc_id": cpu.vpc_id,
        "gpu_nat_eips": eips,
    }


def _ensure_private_zone(
    runner: CommandRunner,
    *,
    cpu: ClusterIdentity,
    gpu_clusters: Sequence[ClusterIdentity],
    site_id: str,
) -> dict[str, Any]:
    zone_name = f"{site_id}.gpu-fault.internal."
    zones = cast(
        list[dict[str, Any]],
        runner.aws_json(
            cpu.region,
            "route53",
            "list-hosted-zones-by-name",
            "--dns-name",
            zone_name,
        ).get("HostedZones", []),
    )
    zone = next(
        (
            item
            for item in zones
            if item.get("Name") == zone_name
            and (item.get("Config") or {}).get("PrivateZone") is True
        ),
        None,
    )
    if zone is not None:
        zone_id = str(zone["Id"]).rsplit("/", 1)[-1]
        tags = (
            runner.aws_json(
                cpu.region,
                "route53",
                "list-tags-for-resource",
                "--resource-type",
                "hostedzone",
                "--resource-id",
                zone_id,
            )
            .get("ResourceTagSet", {})
            .get("Tags", [])
        )
        tagged = assert_site_tag(
            tags,
            site_id=site_id,
            description=f"Route53 hosted zone {zone_id}",
            allow_missing=True,
        )
        if not tagged and not runner.dry_run:
            runner.run(
                [
                    "aws",
                    "route53",
                    "change-tags-for-resource",
                    "--resource-type",
                    "hostedzone",
                    "--resource-id",
                    zone_id,
                    "--add-tags",
                    f"Key={SITE_TAG_KEY},Value={site_id}",
                ],
                mutate=True,
                capture=False,
            )
        zone_ownership = "CREATED"
    elif runner.dry_run:
        zone_id = "ZDRYRUN" + hashlib.sha256(site_id.encode()).hexdigest()[:8]
        zone_ownership = "CREATED"
    else:
        created = runner.aws_json(
            cpu.region,
            "route53",
            "create-hosted-zone",
            "--name",
            zone_name,
            "--caller-reference",
            f"{site_id}-{int(time.time())}",
            "--hosted-zone-config",
            "PrivateZone=true,Comment=GPU fault regional control plane",
            "--vpc",
            f"VPCRegion={cpu.region},VPCId={cpu.vpc_id}",
            mutate=True,
        )
        zone_id = str(created["HostedZone"]["Id"]).rsplit("/", 1)[-1]
        zone_ownership = "CREATED"
        runner.run(
            [
                "aws",
                "route53",
                "change-tags-for-resource",
                "--resource-type",
                "hostedzone",
                "--resource-id",
                zone_id,
                "--add-tags",
                f"Key=gpu-fault:site-id,Value={site_id}",
            ],
            mutate=True,
            capture=False,
        )
    associated_vpcs: set[tuple[str, str]] = set()
    if zone is not None and not runner.dry_run:
        details = runner.aws_json(
            cpu.region,
            "route53",
            "get-hosted-zone",
            "--id",
            zone_id,
        )
        associated_vpcs = {
            (str(item.get("VPCRegion") or ""), str(item.get("VPCId") or ""))
            for item in details.get("VPCs", [])
        }
    associations = [
        {
            "vpc_id": cpu.vpc_id,
            "vpc_region": cpu.region,
            "ownership": "CREATED",
        }
    ]
    associated_vpcs.add((cpu.region, cpu.vpc_id))
    for region, vpc_id in unique_gpu_vpcs(cpu, gpu_clusters):
        association = (region, vpc_id)
        if association in associated_vpcs:
            associations.append(
                {
                    "vpc_id": vpc_id,
                    "vpc_region": region,
                    "ownership": "CREATED",
                }
            )
            continue
        result = subprocess.run(
            [
                "aws",
                "route53",
                "associate-vpc-with-hosted-zone",
                "--hosted-zone-id",
                zone_id,
                "--vpc",
                f"VPCRegion={region},VPCId={vpc_id}",
                "--comment",
                "GPU fault managed data plane",
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode and not any(
            value in result.stderr for value in ("PriorRequestNotComplete",)
        ):
            raise BootstrapError(result.stderr.strip())
        associations.append(
            {
                "vpc_id": vpc_id,
                "vpc_region": region,
                "ownership": "CREATED",
            }
        )
        associated_vpcs.add(association)
    return {
        "zone_name": zone_name.rstrip("."),
        "hosted_zone_id": zone_id,
        "zone_ownership": zone_ownership,
        "vpc_associations": associations,
        "hostname": f"api.{zone_name}".rstrip("."),
    }


def _generate_pki(
    runner: CommandRunner,
    *,
    directory: Path,
    hostname: str,
) -> tuple[Path, Path, Path, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    directory.chmod(0o700)
    ca_key = directory / "ca.key"
    ca_cert = directory / "ca.crt"
    server_key = directory / "server.key"
    server_cert = directory / "server.crt"
    if all(path.is_file() for path in (ca_key, ca_cert, server_key, server_cert)):
        return ca_key, ca_cert, server_key, server_cert
    extension = directory / "server.ext"
    runner.run(
        ["openssl", "genrsa", "-out", str(ca_key), "4096"],
        mutate=True,
        capture=False,
        sensitive=True,
    )
    runner.run(
        [
            "openssl",
            "req",
            "-x509",
            "-new",
            "-sha256",
            "-key",
            str(ca_key),
            "-days",
            "3650",
            "-subj",
            "/O=GPU Fault Regional Control Plane/CN=GPU Fault Private Root CA",
            "-addext",
            "basicConstraints=critical,CA:TRUE,pathlen:0",
            "-addext",
            "keyUsage=critical,keyCertSign,cRLSign",
            "-out",
            str(ca_cert),
        ],
        mutate=True,
        capture=False,
        sensitive=True,
    )
    runner.run(
        ["openssl", "genrsa", "-out", str(server_key), "2048"],
        mutate=True,
        capture=False,
        sensitive=True,
    )
    csr = directory / "server.csr"
    runner.run(
        [
            "openssl",
            "req",
            "-new",
            "-sha256",
            "-key",
            str(server_key),
            "-subj",
            "/O=GPU Fault Regional Control Plane/CN=gpu-fault-regional",
            "-out",
            str(csr),
        ],
        mutate=True,
        capture=False,
        sensitive=True,
    )
    extension.write_text(
        "basicConstraints=critical,CA:FALSE\n"
        "keyUsage=critical,digitalSignature,keyEncipherment\n"
        "extendedKeyUsage=serverAuth\n"
        f"subjectAltName=DNS:{hostname}\n",
        encoding="utf-8",
    )
    runner.run(
        [
            "openssl",
            "x509",
            "-req",
            "-sha256",
            "-in",
            str(csr),
            "-CA",
            str(ca_cert),
            "-CAkey",
            str(ca_key),
            "-CAcreateserial",
            "-days",
            "825",
            "-extfile",
            str(extension),
            "-out",
            str(server_cert),
        ],
        mutate=True,
        capture=False,
        sensitive=True,
    )
    for path in (ca_key, server_key):
        if path.exists():
            path.chmod(0o600)
    return ca_key, ca_cert, server_key, server_cert


def _ensure_pki(
    runner: CommandRunner,
    *,
    cpu: ClusterIdentity,
    gpu_clusters: Sequence[ClusterIdentity],
    state_dir: Path,
    site_id: str,
) -> dict[str, Any]:
    dns = _ensure_private_zone(
        runner,
        cpu=cpu,
        gpu_clusters=gpu_clusters,
        site_id=site_id,
    )
    secret_name = f"gpu-fault/{site_id}/regional-pki"
    existing = (
        subprocess.run(
            [
                "aws",
                "secretsmanager",
                "describe-secret",
                "--region",
                cpu.region,
                "--secret-id",
                secret_name,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ).returncode
        == 0
    )
    ca_output = state_dir / "control-plane-ca.crt"
    if existing:
        secret_description = runner.aws_json(
            cpu.region,
            "secretsmanager",
            "describe-secret",
            "--secret-id",
            secret_name,
        )
        secret = json.loads(
            runner.aws_text(
                cpu.region,
                "secretsmanager",
                "get-secret-value",
                "--secret-id",
                secret_name,
                "--query",
                "SecretString",
                sensitive=True,
            )
        )
        if secret.get("hostname") != dns["hostname"]:
            raise BootstrapError("existing PKI secret belongs to another hostname")
        tagged = assert_site_tag(
            secret_description.get("Tags"),
            site_id=site_id,
            description=f"Secrets Manager secret {secret_name}",
            allow_missing=True,
        )
        if not tagged and not runner.dry_run:
            runner.run(
                [
                    "aws",
                    "secretsmanager",
                    "tag-resource",
                    "--region",
                    cpu.region,
                    "--secret-id",
                    secret_name,
                    "--tags",
                    f"Key={SITE_TAG_KEY},Value={site_id}",
                ],
                mutate=True,
                capture=False,
            )
        certificate_arn = str(secret["certificate_arn"])
        certificate_tags = runner.aws_json(
            cpu.region,
            "acm",
            "list-tags-for-certificate",
            "--certificate-arn",
            certificate_arn,
        ).get("Tags", [])
        certificate_tagged = assert_site_tag(
            certificate_tags,
            site_id=site_id,
            description=f"ACM certificate {certificate_arn}",
            allow_missing=True,
        )
        if not certificate_tagged and not runner.dry_run:
            runner.run(
                [
                    "aws",
                    "acm",
                    "add-tags-to-certificate",
                    "--region",
                    cpu.region,
                    "--certificate-arn",
                    certificate_arn,
                    "--tags",
                    f"Key={SITE_TAG_KEY},Value={site_id}",
                ],
                mutate=True,
                capture=False,
            )
        ca_output.write_text(secret["ca_certificate"], encoding="utf-8")
        ca_output.chmod(0o644)
        return {
            **dns,
            "certificate_arn": certificate_arn,
            "certificate_ownership": "CREATED",
            "ca_file": str(ca_output),
            "pki_secret_id": secret_name,
            "pki_secret_ownership": "CREATED",
        }
    if runner.dry_run:
        ca_output.write_text("DRY-RUN CA\n", encoding="utf-8")
        return {
            **dns,
            "certificate_arn": (
                f"arn:aws:acm:{cpu.region}:{cpu.account_id}:certificate/dry-run"
            ),
            "certificate_ownership": "CREATED",
            "ca_file": str(ca_output),
            "pki_secret_id": secret_name,
            "pki_secret_ownership": "CREATED",
        }
    ca_key, ca_cert, server_key, server_cert = _generate_pki(
        runner,
        directory=state_dir / "pki",
        hostname=dns["hostname"],
    )
    certificate_arn = runner.aws_text(
        cpu.region,
        "acm",
        "import-certificate",
        "--certificate",
        f"fileb://{server_cert}",
        "--private-key",
        f"fileb://{server_key}",
        "--certificate-chain",
        f"fileb://{ca_cert}",
        "--tags",
        f"Key=gpu-fault:site-id,Value={site_id}",
        "--query",
        "CertificateArn",
        mutate=True,
        sensitive=True,
    )
    ca_output.write_bytes(ca_cert.read_bytes())
    ca_output.chmod(0o644)
    payload = {
        "ca_certificate": ca_cert.read_text(encoding="utf-8"),
        "ca_private_key": ca_key.read_text(encoding="utf-8"),
        "server_certificate": server_cert.read_text(encoding="utf-8"),
        "server_private_key": server_key.read_text(encoding="utf-8"),
        "hostname": dns["hostname"],
        "certificate_arn": certificate_arn,
    }
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as handle:
        json.dump(payload, handle)
        secret_file = Path(handle.name)
    secret_file.chmod(0o600)
    try:
        runner.run(
            [
                "aws",
                "secretsmanager",
                "create-secret",
                "--region",
                cpu.region,
                "--name",
                secret_name,
                "--description",
                "GPU fault regional private PKI",
                "--tags",
                f"Key=gpu-fault:site-id,Value={site_id}",
                "--secret-string",
                f"file://{secret_file}",
            ],
            mutate=True,
            sensitive=True,
            capture=False,
        )
    finally:
        secret_file.unlink(missing_ok=True)
    return {
        **dns,
        "certificate_arn": certificate_arn,
        "certificate_ownership": "CREATED",
        "ca_file": str(ca_output),
        "pki_secret_id": secret_name,
        "pki_secret_ownership": "CREATED",
    }


def _private_subnets(
    runner: CommandRunner,
    cluster: ClusterIdentity,
) -> list[dict[str, Any]]:
    all_subnets = _describe_subnets(
        runner,
        region=cluster.region,
        subnet_ids=cluster.subnet_ids,
    )
    private = [
        item
        for item in all_subnets
        if not _public_subnet(runner, region=cluster.region, subnet=item)
    ]
    selected: dict[str, dict[str, Any]] = {}
    for item in private:
        selected.setdefault(item["AvailabilityZone"], item)
    if len(selected) < 2:
        raise BootstrapError("CPU EKS requires private subnets in at least two AZs")
    return list(selected.values())[:2]


def _ensure_rds_site_tag(
    runner: CommandRunner,
    *,
    region: str,
    resource_arn: str,
    tags: object,
    site_id: str,
    description: str,
) -> None:
    tagged = assert_site_tag(
        tags,
        site_id=site_id,
        description=description,
        allow_missing=True,
    )
    if tagged or runner.dry_run:
        return
    runner.run(
        [
            "aws",
            "rds",
            "add-tags-to-resource",
            "--region",
            region,
            "--resource-name",
            resource_arn,
            "--tags",
            f"Key={SITE_TAG_KEY},Value={site_id}",
        ],
        mutate=True,
        capture=False,
    )


def _cpu_node_security_groups(
    runner: CommandRunner,
    *,
    cpu: ClusterIdentity,
    cpu_kubeconfig: Path,
) -> list[str]:
    document = json.loads(
        runner.run(
            [
                "kubectl",
                "--kubeconfig",
                str(cpu_kubeconfig),
                "get",
                "nodes",
                "-l",
                f"sagemaker.amazonaws.com/cluster-name={cpu.hyperpod_name}",
                "-o",
                "json",
            ]
        )
    )
    ips = sorted(
        {
            address["address"]
            for item in document.get("items", [])
            for address in item.get("status", {}).get("addresses", [])
            if address.get("type") == "InternalIP"
        }
    )
    if not ips:
        raise BootstrapError("CPU HyperPod has no node InternalIP")
    interfaces = cast(
        list[dict[str, Any]],
        runner.aws_json(
            cpu.region,
            "ec2",
            "describe-network-interfaces",
            "--filters",
            "Name=addresses.private-ip-address,Values=" + ",".join(ips),
        ).get("NetworkInterfaces", []),
    )
    groups = sorted(
        {
            group["GroupId"]
            for interface in interfaces
            for group in interface.get("Groups", [])
        }
    )
    if not groups:
        raise BootstrapError("cannot discover CPU node Security Groups")
    return groups


def _ensure_aurora(
    runner: CommandRunner,
    *,
    cpu: ClusterIdentity,
    cpu_kubeconfig: Path,
    namespace: str,
    site_id: str,
    capacity: AuroraCapacityConfig,
) -> dict[str, Any]:
    cluster_id = _safe_name(f"gpu-fault-{site_id}-aurora", maximum=63)
    subnets = _private_subnets(runner, cpu)
    subnet_ids = [item["SubnetId"] for item in subnets]
    availability_zones = [item["AvailabilityZone"] for item in subnets]
    subnet_group = cluster_id
    subnet_group_exists = (
        subprocess.run(
            [
                "aws",
                "rds",
                "describe-db-subnet-groups",
                "--region",
                cpu.region,
                "--db-subnet-group-name",
                subnet_group,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ).returncode
        == 0
    )
    if subnet_group_exists:
        subnet_group_details = runner.aws_json(
            cpu.region,
            "rds",
            "describe-db-subnet-groups",
            "--db-subnet-group-name",
            subnet_group,
        )["DBSubnetGroups"][0]
        _ensure_rds_site_tag(
            runner,
            region=cpu.region,
            resource_arn=str(subnet_group_details["DBSubnetGroupArn"]),
            tags=runner.aws_json(
                cpu.region,
                "rds",
                "list-tags-for-resource",
                "--resource-name",
                str(subnet_group_details["DBSubnetGroupArn"]),
            ).get("TagList"),
            site_id=site_id,
            description=f"RDS subnet group {subnet_group}",
        )
        runner.run(
            [
                "aws",
                "rds",
                "modify-db-subnet-group",
                "--region",
                cpu.region,
                "--db-subnet-group-name",
                subnet_group,
                "--subnet-ids",
                *subnet_ids,
            ],
            mutate=True,
            capture=False,
        )
    else:
        runner.run(
            [
                "aws",
                "rds",
                "create-db-subnet-group",
                "--region",
                cpu.region,
                "--db-subnet-group-name",
                subnet_group,
                "--db-subnet-group-description",
                "GPU fault regional Aurora",
                "--subnet-ids",
                *subnet_ids,
                "--tags",
                f"Key=gpu-fault:site-id,Value={site_id}",
            ],
            mutate=True,
            capture=False,
        )
    security_group = _ensure_security_group(
        runner,
        cluster=cpu,
        group_name=cluster_id,
        description="GPU fault regional Aurora PostgreSQL",
        site_id=site_id,
    )
    for source_group in _cpu_node_security_groups(
        runner,
        cpu=cpu,
        cpu_kubeconfig=cpu_kubeconfig,
    ):
        result = subprocess.run(
            [
                "aws",
                "ec2",
                "authorize-security-group-ingress",
                "--region",
                cpu.region,
                "--group-id",
                security_group["group_id"],
                "--protocol",
                "tcp",
                "--port",
                "5432",
                "--source-group",
                source_group,
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode and "InvalidPermission.Duplicate" not in result.stderr:
            raise BootstrapError(result.stderr.strip())
    cluster_exists = (
        subprocess.run(
            [
                "aws",
                "rds",
                "describe-db-clusters",
                "--region",
                cpu.region,
                "--db-cluster-identifier",
                cluster_id,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ).returncode
        == 0
    )
    if cluster_exists:
        existing_cluster = runner.aws_json(
            cpu.region,
            "rds",
            "describe-db-clusters",
            "--db-cluster-identifier",
            cluster_id,
        )["DBClusters"][0]
        _ensure_rds_site_tag(
            runner,
            region=cpu.region,
            resource_arn=str(existing_cluster["DBClusterArn"]),
            tags=runner.aws_json(
                cpu.region,
                "rds",
                "list-tags-for-resource",
                "--resource-name",
                str(existing_cluster["DBClusterArn"]),
            ).get("TagList"),
            site_id=site_id,
            description=f"Aurora cluster {cluster_id}",
        )
        # Diagnostics (lock-wait and slow-statement logging, pg_stat_statements,
        # CloudWatch log export) are reconciled on every deploy like capacity
        # is, so a cluster created before they existed gets them on its next
        # deploy without an operator step.
        parameter_group = ensure_cluster_parameter_group(
            runner,
            aws_region=cpu.region,
            cluster_id=cluster_id,
            engine_version=str(
                existing_cluster.get("EngineVersion") or AURORA_ENGINE_VERSION
            ),
            safe_name=_safe_name,
        )
        reconcile_cluster_diagnostics(
            runner,
            aws_region=cpu.region,
            cluster_id=cluster_id,
            cluster=existing_cluster,
            parameter_group=parameter_group,
        )
    if not cluster_exists:
        parameter_group = ensure_cluster_parameter_group(
            runner,
            aws_region=cpu.region,
            cluster_id=cluster_id,
            engine_version=AURORA_ENGINE_VERSION,
            safe_name=_safe_name,
        )
        # The argument list is owned by ``aurora_capacity`` -- the one writer of
        # the ACU window, shared with the legacy deploy script's ``create``.
        runner.aws_json(
            cpu.region,
            *create_db_cluster_arguments(
                AuroraClusterSpec(
                    cluster_id=cluster_id,
                    subnet_group=subnet_group,
                    security_group_ids=(security_group["group_id"],),
                    parameter_group=parameter_group,
                    capacity=capacity,
                    site_id=site_id,
                )
            ),
            mutate=True,
        )
    instance_ids = ensure_serverless_instances(
        runner,
        aws_region=cpu.region,
        cluster_id=cluster_id,
        availability_zones=availability_zones,
        safe_name=_safe_name,
    )
    if cluster_exists:
        # After the instances: the shared reconciler proves the window on both
        # members, so a resumed bootstrap that had created the cluster but not
        # its instances must not reach it first.
        reconcile_existing_capacity(
            runner,
            aws_region=cpu.region,
            cluster_id=cluster_id,
            cluster=existing_cluster,
            capacity=capacity,
        )
    if runner.dry_run:
        return {
            "cluster_id": cluster_id,
            "cluster_ownership": "CREATED",
            "instance_ids": [
                *instance_ids,
            ],
            "subnet_group": subnet_group,
            "subnet_group_ownership": "CREATED",
            "security_group": security_group["group_id"],
            "security_group_ownership": security_group["ownership"],
            "parameter_group": parameter_group,
            "parameter_group_ownership": "CREATED",
            "master_secret_arn": "arn:aws:secretsmanager:dry-run",
            "master_secret_kms_key_arn": "",
        }
    database = runner.aws_json(
        cpu.region,
        "rds",
        "describe-db-clusters",
        "--db-cluster-identifier",
        cluster_id,
    )["DBClusters"][0]
    secret_arn = database["MasterUserSecret"]["SecretArn"]
    secret_value = json.loads(
        runner.aws_text(
            cpu.region,
            "secretsmanager",
            "get-secret-value",
            "--secret-id",
            secret_arn,
            "--query",
            "SecretString",
            sensitive=True,
        )
    )
    url = (
        "postgresql://"
        + quote(secret_value["username"], safe="")
        + ":"
        + quote(secret_value["password"], safe="")
        + "@"
        + database["Endpoint"]
        + ":5432/gpu_fault?sslmode=require"
    )
    manifest = runner.run(
        [
            "kubectl",
            "--kubeconfig",
            str(cpu_kubeconfig),
            "-n",
            namespace,
            "create",
            "secret",
            "generic",
            "gpu-fault-aurora",
            f"--from-literal=postgres-url={url}",
            f"--from-literal=master-secret-arn={secret_arn}",
            "--dry-run=client",
            "-o",
            "yaml",
        ],
        sensitive=True,
    )
    _kubectl_apply(runner, cpu_kubeconfig, manifest)
    return {
        "cluster_id": cluster_id,
        "cluster_ownership": "CREATED",
        "instance_ids": [
            *instance_ids,
        ],
        "subnet_group": subnet_group,
        "subnet_group_ownership": "CREATED",
        "security_group": security_group["group_id"],
        "security_group_ownership": security_group["ownership"],
        # Created by name (or adopted under it) in ensure_cluster_parameter_group;
        # registered so uninstall deletes it after the cluster.
        "parameter_group": parameter_group,
        "parameter_group_ownership": "CREATED",
        "master_secret_arn": secret_arn,
        "master_secret_kms_key_arn": str(
            database["MasterUserSecret"].get("KmsKeyId") or ""
        ),
    }


def _initial_secure_files(
    *,
    state_dir: Path,
    gpu_clusters: Sequence[ClusterIdentity],
) -> tuple[dict[str, Path], Path]:
    secure = state_dir / "secure"
    secure.mkdir(parents=True, exist_ok=True)
    secure.chmod(0o700)
    tokens = {}
    for cluster in gpu_clusters:
        cluster_id = _safe_name(cluster.hyperpod_name)
        path = secure / f"{cluster_id}.token"
        _write_secret(path, secrets.token_hex(32))
        tokens[cluster_id] = path
    return tokens, secure


def _site_document(
    *,
    request: BootstrapRequest,
    site_id: str,
    cpu: ClusterIdentity,
    gpu_clusters: Sequence[ClusterIdentity],
    cpu_kubeconfig: Path,
    gpu_kubeconfig: Path,
    release: Mapping[str, Any],
    nlb: Mapping[str, Any],
    pki: Mapping[str, Any],
    aurora: Mapping[str, Any],
    monitoring: Mapping[str, Any],
    executor_roles: Mapping[str, str],
    token_files: Mapping[str, Path],
    fleet_master_file: Path,
    adot_image: str,
    admin_email: str,
    routing: NotificationRouting,
) -> dict[str, Any]:
    cluster_documents = []
    for cluster in gpu_clusters:
        cluster_id = _safe_name(cluster.hyperpod_name)
        cluster_documents.append(
            {
                "clusterId": cluster_id,
                "context": cluster.context,
                "hyperpodClusterName": cluster.hyperpod_name,
                "eksClusterArn": cluster.eks_arn,
                "executorIrsaRoleArn": executor_roles[cluster_id],
                "allowedNamespaces": ["gpu-fault-system", "training"],
                "agentEndpointAllowedCidrs": list(cluster.subnet_cidrs),
                "controlPlaneUrl": f"https://{pki['hostname']}",
                "tokenFile": str(token_files[cluster_id]),
                "caFile": pki["ca_file"],
                "fleetMasterFile": str(fleet_master_file),
            }
        )
    first_cluster = cluster_documents[0]["clusterId"]
    profile_source = str(request.repository_root / REGIONAL_PROFILE_SOURCE)
    return {
        "apiVersion": "gpu-fault.aws/v1alpha1",
        "kind": "RegionalSite",
        "metadata": {"name": site_id},
        "spec": {
            "repositoryRoot": str(request.repository_root),
            "gpuKubeconfig": str(gpu_kubeconfig),
            "awsRegion": cpu.region,
            "namespace": "gpu-fault-system",
            "autoRollback": True,
            "cpu": {
                "kubeconfig": str(cpu_kubeconfig),
                "eksArn": cpu.eks_arn,
                "hyperpodClusterName": cpu.hyperpod_name,
            },
            "release": {
                "manifest": release["manifest"],
                "agentConfigDigest": release["agent_config_digest"],
                # 0 = auto: follow the fleet-size cap (4/8/16/32) after the
                # single-node canary wave.
                "upgradeMaxUnavailable": 0,
                "rollbackMaxUnavailable": 2,
            },
            "runtimeProfile": {
                "source": profile_source,
                "templateSource": profile_source,
                "version": "hyperpod-v1",
                "registrationClusterId": first_cluster,
            },
            "nlb": {
                "name": nlb["name"],
                "publicSubnets": list(nlb["public_subnets"]),
                "securityGroup": nlb["security_group"],
                "certificateArn": pki["certificate_arn"],
            },
            "dns": {
                "hostedZoneId": pki["hosted_zone_id"],
                "hostname": pki["hostname"],
            },
            "images": {
                "runtime": release["images"]["runtime"],
                "nodeInstaller": release["images"]["node_installer"],
                "dcgmExporter": release["images"]["dcgm_exporter"],
                "adot": release["images"].get("adot") or adot_image,
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
                "allowEmail": True,
                "acknowledgeExternalAlertChannel": False,
                "adminEmail": admin_email,
                "emailSender": routing.sender,
                "emailRecipients": list(routing.recipients),
                "emailSubjectPrefix": routing.subject_prefix,
            },
            "clusters": cluster_documents,
        },
    }


def bootstrap_from_arns(
    request: BootstrapRequest,
    *,
    runner: CommandRunner | None = None,
) -> BootstrapResult:
    from gpu_fault.admin.bootstrap_services import (
        _ensure_pod_identity_agent,
    )
    from gpu_fault.admin.notification_bootstrap import notification_routing

    validate_bootstrap_dependencies()
    active_runner = runner or CommandRunner(dry_run=request.dry_run)
    existing_site, cpu, gpu_clusters = discover_bootstrap_scope(
        request=request,
        runner=active_runner,
        discover=discover_cluster,
        alias=_cluster_alias,
    )
    _require_same_scope(cpu, gpu_clusters)
    site_id = _site_identifier(cpu, gpu_clusters)
    request.state_dir.mkdir(parents=True, exist_ok=True)
    request.state_dir.chmod(0o700)
    state = BootstrapState(request.state_dir / "bootstrap-state.json", site_id=site_id)
    managed_gpu_clusters = bootstrap_gpu_scope(state, existing_site, cpu, gpu_clusters)
    state.phase("discovered")
    release = prepare_signed_release(
        active_runner,
        request=request,
        cpu=cpu,
        gpu_clusters=tuple(managed_gpu_clusters),
        site_id=site_id,
        state=state,
    )
    admin_email, routing = notification_routing(active_runner, cpu, request, state)
    cpu_kubeconfig, gpu_kubeconfig = _ensure_kubeconfigs(
        active_runner,
        cpu=cpu,
        gpu_clusters=managed_gpu_clusters,
        state_dir=request.state_dir,
    )
    namespace = "gpu-fault-system"
    _ensure_namespace(
        active_runner,
        kubeconfig=cpu_kubeconfig,
        namespace=namespace,
    )
    for cluster in managed_gpu_clusters:
        _ensure_namespace(
            active_runner,
            kubeconfig=gpu_kubeconfig,
            namespace=namespace,
            context=cluster.context,
        )
    token_files, secure_dir = _initial_secure_files(
        state_dir=request.state_dir,
        gpu_clusters=managed_gpu_clusters,
    )
    fleet_master_file = _ensure_base_secrets(
        active_runner,
        cpu_kubeconfig=cpu_kubeconfig,
        namespace=namespace,
        secure_dir=secure_dir,
    )
    # Completed exclusive resources are re-probed without mutation.
    # A healthy probe reuses its checkpoint; detected drift enters ensure.
    # Probe failures other than mutation-required drift remain fail-closed.
    revalidate_pod_identity_agent(
        active_runner,
        state,
        cpu,
        site_id,
        _ensure_pod_identity_agent,
    )
    adot_image = str(release["images"]["adot"])

    first_phase = run_foundation_tasks(
        runner=active_runner,
        state=state,
        cpu=cpu,
        gpu_clusters=managed_gpu_clusters,
        cpu_kubeconfig=cpu_kubeconfig,
        namespace=namespace,
        site_id=site_id,
        state_dir=request.state_dir,
        admin_email=admin_email,
        routing=routing,
        aurora_capacity=bootstrap_aurora_capacity(request.state_dir),
        ensure_nlb_network=_ensure_nlb_network,
        ensure_pki=_ensure_pki,
        ensure_aurora=_ensure_aurora,
    )
    state.phase("aws-infrastructure-ready")
    executor_roles = {
        name.removeprefix("executor_role:"): value["role_arn"]
        for name, value in first_phase.items()
        if name.startswith("executor_role:")
    }
    aurora = cast(dict[str, Any], first_phase["aurora"])
    monitoring = cast(dict[str, Any], first_phase["monitoring_resources"])

    run_platform_prerequisite_tasks(
        runner=active_runner,
        state=state,
        repository_root=request.repository_root,
        cpu=cpu,
        gpu_clusters=managed_gpu_clusters,
        cpu_kubeconfig=cpu_kubeconfig,
        gpu_kubeconfig=gpu_kubeconfig,
        namespace=namespace,
        site_id=site_id,
        monitoring=monitoring,
        adot_image=adot_image,
        alert_email=request.alert_email,
        release_manifest=Path(release["manifest"]),
        runtime_image=str(release["images"]["runtime"]),
        aurora=aurora,
        fleet_master_file=fleet_master_file,
    )
    state.phase("platform-prerequisites-ready")
    site_file = request.state_dir / "site.yaml"
    generated_site = _site_document(
        request=request,
        site_id=site_id,
        cpu=cpu,
        gpu_clusters=managed_gpu_clusters,
        cpu_kubeconfig=cpu_kubeconfig,
        gpu_kubeconfig=gpu_kubeconfig,
        release=release,
        nlb=cast(dict[str, Any], first_phase["nlb_network"]),
        pki=cast(dict[str, Any], first_phase["pki"]),
        aurora=aurora,
        monitoring=monitoring,
        executor_roles=executor_roles,
        token_files=token_files,
        fleet_master_file=fleet_master_file,
        adot_image=adot_image,
        admin_email=admin_email,
        routing=routing,
    )
    return finalize_bootstrap_site(
        site_file, generated_site, existing_site, gpu_clusters, state, _write_yaml
    )

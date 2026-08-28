from __future__ import annotations

import base64
import fnmatch
import hashlib
import json
import os
import re
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlsplit

import regional_deployment_inventory as inventory
from regional_release_config import ReleaseError
from regional_runtime_profile import verify_runtime_profile


ROOT = Path(__file__).resolve().parents[3]
REQUIRED_TOOLS = (
    "aws",
    "kubectl",
    "helm",
    "jq",
    "openssl",
    "sha256sum",
    "python3",
)
CONTROL_API_SCRIPT = r"""
import json
import os
import urllib.request

from gpu_fault.app import ApplicationContext


def get(path):
    request = urllib.request.Request(
        "http://127.0.0.1:8080" + path,
        headers={
            "X-GPU-Fault-Execution-Token":
                os.environ["GPU_FAULT_EXECUTION_TOKEN"],
        },
    )
    with urllib.request.urlopen(request, timeout=20) as response:
        return json.load(response)


def post(path, body):
    request = urllib.request.Request(
        "http://127.0.0.1:8080" + path,
        data=json.dumps(body).encode(),
        method="POST",
        headers={
            "Content-Type": "application/json",
            "X-GPU-Fault-Execution-Token":
                os.environ["GPU_FAULT_EXECUTION_TOKEN"],
        },
    )
    with urllib.request.urlopen(request, timeout=20) as response:
        return json.load(response)


clusters = json.loads(os.environ["ADMIN_CLUSTER_IDS_JSON"])
expected_nodes = json.loads(os.environ["ADMIN_EXPECTED_NODES_JSON"])
result = {
    "healthz": get("/healthz"),
    "version": get("/v1/version"),
    "registry": get("/v1/regional/clusters"),
    "clusters": {},
    "remote_commands": (
        ApplicationContext.from_environment().store.remote_command_stats()
    ),
}
for cluster_id in clusters:
    agents = get("/v1/fleet/agents?cluster_id=" + cluster_id)
    node_ids = sorted(expected_nodes[cluster_id])
    result["clusters"][cluster_id] = {
        "expected_node_ids": node_ids,
        "agents": agents,
        "fleet_readiness": (
            post(
                "/v1/fleet/readiness",
                {"cluster_id": cluster_id, "node_ids": node_ids},
            )
            if node_ids
            else None
        ),
        "collector_readiness": get(
            "/v1/collector-readiness/" + cluster_id
        ),
    }
print(json.dumps(result, separators=(",", ":")))
"""
EXECUTOR_TLS_SCRIPT = r"""
import json
import os
import ssl
import urllib.request

context = ssl.create_default_context(
    cafile=os.environ["GPU_FAULT_CONTROL_PLANE_CA_FILE"]
)
with urllib.request.urlopen(
    os.environ["GPU_FAULT_CONTROL_PLANE_URL"].rstrip("/") + "/healthz",
    context=context,
    timeout=15,
) as response:
    print(json.dumps(json.load(response), separators=(",", ":")))
"""


class CheckSkipped(RuntimeError):
    pass


@dataclass(frozen=True)
class CheckValue:
    summary: str
    details: Any = None


def _check(
    name: str,
    function: Callable[[], CheckValue],
    *,
    skipped_is_failure: bool = True,
) -> dict[str, Any]:
    try:
        value = function()
        return {
            "name": name,
            "status": "PASS",
            "summary": value.summary,
            "details": value.details,
        }
    except CheckSkipped as exc:
        return {
            "name": name,
            "status": "FAIL" if skipped_is_failure else "SKIP",
            "summary": str(exc),
            "details": None,
        }
    except Exception as exc:
        return {
            "name": name,
            "status": "FAIL",
            "summary": str(exc),
            "details": None,
        }


def _report(mode: str, release: Any, checks: list[dict[str, Any]]) -> dict[str, Any]:
    counts = {
        status: sum(item["status"] == status for item in checks)
        for status in ("PASS", "WARN", "FAIL", "SKIP")
    }
    return {
        "mode": mode,
        "site_name": release.config.site_name,
        "healthy": counts["FAIL"] == 0,
        "summary": counts,
        "checks": checks,
    }


def report_exit_code(report: dict[str, Any]) -> int:
    return 0 if report.get("healthy") else 1


def _decode_secret(value: str) -> bytes:
    return base64.b64decode(value.encode(), validate=True)


def _sensitive_mode(path: Path) -> str:
    mode = path.stat().st_mode & 0o777
    if mode & 0o077:
        raise ReleaseError(f"{path} must not grant group/other permissions")
    return f"{mode:03o}"


def _check_tools() -> CheckValue:
    missing = [name for name in REQUIRED_TOOLS if shutil.which(name) is None]
    if missing:
        raise ReleaseError("missing administrator tools: " + ", ".join(missing))
    return CheckValue("all required administrator tools are available")


def _check_local_inputs(release: Any) -> CheckValue:
    config = release.config
    files = [Path(config.cpu_kubeconfig)]
    ca_files: set[Path] = set()
    details: dict[str, Any] = {
        "cpu_kubeconfig": str(config.cpu_kubeconfig),
        "clusters": {},
    }
    for target in config.clusters:
        if not all((target.token_file, target.ca_file, target.fleet_master_file)):
            raise ReleaseError(
                f"{target.cluster_id} requires token_file, ca_file and fleet_master_file"
            )
        token = Path(target.token_file)
        ca = Path(target.ca_file)
        fleet = Path(target.fleet_master_file)
        files.extend((token, ca, fleet))
        ca_files.add(ca)
        for path in (token, ca, fleet):
            if not path.is_file():
                raise ReleaseError(f"required local input is missing: {path}")
        token_text = token.read_text(encoding="utf-8")
        if token_text != token_text.strip() or len(token_text) < 32:
            raise ReleaseError(
                f"{target.cluster_id} token must be at least 32 characters "
                "without surrounding whitespace"
            )
        fleet_text = fleet.read_text(encoding="utf-8")
        if fleet_text != fleet_text.strip() or len(fleet_text) < 32:
            raise ReleaseError(
                f"{target.cluster_id} fleet master must be at least 32 characters"
            )
        details["clusters"][target.cluster_id] = {
            "token_mode": _sensitive_mode(token),
            "fleet_master_mode": _sensitive_mode(fleet),
            "ca_file": str(ca),
        }
    for path in files:
        if not path.is_file():
            raise ReleaseError(f"required local input is missing: {path}")
    for ca_file in sorted(ca_files):
        release.runner.run(
            [
                "openssl",
                "x509",
                "-in",
                str(ca_file),
                "-noout",
                "-checkend",
                str(config.health.certificate_min_validity_days * 86400),
            ],
            capture=True,
        )
    return CheckValue("local kubeconfig and credential references are valid", details)


def _check_aws_identity(release: Any) -> CheckValue:
    value = json.loads(
        release.runner.run(
            ["aws", "sts", "get-caller-identity", "--output", "json"],
            capture=True,
        )
    )
    return CheckValue(
        "AWS caller identity is available",
        {
            "account": value.get("Account"),
            "arn": value.get("Arn"),
        },
    )


def _check_contexts(release: Any) -> CheckValue:
    release._ensure_contexts()
    return CheckValue(
        "CPU/GPU contexts, HyperPod ownership and executor IAM are valid",
        {"clusters": [item.cluster_id for item in release.config.clusters]},
    )


def _ready_nodes(document: dict[str, Any]) -> list[dict[str, Any]]:
    result = []
    for item in document.get("items", []):
        ready = any(
            condition.get("type") == "Ready" and condition.get("status") == "True"
            for condition in item.get("status", {}).get("conditions", [])
        )
        if ready:
            result.append(item)
    return result


def _check_cpu_capacity(release: Any) -> CheckValue:
    document = release._get_json(
        release._cpu(
            "get",
            "nodes",
            "-l",
            "sagemaker.amazonaws.com/cluster-name="
            + release.config.cpu_hyperpod_cluster_name,
        )
    )
    ready = _ready_nodes(document)
    if len(ready) != 3:
        raise ReleaseError(
            f"CPU control plane requires exactly 3 Ready nodes, got {len(ready)}"
        )
    return CheckValue(
        "CPU control plane has three Ready nodes",
        {"nodes": sorted(item["metadata"]["name"] for item in ready)},
    )


def _secret(release: Any, name: str) -> dict[str, Any]:
    return release._get_json(
        release._cpu(
            "-n",
            release.config.namespace,
            "get",
            "secret",
            name,
        )
    )


def check_cpu_secrets(release: Any) -> CheckValue:
    aurora = _secret(release, "gpu-fault-aurora").get("data") or {}
    active = _secret(release, "gpu-fault-control-plane-active").get("data") or {}
    node_keys = _secret(release, "gpu-fault-node-action-keys").get("data") or {}
    required_aurora = {"postgres-url", "master-secret-arn"}
    required_active = {
        "execution-token",
        "processor-replay-secret",
        "node-action-secret",
    }
    if missing := sorted(required_aurora - set(aurora)):
        raise ReleaseError("gpu-fault-aurora is missing: " + ", ".join(missing))
    if missing := sorted(required_active - set(active)):
        raise ReleaseError(
            "gpu-fault-control-plane-active is missing: " + ", ".join(missing)
        )
    decoded = {name: _decode_secret(active[name]) for name in required_active}
    if any(len(value) < 32 for value in decoded.values()):
        raise ReleaseError("control-plane active secrets must be at least 32 bytes")
    digests = {hashlib.sha256(value).hexdigest() for value in decoded.values()}
    if len(digests) != len(required_active):
        raise ReleaseError("control-plane active secrets must be pairwise distinct")
    if release.config.clusters and not node_keys:
        raise ReleaseError("gpu-fault-node-action-keys is empty")
    return CheckValue(
        "required CPU secrets exist without exposing their values",
        {
            "aurora_keys": sorted(aurora),
            "active_keys": sorted(active),
            "node_action_key_count": len(node_keys),
        },
    )


def check_email_notifications(release: Any) -> CheckValue:
    config = release.config.notifications
    if not config.allow_email:
        if config.acknowledge_external_alert_channel:
            return CheckValue(
                "application email is disabled and an external alert channel is acknowledged",
                {"enabled": False},
            )
        raise ReleaseError("no administrator notification channel is enabled")
    if not config.admin_email or not config.email_sender or not config.email_recipients:
        raise ReleaseError("email notification addresses are missing")
    identity = _aws_json(
        release,
        [
            "sesv2",
            "get-email-identity",
            "--email-identity",
            config.email_sender,
        ],
    )
    verified = bool(identity.get("VerifiedForSendingStatus")) or (
        str(identity.get("VerificationStatus") or "").upper() == "SUCCESS"
    )
    if not verified:
        raise ReleaseError("SES sender identity is not verified")
    account = _aws_json(release, ["sesv2", "get-account"])
    if not bool(account.get("SendingEnabled")):
        raise ReleaseError("SES sending is disabled")
    secret = _secret(release, "gpu-fault-email").get("data") or {}
    required = {
        "email-sender",
        "email-recipients",
        "email-subject-prefix",
        "site-id",
        "aws-account-id",
    }
    if missing := sorted(required - set(secret)):
        raise ReleaseError("gpu-fault-email is missing: " + ", ".join(missing))
    sender = _decode_secret(secret["email-sender"]).decode()
    recipients = tuple(
        item.strip()
        for item in _decode_secret(secret["email-recipients"]).decode().split(",")
        if item.strip()
    )
    subject_prefix = _decode_secret(secret["email-subject-prefix"]).decode()
    site_id = _decode_secret(secret["site-id"]).decode()
    account_id = _decode_secret(secret["aws-account-id"]).decode()
    expected_account_id = str(release.config.cpu_eks_arn).split(":")[4]
    if (
        sender != config.email_sender
        or recipients != config.email_recipients
        or subject_prefix != config.email_subject_prefix
        or site_id != release.config.site_name
        or account_id != expected_account_id
    ):
        raise ReleaseError("gpu-fault-email differs from the declared site addresses")
    return CheckValue(
        "SES administrator notification channel is configured",
        {
            "enabled": True,
            "sender_verified": True,
            "sending_enabled": True,
            "production_access_enabled": bool(account.get("ProductionAccessEnabled")),
            "recipient_count": len(recipients),
            "site_id": site_id,
        },
    )


def _check_load_balancer_controller(release: Any) -> CheckValue:
    document = release._get_json(release._cpu("get", "deployment", "-A"))
    matches = []
    for item in document.get("items", []):
        containers = (
            item.get("spec", {})
            .get("template", {})
            .get("spec", {})
            .get("containers", [])
        )
        text = " ".join(
            [
                item.get("metadata", {}).get("name", ""),
                *[str(container.get("image", "")) for container in containers],
            ]
        )
        if not re.search(r"load-balancer|operator-alb", text, re.IGNORECASE):
            continue
        replicas = int(item.get("spec", {}).get("replicas", 0) or 0)
        ready = int(item.get("status", {}).get("readyReplicas", 0) or 0)
        matches.append(
            {
                "namespace": item["metadata"].get("namespace"),
                "name": item["metadata"]["name"],
                "ready": ready,
                "replicas": replicas,
            }
        )
        if replicas <= 0 or ready != replicas:
            raise ReleaseError(
                f"load balancer controller {item['metadata']['name']} "
                f"has {ready}/{replicas} Ready"
            )
    if not matches:
        raise ReleaseError("no AWS Load Balancer Controller deployment was found")
    return CheckValue("AWS Load Balancer Controller is Ready", matches)


def _aws_json(release: Any, arguments: list[str]) -> dict[str, Any]:
    return json.loads(
        release.runner.run(
            [
                "aws",
                *arguments,
                "--region",
                release.config.aws_region,
                "--output",
                "json",
            ],
            capture=True,
        )
    )


def _certificate_details(release: Any) -> dict[str, Any]:
    certificate = (
        _aws_json(
            release,
            [
                "acm",
                "describe-certificate",
                "--certificate-arn",
                release.config.nlb["certificate_arn"],
            ],
        ).get("Certificate")
        or {}
    )
    if certificate.get("Status") != "ISSUED":
        raise ReleaseError(
            f"ACM certificate is {certificate.get('Status')}, expected ISSUED"
        )
    raw_expiry = certificate.get("NotAfter")
    if not raw_expiry:
        raise ReleaseError("ACM certificate has no NotAfter")
    expiry = datetime.fromisoformat(str(raw_expiry).replace("Z", "+00:00"))
    remaining = (expiry - datetime.now(timezone.utc)).total_seconds() / 86400
    if remaining < release.config.health.certificate_min_validity_days:
        raise ReleaseError(
            f"ACM certificate expires in {remaining:.1f} days; "
            f"minimum is {release.config.health.certificate_min_validity_days}"
        )
    return {
        "arn": release.config.nlb["certificate_arn"],
        "status": certificate.get("Status"),
        "not_after": expiry.isoformat(),
        "remaining_days": round(remaining, 1),
        "sans": certificate.get("SubjectAlternativeNames") or [],
    }


def _hostname_matches(pattern: str, hostname: str) -> bool:
    if pattern.startswith("*."):
        return fnmatch.fnmatchcase(hostname, pattern) and hostname.count(".") == (
            pattern.count(".")
        )
    return pattern == hostname


def _subnet_has_igw_default_route(
    release: Any,
    *,
    subnet_id: str,
    vpc_id: str,
) -> bool:
    tables = (
        _aws_json(
            release,
            [
                "ec2",
                "describe-route-tables",
                "--filters",
                f"Name=association.subnet-id,Values={subnet_id}",
            ],
        ).get("RouteTables")
        or []
    )
    if not tables:
        tables = (
            _aws_json(
                release,
                [
                    "ec2",
                    "describe-route-tables",
                    "--filters",
                    f"Name=vpc-id,Values={vpc_id}",
                    "Name=association.main,Values=true",
                ],
            ).get("RouteTables")
            or []
        )
    return any(
        route.get("DestinationCidrBlock") == "0.0.0.0/0"
        and str(route.get("GatewayId", "")).startswith("igw-")
        for table in tables
        for route in table.get("Routes", [])
    )


def _check_nlb_inputs(release: Any) -> CheckValue:
    if not release.config.nlb:
        raise CheckSkipped("NLB health configuration is missing")
    cpu = _aws_json(
        release,
        [
            "eks",
            "describe-cluster",
            "--name",
            release.config.cpu_eks_arn.rsplit("/", 1)[-1],
        ],
    )
    cpu_vpc = cpu["cluster"]["resourcesVpcConfig"]["vpcId"]
    subnet_ids = [
        item.strip()
        for item in release.config.nlb["public_subnets"].split(",")
        if item.strip()
    ]
    subnets = (
        _aws_json(
            release,
            ["ec2", "describe-subnets", "--subnet-ids", *subnet_ids],
        ).get("Subnets")
        or []
    )
    if len(subnets) != len(subnet_ids):
        raise ReleaseError("one or more NLB public subnets do not exist")
    if {item.get("VpcId") for item in subnets} != {cpu_vpc}:
        raise ReleaseError("NLB public subnets are not all in the CPU EKS VPC")
    availability_zones = {item.get("AvailabilityZone") for item in subnets}
    if len(availability_zones) < 2:
        raise ReleaseError("NLB requires public subnets in at least two AZs")
    for subnet_id in subnet_ids:
        if not _subnet_has_igw_default_route(
            release,
            subnet_id=subnet_id,
            vpc_id=cpu_vpc,
        ):
            raise ReleaseError(
                f"NLB subnet {subnet_id} has no 0.0.0.0/0 route to an IGW"
            )
    groups = (
        _aws_json(
            release,
            [
                "ec2",
                "describe-security-groups",
                "--group-ids",
                release.config.nlb["security_group"],
            ],
        ).get("SecurityGroups")
        or []
    )
    if len(groups) != 1 or groups[0].get("VpcId") != cpu_vpc:
        raise ReleaseError("NLB Security Group is not in the CPU EKS VPC")
    for permission in groups[0].get("IpPermissions", []):
        if permission.get("FromPort") != 443 or permission.get("ToPort") != 443:
            continue
        if any(
            item.get("CidrIp") == "0.0.0.0/0" for item in permission.get("IpRanges", [])
        ):
            raise ReleaseError("NLB Security Group exposes TCP 443 to 0.0.0.0/0")
        if any(
            item.get("CidrIpv6") == "::/0" for item in permission.get("Ipv6Ranges", [])
        ):
            raise ReleaseError("NLB Security Group exposes TCP 443 to ::/0")
    certificate = _certificate_details(release)
    hostnames = {
        urlsplit(target.control_plane_url).hostname
        for target in release.config.clusters
    }
    sans = certificate["sans"]
    uncovered = sorted(
        hostname
        for hostname in hostnames
        if hostname
        and not any(_hostname_matches(pattern, hostname) for pattern in sans)
    )
    if uncovered:
        raise ReleaseError(
            "ACM certificate SAN does not cover: " + ", ".join(uncovered)
        )
    return CheckValue(
        "NLB subnets, Security Group and certificate inputs are valid",
        {
            "vpc_id": cpu_vpc,
            "subnets": subnet_ids,
            "availability_zones": sorted(availability_zones),
            "security_group": release.config.nlb["security_group"],
            "certificate": certificate,
        },
    )


def _check_aurora(release: Any) -> CheckValue:
    cluster_id = release.config.health.aurora_cluster_id
    if not cluster_id:
        raise CheckSkipped("health.aurora_cluster_id is not configured")
    document = _aws_json(
        release,
        ["rds", "describe-db-clusters", "--db-cluster-identifier", cluster_id],
    )
    clusters = document.get("DBClusters") or []
    if len(clusters) != 1:
        raise ReleaseError(f"Aurora cluster {cluster_id} was not found")
    cluster = clusters[0]
    if cluster.get("Status") != "available":
        raise ReleaseError(f"Aurora cluster {cluster_id} is {cluster.get('Status')}")
    instances = (
        _aws_json(
            release,
            ["rds", "describe-db-instances"],
        ).get("DBInstances")
        or []
    )
    members = [
        item for item in instances if item.get("DBClusterIdentifier") == cluster_id
    ]
    available = [
        item for item in members if item.get("DBInstanceStatus") == "available"
    ]
    if len(members) < 2 or len(available) != len(members):
        raise ReleaseError(
            f"Aurora requires at least two available instances, got "
            f"{len(available)}/{len(members)}"
        )
    return CheckValue(
        "Aurora cluster and instances are available",
        {
            "cluster_id": cluster_id,
            "endpoint": cluster.get("Endpoint"),
            "reader_endpoint": cluster.get("ReaderEndpoint"),
            "instances": [
                {
                    "id": item.get("DBInstanceIdentifier"),
                    "status": item.get("DBInstanceStatus"),
                }
                for item in members
            ],
        },
    )


def _check_monitoring(release: Any) -> CheckValue:
    health = release.config.health
    if not health.amp_workspace_id or not health.sns_topic_arn:
        raise CheckSkipped("AMP workspace or SNS topic is not configured")
    workspace = (
        _aws_json(
            release,
            [
                "amp",
                "describe-workspace",
                "--workspace-id",
                health.amp_workspace_id,
            ],
        ).get("workspace")
        or {}
    )
    status = (workspace.get("status") or {}).get("statusCode")
    if status != "ACTIVE":
        raise ReleaseError(f"AMP workspace is {status}, expected ACTIVE")
    rules_document = _aws_json(
        release,
        [
            "amp",
            "describe-rule-groups-namespace",
            "--workspace-id",
            health.amp_workspace_id,
            "--name",
            health.amp_rule_namespace,
        ],
    )
    manager_document = _aws_json(
        release,
        [
            "amp",
            "describe-alert-manager-definition",
            "--workspace-id",
            health.amp_workspace_id,
        ],
    )
    subscriptions = (
        _aws_json(
            release,
            [
                "sns",
                "list-subscriptions-by-topic",
                "--topic-arn",
                health.sns_topic_arn,
            ],
        ).get("Subscriptions")
        or []
    )
    confirmed = [
        item
        for item in subscriptions
        if item.get("SubscriptionArn") not in {None, "PendingConfirmation"}
    ]
    if health.require_confirmed_sns_subscription and not confirmed:
        raise ReleaseError("SNS topic has no confirmed subscription")
    rules_data = (rules_document.get("ruleGroupsNamespace") or {}).get("data")
    manager_data = (manager_document.get("alertManagerDefinition") or {}).get("data")
    if not rules_data or not manager_data:
        raise ReleaseError("AMP live rules or Alertmanager definition has no data")
    try:
        rules_text = base64.b64decode(rules_data).decode()
        manager_text = base64.b64decode(manager_data).decode()
    except (ValueError, UnicodeDecodeError) as exc:
        raise ReleaseError("AMP live configuration is not valid base64") from exc
    if health.sns_topic_arn not in manager_text:
        raise ReleaseError(
            "AMP Alertmanager does not reference the configured SNS topic"
        )
    with tempfile.TemporaryDirectory(prefix="gpu-fault-alert-verify-") as directory:
        root = Path(directory)
        rules = root / "rules.yaml"
        manager = root / "alertmanager.yaml"
        rules.write_text(rules_text, encoding="utf-8")
        manager.write_text(manager_text, encoding="utf-8")
        verifier = release.runner.run(
            [
                "python3",
                str(ROOT / "scripts/verify-regional-alerting.py"),
                "--live-rules",
                str(rules),
                "--live-alertmanager",
                str(manager),
            ],
            capture=True,
        )
    return CheckValue(
        "AMP rules, Alertmanager and SNS destination are configured",
        {
            "workspace_id": health.amp_workspace_id,
            "workspace_status": status,
            "rule_namespace": health.amp_rule_namespace,
            "sns_topic_arn": health.sns_topic_arn,
            "confirmed_subscriptions": len(confirmed),
            "verifier": verifier,
        },
    )


def check_aurora_refresh(
    release: Any,
    *,
    require_success: bool = False,
) -> CheckValue:
    cronjob = release._get_json(
        release._cpu(
            "-n",
            release.config.namespace,
            "get",
            "cronjob",
            "gpu-fault-aurora-credential-refresh",
        )
    )
    if cronjob.get("spec", {}).get("suspend") is True:
        raise ReleaseError("Aurora credential refresh CronJob is suspended")
    containers = (
        cronjob.get("spec", {})
        .get("jobTemplate", {})
        .get("spec", {})
        .get("template", {})
        .get("spec", {})
        .get("containers", [])
    )
    environment = {
        item.get("name"): item.get("value")
        for container in containers
        for item in container.get("env", [])
        if item.get("name")
    }
    targets = tuple(
        item.strip()
        for item in str(
            environment.get("GPU_FAULT_AURORA_RESTART_DEPLOYMENTS") or ""
        ).split(",")
        if item.strip()
    )
    if not targets:
        raise ReleaseError(
            "Aurora credential refresh declares no database consumer deployments"
        )
    role = release._get_json(
        release._cpu(
            "-n",
            release.config.namespace,
            "get",
            "role",
            "gpu-fault-aurora-credential-refresh",
        )
    )
    deployment_rules = [
        rule
        for rule in role.get("rules", [])
        if "apps" in (rule.get("apiGroups") or [])
        and "deployments" in (rule.get("resources") or [])
        and {"get", "patch"}.issubset(set(rule.get("verbs") or []))
    ]
    allowed_targets = {
        name for rule in deployment_rules for name in (rule.get("resourceNames") or [])
    }
    if allowed_targets != set(targets):
        raise ReleaseError(
            "Aurora credential refresh deployment RBAC differs from targets: "
            f"targets={sorted(targets)}, allowed={sorted(allowed_targets)}"
        )
    last_successful = cronjob.get("status", {}).get("lastSuccessfulTime")
    if require_success and not last_successful:
        raise ReleaseError("Aurora credential refresh has never completed successfully")
    jobs = (
        release._get_json(
            release._cpu(
                "-n",
                release.config.namespace,
                "get",
                "job",
                "-l",
                "app=gpu-fault-aurora-credential-refresh",
            )
        ).get("items")
        or []
    )
    if jobs:
        latest = max(
            jobs,
            key=lambda item: item.get("metadata", {}).get(
                "creationTimestamp",
                "",
            ),
        )
        conditions = latest.get("status", {}).get("conditions") or []
        failed = any(
            item.get("type") == "Failed" and item.get("status") == "True"
            for item in conditions
        )
        complete = any(
            item.get("type") == "Complete" and item.get("status") == "True"
            for item in conditions
        )
        if failed and not complete:
            raise ReleaseError(
                "latest Aurora credential refresh Job failed: "
                + latest.get("metadata", {}).get("name", "unknown")
            )
    return CheckValue(
        "Aurora credential refresh CronJob is installed",
        {
            "schedule": cronjob.get("spec", {}).get("schedule"),
            "last_schedule_time": cronjob.get("status", {}).get("lastScheduleTime"),
            "last_successful_time": last_successful,
            "deployment_targets": list(targets),
        },
    )


def build_preflight_report(release: Any) -> dict[str, Any]:
    specifications = [
        ("tools", _check_tools),
        ("local_inputs", lambda: _check_local_inputs(release)),
        ("aws_identity", lambda: _check_aws_identity(release)),
        ("regional_contexts", lambda: _check_contexts(release)),
        ("cpu_capacity", lambda: _check_cpu_capacity(release)),
        ("cpu_secrets", lambda: check_cpu_secrets(release)),
        (
            "load_balancer_controller",
            lambda: _check_load_balancer_controller(release),
        ),
        ("nlb_inputs", lambda: _check_nlb_inputs(release)),
        ("aurora", lambda: _check_aurora(release)),
        ("aurora_credential_refresh", lambda: check_aurora_refresh(release)),
        ("email_notifications", lambda: check_email_notifications(release)),
        ("monitoring", lambda: _check_monitoring(release)),
    ]
    with ThreadPoolExecutor(max_workers=min(8, len(specifications))) as executor:
        futures = [
            executor.submit(_check, name, function) for name, function in specifications
        ]
        checks = [future.result() for future in futures]
    return _report("preflight", release, checks)


def _deployment_readiness(item: dict[str, Any], name: str) -> dict[str, Any]:
    replicas = int(item.get("spec", {}).get("replicas", 0) or 0)
    ready = int(item.get("status", {}).get("readyReplicas", 0) or 0)
    if ready != replicas:
        raise ReleaseError(f"{name} has {ready}/{replicas} Ready")
    return {"replicas": replicas, "ready": ready}


def _check_cpu_workloads(release: Any) -> CheckValue:
    names = (
        "gpu-fault-api-ha",
        "gpu-fault-control-worker",
        "gpu-fault-telemetry-spool-worker",
        "gpu-fault-adot",
    )
    details = {}
    for name in names:
        item = release._get_json(
            release._cpu(
                "-n",
                release.config.namespace,
                "get",
                "deployment",
                name,
            )
        )
        direct = {
            env.get("name")
            for env in (
                item.get("spec", {})
                .get("template", {})
                .get("spec", {})
                .get("containers", [{}])[0]
                .get("env", [])
            )
        }
        legacy = {
            "GPU_FAULT_ALLOW_EMAIL",
            "GPU_FAULT_ACKNOWLEDGE_NO_ALERT_CHANNEL",
        }.intersection(direct)
        if legacy:
            raise ReleaseError(
                f"{name} overrides notification ConfigMap values directly: "
                + ", ".join(sorted(legacy))
            )
        details[name] = _deployment_readiness(item, name)
    if details["gpu-fault-api-ha"]["replicas"] != 3:
        raise ReleaseError("gpu-fault-api-ha must run exactly three replicas")
    if details["gpu-fault-control-worker"]["replicas"] <= 0:
        raise ReleaseError("gpu-fault-control-worker is scaled to zero")
    if details["gpu-fault-adot"]["replicas"] <= 0:
        raise ReleaseError("gpu-fault-adot is scaled to zero")
    core = release._config_map_data("gpu-fault-api-ha-config-core")
    if (
        core.get("GPU_FAULT_REQUIRED_RUNTIME_PROFILE_VERSION")
        != release.config.runtime_profile_version
    ):
        raise ReleaseError("CPU required Runtime Profile version does not match site")
    details["runtime_profile_version"] = core.get(
        "GPU_FAULT_REQUIRED_RUNTIME_PROFILE_VERSION"
    )
    expected_allow = str(release.config.notifications.allow_email).lower()
    expected_acknowledge = str(
        release.config.notifications.acknowledge_external_alert_channel
    ).lower()
    if core.get("GPU_FAULT_ALLOW_EMAIL") != expected_allow:
        raise ReleaseError("CPU email enablement does not match site")
    if core.get("GPU_FAULT_ACKNOWLEDGE_NO_ALERT_CHANNEL") != expected_acknowledge:
        raise ReleaseError("CPU external alert acknowledgement does not match site")
    details["email_enabled"] = release.config.notifications.allow_email
    return CheckValue("CPU control-plane workloads are at desired readiness", details)


def _run_read_only_verifiers(release: Any) -> CheckValue:
    control_output = release.runner.run(
        [
            "python3",
            str(ROOT / "deploy/control-plane/tools/verify_control_plane_role_split.py"),
        ],
        env={
            **os.environ,
            "KUBECONFIG": release.config.cpu_kubeconfig,
            "GPU_FAULT_NAMESPACE": release.config.namespace,
            "GPU_FAULT_RUNTIME_IMAGE": release.runtime_image,
        },
        capture=True,
    )
    gpu_outputs = {}
    for target in release.config.clusters:
        gpu_outputs[target.cluster_id] = release.runner.run(
            [
                "python3",
                str(ROOT / "deploy/dataplane/tools/verify_dataplane_executor.py"),
            ],
            env={
                **os.environ,
                "GPU_FAULT_NAMESPACE": release.config.namespace,
                "GPU_FAULT_KUBE_CONTEXT": target.context,
                "GPU_FAULT_CONTROL_PLANE_KUBECONFIG": (release.config.cpu_kubeconfig),
                "GPU_FAULT_EXPECTED_WHEEL_CONFIGMAP": release.executor_wheel_cm,
            },
            capture=True,
        )
    return CheckValue(
        "CPU role split and every GPU executor pass read-only verifiers",
        {"cpu": control_output, "clusters": gpu_outputs},
    )


def _cpu_ingress_pod(release: Any) -> str:
    pod = release.runner.run(
        release._cpu(
            "-n",
            release.config.namespace,
            "get",
            "pod",
            "-l",
            f"app={inventory.CPU_INGRESS_DEPLOYMENT}",
            "--field-selector=status.phase=Running",
            "-o",
            "jsonpath={.items[0].metadata.name}",
        ),
        capture=True,
    )
    if not pod:
        raise ReleaseError("no Running CPU ingress Pod")
    return pod


def _control_api_report(release: Any) -> dict[str, Any]:
    expected_nodes = {}
    for target in release.config.clusters:
        nodes = release._get_json(
            release._gpu(
                target,
                "get",
                "nodes",
                "-l",
                f"sagemaker.amazonaws.com/cluster-name={target.hyperpod_cluster_name}",
            )
        )
        expected_nodes[target.cluster_id] = sorted(
            item["metadata"]["name"] for item in _ready_nodes(nodes)
        )
    return json.loads(
        release.runner.run(
            release._cpu(
                "-n",
                release.config.namespace,
                "exec",
                _cpu_ingress_pod(release),
                "--",
                "env",
                "ADMIN_CLUSTER_IDS_JSON="
                + json.dumps(
                    [item.cluster_id for item in release.config.clusters],
                    separators=(",", ":"),
                ),
                "ADMIN_EXPECTED_NODES_JSON="
                + json.dumps(expected_nodes, separators=(",", ":")),
                "python",
                "-c",
                CONTROL_API_SCRIPT,
            ),
            capture=True,
            sensitive=True,
        )
    )


def _check_control_api(release: Any) -> CheckValue:
    report = _control_api_report(release)
    if report.get("healthz", {}).get("status") != "ok":
        raise ReleaseError("control-plane /healthz is not ok")
    version = report.get("version") or {}
    if version.get("deployment_mode") != "regional":
        raise ReleaseError("control plane is not running in regional mode")
    if version.get("required_agent_artifact_sha256") != release.node_wheel_sha:
        raise ReleaseError(
            "control-plane required Agent artifact does not match release"
        )
    if (
        version.get("required_agent_config_digest")
        != release.config.agent_config_digest
    ):
        raise ReleaseError("control-plane required Agent config digest does not match")
    if version.get("compatible_agent_artifact_sha256s"):
        raise ReleaseError("Agent artifact compatibility window is still open")
    if version.get("required_agent_compatibility_digest") != (
        release.config.component_digests.get("node_runtime") or release.node_wheel_sha
    ):
        raise ReleaseError(
            "control-plane required Agent compatibility digest does not match release"
        )
    if version.get("compatible_agent_compatibility_digests"):
        raise ReleaseError("Agent compatibility digest window is still open")
    if version.get("compatible_agent_protocol_versions"):
        raise ReleaseError("Agent protocol compatibility window is still open")
    if version.get("compatible_agent_config_digests"):
        raise ReleaseError("Agent config compatibility window is still open")
    if (
        version.get("required_runtime_profile_version")
        != release.config.runtime_profile_version
    ):
        raise ReleaseError("control-plane required Runtime Profile does not match site")
    if version.get("required_node_action_key_version") != 2:
        raise ReleaseError(
            "control-plane requires an unexpected Node Action key version"
        )
    if version.get("compatible_regional_executor_protocol_versions"):
        raise ReleaseError(
            "regional executor protocol compatibility window is still open"
        )
    if (
        version.get("required_regional_executor_artifact_sha256")
        != release.executor_wheel_sha
    ):
        raise ReleaseError(
            "control-plane required Executor artifact does not match release"
        )
    if version.get("compatible_regional_executor_artifact_sha256s"):
        raise ReleaseError(
            "regional executor artifact compatibility window is still open"
        )
    if version.get("required_regional_executor_compatibility_digest") != (
        release.config.component_digests.get("executor") or release.executor_wheel_sha
    ):
        raise ReleaseError(
            "control-plane required Executor compatibility digest does not match release"
        )
    if version.get("compatible_regional_executor_compatibility_digests"):
        raise ReleaseError(
            "regional executor compatibility digest window is still open"
        )
    registered = {item.get("cluster_id") for item in report.get("registry", [])}
    configured = {item.cluster_id for item in release.config.clusters}
    if registered != configured:
        raise ReleaseError(
            f"regional registry drift: configured={sorted(configured)}, "
            f"registered={sorted(registered)}"
        )
    for cluster_id, cluster in report.get("clusters", {}).items():
        agents = cluster.get("agents") or []
        if not agents:
            raise ReleaseError(f"{cluster_id} has no registered Node Agents")
        expected_nodes = set(cluster.get("expected_node_ids") or [])
        agent_nodes = {item.get("node_id") for item in agents}
        if agent_nodes != expected_nodes:
            raise ReleaseError(
                f"{cluster_id} Agent coverage drift: "
                f"expected={sorted(expected_nodes)}, agents={sorted(agent_nodes)}"
            )
        if any(item.get("lifecycle_state") != "ACTIVE" for item in agents):
            raise ReleaseError(f"{cluster_id} has a non-ACTIVE Node Agent")
        if any(
            item.get("runtime_profile_version")
            != release.config.runtime_profile_version
            for item in agents
        ):
            raise ReleaseError(f"{cluster_id} Agent Runtime Profile drift")
        if not (cluster.get("fleet_readiness") or {}).get("ready"):
            raise ReleaseError(f"{cluster_id} fleet readiness failed")
        if not (cluster.get("collector_readiness") or {}).get("ready"):
            raise ReleaseError(f"{cluster_id} collector readiness failed")
    remote = report.get("remote_commands") or {}
    oldest = float(remote.get("oldest_unclaimed_age_seconds", 0) or 0)
    if oldest > release.config.health.remote_command_max_unclaimed_seconds:
        raise ReleaseError(
            f"oldest unclaimed remote command is {oldest:.1f}s; "
            f"limit is {release.config.health.remote_command_max_unclaimed_seconds}s"
        )
    if int(remote.get("executor_internal_error_total", 0) or 0) > 0:
        raise ReleaseError("executor internal remote-command errors are non-zero")
    return CheckValue("control-plane API, fleet and collectors are healthy", report)


def _check_gpu_cluster(release: Any, target: Any) -> CheckValue:
    details: dict[str, Any] = {"deployments": {}}
    for deployment in (*inventory.DEPLOYMENTS, inventory.GPU_RECONCILER_DEPLOYMENT):
        item = release._get_json(
            release._gpu(
                target,
                "-n",
                release.config.namespace,
                "get",
                "deployment",
                deployment,
            )
        )
        details["deployments"][deployment] = _deployment_readiness(item, deployment)
    nodes = release._get_json(
        release._gpu(
            target,
            "get",
            "nodes",
            "-l",
            f"sagemaker.amazonaws.com/cluster-name={target.hyperpod_cluster_name}",
        )
    )
    expected_nodes = _ready_nodes(nodes)
    if not expected_nodes:
        raise ReleaseError(f"{target.cluster_id} has no Ready HyperPod nodes")
    for item in expected_nodes:
        annotations = item.get("metadata", {}).get("annotations", {})
        if annotations.get("gpu-fault.io/installer-state") != "Succeeded":
            raise ReleaseError(
                f"{target.cluster_id}/{item['metadata']['name']} installer is not Succeeded"
            )
        if (
            annotations.get("gpu-fault.io/installer-artifact-sha256")
            != release.node_wheel_sha
        ):
            raise ReleaseError(
                f"{target.cluster_id}/{item['metadata']['name']} artifact drift"
            )
    readiness = release.runner.run(
        release._gpu(
            target,
            "-n",
            release.config.namespace,
            "exec",
            "deployment/gpu-fault-cluster-executor",
            "--",
            "gpu-fault-cluster-executor-readiness",
        ),
        capture=True,
        sensitive=True,
    )
    tls = json.loads(
        release.runner.run(
            release._gpu(
                target,
                "-n",
                release.config.namespace,
                "exec",
                "deployment/gpu-fault-cluster-executor",
                "--",
                "python",
                "-c",
                EXECUTOR_TLS_SCRIPT,
            ),
            capture=True,
        )
    )
    if tls.get("status") != "ok":
        raise ReleaseError(f"{target.cluster_id} TLS probe did not return ok")
    details["nodes"] = sorted(item["metadata"]["name"] for item in expected_nodes)
    details["executor_readiness"] = readiness
    details["tls"] = tls
    return CheckValue(f"{target.cluster_id} data plane is healthy", details)


def _check_nlb_runtime(release: Any) -> CheckValue:
    if not release.config.nlb:
        raise CheckSkipped("NLB health configuration is missing")
    name = release.config.nlb.get("name") or (
        f"gpu-fault-regional-{release.config.aws_region}"
    )
    load_balancers = (
        _aws_json(
            release,
            ["elbv2", "describe-load-balancers", "--names", name],
        ).get("LoadBalancers")
        or []
    )
    if len(load_balancers) != 1:
        raise ReleaseError(f"NLB {name} was not found")
    nlb = load_balancers[0]
    if (nlb.get("State") or {}).get("Code") != "active":
        raise ReleaseError(f"NLB {name} is not active")
    arn = nlb["LoadBalancerArn"]
    listeners = (
        _aws_json(
            release,
            ["elbv2", "describe-listeners", "--load-balancer-arn", arn],
        ).get("Listeners")
        or []
    )
    tls = [
        item
        for item in listeners
        if item.get("Port") == 443 and item.get("Protocol") == "TLS"
    ]
    if len(tls) != 1:
        raise ReleaseError("NLB must have exactly one TLS listener on 443")
    certificate_arns = {
        item.get("CertificateArn") for item in tls[0].get("Certificates", [])
    }
    if release.config.nlb["certificate_arn"] not in certificate_arns:
        raise ReleaseError("NLB TLS listener does not use the configured certificate")
    target_groups = (
        _aws_json(
            release,
            ["elbv2", "describe-target-groups", "--load-balancer-arn", arn],
        ).get("TargetGroups")
        or []
    )
    if not target_groups:
        raise ReleaseError("NLB has no target group")
    target_health = (
        _aws_json(
            release,
            [
                "elbv2",
                "describe-target-health",
                "--target-group-arn",
                target_groups[0]["TargetGroupArn"],
            ],
        ).get("TargetHealthDescriptions")
        or []
    )
    healthy = [
        item
        for item in target_health
        if (item.get("TargetHealth") or {}).get("State") == "healthy"
    ]
    if len(healthy) < 3:
        raise ReleaseError(f"NLB has only {len(healthy)} healthy targets")
    return CheckValue(
        "NLB listener, certificate and targets are healthy",
        {
            "name": name,
            "dns": nlb.get("DNSName"),
            "healthy_targets": len(healthy),
            "certificate": _certificate_details(release),
        },
    )


def build_health_report(release: Any, *, mode: str) -> dict[str, Any]:
    specifications = [
        ("regional_contexts", lambda: _check_contexts(release)),
        ("cpu_secrets", lambda: check_cpu_secrets(release)),
        ("cpu_workloads", lambda: _check_cpu_workloads(release)),
        ("email_notifications", lambda: check_email_notifications(release)),
        (
            "aurora_credential_refresh",
            lambda: check_aurora_refresh(release, require_success=True),
        ),
        ("runtime_profile", lambda: _verify_profile(release)),
        ("read_only_verifiers", lambda: _run_read_only_verifiers(release)),
        ("control_api", lambda: _check_control_api(release)),
    ]
    specifications.extend(
        (
            f"gpu_cluster:{target.cluster_id}",
            lambda target=target: _check_gpu_cluster(release, target),
        )
        for target in release.config.clusters
    )
    specifications.extend(
        [
            ("nlb_runtime", lambda: _check_nlb_runtime(release)),
            ("aurora", lambda: _check_aurora(release)),
            ("monitoring", lambda: _check_monitoring(release)),
        ]
    )
    with ThreadPoolExecutor(max_workers=min(8, len(specifications))) as executor:
        futures = [
            executor.submit(_check, name, function) for name, function in specifications
        ]
        checks = [future.result() for future in futures]
    return _report(mode, release, checks)


def _verify_profile(release: Any) -> CheckValue:
    verify_runtime_profile(release)
    return CheckValue(
        "declared Runtime Profile is registered without drift",
        {"version": release.config.runtime_profile_version},
    )

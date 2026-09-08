from __future__ import annotations

import base64
import fnmatch
import hashlib
import json
import os
import re
import shutil
import tempfile
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlsplit

from gpu_fault_release import regional_deployment_inventory as inventory
from gpu_fault_release import regional_monitoring_safety as monitoring_safety
from gpu_fault_release.regional_notifications import check_notification_channel
from gpu_fault_release.regional_release_config import ReleaseError
from gpu_fault_release.regional_release_probes import probe_source
from gpu_fault_release.regional_release_runtime_identity import (
    CONTROL_PLANE_PYTHON,
    EXECUTOR_PYTHON,
    EXECUTOR_READINESS,
    exec_cpu_ingress,
    validate_runtime_component_identity,
)
from gpu_fault_release.regional_release_state import aws_json
from gpu_fault_release.regional_release_workflow_safety import workflow_safety_snapshot
from gpu_fault_release.regional_runtime_profile import verify_runtime_profile
from gpu_fault_release.regional_validation_evidence import (
    QUICK_VALIDATION_EVIDENCE_ENV,
    read_only_verifier_details,
)
from gpu_fault_release.regional_validation_evidence import (
    quick_validation_evidence as _quick_validation_evidence,
)

ROOT = Path(__file__).resolve().parents[2]
REQUIRED_TOOLS = ("aws", "kubectl", "helm", "jq", "openssl", "sha256sum", "python3")


class CheckSkipped(RuntimeError):
    pass


@dataclass(frozen=True)
class CheckValue:
    summary: str
    details: Any = None
    # ``WARN`` is a finding that does not fail the report: something the
    # operator should read, not something the deploy should stop on.
    status: str = "PASS"


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
            "status": value.status,
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


def _read_snapshot(release: Any):
    factory = getattr(release, "_read_snapshot", None)
    return factory() if callable(factory) else nullcontext()


def _prime_deployment_snapshot(release: Any) -> None:
    prime = getattr(release, "_prime_deployment_snapshot", None)
    if callable(prime):
        prime()


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
    summary, details = check_notification_channel(
        release,
        aws_json=lambda arguments: _aws_json(release, arguments),
        read_secret=lambda name: _secret(release, name),
        decode_secret=_decode_secret,
    )
    return CheckValue(summary, details)


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
    # A question repeated inside one report costs one round trip. The two that
    # actually repeat are the IGW default-route test, which falls back to the same
    # VPC main route table once per public subnet, and the executor role expansion
    # `regional_contexts` drives, where every GPU cluster attaches the same managed
    # policies. The second of those arrives from a thread pool inside a thread
    # pool, which is what the shared helper's promise-before-read is built for.
    #
    # It stops at the report boundary on purpose. `_check_nlb_inputs` and
    # `_check_nlb_runtime` both describe the ACM certificate, but they belong to
    # the preflight and the verify report respectively, and the verify report
    # exists to observe what the release changed.
    return aws_json(release, arguments)


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


def _is_exact_https_tcp(permission: dict[str, Any]) -> bool:
    """True only for a rule scoped to exactly tcp/443-443."""

    return (
        str(permission.get("IpProtocol")) == "tcp"
        and permission.get("FromPort") == 443
        and permission.get("ToPort") == 443
    )


def permission_reaches_https(permission: dict[str, Any]) -> bool:
    """True if the rule admits TCP traffic to port 443.

    A `FromPort`-only check misses a wide range (443-65535), the all-traffic
    protocol ("-1"), or a range that starts below 443 and spans past it. Any of
    those admits the HTTPS port and must be inspected, not skipped.
    """

    protocol = str(permission.get("IpProtocol"))
    if protocol == "-1":
        return True
    if protocol != "tcp":
        return False
    from_port = permission.get("FromPort")
    to_port = permission.get("ToPort")
    if from_port is None or to_port is None:
        # A tcp rule with no port range covers every tcp port, 443 included.
        return True
    try:
        return int(from_port) <= 443 <= int(to_port)
    except (TypeError, ValueError):
        return True


def nlb_https_rule_violation(permission: dict[str, Any]) -> str | None:
    """Return why an ingress rule that reaches TCP 443 is unacceptable, or None.

    Only a rule scoped to exactly tcp/443-443 counts as the intended HTTPS
    ingress. A wider port range (e.g. 443-65535), the all-traffic protocol
    ("-1"), or any other shape that still admits 443 opens more than the NLB
    listener and is rejected outright -- otherwise it would bypass the
    0.0.0.0/0 exposure check below by not matching an exact-port comparison.
    """

    if not permission_reaches_https(permission):
        return None
    exposed_v4 = any(
        item.get("CidrIp") == "0.0.0.0/0" for item in permission.get("IpRanges", [])
    )
    exposed_v6 = any(
        item.get("CidrIpv6") == "::/0" for item in permission.get("Ipv6Ranges", [])
    )
    if not _is_exact_https_tcp(permission):
        if exposed_v4 or exposed_v6:
            return (
                "NLB Security Group admits TCP 443 through an overly broad rule "
                "exposed to the internet"
            )
        return (
            "NLB Security Group admits TCP 443 through an overly broad rule "
            "(expected exactly tcp/443-443)"
        )
    if exposed_v4:
        return "NLB Security Group exposes TCP 443 to 0.0.0.0/0"
    if exposed_v6:
        return "NLB Security Group exposes TCP 443 to ::/0"
    return None


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
        violation = nlb_https_rule_violation(permission)
        if violation:
            raise ReleaseError(violation)
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
    # Filtered server-side: without it every RDS instance in the region came
    # back, most of them belonging to other systems, to be discarded here.
    instances = (
        _aws_json(
            release,
            [
                "rds",
                "describe-db-instances",
                "--filters",
                f"Name=db-cluster-id,Values={cluster_id}",
            ],
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
    # The confirmation gate moved to the first minute of ``gpu-fault-admin
    # deploy`` (``notification_precheck``), where the operator can act on it.
    # Here it is a warning: the report still names the topic nobody listens to,
    # but a subscription that lapsed after the deploy does not fail ``verify``.
    unconfirmed = health.require_confirmed_sns_subscription and not confirmed
    email_summary = monitoring_safety.email_subscription_summary(
        subscriptions,
        release.config.notifications.admin_email,
    )
    rules_text, manager_text = monitoring_safety.decode_monitoring_configuration(
        rules_document,
        manager_document,
        health.sns_topic_arn,
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
        (
            "AMP rules and Alertmanager are configured; SNS topic has no "
            "confirmed subscription"
            if unconfirmed
            else "AMP rules, Alertmanager and SNS destination are configured"
        ),
        {
            "workspace_id": health.amp_workspace_id,
            "workspace_status": status,
            "rule_namespace": health.amp_rule_namespace,
            "sns_topic_arn": health.sns_topic_arn,
            "confirmed_subscriptions": len(confirmed),
            "email_subscription": email_summary,
            "verifier": verifier,
        },
        status="WARN" if unconfirmed else "PASS",
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
        (
            "workflow_safety",
            lambda: CheckValue(
                "no active destructive workflow blocks release",
                workflow_safety_snapshot(release),
            ),
        ),
    ]
    with _read_snapshot(release):
        with ThreadPoolExecutor(max_workers=min(8, len(specifications))) as executor:
            futures = [
                executor.submit(_check, name, function)
                for name, function in specifications
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


# The checks a default ``status`` runs. Together they answer "is the control
# plane up and serving this release" from a handful of ``kubectl get`` calls and
# one exec into the ingress Pod, where the full report also probes every GPU
# cluster, the role split, the NLB, Aurora and AMP -- 44 kubectl calls including
# 15 execs, about 44 seconds on production. The full report stays one flag away
# (``status --full``) and is still what acceptance evidence records.
QUICK_HEALTH_CHECKS = ("cpu_workloads", "control_api")


def build_quick_health_report(release: Any) -> dict[str, Any]:
    specifications = [
        ("cpu_workloads", lambda: _check_cpu_workloads(release)),
        ("control_api", lambda: _check_control_api(release)),
    ]
    assert tuple(name for name, _ in specifications) == QUICK_HEALTH_CHECKS
    with _read_snapshot(release):
        _prime_deployment_snapshot(release)
        with ThreadPoolExecutor(max_workers=len(specifications)) as executor:
            futures = [
                executor.submit(_check, name, function)
                for name, function in specifications
            ]
            checks = [future.result() for future in futures]
    report = _report("status", release, checks)
    report["scope"] = "quick"
    report["reused_validation_checks"] = []
    return report


def run_read_only_verifiers(
    release: Any,
    *,
    reused_checks: set[str] | None = None,
) -> CheckValue:
    return CheckValue(
        "CPU role split and every GPU executor pass read-only verifiers",
        read_only_verifier_details(
            release,
            reused_checks=reused_checks,
        ),
    )


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
        exec_cpu_ingress(
            release,
            arguments=(
                "env",
                "ADMIN_CLUSTER_IDS_JSON="
                + json.dumps(
                    [item.cluster_id for item in release.config.clusters],
                    separators=(",", ":"),
                ),
                "ADMIN_EXPECTED_NODES_JSON="
                + json.dumps(expected_nodes, separators=(",", ":")),
                CONTROL_PLANE_PYTHON,
                "-c",
                probe_source("control_api_inspect"),
            ),
            failure="the control API report",
            sensitive=True,
            interactive=False,
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
    registered = {
        item.get("cluster_id")
        for item in report.get("registry", [])
        if item.get("lifecycle_state", "ACTIVE") in {"ACTIVE", "PENDING"}
    }
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
    current_internal_errors = int(remote.get("executor_internal_error_total", 0) or 0)
    state = release._load_state()
    previous = state.get("previous")
    timestamp_field = "executor_internal_error_last_seen_timestamp_seconds"
    raw_baseline_timestamp = (
        previous.get(timestamp_field) if isinstance(previous, dict) else None
    )
    current_raw_timestamp = remote.get(timestamp_field)
    if raw_baseline_timestamp is not None and current_raw_timestamp is not None:
        try:
            baseline_timestamp = float(raw_baseline_timestamp)
            current_timestamp = float(current_raw_timestamp)
        except (TypeError, ValueError) as exc:
            raise ReleaseError(
                "executor internal remote-command error timestamp is invalid"
            ) from exc
        if min(baseline_timestamp, current_timestamp) < 0:
            raise ReleaseError(
                "executor internal remote-command error timestamp is invalid"
            )
        if current_timestamp > baseline_timestamp:
            raise ReleaseError(
                "executor internal remote-command error observed during release: "
                f"{baseline_timestamp:.6f}->{current_timestamp:.6f}"
            )
        return CheckValue("control-plane API, fleet and collectors are healthy", report)
    raw_baseline = (
        previous.get("executor_internal_error_total", 0)
        if isinstance(previous, dict)
        else 0
    )
    try:
        baseline_internal_errors = int(raw_baseline or 0)
    except (TypeError, ValueError) as exc:
        raise ReleaseError(
            "executor internal remote-command error baseline is invalid"
        ) from exc
    if current_internal_errors > baseline_internal_errors:
        raise ReleaseError(
            "executor internal remote-command errors increased during release: "
            f"{baseline_internal_errors}->{current_internal_errors}"
        )
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
            EXECUTOR_READINESS,
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
                EXECUTOR_PYTHON,
                "-c",
                probe_source("executor_tls_healthz"),
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


def _check_runtime_component_identity(release: Any) -> CheckValue:
    return CheckValue(
        "Every runtime Pod loads its declared component wheel",
        validate_runtime_component_identity(release),
    )


def _parallel(fetches: list[Callable[[], Any]]) -> list[Any]:
    """Issue reads that do not depend on each other at once.

    Used inside a single check, where the enclosing thread pool cannot help: one
    check is one task, so a check made of five dependent AWS round trips takes
    five round trips no matter how wide the report is.

    Results and failures are resolved in submission order, so a check whose first
    read is the one that fails reports the same reason it did when it was a
    sequence of statements.
    """

    with ThreadPoolExecutor(max_workers=len(fetches)) as executor:
        futures = [executor.submit(fetch) for fetch in fetches]
    return [future.result() for future in futures]


def _started(fetch: Callable[[], Any]) -> Future[Any]:
    """Start a read now and resolve it where its value is first needed.

    `_parallel` reports the failure of its first fetch, which is right for reads
    the sequential code issued next to each other. A read the sequential code
    issued *last* is different: overlapping it must not let its failure overtake
    the earlier ones, or a check would start reporting the certificate when the
    load balancer it belongs to does not exist. Starting it here and calling
    `result()` at the original statement keeps the reason exactly where it was.

    The executor is released immediately: `shutdown(wait=False)` stops it taking
    more work without cancelling what is already running, and if an earlier check
    raises first, the abandoned read is simply never asked for its answer.
    """

    executor = ThreadPoolExecutor(max_workers=1)
    try:
        return executor.submit(fetch)
    finally:
        executor.shutdown(wait=False)


def _check_nlb_runtime(release: Any) -> CheckValue:
    if not release.config.nlb:
        raise CheckSkipped("NLB health configuration is missing")
    name = release.config.nlb.get("name") or (
        f"gpu-fault-regional-{release.config.aws_region}"
    )
    # The certificate is named by configuration, not by anything the load
    # balancer answers, so the read starts here and is collected at the bottom.
    # The other check that wants it joins the same snapshot read.
    certificate = _started(lambda: _certificate_details(release))
    load_balancers = (
        _aws_json(release, ["elbv2", "describe-load-balancers", "--names", name]).get(
            "LoadBalancers"
        )
        or []
    )
    if len(load_balancers) != 1:
        raise ReleaseError(f"NLB {name} was not found")
    nlb = load_balancers[0]
    if (nlb.get("State") or {}).get("Code") != "active":
        raise ReleaseError(f"NLB {name} is not active")
    arn = nlb["LoadBalancerArn"]
    # Listeners and target groups are both keyed by the load balancer ARN and
    # neither depends on the other; only the target health below has to wait.
    listener_document, target_group_document = _parallel(
        [
            lambda: _aws_json(
                release, ["elbv2", "describe-listeners", "--load-balancer-arn", arn]
            ),
            lambda: _aws_json(
                release, ["elbv2", "describe-target-groups", "--load-balancer-arn", arn]
            ),
        ]
    )
    listeners = listener_document.get("Listeners") or []
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
    target_groups = target_group_document.get("TargetGroups") or []
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
            "certificate": certificate.result(),
        },
    )


def build_health_report(release: Any, *, mode: str) -> dict[str, Any]:
    quick_evidence, evidence_fallback = _quick_validation_evidence(release)
    reusable_checks = (
        set(quick_evidence.get("checks") or []) if quick_evidence is not None else set()
    )

    def runtime_component_identity() -> CheckValue:
        if "runtime_component_identity" in reusable_checks:
            assert quick_evidence is not None
            return CheckValue(
                "runtime component identity reused from the current release quick gate",
                {
                    "evidence": os.getenv(QUICK_VALIDATION_EVIDENCE_ENV),
                    "verified_at_epoch": quick_evidence["verified_at_epoch"],
                    "release_state_sha256": quick_evidence["release_state_sha256"],
                },
            )
        return _check_runtime_component_identity(release)

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
        (
            "read_only_verifiers",
            lambda: run_read_only_verifiers(
                release,
                reused_checks=reusable_checks,
            ),
        ),
        (
            "runtime_component_identity",
            runtime_component_identity,
        ),
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
    with _read_snapshot(release):
        _prime_deployment_snapshot(release)
        with ThreadPoolExecutor(max_workers=min(8, len(specifications))) as executor:
            futures = [
                executor.submit(_check, name, function)
                for name, function in specifications
            ]
            checks = [future.result() for future in futures]
    report = _report(mode, release, checks)
    report["reused_validation_checks"] = sorted(reusable_checks)
    if evidence_fallback is not None:
        report["validation_evidence_fallback"] = evidence_fallback
    return report


def _verify_profile(release: Any) -> CheckValue:
    verify_runtime_profile(release)
    return CheckValue(
        "declared Runtime Profile is registered without drift",
        {"version": release.config.runtime_profile_version},
    )

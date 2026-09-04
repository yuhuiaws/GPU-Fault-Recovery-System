from __future__ import annotations

import base64
import copy
import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping, cast

from gpu_fault.admin.bootstrap_common import (
    SITE_TAG_KEY,
    Arn,
    BootstrapError,
    tag_map,
)
from gpu_fault.admin.resource_records import (
    foundation_ownership as _foundation_ownership,
)
from gpu_fault.admin.resource_records import (
    ownership as _ownership,
)
from gpu_fault.admin.resource_records import (
    policy as _policy,
)
from gpu_fault.admin.resource_records import (
    record as _record,
)
from gpu_fault.admin.resource_records import (
    release_repository_resources as _release_repository_resources,
)
from gpu_fault.admin.site import RenderedSite
from gpu_fault.installation_resources import (
    InstallationResource,
    InstallationResourceDeletePolicy,
    InstallationResourceOwnership,
    InstallationResourceSnapshot,
)

SYNC_SCRIPT = r"""
import base64
import json
import os
import sys
import urllib.request

with urllib.request.urlopen(
    "http://127.0.0.1:8080/openapi.json",
    timeout=30,
) as response:
    paths = json.load(response).get("paths", {})
if "/v1/installation-resources/sync" not in paths:
    print("GPU_FAULT_LEGACY_REGISTRY_API", file=sys.stderr)
    raise SystemExit(44)
payload = base64.b64decode(os.environ["GPU_FAULT_INSTALLATION_SNAPSHOT"])
request = urllib.request.Request(
    "http://127.0.0.1:8080/v1/installation-resources/sync",
    data=payload,
    method="POST",
    headers={
        "Content-Type": "application/json",
        "X-GPU-Fault-Execution-Token": os.environ["GPU_FAULT_EXECUTION_TOKEN"],
    },
)
with urllib.request.urlopen(request, timeout=30) as response:
    print(json.dumps(json.load(response), separators=(",", ":")))
"""
FETCH_SCRIPT = r"""
import json
import os
import sys
import urllib.parse
import urllib.request

with urllib.request.urlopen(
    "http://127.0.0.1:8080/openapi.json",
    timeout=30,
) as response:
    paths = json.load(response).get("paths", {})
if "/v1/installation-resources" not in paths:
    print("GPU_FAULT_LEGACY_REGISTRY_API", file=sys.stderr)
    raise SystemExit(44)
site = urllib.parse.quote(os.environ["GPU_FAULT_INSTALLATION_SITE_ID"], safe="")
request = urllib.request.Request(
    "http://127.0.0.1:8080/v1/installation-resources?site_id=" + site,
    headers={
        "X-GPU-Fault-Execution-Token": os.environ["GPU_FAULT_EXECUTION_TOKEN"],
    },
)
with urllib.request.urlopen(request, timeout=30) as response:
    print(json.dumps(json.load(response), separators=(",", ":")))
"""
DIRECT_SYNC_SCRIPT = r'''
import base64
import json
import os

import psycopg

resources = json.loads(
    base64.b64decode(os.environ["GPU_FAULT_INSTALLATION_SNAPSHOT"])
)
with psycopg.connect(os.environ["GPU_FAULT_STORE_URL"]) as connection:
    with connection.cursor() as cursor:
        for resource in resources["resources"]:
            key = resource["site_id"] + "/" + resource["resource_key"]
            cursor.execute(
                """
                INSERT INTO gpu_fault_objects(kind, key, payload)
                VALUES ('installation_resource', %s, %s::jsonb)
                ON CONFLICT(kind, key)
                DO UPDATE SET payload=excluded.payload
                """,
                (key, json.dumps(resource, separators=(",", ":"))),
            )
print(len(resources["resources"]))
'''


class LegacyInstallationRegistryMissing(BootstrapError):
    pass


def _external_clusters(
    site: RenderedSite,
    *,
    site_id: str,
    region: str,
    account_id: str,
) -> list[InstallationResource]:
    config = site.release_config
    cpu_arn = str(config["cpu_eks_arn"])
    resources = [
        _record(
            site_id=site_id,
            resource_key="cluster/cpu-eks",
            resource_type="cpu_eks",
            resource_id=Arn.parse(cpu_arn).resource_name,
            resource_arn=cpu_arn,
            region=region,
            account_id=account_id,
            ownership=InstallationResourceOwnership.EXTERNAL,
            delete_policy=InstallationResourceDeletePolicy.PRESERVE,
        ),
        _record(
            site_id=site_id,
            resource_key="cluster/cpu-hyperpod",
            resource_type="cpu_hyperpod",
            resource_id=str(config["cpu_hyperpod_cluster_name"]),
            region=region,
            account_id=account_id,
            ownership=InstallationResourceOwnership.EXTERNAL,
            delete_policy=InstallationResourceDeletePolicy.PRESERVE,
            dependencies=["cluster/cpu-eks"],
        ),
    ]
    for cluster in config["clusters"]:
        cluster_id = str(cluster["cluster_id"])
        eks_arn = str(cluster["eks_cluster_arn"])
        resources.extend(
            [
                _record(
                    site_id=site_id,
                    resource_key=f"cluster/{cluster_id}/eks",
                    resource_type="gpu_eks",
                    resource_id=Arn.parse(eks_arn).resource_name,
                    resource_arn=eks_arn,
                    region=region,
                    account_id=account_id,
                    ownership=InstallationResourceOwnership.EXTERNAL,
                    delete_policy=InstallationResourceDeletePolicy.PRESERVE,
                ),
                _record(
                    site_id=site_id,
                    resource_key=f"cluster/{cluster_id}/hyperpod",
                    resource_type="gpu_hyperpod",
                    resource_id=str(cluster["hyperpod_cluster_name"]),
                    region=region,
                    account_id=account_id,
                    ownership=InstallationResourceOwnership.EXTERNAL,
                    delete_policy=InstallationResourceDeletePolicy.PRESERVE,
                    dependencies=[f"cluster/{cluster_id}/eks"],
                ),
            ]
        )
    return resources


def _network_resources(
    *,
    site_id: str,
    region: str,
    account_id: str,
    config: Mapping[str, Any],
    state: Mapping[str, Any],
    runtime: Mapping[str, Any],
) -> list[InstallationResource]:
    resources: list[InstallationResource] = []
    nlb = state.get("nlb_network") or {}
    subnet_keys = []
    for index, subnet in enumerate(nlb.get("subnet_resources") or (), 1):
        ownership = _foundation_ownership(subnet.get("ownership"))
        subnet_key = f"aws/network/public-subnet/{index}"
        subnet_keys.append(subnet_key)
        resources.append(
            _record(
                site_id=site_id,
                resource_key=subnet_key,
                resource_type="ec2_subnet",
                resource_id=str(subnet["subnet_id"]),
                region=region,
                account_id=account_id,
                ownership=ownership,
                delete_policy=_policy(ownership),
                attributes={
                    "vpc_id": subnet.get("vpc_id"),
                    "availability_zone": subnet.get("availability_zone"),
                },
            )
        )
        route_table = subnet.get("route_table_id")
        association = subnet.get("route_table_association_id")
        if route_table:
            route_key = f"aws/network/public-route-table/{index}"
            resources.append(
                _record(
                    site_id=site_id,
                    resource_key=route_key,
                    resource_type="ec2_route_table",
                    resource_id=str(route_table),
                    region=region,
                    account_id=account_id,
                    ownership=ownership,
                    delete_policy=_policy(ownership),
                    attributes={"vpc_id": subnet.get("vpc_id")},
                )
            )
            if association:
                resources.append(
                    _record(
                        site_id=site_id,
                        resource_key=(f"aws/network/public-route-association/{index}"),
                        resource_type="ec2_route_table_association",
                        resource_id=str(association),
                        region=region,
                        account_id=account_id,
                        ownership=ownership,
                        delete_policy=_policy(
                            ownership,
                            created=InstallationResourceDeletePolicy.DETACH,
                        ),
                        dependencies=[subnet_key, route_key],
                    )
                )
    if not subnet_keys:
        raw_subnets = nlb.get("public_subnets")
        if not raw_subnets:
            raw_subnets = str(config["nlb"].get("public_subnets") or "").split(",")
        for index, subnet_id in enumerate(raw_subnets, 1):
            if not subnet_id:
                continue
            subnet_key = f"aws/network/public-subnet/{index}"
            subnet_keys.append(subnet_key)
            resources.append(
                _record(
                    site_id=site_id,
                    resource_key=subnet_key,
                    resource_type="ec2_subnet",
                    resource_id=str(subnet_id),
                    region=region,
                    account_id=account_id,
                    ownership=InstallationResourceOwnership.EXTERNAL,
                    delete_policy=InstallationResourceDeletePolicy.PRESERVE,
                )
            )
    gateway = nlb.get("internet_gateway") or {}
    if gateway.get("internet_gateway_id"):
        ownership = _foundation_ownership(gateway.get("ownership"))
        resources.append(
            _record(
                site_id=site_id,
                resource_key="aws/network/internet-gateway",
                resource_type="internet_gateway",
                resource_id=str(gateway["internet_gateway_id"]),
                region=region,
                account_id=account_id,
                ownership=ownership,
                delete_policy=_policy(ownership),
                attributes={"vpc_id": gateway.get("vpc_id")},
            )
        )
    security_group_id = nlb.get("security_group") or config["nlb"].get("security_group")
    if security_group_id:
        ownership = _ownership(
            nlb.get("security_group_ownership"),
            default=(
                InstallationResourceOwnership.CREATED
                if nlb
                else InstallationResourceOwnership.EXTERNAL
            ),
        )
        resources.append(
            _record(
                site_id=site_id,
                resource_key="aws/nlb/security-group",
                resource_type="security_group",
                resource_id=str(security_group_id),
                region=region,
                account_id=account_id,
                ownership=ownership,
                delete_policy=_policy(ownership),
                attributes={"vpc_id": nlb.get("vpc_id")},
            )
        )
    nlb_details = runtime.get("nlb") or {}
    nlb_dependencies = [
        *subnet_keys,
        *(["aws/nlb/security-group"] if security_group_id else []),
    ]
    resources.append(
        _record(
            site_id=site_id,
            resource_key="aws/nlb",
            resource_type="nlb",
            resource_id=str(config["nlb"]["name"]),
            resource_arn=nlb_details.get("arn"),
            region=region,
            account_id=account_id,
            ownership=InstallationResourceOwnership.CREATED,
            delete_policy=InstallationResourceDeletePolicy.DELETE,
            dependencies=nlb_dependencies,
            attributes={"dns_name": nlb_details.get("dns_name")},
        )
    )
    for index, listener in enumerate(nlb_details.get("listeners") or (), 1):
        resources.append(
            _record(
                site_id=site_id,
                resource_key=f"aws/nlb/listener/{index}",
                resource_type="nlb_listener",
                resource_id=str(listener),
                resource_arn=str(listener),
                region=region,
                account_id=account_id,
                ownership=InstallationResourceOwnership.CREATED,
                delete_policy=InstallationResourceDeletePolicy.DELETE,
                dependencies=["aws/nlb"],
            )
        )
    for index, target_group in enumerate(
        nlb_details.get("target_groups") or (),
        1,
    ):
        resources.append(
            _record(
                site_id=site_id,
                resource_key=f"aws/nlb/target-group/{index}",
                resource_type="nlb_target_group",
                resource_id=str(target_group),
                resource_arn=str(target_group),
                region=region,
                account_id=account_id,
                ownership=InstallationResourceOwnership.CREATED,
                delete_policy=InstallationResourceDeletePolicy.DELETE,
                dependencies=["aws/nlb"],
            )
        )
    return resources


def _pki_resources(
    *,
    site_id: str,
    region: str,
    account_id: str,
    config: Mapping[str, Any],
    state: Mapping[str, Any],
) -> list[InstallationResource]:
    resources: list[InstallationResource] = []
    pki = state.get("pki") or {}
    dns = config.get("dns") or {}
    zone_id = pki.get("hosted_zone_id") or dns.get("hosted_zone_id")
    zone_ownership = _ownership(
        pki.get("zone_ownership"),
        default=(
            InstallationResourceOwnership.CREATED
            if pki
            else InstallationResourceOwnership.EXTERNAL
        ),
    )
    if zone_id:
        resources.append(
            _record(
                site_id=site_id,
                resource_key="aws/route53/zone",
                resource_type="route53_zone",
                resource_id=str(zone_id),
                region=region,
                account_id=account_id,
                ownership=zone_ownership,
                delete_policy=_policy(zone_ownership),
                attributes={"zone_name": pki.get("zone_name")},
            )
        )
    hostname = pki.get("hostname") or dns.get("hostname")
    if zone_id and hostname:
        resources.append(
            _record(
                site_id=site_id,
                resource_key="aws/route53/control-plane-record",
                resource_type="route53_record",
                resource_id=str(hostname),
                region=region,
                account_id=account_id,
                ownership=InstallationResourceOwnership.CREATED,
                delete_policy=InstallationResourceDeletePolicy.DELETE,
                dependencies=["aws/route53/zone", "aws/nlb"],
                attributes={
                    "hosted_zone_id": zone_id,
                    "record_type": "CNAME",
                },
            )
        )
    for index, association in enumerate(pki.get("vpc_associations") or (), 1):
        ownership = _ownership(
            association.get("ownership"),
            default=InstallationResourceOwnership.CREATED,
        )
        if zone_ownership is InstallationResourceOwnership.CREATED:
            continue
        resources.append(
            _record(
                site_id=site_id,
                resource_key=f"aws/route53/vpc-association/{index}",
                resource_type="route53_vpc_association",
                resource_id=(
                    f"{zone_id}:{association.get('vpc_region')}:"
                    f"{association.get('vpc_id')}"
                ),
                region=region,
                account_id=account_id,
                ownership=ownership,
                delete_policy=_policy(
                    ownership,
                    created=InstallationResourceDeletePolicy.DETACH,
                ),
                dependencies=["aws/route53/zone"],
                attributes={
                    "hosted_zone_id": zone_id,
                    "vpc_id": association.get("vpc_id"),
                    "vpc_region": association.get("vpc_region"),
                },
            )
        )
    certificate = pki.get("certificate_arn") or config["nlb"].get("certificate_arn")
    if certificate:
        ownership = _ownership(
            pki.get("certificate_ownership"),
            default=(
                InstallationResourceOwnership.CREATED
                if pki.get("certificate_arn")
                else InstallationResourceOwnership.EXTERNAL
            ),
        )
        resources.append(
            _record(
                site_id=site_id,
                resource_key="aws/acm/certificate",
                resource_type="acm_certificate",
                resource_id=str(certificate),
                resource_arn=str(certificate),
                region=region,
                account_id=account_id,
                ownership=ownership,
                delete_policy=_policy(ownership),
            )
        )
    if pki.get("pki_secret_id"):
        ownership = _ownership(
            pki.get("pki_secret_ownership"),
            default=InstallationResourceOwnership.CREATED,
        )
        resources.append(
            _record(
                site_id=site_id,
                resource_key="aws/secrets/pki",
                resource_type="secretsmanager_secret",
                resource_id=str(pki["pki_secret_id"]),
                region=region,
                account_id=account_id,
                ownership=ownership,
                delete_policy=_policy(ownership),
            )
        )
    return resources


def _monitoring_resources(
    *,
    site_id: str,
    region: str,
    account_id: str,
    config: Mapping[str, Any],
    state: Mapping[str, Any],
) -> list[InstallationResource]:
    resources: list[InstallationResource] = []
    monitoring = state.get("monitoring_resources") or {}
    values = (
        (
            "aws/amp/workspace",
            "amp_workspace",
            monitoring.get("workspace_id") or config["health"].get("amp_workspace_id"),
            monitoring.get("workspace_ownership"),
            monitoring.get("workspace_id") is not None,
        ),
        (
            "aws/sns/topic",
            "sns_topic",
            monitoring.get("sns_topic_arn") or config["health"].get("sns_topic_arn"),
            monitoring.get("sns_topic_ownership"),
            monitoring.get("sns_topic_arn") is not None,
        ),
        (
            "aws/sqs/queue",
            "sqs_queue",
            monitoring.get("sqs_queue_url"),
            monitoring.get("sqs_queue_ownership"),
            monitoring.get("sqs_queue_url") is not None,
        ),
    )
    for key, resource_type, value, raw_ownership, bootstrap_owned in values:
        if not value:
            continue
        ownership = _ownership(
            raw_ownership,
            default=(
                InstallationResourceOwnership.CREATED
                if bootstrap_owned
                else InstallationResourceOwnership.EXTERNAL
            ),
        )
        resources.append(
            _record(
                site_id=site_id,
                resource_key=key,
                resource_type=resource_type,
                resource_id=str(value),
                resource_arn=(str(value) if str(value).startswith("arn:") else None),
                region=region,
                account_id=account_id,
                ownership=ownership,
                delete_policy=_policy(ownership),
            )
        )
    subscription_arn = monitoring.get("queue_subscription_arn")
    if subscription_arn and subscription_arn != "PendingConfirmation":
        ownership = _ownership(
            monitoring.get("queue_subscription_ownership"),
            default=InstallationResourceOwnership.CREATED,
        )
        resources.append(
            _record(
                site_id=site_id,
                resource_key="aws/sns/queue-subscription",
                resource_type="sns_subscription",
                resource_id=str(subscription_arn),
                resource_arn=str(subscription_arn),
                region=region,
                account_id=account_id,
                ownership=ownership,
                delete_policy=_policy(
                    ownership,
                    created=InstallationResourceDeletePolicy.DETACH,
                ),
                dependencies=["aws/sns/topic", "aws/sqs/queue"],
            )
        )
    email_subscription_arn = monitoring.get("email_subscription_arn")
    email_endpoint = str(monitoring.get("email_subscription_endpoint") or "")
    if (
        email_subscription_arn
        and email_subscription_arn != "PendingConfirmation"
        and email_endpoint
    ):
        endpoint_digest = hashlib.sha256(email_endpoint.casefold().encode()).hexdigest()
        resources.append(
            _record(
                site_id=site_id,
                resource_key=("aws/sns/email-subscription/" + endpoint_digest[:16]),
                resource_type="sns_subscription",
                resource_id=str(email_subscription_arn),
                resource_arn=str(email_subscription_arn),
                region=region,
                account_id=account_id,
                ownership=InstallationResourceOwnership.CREATED,
                delete_policy=InstallationResourceDeletePolicy.DETACH,
                dependencies=["aws/sns/topic"],
                attributes={
                    "endpoint_sha256": endpoint_digest,
                    "status": monitoring.get("email_subscription_status"),
                    "topic_generation": monitoring.get("sns_topic_generation"),
                },
            )
        )
    if monitoring.get("sqs_queue_url") and monitoring.get("sns_topic_arn"):
        resources.append(
            _record(
                site_id=site_id,
                resource_key="aws/sqs/topic-policy-binding",
                resource_type="sqs_policy_binding",
                resource_id=str(monitoring["sqs_queue_url"]),
                region=region,
                account_id=account_id,
                ownership=InstallationResourceOwnership.CREATED,
                delete_policy=InstallationResourceDeletePolicy.DETACH,
                dependencies=["aws/sns/topic", "aws/sqs/queue"],
                attributes={"topic_arn": monitoring["sns_topic_arn"]},
            )
        )
    return resources


def _role_resources(
    *,
    site_id: str,
    region: str,
    account_id: str,
    state: Mapping[str, Any],
) -> list[InstallationResource]:
    resources: list[InstallationResource] = []

    def add_role(
        key: str,
        value: Mapping[str, Any],
        *,
        ownership_key: str = "ownership",
    ) -> None:
        role_arn = value.get("role_arn")
        if not role_arn:
            return
        ownership = _ownership(
            value.get(ownership_key),
            default=InstallationResourceOwnership.CREATED,
        )
        role_key = f"aws/iam/{key}/role"
        resources.append(
            _record(
                site_id=site_id,
                resource_key=role_key,
                resource_type="iam_role",
                resource_id=str(role_arn).rsplit("/", 1)[-1],
                resource_arn=str(role_arn),
                region=region,
                account_id=account_id,
                ownership=ownership,
                delete_policy=_policy(ownership),
                attributes={
                    "inline_policy_name": value.get("inline_policy_name"),
                },
            )
        )
        association_id = value.get("association_id")
        if association_id:
            association_ownership = _ownership(
                value.get("association_ownership"),
                default=InstallationResourceOwnership.CREATED,
            )
            resources.append(
                _record(
                    site_id=site_id,
                    resource_key=f"aws/iam/{key}/pod-identity-association",
                    resource_type="eks_pod_identity_association",
                    resource_id=str(association_id),
                    region=region,
                    account_id=account_id,
                    ownership=association_ownership,
                    delete_policy=InstallationResourceDeletePolicy.DETACH,
                    dependencies=[role_key],
                    attributes={
                        "cluster_name": value.get("cluster_name"),
                        "namespace": value.get("namespace"),
                        "service_account": value.get("service_account"),
                    },
                )
            )

    control = state.get("control_plane_role") or {}
    add_role("control-plane", control)
    refresh = state.get("aurora_refresh") or {}
    add_role("aurora-refresh", refresh)
    monitoring = state.get("monitoring_install") or {}
    add_role("adot", monitoring)
    for name, value in state.items():
        if not name.startswith("executor_role:"):
            continue
        cluster_id = name.removeprefix("executor_role:")
        add_role(f"executor/{cluster_id}", value or {})
        provider_arn = (value or {}).get("oidc_provider_arn")
        if provider_arn:
            ownership = _foundation_ownership(
                (value or {}).get("oidc_provider_ownership")
            )
            resources.append(
                _record(
                    site_id=site_id,
                    resource_key=f"aws/iam/executor/{cluster_id}/oidc-provider",
                    resource_type="iam_oidc_provider",
                    resource_id=str(provider_arn),
                    resource_arn=str(provider_arn),
                    region=region,
                    account_id=account_id,
                    ownership=ownership,
                    delete_policy=_policy(
                        ownership,
                        created=InstallationResourceDeletePolicy.DETACH,
                    ),
                    attributes={
                        "cluster_name": (value or {}).get("cluster_name"),
                    },
                )
            )
    lbc = state.get("load_balancer_controller") or {}
    if lbc.get("external") is True:
        pass
    elif lbc.get("reused") is True:
        resources.append(
            _record(
                site_id=site_id,
                resource_key="kubernetes/lbc/helm-release",
                resource_type="helm_release",
                resource_id="aws-load-balancer-controller",
                provider="kubernetes",
                region=region,
                account_id=account_id,
                ownership=InstallationResourceOwnership.CREATED,
                delete_policy=InstallationResourceDeletePolicy.DELETE,
                attributes={"namespace": "kube-system"},
            )
        )
    elif lbc:
        add_role("lbc", lbc, ownership_key="role_ownership")
        resources.append(
            _record(
                site_id=site_id,
                resource_key="kubernetes/lbc/helm-release",
                resource_type="helm_release",
                resource_id="aws-load-balancer-controller",
                provider="kubernetes",
                region=region,
                account_id=account_id,
                ownership=InstallationResourceOwnership.CREATED,
                delete_policy=InstallationResourceDeletePolicy.DELETE,
                dependencies=["aws/iam/lbc/role"],
                attributes={"namespace": "kube-system"},
            )
        )
        policy_arn = lbc.get("policy_arn")
        if policy_arn:
            ownership = _ownership(
                lbc.get("policy_ownership"),
                default=InstallationResourceOwnership.CREATED,
            )
            resources.append(
                _record(
                    site_id=site_id,
                    resource_key="aws/iam/lbc/policy",
                    resource_type="iam_policy",
                    resource_id=str(policy_arn),
                    resource_arn=str(policy_arn),
                    region=region,
                    account_id=account_id,
                    ownership=ownership,
                    delete_policy=_policy(ownership),
                    dependencies=["aws/iam/lbc/role"],
                )
            )
    addon = state.get("pod_identity_agent") or {}
    if addon.get("addon_name"):
        ownership = _foundation_ownership(addon.get("ownership"))
        resources.append(
            _record(
                site_id=site_id,
                resource_key="aws/eks/pod-identity-agent",
                resource_type="eks_addon",
                resource_id=str(addon["addon_name"]),
                region=region,
                account_id=account_id,
                ownership=ownership,
                delete_policy=_policy(ownership),
                attributes={"cluster_name": addon.get("cluster_name")},
            )
        )
    return resources


def _email_resources(
    *,
    site_id: str,
    region: str,
    account_id: str,
    state: Mapping[str, Any],
) -> list[InstallationResource]:
    email = state.get("email_notifications") or {}
    identity = email.get("identity")
    if not identity:
        return []
    ownership = _ownership(email.get("identity_ownership"))
    return [
        _record(
            site_id=site_id,
            resource_key="aws/ses/administrator-email-identity",
            resource_type="ses_email_identity",
            resource_id=str(identity),
            resource_arn=str(email.get("identity_arn") or ""),
            region=region,
            account_id=account_id,
            ownership=ownership,
            delete_policy=InstallationResourceDeletePolicy.PRESERVE,
            attributes={
                "verified": bool(email.get("verified")),
                "production_access_enabled": bool(
                    email.get("production_access_enabled")
                ),
            },
        )
    ]


def _aurora_resources(
    *,
    site_id: str,
    region: str,
    account_id: str,
    config: Mapping[str, Any],
    state: Mapping[str, Any],
) -> list[InstallationResource]:
    resources: list[InstallationResource] = []
    aurora = state.get("aurora") or {}
    cluster_id = aurora.get("cluster_id") or config["health"].get("aurora_cluster_id")
    if not cluster_id:
        return resources
    ownership = _ownership(
        aurora.get("cluster_ownership"),
        default=(
            InstallationResourceOwnership.CREATED
            if aurora
            else InstallationResourceOwnership.EXTERNAL
        ),
    )
    cluster_dependencies = []
    subnet_group = aurora.get("subnet_group")
    if subnet_group:
        subnet_ownership = _ownership(
            aurora.get("subnet_group_ownership"),
            default=InstallationResourceOwnership.CREATED,
        )
        cluster_dependencies.append("aws/aurora/subnet-group")
        resources.append(
            _record(
                site_id=site_id,
                resource_key="aws/aurora/subnet-group",
                resource_type="rds_db_subnet_group",
                resource_id=str(subnet_group),
                region=region,
                account_id=account_id,
                ownership=subnet_ownership,
                delete_policy=_policy(subnet_ownership),
            )
        )
    security_group = aurora.get("security_group")
    if security_group:
        security_ownership = _ownership(
            aurora.get("security_group_ownership"),
            default=InstallationResourceOwnership.CREATED,
        )
        cluster_dependencies.append("aws/aurora/security-group")
        resources.append(
            _record(
                site_id=site_id,
                resource_key="aws/aurora/security-group",
                resource_type="security_group",
                resource_id=str(security_group),
                region=region,
                account_id=account_id,
                ownership=security_ownership,
                delete_policy=_policy(security_ownership),
            )
        )
    resources.append(
        _record(
            site_id=site_id,
            resource_key="aws/aurora/cluster",
            resource_type="aurora_cluster",
            resource_id=str(cluster_id),
            region=region,
            account_id=account_id,
            ownership=ownership,
            delete_policy=_policy(ownership),
            dependencies=cluster_dependencies,
        )
    )
    for instance_id in aurora.get("instance_ids") or ():
        resources.append(
            _record(
                site_id=site_id,
                resource_key=f"aws/aurora/instance/{instance_id}",
                resource_type="aurora_instance",
                resource_id=str(instance_id),
                region=region,
                account_id=account_id,
                ownership=ownership,
                delete_policy=_policy(ownership),
                dependencies=["aws/aurora/cluster"],
            )
        )
    master_arn = aurora.get("master_secret_arn")
    if master_arn:
        resources.append(
            _record(
                site_id=site_id,
                resource_key="aws/aurora/managed-master",
                resource_type="rds_managed_secret",
                resource_id=str(master_arn),
                resource_arn=str(master_arn),
                region=region,
                account_id=account_id,
                ownership=ownership,
                delete_policy=_policy(ownership),
                dependencies=["aws/aurora/cluster"],
            )
        )
    return resources


def build_installation_snapshot(
    site: RenderedSite,
    bootstrap: Mapping[str, Any] | None,
    runtime: Mapping[str, Any] | None = None,
) -> InstallationResourceSnapshot:
    config = site.release_config
    site_id = str(config["site_name"])
    region = str(config["aws_region"])
    account_id = Arn.parse(str(config["cpu_eks_arn"])).account
    state = (bootstrap or {}).get("resources") or {}
    runtime_resources = runtime or {}
    resources = _external_clusters(
        site,
        site_id=site_id,
        region=region,
        account_id=account_id,
    )
    resources.extend(
        _release_repository_resources(
            site_id=site_id,
            region=region,
            account_id=account_id,
            state=state,
        )
    )
    resources.extend(
        _network_resources(
            site_id=site_id,
            region=region,
            account_id=account_id,
            config=config,
            state=state,
            runtime=runtime_resources,
        )
    )
    resources.extend(
        _pki_resources(
            site_id=site_id,
            region=region,
            account_id=account_id,
            config=config,
            state=state,
        )
    )
    resources.extend(
        _monitoring_resources(
            site_id=site_id,
            region=region,
            account_id=account_id,
            config=config,
            state=state,
        )
    )
    resources.extend(
        _role_resources(
            site_id=site_id,
            region=region,
            account_id=account_id,
            state=state,
        )
    )
    resources.extend(
        _email_resources(
            site_id=site_id,
            region=region,
            account_id=account_id,
            state=state,
        )
    )
    resources.extend(
        _aurora_resources(
            site_id=site_id,
            region=region,
            account_id=account_id,
            config=config,
            state=state,
        )
    )
    snapshot = InstallationResourceSnapshot(site_id=site_id, resources=resources)
    return cast(
        InstallationResourceSnapshot,
        snapshot.model_copy(update={"source_sha256": snapshot.digest()}),
    )


def _aws_json(region: str, *arguments: str) -> dict[str, Any]:
    completed = subprocess.run(
        ["aws", *arguments, "--region", region, "--output", "json"],
        text=True,
        capture_output=True,
    )
    if completed.returncode:
        operation = " ".join(arguments[:2])
        raise BootstrapError(
            f"failed to discover deployed AWS resource ({operation}): "
            f"{completed.stderr.strip()}"
        )
    return cast(dict[str, Any], json.loads(completed.stdout or "{}"))


def discover_runtime_resources(site: RenderedSite) -> dict[str, Any]:
    region = str(site.release_config["aws_region"])
    name = str(site.release_config["nlb"]["name"])
    load_balancers = _aws_json(
        region,
        "elbv2",
        "describe-load-balancers",
        "--names",
        name,
    ).get("LoadBalancers", [])
    if len(load_balancers) != 1:
        raise BootstrapError(
            f"deployed NLB {name!r} was not resolved to exactly one load balancer"
        )
    load_balancer = load_balancers[0]
    arn = str(load_balancer["LoadBalancerArn"])
    listeners = _aws_json(
        region,
        "elbv2",
        "describe-listeners",
        "--load-balancer-arn",
        arn,
    ).get("Listeners", [])
    target_groups = _aws_json(
        region,
        "elbv2",
        "describe-target-groups",
        "--load-balancer-arn",
        arn,
    ).get("TargetGroups", [])
    return {
        "nlb": {
            "arn": arn,
            "dns_name": load_balancer.get("DNSName"),
            "listeners": [
                item["ListenerArn"] for item in listeners if item.get("ListenerArn")
            ],
            "target_groups": [
                item["TargetGroupArn"]
                for item in target_groups
                if item.get("TargetGroupArn")
            ],
        }
    }


def _cpu_pod(site: RenderedSite) -> str:
    result = subprocess.run(
        [
            "kubectl",
            "--kubeconfig",
            site.release_config["cpu_kubeconfig"],
            "-n",
            site.release_config["namespace"],
            "get",
            "pod",
            "-l",
            "app=gpu-fault-api-ha",
            "--field-selector=status.phase=Running",
            "-o",
            "jsonpath={.items[0].metadata.name}",
        ],
        text=True,
        capture_output=True,
    )
    pod = (result.stdout or "").strip()
    if result.returncode or not pod:
        raise BootstrapError("no Running CPU ingress Pod for registry synchronization")
    return pod


def _snapshot_payload(snapshot: InstallationResourceSnapshot) -> bytes:
    return json.dumps(
        snapshot.model_dump(mode="json"),
        separators=(",", ":"),
    ).encode()


def write_installation_resource_snapshot(
    site: RenderedSite,
    snapshot: InstallationResourceSnapshot,
    *,
    path: Path | None = None,
) -> Path:
    target = path or site.source.parent / "installation-resources.json"
    target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    payload = json.dumps(
        snapshot.model_dump(mode="json"),
        indent=2,
        sort_keys=True,
    )
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.chmod(0o600)
    temporary.replace(target)
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    digest_path = target.with_suffix(target.suffix + ".sha256")
    digest_path.write_text(f"{digest}  {target.name}\n", encoding="utf-8")
    digest_path.chmod(0o600)
    return target


def load_installation_resource_snapshot(path: Path) -> InstallationResourceSnapshot:
    digest_path = path.with_suffix(path.suffix + ".sha256")
    if not path.is_file() or not digest_path.is_file():
        raise BootstrapError(f"installation registry snapshot is incomplete: {path}")
    expected = digest_path.read_text(encoding="utf-8").split()[0]
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise BootstrapError(f"installation registry snapshot digest mismatch: {path}")
    return cast(
        InstallationResourceSnapshot,
        InstallationResourceSnapshot.model_validate_json(
            path.read_text(encoding="utf-8")
        ),
    )


def find_bootstrap_state(site: RenderedSite) -> dict[str, Any] | None:
    site_id = str(site.release_config["site_name"])
    candidates = [site.source.parent / "bootstrap-state.json"]
    candidates.extend(
        sorted((Path.home() / ".gpu-fault/bootstrap").glob("*/bootstrap-state.json"))
    )
    matches = []
    for path in candidates:
        if not path.is_file():
            continue
        document = cast(
            dict[str, Any],
            json.loads(path.read_text(encoding="utf-8")),
        )
        if document.get("site_id") == site_id:
            matches.append((path.resolve(), document))
    unique = {path: document for path, document in matches}
    if not unique:
        return None
    if len(unique) > 1:
        locations = ", ".join(str(path) for path in sorted(unique))
        raise BootstrapError(
            "multiple bootstrap states match this site; remove stale copies: "
            + locations
        )
    return next(iter(unique.values()))


def build_legacy_installation_snapshot(
    site: RenderedSite,
) -> InstallationResourceSnapshot:
    bootstrap = find_bootstrap_state(site)
    if bootstrap is None:
        raise BootstrapError(
            "legacy deployment has no Aurora resource registry and no matching "
            "bootstrap-state.json; ownership cannot be proven automatically"
        )
    bootstrap = enrich_legacy_bootstrap(site, bootstrap)
    snapshot = build_installation_snapshot(
        site,
        bootstrap,
        discover_runtime_resources(site),
    )
    ambiguous = sorted(
        resource.resource_key
        for resource in snapshot.resources
        if resource.ownership is InstallationResourceOwnership.EXTERNAL
        and resource.resource_type
        not in {
            "cpu_eks",
            "cpu_hyperpod",
            "gpu_eks",
            "gpu_hyperpod",
            "ec2_subnet",
            "internet_gateway",
            "iam_oidc_provider",
            "eks_addon",
            "amp_workspace",
        }
    )
    if ambiguous:
        raise BootstrapError(
            "legacy resource ownership is ambiguous: " + ", ".join(ambiguous)
        )
    return snapshot


def enrich_legacy_bootstrap(
    site: RenderedSite,
    bootstrap: Mapping[str, Any],
) -> dict[str, Any]:
    enriched = copy.deepcopy(dict(bootstrap))
    resources = enriched.setdefault("resources", {})
    nlb = resources.get("nlb_network") or {}
    region = str(site.release_config["aws_region"])
    site_id = str(site.release_config["site_name"])
    subnet_ids = list(nlb.get("public_subnets") or ())
    if subnet_ids and not nlb.get("subnet_resources"):
        subnets = _aws_json(
            region,
            "ec2",
            "describe-subnets",
            "--subnet-ids",
            *subnet_ids,
        ).get("Subnets", [])
        subnet_resources = []
        for subnet in subnets:
            owned = tag_map(subnet.get("Tags")).get(SITE_TAG_KEY) == site_id
            item: dict[str, Any] = {
                "subnet_id": subnet["SubnetId"],
                "availability_zone": subnet["AvailabilityZone"],
                "ownership": "CREATED" if owned else "EXTERNAL",
                "vpc_id": subnet["VpcId"],
            }
            if owned:
                tables = _aws_json(
                    region,
                    "ec2",
                    "describe-route-tables",
                    "--filters",
                    f"Name=association.subnet-id,Values={subnet['SubnetId']}",
                ).get("RouteTables", [])
                if len(tables) != 1:
                    raise BootstrapError(
                        "cannot reconstruct route table for legacy subnet "
                        + str(subnet["SubnetId"])
                    )
                associations = [
                    association
                    for association in tables[0].get("Associations", [])
                    if association.get("SubnetId") == subnet["SubnetId"]
                ]
                if len(associations) != 1:
                    raise BootstrapError(
                        "cannot reconstruct route association for legacy subnet "
                        + str(subnet["SubnetId"])
                    )
                item["route_table_id"] = tables[0]["RouteTableId"]
                item["route_table_association_id"] = associations[0][
                    "RouteTableAssociationId"
                ]
            subnet_resources.append(item)
        nlb["subnet_resources"] = subnet_resources
    security_group = nlb.get("security_group")
    if security_group and not nlb.get("security_group_ownership"):
        groups = _aws_json(
            region,
            "ec2",
            "describe-security-groups",
            "--group-ids",
            str(security_group),
        ).get("SecurityGroups", [])
        if len(groups) != 1:
            raise BootstrapError(
                f"cannot reconstruct NLB security group {security_group}"
            )
        nlb["security_group_ownership"] = (
            "CREATED"
            if tag_map(groups[0].get("Tags")).get(SITE_TAG_KEY) == site_id
            else "EXTERNAL"
        )
    cpu_name = Arn.parse(str(site.release_config["cpu_eks_arn"])).resource_name
    cluster = _aws_json(
        region,
        "eks",
        "describe-cluster",
        "--name",
        cpu_name,
    ).get("cluster", {})
    vpc_id = (cluster.get("resourcesVpcConfig") or {}).get("vpcId")
    if vpc_id and not nlb.get("internet_gateway"):
        gateways = _aws_json(
            region,
            "ec2",
            "describe-internet-gateways",
            "--filters",
            f"Name=attachment.vpc-id,Values={vpc_id}",
        ).get("InternetGateways", [])
        if len(gateways) == 1:
            gateway = gateways[0]
            nlb["internet_gateway"] = {
                "internet_gateway_id": gateway["InternetGatewayId"],
                "ownership": (
                    "CREATED"
                    if tag_map(gateway.get("Tags")).get(SITE_TAG_KEY) == site_id
                    else "EXTERNAL"
                ),
                "vpc_id": vpc_id,
            }
    resources["nlb_network"] = nlb
    return enriched


def sync_installation_resource_snapshot(
    site: RenderedSite,
    snapshot: InstallationResourceSnapshot,
) -> None:
    encoded = base64.b64encode(_snapshot_payload(snapshot)).decode()
    command = [
        "kubectl",
        "--kubeconfig",
        site.release_config["cpu_kubeconfig"],
        "-n",
        site.release_config["namespace"],
        "exec",
        _cpu_pod(site),
        "--",
        "env",
        f"GPU_FAULT_INSTALLATION_SNAPSHOT={encoded}",
        "python",
        "-c",
        SYNC_SCRIPT,
    ]
    delays = (2, 4, 8)
    for attempt in range(len(delays) + 1):
        result = subprocess.run(
            command,
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode == 0:
            return
        message = (result.stdout or "") + "\n" + (result.stderr or "")
        if (
            "GPU_FAULT_LEGACY_REGISTRY_API" in message
            or "HTTP Error 404" in message
            or "404 Not Found" in message
        ):
            raise LegacyInstallationRegistryMissing(
                "control plane predates the installation resource registry API"
            )
        if "HTTP Error 503" not in message:
            raise BootstrapError(
                "installation registry synchronization failed: "
                + (result.stderr or "").strip()
            )
        if attempt < len(delays):
            time.sleep(delays[attempt])
    sync_installation_resource_snapshot_direct(site, snapshot)


def sync_installation_resource_snapshot_direct(
    site: RenderedSite,
    snapshot: InstallationResourceSnapshot,
) -> None:
    encoded = base64.b64encode(_snapshot_payload(snapshot)).decode()
    result = subprocess.run(
        [
            "kubectl",
            "--kubeconfig",
            site.release_config["cpu_kubeconfig"],
            "-n",
            site.release_config["namespace"],
            "exec",
            _cpu_pod(site),
            "--",
            "env",
            f"GPU_FAULT_INSTALLATION_SNAPSHOT={encoded}",
            "python",
            "-c",
            DIRECT_SYNC_SCRIPT,
        ],
        text=True,
        capture_output=True,
    )
    if result.returncode:
        raise BootstrapError(
            "direct Aurora registry synchronization failed: "
            + (result.stderr or "").strip()
        )


def sync_installation_resource_registry(site: RenderedSite) -> Path:
    bootstrap = find_bootstrap_state(site)
    snapshot = build_installation_snapshot(
        site,
        bootstrap,
        discover_runtime_resources(site),
    )
    sync_installation_resource_snapshot(site, snapshot)
    return write_installation_resource_snapshot(site, snapshot)


def fetch_installation_resource_registry(
    site: RenderedSite,
    *,
    output: Path | None = None,
) -> InstallationResourceSnapshot:
    result = subprocess.run(
        [
            "kubectl",
            "--kubeconfig",
            site.release_config["cpu_kubeconfig"],
            "-n",
            site.release_config["namespace"],
            "exec",
            _cpu_pod(site),
            "--",
            "env",
            ("GPU_FAULT_INSTALLATION_SITE_ID=" + str(site.release_config["site_name"])),
            "python",
            "-c",
            FETCH_SCRIPT,
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode:
        message = (result.stdout or "") + "\n" + (result.stderr or "")
        if (
            "GPU_FAULT_LEGACY_REGISTRY_API" in message
            or "HTTP Error 404" in message
            or "404 Not Found" in message
        ):
            raise LegacyInstallationRegistryMissing(
                "control plane predates the installation resource registry API"
            )
        raise BootstrapError(
            "installation registry fetch failed: " + (result.stderr or "").strip()
        )
    resources = [
        InstallationResource.model_validate(item)
        for item in json.loads(result.stdout or "[]")
    ]
    if not resources:
        raise LegacyInstallationRegistryMissing(
            "Aurora installation resource registry is empty"
        )
    snapshot = InstallationResourceSnapshot(
        site_id=str(site.release_config["site_name"]),
        resources=resources,
    )
    snapshot = cast(
        InstallationResourceSnapshot,
        snapshot.model_copy(update={"source_sha256": snapshot.digest()}),
    )
    write_installation_resource_snapshot(site, snapshot, path=output)
    return snapshot

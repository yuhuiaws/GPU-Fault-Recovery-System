from __future__ import annotations

import copy
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Mapping

from gpu_fault.admin.bootstrap_common import (
    SITE_TAG_KEY,
    Arn,
    BootstrapError,
    tag_map,
)
from gpu_fault.admin.diagnostics import diagnostic_text
from gpu_fault.admin.execution import current_deadline, run_command
from gpu_fault.admin.grafana import grafana_installation_resources
from gpu_fault.admin.resource_records import (
    foundation_ownership as _foundation_ownership,
)
from gpu_fault.admin.resource_records import ownership as _ownership
from gpu_fault.admin.resource_records import policy as _policy
from gpu_fault.admin.resource_records import record as _record
from gpu_fault.admin.resource_records import (
    release_repository_resources as _release_repository_resources,
)
from gpu_fault.admin.resource_registry_dns import vpc_association_resources
from gpu_fault.admin.resource_registry_scripts import (
    DIRECT_SYNC_SCRIPT,
    FETCH_SCRIPT,
    LEGACY_REGISTRY_MARKER,
    SYNC_SCRIPT,
    UNAVAILABLE_REGISTRY_MARKER,
)
from gpu_fault.admin.site import RenderedSite
from gpu_fault.installation_resources import (
    InstallationResource,
    InstallationResourceDeletePolicy,
    InstallationResourceOwnership,
    InstallationResourceSnapshot,
)


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
    existing_resources: list[InstallationResource],
) -> list[InstallationResource]:
    resources: list[InstallationResource] = []
    pki = state.get("pki") or {}
    dns = config.get("dns") or {}
    if not isinstance(pki, Mapping) or not isinstance(dns, Mapping):
        raise BootstrapError("Route53 checkpoint or site DNS configuration is invalid")
    zone_id = pki.get("hosted_zone_id") or dns.get("hosted_zone_id")
    if (
        pki.get("hosted_zone_id")
        and dns.get("hosted_zone_id")
        and pki["hosted_zone_id"] != dns["hosted_zone_id"]
    ):
        raise BootstrapError("Route53 checkpoint hosted zone differs from the site")
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
    resources.extend(
        vpc_association_resources(
            site_id=site_id,
            region=region,
            account_id=account_id,
            hosted_zone_id=zone_id,
            state=state,
            existing_resources=existing_resources,
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
    resources = grafana_installation_resources(
        site_id=site_id, region=region, account_id=account_id, state=state
    )
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
        # Legacy row: only older states carry a queue; uninstall still deletes it.
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
    # Per GPU cluster: executor role (owns the shared OIDC row) + ADOT writer role.
    for name, raw in state.items():
        kind, _, cluster_id = name.partition(":")
        value = raw or {}
        if kind == "adot_writer_role":
            add_role(f"adot-writer/{cluster_id}", value)
        if kind != "executor_role":
            continue
        add_role(f"executor/{cluster_id}", value)
        provider_arn = value.get("oidc_provider_arn")
        if provider_arn:
            ownership = _foundation_ownership(value.get("oidc_provider_ownership"))
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
                    attributes={"cluster_name": value.get("cluster_name")},
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
    # The diagnostics parameter group (``<cluster>-pg``) is ours the same way the
    # subnet group is: created by name, attached to the cluster, and deletable
    # only once the cluster is gone -- so the cluster depends on it. States
    # written before the group existed carry no name and register nothing.
    parameter_group = aurora.get("parameter_group")
    if parameter_group:
        group_ownership = _ownership(
            aurora.get("parameter_group_ownership"),
            default=InstallationResourceOwnership.CREATED,
        )
        cluster_dependencies.append("aws/aurora/parameter-group")
        resources.append(
            _record(
                site_id=site_id,
                resource_key="aws/aurora/parameter-group",
                resource_type="rds_cluster_parameter_group",
                resource_id=str(parameter_group),
                region=region,
                account_id=account_id,
                ownership=group_ownership,
                delete_policy=_policy(group_ownership),
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
    # New states record the master Secret ARN under ``aurora_ready``; old ones under ``aurora``.
    readiness = state.get("aurora_ready") or {}
    master_arn = aurora.get("master_secret_arn") or readiness.get("master_secret_arn")
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
    *,
    existing: InstallationResourceSnapshot | None = None,
) -> InstallationResourceSnapshot:
    config = site.release_config
    site_id = site.registry_site_id
    region = str(config["aws_region"])
    account_id = Arn.parse(str(config["cpu_eks_arn"])).account
    state = (bootstrap or {}).get("resources") or {}
    if not isinstance(state, Mapping) or not isinstance(
        state.get("pki") or {}, Mapping
    ):
        raise BootstrapError("bootstrap resource inventory is invalid")
    if existing is not None:
        existing.require_source_binding()
        if existing.site_id != site_id:
            raise BootstrapError("installation registry belongs to another site")
    if (state.get("pki") or {}).get("vpc_associations") and (
        (bootstrap or {}).get("site_id") != config["site_name"]
    ):
        raise BootstrapError("Route53 bootstrap inventory is not bound to this site")
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
            existing_resources=existing.resources if existing is not None else [],
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
    return snapshot.model_copy(update={"source_sha256": snapshot.digest()})


def _aws_json(region: str, *arguments: str) -> dict[str, Any]:
    completed = run_command(
        ["aws", *arguments, "--region", region, "--output", "json"],
    )
    if completed.returncode:
        operation = " ".join(arguments[:2])
        raise BootstrapError(
            f"failed to discover deployed AWS resource ({operation}): "
            f"{diagnostic_text(completed.stderr.strip())}"
        )
    try:
        value = json.loads(completed.stdout)
    except (TypeError, json.JSONDecodeError):
        raise BootstrapError(
            "deployed AWS resource discovery returned invalid JSON"
        ) from None
    if not isinstance(value, dict):
        raise BootstrapError("deployed AWS resource discovery returned no object")
    return value


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
    result = run_command(
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
        timeout_seconds=30,
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
    if snapshot.site_id != site.registry_site_id:
        raise BootstrapError("installation registry snapshot belongs to another site")
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
    fields = digest_path.read_text(encoding="utf-8").split()
    if not fields:
        raise BootstrapError(f"installation registry snapshot digest is empty: {path}")
    expected = fields[0]
    payload = path.read_bytes()
    actual = hashlib.sha256(payload).hexdigest()
    if actual != expected:
        raise BootstrapError(f"installation registry snapshot digest mismatch: {path}")
    try:
        snapshot = InstallationResourceSnapshot.model_validate_json(payload)
        snapshot.require_source_binding()
    except ValueError:
        raise BootstrapError("installation registry snapshot is invalid") from None
    # These digests detect corruption; callers must also bind the snapshot to
    # the intended site and verify live ownership before destructive actions.
    return snapshot


def find_bootstrap_state(site: RenderedSite) -> dict[str, Any] | None:
    site_id = str(site.release_config["site_name"])
    candidates = [site.source.parent / "bootstrap-state.json"]
    if site.registry_site_id != site_id:
        if not candidates[0].is_file():
            raise BootstrapError("current installation bootstrap inventory is missing")
    else:
        candidates.extend(
            sorted(
                (Path.home() / ".gpu-fault/bootstrap").glob("*/bootstrap-state.json")
            )
        )
    matches = []
    for path in candidates:
        if not path.is_file():
            continue
        document = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(document, dict):
            raise BootstrapError("bootstrap state must be an object")
        if document.get("site_id") != site_id:
            if path == candidates[0]:
                raise BootstrapError("local bootstrap state belongs to another site")
            continue
        resources = document.get("resources")
        if not isinstance(resources, dict):
            raise BootstrapError("bootstrap state resource inventory is invalid")
        target = resources.get("initial_deploy_target")
        if target is not None:
            cpu = target.get("cpu") if isinstance(target, dict) else None
            if (
                not isinstance(cpu, dict)
                or cpu.get("eks_arn") != site.release_config["cpu_eks_arn"]
                or cpu.get("hyperpod_name")
                != site.release_config["cpu_hyperpod_cluster_name"]
            ):
                raise BootstrapError("bootstrap state CPU identity does not match site")
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
    snapshot.require_source_binding()
    if snapshot.site_id != site.registry_site_id:
        raise BootstrapError("installation registry snapshot belongs to another site")
    command = [
        "kubectl",
        "--kubeconfig",
        site.release_config["cpu_kubeconfig"],
        "-n",
        site.release_config["namespace"],
        "exec",
        "-i",
        _cpu_pod(site),
        "--",
        "python",
        "-c",
        SYNC_SCRIPT,
    ]
    delays = (2, 4, 8)
    for attempt in range(len(delays) + 1):
        result = run_command(
            command,
            input_text=_snapshot_payload(snapshot).decode(),
            timeout_seconds=90,
        )
        if result.returncode == 0:
            saved = _read_registry_snapshot(site, result.stdout)
            if {item.resource_key: item for item in saved.resources} != {
                item.resource_key: item for item in snapshot.resources
            }:
                raise BootstrapError(
                    "installation registry synchronization proof differs"
                )
            return
        if result.returncode == 44 and result.stdout.strip() == LEGACY_REGISTRY_MARKER:
            raise LegacyInstallationRegistryMissing(
                "control plane predates the installation resource registry API"
            )
        if (
            result.returncode != 75
            or result.stdout.strip() != UNAVAILABLE_REGISTRY_MARKER
        ):
            raise BootstrapError(
                "installation registry synchronization failed: "
                + diagnostic_text((result.stderr or "").strip())
            )
        if attempt < len(delays):
            budget = current_deadline()
            time.sleep(
                min(delays[attempt], budget.remaining()) if budget else delays[attempt]
            )
    sync_installation_resource_snapshot_direct(site, snapshot)


def sync_installation_resource_snapshot_direct(
    site: RenderedSite,
    snapshot: InstallationResourceSnapshot,
) -> None:
    snapshot.require_source_binding()
    if snapshot.site_id != site.registry_site_id:
        raise BootstrapError("installation registry snapshot belongs to another site")
    result = run_command(
        [
            "kubectl",
            "--kubeconfig",
            site.release_config["cpu_kubeconfig"],
            "-n",
            site.release_config["namespace"],
            "exec",
            "-i",
            _cpu_pod(site),
            "--",
            "python",
            "-c",
            DIRECT_SYNC_SCRIPT,
        ],
        input_text=_snapshot_payload(snapshot).decode(),
        timeout_seconds=900,
    )
    if result.returncode:
        raise BootstrapError(
            "direct Aurora registry synchronization failed: "
            + diagnostic_text((result.stderr or "").strip())
        )
    if result.stdout.strip() != str(len(snapshot.resources)):
        raise BootstrapError("direct Aurora registry synchronization proof is invalid")


def sync_installation_resource_registry(site: RenderedSite) -> Path:
    bootstrap = find_bootstrap_state(site)
    existing = None
    if (((bootstrap or {}).get("resources") or {}).get("pki") or {}).get(
        "vpc_associations"
    ):
        existing = fetch_installation_resource_registry(site, allow_empty=True)
    snapshot = build_installation_snapshot(
        site,
        bootstrap,
        discover_runtime_resources(site),
        existing=existing,
    )
    sync_installation_resource_snapshot(site, snapshot)
    return write_installation_resource_snapshot(site, snapshot)


def _read_registry_snapshot(
    site: RenderedSite, output: str
) -> InstallationResourceSnapshot:
    try:
        rows = json.loads(output)
    except (TypeError, json.JSONDecodeError):
        raise BootstrapError("installation registry returned invalid JSON") from None
    if not isinstance(rows, list):
        raise BootstrapError("installation registry returned no resource list")
    try:
        snapshot = InstallationResourceSnapshot(
            site_id=site.registry_site_id,
            resources=[InstallationResource.model_validate(item) for item in rows],
        )
    except ValueError:
        raise BootstrapError(
            "installation registry returned invalid resources"
        ) from None
    return snapshot.model_copy(update={"source_sha256": snapshot.digest()})


def fetch_installation_resource_registry(
    site: RenderedSite,
    *,
    output: Path | None = None,
    allow_empty: bool = False,
) -> InstallationResourceSnapshot:
    result = run_command(
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
            ("GPU_FAULT_INSTALLATION_SITE_ID=" + site.registry_site_id),
            "python",
            "-c",
            FETCH_SCRIPT,
        ],
        timeout_seconds=90,
    )
    if result.returncode:
        if result.returncode == 44 and result.stdout.strip() == LEGACY_REGISTRY_MARKER:
            raise LegacyInstallationRegistryMissing(
                "control plane predates the installation resource registry API"
            )
        raise BootstrapError(
            "installation registry fetch failed: "
            + diagnostic_text((result.stderr or "").strip())
        )
    snapshot = _read_registry_snapshot(site, result.stdout)
    if not snapshot.resources and not allow_empty:
        raise LegacyInstallationRegistryMissing(
            "Aurora installation resource registry is empty"
        )
    write_installation_resource_snapshot(site, snapshot, path=output)
    return snapshot

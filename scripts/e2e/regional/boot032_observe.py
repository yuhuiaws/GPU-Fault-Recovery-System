"""Read-only, identity-bound observations for the two BOOT-032 sites."""

from __future__ import annotations

import base64
import json
import os
from collections.abc import Callable
from typing import Any

import yaml

from gpu_fault.admin.aws_cleanup import ResourceProbe
from gpu_fault.admin.aws_cleanup_ownership import CleanupOwnership
from gpu_fault.admin.aws_commands import json_command
from gpu_fault.admin.bootstrap_common import Arn, CommandRunner
from gpu_fault.admin.resource_registry import fetch_installation_resource_registry
from gpu_fault.admin.site import RenderedSite, materialized_release_config
from gpu_fault.installation_resources import (
    InstallationResourceOwnership,
    InstallationResourceSnapshot,
    InstallationResourceStatus,
)
from scripts.e2e.regional.boot032_contract import (
    CASE_ID,
    CASE_TAG,
    FIXTURE_TAG,
    Settings,
    UninstallCaseError,
    cluster_specs,
    kubeconfig,
    mapping,
    require,
)
from scripts.e2e.regional.boot032_fleet import distinct_fleets, node_inventory
from scripts.e2e.regional.live_driver_guard import details_sha256
from scripts.e2e.regional.regional_live_fixture import (
    RegionalLiveFixture,
    RegionalLiveSettings,
    runtime_identity_errors,
)


def aws(
    region: str,
    service: str,
    operation: str,
    *arguments: str,
    absent: tuple[str, ...] = (),
) -> dict[str, Any] | None:
    return json_command(
        ["aws", service, operation, "--region", region, *arguments],
        not_found=absent,
    )


def one_named(items: Any, name: str) -> dict[str, Any]:
    require(isinstance(items, list), "named identity inventory is missing")
    matches = [
        item for item in items if isinstance(item, dict) and item.get("name") == name
    ]
    require(len(matches) == 1, "named identity is missing or duplicated")
    return matches[0]


def kube_target(site: RenderedSite, spec: dict[str, str], eks: dict[str, Any]) -> str:
    load: Callable[[str], object] = yaml.safe_load
    value = mapping(
        load(kubeconfig(site, spec["plane"]).read_text(encoding="utf-8")),
        "kubeconfig is not an object",
    )
    context = (
        value.get("current-context") if spec["plane"] == "cpu" else spec["context"]
    )
    if not isinstance(context, str) or not context:
        raise UninstallCaseError("kubeconfig context is missing")
    selected = mapping(
        one_named(value.get("contexts"), context).get("context"),
        "kubeconfig context binding is missing",
    )
    cluster_name = selected.get("cluster")
    if not isinstance(cluster_name, str) or not cluster_name:
        raise ValueError("kubeconfig cluster name is missing")
    cluster = mapping(
        one_named(value.get("clusters"), cluster_name).get("cluster"),
        "kubeconfig cluster binding is missing",
    )
    require(
        cluster.get("server") == eks.get("endpoint")
        and isinstance(eks.get("endpoint"), str)
        and eks["endpoint"].startswith("https://")
        and cluster.get("insecure-skip-tls-verify", False) is False,
        "kubeconfig server or TLS validation differs from EKS",
    )
    local = base64.b64decode(
        cluster.get("certificate-authority-data", ""), validate=True
    )
    remote = base64.b64decode(
        eks.get("certificateAuthority", {}).get("data", ""), validate=True
    )
    require(bool(local) and local == remote, "kubeconfig CA differs from EKS")
    return context


def kubernetes_uid(
    site: RenderedSite, spec: dict[str, str], kind: str, name: str, namespace: str = ""
) -> str | None:
    require(
        all(
            isinstance(item, str) and item and not item.startswith("-")
            for item in (kind, name)
        ),
        "Kubernetes resource reference is invalid",
    )
    arguments = ["kubectl", "--kubeconfig", str(kubeconfig(site, spec["plane"]))]
    if spec["plane"] == "gpu":
        arguments += ["--context", spec["context"]]
    if namespace:
        arguments += ["-n", namespace]
    arguments += [
        "get",
        kind,
        name,
        "--ignore-not-found",
        "--request-timeout=30s",
        '-o=jsonpath={"uid="}{.metadata.uid}',
    ]
    output = CommandRunner().run(arguments, timeout_seconds=45).strip()
    if not output:
        return None
    require(
        output.startswith("uid=")
        and bool(output[4:])
        and not any(character.isspace() for character in output[4:]),
        "Kubernetes resource UID is unreadable",
    )
    return output[4:]


def cluster_observation(
    site: RenderedSite,
    spec: dict[str, str],
    *,
    fixture_id: str | None = None,
    cpu_deletion_started: bool = False,
) -> dict[str, Any]:
    eks_document = aws(
        spec["region"],
        "eks",
        "describe-cluster",
        "--name",
        spec["eks_name"],
        absent=("ResourceNotFoundException",),
    )
    hp = aws(
        spec["region"],
        "sagemaker",
        "describe-cluster",
        "--cluster-name",
        spec["hyperpod_name"],
        absent=("ResourceNotFound",),
    )
    eks = (
        None
        if eks_document is None
        else mapping(eks_document.get("cluster"), "EKS observation is malformed")
    )
    if eks is not None:
        require(
            eks.get("arn") == spec["eks_arn"]
            and eks.get("name") == spec["eks_name"]
            and isinstance(eks.get("createdAt"), str)
            and bool(eks["createdAt"]),
            "EKS identity or incarnation is incomplete",
        )
    if hp is not None:
        require(
            hp.get("ClusterName") == spec["hyperpod_name"]
            and isinstance(hp.get("ClusterArn"), str)
            and bool(hp["ClusterArn"])
            and hp.get("Orchestrator", {}).get("Eks", {}).get("ClusterArn")
            == spec["eks_arn"],
            "HyperPod identity or EKS binding is incomplete",
        )
        hp_arn = Arn.parse(hp["ClusterArn"])
        require(
            hp_arn.partition == Arn.parse(spec["eks_arn"]).partition
            and hp_arn.service == "sagemaker"
            and hp_arn.resource.startswith("cluster/")
            and hp_arn.account == spec["account"]
            and hp_arn.region == spec["region"],
            "HyperPod ARN scope differs from the explicit EKS cluster",
        )
    if eks is None or hp is None:
        require(
            spec["plane"] == "cpu" and cpu_deletion_started,
            "required physical cluster is absent before authorized deletion",
        )
        return {
            "eks_absent": eks is None,
            "hyperpod_absent": hp is None,
            **(
                {"eks_arn": eks["arn"], "eks_created_at": eks["createdAt"]}
                if eks is not None
                else {}
            ),
            **(
                {"hyperpod_arn": hp["ClusterArn"], "hyperpod_name": hp["ClusterName"]}
                if hp is not None
                else {}
            ),
        }
    if not cpu_deletion_started:
        require(
            eks.get("status") == "ACTIVE" and hp.get("ClusterStatus") == "InService",
            "physical cluster is not ready",
        )
    if spec["plane"] == "gpu":
        require(
            hp.get("NodeRecovery") == "None", "GPU managed recovery is not disabled"
        )
    if fixture_id is not None:
        tags_document = aws(
            spec["region"], "sagemaker", "list-tags", "--resource-arn", hp["ClusterArn"]
        )
        tags_document = mapping(tags_document, "HyperPod fixture tags are unavailable")
        tag_items = tags_document.get("Tags")
        if not isinstance(tag_items, list):
            raise ValueError("HyperPod fixture tags are malformed")
        tags = {item["Key"]: item["Value"] for item in tag_items}
        require(len(tags) == len(tag_items), "HyperPod fixture tags are duplicated")
        for values in (eks.get("tags"), tags):
            require(
                isinstance(values, dict)
                and values.get(CASE_TAG) == CASE_ID
                and values.get(FIXTURE_TAG) == fixture_id,
                "cluster lacks explicit BOOT-032 fixture-purpose tags",
            )
    identity = {
        "eks_arn": eks["arn"],
        "eks_created_at": eks["createdAt"],
        "hyperpod_arn": hp["ClusterArn"],
        "hyperpod_name": hp["ClusterName"],
    }
    if not cpu_deletion_started:
        identity["kube_context"] = kube_target(site, spec, eks)
        uid = kubernetes_uid(site, spec, "namespace", "kube-system")
        require(uid is not None, "Kubernetes cluster identity is unavailable")
        identity["cluster_uid"] = uid
        identity["namespace_uid"] = kubernetes_uid(
            site, spec, "namespace", site.release_config["namespace"]
        )
    return identity


def regional_fixture(site: RenderedSite, spec: dict[str, str]) -> RegionalLiveFixture:
    return RegionalLiveFixture(
        RegionalLiveSettings(
            cpu_kubeconfig=kubeconfig(site, "cpu"),
            gpu_kubeconfig=kubeconfig(site, "gpu"),
            gpu_context=spec["context"],
            namespace=site.release_config["namespace"],
            cluster_id=spec["cluster_id"],
            region=spec["region"],
        )
    )


def node_observation(site: RenderedSite) -> dict[str, Any]:
    return {
        spec["cluster_id"]: node_inventory(regional_fixture(site, spec))
        for spec in cluster_specs(site)[1:]
    }


def runtime_observation(site: RenderedSite) -> dict[str, Any]:
    result = {}
    for spec in cluster_specs(site)[1:]:
        value = regional_fixture(site, spec).runtime_identity()
        require(
            not runtime_identity_errors(value),
            "runtime identity is not healthy and complete",
        )
        result[spec["cluster_id"]] = value
    return result


def resource_identity(resource: Any, site: RenderedSite) -> tuple[str, ...]:
    kind = resource.resource_type.removeprefix("cpu_").removeprefix("gpu_")
    identifier = resource.resource_arn or (
        resource.resource_id if resource.resource_id.startswith("arn:") else None
    )
    partition = Arn.parse(site.release_config["cpu_eks_arn"]).partition
    if identifier is None and kind == "route53_zone":
        zone = resource.resource_id.removeprefix("/hostedzone/").removeprefix(
            "hostedzone/"
        )
        require(zone, "global hosted zone identity is missing")
        identifier = f"arn:{partition}:route53:::hostedzone/{zone}"
    if identifier is not None:
        arn = Arn.parse(identifier)
        if arn.service in {"iam", "route53"}:
            require(
                not arn.region
                and (bool(arn.account) if arn.service == "iam" else not arn.account),
                "global AWS resource ARN scope is invalid",
            )
        else:
            require(
                arn.region and arn.account,
                "AWS resource ARN scope is unknown",
            )
        return (
            resource.provider,
            "arn",
            arn.partition,
            arn.service,
            arn.region,
            arn.account,
            arn.resource,
        )
    require(
        kind not in {"iam_role", "iam_policy", "iam_oidc_provider"},
        "global IAM resource identity requires its complete ARN",
    )
    if kind in {"route53_record", "route53_vpc_association"}:
        zone = resource.attributes.get("hosted_zone_id")
        require(isinstance(zone, str) and zone, "global DNS binding scope is unknown")
        fields = (
            (
                resource.resource_id.rstrip(".").casefold(),
                resource.attributes.get("record_type"),
            )
            if kind == "route53_record"
            else (
                resource.attributes.get("vpc_region"),
                resource.attributes.get("vpc_id"),
            )
        )
        require(
            all(isinstance(item, str) and item for item in fields),
            "global DNS binding identity is incomplete",
        )
        return resource.provider, kind, partition, zone, *fields
    require(
        resource.region and resource.account_id,
        "regional resource identity scope is unknown",
    )
    scope = site.release_config["cpu_eks_arn"] if kind == "helm_release" else ""
    if kind in {"eks_addon", "eks_pod_identity_association", "grafana_service_account"}:
        key = "workspace_id" if kind == "grafana_service_account" else "cluster_name"
        scope = resource.attributes.get(key)
        require(
            isinstance(scope, str) and scope, "child resource identity scope is unknown"
        )
    return (
        resource.provider,
        kind,
        resource.region or "",
        resource.account_id or "",
        scope,
        resource.resource_id,
    )


def physical_inventory(
    site: RenderedSite, snapshot: InstallationResourceSnapshot
) -> None:
    expected = {
        ("aurora_cluster", site.release_config["health"]["aurora_cluster_id"]),
        *(
            (spec["plane"] + "_" + kind, spec[field])
            for spec in cluster_specs(site)
            for kind, field in (("eks", "eks_name"), ("hyperpod", "hyperpod_name"))
        ),
    }
    kinds = {"cpu_eks", "cpu_hyperpod", "gpu_eks", "gpu_hyperpod", "aurora_cluster"}
    actual = [
        (item.resource_type, item.resource_id)
        for item in snapshot.resources
        if item.resource_type in kinds
    ]
    require(
        set(actual) == expected and len(actual) == len(expected),
        "registered physical inventory differs from the complete explicit site",
    )


def registry(site: RenderedSite, *, fresh: bool) -> InstallationResourceSnapshot:
    snapshot = fetch_installation_resource_registry(site)
    snapshot.require_source_binding()
    require(
        snapshot.site_id == site.registry_site_id and snapshot.resources,
        "registry scope is invalid",
    )
    physical_inventory(site, snapshot)
    probe = ResourceProbe(site)
    probe.validate_supported(snapshot.resources)
    ownership = CleanupOwnership(site)
    ownership.assert_caller()
    for resource in snapshot.resources:
        ownership.validate_scope(resource)
        require(
            resource.status is InstallationResourceStatus.ACTIVE,
            "registry is not wholly active",
        )
        if fresh:
            require(
                resource.ownership is not InstallationResourceOwnership.REUSED,
                "BOOT-032 requires a fresh fixture, not legacy reused solution resources",
            )
        require(
            probe.exists(resource) is True, "registered resource existence is unproved"
        )
    return snapshot


def installed_inventory(site: RenderedSite) -> dict[str, Any]:
    with materialized_release_config(site) as path:
        output = CommandRunner().run(
            [
                "python3",
                str(
                    site.repository_root
                    / "deploy/control-plane/tools/collect_installed_resource_registry.py"
                ),
                "--config",
                str(path),
            ],
            env={
                **os.environ,
                **site.environment,
                "KUBECONFIG": str(kubeconfig(site, "gpu")),
            },
            timeout_seconds=600,
        )
    value = mapping(json.loads(output), "installed inventory is not an object")
    require(
        value.get("unregistered_resources") == [],
        "installed resource inventory has unknown or unregistered resources",
    )
    return value


def inventory_uids(site: RenderedSite, inventory: dict[str, Any]) -> dict[str, Any]:
    result = {}
    for spec in cluster_specs(site):
        section = (
            inventory["cpu"]
            if spec["plane"] == "cpu"
            else inventory["gpu"]["by_context"][spec["context"]]
        )
        resources = section.get("resources")
        require(
            isinstance(resources, list) and resources,
            "installed inventory is incomplete",
        )
        rows = []
        seen = set()
        for item in resources:
            require(
                item.get("scope") in {"cluster", "namespaced"},
                "installed resource scope is unknown",
            )
            namespace = (
                item.get("namespace") or site.release_config["namespace"]
                if item["scope"] == "namespaced"
                else ""
            )
            identity = (item["scope"], item["kind"], namespace, item["name"])
            require(identity not in seen, "installed resource identity is duplicated")
            seen.add(identity)
            rows.append(
                {
                    "identity": list(identity),
                    "uid": kubernetes_uid(
                        site, spec, item["kind"], item["name"], namespace
                    ),
                }
            )
        result[spec["context"]] = rows
    return result


def observe_site(
    site: RenderedSite, *, fixture_id: str | None = None
) -> dict[str, Any]:
    resources = registry(site, fresh=fixture_id is not None)
    clusters = {
        spec["context"]: cluster_observation(site, spec, fixture_id=fixture_id)
        for spec in cluster_specs(site)
    }
    require(
        all(value["namespace_uid"] for value in clusters.values()),
        "solution namespace is absent before the acceptance case",
    )
    require(
        len({value["cluster_uid"] for value in clusters.values()}) == len(clusters),
        "CPU/GPU kubeconfigs alias a physical cluster",
    )
    inventory = installed_inventory(site)
    return {
        "resources": resources.model_dump(mode="json"),
        "clusters": clusters,
        "runtime": runtime_observation(site),
        "nodes": node_observation(site),
        "inventory_sha256": details_sha256(inventory),
        "resource_uids": inventory_uids(site, inventory),
    }


def initial_binding(settings: Settings) -> dict[str, Any]:
    inputs = settings.inputs()
    target = observe_site(settings.target, fixture_id=settings.fixture_id)
    protected = observe_site(settings.protected)
    distinct_fleets(target, protected)
    left = InstallationResourceSnapshot.model_validate(target["resources"])
    right = InstallationResourceSnapshot.model_validate(protected["resources"])
    require(
        not {resource_identity(item, settings.target) for item in left.resources}
        & {resource_identity(item, settings.protected) for item in right.resources},
        "sacrificial resource inventory overlaps the protected accepted site",
    )
    require(
        not {item["cluster_uid"] for item in target["clusters"].values()}
        & {item["cluster_uid"] for item in protected["clusters"].values()},
        "sacrificial Kubernetes cluster aliases the accepted site",
    )
    return {"inputs": inputs, "target": target, "protected": protected}

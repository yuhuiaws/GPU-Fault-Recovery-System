from __future__ import annotations

import json
from typing import Any, Iterable

from gpu_fault.admin.aws_cleanup_clusters import (
    aurora_snapshot,
    cpu_eks_identity,
    delete_aurora_cluster,
    delete_eks_cluster,
    delete_hyperpod_cluster,
    prepare_aurora_deletion,
    prepare_cpu_deletion,
)
from gpu_fault.admin.aws_cleanup_helpers import (
    delete_certificate_once_released,
    object_field,
    objects,
    oidc_provider_users,
    ordered_aurora_instances as ordered_aurora_instances,
    single_object,
    sqs_policy,
    sqs_topic_statements,
)
from gpu_fault.admin.aws_cleanup_ownership import CleanupOwnership
from gpu_fault.admin.aws_commands import (
    FinalSnapshotPolicy,
    helm_release_absent,
)
from gpu_fault.admin.aws_commands import (
    checked_command as _checked,
)
from gpu_fault.admin.aws_commands import (
    json_command as _json,
)
from gpu_fault.admin.aws_commands import (
    matches_not_found as _matches_not_found,
)
from gpu_fault.admin.aws_commands import (
    run_command as _run,
)
from gpu_fault.admin.aws_commands import (
    wait_until as _wait_until,
)
from gpu_fault.admin.bootstrap_common import BootstrapError
from gpu_fault.admin.diagnostics import diagnostic_text
from gpu_fault.admin.site import RenderedSite
from gpu_fault.installation_resources import InstallationResource

SUPPORTED_RESOURCE_TYPES = frozenset(
    {
        "acm_certificate",
        "amp_workspace",
        "aurora_cluster",
        "aurora_instance",
        "cpu_eks",
        "cpu_hyperpod",
        "ec2_route_table",
        "ec2_route_table_association",
        "ec2_subnet",
        "ecr_repository",
        "eks_addon",
        "eks_pod_identity_association",
        "gpu_eks",
        "gpu_hyperpod",
        "grafana_service_account",
        "grafana_workspace",
        "helm_release",
        "iam_oidc_provider",
        "iam_policy",
        "iam_role",
        "internet_gateway",
        "nlb",
        "nlb_listener",
        "nlb_target_group",
        "rds_cluster_parameter_group",
        "rds_db_subnet_group",
        "rds_managed_secret",
        "rds_snapshot",
        "route53_record",
        "route53_vpc_association",
        "route53_zone",
        "ses_email_identity",
        "secretsmanager_secret",
        "security_group",
        "sns_subscription",
        "sns_topic",
        "sqs_policy_binding",
        "sqs_queue",
    }
)

DELETE_PRIORITY = {
    "route53_record": 10,
    "sns_subscription": 10,
    "sqs_policy_binding": 10,
    "eks_pod_identity_association": 10,
    "ec2_route_table_association": 10,
    "route53_vpc_association": 10,
    "grafana_service_account": 10,
    "nlb_listener": 20,
    "nlb": 30,
    "nlb_target_group": 40,
    "helm_release": 50,
    "amp_workspace": 60,
    "grafana_workspace": 60,
    "sns_topic": 60,
    "sqs_queue": 60,
    "acm_certificate": 60,
    "secretsmanager_secret": 60,
    "ses_email_identity": 60,
    "ecr_repository": 60,
    "iam_role": 70,
    "iam_policy": 80,
    "iam_oidc_provider": 80,
    "eks_addon": 80,
    "route53_zone": 90,
    "ec2_route_table": 100,
    "ec2_subnet": 110,
    "internet_gateway": 120,
    "security_group": 130,
}

AURORA_RESOURCE_TYPES = frozenset(
    {
        "aurora_cluster",
        "aurora_instance",
        "rds_cluster_parameter_group",
        "rds_db_subnet_group",
        "rds_managed_secret",
    }
)


class ResourceProbe:
    def __init__(self, site: RenderedSite) -> None:
        self.site = site
        self.region = str(site.release_config["aws_region"])
        self.cpu_kubeconfig = str(site.release_config["cpu_kubeconfig"])
        self._ownership = CleanupOwnership(site)

    def validate_supported(self, resources: Iterable[InstallationResource]) -> None:
        unsupported = sorted(
            {
                resource.resource_type
                for resource in resources
                if resource.resource_type not in SUPPORTED_RESOURCE_TYPES
            }
        )
        if unsupported:
            raise BootstrapError(
                "installation registry contains unsupported resource types: "
                + ", ".join(unsupported)
            )

    def _aws(self, service: str, operation: str, *arguments: str) -> list[str]:
        return ["aws", service, operation, "--region", self.region, *arguments]

    def _exists_command(
        self,
        arguments: list[str],
        *,
        not_found: Iterable[str],
    ) -> bool:
        result = _run(arguments)
        if result.returncode == 0:
            return True
        if _matches_not_found(result, not_found):
            return False
        raise BootstrapError(
            f"resource verification failed: {' '.join(arguments[:3])}: "
            f"{diagnostic_text(result.stderr.strip())}"
        )

    def _route53_records(self, resource: InstallationResource) -> list[dict[str, Any]]:
        document = _json(
            self._aws(
                "route53",
                "list-resource-record-sets",
                "--hosted-zone-id",
                resource.attributes["hosted_zone_id"],
            ),
            not_found=("NoSuchHostedZone",),
        )
        if document is None:
            return []
        name = resource.resource_id.rstrip(".").casefold() + "."
        record_type = resource.attributes.get("record_type", "CNAME")
        records = objects(document, "ResourceRecordSets", required=("Name", "Type"))
        return [
            item
            for item in records
            if item["Name"].casefold() == name and item["Type"] == record_type
        ]

    def _exists_network(self, resource: InstallationResource) -> bool | None:
        resource_type = resource.resource_type
        identifier = resource.resource_id
        arn = resource.resource_arn or identifier
        if resource_type == "nlb":
            if resource.resource_arn:
                arguments = self._aws(
                    "elbv2",
                    "describe-load-balancers",
                    "--load-balancer-arns",
                    resource.resource_arn,
                )
            else:
                arguments = self._aws(
                    "elbv2",
                    "describe-load-balancers",
                    "--names",
                    identifier,
                )
            return self._exists_command(arguments, not_found=("LoadBalancerNotFound",))
        if resource_type == "nlb_listener":
            return self._exists_command(
                self._aws(
                    "elbv2",
                    "describe-listeners",
                    "--listener-arns",
                    arn,
                ),
                not_found=("ListenerNotFound",),
            )
        if resource_type == "nlb_target_group":
            return self._exists_command(
                self._aws(
                    "elbv2",
                    "describe-target-groups",
                    "--target-group-arns",
                    arn,
                ),
                not_found=("TargetGroupNotFound",),
            )
        if resource_type == "security_group":
            return self._exists_command(
                self._aws(
                    "ec2",
                    "describe-security-groups",
                    "--group-ids",
                    identifier,
                ),
                not_found=("InvalidGroup.NotFound",),
            )
        if resource_type == "ec2_subnet":
            return self._exists_command(
                self._aws(
                    "ec2",
                    "describe-subnets",
                    "--subnet-ids",
                    identifier,
                ),
                not_found=("InvalidSubnetID.NotFound",),
            )
        if resource_type == "ec2_route_table":
            return self._exists_command(
                self._aws(
                    "ec2",
                    "describe-route-tables",
                    "--route-table-ids",
                    identifier,
                ),
                not_found=("InvalidRouteTableID.NotFound",),
            )
        if resource_type == "ec2_route_table_association":
            document = _json(
                self._aws(
                    "ec2",
                    "describe-route-tables",
                    "--filters",
                    (
                        "Name=association.route-table-association-id,"
                        f"Values={identifier}"
                    ),
                )
            )
            return bool(objects(document or {}, "RouteTables"))
        if resource_type == "internet_gateway":
            return self._exists_command(
                self._aws(
                    "ec2",
                    "describe-internet-gateways",
                    "--internet-gateway-ids",
                    identifier,
                ),
                not_found=("InvalidInternetGatewayID.NotFound",),
            )
        return None

    def _exists_dns_and_monitoring(self, resource: InstallationResource) -> bool | None:
        resource_type = resource.resource_type
        identifier = resource.resource_id
        arn = resource.resource_arn or identifier
        if resource_type == "route53_record":
            return bool(self._route53_records(resource))
        if resource_type == "route53_zone":
            return self._exists_command(
                self._aws("route53", "get-hosted-zone", "--id", identifier),
                not_found=("NoSuchHostedZone",),
            )
        if resource_type == "route53_vpc_association":
            document = _json(
                self._aws(
                    "route53",
                    "get-hosted-zone",
                    "--id",
                    resource.attributes["hosted_zone_id"],
                ),
                not_found=("NoSuchHostedZone",),
            )
            if document is None:
                return False
            expected = (
                resource.attributes["vpc_region"],
                resource.attributes["vpc_id"],
            )
            return expected in {
                (
                    str(item.get("VPCRegion") or ""),
                    str(item.get("VPCId") or ""),
                )
                for item in objects(document, "VPCs", required=("VPCRegion", "VPCId"))
            }
        if resource_type == "acm_certificate":
            return self._exists_command(
                self._aws(
                    "acm",
                    "describe-certificate",
                    "--certificate-arn",
                    arn,
                ),
                not_found=("ResourceNotFoundException",),
            )
        if resource_type in {"secretsmanager_secret", "rds_managed_secret"}:
            return self._exists_command(
                self._aws(
                    "secretsmanager",
                    "describe-secret",
                    "--secret-id",
                    identifier,
                ),
                not_found=("ResourceNotFoundException",),
            )
        if resource_type in {"amp_workspace", "grafana_workspace"}:
            return self._exists_command(
                self._aws(
                    resource_type.removesuffix("_workspace"),
                    "describe-workspace",
                    "--workspace-id",
                    identifier,
                ),
                not_found=("ResourceNotFoundException",),
            )
        if resource_type == "grafana_service_account":
            document = _json(
                self._aws(
                    "grafana",
                    "list-workspace-service-accounts",
                    "--workspace-id",
                    resource.attributes["workspace_id"],
                ),
                not_found=("ResourceNotFoundException",),
            )
            if document is None:
                return False
            accounts = objects(document, "serviceAccounts")
            if any(
                not isinstance(item.get("id"), (int, str)) or not str(item["id"])
                for item in accounts
            ):
                raise BootstrapError("Grafana service account identity is unavailable")
            return any(str(item["id"]) == identifier for item in accounts)
        if resource_type == "sns_topic":
            return self._exists_command(
                self._aws(
                    "sns",
                    "get-topic-attributes",
                    "--topic-arn",
                    arn,
                ),
                not_found=("NotFound", "NotFoundException"),
            )
        if resource_type == "sns_subscription":
            return self._exists_command(
                self._aws(
                    "sns",
                    "get-subscription-attributes",
                    "--subscription-arn",
                    arn,
                ),
                not_found=("NotFound", "NotFoundException"),
            )
        if resource_type == "ses_email_identity":
            return self._exists_command(
                self._aws(
                    "sesv2",
                    "get-email-identity",
                    "--email-identity",
                    identifier,
                ),
                not_found=("NotFoundException",),
            )
        if resource_type in {"sqs_queue", "sqs_policy_binding"}:
            document = _json(
                self._aws(
                    "sqs",
                    "get-queue-attributes",
                    "--queue-url",
                    identifier,
                    "--attribute-names",
                    "All",
                ),
                not_found=(
                    "AWS.SimpleQueueService.NonExistentQueue",
                    "QueueDoesNotExist",
                ),
            )
            if document is None:
                return False
            if resource_type == "sqs_queue":
                return True
            policy = sqs_policy(object_field(document, "Attributes").get("Policy"))
            return bool(sqs_topic_statements(policy, resource.attributes["topic_arn"]))
        return None

    def _exists_identity_and_runtime(
        self,
        resource: InstallationResource,
    ) -> bool | None:
        resource_type = resource.resource_type
        identifier = resource.resource_id
        arn = resource.resource_arn or identifier
        if resource_type == "iam_role":
            return self._exists_command(
                ["aws", "iam", "get-role", "--role-name", identifier],
                not_found=("NoSuchEntity",),
            )
        if resource_type == "iam_policy":
            return self._exists_command(
                ["aws", "iam", "get-policy", "--policy-arn", arn],
                not_found=("NoSuchEntity",),
            )
        if resource_type == "iam_oidc_provider":
            return self._exists_command(
                [
                    "aws",
                    "iam",
                    "get-open-id-connect-provider",
                    "--open-id-connect-provider-arn",
                    arn,
                ],
                not_found=("NoSuchEntity",),
            )
        if resource_type == "eks_pod_identity_association":
            return self._exists_command(
                self._aws(
                    "eks",
                    "describe-pod-identity-association",
                    "--cluster-name",
                    resource.attributes["cluster_name"],
                    "--association-id",
                    identifier,
                ),
                not_found=("ResourceNotFoundException",),
            )
        if resource_type == "eks_addon":
            return self._exists_command(
                self._aws(
                    "eks",
                    "describe-addon",
                    "--cluster-name",
                    resource.attributes["cluster_name"],
                    "--addon-name",
                    identifier,
                ),
                not_found=("ResourceNotFoundException",),
            )
        if resource_type == "helm_release":
            result = _run(
                [
                    "helm",
                    "--kubeconfig",
                    self.cpu_kubeconfig,
                    "-n",
                    resource.attributes.get("namespace", "default"),
                    "status",
                    identifier,
                ]
            )
            if result.returncode == 0:
                return True
            if helm_release_absent(result):
                return False
            raise BootstrapError(
                f"Helm release verification failed: {diagnostic_text(result.stderr.strip())}"
            )
        if resource_type == "ecr_repository":
            return self._exists_command(
                self._aws(
                    "ecr",
                    "describe-repositories",
                    "--repository-names",
                    identifier,
                ),
                not_found=("RepositoryNotFoundException",),
            )
        return None

    def _exists_database_and_clusters(
        self,
        resource: InstallationResource,
    ) -> bool | None:
        resource_type = resource.resource_type
        identifier = resource.resource_id
        if resource_type == "rds_db_subnet_group":
            return self._exists_command(
                self._aws(
                    "rds",
                    "describe-db-subnet-groups",
                    "--db-subnet-group-name",
                    identifier,
                ),
                not_found=("DBSubnetGroupNotFoundFault",),
            )
        if resource_type == "rds_cluster_parameter_group":
            return self._exists_command(
                self._aws(
                    "rds",
                    "describe-db-cluster-parameter-groups",
                    "--db-cluster-parameter-group-name",
                    identifier,
                ),
                not_found=("DBParameterGroupNotFound",),
            )
        if resource_type == "aurora_cluster":
            return self._exists_command(
                self._aws(
                    "rds",
                    "describe-db-clusters",
                    "--db-cluster-identifier",
                    identifier,
                ),
                not_found=("DBClusterNotFoundFault",),
            )
        if resource_type == "aurora_instance":
            return self._exists_command(
                self._aws(
                    "rds",
                    "describe-db-instances",
                    "--db-instance-identifier",
                    identifier,
                ),
                not_found=("DBInstanceNotFound",),
            )
        if resource_type == "rds_snapshot":
            self._ownership.validate_scope(resource)
            snapshot = aurora_snapshot(
                self,
                identifier,
                cluster_id=str(self.site.release_config["health"]["aurora_cluster_id"]),
                cluster_resource_id=resource.attributes.get("db_cluster_resource_id"),
            )
            return snapshot is not None and snapshot["Status"] == "available"
        if resource_type in {"cpu_eks", "gpu_eks"}:
            return self._exists_command(
                self._aws("eks", "describe-cluster", "--name", identifier),
                not_found=("ResourceNotFoundException",),
            )
        if resource_type in {"cpu_hyperpod", "gpu_hyperpod"}:
            return self._exists_command(
                self._aws(
                    "sagemaker",
                    "describe-cluster",
                    "--cluster-name",
                    identifier,
                ),
                not_found=("ResourceNotFound",),
            )
        return None

    def exists(self, resource: InstallationResource) -> bool:
        for probe in (
            self._exists_network,
            self._exists_dns_and_monitoring,
            self._exists_identity_and_runtime,
            self._exists_database_and_clusters,
        ):
            result = probe(resource)
            if result is not None:
                return result
        raise BootstrapError(
            f"unsupported resource verification: {resource.resource_type}"
        )


class ResourceDeletion(ResourceProbe):
    def _assert_internet_gateway_exclusive(
        self,
        resource: InstallationResource,
    ) -> None:
        document = _json(
            self._aws(
                "ec2",
                "describe-route-tables",
                "--filters",
                f"Name=route.gateway-id,Values={resource.resource_id}",
            )
        )
        foreign = []
        for route_table in objects(document or {}, "RouteTables"):
            tags = {
                str(item.get("Key") or ""): str(item.get("Value") or "")
                for item in route_table.get("Tags", [])
            }
            if tags.get("gpu-fault:site-id") != self._ownership.site_id:
                foreign.append(str(route_table.get("RouteTableId") or "unknown"))
        if foreign:
            raise BootstrapError(
                "internet gateway is now referenced by non-solution route tables: "
                + ", ".join(sorted(foreign))
            )

    def _assert_oidc_provider_unused(self, resource: InstallationResource) -> None:
        document = _json(["aws", "iam", "list-roles"])
        provider = resource.resource_arn or resource.resource_id
        users = oidc_provider_users(document or {}, provider)
        if users:
            raise BootstrapError(
                "OIDC provider is now trusted by remaining IAM roles: "
                + ", ".join(sorted(users))
            )

    def _assert_pod_identity_agent_unused(self, resource: InstallationResource) -> None:
        document = _json(
            self._aws(
                "eks",
                "list-pod-identity-associations",
                "--cluster-name",
                resource.attributes["cluster_name"],
            ),
            not_found=("ResourceNotFoundException",),
        )
        associations = objects(document, "associations") if document is not None else []
        if associations:
            raise BootstrapError(
                "Pod Identity Agent is now used by remaining associations"
            )

    def _assert_lbc_unused(self) -> None:
        document = _json(
            [
                "kubectl",
                "--kubeconfig",
                self.cpu_kubeconfig,
                "get",
                "service",
                "--all-namespaces",
            ]
        )
        users = [
            (
                f"{item.get('metadata', {}).get('namespace', 'default')}/"
                f"{item.get('metadata', {}).get('name', 'unknown')}"
            )
            for item in objects(document or {}, "items")
            if item.get("spec", {}).get("type") == "LoadBalancer"
        ]
        if users:
            raise BootstrapError(
                "Load Balancer Controller is now used by remaining Services: "
                + ", ".join(sorted(users))
            )
        ingresses = _json(
            [
                "kubectl",
                "--kubeconfig",
                self.cpu_kubeconfig,
                "get",
                "ingress",
                "--all-namespaces",
            ]
        )
        items = objects(ingresses or {}, "items")
        if not items:
            return
        classes = _json(
            ["kubectl", "--kubeconfig", self.cpu_kubeconfig, "get", "ingressclass"]
        )
        controllers = {
            str(object_field(item, "metadata").get("name") or ""): object_field(
                item, "spec"
            ).get("controller")
            for item in objects(classes or {}, "items")
        }
        for item in items:
            metadata = object_field(item, "metadata")
            name = object_field(item, "spec").get("ingressClassName") or metadata.get(
                "annotations", {}
            ).get("kubernetes.io/ingress.class")
            if name == "alb" or controllers.get(name) in {
                None,
                "",
                "ingress.k8s.aws/alb",
            }:
                users.append(
                    f"{metadata.get('namespace', 'default')}/{metadata.get('name', 'unknown')}"
                )
        if users:
            raise BootstrapError(
                "Load Balancer Controller may be used by remaining Ingresses: "
                + ", ".join(sorted(users))
            )

    def _route53_nlb_binding(
        self, resource: InstallationResource
    ) -> dict[str, str] | None:
        document = _json(
            self._aws(
                "elbv2",
                "describe-load-balancers",
                "--names",
                str(self.site.release_config["nlb"]["name"]),
            ),
            not_found=("LoadBalancerNotFound",),
        )
        if document is None:
            return None
        details = single_object(document, "LoadBalancers")
        target = str(details.get("DNSName") or "")
        if (
            not target
            or details.get("LoadBalancerName")
            != self.site.release_config["nlb"]["name"]
        ):
            raise BootstrapError("Route53 record has no unique owned NLB target")
        nlb = resource.model_copy(
            update={
                "resource_type": "nlb",
                "resource_id": str(self.site.release_config["nlb"]["name"]),
                "resource_arn": str(details.get("LoadBalancerArn") or ""),
            }
        )
        self._ownership.assert_tags(nlb, self._ownership._nlb_tags(nlb) or {})
        return {"nlb_arn": str(details["LoadBalancerArn"]), "dns_name": target}

    def prepare_dns_delete(self, resource: InstallationResource) -> dict[str, Any]:
        if resource.resource_type != "route53_record":
            raise BootstrapError("DNS cleanup binding requires a Route53 record")
        if self._ownership.before_delete(resource, self.exists) is None:
            return {"record": None}
        records = self._route53_records(resource)
        target = self._route53_nlb_binding(resource)
        if len(records) != 1 or target is None:
            raise BootstrapError("Route53 record has no unique owned NLB target")
        record = records[0]
        values = objects(record, "ResourceRecords")
        if (
            record.get("Type") != "CNAME"
            or len(values) != 1
            or str(values[0].get("Value") or "").rstrip(".").casefold()
            != target["dns_name"].rstrip(".").casefold()
            or set(record) != {"Name", "Type", "TTL", "ResourceRecords"}
        ):
            raise BootstrapError("Route53 record target or routing policy has drifted")
        return {"record": record, **target}

    def _delete_route53_record(self, resource: InstallationResource) -> None:
        records = self._route53_records(resource)
        if not records:
            return
        encoded = resource.attributes.get("uninstall_dns_binding")
        try:
            binding = (
                json.loads(encoded)
                if encoded is not None
                else self.prepare_dns_delete(resource)
            )
        except (TypeError, ValueError):
            raise BootstrapError("invalid saved DNS cleanup binding") from None
        if (
            not isinstance(binding, dict)
            or len(records) != 1
            or records[0] != binding.get("record")
            or any(
                not isinstance(binding.get(key), str) or not binding[key]
                for key in ("nlb_arn", "dns_name")
            )
        ):
            raise BootstrapError("Route53 record differs from its cleanup binding")
        target = self._route53_nlb_binding(resource)
        if target is not None and target != {
            key: binding[key] for key in ("nlb_arn", "dns_name")
        }:
            raise BootstrapError("Route53 NLB target identity has drifted")
        # The Service finalizer can remove the NLB before AWS cleanup. Only an
        # unchanged record with a previously proven target permits that absence.
        batch = json.dumps(
            {"Changes": [{"Action": "DELETE", "ResourceRecordSet": records[0]}]},
            separators=(",", ":"),
        )
        _checked(
            self._aws(
                "route53",
                "change-resource-record-sets",
                "--hosted-zone-id",
                resource.attributes["hosted_zone_id"],
                "--change-batch",
                batch,
            ),
            not_found=("NoSuchHostedZone",),
        )

    def _detach_sqs_policy(self, resource: InstallationResource) -> None:
        document = _json(
            self._aws(
                "sqs",
                "get-queue-attributes",
                "--queue-url",
                resource.resource_id,
                "--attribute-names",
                "Policy",
            ),
            not_found=(
                "AWS.SimpleQueueService.NonExistentQueue",
                "QueueDoesNotExist",
            ),
        )
        if document is None:
            return
        policy = sqs_policy(object_field(document, "Attributes").get("Policy"))
        matched = sqs_topic_statements(policy, resource.attributes["topic_arn"])
        if not matched:
            return
        statements = [
            statement
            for statement in objects(policy, "Statement")
            if statement not in matched
        ]
        policy["Statement"] = statements
        policy_value = json.dumps(policy, separators=(",", ":")) if statements else ""
        _checked(
            self._aws(
                "sqs",
                "set-queue-attributes",
                "--queue-url",
                resource.resource_id,
                "--attributes",
                json.dumps(
                    {"Policy": policy_value},
                    separators=(",", ":"),
                ),
            ),
            not_found=(
                "AWS.SimpleQueueService.NonExistentQueue",
                "QueueDoesNotExist",
            ),
        )

    def _delete_iam_role(self, resource: InstallationResource) -> None:
        role_name = resource.resource_id
        profiles = _json(
            ["aws", "iam", "list-instance-profiles-for-role", "--role-name", role_name],
            not_found=("NoSuchEntity",),
        )
        if profiles is None:
            return
        if objects(profiles, "InstanceProfiles"):
            raise BootstrapError(
                "IAM role remains attached to instance profiles outside the cleanup registry"
            )
        inline = _json(
            ["aws", "iam", "list-role-policies", "--role-name", role_name],
            not_found=("NoSuchEntity",),
        )
        if inline is None:
            return
        for policy_name in inline.get("PolicyNames", []):
            _checked(
                [
                    "aws",
                    "iam",
                    "delete-role-policy",
                    "--role-name",
                    role_name,
                    "--policy-name",
                    policy_name,
                ],
                not_found=("NoSuchEntity",),
            )
        attached = _json(
            ["aws", "iam", "list-attached-role-policies", "--role-name", role_name],
            not_found=("NoSuchEntity",),
        )
        for policy in (attached or {}).get("AttachedPolicies", []):
            _checked(
                [
                    "aws",
                    "iam",
                    "detach-role-policy",
                    "--role-name",
                    role_name,
                    "--policy-arn",
                    policy["PolicyArn"],
                ],
                not_found=("NoSuchEntity",),
            )
        _checked(
            ["aws", "iam", "delete-role", "--role-name", role_name],
            not_found=("NoSuchEntity",),
        )

    def _delete_iam_policy(self, resource: InstallationResource) -> None:
        arn = resource.resource_arn or resource.resource_id
        entities = _json(
            ["aws", "iam", "list-entities-for-policy", "--policy-arn", arn],
            not_found=("NoSuchEntity",),
        )
        if entities is None:
            return
        if any(
            entities.get(name)
            for name in ("PolicyGroups", "PolicyUsers", "PolicyRoles")
        ):
            raise BootstrapError(f"IAM policy remains attached: {arn}")
        versions = _json(
            ["aws", "iam", "list-policy-versions", "--policy-arn", arn],
            not_found=("NoSuchEntity",),
        )
        for version in (versions or {}).get("Versions", []):
            if version.get("IsDefaultVersion"):
                continue
            _checked(
                [
                    "aws",
                    "iam",
                    "delete-policy-version",
                    "--policy-arn",
                    arn,
                    "--version-id",
                    version["VersionId"],
                ],
                not_found=("NoSuchEntity",),
            )
        _checked(
            ["aws", "iam", "delete-policy", "--policy-arn", arn],
            not_found=("NoSuchEntity",),
        )

    def _delete_route53_zone(self, resource: InstallationResource) -> None:
        document = _json(
            self._aws(
                "route53",
                "list-resource-record-sets",
                "--hosted-zone-id",
                resource.resource_id,
            ),
            not_found=("NoSuchHostedZone",),
        )
        if document is None:
            return
        remaining = [
            item
            for item in objects(document, "ResourceRecordSets")
            if item.get("Type") not in {"NS", "SOA"}
        ]
        if remaining:
            raise BootstrapError(
                f"Route53 zone {resource.resource_id} still has non-default records"
            )
        _checked(
            self._aws(
                "route53",
                "delete-hosted-zone",
                "--id",
                resource.resource_id,
            ),
            not_found=("NoSuchHostedZone",),
        )

    def _delete_detachments(self, resource: InstallationResource) -> bool:
        resource_type = resource.resource_type
        identifier = resource.resource_id
        arn = resource.resource_arn or identifier
        if resource_type == "route53_record":
            self._delete_route53_record(resource)
        elif resource_type == "sns_subscription":
            _checked(
                self._aws(
                    "sns",
                    "unsubscribe",
                    "--subscription-arn",
                    arn,
                ),
                not_found=("NotFound", "NotFoundException"),
            )
        elif resource_type == "sqs_policy_binding":
            self._detach_sqs_policy(resource)
        elif resource_type == "eks_pod_identity_association":
            _checked(
                self._aws(
                    "eks",
                    "delete-pod-identity-association",
                    "--cluster-name",
                    resource.attributes["cluster_name"],
                    "--association-id",
                    identifier,
                ),
                not_found=("ResourceNotFoundException",),
            )
        elif resource_type == "ec2_route_table_association":
            _checked(
                self._aws(
                    "ec2",
                    "disassociate-route-table",
                    "--association-id",
                    identifier,
                ),
                not_found=("InvalidAssociationID.NotFound",),
            )
        elif resource_type == "route53_vpc_association":
            _checked(
                self._aws(
                    "route53",
                    "disassociate-vpc-from-hosted-zone",
                    "--hosted-zone-id",
                    resource.attributes["hosted_zone_id"],
                    "--vpc",
                    (
                        f"VPCRegion={resource.attributes['vpc_region']},"
                        f"VPCId={resource.attributes['vpc_id']}"
                    ),
                ),
                not_found=(
                    "NoSuchHostedZone",
                    "VPCAssociationNotFound",
                ),
            )
        else:
            return False
        return True

    def _delete_edge_services(self, resource: InstallationResource) -> bool:
        resource_type = resource.resource_type
        identifier = resource.resource_id
        arn = resource.resource_arn or identifier
        if resource_type == "nlb_listener":
            _checked(
                self._aws(
                    "elbv2",
                    "delete-listener",
                    "--listener-arn",
                    arn,
                ),
                not_found=("ListenerNotFound",),
            )
        elif resource_type == "nlb":
            _checked(
                self._aws("elbv2", "delete-load-balancer", "--load-balancer-arn", arn),
                not_found=("LoadBalancerNotFound",),
            )
        elif resource_type == "nlb_target_group":
            _checked(
                self._aws(
                    "elbv2",
                    "delete-target-group",
                    "--target-group-arn",
                    arn,
                ),
                not_found=("TargetGroupNotFound",),
            )
        elif resource_type == "helm_release":
            if self.exists(resource):
                self._assert_lbc_unused()
                _checked(
                    [
                        "helm",
                        "--kubeconfig",
                        self.cpu_kubeconfig,
                        "-n",
                        resource.attributes.get("namespace", "default"),
                        "uninstall",
                        identifier,
                    ]
                )
        elif resource_type in {"amp_workspace", "grafana_workspace"}:
            _checked(
                self._aws(
                    resource_type.removesuffix("_workspace"),
                    "delete-workspace",
                    "--workspace-id",
                    identifier,
                ),
                not_found=("ResourceNotFoundException",),
            )
        elif resource_type == "grafana_service_account":
            _checked(
                self._aws(
                    "grafana",
                    "delete-workspace-service-account",
                    "--workspace-id",
                    resource.attributes["workspace_id"],
                    "--service-account-id",
                    identifier,
                ),
                not_found=("ResourceNotFoundException",),
            )
        elif resource_type == "sns_topic":
            _checked(
                self._aws("sns", "delete-topic", "--topic-arn", arn),
                not_found=("NotFound", "NotFoundException"),
            )
        elif resource_type == "sqs_queue":
            _checked(
                self._aws(
                    "sqs",
                    "delete-queue",
                    "--queue-url",
                    identifier,
                ),
                not_found=(
                    "AWS.SimpleQueueService.NonExistentQueue",
                    "QueueDoesNotExist",
                ),
            )
        elif resource_type == "acm_certificate":
            delete_certificate_once_released(self, resource, arn)
        elif resource_type == "secretsmanager_secret":
            _checked(
                self._aws(
                    "secretsmanager",
                    "delete-secret",
                    "--secret-id",
                    identifier,
                    "--force-delete-without-recovery",
                ),
                not_found=("ResourceNotFoundException",),
            )
        elif resource_type == "ses_email_identity":
            _checked(
                self._aws(
                    "sesv2",
                    "delete-email-identity",
                    "--email-identity",
                    identifier,
                ),
                not_found=("NotFoundException",),
            )
        elif resource_type == "ecr_repository":
            _checked(
                self._aws(
                    "ecr",
                    "delete-repository",
                    "--repository-name",
                    identifier,
                    "--force",
                ),
                not_found=("RepositoryNotFoundException",),
            )
        else:
            return False
        return True

    def _delete_identity(self, resource: InstallationResource) -> bool:
        resource_type = resource.resource_type
        identifier = resource.resource_id
        arn = resource.resource_arn or identifier
        if resource_type == "iam_role":
            self._delete_iam_role(resource)
        elif resource_type == "iam_policy":
            self._delete_iam_policy(resource)
        elif resource_type == "iam_oidc_provider":
            self._assert_oidc_provider_unused(resource)
            _checked(
                [
                    "aws",
                    "iam",
                    "delete-open-id-connect-provider",
                    "--open-id-connect-provider-arn",
                    arn,
                ],
                not_found=("NoSuchEntity",),
            )
        elif resource_type == "eks_addon":
            self._assert_pod_identity_agent_unused(resource)
            _checked(
                self._aws(
                    "eks",
                    "delete-addon",
                    "--cluster-name",
                    resource.attributes["cluster_name"],
                    "--addon-name",
                    identifier,
                ),
                not_found=("ResourceNotFoundException",),
            )
        elif resource_type == "route53_zone":
            self._delete_route53_zone(resource)
        else:
            return False
        return True

    def _delete_network_support(self, resource: InstallationResource) -> bool:
        resource_type = resource.resource_type
        identifier = resource.resource_id
        if resource_type == "ec2_route_table":
            _checked(
                self._aws(
                    "ec2",
                    "delete-route-table",
                    "--route-table-id",
                    identifier,
                ),
                not_found=("InvalidRouteTableID.NotFound",),
            )
        elif resource_type == "ec2_subnet":
            _checked(
                self._aws("ec2", "delete-subnet", "--subnet-id", identifier),
                not_found=("InvalidSubnetID.NotFound",),
            )
        elif resource_type == "internet_gateway":
            self._assert_internet_gateway_exclusive(resource)
            vpc_id = resource.attributes.get("vpc_id")
            if vpc_id:
                _checked(
                    self._aws(
                        "ec2",
                        "detach-internet-gateway",
                        "--internet-gateway-id",
                        identifier,
                        "--vpc-id",
                        vpc_id,
                    ),
                    not_found=(
                        "Gateway.NotAttached",
                        "InvalidInternetGatewayID.NotFound",
                    ),
                )
            _checked(
                self._aws(
                    "ec2",
                    "delete-internet-gateway",
                    "--internet-gateway-id",
                    identifier,
                ),
                not_found=("InvalidInternetGatewayID.NotFound",),
            )
        elif resource_type == "security_group":
            _checked(
                self._aws(
                    "ec2",
                    "delete-security-group",
                    "--group-id",
                    identifier,
                ),
                not_found=("InvalidGroup.NotFound",),
            )
        elif resource_type == "rds_db_subnet_group":
            _checked(
                self._aws(
                    "rds",
                    "delete-db-subnet-group",
                    "--db-subnet-group-name",
                    identifier,
                ),
                not_found=("DBSubnetGroupNotFoundFault",),
            )
        elif resource_type == "rds_cluster_parameter_group":
            # Reached from the Aurora phase after the cluster is gone; RDS
            # refuses (InvalidDBParameterGroupState) while a cluster uses it.
            _checked(
                self._aws(
                    "rds",
                    "delete-db-cluster-parameter-group",
                    "--db-cluster-parameter-group-name",
                    identifier,
                ),
                not_found=("DBParameterGroupNotFound",),
            )
        else:
            return False
        return True

    def delete(self, resource: InstallationResource) -> None:
        current = self._ownership.before_delete(resource, self.exists)
        if current is None:
            return
        resource = current
        handled = any(
            handler(resource)
            for handler in (
                self._delete_detachments,
                self._delete_edge_services,
                self._delete_identity,
                self._delete_network_support,
            )
        )
        if not handled:
            raise BootstrapError(
                f"resource must be handled by another cleanup phase: "
                f"{resource.resource_key}"
            )
        self.wait_absent(resource)

    def wait_absent(
        self,
        resource: InstallationResource,
        *,
        timeout_seconds: float = 900,
    ) -> None:
        _wait_until(
            lambda: not self.exists(resource),
            description=f"{resource.resource_key} deletion",
            timeout_seconds=timeout_seconds,
        )


class ClusterDeletion(ResourceDeletion):
    def _aurora_cluster(self, cluster_id: str) -> dict[str, Any] | None:
        document = _json(
            self._aws(
                "rds",
                "describe-db-clusters",
                "--db-cluster-identifier",
                cluster_id,
            ),
            not_found=("DBClusterNotFoundFault",),
        )
        if document is None:
            return None
        return single_object(document, "DBClusters")

    def _snapshot_available(self, identifier: str) -> bool:
        snapshot = aurora_snapshot(
            self,
            identifier,
            cluster_id=str(self.site.release_config["health"]["aurora_cluster_id"]),
        )
        return snapshot is not None and snapshot.get("Status") == "available"

    def _delete_hyperpod(self, name: str, *, expected_arn: str | None = None) -> None:
        delete_hyperpod_cluster(self, name, expected_arn=expected_arn)

    def _delete_eks(self, name: str, *, expected_created_at: object = None) -> None:
        delete_eks_cluster(self, name, expected_created_at=expected_created_at)

    def delete_cpu_cluster(
        self,
        cpu_hyperpod: InstallationResource,
        cpu_eks: InstallationResource,
    ) -> None:
        self._ownership.validate_cpu_pair(cpu_hyperpod, cpu_eks)
        eks = cpu_eks_identity(
            self,
            cpu_eks.resource_id,
            expected_created_at=cpu_eks.attributes.get("cpu_eks_created_at"),
        )
        self._delete_hyperpod(
            cpu_hyperpod.resource_id,
            expected_arn=cpu_hyperpod.attributes.get(
                "cpu_hyperpod_arn", cpu_hyperpod.resource_arn
            ),
        )
        if eks is not None:
            self._delete_eks(cpu_eks.resource_id, expected_created_at=eks["createdAt"])

    def prepare_cpu_delete(
        self, cpu_hyperpod: InstallationResource, cpu_eks: InstallationResource
    ) -> dict[str, Any]:
        return prepare_cpu_deletion(self, cpu_hyperpod, cpu_eks)

    def delete_aurora(
        self,
        cluster: InstallationResource,
        *,
        final_snapshot_policy: FinalSnapshotPolicy,
        final_snapshot_identifier: str,
    ) -> str | None:
        return delete_aurora_cluster(
            self,
            cluster,
            final_snapshot_policy=final_snapshot_policy,
            final_snapshot_identifier=final_snapshot_identifier,
        )

    def prepare_aurora_delete(
        self,
        cluster: InstallationResource,
        *,
        final_snapshot_policy: FinalSnapshotPolicy,
        final_snapshot_identifier: str,
    ) -> dict[str, str]:
        return prepare_aurora_deletion(
            self,
            cluster,
            final_snapshot_policy=final_snapshot_policy,
            final_snapshot_identifier=final_snapshot_identifier,
        )


class ResourceCleaner(ClusterDeletion):
    pass


def delete_priority(resource: InstallationResource) -> int:
    try:
        return DELETE_PRIORITY[resource.resource_type]
    except KeyError as exc:
        raise BootstrapError(
            f"resource has no non-Aurora delete priority: {resource.resource_key}"
        ) from exc


def is_aurora_resource(resource: InstallationResource) -> bool:
    return (
        resource.resource_type in AURORA_RESOURCE_TYPES
        or resource.resource_key.startswith("aws/aurora/")
    )

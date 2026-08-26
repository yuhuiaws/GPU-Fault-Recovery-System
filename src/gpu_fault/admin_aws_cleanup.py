from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, cast

from gpu_fault.admin_aws_commands import (
    FinalSnapshotPolicy,
    checked_command as _checked,
    json_command as _json,
    matches_not_found as _matches_not_found,
    run_command as _run,
    wait_until as _wait_until,
)
from gpu_fault.admin_bootstrap_common import BootstrapError
from gpu_fault.admin_site import RenderedSite
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
        "eks_addon",
        "eks_pod_identity_association",
        "gpu_eks",
        "gpu_hyperpod",
        "helm_release",
        "iam_oidc_provider",
        "iam_policy",
        "iam_role",
        "internet_gateway",
        "nlb",
        "nlb_listener",
        "nlb_target_group",
        "rds_db_subnet_group",
        "rds_managed_secret",
        "rds_snapshot",
        "route53_record",
        "route53_vpc_association",
        "route53_zone",
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
    "nlb_listener": 20,
    "nlb": 30,
    "nlb_target_group": 40,
    "helm_release": 50,
    "amp_workspace": 60,
    "sns_topic": 60,
    "sqs_queue": 60,
    "acm_certificate": 60,
    "secretsmanager_secret": 60,
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
        "rds_db_subnet_group",
        "rds_managed_secret",
    }
)


class ResourceProbe:
    def __init__(self, site: RenderedSite) -> None:
        self.site = site
        self.region = str(site.release_config["aws_region"])
        self.cpu_kubeconfig = str(site.release_config["cpu_kubeconfig"])

    def validate_supported(
        self,
        resources: Iterable[InstallationResource],
    ) -> None:
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
        return [
            "aws",
            service,
            operation,
            "--region",
            self.region,
            *arguments,
        ]

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
            f"{result.stderr.strip()}"
        )

    def _route53_records(
        self,
        resource: InstallationResource,
    ) -> list[dict[str, Any]]:
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
        name = resource.resource_id.rstrip(".") + "."
        record_type = resource.attributes.get("record_type", "CNAME")
        return [
            item
            for item in document.get("ResourceRecordSets", [])
            if item.get("Name") == name and item.get("Type") == record_type
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
            return self._exists_command(
                arguments,
                not_found=("LoadBalancerNotFound",),
            )
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
            return bool(document and document.get("RouteTables"))
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

    def _exists_dns_and_monitoring(
        self,
        resource: InstallationResource,
    ) -> bool | None:
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
                for item in document.get("VPCs", [])
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
        if resource_type in {
            "secretsmanager_secret",
            "rds_managed_secret",
        }:
            return self._exists_command(
                self._aws(
                    "secretsmanager",
                    "describe-secret",
                    "--secret-id",
                    identifier,
                ),
                not_found=("ResourceNotFoundException",),
            )
        if resource_type == "amp_workspace":
            return self._exists_command(
                self._aws(
                    "amp",
                    "describe-workspace",
                    "--workspace-id",
                    identifier,
                ),
                not_found=("ResourceNotFoundException",),
            )
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
            policy = (document.get("Attributes") or {}).get("Policy", "")
            return resource.attributes["topic_arn"] in policy
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
            if "release: not found" in result.stderr.lower():
                return False
            raise BootstrapError(
                f"Helm release verification failed: {result.stderr.strip()}"
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
            return self._exists_command(
                self._aws(
                    "rds",
                    "describe-db-cluster-snapshots",
                    "--db-cluster-snapshot-identifier",
                    identifier,
                ),
                not_found=("DBClusterSnapshotNotFoundFault",),
            )
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
        for route_table in cast(
            list[dict[str, Any]],
            (document or {}).get("RouteTables", []),
        ):
            tags = {
                str(item.get("Key") or ""): str(item.get("Value") or "")
                for item in route_table.get("Tags", [])
            }
            if tags.get("gpu-fault:site-id") != resource.site_id:
                foreign.append(str(route_table.get("RouteTableId") or "unknown"))
        if foreign:
            raise BootstrapError(
                "internet gateway is now referenced by non-solution route tables: "
                + ", ".join(sorted(foreign))
            )

    def _assert_oidc_provider_unused(
        self,
        resource: InstallationResource,
    ) -> None:
        document = _json(["aws", "iam", "list-roles"])
        provider = resource.resource_arn or resource.resource_id
        users = [
            str(role.get("RoleName") or "unknown")
            for role in cast(list[dict[str, Any]], (document or {}).get("Roles", []))
            if provider
            in json.dumps(
                role.get("AssumeRolePolicyDocument") or {},
                sort_keys=True,
            )
        ]
        if users:
            raise BootstrapError(
                "OIDC provider is now trusted by remaining IAM roles: "
                + ", ".join(sorted(users))
            )

    def _assert_pod_identity_agent_unused(
        self,
        resource: InstallationResource,
    ) -> None:
        document = _json(
            self._aws(
                "eks",
                "list-pod-identity-associations",
                "--cluster-name",
                resource.attributes["cluster_name"],
            ),
            not_found=("ResourceNotFoundException",),
        )
        associations = (document or {}).get("associations", [])
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
            for item in cast(list[dict[str, Any]], (document or {}).get("items", []))
            if item.get("spec", {}).get("type") == "LoadBalancer"
        ]
        if users:
            raise BootstrapError(
                "Load Balancer Controller is now used by remaining Services: "
                + ", ".join(sorted(users))
            )

    def _delete_route53_record(self, resource: InstallationResource) -> None:
        for record in self._route53_records(resource):
            batch = json.dumps(
                {
                    "Changes": [
                        {
                            "Action": "DELETE",
                            "ResourceRecordSet": record,
                        }
                    ]
                },
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
        raw = (document.get("Attributes") or {}).get("Policy")
        if not raw:
            return
        policy = json.loads(raw)
        topic_arn = resource.attributes["topic_arn"]
        statements = [
            statement
            for statement in policy.get("Statement", [])
            if topic_arn not in json.dumps(statement, sort_keys=True)
        ]
        policy["Statement"] = statements
        _checked(
            self._aws(
                "sqs",
                "set-queue-attributes",
                "--queue-url",
                resource.resource_id,
                "--attributes",
                json.dumps(
                    {
                        "Policy": json.dumps(
                            policy,
                            separators=(",", ":"),
                        )
                    },
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
        profiles = _json(
            [
                "aws",
                "iam",
                "list-instance-profiles-for-role",
                "--role-name",
                role_name,
            ],
            not_found=("NoSuchEntity",),
        )
        for profile in (profiles or {}).get("InstanceProfiles", []):
            _checked(
                [
                    "aws",
                    "iam",
                    "remove-role-from-instance-profile",
                    "--instance-profile-name",
                    profile["InstanceProfileName"],
                    "--role-name",
                    role_name,
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
            for item in document.get("ResourceRecordSets", [])
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
            if self.exists(resource):
                resolved_arn = resource.resource_arn
                if not resolved_arn:
                    document = _json(
                        self._aws(
                            "elbv2",
                            "describe-load-balancers",
                            "--names",
                            identifier,
                        ),
                        not_found=("LoadBalancerNotFound",),
                    )
                    load_balancers = (document or {}).get("LoadBalancers", [])
                    resolved_arn = (
                        load_balancers[0]["LoadBalancerArn"] if load_balancers else None
                    )
                if resolved_arn:
                    _checked(
                        self._aws(
                            "elbv2",
                            "delete-load-balancer",
                            "--load-balancer-arn",
                            resolved_arn,
                        ),
                        not_found=("LoadBalancerNotFound",),
                    )
            self.wait_absent(resource)
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
        elif resource_type == "amp_workspace":
            _checked(
                self._aws(
                    "amp",
                    "delete-workspace",
                    "--workspace-id",
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
            _checked(
                self._aws(
                    "acm",
                    "delete-certificate",
                    "--certificate-arn",
                    arn,
                ),
                not_found=("ResourceNotFoundException",),
            )
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
        else:
            return False
        return True

    def delete(self, resource: InstallationResource) -> None:
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
    def _aurora_cluster(
        self,
        cluster_id: str,
    ) -> dict[str, Any] | None:
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
        clusters = cast(list[dict[str, Any]], document.get("DBClusters", []))
        return clusters[0] if clusters else None

    def _snapshot_available(self, identifier: str) -> bool:
        document = _json(
            self._aws(
                "rds",
                "describe-db-cluster-snapshots",
                "--db-cluster-snapshot-identifier",
                identifier,
            ),
            not_found=("DBClusterSnapshotNotFoundFault",),
        )
        if document is None:
            return False
        snapshots = cast(
            list[dict[str, Any]],
            document.get("DBClusterSnapshots", []),
        )
        return bool(
            snapshots and str(snapshots[0].get("Status") or "").lower() == "available"
        )

    def _delete_hyperpod(self, name: str) -> None:
        if not self._exists_command(
            self._aws(
                "sagemaker",
                "describe-cluster",
                "--cluster-name",
                name,
            ),
            not_found=("ResourceNotFound",),
        ):
            return
        _checked(
            self._aws("sagemaker", "delete-cluster", "--cluster-name", name),
            not_found=("ResourceNotFound",),
        )
        _wait_until(
            lambda: not self._exists_command(
                self._aws(
                    "sagemaker",
                    "describe-cluster",
                    "--cluster-name",
                    name,
                ),
                not_found=("ResourceNotFound",),
            ),
            description=f"HyperPod cluster {name} deletion",
            timeout_seconds=3600,
            interval_seconds=15,
        )

    def _delete_eks(self, name: str) -> None:
        if not self._exists_command(
            self._aws("eks", "describe-cluster", "--name", name),
            not_found=("ResourceNotFoundException",),
        ):
            return
        associations = _json(
            self._aws(
                "eks",
                "list-pod-identity-associations",
                "--cluster-name",
                name,
            ),
            not_found=("ResourceNotFoundException",),
        )
        for association in (associations or {}).get("associations", []):
            _checked(
                self._aws(
                    "eks",
                    "delete-pod-identity-association",
                    "--cluster-name",
                    name,
                    "--association-id",
                    association["associationId"],
                ),
                not_found=("ResourceNotFoundException",),
            )
        nodegroups = _json(
            self._aws("eks", "list-nodegroups", "--cluster-name", name),
            not_found=("ResourceNotFoundException",),
        )
        for nodegroup in cast(
            list[str],
            (nodegroups or {}).get("nodegroups", []),
        ):
            _checked(
                self._aws(
                    "eks",
                    "delete-nodegroup",
                    "--cluster-name",
                    name,
                    "--nodegroup-name",
                    nodegroup,
                ),
                not_found=("ResourceNotFoundException",),
            )

            def nodegroup_absent() -> bool:
                return not self._exists_command(
                    self._aws(
                        "eks",
                        "describe-nodegroup",
                        "--cluster-name",
                        name,
                        "--nodegroup-name",
                        nodegroup,
                    ),
                    not_found=("ResourceNotFoundException",),
                )

            _wait_until(
                nodegroup_absent,
                description=f"EKS nodegroup {nodegroup} deletion",
                timeout_seconds=3600,
                interval_seconds=15,
            )
        profiles = _json(
            self._aws("eks", "list-fargate-profiles", "--cluster-name", name),
            not_found=("ResourceNotFoundException",),
        )
        for profile in cast(
            list[str],
            (profiles or {}).get("fargateProfileNames", []),
        ):
            _checked(
                self._aws(
                    "eks",
                    "delete-fargate-profile",
                    "--cluster-name",
                    name,
                    "--fargate-profile-name",
                    profile,
                ),
                not_found=("ResourceNotFoundException",),
            )

            def profile_absent() -> bool:
                return not self._exists_command(
                    self._aws(
                        "eks",
                        "describe-fargate-profile",
                        "--cluster-name",
                        name,
                        "--fargate-profile-name",
                        profile,
                    ),
                    not_found=("ResourceNotFoundException",),
                )

            _wait_until(
                profile_absent,
                description=f"EKS Fargate profile {profile} deletion",
                timeout_seconds=3600,
                interval_seconds=15,
            )
        addons = _json(
            self._aws("eks", "list-addons", "--cluster-name", name),
            not_found=("ResourceNotFoundException",),
        )
        for addon in (addons or {}).get("addons", []):
            _checked(
                self._aws(
                    "eks",
                    "delete-addon",
                    "--cluster-name",
                    name,
                    "--addon-name",
                    addon,
                ),
                not_found=("ResourceNotFoundException",),
            )
        _checked(
            self._aws("eks", "delete-cluster", "--name", name),
            not_found=("ResourceNotFoundException",),
        )
        _wait_until(
            lambda: not self._exists_command(
                self._aws("eks", "describe-cluster", "--name", name),
                not_found=("ResourceNotFoundException",),
            ),
            description=f"EKS cluster {name} deletion",
            timeout_seconds=3600,
            interval_seconds=15,
        )

    def delete_cpu_cluster(
        self,
        cpu_hyperpod: InstallationResource,
        cpu_eks: InstallationResource,
    ) -> None:
        self._delete_hyperpod(cpu_hyperpod.resource_id)
        self._delete_eks(cpu_eks.resource_id)

    def delete_aurora(
        self,
        cluster: InstallationResource,
        *,
        final_snapshot_policy: FinalSnapshotPolicy,
        final_snapshot_identifier: str,
    ) -> str | None:
        database = self._aurora_cluster(cluster.resource_id)
        retained = (
            final_snapshot_identifier if final_snapshot_policy == "retain" else None
        )
        if database is None:
            if retained and not self._snapshot_available(retained):
                raise BootstrapError(
                    "Aurora cluster is absent but its required final snapshot "
                    f"is not available: {retained}"
                )
            return retained
        document = _json(
            self._aws(
                "rds",
                "describe-db-instances",
                "--filters",
                f"Name=db-cluster-id,Values={cluster.resource_id}",
            )
        )
        for instance in cast(
            list[dict[str, Any]],
            (document or {}).get("DBInstances", []),
        ):
            instance_id = instance["DBInstanceIdentifier"]
            if str(instance.get("DBInstanceStatus") or "") != "deleting":
                _checked(
                    self._aws(
                        "rds",
                        "delete-db-instance",
                        "--db-instance-identifier",
                        instance_id,
                        "--skip-final-snapshot",
                        "--delete-automated-backups",
                    ),
                    not_found=("DBInstanceNotFound",),
                )

            def instance_absent() -> bool:
                return not self._exists_command(
                    self._aws(
                        "rds",
                        "describe-db-instances",
                        "--db-instance-identifier",
                        instance_id,
                    ),
                    not_found=("DBInstanceNotFound",),
                )

            _wait_until(
                instance_absent,
                description=f"Aurora instance {instance_id} deletion",
                timeout_seconds=3600,
                interval_seconds=15,
            )
        database = self._aurora_cluster(cluster.resource_id)
        if database is not None and database.get("Status") != "deleting":
            if database.get("DeletionProtection"):
                _checked(
                    self._aws(
                        "rds",
                        "modify-db-cluster",
                        "--db-cluster-identifier",
                        cluster.resource_id,
                        "--no-deletion-protection",
                        "--apply-immediately",
                    ),
                    not_found=("DBClusterNotFoundFault",),
                )
                _wait_until(
                    lambda: (
                        (current := self._aurora_cluster(cluster.resource_id)) is None
                        or not current.get("DeletionProtection")
                    ),
                    description=(
                        f"Aurora cluster {cluster.resource_id} "
                        "deletion protection disablement"
                    ),
                    timeout_seconds=900,
                    interval_seconds=10,
                )
            arguments = self._aws(
                "rds",
                "delete-db-cluster",
                "--db-cluster-identifier",
                cluster.resource_id,
                "--delete-automated-backups",
            )
            if retained:
                arguments.extend(
                    [
                        "--final-db-snapshot-identifier",
                        retained,
                    ]
                )
            else:
                arguments.append("--skip-final-snapshot")
            _checked(arguments, not_found=("DBClusterNotFoundFault",))
        self.wait_absent(cluster, timeout_seconds=3600)
        if retained:
            _wait_until(
                lambda: self._snapshot_available(retained),
                description=f"Aurora final snapshot {retained} availability",
                timeout_seconds=3600,
                interval_seconds=15,
            )
        return retained


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


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.chmod(0o600)
    temporary.replace(path)

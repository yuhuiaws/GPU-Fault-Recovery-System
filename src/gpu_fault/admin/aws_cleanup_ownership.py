"""Live ownership and binding checks for administrator AWS cleanup."""

from __future__ import annotations

import hashlib
import threading
from typing import TYPE_CHECKING, Callable
from urllib.parse import urlsplit

from gpu_fault.admin.aws_cleanup_helpers import (
    object_field,
    objects,
    single_object,
    strict_tags,
)
from gpu_fault.admin.aws_commands import json_command
from gpu_fault.admin.bootstrap_common import (
    SITE_TAG_KEY,
    Arn,
    BootstrapError,
    assert_site_tag,
)
from gpu_fault.installation_resources import (
    InstallationResource,
    InstallationResourceDeletePolicy,
    InstallationResourceOwnership,
)

if TYPE_CHECKING:
    from gpu_fault.admin.site import RenderedSite


class CleanupOwnership:
    def __init__(self, site: RenderedSite) -> None:
        self.site = site
        self.config = site.release_config
        self.cpu = Arn.parse(str(self.config["cpu_eks_arn"]))
        self.region = str(self.config["aws_region"])
        self.site_id = str(self.config["site_name"])
        self._caller_checked = False
        self._caller_lock = threading.Lock()

    def aws(self, service: str, operation: str, *arguments: str) -> list[str]:
        return ["aws", service, operation, "--region", self.region, *arguments]

    def validate_scope(self, resource: InstallationResource) -> None:
        if (
            resource.site_id != self.site.registry_site_id
            or resource.region != self.region
            or resource.account_id != self.cpu.account
            or resource.provider
            != ("kubernetes" if resource.resource_type == "helm_release" else "aws")
        ):
            raise BootstrapError(
                f"cleanup resource site/Region/account/provider mismatch: "
                f"{resource.resource_key}"
            )
        if resource.resource_arn:
            self.validate_arn(resource.resource_arn)
        if resource.resource_id.startswith("arn:"):
            self.validate_arn(resource.resource_id)
        if (
            resource.resource_id.startswith("arn:")
            and resource.resource_arn
            and resource.resource_id != resource.resource_arn
        ):
            raise BootstrapError("cleanup resource ID and ARN do not match")
        if resource.resource_type in {"sqs_queue", "sqs_policy_binding"}:
            url = urlsplit(resource.resource_id)
            suffix = (
                "amazonaws.com.cn"
                if self.cpu.partition == "aws-cn"
                else "amazonaws.com"
            )
            if (
                url.scheme != "https"
                or url.netloc != f"sqs.{self.region}.{suffix}"
                or len(url.path.split("/")) != 3
                or url.path.split("/")[1] != self.cpu.account
                or not url.path.split("/")[2]
                or url.query
                or url.fragment
            ):
                raise BootstrapError(
                    "SQS queue URL is outside the current site account/Region"
                )
        cluster_name = resource.attributes.get("cluster_name")
        if cluster_name and cluster_name not in {
            self.cpu.resource_name,
            *(
                Arn.parse(str(cluster["eks_cluster_arn"])).resource_name
                for cluster in self.config["clusters"]
            ),
        }:
            raise BootstrapError(
                "cleanup resource refers to a cluster outside the site"
            )

    def validate_arn(self, value: str) -> None:
        try:
            arn = Arn.parse(value)
        except BootstrapError:
            raise BootstrapError("cleanup resource ARN is malformed") from None
        if (
            arn.partition != self.cpu.partition
            or (
                arn.account != self.cpu.account
                and not (
                    arn.service == "route53" and not arn.account and not arn.region
                )
            )
            or (arn.region and arn.region != self.region)
        ):
            raise BootstrapError("cleanup resource ARN is outside the current site")

    def assert_caller(self) -> None:
        with self._caller_lock:
            if self._caller_checked:
                return
            identity = json_command(self.aws("sts", "get-caller-identity"))
            if identity is None or identity.get("Account") != self.cpu.account:
                raise BootstrapError(
                    "cleanup AWS caller account does not match the site"
                )
            self._caller_checked = True

    def before_delete(
        self,
        resource: InstallationResource,
        exists: Callable[[InstallationResource], bool],
    ) -> InstallationResource | None:
        self.validate_scope(resource)
        if (
            not isinstance(resource.delete_policy, InstallationResourceDeletePolicy)
            or not isinstance(resource.ownership, InstallationResourceOwnership)
            or resource.delete_policy is InstallationResourceDeletePolicy.PRESERVE
            or (
                resource.ownership is InstallationResourceOwnership.EXTERNAL
                and resource.delete_policy
                is not InstallationResourceDeletePolicy.DETACH
            )
            or resource.resource_type
            in {"gpu_eks", "gpu_hyperpod", "cpu_eks", "cpu_hyperpod", "rds_snapshot"}
        ):
            raise BootstrapError(
                f"cleanup cannot delete a preserved or external resource: "
                f"{resource.resource_key}"
            )
        self.assert_caller()
        if not exists(resource):
            return None
        if resource.resource_type == "nlb" and not resource.resource_arn:
            document = json_command(
                self.aws(
                    "elbv2", "describe-load-balancers", "--names", resource.resource_id
                ),
                not_found=("LoadBalancerNotFound",),
            )
            if document is None:
                return None
            details = single_object(document, "LoadBalancers")
            if details.get("LoadBalancerName") != resource.resource_id:
                raise BootstrapError("NLB name no longer matches the registry")
            arn = str(details.get("LoadBalancerArn") or "")
            self.validate_arn(arn)
            resource = resource.model_copy(update={"resource_arn": arn})
        binding = self._binding(resource)
        if binding is not None:
            return resource if binding else None
        tags = self._tags(resource)
        if tags is None:
            return None
        self.assert_tags(resource, tags)
        return resource

    def assert_tags(self, resource: InstallationResource, tags: dict[str, str]) -> None:
        if resource.resource_type.startswith("nlb"):
            owner = tags.get(SITE_TAG_KEY)
            if owner is None:
                if (
                    tags.get("elbv2.k8s.aws/cluster") != self.cpu.resource_name
                    or tags.get("service.k8s.aws/stack")
                    != "gpu-fault-system/gpu-fault-api-nlb"
                ):
                    raise BootstrapError("NLB controller ownership has drifted")
                return
        assert_site_tag(tags, site_id=self.site_id, description=resource.resource_key)
        if resource.resource_type == "grafana_workspace":
            from gpu_fault.admin.grafana import CREATED_TAG_KEY, CREATED_TAG_VALUE

            if tags.get(CREATED_TAG_KEY) != CREATED_TAG_VALUE:
                raise BootstrapError("Grafana workspace has no solution creation tag")

    def _arn(self, resource: InstallationResource, service: str, name: str) -> str:
        expected = f"arn:{self.cpu.partition}:{service}:{self.region}:{self.cpu.account}:{name}"
        if resource.resource_arn and resource.resource_arn != expected:
            raise BootstrapError("cleanup resource ID and ARN do not match")
        return expected

    def _tags(self, resource: InstallationResource) -> dict[str, str] | None:
        kind = resource.resource_type
        identifier = resource.resource_id
        arn = resource.resource_arn or identifier
        field = "Tags"
        absent: tuple[str, ...] = ()
        if kind in {
            "security_group",
            "ec2_subnet",
            "ec2_route_table",
            "internet_gateway",
        }:
            operation, option, key, code = {
                "security_group": (
                    "describe-security-groups",
                    "--group-ids",
                    "SecurityGroups",
                    "InvalidGroup.NotFound",
                ),
                "ec2_subnet": (
                    "describe-subnets",
                    "--subnet-ids",
                    "Subnets",
                    "InvalidSubnetID.NotFound",
                ),
                "ec2_route_table": (
                    "describe-route-tables",
                    "--route-table-ids",
                    "RouteTables",
                    "InvalidRouteTableID.NotFound",
                ),
                "internet_gateway": (
                    "describe-internet-gateways",
                    "--internet-gateway-ids",
                    "InternetGateways",
                    "InvalidInternetGatewayID.NotFound",
                ),
            }[kind]
            document = json_command(
                self.aws("ec2", operation, option, identifier), not_found=(code,)
            )
            if document is None:
                return None
            details = single_object(document, key)
            id_field = {
                "security_group": "GroupId",
                "ec2_subnet": "SubnetId",
                "ec2_route_table": "RouteTableId",
                "internet_gateway": "InternetGatewayId",
            }[kind]
            if (
                details.get("OwnerId") != self.cpu.account
                or details.get(id_field) != identifier
            ):
                raise BootstrapError(
                    "EC2 resource owner or ID does not match the site record"
                )
            return strict_tags(details.get("Tags"))
        elif kind == "ecr_repository":
            command = self.aws(
                "ecr",
                "list-tags-for-resource",
                "--resource-arn",
                self._arn(resource, "ecr", f"repository/{identifier}"),
            )
            field, absent = "tags", ("RepositoryNotFoundException",)
        elif kind in {"iam_role", "iam_policy", "iam_oidc_provider"}:
            operation, option = {
                "iam_role": ("list-role-tags", "--role-name"),
                "iam_policy": ("list-policy-tags", "--policy-arn"),
                "iam_oidc_provider": (
                    "list-open-id-connect-provider-tags",
                    "--open-id-connect-provider-arn",
                ),
            }[kind]
            if kind == "iam_role" and resource.resource_arn:
                if Arn.parse(arn).resource.rsplit("/", 1)[-1] != identifier:
                    raise BootstrapError("IAM role name and ARN do not match")
            command = self.aws(
                "iam", operation, option, identifier if kind == "iam_role" else arn
            )
            absent = ("NoSuchEntity",)
        elif kind in {
            "aurora_cluster",
            "aurora_instance",
            "rds_db_subnet_group",
            "rds_cluster_parameter_group",
        }:
            prefix, code = {
                "aurora_cluster": ("cluster", "DBClusterNotFoundFault"),
                "aurora_instance": ("db", "DBInstanceNotFound"),
                "rds_db_subnet_group": ("subgrp", "DBSubnetGroupNotFoundFault"),
                "rds_cluster_parameter_group": (
                    "cluster-pg",
                    "DBParameterGroupNotFound",
                ),
            }[kind]
            command = self.aws(
                "rds",
                "list-tags-for-resource",
                "--resource-name",
                self._arn(resource, "rds", f"{prefix}:{identifier}"),
            )
            field, absent = "TagList", (code,)
        elif kind == "secretsmanager_secret":
            command = self.aws(
                "secretsmanager", "describe-secret", "--secret-id", identifier
            )
            absent = ("ResourceNotFoundException",)
        elif kind == "acm_certificate":
            command = self.aws(
                "acm", "list-tags-for-certificate", "--certificate-arn", arn
            )
            absent = ("ResourceNotFoundException",)
        elif kind in {"amp_workspace", "grafana_workspace"}:
            service, arn_service, prefix = (
                ("amp", "aps", "workspace/")
                if kind == "amp_workspace"
                else ("grafana", "grafana", "/workspaces/")
            )
            command = self.aws(
                service,
                "list-tags-for-resource",
                "--resource-arn",
                self._arn(resource, arn_service, prefix + identifier),
            )
            field, absent = "tags", ("ResourceNotFoundException",)
        elif kind == "sns_topic":
            command = self.aws("sns", "list-tags-for-resource", "--resource-arn", arn)
            absent = ("NotFound", "NotFoundException")
        elif kind == "sqs_queue":
            command = self.aws("sqs", "list-queue-tags", "--queue-url", identifier)
            absent = ("AWS.SimpleQueueService.NonExistentQueue", "QueueDoesNotExist")
        elif kind == "ses_email_identity":
            command = self.aws(
                "sesv2", "get-email-identity", "--email-identity", identifier
            )
            absent = ("NotFoundException",)
        elif kind == "route53_zone":
            document = json_command(
                self.aws(
                    "route53",
                    "list-tags-for-resource",
                    "--resource-type",
                    "hostedzone",
                    "--resource-id",
                    identifier,
                ),
                not_found=("NoSuchHostedZone",),
            )
            return (
                strict_tags(object_field(document, "ResourceTagSet").get("Tags"))
                if document is not None
                else None
            )
        elif kind.startswith("nlb"):
            return self._nlb_tags(resource)
        elif kind == "eks_addon":
            if (
                resource.attributes["cluster_name"] != self.cpu.resource_name
                or identifier != "eks-pod-identity-agent"
            ):
                raise BootstrapError("EKS add-on is outside the CPU platform binding")
            document = json_command(
                self.aws(
                    "eks",
                    "describe-addon",
                    "--cluster-name",
                    resource.attributes["cluster_name"],
                    "--addon-name",
                    identifier,
                ),
                not_found=("ResourceNotFoundException",),
            )
            return (
                strict_tags(object_field(document, "addon").get("tags"))
                if document is not None
                else None
            )
        else:
            raise BootstrapError(f"cleanup has no ownership proof for {kind}")
        document = json_command(command, not_found=absent)
        return strict_tags(document.get(field)) if document is not None else None

    def _nlb_tags(self, resource: InstallationResource) -> dict[str, str] | None:
        arn = resource.resource_arn or resource.resource_id
        if not arn.startswith("arn:"):
            document = json_command(
                self.aws(
                    "elbv2", "describe-load-balancers", "--names", resource.resource_id
                ),
                not_found=("LoadBalancerNotFound",),
            )
            if document is None:
                return None
            arn = str(
                single_object(document, "LoadBalancers").get("LoadBalancerArn") or ""
            )
        self.validate_arn(arn)
        parsed = Arn.parse(arn)
        parts = parsed.resource.split("/")
        if parsed.service != "elasticloadbalancing":
            raise BootstrapError("NLB ARN does not identify an ELB resource")
        if resource.resource_type == "nlb" and (
            len(parts) != 4
            or parts[:2] != ["loadbalancer", "net"]
            or parts[2] != resource.resource_id
        ):
            raise BootstrapError("NLB ARN does not match its registered name")
        if resource.resource_type == "nlb_listener":
            if len(parts) != 5 or parts[:2] != ["listener", "net"]:
                raise BootstrapError(
                    "NLB listener does not identify a network load balancer"
                )
            arn = arn.rsplit(":", 1)[0] + ":loadbalancer/" + "/".join(parts[1:4])
        document = json_command(
            self.aws("elbv2", "describe-tags", "--resource-arns", arn),
            not_found=("LoadBalancerNotFound", "TargetGroupNotFound"),
        )
        if document is None:
            return None
        tags = single_object(document, "TagDescriptions")
        if tags.get("ResourceArn") != arn:
            raise BootstrapError("NLB ownership response identity mismatch")
        return strict_tags(tags.get("Tags"))

    def _binding(self, resource: InstallationResource) -> bool | None:
        kind = resource.resource_type
        attributes = resource.attributes
        if kind == "sqs_policy_binding":
            self.validate_arn(attributes["topic_arn"])
            return True
        if kind == "route53_record":
            if (
                attributes["hosted_zone_id"]
                != self.config.get("dns", {}).get("hosted_zone_id")
                or resource.resource_id.rstrip(".")
                != str(self.config.get("dns", {}).get("hostname") or "").rstrip(".")
                or attributes.get("record_type", "CNAME") != "CNAME"
            ):
                raise BootstrapError("Route53 record is outside the site DNS binding")
            return True
        if kind == "route53_vpc_association":
            if (
                attributes["hosted_zone_id"]
                != self.config.get("dns", {}).get("hosted_zone_id")
                or attributes["vpc_region"] != self.region
                or resource.resource_id
                != f"{attributes['hosted_zone_id']}:{self.region}:{attributes['vpc_id']}"
            ):
                raise BootstrapError("Route53 VPC association identity mismatch")
            return True
        if kind == "ec2_route_table_association":
            document = json_command(
                self.aws(
                    "ec2",
                    "describe-route-tables",
                    "--filters",
                    "Name=association.route-table-association-id,"
                    f"Values={resource.resource_id}",
                )
            )
            tables = objects(document or {}, "RouteTables")
            if not tables:
                return False
            if len(tables) != 1:
                raise BootstrapError("route-table association ownership is ambiguous")
            table = tables[0]
            associations = [
                item
                for item in objects(table, "Associations")
                if item.get("RouteTableAssociationId") == resource.resource_id
            ]
            if (
                len(associations) != 1
                or associations[0].get("Main") is not False
                or table.get("OwnerId") != self.cpu.account
            ):
                raise BootstrapError(
                    "cannot detach a main or unknown route-table association"
                )
            assert_site_tag(
                strict_tags(table.get("Tags")),
                site_id=self.site_id,
                description=resource.resource_key,
            )
            return True
        if kind == "eks_pod_identity_association":
            return self._pod_identity_binding(resource)
        if kind == "sns_subscription":
            return self._subscription_binding(resource)
        if kind == "grafana_service_account":
            document = json_command(
                self.aws(
                    "grafana",
                    "list-workspace-service-accounts",
                    "--workspace-id",
                    attributes["workspace_id"],
                ),
                not_found=("ResourceNotFoundException",),
            )
            if document is None:
                return False
            accounts = [
                item
                for item in objects(document, "serviceAccounts")
                if str(item.get("id")) == resource.resource_id
            ]
            if not accounts:
                return False
            if (
                len(accounts) != 1
                or not attributes.get("name")
                or accounts[0].get("name") != attributes["name"]
            ):
                raise BootstrapError("Grafana service account identity has drifted")
            return True
        if kind == "helm_release":
            if (
                resource.resource_id != "aws-load-balancer-controller"
                or attributes.get("namespace") != "kube-system"
            ):
                raise BootstrapError(
                    "cleanup Helm release is outside the CPU controller binding"
                )
            values = json_command(
                [
                    "helm",
                    "--kubeconfig",
                    str(self.config["cpu_kubeconfig"]),
                    "-n",
                    "kube-system",
                    "get",
                    "values",
                    resource.resource_id,
                ]
            )
            if values is None or values.get("clusterName") != self.cpu.resource_name:
                raise BootstrapError("Helm controller cluster ownership has drifted")
            return True
        return None

    def _pod_identity_binding(self, resource: InstallationResource) -> bool:
        attributes = resource.attributes
        document = json_command(
            self.aws(
                "eks",
                "describe-pod-identity-association",
                "--cluster-name",
                attributes["cluster_name"],
                "--association-id",
                resource.resource_id,
            ),
            not_found=("ResourceNotFoundException",),
        )
        if document is None:
            return False
        association = object_field(document, "association")
        expected = {
            "associationId": resource.resource_id,
            "clusterName": attributes["cluster_name"],
            "namespace": attributes.get("namespace"),
            "serviceAccount": attributes.get("service_account"),
        }
        if any(
            not value or association.get(key) != value
            for key, value in expected.items()
        ):
            raise BootstrapError("Pod Identity association binding has drifted")
        arn = str(association.get("roleArn") or "")
        self.validate_arn(arn)
        role = json_command(
            self.aws(
                "iam",
                "get-role",
                "--role-name",
                Arn.parse(arn).resource.rsplit("/", 1)[-1],
            )
        )
        details = object_field(role or {}, "Role")
        if details.get("Arn") != arn:
            raise BootstrapError("Pod Identity role identity has drifted")
        assert_site_tag(
            strict_tags(details.get("Tags")),
            site_id=self.site_id,
            description=resource.resource_key,
        )
        return True

    def _subscription_binding(self, resource: InstallationResource) -> bool:
        arn = resource.resource_arn or resource.resource_id
        document = json_command(
            self.aws("sns", "get-subscription-attributes", "--subscription-arn", arn),
            not_found=("NotFound", "NotFoundException"),
        )
        if document is None:
            return False
        attributes = object_field(document, "Attributes")
        if (
            attributes.get("SubscriptionArn") != arn
            or attributes.get("TopicArn") != self.config["health"].get("sns_topic_arn")
            or attributes.get("Owner") != self.cpu.account
        ):
            raise BootstrapError("SNS subscription binding has drifted")
        digest = resource.attributes.get("endpoint_sha256")
        if (
            digest
            and digest
            != hashlib.sha256(
                str(attributes.get("Endpoint") or "").casefold().encode()
            ).hexdigest()
        ):
            raise BootstrapError("SNS subscription endpoint identity has drifted")
        generation = resource.attributes.get("topic_generation")
        if generation:
            from gpu_fault.admin.monitoring_policy import SNS_TOPIC_GENERATION_TAG

            tags = json_command(
                self.aws(
                    "sns",
                    "list-tags-for-resource",
                    "--resource-arn",
                    str(attributes["TopicArn"]),
                )
            )
            if (
                strict_tags((tags or {}).get("Tags")).get(SNS_TOPIC_GENERATION_TAG)
                != generation
            ):
                raise BootstrapError("SNS subscription topic generation has drifted")
        return True

    def validate_cpu_pair(
        self, hyperpod: InstallationResource, eks: InstallationResource
    ) -> None:
        for resource in (hyperpod, eks):
            self.validate_scope(resource)
        if (
            hyperpod.resource_type != "cpu_hyperpod"
            or eks.resource_type != "cpu_eks"
            or hyperpod.resource_id != self.config["cpu_hyperpod_cluster_name"]
            or eks.resource_id != self.cpu.resource_name
            or eks.resource_arn != self.config["cpu_eks_arn"]
            or any(
                cluster["eks_cluster_arn"] == eks.resource_arn
                or cluster["hyperpod_cluster_name"] == hyperpod.resource_id
                for cluster in self.config["clusters"]
            )
        ):
            raise BootstrapError(
                "CPU deletion targets do not match the site CPU clusters"
            )
        self.assert_caller()

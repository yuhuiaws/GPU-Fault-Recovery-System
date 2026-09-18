from __future__ import annotations

import base64
import copy
import json
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any

import pytest

from gpu_fault.admin.capacity_defaults import INGRESS_REPLICAS
from gpu_fault.admin.config import default_admin_config
from gpu_fault_release import regional_admin_checks as checks
from gpu_fault_release.regional_release_state import read_snapshot
from tests.regional._cov95_release_support import ResourceRelease, deployment

TOPIC = "arn:aws:sns:us-east-1:123456789012:example"
CERTIFICATE = "arn:aws:acm:us-east-1:123456789012:certificate/example"
HOSTNAME = "api.example.test"


class CheckedRelease(ResourceRelease):
    def __init__(self) -> None:
        super().__init__()
        self.config.clusters = tuple(
            replace(target, control_plane_url=f"https://{HOSTNAME}")
            for target in self.config.clusters
        )
        self.config.cpu_hyperpod_cluster_name = "hp-cpu"
        self.config.admin_config = default_admin_config()
        self.config.notifications = SimpleNamespace(
            allow_email=False,
            acknowledge_external_alert_channel=True,
            admin_email="ops@example.com",
        )
        self.config.health = SimpleNamespace(
            certificate_min_validity_days=7,
            aurora_cluster_id="aurora-example",
            amp_workspace_id="workspace-example",
            sns_topic_arn=TOPIC,
            amp_rule_namespace="example-rules",
            require_confirmed_sns_subscription=True,
            remote_command_max_unclaimed_seconds=30,
        )
        self.config.nlb = {
            "certificate_arn": CERTIFICATE,
            "name": "example-nlb",
            "public_subnets": "subnet-a,subnet-b",
            "security_group": "sg-example",
        }
        self.config.retention = None
        self.context_checks = 0
        self.prime_calls = 0
        self.tls = {"status": "ok"}
        self.api: Any = {
            "healthz": {"status": "ok"},
            "version": {
                "deployment_mode": "regional",
                "required_agent_artifact_sha256": self.node_wheel_sha,
                "required_agent_config_digest": self.config.agent_config_digest,
                "required_agent_compatibility_digest": self.config.component_digests[
                    "node_runtime"
                ],
                "required_runtime_profile_version": self.config.runtime_profile_version,
                "required_node_action_key_version": 2,
                "required_regional_executor_artifact_sha256": self.executor_wheel_sha,
                "required_regional_executor_compatibility_digest": self.config.component_digests[
                    "executor"
                ],
            },
            "registry": [{"cluster_id": "gpu-a", "lifecycle_state": "ACTIVE"}],
            "clusters": {
                "gpu-a": {
                    "expected_node_ids": ["node-a"],
                    "agents": [
                        {
                            "node_id": "node-a",
                            "lifecycle_state": "ACTIVE",
                            "runtime_profile_version": self.config.runtime_profile_version,
                        }
                    ],
                    "fleet_readiness": {"ready": True},
                    "collector_readiness": {"ready": True},
                }
            },
            "remote_commands": {"executor_internal_error_total": 0},
        }
        for name, replicas in {
            "gpu-fault-api-ha": INGRESS_REPLICAS,
            "gpu-fault-control-worker": 3,
            "gpu-fault-telemetry-spool-worker": 0,
            "gpu-fault-adot": 1,
        }.items():
            self.documents[("cpu", "deployment", name)] = deployment(
                name, replicas=replicas
            )
        self.documents[("cpu", "configmap", "gpu-fault-api-ha-config-core")] = {
            "data": {
                "GPU_FAULT_REQUIRED_RUNTIME_PROFILE_VERSION": self.config.runtime_profile_version,
                "GPU_FAULT_ALLOW_EMAIL": "false",
                "GPU_FAULT_ACKNOWLEDGE_NO_ALERT_CHANNEL": "true",
            }
        }
        for item in self.documents[("gpu-a", "nodes", "")]["items"]:
            item["status"] = {"conditions": [{"type": "Ready", "status": "True"}]}
        self.documents[("cpu", "nodes", "")] = {
            "items": [
                {
                    "metadata": {"name": f"cpu-{index}"},
                    "status": {
                        "conditions": [{"type": "Ready", "status": "True"}],
                        "allocatable": {"pods": "30"},
                    },
                }
                for index in range(3)
            ]
        }
        self.documents[("cpu", "pods", "")] = {
            "items": [{"spec": {"nodeName": "not-managed"}}]
        }
        self.documents[("cpu", "deployment", "")] = {
            "items": [
                deployment("unrelated"),
                deployment("aws-load-balancer-controller"),
            ]
        }
        self.aws = self.build_healthy_aws_responses()
        self.runner.handler = self.dispatch

    @staticmethod
    def build_healthy_aws_responses() -> dict[tuple[str, str], Any]:
        return {
            ("sts", "get-caller-identity"): {
                "Account": "123456789012",
                "Arn": "arn:aws:iam::123456789012:role/example",
            },
            ("eks", "describe-cluster"): {
                "cluster": {"resourcesVpcConfig": {"vpcId": "vpc-example"}}
            },
            ("ec2", "describe-subnets"): {
                "Subnets": [
                    {
                        "SubnetId": "subnet-a",
                        "VpcId": "vpc-example",
                        "AvailabilityZone": "us-east-1a",
                    },
                    {
                        "SubnetId": "subnet-b",
                        "VpcId": "vpc-example",
                        "AvailabilityZone": "us-east-1b",
                    },
                ]
            },
            ("ec2", "describe-route-tables"): {
                "RouteTables": [
                    {
                        "Routes": [
                            {
                                "DestinationCidrBlock": "0.0.0.0/0",
                                "GatewayId": "igw-example",
                            }
                        ]
                    }
                ]
            },
            ("ec2", "describe-security-groups"): {
                "SecurityGroups": [
                    {
                        "VpcId": "vpc-example",
                        "IpPermissions": [
                            {
                                "IpProtocol": "tcp",
                                "FromPort": 443,
                                "ToPort": 443,
                                "IpRanges": [{"CidrIp": "10.0.0.0/16"}],
                            }
                        ],
                    }
                ]
            },
            ("acm", "describe-certificate"): {
                "Certificate": {
                    "Status": "ISSUED",
                    "NotAfter": (
                        datetime.now(timezone.utc) + timedelta(days=180)
                    ).isoformat(),
                    "SubjectAlternativeNames": [HOSTNAME],
                }
            },
            ("rds", "describe-db-clusters"): {
                "DBClusters": [{"Status": "available", "Endpoint": "example.invalid"}]
            },
            ("rds", "describe-db-instances"): {
                "DBInstances": [
                    {
                        "DBClusterIdentifier": "aurora-example",
                        "DBInstanceIdentifier": name,
                        "DBInstanceStatus": "available",
                    }
                    for name in ("writer", "reader")
                ]
            },
            ("amp", "describe-workspace"): {
                "workspace": {"status": {"statusCode": "ACTIVE"}}
            },
            ("amp", "describe-rule-groups-namespace"): {
                "ruleGroupsNamespace": {
                    "data": base64.b64encode(b"groups: []\n").decode()
                }
            },
            ("amp", "describe-alert-manager-definition"): {
                "alertManagerDefinition": {
                    "data": base64.b64encode(f"receiver: {TOPIC}\n".encode()).decode()
                }
            },
            ("sns", "list-subscriptions-by-topic"): {
                "Subscriptions": [
                    {
                        "Protocol": "email",
                        "Endpoint": "ops@example.com",
                        "SubscriptionArn": TOPIC + ":confirmed",
                    }
                ]
            },
            ("elbv2", "describe-load-balancers"): {
                "LoadBalancers": [
                    {"LoadBalancerArn": "nlb-example", "State": {"Code": "active"}}
                ]
            },
            ("elbv2", "describe-listeners"): {
                "Listeners": [
                    {
                        "Port": 443,
                        "Protocol": "TLS",
                        "Certificates": [{"CertificateArn": CERTIFICATE}],
                    }
                ]
            },
            ("elbv2", "describe-target-groups"): {
                "TargetGroups": [{"TargetGroupArn": "target-group-example"}]
            },
            ("elbv2", "describe-target-health"): {
                "TargetHealthDescriptions": [
                    {"TargetHealth": {"State": "healthy"}} for _ in range(3)
                ]
            },
        }

    def dispatch(self, arguments: list[str], kwargs: dict[str, Any]) -> str:
        if arguments[0] == "aws":
            value = self.aws[arguments[1], arguments[2]]
            if isinstance(value, Exception):
                raise value
            if callable(value):
                value = value(arguments)
            return json.dumps(value)
        if arguments[0] == "openssl":
            return "fixture certificate valid"
        if arguments[0] == "python3" and arguments[1].endswith(
            (
                "verify-regional-alerting.py",
                "verify_control_plane_role_split.py",
                "verify_dataplane_executor.py",
            )
        ):
            return "fixture verifier result"
        if arguments[0] == "kubectl" and "exec" in arguments:
            if arguments[-1] == checks.EXECUTOR_READINESS:
                return "fixture ready"
            if "-c" in arguments:
                return json.dumps(self.tls)
        raise AssertionError(f"unconfigured fake read: {arguments[:4]}")

    def _ensure_contexts(self) -> None:
        self.context_checks += 1

    def _read_snapshot(self):
        return read_snapshot(self)

    def _prime_deployment_snapshot(self) -> None:
        self.prime_calls += 1


def install_api_transport(
    monkeypatch: pytest.MonkeyPatch, release: CheckedRelease
) -> list[Any]:
    calls = []

    def execute(_release: Any, **kwargs: Any) -> str:
        calls.append(copy.deepcopy(kwargs))
        return json.dumps(release.api)

    monkeypatch.setattr(checks, "exec_cpu_ingress", execute)
    return calls


def finding(report: dict[str, Any], name: str) -> dict[str, Any]:
    matches = [item for item in report["checks"] if item["name"] == name]
    assert len(matches) == 1
    return matches[0]

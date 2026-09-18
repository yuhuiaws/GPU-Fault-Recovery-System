"""Offline Kubernetes/AWS documents for the Aurora binding contract."""

from __future__ import annotations

import base64
import copy
import hashlib
import json
from typing import Any

import yaml

from gpu_fault.admin.bootstrap_common import ClusterIdentity
from gpu_fault.admin.bootstrap_services import pod_identity_trust
from gpu_fault_release.regional_release_aurora_refresh import render_aurora_refresh
from scripts.e2e.regional import aurora_binding as binding

REGION = "us-east-1"
NAMESPACE = "unit"
CLUSTER = "unit-aurora"
CPU_ARN = f"arn:aws:eks:{REGION}:111122223333:cluster/unit-cpu"
MASTER_ARN = f"arn:aws:secretsmanager:{REGION}:111122223333:secret:unit-master"
ROLE_ARN = "arn:aws:iam::111122223333:role/unit-refresh"
CRONJOB = "gpu-fault-aurora-credential-refresh"
SECRET = "gpu-fault-aurora"
IMAGE = "registry.invalid/cpu@sha256:" + "a" * 64
MODULE_DIGEST = "b" * 64
PASSWORD = "REPLACE_WITH_UNIT_PASSWORD"


def encoded(value: str) -> str:
    return base64.b64encode(value.encode()).decode()


def metadata(kind: str, name: str) -> dict[str, Any]:
    return {
        "kind": kind,
        "metadata": {
            "name": name,
            "namespace": "" if kind == "Namespace" else NAMESPACE,
            "uid": f"{kind}-{name}-uid",
            "generation": 1,
        },
    }


def cpu_container() -> dict[str, Any]:
    return {
        "name": "cpu",
        "image": IMAGE,
        "env": [
            {
                "name": "GPU_FAULT_STORE_URL",
                "valueFrom": {"secretKeyRef": {"name": SECRET, "key": "postgres-url"}},
            }
        ],
        "volumeMounts": [
            {
                "name": "aurora-credentials",
                "mountPath": "/etc/gpu-fault/aurora",
                "readOnly": True,
            }
        ],
    }


class FakeAurora:
    def __init__(self) -> None:
        self.calls: list[tuple[str, ...]] = []
        self.probes: list[dict[str, Any]] = []
        self.verified = True
        self.failure: Exception | None = None
        self.dsn = (
            f"postgresql://unit_admin:{PASSWORD}@writer.unit.invalid:5432/gpu_fault"
            "?sslmode=verify-full&sslrootcert=/etc/gpu-fault/rds/ca-bundle.pem"
        )
        self.cluster = {
            "DBClusterIdentifier": CLUSTER,
            "DBClusterArn": f"arn:aws:rds:{REGION}:111122223333:cluster:{CLUSTER}",
            "Engine": "aurora-postgresql",
            "Status": "available",
            "Port": 5432,
            "DbClusterResourceId": "cluster-unit-resource",
            "Endpoint": "writer.unit.invalid",
            "DatabaseName": "gpu_fault",
            "MasterUsername": "unit_admin",
            "MasterUserSecret": {"SecretArn": MASTER_ARN, "SecretStatus": "active"},
            "DBClusterMembers": [
                {"DBInstanceIdentifier": "unit-writer", "IsClusterWriter": True},
                {"DBInstanceIdentifier": "unit-reader", "IsClusterWriter": False},
            ],
        }
        self.eks = {
            "arn": CPU_ARN,
            "status": "ACTIVE",
            "createdAt": "unit-created-at",
            "endpoint": "https://cpu.unit.invalid",
            "identity": {"oidc": {"issuer": "https://oidc.unit.invalid/id/UNIT"}},
        }
        self.config = {
            "contexts": [{"context": {"cluster": CPU_ARN}}],
            "clusters": [
                {"name": CPU_ARN, "cluster": {"server": self.eks["endpoint"]}}
            ],
        }
        self.state = {
            "phase": "complete",
            "transaction_committed": True,
            "release_id": "unit-release",
            "runtime_image": IMAGE,
            "component_digests": {"control_plane": MODULE_DIGEST},
            "wheel_config_map": "unit-wheel",
        }
        self.documents: dict[tuple[str, str], dict[str, Any]] = {
            ("namespace", NAMESPACE): metadata("Namespace", NAMESPACE),
            ("configmap", "gpu-fault-regional-release-state"): {
                **metadata("ConfigMap", "gpu-fault-regional-release-state"),
                "data": {"state.json": json.dumps(self.state)},
            },
            ("secret", SECRET): {
                **metadata("Secret", SECRET),
                "data": {
                    "postgres-url": encoded(self.dsn),
                    "master-secret-arn": encoded(MASTER_ARN),
                },
            },
        }
        self.pods = {}
        for name in binding.CPU_RUNTIME_DEPLOYMENTS:
            replicas = int(name != "gpu-fault-telemetry-spool-worker")
            spec = {
                "containers": [cpu_container()],
                "volumes": [
                    {"name": "aurora-credentials", "secret": {"secretName": SECRET}}
                ],
            }
            self.documents["deployment", name] = {
                **metadata("Deployment", name),
                "spec": {"replicas": replicas, "template": {"spec": spec}},
                "status": {
                    "observedGeneration": 1,
                    "updatedReplicas": replicas,
                    "readyReplicas": replicas,
                    "availableReplicas": replicas,
                },
            }
            pod = {
                **metadata("Pod", f"{name}-pod"),
                "spec": copy.deepcopy(spec),
                "status": {
                    "phase": "Running",
                    "conditions": [{"type": "Ready", "status": "True"}],
                },
            }
            self.pods[name] = [pod] if replicas else []
            if replicas:
                self.documents["pod", pod["metadata"]["name"]] = pod
        rendered = render_aurora_refresh(
            namespace=NAMESPACE,
            runtime_image=IMAGE,
            wheel_config_map="unit-wheel",
            master_secret_arn=MASTER_ARN,
        )
        for document in yaml.safe_load_all(rendered):
            document["metadata"].update(uid=f"{document['kind']}-uid", generation=1)
            self.documents[document["kind"].lower(), document["metadata"]["name"]] = (
                document
            )
        self.associations = [{"associationId": "unit-association"}]
        self.association = {
            "associationId": "unit-association",
            "clusterName": "unit-cpu",
            "namespace": NAMESPACE,
            "serviceAccount": CRONJOB,
            "roleArn": ROLE_ARN,
        }
        self.iam_role = {
            "Arn": ROLE_ARN,
            "RoleId": "unit-role-id",
            "AssumeRolePolicyDocument": pod_identity_trust(
                ClusterIdentity(
                    input_arn=CPU_ARN,
                    role="cpu",
                    region=REGION,
                    account_id="111122223333",
                    hyperpod_arn="",
                    hyperpod_name="",
                    eks_arn=CPU_ARN,
                    eks_name="unit-cpu",
                    vpc_id="",
                    subnet_ids=(),
                    node_recovery="None",
                    context="",
                )
            ),
        }
        self.policy = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Action": ["secretsmanager:GetSecretValue"],
                    "Resource": MASTER_ARN,
                }
            ],
        }
        self.guard = binding.AuroraBinding(
            self.control, self.aws, REGION, CLUSTER, NAMESPACE, cronjob_name=CRONJOB
        )

    def control(self, *args: str, **kwargs: Any) -> str:
        self.calls.append(("kubectl", *args[:3]))
        if self.failure is not None:
            raise self.failure
        if args[0] == "config":
            return json.dumps(self.config)
        if args[0] == "exec":
            expected = json.loads(args[-1])
            self.probes.append(expected)
            return json.dumps(
                {
                    "verified": self.verified,
                    "credential_sha256": hashlib.sha256(self.dsn.encode()).hexdigest(),
                }
            )
        if args[:2] == ("get", "pods"):
            return json.dumps({"items": self.pods[args[3].removeprefix("app=")]})
        if args[0] == "get":
            return json.dumps(self.documents[args[1], args[2]])
        raise AssertionError(f"unexpected or mutating Kubernetes command: {args[0]}")

    def aws(self, service: str, command: str, *args: str) -> dict[str, Any]:
        self.calls.append(("aws", service, command))
        responses = {
            ("eks", "describe-cluster"): {"cluster": self.eks},
            ("rds", "describe-db-clusters"): {"DBClusters": [self.cluster]},
            ("secretsmanager", "describe-secret"): {
                "ARN": MASTER_ARN,
                "OwningService": "rds",
            },
            ("eks", "list-pod-identity-associations"): {
                "associations": self.associations
            },
            ("eks", "describe-pod-identity-association"): {
                "association": self.association
            },
            ("iam", "get-role"): {"Role": self.iam_role},
            ("iam", "get-role-policy"): {"PolicyDocument": self.policy},
        }
        assert (service, command) in responses, (
            "binding must use read-only metadata APIs"
        )
        return copy.deepcopy(responses[service, command])

    def refresh_container(self) -> dict[str, Any]:
        return self.documents["cronjob", CRONJOB]["spec"]["jobTemplate"]["spec"][
            "template"
        ]["spec"]["containers"][0]

    def change_refresh_env(self, name: str, value: str) -> None:
        for item in self.refresh_container()["env"]:
            if item["name"] == name:
                item["value"] = value
                return
        self.refresh_container()["env"].append({"name": name, "value": value})

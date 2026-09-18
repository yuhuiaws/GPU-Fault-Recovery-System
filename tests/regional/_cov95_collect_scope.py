"""Synthetic read-only AWS and Kubernetes answers for the COLLECT-004 scope preflight.

``capture_reboot_scope`` binds the target Node to exactly one HyperPod instance
and to the deployed Executor identity through ``regional.run(["aws", ...])`` and
``regional.kubectl("gpu", ...)``. This responder answers those reads for one
fixture node with placeholder identifiers only; any other command is a harness
gap and fails loudly instead of being silently tolerated.
"""

from __future__ import annotations

import base64
import json
import subprocess
from copy import deepcopy
from typing import Any

REGION = "us-west-2"
INSTANCE = "i-00000000000000001"
SPARE_INSTANCE = "i-00000000000000002"
LOGICAL = "logical-node-001"
SPARE_LOGICAL = "logical-node-002"
CA = base64.b64encode(b"public-test-ca").decode()
IMAGE = "registry.example/executor@sha256:" + "a" * 64
EXECUTOR = "gpu-fault-cluster-executor"
_AWS_READS = {
    ("sagemaker", "describe-cluster"): "cluster",
    ("sagemaker", "list-cluster-nodes"): "inventory",
    ("sagemaker", "describe-cluster-node"): "node_detail",
    ("eks", "describe-cluster"): "eks",
}
_KUBE_READS = {
    "node": "node",
    "namespace": "anchor",
    "deployment": "deployment",
    "pod": "pods",
    "serviceaccount": "serviceaccount",
    "replicaset": "replicaset",
}


class RebootScopeReads:
    """Consistent provider, orchestrator and Executor facts for one target node."""

    def __init__(
        self,
        *,
        node: str,
        hyperpod_cluster: str,
        executor_role_arn: str,
        cluster_id: str,
        namespace: str,
        region: str = REGION,
    ) -> None:
        self.calls: list[tuple[str, ...]] = []
        account = executor_role_arn.split(":")[4]
        role_name = executor_role_arn.rsplit("/", 1)[-1]
        hp_arn = f"arn:aws:sagemaker:{region}:{account}:cluster/{hyperpod_cluster}"
        eks_arn = f"arn:aws:eks:{region}:{account}:cluster/eks-fixture"
        endpoint = "https://eks-fixture.example"
        pod = {
            "metadata": {
                "name": "executor-a",
                "namespace": namespace,
                "uid": "executor-a-uid",
                "ownerReferences": [
                    {
                        "apiVersion": "apps/v1",
                        "kind": "ReplicaSet",
                        "controller": True,
                        "name": "executor-rs",
                        "uid": "executor-rs-uid",
                    }
                ],
            },
            "spec": {
                "serviceAccountName": "executor-sa",
                "nodeName": "cpu-node-a",
                "containers": [{"name": "executor", "image": IMAGE}],
            },
            "status": {
                "phase": "Running",
                "conditions": [{"type": "Ready", "status": "True"}],
                "containerStatuses": [
                    {
                        "name": "executor",
                        "ready": True,
                        "containerID": "containerd://executor-a",
                        "restartCount": 0,
                        "state": {
                            "running": {"startedAt": "2026-09-17T00:00:00+00:00"}
                        },
                    }
                ],
            },
        }
        self.data: dict[str, Any] = {
            "cluster": {
                "ClusterName": hyperpod_cluster,
                "ClusterArn": hp_arn,
                "ClusterStatus": "InService",
                "NodeRecovery": "None",
                "Orchestrator": {"Eks": {"ClusterArn": eks_arn}},
            },
            "eks": {
                "cluster": {
                    "name": "eks-fixture",
                    "arn": eks_arn,
                    "status": "ACTIVE",
                    "endpoint": endpoint,
                    "certificateAuthority": {"data": CA},
                }
            },
            "kubeconfig_clusters": [
                {
                    "name": "fixture-alias",
                    "cluster": {"server": endpoint, "certificate-authority-data": CA},
                }
            ],
            "anchor": {"metadata": {"name": "kube-system", "uid": "kube-system-uid"}},
            "node": {
                "metadata": {"name": node, "uid": f"{node}-uid"},
                "spec": {"providerID": f"aws:///{region}a/{INSTANCE}"},
                "status": {
                    "conditions": [{"type": "Ready", "status": "True"}],
                    "nodeInfo": {"bootID": "boot-a"},
                },
            },
            "inventory": {
                "ClusterNodeSummaries": [
                    {
                        "NodeLogicalId": LOGICAL,
                        "InstanceId": INSTANCE,
                        "InstanceStatus": {"Status": "Running"},
                    },
                    {
                        "NodeLogicalId": SPARE_LOGICAL,
                        "InstanceId": SPARE_INSTANCE,
                        "InstanceStatus": {"Status": "Running"},
                    },
                ]
            },
            "node_detail": {
                "NodeDetails": {"NodeLogicalId": LOGICAL, "InstanceId": INSTANCE}
            },
            "deployment": {
                "metadata": {
                    "name": EXECUTOR,
                    "uid": "executor-deployment-uid",
                    "namespace": namespace,
                    "generation": 1,
                },
                "spec": {
                    "replicas": 1,
                    "template": {
                        "spec": {
                            "serviceAccountName": "executor-sa",
                            "containers": [{"name": "executor", "image": IMAGE}],
                        }
                    },
                },
                "status": {
                    "observedGeneration": 1,
                    "replicas": 1,
                    "readyReplicas": 1,
                    "updatedReplicas": 1,
                    "availableReplicas": 1,
                },
            },
            "pods": {"items": [pod]},
            "replicaset": {
                "metadata": {
                    "name": "executor-rs",
                    "uid": "executor-rs-uid",
                    "namespace": namespace,
                    "ownerReferences": [
                        {
                            "apiVersion": "apps/v1",
                            "kind": "Deployment",
                            "controller": True,
                            "name": EXECUTOR,
                            "uid": "executor-deployment-uid",
                        }
                    ],
                }
            },
            "serviceaccount": {
                "metadata": {
                    "name": "executor-sa",
                    "namespace": namespace,
                    "uid": "executor-sa-uid",
                    "annotations": {"eks.amazonaws.com/role-arn": executor_role_arn},
                }
            },
            "environment": {
                "cluster_id": cluster_id,
                "hyperpod_cluster": hyperpod_cluster,
                "region": region,
                "role_arn": executor_role_arn,
                "allow_reboot": True,
                "allow_replace": False,
                "allow_automatic": False,
                "credential_method": "assume-role-with-web-identity",
                "caller_arn": (
                    f"arn:aws:sts::{account}:assumed-role/{role_name}/pod-session"
                ),
                "caller_account": account,
            },
        }

    def run(
        self, command: list[str], **kwargs: Any
    ) -> subprocess.CompletedProcess[str]:
        self.calls.append(tuple(command))
        if command[0] != "aws" or (command[1], command[2]) not in _AWS_READS:
            raise AssertionError(f"unmocked fixture command {command[:3]}")
        return subprocess.CompletedProcess(
            command,
            0,
            json.dumps(deepcopy(self.data[_AWS_READS[(command[1], command[2])]])),
            "",
        )

    def kubectl(self, plane: str, *arguments: str, **kwargs: Any) -> str:
        self.calls.append(("kubectl", plane, *arguments))
        if plane != "gpu":
            raise AssertionError(f"scope capture must read the GPU plane, not {plane}")
        if arguments[0] == "config":
            return json.dumps(deepcopy(self.data["kubeconfig_clusters"]))
        if arguments[0] == "exec":
            return json.dumps(
                {**deepcopy(self.data["environment"]), "pod": arguments[2]}
            )
        if arguments[0] == "get" and arguments[1] in _KUBE_READS:
            return json.dumps(deepcopy(self.data[_KUBE_READS[arguments[1]]]))
        raise AssertionError(f"unmocked fixture kubectl read {arguments[:2]}")

"""DESTR013 caller wiring with all external commands replaced at public transport."""

from __future__ import annotations

import base64
import copy
import json
import signal
import subprocess
import sys
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from gpu_fault.admin.process_supervisor import ProcessSupervisionLost
from scripts.e2e.regional import audit_destr013_replacement_invariant as audit
from scripts.e2e.regional import destr013_audit_evidence as binding
from scripts.e2e.regional import regional_commands
from scripts.e2e.regional.regional_live_fixture import RegionalFixtureAbort
from tests.regional._cov95_focused_mock_receipts import write_focused_receipt

NOW = datetime(2026, 9, 7, 12, tzinfo=timezone.utc)
REGION = "us-west-2"
ACCOUNT = "123456789012"
EKS = f"arn:aws:eks:{REGION}:{ACCOUNT}:cluster/gpu"
HYPERPOD = f"arn:aws:sagemaker:{REGION}:{ACCOUNT}:cluster/physical-id"
ROLE = f"arn:aws:iam::{ACCOUNT}:role/regional/executor"
SENSITIVE = "private-diagnostic-sentinel"
# The live mixed site's second registration: a plain EKS GPU cluster.
EKS_CLUSTER_ID = "gpu-eks-cluster"
EKS_GPU_NAME = "gpu-eks"
EKS_GPU = f"arn:aws:eks:{REGION}:{ACCOUNT}:cluster/{EKS_GPU_NAME}"
INSTANCES = ("i-00000000000000001", "i-00000000000000002")
NODEGROUP = "gpu-p5en-group"
KARPENTER_ROLE = f"arn:aws:iam::{ACCOUNT}:role/karpenter-controller"
ASG_ROLE = f"arn:aws:iam::{ACCOUNT}:role/aws-service-role/AWSServiceRoleForAutoScaling"


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def population(app: str, container: str) -> tuple[dict[str, Any], dict[str, Any]]:
    image = f"example.invalid/{container}@sha256:" + "e" * 64
    containers = [
        {"name": "sidecar", "image": "example.invalid/sidecar"},
        {"name": container, "image": image},
    ]
    spec = {"containers": containers, "serviceAccountName": app}
    deployment = {
        "metadata": {"uid": f"{app}-uid", "generation": 2},
        "spec": {"replicas": 3, "template": {"spec": copy.deepcopy(spec)}},
        "status": {
            "observedGeneration": 2,
            "replicas": 3,
            "readyReplicas": 3,
            "updatedReplicas": 3,
            "availableReplicas": 3,
        },
    }
    pods = {
        "items": [
            {
                "metadata": {
                    "name": f"{container}-{index}",
                    "uid": f"{container}-uid-{index}",
                },
                "spec": copy.deepcopy(spec),
                "status": {
                    "phase": "Running",
                    "conditions": [{"type": "Ready", "status": "True"}],
                    "containerStatuses": [
                        {
                            "name": row["name"],
                            "ready": True,
                            "containerID": f"container-{row['name']}-{index}",
                            "restartCount": 0,
                        }
                        for row in containers
                    ],
                },
            }
            for index in range(3)
        ]
    }
    return deployment, pods


class LocalAudit:
    def __init__(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self.run_dir = tmp_path / "run"
        self.output = self.run_dir / "cases" / audit.CASE_ID / f"{audit.CASE_ID}.json"
        self.predecessor = (
            self.run_dir
            / "cases"
            / audit.PREDECESSOR_CASE_ID
            / f"{audit.PREDECESSOR_CASE_ID}.json"
        )
        write_json(
            self.predecessor,
            {
                "case_id": audit.PREDECESSOR_CASE_ID,
                "verdict": "PASS",
                "release_id": "release-a",
                "cluster_id": "cluster-a",
            },
        )
        write_json(
            self.run_dir / "cases" / "GF-REGIONAL-DESTR-002" / "result.json",
            {
                "started_at": (NOW - timedelta(hours=2)).isoformat(),
                "completed_at": (NOW - timedelta(minutes=40)).isoformat(),
            },
        )
        self.ca = base64.b64encode(b"public-test-certificate").decode()
        self.cpu_config = tmp_path / "cpu"
        self.gpu_config = tmp_path / "gpu"
        self.kubeconfig = {
            "current-context": "selected",
            "contexts": [{"name": "selected", "context": {"cluster": "selected"}}],
            "clusters": [
                {
                    "name": "selected",
                    "cluster": {
                        "server": "https://gpu.example.invalid",
                        "certificate-authority-data": self.ca,
                    },
                }
            ],
        }
        write_json(self.cpu_config, self.kubeconfig)
        write_json(self.gpu_config, self.kubeconfig)
        # The live two-cluster shape: the HyperPod registration and the plain
        # EKS one share the durable registry; the CLI's cluster names which.
        self.registry = {
            "generation": 3,
            "registrations": [
                {
                    "cluster_id": "cluster-a",
                    "region": REGION,
                    "hyperpod_cluster_name": "managed",
                    "eks_cluster_arn": EKS,
                    "provider": "hyperpod",
                },
                {
                    "cluster_id": EKS_CLUSTER_ID,
                    "region": REGION,
                    "hyperpod_cluster_name": "",
                    "eks_cluster_arn": EKS_GPU,
                    "provider": "eks-managed",
                },
            ],
        }
        self.provider = "hyperpod"
        self.failing: dict[tuple[str, str], tuple[int, str]] = {}
        self.recovery = {
            "ClusterName": "managed",
            "ClusterArn": HYPERPOD,
            "ClusterStatus": "InService",
            "NodeRecovery": "None",
            "Orchestrator": {"Eks": {"ClusterArn": EKS}},
        }
        self.eks = {
            "cluster": {
                "arn": EKS,
                "status": "ACTIVE",
                "endpoint": "https://gpu.example.invalid",
                "certificateAuthority": {"data": self.ca},
            }
        }
        self.deployments: dict[str, dict[str, Any]] = {}
        self.pods: dict[str, dict[str, Any]] = {}
        for app, container in (
            (binding.API_APP, "api"),
            (binding.EXECUTOR_APP, "executor"),
        ):
            self.deployments[app], self.pods[app] = population(app, container)
        self.state = {
            "release_id": "release-a",
            "phase": "complete",
            "transaction_committed": True,
            "cluster_ids": ["cluster-a"],
            "wheel_sha256": "c" * 64,
            "executor_wheel_sha256": "d" * 64,
            "component_digests": {"control_plane": "a" * 64, "executor": "b" * 64},
            "runtime_image": self.deployments[binding.API_APP]["spec"]["template"][
                "spec"
            ]["containers"][1]["image"],
            "executor_image": self.deployments[binding.EXECUTOR_APP]["spec"][
                "template"
            ]["spec"]["containers"][1]["image"],
        }
        self.service_account = {
            "metadata": {
                "uid": "executor-sa-uid",
                "annotations": {"eks.amazonaws.com/role-arn": ROLE},
            }
        }
        # The fleet pins as the product serves them: ``gpu-fault-release-metadata``
        # in the ``kubectl get configmap -o json`` shape. The api-ha Pods only
        # hold the kubelet's start-time snapshot of these, so the fixture never
        # puts the REQUIRED_REGIONAL_EXECUTOR_* values on a cpu Pod.
        self.pins = {
            "required-agent-artifact-sha256": "1" * 64,
            "required-agent-compatibility-digest": "2" * 64,
            "required-agent-config-digest": "3" * 64,
            "required-agent-protocol-version": "4",
            "required-node-action-key-version": "2",
            "required-regional-executor-protocol-version": "4",
            "required-regional-executor-artifact-sha256": "d" * 64,
            "required-regional-executor-compatibility-digest": "b" * 64,
            "compatible-regional-executor-artifact-sha256s": "",
            "compatible-regional-executor-compatibility-digests": "",
            "compatible-regional-executor-protocol-versions": "",
        }
        self.state["executor_protocol_version"] = 4
        self.environments = {}
        for plane, container in (("cpu", "api"), ("gpu", "executor")):
            for index in range(3):
                pod = f"{container}-{index}"
                self.environments[pod] = {
                    "pod": pod,
                    "module_digest": "a" * 64 if plane == "cpu" else "b" * 64,
                    **(
                        {"synthetic_route": None}
                        if plane == "cpu"
                        else {
                            "executor_artifact": "d" * 64,
                            "executor_compatibility": "b" * 64,
                            "allow_replace": "false",
                            "allow_reboot": "true",
                            "allow_automatic": None,
                            "legacy_mutation": None,
                            "cluster_id": "cluster-a",
                            "role_arn": ROLE,
                            "region": REGION,
                            "default_region": REGION,
                            "credential_method": "assume-role-with-web-identity",
                            "caller_account": ACCOUNT,
                            "caller_arn": f"arn:aws:sts::{ACCOUNT}:assumed-role/executor/session",
                        }
                    ),
                }
        self.events = {name: [] for name in audit.EVENT_NAMES}
        self.events[audit.POSITIVE_CONTROL_EVENT] = [
            self.event(audit.POSITIVE_CONTROL_EVENT)
        ]
        self.calls: list[tuple[list[str], dict[str, Any]]] = []
        self.ec2_events: dict[str, list[dict[str, Any]]] = {}
        self.ec2_iam: dict[str, Any] = {}
        self.instance_tags: dict[str, dict[str, str]] = {}
        self.before_command: Callable[[list[str]], None] | None = None
        self.pages: dict[str, dict[str, Any]] | None = None
        self.pytest_code = 0
        self.pytest_receipt = "complete"
        self.pytest_receipts: list[dict[str, Any] | None] = []
        self.argv = [
            "destr013",
            "--run-dir",
            str(self.run_dir),
            "--cpu-kubeconfig",
            str(self.cpu_config),
            "--gpu-kubeconfig",
            str(self.gpu_config),
            "--gpu-context",
            "selected",
            "--region",
            REGION,
            "--hyperpod-cluster",
            "managed",
            "--executor-role-arn",
            ROLE,
            "--window-start",
            (NOW - timedelta(hours=3)).isoformat(),
            "--window-end",
            (NOW - timedelta(minutes=20)).isoformat(),
        ]
        monkeypatch.setattr(sys, "argv", self.argv)
        monkeypatch.setattr(audit, "install_abort_signals", lambda: None)
        monkeypatch.setattr(regional_commands, "run_command", self.command)

        class Clock(datetime):
            @classmethod
            def now(cls, tz: Any = None) -> datetime:
                return NOW

        monkeypatch.setattr(audit, "datetime", Clock)

    def use_eks(self) -> None:
        """Target the registry's eks-managed member instead of the HyperPod one."""

        self.provider = "eks-managed"
        self.argv[self.argv.index("--hyperpod-cluster") + 1] = EKS_CLUSTER_ID
        write_json(
            self.predecessor,
            {
                "case_id": audit.PREDECESSOR_CASE_ID,
                "verdict": "PASS",
                "release_id": "release-a",
                "cluster_id": EKS_CLUSTER_ID,
            },
        )
        self.state["cluster_ids"] = ["cluster-a", EKS_CLUSTER_ID]
        for pod, environment in self.environments.items():
            if pod.startswith("executor"):
                environment["cluster_id"] = EKS_CLUSTER_ID
        self.eks_gpu = {
            "cluster": {
                "name": EKS_GPU_NAME,
                "arn": EKS_GPU,
                "status": "ACTIVE",
                "endpoint": "https://gpu.example.invalid",
                "certificateAuthority": {"data": self.ca},
            }
        }
        self.nodes = {
            "items": [
                {
                    "metadata": {
                        "name": f"ip-10-0-0-{index}.{REGION}.compute.internal",
                        "labels": {"eks.amazonaws.com/nodegroup": NODEGROUP},
                    },
                    "spec": {"providerID": f"aws:///{REGION}c/{instance}"},
                    "status": {"allocatable": {"nvidia.com/gpu": "8"}},
                }
                for index, instance in enumerate(INSTANCES, start=1)
            ]
        }
        self.instance_tags = {
            instance: {
                "eks:nodegroup-name": NODEGROUP,
                "eks:cluster-name": EKS_GPU_NAME,
            }
            for instance in INSTANCES
        }
        self.nodegroup = {
            "nodegroup": {
                "nodegroupName": NODEGROUP,
                "clusterName": EKS_GPU_NAME,
                "status": "ACTIVE",
                "health": {"issues": []},
            }
        }
        self.ec2_iam = {
            "EvaluationResults": [
                {
                    "EvalActionName": action,
                    "EvalDecision": (
                        "allowed" if action == "ec2:RebootInstances" else "implicitDeny"
                    ),
                    "ResourceSpecificResults": [
                        {
                            "EvalResourceName": f"arn:aws:ec2:{REGION}:{ACCOUNT}:instance/{i}",
                            "EvalResourceDecision": (
                                "allowed"
                                if action == "ec2:RebootInstances"
                                else "implicitDeny"
                            ),
                        }
                        for i in INSTANCES
                    ],
                }
                for action in audit.EKS_IAM_ACTIONS
            ]
        }
        self.ec2_events = {name: [] for name in audit.EKS_EVENT_NAMES}
        # Live shape: two reboots by the executor role, Karpenter's refused
        # RunInstances dry-runs about once a minute, one Auto Scaling
        # termination by the service-linked role.
        self.ec2_events["RebootInstances"] = [
            self.ec2_event("RebootInstances", [instance]) for instance in INSTANCES
        ]
        self.ec2_events["RunInstances"] = [
            self.ec2_event(
                "RunInstances",
                None,
                issuer=KARPENTER_ROLE,
                errorCode="Client.DryRunOperation",
            )
            for _ in range(3)
        ]
        self.ec2_events["TerminateInstances"] = [
            self.ec2_event(
                "TerminateInstances", ["i-0000000000000000"], issuer=ASG_ROLE
            )
        ]

    def ec2_event(
        self,
        name: str,
        instance_ids: list[str] | None,
        *,
        issuer: str = ROLE,
        **overrides: Any,
    ) -> dict[str, Any]:
        at = (NOW - timedelta(hours=1)).isoformat()
        event_id = f"{name}-{len(self.calls)}-{abs(hash((name, tuple(instance_ids or []), issuer)))}"
        detail: dict[str, Any] = {
            "eventID": event_id,
            "requestID": "request-" + event_id,
            "eventSource": "ec2.amazonaws.com",
            "eventName": name,
            "eventTime": at,
            "awsRegion": REGION,
            "recipientAccountId": ACCOUNT,
            "requestParameters": (
                {"instancesSet": {"items": [{"instanceId": i} for i in instance_ids]}}
                if instance_ids is not None
                else {"instancesSet": {"items": [{"minCount": 1, "maxCount": 1}]}}
            ),
            "responseElements": {"_return": True} if name == "RebootInstances" else {},
            "userIdentity": {
                "type": "AssumedRole",
                "arn": f"arn:aws:sts::{ACCOUNT}:assumed-role/{issuer.rsplit('/', 1)[-1]}/s",
                "accountId": ACCOUNT,
                "sessionContext": {
                    "sessionIssuer": {
                        "type": "Role",
                        "arn": issuer,
                        "accountId": ACCOUNT,
                    }
                },
            },
            **overrides,
        }
        if "errorCode" in detail:
            detail["responseElements"] = None
        return {
            "EventId": event_id,
            "EventName": name,
            "EventTime": at,
            "CloudTrailEvent": json.dumps(detail),
        }

    def event(self, name: str, **overrides: Any) -> dict[str, Any]:
        at = (NOW - timedelta(hours=1)).isoformat()
        detail = {
            "eventSource": "sagemaker.amazonaws.com",
            "eventName": name,
            "eventTime": at,
            "awsRegion": REGION,
            "recipientAccountId": ACCOUNT,
            "requestParameters": {"clusterName": "managed"},
            "userIdentity": {"sessionContext": {"sessionIssuer": {"arn": ROLE}}},
            **overrides,
        }
        return {
            "EventName": name,
            "EventTime": at,
            "CloudTrailEvent": json.dumps(detail),
        }

    def command(
        self, arguments: list[str], **kwargs: Any
    ) -> subprocess.CompletedProcess[str]:
        command = list(arguments)
        self.calls.append((command, kwargs))
        current = json.loads(self.output.read_text(encoding="utf-8"))
        assert current["verdict"] == "FAIL", (
            "initial FAIL must exist before every external read"
        )
        if self.before_command is not None:
            self.before_command(command)
        if command[1:3] == ["-m", "pytest"]:
            self.pytest_receipts.append(
                write_focused_receipt(
                    command,
                    environment=kwargs["environment"],
                    cwd=kwargs.get("cwd"),
                    returncode=self.pytest_code,
                    defect=self.pytest_receipt,
                )
            )
            return subprocess.CompletedProcess(
                command, self.pytest_code, stdout=SENSITIVE, stderr=""
            )
        if command[0] == "aws":
            if tuple(command[1:3]) in self.failing:
                returncode, stderr = self.failing[tuple(command[1:3])]
                return subprocess.CompletedProcess(
                    command, returncode, stdout="", stderr=stderr
                )
            if self.provider == "eks-managed" and command[1] == "sagemaker":
                pytest.fail(
                    "an eks-managed registration must never be asked the SageMaker API"
                )
            if command[1:3] == ["sagemaker", "describe-cluster"]:
                value = self.recovery
            elif command[1:3] == ["sagemaker", "list-cluster-nodes"]:
                value = {
                    "ClusterNodeSummaries": [
                        {
                            "NodeLogicalId": "logical-a",
                            "InstanceId": "instance-a",
                            "InstanceGroupName": "gpu",
                            "InstanceType": "gpu-type",
                        }
                    ]
                }
            elif command[1:3] == ["eks", "describe-cluster"]:
                name = command[command.index("--name") + 1]
                value = self.eks_gpu if name == EKS_GPU_NAME else self.eks
            elif command[1:3] == ["eks", "describe-nodegroup"]:
                assert command[command.index("--cluster-name") + 1] == EKS_GPU_NAME
                assert command[command.index("--nodegroup-name") + 1] == NODEGROUP
                value = self.nodegroup
            elif command[1:3] == ["ec2", "describe-instances"]:
                wanted = (
                    command[command.index("--filters") + 1]
                    .split("Values=")[1]
                    .split(",")
                )
                value = {
                    "Reservations": [
                        {
                            "Instances": [
                                {
                                    "InstanceId": instance,
                                    "InstanceType": "p5en.48xlarge",
                                    "Placement": {"AvailabilityZone": f"{REGION}c"},
                                    "State": {"Name": "running"},
                                    "Tags": [
                                        {"Key": key, "Value": tag}
                                        for key, tag in self.instance_tags[
                                            instance
                                        ].items()
                                    ],
                                }
                                for instance in wanted
                                if instance in self.instance_tags
                            ]
                        }
                    ]
                }
            elif command[1:3] == ["iam", "simulate-principal-policy"] and command[
                command.index("--action-names") + 1
            ].startswith("ec2:"):
                value = self.ec2_iam
            elif command[1:3] == ["iam", "simulate-principal-policy"]:
                value = {
                    "EvaluationResults": [
                        {
                            "EvalActionName": "sagemaker:BatchReplaceClusterNodes",
                            "EvalDecision": "implicitDeny",
                        },
                        {
                            "EvalActionName": "sagemaker:BatchRebootClusterNodes",
                            "EvalDecision": "allowed",
                        },
                    ]
                }
            elif command[1:3] == ["cloudtrail", "lookup-events"]:
                name = command[command.index("--lookup-attributes") + 1].split(
                    "AttributeValue="
                )[1]
                token = (
                    command[command.index("--next-token") + 1]
                    if "--next-token" in command
                    else ""
                )
                if name in self.ec2_events:
                    value = {"Events": self.ec2_events[name]}
                else:
                    value = (
                        self.pages[token]
                        if self.pages is not None
                        and name == audit.POSITIVE_CONTROL_EVENT
                        else {"Events": self.events[name]}
                    )
            else:
                pytest.fail(f"unexpected AWS command verb: {command[1:3]}")
        elif command[0] == "kubectl":
            operation = command[command.index("-n") + 2 :]
            if operation[:2] == ["get", "deployment"]:
                value = self.deployments[operation[2]]
            elif operation[:2] == ["get", "pod"]:
                app = operation[operation.index("-l") + 1].removeprefix("app=")
                value = self.pods[app]
            elif operation[:2] == ["get", "namespace"]:
                value = {"metadata": {"uid": "gpu-kube-system-uid"}}
            elif operation[:2] == ["get", "nodes"]:
                value = self.nodes
            elif operation[:2] == ["get", "serviceaccount"]:
                value = self.service_account
            elif operation[:3] == ["get", "configmap", "gpu-fault-release-metadata"]:
                assert operation[3:] == ["-o", "json"], operation
                value = {
                    "apiVersion": "v1",
                    "kind": "ConfigMap",
                    "metadata": {
                        "name": "gpu-fault-release-metadata",
                        "namespace": "gpu-fault-system",
                        "resourceVersion": "48517395",
                    },
                    "data": self.pins,
                }
            elif operation[:2] == ["get", "configmap"]:
                value = {"data": {"state.json": json.dumps(self.state)}}
            elif operation[:2] == ["exec", "-i"]:
                pod = operation[2]
                value = (
                    self.registry
                    if kwargs["input_text"] == binding.REGISTRY_PROBE
                    else self.environments[pod]
                )
            else:
                pytest.fail(f"unexpected Kubernetes operation: {operation[:2]}")
        else:
            pytest.fail("unexpected command executable")
        return subprocess.CompletedProcess(
            command, 0, stdout=json.dumps(value), stderr=""
        )

    def run(self) -> tuple[int, dict[str, Any]]:
        code = audit.main()
        return code, json.loads(self.output.read_text(encoding="utf-8"))

    def iam_calls(self) -> list[list[str]]:
        return [
            command
            for command, _ in self.calls
            if command[1:3] == ["iam", "simulate-principal-policy"]
        ]


@pytest.fixture
def local_audit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> LocalAudit:
    return LocalAudit(tmp_path, monkeypatch)


def test_main_binds_real_targets_and_all_ready_replicas(
    local_audit: LocalAudit,
) -> None:
    code, result = local_audit.run()
    assert code == 0 and result["verdict"] == "PASS", result
    assert result["formal_sequence_satisfied"] is True, (
        "settled, bound evidence may satisfy the formal sequence"
    )
    assert (
        result["release_id"] == "release-a" and result["cluster_id"] == "cluster-a"
    ), "persist the deployed predecessor identity"
    assert result["executor_replica_count"] == result["api_replica_count"] == 3, (
        "inspect the actual complete population, not one or two replicas"
    )
    assert result["predecessor"]["expected_release_id"] == "release-a", (
        "the actual predecessor call must receive the release"
    )
    assert result["predecessor"]["expected_cluster_id"] == "cluster-a", (
        "the actual predecessor call must receive the cluster"
    )
    [simulation] = local_audit.iam_calls()
    assert simulation[simulation.index("--policy-source-arn") + 1] == ROLE, (
        "simulate the verified deployed IRSA role"
    )
    assert simulation[simulation.index("--resource-arns") + 1] == HYPERPOD, (
        "simulate the verified physical HyperPod ARN"
    )
    execs = [
        command
        for command, _ in local_audit.calls
        if command[0] == "kubectl" and "exec" in command
    ]
    assert len(execs) == 14, (
        "both snapshots probe all six replicas and the durable registry"
    )
    for command in execs:
        container = command[command.index("-c") + 1]
        plane = "cpu" if container == "api" else "gpu"
        assert command[command.index("--") + 1 :] == [
            binding.component_python(plane),
            "-B",
            "-",
        ], "use the named component container/interpreter even with a sidecar first"
    assert not any(
        "field-selector" in item for command, _ in local_audit.calls for item in command
    ), "do not hide unready replicas from the population check"
    [receipt] = local_audit.pytest_receipts
    assert receipt is not None, "the mocked child must supply a structured receipt"
    assert set(receipt["records"]) == set(receipt["session"]["discovered_nodeids"]), (
        "a successful audit must consume the complete mocked pytest discovery"
    )


@pytest.mark.parametrize(
    "defect",
    ["missing", "failed", "missing-phase", "unexecuted-discovery", "foreign-source"],
)
def test_main_refuses_zero_exit_without_complete_passing_focused_receipt(
    local_audit: LocalAudit, defect: str
) -> None:
    local_audit.pytest_receipt = defect
    code, result = local_audit.run()
    assert local_audit.pytest_code == 0, "the fake child must still exit successfully"
    assert code == 1 and result["verdict"] == "FAIL", result
    assert result["focused_tests"]["passed"] is False, result
    assert "focused regression tests failed" in result["errors"], result
    assert result["formal_sequence_satisfied"] is False, (
        "unproven local regression evidence must not satisfy the formal audit chain"
    )


@pytest.mark.parametrize(
    ("defect", "diagnostic"),
    [
        ("hyperpod", "HyperPod/EKS/IRSA"),
        ("orchestrator", "HyperPod/EKS/IRSA"),
        ("eks", "registered EKS"),
        ("endpoint", "registered EKS"),
        ("ca", "registered EKS"),
        ("registry", "durable registration"),
        ("commit", "committed"),
        ("release", "release pins"),
        ("cluster", "cluster/Region/IRSA"),
        ("region", "cluster/Region/IRSA"),
        ("role", "cluster/Region/IRSA"),
        ("serviceaccount", "cluster/Region/IRSA"),
        ("credentials", "cluster/Region/IRSA"),
        ("caller", "cluster/Region/IRSA"),
        ("cpu-module", "release pins"),
    ],
)
def test_main_rejects_valid_iam_answers_for_an_unbound_target(
    local_audit: LocalAudit, defect: str, diagnostic: str
) -> None:
    if defect == "hyperpod":
        local_audit.recovery["ClusterName"] = "other"
    elif defect == "orchestrator":
        local_audit.recovery["Orchestrator"] = {"Slurm": {}}
    elif defect == "eks":
        local_audit.eks["cluster"]["arn"] = EKS + "-other"
    elif defect == "endpoint":
        local_audit.eks["cluster"]["endpoint"] = "https://other.example.invalid"
    elif defect == "ca":
        local_audit.eks["cluster"]["certificateAuthority"]["data"] = base64.b64encode(
            b"other"
        ).decode()
    elif defect == "registry":
        local_audit.registry["registrations"][0]["hyperpod_cluster_name"] = "other"
    elif defect == "commit":
        local_audit.state["transaction_committed"] = False
    elif defect == "release":
        local_audit.environments["executor-2"]["executor_artifact"] = "f" * 64
    elif defect == "cpu-module":
        local_audit.environments["api-2"]["module_digest"] = "f" * 64
    elif defect == "serviceaccount":
        local_audit.service_account["metadata"]["annotations"][
            "eks.amazonaws.com/role-arn"
        ] = ROLE + "-other"
    else:
        field = {
            "cluster": "cluster_id",
            "region": "region",
            "role": "role_arn",
            "credentials": "credential_method",
            "caller": "caller_arn",
        }[defect]
        local_audit.environments["executor-2"][field] = "other"
    code, result = local_audit.run()
    assert code == 1 and result["verdict"] == "FAIL", result
    assert diagnostic in result["error"], result
    assert not local_audit.iam_calls(), (
        "an unrelated role's correct IAM answers must never be used"
    )


@pytest.mark.parametrize(
    ("option", "value"),
    [("--executor-role-arn", ROLE + "-other"), ("--hyperpod-cluster", "another")],
)
def test_supplied_cli_target_must_match_the_deployed_registration_and_irsa(
    local_audit: LocalAudit, option: str, value: str
) -> None:
    local_audit.argv[local_audit.argv.index(option) + 1] = value
    code, result = local_audit.run()
    assert code == 1 and result["verdict"] == "FAIL", result
    assert "target" in result["error"], "report the target mismatch explicitly"
    assert not local_audit.iam_calls(), (
        "correct IAM answers for an arbitrary CLI target cannot validate this deployment"
    )


@pytest.mark.parametrize("app", [binding.API_APP, binding.EXECUTOR_APP])
@pytest.mark.parametrize(
    "defect",
    [
        "missing",
        "unready",
        "container",
        "pending",
        "terminating",
        "count",
        "generation",
        "extra",
    ],
)
def test_main_requires_complete_ready_deployments(
    local_audit: LocalAudit, app: str, defect: str
) -> None:
    pods = local_audit.pods[app]["items"]
    deployment = local_audit.deployments[app]
    if defect == "missing":
        pods.pop()
    elif defect == "unready":
        pods[-1]["status"]["conditions"][0]["status"] = "False"
    elif defect == "container":
        pods[-1]["status"]["containerStatuses"][0]["ready"] = False
    elif defect == "pending":
        pods[-1]["status"]["phase"] = "Pending"
    elif defect == "terminating":
        pods[-1]["metadata"]["deletionTimestamp"] = NOW.isoformat()
    elif defect == "count":
        deployment["status"]["readyReplicas"] = "3"
    elif defect == "generation":
        deployment["status"]["observedGeneration"] = 1
    else:
        pods.append(copy.deepcopy(pods[0]))
    code, result = local_audit.run()
    assert code == 1 and "complete stable Ready" in result["error"], result
    assert not local_audit.iam_calls(), (
        "incomplete populations cannot establish the audited deployment"
    )


@pytest.mark.parametrize(
    "field", ["release_id", "cluster_id", "status", "formal_sequence_satisfied"]
)
def test_main_binds_predecessor_before_evaluating_iam(
    local_audit: LocalAudit, field: str
) -> None:
    proof = json.loads(local_audit.predecessor.read_text(encoding="utf-8"))
    proof[field] = False if field == "formal_sequence_satisfied" else "another"
    write_json(local_audit.predecessor, proof)
    code, result = local_audit.run()
    assert code == 1 and "predecessor" in result["error"], result
    assert result["predecessor"]["valid"] is False, (
        "the shared predecessor validator must reject the mismatch"
    )
    assert not local_audit.iam_calls(), (
        "invalid predecessors must stop the audit before IAM"
    )


@pytest.mark.parametrize(
    "defect",
    ["wrong-cluster", "wrong-role", "failed", "missing-target", "bad-json", "outside"],
)
def test_main_does_not_accept_unbound_cloudtrail_controls(
    local_audit: LocalAudit, defect: str
) -> None:
    event = local_audit.event(audit.POSITIVE_CONTROL_EVENT)
    detail = json.loads(event["CloudTrailEvent"])
    if defect == "wrong-cluster":
        detail["requestParameters"]["clusterName"] = "another"
    elif defect == "wrong-role":
        detail["userIdentity"]["sessionContext"]["sessionIssuer"]["arn"] = (
            ROLE + "-other"
        )
    elif defect == "failed":
        detail["errorCode"] = "AccessDenied"
    elif defect == "missing-target":
        detail["requestParameters"] = {}
    elif defect == "outside":
        detail["eventTime"] = (NOW - timedelta(days=1)).isoformat()
        event["EventTime"] = detail["eventTime"]
    event["CloudTrailEvent"] = "{" if defect == "bad-json" else json.dumps(detail)
    local_audit.events[audit.POSITIVE_CONTROL_EVENT] = [event]
    code, result = local_audit.run()
    assert code == 1 and result["verdict"] == "FAIL", result
    assert "CloudTrail" in result.get("error", "") or any(
        "wrong region or hours" in error for error in result.get("errors", [])
    ), result


def test_cloudtrail_pagination_and_arn_target_are_consumed(
    local_audit: LocalAudit,
) -> None:
    local_audit.pages = {
        "": {"Events": [], "NextToken": "page-2"},
        "page-2": {
            "Events": [
                local_audit.event(
                    audit.POSITIVE_CONTROL_EVENT,
                    requestParameters={"clusterName": HYPERPOD},
                )
            ]
        },
    }
    code, result = local_audit.run()
    assert code == 0, result
    assert len(result["cloudtrail_events"][audit.POSITIVE_CONTROL_EVENT]) == 1, (
        "the positive control may appear after the first page"
    )
    assert any("--next-token" in command for command, _ in local_audit.calls), (
        "the caller must actually read the next page"
    )


def test_repeated_cloudtrail_page_is_a_failure(local_audit: LocalAudit) -> None:
    local_audit.pages = {
        "": {"Events": [], "NextToken": "repeat"},
        "repeat": {"Events": [], "NextToken": "repeat"},
    }
    code, result = local_audit.run()
    assert code == 1 and "pagination" in result["error"], result
    reboots = [
        command
        for command, _ in local_audit.calls
        if command[1:3] == ["cloudtrail", "lookup-events"]
        and any(audit.POSITIVE_CONTROL_EVENT in item for item in command)
    ]
    assert len(reboots) == 2, "a repeated token must terminate instead of looping"


@pytest.mark.parametrize("event_name", audit.FORBIDDEN_EVENT_NAMES)
def test_forbidden_events_are_not_filtered_by_executor_actor(
    local_audit: LocalAudit, event_name: str
) -> None:
    local_audit.events[event_name] = [
        local_audit.event(event_name, userIdentity={"arn": "other-actor"})
    ]
    code, result = local_audit.run()
    assert code == 1 and result["cloudtrail_events"][event_name], result
    assert any("CloudTrail contains" in error for error in result["errors"]), (
        "all forbidden verbs still invalidate the window"
    )


def test_accepted_recent_window_is_incomplete_not_formal_pass(
    local_audit: LocalAudit,
) -> None:
    local_audit.argv[local_audit.argv.index("--window-end") + 1] = (
        NOW - timedelta(minutes=5)
    ).isoformat()
    local_audit.argv.append("--accept-recent-window")
    code, result = local_audit.run()
    assert code == 1 and result["verdict"] == "FAIL", result
    assert (
        result["status"] == "INCOMPLETE" and result["cloudtrail_provisional"] is True
    ), result
    assert result["formal_sequence_satisfied"] is False, (
        "accepted delivery lag cannot become formal PASS"
    )


def test_identity_drift_after_reads_prevents_pass(local_audit: LocalAudit) -> None:
    def change_after_tests(command: list[str]) -> None:
        if command[1:3] == ["-m", "pytest"]:
            local_audit.service_account["metadata"]["uid"] = "new-sa-uid"

    local_audit.before_command = change_after_tests
    code, result = local_audit.run()
    assert code == 1 and any(
        "identity drifted" in error for error in result["errors"]
    ), result


@pytest.mark.parametrize("failure", ["command", "timeout", "abort", "supervision"])
def test_initial_fail_and_no_further_commands_after_failure(
    local_audit: LocalAudit, capsys: pytest.CaptureFixture[str], failure: str
) -> None:
    write_json(local_audit.output, {"case_id": audit.CASE_ID, "verdict": "PASS"})

    def fail(command: list[str]) -> None:
        if failure == "command":
            raise OSError(SENSITIVE)
        if failure == "timeout":
            raise subprocess.TimeoutExpired(
                command, 180, output=SENSITIVE, stderr=SENSITIVE
            )
        if failure == "abort":
            raise RegionalFixtureAbort(signal.SIGTERM)
        raise ProcessSupervisionLost(SENSITIVE)

    local_audit.before_command = fail
    code, result = local_audit.run()
    assert code != 0 and result["verdict"] == "FAIL", result
    assert len(local_audit.calls) == 1, (
        "failure must stop reads, tests, retries and cleanup commands"
    )
    assert SENSITIVE not in json.dumps(result) + capsys.readouterr().out, (
        "raw command failure payloads must never enter evidence/output"
    )
    if failure == "supervision":
        assert result["status"] == "RECOVERY_REQUIRED", (
            "lost supervision is not ordinary completed failure"
        )
        marker = local_audit.run_dir / "command-supervision-lost.json"
        assert marker.exists(), "use the shared durable supervision-loss marker"
        local_audit.before_command = None
        retry_code, retry = local_audit.run()
        assert retry_code == 1 and retry["verdict"] == "FAIL", retry
        assert len(local_audit.calls) == 1, (
            "a fresh audit invocation must refuse a run with lost supervision"
        )


@pytest.mark.parametrize("end", ["bad", "2026-09-07T12:01:00Z", "2026-09-07T08:00:00Z"])
def test_initial_fail_precedes_window_validation(
    local_audit: LocalAudit, end: str
) -> None:
    write_json(local_audit.output, {"case_id": audit.CASE_ID, "verdict": "PASS"})
    local_audit.argv[local_audit.argv.index("--window-end") + 1] = end
    code, result = local_audit.run()
    assert code == 1 and result["verdict"] == "FAIL", result
    assert "window" in result["error"], (
        "window validation needs an explicit safe diagnostic"
    )
    assert not local_audit.calls, (
        "invalid local windows must fail before external reads"
    )


def test_focused_subprocess_has_no_credentials_and_logs_are_redacted(
    local_audit: LocalAudit, capsys: pytest.CaptureFixture[str]
) -> None:
    local_audit.pytest_code = 1
    code, result = local_audit.run()
    assert code == 1 and "focused regression tests failed" in result["errors"], result
    [environment] = [
        kwargs["environment"]
        for command, kwargs in local_audit.calls
        if command[1:3] == ["-m", "pytest"]
    ]
    assert (
        environment["AWS_CONFIG_FILE"]
        == environment["AWS_SHARED_CREDENTIALS_FILE"]
        == "/dev/null"
    ), "local regression tests must not inherit AWS files"
    assert (
        environment["KUBECONFIG"] == "/dev/null"
        and environment["PYTHONDONTWRITEBYTECODE"] == "1"
    ), "local tests must not inherit deployment access"
    log = (local_audit.output.parent / "focused-tests.log").read_text(encoding="utf-8")
    assert SENSITIVE not in log + json.dumps(result) + capsys.readouterr().out, (
        "no raw test output may leak into an audit artifact"
    )


@pytest.mark.parametrize(
    ("pod", "field"), [("api-2", "synthetic_route"), ("executor-2", "allow_automatic")]
)
def test_missing_switch_evidence_is_not_treated_as_unset(
    local_audit: LocalAudit, pod: str, field: str
) -> None:
    local_audit.environments[pod].pop(field)
    code, result = local_audit.run()
    assert code == 1 and "environment evidence is incomplete" in result["error"], result


def test_missing_target_input_has_a_safe_explicit_error(
    local_audit: LocalAudit,
) -> None:
    local_audit.argv[local_audit.argv.index("--executor-role-arn") + 1] = ""
    code, result = local_audit.run()
    assert code == 1 and result["error"] == "executor role ARN is required", result
    assert not local_audit.calls, (
        "missing target inputs must fail before external reads"
    )


@pytest.mark.parametrize("pod", ["api-2", "executor-2"])
def test_each_replica_switch_is_judged_without_echoing_arbitrary_values(
    local_audit: LocalAudit, capsys: pytest.CaptureFixture[str], pod: str
) -> None:
    field = "synthetic_route" if pod.startswith("api") else "allow_replace"
    local_audit.environments[pod][field] = SENSITIVE
    code, result = local_audit.run()
    assert code == 1 and result["verdict"] == "FAIL", result
    assert SENSITIVE not in json.dumps(result) + capsys.readouterr().out, (
        "invalid switch values must fail without being copied to evidence"
    )


def test_main_rejects_partial_timestamp_inventory_before_external_reads(
    local_audit: LocalAudit,
) -> None:
    write_json(
        local_audit.run_dir / "cases" / "GF-REGIONAL-DESTR-003" / "timeline.json",
        {"entries": [{"observed_at": "invalid"}]},
    )
    code, result = local_audit.run()
    assert code == 1 and "destructive evidence" in result["error"], result
    assert not local_audit.calls, "one good case must not hide another malformed window"


@pytest.mark.parametrize("pod", ["api-2", "executor-2"])
def test_population_is_rechecked_after_pod_probes(
    local_audit: LocalAudit, pod: str
) -> None:
    def change_uid(command: list[str]) -> None:
        if "exec" in command and pod in command:
            app = binding.API_APP if pod.startswith("api") else binding.EXECUTOR_APP
            local_audit.pods[app]["items"][-1]["metadata"]["uid"] += "-replacement"

    local_audit.before_command = change_uid
    code, result = local_audit.run()
    assert code == 1 and "identity drifted during probes" in result["error"], result


@pytest.mark.parametrize("plane", ["cpu", "gpu"])
def test_readonly_environment_probe_reports_only_identity_and_switches(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], plane: str
) -> None:
    import boto3

    import gpu_fault

    calls: list[str] = []

    def client(name: str, **kwargs: Any) -> Any:
        calls.append(name)
        assert kwargs["config"].read_timeout == 10, (
            "STS identity reads must have a finite timeout"
        )
        return SimpleNamespace(
            get_caller_identity=lambda: {
                "Arn": f"arn:aws:sts::{ACCOUNT}:assumed-role/executor/session",
                "Account": ACCOUNT,
            }
        )

    monkeypatch.setattr(
        boto3,
        "Session",
        lambda: SimpleNamespace(
            get_credentials=lambda: SimpleNamespace(
                method="assume-role-with-web-identity"
            ),
            client=client,
        ),
    )
    monkeypatch.setattr(gpu_fault, "module_digest", lambda: "b" * 64)
    monkeypatch.setenv("GPU_FAULT_CONTROL_PLANE_TOKEN", SENSITIVE)
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", SENSITIVE)
    monkeypatch.setenv("GPU_FAULT_CLUSTER_ID", "cluster-a")
    monkeypatch.setenv("AWS_ROLE_ARN", ROLE)
    monkeypatch.setenv("GPU_FAULT_ALLOW_HYPERPOD_REPLACE", "false")
    exec(compile(binding.environment_probe(plane), "<environment-probe>", "exec"), {})
    output = capsys.readouterr().out
    value = json.loads(output)
    assert SENSITIVE not in output, "the probe must never serialize credentials"
    assert value["module_digest"] == "b" * 64, "read the deployed component identity"
    if plane == "gpu":
        assert calls == ["sts"], (
            "only STS get-caller-identity is used by this read-only probe"
        )
        assert value["credential_method"] == "assume-role-with-web-identity", (
            "prove IRSA is the active credential source"
        )
        assert (
            value["allow_replace"] == "false" and value["cluster_id"] == "cluster-a"
        ), "read actual executor settings"
    else:
        assert not calls and "synthetic_route" in value, (
            "the API probe reads no AWS credentials"
        )


def test_stale_api_pod_environment_passes_when_configmap_pins_match(
    local_audit: LocalAudit,
) -> None:
    # Live: api-ha Pods started before a data-plane-only release re-published
    # the executor pins. The kubelet resolved the configMapKeyRef env once at
    # container start, so the Pod env still carried the previous artifact
    # digest while the ConfigMap, the release state and the executor
    # Deployment all agreed. That is a consistent product, not drift.
    for index in range(3):
        local_audit.environments[f"api-{index}"].update(
            executor_artifact="e" * 64, executor_compatibility="f" * 64
        )
    code, result = local_audit.run()
    assert code == 0 and result["verdict"] == "PASS", result
    pin_reads = [
        command
        for command, _ in local_audit.calls
        if command[0] == "kubectl"
        and command[command.index("-n") + 2 :]
        == ["get", "configmap", binding.RELEASE_PIN_CONFIGMAP, "-o", "json"]
    ]
    assert len(pin_reads) == 2, "both snapshots read the pins the product serves"
    assert all(
        command[command.index("--kubeconfig") + 1] == str(local_audit.cpu_config)
        for command in pin_reads
    ), "the release pin ConfigMap lives on the control plane"
    cpu_probes = [
        kwargs["input_text"]
        for command, kwargs in local_audit.calls
        if command[0] == "kubectl"
        and "exec" in command
        and command[command.index("-c") + 1] == "api"
        and kwargs["input_text"] != binding.REGISTRY_PROBE
    ]
    assert len(cpu_probes) == 6, cpu_probes
    assert not any("REQUIRED_REGIONAL_EXECUTOR" in probe for probe in cpu_probes), (
        "fleet pins are never read from a Pod's start-time environment snapshot"
    )


@pytest.mark.parametrize(
    "key",
    [
        "required-regional-executor-artifact-sha256",
        "required-regional-executor-compatibility-digest",
        "required-regional-executor-protocol-version",
    ],
)
def test_configmap_executor_pin_drift_fails_naming_the_configmap(
    local_audit: LocalAudit, key: str
) -> None:
    local_audit.pins[key] = "9" if key.endswith("version") else "e" * 64
    code, result = local_audit.run()
    assert code == 1 and result["verdict"] == "FAIL", result
    assert binding.RELEASE_PIN_CONFIGMAP in result["error"], result
    assert "does not match the deployed release pins" in result["error"], result
    assert not local_audit.iam_calls(), "drifted pins must not reach IAM simulation"


@pytest.mark.parametrize("defect", ["missing", "empty", "blank", "no-data"])
def test_incomplete_release_pin_configmap_is_refused(
    local_audit: LocalAudit, defect: str
) -> None:
    key = "required-regional-executor-compatibility-digest"
    if defect == "missing":
        del local_audit.pins[key]
    elif defect == "empty":
        local_audit.pins[key] = ""
    elif defect == "blank":
        local_audit.pins[key] = "   "
    else:
        local_audit.pins = None
    code, result = local_audit.run()
    assert code == 1 and result["verdict"] == "FAIL", result
    assert binding.RELEASE_PIN_CONFIGMAP in result["error"], result
    assert "incomplete" in result["error"], result
    assert not local_audit.iam_calls(), "incomplete pins must not reach IAM simulation"


def test_executor_pod_own_identity_is_still_read_from_the_pod(
    local_audit: LocalAudit,
) -> None:
    # GPU_FAULT_EXECUTOR_ARTIFACT_SHA256 is a literal value on the executor
    # Deployment (the replica's own identity), not a fleet pin, so the running
    # Pod is exactly what the product runs with.
    local_audit.environments["executor-1"]["executor_compatibility"] = "f" * 64
    code, result = local_audit.run()
    assert code == 1 and result["verdict"] == "FAIL", result
    assert (
        "target gpu replica does not match the deployed release pins"
        in (result["error"])
    ), result


def test_release_pin_table_is_the_product_fleet_pin_table() -> None:
    from gpu_fault import fleet_pins

    assert binding.RELEASE_PIN_CONFIGMAP == fleet_pins.DEFAULT_CONFIG_MAP
    assert set(binding.EXECUTOR_PIN_ENVIRONMENT.values()) <= (
        fleet_pins.PIN_ENVIRONMENT_NAMES
    )
    assert binding.EXECUTOR_PIN_CONFIG_MAP_KEYS == {
        field: next(
            key
            for key, environment in fleet_pins.CONFIG_MAP_KEY_ENVIRONMENT.items()
            if environment == name
        )
        for field, name in binding.EXECUTOR_PIN_ENVIRONMENT.items()
    }


def test_eks_registration_is_audited_through_eks_ec2_and_the_executor_role(
    local_audit: LocalAudit,
) -> None:
    """Live 2026-10-07, mixed site: the audit of the plain-EKS member died in
    ``sagemaker describe-cluster`` (ResourceNotFound). An eks-managed
    registration is read through EKS/EC2 and judged in ec2.amazonaws.com."""

    local_audit.use_eks()
    code, result = local_audit.run()
    assert code == 0 and result["verdict"] == "PASS", result
    assert (
        result["provider"] == "eks-managed" and result["cluster_id"] == EKS_CLUSTER_ID
    )
    assert result["positive_control_event"] == "RebootInstances"
    assert result["cluster"] == {
        "cluster_name": EKS_GPU_NAME,
        "cluster_arn": EKS_GPU,
        "cluster_status": "ACTIVE",
        "node_recovery": "not-applicable",
        "orchestrator": None,
        "provider": "eks-managed",
    }
    assert not [c for c, _ in local_audit.calls if c[:2] == ["aws", "sagemaker"]]
    [simulation] = local_audit.iam_calls()
    actions = simulation[
        simulation.index("--action-names") + 1 : simulation.index("--resource-arns")
    ]
    assert actions == list(audit.EKS_IAM_ACTIONS)
    assert simulation[
        simulation.index("--resource-arns") + 1 : simulation.index("--context-entries")
    ] == [f"arn:aws:ec2:{REGION}:{ACCOUNT}:instance/{i}" for i in INSTANCES], (
        "simulate against the GPU instances, never *"
    )
    assert simulation[simulation.index("--context-entries") + 1] == (
        f"ContextKeyName=aws:ResourceTag/eks:cluster-name,ContextKeyValues={EKS_GPU_NAME},"
        "ContextKeyType=string"
    )
    assert result["iam_decisions"]["ec2:RebootInstances"] == "allowed"
    assert result["cloudtrail_refused_calls"]["RunInstances"] == 3
    assert result["cloudtrail_foreign_calls"]["TerminateInstances"] == 1
    assert result["cloudtrail_events"]["TerminateInstances"] == []
    assert [
        item["instance_ids"] for item in result["cloudtrail_events"]["RebootInstances"]
    ] == [[INSTANCES[0]], [INSTANCES[1]]]
    assert result["reboot_instance_bindings"] == {
        "bound": sorted(INSTANCES),
        "unbound": [],
        "outside_cluster": [],
    }
    inventory = result["inventory_before"]
    assert inventory["count"] == 2 and inventory["cluster_name"] == EKS_GPU_NAME
    assert inventory["nodegroups"] == {
        NODEGROUP: {
            "cluster_name": EKS_GPU_NAME,
            "status": "ACTIVE",
            "health_issue_codes": [],
        }
    }
    assert inventory["instance_cluster_tags"] == {i: EKS_GPU_NAME for i in INSTANCES}
    assert result["inventory_after"] == inventory
    assert result["executor_replica_count"] == 3, (
        "the replace switches are still judged"
    )


def test_eks_audit_fails_on_an_accepted_ec2_replacement_by_the_executor_role(
    local_audit: LocalAudit,
) -> None:
    local_audit.use_eks()
    local_audit.ec2_events["TerminateInstances"].append(
        local_audit.ec2_event("TerminateInstances", [INSTANCES[1]])
    )
    code, result = local_audit.run()
    assert code == 1 and result["verdict"] == "FAIL", result
    assert result["errors"] == [
        "CloudTrail contains an accepted EC2 TerminateInstances by the executor role"
    ], result
    assert result["cloudtrail_foreign_calls"]["TerminateInstances"] == 1, (
        "the Auto Scaling termination stays a counted foreign call"
    )


def test_eks_audit_binds_the_executor_roles_reboots_to_the_registered_cluster(
    local_audit: LocalAudit,
) -> None:
    local_audit.use_eks()
    local_audit.ec2_events["RebootInstances"].append(
        local_audit.ec2_event("RebootInstances", ["i-0000000000000000"])
    )
    local_audit.instance_tags["i-0000000000000000"] = {"eks:cluster-name": "other"}
    code, result = local_audit.run()
    assert code == 1 and result["verdict"] == "FAIL", result
    assert result["errors"] == [
        "CloudTrail contains 1 RebootInstances target(s) by the executor role "
        "outside the registered EKS cluster"
    ], result
    assert result["reboot_instance_bindings"]["outside_cluster"] == [
        "i-0000000000000000"
    ]
    describes = [
        c for c, _ in local_audit.calls if c[1:3] == ["ec2", "describe-instances"]
    ]
    assert any(
        "i-0000000000000000" in c[c.index("--filters") + 1] for c in describes
    ), "a rebooted instance that is no longer a GPU node is looked up by filter"


def test_eks_audit_records_a_rebooted_instance_ec2_no_longer_knows(
    local_audit: LocalAudit,
) -> None:
    local_audit.use_eks()
    local_audit.ec2_events["RebootInstances"].append(
        local_audit.ec2_event("RebootInstances", ["i-0000000000000000"])
    )
    code, result = local_audit.run()
    assert code == 0 and result["verdict"] == "PASS", result
    assert result["reboot_instance_bindings"]["unbound"] == ["i-0000000000000000"]


def test_eks_audit_fails_when_iam_permits_a_replacement_verb(
    local_audit: LocalAudit,
) -> None:
    local_audit.use_eks()
    for item in local_audit.ec2_iam["EvaluationResults"]:
        if item["EvalActionName"] == "ec2:TerminateInstances":
            item["EvalDecision"] = "allowed"
            for row in item["ResourceSpecificResults"]:
                row["EvalResourceDecision"] = "allowed"
    code, result = local_audit.run()
    assert code == 1 and result["errors"] == [
        "executor IAM permits ec2:TerminateInstances"
    ], result


def test_eks_audit_fails_on_an_untagged_instance_or_degraded_node_group(
    local_audit: LocalAudit,
) -> None:
    local_audit.use_eks()
    del local_audit.instance_tags[INSTANCES[1]]["eks:cluster-name"]
    local_audit.nodegroup["nodegroup"]["status"] = "DEGRADED"
    code, result = local_audit.run()
    assert code == 1 and result["verdict"] == "FAIL", result
    # The untagged instance is also the one the role rebooted: no longer bound
    # to the registered cluster, that reboot is judged outside it as well.
    assert result["errors"] == [
        "CloudTrail contains 1 RebootInstances target(s) by the executor role "
        "outside the registered EKS cluster",
        f"1 GPU instance(s) are not tagged eks:cluster-name={EKS_GPU_NAME}",
        f"managed node group {NODEGROUP} is not ACTIVE on the registered EKS cluster",
    ], result


def test_a_failed_provider_command_names_its_class_and_error_code(
    local_audit: LocalAudit, capsys: pytest.CaptureFixture[str]
) -> None:
    """Live 2026-10-07: "RegionalCommandFailed: ResourceNotFound: <sensitive
    output redacted>" named neither the API asked nor the step."""

    local_audit.failing[("sagemaker", "describe-cluster")] = (
        254,
        "\nAn error occurred (ResourceNotFound) when calling the DescribeCluster "
        f"operation: {SENSITIVE}\n",
    )
    code, result = local_audit.run()
    assert code == 1 and result["verdict"] == "FAIL", result
    assert (
        result["error"]
        == "aws sagemaker describe-cluster failed (exit 254) [ResourceNotFound]"
    )
    assert SENSITIVE not in json.dumps(result) + capsys.readouterr().out

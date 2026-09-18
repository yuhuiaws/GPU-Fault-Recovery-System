from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

import pytest

from scripts.e2e.regional import audit_warm_spare_guardrails as audit
from tests.regional.test_guardrail_audit_evidence import population

GLOBALS = (
    "CPU_KUBECONFIG",
    "CPU_CONTEXT",
    "GPU_KUBECONFIG",
    "GPU_CONTEXT",
    "NAMESPACE",
    "AWS_REGION",
    "MANAGED_GPU_CLUSTER",
    "AUTOMATIC_NEGATIVE_CLUSTER",
)
ENVIRONMENT = (
    "GPU_KUBECONFIG",
    "KUBECONFIG",
    "GPU_EKS_CONTEXT",
    "GPU_FAULT_DATAPLANE_CONTEXT",
    "AWS_REGION",
    "AWS_DEFAULT_REGION",
    "GPU_FAULT_HYPERPOD_CLUSTER_NAME",
    "GPU_FAULT_AUTOMATIC_NEGATIVE_CLUSTER_NAME",
    "GPU_FAULT_CONTROL_KUBECONFIG",
    "CPU_KUBECONFIG",
    "GPU_FAULT_CONTROL_CONTEXT",
    "CPU_EKS_CONTEXT",
)


def arguments(tmp_path: Path, **changes):
    for name in ("cpu", "gpu"):
        path = tmp_path / f"{name}.yaml"
        path.write_text("apiVersion: v1\n", encoding="ascii")
        path.chmod(0o600)
    values = {
        "cpu_kubeconfig": str(tmp_path / "cpu.yaml"),
        "cpu_context": "cpu",
        "gpu_kubeconfig": str(tmp_path / "gpu.yaml"),
        "gpu_context": "gpu",
        "namespace": "gpu-fault-system",
        "region": "us-west-2",
        "managed_gpu_cluster_name": "managed",
        "automatic_negative_cluster_name": "negative",
    }
    return argparse.Namespace(**(values | changes))


@pytest.fixture
def configured(tmp_path, monkeypatch):
    for name in GLOBALS:
        monkeypatch.setattr(audit, name, getattr(audit, name))
    for name in ENVIRONMENT:
        monkeypatch.delenv(name, raising=False)
    value = arguments(tmp_path)
    audit.configure(value, {audit.CASE_IDS[0]})
    return value


def node(name="node-a"):
    return {
        "name": name,
        "uid": f"{name}-uid",
        "ready": "True",
        "gpu_allocatable": 8,
        "unschedulable": False,
        "taints": [],
        "ownership_annotations": {name: None for name in audit.OWNERSHIP_ANNOTATIONS},
    }


def raw_node(name="node-a"):
    return {
        "metadata": {"name": name, "uid": f"{name}-uid", "annotations": {}},
        "spec": {"unschedulable": False, "taints": []},
        "status": {
            "allocatable": {"nvidia.com/gpu": "8"},
            "conditions": [{"type": "Ready", "status": "True"}],
        },
    }


class Kubernetes:
    def __init__(self):
        self.deployment, self.pods = population(2)
        self.calls = []
        self.reads = {}
        self.uid = "present"
        self.drift = False
        self.reported_pod = None
        self.probe = {"status": "FAILED", "error": "controlled refusal"}
        self.registry = {
            "generation": 4,
            "registrations": [
                {
                    "cluster_id": "cluster-a",
                    "region": "us-west-2",
                    "hyperpod_cluster_name": "managed",
                    "eks_cluster_arn": "arn:aws:eks:us-west-2:111122223333:cluster/gpu",
                }
            ],
        }
        self.release = {"release_id": "release-a", "phase": "complete"}

    def __call__(self, kubeconfig, context, *arguments, **options):
        self.calls.append((kubeconfig, context, arguments, options))
        if "get" in arguments:
            offset = arguments.index("get")
            kind, name = arguments[offset + 1 : offset + 3]
            if kind == "deployment":
                return json.dumps(self.deployment)
            if kind == "configmap":
                return json.dumps({"data": {"state.json": json.dumps(self.release)}})
            if kind == "pod" and name == "-l":
                return json.dumps(self.pods)
            if kind == "pod":
                metadata = deepcopy(
                    next(
                        pod["metadata"]
                        for pod in self.pods["items"]
                        if pod["metadata"]["name"] == name
                    )
                )
                self.reads[name] = self.reads.get(name, 0) + 1
                if self.uid != "present":
                    metadata["uid"] = self.uid
                if self.drift and self.reads[name] > 1:
                    metadata["uid"] = "replacement-uid"
                return json.dumps({"metadata": metadata})
        if "exec" in arguments:
            if options.get("stdin") == audit.REGISTRY_PROBE:
                return json.dumps(self.registry)
            if options.get("stdin"):
                return "probe banner\n" + json.dumps(self.probe)
            name = arguments[arguments.index("exec") + 1]
            return json.dumps(
                {
                    "pod": self.reported_pod or name,
                    "cluster_id": "cluster-a",
                    "spare_failover": "true",
                    "remote_state": "true",
                    "allow_replace": "false",
                }
            )
        pytest.fail(f"unexpected fake Kubernetes operation: {arguments}")

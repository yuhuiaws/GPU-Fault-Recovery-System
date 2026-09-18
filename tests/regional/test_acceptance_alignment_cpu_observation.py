from __future__ import annotations

import copy
import json
import subprocess
from types import SimpleNamespace

import pytest

from gpu_fault_release.regional_deployment_inventory import CPU_RUNTIME_DEPLOYMENTS
from scripts.e2e.regional import boot_membership_observation as reader


class Inventory:
    def __init__(self):
        self.deployments = [
            {
                "metadata": {"name": name, "uid": name + "-uid", "generation": 1},
                "spec": {
                    "replicas": 1,
                    "selector": {"matchLabels": {"app": name}},
                    "template": {"metadata": {"labels": {"app": name}}},
                },
                "status": {
                    "observedGeneration": 1,
                    "replicas": 1,
                    "updatedReplicas": 1,
                    "readyReplicas": 1,
                    "availableReplicas": 1,
                },
            }
            for name in CPU_RUNTIME_DEPLOYMENTS
        ]
        self.children = {}
        for deployment in self.deployments:
            name = deployment["metadata"]["name"]
            self.children[name] = [
                {
                    "kind": "ReplicaSet",
                    "metadata": {
                        "name": name + "-rs",
                        "uid": name + "-rs-uid",
                        "ownerReferences": [
                            {
                                "apiVersion": "apps/v1",
                                "kind": "Deployment",
                                "name": name,
                                "uid": name + "-uid",
                                "controller": True,
                            }
                        ],
                    },
                },
                {
                    "kind": "Pod",
                    "metadata": {
                        "name": name + "-pod",
                        "uid": name + "-pod-uid",
                        "ownerReferences": [
                            {
                                "apiVersion": "apps/v1",
                                "kind": "ReplicaSet",
                                "name": name + "-rs",
                                "uid": name + "-rs-uid",
                                "controller": True,
                            }
                        ],
                    },
                    "spec": {"containers": [{"name": "api"}], "nodeName": "cpu-node"},
                    "status": {
                        "phase": "Running",
                        "conditions": [{"type": "Ready", "status": "True"}],
                        "containerStatuses": [
                            {
                                "name": "api",
                                "containerID": "container-" + name,
                                "restartCount": 0,
                                "ready": True,
                            }
                        ],
                    },
                },
            ]
        self.failed = ""
        self.calls = []

    def run(self, arguments, **options):
        self.calls.append((arguments, options))
        kind = arguments[arguments.index("get") + 1]
        values = (
            self.deployments
            if kind == "deployment"
            else self.children[
                arguments[arguments.index("-l") + 1].removeprefix("app=")
            ]
        )
        return subprocess.CompletedProcess(
            arguments,
            int(self.failed == kind),
            json.dumps({"items": copy.deepcopy(values)}),
            "",
        )


@pytest.mark.parametrize(
    "problem",
    [
        "none",
        "deployment-read",
        "pod-read",
        "missing-role",
        "not-ready",
        "wrong-rs-name",
        "wrong-pod-owner",
        "missing-process",
        "missing-restart",
        "selector",
    ],
)
def test_membership_observes_complete_owned_cpu_processes(monkeypatch, problem):
    inventory = Inventory()
    name = CPU_RUNTIME_DEPLOYMENTS[0]
    deployment = inventory.deployments[0]
    rs, pod = inventory.children[name]
    if problem == "deployment-read":
        inventory.failed = "deployment"
    elif problem == "pod-read":
        inventory.failed = "pod,replicaset"
    elif problem == "missing-role":
        inventory.deployments.pop()
    elif problem == "not-ready":
        pod["status"]["conditions"][0]["status"] = "False"
    elif problem == "wrong-rs-name":
        rs["metadata"]["ownerReferences"][0]["name"] = "other"
    elif problem == "wrong-pod-owner":
        pod["metadata"]["ownerReferences"][0]["name"] = "other"
    elif problem == "missing-process":
        pod["status"]["containerStatuses"][0]["containerID"] = ""
    elif problem == "missing-restart":
        pod["status"]["containerStatuses"][0]["restartCount"] = None
    elif problem == "selector":
        deployment["spec"]["selector"] = {}
    monkeypatch.setattr(reader, "run_command", inventory.run)
    site = SimpleNamespace(
        release_config={"cpu_kubeconfig": "/unit/cpu", "namespace": "unit"}
    )
    if problem == "none":
        result = reader.cpu_observation(site)
        assert set(result) == set(CPU_RUNTIME_DEPLOYMENTS)
        assert all(len(item["pods"]) == 1 for item in result.values()), (
            "each CPU role must retain its single owned process observation"
        )
        assert result[name]["pods"][0]["containers"][0]["restart_count"] == 0
        assert len(inventory.calls) == 4
    else:
        with pytest.raises(ValueError):
            reader.cpu_observation(site)

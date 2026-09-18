from __future__ import annotations

import copy
import json
import subprocess
from pathlib import Path
from typing import Any

from scripts.e2e.regional.regional_live_fixture import (
    RUNTIME_IDENTITY_DEPLOYMENTS,
    RegionalLiveFixture,
)
from tests.regional.test_live_command_boundaries import fixture, ready_pod


def deployment(name: str) -> dict[str, Any]:
    return {
        "metadata": {"name": name, "uid": name + "-uid", "generation": 2},
        "spec": {
            "replicas": 1,
            "template": {
                "spec": {
                    "initContainers": [{"name": "init", "image": "example/init:1"}],
                    "containers": [{"name": "app", "image": "example/app:1"}],
                }
            },
        },
        "status": {
            "observedGeneration": 2,
            "updatedReplicas": 1,
            "readyReplicas": 1,
            "availableReplicas": 1,
        },
    }


def node(name: str = "node-a", gpus: int = 8) -> dict[str, Any]:
    return {
        "metadata": {
            "name": name,
            "uid": name + "-uid",
            "labels": {
                "nvidia.com/gpu.product": "example-product",
                "other": "untouched",
            },
            "annotations": {
                "gpu-fault.io/owner": "unit",
                "gpu-fault.io/installer-state": "Succeeded",
                "other": "ignored",
            },
        },
        "spec": {"unschedulable": False, "taints": []},
        "status": {
            "nodeInfo": {"bootID": "boot-new"},
            "conditions": [
                {"type": "Other", "status": "False"},
                {"type": "Ready", "status": "True"},
            ],
            "allocatable": {"nvidia.com/gpu": str(gpus)},
        },
    }


class LiveModel(RegionalLiveFixture):
    def __init__(self, root: Path) -> None:
        super().__init__(fixture(root).settings)
        self.calls: list[tuple[list[str], dict[str, Any]]] = []
        self.documents: dict[tuple[str, str, str], Any] = {
            ("cpu", "configmap", "gpu-fault-regional-release-state"): {
                "data": {
                    "state.json": json.dumps(
                        {
                            "release_id": "unit-release",
                            "phase": "complete",
                            "transaction_committed": True,
                        }
                    )
                }
            },
            ("cpu", "pod", ""): {"items": [ready_pod()]},
            ("gpu", "pod", ""): {"items": [ready_pod()]},
            ("gpu", "node", "node-a"): node(),
            ("gpu", "node", ""): {"items": [node(), node("cpu-only", 0)]},
        }
        for plane, names in RUNTIME_IDENTITY_DEPLOYMENTS.items():
            self.documents[(plane, "deployment", "")] = {
                "items": [deployment(name) for name in names]
            }
        self.executions: list[Any] = [{"ok": True}]
        self.events: list[Any] = [{"Events": []}]

    def run(
        self, command: list[str], **kwargs: Any
    ) -> subprocess.CompletedProcess[str]:
        self.calls.append((list(command), copy.deepcopy(kwargs)))
        if command[:3] == ["aws", "cloudtrail", "lookup-events"]:
            value = self.events.pop(0) if len(self.events) > 1 else self.events[0]
        elif command[0] == "kubectl":
            plane = "gpu" if "--context" in command else "cpu"
            if "get" in command:
                offset = command.index("get")
                resource = command[offset + 1]
                tail = command[offset + 2 :]
                name = tail[0] if tail and not tail[0].startswith("-") else ""
                if resource in {"deployment", "pod"}:
                    name = ""
                value = self.documents[(plane, resource, name)]
            elif "exec" in command:
                value = (
                    self.executions.pop(0)
                    if len(self.executions) > 1
                    else self.executions[0]
                )
            else:
                raise AssertionError(f"unmodeled Kubernetes action: {command[:5]}")
        else:
            raise AssertionError(f"unmodeled command: {command[:3]}")
        if isinstance(value, BaseException):
            raise value
        output = value if isinstance(value, str) else json.dumps(value)
        return subprocess.CompletedProcess(command, 0, output, "")

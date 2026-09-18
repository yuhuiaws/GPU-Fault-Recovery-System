"""Controlled API and command boundary for the real activation-fence lifecycle."""

from __future__ import annotations

import copy
import hashlib
import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional.destr008_admission import (
    API_VERSION,
    ActivationFence,
    FenceBinding,
)
from scripts.e2e.regional.regional_live_fixture import (
    RegionalLiveFixture,
    RegionalLiveSettings,
)
from tests.regional._destr008_control import ControlHarness, build_control

PLURALS = {
    "validatingadmissionpolicies": "validatingadmissionpolicy",
    "validatingadmissionpolicybindings": "validatingadmissionpolicybinding",
}


@dataclass
class Harness:
    regional: RegionalLiveFixture
    binding: FenceBinding
    directory: Path
    node: dict[str, Any]
    scope: dict[str, str] = field(
        default_factory=lambda: {"cluster_id": "cluster-a", "release_id": "release-a"}
    )
    objects: dict[str, dict[str, Any]] = field(default_factory=dict)
    calls: list[tuple[str, Any]] = field(default_factory=list)
    lost_create: set[str] = field(default_factory=set)
    failed_create: set[str] = field(default_factory=set)
    failed_delete: set[str] = field(default_factory=set)
    probe_changes: dict[str, Any] = field(default_factory=dict)
    probe_exit: int = 0
    discovery: object | None = None
    read_error: Exception | None = None

    def fence(self) -> ActivationFence:
        return ActivationFence(self.regional, self.binding, self.directory)

    def watchdog(self, fence: ActivationFence, *, bind: bool = True) -> ControlHarness:
        harness = build_control(
            namespace=self.regional.settings.namespace,
            run_id=self.binding.run_id,
            cluster_id=self.binding.cluster_id,
            release_id=self.binding.release_id,
            spare_node=self.binding.node,
            fence=fence.identity(),
        )
        assert harness.tick().state == "ARMED", "only the real watchdog can arm control"
        if bind:
            fence.bind_watchdog(harness.control)
        return harness

    def install(self, manifest: dict[str, Any]) -> dict[str, Any]:
        value = copy.deepcopy(manifest)
        kind = value["kind"].lower()
        value["metadata"].update(uid=f"uid-{kind}", resourceVersion="1")
        self.objects[kind] = value
        return value

    def kube(self, plane: str, *args: str, **kwargs: Any) -> str:
        assert plane == "gpu", "the activation fence may address only its GPU context"
        self.calls.append(("kube", args))
        if args[:2] == ("get", "--raw"):
            value = self.discovery
            if value is None:
                value = {
                    "kind": "APIResourceList",
                    "groupVersion": API_VERSION,
                    "resources": [
                        {
                            "name": name,
                            "namespaced": False,
                            "verbs": ["create", "get", "delete"],
                        }
                        for name in PLURALS
                    ],
                }
            return json.dumps(value)
        if args[0] == "get":
            if self.read_error:
                raise self.read_error
            if args[1] == "node":
                return json.dumps(self.node)
            value = self.objects.get(args[1])
            if value is None:
                return ""
            if args[-1] == "jsonpath={.metadata}":
                return json.dumps(value["metadata"])
            return json.dumps(value)
        if args[0] == "create":
            manifest = json.loads(kwargs["input_text"])
            kind = manifest["kind"].lower()
            journal = json.loads(self.fence().path.read_text())
            assert kind in journal["attempted"], (
                "creation intent must be durable before the API call"
            )
            if kind in self.failed_create:
                raise TimeoutError("create outcome is unknown")
            value = self.install(manifest)
            if kind in self.lost_create:
                raise TimeoutError("create ACK lost after commit")
            return json.dumps(value)
        if args[0] == "delete":
            path = args[2]
            assert path.startswith("/apis/" + API_VERSION + "/"), (
                "cleanup must use the bound admission API"
            )
            assert "/namespaces/" not in path, (
                "admission policies are cluster-scoped resources"
            )
            kind = PLURALS[path.split("/")[-2]]
            value = self.objects[kind]
            options = json.loads(kwargs["input_text"])
            assert options["preconditions"] == {
                "uid": value["metadata"]["uid"],
                "resourceVersion": value["metadata"]["resourceVersion"],
            }, "deletion must carry the recorded UID and current resource version"
            if kind in self.failed_delete:
                raise TimeoutError("delete outcome is unknown")
            del self.objects[kind]
            return "{}"
        raise AssertionError(f"unexpected fixture command: {args}")

    def command(
        self, command: list[str], **kwargs: Any
    ) -> subprocess.CompletedProcess[str]:
        self.calls.append(("probe", command))
        assert kwargs["timeout"] == 60, "the independent process must have a deadline"
        start = command.index("--kubeconfig")
        values = dict(zip(command[start::2], command[start + 1 :: 2], strict=True))
        output = {
            "state": "DENYING_ACTIVATION",
            "node": values["--node"],
            "node_uid": values["--uid"],
            "policy": values["--policy"],
            "binding": values["--binding"],
            "marker": values["--marker"],
            "safe_dry_run_acknowledged": True,
            "activation_dry_run_denied": True,
            "probe_not_persisted": True,
            "source_sha256": hashlib.sha256(kwargs["input_text"].encode()).hexdigest(),
            **self.probe_changes,
        }
        return subprocess.CompletedProcess(
            command, self.probe_exit, json.dumps(output), ""
        )


def build_harness(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Harness:
    cpu = tmp_path / "cpu-config"
    gpu = tmp_path / "gpu-config"
    cpu.write_text("fixture only\n")
    gpu.write_text("fixture only\n")
    regional = RegionalLiveFixture(
        RegionalLiveSettings(
            cpu, gpu, "gpu-a", "gpu-fault-system", "cluster-a", "us-west-2"
        )
    )
    binding = FenceBinding(
        "destr008-example", "cluster-a", "spare-a", "node-uid-a", "release-a"
    )
    harness = Harness(
        regional,
        binding,
        tmp_path / "fences",
        {
            "metadata": {"name": "spare-a", "uid": "node-uid-a", "annotations": {}},
            "spec": {"unschedulable": True},
        },
    )
    monkeypatch.setattr(regional, "evidence_identity", lambda: dict(harness.scope))
    monkeypatch.setattr(regional, "kubectl", harness.kube)
    monkeypatch.setattr(regional, "run", harness.command)
    return harness

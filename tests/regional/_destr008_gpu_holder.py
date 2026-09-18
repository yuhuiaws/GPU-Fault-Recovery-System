"""Local Kubernetes boundary for the real durable holder controller."""

from __future__ import annotations

import copy
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import destr008_gpu_holder as holder
from scripts.e2e.regional.regional_live_fixture import (
    RegionalLiveFixture,
    RegionalLiveSettings,
)
from scripts.e2e.regional.warm_spare_fixture import WarmSpareLiveFixture


@dataclass
class Clock:
    epoch: float = 2000000000.0
    elapsed: float = 0.0

    def time(self) -> float:
        return self.epoch

    def monotonic(self) -> float:
        return self.elapsed

    def sleep(self, seconds: float) -> None:
        self.epoch += seconds
        self.elapsed += seconds


def admitted_pod(manifest: dict[str, Any]) -> dict[str, Any]:
    value = copy.deepcopy(manifest)
    value["metadata"].update(uid="holder-uid", resourceVersion="1")
    value["spec"].update(
        dnsPolicy="ClusterFirst",
        schedulerName="default-scheduler",
        serviceAccountName="default",
        serviceAccount="default",
        securityContext={},
        priority=0,
        preemptionPolicy="PreemptLowerPriority",
    )
    value["spec"]["containers"][0].update(
        terminationMessagePath="/dev/termination-log", terminationMessagePolicy="File"
    )
    value["status"] = {
        "phase": "Running",
        "conditions": [{"type": "Ready", "status": "True"}],
    }
    return value


@dataclass
class HolderHarness:
    regional: RegionalLiveFixture
    directory: Path
    clock: Clock = field(default_factory=Clock)
    identity: dict[str, str] = field(
        default_factory=lambda: {"cluster_id": "cluster-a", "release_id": "release-a"}
    )
    node: dict[str, Any] = field(
        default_factory=lambda: {
            "metadata": {"name": "spare-a", "uid": "node-uid-a"},
            "spec": {},
        }
    )
    pod: dict[str, Any] | None = None
    calls: list[tuple[str, ...]] = field(default_factory=list)
    before: Callable[[tuple[str, ...]], None] | None = None
    create_error: BaseException | None = None
    lost_create: bool = False
    ack_raw: str | None = None
    ack_change: Callable[[dict[str, Any]], None] | None = None
    delete_error: BaseException | None = None
    lost_delete: bool = False
    controller: holder.BoundedGpuHolderFixture = field(init=False)

    def new_holder(self, **changes: Any) -> holder.BoundedGpuHolderFixture:
        values = {
            "node": "spare-a",
            "run_id": "destr008-run",
            "state_directory": self.directory,
            "node_uid": "node-uid-a",
            "plan_sha256": "a" * 64,
            "release_id": "release-a",
            **changes,
        }
        return holder.BoundedGpuHolderFixture(
            WarmSpareLiveFixture(self.regional, "hyperpod-a"), **values
        )

    def journal(self) -> dict[str, Any]:
        value: dict[str, Any] = json.loads(self.controller.path.read_text())
        return value

    def mutations(self) -> list[tuple[str, ...]]:
        return [args for args in self.calls if args[0] != "get"]

    def kube(self, plane: str, *args: str, **kwargs: Any) -> str:
        assert plane == "gpu", "the holder must address only its bound GPU context"
        self.calls.append(args)
        if self.before is not None:
            self.before(args)
        if args[:2] == ("get", "node"):
            assert args[2] == "spare-a", "Node reads must be exactly scoped"
            return json.dumps(self.node)
        if args[:2] == ("get", "pod"):
            assert args[2] == self.controller.name, "Pod reads must be exactly scoped"
            assert 0 < kwargs["timeout"] <= 60, "Pod reads must be bounded"
            if self.pod is None:
                return ""
            value = (
                self.pod["metadata"] if args[-1] == "jsonpath={.metadata}" else self.pod
            )
            return json.dumps(value)
        if args[0] == "create":
            assert args == ("create", "-f", "-", "-o", "json"), "apply is forbidden"
            assert kwargs["timeout"] == 60, "CREATE must be bounded"
            intent = self.journal()
            assert intent["phase"] == "CREATING", "intent must precede CREATE"
            assert intent["create_started"] is True, "intent must be durable"
            assert intent["pod_uid"] is None, "no UID may be invented before ACK"
            assert intent["deadline_at"] > self.clock.time(), "CREATE may not be late"
            if self.create_error is not None:
                raise self.create_error
            assert self.pod is None, "CREATE cannot replace a Pod"
            self.pod = admitted_pod(json.loads(kwargs["input_text"]))
            if self.lost_create:
                raise TimeoutError("local fake: ACK lost after commit")
            if self.ack_raw is not None:
                return self.ack_raw
            value = copy.deepcopy(self.pod)
            if self.ack_change is not None:
                self.ack_change(value)
            return json.dumps(value)
        if args[0] == "delete":
            assert args == (
                "delete",
                "--raw",
                f"/api/v1/namespaces/{self.regional.settings.namespace}/pods/{self.controller.name}",
                "-f",
                "-",
            ), "deletion must address exactly the bound namespaced Pod"
            assert kwargs["timeout"] == 120, "DELETE must be bounded"
            assert self.pod is not None, "only an observed Pod may be deleted"
            options = json.loads(kwargs["input_text"])
            assert options == {
                "apiVersion": "v1",
                "kind": "DeleteOptions",
                "preconditions": {
                    "uid": self.pod["metadata"]["uid"],
                    "resourceVersion": self.pod["metadata"]["resourceVersion"],
                },
                "propagationPolicy": "Foreground",
            }, "DELETE must use the checked UID and resourceVersion"
            if self.delete_error is not None:
                raise self.delete_error
            self.pod = None
            if self.lost_delete:
                raise TimeoutError("local fake: DELETE ACK lost after commit")
            return "{}"
        raise AssertionError(f"unexpected local holder command: {args}")


def build_holder(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> HolderHarness:
    cpu, gpu = tmp_path / "cpu-config", tmp_path / "gpu-config"
    if not cpu.exists():
        cpu.write_text("local test configuration only\n")
        gpu.write_text("local test configuration only\n")
    regional = RegionalLiveFixture(
        RegionalLiveSettings(
            cpu, gpu, "gpu-a", "gpu-fault-system", "cluster-a", "us-west-2"
        )
    )
    harness = HolderHarness(regional, tmp_path / "holder-state")
    monkeypatch.setattr(regional, "evidence_identity", lambda: dict(harness.identity))
    monkeypatch.setattr(regional, "kubectl", harness.kube)
    monkeypatch.setattr(holder, "time", harness.clock)
    harness.controller = harness.new_holder()
    return harness

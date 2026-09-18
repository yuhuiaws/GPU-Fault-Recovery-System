"""Real cancellation/Store logic behind fake CPU and GPU Kubernetes boundaries."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from kubernetes import client

from gpu_fault.host_health import NodeHealthIngestionResult
from gpu_fault.models import WorkflowStatus
from scripts.e2e.regional import destr008_cancellation as lifecycle
from scripts.e2e.regional import destr008_safety as safety_module
from scripts.e2e.regional.destr008_admission import FenceBinding
from scripts.e2e.regional.probes import destr008_cancellation_probe as probe
from scripts.e2e.regional.probes import destr008_cancellation_protocol as wire
from tests.regional._destr008_admission import Harness as GpuApi
from tests.regional._destr008_cancellation_controller import CpuApi, build_api
from tests.regional.test_destr008_cancellation_probe import FakeApi, MemoryCPU, seed


@dataclass
class Observer:
    api: FakeApi
    watchdog: probe.Watchdog
    last_tick: float = -1


@dataclass
class CausalHarness:
    cpu: CpuApi
    gpu: GpuApi
    store: MemoryCPU
    safety: safety_module.ShortageSafety
    observers: dict[str, Observer] = field(default_factory=dict)
    observer_enabled: bool = True

    def progress(self) -> None:
        if not self.observer_enabled:
            return
        for (kind, _), pod in list(self.cpu.objects.items()):
            if (
                kind != "pod"
                or pod.get("status", {}).get("phase") != "Running"
                or ("job", pod["metadata"]["ownerReferences"][0]["name"])
                not in self.cpu.objects
            ):
                continue
            command = pod["spec"]["containers"][0]["command"]
            name = command[command.index("--configmap") + 1]
            current = self.cpu.objects.get(("configmap", name))
            if current is None:
                continue
            metadata = current["metadata"]
            plan = wire.decode(wire.Plan, current["data"]["plan.json"])
            pod_uid = pod["metadata"]["uid"]
            observer = self.observers.get(pod_uid)
            if observer is None:
                api = FakeApi(
                    plan,
                    namespace=metadata["namespace"],
                    name=name,
                    uid=metadata["uid"],
                )
                port = probe.KubernetesControlMap(
                    api,
                    namespace=metadata["namespace"],
                    name=name,
                    uid=metadata["uid"],
                    plan_sha256=wire.digest(plan),
                    probe_sha256=plan.probe_sha256,
                )
                cleanup = "--cleanup-only" in command
                observer = Observer(
                    api,
                    probe.Watchdog(
                        port,
                        self.store,
                        sleep=self.cpu.clock.sleep,
                        monotonic=self.cpu.clock.monotonic,
                        cleanup_only=cleanup,
                        cleanup_seconds=(
                            int(command[command.index("--cleanup-seconds") + 1])
                            if cleanup
                            else wire.MAX_CLEANUP_SECONDS
                        ),
                        cleanup_attempt_id=(
                            command[command.index("--cleanup-attempt-id") + 1]
                            if cleanup
                            else None
                        ),
                    ),
                )
                self.observers[pod_uid] = observer
            if self.cpu.clock.now() - observer.last_tick < 1:
                continue
            observer.api.value = client.V1ConfigMap(
                api_version="v1",
                kind="ConfigMap",
                metadata=client.V1ObjectMeta(
                    namespace=metadata["namespace"],
                    name=name,
                    uid=metadata["uid"],
                    resource_version=metadata["resourceVersion"],
                ),
                data=copy.deepcopy(current["data"]),
            )
            receipt = observer.watchdog.tick(int(self.cpu.clock.now()))
            observer.last_tick = self.cpu.clock.now()
            current["data"] = copy.deepcopy(observer.api.value.data)
            metadata["resourceVersion"] = observer.api.value.metadata.resource_version
            self.cpu.counter = max(self.cpu.counter, int(metadata["resourceVersion"]))
            if receipt.state == "QUIESCENT":
                self.cpu.finish(pod)

    def kube(self, plane: str, *args: str, **kwargs: Any) -> str:
        if plane == "cpu":
            return self.cpu.kube(plane, *args, **kwargs)
        return self.gpu.kube(plane, *args, **kwargs)

    def observation(self) -> dict[str, Any]:
        return {
            "runtime_profile_version": "profile-a",
            "workload_ids": ["training/job-a"],
        }

    def acknowledge(self, claim_id: str) -> None:
        assert self.safety.plan is not None
        seed(self.store, self.safety.plan, status=WorkflowStatus.FAILED)
        self.safety.acknowledge(
            claim_id,
            {
                "status": 200,
                "body": NodeHealthIngestionResult(
                    batch_id=self.safety.event_id,
                    duplicate=False,
                    incident_ids=["incident-a"],
                    workflow_request_ids=["workflow-a"],
                ).model_dump(mode="json"),
            },
        )


def build_causal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> CausalHarness:
    cpu, _, _ = build_api(tmp_path, monkeypatch)
    cpu.source = Path(probe.__file__).parent
    cpu.bind(monkeypatch)
    binding = FenceBinding(
        "destr008-causal", "cluster-a", "spare-a", "node-uid-a", "release-a"
    )
    gpu = GpuApi(
        cpu.regional,
        binding,
        cpu.directory,
        {
            "metadata": {"name": "spare-a", "uid": "node-uid-a", "annotations": {}},
            "spec": {"unschedulable": True},
        },
    )
    monkeypatch.setattr(cpu.regional, "run", gpu.command)
    monkeypatch.setattr(safety_module, "time", SimpleNamespace(time=cpu.clock.now))
    real_controller = lifecycle.CancellationWatchdog

    def construct(*args: Any, **kwargs: Any) -> lifecycle.CancellationWatchdog:
        return real_controller(
            *args,
            **kwargs,
            clock=cpu.clock.now,
            monotonic=cpu.clock.monotonic,
            sleep=cpu.clock.sleep,
        )

    monkeypatch.setattr(lifecycle, "CancellationWatchdog", construct)
    safety = safety_module.ShortageSafety(
        cpu.regional,
        run_id=binding.run_id,
        attempt_id="attempt-a",
        event_id="destr008-event",
        fault_node="fault-a",
        spare_node=binding.node,
        spare_uid=binding.node_uid,
        release_id=binding.release_id,
        directory=cpu.directory,
    )
    harness = CausalHarness(cpu, gpu, MemoryCPU(), safety)
    monkeypatch.setattr(cpu, "progress", harness.progress)
    monkeypatch.setattr(cpu.regional, "kubectl", harness.kube)
    return harness

"""One real watchdog/Memory Store behind a controlled CPU ConfigMap command API."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from kubernetes import client

from gpu_fault.host_health import NodeHealthIngestionResult
from gpu_fault.models import WorkflowStatus
from gpu_fault.remote_command_models import RemoteCommandStatus
from scripts.e2e.regional.destr008_watchdog_control import CancellationControl
from scripts.e2e.regional.probes import destr008_cancellation_probe as probe
from scripts.e2e.regional.probes import destr008_cancellation_protocol as wire
from tests.regional.test_destr008_cancellation_probe import (
    NOW,
    FakeApi,
    MemoryCPU,
    command,
    seed,
)
from tests.regional.test_destr008_cancellation_probe import plan as make_plan


@dataclass
class Clock:
    at: float = float(NOW)
    elapsed: float = 0

    def __call__(self) -> float:
        return self.at

    def monotonic(self) -> float:
        return self.elapsed

    def advance(self, seconds: float) -> None:
        self.at += seconds
        self.elapsed += seconds


@dataclass
class CpuApi:
    api: FakeApi
    calls: list[tuple[tuple[str, ...], float]] = field(default_factory=list)
    reads: list[str | Exception] = field(default_factory=list)
    patches: list[str | Exception] = field(default_factory=list)
    before_patch: Callable[[], None] | None = None

    def document(self) -> dict[str, Any]:
        with client.ApiClient() as serializer:
            value = serializer.sanitize_for_serialization(self.api.value)
        assert isinstance(value, dict), (
            "controlled ConfigMap serialization must be an object"
        )
        return value

    def __call__(
        self, *args: str, stdin: bytes | None = None, timeout: float = 30
    ) -> str:
        self.calls.append((args, timeout))
        assert 0 < timeout <= 30, "CPU commands must have a bounded timeout"
        assert args[:3] in {
            ("get", "configmap", self.api.name),
            ("patch", "configmap", self.api.name),
        }, "parent control may only read/patch its one ConfigMap"
        assert args[args.index("--namespace") + 1] == self.api.namespace
        assert args[-2:] == ("-o", "json")
        if args[0] == "get":
            assert stdin is None
            if self.reads:
                value = self.reads.pop(0)
                if isinstance(value, Exception):
                    raise value
                return value
            self.api.read_namespaced_config_map(
                self.api.name, self.api.namespace, _request_timeout=(3, 5)
            )
        else:
            assert "--type=json" in args and "--patch-file=/dev/stdin" in args
            assert stdin is not None
            operations = json.loads(stdin)
            writes = [item for item in operations if item["op"] != "test"]
            assert len(writes) == 1 and writes[0]["path"] == "/data/control.json"
            assert writes[0]["op"] == "replace", "parent must not write watchdog status"
            if self.before_patch is not None:
                callback, self.before_patch = self.before_patch, None
                callback()
            self.api.patch_namespaced_config_map(
                self.api.name, self.api.namespace, operations, _request_timeout=(3, 5)
            )
            if self.patches:
                value = self.patches.pop(0)
                if isinstance(value, Exception):
                    raise value
                return value
        return wire.encode(self.document())


@dataclass
class ControlHarness:
    plan: wire.Plan
    api: FakeApi
    port: probe.KubernetesControlMap
    store: MemoryCPU
    clock: Clock
    cpu: CpuApi
    control: CancellationControl
    watchdog: probe.Watchdog
    posts: int = 0

    def tick(self) -> wire.Receipt:
        return self.watchdog.tick(int(self.clock()))

    def sleep(self, seconds: float) -> None:
        self.clock.advance(seconds)
        self.tick()

    def submit(
        self,
        *,
        status: WorkflowStatus = WorkflowStatus.FAILED,
        command_status: RemoteCommandStatus | None = None,
    ) -> str:
        claim_id = self.control.claim()
        self.posts += 1
        incident, workflow = seed(self.store, self.plan, status=status)
        if command_status is not None:
            command(self.store, incident, workflow, status=command_status)
        self.control.acknowledge(claim_id, self.response())
        return claim_id

    def response(self) -> dict[str, Any]:
        return {
            "status": 200,
            "body": NodeHealthIngestionResult(
                batch_id=self.plan.event_id,
                incident_ids=["incident-a"],
                workflow_request_ids=["workflow-a"],
                duplicate=False,
            ).model_dump(mode="json"),
        }

    def finish(self) -> wire.Receipt:
        self.control.request_close()
        self.tick()
        self.clock.advance(wire.QUIET_SECONDS)
        result = self.tick()
        assert result.state == "QUIESCENT", (
            "fixture must obtain actual Store drain proof"
        )
        return result

    def cleanup(self, *, attempt_id: str = "cleanup-a", seconds: int = 30) -> None:
        self.watchdog = probe.Watchdog(
            self.port,
            self.store,
            cleanup_only=True,
            cleanup_attempt_id=attempt_id,
            cleanup_seconds=seconds,
            sleep=lambda _: None,
            monotonic=self.clock.monotonic,
        )

    def rewrite(self, key: str, value: Any) -> None:
        self.api.value.data[key] = wire.encode(value)
        self.api.value.metadata.resource_version = str(
            int(self.api.value.metadata.resource_version) + 1
        )


def build_control(
    *,
    namespace: str = "cpu",
    name: str = "watchdog",
    uid: str = "cm-uid",
    **plan_fields: Any,
) -> ControlHarness:
    bound = make_plan(**plan_fields)
    clock = Clock(float(bound.created_at))
    api = FakeApi(bound, namespace=namespace, name=name, uid=uid)
    port = probe.KubernetesControlMap(
        api,
        namespace=namespace,
        name=name,
        uid=uid,
        plan_sha256=wire.digest(bound),
        probe_sha256=wire.source_sha256(),
    )
    store = MemoryCPU()
    cpu = CpuApi(api)
    control = CancellationControl(
        plan=bound,
        namespace=namespace,
        name=name,
        uid=uid,
        cpu=cpu,
        clock=clock,
        monotonic=clock.monotonic,
    )
    watchdog = probe.Watchdog(
        port, store, sleep=lambda _: None, monotonic=clock.monotonic
    )
    return ControlHarness(bound, api, port, store, clock, cpu, control, watchdog)

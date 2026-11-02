"""A node and a control plane for the GF-GENERAL-LIVE-024 ``xid_twin_*`` tests.

The variant's own code runs unchanged; what it reaches is faked at the two
edges a live run has:

* the node -- ``TwinNode`` answers the DCGM field replay controller and the
  kmsg kernel replay probe the way the live probes do, from a virtual clock:
  the metrics collector scrapes the field tool every ``interval`` seconds once
  repointed, the tool serves the installed plan (the gauge at 0 until
  advanced, then the XID) and journals every read with its ``at``, and the
  collector delivers a batch when the real one would (its first scrape after
  the restart -- ``initial-baseline`` -- and a change of the last XID --
  ``xid-changed``); the kernel collector posts each reviewed line the moment
  it is armed;
* the control plane -- ``TwinProduct`` is a real ``ApplicationContext`` over an
  in-memory store, fed through its own collector routes, with the production
  dispatcher and executor over a recording adapter that answers every step
  (``tests/orchestration/_xid_report_twins_support``), the XID correlation
  loop run on the virtual clock, and every store probe of the variant
  (``cpu_python``) executed in-process over that store.

So the verdict is the variant's oracle over the product's own records. A test
removes batch 237's twin rule (``without_twin_rule``) to show the oracle fails
the shape the product had before it.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import dataclasses
import io
import json
import sys
import types
from collections.abc import Iterator
from datetime import datetime, timedelta, timezone
from typing import Any
from unittest import mock

from gpu_fault.app import ApplicationContext, default_simulated_profile
from gpu_fault.channel_registry import GPU_METRICS_PATH, NVIDIA_KERNEL_PATH
from gpu_fault.execution import ProductionExecutorConfig, WorkflowDispatcherConfig
from gpu_fault.gpu_metrics import GpuMetricBatch, GpuMetricSample, GpuMetricSource
from gpu_fault.models import WorkflowOperation, WorkflowStatus
from gpu_fault.store import InMemoryStore
from gpu_fault.telemetry import WorkloadCoverageHeartbeat
from scripts.e2e.regional.host_probe_fixture import HostProbeMissingResponseError
from tests._builders import asgi_client, attempt_observation, container_observation
from tests.orchestration._xid_report_twins_support import RecordingAdapter
from tests.regional._general_live_fakes import FakeProbe
from tests.regional._general_live_gpu_fakes import survey

NODE = "ip-10-0-43-5.us-west-2.compute.internal"
CLUSTER = "gpu-eks-cluster"
BOOT = "33333333-3333-4333-8333-333333333333"
G0 = "GPU-00000000-0000-4000-8000-000000000000"
G1 = "GPU-11111111-1111-4111-8111-111111111111"
#: An eight-GPU p5en node (the unbound-job variant holds four of them).
EIGHT_GPUS = tuple(f"GPU-{index:08d}-0000-4000-8000-{index:012d}" for index in range(8))
#: The product's clock sits in the past of the real one: its records are
#: written at real time, so everything the variant reads "since" is after.
START = datetime.now(timezone.utc).replace(microsecond=0) - timedelta(minutes=40)
EPOCH_START = START.timestamp()


class VirtualClock:
    """The one clock of a test run: the variant's waits and the node's scrapes."""

    def __init__(self) -> None:
        self.now = START
        self.listeners: list[Any] = []

    def clock(self) -> datetime:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += timedelta(seconds=max(0.0, float(seconds)))
        for listener in list(self.listeners):
            listener(self.now)

    def epoch(self) -> float:
        return self.now.timestamp()


class RestartingAdapter(RecordingAdapter):
    """Answers every step like the recording adapter; a RESTART_WORKLOAD
    records the copy's attempt id the way the product's restart does."""

    def execute(self, context):
        outcome = super().execute(context)
        if context.step.operation is WorkflowOperation.RESTART_WORKLOAD:
            return dataclasses.replace(
                outcome,
                details={
                    **(outcome.details or {}),
                    "restart_attempt_id": "train-a001-r-1",
                },
            )
        return outcome


class TwinProduct:
    """The in-memory control plane the faked node reports to."""

    def __init__(self, clock: VirtualClock, *, remediate: bool) -> None:
        self.clock = clock
        self.store = InMemoryStore()
        self.adapter = RestartingAdapter()
        self.adapter.hold = False
        self.remediate = remediate
        options: dict[str, Any] = {}
        if remediate:
            options = {
                "production_executor_config": ProductionExecutorConfig(
                    enabled=True,
                    executor_id="s62-executor",
                    allowed_operations=frozenset(WorkflowOperation),
                ),
                "production_adapters": [self.adapter],
                "dispatcher_config": WorkflowDispatcherConfig(
                    enabled=True, poll_interval_seconds=1, batch_size=50
                ),
            }
        self.context = ApplicationContext(store=self.store, **options)
        self.store.save_profile(default_simulated_profile())
        self.real_start = datetime.now(timezone.utc)
        coordinator = self.context.xid_correlation
        coordinator.now = self.product_now
        self.coverage(START)
        clock.listeners.append(self.pump)

    def product_now(self) -> datetime:
        """The product's own clock: real time advanced by the virtual one."""

        return self.real_start + (self.clock.now - START)

    def coverage(self, at: datetime) -> None:
        self.context.topology.observe_coverage(
            WorkloadCoverageHeartbeat(
                cluster_id=CLUSTER,
                observed_at=at,
                watcher_instance="watcher-a",
                resource_version="1",
            )
        )

    def job(self, job_id: str, attempt_id: str, gpus: list[str]) -> None:
        """A managed attempt on ``gpus`` of the node (the unbound-job variant)."""

        self.store.save_attempt_observation(
            attempt_observation(
                job_id,
                attempt_id,
                START,
                cluster_id=CLUSTER,
                started_at=START - timedelta(minutes=30),
                containers=[
                    container_observation(
                        f"pod-{attempt_id}", "trainer-0", 0, NODE, gpu_uuids=gpus
                    )
                ],
                workload_ids=[f"training/pytorchjob/{job_id}"],
                restart_budget=2,
            )
        )

    def post(self, path: str, body: dict[str, Any]) -> None:
        async def call() -> None:
            async with asgi_client(self.context) as client:
                response = await client.post(path, json=body)
                assert response.status_code == 200, (path, response.text)

        asyncio.run(call())

    def pump(self, now: datetime) -> None:
        """Due XID correlations, and (remediating) the dispatcher's passes."""

        self.coverage(now)
        self.context.xid_correlation.run_once()
        if not self.remediate:
            return
        for _ in range(4):
            for workflow in self.store.list_workflows(set(WorkflowStatus), limit=200):
                if workflow.not_before is not None and workflow.status in {
                    WorkflowStatus.PENDING,
                    WorkflowStatus.SAFETY_PENDING,
                }:
                    self.store.save_workflow(
                        workflow.model_copy(update={"not_before": None}),
                        expected=workflow,
                    )
            self.context.dispatcher.run_once()

    def probe(self, script: str, *arguments: str, **_options: Any) -> dict[str, Any]:
        """A store probe of the variant, run in-process over this store."""

        output = io.StringIO()
        context = types.SimpleNamespace(store=self.store)
        with (
            mock.patch.object(ApplicationContext, "from_environment", lambda: context),
            mock.patch.object(sys, "argv", ["probe", *arguments]),
            contextlib.redirect_stdout(output),
        ):
            try:
                exec(compile(script, "<store-probe>", "exec"), {"__name__": "__main__"})
            except SystemExit:
                pass
        return json.loads(output.getvalue().splitlines()[-1])


def _decoded(arguments: tuple[str, ...]) -> Any:
    return json.loads(base64.b64decode(arguments[arguments.index("--b64") + 1]))


class TwinNode:
    """The node: the field replay behind the metrics collector, the kmsg fixture
    behind the kernel collector, both on the virtual clock."""

    def __init__(
        self,
        product: TwinProduct,
        clock: VirtualClock,
        *,
        interval: int = 15,
        phase: float = 4.0,
        gpus: tuple[str, ...] = (G0, G1),
        exec_away: tuple[float, float] | None = None,
    ) -> None:
        self.product = product
        self.clock = clock
        self.interval = interval
        self.phase = phase
        #: HyperPod's quiesce: ``(after, seconds)`` from the delivery of the
        #: XID change, every probe exec comes back empty (kubelet stopped).
        self.exec_away = exec_away
        self.changed_at: datetime | None = None
        self.refused: list[tuple[str, float]] = []
        self.survey = survey(gpus=gpus)
        self.plan: dict[str, Any] | None = None
        self.armed = False
        self.repointed = False
        self.disarmed = False
        self.step = 0
        self.reads = 0
        self.reads_in_step = 0
        self.next_scrape: datetime | None = None
        self.journal: list[dict[str, Any]] = []
        self.delivered: dict[str, float] = {}
        self.records: list[dict[str, Any]] = []
        self.kernel_posts: list[dict[str, Any]] = []
        clock.listeners.append(self.tick)

    # -- the metrics collector and the field tool ------------------------ #
    def tick(self, now: datetime) -> None:
        while self.next_scrape is not None and self.next_scrape <= now:
            self.scrape(self.next_scrape)
            self.next_scrape += timedelta(seconds=self.interval)

    def _served(self) -> dict[str, float]:
        if not self.armed or self.disarmed or self.plan is None:
            return {}
        steps = self.plan["steps"]
        if self.step >= len(steps):
            return {}
        if self.reads_in_step >= int(steps[self.step]["reads"]):
            if steps[self.step].get("until_advance"):
                return {}
            self.step += 1
            self.reads_in_step = 0
            return self._served()
        values = {}
        for operation in steps[self.step]["ops"]:
            for gpu in operation["gpus"]:
                values[f"{gpu}/{operation['field']}"] = float(operation["value"])
        return values

    def scrape(self, at: datetime) -> None:
        served = self._served()
        self.reads += 1
        if self.armed and not self.disarmed:
            self.reads_in_step += 1
            self.journal.append(
                {
                    "read": self.reads,
                    "armed": True,
                    "at": at.timestamp(),
                    "step": self.step,
                    "served": dict(served),
                    "aborted": None,
                    "proof_scope": "live-replay",
                }
            )
        first = not self.delivered
        changed = {
            key: value
            for key, value in served.items()
            if key in self.delivered and self.delivered[key] != value
        }
        if first or changed:
            self.deliver(
                at, served, reason="initial-baseline" if first else "xid-changed"
            )
            if changed and self.changed_at is None:
                self.changed_at = at
        self.delivered = dict(served) if served else {"none": 0.0}

    def deliver(self, at: datetime, served: dict[str, float], *, reason: str) -> None:
        samples = []
        for key, value in served.items():
            gpu, _field = key.split("/", 1)
            entry = self.survey["exporter"]["gpus"][gpu]
            samples.append(
                GpuMetricSample(
                    metric_name="DCGM_FI_DEV_XID_ERRORS",
                    canonical_name="xid_last_error",
                    value=value,
                    gpu_index=entry["gpu_index"],
                    gpu_uuid=gpu,
                    pci_bdf=entry["pci_bdf"],
                    labels={
                        "gpu": entry["gpu_index"],
                        "UUID": gpu,
                        "pci_bus_id": entry["pci_bdf"],
                        "modelName": "NVIDIA H200",
                    },
                )
            )
        for gpu, entry in self.survey["exporter"]["gpus"].items():
            samples.append(
                GpuMetricSample(
                    metric_name="DCGM_FI_DEV_GPU_TEMP",
                    canonical_name="gpu_temperature_c",
                    value=40.0,
                    gpu_index=entry["gpu_index"],
                    gpu_uuid=gpu,
                    pci_bdf=entry["pci_bdf"],
                    labels={"gpu": entry["gpu_index"], "UUID": gpu},
                )
            )
        batch = GpuMetricBatch(
            batch_id=f"scrape-{self.reads}",
            cluster_id=CLUSTER,
            node_id=NODE,
            observed_at=at,
            source=GpuMetricSource.DCGM_EXPORTER,
            product="NVIDIA H200",
            driver_branch=580,
            cuda_version="13.0",
            runtime_profile_version="simulated-v1",
            samples=samples,
            edge_filter_reasons=[reason],
        )
        self.product.post(GPU_METRICS_PATH, batch.model_dump(mode="json"))

    def away(self) -> bool:
        """Whether kubelet is stopped now (``exec_away``)."""

        if self.exec_away is None or self.changed_at is None:
            return False
        after, seconds = self.exec_away
        elapsed = (self.clock.now - self.changed_at).total_seconds()
        return after <= elapsed < after + seconds

    def guarded(self, script: dict[str, Any]) -> FakeProbe:
        """``script`` as a probe whose every call fails while kubelet is away,
        the way the live exec did (no output, the Node still Ready)."""

        node = self

        def guard(name: str, value: Any) -> Any:
            def call(arguments: tuple[str, ...]) -> dict[str, Any]:
                if node.away():
                    node.refused.append((name, node.clock.epoch()))
                    raise HostProbeMissingResponseError(
                        "host probe response is missing (exit 1): no output came "
                        f"back over kubectl exec into gpu-fault-host-probe-test on "
                        f"{NODE}; an unclassified kubectl error (withheld); node "
                        "Ready=True (the node is Ready, so the exec channel itself "
                        "failed)"
                    )
                return value(arguments) if callable(value) else dict(value)

            return call

        return FakeProbe({name: guard(name, value) for name, value in script.items()})

    def dcgm(self, **extra: Any) -> FakeProbe:
        node = self

        def install(arguments: tuple[str, ...]) -> dict[str, Any]:
            name = arguments[arguments.index("--name") + 1]
            if name == "plan.json":
                node.plan = _decoded(arguments)
            return {"ok": True}

        def arm(arguments: tuple[str, ...]) -> dict[str, Any]:
            assert "--before-repoint" in arguments, "the twin arms before the repoint"
            node.armed = True
            node.journal.append({"event": "ARM", "at": node.clock.epoch()})
            return {"armed": True, "status": {"armed": True}}

        def repoint(_arguments: tuple[str, ...]) -> dict[str, Any]:
            node.repointed = True
            node.delivered = {}
            node.next_scrape = node.clock.now + timedelta(seconds=node.phase)
            return {
                "replay_endpoint": "http://127.0.0.1:9412/metrics",
                "process_environment": {
                    "GPU_FAULT_METRICS_INTERVAL_SECONDS": str(node.interval)
                },
            }

        def advance(_arguments: tuple[str, ...]) -> dict[str, Any]:
            node.tick(node.clock.now)
            assert node.plan is not None and node.plan["steps"][node.step].get(
                "until_advance"
            ), "an advance leaves a held step"
            node.step += 1
            node.reads_in_step = 0
            node.journal.append(
                {"event": "ADVANCE", "read": node.reads, "at": node.clock.epoch()}
            )
            return {"advanced": True}

        def audit(arguments: tuple[str, ...]) -> dict[str, Any]:
            node.tick(node.clock.now)
            offset = int(arguments[arguments.index("--offset") + 1])
            chunk = node.journal[offset:]
            return {
                "records": list(chunk),
                "next_offset": offset + len(chunk),
                "total": len(node.journal),
            }

        def close(_arguments: tuple[str, ...]) -> dict[str, Any]:
            node.disarmed = True
            node.next_scrape = None
            return {
                "clean": True,
                "leftover": [],
                "errors": [],
                "disarm": {"disarmed": True},
                "restore": {"dropin_removed": True},
                "stop": {"run_dir_removed": True},
            }

        script: dict[str, Any] = {
            "survey": self.survey,
            "install-source": install,
            "deploy-proxy": {"status": {"armed": False}},
            "arm": arm,
            "repoint-collector": repoint,
            "advance": advance,
            "audit-read": audit,
            "close-run": close,
        }
        script.update(extra)
        return self.guarded(script)

    # -- the kernel collector and the kmsg fixture ----------------------- #
    def kernel(self) -> FakeProbe:
        node = self

        def install(arguments: tuple[str, ...]) -> dict[str, Any]:
            node.records = _decoded(arguments)
            return {"records": len(node.records)}

        def arm(arguments: tuple[str, ...]) -> dict[str, Any]:
            index = int(arguments[arguments.index("--index") + 1])
            record = node.records[index]
            at = node.clock.now
            body = {
                "cluster_id": CLUSTER,
                "node_id": NODE,
                "record_id": f"kmsg-{BOOT}-{900_000_064 + index}",
                "observed_at": at.isoformat(),
                "source_boot_id": BOOT,
                "message": record["message"],
                "product": "NVIDIA H200",
                "driver_branch": 580,
                "cuda_version": "13.0",
                "runtime_profile_version": "simulated-v1",
            }
            node.kernel_posts.append(body)
            node.product.post(NVIDIA_KERNEL_PATH, body)
            return {
                "armed": True,
                "indexes": [index],
                "written_at": at.timestamp(),
                "lines_written": int(record.get("emissions", 1)),
            }

        return self.guarded(
            {
                "install-records": install,
                "open-window": {"after": {"ActiveState": "active"}},
                "arm": arm,
                "status": {"injected": {"mode": "kmsg", "fixture_lines": 2}},
                "close-window": {"clean": True},
            }
        )


@contextlib.contextmanager
def without_twin_rule() -> Iterator[None]:
    """The product before batch 237: the change is decided as a fault of its
    own -- no twin follow in the engine, no kernel-side join, no incident link
    of the second report."""

    with (
        mock.patch("gpu_fault.policy.engine.follow_report_twin", lambda *a, **k: None),
        mock.patch(
            "gpu_fault.app.ingest.faults.join_gauge_twin",
            lambda event, decision, *a, **k: decision,
        ),
        mock.patch(
            "gpu_fault.orchestration.provider_correlation.report_twin_incident",
            lambda *a, **k: None,
        ),
    ):
        yield


__all__ = [
    "BOOT",
    "RestartingAdapter",
    "CLUSTER",
    "EIGHT_GPUS",
    "EPOCH_START",
    "G0",
    "G1",
    "NODE",
    "START",
    "TwinNode",
    "TwinProduct",
    "VirtualClock",
    "without_twin_rule",
]

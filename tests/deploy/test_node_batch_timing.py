"""Deterministic admission timing model; no subprocess, API or wall-clock timing."""

from __future__ import annotations

import json
from concurrent.futures import Future
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Iterable

import pytest

from gpu_fault.admin.api_budget import START_INTERVAL
from gpu_fault.admin.deploy_limits import DEPLOY_CONCURRENCY
from gpu_fault.node_installer_rendering import InstallerIdentity
from gpu_fault_release import regional_node_batch as batch
from tests.deploy import test_node_batch_rendering

rendered_batch = test_node_batch_rendering.rendered_batch

RenderedBatch = tuple[Path, Path, Path, batch.NodeScope, InstallerIdentity]


@dataclass
class ScheduledCall:
    finishes: float
    error: BaseException | None


class TimingTransport:
    """Paced CLI starts, then startup and serial per-object RTT for each List."""

    def __init__(self, startup: float, rtt: float) -> None:
        self.startup = startup
        self.rtt = rtt
        self.now = 0.0
        self.next_start = 0.0
        self.call_seconds = 0.0
        self.pending: dict[Future[None], ScheduledCall] = {}
        self.admission_finishes: list[float] = []
        self.host_starts: list[float] = []
        self.batch_sizes: list[int] = []
        self.maximum = 0

    def executor(self, *, max_workers: int, thread_name_prefix: str) -> TimingPool:
        assert thread_name_prefix, "the modeled worker pool needs an identity"
        return TimingPool(self, max_workers)

    def wait(
        self, active: Iterable[Future[None]], *, return_when: str
    ) -> tuple[set[Future[None]], set[Future[None]]]:
        assert return_when == "FIRST_COMPLETED", "unexpected admission scheduling"
        selected = set(active)
        self.now = min(self.pending[future].finishes for future in selected)
        done = {
            future for future in selected if self.pending[future].finishes <= self.now
        }
        for future in done:
            call = self.pending.pop(future)
            if call.error is None:
                future.set_result(None)
            else:
                future.set_exception(call.error)
        return done, selected - done

    def run(
        self,
        args: list[str],
        *,
        capture: bool = False,
        input_text: str | None = None,
        timeout_seconds: float | None = None,
    ) -> str:
        if "--dry-run=server" in args:
            assert input_text is not None, "admission omitted its List"
            document = json.loads(input_text)
            assert document["kind"] == "List", "transport expects per-object Lists"
            size = len(document["items"])
            duration = self.startup + self.rtt * size
            starts = max(self.now, self.next_start)
            self.next_start = starts + START_INTERVAL["kubectl"]
            duration += starts - self.now
            assert timeout_seconds is not None and duration < timeout_seconds, (
                "the timing scenario must fit the real command budget"
            )
            self.call_seconds += duration
            self.admission_finishes.append(self.now + self.call_seconds)
            self.batch_sizes.append(size)
        elif "create" in args:
            self.host_starts.append(self.now)
        return ""

    def probe_output(
        self, args: list[str], *, timeout_seconds: float | None = None
    ) -> tuple[int, str, str]:
        return 0, "", ""


class TimingPool:
    def __init__(self, transport: TimingTransport, workers: int) -> None:
        self.transport = transport
        self.workers = workers

    def __enter__(self) -> TimingPool:
        return self

    def __exit__(self, *_args: object) -> None:
        while self.transport.pending:
            self.transport.wait(self.transport.pending, return_when="FIRST_COMPLETED")

    def submit(self, action: Callable[..., None], *args: object) -> Future[None]:
        transport = self.transport
        assert len(transport.pending) < self.workers, (
            "scheduler queued more calls than its bounded worker window"
        )
        future: Future[None] = Future()
        transport.call_seconds = 0.0
        error: BaseException | None = None
        try:
            action(*args)
        except Exception as exc:
            error = exc
        transport.pending[future] = ScheduledCall(
            transport.now + transport.call_seconds, error
        )
        if transport.call_seconds:
            transport.maximum = max(transport.maximum, len(transport.pending))
        return future


@pytest.mark.parametrize("rendered_batch", [8, 32, 128], indirect=True)
@pytest.mark.parametrize("startup,rtt", [(0.100, 0.020), (0.020, 0.100)])
@pytest.mark.parametrize("admission_workers", [2, 8])
def test_admission_startup_and_serial_rtt_model(
    rendered_batch: RenderedBatch,
    monkeypatch: pytest.MonkeyPatch,
    startup: float,
    rtt: float,
    admission_workers: int,
) -> None:
    source, template, output, scope, identity = rendered_batch
    nodes = batch.prepare_node_batch(source, template, output, scope, identity)
    transport = TimingTransport(startup, rtt)
    monkeypatch.setattr(batch, "ThreadPoolExecutor", transport.executor)
    monkeypatch.setattr(batch, "wait", transport.wait)
    monkeypatch.setattr(
        batch,
        "DEPLOY_CONCURRENCY",
        replace(DEPLOY_CONCURRENCY, read_only_checks=admission_workers),
    )

    batch.run_node_preflights(nodes, scope, identity, transport, workers=1)

    assert sum(transport.batch_sizes) == 2 * len(nodes), (
        "the model skipped an installer or host-preflight admission"
    )
    assert 1 <= transport.maximum <= 8, "admission exceeded its API worker ceiling"
    assert min(transport.host_starts) >= max(transport.admission_finishes), (
        "host preflight preceded the last successful admission response"
    )
    print(
        json.dumps(
            {
                "timing": "deterministic-model-not-live",
                "policy": (
                    "previous-two-cli-reference"
                    if admission_workers == 2
                    else "current-eight-cli"
                ),
                "nodes": len(nodes),
                "startup_ms": startup * 1000,
                "serial_object_rtt_ms": rtt * 1000,
                "global_cli_start_interval_ms": START_INTERVAL["kubectl"] * 1000,
                "cli_calls": len(transport.batch_sizes),
                "max_inflight_cli": transport.maximum,
                "admission_seconds": round(max(transport.admission_finishes), 6),
            },
            sort_keys=True,
        )
    )

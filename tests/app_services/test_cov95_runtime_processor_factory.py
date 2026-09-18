from __future__ import annotations

from types import SimpleNamespace

import pytest

from gpu_fault.app import processor_factory
from gpu_fault.store import InMemoryStore
from tests.app_services._cov95_runtime_workers import ThreadRecorder
from tests.regional._cov95_runtime_support import offline_runtime as offline_runtime


def context() -> SimpleNamespace:
    return SimpleNamespace(
        store=InMemoryStore(),
        execution_token="unit-execution-" + "e" * 32,
        processor_replay_secret="unit-replay-" + "r" * 32,
    )


@pytest.mark.parametrize(
    ("defect", "message"),
    [
        ("mode", "direct or active-active"),
        ("owner", "requires POD_UID"),
        ("execution-token", "requires the execution token"),
        ("replay-token", "at least 32"),
        ("shared-token", "must differ"),
        ("remote-url", "loopback"),
        ("negative-grace", "cannot be negative"),
    ],
)
def test_processor_identity_and_replay_config_fail_closed(
    monkeypatch: pytest.MonkeyPatch, defect: str, message: str
) -> None:
    ctx = context()
    monkeypatch.setenv("POD_UID", "unit-pod")
    if defect == "owner":
        monkeypatch.delenv("POD_UID")
    elif defect == "execution-token":
        ctx.execution_token = None
    elif defect == "replay-token":
        ctx.processor_replay_secret = "short"
    elif defect == "shared-token":
        ctx.processor_replay_secret = ctx.execution_token
    elif defect == "remote-url":
        monkeypatch.setenv("GPU_FAULT_PROCESSOR_LOCAL_URL", "https://control.invalid")
    factory = processor_factory.ProcessorFactory(
        ctx,
        mode="active-standby" if defect == "mode" else "active-active",
        exit_grace_seconds=-1 if defect == "negative-grace" else 0,
    )
    with pytest.raises((ValueError, RuntimeError), match=message):
        factory.build()
    assert factory.processor is None


def test_direct_processor_mode_creates_no_queue_consumer() -> None:
    factory = processor_factory.ProcessorFactory(
        context(), mode="direct", exit_grace_seconds=0
    )
    assert factory.build() is None
    assert factory.processor is None


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("grace", [0.0, 0.25])
def test_unhealthy_callback_abandons_owned_work_before_recorded_process_exit(
    monkeypatch: pytest.MonkeyPatch, enabled: bool, grace: float
) -> None:
    monkeypatch.setenv("POD_UID", "unit-pod")
    monkeypatch.setenv("GPU_FAULT_PROCESSOR_EXIT_ON_DEADLINE", str(enabled).lower())
    recorder = ThreadRecorder()
    events = []
    monkeypatch.setattr(processor_factory, "Thread", recorder)
    monkeypatch.setattr(
        processor_factory.time, "sleep", lambda delay: events.append(("sleep", delay))
    )
    monkeypatch.setattr(
        processor_factory.os, "_exit", lambda code: events.append(("exit", code))
    )
    factory = processor_factory.ProcessorFactory(
        context(), mode="active-active", exit_grace_seconds=grace
    )
    processor = factory.build()
    assert processor is not None
    real_abandon = processor.abandon_in_flight

    def abandon() -> dict[str, int]:
        events.append(("abandon", None))
        return real_abandon()

    monkeypatch.setattr(processor, "abandon_in_flight", abandon)
    processor.on_unhealthy("synthetic repeated deadline")
    assert len(recorder.threads) == int(enabled)
    for thread in recorder.threads:
        assert thread.name == "gpu-fault-processor-fatal-exit"
        thread.run()
    assert events == (
        ([("sleep", grace)] if grace else []) + [("abandon", None), ("exit", 70)]
        if enabled
        else []
    )

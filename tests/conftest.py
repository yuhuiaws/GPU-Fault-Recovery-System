from __future__ import annotations

from datetime import datetime, timezone

import pytest

from gpu_fault.app import ApplicationContext
from gpu_fault.models import (
    AllocationEntry,
    Environment,
    RankExitStatus,
    TerminalEvent,
    TerminalStatus,
)
from tests._builders import build_context, build_store


@pytest.fixture(autouse=True)
def processor_replay_secret(monkeypatch) -> None:
    for name in ("GPU_FAULT_DOC_IMPACT", "GPU_FAULT_DOC_IMPACT_REASON"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(
        "GPU_FAULT_PROCESSOR_REPLAY_SECRET", "test-processor-replay-secret-" + "r" * 32
    )


@pytest.fixture
def context() -> ApplicationContext:
    return build_context()


@pytest.fixture
def memory_store():
    return build_store()


@pytest.fixture
def ended_at() -> datetime:
    return datetime(2026, 7, 19, 22, 0, tzinfo=timezone.utc)


@pytest.fixture
def failed_event(ended_at: datetime) -> TerminalEvent:
    return TerminalEvent(
        cluster_id="cluster-a",
        environment=Environment.KUBERNETES,
        job_id="train-123",
        attempt_id="train-123-a1",
        terminal_status=TerminalStatus.FAILED,
        ended_at=ended_at,
        rank_exit_status=[
            RankExitStatus(rank=0, exit_code=1, node_id="node-a", finished_at=ended_at)
        ],
        allocation=[
            AllocationEntry(
                node_id="node-a",
                instance_id="i-a",
                rank=0,
                gpu_uuids=["GPU-a"],
                fabric_partition="fabric-a",
            ),
            AllocationEntry(
                node_id="node-b",
                instance_id="i-b",
                rank=1,
                gpu_uuids=["GPU-b"],
                fabric_partition="fabric-a",
            ),
        ],
        checkpoint_manifest_ref="s3://bucket/checkpoint.json",
        runtime_profile_version="simulated-v1",
    )

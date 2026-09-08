from __future__ import annotations

import os
import random
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
from tests._cluster_binary_guard import MARKER
from tests._cluster_binary_guard import install as install_binary_guard

SHUFFLE_SEED_VARIABLE = "GPU_FAULT_TEST_SHUFFLE_SEED"


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        f"{MARKER}(*names): permit this test to execute the named cluster binaries.",
    )


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Shuffle the run order when a seed is exported, to expose order coupling.

    CI 一直按同一个顺序跑：分片固定、``--dist worksteal`` 的分发也总是从同一个
    收集顺序出发。一条用例如果偷偷依赖别人先跑（模块级缓存、``lru_cache``、
    进程内单例、被别人先 monkeypatch 过的全局量），在这个顺序下永远是绿的，
    换个顺序才炸——而换顺序的那天通常是加了一条不相关的用例。

    ``GPU_FAULT_TEST_SHUFFLE_SEED`` 没设时完全不动顺序，所以既有的分片、
    ``--durations`` 基线和 coverage 证据都不受影响；设了才打乱。

    两个刻意的设计：

    * 先按 ``nodeid`` 排序再打乱，所以最终顺序只是 ``(种子, 用例集合)`` 的函数，
      跟 pytest 自己那套依赖文件系统遍历的收集顺序无关。给出种子就能复现。
    * 种子只认整数，不认 ``random`` 之类的占位符。xdist 的每个 worker 都会各自
      跑一遍这个钩子，要是让它们各自摇一个种子，收集结果就对不上，xdist 会直接
      以 "Different tests were collected" 中止。种子由调用方（``make
      test-shuffled``）摇好再传进来，worker 继承同一个环境变量，顺序才一致。
    """
    raw = os.environ.get(SHUFFLE_SEED_VARIABLE, "").strip()
    if not raw:
        return
    try:
        seed = int(raw)
    except ValueError:
        raise pytest.UsageError(
            f"{SHUFFLE_SEED_VARIABLE} must be an integer seed, got {raw!r}"
        ) from None
    items.sort(key=lambda item: item.nodeid)
    random.Random(seed).shuffle(items)


def pytest_report_header() -> list[str]:
    """Put the seed in the header, so a red CI run can be replayed verbatim.

    读环境变量而不是收集钩子里存的值：``pytest_report_header`` 在收集之前就调用，
    这时候还没有 items 可打乱。种子非法的情况留给上面的钩子报 ``UsageError``。
    """
    raw = os.environ.get(SHUFFLE_SEED_VARIABLE, "").strip()
    if not raw:
        return []
    return [f"test order shuffled: {SHUFFLE_SEED_VARIABLE}={raw}"]


@pytest.fixture(autouse=True)
def block_cluster_binaries(request: pytest.FixtureRequest, monkeypatch) -> None:
    """Fail instead of letting a test reach kubectl, aws, ssh and friends."""
    allowed: set[str] = set()
    for mark in request.node.iter_markers(MARKER):
        allowed.update(mark.args)
    install_binary_guard(monkeypatch, frozenset(allowed))


@pytest.fixture(autouse=True)
def processor_replay_secret(monkeypatch) -> None:
    for name in ("GPU_FAULT_DOC_IMPACT", "GPU_FAULT_DOC_IMPACT_REASON"):
        monkeypatch.delenv(name, raising=False)
    # Tests that set POD_UID would otherwise share counters through the
    # machine's /dev/shm and sum with other live test processes; a test that
    # wants the sharing points the variable at its own tmp_path.
    monkeypatch.setenv("GPU_FAULT_PROCESS_COUNTERS_DIR", "off")
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

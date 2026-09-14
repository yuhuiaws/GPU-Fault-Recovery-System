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
    _isolate_postgres_per_worker()


def _isolate_postgres_per_worker() -> None:
    """Point this xdist worker at its own database before any module imports.

    The Postgres-gated modules read the URL at import and truncate every table
    per test; sharing one database across workers is a race, so the shard ran
    serially. See ``tests/_postgres_worker_database``.
    """

    worker = os.environ.get("PYTEST_XDIST_WORKER", "")
    base_url = os.environ.get("GPU_FAULT_TEST_POSTGRES_URL", "").strip()
    if not worker or not base_url:
        return
    from tests._postgres_worker_database import worker_database_url

    os.environ["GPU_FAULT_TEST_POSTGRES_URL"] = worker_database_url(base_url, worker)


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
def kubeconfig_is_never_ambient(monkeypatch) -> None:
    """A test points the release engine at a kubeconfig it wrote, never one it
    inherits. The deploy runs this suite as a release gate with ``KUBECONFIG``
    set to the live site kubeconfig, whose exec credential is ``aws eks
    get-token``; ``ReleaseKubeconfigCache`` pre-fetches a token for every entry
    in ``KUBECONFIG`` on ``__enter__``. An ambient value made four
    kubeconfig-cache and four rollout tests plus the rollback-marker test exec
    real ``aws`` and trip ``block_cluster_binaries`` in the gate while passing on
    a developer box that has no ``KUBECONFIG``. Clearing it here forces each
    test to export ``KUBECONFIG`` on purpose."""
    monkeypatch.delenv("KUBECONFIG", raising=False)


@pytest.fixture(autouse=True)
def deploy_consent_is_never_ambient(monkeypatch) -> None:
    """The deploy carries operator consent down its process chain as
    environment variables, and it runs this suite as a release gate inside
    that chain: a `deploy --supersede-failed-transaction` on 2026-09-11 handed
    every test `GPU_FAULT_RELEASE_SUPERSEDE_FAILED_TRANSACTION=1`, and six
    resume tests took the supersede branch and failed the gate. Consent is an
    input a test sets on purpose, never something it inherits.

    The signing password rides the same chain: on 2026-09-12 the gate's
    ambient ``COSIGN_PASSWORD`` made ``ensure_signing_material`` skip its
    generation branch, failed three tests, and the assertion diff printed the
    live password into the deploy log."""

    for name in (
        "GPU_FAULT_RELEASE_SUPERSEDE_FAILED_TRANSACTION",
        "GPU_FAULT_RELEASE_ACCEPT_SCHEMA_CHANGE",
        "GPU_FAULT_EXPECTED_RELEASE_STATE_SHA256",
        "COSIGN_PASSWORD",
    ):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture(autouse=True)
def processor_replay_secret(monkeypatch) -> None:
    for name in ("GPU_FAULT_DOC_IMPACT", "GPU_FAULT_DOC_IMPACT_REASON"):
        monkeypatch.delenv(name, raising=False)
    # Tests that set POD_UID would otherwise share /metrics samples through
    # the machine's /dev/shm and merge with other live test processes; a test
    # that wants the sharing points the variable at its own tmp_path.
    monkeypatch.setenv("GPU_FAULT_PROCESS_METRICS_DIR", "off")
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

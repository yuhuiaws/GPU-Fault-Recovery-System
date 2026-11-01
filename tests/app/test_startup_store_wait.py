"""One bounded start-up wait for every store step before a process serves.

Why (GF-REGIONAL-HA-010): an api-ha Pod created two seconds into an Aurora
writer failover crashed thirteen seconds after its container started --
``PoolTimeout`` from ``PostgresStore``'s schema check, the first store round
trip of ``ApplicationContext.from_environment`` -- and restarted once, because
only the regional registry bootstrap, which runs later, had a retry budget. The wait now
covers the store and its schema check, the context's registry sync and
registered-cluster reads, and the registry bootstrap, all inside one
``GPU_FAULT_STARTUP_STORE_RETRY_SECONDS`` budget, and only for an outage.

Every test drives a fake clock: ``sleep`` advances it, and a failed fake
checkout advances it by the 10 s pool timeout the live site runs with, so a
two-minute outage costs milliseconds.
"""

from __future__ import annotations

import logging
import socket
from typing import Any

import pytest

import gpu_fault.app.context as context_module
from gpu_fault.app import ApplicationContext, create_app
from gpu_fault.app.startup_store_wait import (
    AUTHENTICATION,
    CONTENTION,
    CONTEXT_STEP,
    REGISTRY_BOOTSTRAP_STEP,
    STARTUP_STORE_RETRY_ENV,
    STORE_STEP,
    TIMEOUT,
    UNAVAILABLE,
    StartupStoreUnavailable,
    StartupStoreWait,
    startup_store_budget_seconds,
    startup_store_failure_kind,
)
from gpu_fault.env_validation import validate_gpu_fault_environment
from gpu_fault.regional_registry_runtime import (
    STARTUP_RETRY_INITIAL_SECONDS,
    STARTUP_RETRY_MAX_INTERVAL_SECONDS,
)
from gpu_fault.store import InMemoryStore, NotFoundError
from tests.regional._regional_support import regional_environment

psycopg = pytest.importorskip("psycopg")
psycopg_errors = pytest.importorskip("psycopg.errors")
psycopg_pool = pytest.importorskip("psycopg_pool")

POOL_TIMEOUT_SECONDS = 10.0
WAIT_LOGGER = "gpu_fault.app.startup_store_wait"
# A DSN password that must never reach a log line or an error message.
DSN_PASSWORD = "startup-wait-dsn-password-7c1f"
STORE_URL = f"postgresql://gpu-fault:{DSN_PASSWORD}@aurora.invalid:5432/gpu_fault"
# Every store method the three roles' start-up calls before serving.
START_UP_STORE_METHODS = {
    "get_regional_registry_head",
    "list_regional_cluster_ids",
    "save_regional_cluster",
    "list_regional_clusters",
    "publish_regional_registry_revision",
    "get_regional_registry_revision",
    "save_regional_registry_member",
}


class FakeClock:
    def __init__(self) -> None:
        self.now = 1_000.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _wait(budget: float, clock: FakeClock) -> StartupStoreWait:
    return StartupStoreWait(budget, sleep=clock.sleep, clock=clock)


def _wait_lines(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [
        record.getMessage() for record in caplog.records if record.name == WAIT_LOGGER
    ]


def _pool_timeout(clock: FakeClock) -> Exception:
    """The pool's answer to a checkout during a failover, after its timeout."""

    clock.advance(POOL_TIMEOUT_SECONDS)
    return psycopg_pool.PoolTimeout("couldn't get a connection after 10.00 sec")


def _pg(sqlstate: str, message: str = "synthetic") -> Exception:
    return psycopg_errors.lookup(sqlstate)(f"{message} {sqlstate}")


class OutageStore:
    """An in-memory store each of whose methods meets the outage once.

    The first call of every store method (or of each one in ``fail``) raises
    the pool's ``PoolTimeout`` and costs the 10 s checkout timeout; later calls
    go through. A start-up path that reaches any store method outside the wait
    crashes the test.
    """

    def __init__(self, clock: FakeClock, fail: set[str] | None = None) -> None:
        self.inner = InMemoryStore()
        self.clock = clock
        self.fail = fail
        self.failed: list[str] = []
        self.closed = 0

    def close(self) -> None:
        self.closed += 1

    def __getattr__(self, name: str) -> Any:
        value = getattr(self.inner, name)
        if name.startswith("_") or not callable(value):
            return value

        def call(*args: Any, **kwargs: Any) -> Any:
            if name not in self.failed and (self.fail is None or name in self.fail):
                self.failed.append(name)
                raise _pool_timeout(self.clock)
            return value(*args, **kwargs)

        return call


@pytest.mark.parametrize(
    ("make_error", "kind"),
    [
        (lambda: psycopg_pool.PoolTimeout("couldn't get a connection"), UNAVAILABLE),
        (
            lambda: psycopg.OperationalError(
                "connection failed: server closed the connection unexpectedly"
            ),
            UNAVAILABLE,
        ),
        (lambda: _pg("57P03", "the database system is starting up"), UNAVAILABLE),
        (lambda: _pg("57P01", "terminating connection"), UNAVAILABLE),
        (lambda: _pg("08006", "connection failure"), UNAVAILABLE),
        (lambda: _pg("25006", "read-only transaction"), UNAVAILABLE),
        (lambda: _pg("53300", "too many connections"), UNAVAILABLE),
        (lambda: ConnectionRefusedError("refused"), UNAVAILABLE),
        (lambda: psycopg_errors.ConnectionTimeout("timeout expired"), TIMEOUT),
        (lambda: _pg("57014", "canceling statement due to statement timeout"), TIMEOUT),
        (lambda: _pg("55P03", "lock timeout"), TIMEOUT),
        (lambda: TimeoutError("timed out"), TIMEOUT),
        (lambda: _pg("40001", "could not serialize access"), CONTENTION),
        (lambda: _pg("40P01", "deadlock detected"), CONTENTION),
        (lambda: _pg("28P01", "password authentication failed"), AUTHENTICATION),
        (
            lambda: psycopg.OperationalError(
                'connection failed: FATAL:  password authentication failed for user "g"'
            ),
            AUTHENTICATION,
        ),
    ],
)
def test_an_outage_is_waited_for(make_error, kind) -> None:
    assert startup_store_failure_kind(make_error()) == kind


@pytest.mark.parametrize(
    "make_error",
    [
        lambda: RuntimeError(
            "PostgreSQL schema version mismatch: expected 23, got 22; "
            "run gpu-fault-store-migrate --ensure-schema"
        ),
        lambda: RuntimeError("regional registry head digest mismatch"),
        lambda: ValueError("regional registry head validation failed"),
        lambda: NotFoundError("regional_registry_revision:7"),
        lambda: _pg("42P01", "relation does not exist"),
        lambda: _pg("23505", "duplicate key value"),
        lambda: psycopg_pool.PoolClosed("the pool 'pool-1' is already closed"),
        lambda: StartupStoreUnavailable(
            "spent", step=STORE_STEP, attempts=9, kind=UNAVAILABLE
        ),
    ],
)
def test_anything_else_fails_start_up_at_once(make_error) -> None:
    assert startup_store_failure_kind(make_error()) is None


def test_an_outage_wrapped_by_the_caller_is_still_one() -> None:
    try:
        try:
            raise psycopg_pool.PoolTimeout("couldn't get a connection")
        except psycopg_pool.PoolTimeout as exc:
            raise RuntimeError("store probe failed") from exc
    except RuntimeError as wrapped:
        assert startup_store_failure_kind(wrapped) == UNAVAILABLE, "cause chain"


def test_the_budget_is_read_from_the_environment_and_validated(monkeypatch) -> None:
    assert startup_store_budget_seconds({}) == 120.0
    assert startup_store_budget_seconds({STARTUP_STORE_RETRY_ENV: "0"}) == 0.0
    assert startup_store_budget_seconds({STARTUP_STORE_RETRY_ENV: "45.5"}) == 45.5
    for invalid in ("-1", "nan", "inf"):
        with pytest.raises(ValueError, match=STARTUP_STORE_RETRY_ENV):
            startup_store_budget_seconds({STARTUP_STORE_RETRY_ENV: invalid})
    # A value that is no number at all is refused earlier, by name, by the
    # process's environment validation (the inventory records it as a number).
    monkeypatch.setenv(STARTUP_STORE_RETRY_ENV, "soon")
    with pytest.raises(
        RuntimeError, match=f"{STARTUP_STORE_RETRY_ENV} must be a number"
    ):
        validate_gpu_fault_environment(process_name="gpu-fault-api")


def test_a_store_outage_is_waited_out_with_backoff_and_one_success(caplog) -> None:
    clock = FakeClock()
    attempts: list[float] = []

    def check_schema() -> str:
        attempts.append(clock.now)
        if len(attempts) <= 3:
            raise _pool_timeout(clock)
        return "store"

    with caplog.at_level(logging.WARNING, logger=WAIT_LOGGER):
        result = _wait(120, clock).run(STORE_STEP, check_schema)

    assert result == "store"
    assert len(attempts) == 4, "three outages, then exactly one success"
    assert clock.sleeps == [
        STARTUP_RETRY_INITIAL_SECONDS,
        STARTUP_RETRY_INITIAL_SECONDS * 2,
        STARTUP_RETRY_INITIAL_SECONDS * 4,
    ]
    lines = _wait_lines(caplog)
    assert len(lines) == 3, lines
    first = lines[0]
    assert first.startswith(f"start-up store wait: {STORE_STEP} attempt 1 failed "), (
        first
    )
    assert "writer unavailable, PoolTimeout: couldn't get a connection" in first
    assert "retrying in 0.5s with 110s of the 120s" in first
    assert STARTUP_STORE_RETRY_ENV in first
    assert " attempt 3 failed " in lines[2]


def test_the_backoff_reaches_its_cap_and_the_budget_ends_in_one_clear_error() -> None:
    clock = FakeClock()
    started = clock.now
    attempts = []

    def check_schema() -> None:
        attempts.append(clock.now)
        raise _pool_timeout(clock)

    with pytest.raises(StartupStoreUnavailable) as raised:
        _wait(120, clock).run(STORE_STEP, check_schema)

    error = raised.value
    assert isinstance(error.__cause__, psycopg_pool.PoolTimeout), "the last error"
    assert (error.step, error.kind, error.attempts) == (
        STORE_STEP,
        UNAVAILABLE,
        len(attempts),
    )
    # Doubling to the cap; the last sleep is cut to what is left of the budget.
    assert clock.sleeps == [0.5, 1.0, 2.0, 4.0, 8.0, 10.0, 10.0, 4.5]
    assert max(clock.sleeps) == STARTUP_RETRY_MAX_INTERVAL_SECONDS
    # The last attempt starts at the deadline at the latest; only its own
    # checkout timeout runs past it.
    assert clock.now - started <= 120 + POOL_TIMEOUT_SECONDS
    assert attempts[-1] - started <= 120
    message = str(error)
    assert message.startswith("start-up gave up waiting for the store after "), message
    assert f"{STORE_STEP} failed {len(attempts)} time(s)" in message
    assert "writer unavailable (PoolTimeout)" in message
    assert f"120s {STARTUP_STORE_RETRY_ENV} budget is spent" in message


def test_a_non_transient_error_fails_at_once() -> None:
    clock = FakeClock()
    attempts = []

    def check_schema() -> None:
        attempts.append(clock.now)
        raise RuntimeError(
            "PostgreSQL schema version mismatch: expected 23, got 22; "
            "run gpu-fault-store-migrate --ensure-schema"
        )

    with pytest.raises(RuntimeError, match="schema version mismatch"):
        _wait(120, clock).run(STORE_STEP, check_schema)
    assert (len(attempts), clock.sleeps) == (1, [])


def test_a_zero_budget_keeps_the_first_failure() -> None:
    clock = FakeClock()

    def check_schema() -> None:
        raise _pool_timeout(clock)

    with pytest.raises(psycopg_pool.PoolTimeout):
        _wait(0, clock).run(STORE_STEP, check_schema)
    assert clock.sleeps == []


def test_the_steps_share_one_budget_not_one_each() -> None:
    clock = FakeClock()
    wait = _wait(120, clock)
    outage_ends = clock.now + 95

    def step() -> str:
        if clock.now < outage_ends:
            raise _pool_timeout(clock)
        return "up"

    assert wait.run(STORE_STEP, step) == "up"
    spent = clock.now - wait.started_at
    assert 95 <= spent < 120
    remaining = wait.remaining_seconds()

    def bootstrap() -> None:
        raise _pool_timeout(clock)

    before = clock.now
    with pytest.raises(StartupStoreUnavailable) as raised:
        wait.run(REGISTRY_BOOTSTRAP_STEP, bootstrap)
    assert raised.value.step == REGISTRY_BOOTSTRAP_STEP
    assert clock.now - before <= remaining + POOL_TIMEOUT_SECONDS, (
        "the bootstrap only gets what the earlier steps left"
    )


def test_a_retry_line_never_carries_a_dsn(caplog) -> None:
    clock = FakeClock()
    failures = [
        psycopg.OperationalError(f"connection to {STORE_URL} failed: refused"),
        psycopg.OperationalError(
            f"host=aurora.invalid password={DSN_PASSWORD} refused"
        ),
    ]

    def connect() -> str:
        if failures:
            raise failures.pop(0)
        return "connected"

    with caplog.at_level(logging.WARNING, logger=WAIT_LOGGER):
        assert _wait(120, clock).run(STORE_STEP, connect) == "connected"
    assert len(_wait_lines(caplog)) == 2
    assert DSN_PASSWORD not in caplog.text


# --------------------------------------------------------------------------- #
# The context: the step GF-REGIONAL-HA-010 crashed in
# --------------------------------------------------------------------------- #
def _postgres_regional_environment(monkeypatch, tmp_path, *, role: str) -> None:
    regional_environment(monkeypatch, tmp_path)
    monkeypatch.setenv("GPU_FAULT_STORE_URL", STORE_URL)
    monkeypatch.setenv("GPU_FAULT_POSTGRES_AUTO_SCHEMA_INIT", "false")
    monkeypatch.setenv("GPU_FAULT_SERVICE_ROLE", role)
    monkeypatch.setenv("GPU_FAULT_PROCESSOR_MODE", "active-active")
    monkeypatch.setenv("POD_UID", f"pod-startup-store-wait-{role}")
    monkeypatch.setenv("GPU_FAULT_ENABLE_AGENT_REGISTRY", "true")
    monkeypatch.setenv("GPU_FAULT_AGENT_REGISTRATION_SECRET", "s" * 32)
    monkeypatch.setenv("GPU_FAULT_REQUIRED_AGENT_ARTIFACT_SHA256", "a" * 64)
    monkeypatch.setenv("GPU_FAULT_REQUIRED_AGENT_CONFIG_DIGEST", "c" * 64)
    # The three live roles: ingress and control-worker observe managed
    # recovery, the spool worker consumes the spool and observes nothing.
    monkeypatch.setenv(
        "GPU_FAULT_ENABLE_HYPERPOD_MANAGED_OBSERVER",
        "false" if role == "spool-worker" else "true",
    )
    monkeypatch.setenv(
        "GPU_FAULT_TELEMETRY_SPOOL", "true" if role == "spool-worker" else "false"
    )


def _pool_opener(monkeypatch, clock: FakeClock, store: Any, *, outages: int):
    """``PostgresStore`` whose schema check meets ``outages`` pool timeouts."""

    opened: list[dict[str, Any]] = []

    def open_store(url: str, **kwargs: Any) -> Any:
        assert url == STORE_URL
        opened.append(kwargs)
        if len(opened) <= outages:
            raise _pool_timeout(clock)
        return store

    monkeypatch.setattr(context_module, "PostgresStore", open_store)
    return opened


def test_the_context_waits_out_a_pool_timeout_in_the_schema_check(
    monkeypatch, tmp_path, caplog
) -> None:
    _postgres_regional_environment(monkeypatch, tmp_path, role="ingress")
    clock = FakeClock()
    store = InMemoryStore()
    opened = _pool_opener(monkeypatch, clock, store, outages=3)
    wait = _wait(120, clock)

    with caplog.at_level(logging.WARNING, logger=WAIT_LOGGER):
        context = ApplicationContext.from_environment(startup_wait=wait)

    assert context.store is store
    assert context.startup_wait is wait, "the factory's bootstrap shares the budget"
    assert len(opened) == 4, "three pool timeouts, then one store"
    assert opened[-1]["initialize_schema"] is False
    assert clock.sleeps == [0.5, 1.0, 2.0]
    assert [line.split(" failed ")[0] for line in _wait_lines(caplog)] == [
        f"start-up store wait: {STORE_STEP} attempt {attempt}" for attempt in (1, 2, 3)
    ]
    assert DSN_PASSWORD not in caplog.text


def test_the_context_gives_up_once_with_one_clear_error(monkeypatch, tmp_path) -> None:
    _postgres_regional_environment(monkeypatch, tmp_path, role="ingress")
    clock = FakeClock()
    opened = _pool_opener(monkeypatch, clock, InMemoryStore(), outages=10_000)

    with pytest.raises(StartupStoreUnavailable) as raised:
        ApplicationContext.from_environment(startup_wait=_wait(120, clock))

    assert raised.value.step == STORE_STEP
    assert raised.value.attempts == len(opened)
    assert 7 <= len(opened) <= 12, len(opened)
    assert DSN_PASSWORD not in str(raised.value)


def test_a_schema_version_mismatch_fails_start_up_at_once(
    monkeypatch, tmp_path
) -> None:
    _postgres_regional_environment(monkeypatch, tmp_path, role="ingress")
    clock = FakeClock()
    calls = []

    def open_store(url: str, **kwargs: Any) -> Any:
        calls.append(url)
        raise RuntimeError(
            "PostgreSQL schema version mismatch: expected 23, got 22; "
            "run gpu-fault-store-migrate --ensure-schema"
        )

    monkeypatch.setattr(context_module, "PostgresStore", open_store)
    with pytest.raises(RuntimeError, match="schema version mismatch"):
        ApplicationContext.from_environment(startup_wait=_wait(120, clock))
    assert (len(calls), clock.sleeps) == (1, [])


def test_an_outage_after_the_store_opened_rebuilds_the_context_on_it(
    monkeypatch, tmp_path
) -> None:
    _postgres_regional_environment(monkeypatch, tmp_path, role="worker")
    clock = FakeClock()
    store = OutageStore(clock)
    opened = _pool_opener(monkeypatch, clock, store, outages=0)

    context = ApplicationContext.from_environment(startup_wait=_wait(120, clock))

    assert context.store is store
    assert len(opened) == 1, "the open store is kept, not reopened"
    assert store.closed == 0
    assert store.failed == [
        "get_regional_registry_head",
        "list_regional_cluster_ids",
        "save_regional_cluster",
        "list_regional_clusters",
    ], "the registry sync and the agent registry's read each met the outage"
    assert [item.cluster_id for item in store.inner.list_regional_clusters()] == [
        "cluster-a"
    ]


def test_a_context_that_cannot_be_built_closes_its_store(monkeypatch, tmp_path) -> None:
    _postgres_regional_environment(monkeypatch, tmp_path, role="worker")
    clock = FakeClock()
    store = OutageStore(clock)
    _pool_opener(monkeypatch, clock, store, outages=0)

    def corrupt_head() -> Any:
        raise ValueError("regional registry head validation failed")

    store.inner.get_regional_registry_head = corrupt_head
    with pytest.raises(ValueError, match="head validation failed"):
        ApplicationContext.from_environment(startup_wait=_wait(120, clock))
    assert store.closed == 1, "a failed start-up leaves no pool behind"


# --------------------------------------------------------------------------- #
# The whole start-up of each role, registry bootstrap included
# --------------------------------------------------------------------------- #
def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


@pytest.mark.parametrize("role", ["ingress", "worker", "spool-worker"])
def test_every_start_up_store_step_of_each_role_waits_out_the_outage(
    monkeypatch, tmp_path, caplog, role: str
) -> None:
    _postgres_regional_environment(monkeypatch, tmp_path, role=role)
    monkeypatch.setenv("GPU_FAULT_LIVENESS_PORT", str(_free_port()))
    clock = FakeClock()
    store = OutageStore(clock)
    opened = _pool_opener(monkeypatch, clock, store, outages=2)
    wait = _wait(300, clock)

    with caplog.at_level(logging.WARNING, logger=WAIT_LOGGER):
        app = create_app(ApplicationContext.from_environment(startup_wait=wait))

    assert set(store.failed) == START_UP_STORE_METHODS, sorted(store.failed)
    assert len(opened) == 3
    runtime = app.state.regional_registry_runtime
    assert runtime is not None and runtime.is_ready(), runtime and runtime.status()
    assert clock.now <= wait.deadline, "the whole start-up fits one deadline"
    steps = {
        line.split(" attempt ")[0].removeprefix("start-up store wait: ")
        for line in _wait_lines(caplog)
    }
    assert steps == {STORE_STEP, CONTEXT_STEP, REGISTRY_BOOTSTRAP_STEP}
    # GF-REGIONAL-HA-010 records this marker as supplementary evidence.
    assert "regional registry bootstrap attempt" in caplog.text
    # Nothing answers the kubelet while the process is still starting: the
    # probe listener opens at the end of the lifespan start-up.
    assert app.state.liveness_listener.bound_port is None
    assert DSN_PASSWORD not in caplog.text


def test_a_context_built_by_the_caller_takes_its_budget_from_the_environment(
    monkeypatch, caplog
) -> None:
    """A context the caller built carries no start-up wait; the factory's
    registry bootstrap starts one from ``GPU_FAULT_STARTUP_STORE_RETRY_SECONDS``."""

    monkeypatch.setenv("GPU_FAULT_SERVICE_ROLE", "ingress")

    def regional_context() -> tuple[ApplicationContext, OutageStore]:
        store = OutageStore(FakeClock(), fail={"get_regional_registry_revision"})
        context = ApplicationContext(store=store, execution_token="t" * 32)
        context.regional_mode = True
        return context, store

    monkeypatch.setenv(STARTUP_STORE_RETRY_ENV, "0")
    context, store = regional_context()
    with pytest.raises(psycopg_pool.PoolTimeout):
        create_app(context)
    assert store.failed == ["get_regional_registry_revision"]

    monkeypatch.delenv(STARTUP_STORE_RETRY_ENV)
    context, store = regional_context()
    with caplog.at_level(logging.WARNING, logger=WAIT_LOGGER):
        app = create_app(context)
    assert app.state.regional_registry_runtime.is_ready(), "one retry, then up"
    (line,) = _wait_lines(caplog)
    assert line.startswith(
        f"start-up store wait: {REGISTRY_BOOTSTRAP_STEP} attempt 1 failed "
    ), line
    assert "of the 120s" in line, "the default budget"

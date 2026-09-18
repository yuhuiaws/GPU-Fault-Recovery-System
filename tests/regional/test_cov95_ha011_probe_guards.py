from __future__ import annotations

import base64
import json
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from threading import Event, Thread
from types import SimpleNamespace

import pytest

from gpu_fault.store import InMemoryStore
from scripts.e2e.regional import ha011_contracts as contracts
from scripts.e2e.regional.probes import ha011_probe as probe
from scripts.e2e.regional.probes import ha011_workers as workers
from tests.regional._cov95_ha011_runtime import Capture, MemoryWorkers
from tests.regional._cov95_ha011_runtime import isolated_fixture as isolated_fixture
from tests.regional._cov95_ha011_support import INTENT, POD_UID, RUN_ID, Clock
from tests.regional._cov95_ha011_support import (
    blocked_external_transports as blocked_external_transports,
)


class Store(InMemoryStore):
    def __init__(self):
        super().__init__()
        self.closes = 0

    def close(self):
        self.closes += 1


class Database:
    def __init__(self):
        self.store = Store()
        self.version = 160010
        self.aurora = (False,)
        self.existing = (False,)
        self.failures = 0
        self.attempts = 0
        self.connection_closes = 0
        self.constructed = []

    def connect(self, url, **kwargs):
        from psycopg import OperationalError
        from psycopg.conninfo import conninfo_to_dict

        parameters = conninfo_to_dict(url)
        assert parameters["host"] == "127.0.0.1", (
            "the probe must not accept a caller-selected database host"
        )
        assert parameters["port"] == "5432" and parameters["dbname"] == "postgres", (
            "the only allowed database is the Pod's private loopback sidecar"
        )
        assert kwargs["autocommit"] is True, (
            "identity guards must not leave an open transaction"
        )
        self.attempts += 1
        if self.failures:
            self.failures -= 1
            raise OperationalError("controlled unavailable private database")
        owner = self

        class Connection:
            info = SimpleNamespace(server_version=owner.version)

            def __init__(self):
                self.calls = 0

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                owner.connection_closes += 1

            def execute(self, _query):
                self.calls += 1
                result = owner.aurora if self.calls == 1 else owner.existing
                return SimpleNamespace(fetchone=lambda: result)

        return Connection()

    def construct(self, _url, *, initialize_schema, pool_max_size):
        self.constructed.append(initialize_schema)
        assert pool_max_size == 4, "the isolated proof must bound its connection pool"
        return self.store


@pytest.fixture
def database(isolated, monkeypatch):
    fake = Database()
    monkeypatch.setattr("psycopg.connect", fake.connect)
    monkeypatch.setattr(probe, "PostgresStore", fake.construct)
    probe.arm(POD_UID, RUN_ID, INTENT)
    return fake


@pytest.mark.parametrize("initialize", [False, True])
def test_private_database_guards_precede_explicit_schema_or_runtime_open(
    database, initialize: bool
) -> None:
    store = probe.private_store(initialize=initialize)
    assert store is database.store, (
        "only the guarded private transport may supply the runtime Store"
    )
    assert database.constructed == [initialize], (
        "only explicit parent setup may request schema initialization"
    )
    assert database.connection_closes == 1, (
        "the identity-check connection must close before runtime opens"
    )


@pytest.mark.parametrize(
    "failure", ["version", "aurora", "existing", "short-password", "malformed-password"]
)
def test_database_refusal_never_initializes_or_replays(database, failure: str) -> None:
    if failure == "version":
        database.version = 150009
    elif failure == "aurora":
        database.aurora = (True,)
    elif failure == "existing":
        database.existing = (True,)
    elif failure == "short-password":
        probe.PASSWORD_FILE.write_text("short", encoding="utf-8")
    else:
        probe.PASSWORD_FILE.write_text(
            "public-fake-password-with-enough-length\nbut-invalid", encoding="utf-8"
        )
    with pytest.raises(contracts.ProofError):
        probe.private_store(initialize=True)
    assert database.constructed == [], (
        "a failed private database guard must prevent Store construction"
    )


def test_private_database_startup_retries_are_bounded_and_do_not_change_target(
    database, monkeypatch
) -> None:
    database.failures = 1
    monkeypatch.setattr(probe, "time", Clock(step=1))
    assert probe.private_store(initialize=True) is database.store, (
        "a bounded sidecar startup retry may recover"
    )
    assert database.attempts == 2, (
        "startup retry must use only the same private transport"
    )
    database.failures = 3
    database.constructed.clear()
    monkeypatch.setattr(probe, "time", Clock(step=91))
    with pytest.raises(contracts.ProofError, match="deadline"):
        probe.private_store(initialize=False)
    assert database.constructed == [], (
        "startup timeout must not construct a runtime Store"
    )


def test_probe_entry_keeps_real_orchestration_with_fake_boundaries(
    database, monkeypatch, capsys
) -> None:
    processes = MemoryWorkers(database.store)
    monkeypatch.setattr(probe, "WorkerProcess", processes)
    assert probe.main() == 0, (
        "the complete guarded algorithm should pass against declared local fakes"
    )
    result = json.loads(capsys.readouterr().out)
    assert (
        contracts.evidence_errors(
            result, isolation_id=RUN_ID, pod_uid=POD_UID, intent_sha256=INTENT
        )
        == []
    ), "both role proofs must retain the exact same deployed evidence schema"
    assert database.constructed == [True, False] and database.store.closes == 2, (
        "setup and runtime Stores must be separately opened and closed"
    )
    assert len(processes.workers) == 6 and all(
        not worker.alive() for worker in processes.workers
    ), "fake orchestration must drain original, contender and replacement lifetimes"


@pytest.mark.parametrize("failure", ["nonempty", "incomplete-proof"])
def test_probe_entry_refuses_nonempty_setup_or_incomplete_worker_proof(
    database, monkeypatch, capsys, failure: str
) -> None:
    if failure == "nonempty":
        database.store.enqueue_processor_request(probe.sample(RUN_ID, "processor", 8))
    else:
        factory = MemoryWorkers(database.store)
        original = factory.__call__

        def missing_cpu(*args, **kwargs):
            worker = original(*args, **kwargs)
            for event in worker.events:
                if event["kind"] == "busy":
                    event["cpu_seconds"] = 0
            return worker

        monkeypatch.setattr(probe, "WorkerProcess", missing_cpu)
    assert probe.main() == 1, "incomplete proof must never be promoted to acceptance"
    assert json.loads(capsys.readouterr().out)["verdict"] == "FAIL", (
        "entrypoint must emit a non-success verdict"
    )
    assert database.store.closes >= 1, "failure must close every opened owned Store"


def test_worker_entry_reports_only_redacted_failure_and_closes_pipe(database) -> None:
    database.version = 150001
    capture = Capture()
    with pytest.raises(SystemExit) as exit_status:
        probe.worker_entry("spool", RUN_ID + "-spool-0", capture, None, None)
    assert exit_status.value.code == 1, "a child guard failure must be a failing exit"
    assert capture.closed and [event["kind"] for event in capture.events] == [
        "error"
    ], "child failure must close IPC and expose no database diagnostics"


def test_spool_admission_must_not_coalesce_or_silently_lose_test_payload() -> None:
    class Rejected:
        def try_spool_telemetry_requests(self, requests, **_kwargs):
            return [(requests[0], "coalesced")]

    with pytest.raises(contracts.ProofError, match="admission"):
        probe.admit(Rejected(), "spool", [probe.sample(RUN_ID, "spool", 0)])


def test_late_processor_callback_requires_typed_fencing_rejection() -> None:
    item = probe.sample(RUN_ID, "processor", 0)
    claim = {"item": item, "owner": "old"}

    class Accepted:
        def complete_active_processor_request(self, *_args, **_kwargs):
            return item

    assert probe.refuse_old_completion(Accepted(), "processor", claim) is False, (
        "a successful late callback must not be reported as fenced"
    )

    class Unavailable:
        def complete_active_processor_request(self, *_args, **_kwargs):
            raise OSError("controlled transport failure")

    with pytest.raises(OSError):
        probe.refuse_old_completion(Unavailable(), "processor", claim)


def test_spool_observer_only_reports_successful_single_item_completion() -> None:
    store = InMemoryStore()
    request = probe.sample(RUN_ID, "spool", 0)
    capture = Capture()
    observed = workers.ObservedStore(store, workers.Emitter(capture))
    probe.admit(store, "spool", [request])
    rows = observed.claim_telemetry_spool(
        "owner",
        now=datetime.now(timezone.utc),
        lease_duration=timedelta(seconds=8),
        limit=1,
    )
    assert observed.complete_telemetry_spool(rows) == 1, (
        "the accepted claim must be observable"
    )
    assert observed.complete_telemetry_spool(rows) == 0, (
        "a duplicate callback must not become a second completion"
    )
    assert sum(event["kind"] == "complete" for event in capture.events) == 1, (
        "only the durable successful callback may appear in backlog evidence"
    )
    with pytest.raises(contracts.ProofError, match="single-item"):
        observed.complete_telemetry_spool([])


def test_empty_claim_observation_is_lane_specific_and_idempotent() -> None:
    capture = Capture()
    observed = workers.ObservedStore(InMemoryStore(), workers.Emitter(capture))
    options = {
        "now": datetime.now(timezone.utc),
        "lease_duration": timedelta(seconds=8),
        "limit": 1,
    }
    observed.claim_telemetry_spool("owner", path="/other-lane", **options)
    assert capture.events == [], (
        "an unrelated empty lane cannot prove the busy target was unavailable"
    )
    observed.claim_telemetry_spool("owner", path=contracts.SPOOL_PATH, **options)
    observed.claim_active_processor_requests("owner", **options)
    assert [event["kind"] for event in capture.events] == ["idle"], (
        "target-lane emptiness should be reported once without changing claim behavior"
    )


def test_noop_completion_does_not_fabricate_backlog_progress() -> None:
    class Noop:
        def complete_active_processor_request(self, *_args, **_kwargs):
            return None

    capture = Capture()
    observed = workers.ObservedStore(Noop(), workers.Emitter(capture))
    assert observed.complete_active_processor_request("missing") is None, (
        "a no-op result must remain a no-op"
    )
    assert capture.events == [], (
        "completion evidence requires a durable successful result"
    )
    replay = workers.Replay(RUN_ID + "-spool-0", Event(), workers.Emitter(capture))
    assert replay.execute(RUN_ID + "-spool-1")["status"] == 200, (
        "owned backlog items must finish without waiting at the busy-target barrier"
    )


@pytest.mark.parametrize("failure", ["factory-none", "unknown-role"])
def test_worker_startup_failure_still_closes_the_owned_listener(
    isolated, monkeypatch, failure: str
) -> None:
    if failure == "factory-none":
        monkeypatch.setattr(
            workers,
            "ProcessorFactory",
            lambda *_args, **_kwargs: SimpleNamespace(build=lambda: None),
        )
    stop, release = Event(), Event()
    with pytest.raises(contracts.ProofError):
        workers.run_worker(
            InMemoryStore(), "unknown", RUN_ID + "-spool-0", Capture(), stop, release
        )
    assert stop.is_set() and release.is_set(), (
        "worker startup errors must release the owned stop/barrier controls"
    )


def test_worker_entry_closes_a_normally_stopped_runtime_store(database) -> None:
    capture, stop, release = Capture(), Event(), Event()
    stop.set()
    release.set()
    probe.worker_entry("spool", RUN_ID + "-spool-0", capture, stop, release)
    assert capture.closed and database.store.closes == 1, (
        "a normal child entry must close its owned Store and pipe"
    )
    assert [event["kind"] for event in capture.events] == ["ready"], (
        "an already-stopped child must not manufacture claims or completion events"
    )


def test_barrier_can_wait_for_a_later_owned_arm(isolated) -> None:
    results = []
    waiter = Thread(target=lambda: results.append(probe.wait_for_arm(timeout=3)))
    waiter.start()
    probe.arm(POD_UID, RUN_ID, INTENT)
    waiter.join(timeout=4)
    assert not waiter.is_alive() and results[0]["intent_sha256"] == INTENT, (
        "the startup barrier must wait without initializing the database"
    )


@pytest.mark.parametrize("role", contracts.ROLES)
def test_backlog_payload_identity_is_verified_not_just_completion_count(
    role: str,
) -> None:
    class CorruptTransport(InMemoryStore):
        def claim_active_processor_requests(self, *args, **kwargs):
            return [
                item.model_copy(
                    update={
                        "body_base64": base64.b64encode(b'{"changed":true}').decode()
                    }
                )
                if item.request_id.endswith("-1")
                else item
                for item in super().claim_active_processor_requests(*args, **kwargs)
            ]

        def claim_telemetry_spool(self, *args, **kwargs):
            return [
                replace(item, payload={"changed": True})
                if item.request_id.endswith("-1")
                else item
                for item in super().claim_telemetry_spool(*args, **kwargs)
            ]

    store = CorruptTransport()
    with pytest.raises(contracts.ProofError, match="backlog payloads"):
        probe.exercise_role(store, role, RUN_ID, factory=MemoryWorkers(store))


def test_failed_listener_thread_start_closes_its_owned_socket(
    isolated, monkeypatch
) -> None:
    servers = []
    original = workers.http_server

    def server(*args):
        value = original(*args)
        servers.append(value)
        return value

    class UnavailableThread:
        def __init__(self, **_kwargs):
            pass

        def start(self):
            raise RuntimeError("controlled thread allocation failure")

    monkeypatch.setattr(workers, "http_server", server)
    monkeypatch.setattr(workers, "Thread", UnavailableThread)
    with pytest.raises(RuntimeError, match="allocation"):
        workers.run_worker(
            InMemoryStore(), "spool", RUN_ID + "-spool-0", Capture(), Event(), Event()
        )
    assert servers[0].socket.fileno() == -1, (
        "failed thread startup must not leak an owned listener"
    )

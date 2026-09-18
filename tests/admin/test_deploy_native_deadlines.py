from __future__ import annotations

import contextlib
import io
import os
import sqlite3
import subprocess
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from email.message import Message
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from typing import Any, cast

import botocore.auth
import pytest

from gpu_fault.admin import api_budget as budget
from gpu_fault.admin import (
    deadlines,
    execution,
    grafana,
    native_http,
    process_supervisor,
)
from gpu_fault.admin.bootstrap_common import BootstrapError
from gpu_fault.admin.process_supervisor import ProcessSupervisionLost
from gpu_fault_release import regional_adot_self_metrics as adot


class Response(io.BytesIO):
    status = 200

    def __init__(self) -> None:
        super().__init__(b'{"status":"success","data":{"result":[]}}')


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    values = [100.0]
    monkeypatch.setattr(time, "monotonic", lambda: values[0])

    def sleep(seconds: float) -> None:
        values[0] += seconds

    monkeypatch.setattr(time, "sleep", sleep)
    return values


@pytest.fixture
def native_io(monkeypatch: pytest.MonkeyPatch) -> dict[str, list[float]]:
    calls: dict[str, list[float]] = {"http": [], "http_started": [], "credentials": []}

    def urlopen(_request: urllib.request.Request, *, timeout: float) -> Response:
        calls["http"].append(timeout)
        calls["http_started"].append(time.monotonic())
        return Response()

    def worker(
        arguments: list[str], *, input_text: str, **_kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        if arguments[-1] == "credentials":
            with budget.api_slot("aws"):
                calls["credentials"].append(time.monotonic())
                document: dict[str, object] = {
                    "Version": 1,
                    "AccessKeyId": "fixture-access",
                    "SecretAccessKey": "fixture-secret",
                    "SessionToken": "fixture-session",
                }
        else:
            assert arguments[-1] == "request", "unexpected native helper command"
            document = native_http.worker_request(
                native_http.decode_message(input_text)
            )
        return subprocess.CompletedProcess(
            arguments, 0, native_http.encode_message(document), ""
        )

    monkeypatch.setattr(execution, "run_command", worker)
    monkeypatch.setattr(
        botocore.auth,
        "SigV4Auth",
        lambda *_args: SimpleNamespace(add_auth=lambda _request: None),
    )
    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(
        urllib.request,
        "build_opener",
        lambda *_args: SimpleNamespace(
            open=lambda *args, **kwargs: urllib.request.urlopen(*args, **kwargs)
        ),
    )
    return calls


def request(client: str) -> object:
    if client == "grafana":
        return grafana.urllib_transport(
            "GET", "https://grafana.invalid/api/health", {}, None
        )
    return adot.amp_instant_query(
        SimpleNamespace(
            config=SimpleNamespace(
                aws_region="us-west-2",
                health=SimpleNamespace(amp_workspace_id="test-workspace"),
            )
        ),
        "up",
    )


@pytest.mark.parametrize("client,backend", [("grafana", "http"), ("amp", "aws")])
def test_native_timeout_uses_task_deadline_after_admission_wait(
    clock: list[float], native_io: dict[str, list[float]], client: str, backend: str
) -> None:
    with budget.deployment_api_budget(), execution.deployment_deadline("root", 600):
        root = budget.budget_root()
        assert root is not None, "test budget was not created"
        with sqlite3.connect(root / "budget.sqlite3") as database:
            database.execute(
                "UPDATE capacity SET next_start=102 WHERE backend=?", (backend,)
            )

        def task() -> None:
            with execution.deadline_scope("native task", 5):
                request(client)

        with ThreadPoolExecutor(max_workers=1) as pool:
            pool.submit(task).result()
        assert native_io["http"] == [
            pytest.approx(105 - native_io["http_started"][0])
        ], "native HTTP ignored the task deadline or time spent awaiting admission"
        assert 102 <= native_io["http_started"][0] < 105, (
            "HTTP started outside its admission/task window"
        )
        assert os.environ[execution.DEADLINE_ENV] == "700.0", (
            "worker deadline changed the root process environment"
        )


@pytest.mark.parametrize("backend", ["http", "aws"])
def test_blocked_native_admission_cannot_outlive_current_task(
    clock: list[float], backend: str
) -> None:
    with budget.deployment_api_budget(), execution.deployment_deadline("root", 600):
        with contextlib.ExitStack() as holders:
            holders.enter_context(
                budget.api_slot(backend, weight=budget.LIMITS[backend])
            )
            started = clock[0]
            with execution.deadline_scope("short admission", 0.2):
                with pytest.raises(budget.ApiBudgetError, match="deadline"):
                    with budget.api_slot(backend):
                        pytest.fail("blocked native work outlived its task deadline")
            assert clock[0] - started < 0.3, "admission used the longer root deadline"
            reported = cast(
                dict[str, dict[str, object]], budget.statistics()["backends"]
            )
            assert reported[backend]["commands"] == 1, (
                "expired waiter was counted as admitted work"
            )
        root = budget.budget_root()
        assert root is not None, "test budget was not created"
        with sqlite3.connect(root / "budget.sqlite3") as database:
            assert database.execute("SELECT COUNT(*) FROM leases").fetchone()[0] == 0, (
                "expired admission leaked a waiting lease"
            )


def test_sqlite_admission_lock_wait_is_capped_by_task_deadline() -> None:
    with budget.deployment_api_budget():
        root = budget.budget_root()
        assert root is not None, "test budget was not created"
        with sqlite3.connect(root / "budget.sqlite3") as blocker:
            blocker.execute("BEGIN IMMEDIATE")
            started = time.monotonic()
            with execution.deadline_scope("locked admission", 0.1):
                with pytest.raises(sqlite3.OperationalError, match="locked"):
                    with budget.api_slot("http"):
                        pytest.fail("a locked ledger admitted native work")
            assert time.monotonic() - started < 0.6, (
                "admission used the SQLite default timeout beyond its task deadline"
            )
            assert blocker.execute("SELECT COUNT(*) FROM leases").fetchone()[0] == 0, (
                "failed lock acquisition left a lease"
            )


@pytest.mark.parametrize("client", ["grafana", "amp"])
@pytest.mark.parametrize("scoped", [False, True])
def test_expired_native_task_never_starts_http(
    clock: list[float], native_io: dict[str, list[float]], client: str, scoped: bool
) -> None:
    scope = budget.deployment_api_budget() if scoped else contextlib.nullcontext()
    with scope, execution.deadline_scope("expired native task", 1):
        clock[0] = 102
        with pytest.raises(
            (budget.ApiBudgetError, execution.DeploymentDeadlineExceeded),
            match="deadline",
        ):
            request(client)
    assert native_io["http"] == [], "expired task opened a native HTTP connection"
    assert native_io["credentials"] == [], "expired task began credential resolution"


@pytest.mark.parametrize("client", ["grafana", "amp"])
@pytest.mark.parametrize("recovery", [False, True])
def test_native_recovery_in_worker_ignores_expired_normal_but_not_hard_deadline(
    clock: list[float], native_io: dict[str, list[float]], client: str, recovery: bool
) -> None:
    with (
        budget.deployment_api_budget(),
        execution.deployment_deadline("normal", 1, recovery_seconds=10),
    ):
        clock[0] = 102

        def task() -> dict[str, str] | None:
            scope = (
                execution.recovery_deadline("rollback")
                if recovery
                else execution.cleanup_deadline("cleanup", 3)
            )
            with scope:
                request(client)
                return execution.command_environment({})

        with ThreadPoolExecutor(max_workers=1) as pool:
            exported = pool.submit(task).result()
        expected = 9.0 if recovery else 3.0
        assert native_io["http"] == [
            pytest.approx(102 + expected - native_io["http_started"][0])
        ], "native compensation used an expired normal budget or extended hard stop"
        assert exported is not None, "worker did not export its compensation budget"
        assert exported[execution.RECOVERY_ACTIVE_ENV] == "true", (
            "child environment lost its recovery semantics"
        )
        assert float(exported[execution.DEADLINE_ENV]) == 102 + expected, (
            "native HTTP and child processes received different deadlines"
        )
        assert os.environ[execution.DEADLINE_ENV] == "101.0", (
            "worker compensation replaced the root normal deadline"
        )
        clock[0] = 112
        with pytest.raises(execution.DeploymentDeadlineExceeded, match="deadline"):
            with execution.recovery_deadline("too late"):
                request(client)
        assert len(native_io["http"]) == 1, "recovery extended an expired hard stop"


def test_nested_compensation_cannot_extend_current_cleanup(clock: list[float]) -> None:
    with execution.cleanup_deadline("cleanup", 3):
        clock[0] = 101
        with execution.cleanup_deadline("nested cleanup", 120):
            assert deadlines.remaining_timeout(30) == 2, (
                "nested cleanup extended its parent"
            )
        with execution.recovery_deadline("nested rollback"):
            assert deadlines.remaining_timeout(30) == 2, (
                "nested rollback extended cleanup"
            )
        assert deadlines.recovery_active(), "cleanup lost its recovery context"
    assert not deadlines.recovery_active(), "cleanup leaked its recovery context"


@pytest.mark.parametrize("client", ["grafana", "amp"])
def test_native_success_after_task_expiry_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
    clock: list[float],
    native_io: dict[str, list[float]],
    client: str,
) -> None:
    class LateResponse(Response):
        def read1(self, size: int | None = -1, /) -> bytes:
            clock[0] = 106
            return super().read1(size)

    monkeypatch.setattr(
        urllib.request, "urlopen", lambda _request, **_kwargs: LateResponse()
    )
    with execution.deadline_scope("native response", 5):
        with pytest.raises(
            execution.DeploymentDeadlineExceeded,
            match=(
                "^deployment deadline exceeded: native response$"
                if client == "grafana"
                else "^deployment deadline exceeded: AMP query$"
            ),
        ):
            request(client)


@pytest.mark.parametrize("client", ["grafana", "amp"])
@pytest.mark.parametrize("fail_at", [1, 2])
def test_native_http_checks_supervision_at_entry_and_before_starting_a_worker(
    monkeypatch: pytest.MonkeyPatch,
    native_io: dict[str, list[float]],
    client: str,
    fail_at: int,
) -> None:
    checks = 0

    def unsafe(*, allow_interrupted: bool = False) -> None:
        nonlocal checks
        checks += 1
        if checks == fail_at:
            raise ProcessSupervisionLost("owned test supervision is unavailable")

    monkeypatch.setattr(
        grafana if client == "grafana" else adot, "ensure_supervision_safe", unsafe
    )
    monkeypatch.setattr(process_supervisor, "ensure_supervision_safe", unsafe)
    with budget.deployment_api_budget():
        with pytest.raises(ProcessSupervisionLost, match="test supervision"):
            request(client)
        assert native_io["http"] == [], "poisoned process started native HTTP work"
        assert budget.statistics()["backends"] == {}, (
            "poisoned process attempted admission before checking supervision"
        )
        assert native_io["credentials"] == [], (
            "poisoned process began native credential resolution"
        )


def test_grafana_error_body_is_read_inside_admission(
    monkeypatch: pytest.MonkeyPatch, native_io: dict[str, list[float]]
) -> None:
    class ErrorBody(io.BytesIO):
        def read1(self, size: int | None = -1, /) -> bytes:
            root = budget.budget_root()
            assert root is not None, "test budget was not created"
            with sqlite3.connect(root / "budget.sqlite3") as database:
                assert (
                    database.execute(
                        "SELECT COUNT(*) FROM leases WHERE backend='http' AND state='active'"
                    ).fetchone()[0]
                    == 1
                ), "HTTP error body escaped its capacity reservation"
            return super().read1(size)

    body = ErrorBody(b"not found")

    def failed(_request: Any, **_kwargs: Any) -> Iterator[None]:
        raise urllib.error.HTTPError(
            "https://grafana.invalid/api/health", 404, "not found", Message(), body
        )

    monkeypatch.setattr(urllib.request, "urlopen", failed)
    with budget.deployment_api_budget():
        response = grafana.urllib_transport(
            "GET", "https://grafana.invalid/api/health", {}, None
        )
        assert response == grafana.HttpResponse(404, "not found"), (
            "deadline enforcement changed Grafana's HTTP error contract"
        )
        assert body.closed, "Grafana left its HTTP error response open"


@pytest.mark.parametrize("client", ["grafana", "amp"])
@pytest.mark.parametrize("task_seconds", [0.1, 90.0])
def test_slow_streams_stop_at_task_or_request_deadline_without_waiting_for_eof(
    monkeypatch: pytest.MonkeyPatch,
    clock: list[float],
    native_io: dict[str, list[float]],
    client: str,
    task_seconds: float,
) -> None:
    reads = 0

    class StreamingResponse(Response):
        def read(self, size: int | None = -1, /) -> bytes:
            pytest.fail("native HTTP used an EOF-blocking response read")

        def read1(self, size: int | None = -1, /) -> bytes:
            nonlocal reads
            assert size is not None and 0 < size <= 65536, (
                "native response read requested an unbounded chunk"
            )
            reads += 1
            clock[0] += 0.02 if task_seconds < 1 else 1
            return b" "

    response = StreamingResponse()
    monkeypatch.setattr(urllib.request, "urlopen", lambda _request, **_kwargs: response)
    local_grafana_expiry = (
        client == "grafana" and task_seconds > grafana.HTTP_TIMEOUT_SECONDS
    )
    with execution.deadline_scope("streaming task", task_seconds) as task:
        with pytest.raises(
            BootstrapError
            if local_grafana_expiry
            else execution.DeploymentDeadlineExceeded,
            match=(
                "^Grafana HTTP request timed out$"
                if local_grafana_expiry
                else "^deployment deadline exceeded: streaming task$"
                if client == "grafana"
                else "^deployment deadline exceeded: AMP query$"
            ),
        ):
            request(client)
        assert deadlines.current_deadline() is task, (
            "the native request did not restore its enclosing deadline"
        )
        if local_grafana_expiry:
            assert task.remaining() == pytest.approx(
                task_seconds - grafana.HTTP_TIMEOUT_SECONDS
            ), "Grafana degraded an expired outer task or extended its budget"
        assert clock[0] - 100 <= min(task_seconds, 30) + 0.02, (
            "a streaming body silently extended the task or HTTP request deadline"
        )
        assert 1 <= reads <= (6 if task_seconds < 1 else 30), (
            "deadline checks did not run between streaming chunks"
        )
    assert response.closed, "timed-out streaming response was not closed"


@pytest.mark.parametrize("client", ["grafana", "amp", "grafana-error"])
def test_oversized_native_responses_are_closed_and_rejected(
    monkeypatch: pytest.MonkeyPatch, native_io: dict[str, list[float]], client: str
) -> None:
    reads: list[int] = []

    class OversizedResponse(Response):
        def read(self, size: int | None = -1, /) -> bytes:
            pytest.fail("native HTTP attempted an unbounded body read")

        def read1(self, size: int | None = -1, /) -> bytes:
            assert size is not None and 0 < size <= 65536, (
                "native HTTP did not bound its response chunk"
            )
            reads.append(size)
            return b"x" * size

    response = OversizedResponse()

    def urlopen(_request: Any, **_kwargs: Any) -> Response:
        if client == "grafana-error":
            raise urllib.error.HTTPError(
                "https://grafana.invalid/api/health",
                500,
                "unavailable",
                Message(),
                response,
            )
        return response

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    with budget.deployment_api_budget():
        with pytest.raises(
            BootstrapError
            if client.startswith("grafana")
            else deadlines.HttpResponseTooLarge,
            match="response exceeds its size limit",
        ):
            request("grafana" if client.startswith("grafana") else client)
        assert sum(reads) == deadlines.MAX_HTTP_RESPONSE_BYTES + 1, (
            "native HTTP buffered beyond the response limit and one overflow byte"
        )
        assert response.closed, "oversized native response was not closed"
        root = budget.budget_root()
        assert root is not None, "test budget was not created"
        with sqlite3.connect(root / "budget.sqlite3") as database:
            assert database.execute("SELECT COUNT(*) FROM leases").fetchone()[0] == 0, (
                "oversized response leaked its admission slot"
            )


def test_real_urllib_drip_response_cannot_extend_task_until_eof() -> None:
    stopped = threading.Event()
    completed = threading.Event()
    chunks = [0]

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            self.send_response(200)
            self.send_header("Content-Length", "1048576")
            self.end_headers()
            try:
                while not stopped.is_set():
                    self.wfile.write(b" ")
                    self.wfile.flush()
                    chunks[0] += 1
                    stopped.wait(0.01)
            except OSError:
                pass
            finally:
                completed.set()

        def log_message(self, format: str, *args: object) -> None:
            pass

    with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
        thread = threading.Thread(
            target=server.serve_forever, kwargs={"poll_interval": 0.01}
        )
        thread.start()
        started = time.monotonic()
        try:
            with execution.deadline_scope("local streaming fixture", 0.3):
                with pytest.raises(
                    execution.DeploymentDeadlineExceeded,
                    match="^deployment deadline exceeded: local streaming fixture$",
                ):
                    grafana.urllib_transport(
                        "GET",
                        f"http://127.0.0.1:{server.server_port}/fixture",
                        {},
                        None,
                    )
            assert time.monotonic() - started < 1, (
                "urllib waited for a streaming body's EOF beyond the task deadline"
            )
            assert chunks[0] > 1, "local fixture did not exercise streaming reads"
        finally:
            stopped.set()
            server.shutdown()
            thread.join(timeout=2)
            assert not thread.is_alive(), "local HTTP fixture server did not stop"
            if chunks[0]:
                assert completed.wait(timeout=2), (
                    "local HTTP fixture handler did not stop"
                )


@pytest.mark.parametrize("size", [0, deadlines.MAX_HTTP_RESPONSE_BYTES])
def test_bounded_reader_accepts_empty_and_exact_limit_bodies(size: int) -> None:
    with io.BytesIO(b"x" * size) as response, deadlines.deadline_scope("body", 30):
        value = deadlines.read_http_response(response)
    assert len(value) == size, (
        "native HTTP size limit incorrectly truncated a valid body"
    )


@pytest.mark.parametrize("client", ["grafana", "amp"])
def test_native_http_also_obeys_an_earlier_hard_deadline(
    monkeypatch: pytest.MonkeyPatch,
    clock: list[float],
    native_io: dict[str, list[float]],
    client: str,
) -> None:
    monkeypatch.setenv(execution.HARD_DEADLINE_ENV, "105")
    with execution.deadline_scope("long task", 90):
        request(client)
    assert native_io["http"] == [5.0], "native HTTP ignored the earlier hard deadline"


@pytest.mark.parametrize("client", ["grafana", "amp"])
def test_concurrent_native_tasks_keep_independent_deadlines(
    clock: list[float], native_io: dict[str, list[float]], client: str
) -> None:
    barrier = threading.Barrier(2)

    def task(seconds: float) -> None:
        with execution.deadline_scope("parallel native task", seconds):
            barrier.wait(timeout=2)
            request(client)

    with execution.deployment_deadline("root", 600):
        with ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(task, [5.0, 9.0]))
        assert sorted(native_io["http"]) == [5.0, 9.0], (
            "parallel tasks replaced each other's native HTTP deadlines"
        )
        assert os.environ[execution.DEADLINE_ENV] == "700.0", (
            "parallel tasks changed the root deadline environment"
        )

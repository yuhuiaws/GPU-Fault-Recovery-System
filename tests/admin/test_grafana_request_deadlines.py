"""Request-local Grafana failures must not conceal deployment or cleanup expiry."""

from __future__ import annotations

import contextlib
import json
import subprocess
import traceback
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import pytest

from gpu_fault.admin import deadlines, execution, grafana
from gpu_fault.admin.bootstrap_common import (
    SITE_TAG_KEY,
    BootstrapMutationRequired,
    CommandRunner,
)
from gpu_fault.admin.bootstrap_task_inputs import task_input_spec
from gpu_fault.admin.process_supervisor import ProcessSupervisionLost
from tests.admin._bootstrap_support import _cluster
from tests.admin.test_admin_grafana import AMP, HYPERPOD_WORKSPACE, SITE, Runner

PRIVATE_MARKER = "fixture-private-timeout-output"
REQUEST_TIMEOUT = "deployment deadline exceeded: Grafana HTTP request"
TOKEN_DELETE = "delete-workspace-service-account-token"


@dataclass
class Clock:
    value: float = 100.0

    def monotonic(self) -> float:
        return self.value


@dataclass
class HttpWorker:
    clock: Clock
    elapsed: float | None = None
    error: BaseException | None = None
    response: tuple[int, str] | None = None
    timeouts: list[float] = field(default_factory=list)

    def run(
        self, arguments: Sequence[str], *, timeout_seconds: float, **_options: Any
    ) -> subprocess.CompletedProcess[str]:
        assert arguments[-1] == "request", "unexpected non-HTTP worker"
        self.timeouts.append(timeout_seconds)
        if self.error is not None:
            raise self.error
        if self.response is not None:
            status, body = self.response
            return subprocess.CompletedProcess(
                arguments, 0, json.dumps({"status": status, "body": body}), ""
            )
        self.clock.value += timeout_seconds if self.elapsed is None else self.elapsed
        raise subprocess.TimeoutExpired(
            arguments, timeout_seconds, output=PRIVATE_MARKER, stderr=PRIVATE_MARKER
        )


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> Clock:
    value = Clock()
    monkeypatch.setattr(deadlines, "time", value)
    monkeypatch.setattr(grafana, "time", value)
    return value


@pytest.fixture
def worker(monkeypatch: pytest.MonkeyPatch, clock: Clock) -> HttpWorker:
    value = HttpWorker(clock)
    monkeypatch.setattr(execution, "run_command", value.run)
    return value


@pytest.fixture
def runner() -> Runner:
    return Runner([{**HYPERPOD_WORKSPACE, "tags": {SITE_TAG_KEY: SITE}}])


@pytest.fixture
def repository(tmp_path: Path) -> Path:
    directory = tmp_path / grafana.DASHBOARDS_DIRECTORY
    directory.mkdir(parents=True)
    (directory / "fixture.json").write_text(
        json.dumps({"uid": "deadline-fixture", "title": "Deadline Fixture"}),
        encoding="utf-8",
    )
    return tmp_path


def ensure(runner: Runner, repository: Path) -> dict[str, Any]:
    return grafana.ensure_grafana_dashboards(
        cast(CommandRunner, runner),
        settings=grafana.GrafanaSettings(),
        cpu=_cluster(),
        site_id=SITE,
        amp_workspace_id=AMP,
        repository_root=repository,
    )


@pytest.mark.parametrize("scoped", [False, True])
def test_request_expiry_degrades_only_presentation_and_revokes_token(
    scoped: bool,
    worker: HttpWorker,
    runner: Runner,
    repository: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    scope = (
        deadlines.deadline_scope("bootstrap/grafana_install", 900)
        if scoped
        else contextlib.nullcontext()
    )
    with scope:
        outer = deadlines.current_deadline()
        result = ensure(runner, repository)
        assert deadlines.current_deadline() is outer, "request leaked its deadline"
        deadlines.remaining_timeout(30)
    assert result["status"] == "DEGRADED", "request expiry did not degrade presentation"
    assert result["reason_code"] == "PRESENTATION_UNAVAILABLE", (
        "request expiry changed the presentation failure policy"
    )
    assert task_input_spec("grafana_install").result_status({"grafana": result}) == (
        "degraded"
    ), "the registered Grafana policy rejected a presentation outage"
    assert worker.timeouts == [30], "the request lost its bounded HTTP budget"
    assert runner.operations().count(TOKEN_DELETE) == 1, "token cleanup was not run"
    output = capsys.readouterr()
    assert PRIVATE_MARKER not in json.dumps(result) + output.out + output.err, (
        "private worker output reached diagnostics"
    )
    with pytest.raises(BootstrapMutationRequired):
        grafana.ensure_grafana_dashboards(
            cast(CommandRunner, runner),
            settings=grafana.GrafanaSettings(previous=result),
            cpu=_cluster(),
            site_id=SITE,
            amp_workspace_id=AMP,
            repository_root=repository,
            probe_only=True,
        )
    assert worker.timeouts == [30], (
        "a degraded checkpoint was accepted or reprovisioned"
    )


@pytest.mark.parametrize("scope_kind", ["task", "root", "hard"])
def test_enclosing_expiry_is_fatal_even_with_request_labeled_error(
    scope_kind: str,
    monkeypatch: pytest.MonkeyPatch,
    clock: Clock,
    worker: HttpWorker,
    runner: Runner,
    repository: Path,
) -> None:
    if scope_kind == "hard":
        monkeypatch.setenv(deadlines.HARD_DEADLINE_ENV, str(clock.value + 5))
    scope = (
        deadlines.deployment_deadline("deploy root", 5)
        if scope_kind == "root"
        else deadlines.deadline_scope(
            "bootstrap/grafana_install", 900 if scope_kind == "hard" else 5
        )
    )
    with scope:
        outer = deadlines.current_deadline()
        with pytest.raises(deadlines.DeploymentDeadlineExceeded):
            ensure(runner, repository)
        assert deadlines.current_deadline() is outer, "request replaced the outer scope"
    assert worker.timeouts == [5], "the request ignored the enclosing deadline"
    assert runner.operations().count(TOKEN_DELETE) == 1, "cleanup was not attempted"


def test_outer_expiry_during_worker_teardown_is_not_degraded(
    worker: HttpWorker, runner: Runner, repository: Path
) -> None:
    worker.elapsed = 35
    with deadlines.deadline_scope("bootstrap/grafana_install", 35):
        with pytest.raises(deadlines.DeploymentDeadlineExceeded):
            ensure(runner, repository)
    assert worker.timeouts == [30], "the local budget was extended to the task budget"
    assert runner.operations().count(TOKEN_DELETE) == 1, "cleanup was not attempted"


def test_early_worker_timeout_is_not_proof_of_request_deadline_expiry(
    worker: HttpWorker, runner: Runner, repository: Path
) -> None:
    worker.elapsed = 0
    with pytest.raises(deadlines.DeploymentDeadlineExceeded) as failure:
        ensure(runner, repository)
    assert str(failure.value) == REQUEST_TIMEOUT, "fixture lost its request label"
    assert PRIVATE_MARKER not in "".join(traceback.format_exception(failure.value)), (
        "private timeout output survived native exception sanitization"
    )
    assert runner.operations().count(TOKEN_DELETE) == 1, "cleanup was not attempted"


@pytest.mark.parametrize(
    "error",
    [
        deadlines.DeploymentDeadlineExceeded("unidentified enclosing deadline"),
        TimeoutError("unidentified timeout"),
    ],
    ids=["unidentified-deadline", "unclassified-timeout"],
)
def test_unidentified_transport_timeouts_remain_fatal(
    error: TimeoutError,
    monkeypatch: pytest.MonkeyPatch,
    runner: Runner,
    repository: Path,
) -> None:
    def fail(*_args: object, **_kwargs: object) -> tuple[int, str]:
        raise error

    monkeypatch.setattr(grafana, "http_request", fail)
    with pytest.raises(type(error)) as failure:
        ensure(runner, repository)
    assert failure.value is error, "an unidentified timeout became presentation failure"
    assert runner.operations().count(TOKEN_DELETE) == 1, "cleanup was not attempted"


@pytest.mark.parametrize(
    "error",
    [
        ProcessSupervisionLost("fixture ownership lost"),
        KeyboardInterrupt("fixture stop"),
    ],
    ids=["supervision-loss", "interruption"],
)
@pytest.mark.parametrize("after_timeout", [False, True])
def test_supervision_and_interruption_always_remain_fatal(
    error: BaseException,
    after_timeout: bool,
    monkeypatch: pytest.MonkeyPatch,
    worker: HttpWorker,
    runner: Runner,
    repository: Path,
) -> None:
    if after_timeout:
        checks: list[bool] = []

        def guard(*, allow_interrupted: bool = False) -> None:
            checks.append(allow_interrupted)
            if len(checks) > 1:
                raise error

        monkeypatch.setattr(grafana, "ensure_supervision_safe", guard)
    else:
        worker.error = error
    with pytest.raises(type(error)) as failure:
        ensure(runner, repository)
    assert failure.value is error, "fatal supervision state became presentation failure"
    assert runner.operations().count(TOKEN_DELETE) == 1, "cleanup was not attempted"


@pytest.mark.parametrize(
    "error",
    [
        deadlines.DeploymentDeadlineExceeded(REQUEST_TIMEOUT),
        TimeoutError("fixture cleanup timeout"),
        RuntimeError("fixture cleanup failure"),
        ProcessSupervisionLost("fixture cleanup ownership lost"),
        KeyboardInterrupt("fixture cleanup stop"),
    ],
    ids=[
        "request-labeled-deadline",
        "timeout",
        "runtime",
        "supervision",
        "interruption",
    ],
)
def test_cleanup_failures_cannot_be_converted_into_request_degradation(
    error: BaseException,
    monkeypatch: pytest.MonkeyPatch,
    worker: HttpWorker,
    runner: Runner,
    repository: Path,
) -> None:
    original_run = runner.run

    def run(arguments: Sequence[str], **options: Any) -> str:
        result = original_run(arguments, **options)
        if arguments[2] == TOKEN_DELETE:
            raise error
        return result

    monkeypatch.setattr(runner, "run", run)
    with pytest.raises(type(error)) as failure:
        ensure(runner, repository)
    assert failure.value is error, (
        "cleanup failure was mistaken for HTTP request expiry"
    )
    assert worker.timeouts == [30], "fixture did not reach the request deadline"
    assert runner.operations().count(TOKEN_DELETE) == 1, (
        "cleanup was not attempted once"
    )


@pytest.mark.parametrize("status", [200, 503])
def test_existing_http_success_and_failure_policy_is_unchanged(
    status: int, worker: HttpWorker, runner: Runner, repository: Path
) -> None:
    worker.response = (status, "{}")
    with deadlines.deadline_scope("bootstrap/grafana_install", 900):
        result = ensure(runner, repository)
    expected = "PROVISIONED" if status == 200 else "DEGRADED"
    assert result["status"] == expected, "ordinary HTTP handling changed"
    assert worker.timeouts, "the default Grafana task skipped HTTP provisioning"
    assert runner.operations().count(TOKEN_DELETE) == 1, "token cleanup was not run"

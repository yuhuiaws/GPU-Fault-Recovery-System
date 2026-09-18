"""Safe operation context through Grafana's public provisioning entry points."""

from __future__ import annotations

import json
import traceback
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import pytest

from gpu_fault.admin import grafana
from gpu_fault.admin.bootstrap_common import SITE_TAG_KEY, BootstrapError, CommandRunner
from gpu_fault.admin.deadlines import DeploymentDeadlineExceeded
from gpu_fault.admin.process_supervisor import ProcessSupervisionLost
from tests.admin._bootstrap_support import _cluster
from tests.admin.test_admin_grafana import AMP, REGION, SITE, Http, Runner

DATASOURCE = "/api/datasources/uid/gpu-fault-amp"
FOLDER = "/api/folders/gpu-fault-recovery"
TIMEOUT = "Grafana HTTP request timed out"
TOKEN_DELETE = "delete-workspace-service-account-token"
WORKSPACE = {
    "id": "g-operation-fixture",
    "status": "ACTIVE",
    "endpoint": "g-operation-fixture.grafana-workspace.us-east-1.amazonaws.com",
    "tags": {SITE_TAG_KEY: SITE},
}


class FaultingHttp(Http):
    def __init__(
        self,
        failures: Mapping[tuple[str, str], BaseException],
        responses: Mapping[tuple[str, str], Any] | None = None,
    ) -> None:
        super().__init__(responses)
        self.failures = dict(failures)

    def __call__(
        self, method: str, url: str, headers: Mapping[str, str], body: bytes | None
    ) -> grafana.HttpResponse:
        response = super().__call__(method, url, headers, body)
        path = self.requests[-1][1]
        error = self.failures.get((method, path))
        if error is not None:
            raise error
        return response


@pytest.fixture
def runner() -> Runner:
    return Runner([WORKSPACE])


@pytest.fixture
def repository(tmp_path: Path) -> Path:
    directory = tmp_path / grafana.DASHBOARDS_DIRECTORY
    directory.mkdir(parents=True)
    (directory / "fixture.json").write_text(
        json.dumps(
            {
                "id": 17,
                "uid": "operation-fixture",
                "title": "fixture-private-dashboard-title",
            }
        ),
        encoding="utf-8",
    )
    return tmp_path


def provision(
    runner: Runner,
    repository: Path,
    http: Http,
    *,
    endpoint: str = str(WORKSPACE["endpoint"]),
) -> dict[str, Any]:
    return grafana.provision_grafana(
        cast(CommandRunner, runner),
        workspace={"workspace_id": WORKSPACE["id"], "endpoint": endpoint},
        amp_workspace_id=AMP,
        region=REGION,
        dashboards_dir=repository / grafana.DASHBOARDS_DIRECTORY,
        http=http,
    )


@pytest.mark.parametrize(
    "method,path",
    [
        ("GET", "/api/org"),
        ("GET", DATASOURCE),
        ("POST", "/api/datasources"),
        ("PUT", DATASOURCE),
        ("GET", f"{DATASOURCE}/health"),
        ("POST", "/api/ds/query"),
        ("GET", FOLDER),
        ("POST", "/api/folders"),
        ("POST", "/api/dashboards/db"),
    ],
)
def test_transport_failure_identifies_only_the_fixed_operation(
    method: str, path: str, runner: Runner, repository: Path
) -> None:
    responses = {
        ("GET", DATASOURCE): grafana.HttpResponse(404, "{}"),
        ("GET", FOLDER): grafana.HttpResponse(404, "{}"),
    }
    failures = {(method, path): BootstrapError(TIMEOUT)}
    if method == "PUT":
        responses[("GET", DATASOURCE)] = grafana.HttpResponse(200, "{}")
    if path == "/api/ds/query":
        responses[("GET", f"{DATASOURCE}/health")] = grafana.HttpResponse(400, "{}")
    if path.endswith("/health"):
        failures[("POST", "/api/ds/query")] = BootstrapError(
            "Grafana HTTP request failed"
        )
    http = FaultingHttp(failures, responses)
    with pytest.raises(BootstrapError) as failure:
        provision(runner, repository, http)
    message = str(failure.value)
    assert f"{method} {path}: HTTP request timed out" in message, (
        "the sanitized failure lost its API operation"
    )
    assert str(WORKSPACE["endpoint"]) not in message, (
        "workspace URL entered diagnostics"
    )
    assert runner.token_key not in message, (
        "service-account credential entered diagnostics"
    )
    assert http.paths(method).count(path) == 1, "diagnostics retried a failed operation"
    assert runner.operations().count(TOKEN_DELETE) == 1, "token cleanup was not run"


@pytest.mark.parametrize(
    "transport_message,expected",
    [
        (TIMEOUT, "HTTP request timed out"),
        ("Grafana HTTP request failed", "HTTP request failed"),
        (
            "Grafana HTTP response exceeds its size limit",
            "HTTP response exceeds its size limit",
        ),
    ],
)
def test_sanitized_transport_reason_remains_available(
    transport_message: str, expected: str, runner: Runner, repository: Path
) -> None:
    http = FaultingHttp(
        {("POST", "/api/folders"): BootstrapError(transport_message)},
        {("GET", FOLDER): grafana.HttpResponse(404, "{}")},
    )
    with pytest.raises(BootstrapError) as failure:
        provision(runner, repository, http)
    assert str(failure.value) == f"Grafana POST /api/folders: {expected}", (
        "transport classification was lost or included unreviewed detail"
    )


def test_untrusted_transport_details_and_exception_chain_are_discarded(
    runner: Runner, repository: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    runner.token_key = "fixture-private-credential"
    endpoint = "fixture-private-host.grafana-workspace.us-east-1.amazonaws.com"

    class LeakyHttp(Http):
        def __call__(
            self, method: str, url: str, headers: Mapping[str, str], body: bytes | None
        ) -> grafana.HttpResponse:
            response = super().__call__(method, url, headers, body)
            if self.requests[-1][1] == "/api/dashboards/db":
                raise BootstrapError(
                    f"{TIMEOUT}\n{url}?query=fixture-private-query "
                    f"{dict(headers)!r} {body!r}"
                )
            return response

    http = LeakyHttp()
    with pytest.raises(BootstrapError) as failure:
        provision(runner, repository, http, endpoint=endpoint)
    assert str(failure.value) == (
        "Grafana POST /api/dashboards/db: HTTP request failed"
    ), "untrusted transport detail was retained"
    assert failure.value.__cause__ is None, (
        "raw transport error became an explicit cause"
    )
    assert failure.value.__context__ is None, (
        "raw transport error remained in the chain"
    )
    output = capsys.readouterr()
    diagnostic = (
        "".join(traceback.format_exception(failure.value)) + output.out + output.err
    )
    assert "fixture-private-" not in diagnostic, (
        "URL, query, credential, or dashboard body escaped through diagnostics"
    )
    assert http.paths("POST").count("/api/dashboards/db") == 1, (
        "a POST with unknown outcome was retried"
    )
    assert runner.operations().count(TOKEN_DELETE) == 1, "token cleanup was not run"


@pytest.mark.parametrize(
    "uid",
    [
        "gpu-fault-amp?query=fixture-private-query",
        "gpu-fault-amp/fixture-private-path",
        "gpu-fault-amp#fixture-private-fragment",
    ],
)
def test_unexpected_paths_are_not_echoed(
    uid: str, monkeypatch: pytest.MonkeyPatch, runner: Runner, repository: Path
) -> None:
    monkeypatch.setattr(grafana, "DATASOURCE_UID", uid)
    path = f"/api/datasources/uid/{uid}"
    http = FaultingHttp({("GET", path): BootstrapError(TIMEOUT)})
    with pytest.raises(BootstrapError) as failure:
        provision(runner, repository, http)
    assert str(failure.value) == (
        "Grafana unknown API operation: HTTP request timed out"
    ), "an unexpected path was treated as reviewed diagnostic context"
    assert uid not in str(failure.value), "raw unexpected path entered diagnostics"
    assert http.paths("GET")[-1] == path, "diagnostics changed the requested operation"


@pytest.mark.parametrize(
    "error",
    [
        DeploymentDeadlineExceeded("deployment deadline exceeded: bootstrap"),
        TimeoutError("unclassified timeout"),
        ProcessSupervisionLost("fixture ownership lost"),
        KeyboardInterrupt("fixture interruption"),
    ],
    ids=["outer-deadline", "unclassified-timeout", "supervision", "interruption"],
)
def test_fatal_transport_failures_are_not_wrapped_or_retried(
    error: BaseException, runner: Runner, repository: Path
) -> None:
    http = FaultingHttp({("POST", "/api/dashboards/db"): error})
    with pytest.raises(type(error)) as failure:
        provision(runner, repository, http)
    assert failure.value is error, (
        "a fatal transport failure became a presentation error"
    )
    assert http.paths("POST").count("/api/dashboards/db") == 1, (
        "fatal transport failure triggered a retry"
    )
    assert runner.operations().count(TOKEN_DELETE) == 1, "cleanup behavior changed"


def test_operation_context_reaches_the_existing_degraded_result(
    runner: Runner, repository: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    http = FaultingHttp({("GET", FOLDER): BootstrapError(TIMEOUT)})
    result = grafana.ensure_grafana_dashboards(
        cast(CommandRunner, runner),
        settings=grafana.GrafanaSettings(),
        cpu=_cluster(),
        site_id=SITE,
        amp_workspace_id=AMP,
        repository_root=repository,
        http=http,
    )
    expected = f"Grafana GET {FOLDER}: HTTP request timed out"
    assert result["status"] == "DEGRADED", "presentation failure changed task policy"
    assert result["reason_code"] == "PRESENTATION_UNAVAILABLE", "failure policy changed"
    assert result["reason"] == expected, "the checkpoint lost operation context"
    assert expected in capsys.readouterr().err, (
        "operator warning lost operation context"
    )
    assert runner.operations().count(TOKEN_DELETE) == 1, "token cleanup was not run"

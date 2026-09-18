from __future__ import annotations

import io
import sys
import urllib.request
from contextlib import redirect_stdout

import pytest

from scripts.e2e.regional.ha009_observation import observation_errors, pod_observation


@pytest.mark.parametrize("failed", [False, True])
@pytest.mark.parametrize("port", [8080, 8081, 8082])
def test_new_connection_uses_projected_dsn_and_exports_no_connection_diagnostics(
    failed, port, monkeypatch, tmp_path
) -> None:
    import psycopg

    projected = tmp_path / "projected-dsn"
    projected.write_text("postgresql://fixture.invalid/new")
    monkeypatch.setenv("GPU_FAULT_STORE_URL", "postgresql://fixture.invalid/old")
    monkeypatch.setenv("GPU_FAULT_STORE_URL_FILE", str(projected))
    connections = []

    class Connection:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def execute(self, sql):
            assert sql == "SELECT 1"
            return self

        def fetchone(self):
            return (1,)

    def connect(conninfo, **kwargs):
        connections.append(conninfo)
        if failed:
            raise RuntimeError("private diagnostic must not escape")
        return Connection()

    class Response(Connection):
        status = 200

        def read(self):
            return b"gpu_fault_postgres_pool_connections_errors_total 0"

    monkeypatch.setattr(psycopg, "connect", connect)
    urls = []
    monkeypatch.setattr(
        urllib.request, "urlopen", lambda url, **kw: urls.append(url) or Response()
    )
    stdout = io.StringIO()

    def control(*args, **kwargs):
        with monkeypatch.context() as patch:
            patch.setattr(sys, "argv", ["probe", args[-1]])
            with redirect_stdout(stdout):
                exec(args[args.index("-c") + 1], {})
        return stdout.getvalue()

    result = pod_observation(control, "worker", port, python="unit-python")
    assert urls == [
        f"http://127.0.0.1:{port}/healthz",
        f"http://127.0.0.1:{port}/metrics",
    ]
    assert connections == ["postgresql://fixture.invalid/new"]
    assert result["fresh_connection"] is (not failed)
    assert "private diagnostic" not in stdout.getvalue()
    assert "fixture.invalid" not in stdout.getvalue()
    assert result["healthz_status"] == 200


@pytest.mark.parametrize(
    "defect",
    [
        {"fresh_connection": False},
        {"healthz_status": 503},
        {"metrics_status": 503},
        {"metrics": {}},
        {"metrics": {"gpu_fault_postgres_pool_connections_errors_total": float("nan")}},
    ],
)
def test_readiness_without_authenticated_sql_and_complete_metrics_cannot_pass(
    defect,
) -> None:
    sample = {
        "healthz_status": 200,
        "metrics_status": 200,
        "fresh_connection": True,
        "metrics": {"gpu_fault_postgres_pool_connections_errors_total": 0.0},
    }
    observation = {"samples": {"pod": [sample]}, "auth_failures_in_logs": {"pod": 0}}
    propagation = {"digest": "unit-digest", "pods": {"pod": "unit-digest"}}
    assert observation_errors({"pod"}, propagation, observation) == []
    observation["samples"]["pod"] = [{**sample, **defect}]
    assert observation_errors({"pod"}, propagation, observation), observation

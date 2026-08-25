from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from scripts.e2e.isolated_api import (
    IsolatedApiProcess,
    random_execution_token,
    require_assertions_enabled,
    reserve_loopback_port,
    wait_for_isolated_api,
)

ROOT = Path(__file__).resolve().parents[2]
RUNNERS = (
    ROOT / "scripts/e2e/run_hyperpod_dcgm_metrics_e2e.py",
    ROOT / "scripts/e2e/run_hyperpod_efa_traffic_e2e.py",
    ROOT / "scripts/e2e/run_hyperpod_three_source_fault_e2e.py",
)


def test_isolated_api_credentials_and_ports_are_runtime_generated() -> None:
    first = random_execution_token()
    second = random_execution_token()

    assert len(first) >= 32
    assert first != second
    assert 0 < reserve_loopback_port() < 65536


def test_wait_reports_an_early_child_exit(tmp_path: Path) -> None:
    log = tmp_path / "api.log"
    log.write_text("address already in use\n", encoding="utf-8")
    process = subprocess.Popen([sys.executable, "-c", "raise SystemExit(17)"])
    process.wait(timeout=5)
    api = IsolatedApiProcess(process=process, url="http://127.0.0.1:1", log_path=log)

    with pytest.raises(
        RuntimeError, match=r"(?s)exited with code 17.*address already in use"
    ):
        wait_for_isolated_api(api, expected_executor="active", timeout_seconds=0.1)


class _HealthzHandler(BaseHTTPRequestHandler):
    payload: dict = {}

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        body = json.dumps(self.payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args: object) -> None:
        return


@contextmanager
def _healthz_server(payload: dict) -> Iterator[str]:
    handler = type("Handler", (_HealthzHandler,), {"payload": payload})
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@contextmanager
def _live_child() -> Iterator[subprocess.Popen]:
    process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        yield process
    finally:
        process.kill()
        process.wait(timeout=5)


def test_wait_fails_fast_on_a_healthy_api_with_the_wrong_identity(
    tmp_path: Path,
) -> None:
    # 身份不符原来 raise 在 try 里，被同一个 except Exception 吞掉，于是
    # 一个已经在应答的 API 会空转满 timeout，最后报成 "did not become
    # ready" —— 配置错误被伪装成启动超时。所以这里既断言错误信息，也断言
    # 它没有把 timeout 耗完。
    log = tmp_path / "api.log"
    log.write_text("", encoding="utf-8")
    payload = {
        "status": "ok",
        "executor": "simulation-only",
        "service_role": "combined",
    }

    with _live_child() as process, _healthz_server(payload) as url:
        api = IsolatedApiProcess(process=process, url=url, log_path=log)
        started = time.monotonic()
        with pytest.raises(RuntimeError, match="unexpected isolated API identity"):
            wait_for_isolated_api(api, expected_executor="active", timeout_seconds=20)
        assert time.monotonic() - started < 5


def test_wait_keeps_retrying_while_the_api_reports_unhealthy(tmp_path: Path) -> None:
    log = tmp_path / "api.log"
    log.write_text("", encoding="utf-8")
    payload = {"status": "unhealthy", "executor": "active", "service_role": "combined"}

    with _live_child() as process, _healthz_server(payload) as url:
        api = IsolatedApiProcess(process=process, url=url, log_path=log)
        with pytest.raises(RuntimeError, match="did not become ready"):
            wait_for_isolated_api(api, expected_executor="active", timeout_seconds=1)


def test_wait_returns_the_payload_once_identity_and_status_agree(
    tmp_path: Path,
) -> None:
    log = tmp_path / "api.log"
    log.write_text("", encoding="utf-8")
    payload = {"status": "ok", "executor": "active", "service_role": "combined"}

    with _live_child() as process, _healthz_server(payload) as url:
        api = IsolatedApiProcess(process=process, url=url, log_path=log)

        assert (
            wait_for_isolated_api(api, expected_executor="active", timeout_seconds=20)
            == payload
        )


def test_assertion_guard_rejects_optimized_python() -> None:
    command = (
        "from scripts.e2e.isolated_api import "
        "require_assertions_enabled; require_assertions_enabled()"
    )
    result = subprocess.run(
        [sys.executable, "-O", "-c", command],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "assertions are disabled" in result.stderr
    require_assertions_enabled()


def test_e2e_runners_use_the_shared_fail_closed_launcher() -> None:
    for path in RUNNERS:
        text = path.read_text(encoding="utf-8")

        assert "gpu_fault.api:create_app" not in text
        assert '"8080"' not in text
        assert "-e2e-token-" not in text
        assert "launch_isolated_api(env)" in text
        assert "wait_for_isolated_api(" in text
        assert "require_assertions_enabled()" in text
        assert text.index("require_assertions_enabled()") < text.index(
            'report["status"] = "PASSED"'
        )

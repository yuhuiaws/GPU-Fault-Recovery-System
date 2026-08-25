from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import secrets
import socket
import subprocess
import sys
import tempfile
import time
from urllib.request import urlopen


@dataclass(frozen=True)
class IsolatedApiProcess:
    process: subprocess.Popen
    url: str
    log_path: Path


def random_execution_token() -> str:
    return secrets.token_urlsafe(32)


def reserve_loopback_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def launch_isolated_api(
    env: dict[str, str],
) -> IsolatedApiProcess:
    port = reserve_loopback_port()
    log_file = tempfile.NamedTemporaryFile(
        prefix="gpu-fault-isolated-api-",
        suffix=".log",
        delete=False,
    )
    log_path = Path(log_file.name)
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "gpu_fault.app:create_app",
            "--factory",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--no-proxy-headers",
        ],
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
    )
    log_file.close()
    return IsolatedApiProcess(
        process=process,
        url=f"http://127.0.0.1:{port}",
        log_path=log_path,
    )


def _log_tail(path: Path, limit: int = 4000) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")[-limit:]
    except OSError:
        return "<isolated API log unavailable>"


def wait_for_isolated_api(
    api: IsolatedApiProcess,
    *,
    expected_executor: str,
    timeout_seconds: float = 90,
) -> dict:
    deadline = time.monotonic() + timeout_seconds
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        returncode = api.process.poll()
        if returncode is not None:
            raise RuntimeError(
                f"isolated API exited with code {returncode}:\n"
                + _log_tail(api.log_path)
            )
        try:
            with urlopen(f"{api.url}/healthz", timeout=2) as response:
                payload = json.load(response)
        except Exception as exc:
            # 连接被拒和 503（processor 线程还没热起来）都是「还没起来」，
            # 该重试。
            last_error = exc
            time.sleep(0.25)
            continue
        # identity 不是 readiness：executor 和 service_role 直接来自配置，
        # 不会随时间变好。这条 raise 原来就在上面的 try 里，被自己的
        # except Exception 吞掉，于是一个已经健康但配错的 API 会空转满 90
        # 秒，最后报成 "isolated API did not become ready"，把配置错误
        # 伪装成启动超时。
        if (
            payload.get("executor") != expected_executor
            or payload.get("service_role") != "combined"
        ):
            raise RuntimeError(f"unexpected isolated API identity: {payload}")
        if payload.get("status") != "ok":
            last_error = RuntimeError(f"isolated API is not ok yet: {payload}")
            time.sleep(0.25)
            continue
        return payload
    raise RuntimeError(
        f"isolated API did not become ready: {last_error}\n{_log_tail(api.log_path)}"
    )


def stop_isolated_api(api: IsolatedApiProcess) -> None:
    if api.process.poll() is None:
        api.process.terminate()
        try:
            api.process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            api.process.kill()
            api.process.wait(timeout=5)
    api.log_path.unlink(missing_ok=True)


def require_assertions_enabled() -> None:
    if not __debug__:
        raise RuntimeError(
            "E2E assertions are disabled under python -O; refusing to run"
        )

from __future__ import annotations

import contextlib
import hashlib
import json
import stat
import subprocess
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from types import SimpleNamespace

import pytest

from gpu_fault.admin import process_supervisor as supervision
from gpu_fault.admin.execution import deadline_scope
from gpu_fault.admin.grafana import DASHBOARD_FOLDER_UID, urllib_transport
from gpu_fault.admin.process_supervisor import run_owned_command


@pytest.mark.parametrize("characters", [0, 1024, 262144, 1048576])
def test_delayed_child_receives_complete_private_input(characters: int) -> None:
    text = "\u03bb" * characters
    payload = text.encode("utf-8")
    result = run_owned_command(
        [
            sys.executable,
            "-c",
            "import hashlib,json,os,stat,sys,time; "
            "time.sleep(0.35); data=sys.stdin.buffer.read(); s=os.fstat(0); "
            "print(json.dumps({'length':len(data),"
            "'sha256':hashlib.sha256(data).hexdigest(),"
            "'mode':stat.S_IMODE(s.st_mode),'links':s.st_nlink}))",
        ],
        input_text=text,
        capture=True,
        timeout=5,
    )

    assert result.returncode == 0, "the delayed reader did not complete"
    assert json.loads(result.stdout) == {
        "length": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "mode": stat.S_IRUSR | stat.S_IWUSR,
        "links": 0,
    }, "stdin must retain all bytes without a named or publicly readable file"


def test_stdin_preparation_cannot_extend_the_command_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = supervision.tempfile.TemporaryFile
    created = []

    @contextlib.contextmanager
    def delayed_input(**options):
        with original(**options) as stream:
            created.append(stream)
            time.sleep(0.05)
            yield stream

    def no_start(*_arguments, **_options):
        pytest.fail(
            "a command started after its input preparation exhausted the budget"
        )

    monkeypatch.setattr(
        supervision, "tempfile", SimpleNamespace(TemporaryFile=delayed_input)
    )
    monkeypatch.setattr(
        supervision,
        "subprocess",
        SimpleNamespace(Popen=no_start, TimeoutExpired=subprocess.TimeoutExpired),
    )
    with pytest.raises(subprocess.TimeoutExpired):
        run_owned_command(
            [sys.executable, "-c", "pass"],
            input_text="private fixture input",
            timeout=0.01,
        )
    assert len(created) == 1 and created[0].closed, (
        "timed-out stdin preparation must close its private descriptor"
    )


@pytest.mark.parametrize(
    "filename",
    ["gpu-fault-collector-health.json", "gpu-fault-control-plane-capacity.json"],
)
def test_real_dashboard_bodies_cross_the_supervised_http_transport(
    filename: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = Path(__file__).resolve().parents[2]
    model = json.loads(
        (root / "deploy/observability/dashboards" / filename).read_text()
    )
    body = json.dumps(
        {
            "dashboard": model,
            "folderUid": DASHBOARD_FOLDER_UID,
            "overwrite": True,
            "message": "gpu-fault-admin deploy",
        }
    ).encode()
    received: list[bytes] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            received.append(self.rfile.read(int(self.headers["Content-Length"])))
            content = b'{"version":1}'
            self.send_response(200)
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)

        def log_message(self, *_args: object) -> None:
            pass

    class SmallPipes:
        def __getattr__(self, name):
            return getattr(subprocess, name)

        def Popen(self, *arguments, **options):
            return subprocess.Popen(*arguments, **{**options, "pipesize": 8192})

    monkeypatch.setattr(supervision, "subprocess", SmallPipes())
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(
        target=lambda: server.serve_forever(poll_interval=0.01), daemon=True
    )
    thread.start()
    try:
        with deadline_scope("local dashboard transfer", 5):
            result = urllib_transport(
                "POST",
                f"http://127.0.0.1:{server.server_port}/api/dashboards/db",
                {"Content-Type": "application/json"},
                body,
            )
        assert result.status == 200
        assert json.loads(result.body) == {"version": 1}
        assert received == [body], (
            "the first and largest dashboard must each arrive complete, once"
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
        assert not thread.is_alive(), "the local HTTP fixture did not stop"

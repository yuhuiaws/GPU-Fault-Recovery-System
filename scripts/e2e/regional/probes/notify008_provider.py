"""A separate, bounded SIMULATED provider with a durable acceptance ledger."""

from __future__ import annotations

import json
import math
import os
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any
from urllib.request import ProxyHandler, Request, build_opener
from uuid import uuid4

from scripts.e2e.regional.probes.notify008_protocol import (
    MAX_RECEIPTS,
    ProbeError,
    run_identity,
    variants,
)

MAX_BODY = 4096


class ReceiptLog:
    def __init__(self, path: Path, run_id: str) -> None:
        self.run_id = run_identity(run_id)
        self.allowed = {variant.notification_id(run_id) for variant in variants()}
        self.records: list[dict[str, Any]] = []
        self.stream = path.open("xb")
        os.chmod(path, 0o600)

    def accept(self, value: Any) -> dict[str, Any]:
        if (
            not isinstance(value, dict)
            or set(value) != {"run_id", "notification_id"}
            or value["run_id"] != self.run_id
            or not isinstance(value["notification_id"], str)
            or value["notification_id"] not in self.allowed
            or len(self.records) >= MAX_RECEIPTS
        ):
            raise ProbeError("simulated provider refused request identity or budget")
        receipt = {
            "run_id": self.run_id,
            "notification_id": value["notification_id"],
            "message_id": f"simulated-{uuid4().hex}",
            "provider_pid": os.getpid(),
            "sequence": len(self.records) + 1,
        }
        self.stream.write(json.dumps(receipt, sort_keys=True).encode() + b"\n")
        self.stream.flush()
        os.fsync(self.stream.fileno())
        self.records.append(receipt)
        return receipt

    def close(self) -> None:
        self.stream.close()


def handler_for(ledger: ReceiptLog) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if (
                    self.path != "/accept"
                    or not 1 <= length <= MAX_BODY
                    or self.headers.get("Content-Type") != "application/json"
                ):
                    raise ProbeError("invalid simulated-provider request")
                self.connection.settimeout(2)
                receipt = ledger.accept(json.loads(self.rfile.read(length)))
                payload = json.dumps(receipt, sort_keys=True).encode()
            except Exception:
                self.send_error(400, "simulated-provider request refused")
                return
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, format: str, *args: Any) -> None:
            return

    return Handler


def serve_provider(path: Path, run_id: str, ready: Any, seconds: float) -> None:
    ledger = None
    try:
        if (
            type(seconds) not in (int, float)
            or not math.isfinite(seconds)
            or not 0 < seconds <= 600
        ):
            raise ProbeError("invalid provider lifetime")
        ledger = ReceiptLog(path, run_id)
        with HTTPServer(("127.0.0.1", 0), handler_for(ledger)) as server:
            server.timeout = 0.2
            ready.send({"pid": os.getpid(), "port": server.server_port})
            until = time.monotonic() + seconds
            while time.monotonic() < until:
                server.handle_request()
    except Exception as exc:
        ready.send({"error": type(exc).__name__})
    finally:
        if ledger is not None:
            ledger.close()
        ready.close()


def accept_notification(port: int, run_id: str, notification_id: str) -> dict[str, Any]:
    run_identity(run_id)
    if type(port) is not int or not 1 <= port <= 65535:
        raise ProbeError("invalid simulated-provider port")
    request = Request(
        f"http://127.0.0.1:{port}/accept",
        data=json.dumps(
            {"run_id": run_id, "notification_id": notification_id}
        ).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    # No inherited proxy or cloud transport may route this fixture off the Pod.
    with build_opener(ProxyHandler({})).open(request, timeout=3) as response:
        payload = response.read(MAX_BODY + 1)
        if response.status != 200 or len(payload) > MAX_BODY:
            raise ProbeError("simulated provider did not acknowledge acceptance")
    value = json.loads(payload)
    if (
        not isinstance(value, dict)
        or value.get("run_id") != run_id
        or value.get("notification_id") != notification_id
        or not isinstance(value.get("message_id"), str)
    ):
        raise ProbeError("simulated-provider acknowledgement identity differs")
    return value


def read_receipts(
    path: Path, *, run_id: str, notification_id: str | None = None
) -> list[dict[str, Any]]:
    if (
        not path.is_file()
        or path.is_symlink()
        or path.stat().st_size > MAX_BODY * MAX_RECEIPTS
    ):
        raise ProbeError("independent provider ledger is absent or oversized")
    data = path.read_bytes()
    if data and not data.endswith(b"\n"):
        raise ProbeError("independent provider ledger has a partial receipt")
    values = [json.loads(line) for line in data.splitlines()]
    if len(values) > MAX_RECEIPTS:
        raise ProbeError("independent provider ledger exceeded its budget")
    for index, value in enumerate(values, start=1):
        if (
            not isinstance(value, dict)
            or value.get("run_id") != run_identity(run_id)
            or type(value.get("sequence")) is not int
            or value["sequence"] != index
        ):
            raise ProbeError("independent provider ledger identity or sequence differs")
    return [
        item
        for item in values
        if notification_id is None or item.get("notification_id") == notification_id
    ]

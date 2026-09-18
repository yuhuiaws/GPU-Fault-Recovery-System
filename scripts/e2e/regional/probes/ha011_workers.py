"""Public production worker loops with an isolated, controlled replay endpoint."""

from __future__ import annotations

import json
import os
import secrets
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Lock, Thread
from types import SimpleNamespace
from typing import Any

from gpu_fault.app.processor_factory import ProcessorFactory
from scripts.e2e.regional.ha011_contracts import (
    LEASE_SECONDS,
    MIN_CPU_SECONDS,
    SPOOL_PATH,
    ProofError,
)


class Emitter:
    def __init__(self, connection: Any) -> None:
        self.connection = connection
        self.lock = Lock()

    def emit(self, kind: str, **fields: Any) -> None:
        with self.lock:
            self.connection.send(
                {"kind": kind, "pid": os.getpid(), "at": time.monotonic(), **fields}
            )


class ObservedStore:
    """Forward public Store calls unchanged; claim credentials stay on private IPC."""

    def __init__(self, store: Any, emitter: Emitter) -> None:
        self.store = store
        self.emitter = emitter
        self.idle_reported = False

    def __getattr__(self, name: str) -> Any:
        return getattr(self.store, name)

    def report_claim(
        self, rows: list[Any], owner: str, *, report_idle: bool = True
    ) -> list[Any]:
        for row in rows:
            self.emitter.emit("claim", request_id=row.request_id, owner=owner, item=row)
        if not rows and report_idle and not self.idle_reported:
            self.emitter.emit("idle")
            self.idle_reported = True
        return rows

    def claim_active_processor_requests(self, owner: str, **kwargs: Any) -> list[Any]:
        return self.report_claim(
            self.store.claim_active_processor_requests(owner, **kwargs), owner
        )

    def claim_telemetry_spool(self, owner: str, **kwargs: Any) -> list[Any]:
        return self.report_claim(
            self.store.claim_telemetry_spool(owner, **kwargs),
            owner,
            report_idle=kwargs.get("path") in (None, SPOOL_PATH),
        )

    def complete_active_processor_request(
        self, request_id: str, *args: Any, **kwargs: Any
    ) -> Any:
        result = self.store.complete_active_processor_request(
            request_id, *args, **kwargs
        )
        if result is not None:
            self.emitter.emit("complete", request_id=request_id)
        return result

    def complete_telemetry_spool(self, items: list[Any]) -> int:
        if len(items) != 1:
            raise ProofError("the acceptance spool must use single-item replay batches")
        count = self.store.complete_telemetry_spool(items)
        if count == 1:
            self.emitter.emit("complete", request_id=items[0].request_id)
        return int(count)


class Replay:
    def __init__(self, target: str, release: Any, emitter: Emitter) -> None:
        self.target = target
        self.release = release
        self.emitter = emitter

    def execute(self, identity: str) -> dict[str, Any]:
        if not identity.startswith(self.target.rsplit("-", 1)[0] + "-"):
            raise ProofError("replay received work outside the owned acceptance run")
        if identity == self.target:
            started = time.process_time()
            deadline = time.monotonic() + 60
            announced = False
            while not self.release.is_set():
                # Bounded arithmetic is controlled CPU work, not host saturation.
                sum(index * index for index in range(2000))
                cpu = time.process_time() - started
                if cpu >= MIN_CPU_SECONDS and not announced:
                    self.emitter.emit("busy", request_id=identity, cpu_seconds=cpu)
                    announced = True
                if time.monotonic() >= deadline:
                    raise ProofError("controlled replay barrier expired")
                time.sleep(0.001)
        return {"request_id": identity, "status": 200, "body": {"accepted": identity}}

    def batch(self, request: dict[str, Any]) -> dict[str, Any]:
        return {
            "results": [self.execute(item["request_id"]) for item in request["items"]]
        }


def http_server(replay: Replay, replay_secret: str) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            supplied = self.headers.get("X-GPU-Fault-Processor-Replay", "")
            if not secrets.compare_digest(supplied, replay_secret):
                self.send_error(403)
                return
            identity = self.headers.get("X-GPU-Fault-Processor-Request-ID", "")
            result = replay.execute(identity)
            body = json.dumps(result["body"]).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args: Any) -> None:
            return

    return ThreadingHTTPServer(("127.0.0.1", 0), Handler)


def configure_worker_environment(local_url: str) -> None:
    values = {
        "GPU_FAULT_PROCESSOR_LOCAL_URL": local_url,
        "GPU_FAULT_PROCESSOR_WORKERS": "1",
        "GPU_FAULT_PROCESSOR_FAULT_WORKERS": "1",
        "GPU_FAULT_PROCESSOR_OBSERVATION_WORKERS": "0",
        "GPU_FAULT_PROCESSOR_GPU_TELEMETRY_WORKERS": "0",
        "GPU_FAULT_PROCESSOR_HOST_TELEMETRY_WORKERS": "0",
        "GPU_FAULT_PROCESSOR_REQUEST_LEASE_SECONDS": str(LEASE_SECONDS),
        "GPU_FAULT_PROCESSOR_REQUEST_RENEW_SECONDS": "1",
        "GPU_FAULT_PROCESSOR_REQUEST_MAX_EXECUTION_SECONDS": "60",
        "GPU_FAULT_PROCESSOR_EXIT_ON_DEADLINE": "false",
        "GPU_FAULT_TELEMETRY_SPOOL": "true",
        "GPU_FAULT_TELEMETRY_SPOOL_WORKERS": "1",
        "GPU_FAULT_TELEMETRY_SPOOL_LEASE_SECONDS": str(LEASE_SECONDS),
        "GPU_FAULT_TELEMETRY_SPOOL_REPLAY_BATCH_MAX_ITEMS": "1",
        "GPU_FAULT_TELEMETRY_SPOOL_NOTIFICATION_FALLBACK_SECONDS": "2",
    }
    os.environ.update(values)


def run_worker(
    store: Any, role: str, target: str, connection: Any, stop: Any, release: Any
) -> None:
    emitter = Emitter(connection)
    replay = Replay(target, release, emitter)
    replay_secret = secrets.token_hex(32)
    server = http_server(replay, replay_secret)
    serving = Thread(target=server.serve_forever, daemon=True)
    try:
        serving.start()
    except BaseException:
        server.server_close()
        raise
    processor = None
    try:
        configure_worker_environment(f"http://127.0.0.1:{server.server_port}")
        context = SimpleNamespace(
            store=ObservedStore(store, emitter),
            execution_token=secrets.token_hex(32),
            processor_replay_secret=replay_secret,
        )
        processor = ProcessorFactory(
            context, mode="active-active", exit_grace_seconds=0
        ).build()
        if processor is None:
            raise ProofError("production processor factory did not create a worker")
        processor.telemetry_spool_replay_handler = replay.batch

        def stop_requested() -> None:
            stop.wait()
            release.set()
            processor.stop()

        Thread(target=stop_requested, daemon=True).start()
        emitter.emit("ready", owner=processor.owner_id)
        if role == "processor":
            processor.run_processor()
        elif role == "spool":
            processor.run_telemetry_spool()
        else:
            raise ProofError("unknown acceptance worker role")
    finally:
        stop.set()
        release.set()
        if processor is not None:
            processor.stop()
        server.shutdown()
        server.server_close()
        serving.join(timeout=2)

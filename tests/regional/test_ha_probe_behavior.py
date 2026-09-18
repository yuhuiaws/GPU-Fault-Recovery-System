from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from types import SimpleNamespace
from urllib.error import URLError

import pytest

from gpu_fault.collectors import CollectorError
from scripts.e2e.regional.probes import ha001_probe, ha005_probe, ha010_probe


def test_health_sampler_executes_against_local_readiness_refusals(capsys) -> None:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            payload = {"regional_registry": {"ready": False, "secret_drift": False}}
            self.send_response(503 if self.path == "/healthz" else 200)
            self.end_headers()
            self.wfile.write(json.dumps(payload).encode())

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        assert ha010_probe.main(["probe", str(server.server_port), "0.01", "0.01"]) == 0
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
    report = json.loads(capsys.readouterr().out)
    assert report["samples"]
    assert all(
        item["livez"] == 200 and item["healthz"] == 503 for item in report["samples"]
    ), report
    assert not thread.is_alive(), (
        f"local health server thread {thread.name!r} did not stop within 5 seconds"
    )


def test_sampler_distinguishes_transport_failure_from_readiness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(*args, **kwargs):
        raise URLError("unit transport unavailable")

    monkeypatch.setattr(ha010_probe, "urlopen", fail)
    code, body, error = ha010_probe.fetch("http://127.0.0.1:1/healthz")
    assert code is None and body is None
    assert error is not None


def relocate(
    module, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, names: tuple[str, ...]
) -> None:
    monkeypatch.setattr(module, "STATE", tmp_path)
    for name in names:
        monkeypatch.setattr(module, name, tmp_path / getattr(module, name).name)


@pytest.mark.parametrize("failure", [False, True])
def test_continuity_probe_executes_one_bounded_claim_and_event_cycle(
    failure: bool, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    relocate(ha005_probe, monkeypatch, tmp_path, ("READY", "STATS", "STOP", "OUTBOX"))
    tokens = tmp_path / "fixture-input.json"
    tokens.write_text(
        json.dumps([{"cluster_id": "unit-cluster", "token": "fixture-only"}])
    )
    monkeypatch.setattr(ha005_probe, "Path", lambda _: tokens)
    for name in (
        "RUN_ID",
        "CONTROL_PLANE_URL",
        "EXECUTOR_ARTIFACT_SHA256",
        "EXECUTOR_COMPATIBILITY_DIGEST",
    ):
        monkeypatch.setenv(name, "unit-value")
    monkeypatch.setattr(
        ha005_probe,
        "RegionalExecutorClient",
        lambda *a, **kw: SimpleNamespace(claim=lambda *a, **kw: []),
    )
    posts = []

    class Sink:
        def __init__(self, *args, **kwargs):
            self.evidence = kwargs["evidence"]

        def post(self, path, payload):
            posts.append((path, payload))
            if failure:
                raise CollectorError(
                    "unit refusal", status_code=503, buffered=True, replayable=True
                )
            self.evidence.admissions.append(
                {
                    "batch_id": payload["batch_id"],
                    "accepted": True,
                    "replay": False,
                    "spooled": False,
                    "coalesced": False,
                    "request_id": "request-a",
                }
            )
            return {"accepted": True, "processor_request_id": "request-a"}

        def wait_for_outbox_replay(self, seconds):
            return True

    monkeypatch.setattr(ha005_probe, "ObservedSink", Sink)
    monkeypatch.setattr(ha005_probe.time, "sleep", lambda _: ha005_probe.STOP.touch())
    ha005_probe.main()
    result = json.loads(ha005_probe.STATS.read_text())
    assert result["stopped"] is True
    assert result["counters"]["claim_success"] == 1
    assert result["counters"]["event_attempts"] == 1
    assert len(posts) == 1
    if failure:
        assert result["error_types"]["http-503"] == 1
        assert result["counters"]["event_buffered"] == 1
    else:
        assert result["accepted_request_ids"] == ["request-a"]


@pytest.mark.parametrize("failure", [False, True])
def test_cpu_failover_probe_records_failed_claims_without_faking_success(
    failure: bool, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    relocate(ha001_probe, monkeypatch, tmp_path, ("READY", "STATS", "STOP", "LEDGER"))
    for name in (
        "GPU_FAULT_CONTROL_PLANE_URL",
        "GPU_FAULT_CLUSTER_ID",
        "GPU_FAULT_CONTROL_PLANE_TOKEN",
        "GPU_FAULT_CONTROL_PLANE_CA_FILE",
        "GPU_FAULT_EXECUTOR_ARTIFACT_SHA256",
        "GPU_FAULT_EXECUTOR_COMPATIBILITY_DIGEST",
    ):
        monkeypatch.setenv(name, "fixture-only")
    client = SimpleNamespace(
        cluster_id="unit-cluster", _get=lambda path: {"status": "ok"}
    )
    monkeypatch.setattr(ha001_probe, "RegionalExecutorClient", lambda *a, **kw: client)

    def cycle():
        if failure:
            raise RuntimeError("claim unavailable")
        return 0

    executor = SimpleNamespace(
        run_once=cycle,
        executor_id="unit-executor",
        claimed_total=0,
        reported_failures=0,
        unexpected_failures=0,
        lease_renewal_failures=0,
    )
    monkeypatch.setattr(ha001_probe, "ClusterActionExecutor", lambda *a, **kw: executor)
    monkeypatch.setattr(ha001_probe.time, "sleep", lambda _: ha001_probe.STOP.touch())
    ha001_probe.main()
    result = json.loads(ha001_probe.STATS.read_text())
    assert result["counters"]["health_success"] == 1
    assert result["counters"].get("claim_success", 0) == int(not failure)
    assert result["counters"].get("claim_failure", 0) == int(failure)

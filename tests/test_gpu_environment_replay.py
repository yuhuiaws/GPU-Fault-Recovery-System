"""Only local HTTP fixtures; never redirect a collector or mutate a GPU."""

import io
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest
from prometheus_client.parser import text_string_to_metric_families

from tools import gpu_environment_replay as replay_tool

RAW = (
    b'DCGM_FI_DEV_GPU_TEMP{UUID="GPU-a"} 40\n'
    b'DCGM_FI_DEV_CLOCKS_EVENT_REASONS{UUID="GPU-a"} 0\n'
    b'DCGM_FI_DEV_THERMAL_VIOLATION{UUID="GPU-a"} 0\n'
    b'DCGM_FI_DEV_GPU_TEMP{UUID="GPU-b"} 42\n'
)


def samples(body):
    return {
        (sample.name, sample.labels.get("UUID")): sample.value
        for family in text_string_to_metric_families(body.decode())
        for sample in family.samples
    }


def test_replay_only_changes_target_and_converts_us_to_wire_ns():
    output = replay_tool.transform(
        RAW.decode(),
        "GPU-a",
        {"gpu_temperature_c": 95, "thermal_violation_total_us": 10000},
    )
    observed = samples(output)
    assert observed[("DCGM_FI_DEV_GPU_TEMP", "GPU-a")] == 95
    assert observed[("DCGM_FI_DEV_GPU_TEMP", "GPU-b")] == 42
    assert observed[("DCGM_FI_DEV_THERMAL_VIOLATION", "GPU-a")] == 10_000_000


def test_missing_fields_are_not_invented_and_null_omits_a_field():
    with pytest.raises(ValueError, match="absent"):
        replay_tool.transform(RAW.decode(), "GPU-a", {"gpu_tlimit_c": -3})
    with pytest.raises(ValueError, match="absent"):
        replay_tool.transform(RAW.decode(), "GPU-missing", {"gpu_temperature_c": 95})
    output = replay_tool.transform(RAW.decode(), "GPU-a", {"gpu_temperature_c": None})
    assert ("DCGM_FI_DEV_GPU_TEMP", "GPU-a") not in samples(output), output
    assert ("DCGM_FI_DEV_GPU_TEMP", "GPU-b") in samples(output), output


@pytest.mark.parametrize(
    "url",
    [
        "http://169.254.169.254/metrics",
        "https://127.0.0.1:9400/metrics",
        "http://127.0.0.1:9411/metrics",
        "http://user@127.0.0.1:9400/metrics",
        "http://127.0.0.1:9400/other",
    ],
)
def test_upstream_cannot_leave_loopback_or_recurse(url):
    with pytest.raises(ValueError):
        replay_tool.validate_upstream(url, 9411)


@pytest.mark.parametrize(
    "plan",
    [
        {"steps": [{"reads": True, "values": {}}]},
        {"steps": [{"reads": 0, "values": {}}]},
        {"steps": [{"reads": 1, "values": {"gpu_shutdown_temperature_c": 50}}]},
        {"steps": [{"reads": 1, "values": {"gpu_temperature_c": float("nan")}}]},
    ],
)
def test_plan_refuses_unsafe_or_ambiguous_configuration(plan):
    with pytest.raises(ValueError):
        replay_tool.validate_plan(plan)


def test_default_cli_only_validates_and_serve_requires_ack(tmp_path, monkeypatch):
    path = tmp_path / "plan.json"
    path.write_text(
        json.dumps({"steps": [{"reads": 3, "values": {"gpu_temperature_c": 95}}]})
    )
    calls = []
    monkeypatch.setattr(replay_tool, "serve", lambda *args: calls.append(args))
    arguments = ["--plan", str(path), "--gpu-uuid", "GPU-a"]
    assert replay_tool.main(arguments) == 0
    with pytest.raises(SystemExit) as error:
        replay_tool.main([*arguments, "--serve"])
    assert error.value.code == 2
    assert calls == [], "neither path starts a network endpoint"


@pytest.mark.parametrize(
    "body",
    [
        RAW.replace(b"} 40", b"} 90"),
        RAW.replace(b"} 40", b"} -1"),
        RAW.replace(b"} 40", b"} NaN"),
        RAW.replace(b'REASONS{UUID="GPU-a"} 0', b'REASONS{UUID="GPU-a"} 128'),
    ],
)
def test_real_safety_alarm_aborts_injection_and_is_never_masked(body):
    value = replay_tool.Replay(
        "GPU-a", [{"reads": 3, "values": {"clock_throttle_reasons": 0}}], 600
    )
    value.arm()
    output, receipt = value.render(body)
    assert output == body, "real alarms must pass through unchanged"
    assert receipt["aborted"], receipt
    assert value.values() == {}, "the abort is latched for this run"


def test_deadline_and_completed_plan_pass_through_fresh_source(monkeypatch):
    clock = [10.0]
    monkeypatch.setattr(replay_tool.time, "monotonic", lambda: clock[0])
    value = replay_tool.Replay(
        "GPU-a", [{"reads": 1, "values": {"gpu_temperature_c": 95}}], 10
    )
    assert value.render(RAW)[0] == RAW, "starting the proxy does not arm injection"
    value.arm()
    assert value.render(RAW)[0] != RAW, "the initial frame is injected"
    clock[0] = 21
    assert value.render(RAW)[0] == RAW
    clock[0] = 10
    value.reads = 1
    assert value.render(RAW)[0] == RAW


def test_real_loopback_http_preserves_status_and_handles_upstream_failure(monkeypatch):
    failed = [False]

    def upstream(*args, **kwargs):
        if failed[0]:
            raise OSError("upstream unavailable")
        return io.BytesIO(RAW)

    monkeypatch.setattr(replay_tool, "open_upstream", upstream)
    value = replay_tool.Replay(
        "GPU-a", [{"reads": 1, "values": {"gpu_temperature_c": 95}}], 600
    )
    audit = io.StringIO()
    server = replay_tool.make_server("http://127.0.0.1:9400/metrics", 0, value, audit)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        with urlopen(base + "/status", timeout=5) as response:
            assert json.load(response)["reads"] == 0
        with pytest.raises(HTTPError) as unacknowledged:
            urlopen(Request(base + "/arm", method="POST"), timeout=5)
        assert unacknowledged.value.code == 403
        arm = Request(
            base + "/arm",
            method="POST",
            headers={"X-GFRS-Replay-Ack": replay_tool.ACKNOWLEDGEMENT},
        )
        with urlopen(arm, timeout=5) as response:
            assert response.status == 200
        with pytest.raises(HTTPError) as duplicate:
            urlopen(arm, timeout=5)
        assert duplicate.value.code == 409
        with urlopen(base + "/metrics", timeout=5) as response:
            assert samples(response.read())[("DCGM_FI_DEV_GPU_TEMP", "GPU-a")] == 95
        with urlopen(base + "/metrics", timeout=5) as response:
            assert response.read() == RAW
        failed[0] = True
        with pytest.raises(HTTPError) as error:
            urlopen(base + "/metrics", timeout=5)
        assert error.value.code == 503
        assert value.reads == 2, "failure does not consume another replay frame"
        receipts = [json.loads(line) for line in audit.getvalue().splitlines()]
        assert receipts[1]["physical_fault_injected"] is False
        assert receipts[-1]["error"] == "upstream unavailable"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(5)


def test_upstream_redirect_cannot_reach_another_host(monkeypatch):
    class Redirect(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(302)
            self.send_header("Location", "http://169.254.169.254/latest/meta-data/")
            self.end_headers()

    server = HTTPServer(("127.0.0.1", 0), Redirect)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    # A broken inherited proxy must not intercept loopback test telemetry.
    monkeypatch.setenv("http_proxy", "http://127.0.0.1:1")
    monkeypatch.setenv("no_proxy", "")
    try:
        with pytest.raises(ValueError, match="redirects are forbidden"):
            replay_tool.open_upstream(
                f"http://127.0.0.1:{server.server_address[1]}/metrics", timeout=3
            )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(5)

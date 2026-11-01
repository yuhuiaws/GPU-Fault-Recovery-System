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
#: What dcgm-exporter really publishes: typed families, the violation
#: durations declared ``counter`` (deploy/dataplane/dcgm-counters.csv), so the
#: parser renames their samples ``<name>_total``.
TYPED = (
    b"# TYPE DCGM_FI_DEV_GPU_TEMP gauge\n"
    b'DCGM_FI_DEV_GPU_TEMP{gpu="0",UUID="GPU-a"} 29\n'
    b'DCGM_FI_DEV_GPU_TEMP{gpu="1",UUID="GPU-b"} 31\n'
    b"# TYPE DCGM_FI_DEV_CLOCKS_EVENT_REASONS gauge\n"
    b'DCGM_FI_DEV_CLOCKS_EVENT_REASONS{gpu="0",UUID="GPU-a"} 1\n'
    b'DCGM_FI_DEV_CLOCKS_EVENT_REASONS{gpu="1",UUID="GPU-b"} 1\n'
    b"# TYPE DCGM_FI_DEV_POWER_VIOLATION counter\n"
    b'DCGM_FI_DEV_POWER_VIOLATION{gpu="0",UUID="GPU-a"} 0\n'
    b'DCGM_FI_DEV_POWER_VIOLATION{gpu="1",UUID="GPU-b"} 0\n'
    b"# TYPE DCGM_FI_DEV_THERMAL_VIOLATION counter\n"
    b'DCGM_FI_DEV_THERMAL_VIOLATION{gpu="0",UUID="GPU-a"} 0\n'
    b'DCGM_FI_DEV_THERMAL_VIOLATION{gpu="1",UUID="GPU-b"} 5000\n'
)
#: GF-ENV-LIVE-006 ``hw_thermal``, its third read
#: (``env_live_plans.thermal_duration_plan``: 0x40 and a rising duration in us).
HW_THERMAL_STEP = {
    "clock_throttle_reasons": 64,
    "thermal_violation_total_us": 2_000_000.0,
}


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


def test_counter_families_are_matched_by_their_bare_name_and_reach_the_product():
    """Live 2026-10-26, GF-ENV-LIVE-006 ``hw_thermal`` on plain EKS: the tool
    looked the parsed ``DCGM_FI_DEV_THERMAL_VIOLATION_total`` samples up under
    the bare exporter name, found nothing, refused every armed read with 503
    and the collector consumed 0 of 7 reads. The lookup now falls back like the
    collector's own parser, and the transformed text reaches that parser."""

    from gpu_fault.collectors.gpu.dcgm import DCGM_METRICS, parse_dcgm_samples

    assert (
        replay_tool.canonical_mapping("DCGM_FI_DEV_THERMAL_VIOLATION_total")
        == DCGM_METRICS["DCGM_FI_DEV_THERMAL_VIOLATION"]
    )
    assert (
        replay_tool.canonical_mapping("DCGM_FI_DEV_THERMAL_VIOLATION")
        == (DCGM_METRICS["DCGM_FI_DEV_THERMAL_VIOLATION"])
    )
    assert replay_tool.canonical_mapping("DCGM_FI_DEV_NOTHING_total") is None
    output = replay_tool.transform(TYPED.decode(), "GPU-a", HW_THERMAL_STEP)
    seen = {
        (sample.gpu_uuid, sample.canonical_name): sample.value
        for sample in parse_dcgm_samples(output.decode())
    }
    assert seen[("GPU-a", "thermal_violation_total_us")] == 2_000_000.0, (
        "plan us -> wire ns -> collector us"
    )
    assert seen[("GPU-a", "clock_throttle_reasons")] == 64.0
    assert seen[("GPU-b", "thermal_violation_total_us")] == 5.0, (
        "the other GPU keeps its own reading (5000 ns)"
    )
    assert seen[("GPU-b", "clock_throttle_reasons")] == 1.0
    assert seen[("GPU-a", "power_violation_total_us")] == 0.0, (
        "an unplanned counter passes through"
    )
    assert seen[("GPU-a", "gpu_temperature_c")] == 29.0
    # A field the exporter does not publish is still never invented.
    with pytest.raises(ValueError, match="absent"):
        replay_tool.transform(TYPED.decode(), "GPU-a", {"gpu_tlimit_c": -3})


def test_a_refused_read_is_counted_on_status_and_consumes_no_step(monkeypatch):
    # The pre-fix GF-ENV-LIVE-006 shape over real loopback HTTP: a plan naming
    # a field the transform cannot find is answered 503 on every armed read,
    # ``reads`` never advances, and ``/status`` now says so.
    monkeypatch.setattr(replay_tool, "open_upstream", lambda *a, **k: io.BytesIO(RAW))
    value = replay_tool.Replay(
        "GPU-a", [{"reads": 2, "values": {"gpu_tlimit_c": -3}}], 600
    )
    audit = io.StringIO()
    server = replay_tool.make_server("http://127.0.0.1:9400/metrics", 0, value, audit)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        with urlopen(base + "/status", timeout=5) as response:
            before = json.load(response)
        assert before["failed_reads"] == 0 and before["last_error"] is None
        arm = Request(
            base + "/arm",
            method="POST",
            headers={"X-GFRS-Replay-Ack": replay_tool.ACKNOWLEDGEMENT},
        )
        with urlopen(arm, timeout=5) as response:
            assert response.status == 200
        for _attempt in range(2):
            with pytest.raises(HTTPError) as refused:
                urlopen(base + "/metrics", timeout=5)
            assert refused.value.code == 503
        with urlopen(base + "/status", timeout=5) as response:
            status = json.load(response)
        assert status["reads"] == 0 and status["armed"] is True
        assert status["failed_reads"] == 2
        assert "requested fields are absent" in status["last_error"]
        receipts = [json.loads(line) for line in audit.getvalue().splitlines()]
        assert [item.get("read") for item in receipts[-2:]] == [1, 1]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(5)


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
        with urlopen(base + "/status", timeout=5) as response:
            status = json.load(response)
        assert status["reads"] == 2 and status["failed_reads"] == 1
        assert status["last_error"] == "upstream unavailable"
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


#: The live H200 exporter's temperature fields (driver 580, DCGM 4.7.0): the
#: reading, the T.Limit margin and field 159, the absolute shutdown limit.
H200_EXPORTER = (
    "# TYPE DCGM_FI_DEV_GPU_TEMP gauge\n"
    'DCGM_FI_DEV_GPU_TEMP{gpu="0",UUID="GPU-a"} 28\n'
    "# TYPE DCGM_FI_DEV_GPU_TEMP_LIMIT gauge\n"
    'DCGM_FI_DEV_GPU_TEMP_LIMIT{gpu="0",UUID="GPU-a"} 59\n'
    "# TYPE DCGM_FI_DEV_SHUTDOWN_TEMP gauge\n"
    'DCGM_FI_DEV_SHUTDOWN_TEMP{gpu="0",UUID="GPU-a"} 95\n'
)
H200_XML = (
    "<nvidia_smi_log><gpu><uuid>GPU-a</uuid><minor_number>0</minor_number>"
    "<temperature><gpu_temp>28 C</gpu_temp><gpu_temp_tlimit>59 C</gpu_temp_tlimit>"
    "<gpu_temp_max_tlimit_threshold>-8 C</gpu_temp_max_tlimit_threshold>"
    "<gpu_temp_slow_tlimit_threshold>-2 C</gpu_temp_slow_tlimit_threshold>"
    "</temperature></gpu></nvidia_smi_log>"
)


def test_the_shutdown_replay_raises_the_graded_reading_and_never_the_limit():
    """GF-ENV-LIVE-011 on a T.Limit H200 (batch 232).

    The plan raises DCGM_FI_DEV_GPU_TEMP, the reading the product grades
    against the DCGM field 159 limit; the limit is hardware metadata the tool
    refuses to change and passes through, so the collector publishes the
    device's own 95 C and the one replayed read is the emergency.
    """

    import subprocess
    from datetime import datetime, timezone

    from gpu_fault.collectors.gpu.dcgm import DcgmMetricsCollector, parse_dcgm_samples
    from gpu_fault.collectors.gpu.discovery import query_nvidia_temperature_limits
    from gpu_fault.collectors.gpu.shutdown_limit import DCGM_SHUTDOWN_CANDIDATE
    from gpu_fault.collectors.models import CollectorContext
    from gpu_fault.gpu_metrics import GpuMetricBatch, GpuMetricsService

    output = replay_tool.transform(H200_EXPORTER, "GPU-a", {"gpu_temperature_c": 96.0})
    seen = {
        item.canonical_name: item.value for item in parse_dcgm_samples(output.decode())
    }
    assert seen == {
        "gpu_temperature_c": 96.0,
        "gpu_tlimit_c": 59.0,
        DCGM_SHUTDOWN_CANDIDATE: 95.0,
    }, "only the graded reading moves; the T.Limit offset never substitutes"
    for limit in (DCGM_SHUTDOWN_CANDIDATE, "gpu_shutdown_temperature_c"):
        with pytest.raises(ValueError, match="hardware limit metadata"):
            replay_tool.validate_plan(
                {"steps": [{"reads": 1, "values": {limit: 80.0}}]}
            )

    service = GpuMetricsService()
    findings = []

    class Sink:
        def post(self, path, payload):
            result = service.ingest(GpuMetricBatch.model_validate(payload))
            findings.extend(result.new_findings)
            return result.model_dump(mode="json")

    def xml_runner(argv, **_kwargs):
        return subprocess.CompletedProcess(argv, 0, stdout=H200_XML, stderr="")

    DcgmMetricsCollector(
        Sink(), CollectorContext(cluster_id="c"), node_id="node-a"
    ).collect_text(
        output.decode(),
        observed_at=datetime.now(timezone.utc),
        extra_samples=query_nvidia_temperature_limits(xml_runner),
    )
    assert [
        (item.canonical_name, item.severity, item.confirmation_count)
        for item in findings
    ] == [("gpu_temperature_c", "CRITICAL", 1)], findings

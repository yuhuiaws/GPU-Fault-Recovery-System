from __future__ import annotations

import argparse
import io
import json
import urllib.error
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts.e2e.regional.probes import destr019_node_probe as agent
from scripts.e2e.regional.probes import warm_spare_node_probe as warm
from tests.regional._cov95_destr_probes import RUN_ID, ProbeHarness
from tests.regional._destr008_service_window import Host


@pytest.mark.parametrize("service", sorted(warm.ALLOWED_SERVICES))
@pytest.mark.parametrize("delay", [0, 15])
def test_service_stop_arms_recovery_first_and_restore_cancels_delayed_stop_first(
    service: str, delay: int, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("INVOCATION_ID", "a" * 32)
    h = Host(tmp_path, monkeypatch)
    h.binding = h.make_binding(service, delay)
    h.window = warm.ServiceWindow(h.binding)
    h.arm()
    report = h.window.schedule_stop()
    assert report["phase"] == "SCHEDULED" and report["stop_requested"] is False
    assert report["restore_at"] == int(h.now) + 180 + delay
    assert ("stop", "--no-block", "--job-mode=fail", service) not in h.mutations()
    h.fire_stop()
    report = h.window.restore()
    assert report["phase"] == "RESTORED" and report["after"]["ActiveState"] == "active"
    commands = h.mutations()
    assert commands.index(("stop", h.window.units["stop-timer"])) < commands.index(
        ("start", "--no-block", "--job-mode=fail", service)
    )
    assert report["stop_quiescence"]["service"]["job_id"] == 0
    assert h.window.cleanup()["restore_quiescence"]["cgroup_empty"] is True


@pytest.mark.parametrize("defect", ["inactive", "stop-failed", "restore-failed"])
def test_service_refuses_unsafe_baseline_and_reports_unrestored_state(
    defect: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("INVOCATION_ID", "a" * 32)
    h = Host(tmp_path, monkeypatch)
    service = "gpu-fault-node-agent.service"
    h.binding = h.make_binding(service, 0)
    h.window = warm.ServiceWindow(h.binding)
    if defect == "inactive":
        h.units[service].update(ActiveState="inactive", MainPID="0", SubState="dead")
        with pytest.raises(warm.ProbeError, match="baseline"):
            h.window.prepare()
        assert h.mutations() == []
    elif defect == "stop-failed":
        h.arm()
        h.window.schedule_stop()
        h.fail.add(("stop", "--no-block", "--job-mode=fail", service))
        with pytest.raises(warm.ProbeError, match="systemctl failed"):
            h.fire_stop()
        assert h.units[service]["ActiveState"] == "active"
        assert h.window.targets["restore-service"].exists(), h.window.targets
    else:
        h.stopped()
        h.pending_start = True
        with pytest.raises(warm.ProbeError, match="quiescent"):
            h.window.restore()
        assert h.units[service]["Job"] == "91"
        assert ("stop", h.window.units["restore-service"]) not in h.mutations()


@pytest.mark.parametrize(
    ("flag", "value"),
    [
        ("--restore-seconds", "59"),
        ("--restore-seconds", "601"),
        ("--stop-delay-seconds", "-1"),
        ("--stop-delay-seconds", "121"),
    ],
)
def test_service_main_rejects_unbounded_windows_before_any_command(
    flag: str,
    value: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    h = Host(tmp_path, monkeypatch)
    before = list(h.calls)
    code = warm.main(
        [
            "stop-with-failsafe",
            "--run-id",
            RUN_ID,
            "--service",
            "kubelet.service",
            flag,
            value,
        ]
    )
    report = json.loads(capsys.readouterr().out)
    assert code == 1 and report["error"] == "ProbeError", report
    assert h.calls == before


def test_service_snapshot_and_command_error_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    h = Host(tmp_path, monkeypatch)
    code = warm.main(["snapshot", "--service", "gpu-fault-node-agent.service"])
    report = json.loads(capsys.readouterr().out)
    assert code == 0 and report["state"]["ActiveState"] == "active", report
    with pytest.raises(warm.ProbeError, match="allowlist"):
        warm.service_name("foreign.service")
    with pytest.raises(warm.ProbeError, match="unsafe"):
        warm.safe_id("invalid;id", "run ID")
    h.fail.add(("show", "kubelet.service"))
    with pytest.raises(warm.ProbeError, match="systemctl failed"):
        warm.systemctl("show", "kubelet.service")


def test_agent_probe_restart_has_fail_safe_before_restart_and_disarms_last(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = ProbeHarness(agent, tmp_path, monkeypatch)
    args = argparse.Namespace(run_id=RUN_ID, restore_seconds=180)
    agent.restart_agent(args)
    report = h.records[-1]
    assert (
        report["before"]["MainPID"] == "100" and report["after"]["MainPID"] == "101"
    ), report
    commands = [cmd for cmd, _ in h.calls]
    timer = next(i for i, cmd in enumerate(commands) if cmd[0] == "systemd-run")
    restart = commands.index(["systemctl", "restart", agent.AGENT_UNIT])
    assert timer < restart, commands
    assert json.loads(h.state.read_text())["restarted_at"], h.state
    agent.disarm_restore(args)
    assert h.records[-1]["restore_timer"]["ActiveState"] == "inactive", h.records
    assert json.loads(h.state.read_text())["disarmed_at"], h.state
    code, report = h.main(monkeypatch, "snapshot", "--run-id", RUN_ID)
    assert code == 0 and report["boot_id"] == "boot-before", report


@pytest.mark.parametrize("defect", ["inactive", "unchanged-pid", "never-active"])
def test_agent_restart_refuses_missing_baseline_or_unconfirmed_new_process(
    defect: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = ProbeHarness(agent, tmp_path, monkeypatch)
    if defect == "inactive":
        h.agent_active = False
    elif defect == "unchanged-pid":
        h.restart_changes_pid = False
    else:
        h.agent_start_stuck = True
    code, report = h.main(
        monkeypatch, "restart-agent", "--run-id", RUN_ID, "--restore-seconds", "180"
    )
    assert code == 1 and "error" in report, report
    if defect == "inactive":
        assert not h.state.exists(), h.state
    else:
        assert json.loads(h.state.read_text())["armed_at"], h.state


@pytest.mark.parametrize("active", [False, True])
def test_agent_ensure_active_starts_only_when_needed(
    active: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = ProbeHarness(agent, tmp_path, monkeypatch)
    h.agent_active = active
    agent.ensure_agent_active(argparse.Namespace(run_id=RUN_ID))
    assert h.records[-1]["started"] is not active, h.records
    assert h.records[-1]["after"]["ActiveState"] == "active", h.records
    h.agent_active = False
    h.agent_start_stuck = True
    with pytest.raises(agent.ProbeError, match="not active"):
        agent.ensure_agent_active(argparse.Namespace(run_id=RUN_ID))


@pytest.mark.parametrize("body", [b'{"status":"ok"}', b"", b"not-json"])
@pytest.mark.parametrize("http_error", [False, True])
def test_agent_health_uses_fake_verified_transport_and_handles_non_json(
    body: bytes, http_error: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = ProbeHarness(agent, tmp_path, monkeypatch)
    requests: list[Any] = []
    monkeypatch.setattr(
        agent,
        "agent_env",
        lambda: {"GPU_FAULT_NODE_ADVERTISE_URL": "https://node.invalid"},
    )

    class Response(io.BytesIO):
        status = 200

    def open_url(request: Any, **kwargs: Any) -> Any:
        requests.append((request, kwargs))
        if http_error:
            raise urllib.error.HTTPError(
                request.full_url, 503, "fake unavailable", {}, io.BytesIO(body)
            )
        return Response(body)

    monkeypatch.setattr(agent.urllib.request, "urlopen", open_url)
    agent.agent_health(argparse.Namespace())
    report = h.records[-1]
    assert report["http_status"] == (503 if http_error else 200), report
    assert requests[0][0].full_url == "https://node.invalid/healthz", requests
    assert requests[0][1]["timeout"] == 10, requests
    assert report["payload"] == (
        {"status": "ok"}
        if body.startswith(b"{")
        else {"raw": "not-json"}
        if body
        else {}
    ), report


def test_agent_health_pins_certificate_and_keeps_tls_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = ProbeHarness(agent, tmp_path, monkeypatch)
    certificate = str(tmp_path / "public-certificate.pem")
    trusted: list[str] = []
    context = SimpleNamespace(check_hostname=True)

    def tls(*, cafile: str) -> Any:
        trusted.append(cafile)
        return context

    monkeypatch.setattr(agent, "ssl", SimpleNamespace(create_default_context=tls))
    monkeypatch.setattr(
        agent, "agent_env", lambda: {"GPU_FAULT_NODE_AGENT_TLS_CERT": certificate}
    )

    class Response(io.BytesIO):
        status = 200

    def open_url(request: Any, **kwargs: Any) -> Any:
        assert kwargs["context"] is context, kwargs
        assert request.full_url == "https://127.0.0.1:9099/healthz", request.full_url
        return Response(b"{}")

    monkeypatch.setattr(agent.urllib.request, "urlopen", open_url)
    agent.agent_health(argparse.Namespace())
    assert trusted == [certificate] and h.records[-1]["http_status"] == 200, h.records


def test_agent_journal_filters_by_command_and_ignores_unstructured_lines(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = ProbeHarness(agent, tmp_path, monkeypatch)
    output = "\n".join(
        [
            "not-json",
            json.dumps({"MESSAGE": "ordinary service line"}),
            json.dumps(
                {
                    "MESSAGE": "node action accepted command_id=owned attempt=1",
                    "_PID": "10",
                }
            ),
            json.dumps({"MESSAGE": "node action started command_id=other attempt=1"}),
        ]
    )
    original = h.run

    def run(command: list[str], **kwargs: Any) -> Any:
        if command[0] == "journalctl":
            return SimpleNamespace(returncode=0, stdout=output, stderr="")
        return original(command, **kwargs)

    monkeypatch.setattr(agent.subprocess, "run", run)
    agent.journal(argparse.Namespace(command_id="owned", since_epoch=0.0))
    assert h.records[-1]["line_count"] == 1, h.records
    assert h.records[-1]["lines"][0]["fields"]["command_id"] == "owned", h.records

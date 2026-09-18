from __future__ import annotations

import io
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts.e2e.regional import run_ha010_aurora_blackout_liveness as ha010
from tests.regional._cov95_ha010_harness import T0, HA010Harness


@pytest.mark.parametrize("mode", ["timeout", "empty", "malformed"])
def test_sampler_collection_reports_process_failure_without_inventing_samples(
    mode: str,
) -> None:
    calls = []

    def communicate(**kwargs: Any) -> tuple[str, str]:
        calls.append("communicate")
        if mode == "timeout" and len(calls) == 1:
            raise subprocess.TimeoutExpired("unit", 1)
        return ("invalid-json" if mode == "malformed" else "", "unit stderr")

    process = SimpleNamespace(
        communicate=communicate, returncode=7, kill=lambda: calls.append("kill")
    )
    result = ha010.Sampler("unit", process, T0).collect(1)
    assert result["returncode"] == 7
    assert "samples" not in result
    assert ("kill" in calls) is (mode == "timeout")


@pytest.mark.parametrize("stdin_present,alive", [(False, True), (True, False)])
def test_sampler_start_failure_reaps_without_rekilling_an_exited_child(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, stdin_present: bool, alive: bool
) -> None:
    calls = []

    class Broken(io.StringIO):
        def write(self, text: str) -> int:
            raise BrokenPipeError("unit")

    process = SimpleNamespace(
        stdin=Broken() if stdin_present else None,
        poll=lambda: None if alive else 1,
        kill=lambda: calls.append("kill"),
        wait=lambda **kw: calls.append("wait"),
    )
    monkeypatch.setattr(
        ha010,
        "subprocess",
        SimpleNamespace(**{**vars(subprocess), "Popen": lambda *a, **kw: process}),
    )
    regional = SimpleNamespace(
        settings=SimpleNamespace(cpu_kubeconfig=tmp_path / "cpu", namespace="unit")
    )
    with pytest.raises((RuntimeError, BrokenPipeError), match="stdin|unit"):
        ha010.start_sampler(
            regional, {"name": "unit", "port": 8080}, duration_seconds=120
        )
    assert calls == (["kill", "wait"] if alive else ["wait"])


@pytest.mark.parametrize("remote_allowed", [False, True])
def test_cleanup_does_not_accept_running_samplers_and_limits_remote_actions(
    tmp_path: Path, remote_allowed: bool
) -> None:
    calls = []
    process = SimpleNamespace(
        poll=lambda: None, kill=lambda: None, wait=lambda **kw: None
    )

    def unavailable(*args: Any, **kwargs: Any) -> str:
        calls.append("rollout")
        raise OSError("unit rollout unavailable")

    run = SimpleNamespace(
        samplers=[ha010.Sampler("stuck", process, T0)],
        case_dir=tmp_path,
        preflight={"runtime_identity": {}},
        regional=SimpleNamespace(
            kubectl=unavailable,
            verify_runtime_identity=lambda *a, **kw: calls.append("verify") or {},
        ),
    )
    result = ha010.cleanup_case(run, remote_allowed=remote_allowed)
    assert result["samplers_stopped"] == {"stuck": False}
    assert any("still running" in error for error in result["errors"]), result
    assert calls == (["rollout", "verify"] if remote_allowed else [])


def test_health_collection_skips_unready_or_unaddressable_pods() -> None:
    calls = []
    regional = SimpleNamespace(
        kubectl=lambda *args, **kw: calls.append(args[3]) or '{"http_status":200}'
    )
    result = ha010.healthz_by_pod(
        regional,
        {
            "deployment": [
                {"name": "unready", "ready": False, "port": 8080},
                {"name": "no-port", "ready": True, "port": None},
                {"name": "ready", "ready": True, "port": 8080},
            ]
        },
    )
    assert result == {"ready": {"http_status": 200}}
    assert calls == ["ready"]
    assert ha010.deployments_scaled_to_zero(
        SimpleNamespace(kubectl=lambda *a: "0")
    ) == frozenset(ha010.verdicts.CPU_DEPLOYMENTS)


def test_no_ready_ingress_refuses_before_sampler_creation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    harness = HA010Harness(monkeypatch, tmp_path)
    for record in harness.pods[ha010.verdicts.ROLLED_DEPLOYMENT]:
        record["status"]["conditions"][0]["status"] = "False"
    with pytest.raises(RuntimeError, match="no Ready Pod"):
        harness.execute()
    assert harness.processes == []


@pytest.mark.parametrize("field", ["deleted_at", "rds_available_at"])
def test_missing_action_timestamp_cannot_satisfy_the_final_evidence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, field: str
) -> None:
    harness = HA010Harness(monkeypatch, tmp_path)
    request = ha010.request_failover

    def incomplete(run: Any) -> dict:
        value = request(run)
        setattr(run, field, None)
        return value

    monkeypatch.setattr(ha010, "request_failover", incomplete)
    code, report = harness.execute()
    assert code == 1
    assert "deletion" in report["error"] or "failover" in report["error"], report[
        "error"
    ]
    assert all("wait" in process.calls for process in harness.processes), (
        "missing evidence must still reap every owned sampler"
    )

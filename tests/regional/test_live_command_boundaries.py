from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest

from scripts.e2e.regional.acceptance_runner_common import replica_vanished
from scripts.e2e.regional.regional_live_fixture import (
    RegionalCommandTimeout,
    RegionalFixtureError,
    RegionalLiveFixture,
    RegionalLiveSettings,
)


def process_is_running(pid: int) -> bool:
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    except (FileNotFoundError, ProcessLookupError):
        return False
    return fields[0] not in {"Z", "X"}


def test_fixture_timeout_stops_its_local_descendants(tmp_path: Path) -> None:
    receipt = tmp_path / "child-pid"
    script = (
        "import pathlib,subprocess,sys,time;"
        "child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(3)'],"
        "stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL);"
        "pathlib.Path(sys.argv[1]).write_text(str(child.pid));"
        "time.sleep(30)"
    )
    try:
        with pytest.raises(RegionalCommandTimeout):
            RegionalLiveFixture.run(
                [sys.executable, "-c", script, str(receipt)], timeout=1
            )
        assert receipt.exists(), "the owned child must start before the timeout"
        child = int(receipt.read_text())
        assert not process_is_running(child), (
            "the fixture returned while its timed-out command could still act"
        )
    finally:
        # The deliberate red-case child also has a short self-imposed lifetime.
        time.sleep(3.1)


def test_command_failure_does_not_echo_argv_or_unkeyed_stderr() -> None:
    secret = "only-a-local-command-fixture-secret"
    with pytest.raises(RegionalFixtureError) as failed:
        RegionalLiveFixture.run(
            [
                sys.executable,
                "-c",
                "import sys; sys.stderr.write(sys.argv[1]); sys.exit(7)",
                secret,
            ]
        )
    assert secret not in str(failed.value), "command diagnostics exposed private output"


def test_timeout_diagnostic_does_not_echo_arguments() -> None:
    secret = "only-a-local-timeout-fixture-secret"
    failure = RegionalCommandTimeout(["kubectl", "exec", secret], 1)
    assert secret not in str(failure), "timeout diagnostics exposed command arguments"


def test_private_diagnostics_preserve_a_verified_gone_pod_reason() -> None:
    with pytest.raises(RegionalFixtureError) as failed:
        RegionalLiveFixture.run(
            [
                sys.executable,
                "-c",
                "import sys; sys.stderr.write('Error from server (NotFound): "
                'pods "test-pod" not found\'); sys.exit(1)',
            ]
        )
    assert replica_vanished(failed.value), (
        "safe diagnostics lost the Pod-specific reason"
    )


def fixture(tmp_path: Path) -> RegionalLiveFixture:
    kubeconfig = tmp_path / "unused-kubeconfig"
    kubeconfig.touch()
    return RegionalLiveFixture(
        RegionalLiveSettings(
            cpu_kubeconfig=kubeconfig,
            gpu_kubeconfig=kubeconfig,
            gpu_context="unused-context",
            namespace="test-namespace",
            cluster_id="test-cluster",
            region="test-region",
        )
    )


@pytest.mark.parametrize(
    ("plane", "component"), [("cpu", "control-plane"), ("gpu", "executor")]
)
def test_pod_probe_uses_installed_component_interpreter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, plane: str, component: str
) -> None:
    regional = fixture(tmp_path)
    calls = []
    monkeypatch.setattr(regional, "ready_pod", lambda *_args: "test-pod")

    def kubectl(*args, **kwargs):
        calls.append((args, kwargs))
        return '{"ok": true}\n'

    monkeypatch.setattr(regional, "kubectl", kubectl)
    assert regional.pod_python(plane, "test-app", "print('proof')", attempts=1) == {
        "ok": True
    }
    args, kwargs = calls[0]
    assert f"/opt/gpu-fault/{component}/bin/python" in args, calls
    assert "python3" not in args and "python" not in args, calls
    assert kwargs["input_text"] == "print('proof')", calls


def ready_pod() -> dict:
    return {
        "metadata": {"name": "test-pod", "uid": "test-pod-uid"},
        "spec": {"containers": [{"name": "api"}, {"name": "sidecar"}]},
        "status": {
            "phase": "Running",
            "conditions": [{"type": "Ready", "status": "True"}],
            "containerStatuses": [
                {"name": "api", "ready": True},
                {"name": "sidecar", "ready": True},
            ],
        },
    }


@pytest.mark.parametrize("defect", ["terminating", "gate", "partial", "string"])
def test_ready_pods_never_uses_a_partial_or_terminating_ready_proof(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, defect: str
) -> None:
    regional = fixture(tmp_path)
    pod = ready_pod()
    if defect == "terminating":
        pod["metadata"]["deletionTimestamp"] = "2026-01-01T00:00:00Z"
    elif defect == "gate":
        pod["status"]["conditions"][0]["status"] = "False"
    elif defect == "partial":
        pod["status"]["containerStatuses"].pop()
    else:
        pod["status"]["containerStatuses"][0]["ready"] = "false"
    monkeypatch.setattr(
        regional, "kubectl", lambda *_args, **_kwargs: json.dumps({"items": [pod]})
    )
    assert regional.ready_pods("cpu", "test-app") == []


@pytest.mark.parametrize("response", [{}, {"items": None}, {"items": {}}])
def test_malformed_pod_inventory_is_not_an_empty_cluster(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, response: dict
) -> None:
    regional = fixture(tmp_path)
    monkeypatch.setattr(
        regional, "kubectl", lambda *_args, **_kwargs: json.dumps(response)
    )
    with pytest.raises(RegionalFixtureError):
        regional.ready_pods("cpu", "test-app")


def test_complete_pod_inventory_remains_usable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    regional = fixture(tmp_path)
    monkeypatch.setattr(
        regional,
        "kubectl",
        lambda *_args, **_kwargs: json.dumps({"items": [ready_pod()]}),
    )
    assert regional.ready_pods("cpu", "test-app") == [
        {"name": "test-pod", "uid": "test-pod-uid", "node": None}
    ]

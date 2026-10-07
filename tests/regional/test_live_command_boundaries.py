from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest

from scripts.e2e.regional import regional_pod_inventory as inventory_module
from scripts.e2e.regional.acceptance_runner_common import replica_vanished
from scripts.e2e.regional.regional_live_fixture import (
    RegionalCommandTimeout,
    RegionalFixtureError,
    RegionalLiveFixture,
    RegionalLiveSettings,
)
from scripts.e2e.regional.regional_pod_inventory import running_pod, running_pods


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


def running_not_ready_pod() -> dict:
    """An executor Pod of a cluster the case cut off from the control plane:
    Running, every container running, readiness failing by design."""

    pod = ready_pod()
    pod["status"]["conditions"][0]["status"] = "False"
    for status in pod["status"]["containerStatuses"]:
        status["ready"] = False
        status["state"] = {"running": {"startedAt": "2026-01-01T00:00:00Z"}}
    return pod


def test_running_pods_accept_a_not_ready_pod_whose_containers_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    regional = fixture(tmp_path)
    monkeypatch.setattr(
        regional,
        "kubectl",
        lambda *_args, **_kwargs: json.dumps({"items": [running_not_ready_pod()]}),
    )

    assert regional.ready_pods("gpu", "test-app") == [], (
        "readiness is still the control plane's signal; a NotReady Pod is not Ready"
    )
    assert [item["name"] for item in running_pods(regional, "gpu", "test-app")] == [
        "test-pod"
    ]
    assert running_pod(regional, "gpu", "test-app") == "test-pod"


@pytest.mark.parametrize(
    "defect", ["terminating", "pending", "waiting-container", "partial"]
)
def test_running_pods_still_exclude_pods_that_cannot_answer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, defect: str
) -> None:
    from scripts.e2e.regional.regional_commands import RegionalFixtureError

    regional = fixture(tmp_path)
    pod = running_not_ready_pod()
    if defect == "terminating":
        pod["metadata"]["deletionTimestamp"] = "2026-01-01T00:00:00Z"
    elif defect == "pending":
        pod["status"]["phase"] = "Pending"
    elif defect == "waiting-container":
        pod["status"]["containerStatuses"][1]["state"] = {
            "waiting": {"reason": "CrashLoopBackOff"}
        }
    else:
        pod["status"]["containerStatuses"].pop()
    monkeypatch.setattr(
        regional, "kubectl", lambda *_args, **_kwargs: json.dumps({"items": [pod]})
    )

    assert running_pods(regional, "gpu", "test-app") == []
    with pytest.raises(
        RegionalFixtureError, match="no Running gpu Pod for app=test-app"
    ):
        running_pod(regional, "gpu", "test-app")


def test_pod_python_execs_into_a_running_pod_only_when_told(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    regional = fixture(tmp_path)
    selections: list[str] = []
    calls: list[tuple] = []

    def ready_pod_name(*_args: object) -> str:
        selections.append("ready")
        return "ready-pod"

    def running_pod_name(_fixture: object, *_args: object) -> str:
        selections.append("running")
        return "running-pod"

    def kubectl(*args, **kwargs):
        calls.append(args)
        return '{"ok": true}\n'

    monkeypatch.setattr(regional, "ready_pod", ready_pod_name)
    monkeypatch.setattr(inventory_module, "running_pod", running_pod_name)
    monkeypatch.setattr(regional, "kubectl", kubectl)

    regional.pod_python("gpu", "test-app", "print()", attempts=1)
    regional.pod_python("gpu", "test-app", "print()", attempts=1, require_ready=False)

    assert selections == ["ready", "running"], "Ready is the default selection"
    assert "ready-pod" in calls[0] and "running-pod" in calls[1], calls


def test_command_failure_names_the_exception_class_but_not_its_message() -> None:
    """Live 2026-10-07 (gpu-eks-cluster DESTR-003 cleanup): the spare release
    probe raised StopIteration inside the executor Pod and the evidence read
    only ``command failed (1): <sensitive output redacted> (step: kubectl exec)``.
    The class of the failure is not a value; the message may be."""

    from scripts.e2e.regional.regional_commands import (
        RegionalCommandFailed,
        exec_refused_before_script,
        failure_class,
    )

    secret = "only-a-local-traceback-fixture-secret"
    traceback = (
        "Traceback (most recent call last):\n"
        '  File "<stdin>", line 9, in <module>\n'
        f"RuntimeError: no spare coordinator for {secret}\n"
    )
    failed = RegionalCommandFailed(1, traceback, ["kubectl", "exec", "-i", "pod"])
    assert "[RuntimeError]" in str(failed), str(failed)
    assert "(step: kubectl exec)" in str(failed), str(failed)
    assert secret not in str(failed), "the message of the exception is a value"
    assert exec_refused_before_script(failed) is False, (
        "the script ran and raised; another Pod would run it again"
    )

    assert failure_class("Traceback ...\nStopIteration\n") == "StopIteration"
    refused = RegionalCommandFailed(
        1,
        f"error: unable to upgrade connection: container not found ({secret})",
        ["kubectl", "exec", "-i", "pod"],
    )
    assert "[kubectl: unable to upgrade connection]" in str(refused), str(refused)
    assert secret not in str(refused)
    assert exec_refused_before_script(refused) is True, (
        "kubectl never attached; the script did not run"
    )
    assert failure_class("") == "" and failure_class(None) == ""
    assert failure_class(f"some unclassified text {secret}") == ""

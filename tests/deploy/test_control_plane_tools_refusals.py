"""Refusal and fallback paths of the two small control-plane apply tools.

``apply_control_plane_deployment`` must refuse anything that is not one of the
three CPU role Deployments, surface every kubectl failure as a ``ValueError``
and never replace a legacy Deployment whose UID it cannot bind. The pin wait
rejects malformed arguments before any kubectl runs and keeps reading Pods
whose list or port layout is unusual.
"""

from __future__ import annotations

import io
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from tests._script_loader import lazy_script_module

ROOT = Path(__file__).resolve().parents[2]
APPLY = lazy_script_module(
    ROOT / "deploy/control-plane/tools/apply_control_plane_deployment.py"
)
WAIT = lazy_script_module(
    ROOT / "deploy/control-plane/tools/wait_control_plane_pins.py"
)
COMMAND = ["kubectl", "--kubeconfig", "/example/cpu", "-n", "gpu-fault-system"]
DIGEST = "a" * 64


def deployment(
    name: str = "gpu-fault-api-ha",
    *,
    kind: str = "Deployment",
    env: list[dict[str, str]] | None = None,
    uid: str | None = "uid-1",
) -> dict[str, Any]:
    container: dict[str, Any] = {"name": "api"}
    if env is not None:
        container["env"] = env
    metadata: dict[str, Any] = {"name": name, "namespace": "gpu-fault-system"}
    if uid is not None:
        metadata["uid"] = uid
    return {
        "kind": kind,
        "metadata": metadata,
        "spec": {"template": {"spec": {"containers": [container]}}},
    }


class Kubectl:
    """Scripted ``subprocess.run`` answering apply, dry-run and replace."""

    def __init__(self, answers: dict[str, subprocess.CompletedProcess[str]]) -> None:
        self.answers = answers
        self.calls: list[tuple[list[str], Any]] = []

    def __call__(self, arguments: list[str], **options: Any) -> Any:
        self.calls.append((list(arguments), json.loads(options["input"])))
        verb = "dry-run" if "--dry-run=server" in arguments else arguments[-3]
        return self.answers[verb]


def completed(code: int, stdout: str = "", stderr: str = "") -> Any:
    return subprocess.CompletedProcess(["kubectl"], code, stdout=stdout, stderr=stderr)


@pytest.mark.parametrize(
    "desired",
    [deployment("gpu-fault-collector"), deployment(kind="StatefulSet")],
    ids=["foreign-name", "foreign-kind"],
)
def test_apply_refuses_anything_but_a_cpu_role_deployment(
    monkeypatch: pytest.MonkeyPatch, desired: dict[str, Any]
) -> None:
    kubectl = Kubectl({})
    monkeypatch.setattr(subprocess, "run", kubectl)
    with pytest.raises(ValueError, match="expected a CPU role Deployment"):
        APPLY.apply_deployment(desired, {"items": []}, COMMAND)
    assert kubectl.calls == []


def test_apply_failure_is_a_value_error_with_the_diagnostic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kubectl = Kubectl({"apply": completed(1, stderr="server refused the apply")})
    monkeypatch.setattr(subprocess, "run", kubectl)
    with pytest.raises(ValueError, match="apply failed.*server refused"):
        APPLY.apply_deployment(deployment(), {"items": []}, COMMAND)
    assert kubectl.calls[0][0] == [*COMMAND, "apply", "-f", "-"]


def legacy_live(uid: str | None = "uid-1") -> dict[str, Any]:
    return {
        "items": [
            deployment(env=[{"name": "GPU_FAULT_ALLOW_EMAIL", "value": "x"}], uid=uid)
        ]
    }


def test_legacy_cleanup_requires_the_live_deployment_uid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kubectl = Kubectl({})
    monkeypatch.setattr(subprocess, "run", kubectl)
    with pytest.raises(ValueError, match="no UID"):
        APPLY.apply_deployment(deployment(), legacy_live(uid=None), COMMAND)
    assert kubectl.calls == [], "nothing may be sent before the identity is bound"


def test_legacy_cleanup_dry_run_failure_stops_the_replacement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kubectl = Kubectl({"dry-run": completed(1, stderr="admission webhook denied")})
    monkeypatch.setattr(subprocess, "run", kubectl)
    with pytest.raises(ValueError, match="dry-run failed.*admission webhook"):
        APPLY.apply_deployment(deployment(), legacy_live(), COMMAND)
    assert [call[0][-5:] for call in kubectl.calls] == [
        ["--dry-run=server", "-f", "-", "-o", "json"]
    ]


def merged(*containers: dict[str, Any]) -> str:
    return json.dumps(
        {
            "kind": "Deployment",
            "metadata": {
                "name": "gpu-fault-api-ha",
                "namespace": "gpu-fault-system",
                "uid": "uid-1",
                "resourceVersion": "9",
            },
            "spec": {"template": {"spec": {"containers": list(containers)}}},
        }
    )


def test_legacy_cleanup_keeps_containers_without_env_and_rejects_hard_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preview = merged(
        {"name": "api", "env": [{"name": "GPU_FAULT_ALLOW_EMAIL", "value": "x"}]},
        {"name": "sidecar"},
    )
    kubectl = Kubectl(
        {
            "dry-run": completed(0, stdout=preview),
            "replace": completed(1, stderr="Forbidden: field is immutable"),
        }
    )
    monkeypatch.setattr(subprocess, "run", kubectl)
    with pytest.raises(ValueError, match="replacement failed.*Forbidden"):
        APPLY.apply_deployment(deployment(), legacy_live(), COMMAND)
    replaced = kubectl.calls[-1][1]
    assert replaced["spec"]["template"]["spec"]["containers"] == [
        {"name": "api", "env": []},
        {"name": "sidecar"},
    ], "only the legacy names are removed; env-less containers stay untouched"
    assert len(kubectl.calls) == 2, "a non-conflict failure is not retried"


def test_main_requires_a_scoped_kubectl_command(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    live = tmp_path / "live.json"
    live.write_text(json.dumps({"items": []}), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["apply", "--live", str(live)])
    with pytest.raises(SystemExit) as refused:
        APPLY.main()
    assert refused.value.code == 2


def test_main_applies_stdin_with_the_remainder_command(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    live = tmp_path / "live.json"
    live.write_text(json.dumps({"items": []}), encoding="utf-8")
    kubectl = Kubectl({"apply": completed(0, stdout="deployment configured\n")})
    monkeypatch.setattr(subprocess, "run", kubectl)
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(deployment())))
    monkeypatch.setattr(sys, "argv", ["apply", "--live", str(live), "--", *COMMAND])
    APPLY.main()
    assert kubectl.calls[0][0] == [*COMMAND, "apply", "-f", "-"]
    assert capsys.readouterr().out == "deployment configured\n"


def test_main_reports_refusals_as_a_system_exit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    live = tmp_path / "live.json"
    live.write_text(json.dumps({"items": []}), encoding="utf-8")
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(deployment("other"))))
    monkeypatch.setattr(sys, "argv", ["apply", "--live", str(live), *COMMAND])
    with pytest.raises(SystemExit, match="CPU role apply refused"):
        APPLY.main()


# --- wait_control_plane_pins argument validation ---------------------------


def wait_arguments(**overrides: str) -> list[str]:
    values = {
        "--namespace": "gpu-fault-system",
        "--expected-sha256": DIGEST,
        "--timeout-seconds": "30",
        "--deployments": "gpu-fault-api-ha,gpu-fault-control-worker",
        "--poll-seconds": "1",
    }
    values.update(overrides)
    return [item for pair in values.items() for item in pair]


@pytest.mark.parametrize(
    "overrides",
    [
        {"--expected-sha256": "A" * 64},
        {"--timeout-seconds": "29"},
        {"--poll-seconds": "0"},
        {"--namespace": "Gpu_Fault"},
        {"--deployments": "gpu-fault-api-ha,gpu-fault-api-ha"},
    ],
    ids=["uppercase-digest", "short-timeout", "zero-poll", "bad-name", "duplicate"],
)
def test_wait_rejects_invalid_arguments_before_running_kubectl(
    monkeypatch: pytest.MonkeyPatch, overrides: dict[str, str]
) -> None:
    def forbidden(*_arguments: Any, **_options: Any) -> Any:
        raise AssertionError("argument validation must not reach kubectl")

    monkeypatch.setattr(subprocess, "run", forbidden)
    with pytest.raises(SystemExit) as refused:
        WAIT.parse_arguments(wait_arguments(**overrides))
    assert refused.value.code == 2


def test_wait_accepts_the_minimum_timeout_and_keeps_names() -> None:
    options = WAIT.parse_arguments(wait_arguments())
    assert options.timeout_seconds == 30
    assert options.deployments == ("gpu-fault-api-ha", "gpu-fault-control-worker")
    assert options.kubeconfig is None


def test_wait_reports_a_pod_list_without_items_as_a_read_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        subprocess, "run", lambda *_a, **_k: completed(0, stdout=json.dumps({}))
    )
    options = WAIT.parse_arguments(
        wait_arguments(**{"--deployments": "gpu-fault-api-ha"})
    )
    readings, empty = WAIT.poll_once(options, {})
    assert empty == []
    assert [(reading.pod, reading.error) for reading in readings] == [
        (None, "kubectl returned no Pod list")
    ]
    assert readings[0].line() == "gpu-fault-api-ha: error: kubectl returned no Pod list"


def test_http_port_skips_unnamed_ports_and_falls_back_per_deployment() -> None:
    pod = {
        "spec": {
            "containers": [
                {"ports": [{"name": "metrics", "containerPort": 9100}]},
                {"ports": [{"name": "http", "containerPort": "8080"}]},
            ]
        }
    }
    assert WAIT.http_port(pod, "gpu-fault-control-worker", {}) == 8081
    assert WAIT.http_port(pod, "gpu-fault-api-ha", {}) == 8080
    assert WAIT.http_port(pod, "gpu-fault-unknown", {}) == 8080
    assert WAIT.http_port(pod, WAIT.SPOOL_WORKER_DEPLOYMENT, {}) == 8082
    assert (
        WAIT.http_port(
            pod,
            WAIT.SPOOL_WORKER_DEPLOYMENT,
            {WAIT.SPOOL_WORKER_METRICS_PORT_VARIABLE: "9090"},
        )
        == 9090
    )

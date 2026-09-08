"""The rollback verifications judge the previous release by its own snapshot.

Automatic rollback restores the previous release's control-plane Deployments
with their container ``env``/``envFrom`` verbatim (release 459bcc1), because
the previous image fails fast on ``GPU_FAULT_*`` names it does not know. Two
checks in the apply path still judged that restored shape by the *current*
release's rules: ``gpu_fault.config_cli validate`` on the render, and
``verify_control_plane_role_split.py`` on the live Deployments. A release that
drops an env name, tightens a validator or changes a hard-coded port would have
failed a correct rollback.

With ``GPU_FAULT_ROLE_SPLIT_CONTAINER_ENV_FILE`` set both switch to snapshot
mode. These tests pin the verifier's half and the apply script's wiring:

* snapshot mode passes when the live env equals the snapshot, every container
  runs ``GPU_FAULT_RUNTIME_IMAGE`` and every tier is fully ready;
* it fails on env drift (naming Deployment/container/name), on a tier with
  fewer ready than desired replicas or none at all, on an image mismatch, and
  on a missing role Deployment;
* it skips the current-contract assertions -- a worker without
  ``GPU_FAULT_PROCESSOR_WORKERS`` fails default mode and passes snapshot mode;
* it fails closed on a malformed snapshot;
* default mode (variable unset) is byte-for-byte today's behaviour;
* the apply script passes ``--container-env-snapshot`` to the lint only when
  the variable is set, and hands the variable on to the verifier.

Nothing here talks to kubectl: ``subprocess.run`` is replaced by a recorder
that answers ``kubectl get deployment`` from a dict.
"""

from __future__ import annotations

import copy
import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from gpu_fault.container_env_snapshot import pod_container_env
from gpu_fault_release import regional_release_rollback_context as ROLLBACK_CONTEXT
from tests._script_loader import load_script_module

ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "deploy/control-plane/tools"
VERIFIER = TOOLS / "verify_control_plane_role_split.py"
APPLY_SCRIPT = TOOLS / "apply-control-plane-role-split.sh"
VARIABLE = "GPU_FAULT_ROLE_SPLIT_CONTAINER_ENV_FILE"
IMAGE = "123456789012.dkr.ecr.us-west-2.amazonaws.com/gpu-fault:previous"
UVICORN = "exec python -m uvicorn gpu_fault.app:app --no-proxy-headers "


def _deployment(
    name: str,
    container_name: str,
    env: dict[str, str],
    args: str,
    *,
    replicas: int,
    ready: int | None = None,
) -> dict[str, Any]:
    return {
        "metadata": {"name": name},
        "spec": {
            "replicas": replicas,
            "template": {
                "spec": {
                    "containers": [
                        {
                            "name": container_name,
                            "image": IMAGE,
                            "args": [args],
                            "env": [
                                {"name": key, "value": value}
                                for key, value in env.items()
                            ],
                            "envFrom": [
                                {"configMapRef": {"name": f"{name}-config-core"}}
                            ],
                        }
                    ]
                }
            },
        },
        "status": {"readyReplicas": replicas if ready is None else ready},
    }


def _live_control_plane() -> dict[str, dict[str, Any]]:
    """Three Deployments that satisfy every current-contract assertion."""

    return {
        "gpu-fault-api-ha": _deployment(
            "gpu-fault-api-ha",
            "api",
            {
                "GPU_FAULT_SERVICE_ROLE": "ingress",
                "GPU_FAULT_TELEMETRY_SPOOL": "true",
                "GPU_FAULT_POSTGRES_POOL_MAX_SIZE": "40",
                "GPU_FAULT_STORE_IO_WORKERS": "8",
                "GPU_FAULT_FAULT_STORE_IO_WORKERS": "8",
                "GPU_FAULT_EVIDENCE_STORE_IO_WORKERS": "8",
                "GPU_FAULT_TELEMETRY_SPOOL_STORE_IO_WORKERS": "8",
                "GPU_FAULT_TELEMETRY_SPOOL_MAX_ITEM_BYTES": "65536",
            },
            UVICORN + "--port 8080 --workers 4",
            replicas=2,
        ),
        "gpu-fault-control-worker": _deployment(
            "gpu-fault-control-worker",
            "control-worker",
            {
                "GPU_FAULT_SERVICE_ROLE": "worker",
                "GPU_FAULT_TELEMETRY_SPOOL": "false",
                "GPU_FAULT_PROCESSOR_WORKERS": "8",
            },
            UVICORN + "--port 8081 --workers 1",
            replicas=1,
        ),
        "gpu-fault-telemetry-spool-worker": _deployment(
            "gpu-fault-telemetry-spool-worker",
            "telemetry-spool-worker",
            {
                "GPU_FAULT_SERVICE_ROLE": "spool-worker",
                "GPU_FAULT_TELEMETRY_SPOOL": "true",
                "GPU_FAULT_TELEMETRY_SPOOL_MAX_ITEM_BYTES": "65536",
                "GPU_FAULT_TELEMETRY_SPOOL_REPLAY_BATCH_MAX_BYTES": "1048576",
                "GPU_FAULT_TELEMETRY_SPOOL_MAX_IN_FLIGHT_BYTES": "4194304",
                "GPU_FAULT_TELEMETRY_SPOOL_WORKERS": "4",
                "GPU_FAULT_TELEMETRY_SPOOL_REPLAY_BATCH_MAX_ITEMS": "64",
                "GPU_FAULT_TELEMETRY_SPOOL_NOTIFICATION_FALLBACK_SECONDS": "3",
            },
            UVICORN + "--port 8082 --workers 1",
            replicas=1,
        ),
    }


def _env_list(deployment: dict[str, Any]) -> list[dict[str, Any]]:
    (container,) = deployment["spec"]["template"]["spec"]["containers"]
    return list(container["env"])


def _drop_env(deployment: dict[str, Any], name: str) -> None:
    (container,) = deployment["spec"]["template"]["spec"]["containers"]
    container["env"] = [item for item in container["env"] if item["name"] != name]


class _Kubectl:
    """Answers ``kubectl -n <ns> get deployment|configmap <name> -o json``."""

    def __init__(self, deployments: dict[str, dict[str, Any]]) -> None:
        self.deployments = deployments
        self.calls: list[list[str]] = []

    def __call__(
        self, command: list[str], **_kwargs: Any
    ) -> subprocess.CompletedProcess:
        assert command[:2] == ["kubectl", "-n"], command
        kind, name = command[4], command[5]
        self.calls.append(command)
        if kind == "configmap":
            return subprocess.CompletedProcess(
                command, 0, stdout=json.dumps({"data": {}}), stderr=""
            )
        assert kind == "deployment", command
        if name not in self.deployments:
            return subprocess.CompletedProcess(
                command,
                1,
                stdout="",
                stderr=f'Error from server (NotFound): deployments "{name}" not found\n',
            )
        return subprocess.CompletedProcess(
            command, 0, stdout=json.dumps(self.deployments[name]), stderr=""
        )


@pytest.fixture
def verifier(monkeypatch):
    module = load_script_module(VERIFIER)
    monkeypatch.delenv(VARIABLE, raising=False)
    monkeypatch.delenv("GPU_FAULT_RUNTIME_IMAGE", raising=False)
    return module


def _install(monkeypatch, verifier, live: dict[str, dict[str, Any]]) -> _Kubectl:
    kubectl = _Kubectl(live)
    monkeypatch.setattr(verifier.subprocess, "run", kubectl)
    return kubectl


def _snapshot_file(tmp_path: Path, snapshot: object) -> str:
    path = tmp_path / "previous-container-env.json"
    path.write_text(json.dumps(snapshot), encoding="utf-8")
    return str(path)


def _snapshot_of(live: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """What the engine captured, detached from the live objects the tests mutate."""

    return copy.deepcopy({name: pod_container_env(item) for name, item in live.items()})


def test_default_mode_holds_the_current_contract(monkeypatch, verifier, capsys):
    """Unchanged behaviour without the variable: the fixture passes, and the
    tier that claims from the queue must declare its pool size."""

    live = _live_control_plane()
    _install(monkeypatch, verifier, live)
    assert verifier.main() == 0, capsys.readouterr().out
    out = capsys.readouterr().out
    assert "role-split check passed: ingress 2 replicas on 8080" in out
    assert "snapshot mode" not in out

    _drop_env(live["gpu-fault-control-worker"], "GPU_FAULT_PROCESSOR_WORKERS")
    assert verifier.main() == 1
    out = capsys.readouterr().out
    assert "gpu-fault-control-worker has no GPU_FAULT_PROCESSOR_WORKERS" in out


def test_snapshot_mode_skips_the_current_contract_and_passes_on_a_match(
    monkeypatch, verifier, capsys, tmp_path
):
    """The previous release ran without GPU_FAULT_PROCESSOR_WORKERS and with a
    name today's registry does not know; both are what the snapshot says."""

    live = _live_control_plane()
    _drop_env(live["gpu-fault-control-worker"], "GPU_FAULT_PROCESSOR_WORKERS")
    _env_list(live["gpu-fault-api-ha"])  # unchanged; documents the shape
    live["gpu-fault-api-ha"]["spec"]["template"]["spec"]["containers"][0]["env"].append(
        {"name": "GPU_FAULT_ONLY_THE_OLD_IMAGE_KNOWS", "value": "1"}
    )
    kubectl = _install(monkeypatch, verifier, live)
    monkeypatch.setenv(VARIABLE, _snapshot_file(tmp_path, _snapshot_of(live)))
    monkeypatch.setenv("GPU_FAULT_RUNTIME_IMAGE", IMAGE)

    assert verifier.main() == 0
    lines = capsys.readouterr().out.strip().splitlines()
    assert lines[0].startswith(
        "role-split check running in container env snapshot mode"
    ), lines
    assert VARIABLE in lines[0]
    assert "contract assertions are skipped" in lines[0]
    assert lines[1].startswith("role-split check passed (snapshot mode)"), lines
    assert "gpu-fault-control-worker 1 replicas ready" in lines[1]
    # Snapshot mode never resolves ConfigMaps: it compares references, not values.
    assert all(call[4] == "deployment" for call in kubectl.calls), kubectl.calls


def test_snapshot_mode_ignores_env_order(monkeypatch, verifier, tmp_path):
    live = _live_control_plane()
    snapshot = _snapshot_of(live)
    for item in live.values():
        item["spec"]["template"]["spec"]["containers"][0]["env"].reverse()
    _install(monkeypatch, verifier, live)
    monkeypatch.setenv(VARIABLE, _snapshot_file(tmp_path, snapshot))

    assert verifier.main() == 0


def test_snapshot_mode_fails_on_env_drift_naming_the_entry(
    monkeypatch, verifier, capsys, tmp_path
):
    live = _live_control_plane()
    snapshot = _snapshot_of(live)
    container = live["gpu-fault-api-ha"]["spec"]["template"]["spec"]["containers"][0]
    for item in container["env"]:
        if item["name"] == "GPU_FAULT_SERVICE_ROLE":
            item["value"] = "worker"
    _install(monkeypatch, verifier, live)
    monkeypatch.setenv(VARIABLE, _snapshot_file(tmp_path, snapshot))

    assert verifier.main() == 1
    out = capsys.readouterr().out
    assert (
        "role-split check failed: gpu-fault-api-ha/api env GPU_FAULT_SERVICE_ROLE "
        "differs from the snapshot"
    ) in out

    container["envFrom"] = []
    assert verifier.main() == 1
    assert "gpu-fault-api-ha/api envFrom[0] differs from the snapshot" in (
        capsys.readouterr().out
    )


@pytest.mark.parametrize(
    ("replicas", "ready", "problem"),
    [
        (1, 0, "gpu-fault-control-worker has 0/1 ready"),
        (2, 1, "gpu-fault-control-worker has 1/2 ready"),
        (0, 0, "gpu-fault-control-worker is scaled to zero"),
    ],
)
def test_snapshot_mode_requires_every_tier_fully_ready(
    monkeypatch, verifier, capsys, tmp_path, replicas, ready, problem
):
    live = _live_control_plane()
    live["gpu-fault-control-worker"]["spec"]["replicas"] = replicas
    live["gpu-fault-control-worker"]["status"]["readyReplicas"] = ready
    _install(monkeypatch, verifier, live)
    monkeypatch.setenv(VARIABLE, _snapshot_file(tmp_path, _snapshot_of(live)))

    assert verifier.main() == 1
    assert f"role-split check failed: {problem}" in capsys.readouterr().out


def test_snapshot_mode_checks_the_runtime_image_when_given(
    monkeypatch, verifier, capsys, tmp_path
):
    live = _live_control_plane()
    live["gpu-fault-telemetry-spool-worker"]["spec"]["template"]["spec"]["containers"][
        0
    ]["image"] = IMAGE.replace("previous", "current")
    _install(monkeypatch, verifier, live)
    monkeypatch.setenv(VARIABLE, _snapshot_file(tmp_path, _snapshot_of(live)))

    assert verifier.main() == 0, "no expected image, no image assertion"
    capsys.readouterr()

    monkeypatch.setenv("GPU_FAULT_RUNTIME_IMAGE", IMAGE)
    assert verifier.main() == 1
    assert (
        "gpu-fault-telemetry-spool-worker container telemetry-spool-worker "
        "runtime image does not match GPU_FAULT_RUNTIME_IMAGE"
    ) in capsys.readouterr().out


def test_snapshot_mode_still_requires_the_three_role_deployments(
    monkeypatch, verifier, capsys, tmp_path
):
    live = _live_control_plane()
    snapshot = _snapshot_of(live)
    live.pop("gpu-fault-control-worker")
    _install(monkeypatch, verifier, live)
    monkeypatch.setenv(VARIABLE, _snapshot_file(tmp_path, snapshot))

    assert verifier.main() == 1
    out = capsys.readouterr().out
    assert (
        "gpu-fault-control-worker is missing: nothing claims from the processor queue"
        in out
    )


def test_snapshot_mode_fails_closed_on_a_malformed_snapshot(
    monkeypatch, verifier, tmp_path
):
    live = _live_control_plane()
    _install(monkeypatch, verifier, live)
    snapshot = _snapshot_of(live)
    snapshot["gpu-fault-api-ha"]["api"].pop("envFrom")
    monkeypatch.setenv(VARIABLE, _snapshot_file(tmp_path, snapshot))

    with pytest.raises(SystemExit, match="snapshot is invalid") as excinfo:
        verifier.main()
    assert "must carry exactly env and envFrom lists" in str(excinfo.value)

    monkeypatch.setenv(VARIABLE, str(tmp_path / "missing.json"))
    with pytest.raises(SystemExit, match="cannot read"):
        verifier.main()


def test_verifier_names_the_same_variable_as_the_engine_and_renderer(verifier):
    assert verifier.CONTAINER_ENV_FILE_VARIABLE == VARIABLE
    assert ROLLBACK_CONTEXT.CONTAINER_ENV_FILE_VARIABLE == VARIABLE


def test_apply_script_passes_the_snapshot_to_both_verifications() -> None:
    """The lint gets ``--container-env-snapshot`` only when the variable is set;
    the verifier gets the variable itself."""

    script = APPLY_SCRIPT.read_text(encoding="utf-8")
    guard = script.index('if [[ -n "${GPU_FAULT_ROLE_SPLIT_CONTAINER_ENV_FILE:-}" ]]')
    flag = script.index(
        '--container-env-snapshot "${GPU_FAULT_ROLE_SPLIT_CONTAINER_ENV_FILE}"', guard
    )
    end_of_guard = script.index("\nfi\n", flag)
    lint = script.index('gpu_fault.config_cli validate "${config_validate_args[@]}"')
    assert guard < flag < end_of_guard < lint, "flag must be added inside the guard"
    assert script.count("--container-env-snapshot") == 1

    verify = script.index('python3 "${SCRIPT_DIR}/verify_control_plane_role_split.py"')
    handoff = script.rindex(
        'GPU_FAULT_ROLE_SPLIT_CONTAINER_ENV_FILE="${GPU_FAULT_ROLE_SPLIT_CONTAINER_ENV_FILE:-}"',
        0,
        verify,
    )
    assert lint < handoff < verify

    result = subprocess.run(
        ["bash", "-n", str(APPLY_SCRIPT)], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr


def test_snapshot_shape_matches_what_the_engine_captures() -> None:
    """The verifier's loader and the engine's capture agree on the shape, so a
    file the engine wrote is one the verifier reads."""

    live = _live_control_plane()
    snapshot = _snapshot_of(live)
    validated = ROLLBACK_CONTEXT.previous_container_env_snapshot(
        copy.deepcopy(snapshot)
    )
    assert validated == snapshot


def test_snapshot_mode_allows_a_drained_spool_tier_only_when_the_snapshot_disabled_it(
    monkeypatch, verifier, capsys, tmp_path
):
    """A site that ran with spool admission off keeps its spool tier at zero.

    The exemption is read from the snapshot's own ingress env, not from the
    current contract; with admission on in the snapshot a zero is still a
    failure.
    """

    live = _live_control_plane()
    live["gpu-fault-telemetry-spool-worker"]["spec"]["replicas"] = 0
    live["gpu-fault-telemetry-spool-worker"]["status"]["readyReplicas"] = 0
    (api,) = live["gpu-fault-api-ha"]["spec"]["template"]["spec"]["containers"]
    for item in api["env"]:
        if item["name"] == "GPU_FAULT_TELEMETRY_SPOOL":
            item["value"] = "false"
    _install(monkeypatch, verifier, live)
    monkeypatch.setenv(VARIABLE, _snapshot_file(tmp_path, _snapshot_of(live)))

    assert verifier.main() == 0, capsys.readouterr().out

    admitting = _live_control_plane()
    admitting["gpu-fault-telemetry-spool-worker"]["spec"]["replicas"] = 0
    admitting["gpu-fault-telemetry-spool-worker"]["status"]["readyReplicas"] = 0
    _install(monkeypatch, verifier, admitting)
    monkeypatch.setenv(VARIABLE, _snapshot_file(tmp_path, _snapshot_of(admitting)))

    assert verifier.main() == 1
    assert (
        "role-split check failed: gpu-fault-telemetry-spool-worker is scaled to zero"
        in capsys.readouterr().out
    )

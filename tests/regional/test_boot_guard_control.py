from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import boot_guard_control as control
from scripts.e2e.regional import live_driver_guard as guard
from scripts.e2e.regional.regional_live_fixture import predecessor_evidence


@pytest.mark.parametrize("invalid", ["missing", "container", "extra", "node"])
def test_api_gate_cannot_accept_an_unchanged_but_unhealthy_population(
    invalid: str,
) -> None:
    pods = [
        {
            "metadata": {"name": f"api-{index}", "uid": f"uid-{index}"},
            "spec": {"nodeName": f"node-{index}", "containers": [{"name": "api"}]},
            "status": {
                "phase": "Running",
                "conditions": [{"type": "Ready", "status": "True"}],
                "containerStatuses": [{"name": "api", "ready": True}],
            },
        }
        for index in range(3)
    ]
    deployment = {"spec": {"replicas": 3}}
    assert len(control.healthy_api_pods(deployment, {"items": pods})) == 3, (
        "the complete three-node control should pass"
    )
    if invalid == "missing":
        pods.pop()
    elif invalid == "container":
        pods[0]["status"]["containerStatuses"][0]["ready"] = False
    elif invalid == "extra":
        pods.append({**pods[0], "metadata": {"name": "extra", "uid": "extra"}})
    else:
        pods[1]["spec"]["nodeName"] = pods[0]["spec"]["nodeName"]
    with pytest.raises(RuntimeError, match="replicas"):
        control.healthy_api_pods(deployment, {"items": pods})


def scope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, valid: bool = True
) -> tuple[list[str], Path, dict[str, Any]]:
    environment = {
        "RUN_DIR": str(tmp_path),
        "CPU_KUBECONFIG": "/dev/null",
        "NAMESPACE": "synthetic",
        "AWS_REGION": "us-west-2",
        "CPU_HYPERPOD_CLUSTER": "synthetic-cpu",
        "BOOT_GUARD_START_CASE": "1",
        "GUARD_PROBE_BASE": str(tmp_path / "base.json"),
    }
    for name, value in environment.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(control, "install_site_profile", lambda: None)
    monkeypatch.setattr(guard, "applied_site_profile", lambda: None)
    monkeypatch.setattr(guard, "source_digest", lambda: "a" * 64)
    identity = {
        "release_id": "release-a",
        "cpu_deployment_uid": "cpu-uid",
        "generation": 3,
    }
    monkeypatch.setattr(control, "target_identity", lambda _env: identity)
    monkeypatch.setattr(
        control,
        "predecessor",
        lambda *_a: {"valid": valid, "evidence_cluster_id": "cluster-a"},
    )
    arguments = ["boot-guard-control", "--run-dir", str(tmp_path)]
    monkeypatch.setattr(sys, "argv", arguments)
    path = tmp_path / "cases/GF-REGIONAL-BOOT-001"
    return arguments, path, identity


def execute_flags() -> list[str]:
    return [
        "--execute",
        "--confirm",
        control.CONFIRMATION,
        "--maintenance-window-end",
        (datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat(),
    ]


def test_shell_guard_has_bound_schema3_plan_and_canonical_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    arguments, path, _identity = scope(tmp_path, monkeypatch)

    assert control.main() == 0, "the valid read-only prerequisite should plan"
    plan = json.loads((path / "plan.json").read_text(encoding="utf-8"))
    assert plan["schema_version"] == 3 and plan["preflight_passed"] is True, (
        "shell execution must use the same guarded plan schema"
    )
    assert "GF-REGIONAL-BOOT-006" not in plan["details"]["cases"], (
        "retired means unexecutable"
    )
    monkeypatch.setattr(sys, "argv", [*arguments, *execute_flags()])
    monkeypatch.setenv("BOOT_GUARD_RECORD_CASE", "GF-REGIONAL-BOOT-001")
    monkeypatch.setenv("BOOT_GUARD_RECORD_VERDICT", "FAIL")
    assert control.main() == 0, "begin must invalidate any old PASS"
    evidence = path / "GF-REGIONAL-BOOT-001.json"
    assert predecessor_evidence(evidence, "GF-REGIONAL-BOOT-001")["valid"] is False, (
        "a running shell case cannot authorize its successor"
    )
    monkeypatch.setenv("BOOT_GUARD_RECORD_VERDICT", "PASS")
    assert control.main() == 0, "the completed shell case should record its verdict"
    assert (
        predecessor_evidence(
            evidence,
            "GF-REGIONAL-BOOT-001",
            release_id="release-a",
            cluster_id="cluster-a",
        )["valid"]
        is True
    ), "the next family must consume the canonical bound JSON"


@pytest.mark.parametrize("invalid", ["preflight", "target", "window", "scope"])
def test_shell_guard_refuses_failed_or_changed_authorization(
    invalid: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    arguments, _path, identity = scope(
        tmp_path, monkeypatch, valid=invalid != "preflight"
    )
    assert control.main() == (1 if invalid == "preflight" else 0), (
        "plan must preserve preflight"
    )
    flags = execute_flags()
    if invalid == "target":
        identity["cpu_deployment_uid"] = "another-cpu"
    elif invalid == "window":
        flags[-1] = "2020-01-01T00:00:00+00:00"
    elif invalid == "scope":
        monkeypatch.setenv("NAMESPACE", "another-namespace")
    monkeypatch.setattr(sys, "argv", [*arguments, *flags])

    with pytest.raises(RuntimeError, match="preflight|drifted|window"):
        control.main()


def _identity_kubectl(release_state: dict[str, Any] | None):
    """A live-shaped CPU control plane: release-metadata carries compatibility
    digests only (no ``release-id`` key, live 2026-09-20); the release id lives in
    ``gpu-fault-regional-release-state`` ``state.json`` like every other case reads it."""

    pods = {
        "items": [
            {
                "metadata": {"name": f"api-{index}", "uid": f"uid-{index}"},
                "spec": {"nodeName": f"node-{index}", "containers": [{"name": "api"}]},
                "status": {
                    "phase": "Running",
                    "conditions": [{"type": "Ready", "status": "True"}],
                    "containerStatuses": [{"name": "api", "ready": True}],
                },
            }
            for index in range(3)
        ]
    }
    deployment = {
        "metadata": {"uid": "cpu-uid", "generation": 7},
        "spec": {"replicas": 3},
    }
    metadata = {"data": {"required-agent-protocol-version": "4"}}
    documents = {
        ("configmap", "gpu-fault-release-metadata"): metadata,
        ("deployment", "gpu-fault-api-ha"): deployment,
    }
    if release_state is not None:
        documents[("configmap", "gpu-fault-regional-release-state")] = {
            "data": {"state.json": json.dumps(release_state)}
        }

    def run(command: list[str], **_kwargs: Any) -> Any:
        assert command[:2] == ["kubectl", "--kubeconfig"], command
        kind = command[command.index("get") + 1]
        if kind == "pod":
            return type("Completed", (), {"stdout": json.dumps(pods)})()
        name = command[command.index("get") + 2]
        if (kind, name) not in documents:
            raise RuntimeError(f"configmaps {name!r} not found")
        return type("Completed", (), {"stdout": json.dumps(documents[(kind, name)])})()

    return run


def test_target_identity_reads_the_release_id_every_other_case_binds_to(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = {"CPU_KUBECONFIG": "/dev/null", "NAMESPACE": "synthetic"}
    monkeypatch.setattr(
        control,
        "run",
        _identity_kubectl({"phase": "complete", "release_id": "0e813e5eaf80"}),
    )
    identity = control.target_identity(environment)
    assert identity["release_id"] == "0e813e5eaf80"
    assert identity["cpu_deployment_uid"] == "cpu-uid" and identity["generation"] == 7
    assert len(identity["cpu_pods"]) == 3
    monkeypatch.setattr(control, "run", _identity_kubectl({"phase": "complete"}))
    with pytest.raises(RuntimeError, match="identity is incomplete"):
        control.target_identity(environment)

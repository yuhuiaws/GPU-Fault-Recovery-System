from __future__ import annotations

import json
import subprocess
import sys

import pytest

from scripts.e2e.regional import boot_guard_control as control
from tests.regional.test_boot_guard_control import execute_flags, scope
from tests.regional.test_live_command_boundaries import ready_pod


@pytest.mark.parametrize("number", [0, 6, 11])
def test_unknown_or_retired_guard_case_is_not_executable(number) -> None:
    with pytest.raises(RuntimeError, match="unknown or retired"):
        control.case_id(number)


@pytest.mark.parametrize("failure", ["", "release", "uid", "generation"])
def test_target_binding_reads_complete_replica_population_before_identity(
    failure, tmp_path, monkeypatch
) -> None:
    # Live shape (2026-09-20): the release id is bound from the regional release
    # state ConfigMap's state.json, the same source every case's evidence uses;
    # the release-metadata ConfigMap carries no release-id key.
    release_state = {"phase": "complete", "release_id": "fixture-release"}
    deployment = {
        "metadata": {"uid": "fixture-deployment", "generation": 2},
        "spec": {"replicas": 3},
    }
    if failure == "release":
        release_state["release_id"] = None
    elif failure == "uid":
        deployment["metadata"]["uid"] = ""
    elif failure == "generation":
        deployment["metadata"]["generation"] = True
    metadata = {"data": {"state.json": json.dumps(release_state)}}
    pods = []
    for index in range(3):
        pod = ready_pod()
        pod["metadata"] = {"name": f"pod-{index}", "uid": f"uid-{index}"}
        pod["spec"]["nodeName"] = f"node-{index}"
        pods.append(pod)
    replies = iter([metadata, deployment, {"items": pods}])
    calls = []

    def run(command):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, json.dumps(next(replies)), "")

    monkeypatch.setattr(control, "run", run)
    environment = {
        "CPU_KUBECONFIG": str(tmp_path / "empty-kubeconfig"),
        "NAMESPACE": "fixture",
    }
    if failure:
        with pytest.raises(RuntimeError, match="identity is incomplete"):
            control.target_identity(environment)
    else:
        result = control.target_identity(environment)
        assert result["generation"] == 2 and len(result["cpu_pods"]) == 3, (
            "the bound identity needs every Ready API replica"
        )
    assert len(calls) == 3, (
        "target binding must read release state, deployment and Pods"
    )
    assert all(
        command[:5]
        == ["kubectl", "--kubeconfig", environment["CPU_KUBECONFIG"], "-n", "fixture"]
        for command in calls
    ), "all reads must retain the approved CPU scope"


@pytest.mark.parametrize("needed", [False, True])
def test_formal_predecessor_binds_release_or_explicitly_records_none(
    needed, tmp_path, monkeypatch
) -> None:
    calls = []
    monkeypatch.setattr(
        control,
        "predecessor_path",
        lambda *_a: (
            ("previous", tmp_path / "previous.json") if needed else (None, None)
        ),
    )

    def read(path, identifier, **kwargs):
        calls.append((path, identifier, kwargs))
        return {"valid": False}

    monkeypatch.setattr(control, "predecessor_evidence", read)
    result = control.predecessor(tmp_path, "case", "fixture-release")
    assert result == (
        {"valid": False} if needed else {"valid": True, "verdict": "NOT_REQUIRED"}
    ), "missing predecessor requirements must not be confused with failed evidence"
    assert calls == (
        [(tmp_path / "previous.json", "previous", {"release_id": "fixture-release"})]
        if needed
        else []
    ), "predecessor evidence must bind the same release"


@pytest.mark.parametrize("failure", ["environment", "directory", "start"])
def test_entry_rejects_incomplete_shell_scope_before_authorization(
    failure, tmp_path, monkeypatch
) -> None:
    scope(tmp_path, monkeypatch)
    if failure == "environment":
        monkeypatch.delenv("AWS_REGION")
    elif failure == "directory":
        monkeypatch.setenv("RUN_DIR", str(tmp_path / "other"))
    else:
        monkeypatch.setenv("BOOT_GUARD_START_CASE", "2")
    with pytest.raises(RuntimeError, match="incomplete|must match|start must"):
        control.main()
    assert not (tmp_path / "cases").exists(), "invalid shell scope cannot create a plan"


@pytest.mark.parametrize("failure", ["verdict", "case", "predecessor", "none"])
def test_recording_requires_selected_case_and_current_predecessor(
    failure, tmp_path, monkeypatch
) -> None:
    arguments, directory, _identity = scope(tmp_path, monkeypatch)
    assert control.main() == 0, "the initial fake scope must produce its bound plan"
    monkeypatch.setattr(sys, "argv", [*arguments, *execute_flags()])
    if failure != "none":
        monkeypatch.setenv(
            "BOOT_GUARD_RECORD_CASE",
            "foreign" if failure == "case" else "GF-REGIONAL-BOOT-001",
        )
        monkeypatch.setenv(
            "BOOT_GUARD_RECORD_VERDICT", "UNKNOWN" if failure == "verdict" else "PASS"
        )
    if failure == "predecessor":
        responses = iter(
            [
                {"valid": True, "evidence_cluster_id": "cluster-a"},
                {"valid": False, "evidence_cluster_id": "cluster-a"},
            ]
        )
        monkeypatch.setattr(control, "predecessor", lambda *_args: next(responses))
    if failure == "none":
        assert control.main() == 0, (
            "authorization without evidence recording remains valid"
        )
    else:
        with pytest.raises(RuntimeError, match="invalid|predecessor"):
            control.main()
    evidence = directory / "GF-REGIONAL-BOOT-001.json"
    if failure == "predecessor":
        assert json.loads(evidence.read_text())["verdict"] == "FAIL", (
            "a newly failed predecessor must downgrade the requested PASS"
        )
    else:
        assert not evidence.exists(), (
            "invalid or absent recording request must not mint evidence"
        )

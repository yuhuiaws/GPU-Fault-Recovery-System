from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Any

import pytest

from tests._script_loader import lazy_script_module
from tests.deploy._cov95_tools_support import TOOLS

STATE = lazy_script_module(TOOLS / "cleanup_state.py")


def cleanup_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    config = tmp_path / "release.json"
    config.write_text(
        json.dumps(
            {
                "namespace": "gpu-fault-system",
                "cpu_kubeconfig": "/dev/null",
                "clusters": [
                    {"cluster_id": "gpu-a", "context": "context-a"},
                    {"cluster_id": "gpu-b", "context": "context-b"},
                ],
            }
        )
    )
    inventory = tmp_path / "inventory.json"
    inventory.write_text(
        json.dumps(
            {"schema_version": 1, "cpu": {"resources": []}, "gpu": {"resources": []}}
        )
    )
    return config, inventory, tmp_path / "checkpoint" / "cleanup.json"


@pytest.fixture
def cleanup_state(tmp_path: Path) -> dict[str, Any]:
    config, inventory, path = cleanup_inputs(tmp_path)
    return STATE.initialize(
        path,
        config_path=config,
        inventory_path=inventory,
        scope="all",
        mode="reset",
        node_mode="uninstall",
    )


@pytest.mark.parametrize(
    "field,value,problem",
    [
        ("schema_version", 1, "unsupported"),
        ("phase", "invented", "invalid cleanup phase"),
        ("status", "UNKNOWN", "invalid cleanup status"),
        ("history", [], "lacks phase history"),
        ("history", {}, "lacks phase history"),
        ("history", [None], "invalid cleanup phase history"),
        (
            "history",
            [{"phase": "invented", "status": "COMPLETED"}],
            "invalid cleanup phase history",
        ),
        (
            "history",
            [{"phase": "PREFLIGHT", "status": "UNKNOWN"}],
            "invalid cleanup phase history",
        ),
        (
            "history",
            [
                {"phase": "INGRESS_STOPPED", "status": "COMPLETED"},
                {"phase": "PREFLIGHT", "status": "COMPLETED"},
            ],
            "invalid cleanup phase history",
        ),
        (
            "history",
            [{"phase": "PREFLIGHT", "status": "COMPLETED"}],
            "differs from history",
        ),
    ],
)
def test_checkpoint_digest_does_not_authorize_invalid_state(
    cleanup_state: dict[str, Any], field: str, value: Any, problem: str
) -> None:
    cleanup_state[field] = value
    cleanup_state["content_sha256"] = STATE.content_digest(cleanup_state)
    with pytest.raises(STATE.CleanupStateError, match=problem):
        STATE.verify_document(cleanup_state)


def test_checkpoint_must_be_an_object_and_digest_must_match(
    cleanup_state: dict[str, Any],
) -> None:
    with pytest.raises(STATE.CleanupStateError, match="must be an object"):
        STATE.verify_document([])
    cleanup_state["scope"] = "gpu"
    with pytest.raises(STATE.CleanupStateError, match="SHA-256 mismatch"):
        STATE.verify_document(cleanup_state)


def test_rehashed_completion_still_needs_lifecycle_evidence(
    cleanup_state: dict[str, Any],
) -> None:
    cleanup_state["phase"] = "CLEANUP_COMPLETED"
    cleanup_state["status"] = "COMPLETED"
    cleanup_state["history"].append(
        {"phase": "CLEANUP_COMPLETED", "status": "COMPLETED"}
    )
    cleanup_state["content_sha256"] = STATE.content_digest(cleanup_state)
    with pytest.raises(STATE.CleanupStateError, match="lacks completed lifecycle"):
        STATE.verify_document(cleanup_state)


@pytest.mark.parametrize("contents", [None, "{", "null", "[]"])
def test_unreadable_or_malformed_checkpoint_is_never_an_empty_state(
    tmp_path: Path, contents: str | None
) -> None:
    path = tmp_path / "state.json"
    if contents is not None:
        path.write_text(contents)
    with pytest.raises(STATE.CleanupStateError, match="cannot read|must be an object"):
        STATE.read_state(path)


def test_atomic_write_failure_preserves_the_previous_checkpoint_and_removes_temp(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config, inventory, path = cleanup_inputs(tmp_path)
    document = STATE.initialize(
        path,
        config_path=config,
        inventory_path=inventory,
        scope="all",
        mode="stop",
        node_mode="stop",
    )
    original = path.read_bytes()
    STATE.transition(document, phase="PREFLIGHT", status="COMPLETED", message="ready")

    def fail_replace(source: Path, destination: Path) -> None:
        assert destination == path
        assert source.stat().st_mode & 0o777 == 0o600
        raise OSError("simulated disk failure")

    monkeypatch.setattr(STATE.os, "replace", fail_replace)
    with pytest.raises(OSError, match="simulated disk failure"):
        STATE.atomic_write(path, document)
    assert path.read_bytes() == original
    assert list(path.parent.iterdir()) == [path]
    assert STATE.read_state(path)["status"] == "IN_PROGRESS"


@pytest.mark.parametrize(
    "config,scope,ids,problem",
    [
        ({}, "all", [], "must be a list"),
        (
            {"clusters": [{"cluster_id": "a"}, {"cluster_id": "a"}]},
            "all",
            [],
            "duplicate",
        ),
        ({"clusters": []}, "gpu", [], "distinct, known"),
        ({"clusters": []}, "gpu", ["unknown"], "distinct, known"),
        ({"clusters": [{"cluster_id": "a"}]}, "gpu", ["a", "a"], "distinct, known"),
        ({"clusters": []}, "all", ["a"], "invalid cleanup scope"),
        ({"clusters": []}, "unknown", [], "invalid cleanup scope"),
    ],
)
def test_request_targets_reject_ambiguous_ownership(
    config: dict[str, Any], scope: str, ids: list[str], problem: str
) -> None:
    with pytest.raises(STATE.CleanupStateError, match=problem):
        STATE.request_targets(config, scope=scope, cluster_ids=ids)


@pytest.mark.parametrize("mode", ["stop", "clean", "reset"])
def test_required_checkpoints_follow_selected_scope_and_mode(mode: str) -> None:
    gpu = STATE.required_phases({"scope": "gpu", "mode": mode})
    all_scopes = STATE.required_phases({"scope": "all", "mode": mode})
    assert set(all_scopes) - set(gpu) == {
        "CLUSTERS_DRAINING",
        "INGRESS_STOPPED",
        "CONTROL_CONSUMERS_STOPPED",
        "CPU_AUXILIARIES_STOPPED",
    }
    assert ("NAMESPACES_DELETED" in gpu) is (mode == "reset")
    assert ("APPLICATION_OBJECTS_DELETED" in gpu) is (mode != "stop")
    assert gpu == sorted(gpu, key=STATE.PHASES.index)


def test_resource_recapture_is_idempotent_but_cannot_rebind_previous_state(
    cleanup_state: dict[str, Any],
) -> None:
    first = {
        "resource_scope": "cpu",
        "context": "cpu",
        "kind": "deployment",
        "name": "gpu-fault-api-ha",
        "previous": "3",
    }
    second = {**first, "resource_scope": "gpu", "context": "context-a"}
    STATE.record_resource(cleanup_state, **first)
    STATE.record_resource(cleanup_state, **second)
    before = copy.deepcopy(cleanup_state["original_resources"])
    STATE.record_resource(cleanup_state, **second)
    assert cleanup_state["original_resources"] == before
    with pytest.raises(
        STATE.CleanupStateError, match="original resource state changed"
    ):
        STATE.record_resource(cleanup_state, **{**second, "previous": "0"})
    assert cleanup_state["original_resources"] == before


@pytest.mark.parametrize(
    "phase,status,problem",
    [
        ("unknown", "COMPLETED", "unknown cleanup phase"),
        ("PREFLIGHT", "unknown", "unknown cleanup status"),
        ("READY_TO_DELETE_AURORA", "COMPLETED", "completed CLEANUP_COMPLETED"),
        ("AURORA_DELETED", "COMPLETED", "completed READY_TO_DELETE_AURORA"),
        ("CLEANUP_COMPLETED", "COMPLETED", "completed lifecycle"),
    ],
)
def test_transition_refusal_does_not_change_the_journal(
    cleanup_state: dict[str, Any], phase: str, status: str, problem: str
) -> None:
    before = copy.deepcopy(cleanup_state)
    with pytest.raises(STATE.CleanupStateError, match=problem):
        STATE.transition(cleanup_state, phase=phase, status=status, message="refused")
    assert cleanup_state == before


def test_latest_checkpoint_status_controls_completion_and_cannot_go_backwards(
    cleanup_state: dict[str, Any],
) -> None:
    STATE.transition(cleanup_state, phase="PREFLIGHT", status="COMPLETED", message="")
    STATE.transition(cleanup_state, phase="PREFLIGHT", status="FAILED", message="retry")
    assert STATE.completed_phases(cleanup_state) == []
    STATE.transition(
        cleanup_state, phase="INGRESS_STOPPED", status="IN_PROGRESS", message=""
    )
    before = copy.deepcopy(cleanup_state)
    with pytest.raises(STATE.CleanupStateError, match="cannot move backwards"):
        STATE.transition(
            cleanup_state, phase="PREFLIGHT", status="COMPLETED", message=""
        )
    assert cleanup_state == before


def invoke(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], *arguments: str
) -> dict[str, Any]:
    monkeypatch.setattr(sys, "argv", ["cleanup_state", *arguments])
    assert STATE.main() == 0
    return json.loads(capsys.readouterr().out)


def test_local_checkpoint_entrypoints_round_trip_and_refuse_resume_after_handoff(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    config, inventory, path = cleanup_inputs(tmp_path)
    request = [
        "--config",
        str(config),
        "--scope",
        "all",
        "--mode",
        "reset",
        "--node-mode",
        "uninstall",
    ]
    result = invoke(
        monkeypatch,
        capsys,
        "init",
        "--path",
        str(path),
        "--inventory",
        str(inventory),
        *request,
    )
    assert result["completed_phases"] == []
    assert result["status"] == "IN_PROGRESS"
    before = path.read_bytes()
    assert invoke(monkeypatch, capsys, "verify", "--path", str(path)) == result
    assert (
        invoke(monkeypatch, capsys, "verify", "--path", str(path), *request) == result
    )
    assert path.read_bytes() == before, "verification must not rewrite evidence"
    with pytest.raises(STATE.CleanupStateError, match="already exists"):
        invoke(
            monkeypatch,
            capsys,
            "init",
            "--path",
            str(path),
            "--inventory",
            str(inventory),
            *request,
        )

    invoke(
        monkeypatch,
        capsys,
        "record",
        "--path",
        str(path),
        "--resource-scope",
        "cpu",
        "--context",
        "cpu",
        "--kind",
        "deployment",
        "--name",
        "gpu-fault-api-ha",
        "--previous",
        "3",
    )
    fleet = tmp_path / "fleet.json"
    fleet.write_text(json.dumps([{"cluster_id": "gpu-a", "node_id": "node-a"}]))
    invoke(
        monkeypatch, capsys, "attach-fleet", "--path", str(path), "--input", str(fleet)
    )
    stored = STATE.read_state(path)
    assert len(stored["original_resources"]) == 1
    assert stored["fleet_snapshot"] == json.loads(fleet.read_text())
    assert len(stored["fleet_snapshot_sha256"]) == 64
    restored_inventory = tmp_path / "resume-inventory.json"
    invoke(
        monkeypatch,
        capsys,
        "resume",
        "--path",
        str(path),
        *request,
        "--inventory-output",
        str(restored_inventory),
    )
    assert json.loads(restored_inventory.read_text()) == json.loads(
        inventory.read_text()
    )
    assert restored_inventory.stat().st_mode & 0o777 == 0o600
    for phase in [
        *STATE.required_phases(stored),
        "CLEANUP_COMPLETED",
        "READY_TO_DELETE_AURORA",
        "AURORA_DELETED",
    ]:
        result = invoke(
            monkeypatch,
            capsys,
            "transition",
            "--path",
            str(path),
            "--phase",
            phase,
            "--status",
            "COMPLETED",
            "--message",
            "local fixture",
        )
        assert (result["phase"], result["status"]) == (phase, "COMPLETED")
    assert STATE.read_state(path)["phase"] == "AURORA_DELETED"
    before = path.read_bytes()
    with pytest.raises(STATE.CleanupStateError, match="after Aurora handoff"):
        invoke(
            monkeypatch,
            capsys,
            "resume",
            "--path",
            str(path),
            *request,
            "--inventory-output",
            str(restored_inventory),
        )
    assert path.read_bytes() == before


def test_entrypoints_reject_incomplete_verification_or_nonlist_fleet_snapshot(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    config, inventory, path = cleanup_inputs(tmp_path)
    STATE.initialize(
        path,
        config_path=config,
        inventory_path=inventory,
        scope="all",
        mode="stop",
        node_mode="skip",
    )
    before = path.read_bytes()
    with pytest.raises(STATE.CleanupStateError, match="needs scope and modes"):
        invoke(
            monkeypatch, capsys, "verify", "--path", str(path), "--config", str(config)
        )
    fleet = tmp_path / "fleet.json"
    fleet.write_text("{}")
    with pytest.raises(STATE.CleanupStateError, match="must be a JSON list"):
        invoke(
            monkeypatch,
            capsys,
            "attach-fleet",
            "--path",
            str(path),
            "--input",
            str(fleet),
        )
    assert path.read_bytes() == before


def test_a_rematerialized_config_is_accepted_only_with_its_explicit_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cleanup_state: dict[str, Any]
) -> None:
    import hashlib

    config, _inventory, path = cleanup_inputs(tmp_path)
    moved = tmp_path / "moved-release.json"
    moved.write_text(
        json.dumps({**json.loads(config.read_text()), "release": {"manifest": "/b"}})
    )
    digest = hashlib.sha256(moved.read_bytes()).hexdigest()
    request = dict(scope="all", mode="reset", node_mode="uninstall", cluster_ids=[])
    previous = cleanup_state["config_sha256"]

    monkeypatch.delenv(STATE.ACCEPT_CONFIG_ENV, raising=False)
    with pytest.raises(STATE.CleanupStateError, match="differs on config_sha256"):
        STATE.validate_request(cleanup_state, config_path=moved, **request)
    monkeypatch.setenv(STATE.ACCEPT_CONFIG_ENV, "0" * 64)
    with pytest.raises(STATE.CleanupStateError, match="differs on config_sha256"):
        STATE.validate_request(cleanup_state, config_path=moved, **request)
    assert cleanup_state["config_sha256"] == previous, "a wrong digest changes nothing"

    monkeypatch.setenv(STATE.ACCEPT_CONFIG_ENV, digest)
    STATE.validate_request(cleanup_state, config_path=moved, state_path=path, **request)
    persisted = STATE.read_state(path)
    assert persisted["config_sha256"] == digest, "the accepted digest is persisted"
    (entry,) = persisted["config_sha256_history"]
    assert (entry["previous"], entry["accepted"]) == (previous, digest)
    STATE.validate_request(cleanup_state, config_path=moved, state_path=path, **request)
    assert len(STATE.read_state(path)["config_sha256_history"]) == 1, (
        "an accepted digest is not re-recorded"
    )
    with pytest.raises(STATE.CleanupStateError, match="differs on config_sha256"):
        STATE.validate_request(cleanup_state, config_path=config, **request)

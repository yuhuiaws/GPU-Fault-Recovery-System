from __future__ import annotations

import json
import subprocess
from uuid import UUID

import pytest

from scripts.e2e.regional import boot_acceptance_runtime as runtime
from tests.regional._cov95_boot_extra_safety import (
    boot_extra_isolation as boot_extra_isolation,
)
from tests.regional._cov95_boot_owner import OwnerFixture
from tests.regional._cov95_cap_cases import Clock


@pytest.fixture
def owner(monkeypatch):
    clock = Clock()
    fixture = OwnerFixture(clock)
    monkeypatch.setattr(runtime, "time", clock)
    monkeypatch.setattr(runtime, "amp_request", fixture.amp_request)
    monkeypatch.setattr(
        runtime,
        "run",
        lambda args, **kwargs: subprocess.CompletedProcess(args, 0, "", ""),
    )
    return fixture


@pytest.mark.parametrize(
    "fault", ["no-pods", "no-profile", "empty-profile", "numeric-profile", "alert-gate"]
)
def test_owner_case_refuses_unverifiable_preconditions_before_inserting_a_command(
    tmp_path, monkeypatch, owner, fault
):
    if fault == "no-pods":
        monkeypatch.setattr(owner, "pods", lambda *args: [])
    elif fault == "no-profile":
        owner.config.pop("runtime_profile")
    elif fault == "empty-profile":
        owner.config["runtime_profile"]["version"] = ""
    elif fault == "numeric-profile":
        owner.config["runtime_profile"]["version"] = 1
    else:
        monkeypatch.setattr(
            runtime,
            "run",
            lambda args, **kwargs: subprocess.CompletedProcess(
                args, 1, "", "unit alert gate failed"
            ),
        )
    with pytest.raises(runtime.BootAcceptanceError, match="Pods|Profile|reachability"):
        runtime.run_boot015(owner, case_dir=tmp_path, attempt=3)
    assert owner.rows == {"existing-command"}
    assert "insert" not in owner.probes
    assert "delete" not in owner.probes
    assert "baseline" not in owner.probes
    assert not (tmp_path / "boot015-details.json").exists(), (
        "a rejected precondition generated mutation evidence"
    )


@pytest.mark.parametrize(
    "inventory",
    [
        None,
        {},
        {"count": True, "ids": []},
        {"count": -1, "ids": []},
        {"count": 1, "ids": []},
        {"count": 1, "ids": [""]},
        {"count": 1, "ids": [3]},
        {"count": 2, "ids": ["same", "same"]},
    ],
)
def test_cleanup_never_converts_an_unknown_inventory_into_verified_absence(
    monkeypatch, owner, inventory
):
    owner.command_id = "unit-owned-command"
    owner.rows.add(owner.command_id)
    original = owner.cpu_python

    def probe(script, *args, **kwargs):
        if script == runtime.REMOTE_BASELINE_PROBE:
            owner.probes.append("baseline")
            return inventory
        return original(script, *args, **kwargs)

    monkeypatch.setattr(owner, "cpu_python", probe)
    result = runtime.boot015_cleanup(
        owner,
        command_id=owner.command_id,
        pods=["replica-a", "replica-b"],
        baseline_ids={"existing-command"},
    )
    assert result["passed"] is False
    assert result["inventory_verified"] is False
    assert result["synthetic_command_removed"] is False
    assert "resolve_observed" not in result
    assert result["readiness_recovered"] == {"replica-a": True, "replica-b": True}
    assert owner.probes == ["delete", "baseline"]
    assert owner.rows == {"existing-command"}


@pytest.mark.parametrize("fault", ["delete", "introduced", "readiness", "empty-pods"])
def test_cleanup_keeps_removal_inventory_and_readiness_failures_visible(
    monkeypatch, owner, fault
):
    owner.command_id = "unit-owned-command"
    owner.rows.add(owner.command_id)
    if fault == "delete":
        owner.failure = "delete"
    elif fault == "introduced":
        owner.rows.add("unexpected-new-command")
    elif fault == "readiness":

        def execute(*args, **kwargs):
            raise OSError("unit readiness read failed")

        monkeypatch.setattr(owner, "exec", execute)
    pods = [] if fault == "empty-pods" else ["replica-a", "replica-b"]
    result = runtime.boot015_cleanup(
        owner, command_id=owner.command_id, pods=pods, baseline_ids={"existing-command"}
    )
    assert result["passed"] is False
    assert result["inventory_verified"] is True
    assert result["synthetic_command_removed"] is (fault != "delete")
    assert owner.probes == ["delete", "baseline"]
    if fault == "delete":
        assert "synthetic delete failure" in result["delete"]["error"]
        assert result["readiness_recovered"] == {"replica-a": False, "replica-b": False}
        assert "resolve_observed" not in result
    elif fault == "introduced":
        assert result["commands_introduced"] == ["unexpected-new-command"]
        assert "unexpected-new-command" in owner.rows
    elif fault == "readiness":
        assert result["readiness_recovered"] == {
            "error": "OSError: unit readiness read failed"
        }
        assert result["resolve_observed"]["matched"] is True
    else:
        assert result["readiness_recovered"] == {}


@pytest.mark.parametrize("aborted", [False, True])
def test_insert_ack_loss_still_removes_only_the_minted_command_and_persists_fail(
    tmp_path, monkeypatch, owner, aborted
):
    original = owner.cpu_python
    error = (
        KeyboardInterrupt("unit insert abort")
        if aborted
        else OSError("unit insertion acknowledgement lost")
    )
    inserted = []

    def probe(script, *args, **kwargs):
        result = original(script, *args, **kwargs)
        if script == runtime.REMOTE_INJECT_PROBE:
            inserted.append(args[1])
            raise error
        return result

    monkeypatch.setattr(owner, "cpu_python", probe)
    with pytest.raises(type(error)) as caught:
        runtime.run_boot015(owner, case_dir=tmp_path, attempt=4)
    assert caught.value is error
    assert len(inserted) == 1
    assert owner.rows == {"existing-command"}
    assert owner.probes == ["profile", "baseline", "insert", "delete", "baseline"]
    saved = json.loads((tmp_path / "boot015-details.json").read_text(encoding="utf-8"))
    assert saved["verdict"] == "FAIL"
    assert saved["cleanup"]["passed"] is True
    assert saved["cleanup"]["synthetic_command_removed"] is True
    assert "checks" not in saved, "post-insertion checks were fabricated after an abort"


def test_repeated_attempts_get_independent_nonces_and_restore_the_original_inventory(
    tmp_path, owner
):
    ids = []
    for name in ("first", "second"):
        case_dir = tmp_path / name
        case_dir.mkdir()
        result = runtime.run_boot015(owner, case_dir=case_dir, attempt=7)
        assert result["verdict"] == "PASS"
        assert result["cleanup"]["passed"] is True
        assert owner.rows == {"existing-command"}
        ids.append(owner.command_id)
    assert len(set(ids)) == 2, "a repeated attempt reused a remotely owned command ID"
    for command_id in ids:
        assert command_id.startswith("remote-boot015-7-"), "attempt identity was lost"
        assert UUID(command_id.removeprefix("remote-boot015-7-")).version == 4
    assert owner.probes.count("insert") == 2
    assert owner.probes.count("delete") == 2


def test_secret_contract_accepts_nonindented_inline_python_without_changing_key_checks(
    monkeypatch,
):
    manifest = runtime.manifest_deployment()
    container = manifest["spec"]["template"]["spec"]["containers"][0]
    container["command"] = ["python3", "-c"]
    container["args"] = ["print('unit')"]
    monkeypatch.setattr(runtime, "manifest_deployment", lambda: manifest)
    result = runtime.secret_key_contract()
    assert result["passed"] is True
    assert result["python_c_leading_whitespace"] == []
    assert result["missing"] == []
    assert result["required"], "inline command validation skipped the Secret contract"

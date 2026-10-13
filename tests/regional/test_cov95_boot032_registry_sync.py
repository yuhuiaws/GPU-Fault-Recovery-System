"""BOOT-032 and the native deletion gate on the installation registry sync.

Native ``uninstall`` refuses a new attempt while the sacrificial site's
``release-deploy/installation-registry-sync.json`` is failed, missing (a site
last deployed before deploys recorded it) or unreadable
(``resource_registry_outcome.deletion_gate``). The harness never passes the
override flag: its read-only preflight and its execute path name the refusal
and the remedy before any case or native journal exists, and a native attempt
that already started resumes on the verdict it started with.
"""

from __future__ import annotations

import pytest

from gpu_fault.admin.resource_registry_outcome import ACCEPT_FLAG, FIELD, record_path
from scripts.e2e.regional import boot032_contract as contract
from scripts.e2e.regional import boot032_journal as journal
from scripts.e2e.regional import boot032_lifecycle as lifecycle
from tests.admin._registry_sync_support import record_registry_sync
from tests.regional._cov95_boot032_approval import approve, execute, restart
from tests.regional._cov95_boot032_native import NativeHarness
from tests.regional._cov95_boot032_world import World

FAILURE = "fixture registry sync failed"


def unsynchronize(world, record):
    state_dir = world.target.source.parent
    if record == "missing":
        record_path(state_dir).unlink()
    elif record == "failed":
        record_registry_sync(state_dir, error=FAILURE)
    else:
        record_path(state_dir).write_text("{", encoding="utf-8")


@pytest.mark.parametrize(
    ("record", "status", "reason"),
    [
        ("missing", "MISSING", "no installation registry sync outcome is recorded"),
        ("failed", "UNAVAILABLE", FAILURE),
        ("unreadable", "UNREADABLE", "is not valid JSON"),
    ],
    ids=["missing", "failed", "unreadable"],
)
def test_preflight_names_an_unsynchronized_registry_with_its_remedy(
    tmp_path, monkeypatch, record, status, reason
):
    world = World(tmp_path, monkeypatch)
    unsynchronize(world, record)
    preflight = lifecycle.read_only_preflight(world.settings, world.settings.case_dir)
    assert preflight["binding"] is None, "a refused preflight mints no approval binding"
    assert len(preflight["errors"]) == 1, f"only the registry refuses: {preflight}"
    error = preflight["errors"][0]
    assert f"installation registry sync {status}" in error and reason in error, (
        f"the refusal names the recorded outcome: {error}"
    )
    assert (
        f"gpu-fault-admin deploy --state-dir {world.target.source.parent}" in error
    ), f"the refusal names the remedy: {error}"
    assert f"never passes {ACCEPT_FLAG}" in error, (
        "the harness never overrides the native gate"
    )
    assert not world.settings.native_dir.exists(), "a plan starts no native attempt"


def test_preflight_reports_the_registry_beside_any_other_refusal(tmp_path, monkeypatch):
    world = World(tmp_path, monkeypatch)
    unsynchronize(world, "missing")
    world.read_error = True
    errors = lifecycle.read_only_preflight(world.settings, world.settings.case_dir)[
        "errors"
    ]
    assert errors[0] == "uninstall preflight refused: RuntimeError", (
        f"the other refusal keeps its report: {errors}"
    )
    assert len(errors) == 2 and "installation registry sync MISSING" in errors[1], (
        f"the registry refusal is reported as well: {errors}"
    )


def test_execute_refuses_before_any_journal_and_proceeds_after_resynchronizing(
    tmp_path, monkeypatch
):
    world = World(tmp_path, monkeypatch)
    harness = NativeHarness(world, monkeypatch)
    settings, _plan, deadline = approve(world)
    # A config re-roll after the approval failed to synchronize the registry.
    unsynchronize(world, "failed")
    with pytest.raises(contract.UninstallCaseError, match="registry sync UNAVAILABLE"):
        execute(world, deadline)
    assert journal.read_case(settings.case_dir / "boot032-state.json") is None, (
        "no case journal binds the registry that the remedial deploy rewrites"
    )
    assert not settings.native_dir.exists() and not harness.events, (
        "the refusal starts no native attempt and deletes nothing"
    )
    result = contract.read_document(settings.case_dir / f"{contract.CASE_ID}.json")
    assert (result["verdict"], result["status"]) == ("FAIL", "PREFLIGHT"), (
        "a refused execution never reports progress"
    )
    record_registry_sync(world.target.source.parent)
    _settings, _plan, deadline = approve(world)
    assert execute(world, deadline) == contract.RESTART_EXIT, (
        "a new plan over the resynchronized site proceeds"
    )


def test_a_started_native_attempt_resumes_on_the_verdict_it_started_with(
    tmp_path, monkeypatch
):
    world = World(tmp_path, monkeypatch)
    NativeHarness(world, monkeypatch)
    settings, _plan, deadline = approve(world)
    assert execute(world, deadline) == contract.RESTART_EXIT, (
        "the native attempt starts over a synchronized registry"
    )
    unsynchronize(world, "missing")
    assert lifecycle.read_only_preflight(settings, settings.case_dir)["errors"] == [], (
        "a paused native attempt is no new attempt to judge"
    )
    restart(world)
    assert execute(world, deadline) == 0, "the resume completes the retirement"
    native = contract.read_document(settings.native_dir / "state.json")
    assert native[FIELD]["status"] == "SYNCED", (
        "the native journal keeps the verdict its attempt started on"
    )

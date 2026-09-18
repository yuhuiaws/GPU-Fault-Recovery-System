from __future__ import annotations

from types import SimpleNamespace

import pytest

from gpu_fault.admin.bootstrap_common import BootstrapError
from gpu_fault.admin.process_supervisor import ProcessSupervisionLost
from scripts.e2e.regional import boot032_contract as contract
from scripts.e2e.regional import boot032_journal as journal
from scripts.e2e.regional import boot032_lifecycle as lifecycle
from scripts.e2e.regional import boot032_native as adapter
from tests.regional._cov95_boot032_approval import approve, execute, restart
from tests.regional._cov95_boot032_native import NativeHarness
from tests.regional._cov95_boot032_world import World


def test_same_process_resume_fails_without_mutation_but_fresh_retry_works(
    tmp_path, monkeypatch
):
    world = World(tmp_path, monkeypatch)
    harness = NativeHarness(world, monkeypatch)
    settings, _plan, deadline = approve(world)
    assert execute(world, deadline) == contract.RESTART_EXIT, (
        "native checkpoint must be real"
    )
    before = list(harness.events)
    assert execute(world, deadline) == 1, "same-process retry must be refused"
    assert harness.events == before, "same process must not enter AWS deletion"
    saved = journal.read_case(settings.case_dir / "boot032-state.json")
    assert saved["phase"] == "FAILED" and saved["pause"], (
        "retry must retain the original pause proof"
    )
    restart(world)
    assert execute(world, deadline) == 0, (
        "fresh process may retry the identical approved journal"
    )


@pytest.mark.parametrize("after_pause", [False, True])
def test_native_zero_exit_without_required_evidence_is_never_pass(
    tmp_path, monkeypatch, after_pause
):
    world = World(tmp_path, monkeypatch)
    harness = NativeHarness(world, monkeypatch)
    settings, _plan, deadline = approve(world)
    if after_pause:
        assert execute(world, deadline) == contract.RESTART_EXIT, (
            "prepare real cleanup checkpoint"
        )
        restart(world)
    monkeypatch.setattr(adapter, "uninstall", lambda *_a, **_k: {"phase": "COMPLETED"})
    assert execute(world, deadline) == 1, (
        "native exit or response alone cannot prove teardown"
    )
    value = contract.read_document(settings.case_dir / f"{contract.CASE_ID}.json")
    assert value["verdict"] == "FAIL" and not value["checks"], (
        "missing evidence must invalidate PASS"
    )
    assert not any(event.startswith("delete:") for event in harness.events), (
        "a transport success with no native state must never imply resource deletion"
    )


@pytest.mark.parametrize("failure", ["cpu", "aurora"])
def test_native_async_delete_failure_resumes_original_incarnations(
    tmp_path, monkeypatch, failure
):
    world = World(tmp_path, monkeypatch)
    harness = NativeHarness(world, monkeypatch)
    settings, plan, deadline = approve(world)
    assert execute(world, deadline) == contract.RESTART_EXIT, (
        "capture supervised pause boundary"
    )
    pause = journal.read_case(settings.case_dir / "boot032-state.json")["pause"]
    restart(world)
    setattr(harness, "fail_" + failure, True)
    assert execute(world, deadline) == 1, (
        "lost deletion acknowledgement must leave a failed attempt"
    )
    state = harness.state()
    assert state["phase"] == (
        "CPU_DELETE_IN_PROGRESS" if failure == "cpu" else "AURORA_DELETE_IN_PROGRESS"
    ), "native retry must retain the exact in-flight deletion phase"
    assert state["cpu_binding"] == pause["proof"]["native_cpu_binding"], (
        "failure must not erase or recapture CPU incarnation"
    )
    assert lifecycle.read_only_preflight(settings, settings.case_dir)["errors"] == [], (
        "proved native async deletion state must remain eligible for the same approved retry"
    )
    restart(world)
    assert execute(world, deadline) == 0, (
        "native resume must converge after a failed acknowledgement"
    )
    final, _cleanup = journal.final_receipts(
        settings, plan["details"]["binding"], pause["proof"]
    )
    assert all(
        row.status.value in {"DELETED", "DETACHED", "PRESERVED"}
        for row in final.resources
    ), "retry PASS requires all identities to have policy-consistent terminal status"


def test_failed_final_readback_retries_read_only_after_native_completion(
    tmp_path, monkeypatch
):
    world = World(tmp_path, monkeypatch)
    harness = NativeHarness(world, monkeypatch)
    settings, _plan, deadline = approve(world)
    assert execute(world, deadline) == contract.RESTART_EXIT, (
        "prepare original checkpoint"
    )
    restart(world)

    def probe(site):
        delegate = world.probe(site)

        def exists(resource):
            if resource.resource_type == "rds_snapshot":
                raise PermissionError("fake final snapshot read denied")
            return delegate.exists(resource)

        return SimpleNamespace(exists=exists)

    monkeypatch.setattr(adapter, "ResourceProbe", probe)
    assert execute(world, deadline) == 1, (
        "unreadable retained snapshot cannot yield PASS"
    )
    assert harness.state()["phase"] == "COMPLETED", (
        "native completion must remain durable"
    )
    saved = journal.read_case(settings.case_dir / "boot032-state.json")
    assert saved["phase"] == "FAILED" and saved["pause"], (
        "verification failure must preserve original restart proof"
    )
    before = list(harness.events), harness.cleanup_calls
    monkeypatch.setattr(adapter, "ResourceProbe", world.probe)
    harness.no_mutations = True
    assert execute(world, deadline) == 0, (
        "a later independent read may complete the case"
    )
    assert (harness.events, harness.cleanup_calls) == before, (
        "native COMPLETED plus failed case verification must never reenter uninstall"
    )


def test_native_cleanup_success_cannot_hide_a_remaining_namespace(
    tmp_path, monkeypatch
):
    world = World(tmp_path, monkeypatch)
    harness = NativeHarness(world, monkeypatch)
    settings, _plan, deadline = approve(world)
    original = harness.run

    def run(arguments, **kwargs):
        result = original(arguments, **kwargs)
        if arguments[-1] == "verify-targets":
            world.uids[
                (world.target.metadata_name, "cpu", "namespace", "gpu-fault-system", "")
            ] = "leftover-uid"
        return result

    monkeypatch.setattr(harness, "run", run)
    assert execute(world, deadline) == 1, (
        "native zero exit must be followed by actual namespace readback"
    )
    saved = journal.read_case(settings.case_dir / "boot032-state.json")
    assert saved["pause"] is None and harness.events == ["cleanup"], (
        "failed native proof cannot produce a restart checkpoint or start AWS deletion"
    )


def test_native_supervision_loss_blocks_every_future_attempt(tmp_path, monkeypatch):
    world = World(tmp_path, monkeypatch)
    harness = NativeHarness(world, monkeypatch)
    settings, _plan, deadline = approve(world)
    harness.fail_cleanup = ProcessSupervisionLost("fake unproved command termination")
    with pytest.raises(ProcessSupervisionLost):
        execute(world, deadline)
    assert harness.state()["supervision_lost"] is True, (
        "native entry must retain its own safety marker"
    )
    saved = journal.read_case(settings.case_dir / "boot032-state.json")
    assert saved["phase"] == "BLOCKED", (
        "case must not call a supervision failure merely retryable"
    )
    before = list(harness.events), harness.cleanup_calls
    restart(world)
    with pytest.raises(RuntimeError, match="supervision"):
        execute(world, deadline)
    assert (harness.events, harness.cleanup_calls) == before, (
        "fresh process cannot clear unproved in-flight work"
    )
    assert (
        lifecycle.read_only_preflight(settings, settings.case_dir)["binding"] is None
    ), "a blocked case cannot mint another successful approval plan"


def test_completed_pass_is_invalidated_before_new_readback_failure(
    tmp_path, monkeypatch
):
    world = World(tmp_path, monkeypatch)
    harness = NativeHarness(world, monkeypatch)
    settings, _plan, deadline = approve(world)
    assert execute(world, deadline) == contract.RESTART_EXIT, "prepare real pause"
    restart(world)
    assert execute(world, deadline) == 0, "prepare independently verified completion"
    before = list(harness.events)
    world.read_error = True
    with pytest.raises(RuntimeError, match="read failed"):
        execute(world, deadline)
    value = contract.read_document(settings.case_dir / f"{contract.CASE_ID}.json")
    assert value["verdict"] == "FAIL" and value["status"] == "PREFLIGHT", (
        "a new unknown read must not leave a stale PASS as the current attempt result"
    )
    assert harness.events == before, "failed completed replay remains read-only"


def test_cleanup_failure_does_not_delete_aws_or_invent_a_pause(tmp_path, monkeypatch):
    world = World(tmp_path, monkeypatch)
    harness = NativeHarness(world, monkeypatch)
    settings, _plan, deadline = approve(world)
    harness.fail_cleanup = BootstrapError("fake cleanup failed")
    assert execute(world, deadline) == 1, "native cleanup failure must be reported"
    saved = journal.read_case(settings.case_dir / "boot032-state.json")
    assert saved["pause"] is None and not harness.events, (
        "unproved cleanup cannot authorize AWS phases"
    )
    restart(world)
    assert execute(world, deadline) == contract.RESTART_EXIT, (
        "same original native journal may retry failed cleanup"
    )
    restart(world)
    assert execute(world, deadline) == 0, (
        "verified cleanup retry may later complete full retirement"
    )

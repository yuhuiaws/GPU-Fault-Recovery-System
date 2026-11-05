"""GF-REGIONAL-DESTR-026 cleanup: every failure downgrades the verdict.

Each cleanup step runs on its own (one refusal never skips the next), the
workload is deleted and an isolated node restored only once the store proves
quiescence, a node is only ever released through the validated-restore
workflow of a known incident, and residual probe, prewarm or workload
resources fail a case that otherwise passed.
"""

from __future__ import annotations

from typing import Any

import pytest

from scripts.e2e.regional.regional_live_fixture import RegionalFixtureError
from tests.regional._cov95_destr026_harness import (
    Destr026Harness,
    FakeWorkload,
    isolate,
)
from tests.regional.test_destr026_same_node_two_gpu_reset import INCIDENT, NODE


def passing(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> Destr026Harness:
    h = Destr026Harness(tmp_path, monkeypatch)
    preflight = h.plan()
    assert preflight["errors"] == [], preflight["errors"]
    return h


def test_a_test_workload_left_behind_fails_a_case_that_passed(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = passing(tmp_path, monkeypatch)
    h.workload_residual = f"pytorchjob/{FakeWorkload.name}\n"

    code, report = h.execute()

    assert report["errors"] == [], "the case itself passed"
    assert code == 1 and report["verdict"] == "FAIL", report
    assert report["cleanup"]["errors"] == [
        "workload_residual: RegionalFixtureError: test workload still exists: "
        f"pytorchjob/{FakeWorkload.name}"
    ], report["cleanup"]["errors"]
    (residual,) = [
        detail["args"]
        for detail in h.details("kubectl")
        if "--ignore-not-found" in detail["args"]
    ]
    assert residual == [
        "get",
        "pytorchjob",
        FakeWorkload.name,
        "--ignore-not-found",
        "-o",
        "name",
    ], residual


@pytest.mark.parametrize("owner", ["a", "b", "partner", "prewarm"])
def test_residual_probe_or_prewarm_resources_fail_the_cleanup(
    owner: str, tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = passing(tmp_path, monkeypatch)
    if owner == "prewarm":
        h.prewarm_residuals = {"pod": True}
        label = "prewarm_cleanup"
    else:
        h.probe_residuals[owner] = {"pod": True, "configmap": False}
        label = f"probe_cleanup_{owner}"

    code, report = h.execute()

    assert code == 1 and report["errors"] == [], report
    (error,) = report["cleanup"]["errors"]
    assert error.startswith(
        f"{label}: RegionalFixtureError: residual resources remain: "
    ), error
    assert label not in report["cleanup"], "a refused cleanup records no result"
    assert h.names()[-1] == "runtime.verify", "the identity recheck still runs last"


def test_each_cleanup_step_runs_even_when_another_one_fails(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = passing(tmp_path, monkeypatch)
    h.failures["workload.delete"] = RegionalFixtureError("delete refused")
    h.failures["prewarm.cleanup"] = RegionalFixtureError("prewarm API timed out")

    code, report = h.execute()

    assert code == 1, report
    cleanup = report["cleanup"]
    assert cleanup["errors"] == [
        "workload_delete: RegionalFixtureError: delete refused",
        "prewarm_cleanup: RegionalFixtureError: prewarm API timed out",
    ], cleanup["errors"]
    for later in (
        "workload_residual",
        "restore_isolated_nodes",
        "probe_cleanup_a",
        "probe_cleanup_b",
        "probe_cleanup_partner",
        "runtime_identity",
    ):
        assert later in cleanup, f"{later} was skipped after an earlier failure"


def test_a_runtime_identity_that_moved_by_the_end_of_cleanup_fails_the_case(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = passing(tmp_path, monkeypatch)
    h.runtime_after_cleanup = {**h.runtime, "release_state": {"phase": "rolling"}}

    code, report = h.execute()

    assert code == 1 and report["errors"] == [], report
    assert report["cleanup"]["errors"] == [
        "runtime_identity: RegionalFixtureError: runtime identity drifted after "
        "DESTR-026 cleanup"
    ], report["cleanup"]["errors"]
    stages = h.details("runtime.verify")
    assert stages == [
        "before DESTR-026 injection",
        "after DESTR-026 workload restart",
        "after DESTR-026 cleanup",
    ], stages


def test_a_remote_command_that_never_settles_leaves_the_job_and_the_node(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = passing(tmp_path, monkeypatch)
    # The case's own reads all saw a settled workflow; only after them does a
    # remote command reappear LEASED and stay so through cleanup.
    h.hooks["provider.events"] = lambda: setattr(h, "cleanup_busy", True)
    h.advance_at["store.workflow"] = 120

    code, report = h.execute()

    assert report["errors"] == [], "the case passed before cleanup"
    assert code == 1 and report["verdict"] == "FAIL", report
    cleanup = report["cleanup"]
    (error,) = cleanup["errors"]
    assert error.startswith(
        "quiescence: RegionalFixtureError: cleanup could not prove workflow and "
        "remote-command quiescence"
    ), error
    assert cleanup["workload_cleanup_deferred"] is True, cleanup
    assert cleanup["isolation_restore_deferred"] is True, cleanup
    names = h.names()
    assert "workload.delete" not in names, "no delete under a live remote command"
    assert "restore.create" not in names, "no restore under a live remote command"
    assert {"probe_cleanup_a", "probe_cleanup_b", "probe_cleanup_partner"} <= set(
        cleanup
    ), "the probes are still removed"
    quiet = h.evidence("cleanup-quiescence.json")
    assert quiet["safe_to_delete"] is False, quiet
    assert {entry["quiescent"] for entry in quiet["entries"]} == {False}, quiet


def test_a_node_isolated_by_another_incident_is_never_released_by_this_case(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = passing(tmp_path, monkeypatch)
    h.cached = [NODE]  # the case stops before its own fault
    h.hooks["prewarm.create"] = lambda: isolate(h.nodes[NODE], "inc-someone-else")

    code, report = h.execute()

    assert code == 1, report
    assert report["error"] == (
        "RegionalFixtureError: training image is not cached on both nodes"
    ), report
    assert report["cleanup"]["errors"] == [
        "restore_isolated_nodes: RegionalFixtureError: node-a is isolated but no "
        "incident is known"
    ], report["cleanup"]["errors"]
    assert "restore.create" not in h.names(), "no restore without this case's incident"
    assert h.nodes[NODE]["unschedulable"] is True, "the node stays with its owner"


def test_a_validated_restore_that_does_not_succeed_fails_the_cleanup(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = passing(tmp_path, monkeypatch)
    h.workflow["status"] = "FAILED"
    h.restore_status = "FAILED"
    h.hooks["workflow.wait"] = lambda: isolate(h.nodes[NODE], INCIDENT)

    code, report = h.execute()

    assert code == 1, report
    assert report["cleanup"]["errors"] == [
        "restore_isolated_nodes: RegionalFixtureError: node-a restore workflow did "
        "not succeed"
    ], report["cleanup"]["errors"]
    assert h.details("restore.wait") == ["restore-node-a"], h.calls
    assert h.nodes[NODE]["unschedulable"] is True, "never untainted by hand"


def test_a_sampler_that_cannot_be_stopped_is_a_cleanup_failure(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = passing(tmp_path, monkeypatch)
    h.split = True  # stops before the observation, with the sampler running
    h.failures["probe.stop-reset-sampler"] = RegionalFixtureError("unit busy")

    code, report = h.execute()

    assert code == 1, report
    cleanup = report["cleanup"]
    assert cleanup["fault_node_return"]["exec_allowed"] is True, cleanup
    assert cleanup["errors"] == ["sampler_stop: RegionalFixtureError: unit busy"], (
        cleanup["errors"]
    )
    assert sorted(h.details("probe.cleanup")) == ["a", "b", "partner"], (
        "the probes are removed even though the sampler stop failed"
    )

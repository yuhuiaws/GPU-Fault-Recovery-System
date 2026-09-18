from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional.regional_live_fixture import (
    RegionalFixtureAbort,
    RegionalFixtureError,
)
from tests.regional._cov95_destr_actions import (
    ActionHarness,
    fabric,
    reboot,
    workload_case,
)

MODULES = (reboot, workload_case, fabric)


def test_workload_cleanup_cannot_claim_absence_after_a_failed_kubernetes_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = ActionHarness(workload_case, tmp_path, monkeypatch)
    h.plan(tmp_path)

    def unavailable(value: Any, args: Any, kwargs: Any) -> Any:
        if "--ignore-not-found" in args:
            if kwargs.get("check", True):
                raise RegionalFixtureError("fake absence read unavailable")
            return ""
        return value

    h.transforms["regional.kubectl"] = unavailable
    code, report = h.execute(tmp_path)
    assert code == 1 and report["workload_residual"] is True, report
    assert "absence read unavailable" in report["workload_cleanup_error"], report


@pytest.mark.parametrize("module", MODULES)
def test_action_lifecycle_succeeds_and_all_requested_resources_are_cleaned(
    module: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = ActionHarness(module, tmp_path, monkeypatch)
    h.plan(tmp_path)
    code, report = h.execute(tmp_path)
    assert code == 0 and report["verdict"] == "PASS", report
    names = [name for name, _, _ in h.calls]
    if module is workload_case:
        assert (
            names.index("regional.wait_for_workflow")
            < names.index("workload.authorize_restart")
            < names.index("workload.wait_restarted")
        )
        authorized = next(
            args[0] for name, args, _ in h.calls if name == "workload.authorize_restart"
        )
        assert authorized == h.state
        assert names.index("regional.wait_for_workflow") < names.index(
            "workload.delete"
        ), names
        assert (
            report["workload_residual"] is False and report["prewarm_residuals"] == {}
        ), report
    else:
        assert names.index("host.create") < names.index("host.cleanup"), names
        assert report["probe_residuals"] == {}, report


def test_workload_restart_custody_refusal_prevents_pod_wait(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = ActionHarness(workload_case, tmp_path, monkeypatch)
    h.plan(tmp_path)
    h.failures["workload.authorize_restart"] = RegionalFixtureError(
        "restart custody rejected"
    )
    code, report = h.execute(tmp_path)
    assert code == 1 and "custody rejected" in report["error"]
    names = [name for name, _, _ in h.calls]
    assert "workload.wait_restarted" not in names
    assert "workload.delete" in names and "prewarm.cleanup" in names


@pytest.mark.parametrize("module", MODULES)
@pytest.mark.parametrize(
    "phase", ["regional.wait_for_workflow", "regional.provider_events"]
)
def test_action_error_never_skips_cleanup(
    module: Any, phase: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = ActionHarness(module, tmp_path, monkeypatch)
    h.plan(tmp_path)
    h.failures[phase] = RuntimeError(f"fake failure at {phase}")
    code, report = h.execute(tmp_path)
    assert code == 1 and phase in report["error"], report
    names = [name for name, _, _ in h.calls]
    assert (
        "workload.delete" if module is workload_case else "host.cleanup"
    ) in names, names


@pytest.mark.parametrize("module", MODULES)
@pytest.mark.parametrize("phase", ["before", "during-baseline"])
def test_action_expired_window_prevents_following_mutation(
    module: Any, phase: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = ActionHarness(module, tmp_path, monkeypatch)
    h.plan(tmp_path)
    if phase == "before":
        h.clock.sleep(61)
    else:
        h.advance_at[
            "prewarm.create" if module is workload_case else "host.execute"
        ] = 61
    code, report = h.execute(tmp_path, seconds=60)
    assert code == 1 and "window ended" in report["error"], report
    names = [name for name, _, _ in h.calls]
    assert "regional.wait_for_workflow" not in names, names


@pytest.mark.parametrize("module", MODULES)
def test_action_abort_runs_cleanup_and_propagates(
    module: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = ActionHarness(module, tmp_path, monkeypatch)
    h.plan(tmp_path)
    h.failures["regional.wait_for_workflow"] = RegionalFixtureAbort(2)
    with pytest.raises(RegionalFixtureAbort):
        h.execute(tmp_path)
    names = [name for name, _, _ in h.calls]
    assert (
        "workload.delete" if module is workload_case else "host.cleanup"
    ) in names, names


@pytest.mark.parametrize("module", MODULES)
@pytest.mark.parametrize("defect", ["preflight", "plan"])
def test_action_preflight_or_identity_failure_never_creates_resources(
    module: Any, defect: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = ActionHarness(module, tmp_path, monkeypatch)
    h.plan(tmp_path)
    if defect == "preflight":
        h.preflight["errors"] = ["fake preflight refusal"]
    else:
        h.preflight["release_id"] = "changed"
    with pytest.raises(RegionalFixtureError, match="preflight failed|plan drifted"):
        h.execute(tmp_path)
    assert h.calls == [], h.calls


@pytest.mark.parametrize("module", [reboot, fabric])
@pytest.mark.parametrize("defect", ["host-error", "residual", "final-read"])
def test_host_cleanup_or_postflight_failure_downgrades_success(
    module: Any, defect: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = ActionHarness(module, tmp_path, monkeypatch)
    h.plan(tmp_path)
    if defect == "host-error":
        h.failures["host.cleanup"] = RuntimeError("fake cleanup failure")
    elif defect == "residual":
        h.transforms["host.cleanup"] = lambda result, args, kwargs: {"pod": True}
    else:

        def cleanup(result: Any, args: Any, kwargs: Any) -> Any:
            h.failures["regional.node_snapshot"] = RuntimeError(
                "fake final read failure"
            )
            return result

        h.transforms["host.cleanup"] = cleanup
    code, report = h.execute(tmp_path)
    assert code == 1 and report["verdict"] == "FAIL", report
    if defect == "final-read":
        assert "fake final read failure" in report["postflight_error"], report


@pytest.mark.parametrize(
    "defect",
    [
        "not-submitted",
        "missing-command",
        "duplicate-command",
        "no-provider",
        "wrong-actor",
        "final-unready",
        "cpu",
    ],
)
def test_reboot_requires_unique_submission_and_provider_confirmation(
    defect: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = ActionHarness(reboot, tmp_path, monkeypatch)
    h.plan(tmp_path)
    if defect in {"not-submitted", "missing-command", "duplicate-command"}:

        def submission(value: dict[str, Any], args: Any, kwargs: Any) -> dict[str, Any]:
            if defect == "not-submitted":
                value["workflow"]["status"] = "FAILED"
                value["submission"]["state"] = "FAILED"
            elif defect == "missing-command":
                value["commands"] = []
            else:
                value["commands"] *= 2
            return value

        h.transforms["regional.store_snapshot"] = submission
    elif defect == "no-provider":
        h.transforms["regional.wait_provider_events"] = lambda *_a: []
    elif defect == "wrong-actor":
        h.transforms["regional.wait_provider_events"] = lambda *_a: [
            {"event_name": "RebootClusterNodes", "session_issuer_role_name": "foreign"}
        ]
    elif defect == "final-unready":
        h.transforms["regional.node_snapshot"] = lambda value, *_a: {
            **value,
            "ready": "False",
            "unschedulable": True,
        }
    else:
        h.transforms["regional.cpu_blast_snapshot"] = lambda *_a: {"changed": True}
    code, report = h.execute(tmp_path)
    assert code == 1 and (report.get("error") or report.get("errors")), report


@pytest.mark.parametrize(
    "defect",
    [
        "cache",
        "receipt",
        "overlap",
        "placement",
        "budget",
        "provider",
        "quarantine",
        "cpu",
    ],
)
def test_workload_restart_evidence_rejects_incomplete_or_unowned_recovery(
    defect: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = ActionHarness(workload_case, tmp_path, monkeypatch)
    h.plan(tmp_path)
    if defect == "cache":
        h.transforms["prewarm.cached_nodes"] = lambda *_a: []
    elif defect == "receipt":
        h.transforms["regional.post_xid_event"] = lambda *_a: {
            "receipt": {"status": 500}
        }
    elif defect in {"overlap", "placement"}:

        def target(value: dict[str, Any], args: Any, kwargs: Any) -> dict[str, Any]:
            if defect == "overlap":
                value["pods"][0]["uid"] = "src-0"
            else:
                value["pods"][0]["node"] = value["pods"][1]["node"]
            return value

        h.transforms["workload.wait_restarted"] = target
    elif defect == "budget":
        h.state["restart_budget"]["restart_count"] = 2
    elif defect == "provider":
        h.transforms["regional.provider_events"] = lambda *_a: [
            {"event_name": "BatchRebootClusterNodes"}
        ]
    elif defect == "quarantine":
        h.transforms["regional.node_snapshot"] = lambda value, *_a: {
            **value,
            "ownership_annotations": {"gpu-fault.io/incident-id": "foreign"},
        }
    else:
        h.transforms["regional.cpu_blast_snapshot"] = lambda *_a: {"changed": True}
    code, report = h.execute(tmp_path)
    assert code == 1 and (report.get("error") or report.get("errors")), report


@pytest.mark.parametrize(
    "defect",
    [
        "missing-ledger",
        "replay-failed",
        "notifications",
        "provider",
        "workload",
        "recovery",
    ],
)
def test_fabric_restart_replay_is_one_shot_and_preserves_postconditions(
    defect: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = ActionHarness(fabric, tmp_path, monkeypatch)
    h.plan(tmp_path)
    if defect == "missing-ledger":

        def host(value: dict[str, Any], args: Any, kwargs: Any) -> dict[str, Any]:
            if args[0] == "snapshot":
                value["ledger"] = []
            return value

        h.transforms["host.execute"] = host
    elif defect == "replay-failed":
        h.transforms["regional.executor_python"] = lambda *_a: {"status": "FAILED"}
    elif defect == "notifications":
        h.transforms["regional.store_snapshot"] = lambda value, *_a: {
            **value,
            "notifications": [],
        }
    elif defect == "provider":
        h.transforms["regional.provider_events"] = lambda *_a: [
            {"event_name": "BatchRebootClusterNodes"}
        ]
    elif defect == "workload":
        h.transforms["regional.business_workloads"] = lambda *_a: [{"name": "foreign"}]
    else:

        def recovery(value: dict[str, Any], args: Any, kwargs: Any) -> dict[str, Any]:
            if args[0] == "ensure-fabric-manager-active":
                raise RuntimeError("fake recovery failure")
            return value

        h.transforms["host.execute"] = recovery
    code, report = h.execute(tmp_path)
    assert code == 1 and report["verdict"] == "FAIL", report
    if defect == "missing-ledger":
        assert not any(name == "regional.executor_python" for name, _, _ in h.calls), (
            h.calls
        )

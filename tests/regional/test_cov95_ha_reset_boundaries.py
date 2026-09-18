from __future__ import annotations

import copy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import run_ha003_aurora_failover_reset as ha003
from scripts.e2e.regional import run_ha004_waiting_reclaim_reset as ha004
from scripts.e2e.regional.regional_commands import RegionalFixtureError
from tests.regional._cov95_ha_reset_harness import ResetHarness


@pytest.fixture(params=[ha003, ha004], ids=["aurora", "executor"])
def harness(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> ResetHarness:
    return ResetHarness(monkeypatch, tmp_path, request.param)


def test_expired_window_refuses_before_mutating_setup(harness: ResetHarness) -> None:
    harness.deadline = datetime(2000, 1, 1, tzinfo=timezone.utc)
    code, report = harness.execute()
    assert code == 1
    assert "maintenance window" in report["error"]
    assert "host-create" not in harness.events
    assert "watchdog-start" not in harness.events
    assert "write-xid46" not in harness.events


def test_settings_and_entrypoint_keep_their_real_contract(
    harness: ResetHarness, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = harness.module
    regional = harness.settings.regional
    values = [
        "--run-dir",
        str(harness.path),
        "--cpu-kubeconfig",
        str(regional.cpu_kubeconfig),
        "--gpu-kubeconfig",
        str(regional.gpu_kubeconfig),
        "--gpu-context",
        regional.gpu_context,
        "--namespace",
        regional.namespace,
        "--cluster-id",
        regional.cluster_id,
        "--region",
        regional.region,
        "--node",
        harness.settings.node,
        "--host-probe-image",
        harness.settings.host_probe_image,
        "--predecessor-evidence",
        str(harness.settings.predecessor_path),
    ]
    if module is ha003:
        values += ["--rds-cluster-id", harness.settings.rds_cluster_id]
    configured = module.configure(module.parser().parse_args(values))
    assert configured == harness.settings
    assert configured.environment()["GPU_FAULT_TARGET_NODE"] == "node-a"
    monkeypatch.setattr(
        module, "run_standard_case", lambda case: int(case.case_id == module.CASE_ID)
    )
    assert module.main() == 1


@pytest.mark.parametrize("defect", ["node-uid", "release", "agent", "profile"])
def test_plan_identity_drift_stops_before_any_setup(
    harness: ResetHarness, defect: str
) -> None:
    if defect == "node-uid":
        harness.api.node["uid"] = "replacement"
    elif defect == "release":
        harness.api.baseline["release_id"] = "different"
        if harness.module is ha003:
            harness.api.evidence_identity = lambda: {
                "release_id": "different",
                "cluster_id": "unit-cluster",
            }
            harness.settings.predecessor_path.write_text(
                '{"case_id":"GF-REGIONAL-DESTR-008","verdict":"PASS","release_id":"different","cluster_id":"unit-cluster"}'
            )
    elif defect == "agent":
        harness.api.baseline["agent"]["generation"] = 2
    else:
        harness.api.baseline["profile"]["profile_version"] = "different"
    with pytest.raises(RegionalFixtureError, match="plan drifted"):
        harness.execute()
    assert "host-create" not in harness.events


@pytest.mark.parametrize(
    "defect", ["command-id", "command-count", "host-ledger", "blocked"]
)
def test_terminal_evidence_must_match_the_captured_reset(
    harness: ResetHarness, defect: str
) -> None:
    state = harness.api.final
    if defect == "command-id":
        state["commands"][4]["command_id"] = "foreign-command"
    elif defect == "command-count":
        state["commands"].append(copy.deepcopy(state["commands"][4]))
    elif defect == "host-ledger":
        harness.host.after["ledger"] = []
    else:
        state["workflow"]["status"] = "BLOCKED"
        state["workflow"]["blocked_reasons"] = ["unit blocked"]
    code, report = harness.execute()
    assert code == 1
    assert report["errors"], "incorrect terminal evidence must affect the verdict"
    assert report["cleanup"]["errors"] == []


@pytest.mark.parametrize("defect", ["busy-client", "missing-incident", "residuals"])
def test_missing_cleanup_identity_or_dirty_host_is_not_success(
    harness: ResetHarness, defect: str
) -> None:
    if defect == "busy-client":
        harness.host.before["compute_clients"] = [{"pid": "unit-client"}]
    elif defect == "missing-incident":
        for state in harness.api.states:
            state["incident"] = {}
    else:
        harness.host.residuals["pod"] = True
    code, report = harness.execute()
    assert code == 1
    if defect == "busy-client":
        assert "active NVIDIA clients" in report["error"]
        assert "write-xid46" not in harness.events
    elif defect == "missing-incident":
        assert "cleanup incident identity" in report["error"]
    else:
        assert "host probe resources remain" in report["cleanup"]["errors"]


@pytest.mark.parametrize("outcome", ["SUCCEEDED", "FAILED", "exception"])
def test_owned_validation_restore_is_attempted_and_its_result_is_checked(
    harness: ResetHarness, monkeypatch: pytest.MonkeyPatch, outcome: str
) -> None:
    wait = harness.api.wait_for_workflow

    def finish(**kwargs: Any) -> dict[str, Any]:
        harness.api.node["ownership_annotations"] = {"unit-owner": "incident-unit"}
        return wait(**kwargs)

    monkeypatch.setattr(harness.api, "wait_for_workflow", finish)
    if outcome == "exception":
        harness.restore_failure = RuntimeError("unit restore failure")
    else:
        harness.restore_status = outcome
    code, report = harness.execute()
    assert "node-restore" in harness.events
    if outcome == "SUCCEEDED":
        assert report["cleanup"]["errors"] == []
    else:
        assert code == 1
        assert any("restore" in error for error in report["cleanup"]["errors"]), report


@pytest.mark.parametrize(
    "defect", ["ready", "cordon", "provider", "cpu-blast", "result-code", "step-count"]
)
def test_aurora_verdict_checks_recovery_and_blast_radius(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, defect: str
) -> None:
    harness = ResetHarness(monkeypatch, tmp_path, ha003)
    wait = harness.api.wait_for_workflow

    def finish(**kwargs: Any) -> dict:
        if defect == "ready":
            harness.api.node["ready"] = "False"
        elif defect == "cordon":
            harness.api.node["unschedulable"] = False
        elif defect == "provider":
            harness.api.provider_events = [{"event_name": "unit-provider-mutation"}]
        elif defect == "cpu-blast":
            harness.api.cpu_blast = {"unit": "changed"}
        elif defect == "result-code":
            harness.api.log_text = "could not report result\nrejected request (500)\n"
        else:
            harness.api.final["workflow"]["step_executions"] = []
        return wait(**kwargs)

    monkeypatch.setattr(harness.api, "wait_for_workflow", finish)
    code, report = harness.execute()
    assert code == 1
    assert report["errors"], f"{defect} must invalidate failover recovery"


@pytest.mark.parametrize(
    "defect", ["failed-predecessor", "unavailable-rds", "no-reader", "one-executor"]
)
def test_preflight_checks_predecessor_and_failover_capacity(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, defect: str
) -> None:
    module = ha004 if defect == "one-executor" else ha003
    harness = ResetHarness(monkeypatch, tmp_path, module)
    if defect == "failed-predecessor":
        harness.settings.predecessor_path.write_text("{}")
    elif defect == "unavailable-rds":
        harness.api.rds_status = "failing-over"
    elif defect == "no-reader":
        harness.api.rds_members = 1
    else:
        harness.api.document["spec"]["replicas"] = 1
    with pytest.raises(RegionalFixtureError, match="preflight failed"):
        harness.execute()
    assert "host-create" not in harness.events


@pytest.mark.parametrize("mode", ["timeout", "terminal", "ambiguous"])
def test_reset_capture_does_not_invent_an_open_command(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, mode: str
) -> None:
    harness = ResetHarness(monkeypatch, tmp_path, ha003)
    value = copy.deepcopy(harness.api.final)
    if mode == "timeout":
        value["commands"][4]["status"] = "PENDING"
    elif mode == "ambiguous":
        value["commands"].append(copy.deepcopy(value["commands"][4]))
        assert ha003.reset_command_status(value) is None
    harness.api.states = [value]
    with pytest.raises(RegionalFixtureError, match="not observed|completed before"):
        ha003.wait_reset_claim(
            harness.api,
            harness.settings,
            marker="unit",
            observed_after=datetime.now(timezone.utc),
            timeout_seconds=1,
        )


def test_rds_timeout_without_observer_and_missing_object_response_fail_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    harness = ResetHarness(monkeypatch, tmp_path, ha003)
    with pytest.raises(RegionalFixtureError, match="did not converge"):
        ha003.wait_rds_failover(
            harness.settings, previous_writer="old-writer", timeout_seconds=1
        )
    monkeypatch.setattr(
        harness.api, "run", lambda *a, **kw: type("Result", (), {"stdout": "[]"})()
    )
    with pytest.raises(RegionalFixtureError, match="not a JSON object"):
        ha003.aws_rds(harness.settings, "describe-db-clusters")

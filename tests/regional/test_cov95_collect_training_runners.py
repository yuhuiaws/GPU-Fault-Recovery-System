"""Full collector case orchestration and ownership cleanup via fake boundaries."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import run_collect016_training_recovery as c016
from scripts.e2e.regional import run_collect017_efa_plugin as c017
from scripts.e2e.regional import run_collect021_late_xid_after_pod_death as c021
from tests.regional import _cov95_collect_training as training_support
from tests.regional._cov95_collect_net import no_external_effects  # noqa: F401

training_case = training_support.training_case


def execute(harness: Any) -> int:
    return harness.module.execute_case(
        harness.settings,
        harness.root,
        1,
        datetime.now(timezone.utc) + timedelta(hours=1),
    )


def report(harness: Any) -> dict[str, Any]:
    return json.loads(
        (
            harness.root
            / "cases"
            / harness.module.CASE_ID
            / f"{harness.module.CASE_ID}.json"
        ).read_text()
    )


def test_full_case_uses_real_sections_then_cleans_owned_resources(
    training_case: Any,
) -> None:
    harness = training_case
    assert execute(harness) == 0
    result = report(harness)
    assert result["verdict"] == "PASS"
    assert result["cluster_id"] == "cluster-a"
    names = [call[0] for call in harness.calls]
    assert "collector-cleanup" in names
    assert "prewarm-cleanup" in names
    assert "workload-delete" in names
    if harness.module is c016:
        assert set(result) >= {"a", "b", "d"}
        assert names.index("workload-delete") < names.index("reset-section")
        assert names.count("authorize-restart") == names.count("restarted") == 2
    elif harness.module is c017:
        assert set(result) >= {"a", "b", "c"}
        assert len(harness.plugins) == 2
        assert names.count("plugin-restore") == 2
    else:
        assert names.index("kill-workload") < names.index("write-xid")
        assert result["late_xid"]["restart_budget"]["restart_count"] == 1
        authorized = harness.workloads[0].restart_state
        assert authorized["workflow"] is authorized["recovery_workflow"]
        assert authorized["incident"] is authorized["recovery_incident"]
        assert authorized["commands"]
        assert authorized["observations"] == []
        assert names.index("cpu-read") < names.index("authorize-restart")
        assert names.index("authorize-restart") < names.index("restarted")
        assert names.count("cpu-read") == 2


@pytest.mark.parametrize("training_case", [c016, c021], indirect=True)
def test_restart_custody_rejection_prevents_pod_wait_and_preserves_cleanup(
    training_case: Any,
) -> None:
    harness = training_case
    harness.fail_at = "authorize-restart"
    assert execute(harness) == 1
    names = [call[0] for call in harness.calls]
    assert "authorize-restart" in names
    assert "restarted" not in names
    assert "reset-section" not in names
    assert "workload-delete" in names
    assert "collector-cleanup" in names


@pytest.mark.parametrize("training_case", [c021], indirect=True)
def test_late_xid_waits_for_legitimate_passive_containment_before_final_node_check(
    training_case: Any,
) -> None:
    harness = training_case
    harness.problem = "passive-cordon"
    assert execute(harness) == 0, (
        "temporary passive containment is not a MONITOR_ONLY XID mutation"
    )
    readings = [args for name, args in harness.calls if name == "node-snapshot"]
    assert readings[-1] == (True, True), (
        "the final unchanged-node proof follows the replacement attempt"
    )
    assert (True, False) not in readings, (
        "an in-flight passive cordon cannot be mistaken for the final node state"
    )


@pytest.mark.parametrize(
    "failure",
    [
        "collector-create",
        "prewarm-create",
        "running",
        "workload-delete",
        "collector-cleanup",
        "prewarm-cleanup",
    ],
)
def test_stage_failure_still_attempts_registered_cleanup(
    training_case: Any, failure: str
) -> None:
    harness = training_case
    harness.fail_at = failure
    assert execute(harness) == 1
    result = report(harness)
    assert result["verdict"] == "FAIL"
    assert result.get("error") or result.get("errors") or result.get("cleanup_error"), (
        "every failed stage must leave its failure reason"
    )
    names = [call[0] for call in harness.calls]
    if harness.collectors:
        assert "collector-cleanup" in names
    if harness.workloads:
        assert "workload-delete" in names
    if "prewarm-create" in names:
        assert "prewarm-cleanup" in names


@pytest.mark.parametrize(
    "problem", ["section-error", "probe-residual", "prewarm-residual"]
)
def test_bad_phase_or_residual_refuses_success(
    training_case: Any, problem: str
) -> None:
    harness = training_case
    harness.problem = problem
    assert execute(harness) == 1
    result = report(harness)
    assert result["verdict"] == "FAIL"
    if problem == "section-error" and harness.module is c016:
        assert not any(call[0] == "reset-section" for call in harness.calls), (
            "failed restart must not progress to reset"
        )
    if problem == "section-error" and harness.module is c017:
        assert harness.plugins == [], (
            "failed EFA recovery must not progress to plugin exclusion"
        )


@pytest.mark.parametrize(
    "problem", ["predecessor", "site", "focused", "existing-workload"]
)
def test_preflight_rejects_each_missing_premise_without_creating_resources(
    training_case: Any, problem: str
) -> None:
    harness = training_case
    if problem == "predecessor":
        harness.predecessor = False
    elif problem == "site":
        harness.settings.site_file.unlink()
    elif problem == "focused":
        harness.focused_status = 1
    else:
        harness.problem = problem
    preflight = harness.module.read_only_preflight(harness.settings, harness.root)
    assert len(preflight["errors"]) == 1
    with pytest.raises(RuntimeError, match="preflight"):
        execute(harness)
    assert harness.collectors == [] and harness.workloads == [], (
        "failed admission must not allocate fixtures"
    )


@pytest.mark.parametrize("training_case", [c016, c021], indirect=True)
@pytest.mark.parametrize("failure", ["incident-read", "incident-restore", "seed-read"])
def test_seed_cleanup_failures_are_recorded_before_resource_release(
    training_case: Any, failure: str
) -> None:
    harness = training_case
    harness.fail_at = failure
    assert execute(harness) == 1
    result = report(harness)
    assert result["errors"], "failed seed cleanup must downgrade the case"
    assert any(call[0] == "collector-cleanup" for call in harness.calls), (
        "seed cleanup failure must not suppress collector cleanup"
    )


@pytest.mark.parametrize("training_case", [c017], indirect=True)
@pytest.mark.parametrize(
    "problem",
    [
        "incident-delay",
        "planning-delay",
        "planning-timeout",
        "request-id",
        "pod-change",
        "restart-workload",
    ],
)
def test_plugin_phases_wait_for_observation_and_refuse_changed_workload(
    training_case: Any, problem: str
) -> None:
    harness = training_case
    harness.problem = problem
    status = execute(harness)
    assert status == (0 if problem in {"incident-delay", "planning-delay"} else 1)
    if harness.plugins:
        assert any(call[0] == "plugin-restore" for call in harness.calls), (
            "plugin failure must retain a restore attempt"
        )


@pytest.mark.parametrize("training_case", [c017], indirect=True)
@pytest.mark.parametrize(
    "failure",
    [
        "restore-efa",
        "plugin-exclude",
        "plugin-restore",
        "plugin-discover",
        "allocatable",
    ],
)
def test_plugin_and_efa_failure_do_not_suppress_outer_cleanup(
    training_case: Any, failure: str
) -> None:
    harness = training_case
    harness.fail_at = failure
    assert execute(harness) == 1
    assert report(harness)["verdict"] == "FAIL"
    assert any(call[0] == "collector-cleanup" for call in harness.calls), (
        "plugin failure must not suppress collector cleanup"
    )


@pytest.mark.parametrize("module", [c016, c017, c021])
@pytest.mark.parametrize("explicit", [False, True])
def test_configuration_and_plan_keep_bound_identity(
    module: Any, explicit: bool, tmp_path: Path, monkeypatch: Any
) -> None:
    regional = type(
        "Settings", (), {"environment": lambda self: {"CLUSTER": "cluster-a"}}
    )()
    monkeypatch.setattr(module, "settings_from_arguments", lambda args: regional)
    argv = [
        "--run-dir",
        str(tmp_path),
        "--site-file",
        str(tmp_path / "site"),
        "--host-probe-image",
        "example",
        *(["--node", "node-a"] if module is c017 else []),
    ]
    if explicit:
        argv += ["--predecessor-evidence", str(tmp_path / "previous.json")]
    configured = module.configure(module.parser().parse_args(argv))
    assert configured.environment()["CLUSTER"] == "cluster-a"
    assert configured.predecessor_path == (
        tmp_path / "previous.json"
        if explicit
        else tmp_path
        / "cases"
        / module.PREDECESSOR_CASE_ID
        / f"{module.PREDECESSOR_CASE_ID}.json"
    )
    preflight = {
        "release_id": "release-a",
        "predecessor": {"valid": True},
        "candidate_nodes": [{"uid": "uid-a"}],
        "node": {"uid": "uid-a"},
    }
    details = module.plan_details(configured, preflight)
    assert details["preflight_identity"]["release_id"] == "release-a"

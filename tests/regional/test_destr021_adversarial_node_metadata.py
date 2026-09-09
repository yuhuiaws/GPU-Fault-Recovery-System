"""Contract tests for GF-REGIONAL-DESTR-021.

Every verdict is judged against synthetic evidence -- the remote-command bundle
of the EFA remediation workflow, node snapshots, the annotation writer's report
-- once on the intended run and once per way the run can be wrong. The writer
itself runs with an injected fake kubectl. Nothing here touches a cluster; the
runner's live phases only call these functions.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import destr021_verdicts as verdicts
from scripts.e2e.regional import run_destr021_adversarial_node_metadata as destr021
from scripts.e2e.regional.probes import destr021_annotation_writer as writer_module
from scripts.e2e.regional.regional_case_contract import RegionalCaseMetadata

ROOT = Path(__file__).resolve().parents[2]
NODE = "node-a"
FOREIGN = "acceptance-foreign-1"
T0 = datetime(2026, 9, 6, 10, 0, tzinfo=timezone.utc)
EFA = 32


def _text(errors: list[str]) -> str:
    return "\n".join(errors)


# --------------------------------------------------------------------------- #
# Case contract
# --------------------------------------------------------------------------- #
def test_the_confirmation_names_this_case_and_the_predecessor_is_destr020() -> None:
    metadata = RegionalCaseMetadata(
        case_id=destr021.CASE_ID,
        title="",
        category="regional-destructive-acceptance",
        level="staging",
        risk="live-node-mutation",
        automation="manual",
        procedure="docs/x.md#gf-regional-destr-021",
        predecessor=destr021.PREDECESSOR_CASE_ID,
    )
    prefix = metadata.confirmation.removesuffix("EXECUTE")
    assert prefix == "DESTR021_", prefix
    assert destr021.CONFIRMATION.startswith(prefix), destr021.CONFIRMATION
    assert destr021.PREDECESSOR_CASE_ID == "GF-REGIONAL-DESTR-020", (
        "the positional predecessor gates execution"
    )
    assert verdicts.CASE_ID == "GF-REGIONAL-DESTR-021", verdicts.CASE_ID
    assert verdicts.RISK == "live-node-mutation", verdicts.RISK


def test_the_expected_steps_are_collect017_a_and_forbid_every_reset_path() -> None:
    assert verdicts.EXPECTED_STEPS[0] == "FREEZE_EVIDENCE", verdicts.EXPECTED_STEPS
    assert "MARK_UNSCHEDULABLE" in verdicts.EXPECTED_STEPS, verdicts.EXPECTED_STEPS
    assert "RESTART_EFA_DEVICE_PLUGIN" in verdicts.EXPECTED_STEPS, (
        verdicts.EXPECTED_STEPS
    )
    assert verdicts.EXPECTED_STEPS[-1] == "RESTORE_SCHEDULING", verdicts.EXPECTED_STEPS
    for operation in ("RESET_GPU", "RESTART_NODE", "REPLACE_NODE"):
        assert operation in verdicts.FORBIDDEN_OPERATIONS, operation
    assert not set(verdicts.EXPECTED_STEPS) & verdicts.FORBIDDEN_OPERATIONS, (
        "the expected steps must not overlap the forbidden set"
    )


# --------------------------------------------------------------------------- #
# Pre-seed
# --------------------------------------------------------------------------- #
def test_the_preseed_is_stale_by_more_than_the_restart_timeout() -> None:
    annotations = verdicts.preseed_annotations(
        foreign_operation="op-1", foreign_incident=FOREIGN, now=T0
    )
    assert set(annotations) == set(verdicts.PLUGIN_RESTART_ANNOTATIONS), annotations
    started = verdicts.parse_time(annotations[verdicts.ANNOTATION_STARTED])
    assert started is not None, "started-at must parse"
    age = (T0 - started).total_seconds()
    assert age >= verdicts.RESTART_TIMEOUT_SECONDS + verdicts.STALE_MARGIN_SECONDS, age
    assert annotations[verdicts.ANNOTATION_INCIDENT] == FOREIGN, annotations


def test_a_preseed_that_did_not_land_or_is_fresh_is_refused() -> None:
    good = verdicts.preseed_annotations(
        foreign_operation="op-1",
        foreign_incident=FOREIGN,
        now=datetime.now(timezone.utc),
    )
    assert verdicts.preseed_errors(good, foreign_incident=FOREIGN) == [], "intended"
    missing = dict(good)
    missing[verdicts.ANNOTATION_INCIDENT] = "someone-else"
    assert "foreign incident" in _text(
        verdicts.preseed_errors(missing, foreign_incident=FOREIGN)
    ), "a different incident must be refused"
    fresh = dict(good)
    fresh[verdicts.ANNOTATION_STARTED] = datetime.now(timezone.utc).isoformat()
    assert "older than the restart timeout" in _text(
        verdicts.preseed_errors(fresh, foreign_incident=FOREIGN)
    ), "a fresh annotation would be refused by the adapter, not taken over"
    garbage = dict(good)
    garbage[verdicts.ANNOTATION_STARTED] = "yesterday"
    assert "not parsable" in _text(
        verdicts.preseed_errors(garbage, foreign_incident=FOREIGN)
    ), "an unparsable started-at is never stale (fail closed)"


# --------------------------------------------------------------------------- #
# Preflight
# --------------------------------------------------------------------------- #
def _node() -> dict[str, Any]:
    return {
        "uid": "uid-1",
        "boot_id": "boot-1",
        "ready": "True",
        "unschedulable": False,
        "taints": [],
        "ownership_annotations": {},
    }


def _preflight(**overrides: Any) -> list[str]:
    arguments: dict[str, Any] = {
        "node": _node(),
        "node_annotations": {},
        "workloads": [],
        "queue": {"depth": 0},
        "remote_commands": {"open_by_cluster": {}},
        "efa_allocatable": EFA,
        "bound_efa_functions": 4,
        "predecessor_valid": True,
        "tests_passed": True,
        "site_file_exists": True,
    }
    arguments.update(overrides)
    return verdicts.preflight_errors(**arguments)


def test_the_intended_preflight_is_clean() -> None:
    assert _preflight() == [], "intended preflight"


@pytest.mark.parametrize(
    ("overrides", "fragment"),
    [
        ({"node": {**_node(), "ready": "False"}}, "not Ready"),
        ({"node": {**_node(), "unschedulable": True}}, "already unschedulable"),
        ({"node": {**_node(), "taints": [{"key": "x"}]}}, "pre-existing taints"),
        (
            {
                "node": {
                    **_node(),
                    "ownership_annotations": {"gpu-fault.io/incident-id": "i"},
                }
            },
            "workflow ownership",
        ),
        (
            {"node_annotations": {verdicts.ANNOTATION_OPERATION: "op"}},
            "already carries plugin-restart annotations",
        ),
        ({"node_annotations": {verdicts.TICK_ANNOTATION: "3"}}, "tick annotation"),
        ({"workloads": [{"name": "job"}]}, "non-system running Pods"),
        ({"efa_allocatable": 0}, "no allocatable EFA"),
        ({"bound_efa_functions": 0}, "no bound EFA function"),
        ({"queue": {"depth": 1}}, "queue is not empty"),
        ({"remote_commands": {"open_by_cluster": {"c": 1}}}, "remote command queue"),
        ({"predecessor_valid": False}, "predecessor evidence is not PASS"),
        ({"tests_passed": False}, "focused regression tests failed"),
        ({"site_file_exists": False}, "site file does not exist"),
    ],
)
def test_each_preflight_refusal_names_its_cause(
    overrides: dict[str, Any], fragment: str
) -> None:
    errors = _preflight(**overrides)
    assert any(fragment in error for error in errors), (fragment, errors)


# --------------------------------------------------------------------------- #
# Bundle fabrication
# --------------------------------------------------------------------------- #
def _snapshot(unschedulable: bool, taints: list[str], version: str) -> dict[str, Any]:
    return {
        "unschedulable": unschedulable,
        "taint_keys": taints,
        "resource_version": version,
    }


def _bundle(**overrides: Any) -> dict[str, Any]:
    steps = list(verdicts.EXPECTED_STEPS)
    workflow = {
        "request_id": "wf-1",
        "status": "SUCCEEDED",
        "official_steps": [
            {
                "operation": name,
                "parameters": (
                    {"expected_count": EFA}
                    if name == "RESTART_EFA_DEVICE_PLUGIN"
                    else {}
                ),
            }
            for name in steps
        ],
        "step_executions": [
            {"step_index": index, "operation": name, "status": "SUCCEEDED"}
            for index, name in enumerate(steps)
        ],
    }
    commands = [
        {
            "operation": "MARK_UNSCHEDULABLE",
            "status": "SUCCEEDED",
            "result_details": {
                "isolated_nodes": [NODE],
                "node_baselines": {
                    NODE: {
                        "before": _snapshot(False, [], "100"),
                        "after": _snapshot(True, [verdicts.QUARANTINE_TAINT], "104"),
                    }
                },
            },
        },
        {
            "operation": "RESTART_EFA_DEVICE_PLUGIN",
            "status": "SUCCEEDED",
            "step": {"parameters": {"expected_count": EFA}},
            "result_details": {
                "node_results": {
                    NODE: {
                        "allocatable": EFA,
                        "expected": EFA,
                        "already_healthy": False,
                        "replacement_pod_uids": ["pod-2"],
                        "took_over_incident": FOREIGN,
                    }
                }
            },
        },
        {
            "operation": "RESTORE_SCHEDULING",
            "status": "SUCCEEDED",
            "result_details": {
                "restored_nodes": [NODE],
                "node_baselines": {
                    NODE: {
                        "before": _snapshot(True, [verdicts.QUARANTINE_TAINT], "230"),
                        "after": _snapshot(False, [], "233"),
                    }
                },
            },
        },
    ]
    bundle: dict[str, Any] = {
        "workflow": workflow,
        "incident": {"incident_id": "inc-1", "state": "RECOVERED"},
        "remote_commands": commands,
    }
    bundle.update(overrides)
    return bundle


def _with_command(bundle: dict[str, Any], operation: str, **changes: Any) -> dict:
    commands = []
    for item in bundle["remote_commands"]:
        if item["operation"] == operation:
            item = json.loads(json.dumps(item))
            for key, value in changes.items():
                item[key] = value
        commands.append(item)
    return {**bundle, "remote_commands": commands}


# --------------------------------------------------------------------------- #
# Workflow shape
# --------------------------------------------------------------------------- #
def test_the_intended_workflow_passes() -> None:
    assert verdicts.workflow_errors(_bundle()) == [], "intended workflow"


def test_a_workflow_that_escalated_or_did_not_recover_fails() -> None:
    bundle = _bundle()
    bundle["workflow"]["step_executions"].append(
        {"step_index": 9, "operation": "RESET_GPU", "status": "SUCCEEDED"}
    )
    assert "forbidden operations" in _text(verdicts.workflow_errors(bundle)), (
        "a reset must never be reachable from an EFA remediation"
    )
    failed = _bundle()
    failed["workflow"]["status"] = "FAILED"
    failed["incident"]["state"] = "ESCALATED"
    errors = _text(verdicts.workflow_errors(failed))
    assert "not SUCCEEDED" in errors and "not RECOVERED" in errors, errors
    short = _bundle()
    short["workflow"]["official_steps"] = short["workflow"]["official_steps"][:3]
    assert "official steps" in _text(verdicts.workflow_errors(short)), "step list"


# --------------------------------------------------------------------------- #
# A3: takeover and expected_count
# --------------------------------------------------------------------------- #
def test_the_stale_foreign_annotation_is_taken_over_and_cleared() -> None:
    errors = verdicts.takeover_errors(
        _bundle(), node=NODE, foreign_incident=FOREIGN, final_annotations={}
    )
    assert errors == [], errors


def test_no_takeover_or_a_leftover_annotation_fails() -> None:
    bundle = _with_command(
        _bundle(),
        "RESTART_EFA_DEVICE_PLUGIN",
        result_details={"node_results": {NODE: {"allocatable": EFA, "expected": EFA}}},
    )
    assert "did not take over" in _text(
        verdicts.takeover_errors(
            bundle, node=NODE, foreign_incident=FOREIGN, final_annotations={}
        )
    ), "a restart that ignored the poison pill would have refused, not healed"
    left = verdicts.takeover_errors(
        _bundle(),
        node=NODE,
        foreign_incident=FOREIGN,
        final_annotations={verdicts.ANNOTATION_OPERATION: "op-x"},
    )
    assert "left on the node" in _text(left), left
    absent = verdicts.takeover_errors(
        {**_bundle(), "remote_commands": []},
        node=NODE,
        foreign_incident=FOREIGN,
        final_annotations={},
    )
    assert "no RESTART_EFA_DEVICE_PLUGIN remote command" in _text(absent), absent
    other_node = verdicts.takeover_errors(
        _bundle(), node="node-b", foreign_incident=FOREIGN, final_annotations={}
    )
    assert "no node_results" in _text(other_node), other_node


def test_expected_count_must_be_positive_and_match_the_efa_allocatable() -> None:
    assert (
        verdicts.expected_count_errors(_bundle(), node=NODE, baseline_efa=EFA) == []
    ), "intended"
    zero = _with_command(
        _bundle(),
        "RESTART_EFA_DEVICE_PLUGIN",
        step={"parameters": {"expected_count": 0}},
    )
    zero["workflow"]["official_steps"] = []
    assert "fail closed" in _text(
        verdicts.expected_count_errors(zero, node=NODE, baseline_efa=EFA)
    ), "a zero expected_count is exactly what ARCH-A3 rejects"
    missing = _with_command(
        _bundle(), "RESTART_EFA_DEVICE_PLUGIN", step={"parameters": {}}
    )
    missing["workflow"]["official_steps"] = []
    assert "expected_count=None" in _text(
        verdicts.expected_count_errors(missing, node=NODE, baseline_efa=EFA)
    ), "a missing expected_count is reported as such"
    drift = verdicts.expected_count_errors(_bundle(), node=NODE, baseline_efa=EFA + 1)
    assert "differs from the node's EFA allocatable" in _text(drift), drift
    low = _with_command(
        _bundle(),
        "RESTART_EFA_DEVICE_PLUGIN",
        result_details={
            "node_results": {
                NODE: {
                    "allocatable": EFA - 1,
                    "expected": EFA,
                    "took_over_incident": FOREIGN,
                }
            }
        },
    )
    assert "below expected" in _text(
        verdicts.expected_count_errors(low, node=NODE, baseline_efa=EFA)
    ), "a node reporting fewer devices than expected did not recover"


def test_expected_count_falls_back_to_the_official_step_parameters() -> None:
    bundle = _with_command(_bundle(), "RESTART_EFA_DEVICE_PLUGIN", step={})
    assert verdicts.expected_count_errors(bundle, node=NODE, baseline_efa=EFA) == [], (
        "official_steps carry the same parameters when the command copy lacks them"
    )


# --------------------------------------------------------------------------- #
# A8: before/after baselines
# --------------------------------------------------------------------------- #
def test_the_observed_baselines_pass_on_the_intended_run() -> None:
    assert verdicts.baseline_errors(_bundle(), node=NODE) == [], "intended"


def test_missing_or_unchanged_baselines_fail() -> None:
    absent = _with_command(_bundle(), "MARK_UNSCHEDULABLE", result_details={})
    assert "MARK_UNSCHEDULABLE recorded no node_baselines" in _text(
        verdicts.baseline_errors(absent, node=NODE)
    ), "the isolate step must snapshot"
    no_restore = _with_command(_bundle(), "RESTORE_SCHEDULING", result_details={})
    assert "RESTORE_SCHEDULING recorded no node_baselines" in _text(
        verdicts.baseline_errors(no_restore, node=NODE)
    ), "the restore step must snapshot"
    stale = _with_command(
        _bundle(),
        "RESTORE_SCHEDULING",
        result_details={
            "node_baselines": {
                NODE: {
                    "before": _snapshot(True, [verdicts.QUARANTINE_TAINT], "230"),
                    "after": _snapshot(True, [verdicts.QUARANTINE_TAINT], "230"),
                }
            }
        },
    )
    errors = _text(verdicts.baseline_errors(stale, node=NODE))
    assert "does not show the node schedulable" in errors, errors
    assert "still carries the quarantine taint" in errors, errors
    assert "not re-read after the patch" in errors, errors
    partial = _with_command(
        _bundle(),
        "MARK_UNSCHEDULABLE",
        result_details={
            "node_baselines": {NODE: {"before": {"unschedulable": False}, "after": {}}}
        },
    )
    assert "lacks" in _text(verdicts.baseline_errors(partial, node=NODE)), (
        "every snapshot carries the three keys"
    )


# --------------------------------------------------------------------------- #
# A2: RESTORE_SCHEDULING under the race
# --------------------------------------------------------------------------- #
def test_restore_scheduling_succeeded_under_the_race() -> None:
    timeline = [
        {"operation": "RESTORE_SCHEDULING", "status": "WAITING"},
        {"operation": "RESTORE_SCHEDULING", "status": "SUCCEEDED"},
    ]
    assert verdicts.restore_errors(_bundle(), timeline_statuses=timeline) == [], (
        "a WAITING on the way is the retry, not a failure"
    )
    waiting = [
        {
            "step_index": 6,
            "operation": "RESTORE_SCHEDULING",
            "status": "WAITING",
            "details": {"patch_conflict_retry": [NODE]},
        }
    ]
    assert verdicts.conflict_retries_observed(waiting) == 1, "conflicts are counted"
    assert verdicts.conflict_retries_observed([{"details": {}}]) == 0, "none"


def test_a_failed_or_never_run_restore_fails() -> None:
    failed = _bundle()
    failed["workflow"]["step_executions"][-1]["status"] = "FAILED"
    assert "not SUCCEEDED" in _text(
        verdicts.restore_errors(failed, timeline_statuses=[])
    ), "a FAILED restore leaves the node cordoned for good"
    observed = verdicts.restore_errors(
        _bundle(),
        timeline_statuses=[{"operation": "RESTORE_SCHEDULING", "status": "FAILED"}],
    )
    assert "observed FAILED during the race" in _text(observed), observed
    never = _bundle()
    never["workflow"]["step_executions"] = never["workflow"]["step_executions"][:-1]
    assert "never executed" in _text(
        verdicts.restore_errors(never, timeline_statuses=[])
    ), "restore must run"
    command_failed = _with_command(_bundle(), "RESTORE_SCHEDULING", status="FAILED")
    assert "command is 'FAILED'" in _text(
        verdicts.restore_errors(command_failed, timeline_statuses=[])
    ), "the remote command must be SUCCEEDED too"


# --------------------------------------------------------------------------- #
# Writer report
# --------------------------------------------------------------------------- #
def _report(**overrides: Any) -> dict[str, Any]:
    report = {
        "patches": 400,
        "elapsed_seconds": 200.0,
        "conflicts": 3,
        "stopped": True,
        "cleared": True,
        "exceeded_max_seconds": False,
    }
    report.update(overrides)
    return report


def test_the_writer_report_passes_when_the_race_was_real_and_cleaned() -> None:
    assert verdicts.writer_errors(_report()) == [], "intended"


def test_the_writer_is_judged_by_coverage_not_by_the_requested_cadence() -> None:
    """Live 2026-09-08: 34 patches in 62.9 s (one every 1.85 s) spanned a 26 s
    workflow whose node-writing steps each took 5 s or more; that race was
    real even though the 0.5 s request was never achievable from the host."""
    live = _report(patches=34, elapsed_seconds=62.944)
    assert verdicts.writer_errors(live) == [], live
    assert verdicts.writer_achieved_interval(live) == pytest.approx(1.851, abs=0.01)
    assert (
        verdicts.writer_achieved_interval({"patches": 0, "elapsed_seconds": 5}) is None
    )


@pytest.mark.parametrize(
    ("overrides", "fragment"),
    [
        ({"patches": 5, "elapsed_seconds": 2.5}, "race was not real"),
        ({"patches": 20, "elapsed_seconds": 100.0}, "pass between ticks"),
        ({"stopped": False}, "did not stop"),
        ({"cleared": False}, "was not cleared"),
        ({"exceeded_max_seconds": True}, "hard bound"),
    ],
)
def test_each_writer_defect_is_named(overrides: dict[str, Any], fragment: str) -> None:
    errors = verdicts.writer_errors(_report(**overrides))
    assert any(fragment in error for error in errors), (fragment, errors)


# --------------------------------------------------------------------------- #
# Final node
# --------------------------------------------------------------------------- #
def _final(**overrides: Any) -> list[str]:
    arguments: dict[str, Any] = {
        "baseline": _node(),
        "final": _node(),
        "final_annotations": {},
        "baseline_efa": EFA,
        "final_efa": EFA,
    }
    arguments.update(overrides)
    return verdicts.node_final_errors(**arguments)


def test_the_final_node_is_clean_on_the_intended_run() -> None:
    assert _final() == [], "intended"


@pytest.mark.parametrize(
    ("overrides", "fragment"),
    [
        ({"final": {**_node(), "unschedulable": True}}, "left cordoned"),
        (
            {"final": {**_node(), "taints": [{"key": verdicts.QUARANTINE_TAINT}]}},
            "quarantine taint",
        ),
        (
            {
                "final": {
                    **_node(),
                    "ownership_annotations": {"gpu-fault.io/incident-id": "i"},
                }
            },
            "gpu-fault annotations remain",
        ),
        (
            {"final_annotations": {verdicts.TICK_ANNOTATION: "77"}},
            "tick annotation remains",
        ),
        ({"final": {**_node(), "boot_id": "boot-2"}}, "identity changed"),
        ({"final_efa": EFA - 1}, "EFA allocatable"),
        ({"final": {**_node(), "ready": "False"}}, "not Ready"),
    ],
)
def test_each_final_node_defect_is_named(
    overrides: dict[str, Any], fragment: str
) -> None:
    errors = _final(**overrides)
    assert any(fragment in error for error in errors), (fragment, errors)


# --------------------------------------------------------------------------- #
# Annotation writer with a fake kubectl
# --------------------------------------------------------------------------- #
class _FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


def _writer(
    runner: Any, *, max_seconds: float = 10.0, clock: _FakeClock | None = None
) -> writer_module.AnnotationWriter:
    clock = clock or _FakeClock()
    return writer_module.AnnotationWriter(
        ["kubectl", "--kubeconfig", "/k", "--context", "c"],
        NODE,
        interval_seconds=0.5,
        max_seconds=max_seconds,
        runner=runner,
        clock=clock,
        sleep=clock.sleep,
    )


def test_the_writer_is_bounded_and_counts_patches_and_conflicts() -> None:
    commands: list[list[str]] = []

    def runner(command: list[str]) -> int:
        commands.append(command)
        return 1 if len(commands) % 5 == 0 else 0

    writer = _writer(runner, max_seconds=10.0)
    writer.run_loop()
    report = writer.report()
    assert report["exceeded_max_seconds"] is True, report
    assert report["stopped"] is True, report
    assert report["patches"] + report["conflicts"] == len(commands), report
    assert report["conflicts"] == len(commands) // 5, report
    assert 15 <= len(commands) <= 21, len(commands)
    body = json.loads(commands[0][-1])
    assert body == {
        "metadata": {"annotations": {writer_module.TICK_ANNOTATION: "1"}}
    }, body
    assert commands[0][:5] == ["kubectl", "--kubeconfig", "/k", "--context", "c"], (
        commands[0]
    )
    assert "--type=merge" in commands[0], commands[0]


def test_clear_nulls_the_tick_and_reports_the_outcome() -> None:
    commands: list[list[str]] = []
    writer = _writer(lambda command: commands.append(command) or 0)
    assert writer.clear() == 0, "clear returns the kubectl exit code"
    body = json.loads(commands[-1][-1])
    assert body == {
        "metadata": {"annotations": {writer_module.TICK_ANNOTATION: None}}
    }, body
    assert writer.report()["cleared"] is True, writer.report()
    failing = _writer(lambda command: 1)
    assert failing.clear() == 1, "a failed clear is reported, not hidden"
    assert failing.report()["cleared"] is False, failing.report()


def test_stop_ends_a_started_writer() -> None:
    import threading

    started = threading.Event()

    def runner(command: list[str]) -> int:
        started.set()
        return 0

    writer = writer_module.AnnotationWriter(
        ["kubectl"], NODE, interval_seconds=0.2, max_seconds=30.0, runner=runner
    )
    writer.start()
    assert started.wait(5.0), "the loop must issue a patch quickly"
    writer.stop(join_timeout=5.0)
    assert writer.running is False, "stop must end the thread"
    assert writer.report()["stopped"] is True, writer.report()
    assert writer.report()["exceeded_max_seconds"] is False, "stopped before the bound"
    with pytest.raises(RuntimeError, match="already started"):
        writer.start()


def test_the_writer_refuses_a_flood_or_an_unbounded_run() -> None:
    with pytest.raises(ValueError, match="API flood"):
        writer_module.AnnotationWriter(["kubectl"], NODE, interval_seconds=0.05)
    with pytest.raises(ValueError, match="max_seconds"):
        writer_module.AnnotationWriter(["kubectl"], NODE, max_seconds=0)
    with pytest.raises(ValueError, match="max_seconds"):
        writer_module.AnnotationWriter(["kubectl"], NODE, max_seconds=99999)
    with pytest.raises(ValueError, match="node name"):
        writer_module.AnnotationWriter(["kubectl"], "")


# --------------------------------------------------------------------------- #
# Runner guard
# --------------------------------------------------------------------------- #
def _preflight_document() -> dict[str, Any]:
    return {
        "release_id": "rel-1",
        "node": _node(),
        "efa_allocatable": EFA,
        "bound_efa_functions": ["0000:59:00.0"],
    }


def test_a_plan_that_drifted_from_its_preflight_is_refused(tmp_path: Path) -> None:
    case_dir = tmp_path / "cases" / destr021.CASE_ID
    case_dir.mkdir(parents=True)
    (case_dir / "plan.json").write_text(
        json.dumps({"details": {"preflight_identity": {"release_id": "rel-0"}}}),
        encoding="utf-8",
    )
    with pytest.raises(Exception, match="plan drifted"):
        destr021.verify_plan_identity(case_dir, _preflight_document())


def test_plan_details_name_the_risk_stops_and_rollback() -> None:
    settings = destr021.Settings(
        regional=None,  # type: ignore[arg-type]
        node=NODE,
        site_file=Path("/tmp/site.yaml"),
        host_probe_image="img",
        predecessor_path=Path("/tmp/p.json"),
        writer_interval_seconds=0.5,
        writer_max_seconds=900.0,
    )
    details = destr021.plan_details(
        settings, {**_preflight_document(), "predecessor": {"valid": True}}
    )
    assert details["risk"] == "live-node-mutation", details["risk"]
    conditions = "\n".join(details["stop_conditions"])
    assert "plugin-restart annotation" in conditions, conditions
    assert "RESTORE_SCHEDULING fails" in conditions, conditions
    assert details["rollback"]["restore_efa_runs_in_finally"] is True, details
    assert details["rollback"]["no_reset_reboot_or_provider_action_is_authorized"], (
        details
    )
    assert "No reset, no reboot, no provider call" in details["mutation"], details


def test_the_runner_is_plan_by_default_and_needs_an_exact_confirmation() -> None:
    parser = destr021.parser()
    plan = parser.parse_args(["--run-dir", "/tmp/run"])
    assert plan.execute is False, "the runner must default to plan mode"
    assert plan.writer_interval_seconds == verdicts.WRITER_INTERVAL_SECONDS, plan
    assert plan.writer_max_seconds == verdicts.WRITER_MAX_SECONDS, plan
    execute = parser.parse_args(
        [
            "--run-dir",
            "/tmp/run",
            "--execute",
            "--confirm",
            destr021.CONFIRMATION,
            "--maintenance-window-end",
            "2026-09-06T12:00:00+00:00",
            "--node",
            NODE,
        ]
    )
    assert execute.execute is True and execute.confirm == destr021.CONFIRMATION, execute
    with pytest.raises(SystemExit):
        parser.parse_args(["--run-dir", "/tmp/run", "--plan", "--execute"])
    assert isinstance(parser, argparse.ArgumentParser), "parser type"


def test_configure_refuses_a_flooding_or_unbounded_writer(tmp_path: Path) -> None:
    cpu = tmp_path / "cpu.kubeconfig"
    gpu = tmp_path / "gpu.kubeconfig"
    cpu.write_text("apiVersion: v1\n", encoding="utf-8")
    gpu.write_text("apiVersion: v1\n", encoding="utf-8")
    common = [
        "--run-dir",
        str(tmp_path),
        "--cpu-kubeconfig",
        str(cpu),
        "--gpu-kubeconfig",
        str(gpu),
        "--gpu-context",
        "ctx",
        "--cluster-id",
        "cluster-a",
        "--region",
        "us-west-2",
        "--node",
        NODE,
        "--site-file",
        str(tmp_path / "site.yaml"),
        "--host-probe-image",
        "img",
    ]
    flood = destr021.parser().parse_args([*common, "--writer-interval-seconds", "0.01"])
    with pytest.raises(Exception, match="API flood"):
        destr021.configure(flood)
    unbounded = destr021.parser().parse_args([*common, "--writer-max-seconds", "7200"])
    with pytest.raises(Exception, match="writer max seconds"):
        destr021.configure(unbounded)
    settings = destr021.configure(destr021.parser().parse_args(common))
    assert settings.node == NODE, settings
    assert settings.environment()["GPU_FAULT_TARGET_NODE"] == NODE, (
        settings.environment()
    )


def test_the_help_text_offers_the_four_documented_live_flags() -> None:
    help_text = destr021.parser().format_help()
    for flag in ("--plan", "--execute", "--confirm", "--maintenance-window-end"):
        assert flag in help_text, flag


def test_the_runner_and_probe_are_executable_and_carry_no_site_topology() -> None:
    executables = (
        "run_destr021_adversarial_node_metadata.py",
        "probes/destr021_annotation_writer.py",
    )
    for name in (*executables, "destr021_verdicts.py"):
        path = ROOT / "scripts/e2e/regional" / name
        source = path.read_text(encoding="utf-8")
        for token in (
            "/secure/gpu-fault-bootstrap",
            "514385905925",
            "gpu-fault-gpu-1-",
        ):
            assert token not in source, (name, token)
        if name not in executables:
            continue
        mode = path.stat().st_mode & 0o777
        assert mode == 0o775, f"{name} is {oct(mode)}, not 0o775"
        assert source.splitlines()[0] == "#!/usr/bin/env python3", name


def test_the_stale_started_at_uses_the_adapter_timeout_plus_margin() -> None:
    value = verdicts.parse_time(verdicts.stale_started_at(T0))
    assert value is not None, "parsable"
    assert T0 - value == timedelta(
        seconds=verdicts.RESTART_TIMEOUT_SECONDS + verdicts.STALE_MARGIN_SECONDS
    ), value

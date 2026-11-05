"""GF-GENERAL-LIVE-019 memory variants: the readers and exits at their edges.

``test_general_live_gpu_memory`` and ``test_general_live_gpu_memory_reset`` run
the variants over the product's records on their main paths. These cases pin
what is left to the record that produces it: a reset chain that planned
RESET_GPU and never ran it, ran it into FAILED or is still running it; a
post-reset proof with no workflow, no validation or a validation without a
reset; an incident the latest read no longer lists; a pending flag that never
minted an incident (nothing settled, nothing exited); a health summary too
short to keep the confirmation reads clear of it and a plan the guard refuses,
both before anything is armed; the volatile DBE's validated restore run for
real, refused and accepted; a contained exit with no incident, and one whose
restore succeeded while the incident stayed QUARANTINED.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import general_live_common as common
from scripts.e2e.regional import general_live_gpu_memory as mem
from scripts.e2e.regional.regional_commands import RegionalFixtureError
from tests.regional._general_live_fakes import store_poll_clock
from tests.regional._general_live_gpu_fakes import (
    audit_records,
    dcgm_probe,
    gpu_cpu_python,
    gpu_incident,
    gpu_workflow,
    survey,
)
from tests.regional.test_general_live_gpu_memory import _ctx as memory_ctx
from tests.regional.test_general_live_gpu_memory import _names, _remap_ctx
from tests.regional.test_general_live_gpu_memory_reset import (
    PENDING_RECORDS,
    _activity,
    _pending_ctx,
    _pending_incident,
    _succeeded,
)

GPU = "GPU-11111111-1111-4111-8111-111111111111"
LABEL = f"row_remap_pending on {GPU}"


def reset_workflow(*executions: dict[str, Any], status: str = "RUNNING") -> dict:
    return gpu_workflow(
        "wf-r",
        "inc-pend",
        ["QUIESCE_GPU_SERVICES", "RESET_GPU", "VALIDATE_GPU"],
        status=status,
        executions=list(executions),
    )


def execution(operation: str, status: str, **extra: Any) -> dict[str, Any]:
    return {
        "operation": operation,
        "status": status,
        "updated_at": "2026-10-24T00:04:00+00:00",
        **extra,
    }


# --------------------------------------------------------------------------- #
# reset_executed_errors
# --------------------------------------------------------------------------- #
def test_no_incident_has_no_reset_to_judge() -> None:
    assert mem.reset_executed_errors(_activity([], []), [], LABEL) == []


def test_a_planned_reset_that_never_ran_is_named() -> None:
    incidents = [_pending_incident()]
    activity = _activity(incidents, [reset_workflow()])

    assert mem.reset_executed_errors(activity, incidents, LABEL) == [
        f"{LABEL} chain planned RESET_GPU but never executed it"
    ]


def test_a_failed_reset_is_named_with_its_error() -> None:
    incidents = [_pending_incident()]
    failed = execution("RESET_GPU", "FAILED", error="nvidia-smi -r: GPU is busy")
    activity = _activity(incidents, [reset_workflow(failed, status="FAILED")])

    assert mem.reset_executed_errors(activity, incidents, LABEL) == [
        f"{LABEL} RESET_GPU ended FAILED: nvidia-smi -r: GPU is busy"
    ]


def test_a_reset_still_running_has_not_succeeded() -> None:
    incidents = [_pending_incident()]
    activity = _activity(incidents, [reset_workflow(execution("RESET_GPU", "RUNNING"))])

    assert mem.reset_executed_errors(activity, incidents, LABEL) == [
        f"{LABEL} RESET_GPU has not succeeded (RUNNING)"
    ]


# --------------------------------------------------------------------------- #
# recovered_errors / incident_state / post_reset_errors
# --------------------------------------------------------------------------- #
def test_recovery_without_an_incident_has_nothing_to_judge() -> None:
    state = {"active_findings": [{"canonical_name": "row_remap_pending"}]}

    assert mem.recovered_errors(_activity([], []), [], state, label=LABEL) == []


def test_incident_state_reads_the_latest_record_or_falls_back_to_its_own() -> None:
    incident = _pending_incident("ACTION_PENDING")
    activity = _activity(
        [
            {"incident_id": "inc-other", "state": "ESCALATED"},
            {"incident_id": "inc-pend", "state": "RECOVERED"},
        ],
        [],
    )

    assert mem.incident_state(activity, incident) == "RECOVERED"
    assert mem.incident_state(_activity([{"incident_id": "x"}], []), incident) == (
        "ACTION_PENDING"
    ), "an incident the read does not list keeps the state it was read in"


def test_post_reset_without_a_workflow_names_the_missing_reset() -> None:
    incidents = [_pending_incident("RECOVERED")]

    errors = mem.post_reset_errors(_activity(incidents, []), incidents, label=LABEL)

    assert errors == [f"{LABEL} RESET_GPU succeeded 0 times (expected exactly one)"]


def test_post_reset_needs_the_validation_after_the_reset() -> None:
    incidents = [_pending_incident("RECOVERED")]
    workflow = reset_workflow(*_succeeded("QUIESCE_GPU_SERVICES", "RESET_GPU"))

    errors = mem.post_reset_errors(
        _activity(incidents, [workflow]), incidents, label=LABEL
    )

    assert errors == [f"{LABEL} chain never executed VALIDATE_GPU after the reset"]


def test_a_validation_without_a_reset_is_judged_on_the_reset_count_only() -> None:
    incidents = [_pending_incident("RECOVERED")]
    workflow = reset_workflow(*_succeeded("VALIDATE_GPU"), status="SUCCEEDED")

    errors = mem.post_reset_errors(
        _activity(incidents, [workflow]), incidents, label=LABEL
    )

    assert errors == [f"{LABEL} RESET_GPU succeeded 0 times (expected exactly one)"]


# --------------------------------------------------------------------------- #
# pending_supported: no incident, and the refusals before the arm
# --------------------------------------------------------------------------- #
def test_a_pending_flag_that_minted_nothing_settles_and_exits_nothing(
    tmp_path: Path,
) -> None:
    ctx = _pending_ctx(tmp_path, "pending_supported", activity=_activity([], []))
    store_poll_clock(ctx)

    out = mem.pending_supported(ctx)

    assert out["verdict"] == "FAIL", out
    assert out["stages"] == {
        "chain": [f"no {LABEL} finding reached an incident"],
        "hardware": [],
        "recovery": [],
    }
    assert out["incident_ids"] == []
    assert not (ctx.case_dir / "settled-timeline.json").exists(), (
        "no incident, so no settle wait"
    )
    calls = [call[0] for call in ctx.probes[f"dcgm@{ctx.settings.node}"].calls]
    assert calls.count("close-run") == 1, "the replay is still closed once"


def test_a_health_summary_too_short_for_the_confirmation_refuses_the_arm(
    tmp_path: Path,
) -> None:
    short = survey()
    short["collector_env"]["GPU_FAULT_DCGM_HEALTH_SUMMARY_SECONDS"] = "60"
    probe = dcgm_probe(short, audit=PENDING_RECORDS)
    ctx = _pending_ctx(
        tmp_path, "pending_supported", activity=_activity([], []), probe=probe
    )

    with pytest.raises(RegionalFixtureError) as refused:
        mem.pending_supported(ctx)

    assert str(refused.value).startswith(
        "pending replay cannot be armed: no summary-clear window: 3 fault reads at "
        "15 s span 45 s and every 60 s summary excludes"
    ), refused.value
    calls = [call[0] for call in probe.calls]
    assert "deploy-proxy" in calls and "arm" not in calls, calls
    assert ctx.cleanup.pending() == ["dcgm-replay"], "the deployed replay is closed"
    assert not (ctx.case_dir / "summary-phase-clearance.json").exists(), (
        "no clearance is recorded for an arm that was refused"
    )


def test_a_plan_the_guard_refuses_is_a_fixture_error_before_any_deploy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checked: list[dict[str, Any]] = []

    def refuse(document: dict[str, Any], **expected: Any) -> None:
        checked.append({"steps": document["steps"], **expected})
        raise ValueError("the flag step serves 2 read(s)")

    monkeypatch.setattr(mem, "require_pending_plan", refuse)
    ctx = _pending_ctx(tmp_path, "pending_supported", activity=_activity([], []))

    with pytest.raises(
        RegionalFixtureError,
        match=r"^pending replay plan rejected: the flag step serves 2 read\(s\)$",
    ):
        mem.pending_supported(ctx)

    assert checked == [
        {
            "steps": [
                {"reads": 3, "ops": []},
                {
                    "reads": 3,
                    "ops": [
                        {
                            "op": "set",
                            "field": "row_remap_pending",
                            "gpus": [GPU],
                            "value": 1,
                        }
                    ],
                },
            ],
            "gpu": GPU,
            "reads": 3,
            "variant": "pending_supported",
        }
    ]
    calls = [call[0] for call in ctx.probes[f"dcgm@{ctx.settings.node}"].calls]
    assert "deploy-proxy" not in calls, calls
    assert not (ctx.case_dir / "pending-plan-shape.json").exists(), (
        "a refused plan leaves no plan-shape evidence"
    )


# --------------------------------------------------------------------------- #
# volatile_dbe: the validated restore, refused and accepted
# --------------------------------------------------------------------------- #
def volatile_ctx(tmp_path: Path) -> Any:
    incident = gpu_incident(
        "inc-dbe",
        f"b-{GPU}-ecc_dbe_volatile_total",
        state="QUARANTINED",
        action="QUARANTINE",
        workflow="wf-q",
    )
    workflow = gpu_workflow(
        "wf-q",
        "inc-dbe",
        ["FREEZE_EVIDENCE", "MARK_UNSCHEDULABLE", "QUARANTINE"],
        status="BLOCKED",
        executions=_succeeded("FREEZE_EVIDENCE", "MARK_UNSCHEDULABLE", "QUARANTINE"),
    )
    return memory_ctx(
        tmp_path,
        "volatile_dbe",
        probe=dcgm_probe(audit=audit_records({f"{GPU}/ecc_dbe_volatile_total": 1.0})),
        cpu=gpu_cpu_python(activity=_activity([incident], [workflow])),
    )


def test_a_refused_validated_restore_is_the_recovery_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse(ctx: Any, incidents: Any, *, reason: str) -> list[dict[str, Any]]:
        raise RegionalFixtureError("validated restore of inc-dbe ended FAILED")

    monkeypatch.setattr(common, "restore_incidents", refuse)

    out = mem.volatile_dbe(volatile_ctx(tmp_path))

    assert out["verdict"] == "FAIL", out
    assert out["stages"]["recovery"] == [
        "validated restore failed: RegionalFixtureError: validated restore of "
        "inc-dbe ended FAILED"
    ]
    assert out["stages"]["chain"] == [], out["stages"]


def test_an_accepted_validated_restore_readmits_the_quarantined_node(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    restored: list[tuple[list[str], str]] = []

    def restore(ctx: Any, incidents: Any, *, reason: str) -> list[dict[str, Any]]:
        restored.append(([item["incident_id"] for item in incidents], reason))
        return [{"incident_id": "inc-dbe", "status": "SUCCEEDED"}]

    monkeypatch.setattr(common, "restore_incidents", restore)

    out = mem.volatile_dbe(volatile_ctx(tmp_path))

    assert out["verdict"] == "PASS", out["errors"]
    assert out["stages"]["recovery"] == []
    assert restored == [
        (["inc-dbe"], "GF-GENERAL-LIVE-019.volatile_dbe acceptance restore")
    ]


# --------------------------------------------------------------------------- #
# remap_failure: the contained exit's own refusals
# --------------------------------------------------------------------------- #
def test_a_remap_failure_without_an_incident_exits_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ctx, product = _remap_ctx(tmp_path, monkeypatch, activity=_activity([], []))

    out = mem.remap_failure(ctx)

    assert out["verdict"] == "FAIL", out
    assert out["stages"]["recovery"] == [
        "contained exit failed: RegionalFixtureError: no incident to exit from"
    ]
    assert out["contained_exit"] == {}, "nothing was settled, so nothing is recorded"
    assert not (ctx.case_dir / "contained-exit.json").exists(), (
        "an exit that never started writes no exit evidence"
    )
    assert _names(product) == [], "no incident was settled, restored or closed"


def test_a_restore_that_leaves_the_incident_quarantined_is_the_recovery_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ctx, product = _remap_ctx(tmp_path, monkeypatch)

    def lagging(script: str, *arguments: str) -> dict[str, Any]:
        answer = product(script, *arguments)
        if product.calls and product.calls[-1][0] == "restore":
            # The restore released the node; the store still holds the
            # incident QUARANTINED when the fresh read comes.
            product.incident = {**product.incident, "state": "QUARANTINED"}
        return answer

    ctx.regional.cpu_python_responder = lagging

    out = mem.remap_failure(ctx)

    assert out["verdict"] == "FAIL", out
    assert out["stages"]["recovery"] == [
        "contained exit failed: RegionalFixtureError: incident inc-remap is "
        "QUARANTINED after the validated restore succeeded, not RECOVERED"
    ]
    record = json.loads((ctx.case_dir / "contained-exit.json").read_text())
    assert (record["exit"], record["state_after"]) == (
        "validated-restore",
        "QUARANTINED",
    ), record
    assert record["node_after"]["unschedulable"] is False, "the node was released"
    assert _names(product).count("restore") == 1, "the restore is never retried"

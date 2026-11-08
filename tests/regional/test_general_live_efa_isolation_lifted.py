"""GF-GENERAL-LIVE-013 ``post_repair_isolation_lifted`` (b265): a reboot whose
isolation was lifted mid-chain is handed over, not climbed.

Nothing runs a node. The product's records are scripted in the shape the live
site writes: the ``REMEDIATE_EFA_DRIVER`` record FAILED at VALIDATE_FABRIC, its
``workflow-reboot-after`` successor ran FREEZE_EVIDENCE, MARK_UNSCHEDULABLE,
QUARANTINE and then -- with the cordon lifted -- a RESTART_NODE the lifecycle
adapter refused before any provider call (``isolation_not_observed =
NOT_ISOLATED``), and the classifier opened a ``workflow-support-after`` at stage
``containment_or_release`` that put the isolation back and filed one ticket. The
tests pin the PASS / FAIL grading, the INCONCLUSIVE reason tokens and the
evidence each is read off (``LIFT_LATE``, ``LIFT_UNDONE``,
``REPLAY_WITHOUT_OBSERVE``, ``NO_RUNG`` and the non-race ``CONTAINMENT_FAILED``,
``RUNG_UNRESOLVED``, ``HOLD_NOT_ARMED``), the harness-overreach FAIL, the R24 L72
records graded ``LIFT_LATE``, the generation gate, the window evidence and the
whole variant over fakes (``tests/regional/_cordon_lift_fakes`` plays the Node).
"""

from __future__ import annotations

import copy
import json
import types
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import general_live_cordon_lift as lift
from scripts.e2e.regional import general_live_efa_escalation as esc
from scripts.e2e.regional import general_live_efa_isolation_lifted as mod
from scripts.e2e.regional.general_live_fixture import VariantInconclusive, VariantNotRun
from scripts.e2e.regional.general_live_registry import coverage_rows, load_registry
from tests.regional import _r24_l72_evidence as l72
from tests.regional._cordon_lift_fakes import (
    FakeClock,
    FakeNodeServer,
    handoff,
    mark,
    quarantine,
)
from tests.regional._general_live_fakes import (
    FakeProbe,
    FakeRegional,
    make_context,
    make_settings,
    store_script,
)

NODE = "hyperpod-test-node-a"
BDF = "0000:30:00.0"
RECORD = "workflow-aaf326cf4be4f315223ef25b"
REBOOT = "workflow-reboot-after-" + RECORD
REBOOT_INCIDENT = "inc-reboot-after-" + RECORD
SUPPORT = "workflow-support-after-" + REBOOT
SUPPORT_INCIDENT = "inc-support-after-" + REBOOT
BASE_BOOT = "boot-before"
QUARANTINE_TAINT = "gpu-fault.io/quarantined"
ANNOTATION_INCIDENT = "gpu-fault.io/incident-id"
CORDON_PROBLEM = "spec.unschedulable is not true"
REFUSE_REASON = f"RESTART_NODE not submitted, isolation not observed: {CORDON_PROBLEM}"
SPEC = {spec.variant: spec for spec in mod.VARIANTS}["post_repair_isolation_lifted"]
OLD_HOLD = {"stopped": "handed_over", "lifts": [{"patch": {}}]}


# --------------------------------------------------------------------------- #
# scripted records
# --------------------------------------------------------------------------- #
def _ex(operation: str, status: str, index: int, details: dict | None = None):
    return {
        "operation": operation,
        "status": status,
        "step_index": index,
        "error": None,
        "updated_at": f"2026-10-24T00:{10 + index:02d}:00+00:00",
        "details": details or {},
    }


def _steps(names: tuple[str, ...]) -> list[dict[str, Any]]:
    return [
        {"operation": name, "parameters": {}, "node_ids": [NODE], "workload_ids": []}
        for name in names
    ]


def _refused_restart(index: int = 3) -> dict[str, Any]:
    return _ex(
        "RESTART_NODE",
        "FAILED",
        index,
        {
            "safety_rejection": True,
            "isolation_not_observed": "NOT_ISOLATED",
            "node_failures": {NODE: [REFUSE_REASON]},
        },
    )


def _baseline(before: bool, rv_before: str, rv_after: str) -> dict[str, Any]:
    return {
        NODE: {
            "before": {"resource_version": rv_before, "unschedulable": before},
            "after": {"resource_version": rv_after, "unschedulable": True},
        }
    }


def _reboot_after(
    *,
    restart: dict[str, Any] | None,
    steps_done: int = 3,
    quarantine_before: bool = True,
):
    done = [
        _ex("FREEZE_EVIDENCE", "SUCCEEDED", 0),
        _ex(
            "MARK_UNSCHEDULABLE",
            "SUCCEEDED",
            1,
            {"node_baselines": _baseline(True, "100", "101")},
        ),
        _ex(
            "QUARANTINE",
            "SUCCEEDED",
            2,
            # Intact: a no-op write. Lifted: QUARANTINE re-cordoned (rv moved).
            {
                "node_baselines": _baseline(True, "101", "101")
                if quarantine_before
                else _baseline(False, "102", "103")
            },
        ),
    ][:steps_done]
    if restart is not None:
        done.append(restart)
    return {
        "request_id": REBOOT,
        "incident_id": REBOOT_INCIDENT,
        "status": "FAILED" if (restart or {}).get("status") == "FAILED" else "RUNNING",
        "official_action": "REBOOT_NODE",
        "created_at": "2026-10-24T00:09:00+00:00",
        "predecessor_workflow_id": RECORD,
        "official_steps": _steps(esc.REBOOT_SUCCESSOR_CHAIN),
        "step_executions": done,
    }


def _observed(version: str) -> dict[str, Any]:
    return {
        NODE: {
            "kubernetes_node": NODE,
            "resource_version": version,
            "unschedulable": True,
            "incident": REBOOT_INCIDENT,
            "fencing_token": "1",
        }
    }


def _submitted(status: str = "WAITING", version: str = "101") -> dict[str, Any]:
    return _ex(
        "RESTART_NODE",
        status,
        3,
        {"submitted_nodes": [NODE], "observed_isolation": _observed(version)},
    )


def _support_after(*, stage: str = "containment_or_release"):
    return {
        "request_id": SUPPORT,
        "incident_id": SUPPORT_INCIDENT,
        "status": "SUCCEEDED",
        "official_action": "ESCALATE_OPERATOR",
        "created_at": "2026-10-24T00:20:00+00:00",
        "predecessor_workflow_id": REBOOT,
        "official_steps": _steps(esc.CONTAINMENT_RELEASE_SUCCESSOR_CHAIN),
        "step_executions": [
            _ex("FREEZE_EVIDENCE", "SUCCEEDED", 0),
            _ex("MARK_UNSCHEDULABLE", "SUCCEEDED", 1),
            _ex("QUARANTINE", "SUCCEEDED", 2),
            _ex("ESCALATE_SUPPORT", "SUCCEEDED", 3),
        ],
    }


def _incident(
    incident_id: str, state: str, reason: str, *, event_id: str | None = None
) -> dict[str, Any]:
    return {
        "incident_id": incident_id,
        "event_id": event_id or incident_id.removeprefix("inc-"),
        "event_type": "NODE_HEALTH",
        "state": state,
        "created_at": "2026-10-24T00:05:00+00:00",
        "workflow_request_id": incident_id,
        "node_ids": [NODE],
        "reasons": [reason],
    }


def _source_workflow():
    return {
        "request_id": RECORD,
        "incident_id": "inc-efa",
        "status": "FAILED",
        "official_action": "REMEDIATE_EFA_DRIVER",
        "created_at": "2026-10-24T00:05:00+00:00",
        "official_steps": _steps(esc.REMEDIATION_CHAIN),
        "step_executions": [
            _ex("REMEDIATE_EFA_DRIVER", "SUCCEEDED", 2),
            _ex("VALIDATE_FABRIC", "FAILED", 5),
        ],
    }


def _doc(
    *,
    reboot_after: dict[str, Any] | None,
    support: dict[str, Any] | None = None,
    extra_workflows: tuple[dict[str, Any], ...] = (),
) -> dict[str, Any]:
    workflows = [_source_workflow()]
    incidents = [
        _incident(
            "inc-efa",
            "QUARANTINED",
            "EFA driver is not bound",
            event_id=f"host-{NODE}-179-efa_inventory_mismatch-node",
        )
    ]
    if reboot_after is not None:
        workflows.append(reboot_after)
        incidents.append(
            _incident(
                REBOOT_INCIDENT,
                "ACTION_PENDING",
                "efa_recovery remediation failed; failed operation or validation: "
                f"VALIDATE_FABRIC; nodes: {NODE}: VALIDATE_FABRIC failed",
            )
        )
    if support is not None:
        workflows.append(support)
        incidents.append(
            _incident(
                SUPPORT_INCIDENT,
                "ESCALATED",
                "containment_or_release remediation failed; failed operation or "
                f"validation: RESTART_NODE; nodes: {NODE}: {REFUSE_REASON}",
            )
        )
    for item in extra_workflows:
        workflows.append(item)
    return {"incidents": incidents, "workflows": list(workflows), "commands": []}


def _node(*, boot: str = BASE_BOOT, reisolated: bool = True) -> dict[str, Any]:
    taints = (
        [{"key": QUARANTINE_TAINT, "value": SUPPORT_INCIDENT, "effect": "NoSchedule"}]
        if reisolated
        else []
    )
    annotations = {ANNOTATION_INCIDENT: SUPPORT_INCIDENT} if reisolated else {}
    return {
        "name": NODE,
        "boot_id": boot,
        "ready": "True",
        "unschedulable": bool(reisolated),
        "taints": taints,
        "ownership_annotations": annotations,
    }


def _settled(*, boot: str = BASE_BOOT) -> dict[str, Any]:
    return {
        "boot_id_now": boot,
        "rebooted": boot != BASE_BOOT,
        "successors": [REBOOT, SUPPORT],
        "restart_node_executions": [],
    }


def _grade(
    doc: dict[str, Any],
    node: dict[str, Any],
    *,
    boot: str = BASE_BOOT,
    hold: dict[str, Any] | None = None,
):
    pair = esc.efa_chain(doc)
    return mod.grade_isolation_lifted(
        observed=doc,
        pair=pair,
        node_now=node,
        settled=_settled(boot=str(node.get("boot_id"))),
        fixture=types.SimpleNamespace(probe_lost=None),
        bdf=BDF,
        expected=16,
        baseline_boot=boot,
        lift=hold if hold is not None else OLD_HOLD,
        oracle=[{"module_loaded": True, "bound_count": 16}],
    )


def _inconclusive(doc: dict[str, Any], node: dict[str, Any], **kwargs: Any) -> str:
    with pytest.raises(VariantInconclusive) as excinfo:
        _grade(doc, node, **kwargs)
    return str(excinfo.value)


def _hold(*steps: Any) -> dict[str, Any]:
    """A real S74 hold record: the lifter over the fake Node."""

    server = FakeNodeServer(
        FakeClock(),
        incident="inc-efa",
        successor=REBOOT_INCIDENT,
        handoff=SUPPORT_INCIDENT,
    )
    server.script.extend(steps)
    lifter = lift.CordonLifter(
        server, monotonic=server.clock.monotonic, wall=server.clock.wall
    )
    plan = lift.LiftPlan(
        node=NODE,
        source_workflow_id=RECORD,
        source_incident_id="inc-efa",
        baseline_boot=BASE_BOOT,
    )
    return lifter.run(plan, window_seconds=600)


# --------------------------------------------------------------------------- #
# registry + coverage
# --------------------------------------------------------------------------- #
def test_variant_is_registered_native_dangerous_in_its_own_module() -> None:
    reg = load_registry()
    spec = reg["GF-GENERAL-LIVE-013.post_repair_isolation_lifted"]
    assert spec.implemented and spec.proof_mode == "native"
    assert spec.risk == "H" and spec.flags == "X"
    assert spec.lanes == ("eks", "hp")
    assert set(spec.approvals) == {"EFA_DRIVER_UNBIND", "NODE_REBOOT"}
    assert "containment_or_release" in spec.expected
    assert "at most two JSON patches" in spec.injection
    assert "re-cordon the node" in spec.cleanup
    assert "post_repair_isolation_lifted" in coverage_rows()["GF-GENERAL-LIVE-013"]
    # general_live_efa_driver still owns exactly the two bind negatives.
    from scripts.e2e.regional import general_live_efa_driver as driver

    assert {
        item.variant
        for item in driver.VARIANTS
        if item.case_id == "GF-GENERAL-LIVE-013"
    } == {"bind_failure", "post_repair_failure"}


def test_every_inconclusive_reason_is_a_named_token() -> None:
    assert set(mod.RACE_REASONS) == {
        "LIFT_LATE",
        "LIFT_UNDONE",
        "REPLAY_WITHOUT_OBSERVE",
        "NO_RUNG",
    }
    assert set(mod.OTHER_REASONS) == {
        "CONTAINMENT_FAILED",
        "RUNG_UNRESOLVED",
        "HOLD_NOT_ARMED",
    }
    assert mod.reason("NO_RUNG", "x") == "NO_RUNG: x"
    with pytest.raises(ValueError, match="LIFT_EARLY"):
        mod.reason("LIFT_EARLY", "x")


# --------------------------------------------------------------------------- #
# generation gate
# --------------------------------------------------------------------------- #
def _gate_ctx(tmp_path: Path, answer: dict[str, Any]):
    def cpu_python(script: str, *arguments: str) -> dict[str, Any]:
        assert script is mod.GENERATION_PROBE
        return answer

    regional = FakeRegional(node_snapshot=_node(), cpu_python=cpu_python)
    return make_context(
        SPEC,
        tmp_path=tmp_path,
        regional=regional,
        probes={},
        settings=make_settings(SPEC, node=NODE),
    )


def test_generation_gate_passes_when_the_classifier_is_present(tmp_path: Path) -> None:
    ctx = _gate_ctx(
        tmp_path,
        {"present": True, "marks": {"containment_stage": "containment_or_release"}},
    )
    assert mod.require_isolation_refusal_generation(ctx)["present"] is True


def test_generation_gate_not_run_without_b265(tmp_path: Path) -> None:
    ctx = _gate_ctx(
        tmp_path, {"present": False, "marks": {"lifecycle_isolation_refusals": False}}
    )
    with pytest.raises(VariantNotRun) as excinfo:
        mod.require_isolation_refusal_generation(ctx)
    assert excinfo.value.reason == "CAPABILITY_ABSENT"
    assert "82f7b77" in excinfo.value.detail


# --------------------------------------------------------------------------- #
# the plan
# --------------------------------------------------------------------------- #
def test_the_plan_names_the_successor_from_the_detection_read() -> None:
    plan = mod.lift_plan(_doc(reboot_after=None), node=NODE, baseline_boot=BASE_BOOT)
    assert plan.source_workflow_id == RECORD
    assert plan.source_incident_id == "inc-efa"
    assert plan.successor_incident == REBOOT_INCIDENT
    assert plan.handoff_incident == SUPPORT_INCIDENT


def test_the_plan_takes_the_incident_s_workflow_when_the_record_is_not_read() -> None:
    doc = _doc(reboot_after=None)
    doc["workflows"] = []
    doc["incidents"][0]["workflow_request_id"] = RECORD
    plan = mod.lift_plan(doc, node=NODE, baseline_boot=BASE_BOOT)
    assert plan.successor_incident == REBOOT_INCIDENT
    doc["incidents"][0]["workflow_request_id"] = None
    with pytest.raises(VariantInconclusive) as excinfo:
        mod.lift_plan(doc, node=NODE, baseline_boot=BASE_BOOT)
    assert str(excinfo.value).startswith("HOLD_NOT_ARMED: "), str(excinfo.value)


# --------------------------------------------------------------------------- #
# grading: PASS / FAIL
# --------------------------------------------------------------------------- #
def test_pass_on_the_containment_or_release_handoff() -> None:
    doc = _doc(
        reboot_after=_reboot_after(restart=_refused_restart()), support=_support_after()
    )
    result = _grade(doc, _node(reisolated=True))
    assert result["verdict"] == "PASS", result["errors"]
    assert result["refused_restart"]["isolation_not_observed"] == "NOT_ISOLATED"
    assert result["stages"]["harness_overreach"] == []


def test_pass_with_a_real_s74_hold_record() -> None:
    doc = _doc(
        reboot_after=_reboot_after(restart=_refused_restart()), support=_support_after()
    )
    hold = _hold(mark, quarantine, handoff)
    result = _grade(doc, _node(reisolated=True), hold=hold)
    assert result["verdict"] == "PASS", result["errors"]
    assert result["cordon_lift"]["lifts"] == 2
    assert result["cordon_lift"]["roles"]["decisive"] is not None


def test_fail_when_the_ladder_climbs_to_replace_node() -> None:
    replace = {
        "request_id": "workflow-replace-after-" + RECORD,
        "incident_id": "inc-replace",
        "status": "FAILED",
        "official_action": "REPLACE_NODE",
        "created_at": "2026-10-24T00:25:00+00:00",
        "official_steps": _steps(("MARK_UNSCHEDULABLE", "QUARANTINE", "REPLACE_NODE")),
        "step_executions": [_ex("REPLACE_NODE", "FAILED", 2)],
    }
    doc = _doc(
        reboot_after=_reboot_after(restart=_refused_restart()),
        support=_support_after(),
        extra_workflows=(replace,),
    )
    result = _grade(doc, _node(reisolated=True))
    assert result["verdict"] == "FAIL"
    assert any(
        "REPLACE_NODE" in item for item in result["stages"]["isolation_refused"]
    ), result["stages"]["isolation_refused"]


def test_fail_when_the_handoff_is_the_wrong_stage() -> None:
    doc = _doc(
        reboot_after=_reboot_after(restart=_refused_restart()), support=_support_after()
    )
    for incident in doc["incidents"]:
        if incident["incident_id"] == SUPPORT_INCIDENT:
            incident["reasons"] = [
                "replacement remediation failed; failed operation or validation: "
                f"REPLACE_NODE; nodes: {NODE}: REPLACE_NODE failed"
            ]
    result = _grade(doc, _node(reisolated=True))
    assert result["verdict"] == "FAIL"
    assert result["stages"]["containment_or_release_handoff"]


def test_fail_when_the_refused_reboot_rebooted_the_node_anyway() -> None:
    doc = _doc(
        reboot_after=_reboot_after(restart=_refused_restart()), support=_support_after()
    )
    result = _grade(doc, _node(boot="boot-after", reisolated=True))
    assert result["verdict"] == "FAIL"
    assert result["stages"]["no_reboot"], result["stages"]


@pytest.mark.parametrize(
    ("breach", "expected"),
    [
        ("taint", "is not a cordon-only lift patch"),
        ("foreign", "was decided under 'inc-someone-else'"),
        ("after-stop", "after the hold stopped"),
        ("third", "3 cordon lifts were applied"),
    ],
)
def test_harness_overreach_fails_the_run_whatever_the_product_did(
    breach: str, expected: str
) -> None:
    hold = _hold(mark, quarantine, handoff)
    writes = hold["writes"]
    if breach == "taint":
        writes[0]["ops"] = [{"op": "remove", "path": "/spec/taints/0"}]
    elif breach == "foreign":
        writes[1]["incident"] = "inc-someone-else"
    elif breach == "after-stop":
        writes[1]["seq"] = hold["stop_seq"] + 1
    else:
        writes.append(copy.deepcopy(writes[1]))
    doc = _doc(
        reboot_after=_reboot_after(restart=_refused_restart()), support=_support_after()
    )
    result = _grade(doc, _node(reisolated=True), hold=hold)
    assert result["verdict"] == "FAIL"
    assert set(result["stages"]) == {"harness_overreach"}
    assert any(expected in item for item in result["stages"]["harness_overreach"]), (
        result["stages"]
    )


def test_a_hold_aimed_at_another_successor_is_overreach() -> None:
    hold = _hold(mark, quarantine, handoff)
    hold["successor_incident"] = "inc-reboot-after-workflow-other"
    doc = _doc(
        reboot_after=_reboot_after(restart=_refused_restart()), support=_support_after()
    )
    result = _grade(doc, _node(reisolated=True), hold=hold)
    assert result["verdict"] == "FAIL"
    assert "not the successor" in result["stages"]["harness_overreach"][0]


# --------------------------------------------------------------------------- #
# grading: INCONCLUSIVE, read off the records
# --------------------------------------------------------------------------- #
def test_lift_late_when_the_node_rebooted_before_the_lift() -> None:
    doc = _doc(reboot_after=_reboot_after(restart=_submitted("SUCCEEDED")))
    text = _inconclusive(doc, _node(boot="boot-after", reisolated=False))
    assert text.startswith("LIFT_LATE: "), text
    assert "rebooted (boot-before -> boot-after)" in text
    assert "reboot-aware cleanup" in text
    assert "isolation read rv 101" in text


def test_lift_late_when_restart_submitted_without_reboot_yet() -> None:
    doc = _doc(reboot_after=_reboot_after(restart=_submitted()))
    text = _inconclusive(doc, _node(reisolated=True))
    assert text.startswith("LIFT_LATE: "), text
    assert "submitted to the provider" in text


def test_lift_late_when_the_restart_failed_after_its_isolation_read() -> None:
    restart = _ex("RESTART_NODE", "FAILED", 3)
    restart["error"] = "HyperPod preflight failed: cluster must be InService"
    text = _inconclusive(_doc(reboot_after=_reboot_after(restart=restart)), _node())
    assert text.startswith("LIFT_LATE: "), text


def test_lift_undone_when_quarantine_re_cordoned_the_provocation() -> None:
    doc = _doc(
        reboot_after=_reboot_after(restart=_submitted(), quarantine_before=False)
    )
    text = _inconclusive(doc, _node(reisolated=True))
    assert text.startswith("LIFT_UNDONE: "), text
    assert "QUARANTINE re-cordoned the node" in text
    assert "unschedulable before False" in text


def test_lift_undone_when_the_read_saw_a_cordon_written_after_the_decisive_lift() -> (
    None
):
    hold = _hold(mark, quarantine, handoff)
    decisive = lift.decisive_lift(hold)
    assert decisive is not None
    later = [
        e for e in hold["events"] if e["seq"] > decisive["seq"] and e["unschedulable"]
    ]
    doc = _doc(
        reboot_after=_reboot_after(
            restart=_submitted(version=later[0]["resource_version"])
        )
    )
    text = _inconclusive(doc, _node(reisolated=True), hold=hold)
    assert text.startswith("LIFT_UNDONE: "), text
    assert "after the decisive lift" in text


def test_replay_without_observe_when_the_first_run_was_held() -> None:
    restart = _ex(
        "RESTART_NODE",
        "WAITING",
        3,
        {"fleet_preflight_blocked": True, "outcome_unknown": True},
    )
    text = _inconclusive(_doc(reboot_after=_reboot_after(restart=restart)), _node())
    assert text.startswith("REPLAY_WITHOUT_OBSERVE: "), text
    assert "fleet_preflight_blocked" in text and "b272" in text


def test_rung_unresolved_when_the_restart_failed_without_the_marker() -> None:
    restart = _ex("RESTART_NODE", "FAILED", 3, {})
    text = _inconclusive(_doc(reboot_after=_reboot_after(restart=restart)), _node())
    assert text.startswith("RUNG_UNRESOLVED: "), text
    assert "status 'FAILED'" in text


def test_containment_failure_is_not_a_lift_race() -> None:
    reboot = _reboot_after(restart=None, steps_done=3)
    reboot["step_executions"][2]["status"] = "FAILED"  # QUARANTINE FAILED
    text = _inconclusive(_doc(reboot_after=reboot), _node(reisolated=True))
    assert text.startswith("CONTAINMENT_FAILED: "), text
    assert "cannot cause it" in text


def test_no_rung_when_no_reboot_after_opened() -> None:
    text = _inconclusive(_doc(reboot_after=None), _node(reisolated=True))
    assert text.startswith("NO_RUNG: "), text
    assert "no reboot-after rung" in text


def test_the_r24_l72_records_grade_lift_late() -> None:
    """R24, hp, attempt 1: QUARANTINE wrote nothing (rv 182827275 before and
    after), RESTART_NODE's isolation read recorded that same rv, and S72's
    polling hold lifted 1.47 s after the mint, after the read: LIFT_LATE."""

    pair = esc.efa_chain(l72.OBSERVED)
    assert pair is not None
    with pytest.raises(VariantInconclusive) as excinfo:
        mod.grade_isolation_lifted(
            observed=l72.OBSERVED,
            pair=pair,
            node_now={"name": l72.NODE, "boot_id": l72.BOOT_AFTER},
            settled={
                "boot_id_now": l72.BOOT_AFTER,
                "rebooted": True,
                "successors": [],
                "restart_node_executions": [],
            },
            fixture=types.SimpleNamespace(probe_lost=None),
            bdf="0000:55:00.0",
            expected=16,
            baseline_boot=l72.BASELINE_BOOT,
            lift=l72.HOLD,
            oracle=[],
        )
    text = str(excinfo.value)
    assert text.startswith("LIFT_LATE: "), text
    assert "isolation read rv 182827275" in text
    assert "QUARANTINE rv 182827275 -> 182827275, unschedulable before True" in text
    assert "hold stopped 'lift_cap' after 5 lift(s)" in text
    assert lift.overreach_errors(l72.HOLD) == [], "S72's record predates the log"


def test_the_window_evidence_of_the_r24_l72_records() -> None:
    hold = dict(l72.HOLD, successor_workflow="workflow-reboot-after-" + "x")
    window = mod.window_evidence(l72.OBSERVED, hold, node=l72.NODE)
    assert window["marks"]["quarantine_succeeded_at"] is None, "not that workflow"
    record = dict(l72.HOLD)
    record["source_workflow"] = "workflow-2a6df2454fcf9e861eab982b"
    window = mod.window_evidence(l72.OBSERVED, record, node=l72.NODE)
    marks = window["marks"]
    assert marks["quarantine_succeeded_at"] == "2026-11-07T17:04:39.948331Z"
    assert marks["restart_node_minted_at"] == "2026-11-07T17:04:39.965885Z"
    assert window["restart_isolation_read"]["resource_version"] == "182827275"
    assert window["quarantine_baseline"]["before"]["resource_version"] == "182827275"
    assert window["deltas_ms"]["decisive_after_mint_ms"] is None, "no v2 lift"


def test_the_window_evidence_measures_the_lift_against_the_claim() -> None:
    hold = _hold(mark, quarantine, handoff)
    decisive = lift.decisive_lift(hold)
    assert decisive is not None
    hold["executor_claims"] = {
        "claims": [{"at": "2026-11-08T03:00:00.300000Z", "command": "remote-a"}],
        "finishes": [
            {
                "at": "2026-11-08T03:00:01.500000123Z",
                "command": "remote-a",
                "status": "FAILED",
            }
        ],
    }
    reboot = _reboot_after(restart=_refused_restart())
    reboot["request_id"] = hold["successor_workflow"]
    reboot["events"] = [
        {
            "at": "2026-11-08T03:00:00.100000Z",
            "code": "STEP_WAITING",
            "operation": "RESTART_NODE",
        },
        {"at": "bad", "code": "STEP_SUCCEEDED", "operation": "QUARANTINE"},
    ]
    window = mod.window_evidence({"workflows": [reboot]}, hold, node=NODE)
    deltas = window["deltas_ms"]
    assert deltas["claim_after_mint_ms"] == 200.0
    assert deltas["finish_after_claim_ms"] == 1200.0
    assert deltas["decisive_after_quarantine_ms"] is None, "an unreadable time"
    assert window["marks"]["decisive_acked_at"] == decisive["acked_at"]
    assert window["marks"]["executor_finished_status"] == "FAILED"
    json.dumps(window)


# --------------------------------------------------------------------------- #
# the oracle helpers directly
# --------------------------------------------------------------------------- #
def test_isolation_refusal_errors_reads_the_marker_and_reason() -> None:
    doc = _doc(
        reboot_after=_reboot_after(restart=_refused_restart()), support=_support_after()
    )
    pair = esc.efa_chain(doc)
    refused = esc.refused_restart_step(doc["workflows"][1])
    assert esc.isolation_refusal_errors(doc, pair, refused) == []
    refused["details"]["submitted_nodes"] = [NODE]
    assert esc.isolation_refusal_errors(doc, pair, refused), (
        "a refusal that submitted a provider action must not read as clean"
    )


def test_containment_handoff_requires_reisolation_and_one_ticket() -> None:
    doc = _doc(
        reboot_after=_reboot_after(restart=_refused_restart()), support=_support_after()
    )
    reboot_after = doc["workflows"][1]
    assert (
        esc.containment_release_handoff_errors(
            doc, reboot_after, _node(reisolated=True)
        )
        == []
    )
    problems = esc.containment_release_handoff_errors(
        doc, reboot_after, _node(reisolated=False)
    )
    assert any("re-isolate" in item for item in problems), problems


def test_the_record_readers_match_a_node_by_its_kubernetes_name() -> None:
    reboot = _reboot_after(restart=None)
    reboot["step_executions"].append(
        _ex(
            "RESTART_NODE",
            "WAITING",
            3,
            {"observed_isolation": {"logical-id": {"kubernetes_node": NODE, "x": 1}}},
        )
    )
    assert mod.isolation_read(reboot, NODE) == {"kubernetes_node": NODE, "x": 1}
    assert mod.isolation_read(reboot, "other") == {}
    assert mod.quarantine_baseline(None, NODE) == {}
    assert mod.observe_passed([], rebooted=False) is False
    assert mod.replay_markers([{"details": {"retryable_adapter_error": True}}]) == [
        "retryable_adapter_error"
    ]


# --------------------------------------------------------------------------- #
# the whole variant over fakes
# --------------------------------------------------------------------------- #
def _efa_probe() -> FakeProbe:
    functions = [
        {"pci_bdf": f"0000:{0x30 + index:02x}:00.0", "driver": "efa", "bound": True}
        for index in range(16)
    ]
    summary = {
        "bound_count": 15,
        "module_loaded": True,
        "boot_id": BASE_BOOT,
        "trap_fired_at": None,
        "bound": False,
        "pci_bdf": BDF,
    }
    return FakeProbe(
        {
            "functions": {"functions": functions, "module_loaded": True},
            "summary": summary,
            "trap-unbind": {
                "pci_bdf": BDF,
                "bound": False,
                "baseline_bound_count": 16,
                "boot_id": BASE_BOOT,
            },
            "release": {"clean": True, "bound": True, "pci_bdf": BDF},
        }
    )


def test_the_variant_passes_end_to_end_over_fakes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    doc = _doc(
        reboot_after=_reboot_after(restart=_refused_restart()), support=_support_after()
    )
    # The expected-count read, then a detection read that finds nothing yet
    # (the wait polls once), then the records.
    empty: dict[str, Any] = {"incidents": [], "workflows": []}
    store = store_script({"activity": [empty, empty, doc]})
    reads: list[str] = []

    def cpu_python(script: str, *arguments: str) -> dict[str, Any]:
        if script is mod.GENERATION_PROBE:
            return {"present": True, "marks": {}}
        reads.append(script)
        return store(script, *arguments)

    regional = FakeRegional(node_snapshot=_node(reisolated=True), cpu_python=cpu_python)
    ctx = make_context(
        SPEC,
        tmp_path=tmp_path,
        regional=regional,
        probes={f"efa-driver@{NODE}": _efa_probe()},
        settings=make_settings(SPEC, node=NODE),
    )
    server = FakeNodeServer(
        FakeClock(),
        incident="inc-efa",
        successor=REBOOT_INCIDENT,
        handoff=SUPPORT_INCIDENT,
    )
    server.script.extend([mark, quarantine, handoff])
    built: list[tuple[Any, ...]] = []

    def from_kubeconfig(*arguments: Any) -> FakeNodeServer:
        built.append(arguments)
        return server

    monkeypatch.setattr(lift.NodeClient, "from_kubeconfig", from_kubeconfig)
    polls: list[float] = []
    monkeypatch.setattr(ctx, "sleep", polls.append)

    result = mod.post_repair_isolation_lifted(ctx)

    assert result["verdict"] == "PASS", result["errors"]
    assert len(built) == 1, "one client for the whole hold"
    assert mod.DETECTION_POLL_SECONDS in polls, "detection polls at its own pace"
    assert result["cordon_lift"]["lifts"] == 2
    assert [item["kind"] for item in result["cordon_lift"]["writes"]] == ["lift"] * 2
    hold = json.loads((ctx.case_dir / f"{lift.HOLD_EVIDENCE}.json").read_text())
    assert hold["schema"] == lift.SCHEMA and hold["stopped"] == "handed_over"
    window = json.loads((ctx.case_dir / f"{mod.WINDOW_EVIDENCE}.json").read_text())
    assert window["successor_workflow"] == REBOOT
    pending = ctx.cleanup.pending()
    assert pending.index(lift.CLIENT_CLEANUP) < pending.index(lift.RESTORE_CLEANUP)
    assert pending[0] == "validated-restore"
    assert server.cordoned, "the hand-off re-isolated the node"


def test_the_variant_lifts_nothing_without_a_finding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = store_script({"activity": {"incidents": [], "workflows": []}})

    def cpu_python(script: str, *arguments: str) -> dict[str, Any]:
        if script is mod.GENERATION_PROBE:
            return {"present": True, "marks": {}}
        return store(script, *arguments)

    ctx = make_context(
        SPEC,
        tmp_path=tmp_path,
        regional=FakeRegional(node_snapshot=_node(), cpu_python=cpu_python),
        probes={f"efa-driver@{NODE}": _efa_probe()},
        settings=make_settings(SPEC, node=NODE),
    )
    server = FakeNodeServer(FakeClock(), incident="inc-efa")
    monkeypatch.setattr(lift.NodeClient, "from_kubeconfig", lambda *a: server)
    monkeypatch.setattr(mod, "DETECTION_BUDGET", 30)
    with pytest.raises(VariantInconclusive) as excinfo:
        mod.post_repair_isolation_lifted(ctx)
    assert "never minted an efa_inventory_mismatch finding" in str(excinfo.value)
    assert server.patches == [] and server.reads == 0, "the hold was never armed"
    assert lift.RESTORE_CLEANUP not in ctx.cleanup.pending()
    assert lift.CLIENT_CLEANUP in ctx.cleanup.pending()


def test_the_record_readers_skip_entries_without_the_node_or_the_field() -> None:
    reboot = _reboot_after(restart=None)
    reboot["step_executions"].append(_ex("QUARANTINE", "SUCCEEDED", 2, {}))
    assert mod.quarantine_baseline(reboot, NODE)["before"]["resource_version"] == "101"
    hold = _hold(mark, handoff)
    hold["executor_claims"] = {
        "claims": [{"command": "remote-a"}, {"at": "2026-11-08T03:00:00Z"}]
    }
    window = mod.window_evidence({"workflows": []}, hold, node=NODE)
    assert window["marks"]["executor_claimed_at"] == "2026-11-08T03:00:00Z"

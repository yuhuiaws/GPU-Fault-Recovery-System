"""GF-GENERAL-LIVE-013 ``post_repair_isolation_lifted`` (b265): a reboot whose
isolation was lifted mid-chain is handed over, not climbed.

Nothing runs a node. The product's records are scripted in the shape the live
site writes: the ``REMEDIATE_EFA_DRIVER`` record FAILED at VALIDATE_FABRIC, its
``workflow-reboot-after`` successor ran FREEZE_EVIDENCE, MARK_UNSCHEDULABLE,
QUARANTINE and then -- with the cordon lifted -- a RESTART_NODE the lifecycle
adapter refused before any provider call (``isolation_not_observed =
NOT_ISOLATED``), and the classifier opened a ``workflow-support-after`` at stage
``containment_or_release`` that put the isolation back and filed one ticket. The
tests pin the cordon-lift trigger (fires only after QUARANTINE succeeded and
before RESTART_NODE submits), that the lift is one-shot and touches only the
cordon, the PASS / FAIL / both INCONCLUSIVE races, the generation gate and the
reboot-aware cleanup of a lost-late race.
"""

from __future__ import annotations

import types
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import general_live_efa_escalation as esc
from scripts.e2e.regional import general_live_efa_isolation_lifted as mod
from scripts.e2e.regional.general_live_fixture import VariantInconclusive, VariantNotRun
from scripts.e2e.regional.general_live_registry import coverage_rows, load_registry
from tests.regional._general_live_fakes import FakeRegional, make_context, make_settings

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


def _reboot_after(*, restart: dict[str, Any] | None, steps_done: int = 3):
    done = [
        _ex("FREEZE_EVIDENCE", "SUCCEEDED", 0),
        _ex("MARK_UNSCHEDULABLE", "SUCCEEDED", 1),
        _ex("QUARANTINE", "SUCCEEDED", 2),
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
    return {"incidents": incidents, "workflows": list(workflows) + [], "commands": []}


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


def _grade(doc: dict[str, Any], node: dict[str, Any], *, boot: str = BASE_BOOT):
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
        lift={"stopped": "handed_over", "lifts": [{"patch": {}}]},
        oracle=[{"module_loaded": True, "bound_count": 16}],
    )


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
    assert "post_repair_isolation_lifted" in coverage_rows()["GF-GENERAL-LIVE-013"]
    # general_live_efa_driver still owns exactly the two bind negatives.
    from scripts.e2e.regional import general_live_efa_driver as driver

    assert {
        item.variant
        for item in driver.VARIANTS
        if item.case_id == "GF-GENERAL-LIVE-013"
    } == {"bind_failure", "post_repair_failure"}


# --------------------------------------------------------------------------- #
# lift_cordon: cordon only
# --------------------------------------------------------------------------- #
def _snap(incident: str | None, unschedulable: bool, *, boot: str = BASE_BOOT):
    taints = (
        [{"key": QUARANTINE_TAINT, "value": incident, "effect": "NoSchedule"}]
        if incident
        else []
    )
    return {
        "name": NODE,
        "boot_id": boot,
        "ready": "True",
        "unschedulable": unschedulable,
        "taints": taints,
        "ownership_annotations": {ANNOTATION_INCIDENT: incident} if incident else {},
    }


class _SeqRegional(FakeRegional):
    """A FakeRegional whose node_snapshot walks a scripted sequence (the last
    reading repeats), recording every kubectl patch."""

    def __init__(self, snapshots, **kwargs):
        super().__init__(node_snapshot=snapshots[0], **kwargs)
        self._snaps = list(snapshots)
        self._i = 0

    def node_snapshot(self, node: str):
        snap = self._snaps[min(self._i, len(self._snaps) - 1)]
        self._i += 1
        return dict(snap)


def _ctx_for(regional, tmp_path: Path):
    return make_context(
        SPEC,
        tmp_path=tmp_path,
        regional=regional,
        probes={},
        settings=make_settings(SPEC, node=NODE),
    )


def test_lift_patches_only_the_cordon(tmp_path: Path) -> None:
    regional = FakeRegional(node_snapshot=_node(reisolated=True))
    ctx = _ctx_for(regional, tmp_path)
    record: dict[str, Any] = {}
    mod.lift_cordon(ctx, record)
    assert len(record["lifts"]) == 1, record
    assert len(regional.kubectl_calls) == 1
    call = regional.kubectl_calls[0]
    assert call[0] == "gpu" and call[1] == "patch" and call[2] == "node"
    assert call[3] == NODE
    # A merge patch naming only spec.unschedulable=false: it does not mention
    # taints, metadata, annotations or fencing.
    payload = call[call.index("-p") + 1]
    assert '"unschedulable": false' in payload
    assert "taints" not in payload and "metadata" not in payload
    assert "annotations" not in payload and "fencing" not in payload
    assert record["lifts"][0]["patch"] == {"spec": {"unschedulable": False}}


# --------------------------------------------------------------------------- #
# the cordon-hold trigger (fast node reads)
# --------------------------------------------------------------------------- #
SOURCE_INC = "inc-efa"


def test_hold_lifts_on_the_successor_and_stops_when_handed_over(tmp_path: Path) -> None:
    # source-owned (no lift) -> reboot-after cordoned (lift) -> QUARANTINE
    # re-cordoned my lift (lift again) -> support-after re-isolated (stop).
    regional = _SeqRegional(
        [
            _snap(SOURCE_INC, True),
            _snap(REBOOT_INCIDENT, True),
            _snap(REBOOT_INCIDENT, True),
            _snap(SUPPORT_INCIDENT, True),
        ]
    )
    ctx = _ctx_for(regional, tmp_path)
    record = mod.hold_cordon_lifted(
        ctx, source_incident_id=SOURCE_INC, baseline_boot=BASE_BOOT, window=600
    )
    assert record["stopped"] == "handed_over", record
    assert record["successor_incident"] == REBOOT_INCIDENT
    assert len(record["lifts"]) == 2, record
    # Never lifted while the source incident still owned the node.
    assert len(regional.kubectl_calls) == 2


def test_hold_stops_on_a_reboot(tmp_path: Path) -> None:
    regional = _SeqRegional(
        [_snap(REBOOT_INCIDENT, True), _snap(REBOOT_INCIDENT, True, boot="boot-after")]
    )
    ctx = _ctx_for(regional, tmp_path)
    record = mod.hold_cordon_lifted(
        ctx, source_incident_id=SOURCE_INC, baseline_boot=BASE_BOOT, window=600
    )
    assert record["stopped"] == "rebooted", record
    assert len(record["lifts"]) == 1


def test_hold_is_bounded_by_the_lift_cap(tmp_path: Path) -> None:
    # A node the successor keeps re-cordoning forever is bounded to LIFT_CAP.
    regional = _SeqRegional([_snap(REBOOT_INCIDENT, True)])
    ctx = _ctx_for(regional, tmp_path)
    record = mod.hold_cordon_lifted(
        ctx, source_incident_id=SOURCE_INC, baseline_boot=BASE_BOOT, window=600
    )
    assert record["stopped"] == "lift_cap"
    assert len(record["lifts"]) == mod.LIFT_CAP


def test_hold_never_lifts_while_the_source_incident_owns_the_node(
    tmp_path: Path,
) -> None:
    regional = _SeqRegional([_snap(SOURCE_INC, True)])
    ctx = _ctx_for(regional, tmp_path)
    record = mod.hold_cordon_lifted(
        ctx, source_incident_id=SOURCE_INC, baseline_boot=BASE_BOOT, window=2
    )
    assert record["stopped"] == "window"
    assert record["lifts"] == []
    assert regional.kubectl_calls == []


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
# grading: PASS / FAIL / INCONCLUSIVE races
# --------------------------------------------------------------------------- #
def test_pass_on_the_containment_or_release_handoff() -> None:
    doc = _doc(
        reboot_after=_reboot_after(restart=_refused_restart()), support=_support_after()
    )
    result = _grade(doc, _node(reisolated=True))
    assert result["verdict"] == "PASS", result["errors"]
    assert result["refused_restart"]["isolation_not_observed"] == "NOT_ISOLATED"


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
    # Rewrite the support incident's reason to a different stage.
    for incident in doc["incidents"]:
        if incident["incident_id"] == SUPPORT_INCIDENT:
            incident["reasons"] = [
                "replacement remediation failed; failed operation or validation: "
                f"REPLACE_NODE; nodes: {NODE}: REPLACE_NODE failed"
            ]
    result = _grade(doc, _node(reisolated=True))
    assert result["verdict"] == "FAIL"
    assert result["stages"]["containment_or_release_handoff"]


def test_inconclusive_when_the_node_rebooted_before_the_lift() -> None:
    # The late race: RESTART_NODE ran, the node came back on a new boot.
    doc = _doc(
        reboot_after=_reboot_after(
            restart=_ex("RESTART_NODE", "SUCCEEDED", 3, {"submitted_nodes": [NODE]})
        )
    )
    with pytest.raises(VariantInconclusive) as excinfo:
        _grade(doc, _node(boot="boot-after", reisolated=False), boot=BASE_BOOT)
    assert "rebooted" in str(excinfo.value)


def test_inconclusive_when_quarantine_failed_before_the_lift() -> None:
    # The early race: the lift landed before QUARANTINE succeeded.
    reboot = _reboot_after(restart=None, steps_done=3)
    reboot["step_executions"][2]["status"] = "FAILED"  # QUARANTINE FAILED
    doc = _doc(reboot_after=reboot)
    with pytest.raises(VariantInconclusive) as excinfo:
        _grade(doc, _node(reisolated=True))
    assert "QUARANTINE failed its own verification" in str(excinfo.value)


def test_inconclusive_when_no_reboot_rung_opened() -> None:
    doc = _doc(reboot_after=None)
    with pytest.raises(VariantInconclusive) as excinfo:
        _grade(doc, _node(reisolated=True))
    assert "no reboot-after rung" in str(excinfo.value)


def test_inconclusive_when_restart_submitted_without_reboot_yet() -> None:
    doc = _doc(
        reboot_after=_reboot_after(
            restart=_ex("RESTART_NODE", "WAITING", 3, {"submitted_nodes": [NODE]})
        )
    )
    with pytest.raises(VariantInconclusive) as excinfo:
        _grade(doc, _node(reisolated=True), boot=BASE_BOOT)
    assert "submitted to the provider" in str(excinfo.value)


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
    # a refusal that submitted a provider action is not a clean refusal
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
    # a node the hand-off did not re-isolate fails the check
    problems = esc.containment_release_handoff_errors(
        doc, reboot_after, _node(reisolated=False)
    )
    assert any("re-isolate" in item for item in problems), problems


def test_cleanup_after_lost_late_race_is_the_reboot_aware_release() -> None:
    # The late race leaves cleanup to EfaDriverFixture.release, judged by the
    # new boot id and the product's post-boot inventory -- not this variant's
    # grading. Here we assert the variant reports the reboot (INCONCLUSIVE) so
    # the runner's teardown runs that reboot-aware release.
    doc = _doc(
        reboot_after=_reboot_after(
            restart=_ex("RESTART_NODE", "SUCCEEDED", 3, {"submitted_nodes": [NODE]})
        )
    )
    with pytest.raises(VariantInconclusive) as excinfo:
        _grade(doc, _node(boot="boot-after", reisolated=False), boot=BASE_BOOT)
    assert "reboot-aware cleanup" in str(excinfo.value)

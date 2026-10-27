"""GF-GENERAL-LIVE-021 after the product's reboot rung (S48).

Live 2026-10-27 05:04-05:26Z (HyperPod, the hp lane's exclusive node, R13g,
harness int/r13c-live23) ``recovery_failure`` ended FAIL "a fabric that stayed
failed after the reset did not escalate/hold isolation": the held FAILURE
failed the post-reset VALIDATE_GPU, the ladder's RESTART_NODE rung rebooted
the node (boot id aa20cf7c -> b7e017d9, Ready 05:19:58Z), the ``/run`` replay
went with the boot, the post-reboot validations graded the real fabric and the
incident RECOVERED -- the product's designed path -- while the oracle asked the
settled record for ESCALATED / QUARANTINED, the ladder's end state that only a
fabric still failed AFTER the reboot produces. These tests drive the variant
over fake product timelines with and without the reboot and pin the new
reading: the escalation is graded on the record before and at the rung, the
exit on the settled record, and a reset-only variant whose record carries the
rung is INCONCLUSIVE, not PASS and not FAIL.

The advance the tool refuses (S52)
----------------------------------
Live 2026-10-27 13:14-13:28Z (plain EKS ip-10-0-43-5, R13g, harness
int/r13c-live27) ``not_started`` ended INCONCLUSIVE "the node did not answer
the probe in 13 attempts over 187 s" while the node answered every attempt:
the probe's loopback ``POST /advance`` got HTTP 409 Conflict from the replay
tool. The tool's journal (the case's ``oracle.jsonl``) shows why -- the
collector scraped through the quiesce (read 27: upstream ``Connection
refused`` while the exporter was down) and its first scrape after the exporter
returned read the real ``fabric_manager_status`` 2.0 (Fabric Manager
re-training after the reset) against the baseline 3.0, so the tool aborted
("a real reading is never masked"), left its steps for good and passed the
real readings through; ``Replay.advance`` then raises "the replay is not on a
step" and ``do_POST`` answers 409. HyperPod's collector pauses with the
quiesce (27 reads in 589 s, no upstream error) and its first post-restore
scrape read 3.0, so the advance landed there. These tests pin the runner's
reading of the refusal: a probe error that is not a transport failure makes
the runner read the tool's ``/status`` and classify it (``plan_release``) --
an abort or a lapsed hold is the plan's release (the product grades the real
fabric; PASS with a note), a plan already past the hold is an advance whose
receipt was lost, a disarmed or unarmed tool is INCONCLUSIVE with the 409's
meaning, a plan still on its held step is retried -- while transport errors
never consult the tool; ``recovery_failure`` released this way is INCONCLUSIVE,
not FAIL on the product; and the held step's read bound outlasts the
detection and the chain (the live 36-read bound lapsed after 585 s of scrapes,
reset-done came at 569 s).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import general_live_gpu_fabric as fab
from scripts.e2e.regional import general_live_gpu_replay as replay
from scripts.e2e.regional.general_live_fixture import VariantInconclusive
from scripts.e2e.regional.host_probe_fixture import (
    HostProbeError,
    HostProbeTransportError,
)
from tests.regional._general_live_fakes import FakeRegional, make_context, make_settings
from tests.regional._general_live_gpu_fakes import (
    audit_records,
    dcgm_probe,
    gpu_cpu_python,
    gpu_incident,
    gpu_workflow,
)
from tools import gpu_dcgm_field_replay as tool

SPECS = {spec.variant: spec for spec in fab.VARIANTS}
GPUS = [
    "GPU-00000000-0000-4000-8000-000000000000",
    "GPU-11111111-1111-4111-8111-111111111111",
]
#: The product's catalog chain for the fabric finding and the rung's tail the
#: in-place escalation appends (``branch_escalation.ESCALATION_TAIL``).
CHAIN = [
    "FREEZE_EVIDENCE",
    "MARK_UNSCHEDULABLE",
    "QUIESCE_GPU_SERVICES",
    "VERIFY_NO_GPU_CLIENTS",
    "RESET_ALL_GPUS_NVSWITCHES",
    "RESTORE_GPU_SERVICES",
    "VALIDATE_GPU",
    "VALIDATE_FABRIC",
    "RESTORE_SCHEDULING",
]
TAIL = [
    "RESTART_NODE",
    "VALIDATE_GPU",
    "VALIDATE_HOST",
    "VALIDATE_FABRIC",
    "RESTORE_SCHEDULING",
]
NOT_TRAINED = "nvlink_fabric_not_trained(FAILURE: 8/8 GPUs)"


def _at(clock: str) -> str:
    return f"2026-10-27T{clock}+00:00"


def _exec(
    index: int,
    operation: str,
    status: str,
    at: str,
    *,
    started: str | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    return {
        "step_index": index,
        "operation": operation,
        "status": status,
        "started_at": _at(started or at),
        "updated_at": _at(at),
        "error": error,
        "details": {},
    }


#: The executions up to the reset's completion (live: reset done by 05:16).
THROUGH_RESET = [
    _exec(0, "FREEZE_EVIDENCE", "SUCCEEDED", "05:12:40"),
    _exec(1, "MARK_UNSCHEDULABLE", "SUCCEEDED", "05:12:45"),
    _exec(2, "QUIESCE_GPU_SERVICES", "SUCCEEDED", "05:13:30"),
    _exec(3, "VERIFY_NO_GPU_CLIENTS", "SUCCEEDED", "05:13:35"),
    _exec(4, "RESET_ALL_GPUS_NVSWITCHES", "SUCCEEDED", "05:16:00"),
]
#: The held FAILURE read by the post-reset validation and the rung taken.
FAILED_VALIDATION = [
    _exec(5, "RESTORE_GPU_SERVICES", "SUCCEEDED", "05:16:20"),
    _exec(6, "VALIDATE_GPU", "FAILED", "05:16:40", error=NOT_TRAINED),
]
RUNG_WAITING = _exec(9, "RESTART_NODE", "WAITING", "05:17:10")
#: The reboot's tail: the real fabric validated, the node released.
AFTER_REBOOT = [
    _exec(9, "RESTART_NODE", "SUCCEEDED", "05:20:10", started="05:17:10"),
    _exec(10, "VALIDATE_GPU", "SUCCEEDED", "05:23:00"),
    _exec(11, "VALIDATE_HOST", "SUCCEEDED", "05:23:10"),
    _exec(12, "VALIDATE_FABRIC", "SUCCEEDED", "05:23:20"),
    _exec(13, "RESTORE_SCHEDULING", "SUCCEEDED", "05:25:00"),
]


def _doc(
    executions: list[dict[str, Any]],
    *,
    state: str = "ACTION_PENDING",
    status: str = "RUNNING",
    operations: list[str] | None = None,
) -> dict[str, Any]:
    incident = gpu_incident(
        "inc",
        "b-node-fabric_manager_status",
        action="RESET_GPU",
        workflow="wf",
        state=state,
    )
    workflow = gpu_workflow(
        "wf", "inc", operations or CHAIN, status=status, executions=executions
    )
    return {
        "incidents": [incident],
        "workflows": [workflow],
        "batches": [],
        "markers": [],
        "commands": [],
        "notifications": [],
        "metric_latest": [],
        "collector_statuses": [],
        "agent": None,
    }


DETECTED = _doc(THROUGH_RESET[:2])
RESET_DONE = _doc(THROUGH_RESET)
RUNG_TAKEN = _doc(
    [*THROUGH_RESET, *FAILED_VALIDATION, RUNG_WAITING], operations=CHAIN + TAIL
)
REBOOTED_RECOVERED = _doc(
    [*THROUGH_RESET, *FAILED_VALIDATION, *AFTER_REBOOT],
    state="RECOVERED",
    status="SUCCEEDED",
    operations=CHAIN + TAIL,
)
#: A clean reset: the post-reset validations passed, the node was released.
TRAINED = _doc(
    [
        *THROUGH_RESET,
        _exec(5, "RESTORE_GPU_SERVICES", "SUCCEEDED", "05:16:20"),
        _exec(6, "VALIDATE_GPU", "SUCCEEDED", "05:17:40"),
        _exec(7, "VALIDATE_FABRIC", "SUCCEEDED", "05:17:50"),
        _exec(8, "RESTORE_SCHEDULING", "SUCCEEDED", "05:18:00"),
    ],
    state="RECOVERED",
    status="SUCCEEDED",
)
#: The tool's ``/status`` on the plain-EKS node once it had aborted (live
#: 2026-10-27 13:25Z, journal read 27), as ``general_live_dcgm_probe``'s
#: ``status`` relays it.
ABORTED_STATUS = {
    "reads": 27,
    "profile": "fabric",
    "proof_scope": "live-replay",
    "armed": True,
    "disarmed": False,
    "aborted": (
        "real upstream change of GPU-5d4a1121-2b96-2054-1968-e74914e7bc20/"
        "fabric_manager_status: 3.0 -> 2.0; a real reading is never masked"
    ),
    "step": None,
    "offsets": {},
}
#: The runner's error for the probe exit the 409 produced (13 times live).
REFUSED = (
    "host probe failed (exit 1; error ProbeError 4a771a58aed3bc76 at loopback:430; "
    "detail in probe-error.json); output withheld"
)


class Product:
    """The activity probe's answers, one document per read (the last repeats);
    ``reboot_at`` flips the Node object's boot id when that read is served."""

    def __init__(
        self,
        docs: list[dict[str, Any]],
        *,
        regional: FakeRegional,
        reboot_at: int | None,
    ) -> None:
        self.docs = docs
        self.regional = regional
        self.reboot_at = reboot_at
        self.reads = 0

    def __call__(self, arguments: tuple[str, ...]) -> dict[str, Any]:
        index = min(self.reads, len(self.docs) - 1)
        if self.reboot_at is not None and self.reads >= self.reboot_at:
            self.regional.node_snapshot_value["boot_id"] = "boot-2"
        self.reads += 1
        return dict(self.docs[index])


def _run(
    tmp_path: Path,
    variant: str,
    docs: list[dict[str, Any]],
    *,
    reboot_at: int | None = None,
    advance: Any = None,
    extra: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], Any, Any]:
    spec = SPECS[variant]
    regional = FakeRegional()
    product = Product(docs, regional=regional, reboot_at=reboot_at)
    regional.cpu_python_responder = gpu_cpu_python(
        activity=product, gpu_state={"active_findings": []}
    )
    settings = make_settings(spec, node=regional.node_snapshot("")["name"])
    object.__setattr__(settings, "approvals", spec.approvals)
    ctx = make_context(
        spec, tmp_path=tmp_path, regional=regional, probes={}, settings=settings
    )
    scripted = dict(extra or {})
    if advance is not None:
        scripted["advance"] = advance
    probe = dcgm_probe(
        audit=audit_records({f"{gpu}/fabric_manager_status": 4.0 for gpu in GPUS}),
        extra=scripted or None,
    )
    ctx.probes[f"dcgm@{ctx.settings.node}"] = probe
    out = getattr(fab, variant)(ctx)
    return out, ctx, probe


def _evidence(ctx: Any, name: str) -> dict[str, Any]:
    return json.loads((ctx.case_dir / f"{name}.json").read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# recovery_failure: the rung is the escalation
# --------------------------------------------------------------------------- #
def test_recovery_failure_passes_on_the_live_reboot_shape(tmp_path: Path) -> None:
    """The 2026-10-27 05:04Z record: held FAILURE -> VALIDATE_GPU FAILED ->
    RESTART_NODE -> reboot -> real fabric validated -> RECOVERED is a PASS."""

    out, ctx, probe = _run(
        tmp_path,
        "recovery_failure",
        [DETECTED, RESET_DONE, RUNG_TAKEN, REBOOTED_RECOVERED],
        reboot_at=3,
    )
    assert out["verdict"] == "PASS", out["errors"]
    assert out["stages"]["recovery_failure"] == []
    assert out["stages"]["escalation_exit"] == []
    assert out["stages"]["chain"] == []
    (note,) = out["notes"]
    assert "RESTART_NODE rung is the escalation" in note
    assert "boot id boot-1 -> boot-2" in note and "/run" in note
    assert "not observable" in note
    record = _evidence(ctx, "fabric-escalation")
    assert record["rebooted"] is True
    assert record["incident_state"] == "RECOVERED"
    assert record["workflow_status"] == "SUCCEEDED"
    assert record["node_after"] == {
        "ownership_annotations": {},
        "taints": [],
        "unschedulable": False,
    }
    failed = [e for e in record["executions"] if e["status"] == "FAILED"]
    assert failed == [
        {
            "step_index": 6,
            "operation": "VALIDATE_GPU",
            "status": "FAILED",
            "started_at": _at("05:16:40"),
            "updated_at": _at("05:16:40"),
            "error": NOT_TRAINED,
        }
    ]
    assert [r["operation"] for r in record["rungs"]] == ["RESTART_NODE"]
    names = [call[0] for call in probe.calls]
    assert names.index("advance") < names.index("close-run"), names
    assert out["plan_advance"]["attempts"] == 1, "the advance landed first time"
    for timeline in ("detection", "reset-done", "escalation", "settled"):
        assert (ctx.case_dir / f"{timeline}-timeline.json").is_file(), timeline


def test_recovery_failure_passes_when_the_site_hands_the_node_to_an_operator(
    tmp_path: Path,
) -> None:
    """No reboot on this site: the failed validation ends the record FAILED and
    the incident is QUARANTINED -- isolation held, escalation by hand-off."""

    handed_off = _doc(
        [*THROUGH_RESET, *FAILED_VALIDATION], state="QUARANTINED", status="FAILED"
    )
    out, ctx, _ = _run(tmp_path, "recovery_failure", [DETECTED, RESET_DONE, handed_off])
    assert out["verdict"] == "PASS", out["errors"]
    (note,) = out["notes"]
    assert "held the node for an operator (QUARANTINED)" in note
    record = _evidence(ctx, "fabric-escalation")
    assert record["rebooted"] is False and record["rungs"] == []


def test_recovery_failure_fails_when_the_failed_fabric_is_restored(
    tmp_path: Path,
) -> None:
    """The product validated a fabric the replay kept failed and released the
    node: the original defect the variant exists for."""

    restored = _doc(
        [
            *THROUGH_RESET,
            _exec(5, "RESTORE_GPU_SERVICES", "SUCCEEDED", "05:16:20"),
            _exec(6, "VALIDATE_GPU", "SUCCEEDED", "05:16:40"),
            _exec(7, "VALIDATE_FABRIC", "SUCCEEDED", "05:16:50"),
            _exec(8, "RESTORE_SCHEDULING", "SUCCEEDED", "05:17:00"),
        ],
        state="RECOVERED",
        status="SUCCEEDED",
    )
    out, _, _ = _run(tmp_path, "recovery_failure", [DETECTED, RESET_DONE, restored])
    assert out["verdict"] == "FAIL"
    errors = out["stages"]["recovery_failure"]
    for wanted in (
        "did not fail on the fabric the replay kept failed",
        "restored while the fabric still read failed",
        "no rung followed",
        "did not escalate/hold isolation",
    ):
        assert any(wanted in e for e in errors), (wanted, errors)
    assert out["stages"]["escalation_exit"] == [], "the exit itself is clean"
    assert out["notes"] == []


def test_recovery_failure_fails_when_the_node_is_released_before_the_rung(
    tmp_path: Path,
) -> None:
    released_then_rebooted = _doc(
        [
            *THROUGH_RESET,
            *FAILED_VALIDATION,
            _exec(8, "RESTORE_SCHEDULING", "SUCCEEDED", "05:16:50"),
            *AFTER_REBOOT,
        ],
        state="RECOVERED",
        status="SUCCEEDED",
        operations=CHAIN + TAIL,
    )
    out, _, _ = _run(
        tmp_path,
        "recovery_failure",
        [DETECTED, RESET_DONE, released_then_rebooted],
        reboot_at=2,
    )
    assert out["verdict"] == "FAIL"
    (error,) = out["stages"]["recovery_failure"]
    assert "RESTORE_SCHEDULING SUCCEEDED at " + _at("05:16:50") in error
    assert "before the RESTART_NODE rung at " + _at("05:17:10") in error


def test_recovery_failure_fails_on_a_blind_loop(tmp_path: Path) -> None:
    """A second reboot rung and a REPLACE_NODE after it: the reboot did not end
    the failed validation (a replay that survived the boot, or a real fault)."""

    looped = _doc(
        [
            *THROUGH_RESET,
            *FAILED_VALIDATION,
            _exec(9, "RESTART_NODE", "SUCCEEDED", "05:20:10", started="05:17:10"),
            _exec(10, "VALIDATE_GPU", "FAILED", "05:23:00", error=NOT_TRAINED),
            _exec(14, "RESTART_NODE", "SUCCEEDED", "05:27:00"),
            _exec(15, "VALIDATE_GPU", "FAILED", "05:30:00", error=NOT_TRAINED),
            _exec(19, "REPLACE_NODE", "FAILED", "05:31:00"),
        ],
        state="QUARANTINED",
        status="FAILED",
        operations=CHAIN + TAIL + TAIL + ["REPLACE_NODE"],
    )
    out, _, _ = _run(
        tmp_path, "recovery_failure", [DETECTED, RESET_DONE, looped], reboot_at=2
    )
    assert out["verdict"] == "FAIL"
    assert out["stages"]["recovery_failure"] == [], "the first rung was right"
    errors = out["stages"]["escalation_exit"]
    assert any("2 RESTART_NODE rungs" in e for e in errors), errors
    assert any("went on to REPLACE_NODE (FAILED)" in e for e in errors), errors


def test_recovery_failure_fails_when_the_escalation_never_settles(
    tmp_path: Path,
) -> None:
    out, _, _ = _run(
        tmp_path, "recovery_failure", [DETECTED, RESET_DONE, RUNG_TAKEN], reboot_at=3
    )
    assert out["verdict"] == "FAIL"
    assert out["stages"]["recovery_failure"] == []
    (error,) = out["stages"]["escalation_exit"]
    assert (
        "did not settle within 900 s (workflow RUNNING, incident ACTION_PENDING)"
        in error
    )


def test_exit_errors_name_a_reboot_without_a_boot_transition() -> None:
    errors = fab.escalation_exit_errors(
        REBOOTED_RECOVERED,
        REBOOTED_RECOVERED["incidents"],
        boot_before="boot-1",
        boot_now="boot-1",
        node={"unschedulable": False, "taints": [], "ownership_annotations": {}},
    )
    assert errors == [
        "RESTART_NODE SUCCEEDED but the node's boot id did not change (boot-1)"
    ]


def test_exit_errors_name_a_recovered_record_that_left_isolation() -> None:
    errors = fab.escalation_exit_errors(
        REBOOTED_RECOVERED,
        REBOOTED_RECOVERED["incidents"],
        boot_before="boot-1",
        boot_now="boot-2",
        node={
            "unschedulable": True,
            "taints": [{"key": "gpu-fault.io/quarantined", "value": "x"}],
            "ownership_annotations": {},
        },
    )
    (error,) = errors
    assert error.startswith("the record ended SUCCEEDED but the node still carries"), (
        error
    )
    assert "'unschedulable': True" in error


def test_exit_errors_name_a_failed_record_whose_incident_is_not_with_an_operator() -> (
    None
):
    doc = _doc([*THROUGH_RESET, *FAILED_VALIDATION], state="RECOVERED", status="FAILED")
    errors = fab.escalation_exit_errors(
        doc, doc["incidents"], boot_before="boot-1", boot_now="boot-1"
    )
    assert errors == [
        "the record ended FAILED but the incident is RECOVERED: neither recovered "
        "nor handed to an operator"
    ]


def test_rung_reached_reads_the_rung_the_hand_off_or_the_ended_record() -> None:
    incidents = DETECTED["incidents"]
    assert not fab.rung_reached(RESET_DONE, incidents), "the reset alone is no rung"
    assert fab.rung_reached(RUNG_TAKEN, incidents), "a RESTART_NODE execution"
    assert fab.rung_reached(
        _doc(THROUGH_RESET, state="ESCALATED", status="RUNNING"), incidents
    ), "an incident handed to an operator"
    assert fab.rung_reached(_doc(THROUGH_RESET, status="SUCCEEDED"), incidents), (
        "a record that ended without a rung is judged as it is"
    )


# --------------------------------------------------------------------------- #
# the reset-only variants share the exposure
# --------------------------------------------------------------------------- #
def test_failed_variant_is_inconclusive_when_the_product_rebooted(
    tmp_path: Path,
) -> None:
    """The reset-only ``failed`` lost the race (the post-reset validation read
    the held FAILURE before the advance): not PASS on a reboot, not FAIL on
    the product -- INCONCLUSIVE with the reason."""

    out, ctx, _ = _run(
        tmp_path, "failed", [DETECTED, RESET_DONE, REBOOTED_RECOVERED], reboot_at=2
    )
    assert out["verdict"] == "INCONCLUSIVE", out
    assert out["stages"]["recovery"] == [], (
        "the record did settle and the fabric trained"
    )
    reasons = fab.reboot_exposure(
        REBOOTED_RECOVERED,
        REBOOTED_RECOVERED["incidents"],
        boot_before="boot-1",
        boot_now="boot-2",
    )
    assert len(reasons) == 2
    assert "escalated to RESTART_NODE (SUCCEEDED)" in reasons[0]
    assert "not a product error -- rerun" in reasons[0]
    assert "boot id boot-1 -> boot-2" in reasons[1]
    record = _evidence(ctx, "fabric-escalation")
    assert (
        record["rebooted"] is True and record["rungs"][0]["operation"] == "RESTART_NODE"
    )


def test_failed_variant_passes_on_a_clean_reset(tmp_path: Path) -> None:
    trained = _doc(
        [
            *THROUGH_RESET,
            _exec(5, "RESTORE_GPU_SERVICES", "SUCCEEDED", "05:16:20"),
            _exec(6, "VALIDATE_GPU", "SUCCEEDED", "05:17:40"),
            _exec(7, "VALIDATE_FABRIC", "SUCCEEDED", "05:17:50"),
            _exec(8, "RESTORE_SCHEDULING", "SUCCEEDED", "05:18:00"),
        ],
        state="RECOVERED",
        status="SUCCEEDED",
    )
    out, ctx, _ = _run(tmp_path, "failed", [DETECTED, RESET_DONE, trained])
    assert out["verdict"] == "PASS", out["errors"]
    assert out["plan_advance"]["attempts"] == 1
    assert not (ctx.case_dir / "fabric-escalation.json").exists(), (
        "no rung, no exposure record"
    )


def test_chain_requires_the_cordon_of_the_catalog_chain(tmp_path: Path) -> None:
    uncordoned = [op for op in CHAIN if op != "MARK_UNSCHEDULABLE"]
    doc = _doc(
        [e for e in THROUGH_RESET if e["operation"] != "MARK_UNSCHEDULABLE"],
        operations=uncordoned,
    )
    out, _, _ = _run(tmp_path, "failed", [doc, doc, _doc([], status="SUCCEEDED")])
    (error,) = out["stages"]["chain"]
    assert "lack" in error and "MARK_UNSCHEDULABLE" in error


# --------------------------------------------------------------------------- #
# the advance through a quiesce that still has kubelet down
# --------------------------------------------------------------------------- #
def test_advance_retries_while_the_node_cannot_answer(tmp_path: Path) -> None:
    answers = iter([False, False, True])

    def advance(arguments: tuple[str, ...]) -> dict[str, Any]:
        if not next(answers):
            raise HostProbeTransportError("owned host probe Pod is not ready")
        return {"advanced": True}

    trained = _doc(
        [
            *THROUGH_RESET,
            _exec(6, "VALIDATE_GPU", "SUCCEEDED", "05:17:40"),
            _exec(8, "RESTORE_SCHEDULING", "SUCCEEDED", "05:18:00"),
        ],
        state="RECOVERED",
        status="SUCCEEDED",
    )
    out, ctx, probe = _run(
        tmp_path, "failed", [DETECTED, RESET_DONE, trained], advance=advance
    )
    assert out["verdict"] == "PASS", out["errors"]
    assert out["plan_advance"]["attempts"] == 3
    assert out["plan_advance"]["advanced"] is True
    assert out["plan_advance"]["release"] is None
    names = [c[0] for c in probe.calls]
    assert names.count("advance") == 3
    assert names.count("status") == 0, "a transport failure never consults the tool"
    attempts = _evidence(ctx, "plan-advance")["attempts"]
    assert [a.get("ok", False) for a in attempts] == [False, False, True]
    assert "HostProbeTransportError: dcgm advance:" in attempts[0]["error"]


def test_advance_that_never_lands_is_inconclusive_and_says_why(tmp_path: Path) -> None:
    def advance(arguments: tuple[str, ...]) -> dict[str, Any]:
        raise HostProbeTransportError("owned host probe Pod has terminated")

    with pytest.raises(VariantInconclusive) as refused:
        _run(tmp_path, "failed", [DETECTED, RESET_DONE], advance=advance)
    text = str(refused.value)
    assert "could not be advanced after the reset" in text
    assert "RESTART_NODE rung for a FAILURE state" in text
    assert "attempts over" in text


def test_reset_done_wait_polls_faster_than_the_default() -> None:
    recorded: list[dict[str, Any]] = []

    class Ctx:
        regional = None
        settings = None

        def wait_until(self, accept: Any, **kwargs: Any) -> Any:
            recorded.append(kwargs)
            return None

    def quiet(ctx: Any, since: Any, *, metrics: Any = ()) -> dict[str, Any]:
        return {}

    original = replay.activity_now
    replay.activity_now = quiet  # type: ignore[assignment]
    try:
        replay.wait_gpu_incident(
            Ctx(),
            None,
            lambda d: [],
            timeout_seconds=1,
            name="x",  # type: ignore[arg-type]
        )
        replay.wait_gpu_incident(
            Ctx(),  # type: ignore[arg-type]
            None,
            lambda d: [],
            timeout_seconds=1,
            name="reset-done",
            poll_seconds=fab.RESET_POLL_SECONDS,
        )
    finally:
        replay.activity_now = original  # type: ignore[assignment]
    assert [item["poll_seconds"] for item in recorded] == [20, 10]


def test_specs_tell_the_reader_about_the_rung() -> None:
    spec = SPECS["recovery_failure"]
    assert "RESTART_NODE rung" in spec.expected and "not observable" in spec.expected
    assert spec.approvals == ("GPU_FULL_FABRIC_RESET", "NODE_REBOOT")
    for variant in (
        "not_started",
        "in_progress",
        "failed",
        "full_reset",
        "every_gpu_success",
    ):
        assert "INCONCLUSIVE, not a product error" in SPECS[variant].expected, variant
    assert fab.FULL_RESET[0] == "MARK_UNSCHEDULABLE"


# --------------------------------------------------------------------------- #
# the advance the tool refuses (HTTP 409 Conflict): plain EKS 2026-10-27 (S52)
# --------------------------------------------------------------------------- #
def _refusing(
    status: dict[str, Any], *, land_after: int | None = None
) -> dict[str, Any]:
    """``advance`` exits 1 the way the probe did when its loopback POST got the
    tool's 409 (every attempt, or the first ``land_after``); ``status`` answers
    with the tool's ``/status``."""

    seen = {"advance": 0}

    def advance(arguments: tuple[str, ...]) -> dict[str, Any]:
        seen["advance"] += 1
        if land_after is not None and seen["advance"] > land_after:
            return {"advanced": True, "status": {**status, "step": 2}}
        raise HostProbeError(REFUSED)

    return {"advance": advance, "status": {"run_id": "r", "status": status}}


def test_plan_release_reads_the_tools_status() -> None:
    plan = {
        "steps": [
            {"reads": 3, "ops": []},
            {"reads": 124, "ops": [], "until_advance": True},
            {"reads": 40, "ops": []},
        ]
    }
    aborted = fab.plan_release(plan, ABORTED_STATUS)
    assert aborted is not None and aborted["kind"] == "aborted"
    assert "3.0 -> 2.0" in aborted["reason"], aborted
    lapsed = fab.plan_release(plan, {**ABORTED_STATUS, "aborted": None})
    assert lapsed is not None and lapsed["kind"] == "lapsed"
    advanced = fab.plan_release(plan, {**ABORTED_STATUS, "aborted": None, "step": 2})
    assert advanced is not None and advanced["kind"] == "advanced"
    disarmed = fab.plan_release(
        plan, {**ABORTED_STATUS, "aborted": None, "disarmed": True}
    )
    assert disarmed is not None and disarmed["kind"] == "disarmed"
    unarmed = fab.plan_release(
        plan, {**ABORTED_STATUS, "aborted": None, "armed": False}
    )
    assert unarmed is not None and unarmed["kind"] == "unarmed"
    for release in (aborted, lapsed, advanced, disarmed, unarmed):
        assert "HTTP 409" in release["meaning"] and "held step" in release["meaning"]
    # Still on the held step: the exec failed, not the tool -- retry. An older
    # probe's status without the tool's fields explains nothing either.
    assert (
        fab.plan_release(plan, {**ABORTED_STATUS, "aborted": None, "step": 1}) is None
    )
    assert fab.plan_release(plan, {"armed": True}) is None
    assert fab.plan_release(plan, None) is None


def test_advance_refused_after_the_tools_abort_is_the_plans_release(
    tmp_path: Path,
) -> None:
    """The live plain-EKS shape: 409 on the advance, ``/status`` aborted on the
    real post-reset fabric transition; the product graded the real fabric."""

    out, ctx, probe = _run(
        tmp_path,
        "not_started",
        [DETECTED, RESET_DONE, TRAINED],
        extra=_refusing(ABORTED_STATUS),
    )
    assert out["verdict"] == "PASS", out
    advance = out["plan_advance"]
    assert advance["attempts"] == 1 and advance["advanced"] is False
    assert advance["release"]["kind"] == "aborted"
    names = [c[0] for c in probe.calls]
    assert names.count("advance") == 1 and names.count("status") == 1, names
    assert names.index("status") < names.index("close-run"), names
    record = _evidence(ctx, "plan-advance")
    assert record["attempts"][0]["error"] == f"HostProbeError: dcgm advance: {REFUSED}"
    assert record["release"]["kind"] == "aborted"
    assert record["release"]["status"] == ABORTED_STATUS
    (note,) = out["notes"]
    assert "HTTP 409" in note and "not on a held step" in note, note
    assert "3.0 -> 2.0" in note and "never masked" in note, note
    assert "real fabric" in note, note
    for timeline in ("detection", "reset-done", "settled"):
        assert (ctx.case_dir / f"{timeline}-timeline.json").exists(), timeline


def test_advance_refused_on_a_lapsed_hold_is_a_release_with_the_409_meaning(
    tmp_path: Path,
) -> None:
    status = {**ABORTED_STATUS, "aborted": None, "reads": 40}
    out, _ctx, _probe = _run(
        tmp_path,
        "in_progress",
        [DETECTED, RESET_DONE, TRAINED],
        extra=_refusing(status),
    )
    assert out["verdict"] == "PASS", out
    assert out["plan_advance"]["advanced"] is False
    assert out["plan_advance"]["release"]["kind"] == "lapsed"
    (note,) = out["notes"]
    assert "HTTP 409" in note and "read bound" in note and "40 reads" in note, note


def test_advance_refused_past_the_hold_counts_as_an_advance_that_landed(
    tmp_path: Path,
) -> None:
    status = {**ABORTED_STATUS, "aborted": None, "step": 2, "reads": 28}
    out, _ctx, probe = _run(
        tmp_path, "failed", [DETECTED, RESET_DONE, TRAINED], extra=_refusing(status)
    )
    assert out["verdict"] == "PASS", out
    assert out["plan_advance"]["attempts"] == 1
    assert out["plan_advance"]["advanced"] is True
    assert out["plan_advance"]["release"]["kind"] == "advanced"
    assert out["notes"] == [], "an advance that landed needs no note"
    assert [c[0] for c in probe.calls].count("status") == 1


def test_advance_failure_on_the_held_step_is_retried_not_classified(
    tmp_path: Path,
) -> None:
    status = {**ABORTED_STATUS, "aborted": None, "step": 1, "reads": 20}
    out, ctx, probe = _run(
        tmp_path,
        "full_reset",
        [DETECTED, RESET_DONE, TRAINED],
        extra=_refusing(status, land_after=1),
    )
    assert out["verdict"] == "PASS", out
    assert out["plan_advance"]["attempts"] == 2
    assert out["plan_advance"]["advanced"] is True
    assert out["plan_advance"]["release"] is None
    names = [c[0] for c in probe.calls]
    assert names.count("advance") == 2 and names.count("status") == 1, names
    attempts = _evidence(ctx, "plan-advance")["attempts"]
    assert [a.get("ok", False) for a in attempts] == [False, True]


@pytest.mark.parametrize(
    ("status", "word"),
    [
        ({**ABORTED_STATUS, "aborted": None, "disarmed": True}, "disarmed"),
        ({**ABORTED_STATUS, "aborted": None, "armed": False}, "not armed"),
    ],
)
def test_advance_refused_by_an_ended_tool_is_inconclusive_and_names_the_409(
    tmp_path: Path, status: dict[str, Any], word: str
) -> None:
    with pytest.raises(VariantInconclusive) as refused:
        _run(
            tmp_path,
            "every_gpu_success",
            [DETECTED, RESET_DONE, TRAINED],
            extra=_refusing(status),
        )
    text = str(refused.value)
    assert "could not be advanced after the reset" in text
    assert "HTTP 409" in text and "not on a held step" in text, text
    assert word in text and "not a product error" in text, text
    assert "did not answer" not in text, text


def test_advance_that_never_lands_names_the_probe_failure_not_the_node(
    tmp_path: Path,
) -> None:
    """A probe exit whose status read fails too (the tool itself is gone) is
    not "the node did not answer"; the text says what was and was not seen."""

    def status(arguments: tuple[str, ...]) -> dict[str, Any]:
        raise HostProbeError("dcgm status: host probe failed (exit 1); output withheld")

    extra = {**_refusing(ABORTED_STATUS), "status": status}
    with pytest.raises(VariantInconclusive) as refused:
        _run(tmp_path, "not_started", [DETECTED, RESET_DONE], extra=extra)
    text = str(refused.value)
    assert "could not be advanced after the reset" in text
    assert "the probe ran" in text and "status" in text, text
    assert "HTTP 409" in text, text
    assert "did not answer the probe" not in text, text


def test_recovery_failure_released_by_the_tools_abort_is_inconclusive_not_fail(
    tmp_path: Path,
) -> None:
    with pytest.raises(VariantInconclusive) as refused:
        _run(
            tmp_path,
            "recovery_failure",
            [DETECTED, RESET_DONE, REBOOTED_RECOVERED],
            extra=_refusing(ABORTED_STATUS),
        )
    text = str(refused.value)
    assert "held FAILURE" in text and "HTTP 409" in text, text
    assert "RESTART_NODE rung cannot be driven" in text, text
    assert "3.0 -> 2.0" in text and "not a product error" in text, text


def test_reset_variants_hold_outlasts_detection_and_the_chain(tmp_path: Path) -> None:
    """The held step's read bound: the live 36 reads ((300 + 180) / 15 + 4)
    lapsed after 39 scrapes, 585 s at 15 s; plain EKS's reset-done came 569 s
    after arm with the collector scraping through the quiesce."""

    _out, ctx, _probe = _run(tmp_path, "not_started", [DETECTED, RESET_DONE, TRAINED])
    plan = json.loads((ctx.case_dir / "plan.json").read_text(encoding="utf-8"))
    held = [step for step in plan["steps"] if step.get("until_advance")]
    assert len(held) == 1, plan
    assert held[0]["reads"] == fab.reset_hold_reads(15, 300) == 124
    assert held[0]["reads"] == replay.held_reads(
        15, fab.WINDOW + 300 + fab.RESET_CHAIN_SECONDS
    )
    assert held[0]["reads"] > 36, "the live bound was a near miss"
    assert sum(step["reads"] for step in plan["steps"]) <= tool.MAX_TOTAL_READS
    assert tool.validate_plan(plan)["steps"][1]["until_advance"] is True
    # The bound outlasts the runner's own detection horizon plus the chain.
    assert (3 + 36) * 15 < 600, "the live bound did not reach the detection horizon"
    assert held[0]["reads"] * 15 >= fab.WINDOW + 300 + fab.RESET_CHAIN_SECONDS
    with pytest.raises(VariantInconclusive, match="at most 200 per step"):
        fab.reset_hold_reads(5, 300)

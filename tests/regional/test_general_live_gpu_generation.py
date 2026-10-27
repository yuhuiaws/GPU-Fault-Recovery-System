"""The pending-remap reset chain's product generation (024 ``wrong_scope_source``,
019 ``pending_supported`` / ``post_reset``), decided before the deploy.

Live 2026-10-27 14:33-14:41Z (HyperPod, R13g, harness int/r13c-live28) the
one-read plan served the pair at read 4, reads 5-9 were the device's own, and
the repair GPU's ``row_remap_pending`` incident still ended QUARANTINED (record
settled 14:38:43Z, no reboot: boot id cc70db9b unchanged since the 14:11Z
boot) -- the post-reset VALIDATE_GPU judged the finding the minting batch left
at once. Over the real adapter of this tree and scripted control planes: the
probe reads the generation from the deployed code, a product that judges the
flag at once ends the variants INCONCLUSIVE with nothing deployed, a product
that re-judges on newer data runs the full oracle to PASS, and a chain that
ends QUARANTINED on such a product is a FAIL whose record is kept and whose
isolation the validated restore releases.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import general_live_gpu_composite as comp
from scripts.e2e.regional import general_live_gpu_generation as gen
from scripts.e2e.regional import general_live_gpu_memory as mem
from scripts.e2e.regional.general_live_fixture import VariantInconclusive
from tests.regional._general_live_fakes import FakeRegional, make_context, make_settings
from tests.regional._general_live_gpu_fakes import (
    audit_records,
    dcgm_probe,
    gpu_cpu_python,
    gpu_incident,
    gpu_workflow,
)

G0 = "GPU-00000000-0000-4000-8000-000000000000"
G1 = "GPU-11111111-1111-4111-8111-111111111111"
RESET_CHAIN = (
    "FREEZE_EVIDENCE",
    "MARK_UNSCHEDULABLE",
    "QUARANTINE",
    "QUIESCE_GPU_SERVICES",
    "VERIFY_NO_GPU_CLIENTS",
    "RESET_GPU",
    "RESTORE_GPU_SERVICES",
    "VALIDATE_GPU",
    "RESTORE_SCHEDULING",
)


def _probe(flag: str) -> dict[str, Any]:
    """The probe script run as the control plane runs it, against this tree."""

    out = subprocess.run(
        [sys.executable, "-c", gen.PENDING_FLAG_VERDICT_PROBE, flag],
        capture_output=True,
        text=True,
        timeout=180,
        check=True,
    )
    return json.loads(out.stdout.strip().splitlines()[-1])


# --------------------------------------------------------------------------- #
# The probe against the real adapter
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("flag", gen.PENDING_FLAGS)
def test_this_product_judges_the_pending_flag_at_once(flag: str) -> None:
    """The product fact behind the live FAIL, pinned on the real adapter: a
    WARNING pending-flag finding gets no grace and no wait -- the verdict is
    the state the minting batch left (R13g; the R13h candidate's
    ``gpu_validation.py`` answers the same for these two flags)."""

    answer = _probe(flag)
    assert answer["present"] is True
    assert (answer["grace"], answer["rejudges"]) == (False, False)
    assert answer["inside"] is None and answer["past"] is None
    assert answer["past_failures"] == ["active_gpu_health_findings"]
    generation = gen.PendingFlagGeneration(flag, answer)
    assert generation.rejudges is False
    assert "judges a WARNING" in generation.describe()
    assert "at once" in generation.describe()


def test_the_probe_grants_the_thermal_family_its_grace_on_the_same_adapter() -> None:
    """The classifier tells a grace from the wait: the thermal growth finding
    has the warning grace here (a cooldown inside the window) and, on this
    tree, no newer-metrics wait past it."""

    answer = _probe("thermal_violation_total_us")
    assert answer["grace"] is True
    assert answer["inside"] == ["transient_gpu_warning_cooldown"]
    assert answer["rejudges"] is (answer["past"] is not None)


def test_the_probe_reads_nothing_from_the_store_and_never_raises() -> None:
    # The probe's metrics service is its own; an import failure or an adapter
    # whose shape changed is an undecided answer, not a traceback.
    assert "store" not in gen.PENDING_FLAG_VERDICT_PROBE.split("class Metrics")[0]
    broken = gen.PENDING_FLAG_VERDICT_PROBE.replace(
        "_validate_active_findings", "_no_such_method"
    )
    out = subprocess.run(
        [sys.executable, "-c", broken, "row_remap_pending"],
        capture_output=True,
        text=True,
        timeout=180,
        check=True,
    )
    answer = json.loads(out.stdout.strip().splitlines()[-1])
    assert answer["present"] is False and answer["rejudges"] is None
    assert "AttributeError" in answer["error"]
    assert gen.PendingFlagGeneration("row_remap_pending", answer).rejudges is None


# --------------------------------------------------------------------------- #
# The classifier and its texts
# --------------------------------------------------------------------------- #
def test_generation_record_describes_both_generations() -> None:
    waits = gen.PendingFlagGeneration("row_remap_pending", gen.rejudging_answer())
    assert waits.rejudges is True and waits.grace is True
    assert "keeps WAITING past the warning grace" in waits.describe()
    assert waits.record()["rejudges"] is True

    r13g = gen.PendingFlagGeneration("row_remap_pending", gen.judging_at_once_answer())
    assert r13g.rejudges is False and r13g.grace is False
    assert "no newer-metrics wait exists at all" in r13g.describe()
    r13h = gen.PendingFlagGeneration(
        "row_remap_pending", gen.judging_at_once_answer(newer_metrics_wait=True)
    )
    assert r13h.rejudges is False
    assert "exists for other finding families" in r13h.describe()

    unknown = gen.PendingFlagGeneration(
        "row_remap_pending", {"present": False, "error": "ImportError: x"}
    )
    assert unknown.rejudges is None and unknown.grace is None
    assert "could not be read (ImportError: x)" in unknown.describe()
    # A malformed answer is undecided too, never a verdict.
    assert gen.PendingFlagGeneration("row_remap_pending", {}).rejudges is None
    assert (
        gen.PendingFlagGeneration(
            "row_remap_pending", {"present": True, "rejudges": "yes"}
        ).rejudges
        is None
    )


def test_the_inconclusive_texts_cite_the_mechanism() -> None:
    r13g = gen.PendingFlagGeneration("row_remap_pending", gen.judging_at_once_answer())
    text = gen.judges_at_once_detail(r13g, chain="the repair chain")
    for needle in (
        "expected on this product generation",
        "_validate_active_findings",
        "b202 / gap 26",
        "QUARANTINED",
        "2026-10-27 14:36Z",
        "node was not touched",
    ):
        assert needle in text, needle
    undecided = gen.undecided_detail(
        gen.PendingFlagGeneration("row_remap_pending", {}), chain="the repair chain"
    )
    assert "could not be decided" in undecided and "not touched" in undecided


# --------------------------------------------------------------------------- #
# 024 wrong_scope_source on both generations
# --------------------------------------------------------------------------- #
def _activity(incidents: list, workflows: list = ()) -> dict[str, Any]:
    return {
        "incidents": incidents,
        "workflows": list(workflows),
        "batches": [{"observed_at": "t"}],
        "markers": [],
        "commands": [],
        "notifications": [],
        "metric_latest": [],
        "collector_statuses": [],
        "agent": None,
    }


def _records(
    *, repair_state: str = "RECOVERED", repair_status: str = "SUCCEEDED"
) -> tuple[list, list]:
    error = gpu_incident(
        "inc", f"b-{G0}-ecc_dbe_volatile_total", action="QUARANTINE", workflow="wf"
    )
    isolation = gpu_workflow("wf", "inc", ["MARK_UNSCHEDULABLE", "QUARANTINE"])
    repair = gpu_incident(
        "inc-r",
        f"b-{G1}-row_remap_pending",
        state=repair_state,
        action="RESET_GPU",
        workflow="wf-r",
    )
    executions = [
        {
            "operation": operation,
            "status": "FAILED"
            if repair_status != "SUCCEEDED" and operation == "VALIDATE_GPU"
            else "SUCCEEDED",
        }
        for operation in RESET_CHAIN
        if repair_status == "SUCCEEDED" or operation != "RESTORE_SCHEDULING"
    ]
    chain = gpu_workflow(
        "wf-r", "inc-r", list(RESET_CHAIN), status=repair_status, executions=executions
    )
    return [error, repair], [isolation, chain]


def _run_024(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    generation: Any,
    incidents: list,
    workflows: list,
) -> dict[str, Any]:
    spec = {item.variant: item for item in comp.VARIANTS}["wrong_scope_source"]
    restored: list[list[str]] = []
    monkeypatch.setattr(
        comp,
        "_validated_restore",
        lambda ctx, items: restored.append([str(i["incident_id"]) for i in items])
        or [],
    )
    probe = dcgm_probe(audit=audit_records({f"{G0}/ecc_dbe_volatile_total": 1.0}))
    regional = FakeRegional(
        cpu_python=gpu_cpu_python(
            activity=_activity(incidents, workflows),
            gpu_state={"active_findings": []},
            generation=generation,
        )
    )
    settings = make_settings(spec, node=regional.node_snapshot("")["name"])
    ctx = make_context(
        spec, tmp_path=tmp_path, regional=regional, probes={}, settings=settings
    )
    ctx.probes[f"dcgm@{ctx.settings.node}"] = probe
    try:
        out = comp.wrong_scope_source(ctx)
    except VariantInconclusive as inconclusive:
        out = {"verdict": "INCONCLUSIVE", "detail": str(inconclusive)}
    out["_restored"] = restored
    out["_probe"] = probe
    out["_ctx"] = ctx
    return out


@pytest.mark.parametrize(
    "answer",
    [gen.judging_at_once_answer(), gen.judging_at_once_answer(newer_metrics_wait=True)],
    ids=["r13g", "r13h-candidate"],
)
def test_wrong_scope_source_is_inconclusive_before_the_deploy_on_a_product_that_judges_at_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, answer: dict[str, Any]
) -> None:
    """The live R13g shape: the generation-aware verdict, nothing on the node."""

    incidents, workflows = _records()
    out = _run_024(
        tmp_path,
        monkeypatch,
        generation=answer,
        incidents=incidents,
        workflows=workflows,
    )
    assert out["verdict"] == "INCONCLUSIVE"
    assert out["detail"].startswith("expected on this product generation"), out[
        "detail"
    ]
    assert "row_remap_pending WARNING finding at once" in out["detail"]
    assert "the repair GPU's service-window RESET_GPU chain" in out["detail"]
    # The survey ran (read-only); nothing was installed, deployed or armed.
    assert [call[0] for call in out["_probe"].calls] == ["survey"]
    assert out["_restored"] == []
    record = json.loads(
        (out["_ctx"].case_dir / f"{gen.GENERATION_EVIDENCE}.json").read_text()
    )
    assert record["rejudges"] is False and record["flag"] == "row_remap_pending"
    assert record["control_plane"]["newer_metrics_wait"] is answer["newer_metrics_wait"]


def test_wrong_scope_source_refuses_an_undecided_generation_the_same_way(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    incidents, workflows = _records()
    out = _run_024(
        tmp_path,
        monkeypatch,
        generation={"present": False, "error": "ImportError: no module"},
        incidents=incidents,
        workflows=workflows,
    )
    assert out["verdict"] == "INCONCLUSIVE"
    assert "could not be decided" in out["detail"]
    assert "ImportError: no module" in out["detail"]
    assert [call[0] for call in out["_probe"].calls] == ["survey"]


def test_wrong_scope_source_runs_the_full_oracle_on_a_product_that_rejudges(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The b202-shaped product that also waits for the pending flag: the chain
    ends RECOVERED by itself and the variant PASSES; the record is kept."""

    incidents, workflows = _records()
    out = _run_024(
        tmp_path,
        monkeypatch,
        generation=gen.rejudging_answer(),
        incidents=incidents,
        workflows=workflows,
    )
    assert out["verdict"] == "PASS", out["errors"]
    calls = [call[0] for call in out["_probe"].calls]
    assert "deploy-proxy" in calls and "arm" in calls and "close-run" in calls
    assert out["_restored"] == [["inc"]], "a RECOVERED repair incident is left alone"
    assert out["repair_incidents_restored"] == []
    assert out["product_generation"]["rejudges"] is True
    assert any(
        "product generation decided before the deploy" in n for n in out["notes"]
    ), out["notes"]
    record = json.loads((out["_ctx"].case_dir / "repair-record.json").read_text())
    assert {item["incident_id"] for item in record["incidents"]} == {"inc", "inc-r"}
    assert {item["request_id"] for item in record["workflows"]} == {"wf", "wf-r"}
    assert record["workflows"][1]["step_executions"][7]["operation"] == "VALIDATE_GPU"


def test_a_quarantined_repair_chain_on_a_rejudging_product_fails_and_is_released(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The live 2026-10-27 record shape against a product that should have
    re-judged: FAIL on ``repair_settled`` (no generation excuse), the
    QUARANTINED repair incident released by the validated restore so the node
    is left clean, the record in the evidence."""

    incidents, workflows = _records(repair_state="QUARANTINED", repair_status="FAILED")
    out = _run_024(
        tmp_path,
        monkeypatch,
        generation=gen.rejudging_answer(),
        incidents=incidents,
        workflows=workflows,
    )
    assert out["verdict"] == "FAIL"
    assert out["stages"]["repair_settled"] == [
        f"the row_remap_pending incident inc-r on {G1} is QUARANTINED, not RECOVERED "
        "by its own chain"
    ]
    assert out["_restored"] == [["inc", "inc-r"]]
    assert out["repair_incidents_restored"] == ["inc-r"]
    record = json.loads((out["_ctx"].case_dir / "repair-record.json").read_text())
    assert [item["status"] for item in record["workflows"]] == ["RUNNING", "FAILED"]


def test_a_repair_chain_still_running_is_not_restored_over(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    incidents, workflows = _records(
        repair_state="ACTION_PENDING", repair_status="RUNNING"
    )
    out = _run_024(
        tmp_path,
        monkeypatch,
        generation=gen.rejudging_answer(),
        incidents=incidents,
        workflows=workflows,
    )
    assert out["verdict"] == "FAIL"
    assert out["_restored"] == [["inc"]]
    assert out["repair_incidents_restored"] == []


def test_isolated_repair_and_record_helpers() -> None:
    incidents, workflows = _records(repair_state="QUARANTINED", repair_status="FAILED")
    activity = _activity(incidents, workflows)
    repair = [incidents[1]]
    assert [i["incident_id"] for i in comp.isolated_repair(activity, repair)] == [
        "inc-r"
    ]
    recovered = dict(incidents[1], state="RECOVERED")
    assert (
        comp.isolated_repair(_activity([incidents[0], recovered], workflows), repair)
        == []
    )
    running = gpu_workflow("wf-r", "inc-r", list(RESET_CHAIN), status="RUNNING")
    assert (
        comp.isolated_repair(_activity(incidents, [workflows[0], running]), repair)
        == []
    )
    record = comp.record_of(activity, repair)
    assert [i["incident_id"] for i in record["incidents"]] == ["inc-r"]
    assert [w["request_id"] for w in record["workflows"]] == ["wf-r"]


# --------------------------------------------------------------------------- #
# 019 pending_supported / post_reset share the dependence; the other 024
# variants do not
# --------------------------------------------------------------------------- #
def test_019_pending_variants_are_inconclusive_before_the_deploy_on_r13g(
    tmp_path: Path,
) -> None:
    for variant in ("pending_supported", "post_reset"):
        spec = {item.variant: item for item in mem.VARIANTS}[variant]
        probe = dcgm_probe()
        regional = FakeRegional(
            cpu_python=gpu_cpu_python(generation=gen.judging_at_once_answer())
        )
        settings = make_settings(spec, node=regional.node_snapshot("")["name"])
        ctx = make_context(
            spec,
            tmp_path=tmp_path / variant,
            regional=regional,
            probes={},
            settings=settings,
        )
        ctx.probes[f"dcgm@{ctx.settings.node}"] = probe
        with pytest.raises(VariantInconclusive, match="expected on this product"):
            spec.execute(ctx)
        assert [call[0] for call in probe.calls] == ["survey"]
        assert (ctx.case_dir / f"{gen.GENERATION_EVIDENCE}.json").is_file(), (
            "the probe's answer is evidence even when the variant refuses"
        )


def test_only_the_reset_variants_depend_on_the_generation() -> None:
    """The sibling 024 variants plan no reset (QUARANTINE or diagnostic
    composites), so no post-reset VALIDATE_GPU judges a held flag there: the
    gate is declared exactly where ``GPU_RESET`` is."""

    reset_variants = {
        spec.variant for spec in comp.VARIANTS if "GPU_RESET" in spec.approvals
    }
    assert reset_variants == {"wrong_scope_source"}
    assert {spec.variant for spec in mem.VARIANTS if "GPU_RESET" in spec.approvals} == {
        "pending_supported",
        "post_reset",
    }
    assert "re-judges the pending flag" in comp.VARIANTS[5].expected

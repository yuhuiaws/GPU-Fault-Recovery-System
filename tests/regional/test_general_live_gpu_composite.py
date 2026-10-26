"""GF-GENERAL-LIVE-024 composite injector/verdict contract (scripted)."""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import general_live_gpu_composite as comp
from scripts.e2e.regional.general_live_fixture import VariantNotRun
from scripts.e2e.regional.general_live_gpu_replay import GpuSurvey, add, step
from tests.regional._general_live_fakes import (
    FakeProbe,
    FakeRegional,
    make_context,
    make_settings,
)
from tests.regional._general_live_gpu_fakes import (
    audit_records,
    dcgm_probe,
    gpu_cpu_python,
    gpu_incident,
    gpu_workflow,
    survey,
)

SPECS = {spec.variant: spec for spec in comp.VARIANTS}
G0 = "GPU-00000000-0000-4000-8000-000000000000"
G1 = "GPU-11111111-1111-4111-8111-111111111111"
PRIMARY = "ecc_sbe_volatile_total"
#: The second correctable-memory component per exporter generation, under the
#: exporter sample name the survey records (typed counters arrive ``_total``):
#: the retired-page counter of parts that still retire pages, the correctable
#: row-remap counter of Ampere/Hopper and later (the live 2026-10-26 H200
#: reports the latter and nothing of ``retired_pages_*``).
LEGACY_SBE = ("retired_pages_sbe_total", "DCGM_FI_DEV_RETIRED_SBE_total")
HOPPER_SBE = (
    "row_remap_correctable_total",
    "DCGM_FI_DEV_CORRECTABLE_REMAPPED_ROWS_total",
)
#: The pending-repair flag per generation (the default fake survey reports the
#: modern one, ``row_remap_pending``).
LEGACY_PENDING = ("retired_pages_pending", "DCGM_FI_DEV_RETIRED_PENDING")
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


def _ctx(tmp_path: Path, variant: str, *, probe: Any, cpu: Any):
    spec = SPECS[variant]
    regional = FakeRegional(cpu_python=cpu)
    settings = make_settings(spec, node=regional.node_snapshot("")["name"])
    ctx = make_context(
        spec, tmp_path=tmp_path, regional=regional, probes={}, settings=settings
    )
    ctx.probes[f"dcgm@{ctx.settings.node}"] = probe
    return ctx


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


def _memory_survey(
    *reported: tuple[str, str],
    without: tuple[str, ...] = (),
    only_on: dict[str, tuple[str, ...]] | None = None,
) -> dict[str, Any]:
    """The default survey plus ``reported`` fields under their exporter names.

    ``without`` drops default fields (a legacy part has no row-remap flag);
    ``only_on`` restricts a reported field to some GPUs, as a part-populated
    exporter would.
    """

    value = survey()
    for gpu, entry in value["exporter"]["gpus"].items():
        for name in without:
            entry["values"].pop(name, None)
            entry["metric_names"].pop(name, None)
        for canonical, metric in reported:
            if gpu not in (only_on or {}).get(canonical, (G0, G1)):
                continue
            entry["values"][canonical] = 0.0
            entry["metric_names"][canonical] = [metric]
    return value


def _installed_plan(probe: FakeProbe) -> dict[str, Any]:
    """The field plan the controller installed on the node (``install-source``)."""

    for call in probe.calls:
        if (
            call[0] == "install-source"
            and call[call.index("--name") + 1] == "plan.json"
        ):
            return json.loads(base64.b64decode(call[call.index("--b64") + 1]))
    raise AssertionError("no plan.json was installed")


def _plan_targets(document: dict[str, Any]) -> set[tuple[str, str]]:
    return {
        (operation["field"], gpu)
        for item in document["steps"]
        for operation in item["ops"]
        if "field" in operation
        for gpu in operation["gpus"]
    }


def _correctable_cpu() -> Any:
    inc = gpu_incident(
        "inc", f"b-{G1}-composite-CORRECTABLE_MEMORY_DEGRADATION", state="RECOVERED"
    )
    wf = gpu_workflow("wf", "inc", ["FREEZE_EVIDENCE", "VALIDATE_GPU"], status="FAILED")
    return gpu_cpu_python(
        activity=_activity([inc], [wf]),
        gpu_state={"active_findings": [], "batches": [{"observed_at": "t"}]},
    )


def test_registry_covers_024() -> None:
    from scripts.e2e.regional import general_live_gpu_environment as envc
    from scripts.e2e.regional import general_live_registry as reg

    environment = {spec.variant for spec in envc.VARIANTS}
    assert environment == {"thermal_warning", "power_performance"}
    assert set(SPECS) | environment == set(reg.coverage_rows()["GF-GENERAL-LIVE-024"])
    assert [variant for variant, spec in SPECS.items() if spec.deferred] == []
    assert SPECS["independent_xid"].approvals == ("KERNEL_SOURCE_REPLAY",)
    assert SPECS["independent_xid"].risk == "M"
    # The repair GPU's own finding runs the product's service-window reset:
    # the same token, exclusivity and risk as 019 pending_supported.
    spec = SPECS["wrong_scope_source"]
    assert spec.approvals == ("GPU_RESET",)
    assert (spec.flags, spec.risk) == ("X", "H")
    assert "row-remap-pending" in reg.APPROVALS["GPU_RESET"], (
        "the GPU_RESET token names the replayed reading"
    )
    assert "RESET_GPU" in spec.injection and "RECOVERED by itself" in spec.expected
    for variant in ("corrected_memory", "fresh_components", "reused_components"):
        text = SPECS[variant].injection
        assert "retired-page counter on pre-Hopper" in text, variant
        assert "retired_pages_sbe_total" not in text, variant


def _xid_decision(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "event_id": "kernel-log-kmsg-boot-900000064-xid-92-cluster-abc",
        "xid": comp.INDEPENDENT_XID,
        "gpu_uuid": G1,
        "event_source": "KERNEL_LOG",
        "decision": True,
        "disposition": "MONITOR_ONLY",
        "official_action": "IGNORE",
        "action": "NO_ACTION",
        "safety_action": None,
        "requires_operator": False,
        "incident_id": "inc-xid-92",
        "incident_state": "RECOVERED",
        "workflow_request_id": None,
        "duplicate": False,
        "correlated_event_id": None,
        "reasons": [],
    }
    base.update(overrides)
    return base


def _kernel_script() -> dict[str, Any]:
    return {
        "install-records": {"records": 1},
        "open-window": {"after": {"ActiveState": "active"}},
        "arm": {"armed": True, "lines_written": 6},
        "status": {"injected": {"mode": "kmsg", "fixture_lines": 6}},
        "close-window": {"clean": True},
    }


def _independent_ctx(
    tmp_path: Path,
    decisions: list[dict[str, Any]],
    *,
    second: tuple[str, str] = HOPPER_SBE,
) -> Any:
    from scripts.e2e.regional.general_live_kernel_xid import XID_DECISION_PROBE

    components = [PRIMARY, second[0]]
    inc = gpu_incident(
        "inc-c", f"b-{G1}-composite-CORRECTABLE_MEMORY_DEGRADATION", state="RECOVERED"
    )
    wf = gpu_workflow(
        "wf", "inc-c", ["FREEZE_EVIDENCE", "VALIDATE_GPU"], status="FAILED"
    )
    gpu_state = {
        "active_findings": [],
        "findings": [
            {
                "finding_id": f"b-{G1}-composite-CORRECTABLE_MEMORY_DEGRADATION",
                "correlation_rule_id": "CORRECTABLE_MEMORY_DEGRADATION",
                "component_metrics": components,
                "component_finding_ids": [f"b-{G1}-{name}" for name in components],
                "gpu_uuid": G1,
                "observed_at": "t",
            }
        ],
        "batches": [{"observed_at": "t"}],
    }
    records = gpu_cpu_python(activity=_activity([inc], [wf]), gpu_state=gpu_state)

    def cpu(script: str, *arguments: str) -> dict[str, Any]:
        if script == XID_DECISION_PROBE:
            return {"decisions": decisions}
        return records(script, *arguments)

    probe = dcgm_probe(
        _memory_survey(second),
        audit=audit_records({f"{G1}/{name}": 1.0 for name in components}),
    )
    ctx = _ctx(tmp_path, "independent_xid", probe=probe, cpu=cpu)
    ctx.probes[f"kernel-replay-kmsg@{ctx.settings.node}"] = FakeProbe(_kernel_script())
    return ctx


def test_independent_xid_keeps_its_own_decision_beside_the_weak_composite(
    tmp_path: Path,
) -> None:
    ctx = _independent_ctx(tmp_path, [_xid_decision()])
    out = comp.independent_xid(ctx)
    assert out["verdict"] == "PASS", out["errors"]
    assert out["xid"] == 92 and out["pci_bdf"] == "0000:02:00"
    assert out["kernel_replay"]["mode"] == "kmsg"
    assert out["second_component"] == "row_remap_correctable_total"
    assert out["component_metrics"] == [PRIMARY, "row_remap_correctable_total"]
    kernel = ctx.probes[f"kernel-replay-kmsg@{ctx.settings.node}"]
    verbs = [call[0] for call in kernel.calls]
    assert verbs == ["install-records", "open-window", "arm", "status", "close-window"]
    assert set(out["stages"]) == {
        "composite_chain",
        "composite_read_only",
        "independent",
        "not_covered",
        "hardware",
        "isolation",
    }


def test_independent_xid_holds_the_legacy_pair_where_pages_are_retired(
    tmp_path: Path,
) -> None:
    # The composite's own components are the replayed pair, whichever the node
    # has; the not_covered oracle allows exactly those two.
    ctx = _independent_ctx(tmp_path, [_xid_decision()], second=LEGACY_SBE)
    out = comp.independent_xid(ctx)
    assert out["verdict"] == "PASS", out["errors"]
    assert out["second_component"] == "retired_pages_sbe_total"
    probe = ctx.probes[f"dcgm@{ctx.settings.node}"]
    assert _plan_targets(_installed_plan(probe)) == {
        (PRIMARY, G1),
        ("retired_pages_sbe_total", G1),
    }


def test_independent_xid_fails_when_absorbed_into_the_composite_incident(
    tmp_path: Path,
) -> None:
    ctx = _independent_ctx(tmp_path, [_xid_decision(incident_id="inc-c")])
    out = comp.independent_xid(ctx)
    assert out["verdict"] == "FAIL"
    assert any("absorbed" in error for error in out["errors"]), out["errors"]


def test_independent_xid_fails_when_the_xid_is_dropped(tmp_path: Path) -> None:
    out = comp.independent_xid(_independent_ctx(tmp_path, []))
    assert out["verdict"] == "FAIL"
    assert any("reached no decision" in error for error in out["errors"]), out["errors"]


def _hopper_survey() -> dict[str, Any]:
    """The live 2026-10-26 H200 shape: NVLink counters only as DL aggregates."""

    aggregates = {
        "nvlink_crc_aggregate_error_total": "DCGM_FI_DEV_NVLINK_ERROR_DL_CRC_total",
        "nvlink_replay_aggregate_error_total": "DCGM_FI_DEV_NVLINK_ERROR_DL_REPLAY_total",
        "nvlink_recovery_aggregate_error_total": (
            "DCGM_FI_DEV_NVLINK_ERROR_DL_RECOVERY_total"
        ),
    }
    value = survey(
        fields=("ecc_sbe_volatile_total", "ecc_dbe_volatile_total", *aggregates)
    )
    for entry in value["exporter"]["gpus"].values():
        for canonical, metric in aggregates.items():
            entry["metric_names"][canonical] = [metric]
    return value


def test_nvlink_multiple_gpus_targets_the_counter_the_node_reports(
    tmp_path: Path,
) -> None:
    canonical = "nvlink_crc_aggregate_error_total"
    comp_incident = gpu_incident(
        "inc-c",
        "b-node-composite-MULTI_GPU_NVLINK_FABRIC_FAILURE",
        state="RECOVERED",
        workflow="wf-c",
    )
    wf = gpu_workflow("wf-c", "inc-c", ["FREEZE_EVIDENCE"], status="SUCCEEDED")
    probe = dcgm_probe(
        _hopper_survey(),
        audit=audit_records({f"{G0}/{canonical}": 1.0, f"{G1}/{canonical}": 1.0}),
    )
    out = comp.nvlink_multiple_gpus(
        _ctx(
            tmp_path,
            "nvlink_multiple_gpus",
            probe=probe,
            cpu=gpu_cpu_python(
                activity=_activity([comp_incident], [wf]),
                gpu_state={"active_findings": [], "batches": [{"observed_at": "t"}]},
            ),
        )
    )
    assert out["verdict"] == "PASS", out["errors"]
    assert out["gpus"] == [G0, G1]
    assert out["canonical_name"] == canonical
    assert out["exporter_metrics"] == ["DCGM_FI_DEV_NVLINK_ERROR_DL_CRC_total"]
    assert any(canonical in note for note in out["notes"]), out["notes"]
    document = _installed_plan(probe)
    assert document["profile"] == "composite"
    assert {
        operation["field"] for item in document["steps"] for operation in item["ops"]
    } == {canonical}


# --------------------------------------------------------------------------- #
# The second correctable-memory component: what the exporter has
# --------------------------------------------------------------------------- #
def test_correctable_pair_follows_the_exporter_generation() -> None:
    primary, candidates = comp.CORRECTABLE_PRIMARY, comp.CORRECTABLE_COMPANIONS
    hopper = GpuSurvey(_memory_survey(HOPPER_SBE)).component_pair(primary, candidates)
    assert (hopper.gpu, hopper.companion) == (G1, "row_remap_correctable_total")
    assert hopper.components == (PRIMARY, "row_remap_correctable_total")
    assert hopper.exporter_metrics == ("DCGM_FI_DEV_CORRECTABLE_REMAPPED_ROWS_total",)
    assert hopper.keys() == [f"{G1}/{PRIMARY}", f"{G1}/row_remap_correctable_total"]
    assert "row_remap_correctable_total" in hopper.note(), hopper.note()
    assert hopper.evidence()["second_component"] == "row_remap_correctable_total"
    legacy = GpuSurvey(_memory_survey(LEGACY_SBE)).component_pair(primary, candidates)
    assert legacy.companion == "retired_pages_sbe_total"
    # A part reporting both still retires pages: the retired-page counter first.
    both = GpuSurvey(_memory_survey(LEGACY_SBE, HOPPER_SBE))
    assert (
        both.component_pair(primary, candidates).companion == "retired_pages_sbe_total"
    )
    # The same GPU must report both: the companion on GPU 0 only moves the target.
    partial = GpuSurvey(
        _memory_survey(HOPPER_SBE, only_on={"row_remap_correctable_total": (G0,)})
    )
    assert partial.component_pair(primary, candidates).gpu == G0
    assert partial.component_pair(primary, candidates, avoid=[G1]).gpu == G0
    with pytest.raises(VariantNotRun) as absent:
        GpuSurvey(survey()).component_pair(primary, candidates)
    assert absent.value.reason == "CAPABILITY_ABSENT"
    assert "retired_pages_sbe_total" in str(absent.value)
    assert "row_remap_correctable_total" in str(absent.value)
    with pytest.raises(VariantNotRun, match=f"no GPU on the node reports {PRIMARY}"):
        GpuSurvey(survey(fields=("ecc_dbe_volatile_total",))).component_pair(
            primary, candidates
        )


@pytest.mark.parametrize(
    "variant", ["corrected_memory", "fresh_components", "reused_components"]
)
@pytest.mark.parametrize("second", [LEGACY_SBE, HOPPER_SBE], ids=["legacy", "hopper"])
def test_correctable_variants_replay_the_second_component_the_node_reports(
    tmp_path: Path, variant: str, second: tuple[str, str]
) -> None:
    companion, metric = second
    probe = dcgm_probe(
        _memory_survey(second),
        audit=audit_records({f"{G1}/{PRIMARY}": 1.0, f"{G1}/{companion}": 1.0}),
    )
    out = getattr(comp, variant)(
        _ctx(tmp_path, variant, probe=probe, cpu=_correctable_cpu())
    )
    assert out["verdict"] == "PASS", out["errors"]
    assert out["gpu_uuid"] == G1
    assert out["second_component"] == companion
    assert out["component_metrics"] == [PRIMARY, companion]
    assert out["exporter_metrics"] == [metric]
    assert any(companion in note and metric in note for note in out["notes"]), out[
        "notes"
    ]
    document = _installed_plan(probe)
    assert document["profile"] == "composite"
    assert _plan_targets(document) == {(PRIMARY, G1), (companion, G1)}
    assert "retired_pages_sbe_total" not in json.dumps(document) or (
        companion == "retired_pages_sbe_total"
    )


@pytest.mark.parametrize(
    "variant",
    ["corrected_memory", "fresh_components", "reused_components", "independent_xid"],
)
def test_correctable_variants_are_not_run_without_a_second_component(
    tmp_path: Path, variant: str
) -> None:
    # Fail fast: the default survey reports neither candidate, so nothing is
    # installed on the node and the refusal names both (the live 2026-10-26
    # H200 run aborted after arm instead: INCONCLUSIVE "refusing invented
    # samples ... retired_pages_sbe_total").
    probe = dcgm_probe()
    ctx = _ctx(tmp_path, variant, probe=probe, cpu=gpu_cpu_python())
    with pytest.raises(VariantNotRun) as absent:
        getattr(comp, variant)(ctx)
    assert absent.value.reason == "CAPABILITY_ABSENT"
    assert "retired_pages_sbe_total" in str(absent.value), variant
    assert "row_remap_correctable_total" in str(absent.value), variant
    assert [call[0] for call in probe.calls] == ["survey"], variant


def test_composite_replay_refuses_a_plan_naming_an_unreported_reading(
    tmp_path: Path,
) -> None:
    probe = dcgm_probe()
    ctx = _ctx(tmp_path, "fresh_components", probe=probe, cpu=gpu_cpu_python())
    with pytest.raises(VariantNotRun) as refused:
        comp.composite_replay(
            ctx,
            GpuSurvey(survey()),
            step(
                40, add(PRIMARY, [G1]), add("retired_pages_sbe_total", [G1]), hold=True
            ),
        )
    assert refused.value.reason == "CAPABILITY_ABSENT"
    assert f"{G1}/retired_pages_sbe_total" in str(refused.value)
    assert f"{G1}/{PRIMARY}" not in str(refused.value), "only the absent reading"
    assert probe.calls == [], "refused before anything reached the node"


def test_hopper_fresh_components_keeps_the_replay_call_sequence(tmp_path: Path) -> None:
    # Pinned deliberately: the pair and the preflight are answered by the one
    # survey the variant already made, so a Hopper node costs no extra probe
    # call -- the replay deploys (tool + plan), repoints, arms, reads the
    # journal until both components are served, and is released through
    # close-run (its own journal read first).
    probe = dcgm_probe(
        _memory_survey(HOPPER_SBE),
        audit=audit_records(
            {f"{G1}/{PRIMARY}": 1.0, f"{G1}/row_remap_correctable_total": 1.0}
        ),
    )
    out = comp.fresh_components(
        _ctx(tmp_path, "fresh_components", probe=probe, cpu=_correctable_cpu())
    )
    assert out["verdict"] == "PASS", out["errors"]
    assert [call[0] for call in probe.calls] == [
        "survey",
        "install-source",
        "install-source",
        "deploy-proxy",
        "repoint-collector",
        "arm",
        "audit-read",
        "audit-read",
        "close-run",
    ]


def test_corrected_memory_forms_a_diagnostic_composite(tmp_path: Path) -> None:
    probe = dcgm_probe(
        _memory_survey(HOPPER_SBE),
        audit=audit_records(
            {f"{G1}/{PRIMARY}": 1.0, f"{G1}/row_remap_correctable_total": 1.0}
        ),
    )
    out = comp.corrected_memory(
        _ctx(tmp_path, "corrected_memory", probe=probe, cpu=_correctable_cpu())
    )
    assert out["verdict"] == "PASS", out["errors"]
    assert set(out["stages"]) == {"chain", "read_only", "hardware"}


# --------------------------------------------------------------------------- #
# The pending-repair gauge beside the uncorrectable error
# --------------------------------------------------------------------------- #
def _memory_failure_cpu() -> Any:
    inc = gpu_incident(
        "inc",
        f"b-{G1}-composite-GPU_MEMORY_DEGRADATION",
        state="QUARANTINED",
        action="QUARANTINE",
        workflow="wf",
    )
    # An ended record is judged on executed steps (live 2026-10-26: a record
    # that only planned its containment passed the chain): the steps ran.
    wf = gpu_workflow(
        "wf",
        "inc",
        ["FREEZE_EVIDENCE", "MARK_UNSCHEDULABLE", "QUARANTINE"],
        status="BLOCKED",
        executions=[
            {"operation": op, "status": "SUCCEEDED", "updated_at": f"t{i}"}
            for i, op in enumerate(
                ["FREEZE_EVIDENCE", "MARK_UNSCHEDULABLE", "QUARANTINE"]
            )
        ],
    )
    return gpu_cpu_python(
        activity=_activity([inc], [wf]), gpu_state={"active_findings": []}
    )


def test_memory_failure_is_critical_isolation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(comp, "_validated_restore", lambda ctx, incidents: [])
    probe = dcgm_probe(audit=audit_records({f"{G1}/ecc_dbe_volatile_total": 1.0}))
    out = comp.memory_failure(
        _ctx(tmp_path, "memory_failure", probe=probe, cpu=_memory_failure_cpu())
    )
    assert out["verdict"] == "PASS", out["errors"]
    assert out["second_component"] == "row_remap_pending"
    assert out["component_metrics"] == ["ecc_dbe_volatile_total", "row_remap_pending"]
    assert _plan_targets(_installed_plan(probe)) == {
        ("ecc_dbe_volatile_total", G1),
        ("row_remap_pending", G1),
    }


def test_memory_failure_uses_the_retirement_flag_on_a_legacy_part(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(comp, "_validated_restore", lambda ctx, incidents: [])
    probe = dcgm_probe(
        _memory_survey(LEGACY_PENDING, without=("row_remap_pending",)),
        audit=audit_records({f"{G1}/ecc_dbe_volatile_total": 1.0}),
    )
    out = comp.memory_failure(
        _ctx(tmp_path, "memory_failure", probe=probe, cpu=_memory_failure_cpu())
    )
    assert out["verdict"] == "PASS", out["errors"]
    assert out["second_component"] == "retired_pages_pending"
    assert out["exporter_metrics"] == ["DCGM_FI_DEV_RETIRED_PENDING"]
    document = _installed_plan(probe)
    assert _plan_targets(document) == {
        ("ecc_dbe_volatile_total", G1),
        ("retired_pages_pending", G1),
    }
    [flag] = [
        operation
        for item in document["steps"]
        for operation in item["ops"]
        if operation.get("field") == "retired_pages_pending"
    ]
    assert (flag["op"], flag["value"]) == ("set", 1)


@pytest.mark.parametrize("variant", ["memory_failure", "wrong_scope_source"])
def test_repair_variants_are_not_run_without_a_pending_flag(
    tmp_path: Path, variant: str
) -> None:
    probe = dcgm_probe(_memory_survey(without=("row_remap_pending",)))
    with pytest.raises(VariantNotRun) as absent:
        getattr(comp, variant)(
            _ctx(tmp_path, variant, probe=probe, cpu=gpu_cpu_python())
        )
    assert absent.value.reason == "CAPABILITY_ABSENT"
    assert "row_remap_pending" in str(absent.value), variant
    assert "retired_pages_pending" in str(absent.value), variant
    assert [call[0] for call in probe.calls] == ["survey"], variant


# --------------------------------------------------------------------------- #
# wrong_scope_source: hardware judged per incident, the repair chain settles
# --------------------------------------------------------------------------- #
def _repair_incident(
    *, state: str = "RECOVERED", status: str = "SUCCEEDED", chain=RESET_CHAIN
) -> tuple[dict[str, Any], dict[str, Any]]:
    """The repair GPU's own row_remap_pending incident and its reset chain
    (the live 2026-10-26 07:47-07:50Z eksb record: every step SUCCEEDED)."""

    incident = gpu_incident(
        "inc-r",
        f"b-{G1}-row_remap_pending",
        state=state,
        action="RESET_GPU",
        workflow="wf-r",
    )
    executions = [
        {"operation": operation, "status": "SUCCEEDED"}
        for operation in chain
        if status == "SUCCEEDED"
    ]
    workflow = gpu_workflow(
        "wf-r", "inc-r", list(chain), status=status, executions=executions
    )
    return incident, workflow


def _error_incident(
    operations: tuple[str, ...] = ("MARK_UNSCHEDULABLE", "QUARANTINE"),
) -> tuple[dict[str, Any], dict[str, Any]]:
    incident = gpu_incident(
        "inc", f"b-{G0}-ecc_dbe_volatile_total", action="QUARANTINE", workflow="wf"
    )
    return incident, gpu_workflow("wf", "inc", list(operations))


def _wrong_scope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, incidents: list, workflows: list
) -> dict[str, Any]:
    restored: list[list[str]] = []
    monkeypatch.setattr(
        comp,
        "_validated_restore",
        lambda ctx, items: restored.append([str(i["incident_id"]) for i in items])
        or [],
    )
    out = comp.wrong_scope_source(
        _ctx(
            tmp_path,
            "wrong_scope_source",
            probe=dcgm_probe(
                audit=audit_records({f"{G0}/ecc_dbe_volatile_total": 1.0})
            ),
            cpu=gpu_cpu_python(
                activity=_activity(incidents, workflows),
                gpu_state={"active_findings": []},
            ),
        )
    )
    out["_restored"] = restored
    return out


def test_wrong_scope_source_passes_when_no_composite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    error, isolation = _error_incident()
    repair, chain = _repair_incident()
    out = _wrong_scope(tmp_path, monkeypatch, [error, repair], [isolation, chain])
    assert out["verdict"] == "PASS", out["errors"]
    assert set(out["stages"]) == {
        "no_cross_gpu_composite",
        "hardware",
        "repair_settled",
        "recovery",
    }
    assert (out["error_gpu"], out["repair_gpu"]) == (G0, G1)
    assert out["second_component"] == "row_remap_pending"
    assert out["component_metrics"] == ["ecc_dbe_volatile_total", "row_remap_pending"]
    assert out["repair_incidents"] == ["inc-r"]
    assert out["_restored"] == [["inc"]], "only the error GPU's incident is restored"
    assert any("settles RECOVERED by itself" in note for note in out["notes"]), out[
        "notes"
    ]


def test_wrong_scope_source_fails_if_a_composite_forms(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bad = gpu_incident(
        "inc-c",
        f"b-{G0}-composite-GPU_MEMORY_DEGRADATION",
        action="QUARANTINE",
        workflow="wf",
    )
    wf = gpu_workflow("wf", "inc-c", ["MARK_UNSCHEDULABLE", "QUARANTINE"])
    repair, chain = _repair_incident()
    out = _wrong_scope(tmp_path, monkeypatch, [bad, repair], [wf, chain])
    assert out["verdict"] == "FAIL" and any(
        "another GPU" in e for e in out["errors"]
    ), out["errors"]


def test_wrong_scope_source_allows_the_repair_gpus_own_reset_chain_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The error GPU's isolation planning a reset is the hardware stage's error
    # even while the repair GPU's chain (QUIESCE + RESET_GPU) is allowed.
    error, isolation = _error_incident(
        ("MARK_UNSCHEDULABLE", "QUARANTINE", "RESET_GPU")
    )
    repair, chain = _repair_incident()
    out = _wrong_scope(tmp_path, monkeypatch, [error, repair], [isolation, chain])
    assert out["verdict"] == "FAIL"
    assert out["stages"]["hardware"] == ["workflow wf plans ['RESET_GPU']"]
    assert out["stages"]["repair_settled"] == []
    # The repair chain itself may not escalate beyond the single-GPU reset.
    error, isolation = _error_incident()
    repair, chain = _repair_incident(chain=(*RESET_CHAIN, "RESTART_NODE"))
    out = _wrong_scope(tmp_path, monkeypatch, [error, repair], [isolation, chain])
    assert out["stages"]["hardware"] == ["workflow wf-r plans ['RESTART_NODE']"]


def test_wrong_scope_source_requires_the_repair_incident_to_settle_itself(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    error, isolation = _error_incident()
    repair, chain = _repair_incident(state="ACTION_PENDING", status="RUNNING")
    out = _wrong_scope(tmp_path, monkeypatch, [error, repair], [isolation, chain])
    assert out["verdict"] == "FAIL"
    assert [e for e in out["stages"]["repair_settled"] if "did not settle" in e], out[
        "errors"
    ]
    assert any("ACTION_PENDING, not RECOVERED" in e for e in out["errors"]), out[
        "errors"
    ]
    assert out["_restored"] == [["inc"]], "the repair incident is never restored here"
    # A missing repair incident is named too (the replayed flag minted nothing).
    out = _wrong_scope(tmp_path, monkeypatch, [error], [isolation])
    assert out["stages"]["repair_settled"] == [
        f"the replayed row_remap_pending on {G1} minted no incident of its own"
    ]


def test_scoped_hardware_errors_judge_each_incident_on_its_own() -> None:
    repair, chain = _repair_incident()
    _error, isolation = _error_incident()
    activity = _activity([repair], [chain, isolation])
    assert comp.scoped_hardware_errors(activity, repair_incidents=[repair]) == []
    # Without the scope the same record fails exactly as it did live.
    assert comp.scoped_hardware_errors(activity, repair_incidents=[]) == [
        "workflow wf-r plans ['QUIESCE_GPU_SERVICES', 'RESET_GPU']"
    ]
    other = gpu_workflow("wf-x", "inc-x", ["RESET_GPU"])
    assert comp.scoped_hardware_errors(
        _activity([repair], [chain, other]), repair_incidents=[repair]
    ) == ["workflow wf-x plans ['RESET_GPU']"]
    assert (
        comp.repair_settled_errors(
            activity, [repair], gpu=G1, component="row_remap_pending"
        )
        == []
    )


# --------------------------------------------------------------------------- #
# The product's rules behind the survey-chosen pairs
# --------------------------------------------------------------------------- #
def test_product_composes_either_second_component_with_the_primary() -> None:
    """The survey-chosen pair rests on the product's composite rules taking
    either candidate: CORRECTABLE_MEMORY_DEGRADATION (RUN_DIAGNOSTICS) from the
    volatile SBE counter with the retired-page or the correctable row-remap
    counter, GPU_MEMORY_DEGRADATION (QUARANTINE, the error side's action) from
    the volatile DBE counter with the row-remap or the page-retirement pending
    flag -- so a plan may pick whichever the exporter publishes.
    """

    from datetime import datetime, timedelta, timezone

    from gpu_fault.gpu_metrics import (
        GpuMetricBatch,
        GpuMetricSample,
        GpuMetricSource,
        GpuMetricsService,
    )

    start = datetime(2026, 10, 26, 7, 41, tzinfo=timezone.utc)

    def sample(name: str, value: float) -> GpuMetricSample:
        return GpuMetricSample(
            metric_name=f"DCGM_{name.upper()}",
            canonical_name=name,
            value=value,
            gpu_index="1",
            gpu_uuid=G1,
            pci_bdf="0000:01:00.0",
        )

    def composite_of(primary: str, companion: str):
        service = GpuMetricsService()
        for tick, value in ((0, 0), (1, 1)):
            outcome = service.ingest(
                GpuMetricBatch(
                    batch_id=f"b{tick}-{companion}",
                    cluster_id="gpu-eks-cluster",
                    node_id="ip-10-0-43-5.us-west-2.compute.internal",
                    observed_at=start + timedelta(seconds=15 * tick),
                    source=GpuMetricSource.DCGM_EXPORTER,
                    samples=[sample(primary, value), sample(companion, value)],
                )
            )
        [finding] = outcome.new_composite_findings
        assert sorted(finding.component_metrics) == sorted([primary, companion])
        return finding

    for companion in comp.CORRECTABLE_COMPANIONS:
        finding = composite_of(comp.CORRECTABLE_PRIMARY, companion)
        assert finding.correlation_rule_id == comp.CORRECTABLE_RULE, companion
        assert finding.automatic_action == "RUN_DIAGNOSTICS", companion
    for companion in comp.REPAIR_COMPANIONS:
        finding = composite_of(comp.UNCORRECTABLE_PRIMARY, companion)
        assert finding.correlation_rule_id == comp.MEMORY_RULE, companion
        assert finding.automatic_action == "QUARANTINE", companion


@pytest.mark.xfail(
    strict=True,
    reason=(
        "product gap, live 2026-10-26 08:07Z (eks ip-10-0-43-5, R13g, 024 "
        "independent_xid): the CORRECTABLE_MEMORY_DEGRADATION composite turns "
        "CRITICAL at correctable_memory_drain_consecutive_samples "
        "(gpu_metrics._add_memory_composites) and the severity change is a new "
        "activation -- a batch-stamped second finding id (gpu_composites.composite_finding) "
        "and a second NODE_HEALTH incident (app/ingest/gpu_findings: event_id "
        "gpu-<finding id>) for one sustained episode; awaiting the product lane"
    ),
)
def test_a_sustained_correctable_pair_activates_one_composite() -> None:
    """One composite activation per sustained episode -- the 024 contract that
    reused components keep one composite incident (``reused_components``) and
    what ``independent_xid``'s single-incident chain check expects. Live, the
    held pair (both counters +1 on every scrape) minted a WARNING composite on
    the first replayed scrape (batch 07:54:39.12Z) and a CRITICAL one on the
    third (07:55:09.24Z), 30 s apart and before the kernel XID was injected.
    """

    from datetime import datetime, timedelta, timezone

    from gpu_fault.gpu_metrics import (
        GpuMetricBatch,
        GpuMetricSample,
        GpuMetricSource,
        GpuMetricsService,
    )

    start = datetime(2026, 10, 26, 7, 54, 9, tzinfo=timezone.utc)
    service = GpuMetricsService()
    activations = []
    for tick in range(6):
        outcome = service.ingest(
            GpuMetricBatch(
                batch_id=f"b{tick}",
                cluster_id="gpu-eks-cluster",
                node_id="ip-10-0-43-5.us-west-2.compute.internal",
                observed_at=start + timedelta(seconds=15 * tick),
                source=GpuMetricSource.DCGM_EXPORTER,
                samples=[
                    GpuMetricSample(
                        metric_name=f"DCGM_{name.upper()}",
                        canonical_name=name,
                        value=tick,
                        gpu_index="1",
                        gpu_uuid=G1,
                        pci_bdf="0000:01:00.0",
                    )
                    for name in (PRIMARY, "row_remap_correctable_total")
                ],
            )
        )
        activations.append(
            [
                (finding.correlation_rule_id, str(finding.severity.value))
                for finding in outcome.new_composite_findings
            ]
        )
    assert activations == [[], [(comp.CORRECTABLE_RULE, "WARNING")], [], [], [], []]

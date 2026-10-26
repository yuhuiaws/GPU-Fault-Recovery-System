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
from scripts.e2e.regional.host_probe_fixture import HostProbeMissingResponseError
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


def _ctx(
    tmp_path: Path,
    variant: str,
    *,
    probe: Any,
    cpu: Any,
    node: dict[str, Any] | None = None,
):
    spec = SPECS[variant]
    regional = FakeRegional(cpu_python=cpu, node_snapshot=node)
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
    # The plan is bounded to the one scrape that mints and the chain must end in
    # RECOVERED, never in the RESTART_NODE escalation a held flag provoked live.
    assert "exactly the one scrape" in spec.injection
    assert "no RESTART_NODE escalation" in spec.expected
    assert "on the records" in spec.expected
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
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    incidents: list,
    workflows: list,
    *,
    gpu_state: dict[str, Any] | None = None,
    probe: FakeProbe | None = None,
    node: dict[str, Any] | None = None,
) -> dict[str, Any]:
    restored: list[list[str]] = []
    monkeypatch.setattr(
        comp,
        "_validated_restore",
        lambda ctx, items: restored.append([str(i["incident_id"]) for i in items])
        or [],
    )
    probe = probe or dcgm_probe(
        audit=audit_records({f"{G0}/ecc_dbe_volatile_total": 1.0})
    )
    ctx = _ctx(
        tmp_path,
        "wrong_scope_source",
        probe=probe,
        cpu=gpu_cpu_python(
            activity=_activity(incidents, workflows),
            gpu_state=gpu_state or {"active_findings": []},
        ),
        node=node,
    )
    out = comp.wrong_scope_source(ctx)
    out["_restored"] = restored
    out["_probe"] = probe
    out["_ctx"] = ctx
    return out


def _exposition(dbe: float = 0, pending: float = 0) -> bytes:
    """A two-GPU scrape of the two replayed fields, in the exporter's format."""

    lines = []
    for index, uuid in ((0, G0), (1, G1)):
        labels = (
            f'gpu="{index}",UUID="{uuid}",pci_bus_id="00000000:{index + 1:02d}:00.0",'
            'modelName="NVIDIA H200"'
        )
        lines.append(f"DCGM_FI_DEV_ECC_DBE_VOL_TOTAL{{{labels}}} {dbe}")
        lines.append(f"DCGM_FI_DEV_ROW_REMAP_PENDING{{{labels}}} {pending}")
    return ("\n".join(lines) + "\n").encode()


def _view(body: bytes) -> dict[tuple[str, str], float]:
    """What the product's own DCGM parser reads from a served scrape."""

    from gpu_fault.collectors.gpu.dcgm import parse_dcgm_samples

    return {
        (sample.gpu_uuid, sample.canonical_name): sample.value
        for sample in parse_dcgm_samples(body.decode())
    }


def test_wrong_scope_source_passes_when_no_composite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    error, isolation = _error_incident()
    repair, chain = _repair_incident()
    out = _wrong_scope(tmp_path, monkeypatch, [error, repair], [isolation, chain])
    assert out["verdict"] == "PASS", out["errors"]
    assert set(out["stages"]) == {
        "no_cross_gpu_composite",
        "scoped_incidents",
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
    # The fault pair is served for exactly the one scrape that mints it, in a
    # bounded step -- never held -- and nothing advances.
    assert (out["fault_reads"], out["oracle_gate"]) == (1, "journal")
    document = _installed_plan(out["_probe"])
    assert [
        (item["reads"], item.get("until_advance", False)) for item in document["steps"]
    ] == [(3, False), (1, False)]
    assert "advance" not in [call[0] for call in out["_probe"].calls]
    assert any("returned to the real readings by itself" in n for n in out["notes"]), (
        out["notes"]
    )


def test_the_fault_pair_mints_on_one_scrape_and_a_spare_would_land_after_the_restore(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Why the plan serves the pair for exactly one scrape (``MINTING_READS``).

    The product mints every record the variant judges on the batch that
    carries the pair; the repair chain's quiesce stops the collector and the
    exporter on both providers; the field tool consumes a read only on a
    scrape it could serve, so a spare fault read waits for the first scrape
    after the restore -- the batch the post-reset VALIDATE_GPU grades -- and a
    validation failed after a reset is the RESTART_NODE rung.
    """

    import io
    import threading
    from datetime import datetime, timedelta, timezone
    from urllib.error import HTTPError
    from urllib.request import Request, urlopen

    from gpu_fault.adapters.gpu_validation import TELEMETRY_READINESS_OPERATIONS
    from gpu_fault.gpu_metrics import (
        GpuMetricBatch,
        GpuMetricSample,
        GpuMetricSource,
        GpuMetricsService,
    )
    from gpu_fault.models import WorkflowOperation
    from gpu_fault.node_agent.common import (
        DEFAULT_QUIESCE_CONTAINERS,
        DEFAULT_QUIESCE_SERVICES,
    )
    from gpu_fault.orchestration.escalation import next_rung
    from tools import gpu_dcgm_field_replay as tool

    assert comp.MINTING_READS == 1
    node = "ip-10-0-43-5.us-west-2.compute.internal"
    start = datetime(2026, 10, 26, 14, 53, 59, tzinfo=timezone.utc)

    def sample(name: str, value: float, gpu: str, index: str) -> GpuMetricSample:
        return GpuMetricSample(
            metric_name=f"DCGM_{name.upper()}",
            canonical_name=name,
            value=value,
            gpu_index=index,
            gpu_uuid=gpu,
            pci_bdf=f"00000000:0{int(index) + 1}:00.0",
            labels={"modelName": "NVIDIA H200"},
        )

    def ingest(service: GpuMetricsService, tick: int, samples: list) -> Any:
        return service.ingest(
            GpuMetricBatch(
                batch_id=f"b{tick}",
                cluster_id="gpu-eks-cluster",
                node_id=node,
                observed_at=start + timedelta(seconds=15 * tick),
                source=GpuMetricSource.DCGM_EXPORTER,
                samples=samples,
            )
        )

    # One batch carrying the pair on two GPUs mints two per-GPU findings and no
    # composite (the live 14:54:03Z records, 4 s behind the first replayed
    # scrape); the next real reading clears both, nothing waits for a later one.
    service = GpuMetricsService()
    first = ingest(
        service,
        0,
        [
            sample("ecc_dbe_volatile_total", 1.0, G0, "0"),
            sample("row_remap_pending", 1.0, G1, "1"),
        ],
    )
    assert {
        (item.gpu_uuid, item.canonical_name): item.automatic_action
        for item in first.new_findings
    } == {
        (G0, "ecc_dbe_volatile_total"): "QUARANTINE",
        (G1, "row_remap_pending"): "RESET_GPU",
    }
    assert first.new_composite_findings == []
    second = ingest(
        service,
        1,
        [
            sample("ecc_dbe_volatile_total", 0.0, G0, "0"),
            sample("row_remap_pending", 0.0, G1, "1"),
        ],
    )
    assert second.new_findings == [] and service.findings("gpu-eks-cluster", node) == []
    # The same pair on one GPU composes on that first batch too (memory_failure).
    same = ingest(
        GpuMetricsService(),
        0,
        [
            sample("ecc_dbe_volatile_total", 1.0, G1, "1"),
            sample("row_remap_pending", 1.0, G1, "1"),
        ],
    )
    assert [item.correlation_rule_id for item in same.new_composite_findings] == [
        comp.MEMORY_RULE
    ]

    # The quiesce stops the collector and the exporter container on both
    # providers; the post-reset VALIDATE_GPU grades readings newer than the
    # reset; a validation failed after a reset is the reboot rung.
    assert "gpu-fault-metrics-collector" in DEFAULT_QUIESCE_SERVICES
    assert "gpu-fault-system/exporter" in DEFAULT_QUIESCE_CONTAINERS
    assert WorkflowOperation.RESET_GPU in TELEMETRY_READINESS_OPERATIONS
    assert (
        next_rung(WorkflowOperation.VALIDATE_GPU, {WorkflowOperation.RESET_GPU})
        is WorkflowOperation.RESTART_NODE
    )

    # The tool consumes a read only on a scrape it could serve: a two-read
    # fault step keeps its second read for the first scrape after the exporter
    # is back (the validation's batch); the one-read step is already done.
    exporter_down = [False]

    def open_upstream(*_args: Any, **_kwargs: Any) -> io.BytesIO:
        if exporter_down[0]:
            raise OSError("Connection refused")
        return io.BytesIO(_exposition())

    monkeypatch.setattr(tool, "open_upstream", open_upstream)
    pair = [
        {"op": "add", "field": "ecc_dbe_volatile_total", "gpus": [G0], "value": 1},
        {"op": "set", "field": "row_remap_pending", "gpus": [G1], "value": 1},
    ]

    def scrape_through(reads: int) -> tuple[list[Any], list[dict[str, Any]]]:
        replay = tool.Replay(
            tool.validate_plan(
                {
                    "profile": "composite",
                    "steps": [{"reads": 3, "ops": []}, {"reads": reads, "ops": pair}],
                }
            ),
            600,
        )
        audit = io.StringIO()
        server = tool.make_server("http://127.0.0.1:9400/metrics", 0, replay, audit)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = f"http://127.0.0.1:{server.server_address[1]}"
        pendings: list[Any] = []
        try:
            with urlopen(
                Request(
                    base + "/arm",
                    method="POST",
                    headers={"X-GFRS-Replay-Ack": tool.ACKNOWLEDGEMENT},
                ),
                timeout=5,
            ):
                pass
            for _ in range(4):  # three baseline scrapes and the minting one
                with urlopen(base + "/metrics", timeout=5) as response:
                    pendings.append(_view(response.read())[(G1, "row_remap_pending")])
            exporter_down[0] = True  # the quiesce: the exporter is gone
            with pytest.raises(HTTPError) as refused:
                urlopen(base + "/metrics", timeout=5)
            assert refused.value.code == 503
            assert replay.reads == 4, "a scrape the tool could not serve is no read"
            exporter_down[0] = False  # RESTORE_GPU_SERVICES: the first scrape after
            with urlopen(base + "/metrics", timeout=5) as response:
                pendings.append(_view(response.read())[(G1, "row_remap_pending")])
        finally:
            server.shutdown()
            server.server_close()
            thread.join(5)
        return pendings, [json.loads(line) for line in audit.getvalue().splitlines()]

    spare, journal = scrape_through(2)
    assert spare == [0.0, 0.0, 0.0, 1.0, 1.0], (
        "the spare read is served after the restore: the escalation"
    )
    assert [item.get("read") for item in journal if "error" in item] == [5]
    minting, _journal = scrape_through(comp.MINTING_READS)
    assert minting == [0.0, 0.0, 0.0, 1.0, 0.0], (
        "the one-read plan is done before the quiesce; the restore reads the device"
    )


def test_wrong_scope_source_closes_the_replay_after_the_repair_record_ended(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The close restarts the collector; inside the chain that would fight the
    # quiesce, so it comes after the settled record and before the restore.
    order: list[str] = []
    real_settled = comp.settled

    def settled(activity: dict[str, Any], incidents: Any) -> bool:
        value = real_settled(activity, incidents)
        if value:
            order.append("settled")
        return value

    def close_run(arguments: tuple[str, ...]) -> dict[str, Any]:
        order.append("close-run")
        return dcgm_probe().script["close-run"]

    monkeypatch.setattr(comp, "settled", settled)
    probe = dcgm_probe(
        audit=audit_records({f"{G0}/ecc_dbe_volatile_total": 1.0}),
        extra={"close-run": close_run},
    )
    error, isolation = _error_incident()
    repair, chain = _repair_incident()
    out = _wrong_scope(
        tmp_path, monkeypatch, [error, repair], [isolation, chain], probe=probe
    )
    assert out["verdict"] == "PASS", out["errors"]
    assert order[0] == "settled" and order[-1] == "close-run", order
    assert order.count("close-run") == 1
    assert out["_restored"] == [["inc"]]
    assert any("closed after the chain" in note for note in out["notes"]), out["notes"]
    assert "composite-window-hold.json" not in {
        p.name for p in out["_ctx"].case_dir.iterdir()
    }


def test_wrong_scope_source_plan_returns_to_the_real_readings_by_itself(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The installed plan, run through the field tool over a real-shaped
    scrape: three pass-through reads, one read of the pair, then the real
    readings for good -- what the reset chain's post-reset VALIDATE_GPU reads
    (live 2026-10-26 14:54Z the held plan still said row_remap_pending=1 there
    and the product escalated to RESTART_NODE)."""

    from tools import gpu_dcgm_field_replay as tool

    error, isolation = _error_incident()
    repair, chain = _repair_incident()
    out = _wrong_scope(tmp_path, monkeypatch, [error, repair], [isolation, chain])
    replay = tool.Replay(tool.validate_plan(_installed_plan(out["_probe"])), 600)
    replay.arm()
    body = _exposition()
    served = []
    for _ in range(8):
        output, receipt = replay.render(body)
        replay.reads += 1
        view = _view(output)
        served.append(
            (
                output == body,
                view[(G0, "ecc_dbe_volatile_total")],
                view[(G1, "row_remap_pending")],
                receipt.get("step"),
            )
        )
    assert [item[1:] for item in served] == [
        (0.0, 0.0, 0),
        (0.0, 0.0, 0),
        (0.0, 0.0, 0),
        (1.0, 1.0, 1),
        (0.0, 0.0, None),
        (0.0, 0.0, None),
        (0.0, 0.0, None),
        (0.0, 0.0, None),
    ]
    assert [item[0] for item in served[4:]] == [True] * 4, (
        "after the plan the scrape is the exporter's own, byte for byte"
    )
    assert replay.active() is None and not replay.disarmed
    with pytest.raises(ValueError, match="not on a step"):
        replay.advance()


def _quiesced_probe(
    records: list[dict[str, Any]], *, node: dict[str, bool]
) -> tuple[FakeProbe, dict[str, int]]:
    """A probe whose node stops answering while ``node["down"]``: the audit
    read fails, and the first ``close-run`` fails and brings the node back
    (the chain's RESTORE_GPU_SERVICES restarted kubelet before the next poll)."""

    closes = {"count": 0}

    def audit_read(arguments: tuple[str, ...]) -> dict[str, Any]:
        if node["down"]:
            raise HostProbeMissingResponseError(
                "host probe response is missing (exit 1)"
            )
        offset = int(arguments[arguments.index("--offset") + 1])
        chunk = records[offset:]
        return {
            "records": list(chunk),
            "next_offset": offset + len(chunk),
            "total": len(records),
        }

    def close_run(arguments: tuple[str, ...]) -> dict[str, Any]:
        closes["count"] += 1
        if node["down"]:
            node["down"] = False
            raise HostProbeMissingResponseError(
                "host probe response is missing (exit 1)"
            )
        return dcgm_probe().script["close-run"]

    return dcgm_probe(extra={"audit-read": audit_read, "close-run": close_run}), closes


def test_wrong_scope_source_gates_on_the_records_while_the_node_is_quiesced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Live 2026-10-26 14:54Z (HyperPod): the repair chain's QUIESCE stopped
    kubelet ~20 s after the mint; the oracle gate's audit-read and the close
    failed "response is missing". The records now gate the replay, the close
    waits for the node and reads the journal before it closes."""

    error, isolation = _error_incident()
    repair, chain = _repair_incident()
    records = audit_records(
        {f"{G0}/ecc_dbe_volatile_total": 1.0, f"{G1}/row_remap_pending": 1.0}
    )
    node_state = {"down": True}
    probe, closes = _quiesced_probe(records, node=node_state)
    node = {**FakeRegional().node_snapshot(""), "boot_id": survey()["boot_id"]}
    out = _wrong_scope(
        tmp_path,
        monkeypatch,
        [error, repair],
        [isolation, chain],
        probe=probe,
        node=node,
    )
    assert out["verdict"] == "PASS", out["errors"]
    assert out["oracle_gate"] == "records"
    ctx = out["_ctx"]
    deferred = json.loads((ctx.case_dir / "replay-gate-deferred.json").read_text())
    assert deferred["keys"] == [f"{G0}/ecc_dbe_volatile_total"]
    assert deferred["probe_error"].startswith(
        "HostProbeMissingResponseError: dcgm audit-read: host probe response is missing"
    ), deferred
    close = json.loads((ctx.case_dir / "dcgm-replay-close.json").read_text())
    assert close["close_error"].startswith(
        "HostProbeMissingResponseError: dcgm close-run: host probe response is missing"
    ), close
    assert close["close"]["clean"] is True and "judged_by" not in close["close"]
    assert "rebooted" not in close["close"], "the same boot id: no reboot was read"
    assert closes["count"] == 2
    assert close["close_after_probe_loss"][-1]["close_run"] == "ok"
    assert "audit_error" not in close, "the journal was read once the node answered"
    oracle = [
        json.loads(line)
        for line in (ctx.case_dir / "oracle.jsonl").read_text().splitlines()
    ]
    assert [item["served"] for item in oracle] == [records[0]["served"]]
    assert ctx.cleanup.pending() == []


def test_wrong_scope_source_keeps_a_probe_failure_the_records_do_not_explain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # No incident on either replayed reading: the lost probe is a fixture
    # failure, raised as before, and the cleanup stack still owns the close.
    monkeypatch.setattr(comp, "_validated_restore", lambda ctx, items: [])

    def audit_read(arguments: tuple[str, ...]) -> dict[str, Any]:
        raise HostProbeMissingResponseError("host probe response is missing (exit 1)")

    probe = dcgm_probe(extra={"audit-read": audit_read})
    ctx = _ctx(
        tmp_path,
        "wrong_scope_source",
        probe=probe,
        cpu=gpu_cpu_python(
            activity=_activity([], []), gpu_state={"active_findings": []}
        ),
    )
    with pytest.raises(
        HostProbeMissingResponseError, match="^dcgm audit-read: host probe response"
    ):
        comp.wrong_scope_source(ctx)
    assert ctx.cleanup.pending() == ["dcgm-replay"]
    assert not (ctx.case_dir / "replay-gate-deferred.json").exists(), (
        "nothing stood in for the journal"
    )
    assert "close-run" not in [call[0] for call in probe.calls]


def test_wrong_scope_source_fails_when_a_finding_spans_the_gpus(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The incident list alone would pass: the composite is in the finding
    # store, which the scoping oracle reads too.
    error, isolation = _error_incident()
    repair, chain = _repair_incident()
    finding = {
        "finding_id": f"b-{G1}-composite-GPU_MEMORY_DEGRADATION",
        "correlation_rule_id": "GPU_MEMORY_DEGRADATION",
        "component_metrics": ["row_remap_pending", "ecc_dbe_volatile_total"],
        "component_finding_ids": [f"b-{G0}-ecc_dbe_volatile_total"],
        "gpu_uuid": G1,
        "observed_at": "t",
    }
    out = _wrong_scope(
        tmp_path,
        monkeypatch,
        [error, repair],
        [isolation, chain],
        gpu_state={"active_findings": [], "findings": [finding]},
    )
    assert out["verdict"] == "FAIL"
    assert out["stages"]["no_cross_gpu_composite"] == []
    assert out["stages"]["scoped_incidents"] == [
        f"GPU_MEMORY_DEGRADATION finding b-{G1}-composite-GPU_MEMORY_DEGRADATION "
        f"formed on {G1} from ['ecc_dbe_volatile_total', 'row_remap_pending']"
    ]


def test_scoping_errors_name_each_record_out_of_scope() -> None:
    error, isolation = _error_incident()
    repair, chain = _repair_incident()
    kwargs: dict[str, Any] = {
        "error_gpu": G0,
        "repair_gpu": G1,
        "companion": "row_remap_pending",
    }
    clean = _activity([error, repair], [isolation, chain])
    assert comp.scoping_errors(clean, {"findings": []}, **kwargs) == []
    stray = gpu_incident(
        "inc-s", f"b-{G1}-ecc_dbe_volatile_total", action="QUARANTINE", workflow="wf-s"
    )
    swapped = gpu_incident(
        "inc-t", f"b-{G0}-row_remap_pending", action="RESET_GPU", workflow="wf-t"
    )
    assert comp.scoping_errors(_activity([repair, stray, swapped]), {}, **kwargs) == [
        f"the replayed ecc_dbe_volatile_total on {G0} minted no incident of its own",
        f"ecc_dbe_volatile_total minted incident inc-s on {G1}, which did not replay it",
        f"row_remap_pending minted incident inc-t on {G0}, which did not replay it",
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


def test_a_sustained_correctable_pair_activates_one_composite() -> None:
    """One composite activation per sustained episode -- the 024 contract that
    reused components keep one composite incident (``reused_components``) and
    what ``independent_xid``'s single-incident chain check expects. Live, the
    held pair (both counters +1 on every scrape) minted a WARNING composite on
    the first replayed scrape (batch 07:54:39.12Z) and a CRITICAL one on the
    third (07:55:09.24Z), 30 s apart and before the kernel XID was injected
    (2026-10-26 08:07Z, eks ip-10-0-43-5, R13g). The product now keeps the
    episode's finding id and reports the third sample's CRITICAL as an update
    of the open composite (``updated_composite_findings``), not an activation.
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
    regrades = []
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
        regrades.append(
            [
                (finding.finding_id, str(finding.severity.value))
                for finding in outcome.updated_composite_findings
            ]
        )
    assert activations == [[], [(comp.CORRECTABLE_RULE, "WARNING")], [], [], [], []]
    episode = f"b1-{G1}-composite-{comp.CORRECTABLE_RULE}"
    assert regrades == [[], [], [], [(episode, "CRITICAL")], [], []]

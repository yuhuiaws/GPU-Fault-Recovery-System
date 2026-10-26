"""GF-GENERAL-LIVE-020/022 injector/verdict contract over scripted records."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import general_live_gpu_links as links
from scripts.e2e.regional.general_live_fixture import VariantNotRun
from scripts.e2e.regional.general_live_gpu_replay import (
    NVLINK_FAMILY_MEMBERS,
    GpuSurvey,
)
from tests.regional._general_live_fakes import FakeRegional, make_context, make_settings
from tests.regional._general_live_gpu_fakes import (
    audit_records,
    dcgm_probe,
    gpu_cpu_python,
    gpu_incident,
    gpu_workflow,
)

SPECS = {spec.variant: spec for spec in links.VARIANTS}
#: 020 and 022 both have a ``counter_reset``; ``SPECS`` keeps 022's.
SPECS_020 = {
    spec.variant: spec for spec in links.VARIANTS if spec.case_id == links.CASE_020
}
G0 = "GPU-00000000-0000-4000-8000-000000000000"
G1 = "GPU-11111111-1111-4111-8111-111111111111"


def _ctx(
    tmp_path: Path,
    variant: str,
    *,
    probe: Any,
    cpu: Any,
    approvals: tuple[str, ...] = (),
    case: str = "",
):
    spec = (SPECS_020 if case == links.CASE_020 else SPECS)[variant]
    regional = FakeRegional(cpu_python=cpu)
    settings = make_settings(spec, node=regional.node_snapshot("")["name"])
    object.__setattr__(settings, "approvals", approvals)
    ctx = make_context(
        spec, tmp_path=tmp_path, regional=regional, probes={}, settings=settings
    )
    ctx.probes[f"dcgm@{ctx.settings.node}"] = probe
    return ctx


def _activity(incidents: list, workflows: list = (), **extra: Any) -> dict[str, Any]:
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
        **extra,
    }


def test_registry_covers_020_and_022() -> None:
    from scripts.e2e.regional import general_live_registry as reg

    rows = reg.coverage_rows()
    assert {s.variant for s in links.VARIANTS if s.case_id.endswith("020")} == set(
        rows["GF-GENERAL-LIVE-020"]
    )
    assert {s.variant for s in links.VARIANTS if s.case_id.endswith("022")} == set(
        rows["GF-GENERAL-LIVE-022"]
    )


def test_crc_is_a_clean_nvlink_diagnostic(tmp_path: Path) -> None:
    canonical = "nvlink_crc_flit_error_total"
    inc = gpu_incident("inc", f"b-{G1}-{canonical}", state="RECOVERED")
    wf = gpu_workflow(
        "wf-1",
        "inc",
        ["FREEZE_EVIDENCE", "VALIDATE_GPU"],
        status="FAILED",
        executions=[
            {
                "operation": "VALIDATE_GPU",
                "status": "FAILED",
                "details": {"host_diagnostic_failure": True},
            }
        ],
    )
    cpu = gpu_cpu_python(
        activity=_activity([inc], [wf]),
        gpu_state={"active_findings": [], "batches": [{"observed_at": "t"}]},
    )
    out = links.crc(
        _ctx(
            tmp_path,
            "crc",
            probe=dcgm_probe(audit=audit_records({f"{G1}/{canonical}": 1.0})),
            cpu=cpu,
        )
    )
    assert out["verdict"] == "PASS", out["errors"]


def test_at_rate_limit_passes_when_no_finding(tmp_path: Path) -> None:
    cpu = gpu_cpu_python(
        activity=_activity([]),
        gpu_state={"active_findings": [], "batches": [{"observed_at": "t"}]},
    )
    out = links.at_rate_limit(
        _ctx(
            tmp_path,
            "at_rate_limit",
            probe=dcgm_probe(audit=audit_records({f"{G1}/pcie_replay_total": 2.0})),
            cpu=cpu,
        )
    )
    assert out["verdict"] == "PASS", out["errors"]


def test_at_rate_limit_fails_if_a_finding_appears(tmp_path: Path) -> None:
    inc = gpu_incident("inc", f"b-{G1}-pcie_replay_total")
    wf = gpu_workflow("wf-1", "inc", ["FREEZE_EVIDENCE", "VALIDATE_GPU"])
    cpu = gpu_cpu_python(
        activity=_activity([inc], [wf]),
        gpu_state={"active_findings": [], "batches": [{"observed_at": "t"}]},
    )
    out = links.at_rate_limit(
        _ctx(
            tmp_path,
            "at_rate_limit",
            probe=dcgm_probe(audit=audit_records({f"{G1}/pcie_replay_total": 2.0})),
            cpu=cpu,
        )
    )
    assert out["verdict"] == "FAIL" and any("limit" in e for e in out["errors"])


def test_same_gpu_xid_requires_the_composite(tmp_path: Path) -> None:
    comp = gpu_incident(
        "inc-c",
        f"b-{G1}-composite-PCIE_XID_LINK_FAILURE",
        action="DRAIN",
        workflow="wf-c",
    )
    wf = gpu_workflow(
        "wf-c", "inc-c", ["FREEZE_EVIDENCE", "MARK_UNSCHEDULABLE", "QUARANTINE"]
    )
    cpu = gpu_cpu_python(
        activity=_activity([comp], [wf]),
        gpu_state={"active_findings": [], "batches": [{"observed_at": "t"}]},
    )
    out = links.same_gpu_xid(
        _ctx(
            tmp_path,
            "same_gpu_xid",
            probe=dcgm_probe(audit=audit_records({f"{G1}/pcie_replay_total": 9.0})),
            cpu=cpu,
            approvals=("DCGM_XID_GAUGE",),
        )
    )
    assert out["verdict"] == "PASS", out["errors"]


def test_wrong_gpu_xid_fails_if_the_composite_forms(tmp_path: Path) -> None:
    comp = gpu_incident(
        "inc-c",
        f"b-{G1}-composite-PCIE_XID_LINK_FAILURE",
        action="DRAIN",
        workflow="wf-c",
    )
    wf = gpu_workflow("wf-c", "inc-c", ["MARK_UNSCHEDULABLE", "QUARANTINE"])
    cpu = gpu_cpu_python(
        activity=_activity([comp], [wf]),
        gpu_state={"active_findings": [], "batches": [{"observed_at": "t"}]},
    )
    out = links.wrong_gpu_xid(
        _ctx(
            tmp_path,
            "wrong_gpu_xid",
            probe=dcgm_probe(audit=audit_records({f"{G1}/pcie_replay_total": 9.0})),
            cpu=cpu,
            approvals=("DCGM_XID_GAUGE",),
        )
    )
    assert out["verdict"] == "FAIL" and any("different GPU" in e for e in out["errors"])


def test_expired_xid_fails_while_the_composite_joins(tmp_path: Path) -> None:
    # Batch 160 dates the gauge by its change, so the variant expects PASS live;
    # a composite that still joined an expired XID makes the runner FAIL it.
    spec = SPECS["expired_xid"]
    assert spec.expected_outcome == "PASS" and not spec.product_gap, spec
    comp = gpu_incident(
        "inc-c",
        f"b-{G1}-composite-PCIE_XID_LINK_FAILURE",
        action="DRAIN",
        workflow="wf-c",
    )
    wf = gpu_workflow("wf-c", "inc-c", ["MARK_UNSCHEDULABLE", "QUARANTINE"])
    cpu = gpu_cpu_python(
        activity=_activity([comp], [wf]),
        gpu_state={"active_findings": [], "batches": [{"observed_at": "t"}]},
    )
    out = links.expired_xid(
        _ctx(
            tmp_path,
            "expired_xid",
            probe=dcgm_probe(audit=audit_records({f"{G1}/pcie_replay_total": 9.0})),
            cpu=cpu,
            approvals=("DCGM_XID_GAUGE",),
        )
    )
    assert out["verdict"] == "FAIL", out
    assert "expired" in out["errors"][0]


def _xid_decision(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "event_id": "kernel-log-kmsg-boot-900000064-xid-137-cluster-abc",
        "xid": links.INDEPENDENT_XID,
        "gpu_uuid": G1,
        "event_source": "KERNEL_LOG",
        "decision": True,
        "disposition": "MONITOR_ONLY",
        "official_action": "IGNORE",
        "action": "NO_ACTION",
        "safety_action": None,
        "requires_operator": False,
        "incident_id": "inc-xid-137",
        "incident_state": "RECOVERED",
        "workflow_request_id": None,
        "duplicate": False,
        "correlated_event_id": None,
        "reasons": [],
    }
    base.update(overrides)
    return base


def _independent_ctx(
    tmp_path: Path,
    decisions: list[dict[str, Any]],
    *,
    counter_incidents: int = 1,
    survey_value: dict[str, Any] | None = None,
    canonical: str = "nvlink_crc_flit_error_total",
) -> Any:
    from scripts.e2e.regional.general_live_kernel_xid import XID_DECISION_PROBE
    from tests.regional._general_live_fakes import FakeProbe

    incidents = [
        gpu_incident(
            f"inc-{index}",
            f"b{index}-{G1}-{canonical}",
            state="RECOVERED",
            workflow="wf-1",
        )
        for index in range(counter_incidents)
    ]
    wf = gpu_workflow(
        "wf-1",
        "inc-0",
        ["FREEZE_EVIDENCE", "VALIDATE_GPU"],
        status="FAILED",
        executions=[
            {
                "operation": "VALIDATE_GPU",
                "status": "FAILED",
                "details": {"host_diagnostic_failure": True},
            }
        ],
    )
    records = gpu_cpu_python(
        activity=_activity(incidents, [wf]),
        gpu_state={"active_findings": [], "batches": [{"observed_at": "t"}]},
    )

    def cpu(script: str, *arguments: str) -> dict[str, Any]:
        if script == XID_DECISION_PROBE:
            return {"decisions": decisions}
        return records(script, *arguments)

    ctx = _ctx(
        tmp_path,
        "independent_fatal_event",
        probe=dcgm_probe(survey_value, audit=audit_records({f"{G1}/{canonical}": 1.0})),
        cpu=cpu,
        approvals=("KERNEL_SOURCE_REPLAY",),
    )
    ctx.probes[f"kernel-replay-kmsg@{ctx.settings.node}"] = FakeProbe(
        {
            "install-records": {"records": 1},
            "open-window": {"after": {"ActiveState": "active"}},
            "arm": {"armed": True, "lines_written": 6},
            "status": {"injected": {"mode": "kmsg", "fixture_lines": 6}},
            "close-window": {"clean": True},
        }
    )
    return ctx


def test_independent_fatal_event_keeps_its_catalog_decision_beside_the_counter(
    tmp_path: Path,
) -> None:
    spec = SPECS["independent_fatal_event"]
    assert spec.approvals == ("KERNEL_SOURCE_REPLAY",) and spec.lanes == ("eks", "hp")
    ctx = _independent_ctx(tmp_path, [_xid_decision()])
    out = links.independent_fatal_event(ctx)
    assert out["verdict"] == "PASS", out["errors"]
    assert out["xid"] == 137 and out["gpu_uuid"] == G1
    assert out["pci_bdf"] == "0000:02:00", "the kernel line names the target GPU's BDF"
    kernel = ctx.probes[f"kernel-replay-kmsg@{ctx.settings.node}"]
    verbs = [call[0] for call in kernel.calls]
    assert verbs == ["install-records", "open-window", "arm", "status", "close-window"]
    dcgm = [call[0] for call in ctx.probes[f"dcgm@{ctx.settings.node}"].calls]
    # since the 019 cleanup fix the replay is released by one probe call that restores the endpoint, stops the
    # field tool and removes the run directory in order (close-run), instead of restore-collector + stop-proxy
    assert "close-run" in dcgm, (
        "the counter is released through the fixture's close-run"
    )


def test_independent_fatal_event_fails_when_the_counter_absorbs_the_xid(
    tmp_path: Path,
) -> None:
    out = links.independent_fatal_event(
        _independent_ctx(tmp_path, [_xid_decision(incident_id="inc-0")])
    )
    assert out["verdict"] == "FAIL"
    assert any("absorbed" in error for error in out["errors"]), out["errors"]


def test_independent_fatal_event_fails_when_the_xid_earns_an_action(
    tmp_path: Path,
) -> None:
    out = links.independent_fatal_event(
        _independent_ctx(
            tmp_path,
            [
                _xid_decision(
                    disposition="EXECUTABLE",
                    action="RESET_GPU",
                    incident_state="ACTION_PENDING",
                )
            ],
        )
    )
    assert out["verdict"] == "FAIL"
    assert any("RESET_GPU" in error for error in out["errors"]), out["errors"]


def test_independent_fatal_event_fails_when_the_counter_chain_is_perturbed(
    tmp_path: Path,
) -> None:
    out = links.independent_fatal_event(
        _independent_ctx(tmp_path, [_xid_decision()], counter_incidents=2)
    )
    assert out["verdict"] == "FAIL"
    assert any("expected one" in error for error in out["errors"]), out["errors"]


# --------------------------------------------------------------------------- #
# NVLink family targeting: the counter the node really reports
# --------------------------------------------------------------------------- #
#: Each NVLink family's canonical counter and the exporter sample name behind
#: it, per exporter generation. LEGACY is the per-class counter of NVLink3 and
#: earlier parts; HOPPER is the live 2026-10-26 p5en H200 survey shape (DCGM
#: 4.6/4.7 publishes only the DL aggregates, as ``_total`` counters).
LEGACY = {
    "nvlink_crc": (
        "nvlink_crc_flit_error_total",
        "DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_TOTAL_total",
    ),
    "nvlink_replay": (
        "nvlink_replay_error_total",
        "DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_TOTAL_total",
    ),
    "nvlink_recovery": (
        "nvlink_recovery_error_total",
        "DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_TOTAL_total",
    ),
}
HOPPER = {
    "nvlink_crc": (
        "nvlink_crc_aggregate_error_total",
        "DCGM_FI_DEV_NVLINK_ERROR_DL_CRC_total",
    ),
    "nvlink_replay": (
        "nvlink_replay_aggregate_error_total",
        "DCGM_FI_DEV_NVLINK_ERROR_DL_REPLAY_total",
    ),
    "nvlink_recovery": (
        "nvlink_recovery_aggregate_error_total",
        "DCGM_FI_DEV_NVLINK_ERROR_DL_RECOVERY_total",
    ),
}
OTHER_FIELDS = (
    "ecc_sbe_volatile_total",
    "ecc_dbe_volatile_total",
    "pcie_replay_total",
    "row_remap_failure",
    "fabric_manager_status",
)
#: The NVLink family each 020 variant drives (``links.NVLINK`` by variant).
FAMILY_OF = {
    "crc": "nvlink_crc",
    "replay": "nvlink_replay",
    "recovery_counter": "nvlink_recovery",
    "multiple_gpus": "nvlink_crc",
    "counter_reset": "nvlink_crc",
    "stale_sample": "nvlink_crc",
    "independent_fatal_event": "nvlink_crc",
    "recovered": "nvlink_replay",
}


def _survey(
    *counters: tuple[str, str],
    gpus: tuple[str, ...] = (G0, G1),
    only_on: dict[str, tuple[str, ...]] | None = None,
) -> dict[str, Any]:
    """A survey whose NVLink counters arrive under the given exporter names.

    ``only_on`` restricts a canonical counter to some GPUs (the others neither
    value nor name it), as a part-populated exporter would.
    """

    from tests.regional._general_live_gpu_fakes import survey

    value = survey(
        gpus=gpus, fields=(*OTHER_FIELDS, *(canonical for canonical, _ in counters))
    )
    for gpu, entry in value["exporter"]["gpus"].items():
        for canonical, metric in counters:
            if gpu not in (only_on or {}).get(canonical, gpus):
                del entry["values"][canonical]
                del entry["metric_names"][canonical]
                continue
            entry["metric_names"][canonical] = [metric]
    return value


def _flag(arguments: tuple[str, ...], name: str) -> str:
    for index, value in enumerate(arguments):
        if value == name and index + 1 < len(arguments):
            return arguments[index + 1]
    raise AssertionError(f"{name} missing from {arguments}")


def _installed_plan(probe: Any) -> dict[str, Any]:
    """The field plan the controller installed on the node (``install-source``)."""

    import base64

    for call in probe.calls:
        if call[0] == "install-source" and _flag(call, "--name") == "plan.json":
            return json.loads(base64.b64decode(_flag(call, "--b64")))
    raise AssertionError("no plan.json was installed")


def _plan_targets(plan: dict[str, Any]) -> set[tuple[str, str]]:
    return {
        (operation["field"], gpu)
        for item in plan["steps"]
        for operation in item["ops"]
        if "field" in operation
        for gpu in operation["gpus"]
    }


def test_nvlink_family_members_are_the_product_groups() -> None:
    from gpu_fault.dcgm_fields import DCGM_FABRIC_CANONICAL_GROUPS

    assert {
        family: set(members) for family, members in NVLINK_FAMILY_MEMBERS.items()
    } == dict(DCGM_FABRIC_CANONICAL_GROUPS)
    assert set(links.NVLINK.values()) == set(NVLINK_FAMILY_MEMBERS)
    assert set(SPECS_020) == set(FAMILY_OF)
    for family, members in NVLINK_FAMILY_MEMBERS.items():
        assert members[0] == LEGACY[family][0], "the legacy per-class counter first"
        assert members[1] == HOPPER[family][0], "then the DL aggregate"


def test_nvlink_counter_follows_the_members_the_node_reports() -> None:
    hopper = GpuSurvey(_survey(*HOPPER.values()))
    counter = hopper.nvlink_counter("nvlink_crc")
    assert counter.canonical == "nvlink_crc_aggregate_error_total"
    assert counter.exporter_metrics == ("DCGM_FI_DEV_NVLINK_ERROR_DL_CRC_total",)
    assert counter.gpus == (G0, G1)
    assert "DL aggregate" in counter.note()
    assert "DCGM_FI_DEV_NVLINK_ERROR_DL_CRC_total" in counter.note()
    assert counter.evidence() == {
        "nvlink_family": "nvlink_crc",
        "canonical_name": "nvlink_crc_aggregate_error_total",
        "exporter_metrics": ["DCGM_FI_DEV_NVLINK_ERROR_DL_CRC_total"],
    }
    assert hopper.nvlink_counter("nvlink_replay").canonical == (
        "nvlink_replay_aggregate_error_total"
    )
    assert hopper.nvlink_counter("nvlink_recovery").canonical == (
        "nvlink_recovery_aggregate_error_total"
    )
    legacy = GpuSurvey(_survey(*LEGACY.values()))
    recovery = legacy.nvlink_counter("nvlink_recovery")
    assert recovery.canonical == "nvlink_recovery_error_total"
    assert recovery.exporter_metrics == (
        "DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_TOTAL_total",
    )
    assert "legacy per-class" in recovery.note()
    # A part publishing both generations keeps the historical per-class target.
    both = GpuSurvey(_survey(*LEGACY.values(), *HOPPER.values()))
    assert both.nvlink_counter("nvlink_crc").canonical == "nvlink_crc_flit_error_total"
    # ... unless more GPUs report the other member: multiple_gpus needs two.
    partial = GpuSurvey(
        _survey(
            LEGACY["nvlink_crc"],
            HOPPER["nvlink_crc"],
            only_on={"nvlink_crc_flit_error_total": (G1,)},
        )
    )
    counter = partial.nvlink_counter("nvlink_crc")
    assert counter.canonical == "nvlink_crc_aggregate_error_total"
    assert counter.gpus == (G0, G1)


def test_nvlink_counter_names_the_whole_family_when_absent() -> None:
    none = GpuSurvey(_survey())
    with pytest.raises(VariantNotRun) as absent:
        none.nvlink_counter("nvlink_crc")
    assert absent.value.reason == "CAPABILITY_ABSENT"
    for member in NVLINK_FAMILY_MEMBERS["nvlink_crc"]:
        assert member in str(absent.value), absent.value
    assert "NVLink crc counter" in str(absent.value)
    with pytest.raises(KeyError):
        none.nvlink_counter("nvlink_bogus")


def _pass_records(
    variant: str, canonical: str
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    """Product records and a replay journal under which ``variant`` passes."""

    clean_state = {"active_findings": [], "batches": [{"observed_at": "t"}]}
    diagnostic = gpu_workflow(
        "wf-1",
        "inc",
        ["FREEZE_EVIDENCE", "VALIDATE_GPU"],
        status="FAILED",
        executions=[
            {
                "operation": "VALIDATE_GPU",
                "status": "FAILED",
                "details": {"host_diagnostic_failure": True},
            }
        ],
    )
    served = audit_records({f"{G1}/{canonical}": 1.0})
    if variant in {"crc", "replay", "recovery_counter", "recovered"}:
        inc = gpu_incident("inc", f"b-{G1}-{canonical}", state="RECOVERED")
        return _activity([inc], [diagnostic]), clean_state, served
    if variant == "multiple_gpus":
        comp = gpu_incident(
            "inc-c",
            "b-node-composite-MULTI_GPU_NVLINK_FABRIC_FAILURE",
            state="RECOVERED",
            workflow="wf-c",
        )
        wf = gpu_workflow("wf-c", "inc-c", ["FREEZE_EVIDENCE"], status="SUCCEEDED")
        return (
            _activity([comp], [wf]),
            clean_state,
            audit_records({f"{G0}/{canonical}": 1.0, f"{G1}/{canonical}": 1.0}),
        )
    if variant == "counter_reset":
        # The raise mints (every NVLink increase is the WARNING); the drop is a
        # counter-reset edge that keeps that finding active and re-arms the
        # baseline. ``test_general_live_gpu_links_reset.py`` grades the rest.
        at = "2026-10-26T13:{:02d}:{:02d}+00:00".format
        finding = {
            "finding_id": f"b-{G1}-{canonical}",
            "canonical_name": canonical,
            "severity": "WARNING",
            "value": 1,
            "delta": 1,
            "rate_per_minute": None,
            "gpu_uuid": G1,
            "observed_at": at(24, 45),
        }

        def batch(value: float, when: str, reason: str) -> dict[str, Any]:
            return {
                "batch_id": f"b-{when}",
                "observed_at": when,
                "edge_filter_reasons": [reason],
                "collection_errors": [],
                "samples": [
                    {"gpu": G1, "name": canonical, "value": value, "labels": {}}
                ],
            }

        state = {
            "latest": [{"gpu": G1, "name": canonical, "value": 0, "labels": {}}],
            "active_findings": [finding],
            "findings": [finding],
            "batches": [
                batch(0, at(23, 45), "initial-baseline"),
                batch(1, at(24, 45), "counter-increased"),
                batch(2, at(25, 0), "counter-increased"),
                batch(0, at(25, 15), "counter-reset"),
            ],
        }
        return _activity([gpu_incident("inc", finding["finding_id"])]), state, served
    # stale_sample: a quiet product is the expected outcome.
    return _activity([]), clean_state, served


def _run_variant(
    tmp_path: Path, variant: str, survey_value: dict[str, Any], canonical: str
) -> tuple[dict[str, Any], Any]:
    if variant == "independent_fatal_event":
        ctx = _independent_ctx(
            tmp_path, [_xid_decision()], survey_value=survey_value, canonical=canonical
        )
    else:
        activity, gpu_state, served = _pass_records(variant, canonical)
        ctx = _ctx(
            tmp_path,
            variant,
            probe=dcgm_probe(survey_value, audit=served),
            cpu=gpu_cpu_python(activity=activity, gpu_state=gpu_state),
            case=links.CASE_020,
        )
    execute = SPECS_020[variant].execute
    assert execute is not None
    return execute(ctx), ctx.probes[f"dcgm@{ctx.settings.node}"]


@pytest.mark.parametrize("variant", sorted(FAMILY_OF))
@pytest.mark.parametrize(
    "generation", [pytest.param(LEGACY, id="legacy"), pytest.param(HOPPER, id="hopper")]
)
def test_020_variants_target_the_counter_the_node_reports(
    tmp_path: Path, variant: str, generation: dict[str, tuple[str, str]]
) -> None:
    family = FAMILY_OF[variant]
    canonical, metric = generation[family]
    out, probe = _run_variant(
        tmp_path, variant, _survey(*generation.values()), canonical
    )
    assert out["verdict"] == "PASS", out["errors"]
    gpus = (G0, G1) if variant == "multiple_gpus" else (G1,)
    assert _plan_targets(_installed_plan(probe)) == {(canonical, gpu) for gpu in gpus}
    assert out["nvlink_family"] == family
    assert out["canonical_name"] == canonical
    assert out["exporter_metrics"] == [metric]
    assert any(canonical in note and metric in note for note in out["notes"]), out[
        "notes"
    ]
    if variant == "multiple_gpus":
        assert out["gpus"] == [G0, G1]
    else:
        assert out["gpu_uuid"] == G1


def test_020_variants_are_capability_absent_without_an_nvlink_family(
    tmp_path: Path,
) -> None:
    for variant in sorted(FAMILY_OF):
        with pytest.raises(VariantNotRun) as absent:
            _run_variant(tmp_path / variant, variant, _survey(), "unused")
        assert absent.value.reason == "CAPABILITY_ABSENT", variant
        assert FAMILY_OF[variant].removeprefix("nvlink_") in str(absent.value), variant


def test_multiple_gpus_needs_two_gpus_on_the_chosen_counter(tmp_path: Path) -> None:
    one_gpu = _survey(
        *HOPPER.values(), only_on={"nvlink_crc_aggregate_error_total": (G1,)}
    )
    with pytest.raises(VariantNotRun) as absent:
        _run_variant(
            tmp_path, "multiple_gpus", one_gpu, "nvlink_crc_aggregate_error_total"
        )
    assert absent.value.reason == "CAPABILITY_ABSENT"
    assert "fewer than two GPUs" in str(absent.value)
    assert "nvlink_crc_aggregate_error_total" in str(absent.value)


def test_hopper_crc_run_keeps_the_replay_call_sequence(tmp_path: Path) -> None:
    # Pinned deliberately: the family lookup is answered by the one survey the
    # variant already made, so a Hopper node costs no extra probe call -- the
    # replay deploys (tool + plan), repoints, arms, reads the journal until the
    # aggregate is served, and is released through close-run (its own journal
    # read first).
    _out, probe = _run_variant(
        tmp_path, "crc", _survey(*HOPPER.values()), "nvlink_crc_aggregate_error_total"
    )
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


def test_020_plan_text_promises_no_flit_counter() -> None:
    for spec in links.VARIANTS:
        if spec.case_id != links.CASE_020:
            continue
        text = f"{spec.injection} {spec.expected}"
        assert "nvlink_crc_flit_error_total" not in text, spec.key
        assert "nvlink_replay_error_total" not in text, spec.key
        assert "nvlink_recovery_error_total" not in text, spec.key
    assert "flit total on NVLink3, DL aggregate on NVLink4+" in SPECS["crc"].injection
    assert "DL aggregate on NVLink4+" in SPECS["independent_fatal_event"].injection
    assert "DL aggregate on NVLink4+" in SPECS["replay"].injection
    assert "DL aggregate on NVLink4+" in SPECS["recovery_counter"].injection


def test_product_grades_every_nvlink_family_member_alike() -> None:
    """The family-aware targeting rests on the product grading each member of a
    family identically: one WARNING RUN_DIAGNOSTICS finding on an increment (the
    delta threshold), no finding and the held finding kept on a counter decrease
    (a reset is neither a fault nor a recovery), the finding cleared by a later
    zero-delta sample (recovery), the GPU_FABRIC fault class and the multi-GPU
    composite -- for the legacy per-class counters and the DL aggregates alike.
    """

    from datetime import datetime, timedelta, timezone

    from gpu_fault.gpu_metrics import (
        GpuMetricBatch,
        GpuMetricSample,
        GpuMetricSource,
        GpuMetricsService,
    )
    from gpu_fault.host_health import _GPU_METRIC_FAULT_CLASSES

    start = datetime(2026, 10, 26, 7, 0, tzinfo=timezone.utc)

    def sample(name: str, value: float, gpu: str) -> GpuMetricSample:
        index = "0" if gpu == G0 else "1"
        return GpuMetricSample(
            metric_name=f"DCGM_{name.upper()}",
            canonical_name=name,
            value=value,
            gpu_index=index,
            gpu_uuid=gpu,
            pci_bdf=f"0000:0{index}:00.0",
        )

    def ingest(service: GpuMetricsService, tick: int, samples: list[GpuMetricSample]):
        return service.ingest(
            GpuMetricBatch(
                batch_id=f"b{tick}",
                cluster_id="gpu-eks-cluster",
                node_id="ip-10-0-43-5.us-west-2.compute.internal",
                observed_at=start + timedelta(seconds=15 * tick),
                source=GpuMetricSource.DCGM_EXPORTER,
                samples=samples,
            )
        )

    def active(service: GpuMetricsService) -> list[str]:
        return sorted(
            finding.canonical_name
            for finding in service.store.list_gpu_findings(
                "gpu-eks-cluster",
                "ip-10-0-43-5.us-west-2.compute.internal",
                active_only=True,
            )
        )

    gradings = {}
    for family, members in NVLINK_FAMILY_MEMBERS.items():
        for name in members:
            assert _GPU_METRIC_FAULT_CLASSES[name] == "GPU_FABRIC", name
            service = GpuMetricsService()
            ingest(service, 0, [sample(name, 0, G1)])
            increased = ingest(service, 1, [sample(name, 1, G1)])
            [finding] = increased.new_findings
            gradings[name] = (
                finding.severity,
                finding.automatic_action,
                finding.policy_source,
                finding.official_action,
                finding.delta,
                finding.reason,
            )
            assert active(service) == [name]
            reset = ingest(service, 2, [sample(name, 0, G1)])
            assert reset.findings == [] and active(service) == [name], (
                f"{name}: a counter reset minted or cleared a finding"
            )
            quiet = ingest(service, 3, [sample(name, 0, G1)])
            assert quiet.findings == [] and active(service) == [], (
                f"{name}: a zero-delta sample did not recover the finding"
            )
            fabric = GpuMetricsService()
            ingest(fabric, 0, [sample(name, 0, G0), sample(name, 0, G1)])
            composite = ingest(fabric, 1, [sample(name, 1, G0), sample(name, 1, G1)])
            [multi] = composite.new_composite_findings
            assert multi.correlation_rule_id == "MULTI_GPU_NVLINK_FABRIC_FAILURE", name
            assert multi.automatic_action == "RUN_DIAGNOSTICS", name
        assert len({gradings[name] for name in members}) == 1, (family, gradings)
    assert len(set(gradings.values())) == 1, gradings
    assert gradings["nvlink_crc_aggregate_error_total"][:2] == (
        "WARNING",
        "RUN_DIAGNOSTICS",
    )

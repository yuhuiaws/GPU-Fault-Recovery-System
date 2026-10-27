"""GF-GENERAL-LIVE-024 thermal / power composites over the environment replay (scripted)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from gpu_fault.gpu_metrics import GpuMetricsThresholds
from scripts.e2e.regional import general_live_gpu_composite as comp
from scripts.e2e.regional import general_live_gpu_environment as envc
from scripts.e2e.regional import general_live_registry as reg
from scripts.e2e.regional.general_live_env_replay import PROBE_KEY
from scripts.e2e.regional.general_live_fixture import VariantInconclusive, VariantNotRun
from scripts.e2e.regional.general_live_gpu_replay import GpuSurvey
from scripts.e2e.regional.regional_commands import RegionalFixtureError
from tests.regional._general_live_fakes import (
    FakeProbe,
    FakeRegional,
    make_context,
    make_settings,
)
from tests.regional._general_live_gpu_fakes import (
    dcgm_probe,
    gpu_cpu_python,
    gpu_incident,
    gpu_workflow,
)
from tests.regional._general_live_gpu_fakes import survey as base_survey

SPECS = {spec.variant: spec for spec in envc.VARIANTS}
G1 = "GPU-11111111-1111-4111-8111-111111111111"
REAL = "http://127.0.0.1:9400/metrics"
PROXY = "http://127.0.0.1:9411/metrics"
FIELDS = (
    "gpu_tlimit_c",
    "thermal_violation_total_us",
    "clock_throttle_reasons",
    "power_violation_total_us",
    "power_usage_w",
    "power_limit_w",
    "gpu_utilization_percent",
    "ecc_sbe_volatile_total",
)
# Live p5en H200 (2026-10-24): only the T.Limit scale is published.
H200_LIMITS = {
    "gpu_shutdown_tlimit_c": -8.0,
    "gpu_slowdown_tlimit_c": -2.0,
    "gpu_max_operating_tlimit_c": 0.0,
    "memory_max_operating_tlimit_c": 0.0,
}
#: slowdown -2 -> critical -2, warning 3 -> the reading inside the band is 0.5
WARNING_READING = 0.5


def _survey(
    *,
    tlimit: float = 59.0,
    limit: float = 700.0,
    mask: int = 0,
    with_limits: bool = True,
) -> dict[str, Any]:
    value = base_survey(fields=FIELDS)
    for entry in value["exporter"]["gpus"].values():
        entry["values"].update(
            {
                "gpu_tlimit_c": tlimit,
                "power_limit_w": limit,
                "power_usage_w": 70.0,
                "gpu_utilization_percent": 0.0,
                "clock_throttle_reasons": float(mask),
                "thermal_violation_total_us": 5_000_000.0,
                "power_violation_total_us": 100.0,
            }
        )
    if with_limits:
        value["temperature_limits"] = [
            {
                "gpu_uuid": gpu,
                "gpu_index": entry["gpu_index"],
                "pci_bdf": entry["pci_bdf"],
                "product": "NVIDIA H200",
                "limits": dict(H200_LIMITS),
            }
            for gpu, entry in value["exporter"]["gpus"].items()
        ]
    return value


def _env_script() -> dict[str, Any]:
    return {
        "preflight": {"boot_id": "b"},
        "install-source": {"ok": True},
        "deploy-proxy": {"status": {"armed": False}, "endpoint": PROXY},
        # The probe's proof rides in the repoint receipt: the restarted
        # collector process reads the proxy and the proxy counted its first
        # pass-through read.
        "repoint-collector": {
            "repointed": True,
            "original_endpoint": REAL,
            "proxy_endpoint": PROXY,
            "process_environment": {"GPU_FAULT_DCGM_METRICS_URL": PROXY},
            "first_read": {
                "observed": True,
                "reads_before": 0,
                "reads_after": 1,
                "waited_seconds": 21.0,
                "budget_seconds": 65,
            },
            "after": {"ActiveState": "active"},
        },
        "arm": {"armed": True, "status": {"armed": True, "reads": 3}},
        "status": [
            {"status": {"armed": False, "reads": 2, "aborted": None}},
            {"status": {"armed": True, "reads": 40, "aborted": None}},
        ],
        "restore-collector": {
            "restored_endpoint": REAL,
            "original_endpoint": REAL,
            "after": {"ActiveState": "active"},
        },
        "stop-proxy": {"audit_was_present": True},
    }


def _finding(
    rule: str,
    components: list[str],
    *,
    severity: str = "WARNING",
    action: str = "RUN_DIAGNOSTICS",
) -> dict[str, Any]:
    return {
        "finding_id": f"b-{G1}-composite-{rule}",
        "canonical_name": f"composite:{rule}",
        "correlation_rule_id": rule,
        "finding_kind": "COMPOSITE",
        "severity": severity,
        "automatic_action": action,
        "component_metrics": components,
        "component_finding_ids": [f"b-{G1}-{name}" for name in components[:2]],
        "gpu_uuid": G1,
        "observed_at": "2026-10-24T00:03:00+00:00",
    }


def _state(
    findings: list[dict[str, Any]], samples: list[dict[str, Any]]
) -> dict[str, Any]:
    return {
        "active_findings": [],
        "findings": findings,
        "batches": [{"observed_at": "t", "samples": samples}],
    }


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


def _diagnostic(rule: str, *, action: str = "RUN_DIAGNOSTICS") -> tuple[dict, dict]:
    incident = gpu_incident(
        "inc-c",
        f"b-{G1}-composite-{rule}",
        state="RECOVERED",
        action=action,
        workflow="wf",
    )
    workflow = gpu_workflow(
        "wf",
        "inc-c",
        ["FREEZE_EVIDENCE", "VALIDATE_GPU"],
        status="FAILED",
        executions=[{"operation": "VALIDATE_GPU", "status": "FAILED", "details": {}}],
    )
    return incident, workflow


def _ctx(
    tmp_path: Path,
    variant: str,
    *,
    survey_value: dict[str, Any],
    cpu: Any,
    env_script: dict[str, Any] | None = None,
) -> Any:
    spec = SPECS[variant]
    regional = FakeRegional(cpu_python=cpu)
    settings = make_settings(spec, node=regional.node_snapshot("")["name"])
    ctx = make_context(
        spec, tmp_path=tmp_path, regional=regional, probes={}, settings=settings
    )
    ctx.probes[f"dcgm@{ctx.settings.node}"] = dcgm_probe(survey_value)
    ctx.probes[f"{PROBE_KEY}@{ctx.settings.node}"] = FakeProbe(
        env_script or _env_script()
    )
    return ctx


def _plan(ctx: Any) -> dict[str, Any]:
    return json.loads((ctx.case_dir / "plan.json").read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# registry
# --------------------------------------------------------------------------- #
def test_024_is_fully_implemented_across_the_two_modules() -> None:
    specs = {s.variant: s for s in [*comp.VARIANTS, *envc.VARIANTS]}
    assert set(specs) == set(reg.coverage_rows()["GF-GENERAL-LIVE-024"])
    assert [s.variant for s in specs.values() if s.deferred] == []
    for variant in ("thermal_warning", "power_performance"):
        assert specs[variant].approvals == (), "the environment replay needs no token"
        assert specs[variant].lanes == ("eks", "hp")
    assert specs["independent_xid"].approvals == ("KERNEL_SOURCE_REPLAY",)
    assert envc.VARIANTS[0].proof_mode == "live-replay"
    reg.load_registry()


def test_survey_exposes_limits_and_pci_bdf() -> None:
    survey = GpuSurvey(_survey())
    assert survey.temperature_limits(G1) == H200_LIMITS
    assert survey.temperature_limits("GPU-missing") == {}
    assert survey.pci_bdf(G1) == "00000000:02:00.0"


# --------------------------------------------------------------------------- #
# thermal_warning
# --------------------------------------------------------------------------- #
THERMAL_COMPONENTS = [
    "gpu_tlimit_c",
    "thermal_violation_total_us",
    "clock_throttle_reasons",
    "sm_clock_mhz",
]


def test_thermal_target_grades_the_h200_on_the_tlimit_scale() -> None:
    target = envc.thermal_target(GpuSurvey(_survey()), GpuMetricsThresholds())
    assert target.gpu == G1, "the highest-index capable GPU"
    assert target.decision.scale == "tlimit"
    assert target.field == "gpu_tlimit_c"
    assert target.warning == WARNING_READING
    assert target.baseline_us == 5_000_000.0


def test_thermal_target_is_not_run_without_a_gradable_scale() -> None:
    with pytest.raises(VariantNotRun) as info:
        envc.thermal_target(
            GpuSurvey(_survey(with_limits=False)), GpuMetricsThresholds()
        )
    assert info.value.reason == "CAPABILITY_ABSENT"


@pytest.mark.parametrize(
    ("tlimit", "mask", "fragment"),
    [
        (5.0, 0, "safely above 0"),
        (59.0, 0x20, "thermal/brake"),
        (59.0, 0x80, "thermal/brake"),
    ],
)
def test_require_cool_refuses_a_warm_or_throttling_gpu(
    tlimit: float, mask: int, fragment: str
) -> None:
    survey = GpuSurvey(_survey(tlimit=tlimit, mask=mask))
    target = envc.thermal_target(survey, GpuMetricsThresholds())
    with pytest.raises(VariantNotRun) as info:
        envc.require_cool(survey, target)
    assert info.value.reason == "FIXTURE_NOT_READY"
    assert fragment in info.value.detail


def test_thermal_warning_is_a_warning_diagnostic_composite(tmp_path: Path) -> None:
    incident, workflow = _diagnostic("THERMAL_STRESS")
    cpu = gpu_cpu_python(
        activity=_activity([incident], [workflow]),
        gpu_state=_state(
            [_finding("THERMAL_STRESS", THERMAL_COMPONENTS)],
            [{"gpu": G1, "name": "gpu_tlimit_c", "value": WARNING_READING}],
        ),
    )
    ctx = _ctx(tmp_path, "thermal_warning", survey_value=_survey(), cpu=cpu)
    out = envc.thermal_warning(ctx)
    assert out["verdict"] == "PASS", out["errors"]
    assert out["proof_scope"] == "live-replay" and out["rule_id"] == "THERMAL_STRESS"
    assert out["gpu_uuid"] == G1 and out["scale"] == "tlimit"
    assert set(out["stages"]) == {
        "chain",
        "read_only",
        "diagnostic",
        "own_components",
        "dedup",
        "hardware",
        "recovery",
    }
    plan = _plan(ctx)
    breach = [step for step in plan["steps"] if "gpu_tlimit_c" in step["values"]]
    assert len(breach) == 3 and all(
        step["values"]["gpu_tlimit_c"] == WARNING_READING
        and step["values"]["clock_throttle_reasons"] == 0x20
        for step in breach
    ), "one warning-band read per confirmation sample of the surveyed node"
    durations = [step["values"]["thermal_violation_total_us"] for step in breach]
    assert durations[0] > 5_000_000.0 and durations == sorted(durations), (
        "the duration rises from above the device's real counter"
    )
    assert plan["steps"][-1]["values"]["thermal_violation_total_us"] == durations[-1], (
        "the cleared read repeats the confirmed read's duration"
    )
    env = ctx.probes[f"{PROBE_KEY}@{ctx.settings.node}"]
    verbs = [call[0] for call in env.calls]
    assert verbs.index("repoint-collector") < verbs.index("arm")
    assert verbs.index("restore-collector") < verbs.index("stop-proxy")
    assert ctx.cleanup.pending() == [], "the replay closed itself before recovery"
    assert out["replay"]["collector_process_endpoint"] == PROXY
    assert out["replay"]["collector_first_read"]["observed"] is True


def test_thermal_warning_keeps_the_env_replay_call_sequence(tmp_path: Path) -> None:
    # Pinned deliberately: the proof that the restarted collector reads the
    # proxy (process environment + first pass-through read) rides in the
    # repoint-collector receipt, so the variant arms after one pass-through
    # status with no extra probe call; one served status, then the endpoint is
    # restored before the proxy stops and before the recovery wait.
    incident, workflow = _diagnostic("THERMAL_STRESS")
    cpu = gpu_cpu_python(
        activity=_activity([incident], [workflow]),
        gpu_state=_state(
            [_finding("THERMAL_STRESS", THERMAL_COMPONENTS)],
            [{"gpu": G1, "name": "gpu_tlimit_c", "value": WARNING_READING}],
        ),
    )
    ctx = _ctx(tmp_path, "thermal_warning", survey_value=_survey(), cpu=cpu)
    assert envc.thermal_warning(ctx)["verdict"] == "PASS"
    env = ctx.probes[f"{PROBE_KEY}@{ctx.settings.node}"]
    assert [call[0] for call in env.calls] == [
        "preflight",
        "install-source",
        "install-source",
        "deploy-proxy",
        "repoint-collector",
        "status",
        "arm",
        "status",
        "restore-collector",
        "stop-proxy",
    ]


def test_thermal_warning_is_a_fixture_error_when_the_repoint_did_not_take_effect(
    tmp_path: Path,
) -> None:
    # Live 2026-10-26 (plain EKS): the collector restarted into the drop-in
    # but kept reading the real exporter, and the variant spent its 12-read
    # budget on INCONCLUSIVE "consumed 0 of the 12 armed replay reads". The
    # probe now judges that on the node and rolls back; the variant fails as
    # a fixture error with the reason before anything is armed, and the
    # runner's teardown still restores the unit before it stops the proxy.
    incident, workflow = _diagnostic("THERMAL_STRESS")
    cpu = gpu_cpu_python(
        activity=_activity([incident], [workflow]),
        gpu_state=_state([_finding("THERMAL_STRESS", THERMAL_COMPONENTS)], []),
    )
    script = _env_script()
    script["repoint-collector"] = {
        "repointed": False,
        "reason": "the restarted collector process does not read the proxy "
        "endpoint; the override never reached it",
        "original_endpoint": REAL,
        "proxy_endpoint": PROXY,
        "process_environment": {"GPU_FAULT_DCGM_METRICS_URL": REAL},
        "rollback": {"restarted": True, "restored_endpoint": REAL},
        "after": {"ActiveState": "active"},
    }
    ctx = _ctx(
        tmp_path, "thermal_warning", survey_value=_survey(), cpu=cpu, env_script=script
    )
    with pytest.raises(RegionalFixtureError, match="did not take effect"):
        envc.thermal_warning(ctx)
    env = ctx.probes[f"{PROBE_KEY}@{ctx.settings.node}"]
    verbs = [call[0] for call in env.calls]
    assert "arm" not in verbs, "nothing is armed on an unproven repoint"
    assert verbs[-1] == "repoint-collector"
    assert ctx.cleanup.pending() == ["env-replay"]
    results = ctx.cleanup.run()
    assert all(item["ok"] for item in results), results
    verbs = [call[0] for call in env.calls]
    assert verbs[-2:] == ["restore-collector", "stop-proxy"]


def _old_shape(plan: dict[str, Any]) -> dict[str, Any]:
    """The 2026-10-26 live shape: three more rising reads, the tail at the last."""

    rising = [
        step for step in plan["steps"] if step["values"].get("clock_throttle_reasons")
    ]
    step_us = (
        rising[1]["values"]["thermal_violation_total_us"]
        - rising[0]["values"]["thermal_violation_total_us"]
    )
    extra = [
        {
            "reads": 1,
            "values": {
                **rising[-1]["values"],
                "thermal_violation_total_us": (
                    rising[-1]["values"]["thermal_violation_total_us"] + index * step_us
                ),
            },
        }
        for index in range(1, 4)
    ]
    tail = dict(plan["steps"][-1])
    tail["values"] = {
        **tail["values"],
        "thermal_violation_total_us": extra[-1]["values"]["thermal_violation_total_us"],
    }
    return {"steps": [plan["steps"][0], *rising, *extra, tail]}


def test_checked_tail_refuses_a_rise_on_the_cleared_read(tmp_path: Path) -> None:
    """The guard reads the surveyed node's own confirmation count (3 here)."""

    survey = GpuSurvey(_survey())
    target = envc.thermal_target(survey, GpuMetricsThresholds())
    good = envc.plans.thermal_stress_plan(
        target.decision, confirmation_samples=3, baseline_us=target.baseline_us
    )
    assert (
        envc.checked_tail(
            good,
            survey,
            variant="thermal_warning",
            counter="thermal_violation_total_us",
            marker="clock_throttle_reasons",
        )
        is good
    )
    with pytest.raises(RegionalFixtureError, match="plan rejected") as info:
        envc.checked_tail(
            _old_shape(good),
            survey,
            variant="thermal_warning",
            counter="thermal_violation_total_us",
            marker="clock_throttle_reasons",
        )
    assert "read 10 clears clock_throttle_reasons and serves 1.1e+07" in str(info.value)
    assert "read 6 (the 3-sample confirmation) with 8e+06" in str(info.value)


def test_thermal_warning_refuses_a_rising_tail_before_anything_is_deployed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A plan builder that regresses to the live 2026-10-26 shape is caught by
    the pre-deploy guard: a fixture error, no probe call, nothing to clean."""

    builder = envc.plans.thermal_stress_plan
    monkeypatch.setattr(
        envc.plans,
        "thermal_stress_plan",
        lambda *args, **kwargs: _old_shape(builder(*args, **kwargs)),
    )
    cpu = gpu_cpu_python(activity=_activity([]), gpu_state=_state([], []))
    ctx = _ctx(tmp_path, "thermal_warning", survey_value=_survey(), cpu=cpu)
    with pytest.raises(RegionalFixtureError, match="change the counter"):
        envc.thermal_warning(ctx)
    env = ctx.probes[f"{PROBE_KEY}@{ctx.settings.node}"]
    assert env.calls == [], "nothing was deployed"
    assert ctx.cleanup.pending() == []


def test_thermal_warning_fails_when_the_composite_escalates(tmp_path: Path) -> None:
    incident = gpu_incident(
        "inc-c", f"b-{G1}-composite-THERMAL_STRESS", action="DRAIN", workflow="wf"
    )
    workflow = gpu_workflow(
        "wf", "inc-c", ["FREEZE_EVIDENCE", "MARK_UNSCHEDULABLE", "DRAIN"]
    )
    cpu = gpu_cpu_python(
        activity=_activity([incident], [workflow]),
        gpu_state=_state(
            [
                _finding(
                    "THERMAL_STRESS",
                    THERMAL_COMPONENTS,
                    severity="CRITICAL",
                    action="DRAIN",
                )
            ],
            [{"gpu": G1, "name": "gpu_tlimit_c", "value": WARNING_READING}],
        ),
    )
    out = envc.thermal_warning(
        _ctx(tmp_path, "thermal_warning", survey_value=_survey(), cpu=cpu)
    )
    assert out["verdict"] == "FAIL"
    assert any("CRITICAL" in error for error in out["errors"]), out["errors"]
    assert any("isolates" in error for error in out["errors"]), out["errors"]


def test_thermal_warning_fails_when_a_component_is_minted_beside_the_composite(
    tmp_path: Path,
) -> None:
    incident, workflow = _diagnostic("THERMAL_STRESS")
    late = gpu_incident(
        "inc-late",
        f"b2-{G1}-thermal_violation_total_us",
        state="RECOVERED",
        workflow="wf-2",
    )
    late["created_at"] = "2026-10-24T00:05:00+00:00"
    cpu = gpu_cpu_python(
        activity=_activity([incident, late], [workflow]),
        gpu_state=_state(
            [_finding("THERMAL_STRESS", THERMAL_COMPONENTS)],
            [{"gpu": G1, "name": "gpu_tlimit_c", "value": WARNING_READING}],
        ),
    )
    out = envc.thermal_warning(
        _ctx(tmp_path, "thermal_warning", survey_value=_survey(), cpu=cpu)
    )
    assert out["verdict"] == "FAIL"
    assert any("after the composite formed" in error for error in out["errors"]), out[
        "errors"
    ]


def test_thermal_warning_fails_when_the_composite_absorbs_a_foreign_component(
    tmp_path: Path,
) -> None:
    incident, workflow = _diagnostic("THERMAL_STRESS")
    finding = _finding("THERMAL_STRESS", [*THERMAL_COMPONENTS, "kernel_xid"])
    finding["component_finding_ids"].append("kernel-xid-event-1")
    cpu = gpu_cpu_python(
        activity=_activity([incident], [workflow]),
        gpu_state=_state(
            [finding], [{"gpu": G1, "name": "gpu_tlimit_c", "value": WARNING_READING}]
        ),
    )
    out = envc.thermal_warning(
        _ctx(tmp_path, "thermal_warning", survey_value=_survey(), cpu=cpu)
    )
    assert out["verdict"] == "FAIL"
    assert any("foreign components" in error for error in out["errors"]), out["errors"]
    assert any("XID components" in error for error in out["errors"]), out["errors"]


def test_thermal_warning_is_inconclusive_without_an_accepted_replayed_sample(
    tmp_path: Path,
) -> None:
    incident, workflow = _diagnostic("THERMAL_STRESS")
    cpu = gpu_cpu_python(
        activity=_activity([incident], [workflow]),
        gpu_state=_state([_finding("THERMAL_STRESS", THERMAL_COMPONENTS)], []),
    )
    with pytest.raises(VariantInconclusive, match="accepted no replayed gpu_tlimit_c"):
        envc.thermal_warning(
            _ctx(tmp_path, "thermal_warning", survey_value=_survey(), cpu=cpu)
        )


def test_thermal_warning_fails_when_the_components_stay_active(tmp_path: Path) -> None:
    incident, workflow = _diagnostic("THERMAL_STRESS")
    state = _state(
        [_finding("THERMAL_STRESS", THERMAL_COMPONENTS)],
        [{"gpu": G1, "name": "gpu_tlimit_c", "value": WARNING_READING}],
    )
    state["active_findings"] = [
        {"canonical_name": "thermal_violation_total_us", "gpu_uuid": G1}
    ]
    cpu = gpu_cpu_python(activity=_activity([incident], [workflow]), gpu_state=state)
    ctx = _ctx(tmp_path, "thermal_warning", survey_value=_survey(), cpu=cpu)
    out = envc.thermal_warning(ctx)
    assert out["verdict"] == "FAIL"
    assert any("stayed active" in error for error in out["errors"]), out["errors"]


# --------------------------------------------------------------------------- #
# power_performance
# --------------------------------------------------------------------------- #
POWER_COMPONENTS = [
    "power_violation_total_us",
    "power_usage_w",
    "power_limit_w",
    "gpu_utilization_percent",
]
SERVED_USAGE = round(700.0 * 0.975, 1)


def test_power_target_needs_every_field_and_a_positive_limit() -> None:
    assert envc.power_target(GpuSurvey(_survey())) == (G1, 700.0)
    with pytest.raises(VariantNotRun) as info:
        envc.power_target(GpuSurvey(_survey(limit=0.0)))
    assert info.value.reason == "CAPABILITY_ABSENT"


def test_power_performance_is_a_warning_diagnostic_composite(tmp_path: Path) -> None:
    incident, workflow = _diagnostic("POWER_LIMIT_THROTTLING")
    cpu = gpu_cpu_python(
        activity=_activity([incident], [workflow]),
        gpu_state=_state(
            [_finding("POWER_LIMIT_THROTTLING", POWER_COMPONENTS)],
            [{"gpu": G1, "name": "power_usage_w", "value": SERVED_USAGE}],
        ),
    )
    ctx = _ctx(tmp_path, "power_performance", survey_value=_survey(), cpu=cpu)
    out = envc.power_performance(ctx)
    assert out["verdict"] == "PASS", out["errors"]
    assert out["power_limit_w"] == 700.0 and out["served_power_usage_w"] == SERVED_USAGE
    plan = _plan(ctx)
    breach = [step for step in plan["steps"] if "power_usage_w" in step["values"]]
    assert len(breach) == 3, "one at-the-cap read per confirmation sample"
    for step in breach:
        values = step["values"]
        assert 700.0 * 0.95 <= values["power_usage_w"] <= 700.0, values
        assert values["gpu_utilization_percent"] >= 80.0, values
        assert "clock_throttle_reasons" not in values, (
            "no reason bit: it would be a temperature finding"
        )
        assert "power_limit_w" not in values, (
            "the enforced limit stays the device's real reading"
        )
    assert plan["steps"][-1]["values"] == {
        "power_violation_total_us": breach[-1]["values"]["power_violation_total_us"]
    }


def test_power_performance_refuses_a_really_throttling_gpu(tmp_path: Path) -> None:
    cpu = gpu_cpu_python(activity=_activity([]), gpu_state=_state([], []))
    ctx = _ctx(tmp_path, "power_performance", survey_value=_survey(mask=0x40), cpu=cpu)
    with pytest.raises(VariantNotRun) as info:
        envc.power_performance(ctx)
    assert info.value.reason == "FIXTURE_NOT_READY"
    env = ctx.probes[f"{PROBE_KEY}@{ctx.settings.node}"]
    assert env.calls == [], "nothing was deployed"

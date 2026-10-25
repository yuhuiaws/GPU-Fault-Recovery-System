"""GF-GENERAL-LIVE-007 SMART variants over the shared host source replay.

No cluster, no node: a scripted source-replay ``FakeProbe`` receives the plan
and window the variant sends and answers the wrapper journal; a scripted store
plays the product's records before and after the window closes. The tests pin
what is sent (the reviewed rule per variant, ``--wrapper smartctl`` or
``--inaccessible tool:smartctl``), that the verdict reads only the product's
records (marker + collection error for UNKNOWN, marker without error for
NOT_SUPPORTED, the QUARANTINE chain and restore for a failed drive), that a rule
the product never had served is INCONCLUSIVE, and that the real tool must answer
again after the close.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import general_live_smart as smart
from scripts.e2e.regional.general_live_fixture import VariantInconclusive, VariantNotRun
from scripts.e2e.regional.general_live_registry import (
    LIVE_REPLAY,
    OBSERVATION,
    coverage_rows,
)
from scripts.e2e.regional.general_live_source_replay import HOST_SOURCE_REPLAY
from tests.regional._general_live_fakes import (
    FakeProbe,
    FakeRegional,
    incident,
    make_context,
    make_settings,
    workflow,
)
from tools import host_command_replay as wrapper

NODE = "ip-10-0-43-5.us-west-2.compute.internal"
DEV0 = "/dev/nvme0"
DEV1 = "/dev/nvme1"
SPECS = {spec.variant: spec for spec in smart.VARIANTS}
T0 = "2026-10-24T00:01:00+00:00"
T1 = "2026-10-24T00:02:00+00:00"
T2 = "2026-10-24T00:03:00+00:00"


# --------------------------------------------------------------------------- #
# scripted product records
# --------------------------------------------------------------------------- #
def sample(
    name: str, value: float, device: str | None = None, **labels: str
) -> dict[str, Any]:
    return {"name": name, "device": device, "value": value, "labels": labels}


def batch(
    observed_at: str, *samples: dict[str, Any], errors: tuple[str, ...] = ()
) -> dict[str, Any]:
    return {
        "observed_at": observed_at,
        "batch_id": f"host-{observed_at}",
        "edge_filter_reasons": [],
        "collection_errors": list(errors),
        "samples": list(samples),
    }


def doc(*batches: dict[str, Any], **extra: Any) -> dict[str, Any]:
    document: dict[str, Any] = {
        "incidents": [],
        "workflows": [],
        "batches": list(batches),
        "markers": [],
        "commands": [],
        "health_signals": {},
        "notifications": [],
        "metric_latest": [
            {"name": smart.HEALTH, "device": DEV0, "value": 0.0, "labels": {}},
            {"name": smart.HEALTH, "device": DEV1, "value": 0.0, "labels": {}},
        ],
        "collector_statuses": [],
    }
    document.update(extra)
    return document


def healthy(*times: str) -> dict[str, Any]:
    return doc(
        *[
            batch(
                moment, sample(smart.HEALTH, 0.0, DEV0), sample(smart.HEALTH, 0.0, DEV1)
            )
            for moment in (times or (T1, T2))
        ]
    )


def survey(smartctl: bool = True) -> dict[str, Any]:
    return {
        "cpu_count": 8,
        "tools": {"smartctl": smartctl, "systemd-run": True},
        "modules": {},
        "backing": [],
        "collector_env": {
            "GPU_FAULT_HOST_INTERVAL_SECONDS": "15",
            "GPU_FAULT_HOST_HEALTH_SUMMARY_SECONDS": "300",
        },
        "residue": {"units": [], "paused_units": {}, "paused_processes": {}},
    }


class Scenario:
    """The product's records before the window closes, and after."""

    def __init__(self, during: dict[str, Any], after: dict[str, Any] | None = None):
        self.during = during
        self.after = healthy() if after is None else after
        self.closed = False
        self.events: list[str] = []

    def activity(self, _script: str, *_arguments: str) -> dict[str, Any]:
        return self.after if self.closed else self.during

    def close(self, _arguments: tuple[str, ...]) -> dict[str, Any]:
        self.closed = True
        self.events.append("close-window")
        return {"clean": True, "after": {"ActiveState": "active"}}


def audit_entry(query: str, *argv: str, applied: bool = True) -> dict[str, Any]:
    return {
        "query": query,
        "argv": list(argv),
        "applied_rules": [0] if applied else [],
        "served_returncode": 0,
        "real_returncode": 0,
        "proof_scope": "live-replay",
    }


def source_probe(scenario: Scenario, entries: list[dict[str, Any]]) -> FakeProbe:
    return FakeProbe(
        {
            "install-source": {"bytes": 1},
            "install-wrapper": {
                "shim": "/run/gfrs-live/source-x/bin/smartctl",
                "real": "/usr/sbin/smartctl",
            },
            "open-window": {
                "after": {"ActiveState": "active"},
                "proof_scope": "live-replay",
                "verification": {"ok": True, "checks": []},
            },
            "audit-read": {"entries": entries, "calls": {}},
            "status": {"fixtures": {}},
            "close-window": scenario.close,
        }
    )


def run(
    variant: str,
    scenario: Scenario,
    tmp_path: Path,
    *,
    entries: list[dict[str, Any]] | None = None,
    smartctl: bool = True,
) -> tuple[dict[str, Any], FakeProbe, FakeProbe]:
    spec = SPECS[variant]
    if entries is None:
        entries = [
            audit_entry("health", "-H", "-j", DEV0),
            audit_entry("scan", "--scan-open"),
        ]
    node = FakeProbe({"survey": survey(smartctl)})
    source = source_probe(scenario, entries)
    regional = FakeRegional(cpu_python=scenario.activity)
    ctx = make_context(
        spec,
        tmp_path=tmp_path,
        regional=regional,
        probes={f"node@{NODE}": node, f"source-replay@{NODE}": source},
        settings=make_settings(spec, node=NODE),
    )
    return spec.execute(ctx), source, node


def sent_plan(source: FakeProbe) -> dict[str, Any]:
    installs = [
        call
        for call in source.calls
        if call[0] == "install-source" and "plan-smartctl.json" in call
    ]
    flags = dict(zip(installs[-1][1::2], installs[-1][2::2]))
    return json.loads(base64.b64decode(flags["--b64"]))


def open_flags(source: FakeProbe) -> tuple[str, ...]:
    return next(call for call in source.calls if call[0] == "open-window")[1:]


def restore_recorder(monkeypatch: pytest.MonkeyPatch, scenario: Scenario) -> list:
    restored: list[list[str]] = []

    def record(ctx: Any, incidents: Any, reason: str) -> list[dict[str, Any]]:
        scenario.events.append("restore")
        restored.append([item["incident_id"] for item in incidents])
        return [{"status": "SUCCEEDED"}]

    monkeypatch.setattr(smart, "restore_incidents", record)
    return restored


# --------------------------------------------------------------------------- #
# the registry
# --------------------------------------------------------------------------- #
def test_every_007_variant_is_registered_on_both_lanes() -> None:
    assert sorted(SPECS) == sorted(coverage_rows()["GF-GENERAL-LIVE-007"])
    for spec in smart.VARIANTS:
        assert spec.lanes == ("eks", "hp"), spec.key
        assert spec.implemented, spec.key
        if spec.variant == "healthy":
            assert spec.proof_mode == OBSERVATION and spec.approvals == ()
        else:
            assert spec.proof_mode == LIVE_REPLAY, spec.key
            assert spec.approvals == (HOST_SOURCE_REPLAY,), spec.key
            assert spec.expected_outcome == "PASS", spec.key


def test_rules_are_valid_wrapper_plans() -> None:
    for rules in (
        smart.health_failed_rules(DEV0),
        smart.query_failed_rules(DEV0),
        smart.malformed_json_rules(DEV0),
        smart.scan_failed_rules(),
        smart.unsupported_device_rules(DEV0),
    ):
        plan = wrapper.build_plan("smartctl", rules, not_after=10.0)
        assert plan["rules"] == rules, rules
    failed = json.loads(smart.failed_query_json(DEV0))
    assert failed["smartctl"]["exit_status"] == 2 and "smart_status" not in failed
    with pytest.raises(json.JSONDecodeError):
        json.loads(smart.MALFORMED_JSON)


# --------------------------------------------------------------------------- #
# health_failed: verdict -> QUARANTINE -> real tool -> restore
# --------------------------------------------------------------------------- #
def quarantine_doc(device: str) -> dict[str, Any]:
    inc = incident(
        "inc-s", smart.HEALTH, device=device, state="QUARANTINED", workflow="wf-s"
    )
    wf = workflow(
        "wf-s",
        "inc-s",
        ["FREEZE_EVIDENCE", "MARK_UNSCHEDULABLE", "QUARANTINE"],
        status="SUCCEEDED",
    )
    return doc(
        batch(T1, sample(smart.HEALTH, 1.0, device), sample(smart.HEALTH, 0.0, DEV1)),
        incidents=[inc],
        workflows=[wf],
    )


def test_health_failed_quarantines_then_restores_after_the_real_tool_answers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scenario = Scenario(quarantine_doc(DEV0))
    restored = restore_recorder(monkeypatch, scenario)
    result, source, _node = run("health_failed", scenario, tmp_path)
    assert result["verdict"] == "PASS", result["errors"]
    assert result["device"] == DEV0 and result["proof_scope"] == "live-replay"
    plan = sent_plan(source)
    assert plan["rules"] == [
        {
            "query": "health",
            "action": "set_json",
            "device": DEV0,
            "field": "smart_status.passed",
            "value": False,
        }
    ], plan
    flags = open_flags(source)
    assert flags[:6] == (
        "--mode",
        "command",
        "--restore-seconds",
        "1920",
        "--wrapper",
        "smartctl",
    )
    assert restored == [["inc-s"]], restored
    assert scenario.events == ["close-window", "restore"], (
        "the window closes (real smartctl answers again) before the validated restore"
    )
    audit = [call for call in source.calls if call[0] == "audit-read"]
    assert audit and audit[0][1:5] == ("--tool", "smartctl", "--tail", "400"), audit
    assert (
        tmp_path / "cases" / SPECS["health_failed"].key / "smart-plan.json"
    ).is_file(), "the plan is kept as evidence"


def test_health_failed_is_inconclusive_when_the_rule_was_never_served(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scenario = Scenario(quarantine_doc(DEV0))
    restore_recorder(monkeypatch, scenario)
    with pytest.raises(VariantInconclusive, match="never had the smartctl health rule"):
        run(
            "health_failed",
            scenario,
            tmp_path,
            entries=[
                audit_entry("health", "-H", "-j", DEV1),
                audit_entry("scan", "--scan-open", applied=False),
            ],
        )


def test_health_failed_fails_when_the_verdict_reaches_no_incident(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    during = doc(batch(T1, sample(smart.HEALTH, 1.0, DEV0)))
    scenario = Scenario(during)
    restore_recorder(monkeypatch, scenario)
    result, _source, _node = run("health_failed", scenario, tmp_path)
    assert result["verdict"] == "FAIL"
    assert any("finding reached an incident" in error for error in result["errors"]), (
        result
    )


def test_health_failed_fails_when_the_real_tool_still_reads_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    after = doc(
        batch(T2, sample(smart.HEALTH, 1.0, DEV0), sample(smart.HEALTH, 0.0, DEV1))
    )
    scenario = Scenario(quarantine_doc(DEV0), after=after)
    restore_recorder(monkeypatch, scenario)
    result, _source, _node = run("health_failed", scenario, tmp_path)
    assert result["verdict"] == "FAIL"
    assert any(error.startswith("real_tool_again:") for error in result["errors"]), (
        result
    )


def test_a_node_without_a_smart_verdict_is_not_run(tmp_path: Path) -> None:
    scenario = Scenario(doc(metric_latest=[]))
    with pytest.raises(VariantNotRun, match="CAPABILITY_ABSENT"):
        run("health_failed", scenario, tmp_path)
    with pytest.raises(VariantNotRun, match="lacks smartctl"):
        run("health_failed", Scenario(healthy()), tmp_path, smartctl=False)


# --------------------------------------------------------------------------- #
# UNKNOWN: query_failed / malformed_json / scan_failed
# --------------------------------------------------------------------------- #
def unknown_doc(device: str | None, failure_mode: str, error: str) -> dict[str, Any]:
    return doc(
        batch(
            T1,
            sample(smart.QUERY_FAILED, 1.0, device, failure_mode=failure_mode),
            *([sample(smart.HEALTH, 0.0, DEV1)] if device is not None else []),
            errors=(f"_smart: HostObservationError: {error}",),
        )
    )


def test_query_failed_is_unknown_for_the_drive_only(tmp_path: Path) -> None:
    error = f"SMART health is unknown for {DEV0}: no verdict, exit status 2"
    result, source, _node = run(
        "query_failed", Scenario(unknown_doc(DEV0, "QUERY_FAILED", error)), tmp_path
    )
    assert result["verdict"] == "PASS", result["errors"]
    plan = sent_plan(source)
    assert plan["rules"][0]["action"] == "replace_output"
    assert plan["rules"][0]["exit_code"] == 2 and plan["rules"][0]["device"] == DEV0
    assert json.loads(plan["rules"][0]["stdout"])["smartctl"]["exit_status"] == 2
    assert result["collection_errors"] == [f"_smart: HostObservationError: {error}"]


def test_query_failed_fails_on_a_fake_verdict_or_a_dropped_drive(
    tmp_path: Path,
) -> None:
    error = f"SMART health is unknown for {DEV0}: no verdict, exit status 2"
    faked = unknown_doc(DEV0, "QUERY_FAILED", error)
    faked["batches"][0]["samples"].append(sample(smart.HEALTH, 0.0, DEV0))
    result, _s, _n = run("query_failed", Scenario(faked), tmp_path)
    assert result["verdict"] == "FAIL"
    assert any(error.startswith("no_verdict:") for error in result["errors"]), result
    dropped = unknown_doc(DEV0, "QUERY_FAILED", error)
    dropped["batches"][0]["samples"] = [dropped["batches"][0]["samples"][0]]
    result, _s, _n = run("query_failed", Scenario(dropped), tmp_path)
    assert any(error.startswith("other_drives_kept:") for error in result["errors"]), (
        result
    )


def test_query_failed_fails_without_the_unknown_error(tmp_path: Path) -> None:
    silent = unknown_doc(DEV0, "QUERY_FAILED", "unrelated")
    silent["batches"][0]["collection_errors"] = []
    result, _s, _n = run("query_failed", Scenario(silent), tmp_path)
    assert result["verdict"] == "FAIL"
    assert any(error.startswith("collection_error:") for error in result["errors"]), (
        result
    )


def test_malformed_json_is_unknown(tmp_path: Path) -> None:
    error = f"SMART health is unknown for {DEV0}: exit status 0, output is not JSON"
    result, source, _node = run(
        "malformed_json", Scenario(unknown_doc(DEV0, "QUERY_FAILED", error)), tmp_path
    )
    assert result["verdict"] == "PASS", result["errors"]
    rule = sent_plan(source)["rules"][0]
    assert rule["action"] == "replace_output" and rule["stdout"] == smart.MALFORMED_JSON
    assert "exit_code" not in rule, "a parse failure keeps smartctl's exit 0"


def test_scan_failed_is_an_unknown_drive_set_not_no_drives(tmp_path: Path) -> None:
    error = "smartctl --scan-open failed, so the drive set is unknown: exit status 1"
    result, source, _node = run(
        "scan_failed",
        Scenario(unknown_doc(None, "SCAN_FAILED", error)),
        tmp_path,
        entries=[audit_entry("scan", "--scan-open")],
    )
    assert result["verdict"] == "PASS", result["errors"]
    assert sent_plan(source)["rules"] == [
        {
            "query": "scan",
            "action": "exit",
            "exit_code": 1,
            "stderr": smart.SCAN_FAILED_STDERR,
        }
    ]
    as_no_drives = unknown_doc(None, "SCAN_FAILED", error)
    as_no_drives["batches"][0]["samples"].append(
        sample(smart.UNSUPPORTED, 1.0, None, reason="NO_DEVICES")
    )
    result, _s, _n = run(
        "scan_failed",
        Scenario(as_no_drives),
        tmp_path / "b",
        entries=[audit_entry("scan", "--scan-open")],
    )
    assert any(error.startswith("not_no_drives:") for error in result["errors"]), result


def test_unknown_variants_fail_when_the_error_outlives_the_window(
    tmp_path: Path,
) -> None:
    error = f"SMART health is unknown for {DEV0}: no verdict, exit status 2"
    after = healthy()
    after["batches"][0]["collection_errors"] = [
        f"_smart: HostObservationError: {error}"
    ]
    result, _s, _n = run(
        "query_failed",
        Scenario(unknown_doc(DEV0, "QUERY_FAILED", error), after=after),
        tmp_path,
    )
    assert result["verdict"] == "FAIL"
    assert any(
        "still reports a _smart collection error" in e for e in result["errors"]
    ), result


# --------------------------------------------------------------------------- #
# NOT_SUPPORTED: tool_missing / unsupported_device
# --------------------------------------------------------------------------- #
def test_tool_missing_hides_smartctl_inside_the_unit_only(tmp_path: Path) -> None:
    during = doc(batch(T1, sample(smart.UNSUPPORTED, 1.0, None, reason="TOOL_MISSING")))
    result, source, _node = run("tool_missing", Scenario(during), tmp_path, entries=[])
    assert result["verdict"] == "PASS", result["errors"]
    flags = open_flags(source)
    assert flags[:6] == (
        "--mode",
        "mixed",
        "--restore-seconds",
        "960",
        "--inaccessible",
        "tool:smartctl",
    )
    assert not any(call[0] == "install-wrapper" for call in source.calls), (
        "hiding the tool installs no wrapper"
    )
    assert not any(call[0] == "audit-read" for call in source.calls), (
        "no wrapper, no journal to read"
    )


def test_tool_missing_is_inconclusive_when_the_product_still_found_smartctl(
    tmp_path: Path,
) -> None:
    with pytest.raises(VariantInconclusive, match="still found smartctl"):
        run("tool_missing", Scenario(healthy()), tmp_path, entries=[])


def test_tool_missing_fails_on_an_error_or_a_sample(tmp_path: Path) -> None:
    during = doc(
        batch(
            T1,
            sample(smart.UNSUPPORTED, 1.0, None, reason="TOOL_MISSING"),
            sample(smart.QUERY_FAILED, 1.0, None, failure_mode="SCAN_FAILED"),
            errors=("_smart: HostObservationError: smartctl missing",),
        )
    )
    result, _s, _n = run("tool_missing", Scenario(during), tmp_path, entries=[])
    assert result["verdict"] == "FAIL"
    stages = {error.split(":", 1)[0] for error in result["errors"]}
    assert {"no_error", "no_verdict"} <= stages, result["errors"]


def test_unsupported_device_is_neither_failed_nor_healthy(tmp_path: Path) -> None:
    during = doc(
        batch(
            T1,
            sample(smart.UNSUPPORTED, 1.0, DEV0, reason="NO_SMART_STATUS"),
            sample(smart.HEALTH, 0.0, DEV1),
        )
    )
    result, source, _node = run("unsupported_device", Scenario(during), tmp_path)
    assert result["verdict"] == "PASS", result["errors"]
    assert sent_plan(source)["rules"] == [
        {
            "query": "health",
            "action": "delete_json",
            "device": DEV0,
            "field": "smart_status",
        }
    ]
    healthy_scan = doc(
        batch(
            T1,
            sample(smart.UNSUPPORTED, 1.0, DEV0, reason="NO_SMART_STATUS"),
            sample(smart.HEALTH, 0.0, DEV0),
            sample(smart.HEALTH, 0.0, DEV1),
        )
    )
    result, _s, _n = run("unsupported_device", Scenario(healthy_scan), tmp_path / "b")
    assert any(error.startswith("not_healthy:") for error in result["errors"]), result


def test_unsupported_device_fails_when_reported_as_a_failed_query(
    tmp_path: Path,
) -> None:
    during = doc(
        batch(
            T1,
            sample(smart.QUERY_FAILED, 1.0, DEV0, failure_mode="QUERY_FAILED"),
            sample(smart.HEALTH, 0.0, DEV1),
            errors=("_smart: HostObservationError: SMART health is unknown",),
        )
    )
    result, _s, _n = run("unsupported_device", Scenario(during), tmp_path)
    assert result["verdict"] == "FAIL"
    stages = {error.split(":", 1)[0] for error in result["errors"]}
    assert {"unsupported_marker", "not_failed", "no_error"} <= stages, result["errors"]


# --------------------------------------------------------------------------- #
# helpers other categories reuse
# --------------------------------------------------------------------------- #
def test_served_calls_filters_query_device_and_applied_rules() -> None:
    audit = {
        "entries": [
            audit_entry("health", "-H", "-j", DEV0),
            audit_entry("health", "-H", "-j", DEV1),
            audit_entry("health", "-H", "-j", DEV0, applied=False),
            audit_entry("scan", "--scan-open"),
            {"unparsable": "x"},
        ]
    }
    assert [e["argv"] for e in smart.served_calls(audit, "health", DEV0)] == [
        ["-H", "-j", DEV0]
    ]
    assert len(smart.served_calls(audit, "health")) == 2
    assert len(smart.served_calls(audit, "scan")) == 1
    assert smart.smart_errors(
        {"batches": [{"collection_errors": ["_smart: x", "_bmc: y"]}]}
    ) == ["_smart: x"]

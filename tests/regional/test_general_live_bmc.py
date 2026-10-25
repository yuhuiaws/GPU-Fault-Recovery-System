"""GF-GENERAL-LIVE-015 BMC variants over the shared host source replay.

No cluster, no node: a scripted source-replay ``FakeProbe`` receives the
``ipmitool`` plan (installed with ``--allow-missing-real``: EC2 has no BMC) and
answers the wrapper journal; a scripted store plays the product's records in
three phases -- the replayed table, the healthy table after the replan, and
after the window closes. The tests pin the reviewed tables, the QUARANTINE
chain and its recovery/restore order, the zero-critical and UNKNOWN gradings,
the ``critical`` variant's expected FAIL on sensor identity (the registered
product gap), and the read-only capability record of ``unsupported_bmc``.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import general_live_bmc as bmc
from scripts.e2e.regional.general_live_fixture import (
    VariantBlocked,
    VariantInconclusive,
)
from scripts.e2e.regional.general_live_registry import (
    CAPABILITY,
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
SPECS = {spec.variant: spec for spec in bmc.VARIANTS}
T1 = "2026-10-24T00:02:00+00:00"
T2 = "2026-10-24T00:03:00+00:00"
T3 = "2026-10-24T00:04:00+00:00"


# --------------------------------------------------------------------------- #
# scripted product records
# --------------------------------------------------------------------------- #
def reading(value: float, **labels: str) -> dict[str, Any]:
    return {"name": bmc.METRIC, "device": None, "value": value, "labels": labels}


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
        "metric_latest": [],
        "collector_statuses": [],
    }
    document.update(extra)
    return document


def silent() -> dict[str, Any]:
    """A BMC-less node after the window: host batches with no BMC reading."""

    return doc(batch(T3))


def survey(ipmitool: bool = False) -> dict[str, Any]:
    return {
        "cpu_count": 8,
        "tools": {"ipmitool": ipmitool, "systemd-run": True},
        "modules": {},
        "backing": [],
        "collector_env": {
            "GPU_FAULT_HOST_INTERVAL_SECONDS": "15",
            "GPU_FAULT_HOST_HEALTH_SUMMARY_SECONDS": "300",
        },
        "residue": {"units": [], "paused_units": {}, "paused_processes": {}},
    }


class Scenario:
    """Records during the replayed table, after the healthy replan, after close."""

    def __init__(
        self,
        during: dict[str, Any],
        recovered: dict[str, Any] | None = None,
        after: dict[str, Any] | None = None,
    ) -> None:
        self.phases = {
            "during": during,
            "recovered": during if recovered is None else recovered,
            "after": silent() if after is None else after,
        }
        self.phase = "during"
        self.plans = 0
        self.events: list[str] = []

    def activity(self, _script: str, *_arguments: str) -> dict[str, Any]:
        return self.phases[self.phase]

    def install_source(self, arguments: tuple[str, ...]) -> dict[str, Any]:
        if "plan-ipmitool.json" in arguments:
            self.plans += 1
            if self.plans >= 2:
                self.phase = "recovered"
                self.events.append("replan")
        return {"bytes": 1}

    def close(self, _arguments: tuple[str, ...]) -> dict[str, Any]:
        self.phase = "after"
        self.events.append("close-window")
        return {"clean": True, "after": {"ActiveState": "active"}}


def audit_entry(applied: bool = True) -> dict[str, Any]:
    return {
        "query": "sensor",
        "argv": ["sensor"],
        "applied_rules": [0] if applied else [],
        "real": None,
        "served_returncode": 0,
        "proof_scope": "live-replay",
    }


def source_probe(scenario: Scenario, entries: list[dict[str, Any]]) -> FakeProbe:
    return FakeProbe(
        {
            "install-source": scenario.install_source,
            "install-wrapper": {
                "shim": "/run/gfrs-live/source-x/bin/ipmitool",
                "real": None,
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
    ipmitool: bool = False,
) -> tuple[dict[str, Any], FakeProbe]:
    spec = SPECS[variant]
    node = FakeProbe({"survey": survey(ipmitool)})
    source = source_probe(scenario, [audit_entry()] if entries is None else entries)
    regional = FakeRegional(cpu_python=scenario.activity)
    ctx = make_context(
        spec,
        tmp_path=tmp_path,
        regional=regional,
        probes={f"node@{NODE}": node, f"source-replay@{NODE}": source},
        settings=make_settings(spec, node=NODE),
    )
    return spec.execute(ctx), source


def sent_plans(source: FakeProbe) -> list[dict[str, Any]]:
    plans = []
    for call in source.calls:
        if call[0] == "install-source" and "plan-ipmitool.json" in call:
            flags = dict(zip(call[1::2], call[2::2]))
            plans.append(json.loads(base64.b64decode(flags["--b64"])))
    return plans


def restore_recorder(monkeypatch: pytest.MonkeyPatch, scenario: Scenario) -> list:
    restored: list[list[str]] = []

    def record(ctx: Any, incidents: Any, reason: str) -> list[dict[str, Any]]:
        scenario.events.append("restore")
        restored.append([item["incident_id"] for item in incidents])
        return [{"status": "SUCCEEDED"}]

    monkeypatch.setattr(bmc, "restore_incidents", record)
    return restored


# --------------------------------------------------------------------------- #
# the registry and the reviewed tables
# --------------------------------------------------------------------------- #
def test_every_015_variant_is_registered_on_both_lanes() -> None:
    assert sorted(SPECS) == sorted(coverage_rows()["GF-GENERAL-LIVE-015"])
    for spec in bmc.VARIANTS:
        assert spec.lanes == ("eks", "hp"), spec.key
        if spec.variant == "healthy":
            assert spec.proof_mode == CAPABILITY
            assert spec.deferred.startswith("CAPABILITY_ABSENT:"), spec.deferred
        elif spec.variant == "unsupported_bmc":
            assert spec.proof_mode == OBSERVATION and spec.implemented
        else:
            assert spec.proof_mode == LIVE_REPLAY and spec.implemented, spec.key
            assert spec.approvals == (HOST_SOURCE_REPLAY,), spec.key
    assert SPECS["critical"].expected_outcome == "FAIL"
    assert "_bmc" in SPECS["critical"].product_gap
    assert SPECS["nonrecoverable"].expected_outcome == "PASS"


def statuses(table: str) -> list[str]:
    """The 4th column of every row with at least four, as the product reads it."""

    result = []
    for line in table.splitlines():
        columns = [item.strip().lower() for item in line.split("|")]
        if len(columns) >= 4:
            result.append(columns[3])
    return result


def test_tables_carry_the_intended_fourth_column_statuses() -> None:
    assert statuses(bmc.HEALTHY_TABLE).count("cr") == 0
    assert statuses(bmc.CRITICAL_TABLE).count("cr") == len(bmc.CRITICAL_SENSORS) == 2
    assert statuses(bmc.NONRECOVERABLE_TABLE).count("nr") == 1
    assert all(item not in {"cr", "nr"} for item in statuses(bmc.WRONG_COLUMN_TABLE)), (
        "the wrong-column table has no alert in its status column"
    )
    assert "cr" in bmc.WRONG_COLUMN_TABLE and "| nr |" not in bmc.WRONG_COLUMN_TABLE
    assert any(
        len(line.split("|")) == 3 for line in bmc.WRONG_COLUMN_TABLE.splitlines()
    ), "the wrong-column table carries a three-column row"
    assert statuses(bmc.UNAVAILABLE_TABLE).count("na") == 1
    assert f"{bmc.UNAVAILABLE_SENSOR} | na |" in bmc.UNAVAILABLE_TABLE
    for table in (
        bmc.HEALTHY_TABLE,
        bmc.CRITICAL_TABLE,
        bmc.NONRECOVERABLE_TABLE,
        bmc.WRONG_COLUMN_TABLE,
        bmc.UNAVAILABLE_TABLE,
    ):
        wrapper.build_plan("ipmitool", bmc.table_rules(table), not_after=10.0)
    wrapper.build_plan("ipmitool", bmc.command_failure_rules(), not_after=10.0)


# --------------------------------------------------------------------------- #
# critical / nonrecoverable: QUARANTINE, healthy table, close, restore
# --------------------------------------------------------------------------- #
def quarantine_doc(count: float) -> dict[str, Any]:
    inc = incident("inc-b", bmc.METRIC, state="QUARANTINED", workflow="wf-b")
    wf = workflow(
        "wf-b",
        "inc-b",
        ["FREEZE_EVIDENCE", "MARK_UNSCHEDULABLE", "QUARANTINE"],
        status="SUCCEEDED",
    )
    return doc(batch(T1, reading(count)), incidents=[inc], workflows=[wf])


def recovered_doc(count: float) -> dict[str, Any]:
    document = quarantine_doc(count)
    document["batches"] += [batch(T2, reading(0.0)), batch(T3, reading(0.0))]
    return document


def test_nonrecoverable_quarantines_recovers_on_the_healthy_table_then_restores(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scenario = Scenario(quarantine_doc(1.0), recovered_doc(1.0))
    restored = restore_recorder(monkeypatch, scenario)
    result, source = run("nonrecoverable", scenario, tmp_path)
    assert result["verdict"] == "PASS", result["errors"]
    plans = sent_plans(source)
    assert [plan["rules"][0]["stdout"] for plan in plans] == [
        bmc.NONRECOVERABLE_TABLE,
        bmc.HEALTHY_TABLE,
    ], "the critical table first, the healthy table as the recovery"
    wrap = next(call for call in source.calls if call[0] == "install-wrapper")
    assert "--allow-missing-real" in wrap and "ipmitool" in wrap, wrap
    opened = next(call for call in source.calls if call[0] == "open-window")
    assert opened[1:7] == (
        "--mode",
        "command",
        "--restore-seconds",
        "1920",
        "--wrapper",
        "ipmitool",
    )
    assert scenario.events == ["replan", "close-window", "restore"], scenario.events
    assert restored == [["inc-b"]]
    assert result["proof_scope"] == "live-replay" and result["counts"] == [1.0]


def test_critical_is_the_registered_sensor_identity_gap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scenario = Scenario(quarantine_doc(2.0), recovered_doc(2.0))
    restore_recorder(monkeypatch, scenario)
    result, source = run("critical", scenario, tmp_path)
    assert result["verdict"] == "FAIL"
    assert [error.split(":", 1)[0] for error in result["errors"]] == [
        "sensor_identity",
        "sensor_identity",
    ], result["errors"]
    assert all(
        not result["stages"][stage]
        for stage in ("chain", "count", "recovered", "window_closed")
    ), result["stages"]
    assert sent_plans(source)[0]["rules"][0]["stdout"] == bmc.CRITICAL_TABLE
    assert scenario.events == ["replan", "close-window", "restore"]


def test_critical_passes_once_the_product_names_the_sensors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    during = quarantine_doc(2.0)
    during["batches"][0]["samples"][0]["labels"] = {"sensors": "CPU1 Temp,Fan1"}
    scenario = Scenario(during, recovered_doc(2.0))
    restore_recorder(monkeypatch, scenario)
    result, _source = run("critical", scenario, tmp_path)
    assert result["verdict"] == "PASS", result["errors"]


def test_isolation_fails_on_a_wrong_count_or_a_lingering_critical_reading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scenario = Scenario(quarantine_doc(3.0), recovered_doc(3.0))
    restore_recorder(monkeypatch, scenario)
    result, _source = run("nonrecoverable", scenario, tmp_path)
    assert any(error.startswith("count:") for error in result["errors"]), result
    lingering = Scenario(
        quarantine_doc(1.0), recovered_doc(1.0), after=doc(batch(T3, reading(1.0)))
    )
    restore_recorder(monkeypatch, lingering)
    result, _source = run("nonrecoverable", lingering, tmp_path / "b")
    assert any(error.startswith("window_closed:") for error in result["errors"]), result


def test_isolation_is_inconclusive_when_the_table_was_never_served(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scenario = Scenario(quarantine_doc(1.0), recovered_doc(1.0))
    restore_recorder(monkeypatch, scenario)
    with pytest.raises(VariantInconclusive, match="never had the ipmitool sensor rule"):
        run("nonrecoverable", scenario, tmp_path, entries=[audit_entry(applied=False)])


# --------------------------------------------------------------------------- #
# wrong_column / unavailable_sensor / command_failure
# --------------------------------------------------------------------------- #
def test_wrong_column_and_unavailable_sensor_grade_zero(tmp_path: Path) -> None:
    for variant, table in (
        ("wrong_column", bmc.WRONG_COLUMN_TABLE),
        ("unavailable_sensor", bmc.UNAVAILABLE_TABLE),
    ):
        result, source = run(
            variant, Scenario(doc(batch(T1, reading(0.0)))), tmp_path / variant
        )
        assert result["verdict"] == "PASS", (variant, result["errors"])
        assert sent_plans(source)[0]["rules"][0]["stdout"] == table, variant
        assert "close-window" in [call[0] for call in source.calls], variant


def test_a_zero_table_graded_critical_or_unknown_fails(tmp_path: Path) -> None:
    graded = doc(
        batch(T1, reading(1.0)),
        incidents=[incident("inc-w", bmc.METRIC, state="QUARANTINED")],
    )
    result, _source = run("wrong_column", Scenario(graded), tmp_path)
    stages = {error.split(":", 1)[0] for error in result["errors"]}
    assert {"zero_critical", "no_finding"} <= stages, result["errors"]
    unknown = doc(
        batch(
            T1,
            errors=(
                "_bmc: HostObservationError: BMC sensor query returned no valid sensor rows",
            ),
        )
    )
    result, _source = run("unavailable_sensor", Scenario(unknown), tmp_path / "b")
    stages = {error.split(":", 1)[0] for error in result["errors"]}
    assert {"sample_present", "no_error"} <= stages, result["errors"]


def test_command_failure_is_unknown_without_a_sample(tmp_path: Path) -> None:
    during = doc(
        batch(T1, errors=("_bmc: HostObservationError: BMC sensor query failed",))
    )
    result, source = run("command_failure", Scenario(during), tmp_path)
    assert result["verdict"] == "PASS", result["errors"]
    assert sent_plans(source)[0]["rules"] == [
        {
            "query": "sensor",
            "action": "exit",
            "exit_code": 1,
            "stderr": bmc.IPMI_NO_DEVICE_STDERR,
        }
    ]
    faked = doc(
        batch(
            T1,
            reading(0.0),
            errors=("_bmc: HostObservationError: BMC sensor query failed",),
        )
    )
    result, _source = run("command_failure", Scenario(faked), tmp_path / "b")
    assert any(error.startswith("no_sample:") for error in result["errors"]), result
    healthy = doc(batch(T1, reading(0.0)))
    result, _source = run("command_failure", Scenario(healthy), tmp_path / "c")
    assert any(error.startswith("unknown_error:") for error in result["errors"]), result


# --------------------------------------------------------------------------- #
# unsupported_bmc: the read-only capability record
# --------------------------------------------------------------------------- #
def test_unsupported_bmc_records_the_capability_gap(tmp_path: Path) -> None:
    result, source = run("unsupported_bmc", Scenario(silent()), tmp_path)
    assert result["verdict"] == "PASS", result["errors"]
    assert result["capability"] == {
        "ipmitool_installed": False,
        "bmc_samples": 0,
        "bmc_collection_errors": 0,
        "batches": 1,
    }
    assert any("capability record" in note for note in result["notes"]), result["notes"]
    assert source.calls == [], "a read-only observation touches no source window"
    assert (
        tmp_path / "cases" / SPECS["unsupported_bmc"].key / "bmc-capability.json"
    ).is_file(), "the capability record is kept as evidence"


def test_unsupported_bmc_fails_on_a_permanent_error_and_blocks_on_a_real_bmc(
    tmp_path: Path,
) -> None:
    erroring = doc(
        batch(T1, errors=("_bmc: HostObservationError: BMC sensor query failed",))
    )
    result, _source = run(
        "unsupported_bmc", Scenario(erroring), tmp_path, ipmitool=True
    )
    assert result["verdict"] == "FAIL"
    assert any(error.startswith("no_permanent_error:") for error in result["errors"]), (
        result
    )
    with pytest.raises(VariantBlocked, match="answers ipmitool sensor"):
        run(
            "unsupported_bmc",
            Scenario(doc(batch(T1, reading(0.0)))),
            tmp_path / "b",
            ipmitool=True,
        )


def test_sensor_identity_looks_in_labels_incidents_markers_and_mail() -> None:
    bare = doc(batch(T1, reading(2.0)))
    assert len(bmc.sensor_identity(bare, ("CPU1 Temp", "Fan1"))) == 2
    named = doc(
        batch(T1, reading(2.0)), markers=[{"text": "critical sensors: CPU1 Temp, Fan1"}]
    )
    assert bmc.sensor_identity(named, ("CPU1 Temp", "Fan1")) == []
    assert bmc.bmc_errors(
        {"batches": [{"collection_errors": ["_smart: x", "_bmc: y"]}]}
    ) == ["_bmc: y"]

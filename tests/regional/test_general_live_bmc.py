"""GF-GENERAL-LIVE-015 BMC variants over the shared host source replay.

No cluster, no node: a scripted source-replay ``FakeProbe`` receives the
``ipmitool`` plan (installed with ``--allow-missing-real``: EC2 has no BMC) and
answers the wrapper journal; a scripted store plays the product's records in
three phases -- the replayed table, the healthy table after the replan, and
after the window closes. The tests pin the reviewed tables, the QUARANTINE
chain and its recovery/restore order, the sensor identity the ``critical``
variant demands of the sample labels, the incident reasons and the marker,
the zero-critical, UNKNOWN and NOT_SUPPORTED gradings, the read-only
capability record of ``unsupported_bmc``, and where the capability marker is
read from: the accepted batches, or the product's latest record when no batch
in the read carries it (live 2026-10-26 on both plain-EKS lanes).
"""

from __future__ import annotations

import base64
import json
from collections.abc import Callable
from datetime import datetime, timedelta
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


def unsupported(reason: str) -> dict[str, Any]:
    return {
        "name": bmc.UNSUPPORTED,
        "device": None,
        "value": 1.0,
        "labels": {"reason": reason},
    }


def latest(reason: str, observed_at: datetime) -> dict[str, Any]:
    """The product's ``telemetry_metric_latest`` row of the marker, as the probe lists it."""

    return {
        "name": bmc.UNSUPPORTED,
        "device": None,
        "value": 1.0,
        "observed_at": observed_at.isoformat(),
        "labels": {"reason": reason},
    }


SENSOR_ALERTS = (
    "CPU1 Temp=96.000 degrees C [cr; unc 90.000, ucr 95.000, unr 100.000]; "
    "Fan1=0.000 RPM [cr; lcr 300.000, lnc 500.000]"
)
NAMED_REASON = f"BMC reports 2 critical sensors: {SENSOR_ALERTS}"
BARE_REASON = "BMC reports a critical sensor"


def marker(incident_id: str, raw_reason: str) -> dict[str, Any]:
    return {
        "marker_id": f"marker-{incident_id}",
        "incident_id": incident_id,
        "raw_reason": raw_reason,
        "severity": "critical",
        "recommended_action": "QUARANTINE",
    }


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


Phase = dict[str, Any] | Callable[[datetime], dict[str, Any]]


class Scenario:
    """Records during the replayed table, after the healthy replan, after close.

    A phase is the document itself, or a callable given the read's ``since``
    (the activity probe's third argument) for records whose place relative to
    the read matters -- the latest-record readers count only a record observed
    inside the read, and a finding only when it was minted inside it.
    """

    def __init__(
        self, during: Phase, recovered: Phase | None = None, after: Phase | None = None
    ) -> None:
        self.phases: dict[str, Phase] = {
            "during": during,
            "recovered": during if recovered is None else recovered,
            "after": silent() if after is None else after,
        }
        self.phase = "during"
        self.plans = 0
        self.events: list[str] = []

    def activity(self, _script: str, *arguments: str) -> dict[str, Any]:
        phase = self.phases[self.phase]
        if callable(phase):
            return phase(datetime.fromisoformat(arguments[2]))
        return phase

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
    assert all(spec.expected_outcome == "PASS" for spec in bmc.VARIANTS), [
        (spec.variant, spec.expected_outcome) for spec in bmc.VARIANTS
    ]
    assert all(not spec.product_gap for spec in bmc.VARIANTS), (
        "the sensor identity gap is closed in the product; no variant cites one"
    )


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
    wrapper.build_plan("ipmitool", bmc.no_device_rules(), not_after=10.0)
    assert bmc.command_failure_rules()[0]["stderr"] == bmc.IPMI_TRANSIENT_STDERR
    assert bmc.no_device_rules()[0]["stderr"] == bmc.IPMI_NO_DEVICE_STDERR
    assert "Could not open device" in bmc.IPMI_NO_DEVICE_STDERR
    assert "Could not open device" not in bmc.IPMI_TRANSIENT_STDERR


# --------------------------------------------------------------------------- #
# critical / nonrecoverable: QUARANTINE, healthy table, close, restore
# --------------------------------------------------------------------------- #
def quarantine_doc(
    count: float,
    *,
    labels: dict[str, str] | None = None,
    reason: str = NAMED_REASON,
    marker_reason: str | None = None,
) -> dict[str, Any]:
    """The product's records for a critical table: by default the sample labels,
    the incident reason and the marker all name the sensors (the fixed product)."""

    inc = incident("inc-b", bmc.METRIC, state="QUARANTINED", workflow="wf-b")
    inc["reasons"] = [reason]
    wf = workflow(
        "wf-b",
        "inc-b",
        ["FREEZE_EVIDENCE", "MARK_UNSCHEDULABLE", "QUARANTINE"],
        status="SUCCEEDED",
    )
    sample = reading(
        count,
        **(
            {"sensors": "CPU1 Temp, Fan1", "sensor_alerts": SENSOR_ALERTS}
            if labels is None
            else labels
        ),
    )
    return doc(
        batch(T1, sample),
        incidents=[inc],
        workflows=[wf],
        markers=[marker("inc-b", reason if marker_reason is None else marker_reason)],
    )


def recovered_doc(count: float, **named: Any) -> dict[str, Any]:
    document = quarantine_doc(count, **named)
    document["batches"] += [batch(T2, reading(0.0)), batch(T3, reading(0.0))]
    return document


def test_nonrecoverable_quarantines_recovers_on_the_healthy_table_then_restores(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    nr_reason = (
        "BMC reports 1 critical sensor: CPU1 Temp=104.000 degrees C "
        "[nr; unc 90.000, ucr 95.000, unr 100.000]"
    )
    scenario = Scenario(
        quarantine_doc(1.0, labels={"sensors": "CPU1 Temp"}, reason=nr_reason),
        recovered_doc(1.0, labels={"sensors": "CPU1 Temp"}, reason=nr_reason),
    )
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


def test_critical_passes_when_labels_reasons_and_marker_name_the_sensors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scenario = Scenario(quarantine_doc(2.0), recovered_doc(2.0))
    restore_recorder(monkeypatch, scenario)
    result, source = run("critical", scenario, tmp_path)
    assert result["verdict"] == "PASS", result["errors"]
    assert result["stages"]["sensor_identity"] == []
    assert sent_plans(source)[0]["rules"][0]["stdout"] == bmc.CRITICAL_TABLE
    assert scenario.events == ["replan", "close-window", "restore"]


def test_critical_fails_where_a_sensor_goes_unnamed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A bare count (the old product) fails sensor_identity in all three places;
    a label without the reason, or a reason without the marker, fails that place."""

    bare = Scenario(
        quarantine_doc(2.0, labels={}, reason=BARE_REASON),
        recovered_doc(2.0, labels={}, reason=BARE_REASON),
    )
    restore_recorder(monkeypatch, bare)
    result, _source = run("critical", bare, tmp_path)
    assert result["verdict"] == "FAIL"
    assert [error.split(":", 1)[0] for error in result["errors"]] == [
        "sensor_identity"
    ] * 6, result["errors"]
    assert sum("'CPU1 Temp'" in error for error in result["errors"]) == 3
    assert sum("'Fan1'" in error for error in result["errors"]) == 3
    assert all(
        not result["stages"][stage]
        for stage in ("chain", "count", "recovered", "window_closed")
    ), result["stages"]

    unreasoned = Scenario(
        quarantine_doc(2.0, reason=BARE_REASON, marker_reason=NAMED_REASON),
        recovered_doc(2.0, reason=BARE_REASON, marker_reason=NAMED_REASON),
    )
    restore_recorder(monkeypatch, unreasoned)
    result, _source = run("critical", unreasoned, tmp_path / "b")
    assert [error.split(":", 1)[0] for error in result["errors"]] == [
        "sensor_identity"
    ] * 2, result["errors"]
    assert all("reasons of its incident" in error for error in result["errors"]), (
        result["errors"]
    )

    unmarked = Scenario(
        quarantine_doc(2.0, marker_reason=BARE_REASON),
        recovered_doc(2.0, marker_reason=BARE_REASON),
    )
    restore_recorder(monkeypatch, unmarked)
    result, _source = run("critical", unmarked, tmp_path / "c")
    assert all("marker" in error for error in result["errors"]), result["errors"]
    assert len(result["errors"]) == 2, result["errors"]


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


UNKNOWN_BATCH = batch(
    T1, errors=("_bmc: HostObservationError: BMC sensor query failed",)
)


def no_device_doc(*extra_batches: dict[str, Any]) -> dict[str, Any]:
    """After the no-device replan: the NO_DEVICE marker, the error gone."""

    return doc(UNKNOWN_BATCH, batch(T2, unsupported("NO_DEVICE")), *extra_batches)


def test_command_failure_is_unknown_then_a_missing_device_is_not_supported(
    tmp_path: Path,
) -> None:
    scenario = Scenario(doc(UNKNOWN_BATCH), no_device_doc())
    result, source = run("command_failure", scenario, tmp_path)
    assert result["verdict"] == "PASS", result["errors"]
    assert [plan["rules"] for plan in sent_plans(source)] == [
        [
            {
                "query": "sensor",
                "action": "exit",
                "exit_code": 1,
                "stderr": bmc.IPMI_TRANSIENT_STDERR,
            }
        ],
        [
            {
                "query": "sensor",
                "action": "exit",
                "exit_code": 1,
                "stderr": bmc.IPMI_NO_DEVICE_STDERR,
            }
        ],
    ], "the BMC that does not answer first, the BMC-less host second"
    assert scenario.events == ["replan", "close-window"], scenario.events
    assert result["no_device_markers"][-1]["labels"] == {"reason": "NO_DEVICE"}


def test_command_failure_fails_on_a_sample_a_missing_error_or_an_unsupported_marker(
    tmp_path: Path,
) -> None:
    faked = doc(
        batch(
            T1,
            reading(0.0),
            errors=("_bmc: HostObservationError: BMC sensor query failed",),
        )
    )
    result, _source = run("command_failure", Scenario(faked, no_device_doc()), tmp_path)
    assert any(error.startswith("no_sample:") for error in result["errors"]), result
    healthy = doc(batch(T1, reading(0.0)))
    result, _source = run(
        "command_failure", Scenario(healthy, no_device_doc()), tmp_path / "b"
    )
    assert any(error.startswith("unknown_error:") for error in result["errors"]), result
    unsupported_instead = doc(batch(T1, unsupported("NO_DEVICE")))
    result, _source = run(
        "command_failure",
        Scenario(unsupported_instead, no_device_doc()),
        tmp_path / "c",
    )
    stages = {error.split(":", 1)[0] for error in result["errors"]}
    assert {"unknown_error", "not_unsupported"} <= stages, result["errors"]


def test_command_failure_fails_when_the_missing_device_stays_an_error(
    tmp_path: Path,
) -> None:
    """The old product: ipmitool's 'Could not open device' every tick was a
    permanent _bmc collection error and never a capability marker."""

    permanent = doc(UNKNOWN_BATCH, batch(T2, errors=UNKNOWN_BATCH["collection_errors"]))
    result, _source = run(
        "command_failure", Scenario(doc(UNKNOWN_BATCH), permanent), tmp_path
    )
    assert result["verdict"] == "FAIL"
    stages = {error.split(":", 1)[0] for error in result["errors"]}
    assert {"no_device_unsupported", "no_device_no_error"} <= stages, result["errors"]

    marked_but_erroring = doc(
        UNKNOWN_BATCH,
        batch(T2, unsupported("NO_DEVICE"), errors=UNKNOWN_BATCH["collection_errors"]),
    )
    result, _source = run(
        "command_failure",
        Scenario(doc(UNKNOWN_BATCH), marked_but_erroring),
        tmp_path / "b",
    )
    assert any(error.startswith("no_device_no_error:") for error in result["errors"]), (
        result["errors"]
    )
    graded_zero = no_device_doc(batch(T3, reading(0.0)))
    result, _source = run(
        "command_failure", Scenario(doc(UNKNOWN_BATCH), graded_zero), tmp_path / "c"
    )
    assert any(
        error.startswith("no_device_no_sample:") for error in result["errors"]
    ), result["errors"]


def with_since(since: datetime, document: dict[str, Any]) -> dict[str, Any]:
    """A document the way the activity probe returns it: naming its read."""

    return {
        **document,
        "since": since.isoformat(),
        "cluster_id": "gpu-eks-cluster",
        "node_id": NODE,
    }


def test_command_failure_reads_the_marker_from_the_latest_record_when_no_batch_has_it(
    tmp_path: Path,
) -> None:
    """The live 2026-10-26 eks timeline: the one tick that carried NO_DEVICE was
    stamped a few seconds before the read's ``since`` and nothing after it was
    an edge, so the read held no batch; the product's latest record, moved by
    that tick, says NO_DEVICE and is admitted from one collector interval back."""

    def after_replan(since: datetime) -> dict[str, Any]:
        return with_since(
            since,
            doc(metric_latest=[latest("NO_DEVICE", since - timedelta(seconds=9))]),
        )

    result, _source = run(
        "command_failure", Scenario(doc(UNKNOWN_BATCH), after_replan), tmp_path
    )
    assert result["verdict"] == "PASS", result["errors"]
    assert result["no_device_markers"] == [], "no accepted batch carried the marker"
    witness = result["no_device_marker"]
    assert witness["source"] == "metric_latest", witness
    assert witness["latest"]["labels"] == {"reason": "NO_DEVICE"}
    assert any(
        note.startswith("the NO_DEVICE marker was read from metric_latest")
        for note in result["notes"]
    ), result["notes"]


def test_command_failure_fails_on_a_stale_latest_record_or_another_reason(
    tmp_path: Path,
) -> None:
    """A latest record older than one collector interval before the read is the
    node's state before the replan, not the answer to it; a fresh record with
    another reason is the wrong answer. Both name what the product has."""

    def stale(since: datetime) -> dict[str, Any]:
        return with_since(
            since,
            doc(metric_latest=[latest("NO_DEVICE", since - timedelta(seconds=16))]),
        )

    result, _source = run(
        "command_failure", Scenario(doc(UNKNOWN_BATCH), stale), tmp_path
    )
    assert result["verdict"] == "FAIL"
    (error,) = [e for e in result["errors"] if e.startswith("no_device_unsupported:")]
    assert "none of the 0 accepted batches" in error and "no latest" in error, error

    def wrong_reason(since: datetime) -> dict[str, Any]:
        return with_since(
            since,
            doc(metric_latest=[latest("TOOL_MISSING", since + timedelta(seconds=30))]),
        )

    result, _source = run(
        "command_failure", Scenario(doc(UNKNOWN_BATCH), wrong_reason), tmp_path / "b"
    )
    assert result["verdict"] == "FAIL"
    (error,) = [e for e in result["errors"] if e.startswith("no_device_unsupported:")]
    assert "reason='TOOL_MISSING'" in error, error
    assert result["no_device_marker"]["source"] is None


def test_command_failure_still_fails_on_a_finding_beside_a_latest_record_marker(
    tmp_path: Path,
) -> None:
    """The latest record proves the marker, never the grading: an incident the
    BMC metric minted inside the read fails the variant as before."""

    def minted(since: datetime) -> dict[str, Any]:
        inc = incident("inc-m", bmc.METRIC, state="QUARANTINED", workflow="wf-m")
        inc["created_at"] = (since + timedelta(seconds=20)).isoformat()
        inc["cluster_id"] = "gpu-eks-cluster"
        return with_since(
            since,
            doc(
                incidents=[inc],
                workflows=[
                    workflow("wf-m", "inc-m", ["FREEZE_EVIDENCE", "QUARANTINE"])
                ],
                metric_latest=[latest("NO_DEVICE", since + timedelta(seconds=10))],
            ),
        )

    result, _source = run(
        "command_failure", Scenario(doc(UNKNOWN_BATCH), minted), tmp_path
    )
    assert result["verdict"] == "FAIL"
    stages = {error.split(":", 1)[0] for error in result["errors"]}
    assert stages == {"no_device_no_finding"}, result["errors"]
    assert result["no_device_marker"]["source"] == "metric_latest"


# --------------------------------------------------------------------------- #
# unsupported_bmc: the read-only capability record
# --------------------------------------------------------------------------- #
def test_unsupported_bmc_records_the_product_capability_marker(tmp_path: Path) -> None:
    """No ipmitool: the product says TOOL_MISSING on its host batches, nothing else."""

    recorded = doc(batch(T1, unsupported("TOOL_MISSING")))
    result, source = run("unsupported_bmc", Scenario(recorded), tmp_path)
    assert result["verdict"] == "PASS", result["errors"]
    assert result["capability"] == {
        "ipmitool_installed": False,
        "bmc_samples": 0,
        "bmc_collection_errors": 0,
        "unsupported_reasons": ["TOOL_MISSING"],
        "batches": 1,
        "marker_source": "evidence",
        "latest_marker": None,
    }
    assert any("capability record" in note for note in result["notes"]), result["notes"]
    assert source.calls == [], "a read-only observation touches no source window"
    assert (
        tmp_path / "cases" / SPECS["unsupported_bmc"].key / "bmc-capability.json"
    ).is_file(), "the capability record is kept as evidence"

    with_tool = doc(batch(T1, unsupported("NO_DEVICE")))
    result, _source = run(
        "unsupported_bmc", Scenario(with_tool), tmp_path / "b", ipmitool=True
    )
    assert result["verdict"] == "PASS", result["errors"]
    assert result["capability"]["unsupported_reasons"] == ["NO_DEVICE"]


def test_unsupported_bmc_reads_the_steady_marker_from_the_latest_record(
    tmp_path: Path,
) -> None:
    """The live 2026-10-26 eksb timeline: zero batches in a whole summary period
    (the steady marker is no edge and the summary is not evidence); the latest
    record the summary moved inside the read is the capability record."""

    def summary_only(since: datetime) -> dict[str, Any]:
        return with_since(
            since,
            doc(metric_latest=[latest("TOOL_MISSING", since + timedelta(seconds=290))]),
        )

    result, _source = run("unsupported_bmc", Scenario(summary_only), tmp_path)
    assert result["verdict"] == "PASS", result["errors"]
    assert result["capability"]["batches"] == 0
    assert result["capability"]["marker_source"] == "metric_latest"
    assert result["capability"]["unsupported_reasons"] == ["TOOL_MISSING"]
    assert result["capability"]["latest_marker"]["labels"] == {"reason": "TOOL_MISSING"}
    assert any(
        note.startswith("the marker was read from metric_latest")
        for note in result["notes"]
    ), result["notes"]


def test_unsupported_bmc_fails_when_the_latest_record_predates_the_read(
    tmp_path: Path,
) -> None:
    """A record from before the read is not this window's witness."""

    def stale(since: datetime) -> dict[str, Any]:
        return with_since(
            since,
            doc(metric_latest=[latest("TOOL_MISSING", since - timedelta(seconds=60))]),
        )

    result, _source = run("unsupported_bmc", Scenario(stale), tmp_path)
    assert result["verdict"] == "FAIL"
    (error,) = result["errors"]
    assert error.startswith("capability_marker:") and "no latest" in error, error
    assert result["capability"]["marker_source"] is None


def test_unsupported_bmc_fails_without_the_marker_or_with_a_permanent_error(
    tmp_path: Path,
) -> None:
    """A silent product (no marker at all), the wrong marker for the node, or
    the old per-tick collection error each fail the capability record."""

    result, _source = run("unsupported_bmc", Scenario(silent()), tmp_path)
    assert result["verdict"] == "FAIL"
    assert [error.split(":", 1)[0] for error in result["errors"]] == [
        "capability_marker"
    ], result["errors"]
    assert "reason=TOOL_MISSING" in result["errors"][0]
    assert "from nowhere" in " ".join(result["notes"]), result["notes"]

    wrong_reason = doc(batch(T1, unsupported("TOOL_MISSING")))
    result, _source = run(
        "unsupported_bmc", Scenario(wrong_reason), tmp_path / "b", ipmitool=True
    )
    assert any("reason=NO_DEVICE" in error for error in result["errors"]), result

    erroring = doc(
        batch(T1, errors=("_bmc: HostObservationError: BMC sensor query failed",))
    )
    result, _source = run(
        "unsupported_bmc", Scenario(erroring), tmp_path / "c", ipmitool=True
    )
    assert result["verdict"] == "FAIL"
    stages = {error.split(":", 1)[0] for error in result["errors"]}
    assert {"no_permanent_error", "capability_marker"} <= stages, result["errors"]


def test_unsupported_bmc_blocks_on_a_real_bmc(tmp_path: Path) -> None:
    with pytest.raises(VariantBlocked, match="answers ipmitool sensor"):
        run(
            "unsupported_bmc",
            Scenario(doc(batch(T1, reading(0.0)))),
            tmp_path,
            ipmitool=True,
        )


def test_unsupported_bmc_waits_one_summary_period_for_the_steady_marker(
    tmp_path: Path,
) -> None:
    """The marker rides the routine health summary: no edge announces it, and
    the latest record it moves is read within one summary period."""

    scenario = Scenario(doc(batch(T1, unsupported("TOOL_MISSING"))))
    spec = SPECS["unsupported_bmc"]
    node = FakeProbe({"survey": survey(False)})
    regional = FakeRegional(cpu_python=scenario.activity)
    waits: list[tuple[str, float]] = []
    ctx = make_context(
        spec,
        tmp_path=tmp_path,
        regional=regional,
        probes={f"node@{NODE}": node},
        settings=make_settings(spec, node=NODE),
    )
    original = ctx.wait_until

    def recording(
        accept: Any, *, timeout_seconds: float, poll_seconds: float, name: str
    ):
        waits.append((name, timeout_seconds))
        return original(
            accept,
            timeout_seconds=timeout_seconds,
            poll_seconds=poll_seconds,
            name=name,
        )

    ctx.wait_until = recording  # type: ignore[method-assign]
    result = spec.execute(ctx)
    assert result["verdict"] == "PASS", result["errors"]
    assert waits == [("bmc-capability", ctx.host_summary + ctx.host_interval * 8 + 60)]
    assert spec.max_minutes * 60 > waits[0][1], "the variant budget covers the wait"


def test_sensor_identity_checks_labels_incident_reasons_and_marker() -> None:
    bare = doc(batch(T1, reading(2.0)))
    assert len(bmc.sensor_identity(bare, ("CPU1 Temp", "Fan1"))) == 6, (
        "no incident and a bare count: every place misses every sensor"
    )
    assert bmc.sensor_identity(quarantine_doc(2.0), ("CPU1 Temp", "Fan1")) == []
    zero_only = doc(batch(T1, reading(0.0, sensors="CPU1 Temp, Fan1")))
    assert any(
        "sensors label" in error for error in bmc.sensor_identity(zero_only, ("Fan1",))
    ), "a zero reading's labels do not name a critical sensor"
    foreign = quarantine_doc(2.0)
    foreign["markers"] = [marker("inc-other", NAMED_REASON)]
    assert [
        error for error in bmc.sensor_identity(foreign, ("Fan1",)) if "marker" in error
    ], "a marker of another incident does not name this finding's sensors"
    assert bmc.bmc_errors(
        {"batches": [{"collection_errors": ["_smart: x", "_bmc: y"]}]}
    ) == ["_bmc: y"]
    assert bmc.unsupported_markers(
        doc(batch(T1, unsupported("NO_DEVICE"), unsupported("TOOL_MISSING"))),
        "NO_DEVICE",
    )[0]["labels"] == {"reason": "NO_DEVICE"}

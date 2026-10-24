"""Unit coverage of the GF-GENERAL-LIVE registry, verdict helpers and store reads."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from scripts.e2e.regional import general_live_registry as reg
from scripts.e2e.regional import general_live_store as store
from scripts.e2e.regional import general_live_verdicts as v


def spec(**overrides):
    base = dict(
        case_id="GF-GENERAL-LIVE-001",
        variant="cpu_busy",
        proof_mode=reg.NATIVE,
        lanes=("eks",),
        risk="L",
        injection="i",
        expected="e",
        cleanup="c",
        max_minutes=10,
        execute=lambda ctx: {"verdict": "PASS"},
    )
    base.update(overrides)
    return reg.VariantSpec(**base)


# --------------------------------------------------------------------------- #
# registry
# --------------------------------------------------------------------------- #
def test_variant_spec_properties() -> None:
    s = spec()
    assert s.key == "GF-GENERAL-LIVE-001.cpu_busy"
    assert s.number == 1
    assert s.confirmation == "GENERALLIVE001_EXECUTE"
    assert s.implemented is True
    described = s.describe()
    assert described["key"] == s.key and "execute" not in described


def test_deferred_variant_is_not_implemented() -> None:
    s = reg.deferred_variant(
        "GF-GENERAL-LIVE-004",
        "probe_error",
        proof_mode="live-replay",
        reason="FIXTURE_NOT_READY: x",
    )
    assert s.implemented is False and s.execute is None
    reg.validate_spec(s)


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"case_id": "GF-X-1"}, "bad case id"),
        ({"variant": "Bad Name"}, "bad variant"),
        ({"proof_mode": "nope"}, "unknown proof mode"),
        ({"lanes": ("mars",)}, "unknown lane"),
        ({"risk": "Z"}, "unknown risk"),
        ({"flags": "ZZ"}, "bad list flags"),
        ({"approvals": ("NOPE",)}, "unknown approvals"),
        ({"expected_outcome": "MAYBE"}, "expected outcome"),
        ({"expected_outcome": "FAIL"}, "must cite its product gap"),
        ({"max_minutes": 999}, "max_minutes is outside"),
        ({"proof_mode": "live-replay"}, "only native and observation"),
        ({"execute": None}, "needs execute"),
    ],
)
def test_validate_spec_rejects(overrides, match) -> None:
    with pytest.raises(reg.RegistryError, match=match):
        reg.validate_spec(spec(**overrides))


def test_validate_spec_deferred_reason_and_executor() -> None:
    with pytest.raises(reg.RegistryError, match="deferred reason"):
        reg.validate_spec(spec(deferred="NOPE: x", execute=None))
    with pytest.raises(reg.RegistryError, match="has no executor"):
        reg.validate_spec(spec(deferred="FIXTURE_NOT_READY: x"))


def test_build_registry_detects_missing_and_extra() -> None:
    rows = {"GF-GENERAL-LIVE-001": ["cpu_busy", "recovery"]}
    with pytest.raises(reg.RegistryError, match="missing="):
        reg.build_registry([spec()], rows=rows, categories=[1])
    with pytest.raises(reg.RegistryError, match="extra="):
        reg.build_registry(
            [spec(), spec(variant="recovery"), spec(variant="extra")],
            rows={"GF-GENERAL-LIVE-001": ["cpu_busy", "recovery"]},
            categories=[1],
        )
    with pytest.raises(reg.RegistryError, match="duplicate"):
        reg.build_registry(
            [spec(), spec()], rows={"GF-GENERAL-LIVE-001": ["cpu_busy"]}, categories=[1]
        )


def test_select_and_coverage_rows() -> None:
    registry = {spec().key: spec()}
    assert reg.select(registry, "GF-GENERAL-LIVE-001", "cpu_busy").variant == "cpu_busy"
    with pytest.raises(reg.RegistryError, match="no variant"):
        reg.select(registry, "GF-GENERAL-LIVE-001", "ghost")
    with pytest.raises(reg.RegistryError, match="not implemented"):
        reg.select(registry, "GF-GENERAL-LIVE-099", "x")
    rows = reg.coverage_rows()
    assert rows["GF-GENERAL-LIVE-001"][0] == "cpu_busy"


def test_full_registry_matches_coverage() -> None:
    registry = reg.load_registry()
    rows = reg.coverage_rows()
    for number in reg.PHASE_ONE_CATEGORIES:
        case = f"GF-GENERAL-LIVE-{number:03d}"
        for variant in rows[case]:
            assert f"{case}.{variant}" in registry


# --------------------------------------------------------------------------- #
# verdicts
# --------------------------------------------------------------------------- #
def _incident(metric, device="node", rule=None, state="ACTION_PENDING"):
    event = f"b-{metric}-{v.safe_device(device)}" + (f"-{rule.lower()}" if rule else "")
    return {
        "incident_id": f"inc-{metric}",
        "event_id": event,
        "event_type": "NODE_HEALTH",
        "state": state,
        "created_at": "2026-10-24T00:03:00+00:00",
    }


def _wf(incident_id, ops, status="RUNNING", execs=(), probe=None):
    steps = [
        {
            "operation": op,
            "parameters": (
                {"host_diagnostic_probe": {"metric": probe}}
                if probe and op.startswith("VALIDATE")
                else {}
            ),
        }
        for op in ops
    ]
    return {
        "request_id": "wf",
        "incident_id": incident_id,
        "status": status,
        "official_steps": steps,
        "step_executions": list(execs),
    }


def test_finding_incidents_device_and_rule() -> None:
    doc = {
        "incidents": [
            _incident("cpu_usage_percent"),
            _incident("load1_per_cpu", rule="LOW"),
        ]
    }
    assert v.finding_incidents(doc, "cpu_usage_percent")
    assert v.finding_incidents(doc, "load1_per_cpu", rule_id="LOW")
    assert not v.finding_incidents(doc, "load1_per_cpu", rule_id="OTHER")
    assert not v.finding_incidents(doc, "cpu_usage_percent", device="dm-7")


def test_chain_errors_present_and_absent() -> None:
    doc = {
        "incidents": [_incident("cpu_usage_percent")],
        "workflows": [
            _wf(
                "inc-cpu_usage_percent",
                ["FREEZE_EVIDENCE", "VALIDATE_HOST"],
                status="FAILED",
                execs=[
                    {
                        "operation": "VALIDATE_HOST",
                        "status": "FAILED",
                        "details": {"host_diagnostic_failure": True},
                    }
                ],
                probe="cpu_usage_percent",
            )
        ],
    }
    errors = v.chain_errors(
        doc,
        v.ChainExpectation(
            metric="cpu_usage_percent",
            operations=("FREEZE_EVIDENCE", "VALIDATE_HOST"),
            validation="VALIDATE_HOST",
            validation_must_fail=True,
            probe_metric="cpu_usage_percent",
            incident_states=frozenset({"ACTION_PENDING"}),
        ),
    )
    assert errors == []
    assert v.chain_errors(
        {"incidents": []}, v.ChainExpectation(metric="cpu_usage_percent")
    )
    assert v.chain_errors(
        doc, v.ChainExpectation(metric="cpu_usage_percent", present=False)
    )


def test_chain_errors_detects_every_shape() -> None:
    doc = {
        "incidents": [_incident("m", state="RECOVERED"), _incident("m")],
        "workflows": [
            _wf(
                "inc-m",
                ["FREEZE_EVIDENCE"],
                status="SUCCEEDED",
                execs=[{"operation": "VALIDATE_HOST", "status": "SUCCEEDED"}],
            )
        ],
    }
    errors = v.chain_errors(
        doc,
        v.ChainExpectation(
            metric="m",
            operations=("VALIDATE_HOST",),
            validation="VALIDATE_HOST",
            validation_must_fail=True,
            probe_metric="m",
            incident_states=frozenset({"ACTION_PENDING"}),
            single=True,
        ),
    )
    joined = " ".join(errors)
    assert "incidents (expected one)" in joined
    assert "lack" in joined
    assert "succeeded while the fault" in joined
    assert "no validation step carries" in joined
    doc2 = {"incidents": [_incident("m")], "workflows": []}
    assert "has no workflow" in " ".join(
        v.chain_errors(doc2, v.ChainExpectation(metric="m"))
    )


def test_validation_missing_and_failed_without_marker() -> None:
    doc = {
        "incidents": [_incident("m")],
        "workflows": [
            _wf(
                "inc-m",
                ["VALIDATE_HOST"],
                execs=[
                    {"operation": "VALIDATE_HOST", "status": "FAILED", "details": {}}
                ],
            )
        ],
    }
    assert "without host_diagnostic_failure" in " ".join(
        v.chain_errors(
            doc,
            v.ChainExpectation(
                metric="m", validation="VALIDATE_HOST", validation_must_fail=True
            ),
        )
    )
    doc2 = {
        "incidents": [_incident("m")],
        "workflows": [_wf("inc-m", ["FREEZE_EVIDENCE"])],
    }
    assert "never executed VALIDATE_HOST" in " ".join(
        v.chain_errors(doc2, v.ChainExpectation(metric="m", validation="VALIDATE_HOST"))
    )
    doc3 = {
        "incidents": [_incident("m")],
        "workflows": [
            _wf(
                "inc-m",
                ["VALIDATE_HOST"],
                execs=[
                    {"operation": "VALIDATE_HOST", "status": "FAILED", "error": "boom"}
                ],
            )
        ],
    }
    assert "ended FAILED" in " ".join(
        v.chain_errors(doc3, v.ChainExpectation(metric="m", validation="VALIDATE_HOST"))
    )


def test_hardware_isolation_settled_detection_notifications() -> None:
    doc = {
        "workflows": [
            _wf(
                "i",
                ["RESET_GPU", "MARK_UNSCHEDULABLE", "STOP_WORKLOADS"],
                status="SUCCEEDED",
            )
        ]
    }
    assert v.hardware_errors(doc)
    assert v.hardware_errors(doc, allowed={"RESET_GPU"}) == []
    assert set(x.split(":")[1] for x in v.isolation_operations(doc)) == {
        "MARK_UNSCHEDULABLE",
        "STOP_WORKLOADS",
    }
    assert v.settled({"workflows": [_wf("i", ["x"], status="SUCCEEDED")]}, ["i"])
    assert not v.settled({"workflows": [_wf("i", ["x"], status="RUNNING")]}, ["i"])
    assert (
        v.detection_seconds(
            {}, _incident("m"), datetime(2026, 10, 24, tzinfo=timezone.utc)
        )
        == 180.0
    )
    assert v.notification_errors(
        {"notifications": [{"incident_id": "i"}, {"incident_id": "i"}]},
        incident_ids=["i"],
    )
    assert (
        v.notification_errors(
            {"notifications": [{"incident_id": "i"}]}, incident_ids=["i"]
        )
        == []
    )


def test_oracle_and_sustained_and_node_isolated() -> None:
    assert v.oracle_values(
        [{"a": 1.0}, {"a": "x"}, {"a": float("nan")}], lambda r: r.get("a")
    ) == [1.0]
    assert v.sustained([1, 1, 1], lambda x: x >= 1, 3)
    assert not v.sustained([1, 0, 1], lambda x: x >= 1, 2)
    assert v.node_isolated({"unschedulable": True})
    assert v.node_isolated({"taints": [{"key": "gpu-fault.io/quarantined"}]})
    assert not v.node_isolated(
        {"unschedulable": False, "taints": [], "ownership_annotations": {}}
    )
    assert v.verdict({"a": []}) == "PASS"
    assert v.verdict({"a": ["x"]}) == "FAIL"
    assert v.verdict({"a": []}, inconclusive=["y"]) == "INCONCLUSIVE"


def test_samples_breaching_steps_and_subsequence() -> None:
    doc = {
        "batches": [
            {
                "observed_at": "t",
                "batch_id": "b",
                "samples": [{"name": "m", "device": "d", "value": 5.0}],
            }
        ]
    }
    assert v.samples(doc, "m", device="d")[0]["value"] == 5.0
    assert v.breaching({"value": 99.0}, 98.0)
    assert v.breaching({"value": 1.0}, 2.0, "lte")
    assert not v.breaching({"value": float("inf")}, 1.0)
    wf = _wf("i", ["VALIDATE_HOST"], probe="m")
    assert (
        v.step_parameters(wf, "VALIDATE_HOST")[0]["host_diagnostic_probe"]["metric"]
        == "m"
    )
    assert v.subsequence(["a", "c"], ["a", "b", "c"])
    assert not v.subsequence(["c", "a"], ["a", "b", "c"])
    assert (
        v.final_execution(
            _wf("i", ["X"], execs=[{"operation": "X", "status": "WAITING"}]), "X"
        )["status"]
        == "WAITING"
    )


# --------------------------------------------------------------------------- #
# store reads
# --------------------------------------------------------------------------- #
class Recorder:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def cpu_python(self, script, *arguments, **_):
        self.calls.append((script, arguments))
        return {"ok": True}


def test_store_read_helpers_pass_through_and_validate() -> None:
    rec = Recorder()
    store.preflight_state(rec, "c", "n")
    assert rec.calls[-1][0] is store.PREFLIGHT_PROBE
    store.node_activity(
        rec,
        "c",
        "n",
        datetime(2026, 10, 24, tzinfo=timezone.utc),
        metrics=["m"],
        signal_keys=["s"],
        kinds=["HOST_TELEMETRY"],
    )
    assert rec.calls[-1][0] is store.ACTIVITY_PROBE
    store.attempt_records(rec, "c", "job", ["a1"])
    assert rec.calls[-1][0] is store.ATTEMPT_PROBE
    assert store.signal_key("c", "n", "m", "dev", "RULE") == "c/n/m/dev/RULE"
    assert store.signal_key("c", "n", "m") == "c/n/m/node"
    with pytest.raises(ValueError):
        store.node_activity(rec, "c", "n", datetime.now(timezone.utc), metrics=["a,b"])
    with pytest.raises(ValueError):
        store.attempt_records(rec, "c", "job", [])

"""GF-GENERAL-LIVE-032 unknown error: category-module verdict tests.

Driven over fakes: a ``FakeProbe`` stands in for the kmsg-mode replay host and a
``FakeRegional`` answers both the XID policy-decision read and the NVIDIA_KERNEL
activity read, so the unknown-code -> SITE_SAFETY grading and the dedup of a
repeated event are checked on the product's own ``FaultPolicyDecision`` shape.
"""

from __future__ import annotations

from pathlib import Path

from scripts.e2e.regional import general_live_unknown as mod
from scripts.e2e.regional.general_live_registry import coverage_rows
from tests.regional._general_live_fakes import (
    FakeProbe,
    FakeRegional,
    make_context,
    make_settings,
)

NODE = "ip-10-0-43-5.us-west-2.compute.internal"


def _spec(variant: str) -> object:
    return next(item for item in mod.VARIANTS if item.variant == variant)


def _host_script() -> dict[str, object]:
    return {
        "install-records": {"records": 1},
        "open-window": {"after": {"ActiveState": "active"}},
        "arm": {"armed": True, "fixture": "/run/.../kmsg.fixture"},
        "status": {"injected": {"fixture_lines": 5}},
        "close-window": {"clean": True},
    }


def _cpu_python(decisions: list[dict[str, object]]):
    def responder(script: str, *arguments: str) -> dict[str, object]:
        if "list_xid_events" in script:
            return {"decisions": decisions}
        return {
            "incidents": [],
            "workflows": [],
            "markers": [],
            "notifications": [],
            "batches": [],
            "metric_latest": [],
            "collector_statuses": [],
        }

    return responder


def _decision(**overrides: object) -> dict[str, object]:
    base = {
        "event_id": "xid-987-1",
        "xid": mod.UNKNOWN_XID,
        "disposition": "SITE_SAFETY",
        "action": None,
        "requires_operator": True,
        "incident_id": "inc-xid-987",
        "workflow_request_id": None,
        "duplicate": False,
        "reasons": ["XID is absent from the pinned NVIDIA Catalog"],
    }
    base.update(overrides)
    return base


def _ctx(variant: str, *, decisions: list[dict[str, object]], tmp_path: Path):
    spec = _spec(variant)
    fake = FakeProbe(_host_script())
    regional = FakeRegional(cpu_python=_cpu_python(decisions))
    ctx = make_context(
        spec,
        tmp_path=tmp_path,
        regional=regional,
        probes={f"kernel-replay-kmsg@{NODE}": fake},
        settings=make_settings(spec, node=NODE),
        window_minutes=60,
    )
    return ctx


def test_unknown_xid_is_site_safety_without_action(tmp_path: Path) -> None:
    ctx = _ctx("unknown_xid", decisions=[_decision()], tmp_path=tmp_path)
    result = mod.unknown_xid(ctx)
    assert result["verdict"] == "PASS", result["errors"]
    assert result["proof_scope"] == "live-replay", result


def test_unknown_xid_fails_if_it_earns_an_nvidia_action(tmp_path: Path) -> None:
    ctx = _ctx(
        "unknown_xid",
        decisions=[_decision(disposition="EXECUTABLE", action="RESET_GPU")],
        tmp_path=tmp_path,
    )
    result = mod.unknown_xid(ctx)
    assert result["verdict"] == "FAIL", result
    assert any(
        "recovery action" in error or "SITE_SAFETY" in error
        for error in result["errors"]
    ), result["errors"]


def test_unknown_xid_inconclusive_without_injection(tmp_path: Path) -> None:
    from scripts.e2e.regional.general_live_fixture import VariantInconclusive

    spec = _spec("unknown_xid")
    fake = FakeProbe({**_host_script(), "status": {"injected": {"fixture_lines": 0}}})
    regional = FakeRegional(cpu_python=_cpu_python([]))
    ctx = make_context(
        spec,
        tmp_path=tmp_path,
        regional=regional,
        probes={f"kernel-replay-kmsg@{NODE}": fake},
        settings=make_settings(spec, node=NODE),
        window_minutes=60,
    )
    try:
        mod.unknown_xid(ctx)
    except VariantInconclusive as exc:
        assert "no reviewed records" in str(exc), str(exc)
    else:
        raise AssertionError("an empty kmsg fixture must be INCONCLUSIVE")


def test_duplicate_event_isolates_once(tmp_path: Path) -> None:
    decisions = [
        _decision(event_id="xid-987-1"),
        _decision(event_id="xid-987-2", duplicate=True, incident_id="inc-xid-987"),
    ]
    ctx = _ctx("duplicate", decisions=decisions, tmp_path=tmp_path)
    result = mod.duplicate(ctx)
    assert result["verdict"] == "PASS", result["errors"]


def test_duplicate_fails_on_second_isolation(tmp_path: Path) -> None:
    decisions = [
        _decision(event_id="xid-987-1", incident_id="inc-a"),
        _decision(event_id="xid-987-2", incident_id="inc-b"),
    ]
    ctx = _ctx("duplicate", decisions=decisions, tmp_path=tmp_path)
    result = mod.duplicate(ctx)
    assert result["verdict"] == "FAIL", result
    assert any("isolated" in error for error in result["errors"]), result["errors"]


def test_variants_match_the_coverage_row() -> None:
    expected = set(coverage_rows()["GF-GENERAL-LIVE-032"])
    assert {item.variant for item in mod.VARIANTS} == expected, (
        "032 registers every coverage variant"
    )
    implemented = {item.variant for item in mod.VARIANTS if item.execute is not None}
    assert implemented == {"unknown_xid", "duplicate"}, (
        "032 implements the reachable kmsg variants"
    )
    deferred = [item for item in mod.VARIANTS if item.deferred]
    assert len(deferred) == 6, (
        "the other six 032 variants stay in the denominator as NOT_RUN"
    )

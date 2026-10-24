"""GF-GENERAL-LIVE-013/014 inventory: category-module verdict tests.

013 unreadable_state drives the host-collector PCI-root repoint (a source-replay
FakeProbe) and reads the host telemetry back: a malformed reading is UNKNOWN,
never a missing-device finding or a reboot. Every 014 variant (and the rest of
013) stays NOT_RUN with its reason -- those are asserted to keep the denominator.
"""

from __future__ import annotations

from pathlib import Path

from scripts.e2e.regional import general_live_inventory as mod
from scripts.e2e.regional.general_live_registry import coverage_rows
from tests.regional._general_live_fakes import (
    FakeProbe,
    FakeRegional,
    make_context,
    make_settings,
    store_script,
)

NODE = "ip-10-0-43-5.us-west-2.compute.internal"
SPECS = {spec.variant: spec for spec in mod.VARIANTS}


def _source_probe() -> FakeProbe:
    return FakeProbe(
        {
            "build-pci-fixture": {"efa_devices": 16, "malformed": True},
            "open-window": {"after": {"ActiveState": "active"}, "proof_scope": "live-replay"},
            "status": {"fixtures": {}},
            "close-window": {"clean": True},
        }
    )


def _doc(*, collection_error: bool, incidents=None, workflows=None) -> dict[str, object]:
    batch = {
        "observed_at": "2026-10-24T00:03:00+00:00",
        "batch_id": "host-1",
        "edge_filter_reasons": [],
        "collection_errors": (
            ["EFA inventory observation unavailable: invalid PCI device ID"] if collection_error else []
        ),
        "samples": [],
    }
    return {
        "incidents": incidents or [],
        "workflows": workflows or [],
        "batches": [batch],
        "markers": [],
        "metric_latest": [],
        "notifications": [],
        "collector_statuses": [],
    }


def _ctx(variant: str, *, doc: dict[str, object], node_snapshot=None, tmp_path: Path):
    spec = SPECS[variant]
    regional = FakeRegional(cpu_python=store_script({"activity": doc}), node_snapshot=node_snapshot)
    return make_context(
        spec,
        tmp_path=tmp_path,
        regional=regional,
        probes={f"source-replay@{NODE}": _source_probe()},
        settings=make_settings(spec, node=NODE),
        window_minutes=60,
    )


def test_unreadable_state_is_unknown_not_zero(tmp_path: Path) -> None:
    ctx = _ctx("unreadable_state", doc=_doc(collection_error=True), tmp_path=tmp_path)
    result = mod.unreadable_state(ctx)
    assert result["verdict"] == "PASS", result["errors"]
    assert result["proof_scope"] == "live-replay", result


def test_unreadable_state_fails_without_an_unknown_marker(tmp_path: Path) -> None:
    ctx = _ctx("unreadable_state", doc=_doc(collection_error=False), tmp_path=tmp_path)
    result = mod.unreadable_state(ctx)
    assert result["verdict"] == "FAIL", result
    assert any("collection error" in error for error in result["errors"]), result["errors"]


def test_unreadable_state_fails_on_a_reboot(tmp_path: Path) -> None:
    incidents = [
        {
            "incident_id": "inc-efa",
            "event_id": "host0-efa_inventory_mismatch-node",
            "event_type": "NODE_HEALTH",
            "effective_action": "REBOOT_NODE",
            "state": "ACTION_PENDING",
            "created_at": "2026-10-24T00:03:00+00:00",
            "workflow_request_id": "wf-efa",
            "node_ids": [NODE],
        }
    ]
    workflows = [
        {
            "request_id": "wf-efa",
            "incident_id": "inc-efa",
            "status": "RUNNING",
            "created_at": "2026-10-24T00:03:00+00:00",
            "official_steps": [{"operation": "RESTART_NODE", "parameters": {}}],
            "step_executions": [],
        }
    ]
    doc = _doc(collection_error=True, incidents=incidents, workflows=workflows)
    ctx = _ctx("unreadable_state", doc=doc, tmp_path=tmp_path)
    result = mod.unreadable_state(ctx)
    assert result["verdict"] == "FAIL", result
    assert any("RESTART_NODE" in error or "efa_inventory_mismatch" in error for error in result["errors"]), result["errors"]


def test_013_and_014_cover_every_variant() -> None:
    for case in ("GF-GENERAL-LIVE-013", "GF-GENERAL-LIVE-014"):
        expected = set(coverage_rows()[case])
        got = {item.variant for item in mod.VARIANTS if item.case_id == case}
        assert got == expected, f"{case} registers every coverage variant"
    implemented = {item.variant for item in mod.VARIANTS if item.execute is not None}
    assert implemented == {"unreadable_state"}, implemented
    assert all(item.deferred for item in mod.VARIANTS if item.case_id == "GF-GENERAL-LIVE-014"), (
        "014 has no valid replay path; every variant is NOT_RUN"
    )

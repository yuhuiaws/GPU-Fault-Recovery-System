"""GF-GENERAL-LIVE-026 inventory-identity injector/verdict contract (scripted)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from scripts.e2e.regional import general_live_gpu_identity as ident
from tests.regional._general_live_fakes import FakeRegional, make_context, make_settings
from tests.regional._general_live_gpu_fakes import (
    dcgm_probe,
    gpu_cpu_python,
    gpu_incident,
    gpu_workflow,
)

SPECS = {spec.variant: spec for spec in ident.VARIANTS}


def inventory_rewritten(rewritten: bool = True) -> list[dict[str, Any]]:
    return [
        {
            "at": 1.0,
            "proof_scope": "live-replay",
            "rewritten": rewritten,
            "returncode": 0,
            "unchanged_reason": [] if rewritten else ["no reviewed plan"],
        }
    ]


def _ctx(tmp_path: Path, variant: str, *, probe: Any, cpu: Any):
    spec = SPECS[variant]
    regional = FakeRegional(cpu_python=cpu)
    settings = make_settings(spec, node=regional.node_snapshot("")["name"])
    ctx = make_context(
        spec, tmp_path=tmp_path, regional=regional, probes={}, settings=settings
    )
    ctx.probes[f"dcgm@{ctx.settings.node}"] = probe
    return ctx


def _probe(*, rewritten: bool = True) -> Any:
    return dcgm_probe(inventory_audit=inventory_rewritten(rewritten))


def _activity(
    incidents: list, workflows: list = (), notifications: list = ()
) -> dict[str, Any]:
    return {
        "incidents": incidents,
        "workflows": list(workflows),
        "batches": [],
        "markers": [],
        "commands": [],
        "notifications": list(notifications),
        "metric_latest": [],
        "collector_statuses": [],
        "agent": None,
    }


def _notification(incident_id: str) -> dict[str, Any]:
    return {
        "notification_id": "n1",
        "incident_id": incident_id,
        "category": "OPERATOR_REVIEW",
        "deduplication_key": "k",
        "subject": "s",
        "status": "SENT",
    }


def test_registry_covers_026_and_expects_every_variant_to_pass() -> None:
    from scripts.e2e.regional import general_live_registry as reg

    assert set(SPECS) == set(reg.coverage_rows()["GF-GENERAL-LIVE-026"])
    # Batch 160 closed the bdf_changed and stale_boot product gaps.
    fails = {v for v, s in SPECS.items() if s.expected_outcome == "FAIL"}
    assert fails == set(), fails


def test_uuid_changed_is_a_read_only_identity_diagnostic(tmp_path: Path) -> None:
    inc = gpu_incident(
        "inc", "snap-1-gpu_inventory_identity_changed", state="RECOVERED", workflow="wf"
    )
    inc["event_id"] = (
        "snap-1-gpu_inventory_identity_changed"  # inventory finding: no "gpu-" prefix
    )
    wf = gpu_workflow("wf", "inc", ["FREEZE_EVIDENCE", "VALIDATE_GPU"], status="FAILED")
    cpu = gpu_cpu_python(activity=_activity([inc], [wf], [_notification("inc")]))
    out = ident.uuid_changed(_ctx(tmp_path, "uuid_changed", probe=_probe(), cpu=cpu))
    assert out["verdict"] == "PASS", out["errors"]


def test_uuid_changed_fails_on_a_reset(tmp_path: Path) -> None:
    inc = {
        "incident_id": "inc",
        "event_id": "snap-1-gpu_inventory_identity_changed",
        "event_type": "NODE_HEALTH",
        "state": "ACTION_PENDING",
        "effective_action": "RUN_DIAGNOSTICS",
        "created_at": "t",
        "workflow_request_id": "wf",
        "node_ids": ["n"],
    }
    wf = gpu_workflow("wf", "inc", ["FREEZE_EVIDENCE", "RESTART_NODE", "VALIDATE_GPU"])
    cpu = gpu_cpu_python(activity=_activity([inc], [wf], [_notification("inc")]))
    out = ident.uuid_changed(_ctx(tmp_path, "uuid_changed", probe=_probe(), cpu=cpu))
    assert out["verdict"] == "FAIL" and any("RESTART_NODE" in e for e in out["errors"])


def test_bdf_changed_fails_when_undiagnosed(tmp_path: Path) -> None:
    # Batch 160 diagnoses a BDF change; a product that still did not would FAIL.
    assert not SPECS["bdf_changed"].product_gap
    cpu = gpu_cpu_python(activity=_activity([]))  # no finding -> gap present
    out = ident.bdf_changed(_ctx(tmp_path, "bdf_changed", probe=_probe(), cpu=cpu))
    assert out["verdict"] == "FAIL" and "UUIDs" in out["errors"][0]


def test_stale_boot_fails_when_acted_on(tmp_path: Path) -> None:
    inc = {
        "incident_id": "inc",
        "event_id": "snap-2-gpu_inventory_identity_changed",
        "event_type": "NODE_HEALTH",
        "state": "ACTION_PENDING",
        "effective_action": "RUN_DIAGNOSTICS",
        "created_at": "t",
        "workflow_request_id": "wf",
        "node_ids": ["n"],
    }
    cpu = gpu_cpu_python(
        activity=_activity(
            [inc], [gpu_workflow("wf", "inc", ["FREEZE_EVIDENCE", "VALIDATE_GPU"])]
        )
    )
    out = ident.stale_boot(_ctx(tmp_path, "stale_boot", probe=_probe(), cpu=cpu))
    assert out["verdict"] == "FAIL" and "old boot" in out["errors"][0]


def test_unknown_expected_count_is_evidence_only(tmp_path: Path) -> None:
    inc = {
        "incident_id": "inc",
        "event_id": "snap-3-gpu_expected_count_unknown",
        "event_type": "NODE_HEALTH",
        "state": "ACTION_PENDING",
        "effective_action": "COLLECT_EVIDENCE",
        "created_at": "t",
        "workflow_request_id": "wf",
        "node_ids": ["n"],
    }
    wf = gpu_workflow("wf", "inc", ["FREEZE_EVIDENCE"])
    cpu = gpu_cpu_python(activity=_activity([inc], [wf], [_notification("inc")]))
    out = ident.unknown_expected_count(
        _ctx(tmp_path, "unknown_expected_count", probe=_probe(), cpu=cpu)
    )
    assert out["verdict"] == "PASS", out["errors"]


def test_count_unchanged_rejects_a_reboot_path(tmp_path: Path) -> None:
    identity = {
        "incident_id": "i1",
        "event_id": "snap-4-gpu_inventory_identity_changed",
        "event_type": "NODE_HEALTH",
        "state": "ACTION_PENDING",
        "effective_action": "RUN_DIAGNOSTICS",
        "created_at": "t",
        "workflow_request_id": "w1",
        "node_ids": ["n"],
    }
    mismatch = {
        "incident_id": "i2",
        "event_id": "host-x-gpu_inventory_mismatch-node",
        "event_type": "NODE_HEALTH",
        "state": "ACTION_PENDING",
        "effective_action": "REBOOT_NODE",
        "created_at": "t",
        "workflow_request_id": "w2",
        "node_ids": ["n"],
    }
    cpu = gpu_cpu_python(
        activity=_activity(
            [identity, mismatch],
            [
                gpu_workflow("w1", "i1", ["FREEZE_EVIDENCE", "VALIDATE_GPU"]),
                gpu_workflow("w2", "i2", ["RESTART_NODE"]),
            ],
        )
    )
    out = ident.count_unchanged(
        _ctx(tmp_path, "count_unchanged", probe=_probe(), cpu=cpu)
    )
    assert out["verdict"] == "FAIL" and any(
        "count mismatch" in e for e in out["errors"]
    )


def test_no_unapproved_reset_checks_cleanup(tmp_path: Path) -> None:
    inc = {
        "incident_id": "inc",
        "event_id": "snap-5-gpu_inventory_identity_changed",
        "event_type": "NODE_HEALTH",
        "state": "RECOVERED",
        "effective_action": "RUN_DIAGNOSTICS",
        "created_at": "t",
        "workflow_request_id": "wf",
        "node_ids": ["n"],
    }
    cpu = gpu_cpu_python(
        activity=_activity(
            [inc],
            [
                gpu_workflow(
                    "wf", "inc", ["FREEZE_EVIDENCE", "VALIDATE_GPU"], status="FAILED"
                )
            ],
        )
    )
    out = ident.no_unapproved_reset(
        _ctx(tmp_path, "no_unapproved_reset", probe=_probe(), cpu=cpu)
    )
    assert out["verdict"] == "PASS", out["errors"]


def test_inventory_oracle_blocks_when_not_rewritten(tmp_path: Path) -> None:
    import pytest

    from scripts.e2e.regional.general_live_fixture import VariantInconclusive

    cpu = gpu_cpu_python(activity=_activity([]))
    with pytest.raises(VariantInconclusive, match="inventory replay was not effective"):
        ident.uuid_changed(
            _ctx(tmp_path, "uuid_changed", probe=_probe(rewritten=False), cpu=cpu)
        )

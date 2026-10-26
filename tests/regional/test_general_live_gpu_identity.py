"""GF-GENERAL-LIVE-026 inventory-identity injector/verdict contract (scripted).

The inventory oracle gate is waited for, from the repoint on: live 2026-10-26
(plain EKS, release R13g) ``bdf_changed`` (ip-10-0-43-5) and
``count_unchanged`` (ip-10-0-32-162) ended INCONCLUSIVE "the wrapper never
rewrote a real inventory query" because ``require_inventory`` read the wrapper's
journal exactly once, 14-15 s after the collector restart; the restarted
collector's first inventory query -- its first tick, start-up spread plus
start-up jitter -- came 20 s after the restart, 5 s after the verdict (the
record is in the case's ``oracle.jsonl``), while ``uuid_changed`` had passed at
09:03Z with a first tick 6 s after the restart. The reviewed plan's deadline
(``not_after`` = 09:41:45Z) had not expired in either run.
"""

from __future__ import annotations

import itertools
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import general_live_gpu_identity as ident
from scripts.e2e.regional.general_live_fixture import VariantInconclusive
from scripts.e2e.regional.general_live_gpu_replay import GpuSurvey
from tests.regional._general_live_fakes import FakeRegional, make_context, make_settings
from tests.regional._general_live_gpu_fakes import (
    dcgm_probe,
    gpu_cpu_python,
    gpu_incident,
    gpu_workflow,
    survey,
)

SPECS = {spec.variant: spec for spec in ident.VARIANTS}
#: The fake survey's collector environment carries no start-up spread, jitter
#: or inventory interval, so the window is the product defaults':
#: 15 + 20 + 2 x 60 + 30.
DEFAULT_WINDOW = 185


def inventory_record(
    *, rewritten: bool = True, returncode: int = 0, reason: str = "no reviewed plan"
) -> dict[str, Any]:
    """One inventory call as ``tools/nvidia_smi_inventory_replay.py`` journals it."""

    return {
        "at": 1793006221.914686,
        "proof_scope": "live-replay",
        "physical_fault_injected": False,
        "rewritten": rewritten,
        "returncode": returncode,
        "unchanged_reason": []
        if rewritten
        else ["the real query failed"]
        if returncode
        else [reason],
        "operations": [{"op": "replace_uuid", "index": 7, "uuid": "GPU-x"}]
        if rewritten
        else [],
        "real_sha256": "a" * 64,
        "served_sha256": "b" * 64 if rewritten else "a" * 64,
        "served": "0, GPU-x, 00000000:59:00.0, NVIDIA H200\n" if rewritten else None,
    }


def inventory_rewritten(rewritten: bool = True) -> list[dict[str, Any]]:
    return [inventory_record(rewritten=rewritten)]


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


def _flag(arguments: tuple[str, ...], name: str, default: str) -> str:
    for index, value in enumerate(arguments):
        if value == name and index + 1 < len(arguments):
            return arguments[index + 1]
    return default


def _growing_inventory_journal(*stages: list[dict[str, Any]]):
    """An ``audit-read`` whose inventory journal is ``stages[n]`` on its n-th
    read; the replay journal (read by ``close``) stays empty."""

    calls = {"count": 0}

    def audit_read(arguments: tuple[str, ...]) -> dict[str, Any]:
        offset = int(_flag(arguments, "--offset", "0"))
        if _flag(arguments, "--journal", "replay.jsonl") != "inventory.jsonl":
            return {"records": [], "next_offset": offset, "total": 0}
        source = stages[min(calls["count"], len(stages) - 1)]
        calls["count"] += 1
        chunk = source[offset:]
        return {
            "records": list(chunk),
            "next_offset": offset + len(chunk),
            "total": len(source),
        }

    return audit_read


def _growing_probe(*stages: list[dict[str, Any]]) -> Any:
    return dcgm_probe(extra={"audit-read": _growing_inventory_journal(*stages)})


def _inventory_reads(probe: Any) -> int:
    return sum(
        1
        for call in probe.calls
        if call[0] == "audit-read" and "inventory.jsonl" in call
    )


def _gate_evidence(ctx: Any) -> dict[str, Any]:
    return json.loads((ctx.case_dir / "inventory-gate.json").read_text())


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
    assert not SPECS["bdf_changed"].product_gap, SPECS["bdf_changed"].product_gap
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
    # A real query the wrapper passed through unchanged is final: the reason
    # (here an unreadable plan) holds for the run's whole life.
    cpu = gpu_cpu_python(activity=_activity([]))
    probe = _probe(rewritten=False)
    with pytest.raises(VariantInconclusive) as raised:
        ident.uuid_changed(_ctx(tmp_path, "uuid_changed", probe=probe, cpu=cpu))
    assert str(raised.value) == (
        "the inventory replay was not effective: the wrapper left a real inventory "
        "query unchanged: no reviewed plan"
    )
    assert _inventory_reads(probe) == 1, "a refusal is final; nothing to wait for"


# --------------------------------------------------------------------------- #
# the oracle gate waits for the restarted collector's first inventory query
# --------------------------------------------------------------------------- #
def _passing_identity_cpu() -> Any:
    inc = gpu_incident(
        "inc", "snap-1-gpu_inventory_identity_changed", state="RECOVERED", workflow="wf"
    )
    inc["event_id"] = "snap-1-gpu_inventory_identity_changed"
    wf = gpu_workflow("wf", "inc", ["FREEZE_EVIDENCE", "VALIDATE_GPU"], status="FAILED")
    return gpu_cpu_python(activity=_activity([inc], [wf], [_notification("inc")]))


def test_the_gate_waits_for_the_first_rewritten_inventory_query(tmp_path: Path) -> None:
    # Live 2026-10-26: the journal was empty 14-15 s after the restart and held
    # the rewritten first query 20 s after it.
    probe = _growing_probe([], [], [inventory_record()])
    ctx = _ctx(tmp_path, "uuid_changed", probe=probe, cpu=_passing_identity_cpu())

    out = ident.uuid_changed(ctx)

    assert out["verdict"] == "PASS", out["errors"]
    assert _inventory_reads(probe) >= 3, "the journal was polled until the query came"
    timeline = json.loads(
        (ctx.case_dir / "inventory-rewritten-timeline.json").read_text()
    )
    accepted = [entry["accepted"] for entry in timeline["entries"]]
    assert accepted[:-1] == [False] * (len(accepted) - 1) and accepted[-1] is True
    gate = _gate_evidence(ctx)
    assert gate["window_seconds"] == DEFAULT_WINDOW
    assert gate["rewritten"] == {
        "rewritten_at": 1793006221.914686,
        "inventory_queries": 1,
    }


def test_an_empty_journal_at_the_deadline_names_the_window(tmp_path: Path) -> None:
    probe = _growing_probe([])
    ctx = _ctx(tmp_path, "uuid_changed", probe=probe, cpu=gpu_cpu_python())

    with pytest.raises(VariantInconclusive) as raised:
        ident.uuid_changed(ctx)

    assert str(raised.value) == (
        "the inventory replay was not effective: the wrapper never rewrote a real "
        f"inventory query within {DEFAULT_WINDOW} s of the repoint: no inventory "
        "query reached the wrapper"
    )
    assert _inventory_reads(probe) > 1, "the gate polled across the window, not once"


def test_a_failed_real_query_is_waited_out_not_final(tmp_path: Path) -> None:
    failed = inventory_record(rewritten=False, returncode=6)
    probe = _growing_probe([failed], [failed, inventory_record()])
    ctx = _ctx(tmp_path, "uuid_changed", probe=probe, cpu=_passing_identity_cpu())

    out = ident.uuid_changed(ctx)

    assert out["verdict"] == "PASS", out["errors"]
    assert _gate_evidence(ctx)["rewritten"]["inventory_queries"] == 2


def test_only_failed_queries_at_the_deadline_are_counted(tmp_path: Path) -> None:
    failed = inventory_record(rewritten=False, returncode=6)
    probe = _growing_probe([failed], [failed, failed])

    with pytest.raises(
        VariantInconclusive, match="2 inventory queries journaled, 2 failed$"
    ):
        ident.uuid_changed(
            _ctx(tmp_path, "uuid_changed", probe=probe, cpu=gpu_cpu_python())
        )


def test_a_refused_rewrite_names_the_wrappers_reason(tmp_path: Path) -> None:
    refused = inventory_record(
        rewritten=False, reason="the replay UUID names a real GPU"
    )
    probe = _growing_probe([], [refused])

    with pytest.raises(
        VariantInconclusive, match="unchanged: the replay UUID names a real GPU$"
    ):
        ident.uuid_changed(
            _ctx(tmp_path, "uuid_changed", probe=probe, cpu=gpu_cpu_python())
        )
    assert _inventory_reads(probe) == 2


@pytest.mark.parametrize("variant", sorted(SPECS))
def test_every_variant_waits_for_the_gate(tmp_path: Path, variant: str) -> None:
    probe = _growing_probe([], [inventory_record()])
    ctx = _ctx(tmp_path, variant, probe=probe, cpu=gpu_cpu_python())

    getattr(ident, variant)(ctx)  # the verdict is the variant's; the gate passed

    assert _inventory_reads(probe) >= 2, variant
    assert _gate_evidence(ctx)["window_seconds"] == DEFAULT_WINDOW


# --------------------------------------------------------------------------- #
# the window follows the survey; the plan covers the deploy and the hold
# --------------------------------------------------------------------------- #
def _survey(**env: str) -> GpuSurvey:
    raw = survey()
    raw["collector_env"] = {**raw["collector_env"], **env}
    return GpuSurvey(raw)


def test_the_window_follows_the_collectors_real_settings() -> None:
    assert ident.inventory_window_seconds(_survey()) == DEFAULT_WINDOW
    assert (
        ident.inventory_window_seconds(
            _survey(
                GPU_FAULT_GPU_INVENTORY_INTERVAL_SECONDS="120",
                GPU_FAULT_COLLECTOR_STARTUP_SPREAD_SECONDS="30",
                GPU_FAULT_COLLECTOR_STARTUP_JITTER_SECONDS="10.0",
            )
        )
        == 30 + 10 + 2 * 120 + 30
    )
    # Malformed or negative settings keep the product defaults.
    assert (
        ident.inventory_window_seconds(
            _survey(
                GPU_FAULT_COLLECTOR_STARTUP_SPREAD_SECONDS="soon",
                GPU_FAULT_COLLECTOR_STARTUP_JITTER_SECONDS="-1",
            )
        )
        == DEFAULT_WINDOW
    )


def test_the_plan_deadline_is_the_deploy_allowance_plus_the_hold() -> None:
    plan = ident.inventory_plan(
        [{"op": "swap_bdf", "indexes": [0, 7]}], composed_at=1000.0, hold=960
    )
    assert plan == {
        "operations": [{"op": "swap_bdf", "indexes": [0, 7]}],
        "not_after": 1000.0 + 600 + 960,
    }
    assert ident.INVENTORY_DEPLOY_ALLOWANCE_SECONDS == 600


def test_the_plan_is_composed_on_the_context_clock(tmp_path: Path) -> None:
    probe = _growing_probe([inventory_record()])
    ctx = _ctx(tmp_path, "bdf_changed", probe=probe, cpu=gpu_cpu_python())
    composed = ctx.clock()
    readings = itertools.count()

    def clock() -> datetime:
        # The plan is composed and the repoint returns on one reading (the
        # deploy takes 0 s); the waits after it see a clock that advances.
        number = next(readings)
        return composed if number < 2 else composed + timedelta(seconds=number)

    ctx.clock = clock

    ident.bdf_changed(ctx)

    plan = json.loads((ctx.case_dir / "inventory-plan.json").read_text())
    assert plan["not_after"] == composed.timestamp() + 600 + 16 * 60
    gate = _gate_evidence(ctx)
    assert gate["deploy_seconds"] == 0.0
    assert gate["plan_remaining_seconds"] == 600 + 16 * 60
    assert gate["hold_seconds"] == 16 * 60


def test_a_deploy_past_its_allowance_is_a_named_fixture_failure(tmp_path: Path) -> None:
    probe = _growing_probe([inventory_record()])
    ctx = _ctx(tmp_path, "uuid_changed", probe=probe, cpu=gpu_cpu_python())
    origin = datetime(2026, 10, 26, 9, 15, 45, tzinfo=timezone.utc)
    readings = iter([origin, origin + timedelta(seconds=700)])
    late = origin + timedelta(seconds=700)
    ctx.clock = lambda: next(readings, late)

    with pytest.raises(VariantInconclusive) as raised:
        ident.uuid_changed(ctx)

    assert str(raised.value) == (
        "harness fixture failed: the reviewed inventory plan stays valid 1100 s "
        "after the repoint, below the 1200 s the variant needs (the deploy took "
        "700 s against a 600 s allowance; hold 1200 s, first-query window 185 s); "
        "nothing was graded"
    )
    assert _inventory_reads(probe) == 0, "no wait spends an expiring plan"
    gate = _gate_evidence(ctx)
    assert gate["plan_not_after"] == "2026-10-26T09:45:45+00:00"
    assert "rewritten" not in gate

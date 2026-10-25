"""Lane N1 phase-2 denominator and the 007 healthy observation.

Every required variant of the host/device categories 002/005/007/010/012/015
must be in the registry -- implemented, or NOT_RUN with a reviewed reason --
so none silently drops out of the coverage denominator. The read-only
observation 007 healthy is exercised here against scripted product records;
the replay variants of 007 and 015 have their own modules and tests
(``test_general_live_smart.py``, ``test_general_live_bmc.py``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import general_live_registry as reg
from scripts.e2e.regional import general_live_smart as smart
from tests.regional._general_live_fakes import (
    FakeProbe,
    FakeRegional,
    host_batch,
    incident,
    make_context,
    store_script,
)

LANE_CATEGORIES = (2, 5, 7, 10, 12, 15)
NODE = "ip-10-0-43-5.us-west-2.compute.internal"


def test_phase_two_categories_include_the_host_device_lane() -> None:
    # The integrated registry unions every phase-2 lane's categories.
    assert set(LANE_CATEGORIES) <= set(reg.PHASE_TWO_CATEGORIES), (
        reg.PHASE_TWO_CATEGORIES
    )
    assert list(reg.PHASE_TWO_CATEGORIES) == sorted(set(reg.PHASE_TWO_CATEGORIES)), (
        "phase-2 categories are listed once, in case order"
    )
    assert set(reg.PHASE_ONE_CATEGORIES).isdisjoint(reg.PHASE_TWO_CATEGORIES), (
        "phase-2 categories must not overlap phase 1"
    )


def test_every_lane_variant_is_in_the_registry_denominator() -> None:
    registry = reg.load_registry()
    rows = reg.coverage_rows()
    for number in LANE_CATEGORIES:
        case = f"GF-GENERAL-LIVE-{number:03d}"
        expected = set(rows[case])
        present = {spec.variant for spec in registry.values() if spec.case_id == case}
        assert present == expected, f"{case}: {present ^ expected}"


def test_deferred_lane_variants_carry_a_reviewed_reason() -> None:
    registry = reg.load_registry()
    for spec in registry.values():
        if spec.number not in LANE_CATEGORIES or spec.implemented:
            continue
        reason = spec.deferred.split(":", 1)[0]
        assert reason in reg.NOT_RUN_REASONS, (spec.key, spec.deferred)
        assert len(spec.deferred.split(":", 1)[1].strip()) >= 12, (
            f"{spec.key}: a deferred reason must explain itself"
        )


def test_live_replay_executable_is_now_allowed_but_vendor_lab_is_not() -> None:
    base = dict(
        case_id="GF-GENERAL-LIVE-007",
        variant="probe",
        lanes=("eks",),
        risk="L",
        injection="i",
        expected="e",
        cleanup="c",
        max_minutes=5,
        execute=lambda ctx: {"verdict": "PASS"},
    )
    reg.validate_spec(reg.VariantSpec(proof_mode=reg.LIVE_REPLAY, **base))
    with pytest.raises(reg.RegistryError, match="only native, observation and live"):
        reg.validate_spec(reg.VariantSpec(proof_mode=reg.VENDOR_LAB, **base))


def test_the_executable_lane_variants_are_the_wired_ones() -> None:
    registry = reg.load_registry()
    implemented = sorted(
        spec.key
        for spec in registry.values()
        if spec.number in LANE_CATEGORIES and spec.implemented
    )
    assert implemented == [
        "GF-GENERAL-LIVE-002.cgroup_only",
        "GF-GENERAL-LIVE-002.memory_pressure",
        "GF-GENERAL-LIVE-002.no_swap",
        "GF-GENERAL-LIVE-002.read_failure",
        "GF-GENERAL-LIVE-002.recovery",
        "GF-GENERAL-LIVE-002.swap_pressure",
        "GF-GENERAL-LIVE-007.health_failed",
        "GF-GENERAL-LIVE-007.healthy",
        "GF-GENERAL-LIVE-007.malformed_json",
        "GF-GENERAL-LIVE-007.query_failed",
        "GF-GENERAL-LIVE-007.scan_failed",
        "GF-GENERAL-LIVE-007.tool_missing",
        "GF-GENERAL-LIVE-007.unsupported_device",
        # 010 is wired on the shared host source replay (general_live_efa.py).
        "GF-GENERAL-LIVE-010.counter_reset",
        "GF-GENERAL-LIVE-010.cq_error",
        "GF-GENERAL-LIVE-010.ethtool_failed",
        "GF-GENERAL-LIVE-010.link_inactive",
        "GF-GENERAL-LIVE-010.missing_counter",
        "GF-GENERAL-LIVE-010.rdma_error",
        "GF-GENERAL-LIVE-010.retry",
        "GF-GENERAL-LIVE-010.rnr",
        "GF-GENERAL-LIVE-012.expectation_wrong",
        "GF-GENERAL-LIVE-012.proc_unreadable",
        "GF-GENERAL-LIVE-012.query_failed",
        "GF-GENERAL-LIVE-012.single_sample",
        "GF-GENERAL-LIVE-015.command_failure",
        "GF-GENERAL-LIVE-015.critical",
        "GF-GENERAL-LIVE-015.nonrecoverable",
        "GF-GENERAL-LIVE-015.unavailable_sensor",
        "GF-GENERAL-LIVE-015.unsupported_bmc",
        "GF-GENERAL-LIVE-015.wrong_column",
    ], implemented
    for key in implemented:
        spec = registry[key]
        if spec.proof_mode == reg.LIVE_REPLAY:
            assert "HOST_SOURCE_REPLAY" in spec.approvals, key
            assert spec.lanes == ("eks", "hp"), key


# --------------------------------------------------------------------------- #
# 007 healthy (read-only observation)
# --------------------------------------------------------------------------- #
def _healthy_activity(*, failed: bool = False) -> dict[str, Any]:
    document: dict[str, Any] = {
        "incidents": [],
        "workflows": [],
        "batches": [
            host_batch(
                "2026-10-24T00:02:00+00:00",
                "smart_health_failed",
                1.0 if failed else 0.0,
                device="/dev/nvme0",
            )
        ],
        "markers": [],
        "commands": [],
        "health_signals": {},
        "notifications": [],
        "metric_latest": [
            {
                "name": "smart_health_failed",
                "device": "/dev/nvme0",
                "value": 1.0 if failed else 0.0,
                "observed_at": "2026-10-24T00:02:00+00:00",
                "labels": {},
            }
        ],
    }
    if failed:
        document["incidents"] = [
            incident(
                "inc-s", "smart_health_failed", device="/dev/nvme0", state="QUARANTINED"
            )
        ]
        document["workflows"] = []
    return document


def _devices_context(
    case_id: str, variant: str, tmp_path: Path, document: dict[str, Any]
):
    spec = next(
        item
        for item in smart.VARIANTS
        if item.case_id == case_id and item.variant == variant
    )
    regional = FakeRegional(cpu_python=store_script({"activity": document}))
    probe = FakeProbe(
        {
            "survey": {
                "tools": {},
                "modules": {},
                "backing": [],
                "residue": {"units": [], "paused_units": {}, "paused_processes": {}},
            }
        }
    )
    return spec, make_context(
        spec, tmp_path=tmp_path, regional=regional, probes={f"node@{NODE}": probe}
    )


def test_smart_healthy_passes_when_no_finding(tmp_path: Path) -> None:
    spec, ctx = _devices_context(
        "GF-GENERAL-LIVE-007", "healthy", tmp_path, _healthy_activity()
    )
    result = spec.execute(ctx)
    assert result["verdict"] == "PASS", result["errors"]


def test_smart_healthy_fails_when_a_smart_finding_exists(tmp_path: Path) -> None:
    spec, ctx = _devices_context(
        "GF-GENERAL-LIVE-007", "healthy", tmp_path, _healthy_activity(failed=True)
    )
    result = spec.execute(ctx)
    assert result["verdict"] == "FAIL"
    assert any("smart_health_failed" in error for error in result["errors"]), result[
        "errors"
    ]

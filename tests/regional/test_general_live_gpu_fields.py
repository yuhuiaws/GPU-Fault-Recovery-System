"""GF-GENERAL-LIVE-025 field-variant injector/verdict contract (scripted)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from scripts.e2e.regional import general_live_gpu_fields as fields
from tests.regional._general_live_fakes import FakeRegional, make_context, make_settings
from tests.regional._general_live_gpu_fakes import (
    audit_records,
    dcgm_probe,
    gpu_cpu_python,
)

SPECS = {spec.variant: spec for spec in fields.VARIANTS}
G0 = "GPU-00000000-0000-4000-8000-000000000000"
G1 = "GPU-11111111-1111-4111-8111-111111111111"


def _ctx(tmp_path: Path, variant: str, *, probe: Any, cpu: Any):
    spec = SPECS[variant]
    regional = FakeRegional(cpu_python=cpu)
    settings = make_settings(spec, node=regional.node_snapshot("")["name"])
    ctx = make_context(
        spec, tmp_path=tmp_path, regional=regional, probes={}, settings=settings
    )
    ctx.probes[f"dcgm@{ctx.settings.node}"] = probe
    return ctx


def _state(*, batches: list, latest: list = (), active: list = ()) -> dict[str, Any]:
    return {
        "latest": list(latest),
        "active_findings": list(active),
        "findings": [],
        "xid_events": [],
        "batches": batches,
        "collector_statuses": [],
        "agent": None,
    }


def _batch(collection_errors: list, samples: list = ()) -> dict[str, Any]:
    return {
        "observed_at": "t",
        "batch_id": "b",
        "collection_errors": list(collection_errors),
        "edge_filter_reasons": [],
        "sample_count": len(samples),
        "gpus": [],
        "samples": list(samples),
    }


def test_registry_covers_the_five_field_variants() -> None:
    assert set(SPECS) == {
        "missing_field",
        "one_gpu_missing",
        "conflicting_alias",
        "nan_negative",
        "duplicate_slot",
    }
    assert all(s.proof_mode == "live-replay" for s in fields.VARIANTS), (
        "every 025 field variant is a live-replay"
    )


def test_missing_field_reports_a_batch_wide_omission(tmp_path: Path) -> None:
    served = audit_records(
        {f"{G0}/pcie_replay_total": "omitted", f"{G1}/pcie_replay_total": "omitted"}
    )
    state = _state(
        batches=[
            _batch(["DCGM exporter is missing required fields: pcie_replay_total"])
        ]
    )
    out = fields.missing_field(
        _ctx(
            tmp_path,
            "missing_field",
            probe=dcgm_probe(audit=served),
            cpu=gpu_cpu_python(gpu_state=[state, state]),
        )
    )
    assert out["verdict"] == "PASS", out["errors"]


def test_missing_field_fails_when_not_reported(tmp_path: Path) -> None:
    served = audit_records(
        {f"{G0}/pcie_replay_total": "omitted", f"{G1}/pcie_replay_total": "omitted"}
    )
    state = _state(batches=[_batch([])])
    out = fields.missing_field(
        _ctx(
            tmp_path,
            "missing_field",
            probe=dcgm_probe(audit=served),
            cpu=gpu_cpu_python(gpu_state=[state, state]),
        )
    )
    assert out["verdict"] == "FAIL", out


def test_one_gpu_missing_fails_when_the_omission_stays_hidden(tmp_path: Path) -> None:
    # Batch 160 judges completeness per GPU, so the variant expects PASS live;
    # a product that still hid the omission makes the runner FAIL it.
    spec = SPECS["one_gpu_missing"]
    assert spec.expected_outcome == "PASS" and not spec.product_gap, spec
    served = audit_records({f"{G1}/pcie_replay_total": "omitted"})
    # The batch union still has pcie_replay_total (from G0), so no error fires.
    state = _state(
        batches=[
            _batch([], samples=[{"gpu": G0, "name": "pcie_replay_total", "value": 3.0}])
        ]
    )
    out = fields.one_gpu_missing(
        _ctx(
            tmp_path,
            "one_gpu_missing",
            probe=dcgm_probe(audit=served),
            cpu=gpu_cpu_python(gpu_state=[state, state]),
        )
    )
    assert out["verdict"] == "FAIL", out
    assert "whole-batch" in out["errors"][0]


def test_conflicting_alias_rejects_the_slot(tmp_path: Path) -> None:
    served = audit_records(
        {f"{G1}/clock_throttle_reasons#alias:DCGM_FI_DEV_CLOCK_THROTTLE_REASONS": 64}
    )
    state = _state(
        batches=[
            _batch(
                [f"invalid or ambiguous GPU observation: {G1}/clock_throttle_reasons"]
            )
        ]
    )
    out = fields.conflicting_alias(
        _ctx(
            tmp_path,
            "conflicting_alias",
            probe=dcgm_probe(audit=served),
            cpu=gpu_cpu_python(gpu_state=[state, state]),
        )
    )
    assert out["verdict"] == "PASS", out["errors"]


def test_nan_negative_rejects_and_never_zeroes(tmp_path: Path) -> None:
    state = _state(
        batches=[
            _batch([f"invalid or ambiguous GPU observation: {G1}/pcie_replay_total"])
        ],
        latest=[],  # negative rejected, NaN dropped: neither reaches latest
    )
    out = fields.nan_negative(
        _ctx(
            tmp_path,
            "nan_negative",
            probe=dcgm_probe(audit=audit_records()),
            cpu=gpu_cpu_python(gpu_state=[state, state]),
        )
    )
    assert out["verdict"] == "PASS", out["errors"]


def test_nan_negative_fails_if_a_slot_becomes_zero(tmp_path: Path) -> None:
    state = _state(
        batches=[
            _batch([f"invalid or ambiguous GPU observation: {G1}/pcie_replay_total"])
        ],
        latest=[{"name": "row_remap_failure", "gpu": G1, "value": 0.0}],
    )
    out = fields.nan_negative(
        _ctx(
            tmp_path,
            "nan_negative",
            probe=dcgm_probe(audit=audit_records()),
            cpu=gpu_cpu_python(gpu_state=[state, state]),
        )
    )
    assert out["verdict"] == "FAIL" and any("healthy zero" in e for e in out["errors"])


def test_duplicate_slot_rejects_and_keeps_other_gpus(tmp_path: Path) -> None:
    served = audit_records({f"{G1}/pcie_replay_total#duplicate": 7})
    state = _state(
        batches=[
            _batch([f"invalid or ambiguous GPU observation: {G1}/pcie_replay_total"])
        ],
        latest=[{"name": "pcie_replay_total", "gpu": G0, "value": 4.0}],
    )
    out = fields.duplicate_slot(
        _ctx(
            tmp_path,
            "duplicate_slot",
            probe=dcgm_probe(audit=served),
            cpu=gpu_cpu_python(gpu_state=[state, state]),
        )
    )
    assert out["verdict"] == "PASS", out["errors"]

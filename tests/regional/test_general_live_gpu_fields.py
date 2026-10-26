"""GF-GENERAL-LIVE-025 field-variant injector/verdict contract (scripted)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import general_live_gpu_fields as fields
from tests.regional._general_live_fakes import FakeRegional, make_context, make_settings
from tests.regional._general_live_gpu_fakes import (
    audit_records,
    dcgm_probe,
    gpu_cpu_python,
    survey,
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


def _batch(
    collection_errors: list, samples: list = (), at: str = "t"
) -> dict[str, Any]:
    return {
        "observed_at": at,
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


# The live 2026-10-26 shape (plain EKS, H200): the replayed scrape at REPLAYED_AT
# carried the product's two rejections; both slots' latest accepted samples were
# the real zeros from an earlier scrape.
REPLAYED_AT = "2026-10-26T08:56:40.123456+00:00"
EARLIER = "2026-10-26T08:53:33.466922+00:00"
REJECTIONS = [
    f"DCGM exporter is missing required fields on {G1}: row_remap_failure",
    f"invalid or ambiguous GPU observation: {G1}/pcie_replay_total",
]


def _nan_negative_served() -> list[dict[str, Any]]:
    return audit_records(
        {f"{G1}/pcie_replay_total": -1.0, f"{G1}/row_remap_failure": "NaN"}
    )


def _latest(name: str, value: float, at: str, gpu: str = G1) -> dict[str, Any]:
    return {"name": name, "gpu": gpu, "value": value, "observed_at": at}


def _nan_negative(tmp_path: Path, state: dict[str, Any], **options: Any):
    return fields.nan_negative(
        _ctx(
            tmp_path,
            "nan_negative",
            probe=dcgm_probe(options.get("survey"), audit=_nan_negative_served()),
            cpu=gpu_cpu_python(gpu_state=[state, options.get("after", state)]),
        )
    )


def test_nan_negative_passes_when_the_earlier_value_is_retained(tmp_path: Path) -> None:
    """The product drops both readings and keeps the slots' earlier accepted
    zeros: a retained real zero from before the replayed scrape is no
    conversion (the live run failed on exactly this shape)."""

    state = _state(
        batches=[_batch(REJECTIONS, at=REPLAYED_AT)],
        latest=[
            _latest("pcie_replay_total", 0.0, EARLIER),
            _latest("row_remap_failure", 0.0, EARLIER),
        ],
    )
    out = _nan_negative(tmp_path, state)
    assert out["verdict"] == "PASS", out["errors"]
    assert out["first_replayed_batch_at"] == REPLAYED_AT
    assert len(out["notes"]) == 2 and all(
        "kept its earlier accepted value 0.0" in note for note in out["notes"]
    ), out["notes"]
    assert set(out["latest"]) == {"pcie_replay_total", "row_remap_failure"}


def test_nan_negative_fails_if_a_slot_becomes_zero(tmp_path: Path) -> None:
    """A latest row timestamped by the replayed scrape is an accepted value."""

    state = _state(
        batches=[_batch(REJECTIONS, at=REPLAYED_AT)],
        latest=[
            _latest("pcie_replay_total", 0.0, EARLIER),
            _latest("row_remap_failure", 0.0, REPLAYED_AT),
        ],
    )
    out = _nan_negative(tmp_path, state)
    assert out["verdict"] == "FAIL"
    assert [e for e in out["errors"] if "no_healthy_zero" in e] == [
        "no_healthy_zero: a NaN row_remap_failure reading became the latest accepted "
        f"value 0.0 at {REPLAYED_AT}, not before the first replayed scrape at "
        f"{REPLAYED_AT}"
    ], out["errors"]


def test_nan_negative_fails_if_the_negative_is_accepted(tmp_path: Path) -> None:
    state = _state(
        batches=[_batch(REJECTIONS[:1], at=REPLAYED_AT)],
        latest=[
            _latest("pcie_replay_total", -1.0, REPLAYED_AT),
            _latest("row_remap_failure", 0.0, EARLIER),
        ],
    )
    out = _nan_negative(tmp_path, state)
    assert out["verdict"] == "FAIL"
    assert any("not reported as an invalid observation" in e for e in out["errors"]), (
        out["errors"]
    )
    assert any(
        "a negative pcie_replay_total reading is the latest accepted value: -1.0" in e
        for e in out["errors"]
    ), out["errors"]


def test_nan_negative_fails_if_a_sample_is_accepted_while_served(
    tmp_path: Path,
) -> None:
    state = _state(
        batches=[
            _batch(
                REJECTIONS,
                samples=[{"gpu": G1, "name": "row_remap_failure", "value": 0.0}],
                at=REPLAYED_AT,
            )
        ],
        latest=[
            _latest("pcie_replay_total", 0.0, EARLIER),
            _latest("row_remap_failure", 0.0, EARLIER),
        ],
    )
    out = _nan_negative(tmp_path, state)
    assert out["verdict"] == "FAIL"
    assert any(
        "a NaN row_remap_failure reading was accepted as a sample while it was served"
        in e
        for e in out["errors"]
    ), out["errors"]


def test_nan_negative_cannot_judge_without_the_products_rejection(
    tmp_path: Path,
) -> None:
    """No batch carries a rejection: the stages that read them fail, and the
    retained-value check says it has no replayed scrape to judge against."""

    state = _state(
        batches=[_batch([], at=REPLAYED_AT)],
        latest=[
            _latest("pcie_replay_total", 0.0, EARLIER),
            _latest("row_remap_failure", 0.0, EARLIER),
        ],
    )
    out = _nan_negative(tmp_path, state)
    assert out["verdict"] == "FAIL"
    assert out["first_replayed_batch_at"] is None
    assert any("not reported as an invalid observation" in e for e in out["errors"]), (
        out["errors"]
    )
    assert any("not reported as a missing field" in e for e in out["errors"]), out[
        "errors"
    ]
    assert sum("cannot be judged" in e for e in out["errors"]) == 2, out["errors"]


def test_nan_negative_accepts_the_batch_wide_message_on_a_one_gpu_node(
    tmp_path: Path,
) -> None:
    """With one GPU the dropped NaN slot is a scrape-wide gap, reported without
    the GPU's name."""

    one_gpu = survey(gpus=(G1,))
    state = _state(
        batches=[
            _batch(
                [
                    "DCGM exporter is missing required fields: row_remap_failure",
                    f"invalid or ambiguous GPU observation: {G1}/pcie_replay_total",
                ],
                at=REPLAYED_AT,
            )
        ],
        latest=[
            _latest("pcie_replay_total", 0.0, EARLIER),
            _latest("row_remap_failure", 0.0, EARLIER),
        ],
    )
    out = _nan_negative(tmp_path, state, survey=one_gpu)
    assert out["verdict"] == "PASS", out["errors"]


def test_nan_negative_needs_a_gpu_reporting_both_fields(tmp_path: Path) -> None:
    from scripts.e2e.regional.general_live_fixture import VariantNotRun

    without = survey(
        fields=("pcie_replay_total", "ecc_dbe_volatile_total", "clock_throttle_reasons")
    )
    with pytest.raises(VariantNotRun, match="CAPABILITY_ABSENT.*row_remap_failure"):
        _nan_negative(tmp_path, _state(batches=[]), survey=without)


def test_nan_negative_is_inconclusive_when_the_replay_never_served(
    tmp_path: Path,
) -> None:
    from scripts.e2e.regional.general_live_fixture import VariantInconclusive

    state = _state(batches=[_batch(REJECTIONS, at=REPLAYED_AT)])
    with pytest.raises(VariantInconclusive, match="never served"):
        fields.nan_negative(
            _ctx(
                tmp_path,
                "nan_negative",
                probe=dcgm_probe(audit=audit_records()),
                cpu=gpu_cpu_python(gpu_state=[state, state]),
            )
        )


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

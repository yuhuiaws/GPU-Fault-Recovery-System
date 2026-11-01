"""GF-GENERAL-LIVE-025 field-variant injector/verdict contract (scripted)."""

from __future__ import annotations

import base64
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import general_live_gpu_fields as fields
from scripts.e2e.regional.general_live_fixture import VariantNotRun
from tests.regional._general_live_fakes import (
    FakeRegional,
    make_context,
    make_settings,
    store_poll_clock,
)
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


LEGACY_ALIAS = "DCGM_FI_DEV_CLOCK_THROTTLE_REASONS"
CURRENT_ALIAS = "DCGM_FI_DEV_CLOCKS_EVENT_REASONS"
#: Batch 235: the NVLink5 link-recovery counters are NVLink completeness members
#: read under two names (DCGM 4.6, then 4.5.3); a Hopper survey shows neither.
BLACKWELL_ALIAS_FAMILIES = {
    "nvlink_link_recovery_failed_total": (
        "DCGM_FI_DEV_NVLINK_COUNT_LINK_RECOVERY_FAILED_EVENTS",
        "DCGM_FI_DEV_NVLINK_RECOVERY_FAILED_TOTAL",
    ),
    "nvlink_link_recovery_successful_total": (
        "DCGM_FI_DEV_NVLINK_COUNT_LINK_RECOVERY_SUCCESSFUL_EVENTS",
        "DCGM_FI_DEV_NVLINK_RECOVERY_SUCCESSFUL_TOTAL",
    ),
    "nvlink_link_recovery_total": (
        "DCGM_FI_DEV_NVLINK_COUNT_LINK_RECOVERY_EVENTS",
        "DCGM_FI_DEV_NVLINK_RECOVERY_EVENT_TOTAL",
    ),
}


def alias_survey(*published: str) -> dict[str, Any]:
    """A Hopper-style survey whose exporter publishes ``clock_throttle_reasons``
    under ``published`` (one name on a real exporter; the fake's default uses
    the canonical name, which no exporter does)."""

    value = survey()
    for entry in value["exporter"]["gpus"].values():
        entry["metric_names"]["clock_throttle_reasons"] = list(published)
    return value


def shipped_plan(probe: Any) -> dict[str, Any]:
    """The ``plan.json`` the replay installed on the node, decoded."""

    for call in probe.calls:
        if (
            call[0] == "install-source"
            and call[call.index("--name") + 1] == "plan.json"
        ):
            return json.loads(base64.b64decode(call[call.index("--b64") + 1]))
    raise AssertionError(f"no plan.json was installed: {[c[0] for c in probe.calls]}")


def test_alias_families_are_the_products_pairs_the_profile_allows() -> None:
    """``collectors/gpu/dcgm.DCGM_METRICS`` inverted and limited to the
    completeness profile: the clock event reasons pair; the T.Limit pair is a
    thermal-safety reading the profile excludes."""

    assert fields.alias_families() == {
        "clock_throttle_reasons": (CURRENT_ALIAS, LEGACY_ALIAS),
        **BLACKWELL_ALIAS_FAMILIES,
    }, fields.alias_families()


@pytest.mark.parametrize(
    "published,injected", [(LEGACY_ALIAS, CURRENT_ALIAS), (CURRENT_ALIAS, LEGACY_ALIAS)]
)
def test_conflicting_alias_injects_the_name_the_node_does_not_publish(
    tmp_path: Path, published: str, injected: str
) -> None:
    """Live 2026-10-26 (eksb, H200, dcgm-exporter 4.6.1): the exporter publishes
    the legacy name the variant hard-coded as its alias, and the replay aborted
    ("not a distinct alias"). The injected alias is now the family member the
    survey does not show, for either exporter generation."""

    served = audit_records({f"{G1}/clock_throttle_reasons#alias:{injected}": 64})
    state = _state(
        batches=[
            _batch(
                [f"invalid or ambiguous GPU observation: {G1}/clock_throttle_reasons"],
                at="2026-10-26T15:33:01+00:00",
            )
        ]
    )
    probe = dcgm_probe(alias_survey(published), audit=served)
    out = fields.conflicting_alias(
        _ctx(
            tmp_path,
            "conflicting_alias",
            probe=probe,
            cpu=gpu_cpu_python(gpu_state=[state, state]),
        )
    )
    assert out["verdict"] == "PASS", out["errors"]
    assert out["notes"] == [
        f"{G1}/clock_throttle_reasons has no latest accepted sample"
    ]
    assert out["alias_pair"] == {
        "field": "clock_throttle_reasons",
        "published": published,
        "injected": injected,
        "value": 64,
    }, out["alias_pair"]
    assert out["gpu_uuid"] == G1, "the highest-index GPU reporting the field"
    ops = [op for step in shipped_plan(probe)["steps"] for op in step["ops"]]
    assert ops == [
        {
            "op": "alias",
            "field": "clock_throttle_reasons",
            "gpus": [G1],
            "name": injected,
            "value": 64,
        }
    ], ops


def test_conflicting_alias_without_a_distinct_pair_is_not_run_before_arming(
    tmp_path: Path,
) -> None:
    """An exporter publishing both names of the family leaves no distinct alias
    to inject: NOT_RUN(CAPABILITY_ABSENT) naming the candidates, after the
    survey and before anything is installed -- never an abort mid-run."""

    probe = dcgm_probe(alias_survey(LEGACY_ALIAS, CURRENT_ALIAS))
    ctx = _ctx(
        tmp_path, "conflicting_alias", probe=probe, cpu=gpu_cpu_python(gpu_state=None)
    )
    with pytest.raises(VariantNotRun) as caught:
        fields.conflicting_alias(ctx)
    assert caught.value.reason == "CAPABILITY_ABSENT", caught.value.args
    assert caught.value.detail == (
        "no canonical field has two distinct exporter aliases on this node: "
        f"clock_throttle_reasons ({CURRENT_ALIAS}|{LEGACY_ALIAS}; published: "
        f"{CURRENT_ALIAS}, {LEGACY_ALIAS})"
        + "".join(
            f"; {field} ({'|'.join(members)}; published: none)"
            for field, members in sorted(BLACKWELL_ALIAS_FAMILIES.items())
        )
    ), caught.value.args
    assert [call[0] for call in probe.calls] == ["survey"], probe.calls


def test_conflicting_alias_value_always_differs_from_the_real_reading() -> None:
    assert fields.conflicting_value(1.0) == 64, "the usual idle bitmask"
    assert fields.conflicting_value(None) == 64, "no real reading"
    assert fields.conflicting_value(64.0) == 32, "the real reading is the alias value"


# The live 2026-10-26 15:30Z chronology (HyperPod hpb, H200, R13g): the repoint
# restarted the collector at 15:31:18 and its initial-baseline scrape delivered
# the real clock reading 1.0; the plan's three pass-through reads followed
# (15:31:43, 15:32:04, 15:32:24; one of them a delivered summary carrying 1.0
# again); the alias was served from read 5 (~15:32:30) and the first conflicting
# batch the product delivered and rejected was 15:33:01 (the collector confirms a
# clock-reason candidate over three scrapes before it ships the edge). Every
# conflicting batch holds the product's rejection and no sample of the slot; the
# other GPU's reading of the same field is accepted beside it.
CLOCK = "clock_throttle_reasons"
AMBIGUOUS = f"invalid or ambiguous GPU observation: {G1}/{CLOCK}"
BASELINE_AT = "2026-10-26T15:31:33.100000+00:00"
PASS_THROUGH_AT = "2026-10-26T15:32:04.204132+00:00"
CONFLICT_AT = "2026-10-26T15:33:01.312457+00:00"
CONFLICT_LATER = (
    "2026-10-26T15:33:16.412457+00:00",
    "2026-10-26T15:33:31.512457+00:00",
)


def _clock_sample(gpu: str, value: float) -> dict[str, Any]:
    return {"gpu": gpu, "name": CLOCK, "value": value}


def _alias_chronology(
    *,
    inside: list[dict[str, Any]] | None = None,
    latest_at: str = PASS_THROUGH_AT,
    latest_value: float = 1.0,
    rejected: bool = True,
) -> dict[str, Any]:
    """The live shape; ``inside`` adds samples to the first conflicting batch."""

    rejection = [AMBIGUOUS] if rejected else []
    return _state(
        batches=[
            _batch(
                [],
                samples=[_clock_sample(G0, 1.0), _clock_sample(G1, 1.0)],
                at=BASELINE_AT,
            ),
            _batch(
                [],
                samples=[_clock_sample(G0, 1.0), _clock_sample(G1, 1.0)],
                at=PASS_THROUGH_AT,
            ),
            _batch(
                rejection,
                samples=[_clock_sample(G0, 1.0), *(inside or [])],
                at=CONFLICT_AT,
            ),
            *[
                _batch(rejection, samples=[_clock_sample(G0, 1.0)], at=at)
                for at in CONFLICT_LATER
            ],
        ],
        latest=[
            _latest(CLOCK, 1.0, CONFLICT_LATER[-1], gpu=G0),
            _latest(CLOCK, latest_value, latest_at, gpu=G1),
        ],
    )


def _conflicting_alias(tmp_path: Path, state: dict[str, Any], *, ctx_hook=None):
    served = audit_records({f"{G1}/{CLOCK}#alias:{CURRENT_ALIAS}": 64})
    ctx = _ctx(
        tmp_path,
        "conflicting_alias",
        probe=dcgm_probe(alias_survey(LEGACY_ALIAS), audit=served),
        cpu=gpu_cpu_python(gpu_state=[state, state]),
    )
    if ctx_hook is not None:
        ctx_hook(ctx)
    return fields.conflicting_alias(ctx)


def test_conflicting_alias_passes_the_live_chronology(tmp_path: Path) -> None:
    """The pass-through reads before the conflict delivered the real 1.0 and are
    not graded; from the first conflicting batch the product dropped the whole
    slot and kept the earlier reading as the latest row. The live run FAILed on
    exactly this shape ("a conflicting alias slot was accepted as a value: 1.0")
    because it graded every sample since a moment before the arm."""

    out = _conflicting_alias(tmp_path, _alias_chronology())
    assert out["verdict"] == "PASS", out["errors"]
    assert out["first_conflicting_batch_at"] == CONFLICT_AT
    assert out["window_opened_at"] == CONFLICT_AT
    assert out["latest"] == _latest(CLOCK, 1.0, PASS_THROUGH_AT), out["latest"]
    (note,) = out["notes"]
    assert (
        f"{G1}/{CLOCK} kept its earlier accepted value 1.0 from {PASS_THROUGH_AT}, "
        f"before the first conflicting batch at {CONFLICT_AT}; the ambiguous slot "
        "was dropped" == note
    ), note


def test_conflicting_alias_fails_when_the_published_reading_continues(
    tmp_path: Path,
) -> None:
    """The product drops both readings of an ambiguous slot: the published name's
    1.0 accepted beside the rejection is a sample the slot must not produce."""

    out = _conflicting_alias(
        tmp_path, _alias_chronology(inside=[_clock_sample(G1, 1.0)])
    )
    assert out["verdict"] == "FAIL"
    assert out["stages"]["ambiguous_rejected"] == []
    assert out["stages"]["no_silent_overwrite"] == [
        f"a {CLOCK} sample of {G1} was accepted while the conflicting alias was "
        f"served (from {CONFLICT_AT}): values ['1.0']; the product drops the whole "
        "slot of an ambiguous batch"
    ], out["stages"]


def test_conflicting_alias_fails_when_the_injected_value_becomes_a_sample(
    tmp_path: Path,
) -> None:
    """The alias value is never a reading of the slot, whatever the window and
    whether or not the product also reported the ambiguity. With no rejection
    to anchor on the window opens at the oracle gate (here between the last
    pass-through read and the conflict), so the pass-through 1.0s stay out."""

    gate = datetime(2026, 10, 26, 15, 32, 44, tzinfo=timezone.utc)
    out = _conflicting_alias(
        tmp_path,
        _alias_chronology(
            inside=[_clock_sample(G1, 64.0)],
            latest_at=CONFLICT_AT,
            latest_value=64.0,
            rejected=False,
        ),
        ctx_hook=lambda ctx: store_poll_clock(ctx, start=gate),
    )
    assert out["verdict"] == "FAIL"
    assert out["first_conflicting_batch_at"] is None
    assert datetime.fromisoformat(out["window_opened_at"]) >= gate
    assert out["stages"]["ambiguous_rejected"] == [
        "a conflicting field alias was not reported as an ambiguous slot"
    ]
    assert out["stages"]["no_silent_overwrite"] == [
        f"the injected alias value 64 was accepted as a {CLOCK} sample of {G1} at "
        f"{CONFLICT_AT}",
        f"the injected alias value 64 is the latest accepted {CLOCK} value of {G1}",
    ], out["stages"]


def test_conflicting_alias_fails_when_the_latest_row_moves_into_the_conflict(
    tmp_path: Path,
) -> None:
    out = _conflicting_alias(tmp_path, _alias_chronology(latest_at=CONFLICT_LATER[0]))
    assert out["verdict"] == "FAIL"
    assert out["stages"]["no_silent_overwrite"] == [
        f"a {CLOCK} reading of {G1} became the latest accepted value 1.0 at "
        f"{CONFLICT_LATER[0]}, not before the first conflicting batch at {CONFLICT_AT}"
    ], out["stages"]


def test_conflicting_alias_window_opens_at_the_oracle_gate_without_a_rejection(
    tmp_path: Path,
) -> None:
    """A product that never reported the ambiguity leaves no batch to anchor on;
    the window then opens when the oracle gate saw the alias served (every
    later read is on the held step). Samples before it are the pass-through
    reads; a sample after it is the published reading kept beside an
    unreported conflict."""

    gate = datetime(2026, 10, 26, 15, 32, 44, tzinfo=timezone.utc)
    later = "2026-10-26T15:40:00.123456+00:00"
    state = _state(
        batches=[
            _batch([], samples=[_clock_sample(G1, 1.0)], at=BASELINE_AT),
            _batch([], samples=[_clock_sample(G1, 1.0)], at=later),
        ],
        latest=[_latest(CLOCK, 1.0, later)],
    )
    out = _conflicting_alias(
        tmp_path, state, ctx_hook=lambda ctx: store_poll_clock(ctx, start=gate)
    )
    assert out["verdict"] == "FAIL"
    assert out["first_conflicting_batch_at"] is None
    opened = datetime.fromisoformat(out["window_opened_at"])
    assert gate <= opened < datetime.fromisoformat(later), out["window_opened_at"]
    assert out["stages"]["no_silent_overwrite"] == [
        f"a {CLOCK} sample of {G1} was accepted while the conflicting alias was "
        f"served (from {opened.isoformat()}): values ['1.0']; the product drops the "
        "whole slot of an ambiguous batch",
        f"a {CLOCK} reading of {G1} became the latest accepted value 1.0 at {later}, "
        f"not before the oracle gate at {opened.isoformat()}",
    ], out["stages"]


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


DUPLICATE_REJECTED = f"invalid or ambiguous GPU observation: {G1}/pcie_replay_total"


def _duplicate_slot(tmp_path: Path, state: dict[str, Any]):
    served = audit_records({f"{G1}/pcie_replay_total#duplicate": 7})
    return fields.duplicate_slot(
        _ctx(
            tmp_path,
            "duplicate_slot",
            probe=dcgm_probe(audit=served),
            cpu=gpu_cpu_python(gpu_state=[state, state]),
        )
    )


def test_duplicate_slot_rejects_and_keeps_other_gpus(tmp_path: Path) -> None:
    """The other GPU's reading of the duplicated field is accepted in the very
    batch that rejects the duplicate: its latest row moves to that scrape."""

    state = _state(
        batches=[
            _batch(
                [DUPLICATE_REJECTED],
                samples=[{"gpu": G0, "name": "pcie_replay_total", "value": 4.0}],
                at=REPLAYED_AT,
            )
        ],
        latest=[_latest("pcie_replay_total", 4.0, REPLAYED_AT, gpu=G0)],
    )
    out = _duplicate_slot(tmp_path, state)
    assert out["verdict"] == "PASS", out["errors"]
    assert out["first_replayed_batch_at"] == REPLAYED_AT


def test_duplicate_slot_accepts_the_other_gpus_latest_row_inside_the_replay(
    tmp_path: Path,
) -> None:
    """A read whose batches carry no sample of the other GPU (the field filter
    of an older probe) still proves it usable by the latest row's time."""

    state = _state(
        batches=[_batch([DUPLICATE_REJECTED], at=REPLAYED_AT)],
        latest=[_latest("pcie_replay_total", 4.0, REPLAYED_AT, gpu=G0)],
    )
    assert _duplicate_slot(tmp_path, state)["verdict"] == "PASS"


def test_duplicate_slot_fails_when_the_other_gpu_stopped_at_the_replay(
    tmp_path: Path,
) -> None:
    """A latest row from before the arm proved nothing about the replay: the
    stage used to pass on any row the other GPU ever had."""

    state = _state(
        batches=[_batch([DUPLICATE_REJECTED], at=REPLAYED_AT)],
        latest=[_latest("pcie_replay_total", 4.0, EARLIER, gpu=G0)],
    )
    out = _duplicate_slot(tmp_path, state)
    assert out["verdict"] == "FAIL"
    assert out["stages"]["duplicate_rejected"] == []
    assert out["stages"]["others_usable"] == [
        f"a duplicate slot on {G1} also dropped {G0}'s independent sample: no "
        f"pcie_replay_total sample of {G0} was accepted at or after the first "
        f"replayed scrape at {REPLAYED_AT} (latest row: {EARLIER})"
    ], out["stages"]


def test_duplicate_slot_cannot_judge_the_others_without_the_rejection(
    tmp_path: Path,
) -> None:
    state = _state(
        batches=[_batch([], at=REPLAYED_AT)],
        latest=[_latest("pcie_replay_total", 4.0, REPLAYED_AT, gpu=G0)],
    )
    out = _duplicate_slot(tmp_path, state)
    assert out["verdict"] == "FAIL"
    assert out["first_replayed_batch_at"] is None
    assert out["stages"]["duplicate_rejected"] == [
        "a duplicate field slot was not reported as ambiguous"
    ]
    assert out["stages"]["others_usable"] == [
        f"{G0}'s samples cannot be judged: no batch since the arm carries the "
        f"product's rejection of the duplicate slot on {G1}"
    ]


@pytest.mark.parametrize(
    "text",
    [
        f"DCGM exporter is missing required fields on {G1}: pcie_replay_total",
        "DCGM exporter is missing required fields: row_remap_failure",
        f"invalid or ambiguous GPU observation: {G1}/pcie_replay_total",
    ],
)
def test_missing_field_accepts_only_the_batch_wide_omission_of_its_field(
    tmp_path: Path, text: str
) -> None:
    """A GPU-scoped gap, another field's gap or an invalid slot inside the
    window (it opens before the arm) is not the replay's scrape-wide omission;
    the stage used to pass on any "missing required fields" text."""

    served = audit_records(
        {f"{G0}/pcie_replay_total": "omitted", f"{G1}/pcie_replay_total": "omitted"}
    )
    state = _state(batches=[_batch([text])])
    out = fields.missing_field(
        _ctx(
            tmp_path,
            "missing_field",
            probe=dcgm_probe(audit=served),
            cpu=gpu_cpu_python(gpu_state=[state, state]),
        )
    )
    assert out["verdict"] == "FAIL", out
    assert fields.BATCH_WIDE_OMISSION.search(
        "DCGM exporter is missing required fields: row_remap_failure, pcie_replay_total"
    ), "a scrape-wide message listing several fields names the omitted one"


def test_missing_field_needs_a_gpu_reporting_the_field(tmp_path: Path) -> None:
    without = survey(fields=("ecc_dbe_volatile_total", "clock_throttle_reasons"))
    probe = dcgm_probe(without)
    ctx = _ctx(tmp_path, "missing_field", probe=probe, cpu=gpu_cpu_python())
    with pytest.raises(VariantNotRun, match="CAPABILITY_ABSENT.*pcie_replay_total"):
        fields.missing_field(ctx)
    assert [call[0] for call in probe.calls] == ["survey"], probe.calls


def test_one_gpu_missing_is_inconclusive_when_the_omit_was_never_served(
    tmp_path: Path,
) -> None:
    """The gate ``missing_field`` has: the settle starts once the omit is served,
    and a replay that never served it ends INCONCLUSIVE instead of grading the
    pass-through reads."""

    from scripts.e2e.regional.general_live_fixture import VariantInconclusive

    state = _state(batches=[_batch([], at=REPLAYED_AT)])
    with pytest.raises(VariantInconclusive, match="never served"):
        fields.one_gpu_missing(
            _ctx(
                tmp_path,
                "one_gpu_missing",
                probe=dcgm_probe(audit=audit_records()),
                cpu=gpu_cpu_python(gpu_state=[state, state]),
            )
        )

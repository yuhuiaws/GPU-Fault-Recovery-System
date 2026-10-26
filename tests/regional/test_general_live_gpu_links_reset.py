"""GF-GENERAL-LIVE-020 ``counter_reset`` judges the drop against the raise's finding.

Live 2026-10-26 (HyperPod, H200, the DL aggregate
``nvlink_crc_aggregate_error_total``, harness live17) the variant FAILed with
"no_fault: a counter reset minted a finding; active: a counter reset left an
active finding". Both stages were the oracle's own error. The NVLink rule has
no under-limit raise -- ``gpu_metrics._link_decision`` grades every increase
of an NVLink error counter (``delta >= nvlink_error_delta_critical``, default
1) as the WARNING -- so the plan's +5 per read raise minted the one incident
the oracle blamed on the reset; and the product's contract keeps an active
finding across a decrease (an invalid delta, ``gpu_metrics.ingest``), so "no
active finding after the reset" demanded the recovery the variant's own note
denied. Replayed through the real collector filter and ingest, the live reads
(0,0,0,5,10,15,0...) mint at the first raised read, ship the drop as a
``counter-reset`` edge, mint nothing at it and leave the finding active with
the latest reading at 0. The variant now raises by one per read and grades
the snapshot that first holds the drop: raise / reset / no_fault /
no_recovery / hardware, on the product's own records.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from scripts.e2e.regional import general_live_gpu_links as links
from tests.regional._general_live_fakes import FakeRegional, make_context, make_settings
from tests.regional._general_live_gpu_fakes import (
    audit_records,
    dcgm_probe,
    gpu_cpu_python,
    gpu_incident,
    survey,
)

SPEC = next(
    spec
    for spec in links.VARIANTS
    if spec.case_id == links.CASE_020 and spec.variant == "counter_reset"
)
G1 = "GPU-11111111-1111-4111-8111-111111111111"
#: The default fake survey reports the legacy per-class counter on both GPUs.
FLIT = "nvlink_crc_flit_error_total"
AGGREGATE = "nvlink_crc_aggregate_error_total"
DL_CRC = "DCGM_FI_DEV_NVLINK_ERROR_DL_CRC_total"


def _at(minute: int, second: int = 0) -> str:
    return f"2026-10-26T13:{minute:02d}:{second:02d}+00:00"


def _ctx(tmp_path: Path, *, probe: Any, cpu: Any):
    regional = FakeRegional(cpu_python=cpu)
    settings = make_settings(SPEC, node=regional.node_snapshot("")["name"])
    ctx = make_context(
        SPEC, tmp_path=tmp_path, regional=regional, probes={}, settings=settings
    )
    ctx.probes[f"dcgm@{ctx.settings.node}"] = probe
    return ctx


def _activity(incidents: list[dict[str, Any]] = ()) -> dict[str, Any]:
    return {
        "incidents": list(incidents),
        "workflows": [],
        "batches": [{"observed_at": "t"}],
        "markers": [],
        "commands": [],
        "notifications": [],
        "metric_latest": [],
        "collector_statuses": [],
        "agent": None,
    }


def _batch(
    value: float, observed_at: str, *reasons: str, canonical: str = FLIT
) -> dict[str, Any]:
    return {
        "batch_id": f"dcgm-node-{observed_at}",
        "observed_at": observed_at,
        "edge_filter_reasons": list(reasons),
        "collection_errors": [],
        "sample_count": 1,
        "gpus": [G1],
        "samples": [
            {
                "gpu": G1,
                "name": canonical,
                "value": value,
                "labels": {"source_boot_id": survey()["boot_id"]},
            }
        ],
    }


def _finding(
    value: float, observed_at: str, *, delta: float | None, canonical: str = FLIT
) -> dict[str, Any]:
    return {
        "finding_id": f"dcgm-node-{observed_at}-{G1}-{canonical}",
        "canonical_name": canonical,
        "severity": "WARNING",
        "value": value,
        "delta": delta,
        "rate_per_minute": None,
        "gpu_uuid": G1,
        "observed_at": observed_at,
        "automatic_action": "RUN_DIAGNOSTICS",
        "reason": "NVLink error counter increased; diagnose link errors and rate",
    }


def _state(
    batches: list[dict[str, Any]],
    *,
    latest: dict[str, Any] | None = None,
    findings: list[dict[str, Any]] = (),
    active: list[dict[str, Any]] = (),
) -> dict[str, Any]:
    last = batches[-1]["samples"][0] if batches else None
    if latest is None and last is not None:
        latest = {
            "gpu": G1,
            "name": last["name"],
            "value": last["value"],
            "observed_at": batches[-1]["observed_at"],
            "source": "DCGM_EXPORTER",
            "labels": dict(last["labels"]),
        }
    return {
        "latest": [latest] if latest else [],
        "active_findings": list(active),
        "findings": list(findings),
        "xid_events": [],
        "batches": batches,
        "inventory_records": [],
        "inventory_snapshot": None,
        "collector_statuses": [],
        "agent": None,
        "capabilities": {},
    }


#: The live run's shape with the new raise: baseline 0, +1 per read, the drop.
RAISE_FINDING = _finding(1, _at(24, 45), delta=1)
RAISE_INCIDENT = gpu_incident("inc-raise", RAISE_FINDING["finding_id"])


def _records(canonical: str = FLIT) -> list[dict[str, Any]]:
    return [
        _batch(0, _at(23, 45), "initial-baseline", canonical=canonical),
        _batch(1, _at(24, 45), "counter-increased", canonical=canonical),
        _batch(2, _at(25, 0), "counter-increased", canonical=canonical),
        _batch(3, _at(25, 15), "counter-increased", canonical=canonical),
        _batch(0, _at(25, 30), "counter-reset", canonical=canonical),
    ]


def _run(
    tmp_path: Path,
    state: dict[str, Any],
    incidents: list[dict[str, Any]] = (),
    *,
    probe: Any = None,
) -> tuple[dict[str, Any], Any]:
    ctx = _ctx(
        tmp_path,
        probe=probe or dcgm_probe(audit=audit_records({f"{G1}/{FLIT}": 1.0})),
        cpu=gpu_cpu_python(activity=_activity(incidents), gpu_state=state),
    )
    return links.counter_reset(ctx), ctx


def _plan(ctx: Any) -> dict[str, Any]:
    return json.loads((ctx.case_dir / "plan.json").read_text(encoding="utf-8"))


def _readings(ctx: Any) -> dict[str, Any]:
    return json.loads((ctx.case_dir / "reset-readings.json").read_text())


# --------------------------------------------------------------------------- #
# the plan
# --------------------------------------------------------------------------- #
def test_counter_reset_raises_by_the_smallest_increase_then_clears(tmp_path: Path):
    state = _state(_records(), findings=[RAISE_FINDING], active=[RAISE_FINDING])

    out, ctx = _run(tmp_path, state, [RAISE_INCIDENT])

    assert out["verdict"] == "PASS", out["errors"]
    assert links.NVLINK_RAISE == 1, (
        "the NVLink rule is delta >= 1: no under-limit raise"
    )
    steps = _plan(ctx)["steps"]
    assert [step["reads"] for step in steps] == [3, 3, 20]
    assert steps[1]["ops"] == [
        {"op": "add", "field": FLIT, "gpus": [G1], "value": 1}
    ], (
        "live 2026-10-26 the raise was +5 per read; any size mints the WARNING, "
        "one per read is the smallest honest increase"
    )
    assert steps[2]["ops"] == [{"op": "clear", "field": FLIT, "gpus": [G1]}]
    assert steps[2]["until_advance"] is True


# --------------------------------------------------------------------------- #
# the oracle over the product's records
# --------------------------------------------------------------------------- #
def test_counter_reset_passes_when_the_drop_keeps_the_raises_finding(tmp_path: Path):
    """The live run's records: one incident from the raise, the finding still
    active after the drop, nothing minted at it -- the product's contract."""

    state = _state(_records(), findings=[RAISE_FINDING], active=[RAISE_FINDING])

    out, ctx = _run(tmp_path, state, [RAISE_INCIDENT])

    assert out["verdict"] == "PASS", out["errors"]
    assert set(out["stages"]) == {
        "raise",
        "reset",
        "no_fault",
        "no_recovery",
        "hardware",
    }
    readings = _readings(ctx)
    assert readings["reset_ticks"][0]["previous"] == 3
    assert readings["reset_ticks"][0]["value"] == 0
    assert readings["reset_ticks"][0]["edge_filter_reasons"] == ["counter-reset"]
    assert readings["latest"]["value"] == 0
    assert readings["incidents"] == ["inc-raise"]
    assert readings["no_recovery_judgement"] == (
        "the raise's finding is still active after the drop"
    )
    assert out["canonical_name"] == FLIT and out["nvlink_family"] == "nvlink_crc"
    assert any("neither a new fault nor a recovery" in note for note in out["notes"]), (
        out["notes"]
    )


def test_counter_reset_fails_when_the_drop_mints_a_second_incident(tmp_path: Path):
    late = _finding(0, _at(25, 30), delta=None)
    second = gpu_incident("inc-reset", late["finding_id"])
    state = _state(
        _records(), findings=[RAISE_FINDING, late], active=[RAISE_FINDING, late]
    )

    out, _ctx_ = _run(tmp_path, state, [RAISE_INCIDENT, second])

    assert out["verdict"] == "FAIL"
    assert out["stages"]["no_fault"] == [
        f"a counter reset minted a finding: value=0 delta=None rate=None/min at {_at(25, 30)}",
        "a counter reset minted a second incident: value=0 delta=None rate=None/min "
        f"at {_at(25, 30)}; incident inc-reset",
    ]
    assert out["stages"]["raise"] == [] and out["stages"]["reset"] == []


def test_counter_reset_fails_when_the_drop_clears_the_finding(tmp_path: Path):
    """The old oracle demanded this; the product's contract forbids it."""

    state = _state(_records(), findings=[RAISE_FINDING], active=[])

    out, ctx = _run(tmp_path, state, [RAISE_INCIDENT])

    assert out["verdict"] == "FAIL"
    assert out["stages"]["no_recovery"] == [
        f"the counter reset cleared the active {FLIT} finding on {G1}: a decrease "
        "is an invalid delta that keeps the finding, not a recovery"
    ]
    assert out["stages"]["no_fault"] == []
    assert _readings(ctx)["no_recovery_judgement"] == "the drop cleared the finding"


def test_counter_reset_does_not_judge_recovery_once_a_later_batch_intervened(
    tmp_path: Path,
):
    """A health summary after the drop is a zero delta over a real window: the
    product's legitimate recovery, so the active state is not attributable."""

    batches = [*_records(), _batch(0, _at(28, 45), "health-summary")]
    state = _state(batches, findings=[RAISE_FINDING], active=[])

    out, ctx = _run(tmp_path, state, [RAISE_INCIDENT])

    assert out["verdict"] == "PASS", out["errors"]
    judgement = _readings(ctx)["no_recovery_judgement"]
    assert judgement.startswith(
        f"not judged: a later batch of the counter intervened before the snapshot ({_at(28, 45)})"
    ), judgement


def test_counter_reset_requires_the_raise_to_have_minted(tmp_path: Path):
    state = _state(_records())

    out, _ctx_ = _run(tmp_path, state)

    assert out["verdict"] == "FAIL"
    assert out["stages"]["raise"] == [
        f"the raise minted no {FLIT} finding on {G1}: every NVLink error increase "
        "(delta >= 1) is the WARNING the reset is judged against"
    ]
    assert out["stages"]["no_recovery"] == [], "nothing to judge without the finding"


def test_counter_reset_requires_the_raise_and_the_drop_to_be_accepted(tmp_path: Path):
    only_baseline = _state([_batch(0, _at(23, 45), "initial-baseline")])
    out, _ctx_ = _run(tmp_path, only_baseline)
    assert out["verdict"] == "FAIL"
    assert out["stages"]["raise"] == [
        f"the product accepted no raised {FLIT} reading of {G1} (accepted values: 0)"
    ]
    assert out["stages"]["reset"] == [], "no raise, nothing to drop from"

    no_drop = _state(_records()[:4], findings=[RAISE_FINDING], active=[RAISE_FINDING])
    out, _ctx_ = _run(tmp_path / "no-drop", no_drop, [RAISE_INCIDENT])
    assert out["verdict"] == "FAIL"
    assert out["stages"]["reset"] == [
        f"the product accepted no {FLIT} reading of {G1} below its predecessor "
        "(accepted values: 0, 1, 2, 3)"
    ]


def test_counter_reset_requires_the_counter_reset_edge_and_the_rearmed_baseline(
    tmp_path: Path,
):
    batches = [
        _batch(0, _at(23, 45), "initial-baseline"),
        _batch(3, _at(25, 15), "counter-increased"),
        _batch(0, _at(25, 30), "health-summary"),
    ]
    latest = {
        "gpu": G1,
        "name": FLIT,
        "value": 3,
        "observed_at": _at(25, 15),
        "source": "DCGM_EXPORTER",
        "labels": {},
    }
    raise_finding = _finding(3, _at(25, 15), delta=3)
    state = _state(
        batches, latest=latest, findings=[raise_finding], active=[raise_finding]
    )

    out, _ctx_ = _run(
        tmp_path, state, [gpu_incident("inc-raise", raise_finding["finding_id"])]
    )

    assert out["verdict"] == "FAIL"
    assert out["stages"]["reset"] == [
        f"the drop 3 -> 0 at {_at(25, 30)} was delivered without a counter-reset "
        "edge reason (health-summary)",
        f"the product's latest {FLIT} reading of {G1} is 3, not the reset value 0: "
        "the baseline was not re-armed from the drop",
    ]


def test_counter_reset_drives_the_dl_aggregate_a_hopper_exporter_reports(
    tmp_path: Path,
):
    """S18 targeting on the live H200 shape: the plan, the served key and the
    stages all follow the family member the node really publishes."""

    hopper = survey(
        fields=(
            "ecc_sbe_volatile_total",
            "ecc_dbe_volatile_total",
            "pcie_replay_total",
            "row_remap_failure",
            "fabric_manager_status",
            AGGREGATE,
        )
    )
    for entry in hopper["exporter"]["gpus"].values():
        entry["metric_names"][AGGREGATE] = [DL_CRC]
    finding = _finding(1, _at(24, 45), delta=1, canonical=AGGREGATE)
    state = _state(_records(AGGREGATE), findings=[finding], active=[finding])
    probe = dcgm_probe(hopper, audit=audit_records({f"{G1}/{AGGREGATE}": 1.0}))

    out, ctx = _run(
        tmp_path, state, [gpu_incident("inc-raise", finding["finding_id"])], probe=probe
    )

    assert out["verdict"] == "PASS", out["errors"]
    assert out["canonical_name"] == AGGREGATE
    assert out["exporter_metrics"] == [DL_CRC]
    assert _plan(ctx)["steps"][1]["ops"][0]["field"] == AGGREGATE
    assert _readings(ctx)["reset_ticks"][0]["previous"] == 3


def test_the_registry_text_names_the_rule_and_the_kept_finding() -> None:
    assert "delta >= 1" in SPEC.injection and "by one per read" in SPEC.injection
    assert (
        "counter-reset" in SPEC.expected
        and "keeps the raise's finding" in SPEC.expected
    )
    assert SPEC.expected_outcome == "PASS" and not SPEC.product_gap

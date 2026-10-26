"""GF-GENERAL-LIVE-022 ``counter_reset`` / ``source_boot_change`` grade the boundary.

Live 2026-10-26 (plain EKS, release R13g, harness live14) both variants
FAILed on a WARNING the plan's own raise had minted: ``counter_reset`` raised
the PCIe replay counter by 50 per 15-s read (200/min against the 8/min rule)
and read the storm's incident as "a PCIe counter reset minted a finding";
``source_boot_change`` ramped +5 per read (20/min, the ``over_rate_limit``
increment) inside the replayed boot after the repointed collector's
pass-through first scrape, so the only boundary it crossed was (0, real boot)
-> (0, replayed boot), and it read the same-generation storm as "a
cross-generation PCIe rate". The product's records (pinned in
``tests/metrics/test_pcie_counter_boundaries.py``) grade both boundaries as
unknown and start a new baseline. The variants now drive the boundary itself
-- an under-limit raise before the drop; the jump as the new boot's very
first reading, armed before the repoint -- and judge the product's own
records: accepted readings, the ``counter-reset`` edge, the latest reading's
value and boot, findings placed before/at/after the boundary.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import general_live_gpu_links as links
from scripts.e2e.regional.general_live_fixture import VariantInconclusive
from scripts.e2e.regional.general_live_gpu_replay import (
    at_limit_increment,
    cross_boot_jump,
    over_limit_increment,
    under_limit_increment,
)
from tests.regional._general_live_fakes import FakeRegional, make_context, make_settings
from tests.regional._general_live_gpu_fakes import (
    audit_records,
    dcgm_probe,
    gpu_cpu_python,
    gpu_incident,
    survey,
)

SPECS = {spec.variant: spec for spec in links.VARIANTS}
G1 = "GPU-11111111-1111-4111-8111-111111111111"
PCIE = "pcie_replay_total"
KEY = f"{G1}/{PCIE}"
REAL_BOOT = survey()["boot_id"]
PRIOR_BOOT = survey()["prior_boot_ids"][0]
#: The jump the variant serves on a 15-s collector with a 300-s health summary.
JUMP = cross_boot_jump(15, 300)


def _at(minute: int, second: int = 0) -> str:
    return f"2026-10-26T07:{minute:02d}:{second:02d}+00:00"


def _ctx(tmp_path: Path, variant: str, *, probe: Any, cpu: Any):
    spec = SPECS[variant]
    regional = FakeRegional(cpu_python=cpu)
    settings = make_settings(spec, node=regional.node_snapshot("")["name"])
    ctx = make_context(
        spec, tmp_path=tmp_path, regional=regional, probes={}, settings=settings
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
    value: float, observed_at: str, *reasons: str, boot: str = REAL_BOOT
) -> dict[str, Any]:
    """One accepted GPU_METRICS batch as ``general_live_gpu_store`` lists it."""

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
                "name": PCIE,
                "value": value,
                "labels": {"source_boot_id": boot, "modelName": "NVIDIA H200"},
            }
        ],
    }


def _finding(
    value: float, observed_at: str, *, delta: float | None, rate: float | None
) -> dict[str, Any]:
    return {
        "finding_id": f"dcgm-node-{observed_at}-{G1}-{PCIE}",
        "canonical_name": PCIE,
        "severity": "WARNING",
        "value": value,
        "delta": delta,
        "rate_per_minute": rate,
        "gpu_uuid": G1,
        "observed_at": observed_at,
        "automatic_action": "RUN_DIAGNOSTICS",
        "reason": "PCIe replay rate exceeded DCGM health threshold",
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
            "name": PCIE,
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


def _plan(ctx: Any) -> dict[str, Any]:
    return json.loads((ctx.case_dir / "plan.json").read_text(encoding="utf-8"))


def _calls(ctx: Any) -> list[str]:
    return [call[0] for call in ctx.probes[f"dcgm@{ctx.settings.node}"].calls]


# --------------------------------------------------------------------------- #
# the increments
# --------------------------------------------------------------------------- #
def test_the_under_limit_increment_stays_clear_of_the_rule_on_every_interval() -> None:
    for interval in (10, 15, 20, 30, 60):
        increment = under_limit_increment(interval)
        assert 1 <= increment < max(2, at_limit_increment(interval)) or increment == 1
        assert increment * 60 / interval < 8, (interval, increment)
    assert under_limit_increment(15) == 1, "4/min on the product's default interval"
    assert over_limit_increment(15) == 5, "the live source_boot_change ramp: 20/min"


def test_the_cross_boot_jump_rates_over_the_rule_against_any_old_boot_reading() -> None:
    assert JUMP == 93
    # The old boot's latest sample is at most one health summary, one scrape
    # and the restart old; the jump still rates over 8/min over twice that.
    oldest = 2 * (300 + 15) + 60
    assert JUMP * 60 / oldest > 8
    assert JUMP * 60 / 300 > 8


# --------------------------------------------------------------------------- #
# 022 counter_reset
# --------------------------------------------------------------------------- #
def _reset_state(**overrides: Any) -> dict[str, Any]:
    batches = [
        _batch(0, _at(7, 30), "initial-baseline"),
        _batch(1, _at(8, 30), "counter-increased"),
        _batch(2, _at(8, 45), "counter-increased"),
        _batch(3, _at(9, 0), "counter-increased"),
        _batch(0, _at(9, 15), "counter-reset"),
    ]
    return _state(batches, **overrides)


def _reset_ctx(tmp_path: Path, state: dict[str, Any], incidents: list = ()):
    return _ctx(
        tmp_path,
        "counter_reset",
        probe=dcgm_probe(audit=audit_records({KEY: 1.0})),
        cpu=gpu_cpu_python(activity=_activity(incidents), gpu_state=state),
    )


def test_counter_reset_raises_under_the_rate_limit_then_clears(tmp_path: Path) -> None:
    ctx = _reset_ctx(tmp_path, _reset_state())

    out = links.pcie_counter_reset(ctx)

    assert out["verdict"] == "PASS", out["errors"]
    steps = _plan(ctx)["steps"]
    assert [step["reads"] for step in steps] == [3, 3, 20]
    assert steps[1]["ops"] == [
        {"op": "add", "field": PCIE, "gpus": [G1], "value": 1}
    ], (
        "live 2026-10-26 the raise was +50 per read (200/min): the storm, not "
        "the reset, minted the WARNING the oracle then blamed on the reset"
    )
    assert steps[2]["ops"] == [{"op": "clear", "field": PCIE, "gpus": [G1]}]
    assert steps[2]["until_advance"] is True


def test_counter_reset_passes_on_the_products_own_records(tmp_path: Path) -> None:
    ctx = _reset_ctx(tmp_path, _reset_state())

    out = links.pcie_counter_reset(ctx)

    assert out["verdict"] == "PASS", out["errors"]
    assert set(out["stages"]) == {"raise", "reset", "no_fault", "hardware"}
    readings = json.loads((ctx.case_dir / "reset-readings.json").read_text())
    assert readings["reset_ticks"][0]["previous"] == 3
    assert readings["reset_ticks"][0]["value"] == 0
    assert readings["reset_ticks"][0]["edge_filter_reasons"] == ["counter-reset"]
    assert readings["latest"]["value"] == 0


def test_counter_reset_names_a_finding_minted_at_the_reset_tick(tmp_path: Path) -> None:
    finding = _finding(0, _at(9, 15), delta=None, rate=None)
    inc = gpu_incident("inc-reset", finding["finding_id"])
    ctx = _reset_ctx(tmp_path, _reset_state(findings=[finding]), incidents=[inc])

    out = links.pcie_counter_reset(ctx)

    assert out["verdict"] == "FAIL"
    assert out["stages"]["no_fault"] == [
        "a PCIe counter reset minted a finding: value=0 delta=None rate=None/min "
        f"at {_at(9, 15)}; incident inc-reset"
    ]


def test_counter_reset_tells_a_finding_of_the_raise_from_one_of_the_reset(
    tmp_path: Path,
) -> None:
    """The live run's shape: the finding came from the raise, before the drop."""

    finding = _finding(50, _at(8, 30), delta=50, rate=200)
    inc = gpu_incident("inc-storm", finding["finding_id"])
    batches = [
        _batch(0, _at(7, 30), "initial-baseline"),
        _batch(50, _at(8, 30), "counter-increased"),
        _batch(100, _at(8, 45), "counter-increased"),
        _batch(150, _at(9, 0), "counter-increased"),
        _batch(0, _at(9, 15), "counter-reset"),
    ]
    ctx = _reset_ctx(
        tmp_path, _state(batches, findings=[finding], active=[finding]), incidents=[inc]
    )

    out = links.pcie_counter_reset(ctx)

    assert out["verdict"] == "FAIL"
    assert out["stages"]["no_fault"] == [
        "the raise before the reset minted a finding (the plan's increment must stay "
        "under 8/min): value=50 delta=50 rate=200/min at "
        f"{_at(8, 30)}; incident inc-storm",
        "a counter reset left an active finding: value=50 delta=50 rate=200/min at "
        f"{_at(8, 30)}",
    ]
    assert out["stages"]["reset"] == [], "the drop itself was recorded as a reset"


def test_counter_reset_fails_when_the_product_never_recorded_the_drop(
    tmp_path: Path,
) -> None:
    batches = [
        _batch(0, _at(7, 30), "initial-baseline"),
        _batch(1, _at(8, 30), "counter-increased"),
        _batch(2, _at(8, 45), "counter-increased"),
        _batch(3, _at(9, 0), "counter-increased"),
    ]
    ctx = _reset_ctx(tmp_path, _state(batches))

    out = links.pcie_counter_reset(ctx)

    assert out["verdict"] == "FAIL"
    assert out["stages"]["raise"] == []
    assert out["stages"]["reset"] == [
        f"the product accepted no {PCIE} reading of {G1} below its predecessor "
        "(accepted values: 0, 1, 2, 3)"
    ]


def test_counter_reset_requires_the_counter_reset_edge_and_the_rearmed_baseline(
    tmp_path: Path,
) -> None:
    batches = [
        _batch(0, _at(7, 30), "initial-baseline"),
        _batch(3, _at(9, 0), "counter-increased"),
        _batch(0, _at(9, 15), "health-summary"),
    ]
    latest = {
        "gpu": G1,
        "name": PCIE,
        "value": 3,
        "observed_at": _at(9, 0),
        "source": "DCGM_EXPORTER",
        "labels": {"source_boot_id": REAL_BOOT},
    }
    ctx = _reset_ctx(tmp_path, _state(batches, latest=latest))

    out = links.pcie_counter_reset(ctx)

    assert out["verdict"] == "FAIL"
    assert out["stages"]["reset"] == [
        f"the drop 3 -> 0 at {_at(9, 15)} was delivered without a counter-reset "
        "edge reason (health-summary)",
        f"the product's latest {PCIE} reading of {G1} is 3, not the reset value 0: "
        "the baseline was not re-armed from the drop",
    ]


def test_counter_reset_fails_when_the_raise_was_never_accepted(tmp_path: Path) -> None:
    ctx = _reset_ctx(tmp_path, _state([_batch(0, _at(7, 30), "initial-baseline")]))

    out = links.pcie_counter_reset(ctx)

    assert out["verdict"] == "FAIL"
    assert out["stages"]["raise"] == [
        f"the product accepted no raised {PCIE} reading of {G1} (accepted values: 0)"
    ]


# --------------------------------------------------------------------------- #
# 022 source_boot_change
# --------------------------------------------------------------------------- #
def _boot_ctx(tmp_path: Path, state: dict[str, Any], incidents: list = ()):
    return _ctx(
        tmp_path,
        "source_boot_change",
        probe=dcgm_probe(audit=audit_records({KEY: float(JUMP)})),
        cpu=gpu_cpu_python(activity=_activity(incidents), gpu_state=state),
    )


def _boot_state(**overrides: Any) -> dict[str, Any]:
    return _state(
        [_batch(JUMP, _at(10, 55), "initial-baseline", boot=PRIOR_BOOT)], **overrides
    )


def test_source_boot_change_arms_before_the_repoint_and_jumps_on_the_first_read(
    tmp_path: Path,
) -> None:
    ctx = _boot_ctx(tmp_path, _boot_state())

    out = links.source_boot_change(ctx)

    assert out["verdict"] == "PASS", out["errors"]
    calls = _calls(ctx)
    assert (
        calls.index("deploy-proxy")
        < calls.index("arm")
        < calls.index("repoint-collector")
    ), (
        "live 2026-10-26 the collector was repointed (restarted under the new "
        "boot) before the arm: its first scrape was pass-through and the boundary "
        "carried a zero delta; the ramp after it was a same-generation storm"
    )
    repoint = next(
        c
        for c in ctx.probes[f"dcgm@{ctx.settings.node}"].calls
        if c[0] == "repoint-collector"
    )
    assert repoint[repoint.index("--boot-id") + 1] == PRIOR_BOOT
    assert "--before-repoint" in _arm_call(ctx), (
        "live 2026-10-26 14:12Z (hp, then eks) the probe's one arm guard still "
        "required the repointed endpoint and refused this order: 'the collector "
        "does not read the replay', INCONCLUSIVE at arm"
    )
    steps = _plan(ctx)["steps"]
    assert steps[0] == {
        "reads": 1,
        "ops": [{"op": "add", "field": PCIE, "gpus": [G1], "value": JUMP}],
    }, "the new boot's very first reading is the jump"
    assert steps[1]["ops"] == [] and steps[1]["until_advance"] is True


def _arm_call(ctx: Any) -> tuple[str, ...]:
    [call] = [c for c in ctx.probes[f"dcgm@{ctx.settings.node}"].calls if c[0] == "arm"]
    return call


def test_only_arm_before_repoint_tells_the_probe_the_order(tmp_path: Path) -> None:
    """The flag is threaded from ``arm_before_repoint`` to the probe command;
    the repoint-then-arm variants (``arm_and_confirm``) keep the default guard."""

    before = _boot_ctx(tmp_path, _boot_state())
    after = _reset_ctx(tmp_path, _reset_state())

    assert links.source_boot_change(before)["verdict"] == "PASS"
    assert links.pcie_counter_reset(after)["verdict"] == "PASS"

    assert _arm_call(before) == (
        "arm",
        "--port",
        "9412",
        "--before-repoint",
        "--run-id",
        before.run_id,
    )
    assert _arm_call(after) == ("arm", "--port", "9412", "--run-id", after.run_id)


def test_source_boot_change_passes_when_the_new_boot_starts_a_new_baseline(
    tmp_path: Path,
) -> None:
    ctx = _boot_ctx(tmp_path, _boot_state())

    out = links.source_boot_change(ctx)

    assert out["verdict"] == "PASS", out["errors"]
    assert set(out["stages"]) == {"no_cross_boot_rate", "new_baseline", "hardware"}
    assert out["boot_id"] == PRIOR_BOOT and out["jump"] == JUMP
    assert any("journal --list-boots" in note for note in out["notes"]), out["notes"]
    readings = json.loads((ctx.case_dir / "boot-readings.json").read_text())
    assert readings["first_new_boot_reading"]["value"] == JUMP
    assert readings["latest"]["labels"]["source_boot_id"] == PRIOR_BOOT


def test_source_boot_change_fails_on_a_cross_generation_rate(tmp_path: Path) -> None:
    finding = _finding(JUMP, _at(10, 55), delta=JUMP, rate=18.6)
    inc = gpu_incident("inc-boot", finding["finding_id"])
    ctx = _boot_ctx(
        tmp_path, _boot_state(findings=[finding], active=[finding]), incidents=[inc]
    )

    out = links.source_boot_change(ctx)

    assert out["verdict"] == "FAIL"
    assert out["stages"]["no_cross_boot_rate"] == [
        "a source/boot change formed a cross-generation PCIe rate: "
        f"value={JUMP} delta={JUMP} rate=18.6/min at {_at(10, 55)}; incident inc-boot",
        "a source/boot change left an active finding: "
        f"value={JUMP} delta={JUMP} rate=18.6/min at {_at(10, 55)}",
    ]
    assert out["stages"]["new_baseline"] == []


def test_source_boot_change_fails_when_the_latest_reading_stays_in_the_real_boot(
    tmp_path: Path,
) -> None:
    latest = {
        "gpu": G1,
        "name": PCIE,
        "value": 0,
        "observed_at": _at(10, 0),
        "source": "DCGM_EXPORTER",
        "labels": {"source_boot_id": REAL_BOOT},
    }
    ctx = _boot_ctx(tmp_path, _boot_state(latest=latest))

    out = links.source_boot_change(ctx)

    assert out["verdict"] == "FAIL"
    assert out["stages"]["new_baseline"] == [
        f"the product's latest {PCIE} reading of {G1} is value=0 under boot "
        f"{REAL_BOOT}, not the new boot's first reading ({JUMP} under {PRIOR_BOOT}): "
        "the boot change did not start a new baseline"
    ]


def test_source_boot_change_fails_when_no_reading_of_the_new_boot_was_accepted(
    tmp_path: Path,
) -> None:
    ctx = _boot_ctx(tmp_path, _state([_batch(0, _at(10, 0), "health-summary")]))

    out = links.source_boot_change(ctx)

    assert out["verdict"] == "FAIL"
    assert out["stages"]["new_baseline"] == [
        f"the product accepted no {PCIE} reading of {G1} under the replayed boot "
        f"{PRIOR_BOOT}: the new boot's readings never became a baseline"
    ]


def test_source_boot_change_is_inconclusive_when_the_first_new_boot_reading_is_not_the_jump(
    tmp_path: Path,
) -> None:
    batches = [
        _batch(0, _at(10, 41), "initial-baseline", boot=PRIOR_BOOT),
        _batch(JUMP, _at(10, 56), "counter-increased", boot=PRIOR_BOOT),
    ]
    ctx = _boot_ctx(tmp_path, _state(batches))

    with pytest.raises(VariantInconclusive) as raised:
        links.source_boot_change(ctx)

    assert str(raised.value) == (
        f"the new boot's first accepted {PCIE} reading of {G1} was 0, not the "
        f"replayed jump {JUMP}: the collector scraped the tool before the arm "
        "reached it, so the boundary was crossed at the real value and not exercised"
    )
    # The runner records it INCONCLUSIVE and runs the cleanup stack (close-run).
    assert "dcgm-replay" in ctx.cleanup.pending()


def test_the_registry_text_names_the_boundary_each_variant_drives() -> None:
    reset, boot = SPECS["counter_reset"], SPECS["source_boot_change"]
    assert "under 8/min" in reset.injection and "counter-reset" in reset.expected
    assert "first reading" in boot.injection and "new baseline" in boot.expected
    assert reset.expected_outcome == boot.expected_outcome == "PASS"

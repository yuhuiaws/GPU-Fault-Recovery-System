"""GF-GENERAL-LIVE-009 counter_reset over scripted records (fakes, no node).

A recreated drop interface is a reset tick, never a recovery and never a new
fault: the first burst's edge and recovered zero, the one-process recreate,
the reset tick as a ``_network`` collection error with no reading, the second
burst's delta from the new baseline, and the gap-tick readmission.
"""

from __future__ import annotations

import itertools
import json
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import general_live_network_reset as reset
from scripts.e2e.regional.general_live_fixture import VariantInconclusive
from scripts.e2e.regional.regional_commands import RegionalFixtureError
from tests.regional._general_live_fakes import (
    FakeProbe,
    FakeRegional,
    incident,
    make_context,
    store_script,
    workflow,
)
from tests.regional._general_live_network_fakes import (
    IFA,
    IFB,
    R0,
    RECREATED_AT,
    RESET_ERROR,
    SPECS,
    link_down_sample,
    net_probe,
    node_probe,
    probes,
    recreate_output,
    reset_at_text,
    reset_batch,
    reset_doc,
    reset_oracle,
    survey,
    window_probe,
)


@pytest.fixture
def reset_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    moments = (R0 + timedelta(seconds=60 * step) for step in itertools.count())
    monkeypatch.setattr(reset, "utc_now", lambda: next(moments))


def _run_counter_reset(
    tmp_path: Path,
    document: dict[str, Any],
    *,
    oracle: list[dict[str, Any]] | None = None,
    net_extra: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], Any, FakeProbe]:
    spec = SPECS["counter_reset"]
    regional = FakeRegional(cpu_python=store_script({"activity": document}))
    net_fake = net_probe(**(net_extra or {}))
    ctx = make_context(
        spec,
        tmp_path=tmp_path,
        regional=regional,
        probes=probes(
            node_probe(survey(), oracle or reset_oracle()), net_fake, window_probe()
        ),
    )
    return spec.execute(ctx), ctx, net_fake


@pytest.mark.usefixtures("reset_clock")
def test_counter_reset_reads_the_reset_tick_as_unknown(tmp_path: Path) -> None:
    result, ctx, net_fake = _run_counter_reset(tmp_path, reset_doc())
    assert result["verdict"] == "PASS", result["errors"]
    assert result["stages"]["reset_unknown"] == [] and result["stages"]["rearmed"] == []
    assert (result["oracle"]["ifindex_before"], result["oracle"]["ifindex_after"]) == (
        ["31"],
        ["33"],
    )
    assert (
        result["oracle"]["pre_reset_drops"],
        result["oracle"]["post_reset_drops"],
    ) == (800.0, 300.0)
    # One burst, the one-process recreate, a second burst on the new pair.
    assert [call[0] for call in net_fake.calls] == [
        "create-veth",
        "start-drops",
        "stop-drops",
        "recreate-veth",
        "start-drops",
        "stop-drops",
    ]
    recreate = next(call for call in net_fake.calls if call[0] == "recreate-veth")
    assert recreate[:5] == ("recreate-veth", "--slot", "a", "--peer-slot", "b")
    evidence = json.loads((ctx.case_dir / "reset-evidence.json").read_text())
    assert evidence["errors"] == [RESET_ERROR] and evidence["reset_tick"] == []
    assert [item["value"] for item in evidence["readings"]] == [150.0, 0.0]
    assert (evidence["gap_ms"], evidence["recreated_at"]) == (
        2.4,
        reset_at_text(RECREATED_AT),
    )
    assert evidence["anchor"] == reset_at_text(RECREATED_AT - 15), (
        "one interval before the recreate"
    )
    for name in ("reset-edge", "reset-quiet", "reset-tick", "reset-rearmed"):
        assert (ctx.case_dir / f"{name}-timeline.json").is_file(), name
    spec = SPECS["counter_reset"]
    assert spec.expected_outcome == "PASS" and spec.proof_mode == "native"
    assert "delta unknown" in spec.expected and "clamped" in spec.expected
    assert "recreate-veth" in spec.injection and spec.max_minutes == 25


@pytest.mark.usefixtures("reset_clock")
def test_counter_reset_fails_on_the_old_clamp(tmp_path: Path) -> None:
    # The old product's trace: no reset error ever named, the recreated pair's
    # first delta the first record after the recreate.
    result, _ctx, _net = _run_counter_reset(tmp_path, reset_doc(marked=False))
    assert result["verdict"] == "FAIL", result
    assert any(
        "old clamp" in error and "delta unknown" in error for error in result["errors"]
    ), result["errors"]


@pytest.mark.usefixtures("reset_clock")
def test_counter_reset_fails_on_a_reading_in_the_reset_tick(tmp_path: Path) -> None:
    result, _ctx, _net = _run_counter_reset(
        tmp_path, reset_doc(reset_tick_reading=True)
    )
    assert result["verdict"] == "FAIL", result
    assert any("in the reset tick" in error for error in result["errors"]), result[
        "errors"
    ]


@pytest.mark.usefixtures("reset_clock")
def test_counter_reset_fails_when_the_recreated_pair_never_counts_again(
    tmp_path: Path,
) -> None:
    # The reset error came, but no delta from the new baseline followed the
    # second burst although the oracle counted frames on the new pair.
    result, _ctx, _net = _run_counter_reset(tmp_path, reset_doc(ramp=False))
    assert result["verdict"] == "FAIL", result
    assert any(
        error.startswith("rearmed:") and "no positive network_drops_delta" in error
        for error in result["errors"]
    ), result["errors"]


@pytest.mark.usefixtures("reset_clock")
def test_counter_reset_fails_when_batches_arrive_without_the_interface(
    tmp_path: Path,
) -> None:
    # The host collector delivered after the recreate (its status row moved)
    # but neither named the reset nor read the pair again.
    document = reset_doc(marked=False, ramp=False, status_at=200.0)
    result, _ctx, _net = _run_counter_reset(tmp_path, document)
    assert result["verdict"] == "FAIL", result
    assert any(
        "recorded neither a reset error" in error for error in result["errors"]
    ), result["errors"]


@pytest.mark.usefixtures("reset_clock")
def test_counter_reset_fails_on_a_drops_finding(tmp_path: Path) -> None:
    found = incident("inc-d", "network_drops_delta", device=IFB, workflow="wf-d")
    steps = workflow("wf-d", "inc-d", ["FREEZE_EVIDENCE", "VALIDATE_FABRIC"])
    document = reset_doc(incidents=[found], workflows=[steps])
    result, _ctx, _net = _run_counter_reset(tmp_path, document)
    assert result["verdict"] == "FAIL", result
    assert any(error.startswith("no_finding:") for error in result["errors"]), result[
        "errors"
    ]


@pytest.mark.usefixtures("reset_clock")
def test_counter_reset_is_inconclusive_when_the_burst_never_reached_the_product(
    tmp_path: Path,
) -> None:
    with pytest.raises(VariantInconclusive, match="baseline the recreate must fall"):
        _run_counter_reset(tmp_path, reset_doc(burst=False, marked=False, ramp=False))


@pytest.mark.usefixtures("reset_clock")
def test_counter_reset_is_inconclusive_when_the_counters_never_rose(
    tmp_path: Path,
) -> None:
    with pytest.raises(VariantInconclusive, match="did not move the drop counters"):
        _run_counter_reset(tmp_path, reset_doc(), oracle=reset_oracle(rise=False))


@pytest.mark.usefixtures("reset_clock")
def test_counter_reset_is_inconclusive_without_the_bursts_recovered_edge(
    tmp_path: Path,
) -> None:
    document = reset_doc(burst_zero=False, marked=False, ramp=False)
    with pytest.raises(VariantInconclusive, match="recovered edge"):
        _run_counter_reset(tmp_path, document)


@pytest.mark.usefixtures("reset_clock")
def test_counter_reset_is_inconclusive_on_a_silent_channel_after_the_recreate(
    tmp_path: Path,
) -> None:
    document = reset_doc(marked=False, ramp=False, status_at=20.0)
    with pytest.raises(VariantInconclusive, match="silent channel"):
        _run_counter_reset(tmp_path, document)


@pytest.mark.usefixtures("reset_clock")
def test_counter_reset_refuses_a_recreate_that_kept_the_interface(
    tmp_path: Path,
) -> None:
    with pytest.raises(RegionalFixtureError, match="did not replace the interface"):
        _run_counter_reset(
            tmp_path,
            reset_doc(),
            net_extra={"recreate-veth": recreate_output(new_index="31")},
        )


@pytest.mark.usefixtures("reset_clock")
def test_counter_reset_readmits_the_node_when_a_tick_saw_the_pair_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gap = incident(
        "inc-gap",
        "network_link_down",
        device=IFA,
        state="QUARANTINED",
        workflow="wf-gap",
    )
    steps = workflow(
        "wf-gap",
        "inc-gap",
        ["FREEZE_EVIDENCE", "MARK_UNSCHEDULABLE", "QUARANTINE"],
        status="SUCCEEDED",
    )
    document = reset_doc(marked=False, ramp=False, incidents=[gap], workflows=[steps])
    # The pair is back up: fresh link readings of both ends after the recreate.
    for seconds in (140, 160):
        document["batches"].append(
            reset_batch(
                seconds,
                [link_down_sample(0.0, IFA), link_down_sample(0.0, IFB)],
                reasons=("recovered",),
            )
        )
    restored: list[tuple[list[str], str]] = []

    def fake_restore(ctx: Any, incidents: Any, *, reason: str) -> list[dict[str, Any]]:
        restored.append(([item["incident_id"] for item in incidents], reason))
        return [{"incident_id": "inc-gap"}]

    monkeypatch.setattr(reset, "restore_incidents", fake_restore)
    with pytest.raises(
        VariantInconclusive, match="straddled a collector tick"
    ) as caught:
        _run_counter_reset(tmp_path, document)
    assert "2.4 ms" in str(caught.value) and "not a verdict" in str(caught.value)
    assert restored == [
        (["inc-gap"], "GF-GENERAL-LIVE-009 counter_reset straddled recreate")
    ]


def test_reset_evidence_keeps_only_the_interfaces_own_records() -> None:
    document = reset_doc()
    since = R0 + timedelta(seconds=110)
    evidence = reset.reset_evidence(document, IFA, since)
    assert evidence["errors"] == [RESET_ERROR]
    assert evidence["reset_tick"] == [], "the reset tick carries no reading"
    assert [item["value"] for item in evidence["readings"]] == [150.0, 0.0]
    assert reset.reset_evidence(document, IFB, since)["errors"] == [], (
        "an error naming another interface does not count"
    )
    # Before the anchor nothing counts: the first burst is not the reset's.
    assert reset.reset_evidence(document, IFA, R0 + timedelta(seconds=300)) == {
        "errors": [],
        "reset_tick": [],
        "readings": [],
    }

"""The oracle gate after ``arm`` waits for the collector's own scrapes.

Live 2026-10-25 (release 6a95b90fcd82, both plain-EKS nodes, every replay
variant of GF-GENERAL-LIVE-019): ``arm_and_confirm`` judged the replay
journal about 30 s after ``arm``, when the 15-s collector had made two armed
reads -- both inside the plan's three pass-through baseline reads, which serve
nothing by design -- and every variant ended INCONCLUSIVE "the replay never
served". The gate must wait for the first replayed reading, bounded by the
plan's leading pass-through reads and the collector's real interval, end at
once on an abort, and name what it saw when it gives up.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import general_live_gpu_memory as mem
from scripts.e2e.regional.general_live_fixture import VariantInconclusive
from scripts.e2e.regional.general_live_gpu_replay import (
    DEFAULT_INTERVAL_SECONDS,
    DcgmReplay,
    add,
    arm_and_confirm,
    baseline,
    leading_passthrough_reads,
    plan,
    step,
)
from tests.regional._general_live_fakes import FakeRegional, make_context, make_settings
from tests.regional._general_live_gpu_fakes import dcgm_probe, gpu_cpu_python

SPEC = {spec.variant: spec for spec in mem.VARIANTS}["corrected_ecc"]
GPU = "GPU-11111111-1111-4111-8111-111111111111"
FIELD = "ecc_sbe_volatile_total"
KEY = f"{GPU}/{FIELD}"
NODE = "ip-10-0-43-5.us-west-2.compute.internal"
#: 019's plan: three pass-through baseline reads, then the held increment.
PLAN = plan("memory", baseline(), step(40, add(FIELD, [GPU]), hold=True))


def _ctx(tmp_path: Path, probe: Any):
    regional = FakeRegional(cpu_python=gpu_cpu_python())
    settings = make_settings(SPEC, node=NODE)
    ctx = make_context(
        SPEC, tmp_path=tmp_path, regional=regional, probes={}, settings=settings
    )
    ctx.probes[f"dcgm@{NODE}"] = probe
    return ctx


def _flag(arguments: tuple[str, ...], name: str, default: str) -> str:
    for index, value in enumerate(arguments):
        if value == name and index + 1 < len(arguments):
            return arguments[index + 1]
    return default


def _read(number: int, *, step_index: int, served: dict[str, Any]) -> dict[str, Any]:
    """One armed read as ``tools/gpu_dcgm_field_replay.py`` journals it."""

    return {
        "read": number,
        "armed": True,
        "disarmed": False,
        "step": step_index,
        "ops": [] if step_index == 0 else [{"op": "add", "field": FIELD}],
        "real": {KEY: 0.0},
        "served": served,
        "aborted": None,
        "profile": "memory",
        "proof_scope": "live-replay",
        "physical_fault_injected": False,
        "delivery": "prepared",
    }


ARM = {"event": "ARM", "start_read": 0}
#: What the live journals held when the runner judged them: two baseline reads.
TWO_BASELINE_READS = [
    ARM,
    _read(1, step_index=0, served={}),
    _read(2, step_index=0, served={}),
]
FIRST_REPLAYED_READ = _read(4, step_index=1, served={KEY: 1.0})


def _growing_journal(*stages: list[dict[str, Any]]):
    """An ``audit-read`` whose journal is ``stages[n]`` on its n-th call."""

    calls = {"count": 0}

    def audit_read(arguments: tuple[str, ...]) -> dict[str, Any]:
        offset = int(_flag(arguments, "--offset", "0"))
        source = stages[min(calls["count"], len(stages) - 1)]
        calls["count"] += 1
        chunk = source[offset:]
        return {
            "records": list(chunk),
            "next_offset": offset + len(chunk),
            "total": len(source),
        }

    return audit_read


def _audit_reads(probe: Any) -> int:
    return sum(1 for call in probe.calls if call[0] == "audit-read")


# --------------------------------------------------------------------------- #
# the gate waits for the collector to reach the replayed step
# --------------------------------------------------------------------------- #
def test_arm_and_confirm_waits_for_the_first_replayed_reading(tmp_path: Path) -> None:
    probe = dcgm_probe(
        extra={
            "audit-read": _growing_journal(
                TWO_BASELINE_READS,
                [*TWO_BASELINE_READS, _read(3, step_index=0, served={})],
                [
                    *TWO_BASELINE_READS,
                    _read(3, step_index=0, served={}),
                    FIRST_REPLAYED_READ,
                ],
            )
        }
    )
    ctx = _ctx(tmp_path, probe)
    replay = DcgmReplay(ctx, PLAN, 600)

    arm_and_confirm(replay, KEY)  # live 2026-10-25: raised "never served" here

    assert _audit_reads(probe) >= 3, "the journal was polled until the read appeared"
    assert replay.records[-1]["served"] == {KEY: 1.0}
    timeline = json.loads((ctx.case_dir / "replay-served-timeline.json").read_text())
    accepted = [entry["accepted"] for entry in timeline["entries"]]
    assert accepted[:-1] == [False] * (len(accepted) - 1) and accepted[-1] is True


def test_the_gate_names_the_reads_it_saw_when_it_gives_up(tmp_path: Path) -> None:
    probe = dcgm_probe(extra={"audit-read": _growing_journal(TWO_BASELINE_READS)})
    replay = DcgmReplay(_ctx(tmp_path, probe), PLAN, 600)

    with pytest.raises(VariantInconclusive) as raised:
        arm_and_confirm(replay, KEY)

    assert str(raised.value) == (
        f"the replay never served {KEY} within 180 s: "
        "2 armed reads, the last at plan step 0"
    )
    assert _audit_reads(probe) > 2, "the gate polled across the deadline, not once"


def test_a_journal_without_any_read_says_so(tmp_path: Path) -> None:
    probe = dcgm_probe(extra={"audit-read": _growing_journal([ARM])})
    replay = DcgmReplay(_ctx(tmp_path, probe), PLAN, 600)

    with pytest.raises(VariantInconclusive, match="within 180 s: 0 armed reads$"):
        arm_and_confirm(replay, KEY)


def test_an_abort_ends_the_wait_at_once(tmp_path: Path) -> None:
    aborted = {
        **_read(3, step_index=0, served={}),
        "aborted": f"real upstream change of {KEY}: 0.0 -> 1.0; a real reading is never masked",
    }
    probe = dcgm_probe(
        extra={"audit-read": _growing_journal([*TWO_BASELINE_READS, aborted])}
    )
    replay = DcgmReplay(_ctx(tmp_path, probe), PLAN, 600)

    with pytest.raises(VariantInconclusive, match="^the replay aborted: real upstream"):
        arm_and_confirm(replay, KEY)

    assert _audit_reads(probe) == 1, "an abort is final; nothing to wait for"


def test_a_gate_without_keys_does_not_wait(tmp_path: Path) -> None:
    probe = dcgm_probe(extra={"audit-read": _growing_journal([ARM])})
    replay = DcgmReplay(_ctx(tmp_path, probe), PLAN, 600)

    arm_and_confirm(replay)

    assert _audit_reads(probe) == 0


# --------------------------------------------------------------------------- #
# the deadline follows the plan and the collector's real interval
# --------------------------------------------------------------------------- #
def test_serve_timeout_follows_the_plan_and_the_collector_interval(
    tmp_path: Path,
) -> None:
    ctx = _ctx(tmp_path, dcgm_probe())
    assert leading_passthrough_reads(PLAN) == 3
    assert DEFAULT_INTERVAL_SECONDS == 15, "the product's default scrape interval"
    # 3 baseline reads + the first replayed one + 2 slack scrapes, then 90 s grace.
    assert DcgmReplay(ctx, PLAN, 600).serve_timeout_seconds() == 6 * 15 + 90
    assert (
        DcgmReplay(ctx, PLAN, 600, interval=30).serve_timeout_seconds() == 6 * 30 + 90
    )
    immediate = plan("memory", step(40, add(FIELD, [GPU]), hold=True))
    assert leading_passthrough_reads(immediate) == 0
    assert DcgmReplay(ctx, immediate, 600).serve_timeout_seconds() == 3 * 15 + 90


def test_deploy_learns_the_collector_interval_from_the_repoint(tmp_path: Path) -> None:
    def repoint(environment: dict[str, str]):
        return {
            "replay_endpoint": "http://127.0.0.1:9412/metrics",
            "process_environment": environment,
        }

    probe = dcgm_probe(
        extra={
            "repoint-collector": repoint(
                {
                    "GPU_FAULT_DCGM_METRICS_URL": "http://127.0.0.1:9412/metrics",
                    "GPU_FAULT_METRICS_INTERVAL_SECONDS": "30",
                }
            )
        }
    )
    replay = DcgmReplay(_ctx(tmp_path, probe), PLAN, 600)
    replay.deploy()
    assert replay.interval == 30

    for malformed in (
        {},
        {"GPU_FAULT_METRICS_INTERVAL_SECONDS": "soon"},
        {"GPU_FAULT_METRICS_INTERVAL_SECONDS": "0"},
    ):
        probe = dcgm_probe(extra={"repoint-collector": repoint(malformed)})
        replay = DcgmReplay(_ctx(tmp_path, probe), PLAN, 600)
        replay.deploy()
        assert replay.interval == DEFAULT_INTERVAL_SECONDS, malformed

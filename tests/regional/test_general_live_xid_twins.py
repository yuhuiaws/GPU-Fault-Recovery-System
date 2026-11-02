"""GF-GENERAL-LIVE-024 ``xid_twin_*``: registry, product-generation gate,
reviewed inputs, report placement and the oracles' premises.

The end-to-end runs over the product's own records are in
``test_general_live_xid_twins_product.py``; here each piece is pinned on its
own: the variants' keys, flags and tokens; the generation probe run as the
control plane runs it against this tree (batch 237 present), against the same
tree with ``xid_cross_source`` taken away (the product before it) and against a
changed code shape (undecided) -- the last two INCONCLUSIVE before anything is
deployed; the catalog and Agent prerequisites; the kernel lines and the field
plan; the kernel-first placement (the change on the first scrape 15-30 s after
the line, from the journal's own clock) and its premise failure; the
negative-control and twin-window premises; a kmsg close the product's quiesce
defers.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from gpu_fault.policy import DynamicRecoveryAction, parse_xid154_action
from gpu_fault.policy import xid_cross_source as product_twins
from scripts.e2e.regional import general_live_registry as reg
from scripts.e2e.regional import general_live_xid_twin_delivery as placement
from scripts.e2e.regional import general_live_xid_twin_records as twin_records
from scripts.e2e.regional import general_live_xid_twins as twins
from scripts.e2e.regional.general_live_fixture import VariantInconclusive, VariantNotRun
from scripts.e2e.regional.general_live_gpu_replay import GpuSurvey
from scripts.e2e.regional.general_live_kernel_replay import KernelLogReplay
from scripts.e2e.regional.general_live_kernel_xid import CatalogEntry
from scripts.e2e.regional.host_probe_fixture import HostProbeMissingResponseError
from tests.regional._general_live_fakes import (
    FakeProbe,
    FakeRegional,
    make_context,
    make_settings,
)
from tests.regional._general_live_gpu_fakes import survey
from tools.gpu_dcgm_field_replay import validate_plan

CASE = "GF-GENERAL-LIVE-024"
NODE = "ip-10-0-43-5.us-west-2.compute.internal"
G0 = "GPU-00000000-0000-4000-8000-000000000000"
G1 = "GPU-11111111-1111-4111-8111-111111111111"
FLAGS = {
    "xid_twin_kernel_first": "S",
    "xid_twin_gauge_first": "S",
    "xid_twin_reset_gpu": "X",
    "xid_twin_unbound_job": "X",
    "xid_twin_drain_p2p": "X",
}
RUN_ID = "gl024-xid-twin-kernel-first-a1-deadbeef"


def _spec(variant: str) -> Any:
    return next(item for item in twins.VARIANTS if item.variant == variant)


def _ctx(
    tmp_path: Path, variant: str, *, cpu_python: Any = None, probes: Any = None
) -> Any:
    spec = _spec(variant)
    regional = FakeRegional(cpu_python=cpu_python)
    ctx = make_context(
        spec,
        tmp_path=tmp_path,
        regional=regional,
        probes=probes or {},
        settings=make_settings(spec, node=NODE),
    )
    return ctx


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
def test_the_twin_variants_are_registered_with_their_flags_and_tokens() -> None:
    registry = reg.load_registry()
    assert twins.__name__ in reg.CATEGORY_MODULES, "the module is a category module"
    row = reg.coverage_rows()[CASE]
    for variant, flags in FLAGS.items():
        spec = registry[f"{CASE}.{variant}"]
        assert variant in row, f"{variant} is not in the 024 coverage row"
        assert (spec.flags, spec.lanes, spec.proof_mode) == (
            flags,
            ("eks", "hp"),
            reg.LIVE_REPLAY,
        ), spec.key
        assert spec.implemented and not spec.deferred, spec.key
        assert {twins.KERNEL_SOURCE_REPLAY, twins.DCGM_XID_TWIN} <= set(
            spec.approvals
        ), spec.key
        destructive = {"GPU_RESET", "GPU_DRAIN_P2P"} & set(spec.approvals)
        assert bool(destructive) == (flags == "X"), (spec.key, spec.approvals)
        assert spec.risk == ("H" if flags == "X" else "M"), spec.key
    assert registry[f"{CASE}.xid_twin_drain_p2p"].approvals[0] == "GPU_DRAIN_P2P"
    for token in ("DCGM_XID_TWIN", "GPU_DRAIN_P2P"):
        assert token in reg.APPROVALS, token
    assert "024 xid_twin_reset_gpu" in reg.APPROVALS["GPU_RESET"], (
        "the reset token names the variants it now covers"
    )
    assert twins.CASES.keys() == FLAGS.keys()


def test_the_window_and_the_decoded_xids_are_the_products() -> None:
    assert twin_records.TWIN_WINDOW_SECONDS == (
        product_twins.XID_REPORT_TWIN_WINDOW.total_seconds()
    )
    assert twin_records.KERNEL_SOURCE == product_twins.KERNEL_XID_SOURCE
    assert twin_records.GAUGE_SOURCE == product_twins.DCGM_XID_GAUGE_SOURCE
    assert placement.PRODUCTION_LAG_SECONDS == (15, 30)
    assert 154 in twin_records.KERNEL_DECODED_XIDS
    assert {48, 92}.isdisjoint(twin_records.KERNEL_DECODED_XIDS), (
        "XID 48 and 92 are catalog-named, not kernel-decoded"
    )


# --------------------------------------------------------------------------- #
# The product generation, run as the control plane runs it
# --------------------------------------------------------------------------- #
#: Takes ``gpu_fault.policy.xid_cross_source`` away: the product before
#: batch 237 (the module did not exist; the policy package imports it now).
WITHOUT_TWINS = """
import importlib.abc
import sys


class _Before237(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name == "gpu_fault.policy.xid_cross_source":
            raise ImportError("No module named 'gpu_fault.policy.xid_cross_source'")
        return None


sys.meta_path.insert(0, _Before237())
"""


TREE = Path(__file__).resolve().parents[2]


def _answer(xid: int, tail: str, *, prefix: str = "", script: str = "") -> Any:
    out = subprocess.run(
        [
            sys.executable,
            "-c",
            prefix + (script or twin_records.XID_TWIN_GENERATION_PROBE),
            str(xid),
            tail,
        ],
        capture_output=True,
        text=True,
        timeout=240,
        check=True,
        cwd=TREE,
        # The child imports this tree's gpu_fault, never the interpreter's installed one (the release gate runs the
        # suite with the deployer venv, whose installed product is the previous release).
        env={
            **os.environ,
            "PYTHONPATH": os.pathsep.join(
                [
                    str(TREE / "src"),
                    str(TREE),
                    *filter(None, [os.environ.get("PYTHONPATH")]),
                ]
            ),
        },
    )
    return json.loads(out.stdout.strip().splitlines()[-1])


@pytest.mark.parametrize(
    "variant", ["xid_twin_kernel_first", "xid_twin_reset_gpu", "xid_twin_drain_p2p"]
)
def test_this_tree_answers_the_twin_rule(variant: str) -> None:
    case = twins.CASES[variant]
    answer = _answer(case.xid, twins.kernel_tail(case, RUN_ID))
    generation = twin_records.TwinGeneration(case.xid, answer)

    assert generation.twin_rule, generation.shortfalls()
    primary = answer["primary"]
    assert (primary["disposition"], primary["official_action"]) == (
        case.expectation.disposition,
        case.expectation.official_action,
    ), primary
    assert primary["action"] in case.expectation.actions, primary
    assert answer["followed"]["correlated_event_id"] == "generation-probe-kernel"
    if case.xid == 154:
        assert answer["alone"]["disposition"] == "PENDING_CORRELATION", answer
    assert "batch 237's twin rule" in generation.describe()


def test_a_product_without_xid_cross_source_predates_the_twins() -> None:
    answer = _answer(92, "pid=0, name=gfrs-live, probe", prefix=WITHOUT_TWINS)
    generation = twin_records.TwinGeneration(92, answer)

    assert answer["present"] is False and "xid_cross_source" in answer["error"]
    assert generation.predates_twins and not generation.twin_rule, answer
    assert "predates batch 237" in generation.describe(), generation.describe()


def test_a_changed_code_shape_is_undecided_never_a_traceback() -> None:
    broken = twin_records.XID_TWIN_GENERATION_PROBE.replace(
        "report_twin_incident", "report_twin_incident_renamed"
    )
    answer = _answer(48, "pid=0, name=gfrs-live, probe", script=broken)
    generation = twin_records.TwinGeneration(48, answer)

    assert answer["present"] is True, answer
    assert not generation.twin_rule and not generation.predates_twins
    assert "no incident link of the second report" in generation.shortfalls()


def test_the_gate_refuses_an_earlier_product_before_touching_the_node(
    tmp_path: Path,
) -> None:
    calls: list[str] = []

    def cpu_python(script: str, *arguments: str) -> dict[str, Any]:
        calls.append(script)
        return {
            "xid": 48,
            "present": False,
            "error": "ImportError: no xid_cross_source",
        }

    probe = FakeProbe({})
    ctx = _ctx(
        tmp_path,
        "xid_twin_reset_gpu",
        cpu_python=cpu_python,
        probes={f"dcgm@{NODE}": probe, f"kernel-replay-kmsg@{NODE}": probe},
    )
    with pytest.raises(VariantInconclusive) as refused:
        twins.run_twin(ctx, twins.CASES["xid_twin_reset_gpu"])

    text = str(refused.value)
    assert text.startswith(twin_records.PRE_TWIN_GENERATION), text
    assert "a second RESET_GPU" in text and "Nothing was deployed" in text, text
    assert calls == [twin_records.XID_TWIN_GENERATION_PROBE], "only the code probe ran"
    assert probe.calls == [], "no host probe was called"
    record = json.loads((ctx.case_dir / "product-generation.json").read_text())
    assert record["predates_twins"] is True and record["twin_rule"] is False, record


def test_the_gate_refuses_an_engine_that_decides_the_xid_another_way(
    tmp_path: Path,
) -> None:
    answer = _answer(48, "pid=0, name=gfrs-live, probe")
    answer["primary"] = {**answer["primary"], "disposition": "SITE_SAFETY"}
    ctx = _ctx(
        tmp_path, "xid_twin_reset_gpu", cpu_python=lambda _script, *_args: dict(answer)
    )
    with pytest.raises(VariantInconclusive, match="the deployed engine decides XID 48"):
        twin_records.require_twin_generation(
            ctx, twins.RESET_EXPECTATION, "tail", consequence="x"
        )


# --------------------------------------------------------------------------- #
# Before anything is touched: catalog, Agent, GPU
# --------------------------------------------------------------------------- #
def test_the_pinned_catalog_rows_are_the_ones_the_variants_rely_on(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for case in twins.CASES.values():
        twins.require_catalog(case)
    drifted = CatalogEntry(
        xid=48,
        mnemonic="ROBUST_CHANNEL_GPU_ECC_DBE",
        immediate_action="RESTART_BM",
        investigatory_action=None,
        products=("H100",),
        catalog_version="999",
    )
    monkeypatch.setattr(twins, "catalog_entry", lambda _xid: drifted)
    with pytest.raises(VariantNotRun) as refused:
        twins.require_catalog(twins.CASES["xid_twin_reset_gpu"])
    assert refused.value.reason == "FIXTURE_NOT_READY", refused.value
    assert "RESTART_BM in catalog 999" in refused.value.detail, refused.value.detail


def _agent_state(**agent: Any) -> Any:
    def cpu_python(_script: str, *_arguments: str) -> dict[str, Any]:
        return {
            "latest": [],
            "agent": {"allowed_operations": agent.get("ops", [])},
            "capabilities": agent.get("capabilities", {}),
        }

    return cpu_python


@pytest.mark.parametrize(
    ("ops", "capabilities", "refusal"),
    [
        (["QUIESCE_GPU_SERVICES"], {}, "does not allow ['RESET_GPU']"),
        (
            ["QUIESCE_GPU_SERVICES", "RESET_GPU"],
            {"gpuReset": {"mode": "OBSERVE"}},
            "gpuReset OBSERVE",
        ),
        (["QUIESCE_GPU_SERVICES", "RESET_GPU"], {"gpuReset": {"mode": "OWN"}}, None),
    ],
)
def test_the_chain_needs_the_agents_operations_and_an_owned_gpu_reset(
    tmp_path: Path, ops: list[str], capabilities: dict[str, Any], refusal: str | None
) -> None:
    ctx = _ctx(
        tmp_path,
        "xid_twin_reset_gpu",
        cpu_python=_agent_state(ops=ops, capabilities=capabilities),
    )
    since = datetime(2026, 11, 1, tzinfo=timezone.utc)
    if refusal is None:
        twins.require_agent(ctx, twins.CASES["xid_twin_reset_gpu"], since)
        return
    with pytest.raises(VariantNotRun) as refused:
        twins.require_agent(ctx, twins.CASES["xid_twin_reset_gpu"], since)
    assert refused.value.reason == "CAPABILITY_ABSENT", refused.value
    assert refusal in refused.value.detail, refused.value.detail


def test_the_safe_twin_needs_no_agent_operation(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path, "xid_twin_kernel_first", cpu_python=_agent_state())
    twins.require_agent(
        ctx, twins.CASES["xid_twin_kernel_first"], datetime.now(timezone.utc)
    )


def test_the_gpu_is_the_highest_one_without_a_real_xid_outside_the_job() -> None:
    value = survey(gpus=(G0, G1), fields=("pcie_replay_total", "xid_last_error"))
    case = twins.CASES["xid_twin_kernel_first"]
    assert twins.choose_gpu(GpuSurvey(value), case) == (G1, "0000:02:00")
    value["exporter"]["gpus"][G1]["values"]["xid_last_error"] = 79.0
    assert twins.choose_gpu(GpuSurvey(value), case) == (G0, "0000:01:00"), (
        "a real last XID would be masked by the plan's 0"
    )
    with pytest.raises(VariantNotRun) as refused:
        twins.choose_gpu(GpuSurvey(value), case, avoid={G0})
    assert "real last XID 79" in refused.value.detail, refused.value.detail


# --------------------------------------------------------------------------- #
# The reviewed inputs
# --------------------------------------------------------------------------- #
def test_the_kernel_lines_are_the_reviewed_shape_and_name_the_run() -> None:
    for variant in ("xid_twin_kernel_first", "xid_twin_reset_gpu"):
        case = twins.CASES[variant]
        line = twins.kernel_line(case, "0000:b9:00", RUN_ID)
        assert line.startswith(f"NVRM: Xid (PCI:0000:b9:00): {case.xid}, pid=0, "), line
        assert RUN_ID in line and "reviewed kernel-source replay" in line, line
    drain = twins.CASES["xid_twin_drain_p2p"]
    line = twins.kernel_line(drain, "0000:b9:00", RUN_ID)
    assert parse_xid154_action(line) is DynamicRecoveryAction.DRAIN_P2P, line
    first, repeat = twins.kernel_records(
        twins.CASES["xid_twin_gauge_first"], "0000:b9:00", RUN_ID
    )
    assert first["message"] != repeat["message"], "the repeat is its own record"
    assert {first["emissions"], repeat["emissions"]} == {placement.KMSG_EMISSIONS}
    assert len(twins.kernel_records(drain, "0000:b9:00", RUN_ID)) == 1


@pytest.mark.parametrize("xid", [92, 48, 154])
def test_the_field_plan_baselines_at_zero_and_holds_the_xid(xid: int) -> None:
    document = placement.twin_plan(G1, xid, interval=15, hold_seconds=300)

    assert validate_plan(document)["profile"] == placement.TWIN_PROFILE
    assert document["fields"] == {"xid_last_error": {"absent_when_healthy": True}}
    zero, held = document["steps"]
    assert zero["ops"] == [
        {"op": "set", "field": "xid_last_error", "gpus": [G1], "value": 0}
    ]
    assert held["ops"][0]["value"] == xid and held["until_advance"] is True
    assert zero["until_advance"] is True and zero["reads"] >= 60, zero


# --------------------------------------------------------------------------- #
# Placing the reports
# --------------------------------------------------------------------------- #
class ScriptedReplay:
    """The journal of a collector scraping every ``interval`` s from
    ``first_scrape``; the plan moves to the XID one scrape after ``advance``."""

    def __init__(
        self, ctx: Any, *, interval: int, first_scrape: float, xid: int
    ) -> None:
        self.ctx = ctx
        self.interval = interval
        self.first_scrape = first_scrape
        self.xid = xid
        self.records: list[dict[str, Any]] = []
        self.advanced_at: float | None = None

    def _scrapes_until(self, now: float) -> None:
        index = len(self.records)
        while True:
            at = self.first_scrape + index * self.interval
            if at > now:
                return
            value = (
                float(self.xid)
                if self.advanced_at is not None and at > self.advanced_at
                else 0.0
            )
            self.records.append(
                {
                    "read": index + 1,
                    "armed": True,
                    "at": at,
                    "served": {f"{G1}/xid_last_error": value},
                    "aborted": None,
                }
            )
            index += 1

    def audit(self) -> list[dict[str, Any]]:
        self._scrapes_until(self.ctx.clock().timestamp())
        return self.records

    def armed_reads(self) -> list[dict[str, Any]]:
        return list(self.records)

    def advance(self) -> dict[str, Any]:
        self.advanced_at = self.ctx.clock().timestamp()
        return {"advanced": True}


class ScriptedKernel:
    def __init__(self, ctx: Any) -> None:
        self.ctx = ctx
        self.armed: list[tuple[int, float]] = []

    def arm(self, *, index: int | None = None) -> dict[str, Any]:
        at = self.ctx.clock().timestamp()
        self.armed.append((int(index or 0), at))
        return {"armed": True, "written_at": at}


def _clocked(tmp_path: Path, variant: str) -> Any:
    ctx = _ctx(tmp_path, variant)
    now = {"value": datetime(2026, 11, 1, 12, tzinfo=timezone.utc)}
    ctx.clock = lambda: now["value"]

    def sleep(seconds: float) -> None:
        now["value"] += timedelta(seconds=seconds)

    ctx.sleep = sleep
    return ctx, now


@pytest.mark.parametrize("phase", [0.5, 3.0, 7.5, 14.5])
def test_kernel_first_lands_the_change_15_to_30_s_after_the_line(
    tmp_path: Path, phase: float
) -> None:
    ctx, now = _clocked(tmp_path, "xid_twin_kernel_first")
    start = now["value"].timestamp()
    replay = ScriptedReplay(ctx, interval=15, first_scrape=start + phase, xid=92)
    kernel = ScriptedKernel(ctx)

    delivery = placement.deliver_kernel_first(ctx, replay, kernel, gpu=G1, xid=92)  # type: ignore[arg-type]

    lag = delivery.gauge_served_at - delivery.kernel_written_at  # type: ignore[operator]
    assert 15 <= lag <= 30 and delivery.problems == [], (lag, delivery.problems)
    assert kernel.armed == [(0, start)], kernel.armed


def test_kernel_first_names_a_change_outside_the_production_lag(tmp_path: Path) -> None:
    """A 30 s collector cannot always place the change inside 15-30 s."""

    ctx, now = _clocked(tmp_path, "xid_twin_kernel_first")
    start = now["value"].timestamp()
    replay = ScriptedReplay(ctx, interval=30, first_scrape=start + 14.0, xid=92)

    delivery = placement.deliver_kernel_first(
        ctx,
        replay,  # type: ignore[arg-type]
        ScriptedKernel(ctx),  # type: ignore[arg-type]
        gpu=G1,
        xid=92,
    )

    assert delivery.problems, delivery.record()
    assert "outside the 15-30 s production lag" in delivery.problems[0]


def test_the_repeat_lands_past_the_twin_window(tmp_path: Path) -> None:
    ctx, now = _clocked(tmp_path, "xid_twin_gauge_first")
    delivery = placement.Delivery(
        order="gauge-first", gauge_served_at=now["value"].timestamp()
    )
    kernel = ScriptedKernel(ctx)

    placement.deliver_repeat(ctx, kernel, delivery)  # type: ignore[arg-type]

    ((index, at),) = kernel.armed
    assert index == 1, "the repeat is the second reviewed record"
    assert at - delivery.gauge_served_at >= (  # type: ignore[operator]
        twin_records.TWIN_WINDOW_SECONDS + placement.REPEAT_MARGIN_SECONDS
    )


# --------------------------------------------------------------------------- #
# Premises: what makes a run INCONCLUSIVE rather than graded
# --------------------------------------------------------------------------- #
def _event(
    event_id: str, source: str, seconds: float, **decision: Any
) -> dict[str, Any]:
    at = datetime(2026, 11, 1, 12, tzinfo=timezone.utc) + timedelta(seconds=seconds)
    return {
        "event_id": event_id,
        "event_source": source,
        "gpu_uuid": G1,
        "observed_at": at.isoformat(),
        "linked_incident_id": decision.get("incident_id"),
        "decision": {
            "disposition": "MONITOR_ONLY",
            "official_action": "IGNORE",
            "action": "NO_ACTION",
            **decision,
        },
    }


def test_a_missing_or_late_report_is_a_premise_not_a_verdict() -> None:
    kernel = _event("k", "KERNEL_LOG", 0, incident_id="inc-k")
    late = _event("g", "DCGM_EXPORTER", 125, incident_id="inc-k", duplicate=True)
    pair, problems = twin_records.twin_pair(
        {"events": [kernel]}, gpu=G1, first_source="KERNEL_LOG", xid=92
    )
    assert pair is None and "no field-230 change" in problems[0], problems
    pair, problems = twin_records.twin_pair(
        {"events": [kernel, late]}, gpu=G1, first_source="KERNEL_LOG", xid=92
    )
    assert pair is None and "outside the 120 s twin window" in problems[0], problems
    early = _event("k", "KERNEL_LOG", -5, incident_id="inc-k")
    gauge = _event("g", "DCGM_EXPORTER", 0, incident_id="inc-g")
    pair, problems = twin_records.twin_pair(
        {"events": [early, gauge]}, gpu=G1, first_source="DCGM_EXPORTER", xid=92
    )
    assert pair is None and "before the change" in problems[0], problems


def test_the_joiner_of_a_kernel_decoded_xid_is_always_the_change() -> None:
    gauge = _event("g", "DCGM_EXPORTER", 0, incident_id="inc-k")
    kernel = _event("k", "KERNEL_LOG", 30, incident_id="inc-k")
    pair, problems = twin_records.twin_pair(
        {"events": [gauge, kernel]}, gpu=G1, first_source="DCGM_EXPORTER", xid=154
    )
    assert problems == [] and pair is not None
    assert (pair.joiner["event_id"], pair.primary["event_id"]) == ("g", "k")
    pair, _ = twin_records.twin_pair(
        {"events": [gauge, kernel]}, gpu=G1, first_source="DCGM_EXPORTER", xid=92
    )
    assert pair is not None and pair.joiner["event_id"] == "k", (
        "otherwise the later report joins the first"
    )


def test_the_repeat_inside_the_window_is_a_premise_and_outside_its_own() -> None:
    case = twins.CASES["xid_twin_kernel_first"]
    kernel = _event("k", "KERNEL_LOG", 0, incident_id="inc-k")
    gauge = _event(
        "g",
        "DCGM_EXPORTER",
        20,
        incident_id="inc-k",
        duplicate=True,
        correlated_event_id="k",
    )
    delivery = placement.Delivery(order="kernel-first", repeat_written_at=1.0)
    inside = _event("r", "KERNEL_LOG", 100, incident_id="inc-r")
    errors, premise, _ = twins.grade_repeat(
        {"events": [kernel, gauge, inside], "incidents": []},
        case,
        gpu=G1,
        delivery=delivery,
        incident_id="inc-k",
    )
    assert errors == [] and "not outside the 120 s twin window" in premise[0], premise
    joined = _event(
        "r", "KERNEL_LOG", 175, incident_id="inc-k", correlated_event_id="g"
    )
    errors, premise, repeat = twins.grade_repeat(
        {
            "events": [kernel, gauge, joined],
            "incidents": [{"incident_id": "inc-k", "state": "RECOVERED"}],
        },
        case,
        gpu=G1,
        delivery=delivery,
        incident_id="inc-k",
    )
    assert premise == [] and repeat == "r", premise
    assert any("names g" in item for item in errors), errors
    assert any("not its own inc-r" in item for item in errors), errors


# --------------------------------------------------------------------------- #
# The kmsg close the product's quiesce defers
# --------------------------------------------------------------------------- #
def test_a_kmsg_close_the_quiesce_refuses_is_retried_when_the_node_is_back(
    tmp_path: Path,
) -> None:
    attempts = {"close": 0}

    def close(_arguments: tuple[str, ...]) -> dict[str, Any]:
        attempts["close"] += 1
        if attempts["close"] == 1:
            raise HostProbeMissingResponseError(
                "host probe response is missing (exit 1)"
            )
        return {"clean": True}

    probe = FakeProbe(
        {"open-window": {"after": {"ActiveState": "active"}}, "close-window": close}
    )
    ctx = _ctx(tmp_path, "xid_twin_reset_gpu")
    ctx.regional.node_snapshot_value = {
        **ctx.regional.node_snapshot_value,
        "ready_since": "2026-10-23T00:00:00Z",
    }
    replay = KernelLogReplay(
        ctx.regional,
        node=NODE,
        image="img@sha256:" + "0" * 64,
        case_id=ctx.spec.key,
        run_id=ctx.run_id,
        case_dir=ctx.case_dir,
        mode="kmsg",
        host=probe,
    )
    replay.open(approved=True)
    receipt = placement.close_kernel_window(ctx, replay, "boot-1")

    assert attempts["close"] == 2, "closed again once the node answered"
    assert receipt["close"]["clean"] is True, receipt


# --------------------------------------------------------------------------- #
# The kernel line through a quiesce that took the node's exec channel
# --------------------------------------------------------------------------- #
STOPPED_KUBELET = (
    "host probe response is missing (exit 1): no output came back over kubectl "
    "exec into gpu-fault-host-probe-test; an unclassified kubectl error "
    "(withheld); node Ready=True (the node is Ready, so the exec channel itself "
    "failed)"
)


class QuiescedKernel:
    """The kmsg probe of a node whose kubelet is away until ``back`` (epoch)."""

    def __init__(self, ctx: Any, *, back: float, refuse: bool = False) -> None:
        self.ctx = ctx
        self.back = back
        self.refuse = refuse
        self.calls: list[tuple[float, int]] = []

    def arm(self, *, index: int | None = None, timeout: int = 120) -> dict[str, Any]:
        at = self.ctx.clock().timestamp()
        self.calls.append((at, timeout))
        if self.refuse:
            raise placement.HostProbeError(
                "host probe failed (exit 1); output withheld"
            )
        if at < self.back:
            raise HostProbeMissingResponseError(STOPPED_KUBELET)
        return {"armed": True, "written_at": at, "indexes": [int(index or 0)]}


def _quiesced(
    tmp_path: Path, *, lines: list[dict[str, Any]] | None = None
) -> tuple[Any, dict[str, Any], datetime]:
    records = {"events": list(lines or [])}
    ctx, now = _clocked(tmp_path, "xid_twin_reset_gpu")
    ctx.regional.cpu_python_responder = lambda script, *arguments: records
    return ctx, records, now["value"]


def test_a_line_the_quiesce_refused_is_written_once_the_node_answers(
    tmp_path: Path,
) -> None:
    ctx, _records, start = _quiesced(tmp_path)
    kernel = QuiescedKernel(ctx, back=start.timestamp() + 70)

    receipt = placement.arm_kernel_line(
        ctx,
        kernel,  # type: ignore[arg-type]
        since=start,
        gpu=G1,
        xid=48,
        deadline=start.timestamp() + 90,
    )

    assert receipt["written_from"] == "probe-retry", receipt
    assert (
        70
        <= receipt["written_at"] - start.timestamp()
        <= 70 + (placement.ARM_RETRY_SECONDS)
    ), receipt
    assert kernel.calls[0][1] == 120 and {
        timeout for _at, timeout in kernel.calls[1:]
    } == {placement.ARM_RETRY_TIMEOUT_SECONDS}, kernel.calls
    retry = json.loads((ctx.case_dir / "twin-arm-retry.json").read_text())
    assert retry["first_error"].startswith("HostProbeMissingResponseError"), retry
    assert retry["outcome"] == "probe-retry" and retry["attempts"][-1]["arm"] == "ok"


def test_a_line_a_lost_call_wrote_is_taken_from_the_products_record(
    tmp_path: Path,
) -> None:
    """The exec the stop cut had written the line: the product holds it, and it
    is not written again."""

    ctx, records, start = _quiesced(tmp_path)
    kernel = QuiescedKernel(ctx, back=start.timestamp() + 70)
    line = _event("k", "KERNEL_LOG", 1, incident_id="inc-g", correlated_event_id="g")
    line["gpu_uuid"] = G1
    records["events"].append(line)

    receipt = placement.arm_kernel_line(
        ctx,
        kernel,  # type: ignore[arg-type]
        since=start,
        gpu=G1,
        xid=48,
        deadline=start.timestamp() + 90,
    )

    assert receipt["written_from"] == "product record", receipt
    assert receipt["written_at"] == start.timestamp() + 1, receipt
    assert len(kernel.calls) == 1, "no second write of a line the product holds"


def test_a_node_that_stays_away_past_the_window_is_inconclusive(tmp_path: Path) -> None:
    ctx, _records, start = _quiesced(tmp_path)
    kernel = QuiescedKernel(ctx, back=start.timestamp() + 200)

    with pytest.raises(VariantInconclusive, match="inside the twin window"):
        placement.arm_kernel_line(
            ctx,
            kernel,  # type: ignore[arg-type]
            since=start,
            gpu=G1,
            xid=48,
            deadline=start.timestamp() + 90,
        )
    assert max(at for at, _timeout in kernel.calls) < start.timestamp() + 90, (
        "nothing is written past the window"
    )


def test_a_refused_line_is_not_retried(tmp_path: Path) -> None:
    """A probe that ran and refused is the fixture's failure, not the quiesce."""

    ctx, _records, start = _quiesced(tmp_path)
    kernel = QuiescedKernel(ctx, back=0.0, refuse=True)

    with pytest.raises(placement.HostProbeError, match="output withheld"):
        placement.arm_kernel_line(
            ctx,
            kernel,  # type: ignore[arg-type]
            since=start,
            gpu=G1,
            xid=48,
            deadline=start.timestamp() + 90,
        )
    assert len(kernel.calls) == 1, kernel.calls


def test_no_line_is_tried_while_the_node_reads_not_ready(tmp_path: Path) -> None:
    ctx, _records, start = _quiesced(tmp_path)
    ctx.regional.node_snapshot_value = {
        **ctx.regional.node_snapshot_value,
        "ready": "Unknown",
    }
    kernel = QuiescedKernel(ctx, back=start.timestamp() + 1)

    with pytest.raises(VariantInconclusive):
        placement.arm_kernel_line(
            ctx,
            kernel,  # type: ignore[arg-type]
            since=start,
            gpu=G1,
            xid=48,
            deadline=start.timestamp() + 30,
        )
    assert len(kernel.calls) == 1, "only the first call; none while kubelet is away"


def test_a_journal_the_node_cannot_read_leaves_the_moment_to_the_products_record(
    tmp_path: Path,
) -> None:
    """Plain EKS reads the journal first; when that read cannot reach the node,
    the product's record of the change gives the served moment."""

    ctx, records, start = _quiesced(tmp_path)
    change = _event("g", "DCGM_EXPORTER", 12, incident_id="inc-g")
    change["gpu_uuid"] = G1
    records["events"].append(change)

    class LostJournal:
        interval = 15

        def advance(self) -> dict[str, Any]:
            return {"advanced": True}

        def audit(self) -> list[dict[str, Any]]:
            raise HostProbeMissingResponseError(f"dcgm audit-read: {STOPPED_KUBELET}")

    kernel = QuiescedKernel(ctx, back=0.0)
    delivery = placement.deliver_gauge_first(
        ctx,
        LostJournal(),  # type: ignore[arg-type]
        kernel,  # type: ignore[arg-type]
        since=start,
        gpu=G1,
        xid=48,
        delay_seconds=0,
    )

    assert (delivery.served_from, delivery.written_from) == ("product record", "probe")
    assert delivery.gauge_served_at == start.timestamp() + 12, delivery.record()
    deferred = json.loads((ctx.case_dir / "twin-change-deferred.json").read_text())
    assert deferred["reason"].startswith("HostProbeMissingResponseError"), deferred
    assert deferred["event_id"] == "g", deferred

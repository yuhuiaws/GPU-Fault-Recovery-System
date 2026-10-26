"""The DCGM replay's close when the product took the probe away.

Live 2026-10-26 14:54Z (the HyperPod lane's node, R13g, 024
``wrong_scope_source``): the repair GPU's reset chain quiesced the node
(kubelet stopped) ~20 s after the mint, ``close-run`` returned "host probe
response is missing (exit 1)" and the variant ended FAIL "cleanup did not
prove the node clean"; the chain then escalated to RESTART_NODE and the node
rebooted (NotReady ~14:57Z, Ready 14:59:45Z), which emptied ``/run`` and with
it every replay object. Over the scripted probe and a scripted Node object:
the close waits for the node and retries with the journal read first; a node
whose boot id changed closes the run by its reboot; a node that never answers
again without rebooting leaves the close unproven, inside the budget.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import general_live_gpu_memory as mem
from scripts.e2e.regional.general_live_gpu_replay import (
    DcgmReplay,
    ReplayCleanupError,
    add,
    baseline,
    plan,
    step,
)
from scripts.e2e.regional.general_live_gpu_replay_close import (
    JUDGED_BY_REBOOT,
    QUIESCE_TAINT,
    READY_SETTLE_SECONDS,
    RETRY_INTERVAL_SECONDS,
    NodeFacts,
    node_facts,
    unreachable_detail,
    wait_served_or_records,
)
from scripts.e2e.regional.host_probe_fixture import (
    HostProbeError,
    HostProbeMissingResponseError,
    HostProbeTransportError,
)
from tests.regional._general_live_fakes import FakeRegional, make_context, make_settings
from tests.regional._general_live_gpu_fakes import (
    audit_records,
    dcgm_probe,
    gpu_cpu_python,
    survey,
)

SPEC = {spec.variant: spec for spec in mem.VARIANTS}["corrected_ecc"]
GPU = "GPU-11111111-1111-4111-8111-111111111111"
NODE = "ip-10-0-43-5.us-west-2.compute.internal"
#: The node's boot id as the Node object and the survey report it.
BOOT = survey()["boot_id"]
NEW_BOOT = "44444444-4444-4444-8444-444444444444"
MISSING = "host probe response is missing (exit 1)"
TERMINATED = "owned host probe Pod has terminated"
#: ``make_context``'s clock starts here and advances one second per read.
START = datetime(2026, 10, 24, tzinfo=timezone.utc)
PLAN = plan("memory", baseline(), step(40, add("ecc_sbe_volatile_total", [GPU])))
RECORDS = audit_records({f"{GPU}/ecc_sbe_volatile_total": 1.0})


def _node(**overrides: Any) -> dict[str, Any]:
    value = {**FakeRegional().node_snapshot(""), "name": NODE, "boot_id": BOOT}
    value.update(overrides)
    return value


def _ctx(
    tmp_path: Path,
    probe: Any,
    *,
    nodes: list[dict[str, Any]],
    window_minutes: int = 120,
) -> Any:
    """A context whose Node object answers ``nodes`` in order (the first at the
    deploy, the rest per poll; the last repeats)."""

    regional = FakeRegional(cpu_python=gpu_cpu_python(), node_snapshot=nodes[0])
    sequence = iter(nodes)
    current = {"value": nodes[0]}

    def node_snapshot(node: str) -> dict[str, Any]:
        try:
            current["value"] = next(sequence)
        except StopIteration:
            pass
        return dict(current["value"])

    regional.node_snapshot = node_snapshot  # type: ignore[method-assign]
    settings = make_settings(SPEC, node=NODE)
    ctx = make_context(
        SPEC,
        tmp_path=tmp_path,
        regional=regional,
        probes={},
        settings=settings,
        window_minutes=window_minutes,
    )
    ctx.probes[f"dcgm@{NODE}"] = probe
    return ctx


def _probe(
    *, close_errors: list[Exception], journal_down_while_closing: bool = True
) -> tuple[Any, dict[str, int]]:
    """A probe whose ``close-run`` raises ``close_errors`` in turn and then
    answers clean; ``audit-read`` fails while a close error is still pending
    (the node is away) and serves the journal once it is not."""

    pending = list(close_errors)
    counts = {"close-run": 0, "audit-read": 0}

    def audit_read(arguments: tuple[str, ...]) -> dict[str, Any]:
        counts["audit-read"] += 1
        if pending and journal_down_while_closing:
            raise HostProbeMissingResponseError(MISSING)
        offset = int(arguments[arguments.index("--offset") + 1])
        chunk = RECORDS[offset:]
        return {
            "records": list(chunk),
            "next_offset": offset + len(chunk),
            "total": len(RECORDS),
        }

    def close_run(arguments: tuple[str, ...]) -> dict[str, Any]:
        counts["close-run"] += 1
        if pending:
            raise pending.pop(0)
        return dcgm_probe().script["close-run"]

    return dcgm_probe(extra={"audit-read": audit_read, "close-run": close_run}), counts


def _deployed(ctx: Any) -> DcgmReplay:
    replay = DcgmReplay(ctx, PLAN, 600)
    replay.deploy()
    replay.arm()
    return replay


def _close_evidence(ctx: Any) -> dict[str, Any]:
    return json.loads((ctx.case_dir / "dcgm-replay-close.json").read_text())


# --------------------------------------------------------------------------- #
# the node comes back (HyperPod quiesce: kubelet stopped, then restarted)
# --------------------------------------------------------------------------- #
def test_close_waits_for_the_node_and_retries_with_the_journal_first(
    tmp_path: Path,
) -> None:
    probe, counts = _probe(close_errors=[HostProbeMissingResponseError(MISSING)])
    ctx = _ctx(tmp_path, probe, nodes=[_node(), _node(ready="Unknown"), _node()])
    replay = _deployed(ctx)
    assert replay.node_boot_id == BOOT, "the deploy records the node's boot id"

    receipts = replay.close()

    assert receipts["close"]["clean"] is True
    assert "rebooted" not in receipts["close"]
    assert receipts["close_error"] == (
        f"HostProbeMissingResponseError: dcgm close-run: {MISSING}"
    )
    polls = receipts["close_after_probe_loss"]
    assert [(poll["ready"], poll["settled_ready"]) for poll in polls] == [
        ("Unknown", False),
        ("True", True),
    ]
    assert polls[-1]["close_run"] == "ok" and not any(p["rebooted"] for p in polls)
    assert counts == {"close-run": 2, "audit-read": 2}
    assert "audit_error" not in receipts, "the journal was read before the retry"
    oracle = [
        json.loads(line)
        for line in (ctx.case_dir / "oracle.jsonl").read_text().splitlines()
    ]
    assert [item["served"] for item in oracle] == [RECORDS[0]["served"]]
    timeline = json.loads((ctx.case_dir / "node-return-timeline.json").read_text())
    assert len(timeline["entries"]) == 2
    assert _close_evidence(ctx)["close"]["clean"] is True


def test_close_lets_ready_settle_before_it_tries_the_probe_again(
    tmp_path: Path,
) -> None:
    # A node that just flipped Ready flips once more when HyperPod restarts
    # kubelet with its final config; the retry waits READY_SETTLE_SECONDS.
    probe, counts = _probe(close_errors=[HostProbeMissingResponseError(MISSING)])
    flipped = (START + timedelta(seconds=20)).isoformat()
    ctx = _ctx(tmp_path, probe, nodes=[_node(), _node(ready_since=flipped)])
    receipts = _deployed(ctx).close()
    polls = receipts["close_after_probe_loss"]
    assert polls[0]["settled_ready"] is False and polls[-1]["settled_ready"] is True
    last = datetime.fromisoformat(polls[-1]["observed_at"])
    assert (
        last - datetime.fromisoformat(flipped)
    ).total_seconds() >= READY_SETTLE_SECONDS
    assert counts["close-run"] == 2 and receipts["close"]["clean"] is True


def test_close_waits_out_a_quiesce_taint_on_an_eks_node(tmp_path: Path) -> None:
    probe, counts = _probe(close_errors=[HostProbeMissingResponseError(MISSING)])
    tainted = _node(taints=[{"key": QUIESCE_TAINT, "effect": "NoExecute"}])
    ctx = _ctx(tmp_path, probe, nodes=[_node(), tainted, tainted, _node()])
    receipts = _deployed(ctx).close()
    polls = receipts["close_after_probe_loss"]
    assert [poll["quiesced"] for poll in polls] == [True, True, False]
    assert [poll["settled_ready"] for poll in polls] == [False, False, True]
    assert counts["close-run"] == 2 and receipts["close"]["clean"] is True


# --------------------------------------------------------------------------- #
# the node rebooted (RESTART_NODE): /run is empty by construction
# --------------------------------------------------------------------------- #
def test_a_rebooted_node_closes_the_run_by_its_boot_id(tmp_path: Path) -> None:
    probe, counts = _probe(
        close_errors=[
            HostProbeMissingResponseError(MISSING),
            HostProbeTransportError(TERMINATED),
        ]
    )
    ctx = _ctx(
        tmp_path, probe, nodes=[_node(), _node(ready="False"), _node(boot_id=NEW_BOOT)]
    )
    replay = _deployed(ctx)

    receipts = replay.close()

    closed = receipts["close"]
    assert closed["judged_by"] == JUDGED_BY_REBOOT
    assert (
        closed["clean"] is True and closed["leftover"] == [] and closed["errors"] == []
    )
    assert closed["rebooted"] == {"boot_id_before": BOOT, "boot_id_after": NEW_BOOT}
    assert "cannot have survived the boot" in closed["note"]
    polls = receipts["close_after_probe_loss"]
    assert [(poll["ready"], poll["rebooted"]) for poll in polls] == [
        ("False", False),
        ("True", True),
    ]
    assert (
        polls[-1]["close_run"]
        == f"HostProbeTransportError: dcgm close-run: {TERMINATED}"
    )
    assert counts["close-run"] == 2, "one retry on the rebooted node, then the verdict"
    assert receipts["audit_error"].startswith("HostProbeMissingResponseError: "), (
        "the journal could not be read on the rebooted node"
    )
    assert not replay.armed and not replay.repointed
    assert _close_evidence(ctx)["close"]["judged_by"] == JUDGED_BY_REBOOT


def test_a_rebooted_node_that_answers_the_retry_keeps_the_probes_proof(
    tmp_path: Path,
) -> None:
    probe, counts = _probe(close_errors=[HostProbeMissingResponseError(MISSING)])
    ctx = _ctx(tmp_path, probe, nodes=[_node(), _node(boot_id=NEW_BOOT)])
    receipts = _deployed(ctx).close()
    closed = receipts["close"]
    assert closed["clean"] is True and "judged_by" not in closed
    assert closed["rebooted"] == {"boot_id_before": BOOT, "boot_id_after": NEW_BOOT}
    assert counts["close-run"] == 2


def test_the_cleanup_stack_reports_a_reboot_closed_run_as_ok(tmp_path: Path) -> None:
    probe, _counts = _probe(
        close_errors=[
            HostProbeMissingResponseError(MISSING),
            HostProbeTransportError(TERMINATED),
        ]
    )
    ctx = _ctx(tmp_path, probe, nodes=[_node(), _node(boot_id=NEW_BOOT)])
    _deployed(ctx)
    results = ctx.cleanup.run()
    assert [(item["name"], item["ok"]) for item in results] == [("dcgm-replay", True)]
    assert results[0]["result"]["close"]["judged_by"] == JUDGED_BY_REBOOT


# --------------------------------------------------------------------------- #
# the node never answers again and did not reboot: unproven, inside the budget
# --------------------------------------------------------------------------- #
def test_a_node_that_never_answers_again_leaves_the_close_unproven(
    tmp_path: Path,
) -> None:
    probe, counts = _probe(close_errors=[HostProbeMissingResponseError(MISSING)] * 50)
    ctx = _ctx(
        tmp_path, probe, nodes=[_node(), _node(ready="Unknown")], window_minutes=8
    )
    replay = _deployed(ctx)
    with pytest.raises(ReplayCleanupError) as info:
        replay.close()
    message = str(info.value)
    assert message.startswith(
        f"replay cleanup on {NODE} -- failed steps: close-run: "
    ), message
    assert f"HostProbeMissingResponseError: dcgm close-run: {MISSING}" in message
    assert f"the node did not reboot (boot id {BOOT} unchanged)" in message
    assert "(Ready=Unknown)" in message
    assert counts["close-run"] == 1, "a node that is not Ready is never probed"
    evidence = _close_evidence(ctx)
    polls = evidence["close_after_probe_loss"]
    assert len(polls) >= 2 and all(poll["ready"] == "Unknown" for poll in polls)
    first = datetime.fromisoformat(polls[0]["observed_at"])
    last = datetime.fromisoformat(polls[-1]["observed_at"])
    # 8 min of budget less the 3-min reserve: the wait ends inside the window.
    assert 240 <= (last - first).total_seconds() <= 330
    assert ctx.remaining_seconds() > 150


def test_a_ready_node_that_still_refuses_the_probe_is_unproven_after_the_budget(
    tmp_path: Path,
) -> None:
    # Out of budget (the deadline is a minute away): one poll, one retry, done.
    probe, counts = _probe(close_errors=[HostProbeMissingResponseError(MISSING)] * 3)
    ctx = _ctx(tmp_path, probe, nodes=[_node(), _node()], window_minutes=1)
    with pytest.raises(ReplayCleanupError, match="did not reboot") as info:
        _deployed(ctx).close()
    assert "after 1 poll (Ready=True)" in str(info.value)
    assert counts["close-run"] == 2


def test_a_reboot_read_outside_the_budget_still_closes_the_run(tmp_path: Path) -> None:
    probe, counts = _probe(close_errors=[HostProbeMissingResponseError(MISSING)] * 3)
    ctx = _ctx(
        tmp_path,
        probe,
        nodes=[_node(), _node(ready="False", boot_id=NEW_BOOT)],
        window_minutes=1,
    )
    receipts = _deployed(ctx).close()
    assert receipts["close"]["judged_by"] == JUDGED_BY_REBOOT
    assert counts["close-run"] == 1, "not Ready: no retry; the reboot decides"


def test_a_ready_node_that_keeps_refusing_is_retried_per_interval_not_per_poll(
    tmp_path: Path,
) -> None:
    # Ready as the API sees it, kubelet down underneath (the node-monitor
    # grace period): the probe is tried again every RETRY_INTERVAL_SECONDS or
    # on a Ready transition, while the node is polled every NODE_POLL_SECONDS.
    probe, counts = _probe(close_errors=[HostProbeMissingResponseError(MISSING)] * 50)
    ctx = _ctx(tmp_path, probe, nodes=[_node(), _node()], window_minutes=8)
    with pytest.raises(ReplayCleanupError, match="did not reboot"):
        _deployed(ctx).close()
    polls = _close_evidence(ctx)["close_after_probe_loss"]
    tried = [poll for poll in polls if "close_run" in poll]
    assert len(polls) > 20, "the node was polled through the budget"
    assert 2 <= counts["close-run"] <= 4, counts
    assert counts["close-run"] == 1 + len(tried)
    gaps = [
        (
            datetime.fromisoformat(later["observed_at"])
            - datetime.fromisoformat(earlier["observed_at"])
        ).total_seconds()
        for earlier, later in zip(tried, tried[1:], strict=False)
    ]
    assert all(gap >= RETRY_INTERVAL_SECONDS for gap in gaps), gaps


def test_a_probe_error_that_is_not_the_nodes_absence_is_still_the_close_error(
    tmp_path: Path,
) -> None:
    # Same boot, Ready, the probe answers but refuses: unchanged behaviour --
    # the refusal is retried once and then named.
    probe, counts = _probe(
        close_errors=[HostProbeError("host probe failed (exit 1); output withheld")]
        * 3,
        journal_down_while_closing=False,
    )
    ctx = _ctx(tmp_path, probe, nodes=[_node(), _node()], window_minutes=1)
    with pytest.raises(ReplayCleanupError, match="host probe failed"):
        _deployed(ctx).close()
    assert counts["close-run"] == 2


# --------------------------------------------------------------------------- #
# the pieces
# --------------------------------------------------------------------------- #
def test_node_facts_read_the_node_object(tmp_path: Path) -> None:
    regional = FakeRegional(
        node_snapshot=_node(
            ready="True",
            ready_since="2026-10-26T14:59:45Z",
            taints=[{"key": QUIESCE_TAINT, "effect": "NoExecute"}, {"key": "x"}],
            unschedulable=True,
        )
    )
    ctx = make_context(SPEC, tmp_path=tmp_path, regional=regional, probes={})
    facts = node_facts(ctx)
    assert facts == NodeFacts(
        boot_id=BOOT,
        ready="True",
        ready_since="2026-10-26T14:59:45Z",
        unschedulable=True,
        taints=(QUIESCE_TAINT, "x"),
    )
    now = datetime(2026, 10, 26, 15, 5, tzinfo=timezone.utc)
    assert facts.quiesced and not facts.settled_ready(now)
    assert facts.as_dict()["quiesced"] is True
    settled = NodeFacts(BOOT, "True", "2026-10-26T14:59:45Z", False, ())
    assert settled.settled_ready(now), "Ready for over five minutes"
    assert not settled.settled_ready(
        datetime(2026, 10, 26, 15, 0, tzinfo=timezone.utc)
    ), "Ready for 15 s only"
    assert NodeFacts(BOOT, "True", "", False, ()).settled_ready(now), (
        "no transition time: Ready is taken as settled"
    )
    assert not NodeFacts(BOOT, "Unknown", "", False, ()).settled_ready(now), "not Ready"
    assert NodeFacts(BOOT, "True", "not a time", False, ()).settled_ready(now), (
        "an unreadable transition time is no reason to keep waiting"
    )


def test_unreachable_detail_names_the_last_poll() -> None:
    assert unreachable_detail([]) == "the node was not polled"
    detail = unreachable_detail(
        [{"ready": "Unknown", "boot_id": BOOT}, {"ready": "True", "boot_id": BOOT}]
    )
    assert detail == (
        f"the node did not reboot (boot id {BOOT} unchanged) and the probe could "
        "not run after 2 polls (Ready=True)"
    )
    assert "quiesce taint present" in unreachable_detail(
        [{"ready": "True", "boot_id": BOOT, "quiesced": True}]
    )


class _Gate:
    def __init__(self, ctx: Any, error: Exception | None) -> None:
        self.ctx = ctx
        self.error = error
        self.keys: list[tuple[str, ...]] = []

    def wait_served(self, *keys: str) -> None:
        self.keys.append(keys)
        if self.error is not None:
            raise self.error


def test_the_gate_falls_back_to_the_records_only_for_a_lost_probe(
    tmp_path: Path,
) -> None:
    ctx = make_context(SPEC, tmp_path=tmp_path, regional=FakeRegional(), probes={})
    key = f"{GPU}/ecc_sbe_volatile_total"
    assert (
        wait_served_or_records(
            _Gate(ctx, None), [key], records_show=lambda: False, label="x"
        )
        == "journal"
    )
    lost = _Gate(ctx, HostProbeMissingResponseError(f"dcgm audit-read: {MISSING}"))
    assert (
        wait_served_or_records(lost, [key], records_show=lambda: True, label="x")
        == "records"
    )
    deferred = json.loads((ctx.case_dir / "replay-gate-deferred.json").read_text())
    assert deferred["keys"] == [key] and deferred["label"] == "x"
    assert deferred["probe_error"] == (
        f"HostProbeMissingResponseError: dcgm audit-read: {MISSING}"
    )
    with pytest.raises(HostProbeMissingResponseError):
        wait_served_or_records(lost, [key], records_show=lambda: False, label="x")
    with pytest.raises(ValueError, match="not the probe"):
        wait_served_or_records(
            _Gate(ctx, ValueError("not the probe")),
            [key],
            records_show=lambda: True,
            label="x",
        )

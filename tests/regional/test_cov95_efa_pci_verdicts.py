"""GF-GENERAL-LIVE-013 ``pci_missing``: the graders and waits at their edges.

``test_general_live_efa_pci`` runs the variant end to end on the support
module's phased world. These cases feed each grader the record that makes it
name a problem -- an expected count the product never recorded as a number,
a node cordoned during the unreadable-root hold, a confirmed sample whose
streak is short, a chain that never plans RESTART_NODE, a failed reboot
command, a node that is back NotReady under another UID, a lane without a
reboot API, a node without an instance id, two receipts for one reboot, a
peer that rebooted, went NotReady, was cordoned and opened an incident -- and
drive the variant through a positive window whose close the rebooting node
refuses and a rebooted node that is not settled at the first look.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts.e2e.regional import general_live_efa_pci as mod
from scripts.e2e.regional.general_live_source_replay import host_source_replay
from scripts.e2e.regional.host_probe_fixture import HostProbeError
from tests.regional._general_live_efa_pci_support import (
    EXPECTED,
    INSTANCE,
    NODE,
    PEER,
    PEER_NODE,
    SPEC,
    RebootableProbe,
    World,
    _base_doc,
    _command,
    _confirmed_batches,
    _efa_batch,
    _event,
    _incident,
    _node,
    _probe_script,
    _settled_doc,
    _unknown_batch,
    _workflow,
    _world,
)
from tests.regional._general_live_fakes import make_context, make_settings

SINCE = datetime(2026, 10, 24, tzinfo=timezone.utc)


def fast_clock(world: SimpleNamespace, step: int = 120) -> None:
    ticks = iter(range(0, 10**7, step))
    world.ctx.clock = lambda: SINCE + timedelta(seconds=next(ticks))


def build(tmp_path: Path, regional_type: type[World], **world: Any) -> SimpleNamespace:
    """``_world`` with a ``World`` subclass (the support builder fixes the class)."""

    fake = RebootableProbe(_probe_script(**world.pop("probe_extra", {})))
    regional = regional_type(
        fake,
        negative=world.pop(
            "negative", _base_doc(batches=[_unknown_batch("2026-10-24T00:00:15+00:00")])
        ),
        detection=world.pop(
            "detection",
            _base_doc(
                incidents=[_incident("ACTION_PENDING")],
                workflows=[_workflow("RUNNING")],
                batches=_confirmed_batches(),
            ),
        ),
        settled=world.pop("settled", _settled_doc()),
        **world,
    )
    ctx = make_context(
        SPEC,
        tmp_path=tmp_path,
        regional=regional,
        probes={f"source-replay@{NODE}": fake},
        settings=make_settings(SPEC, node=NODE, peer=PEER),
        window_minutes=SPEC.max_minutes,
    )
    return SimpleNamespace(ctx=ctx, probe=fake, regional=regional)


# --------------------------------------------------------------------------- #
# readers of the product's records
# --------------------------------------------------------------------------- #
def test_an_expected_count_the_product_never_recorded_as_a_number_falls_back(
    tmp_path: Path,
) -> None:
    negative = _base_doc(
        metric_latest=[
            {"name": "efa_inventory_active_count", "value": EXPECTED},
            {"name": "efa_inventory_expected_count", "value": None},
        ]
    )
    world = _world(tmp_path, negative=negative)

    assert mod.expected_efa_count(world.ctx, 7) == 7
    assert mod.expected_efa_count(_world(tmp_path / "b").ctx, 7) == EXPECTED, (
        "a numeric expected count on the record wins over the fallback"
    )


@pytest.mark.parametrize(
    "streak, required",
    [("1", "2"), ("?", "2"), ("2", None)],
    ids=["short", "unreadable", "no-requirement"],
)
def test_a_confirmed_sample_needs_a_full_readable_streak(
    streak: str, required: str | None
) -> None:
    batch = _efa_batch(
        "2026-10-24T00:00:30+00:00",
        failure_mode="PCI_DEVICE_MISSING",
        active=EXPECTED - 1,
        mismatch=1,
        streak=2,
    )
    for sample in batch["samples"]:
        sample["labels"]["consecutive_mismatch_samples"] = streak
        sample["labels"]["required_consecutive_samples"] = required

    errors = mod.confirmed_sample_errors(_base_doc(batches=[batch]), expected=EXPECTED)

    assert errors == [
        f"confirmed sample streak {streak!r} is below the required {required!r}"
    ]


def test_the_reboot_wait_gives_up_on_a_chain_that_never_plans_restart_node(
    tmp_path: Path,
) -> None:
    detection = _base_doc(
        incidents=[_incident("ACTION_PENDING")],
        workflows=[
            _workflow("RUNNING", operations=("MARK_UNSCHEDULABLE", "QUARANTINE"))
        ],
    )
    world = _world(tmp_path, negative=detection)
    fast_clock(world)

    planned = mod.wait_reboot_planned(world.ctx, SINCE, ["inc-silent", "inc-efa"])

    assert planned is None
    entries = json.loads(
        (world.ctx.case_dir / "reboot-planned-timeline.json").read_text("utf-8")
    )["entries"]
    assert len(entries) > 1 and not any(item["accepted"] for item in entries), entries


def test_a_failed_reboot_command_is_named_with_its_error() -> None:
    failed = {**_command(), "status": "FAILED", "error": "SSM command timed out"}
    observed = _settled_doc(commands=[failed])

    errors = mod.reboot_receipt_errors(observed, ["inc-efa"], NODE)

    assert errors == [
        "RESTART_NODE command cmd-restart_node ended FAILED: SSM command timed out"
    ]


def test_a_node_back_not_ready_under_another_uid_is_not_restored() -> None:
    before = {**_node("boot-1"), "efa_allocatable": "32"}
    after = {
        **_node("boot-2"),
        "ready": "Unknown",
        "uid": "node-uid-replacement",
        "efa_allocatable": "32",
    }

    errors = mod.node_restored_errors(before, after, {"boot_id": "boot-2"})

    assert errors == ["the node is Ready=Unknown", "the Node UID changed"]


# --------------------------------------------------------------------------- #
# the provider's receipt
# --------------------------------------------------------------------------- #
def test_a_lane_without_a_reboot_api_or_an_instance_has_no_receipt(
    tmp_path: Path,
) -> None:
    world = _world(tmp_path)

    unknown = mod.provider_receipt(
        world.ctx, provider="gke", instance_id=INSTANCE, since=SINCE
    )
    anonymous = mod.provider_receipt(
        world.ctx, provider="eks", instance_id=None, since=SINCE
    )

    assert unknown == (["no provider reboot API is known for lane 'gke'"], [])
    assert anonymous == (["the node's providerID names no EC2 instance"], [])
    assert world.regional.provider_waits == [], "neither read reaches CloudTrail"


def test_two_receipts_for_one_reboot_are_one_too_many(tmp_path: Path) -> None:
    events = (_event(), _event(event_time="2026-10-24T00:06:00Z"))
    world = _world(tmp_path, events=events)

    errors, accepted = mod.provider_receipt(
        world.ctx, provider="eks", instance_id=INSTANCE, since=SINCE
    )

    assert errors == [f"2 reboot events name {INSTANCE} (expected one)"]
    assert accepted == list(events)


# --------------------------------------------------------------------------- #
# the pair node
# --------------------------------------------------------------------------- #
class DisturbedPeer(World):
    """A peer that rebooted, is NotReady, cordoned and has an incident."""

    def node_snapshot(self, node: str) -> dict[str, Any]:
        if node == PEER:
            return {
                **PEER_NODE,
                "boot_id": "peer-boot-2",
                "ready": "False",
                "unschedulable": True,
            }
        return super().node_snapshot(node)

    def _activity(self, arguments: tuple[str, ...]) -> dict[str, Any]:
        if arguments[1] == PEER:
            return _base_doc(node_id=PEER, incidents=[{"incident_id": "inc-peer-1"}])
        return super()._activity(arguments)


def test_a_disturbed_peer_names_every_change(tmp_path: Path) -> None:
    world = build(tmp_path, DisturbedPeer)

    errors = mod.peer_errors(world.ctx, dict(PEER_NODE), SINCE)

    assert errors == [
        f"{PEER} rebooted (peer-boot-1 -> peer-boot-2)",
        f"{PEER} is Ready=False",
        f"{PEER} is isolated",
        f"{PEER} has incidents since the window opened: ['inc-peer-1']",
    ]


# --------------------------------------------------------------------------- #
# the negative stage
# --------------------------------------------------------------------------- #
class CordonedNode(World):
    def node_snapshot(self, node: str) -> dict[str, Any]:
        value = super().node_snapshot(node)
        if node == NODE:
            value["unschedulable"] = True
        return value


def test_a_node_cordoned_during_the_unreadable_root_hold_fails_the_negative(
    tmp_path: Path,
) -> None:
    world = build(tmp_path, CordonedNode)
    replay = host_source_replay(world.ctx)

    negative = mod.unreadable_root(world.ctx, replay)

    assert negative["stages"]["negative_not_isolated"] == ["the node is isolated"]
    assert negative["stages"]["negative_unknown"] == [], (
        "the unreadable root still reads UNKNOWN"
    )


# --------------------------------------------------------------------------- #
# the variant
# --------------------------------------------------------------------------- #
def test_a_chain_that_never_plans_a_reboot_is_not_waited_out_twice(
    tmp_path: Path,
) -> None:
    settled = _settled_doc(
        incident_state="ESCALATED",
        workflows=[
            _workflow(
                "SUCCEEDED",
                operations=("MARK_UNSCHEDULABLE", "QUARANTINE", "VALIDATE_FABRIC"),
            )
        ],
        commands=[],
        fresh=False,
    )
    world = _world(tmp_path, reboots=False, settled=settled)
    fast_clock(world)

    result = mod.pci_missing(world.ctx)

    assert result["verdict"] == "FAIL", result
    assert result["stages"]["reboot_planned"] == [
        "no workflow of ['inc-efa'] planned RESTART_NODE within "
        f"{mod.STEP_TIMEOUT_SECONDS}s"
    ]
    assert result["stages"]["rebooted"] == [
        "no RESTART_NODE was planned, so no reboot was waited for"
    ]
    assert not (world.ctx.case_dir / "reboot-timeline.json").exists(), (
        "without a planned reboot no reboot wait runs"
    )
    assert result["boot_id_after"] is None


def test_a_positive_window_close_the_rebooting_node_refuses_is_harmless(
    tmp_path: Path,
) -> None:
    closes = {"n": 0}

    def close_window(_arguments: tuple[str, ...]) -> dict[str, Any]:
        closes["n"] += 1
        if closes["n"] == 2:
            raise HostProbeError("host probe Pod is gone (node rebooting)")
        return {"clean": True}

    world = _world(tmp_path, probe_extra={"close-window": close_window})

    result = mod.pci_missing(world.ctx)

    assert result["verdict"] == "PASS", result["errors"]
    assert result["window_close"]["clean"] is False
    assert result["window_close"]["error"] == (
        "HostProbeError: host probe Pod is gone (node rebooting)"
    )
    recorded = json.loads(
        (world.ctx.case_dir / "source-window-close.json").read_text("utf-8")
    )
    assert "post_reboot_clean" in recorded["note"]


#: The settle wait's own poll (``wait_node_settled_after_reboot``); every other
#: wait of the variant sleeps 15 s, 20 s or 30 s.
SETTLE_POLL = 5.0


class InstallerCatchesUp(World):
    """The new boot's installer reconciles only after one settle poll."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.waited: list[float] = []

    def node_snapshot(self, node: str) -> dict[str, Any]:
        value = super().node_snapshot(node)
        if node == NODE and value["boot_id"] == "boot-2":
            reconciled = SETTLE_POLL in self.waited
            value["installer_boot_id"] = "boot-2" if reconciled else "boot-1"
        return value


def test_the_rebooted_node_is_given_time_to_settle_before_the_probe_returns(
    tmp_path: Path,
) -> None:
    world = build(tmp_path, InstallerCatchesUp)
    world.ctx.sleep = world.regional.waited.append

    result = mod.pci_missing(world.ctx)

    assert result["verdict"] == "PASS", result["errors"]
    settled = json.loads((world.ctx.case_dir / "node-settled.json").read_text("utf-8"))
    assert settled["polls"] == 2 and settled["installer_reconciled"] is True, settled
    assert world.regional.waited.count(SETTLE_POLL) == 1, (
        "the settle wait slept one poll on the variant's clock before it settled"
    )

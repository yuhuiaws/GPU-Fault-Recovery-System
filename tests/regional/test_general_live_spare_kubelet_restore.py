"""GF-GENERAL-LIVE-030 ``not_ready``: the kubelet undo is awaited through the node.

Live 2026-10-27 (plain EKS, release 37db33f254a1, attempt 7): the explicit undo
of the kubelet stop -- the service window's ``cleanup`` over ``kubectl exec``
into the host-probe Pod -- was issued at 14:19:05Z with kubelet stopped since
14:17:47Z, got ``HostProbeMissingResponseError: host probe response is missing
(exit 1)`` (the exec channel *is* kubelet), and the variant judged "the
candidate is not Ready after the release (Unknown)" and "cleanup did not prove
the node clean" at 14:19:20Z. The window's on-node restore unit fired at
``restore_at`` 14:27:22Z and the node was Ready at 14:27:23Z: the node-side
deadman worked; the harness had judged the hand-back eight minutes early.

These tests drive ``service_window_node_restore.restore_through_node`` and the
030 ``SpareSession`` undo on a virtual clock with that live shape.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts.e2e.regional import general_live_spares as sp
from scripts.e2e.regional import service_window_node_restore as restore
from scripts.e2e.regional import warm_spare_fixture as wsf
from scripts.e2e.regional.host_probe_fixture import (
    HostProbeError,
    HostProbeMissingResponseError,
    HostProbeTransportError,
)
from scripts.e2e.regional.regional_commands import RegionalFixtureError
from tests.regional._general_live_fakes import FakeRegional, make_context, make_settings

NODE = "ip-10-0-32-162.us-west-2.compute.internal"
NODE_UID = "11111111-2222-3333-4444-555555555555"
TARGET = "ip-10-0-43-5.us-west-2.compute.internal"
#: The live timeline: undo issued 14:19:05Z, restore_at 14:27:22Z,
#: expires_at = restore_at + RECOVERY_SECONDS (180) = 14:30:22Z.
UNDO_AT = datetime(2026, 10, 27, 14, 19, 5, tzinfo=timezone.utc)
RESTORE_AT = datetime(2026, 10, 27, 14, 27, 22, tzinfo=timezone.utc)
EXPIRES_AT = RESTORE_AT + timedelta(seconds=180)
RESTORE_UNIT = "gpu-fault-destr008-24f58fd009dd8190aa84819793d1b1b8-restore.service"
BINDING = {
    "service": "kubelet.service",
    "restore_at": int(RESTORE_AT.timestamp()),
    "expires_at": int(EXPIRES_AT.timestamp()),
}
CLOSED = {
    "phase": "CLOSED",
    "record_kind": "TOMBSTONE",
    "after": {"Id": "kubelet.service", "ActiveState": "active", "SubState": "running"},
    "restore_reason": "deadline",
    "stop_requested": True,
    "stop_acknowledged": True,
}


class Clock:
    """A virtual wall clock; ``sleep`` advances it."""

    def __init__(self, start: datetime) -> None:
        self.now = start

    def __call__(self) -> datetime:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += timedelta(seconds=seconds)


class VirtualNode:
    """The spare as the API shows it: Ready=Unknown until ``ready_at``."""

    def __init__(self, clock: Clock, ready_at: datetime | None) -> None:
        self.clock = clock
        self.ready_at = ready_at
        self.reads = 0

    def ready(self) -> str:
        if self.ready_at is not None and self.clock() >= self.ready_at:
            return "True"
        return "Unknown"

    def snapshot(self) -> dict[str, Any]:
        self.reads += 1
        return {"name": NODE, "uid": NODE_UID, "ready": self.ready(), "labels": {}}


class ChannelUndo:
    """The explicit undo over the probe channel: no answer while kubelet is down."""

    def __init__(
        self, node: VirtualNode, error: type[Exception] = HostProbeMissingResponseError
    ):
        self.node = node
        self.error = error
        self.calls: list[str] = []

    def __call__(self) -> dict[str, Any]:
        self.calls.append(self.node.ready())
        if self.node.ready() != "True":
            raise self.error("host probe response is missing (exit 1)")
        return {"residuals": {"pod": False, "configmap": False}, "window": CLOSED}


def _restore(
    clock: Clock, node: VirtualNode, undo: Any, **overrides: Any
) -> dict[str, Any]:
    arguments: dict[str, Any] = dict(
        service="kubelet.service",
        node=NODE,
        undo=undo,
        node_snapshot=node.snapshot,
        binding=BINDING,
        restore_unit=RESTORE_UNIT,
        clock=clock,
        sleep=clock.sleep,
    )
    arguments.update(overrides)
    return restore.restore_through_node(**arguments)


# --------------------------------------------------------------------------- #
# restore_through_node
# --------------------------------------------------------------------------- #
def test_live_shape_missing_response_is_deferred_and_the_node_return_is_awaited() -> (
    None
):
    """The live run: no response through the stopped kubelet, the node back at
    restore_at + 1 s -- a PASS of the undo, with the restore observed."""

    clock = Clock(UNDO_AT)
    node = VirtualNode(clock, ready_at=RESTORE_AT + timedelta(seconds=1))
    undo = ChannelUndo(node)

    report = _restore(clock, node, undo)

    attempt = report["explicit_undo"]
    assert attempt["outcome"] == "deferred", attempt
    assert attempt["error"].startswith("HostProbeMissingResponseError: "), attempt
    assert attempt["node_ready"] == "Unknown" and attempt["deferred_to"] == RESTORE_UNIT
    assert report["restore_at"] == RESTORE_AT.isoformat()
    assert report["expires_at"] == EXPIRES_AT.isoformat()
    assert (
        report["ready_by"]
        == (
            EXPIRES_AT + timedelta(seconds=restore.NODE_READY_MARGIN_SECONDS)
        ).isoformat()
    ), "the wait outlasts the restore unit's authority by the margin"
    observed = report["node_ready"]
    assert observed["ready"] == "True"
    first_ready = datetime.fromisoformat(observed["observed_at"])
    assert (
        RESTORE_AT
        < first_ready
        <= RESTORE_AT + timedelta(seconds=restore.POLL_SECONDS + 1)
    )
    assert [entry["ready"] for entry in observed["timeline"]] == ["Unknown", "True"], (
        "the timeline keeps transitions, not every poll"
    )
    assert observed["polls"] > 60, "polled every POLL_SECONDS through the whole stop"
    # The explicit undo was repeated over the restored channel: the window's
    # CLOSED report is the proof of the restoration.
    assert undo.calls == ["Unknown", "True"]
    assert report["confirmation"]["window"] == CLOSED
    assert clock.now < EXPIRES_AT, "no waiting past the node's return"


def test_deadline_exceeded_fails_with_the_window_deadlines_and_the_last_condition() -> (
    None
):
    clock = Clock(UNDO_AT)
    node = VirtualNode(clock, ready_at=None)
    undo = ChannelUndo(node)

    with pytest.raises(restore.NodeRestoreOverdue) as info:
        _restore(clock, node, undo)

    message = str(info.value)
    ready_by = EXPIRES_AT + timedelta(seconds=restore.NODE_READY_MARGIN_SECONDS)
    for needle in (
        f"kubelet.service on {NODE} was not restored through the node by {ready_by.isoformat()}",
        f"restore unit {RESTORE_UNIT} was due at {RESTORE_AT.isoformat()}",
        f"authority expired at {EXPIRES_AT.isoformat()}",
        "last observed node condition Ready=Unknown at",
        f"the explicit undo at {UNDO_AT.isoformat()} got HostProbeMissingResponseError",
    ):
        assert needle in message, (needle, message)
    assert isinstance(info.value, RegionalFixtureError), (
        "the overdue restore is a fixture error the cleanup stack records"
    )
    assert undo.calls == ["Unknown"], (
        "no second undo over a channel that never came back"
    )
    assert clock.now >= ready_by, "the whole window was waited out"
    report = info.value.report
    assert (
        report["node_ready"]["ready"] == "Unknown"
        and report["node_ready"]["polls"] > 100
    )
    assert report["explicit_undo"]["outcome"] == "deferred"


def test_probe_channel_available_takes_the_explicit_undo_path() -> None:
    """kubelet up (stop timer not yet fired, or already restored; HyperPod or
    EKS alike): the explicit undo is made and is the restoration."""

    clock = Clock(UNDO_AT)
    node = VirtualNode(clock, ready_at=UNDO_AT)
    undo = ChannelUndo(node)

    report = _restore(clock, node, undo)

    assert report["explicit_undo"]["outcome"] == "restored"
    assert report["explicit_undo"]["result"]["window"] == CLOSED
    assert undo.calls == ["True"] and "confirmation" not in report
    assert report["ready_by"] == (UNDO_AT + timedelta(seconds=300)).isoformat(), (
        "the explicit path keeps the plain Ready wait"
    )
    assert report["node_ready"]["polls"] == 1 and clock.now == UNDO_AT


def test_transport_error_is_deferred_like_a_missing_response() -> None:
    clock = Clock(UNDO_AT)
    node = VirtualNode(clock, ready_at=RESTORE_AT + timedelta(seconds=3))
    undo = ChannelUndo(node, error=HostProbeTransportError)

    report = _restore(clock, node, undo)

    assert report["explicit_undo"]["outcome"] == "deferred"
    assert report["explicit_undo"]["error"].startswith("HostProbeTransportError: "), (
        report["explicit_undo"]
    )
    assert report["node_ready"]["ready"] == "True" and undo.calls == ["Unknown", "True"]


def test_a_rejected_host_action_is_not_deferred() -> None:
    """A host action the probe refused is a real failure: nothing to wait for."""

    clock = Clock(UNDO_AT)
    node = VirtualNode(clock, ready_at=None)

    def refused() -> dict[str, Any]:
        raise HostProbeError("host probe failed (exit 1); output withheld")

    with pytest.raises(HostProbeError, match="host probe failed"):
        _restore(clock, node, refused)
    assert clock.now == UNDO_AT and node.reads == 0


def test_an_unbound_window_waits_the_plain_timeout_and_names_it() -> None:
    """No binding (the stop never reached the node): nothing on the node will
    restore anything, so a missing response gets the plain Ready wait only."""

    clock = Clock(UNDO_AT)
    node = VirtualNode(clock, ready_at=None)
    undo = ChannelUndo(node)

    with pytest.raises(
        restore.NodeRestoreOverdue, match="no service window binding"
    ) as info:
        _restore(
            clock, node, undo, binding=None, restore_unit="", ready_timeout_seconds=60
        )
    assert info.value.report["restore_unit"] is None
    assert clock.now == UNDO_AT + timedelta(seconds=60)


def test_window_deadlines_read_the_live_report_shape_and_refuse_junk() -> None:
    """The SCHEDULED report the live run recorded: ``after`` null, the stop not
    yet requested or acknowledged (the stop timer had not fired when the probe
    answered), and the epoch deadlines the on-node unit runs on."""

    live = {
        "phase": "SCHEDULED",
        "record_kind": "RECOVERY_REQUIRED",
        "after": None,
        "stop_requested": False,
        "stop_acknowledged": False,
        "restore_at": 1793111242,
        "expires_at": 1793111422,
        "restore_unit": RESTORE_UNIT,
    }
    deadlines = restore.window_deadlines(live)
    assert deadlines is not None
    assert deadlines["restore_at"] == datetime(
        2026, 10, 27, 14, 27, 22, tzinfo=timezone.utc
    )
    assert deadlines["expires_at"] == datetime(
        2026, 10, 27, 14, 30, 22, tzinfo=timezone.utc
    )
    assert deadlines["ready_by"] == deadlines["expires_at"] + timedelta(
        seconds=restore.NODE_READY_MARGIN_SECONDS
    )
    assert restore.window_deadlines(None) is None
    assert restore.window_deadlines({}) is None
    assert restore.window_deadlines({"restore_at": "soon", "expires_at": 1}) is None
    assert restore.window_deadlines({"restore_at": True, "expires_at": 1}) is None


def test_restore_unit_name_comes_from_the_binding_when_the_report_was_lost() -> None:
    from scripts.e2e.regional.probes import warm_spare_node_probe as probe

    binding = {
        "case_id": "GF-REGIONAL-DESTR-008",
        "run_id": "gl030-not-ready-a7-4116e7cc",
        "owner": "0" * 32,
        "cluster_id": "gpu-eks-cluster",
        "node": NODE,
        "node_uid": NODE_UID,
        # eks-managed pins the Agent to the EC2 instance id; any safe id does here.
        "agent_identity": "instance-a",
        "release_id": "rel-test",
        "plan_sha256": "a" * 64,
        "helper_sha256": "b" * 64,
        "baseline_sha256": "c" * 64,
        "host_sha256": "d" * 64,
        "boot_id": "boot-1",
        "service": "kubelet.service",
        "restore_at": 1793111242,
        "expires_at": 1793111242 + probe.RECOVERY_SECONDS,
        "stop_delay_seconds": 15,
    }
    name = restore.restore_unit_name(binding)
    assert name.startswith("gpu-fault-destr008-") and name.endswith("-restore.service")
    assert restore.restore_unit_name(None) == ""
    assert restore.restore_unit_name({"restore_at": 1}) == ""


# --------------------------------------------------------------------------- #
# the 030 SpareSession undo
# --------------------------------------------------------------------------- #
def _service_fixture_fake(clock: Clock, node: VirtualNode) -> tuple[type, list[Any]]:
    """``WarmSpareServiceFixture`` as the live run saw it: the stop is
    SCHEDULED (the stop timer fires 15 s later), the on-node unit restores
    kubelet at restore_at, and ``cleanup`` over the probe channel has no answer
    while kubelet is down."""

    instances: list[Any] = []

    class FakeService:
        def __init__(self, warm: Any, **kwargs: Any) -> None:
            self.kwargs = kwargs
            self.window = SimpleNamespace(data={})
            self.cleanups: list[str] = []
            instances.append(self)

        def create(self) -> None:
            self.created = True

        def stop(
            self, service: str, *, restore_seconds: int, delay_seconds: int
        ) -> dict[str, Any]:
            now = clock()
            restore_at = now + timedelta(seconds=delay_seconds + restore_seconds)
            expires_at = restore_at + timedelta(seconds=180)
            self.window.data["binding"] = {
                "service": service,
                "restore_at": int(restore_at.timestamp()),
                "expires_at": int(expires_at.timestamp()),
            }
            node.ready_at = restore_at + timedelta(seconds=1)
            self.down_at = now + timedelta(seconds=delay_seconds)
            return {
                "phase": "SCHEDULED",
                "record_kind": "RECOVERY_REQUIRED",
                "service": service,
                "after": None,
                "stop_requested": False,
                "stop_acknowledged": False,
                "restore_at": int(restore_at.timestamp()),
                "expires_at": int(expires_at.timestamp()),
                "restore_unit": RESTORE_UNIT,
            }

        def cleanup(self) -> dict[str, bool]:
            self.cleanups.append(node.ready())
            if node.ready() != "True":
                raise HostProbeMissingResponseError(
                    "host probe response is missing (exit 1)"
                )
            self.window.data["host_cleanup"] = dict(CLOSED)
            return {"pod": False, "configmap": False}

    return FakeService, instances


def _session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, clock: Clock, node: VirtualNode
) -> Any:
    spec = next(s for s in sp.VARIANTS if s.variant == "not_ready")
    settings = make_settings(spec, node=TARGET, peer=NODE)
    ctx = make_context(
        spec, tmp_path=tmp_path, regional=FakeRegional(), probes={}, settings=settings
    )
    ctx.clock = clock
    ctx.sleep = clock.sleep
    monkeypatch.setattr(
        wsf.WarmSpareLiveFixture,
        "node_snapshot",
        lambda self, name, **_: node.snapshot(),
    )
    monkeypatch.setattr(sp, "pods_not_ready", lambda regional, name: [])
    return sp.SpareSession(ctx)


def test_stop_kubelet_undo_waits_through_the_node_and_records_the_restore(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = Clock(UNDO_AT - timedelta(minutes=2))
    node = VirtualNode(clock, ready_at=None)
    fake, instances = _service_fixture_fake(clock, node)
    monkeypatch.setattr(wsf, "WarmSpareServiceFixture", fake)
    session = _session(tmp_path, monkeypatch, clock, node)

    stopped = session.stop_kubelet()
    assert stopped["phase"] == "SCHEDULED" and node.ready() == "Unknown"
    (service,) = instances
    assert service.kwargs["case_id"] == sp.SERVICE_WINDOW_FIXTURE_CASE_ID
    # The variant graded here (~80 s into the stop live); then the undo.
    clock.sleep(80)
    results = session.ctx.cleanup.run()

    (entry,) = results
    assert entry["name"] == "spare-kubelet" and entry["ok"], entry
    report = entry["result"]
    assert report["explicit_undo"]["outcome"] == "deferred"
    assert report["explicit_undo"]["deferred_to"] == RESTORE_UNIT
    assert report["node_ready"]["ready"] == "True"
    assert report["confirmation"]["window"]["phase"] == "CLOSED"
    assert report["confirmation"]["window"]["restore_reason"] == "deadline", (
        "the window names the on-node unit as the restorer"
    )
    assert report["pods_not_ready"] == []
    assert service.cleanups == ["Unknown", "True"]
    restore_at = datetime.fromtimestamp(stopped["restore_at"], timezone.utc)
    assert (
        restore_at
        < clock.now
        < restore_at + timedelta(seconds=restore.POLL_SECONDS + 2)
    )
    written = json.loads(
        (session.ctx.case_dir / "spare-kubelet-restore.json").read_text(
            encoding="utf-8"
        )
    )
    assert written["restore_at"] == restore_at.isoformat()
    assert written["explicit_undo"]["outcome"] == "deferred"


def test_stop_kubelet_undo_names_the_deadlines_when_the_node_never_returns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = Clock(UNDO_AT - timedelta(minutes=2))
    node = VirtualNode(clock, ready_at=None)
    fake, instances = _service_fixture_fake(clock, node)
    monkeypatch.setattr(wsf, "WarmSpareServiceFixture", fake)
    session = _session(tmp_path, monkeypatch, clock, node)

    stopped = session.stop_kubelet()
    node.ready_at = None  # the on-node unit never brought kubelet back
    clock.sleep(80)
    (entry,) = session.ctx.cleanup.run()

    assert entry["name"] == "spare-kubelet" and not entry["ok"]
    error = entry["error"]
    restore_at = datetime.fromtimestamp(stopped["restore_at"], timezone.utc)
    expires_at = datetime.fromtimestamp(stopped["expires_at"], timezone.utc)
    assert error.startswith("NodeRestoreOverdue: kubelet.service on "), error
    assert f"was due at {restore_at.isoformat()}" in error
    assert f"authority expired at {expires_at.isoformat()}" in error
    assert "Ready=Unknown" in error
    assert instances[0].cleanups == ["Unknown"]
    # The partial trail is still evidence.
    written = json.loads(
        (session.ctx.case_dir / "spare-kubelet-restore.json").read_text(
            encoding="utf-8"
        )
    )
    assert written["node_ready"]["ready"] == "Unknown"
    # ``restored_errors`` then names both: the node not Ready and the failed undo.
    errors = sp.restored_errors(
        {
            "spare_before": {"ready": "True", "labels": {}, "annotations": {}},
            "spare_after": {"ready": "Unknown", "labels": {}, "annotations": {}},
            "residue": {},
            "cleanup": [entry],
        }
    )
    assert errors[0] == "the candidate is not Ready after the release (Unknown)"
    assert errors[1].startswith("undo spare-kubelet failed: NodeRestoreOverdue: "), (
        errors
    )


def test_stop_kubelet_undo_with_kubelet_up_is_explicit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The stop timer has not fired yet (or the unit already restored): the
    explicit undo closes the window at once; no deadline wait."""

    clock = Clock(UNDO_AT)
    node = VirtualNode(clock, ready_at=UNDO_AT)
    fake, instances = _service_fixture_fake(clock, node)
    monkeypatch.setattr(wsf, "WarmSpareServiceFixture", fake)
    session = _session(tmp_path, monkeypatch, clock, node)

    session.stop_kubelet()
    node.ready_at = UNDO_AT  # kubelet never went down before the undo
    (entry,) = session.ctx.cleanup.run()

    assert entry["ok"] and entry["result"]["explicit_undo"]["outcome"] == "restored"
    assert instances[0].cleanups == ["True"] and "confirmation" not in entry["result"]
    assert clock.now == UNDO_AT, "nothing was waited for"


# --------------------------------------------------------------------------- #
# the close over the restored channel
# --------------------------------------------------------------------------- #
class LateChannel(ChannelUndo):
    """The node is back, but the exec channel answers only from ``answers_from``."""

    def __init__(self, node: VirtualNode, answers_from: int | None) -> None:
        super().__init__(node)
        self.answers_from = answers_from

    def __call__(self) -> dict[str, Any]:
        self.calls.append(self.node.ready())
        if self.node.ready() != "True" or (
            self.answers_from is None or len(self.calls) < self.answers_from
        ):
            raise HostProbeMissingResponseError(
                "host probe response is missing (exit 1): no output came back"
            )
        return {"residuals": {"pod": False, "configmap": False}, "window": CLOSED}


def test_a_close_unanswered_right_after_the_node_returns_is_retried() -> None:
    """kubelet posts Ready a moment before it serves exec into the old Pod:
    the close is repeated, the misses kept beside the proof."""

    clock = Clock(UNDO_AT)
    node = VirtualNode(clock, ready_at=RESTORE_AT + timedelta(seconds=1))
    undo = LateChannel(node, answers_from=4)  # deferred, two misses, then CLOSED

    report = _restore(clock, node, undo)

    assert report["confirmation"]["window"] == CLOSED
    misses = report["confirmation_misses"]
    assert [miss["node_ready"] for miss in misses] == ["True", "True"], misses
    assert all(
        m["error"].startswith("HostProbeMissingResponseError: ") for m in misses
    ), misses
    first, second = (datetime.fromisoformat(m["at"]) for m in misses)
    assert second - first == timedelta(seconds=restore.CONFIRM_RETRY_SECONDS)
    assert undo.calls == ["Unknown", "True", "True", "True"]


def test_a_node_back_whose_close_never_answers_fails_naming_every_miss() -> None:
    clock = Clock(UNDO_AT)
    node = VirtualNode(clock, ready_at=RESTORE_AT + timedelta(seconds=1))
    undo = LateChannel(node, answers_from=None)

    with pytest.raises(restore.NodeRestoreUnconfirmed) as info:
        _restore(clock, node, undo)

    message = str(info.value)
    returned = info.value.report["node_ready"]["observed_at"]
    for needle in (
        f"kubelet.service on {NODE} is back (the node posted Ready at {returned})",
        f"got no answer in {restore.CONFIRM_ATTEMPTS} attempts "
        f"{restore.CONFIRM_RETRY_SECONDS:g} s apart",
        "HostProbeMissingResponseError: host probe response is missing (exit 1)",
        "node Ready=True",
        "the window stays open -- its claim, drop-in and units remain on the node",
        f"(restore unit {RESTORE_UNIT})",
    ):
        assert needle in message, (needle, message)
    assert isinstance(info.value, restore.NodeRestoreError), (
        "the 030 undo records the report of every NodeRestoreError"
    )
    assert isinstance(info.value, RegionalFixtureError), (
        "an unconfirmed restore is a fixture error the cleanup stack records"
    )
    assert len(info.value.report["confirmation_misses"]) == restore.CONFIRM_ATTEMPTS
    assert len(undo.calls) == 1 + restore.CONFIRM_ATTEMPTS, "bounded, then failed"


def test_a_refused_close_over_the_restored_channel_is_not_retried() -> None:
    clock = Clock(UNDO_AT)
    node = VirtualNode(clock, ready_at=RESTORE_AT + timedelta(seconds=1))
    calls: list[str] = []

    def undo() -> dict[str, Any]:
        calls.append(node.ready())
        if node.ready() != "True":
            raise HostProbeMissingResponseError("host probe response is missing")
        raise HostProbeError("host probe failed (exit 1; error ProbeError)")

    with pytest.raises(HostProbeError, match="host probe failed"):
        _restore(clock, node, undo)
    assert calls == ["Unknown", "True"], "a host refusal is a real failure at once"


def test_stop_kubelet_undo_records_the_trail_of_an_unanswered_close(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = Clock(UNDO_AT - timedelta(minutes=2))
    node = VirtualNode(clock, ready_at=None)
    fake, instances = _service_fixture_fake(clock, node)
    monkeypatch.setattr(wsf, "WarmSpareServiceFixture", fake)
    session = _session(tmp_path, monkeypatch, clock, node)
    session.stop_kubelet()

    def silent() -> dict[str, bool]:
        instances[0].cleanups.append(node.ready())
        raise HostProbeMissingResponseError("host probe response is missing (exit 1)")

    instances[0].cleanup = silent
    clock.sleep(80)
    (entry,) = session.ctx.cleanup.run()

    assert entry["name"] == "spare-kubelet" and not entry["ok"]
    assert entry["error"].startswith("NodeRestoreUnconfirmed: kubelet.service on "), (
        entry["error"]
    )
    written = json.loads(
        (session.ctx.case_dir / "spare-kubelet-restore.json").read_text(
            encoding="utf-8"
        )
    )
    assert written["node_ready"]["ready"] == "True"
    assert len(written["confirmation_misses"]) == restore.CONFIRM_ATTEMPTS

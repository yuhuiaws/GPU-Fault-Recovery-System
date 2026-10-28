"""A service window left open on the node is named, never stepped on.

Live 2026-10-27 (plain EKS, GF-GENERAL-LIVE-030 ``not_ready``, release
37db33f254a1): the spare's kubelet window was restored by its on-node unit at
``restore_at`` but never cleaned -- the undo got "host probe response is
missing (exit 1)" while kubelet was down -- so the node kept the window's
claim, drop-in and units and the case directory its controller journal at
CLEANUP_REQUIRED. The rerun of the same lane and attempt on the next release
(2026-10-28 03:06Z) got the same run id, met that journal and refused only
"service controller identity or state changed". On HyperPod (2026-10-24/25) a
window left the same way on one node made the next run's window refuse the
claim at ``prepare`` -- after writing its own journal -- and that window's
cleanup then stopped at the same claim: a second window nobody could close.

These tests pin the probe's read-only claim report, the controller's refusal
of a claimed service before anything is armed (and the clean close of the
refused window), and the journal refusal that names what differs.
"""

from __future__ import annotations

import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import pytest

from scripts.e2e.regional import destr008_service_window as controller
from scripts.e2e.regional.host_probe_fixture import (
    HostProbeFixture,
    HostProbeMissingResponseError,
)
from scripts.e2e.regional.probes import warm_spare_node_probe as probe
from scripts.e2e.regional.regional_commands import RegionalFixtureError
from tests.regional._destr008_service_window import PINS, Host, Transport, write

EXPIRES = datetime.fromtimestamp(2_000_001_200, timezone.utc)


def build(transport: Transport, **overrides: Any) -> controller.ServiceWindowController:
    kwargs: dict[str, Any] = {
        "cluster_id": PINS["cluster_id"],
        "node_uid": PINS["node_uid"],
        "agent_identity": PINS["agent_identity"],
        "plan_sha256": "d" * 64,
        "release_id": "test-release",
        "maintenance_expires_at": EXPIRES,
        "read_node_uid": lambda: PINS["node_uid"],
        **overrides,
    }
    return controller.ServiceWindowController(
        cast(HostProbeFixture, transport), **kwargs
    )


@pytest.fixture
def host(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Host:
    monkeypatch.setenv("INVOCATION_ID", "a" * 32)
    return Host(tmp_path, monkeypatch)


def _restored_by_deadline(host: Host) -> None:
    """The live shape: stopped, then restored by the on-node unit, never cleaned."""

    host.stopped()
    host.now = host.binding["restore_at"]
    host.elapsed = host.read()["clock"]["restore"]
    assert host.tick()["restore_reason"] == "deadline"


# --------------------------------------------------------------------------- #
# the probe's read-only claim report
# --------------------------------------------------------------------------- #
def test_a_clean_host_reports_no_claim_and_a_closed_window_leaves_none(
    host: Host,
) -> None:
    assert probe.capture("kubelet.service")["claim"] is None
    host.stopped()
    host.window.restore()
    host.window.cleanup()
    assert probe.capture("kubelet.service")["claim"] is None, (
        "cleanup removes the claim last; a closed window holds nothing"
    )


def test_the_claim_names_the_window_left_restored_by_its_deadline(host: Host) -> None:
    _restored_by_deadline(host)
    before = list(host.calls)

    claim = probe.capture("kubelet.service")["claim"]

    assert claim == {
        "path": str(probe.ROOT / "kubelet.service.claim"),
        "key": host.window.key,
        "unreadable": False,
        "phase": "RESTORED",
        "restore_reason": "deadline",
        "case_id": "GF-REGIONAL-DESTR-008",
        "run_id": host.binding["run_id"],
        "release_id": host.binding["release_id"],
        "restore_at": host.binding["restore_at"],
        "expires_at": host.binding["expires_at"],
    }
    assert [call for call in host.calls[len(before) :] if call[0] != "show"] == [], (
        "the snapshot changes nothing on the host"
    )
    assert probe.capture("gpu-fault-node-agent.service")["claim"] is None, (
        "a claim is per service"
    )


@pytest.mark.parametrize("defect", ["no-journal", "garbage", "symlink", "directory"])
def test_a_claim_that_cannot_be_read_is_reported_as_held(
    host: Host, defect: str
) -> None:
    _restored_by_deadline(host)
    path = host.window.targets["claim"]
    if defect == "no-journal":
        host.window.state_path.unlink()
    elif defect == "garbage":
        path.unlink()
        write(path, "not a window key")
    elif defect == "symlink":
        path.unlink()
        path.symlink_to(host.window.targets["helper"])
    else:
        path.unlink()
        path.mkdir(mode=0o700)

    claim = probe.capture("kubelet.service")["claim"]

    assert claim is not None and claim["unreadable"] is True, claim
    assert claim["key"] == (host.window.key if defect == "no-journal" else None)


# --------------------------------------------------------------------------- #
# the controller refuses a claimed service before anything is armed
# --------------------------------------------------------------------------- #
def test_a_claimed_service_is_refused_before_arming_and_the_refused_window_closes(
    host: Host, tmp_path: Path
) -> None:
    first = build(Transport(host, tmp_path / "earlier-run"))
    first.create()
    first.stop("kubelet.service", delay_seconds=15)
    held = dict(host.binding)
    host.fire_stop()
    host.now = held["restore_at"]
    host.elapsed = host.read()["clock"]["restore"]
    assert host.tick()["restore_reason"] == "deadline"
    first.close()  # the earlier run's undo never closed it
    holder = probe.ServiceWindow(held)
    on_host = probe.private_read(holder.state_path)

    transport = Transport(host, tmp_path / "rerun")
    window = build(transport, release_id="next-release")
    try:
        window.create()
        with pytest.raises(RegionalFixtureError) as refused:
            window.stop("kubelet.service", delay_seconds=15)
        message = str(refused.value)
        assert message.startswith(
            "service baseline does not match the approved target: kubelet.service "
            f"is already claimed by service window {holder.key[:32]} ("
        ), message
        for needle in (
            "case_id 'GF-REGIONAL-DESTR-008'",
            f"run_id {held['run_id']!r}",
            "release_id 'test-release'",
            "phase 'RESTORED'",
            "restore_reason 'deadline'",
            "restore_at "
            + datetime.fromtimestamp(held["restore_at"], timezone.utc).isoformat(),
            "expires_at "
            + datetime.fromtimestamp(held["expires_at"], timezone.utc).isoformat(),
            "until that window is closed by its own cleanup or retired on the node",
            str(probe.ROOT / "kubelet.service.claim"),
        ):
            assert needle in message, (needle, message)
        assert transport.commands == ["snapshot"], "nothing was prepared or armed"
        assert "binding" not in window.data
        residuals = window.cleanup()
    finally:
        window.close()

    assert all(value is False for value in residuals.values()), residuals
    assert probe.private_read(window.path)["phase"] == "CLOSED", (
        "the refused window closes: no second open journal"
    )
    assert transport.commands == ["snapshot"], "its cleanup sent the host nothing"
    assert probe.private_read(holder.state_path) == on_host, (
        "the holder's journal is untouched"
    )
    assert holder.targets["claim"].exists() and holder.dropin.exists(), (
        "the holder's claim and drop-in are never another window's to remove"
    )


def test_baseline_refusals_name_the_claim_or_its_absence() -> None:
    scope = {"cluster_id": "c", "node": "n", "agent_identity": "i"}
    state = {
        "Id": "kubelet.service",
        "LoadState": "loaded",
        "ActiveState": "active",
        "SubState": "running",
        "UnitFileState": "enabled",
        "MainPID": "42",
        "ControlPID": "0",
        "InvocationID": "b" * 32,
        "ExecMainStartTimestampMonotonic": "1",
        "Job": "",
    }
    base = {"service": "kubelet.service", "pins": dict(scope), "state": state}
    refusals = controller.baseline_refusals
    assert refusals(base, service="kubelet.service", scope=scope) == [
        "snapshot does not report the service's window claim"
    ], "a probe that does not report the claim cannot prove the service unheld"
    assert (
        refusals({**base, "claim": None}, service="kubelet.service", scope=scope) == []
    )
    (unknown,) = refusals(
        {**base, "claim": {"path": "/x.claim", "key": None, "unreadable": True}},
        service="kubelet.service",
        scope=scope,
    )
    assert unknown == (
        "kubelet.service is already claimed by an unidentified service window "
        "(its on-node journal could not be read): a new window cannot own "
        "kubelet.service until that window is closed by its own cleanup or "
        "retired on the node ('/x.claim')"
    )
    (odd,) = refusals(
        {**base, "claim": ["held"]}, service="kubelet.service", scope=scope
    )
    assert odd == "the service claim report is list, not a JSON object"


# --------------------------------------------------------------------------- #
# the journal refusal names what differs
# --------------------------------------------------------------------------- #
def _abandoned_journal(host: Host, transport: Transport) -> dict[str, Any]:
    """The earlier execution: stopped, then an undo that got no response."""

    first = build(transport)
    first.create()
    first.stop("kubelet.service", delay_seconds=15)
    host.fire_stop()
    execute = transport.execute

    def no_response(command: str, *args: str, timeout: int = 180) -> dict[str, Any]:
        if command == "cleanup":
            raise HostProbeMissingResponseError(
                "host probe response is missing (exit 1)"
            )
        return execute(command, *args, timeout=timeout)

    transport.execute = no_response  # type: ignore[method-assign]
    with pytest.raises(HostProbeMissingResponseError):
        first.cleanup()
    transport.execute = execute  # type: ignore[method-assign]
    first.close()
    data = probe.private_read(first.path)
    assert data["phase"] == "CLEANUP_REQUIRED" and data["last_request"] == "cleanup"
    assert data["last_failure"] == "HostProbeMissingResponseError"
    return data


def test_a_rerun_under_the_same_run_id_is_told_which_journal_and_what_differs(
    host: Host, tmp_path: Path
) -> None:
    transport = Transport(host, tmp_path / "spare-service")
    data = _abandoned_journal(host, transport)
    commands = list(transport.commands)
    later = datetime.fromtimestamp(2_000_002_400, timezone.utc)

    rerun = build(transport, release_id="next-release", maintenance_expires_at=later)
    try:
        with pytest.raises(RegionalFixtureError) as refused:
            rerun.create()
    finally:
        rerun.close()

    message = str(refused.value)
    binding = data["binding"]
    assert message.startswith(
        "service controller identity or state changed: the controller journal "
        f"{rerun.path.name} (phase 'CLEANUP_REQUIRED', last request 'cleanup' "
        "failed with HostProbeMissingResponseError, a window bound for "
        f"'kubelet.service' on {PINS['node']!r} (restore_at "
        f"{datetime.fromtimestamp(binding['restore_at'], timezone.utc).isoformat()}"
        ", expires_at "
        f"{datetime.fromtimestamp(binding['expires_at'], timezone.utc).isoformat()}"
        ")) is not this controller's: "
    ), message
    assert (
        "maintenance_expires_at recorded 2033-05-18T03:53:20+00:00, this run "
        "2033-05-18T04:13:20+00:00; release_id recorded 'test-release', this run "
        "'next-release'."
    ) in message, message
    assert "Only the execution that wrote it can resume it" in message, message
    assert transport.commands == commands, "no request reached the host"
    assert probe.private_read(rerun.path) == data, "the journal is left as found"


def test_a_digest_or_path_drift_is_named_without_its_value(
    host: Host, tmp_path: Path
) -> None:
    transport = Transport(host, tmp_path / "spare-service")
    window = build(transport)
    window.create()
    window.close()
    data = probe.private_read(window.path)
    data["scope"] = {
        **data["scope"],
        "controller_sha256": "f" * 64,
        "host_state_path": "/elsewhere/record.json",
    }
    probe.write_record(window.path, data)

    fresh = build(transport)
    try:
        with pytest.raises(RegionalFixtureError) as refused:
            fresh.resume_cleanup()
    finally:
        fresh.close()

    message = str(refused.value)
    assert "controller_sha256 differs; host_state_path differs" in message, message
    assert "phase 'CREATED'" in message and "no service window bound" in message
    assert re.search(r"[0-9a-f]{40,}", message) is None, "no digest is printed"
    assert "/elsewhere" not in message, "no path is printed"


@pytest.mark.parametrize(
    ("field", "value", "needle"),
    [
        ("phase", {}, "phase is {}"),
        ("owner", "nope", "owner is not a 32-hex token"),
        ("supervision_lost", "no", "supervision_lost is 'no'"),
        ("schema_version", 2, "schema_version is 2, not 1"),
        ("scope", [], "the journal has no scope (list)"),
    ],
)
def test_a_malformed_journal_names_its_defect(
    host: Host, tmp_path: Path, field: str, value: Any, needle: str
) -> None:
    transport = Transport(host, tmp_path / "spare-service")
    window = build(transport)
    window.create()
    window.close()
    data = probe.private_read(window.path)
    data[field] = value
    probe.write_record(window.path, data)

    fresh = build(transport)
    try:
        with pytest.raises(RegionalFixtureError, match="identity or state") as refused:
            fresh.resume_cleanup()
    finally:
        fresh.close()
    assert needle in str(refused.value), str(refused.value)


def test_the_journal_lock_is_released_after_a_refusal(
    host: Host, tmp_path: Path
) -> None:
    """A refused acquire closes its lock descriptor: the next reader is not
    told "another process owns" a journal nobody holds."""

    transport = Transport(host, tmp_path / "spare-service")
    _abandoned_journal(host, transport)
    for _ in range(2):
        rerun = build(transport, release_id="next-release")
        with pytest.raises(RegionalFixtureError, match="is not this controller's"):
            rerun.create()
        assert rerun.fd is None
    assert os.listdir(tmp_path / "spare-service"), "the journal stays for review"


def test_a_scope_key_added_or_dropped_is_named_and_still_refused(
    host: Host, tmp_path: Path
) -> None:
    """A recorded key this run does not have -- even one whose value is null --
    and a key the journal lacks are refusals, as the whole-scope equality was."""

    transport = Transport(host, tmp_path / "spare-service")
    window = build(transport)
    window.create()
    window.close()
    data = probe.private_read(window.path)
    del data["scope"]["release_id"]
    data["scope"]["extra"] = None
    probe.write_record(window.path, data)

    fresh = build(transport)
    try:
        with pytest.raises(RegionalFixtureError) as refused:
            fresh.resume_cleanup()
    finally:
        fresh.close()

    message = str(refused.value)
    assert (
        "extra is recorded but not in this run's scope; "
        "release_id is missing from the journal"
    ) in message, message

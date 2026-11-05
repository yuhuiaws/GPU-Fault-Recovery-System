"""Retiring a stale service window: the proofs and exits the first suite leaves.

``test_warm_spare_retire.py`` pins the live shape and the common refusals. These
tests run the same fake systemd host (``_destr008_service_window.Host``) through
the remaining paths of ``probes/warm_spare_retire.py``: a host that never wrote a
window, a journal that cannot be opened, a window its own cleanup already
closed, the per-artifact and per-neighbour refusals, the two in-retirement
failures that leave the CLOSING marker for the next call, and the probe's own
command line as the host runs it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional.probes import warm_spare_node_probe as probe
from scripts.e2e.regional.probes import warm_spare_retire as retire
from tests.regional._destr008_service_window import Host, write
from tests.regional.test_warm_spare_retire import (
    STALE,
    restored_by_deadline,
    stale_script,
    window_files,
)

OTHER = "e" * 64


@pytest.fixture
def host(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Host:
    monkeypatch.setenv("INVOCATION_ID", "a" * 32)
    value = Host(tmp_path, monkeypatch)
    run = tmp_path / "host-run"
    run.mkdir(mode=0o700)
    monkeypatch.setattr(retire, "RETIRE_RUN", run)
    return value


def retire_bound(host: Host, *arguments: str) -> dict[str, Any]:
    return retire.retire_window(probe.ServiceWindow(host.binding), *arguments)


def other_window(record: dict[str, Any] | None = None) -> Path:
    """Another window's directory beside this one's, with an optional journal."""

    path = probe.ROOT / OTHER
    path.mkdir(mode=0o700)
    if record is not None:
        probe.write_record(path / "state.json", record)
    return path


# --------------------------------------------------------------------------- #
# before the lock: nothing written, or a journal that cannot be opened
# --------------------------------------------------------------------------- #
def test_a_host_that_never_wrote_a_window_retires_only_the_named_script(
    host: Host,
) -> None:
    assert not probe.ROOT.exists(), "the fake host starts without a window root"
    script, digest = stale_script()
    window = probe.ServiceWindow(host.binding)

    refused = retire.retire_window(window, script, "0" * 64)

    assert refused == {
        "binding_sha256": window.key,
        "service": "kubelet.service",
        "retired": False,
        "refusals": [f"{STALE} is not the earlier probe's script (digest differs)"],
        "proofs": {"shape": "absent", "root": "absent"},
        "removed": [],
    }
    assert (retire.RETIRE_RUN / Path(STALE).name).exists(), "a refused copy is kept"

    report = retire.retire_window(window, script, digest)

    assert report["retired"] is True and report["removed"] == [STALE], report
    assert report["proofs"] == {"shape": "absent", "root": "absent"}
    assert not (retire.RETIRE_RUN / Path(STALE).name).exists(), "the copy is gone"
    assert not probe.ROOT.exists(), "no lock directory is created on a clean host"
    assert host.mutations() == [], "a clean host gets no systemd call"


def _orphan_resource(window: probe.ServiceWindow) -> None:
    probe.ROOT.parent.mkdir(mode=0o700)
    probe.ROOT.mkdir(mode=0o700)
    window.directory.mkdir(mode=0o700)
    write(window.directory / "helper.py", "a helper without its journal")


def _shared_root(window: probe.ServiceWindow) -> None:
    probe.ROOT.parent.mkdir(mode=0o700)
    probe.ROOT.mkdir(mode=0o700)
    probe.ROOT.chmod(0o755)  # readable by others, whatever the umask


@pytest.mark.parametrize(
    ("defect", "reason"),
    [
        (_orphan_resource, "service window journal is missing beside resources"),
        (_shared_root, "service window directory ownership is invalid"),
    ],
    ids=["resource-without-journal", "root-not-private"],
)
def test_a_journal_that_cannot_be_opened_is_a_refusal_not_an_error(
    host: Host, defect: Any, reason: str
) -> None:
    window = probe.ServiceWindow(host.binding)
    defect(window)
    present = sorted(path.name for path in probe.ROOT.rglob("*"))

    report = retire.retire_window(window)

    assert report == {
        "binding_sha256": window.key,
        "service": "kubelet.service",
        "retired": False,
        "refusals": [f"the window's host journal is unusable: {reason}"],
        "proofs": {},
        "removed": [],
    }
    after = sorted(path.name for path in probe.ROOT.rglob("*") if path.name != "lock")
    assert after == [name for name in present if name != "lock"], "nothing removed"
    assert host.mutations() == [], "no systemd call before the journal is read"


# --------------------------------------------------------------------------- #
# shapes and proofs
# --------------------------------------------------------------------------- #
def test_a_window_its_own_cleanup_closed_retires_with_its_tombstone(host: Host) -> None:
    restored_by_deadline(host)
    assert probe.ServiceWindow(host.binding).cleanup()["phase"] == "CLOSED"
    held = sorted(str(path) for path in host.window.directory.iterdir())
    before = len(host.calls)

    report = retire_bound(host)

    assert report["retired"] is True, report["refusals"]
    assert report["proofs"]["shape"] == "closed"
    assert report["proofs"]["phase"] == "CLOSED"
    assert "inert_windows" not in report["proofs"], "only RESTORED surveys others"
    assert sorted(report["removed"][:-1]) == held, report["removed"]
    assert report["removed"][-2:] == [
        str(host.window.state_path),
        str(host.window.directory),
    ], "the journal goes last, then its directory"
    assert not host.window.directory.exists(), "the tombstone is retired"
    assert [c for c in host.calls[before:] if c[0] != "show"] == [], (
        "a closed window needs no daemon-reload"
    )


def test_a_restored_journal_that_names_no_restorer_is_refused(host: Host) -> None:
    restored_by_deadline(host)
    record = host.read()
    record["restore_reason"] = "operator"
    host.save(record)

    report = retire_bound(host)

    assert report["retired"] is False
    assert report["refusals"] == ["the journal names no restorer for the service"]
    assert report["proofs"]["restored_by"] == "operator"
    assert host.read()["phase"] == "RESTORED", "the journal is left as found"


def test_an_artifact_not_named_by_the_window_key_is_refused(host: Host) -> None:
    window = probe.ServiceWindow(host.binding)
    foreign = probe.SYSTEMD / "gpu-fault-destr008-foreign-stop.timer"
    window.targets["stop-timer"] = foreign

    refusals = retire.retire_artifact_refusals(window, "absent")

    # Nothing else of a never-written window refuses: no artifact, no claim and
    # no window directory to inspect.
    assert refusals == [f"{foreign} does not carry the window's key"]
    assert not window.directory.exists(), "the proof reads, it never creates"


def test_a_unit_loaded_from_another_fragment_is_refused(host: Host) -> None:
    restored_by_deadline(host)
    unit = host.window.units["restore-service"]
    host.overrides[unit] = {"FragmentPath": "/usr/lib/systemd/system/elsewhere"}

    report = retire_bound(host)

    assert report["retired"] is False
    assert report["refusals"] == [
        f"{unit} is loaded from /usr/lib/systemd/system/elsewhere"
    ]
    assert all(path.exists() for path in window_files(host)), "nothing removed"


def test_an_unreadable_claim_blocks_the_never_published_window(host: Host) -> None:
    restored_by_deadline(host)
    later = {**host.make_binding(), "owner": "d" * 32}
    with pytest.raises(probe.ProbeError, match="no owner receipt"):
        probe.ServiceWindow(later).prepare()
    claim = host.window.targets["claim"]
    claim.unlink()
    write(claim, "not a window key")

    report = retire.retire_window(probe.ServiceWindow(later))

    assert report["retired"] is False
    assert report["proofs"]["shape"] == "unpublished"
    assert report["refusals"] == ["the kubelet.service claim cannot be read"]
    assert claim.read_text() == "not a window key", "the claim is left as found"
    assert probe.ServiceWindow(later).directory.exists(), "the window is kept"


def test_a_window_directory_entry_that_is_not_a_file_is_refused(host: Host) -> None:
    restored_by_deadline(host)
    (host.window.directory / ".state-orphan").mkdir(mode=0o700)

    report = retire_bound(host)

    assert report["retired"] is False
    assert report["refusals"] == [
        "the window directory entry .state-orphan is not a file"
    ]


# --------------------------------------------------------------------------- #
# the other windows of the service
# --------------------------------------------------------------------------- #
def _file_entry() -> None:
    write(probe.ROOT / OTHER, "not a window")


def _symlink_entry() -> None:
    (probe.ROOT / OTHER).symlink_to(probe.ROOT / "elsewhere")


def _resources_without_journal() -> None:
    write(other_window() / "helper.py", "a helper without its journal")


def _shared_journal() -> None:
    write(other_window() / "state.json", "{}").chmod(0o644)


def _torn_journal() -> None:
    write(other_window() / "state.json", "{not json")


@pytest.mark.parametrize(
    ("neighbour", "refusal"),
    [
        (_file_entry, f"window entry {OTHER[:32]} is not a directory"),
        (_symlink_entry, f"window entry {OTHER[:32]} is not a directory"),
        (_resources_without_journal, f"window {OTHER[:32]} has no journal"),
        (_shared_journal, f"window {OTHER[:32]} cannot be read"),
        (_torn_journal, f"window {OTHER[:32]} cannot be read"),
    ],
    ids=["file", "symlink", "no-journal", "not-private", "not-json"],
)
def test_another_window_that_cannot_be_proven_inert_refuses(
    host: Host, neighbour: Any, refusal: str
) -> None:
    restored_by_deadline(host)
    neighbour()

    report = retire_bound(host)

    assert report["retired"] is False
    assert report["refusals"] == [refusal]
    assert all(path.exists() for path in window_files(host)), "nothing removed"


def test_an_empty_neighbour_directory_is_noted_as_inert(host: Host) -> None:
    restored_by_deadline(host)
    neighbour = other_window()

    report = retire_bound(host)

    assert report["retired"] is True, report["refusals"]
    assert report["proofs"]["inert_windows"] == [f"window {OTHER[:32]} has no journal"]
    assert neighbour.is_dir(), "another window's directory is never removed"


@pytest.mark.parametrize(
    "record",
    [
        {"binding": {"service": "gpu-fault-node-agent.service"}, "installed": True},
        {"binding": "not a binding", "phase": "INSTALLED", "installed": True},
        {
            "binding": {"service": "kubelet.service"},
            "phase": "CLOSED",
            "installed": True,
        },
    ],
    ids=["other-service", "no-binding", "closed"],
)
def test_a_window_of_another_service_or_a_closed_one_is_ignored(
    host: Host, record: dict[str, Any]
) -> None:
    restored_by_deadline(host)
    neighbour = other_window(record)

    report = retire_bound(host)

    assert report["retired"] is True, report["refusals"]
    assert report["proofs"]["inert_windows"] == [], "neither live nor inert"
    assert (neighbour / "state.json").is_file(), "its journal is untouched"


def test_a_drop_in_directory_removed_by_hand_does_not_block_retirement(
    host: Host,
) -> None:
    restored_by_deadline(host)
    dropin = host.window.dropin
    dropin.unlink()
    dropin.parent.rmdir()

    report = retire_bound(host)

    assert report["retired"] is True, report["refusals"]
    assert str(dropin) not in report["removed"], "an absent drop-in is not removed"
    assert not host.window.directory.exists(), "the rest of the window is retired"


def test_a_claim_removed_by_hand_does_not_block_retirement(host: Host) -> None:
    restored_by_deadline(host)
    claim = host.window.targets["claim"]
    claim.unlink()

    report = retire_bound(host)

    assert report["retired"] is True, report["refusals"]
    assert str(claim) not in report["removed"], report["removed"]
    assert str(host.window.dropin) in report["removed"], "the drop-in still goes"


def test_a_stale_script_path_that_is_a_symlink_is_never_followed(host: Host) -> None:
    target = retire.RETIRE_RUN / Path(STALE).name
    decoy = write(retire.RETIRE_RUN / "decoy", "someone else's file")
    target.symlink_to(decoy)

    report = retire.retire_script(STALE, "a" * 64)

    assert report == {
        "retired": False,
        "refusals": [f"{STALE} is not a regular file of this user"],
        "proofs": {"shape": "unbound", "stale_script": STALE},
        "removed": [],
    }
    assert target.is_symlink() and decoy.exists(), "the link and its target stay"


# --------------------------------------------------------------------------- #
# failures inside the retirement leave the CLOSING marker
# --------------------------------------------------------------------------- #
def test_a_unit_still_loaded_after_the_reload_stops_and_resumes(host: Host) -> None:
    restored_by_deadline(host)
    unit = host.window.units["stop-timer"]
    host.overrides[unit] = {"LoadState": "loaded"}

    with pytest.raises(probe.ProbeError, match="a retired window unit is still loaded"):
        retire_bound(host)

    record = host.read()
    assert record["phase"] == "CLOSING" and "started_at" in record["retirement"]
    assert host.window.targets["claim"].exists(), "the claim waits for the proof"
    host.overrides.pop(unit)

    report = retire_bound(host)

    assert report["retired"] is True, report["refusals"]
    assert report["proofs"]["shape"] == "restored", "the marker resumes it"
    assert not host.window.directory.exists(), "the resumed retirement finished"


def test_a_service_that_changes_during_retirement_stops_it(
    host: Host, monkeypatch: pytest.MonkeyPatch
) -> None:
    restored_by_deadline(host)
    real = probe.systemctl

    def reload_restarts_kubelet(*args: str) -> dict[str, str]:
        value = real(*args)
        if args == ("daemon-reload",):
            host.units["kubelet.service"]["InvocationID"] = "9" * 32
        return value

    monkeypatch.setattr(probe, "systemctl", reload_restarts_kubelet)

    with pytest.raises(probe.ProbeError, match="the service changed while its window"):
        retire_bound(host)

    assert host.read()["phase"] == "CLOSING", "the journal records the attempt"
    assert host.window.targets["claim"].exists(), "the claim is kept"


def test_without_journal_or_directory_only_the_script_copy_goes(host: Host) -> None:
    script, _digest = stale_script()
    window = probe.ServiceWindow(host.binding)

    removed = retire.retire_locked(window, "absent", script)

    assert removed == [STALE]
    assert not (retire.RETIRE_RUN / Path(STALE).name).exists(), "the copy is gone"
    assert not window.directory.exists(), "no directory is created"
    assert host.mutations() == [], "no systemd call"


# --------------------------------------------------------------------------- #
# the command line the host runs
# --------------------------------------------------------------------------- #
def test_the_command_line_emits_one_report_per_call(
    host: Host, capsys: pytest.CaptureFixture[str]
) -> None:
    restored_by_deadline(host)

    code = retire.retire_main(["retire", "--binding", json.dumps(host.binding)])

    report = json.loads(capsys.readouterr().out)
    assert code == 0, report
    assert report["retired"] is True and report["proofs"]["shape"] == "restored"
    assert report["binding_sha256"] == host.window.key
    script, digest = stale_script()

    code = retire.retire_main(
        ["retire-script", "--stale-script", script, "--stale-script-sha256", digest]
    )

    assert code == 0
    assert json.loads(capsys.readouterr().out) == {
        "retired": True,
        "refusals": [],
        "proofs": {"shape": "unbound", "stale_script": STALE},
        "removed": [STALE],
    }


@pytest.mark.parametrize(
    ("argv", "error_class", "message"),
    [
        (
            ["retire", "--binding", "{}"],
            "ProbeError",
            "durable service binding is incomplete",
        ),
        (
            ["retire"],
            "ProbeError",
            "durable binding required; service/run names are not ownership",
        ),
        (["retire", "--binding", "not json"], "JSONDecodeError", "JSONDecodeError"),
    ],
    ids=["incomplete-binding", "no-binding", "not-json"],
)
def test_a_retirement_that_cannot_run_is_one_error_document(
    host: Host,
    capsys: pytest.CaptureFixture[str],
    argv: list[str],
    error_class: str,
    message: str,
) -> None:
    code = retire.retire_main(argv)

    assert code == 1
    assert json.loads(capsys.readouterr().out) == {
        "error": error_class,
        "error_kind": "probe",
        "error_class": error_class,
        "error_reason": "retire_unfinished",
        # Only the probe's own errors carry their message to the controller.
        "error_message": message,
        "recovery_required": True,
    }
    assert host.mutations() == [], "nothing ran"

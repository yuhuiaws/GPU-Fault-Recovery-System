"""The host half of retiring a stale service window acts only on proof.

Live shape (plain EKS, 2026-10-27/28): the 030 window stopped kubelet, its own
restore unit started kubelet again at ``restore_at`` (RESTORED, restored by the
deadline), its undo never closed it, and the R13h node Agent wave then rewrote
``node-agent.env`` and ``/opt/gpu-fault/current`` -- fingerprints the window's
own cleanup compares, so it could never close itself. These tests run
``probes/warm_spare_retire.py`` on the fake systemd host of the service-window
tests: the live shape retires (kubelet untouched), every failed proof is a
named refusal with nothing changed, an interrupted retirement resumes, the
HyperPod cascade's never-published window retires beside its claim holder, and
the combined retire probe runs as one stdlib-only script.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional.probes import warm_spare_node_probe as probe
from scripts.e2e.regional.probes import warm_spare_retire as retire
from scripts.e2e.regional.service_window_retire import write_retire_probe
from tests.regional._destr008_service_window import Host, write

STALE = "/run/gpu-fault-host-probe-aababed795.py"


@pytest.fixture
def host(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Host:
    monkeypatch.setenv("INVOCATION_ID", "a" * 32)
    value = Host(tmp_path, monkeypatch)
    run = tmp_path / "host-run"
    run.mkdir(mode=0o700)
    monkeypatch.setattr(retire, "RETIRE_RUN", run)
    return value


def restored_by_deadline(host: Host) -> None:
    host.stopped()
    host.now = host.binding["restore_at"]
    host.elapsed = host.read()["clock"]["restore"]
    assert host.tick()["restore_reason"] == "deadline"
    # The independent watcher returns once RESTORED and its unit goes inactive.
    host.units[host.window.units["restore-service"]].update(
        ActiveState="inactive", SubState="dead", MainPID="0", ControlPID="0", Job="0"
    )


def agent_wave(host: Host) -> None:
    """A later release rewrites the Agent's env and runtime pointer."""

    env = probe.AGENT_ENV
    env.write_text(env.read_text() + "\nGPU_FAULT_NODE_RELEASE=next\n")
    marker = probe.CURRENT.parent / "current-marker"
    marker.write_text("the next release's runtime")


def stale_script(content: str = "the earlier probe's script\n") -> tuple[str, str]:
    path = write(retire.RETIRE_RUN / Path(STALE).name, content)
    return STALE, hashlib.sha256(path.read_bytes()).hexdigest()


def window_files(host: Host) -> list[Path]:
    window = host.window
    return [
        window.dropin,
        *(probe.SYSTEMD / name for name in window.units.values()),
        window.targets["claim"],
        window.state_path,
    ]


def test_the_live_shape_is_retired_and_kubelet_is_never_touched(host: Host) -> None:
    restored_by_deadline(host)
    agent_wave(host)
    with pytest.raises(probe.ProbeError, match="replaced"):
        probe.ServiceWindow(host.binding).cleanup()  # why it stayed stale live
    script, digest = stale_script()
    identity = ("InvocationID", "MainPID", "ActiveState", "SubState")
    kubelet = {key: host.units["kubelet.service"][key] for key in identity}
    before = len(host.calls)

    report = retire.retire_window(probe.ServiceWindow(host.binding), script, digest)

    assert report["retired"] is True and report["refusals"] == [], report
    proofs = report["proofs"]
    assert proofs["shape"] == "restored" and proofs["restored_by"] == "deadline"
    assert (
        proofs["restored_invocation"] == "2" * 32 == proofs["service"]["InvocationID"]
    )
    assert proofs["inert_windows"] == [] and proofs["same_boot"] is True
    assert not any(path.exists() for path in window_files(host)), report["removed"]
    assert not host.window.directory.exists(), "the window directory goes last"
    assert not (retire.RETIRE_RUN / Path(STALE).name).exists(), "script copy removed"
    for path in window_files(host)[:-1]:
        assert str(path) in report["removed"], (path, report["removed"])
    mutations = [call for call in host.calls[before:] if call[0] != "show"]
    assert mutations == [("daemon-reload",)], "only a reload, never a start or stop"
    assert {key: host.units["kubelet.service"][key] for key in identity} == kubelet, (
        "kubelet runs the same invocation it ran before"
    )
    assert probe.AGENT_ENV.exists(), "nothing outside the window is touched"


def test_a_second_retirement_finds_nothing_and_changes_nothing(host: Host) -> None:
    restored_by_deadline(host)
    first = retire.retire_window(probe.ServiceWindow(host.binding))
    assert first["retired"] is True, first
    before = len(host.calls)

    again = retire.retire_window(probe.ServiceWindow(host.binding))

    assert again["retired"] is True and again["proofs"]["shape"] == "absent", again
    assert again["removed"] == [], "the lock's own empty directory is not news"
    assert not host.window.directory.exists(), "nothing was left behind"
    assert [c for c in host.calls[before:] if c[0] != "show"] == []


def _restarted(host: Host) -> None:
    host.units["kubelet.service"]["InvocationID"] = "9" * 32


def _stopped(host: Host) -> None:
    host.units["kubelet.service"].update(
        ActiveState="inactive", SubState="dead", MainPID="0"
    )


def _active_watcher(host: Host) -> None:
    host.units[host.window.units["restore-service"]].update(
        ActiveState="active", SubState="running", MainPID="321"
    )


def _replaced_dropin(host: Host) -> None:
    dropin = host.window.dropin
    dropin.unlink()
    write(dropin, "[Service]\nTimeoutStopSec=1s\n")


def _live_window(host: Host) -> None:
    other = probe.ROOT / ("e" * 64)
    other.mkdir(mode=0o700)
    record = host.read()
    record.update(phase="INSTALLED", installed=True)
    record["binding"] = {**record["binding"], "owner": "e" * 32}
    probe.write_record(other / "state.json", record)


def _foreign_dropin(host: Host) -> None:
    write(host.window.dropin.parent / f"90-gpu-fault-destr008-{'f' * 32}.conf", "x")


def _foreign_file(host: Host) -> None:
    write(host.window.directory / "notes.txt", "not the window's")


def _other_machine(host: Host) -> None:
    probe.MACHINE_ID.write_text("machine-two")


def _foreign_claim(host: Host) -> None:
    claim = host.window.targets["claim"]
    claim.unlink()
    write(claim, "f" * 64)


@pytest.mark.parametrize(
    ("defect", "needle"),
    [
        (_restarted, "runs invocation '99999999"),
        (_stopped, "kubelet.service is not running (inactive/dead)"),
        (_active_watcher, "-restore.service is active/running, not inactive"),
        (_replaced_dropin, ".conf is not this window's own publication"),
        (_live_window, f"another window {'e' * 32} (INSTALLED) still holds"),
        (_foreign_dropin, "another window's drop-in"),
        (_foreign_file, "the window directory holds notes.txt"),
        (_other_machine, "this is not the machine the window was bound on"),
        (_foreign_claim, f"claim names window {'f' * 32}, not this one"),
    ],
)
def test_a_failed_proof_is_named_and_nothing_changes(
    host: Host, defect: Any, needle: str
) -> None:
    restored_by_deadline(host)
    defect(host)
    present = [path for path in window_files(host) if path.exists()]
    before = len(host.calls)

    report = retire.retire_window(probe.ServiceWindow(host.binding))

    assert report["retired"] is False, report
    assert any(needle in reason for reason in report["refusals"]), report["refusals"]
    assert [path for path in window_files(host) if path.exists()] == present
    assert host.read()["phase"] == "RESTORED", "the journal is left as found"
    assert [c for c in host.calls[before:] if c[0] != "show"] == []


def test_a_window_that_did_not_restore_is_never_retired(host: Host) -> None:
    host.arm()
    host.window.schedule_stop()  # SCHEDULED: the stop has not fired yet

    report = retire.retire_window(probe.ServiceWindow(host.binding))

    assert report["retired"] is False
    assert report["refusals"][0].startswith("the window is SCHEDULED: only one"), (
        report["refusals"]
    )
    assert host.window.targets["claim"].exists(), "the claim is kept"


def test_a_stale_script_copy_with_another_digest_is_refused(host: Host) -> None:
    restored_by_deadline(host)
    script, _digest = stale_script()

    report = retire.retire_window(probe.ServiceWindow(host.binding), script, "0" * 64)

    assert report["retired"] is False
    assert report["refusals"] == [
        f"{STALE} is not the earlier probe's script (digest differs)"
    ]
    assert (retire.RETIRE_RUN / Path(STALE).name).exists(), "the copy is kept"


def test_an_interrupted_retirement_resumes_from_its_marker(
    host: Host, monkeypatch: pytest.MonkeyPatch
) -> None:
    restored_by_deadline(host)
    real = probe.systemctl
    calls = {"reload": 0}

    def lost_reload(*args: str) -> dict[str, str]:
        if args == ("daemon-reload",) and not calls["reload"]:
            calls["reload"] += 1
            raise probe.ProbeError("service window systemctl failed; output withheld")
        return real(*args)

    monkeypatch.setattr(probe, "systemctl", lost_reload)
    with pytest.raises(probe.ProbeError, match="systemctl failed"):
        retire.retire_window(probe.ServiceWindow(host.binding))
    record = host.read()
    assert record["phase"] == "CLOSING" and "started_at" in record["retirement"]

    report = retire.retire_window(probe.ServiceWindow(host.binding))

    assert report["retired"] is True and report["proofs"]["shape"] == "restored"
    assert not host.window.directory.exists(), "the resumed retirement finished"
    assert not host.window.targets["claim"].exists(), "the claim went last"


def test_the_hyperpod_cascade_retires_the_holder_then_the_inert_window(
    host: Host,
) -> None:
    """2026-10-24/25: window A (the claim holder) was left RESTORED; window B's
    prepare met A's claim after writing its own PREPARING journal."""

    restored_by_deadline(host)
    holder = dict(host.binding)
    later = {**host.make_binding(), "owner": "d" * 32}
    inert = probe.ServiceWindow(later)
    with pytest.raises(probe.ProbeError, match="no owner receipt"):
        inert.prepare()
    assert probe.private_read(inert.state_path)["phase"] == "PREPARING"

    first = retire.retire_window(probe.ServiceWindow(holder))
    assert first["retired"] is True, first["refusals"]
    assert first["proofs"]["inert_windows"] == [
        f"{inert.key[:32]} (PREPARING, published nothing)"
    ]
    second = retire.retire_window(probe.ServiceWindow(later))
    assert second["retired"] is True and second["proofs"]["shape"] == "unpublished"
    assert (
        not inert.directory.exists()
        and not probe.ServiceWindow(holder).directory.exists()
    )


def test_the_never_published_window_retires_beside_a_live_holder(host: Host) -> None:
    restored_by_deadline(host)
    later = {**host.make_binding(), "owner": "d" * 32}
    with pytest.raises(probe.ProbeError, match="no owner receipt"):
        probe.ServiceWindow(later).prepare()

    report = retire.retire_window(probe.ServiceWindow(later))

    assert report["retired"] is True, report["refusals"]
    assert host.window.targets["claim"].exists(), "the holder's claim is untouched"
    assert host.read()["phase"] == "RESTORED"


def test_an_unbound_journal_retires_only_its_script_copy(host: Host) -> None:
    script, digest = stale_script()
    assert retire.retire_script(script, "0" * 64)["retired"] is False
    report = retire.retire_script(script, digest)
    assert report == {
        "retired": True,
        "refusals": [],
        "proofs": {"shape": "unbound", "stale_script": STALE},
        "removed": [STALE],
    }
    assert retire.retire_script(script, digest)["removed"] == [], "idempotent"
    refused = retire.retire_script("/etc/passwd", digest)
    assert refused["refusals"] == [
        "/etc/passwd is not a host-probe script copy under /run"
    ]


def test_the_retire_probe_runs_as_one_stdlib_script(tmp_path: Path) -> None:
    path = write_retire_probe(tmp_path / "retire")
    source = path.read_text(encoding="utf-8")
    assert (
        source.count("def retire_window(") == 1
        and source.count("class ServiceWindow") == 1
    )
    assert "from scripts" not in source, "nothing outside the stdlib is imported"
    assert path.stat().st_mode & 0o077 == 0
    assert write_retire_probe(tmp_path / "retire") == path, "idempotent"

    def run(*arguments: str) -> tuple[int, dict[str, Any]]:
        completed = subprocess.run(
            [sys.executable, "-I", str(path), *arguments],
            capture_output=True,
            text=True,
            timeout=60,
            cwd=tmp_path,
            check=False,
        )
        return completed.returncode, json.loads(
            completed.stdout.strip().splitlines()[-1]
        )

    code, report = run("retire-script")
    assert code == 0 and report["retired"] is True, report
    code, report = run("retire", "--binding", "{}")
    assert code == 1 and report["error_reason"] == "retire_unfinished", report
    assert report["error_message"] == "durable service binding is incomplete"

"""Window contract tests of the host-collector source replay probe (continued).

The ``FakeHost`` of ``test_general_live_source_probe`` models systemd's
environment assembly (``Environment=``, then every ``EnvironmentFile=`` in order,
``UnsetEnvironment=`` last), the drop-ins it has loaded (``DropInPaths`` until a
``daemon-reload``) and the collector's ``WATCHDOG=1`` stamps; these tests cover
the env window through the run's ``EnvironmentFile=`` (the live 2026-10-26
failure of a drop-in ``Environment=`` against the unit's own file), the close's
reload re-check and read-back of the real configuration, and ``await-ticks``.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from scripts.e2e.regional.probes import general_live_source_probe as probe
from tests.regional.test_general_live_source_probe import (
    RUN,
    UNIT,
    FakeHost,
    _install,
    _main,
    make_host,
)


@pytest.fixture
def host(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> FakeHost:
    return make_host(tmp_path, monkeypatch)


def test_env_override_wins_over_the_units_own_environment_file(
    host: FakeHost, capsys: pytest.CaptureFixture[str]
) -> None:
    """The live 2026-10-26 failure: the installer writes
    GPU_FAULT_EXPECTED_GPU_COUNT="8" into the unit's EnvironmentFile and a drop-in
    Environment= loses to it (systemd applies files after Environment=). The
    window hands the override over as a later EnvironmentFile=, which wins."""

    # The old shape, replayed on the model: the drop-in's Environment= loses.
    dropin = probe.dropin_path(RUN)
    dropin.parent.mkdir(parents=True)
    dropin.write_text("[Service]\nEnvironment=GPU_FAULT_EXPECTED_GPU_COUNT=9\n")
    host.apply_dropin()
    assert host.environment["GPU_FAULT_EXPECTED_GPU_COUNT"] == "8", "the unit file wins"
    dropin.unlink()
    host.apply_dropin()
    code, opened = _main(
        "open-window",
        "--run-id",
        RUN,
        "--mode",
        "mixed",
        "--env",
        "GPU_FAULT_EXPECTED_GPU_COUNT=9",
        capsys=capsys,
    )
    assert code == 0, opened
    assert host.environment["GPU_FAULT_EXPECTED_GPU_COUNT"] == "9"
    lines = dropin.read_text().splitlines()
    assert lines == ["[Service]", f"EnvironmentFile={probe.env_file(RUN)}"], lines
    assert not any(line.startswith("Environment=") for line in lines), (
        "the override never travels as Environment= (the unit file would win)"
    )
    check = next(c for c in opened["verification"]["checks"] if c["kind"] == "env")
    assert check["unit_value"] == "9" and check["ok"] is True
    assert opened["baseline_env"] == {"GPU_FAULT_EXPECTED_GPU_COUNT": "8"}
    code, closed = _main("close-window", "--run-id", RUN, capsys=capsys)
    assert code == 0 and closed["clean"] is True and closed["reloads"] == 1
    assert host.environment == {"GPU_FAULT_EXPECTED_GPU_COUNT": "8"}
    assert closed["restore"] == {
        "checks": [
            {
                "kind": "env",
                "dest": "GPU_FAULT_EXPECTED_GPU_COUNT",
                "baseline": "8",
                "unit_value": "8",
                "ok": True,
            }
        ],
        "ok": True,
    }
    assert closed["residue"]["dropin_loaded"] is False
    # An empty override is an empty string, as the file form keeps it.
    code, opened = _main(
        "open-window",
        "--run-id",
        RUN,
        "--mode",
        "mixed",
        "--env",
        "GPU_FAULT_EXPECTED_GPU_COUNT=",
        capsys=capsys,
    )
    assert code == 0 and host.environment["GPU_FAULT_EXPECTED_GPU_COUNT"] == "", opened
    code, closed = _main("close-window", "--run-id", RUN, capsys=capsys)
    assert code == 0 and closed["restore"]["ok"] is True


def test_close_reloads_again_while_systemd_still_lists_the_dropin(
    host: FakeHost, capsys: pytest.CaptureFixture[str]
) -> None:
    """After the live 013 close systemd still listed the removed drop-in in
    DropInPaths: the close reads the listing back after its reload and reloads
    once more, recording both, so the unit restarts on the real configuration."""

    _install("stat", b"cpu  1 0 1 1 0 0 0 0 0 0\n", capsys)
    code, _opened = _main(
        "open-window",
        "--run-id",
        RUN,
        "--mode",
        "bind",
        "--bind",
        "stat:/proc/stat",
        capsys=capsys,
    )
    assert code == 0
    assert host.loaded_dropins[UNIT] == [str(probe.dropin_path(RUN))]
    host.stale_reloads = 1
    code, closed = _main("close-window", "--run-id", RUN, capsys=capsys)
    assert code == 0 and closed["clean"] is True, closed
    assert closed["reloads"] == 2 and closed["residue"]["dropin_loaded"] is False
    assert host.loaded_dropins[UNIT] == [] and host.binds == {}
    assert closed["restore"]["checks"] == [
        {"kind": "source", "dest": "/proc/stat", "unit_readable": True, "ok": True},
        {
            "kind": "env",
            "dest": "GPU_FAULT_EXPECTED_GPU_COUNT",
            "baseline": "8",
            "unit_value": "8",
            "ok": True,
        },
    ], "the real file on PID 1's device again; the unit's own count untouched"


def test_await_ticks_closes_after_the_collectors_required_ticks(
    host: FakeHost, capsys: pytest.CaptureFixture[str]
) -> None:
    """The exposure follows the ticks the collector took: WATCHDOG=1 pings a
    contributor apart form one tick, a whole interval of silence separates
    ticks. The window closes once the second tick ended, before a third."""

    _install("stat", b"cpu  1 0 1 1 0 0 0 0 0 0\n", capsys)
    code, _opened = _main(
        "open-window",
        "--run-id",
        RUN,
        "--mode",
        "bind",
        "--bind",
        "stat:/proc/stat",
        "--restore-seconds",
        "105",
        capsys=capsys,
    )
    assert code == 0
    # The startup phase (13 s, as live) then two ticks of three pings, 15 s apart.
    host.pings = [13.0, 13.4, 14.1, 29.1, 29.5, 30.3, 45.3, 45.9, 46.2]
    code, waited = _main(
        "await-ticks",
        "--run-id",
        RUN,
        "--count",
        "2",
        "--interval",
        "15",
        "--timeout",
        "105",
        "--close",
        capsys=capsys,
    )
    assert code == 0, waited
    assert waited["completed"] is True and waited["count"] == 2
    assert waited["gap_seconds"] == 6.0, "0.4 x 15 s, at least 5 s"
    assert len(waited["ticks"]) == 2 and all(
        2 <= tick["pings_seen"] <= 3 for tick in waited["ticks"]
    ), "a 0.5 s poll may skip a ping inside a tick, never a tick"
    assert 36.0 <= waited["waited_seconds"] <= 37.5, (
        "the second tick ended at 30.3 s; the gap passed in silence at 36.3 s, "
        "before the third tick at 45.3 s"
    )
    assert waited["close"]["clean"] is True and waited["close"]["closed_at"]
    assert not probe.dropin_path(RUN).exists() and host.binds == {} and not host.timers
    # No ticks within the bound: closed anyway, saying so.
    _install("stat", b"cpu  1 0 1 1 0 0 0 0 0 0\n", capsys)
    code, _opened = _main(
        "open-window",
        "--run-id",
        RUN,
        "--mode",
        "bind",
        "--bind",
        "stat:/proc/stat",
        "--restore-seconds",
        "40",
        capsys=capsys,
    )
    assert code == 0
    host.pings = []
    code, waited = _main(
        "await-ticks",
        "--run-id",
        RUN,
        "--count",
        "2",
        "--interval",
        "15",
        "--timeout",
        "400",
        "--close",
        capsys=capsys,
    )
    assert code == 0 and waited["completed"] is False and waited["ticks"] == []
    assert waited["waited_seconds"] <= 40.5, "the wait never outlasts the deadman"
    assert waited["close"]["clean"] is True and not probe.dropin_path(RUN).exists()
    # Nothing open: the deadman got there first.
    code, waited = _main("await-ticks", "--run-id", RUN, "--close", capsys=capsys)
    assert code == 0 and waited == {
        "run_id": RUN,
        "already_closed": True,
        "completed": False,
        "ticks": [],
    }


def test_await_ticks_skips_the_units_start_post_watchdog_stamp(
    host: FakeHost, capsys: pytest.CaptureFixture[str]
) -> None:
    """The live eks 013 ``link_inactive`` run of 2026-10-26 (18:59Z, R13g)
    replayed: systemd stamps a ``Type=notify`` unit's watchdog when it enters
    start-post on READY=1 -- 30 us before ``ActiveEnterTimestampMonotonic``,
    before the collector's startup phase (systemd 255, reproduced on a transient
    unit) -- and the restarted collector's READY came 1.9 s in; its first real
    tick read the port 12.6 s in and pinged until 13.0 s; the probe's loop
    began 11.1 s in (the exec round trip). Counting every stamp since the main
    process started took the READY stamp for tick 1 and closed ~19-21 s in,
    after one real read (one shortfall reading inside, no finding: the second
    confirming read never happened). Only stamps after the activation are
    ticks: the close follows the second real tick (27.6-28.0 s) and precedes
    the third (42.6 s); the receipt names when the unit became active."""

    _install("stat", b"cpu  1 0 1 1 0 0 0 0 0 0\n", capsys)
    code, _opened = _main(
        "open-window",
        "--run-id",
        RUN,
        "--mode",
        "bind",
        "--bind",
        "stat:/proc/stat",
        "--restore-seconds",
        "105",
        capsys=capsys,
    )
    assert code == 0
    host.ready_offset = 1.9
    host.pings = [12.6, 12.8, 13.0, 27.6, 27.8, 28.0, 42.6, 42.8, 43.0]
    host.pause(11.1)  # the exec round trip: the loop starts before the first tick
    assert host.watchdog_stamp() == host.active_enter_us() - 30, (
        "what the unit shows when the loop starts is the READY stamp, no ping yet"
    )
    code, waited = _main(
        "await-ticks",
        "--run-id",
        RUN,
        "--count",
        "2",
        "--interval",
        "15",
        "--timeout",
        "105",
        "--close",
        capsys=capsys,
    )
    assert code == 0, waited
    assert waited["completed"] is True and len(waited["ticks"]) == 2, waited
    closed = 11.1 + waited["waited_seconds"]
    assert 34.0 <= closed < 42.6, (
        f"closed {closed} s after the restart: the second real tick ended at 28.0 s "
        "and its gap passed in silence at 34.0 s, before the third tick at 42.6 s "
        "(the live close after the READY stamp and one real tick came 19-21 s in)"
    )
    assert 22.5 <= waited["waited_seconds"] <= 24.0, waited
    ticks = waited["ticks"]
    assert all(1 <= tick["pings_seen"] <= 3 for tick in ticks), ticks
    active = datetime.fromisoformat(waited["unit_active_at"])
    first = datetime.fromisoformat(ticks[0]["first_ping_at"])
    second = datetime.fromisoformat(ticks[1]["first_ping_at"])
    assert 10.5 <= (first - active).total_seconds() <= 11.0, (
        "the first tick is the collector's first read, 12.6 s in, not the READY "
        "stamp 1.9 s in"
    )
    assert 14.5 <= (second - first).total_seconds() <= 15.5, "a whole interval apart"
    assert waited["close"]["clean"] is True and not probe.dropin_path(RUN).exists()

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

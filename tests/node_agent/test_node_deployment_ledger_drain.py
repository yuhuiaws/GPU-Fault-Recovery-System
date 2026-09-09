"""The installer drains the node action ledger before it touches the Agent.

Runs the installer's own drain block: an IN_PROGRESS row blocks the restart,
an unreadable ledger (corrupt, tableless or SQLITE_BUSY) counts as in
flight, the wait loop really waits for a settling command, the readers
carry a busy timeout, and the drain calls are ordered before the quiesce
restore, the first unit write, the stop loop and the restart. The Job
budget arithmetic lives in ``test_node_installer_drain_budget.py``.
"""

from __future__ import annotations

import re
import shlex
import sqlite3
import subprocess
from pathlib import Path

from gpu_fault.node_agent.ledger import IN_PROGRESS_STATE
from tests.node_agent._deployment_support import (
    NODE_SCRIPTS,
    _ledger_drain_probe,
    _ledger_with_state,
    _write_stub,
)


def test_installer_refuses_to_restart_the_agent_mid_operation(tmp_path: Path) -> None:
    probe = _ledger_drain_probe(tmp_path)
    database = tmp_path / "node-actions.db"
    _ledger_with_state(database, IN_PROGRESS_STATE)

    def drain(
        sqlite_command: str, db: Path, *, timeout: int = 0, poll: int = 1
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["bash", str(probe), str(db), sqlite_command, str(timeout), str(poll)],
            capture_output=True,
            text=True,
            check=False,
            env={"PATH": "/usr/bin:/bin"},
        )

    blocked = drain("", database)
    assert blocked.returncode != 0, (
        f"an IN_PROGRESS row must block the restart: {blocked.stdout}{blocked.stderr}"
    )
    assert "cmd-in-flight" in blocked.stdout, blocked.stdout

    binaries = tmp_path / "bin"
    binaries.mkdir()
    _write_stub(binaries, "sqlite3", "printf 'cmd-from-sqlite3\\n'\n")
    via_sqlite3 = drain(str(binaries / "sqlite3"), database)
    assert via_sqlite3.returncode != 0, via_sqlite3.stdout + via_sqlite3.stderr
    assert "cmd-from-sqlite3" in via_sqlite3.stdout, via_sqlite3.stdout

    finished = tmp_path / "finished.db"
    _ledger_with_state(finished, "COMPLETED")
    assert "DRAINED" in drain("", finished).stdout, "a settled ledger must not wait"
    assert "DRAINED" in drain("", tmp_path / "absent.db").stdout, (
        "a first install has no ledger and must not wait"
    )


def test_installer_drains_the_ledger_before_it_touches_the_agent_unit() -> None:
    installer = NODE_SCRIPTS[0].read_text()
    assert 'NODE_AGENT_STOP_TIMEOUT_SECONDS="1900"' in installer, (
        "the installer must wait as long as the unit's TimeoutStopSec before dying"
    )
    # The reserve-bounded drain runs before the quiesce restore (an IN_PROGRESS
    # quiesce must not have its GPU services brought back) and the first write;
    # the budget arithmetic itself lives in test_node_installer_drain_budget.py.
    first_write = installer.index("> /etc/systemd/system/gpu-fault-gpu-persistence")
    quiesce_restore = installer.index("existing_quiesce_states=(")
    stop_loop = installer.index('if systemctl is-active --quiet "${runtime_unit}"')
    restart = installer.index("systemctl restart gpu-fault-node-agent.service")
    install = installer.index("\n    drain_node_agent_before_install\n")
    pattern = r"^ +drain_node_agent_before_restart$"
    calls = [match.start() for match in re.finditer(pattern, installer, re.M)]
    assert len(calls) == 2, f"expected drains before the stop and the restart: {calls}"
    assert install < min(quiesce_restore, first_write), "the install drain came late"
    assert calls[0] < stop_loop, "stopping the unit interrupts the same operation"
    assert calls[1] < restart, "the restart must not preempt an in-flight command"


def _drain(
    probe: Path,
    database: Path,
    *,
    sqlite_command: str = "",
    timeout: int = 0,
    poll: int = 1,
    path: str = "/usr/bin:/bin",
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(probe), str(database), sqlite_command, str(timeout), str(poll)],
        capture_output=True,
        text=True,
        check=False,
        env={"PATH": path},
    )


def test_installer_treats_an_unreadable_ledger_as_in_flight(tmp_path: Path) -> None:
    """A ledger that cannot be read must not be mistaken for an idle one.

    Both readers used to swallow every error and print nothing, which the wait
    loop scored as "no in-flight commands" -- so a corrupt database, a ledger
    whose ``results`` table has not been created yet and a ``SQLITE_BUSY`` from
    the very write the guard is meant to notice all let the installer stop the
    Agent in the middle of a node action. Read failure now keeps the installer
    waiting and finally kills the install.
    """

    probe = _ledger_drain_probe(tmp_path)

    corrupt = tmp_path / "corrupt.db"
    corrupt.write_bytes(b"this is not a SQLite database\n" * 64)
    corrupted = _drain(probe, corrupt)
    assert corrupted.returncode != 0, (
        f"a corrupt ledger must block the restart: {corrupted.stdout}"
    )
    assert "WARN  node action ledger unreadable" in corrupted.stdout, corrupted.stdout
    assert "DRAINED" not in corrupted.stdout, (
        "an unreadable ledger must never report the Agent as drained"
    )

    tableless = tmp_path / "tableless.db"
    with sqlite3.connect(tableless) as connection:
        connection.execute("CREATE TABLE unrelated (id INTEGER PRIMARY KEY)")
    without_table = _drain(probe, tableless)
    assert without_table.returncode != 0, (
        f"a ledger without the results table must block: {without_table.stdout}"
    )
    assert "DRAINED" not in without_table.stdout, without_table.stdout

    binaries = tmp_path / "bin"
    binaries.mkdir()
    argument_log = tmp_path / "sqlite3-args.txt"
    _write_stub(
        binaries,
        "sqlite3",
        f"printf '%s\\n' \"$*\" >> {shlex.quote(str(argument_log))}\n"
        f"if [[ -f {shlex.quote(str(tmp_path / 'BUSY_MARKER'))} ]];"
        " then exit 0; fi\n"
        "printf 'Error: database is locked\\n' >&2\n"
        "exit 1\n",
    )
    settled = tmp_path / "settled.db"
    _ledger_with_state(settled, "COMPLETED")
    busy = _drain(probe, settled, sqlite_command=str(binaries / "sqlite3"))
    assert busy.returncode != 0, f"a locked ledger must block: {busy.stdout}"
    assert "WARN  node action ledger unreadable" in busy.stdout, busy.stdout
    assert "DRAINED" not in busy.stdout, busy.stdout
    assert ".timeout 5000" in argument_log.read_text(encoding="utf-8"), (
        "the sqlite3 CLI has no busy timeout of its own, so the install must set one"
    )

    (tmp_path / "BUSY_MARKER").write_text("readable now\n", encoding="utf-8")
    readable = _drain(
        probe,
        settled,
        sqlite_command=str(binaries / "sqlite3"),
        path=f"{binaries}:/usr/bin:/bin",
    )
    assert "DRAINED" in readable.stdout, (
        f"a ledger that becomes readable must let the install proceed: {readable.stdout}"
    )
    assert readable.returncode == 0, readable.stdout + readable.stderr


def test_installer_waits_for_an_in_flight_command_to_settle(tmp_path: Path) -> None:
    """The wait loop must actually wait, not only die at the deadline."""

    probe = _ledger_drain_probe(tmp_path)
    binaries = tmp_path / "bin"
    binaries.mkdir()
    _write_stub(
        binaries,
        "sqlite3",
        f"marker={shlex.quote(str(tmp_path / 'polled'))}\n"
        'if [[ -f "${marker}" ]]; then exit 0; fi\n'
        'printf served > "${marker}"\n'
        "printf 'cmd-slow\\n'\n",
    )
    database = tmp_path / "node-actions.db"
    _ledger_with_state(database, IN_PROGRESS_STATE)

    result = _drain(
        probe, database, sqlite_command=str(binaries / "sqlite3"), timeout=5, poll=1
    )

    assert "waiting for in-flight node actions: cmd-slow" in result.stdout, (
        f"the operator must see why the upgrade is paused: {result.stdout}"
    )
    assert "DRAINED" in result.stdout, (
        f"the install must proceed once the command settles: {result.stdout}"
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_installer_python_ledger_reader_has_a_busy_timeout() -> None:
    installer = NODE_SCRIPTS[0].read_text()

    assert 'f"file:{sys.argv[1]}?mode=ro", uri=True, timeout=5' in installer, (
        "the python ledger reader must wait out a writer instead of erroring at once"
    )
    assert "2>/dev/null || true\n    fi\n}" not in installer, (
        "the ledger readers must not swallow their own failures"
    )

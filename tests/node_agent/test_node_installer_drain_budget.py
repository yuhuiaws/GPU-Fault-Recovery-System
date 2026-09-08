"""The node installer's ledger drain against the regional installer Job's budget.

``NODE_AGENT_STOP_TIMEOUT_SECONDS`` is 1900 s (the Agent unit's
``TimeoutStopSec``), but on the regional path the installer runs inside a
hostPID + chroot Job whose ``activeDeadlineSeconds`` is 840 s. With a field
diagnostic or firmware action in flight (up to 1800 s) the drain outlived the
Job, the kubelet SIGKILLed the whole cgroup, no EXIT trap ran, and the node was
left with new unit and env files, the old wheel and no daemon-reload. The Job
now hands its start time and budget into the chroot, the installer bounds the
wait by ``min(1900, remaining budget - margin)`` and fails with a status the
operator can act on; the Job goes Failed and the reconciler retries it with
backoff. Ordering (the drain runs before the first unit/env write) is asserted
in ``test_node_deployment.py`` next to the other drain contracts.
"""

from __future__ import annotations

import shlex
import subprocess
import time
from pathlib import Path

from gpu_fault.node_agent.ledger import IN_PROGRESS_STATE
from tests.node_agent.test_node_deployment import (
    NODE_SCRIPTS,
    _ledger_drain_probe,
    _ledger_with_state,
    _write_stub,
)

INSTALLER = NODE_SCRIPTS[0]
INSTALLER_JOB = NODE_SCRIPTS[4]


def _drain_with_budget(
    probe: Path,
    database: Path,
    *,
    started_epoch: int,
    active_deadline_seconds: int = 840,
    sqlite_command: str = "",
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(probe), str(database), sqlite_command, "1900", "1"],
        capture_output=True,
        text=True,
        check=False,
        env={
            "PATH": "/usr/bin:/bin",
            "INSTALLER_STARTED_EPOCH": str(started_epoch),
            "INSTALLER_ACTIVE_DEADLINE_SECONDS": str(active_deadline_seconds),
        },
    )


def test_installer_drain_gives_up_before_the_installer_job_deadline(
    tmp_path: Path,
) -> None:
    probe = _ledger_drain_probe(tmp_path)
    database = tmp_path / "node-actions.db"
    _ledger_with_state(database, IN_PROGRESS_STATE)
    now = int(time.time())

    # The Job started 800 s ago with an 840 s budget: the 120 s margin has
    # already been crossed, so the 1900 s drain budget must not be honoured.
    expired = _drain_with_budget(probe, database, started_epoch=now - 800)

    assert expired.returncode != 0, expired.stdout + expired.stderr
    assert "DRAINED" not in expired.stdout, expired.stdout
    assert "node agent still has an in-flight command; retry later" in expired.stdout, (
        f"the operator must be told to retry, not shown a generic failure: "
        f"{expired.stdout}"
    )
    assert "cmd-in-flight" in expired.stdout, expired.stdout


def test_a_job_with_budget_left_does_not_cut_the_drain_short(tmp_path: Path) -> None:
    """``min(1900, remaining)`` is the remaining budget; the command settles."""
    probe = _ledger_drain_probe(tmp_path)
    database = tmp_path / "node-actions.db"
    _ledger_with_state(database, IN_PROGRESS_STATE)
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

    settled = _drain_with_budget(
        probe,
        database,
        started_epoch=int(time.time()),
        sqlite_command=str(binaries / "sqlite3"),
    )

    assert "waiting for in-flight node actions: cmd-slow" in settled.stdout, (
        settled.stdout
    )
    assert "DRAINED" in settled.stdout, settled.stdout + settled.stderr
    assert settled.returncode == 0, settled.stdout + settled.stderr


RESERVE_PROBE_END = (
    "drain_node_agent_before_install() {\n"
    "    systemctl is-active --quiet gpu-fault-node-agent.service || return 0\n"
    '    wait_for_node_action_ledger_idle "${INSTALL_RESERVE_SECONDS}"\n'
    "}"
)


def _reserve_probe(target: Path) -> Path:
    """The ledger drain as the pre-write call runs it: with the install reserve."""
    installer = INSTALLER.read_text()
    start = installer.index("in_progress_node_action_ids() {")
    assert installer.count(RESERVE_PROBE_END) == 1, (
        "the pre-write drain must pass the install reserve to the ledger wait"
    )
    end = installer.index(RESERVE_PROBE_END) + len(RESERVE_PROBE_END)
    probe = target / "reserve-probe.sh"
    probe.write_text(
        "set -euo pipefail\n"
        "PYTHON_COMMAND=python3\n"
        'NODE_ACTION_DB="$1"\n'
        'SQLITE_COMMAND="$2"\n'
        'NODE_AGENT_STOP_TIMEOUT_SECONDS="1900"\n'
        'NODE_AGENT_LEDGER_POLL_SECONDS="1"\n'
        "die() { printf 'DIE: %s\\n' \"$*\"; exit 1; }\n"
        f"{installer[start:end]}\n"
        'wait_for_node_action_ledger_idle "${INSTALL_RESERVE_SECONDS}"\n'
        "printf 'DRAINED\\n'\n",
        encoding="utf-8",
    )
    return probe


def _busy_then_idle_sqlite3(tmp_path: Path) -> Path:
    """A ledger reader that reports one in-flight command on its first call only."""
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
    return binaries / "sqlite3"


def test_pre_write_drain_refuses_a_job_that_cannot_fit_the_install(
    tmp_path: Path,
) -> None:
    """Draining is not enough: the install after it needs its own budget.

    A command that ends at Job second 600-720 used to let the installer enter
    the venv build, eight unit and three env writes, stop, slot switch,
    daemon-reload, restart and verify with at most 120 s left -- and be
    SIGKILLed without the EXIT trap, the very half-upgraded node the drain was
    added to prevent. The pre-write drain is bounded by
    ``remaining - margin - INSTALL_RESERVE_SECONDS`` and dies with the retry
    status when that is already gone, even though the ledger would have gone
    idle a second later.
    """
    probe = _reserve_probe(tmp_path)
    database = tmp_path / "node-actions.db"
    _ledger_with_state(database, IN_PROGRESS_STATE)
    sqlite3 = _busy_then_idle_sqlite3(tmp_path)
    started = time.monotonic()

    # 240 s of an 840 s Job left: 240 - 120 - 420 < 0.
    refused = _drain_with_budget(
        probe,
        database,
        started_epoch=int(time.time()) - 600,
        sqlite_command=str(sqlite3),
    )

    assert refused.returncode != 0, refused.stdout + refused.stderr
    assert "DRAINED" not in refused.stdout, (
        f"the install must not start with less than the reserve left: {refused.stdout}"
    )
    assert "node agent still has an in-flight command; retry later" in refused.stdout, (
        refused.stdout
    )
    assert "cmd-slow" in refused.stdout, refused.stdout
    assert time.monotonic() - started < 30, "the refusal must not wait out the drain"


def test_an_idle_ledger_still_refuses_a_job_below_the_install_reserve(
    tmp_path: Path,
) -> None:
    probe = _reserve_probe(tmp_path)
    database = tmp_path / "node-actions.db"
    _ledger_with_state(database, "COMPLETED")

    # Nothing in flight, but only 300 s of the Job remain for a 420 s install.
    refused = _drain_with_budget(probe, database, started_epoch=int(time.time()) - 540)

    assert refused.returncode != 0, refused.stdout + refused.stderr
    assert "DRAINED" not in refused.stdout, refused.stdout
    assert "retry later" in refused.stdout and "install" in refused.stdout, (
        f"the operator must learn the Job budget, not a phantom command: "
        f"{refused.stdout}"
    )

    # A fresh Job (840 s left) proceeds, and the reserve can be lowered by env
    # for a slower or faster site with the same integer validation as its siblings.
    fresh = _drain_with_budget(probe, database, started_epoch=int(time.time()))
    assert "DRAINED" in fresh.stdout, fresh.stdout + fresh.stderr
    tuned = subprocess.run(
        ["bash", str(probe), str(database), "", "1900", "1"],
        capture_output=True,
        text=True,
        check=False,
        env={
            "PATH": "/usr/bin:/bin",
            "INSTALLER_STARTED_EPOCH": str(int(time.time()) - 540),
            "INSTALLER_ACTIVE_DEADLINE_SECONDS": "840",
            "INSTALLER_INSTALL_RESERVE_SECONDS": "60",
        },
    )
    assert "DRAINED" in tuned.stdout, tuned.stdout + tuned.stderr
    invalid = subprocess.run(
        ["bash", str(probe), str(database), "", "1900", "1"],
        capture_output=True,
        text=True,
        check=False,
        env={"PATH": "/usr/bin:/bin", "INSTALLER_INSTALL_RESERVE_SECONDS": "soon"},
    )
    assert invalid.returncode != 0, invalid.stdout
    assert "INSTALLER_INSTALL_RESERVE_SECONDS" in invalid.stdout, invalid.stdout


def test_install_reserve_is_declared_measured_and_only_bounds_the_pre_write_drain() -> (
    None
):
    installer = INSTALLER.read_text()

    assert 'INSTALL_RESERVE_SECONDS="${INSTALLER_INSTALL_RESERVE_SECONDS:-420}"' in (
        installer
    ), "the install reserve must default to the measured 420 s"
    assert "- NODE_AGENT_DRAIN_JOB_MARGIN_SECONDS - reserve" in installer, (
        "the pre-write bound is remaining - margin - reserve"
    )
    assert "job_deadline - $(date +%s) >= reserve" in installer, (
        "a drain that ends with less than the reserve left must still die"
    )
    assert (
        "drain_node_agent_before_restart() {\n"
        "    systemctl is-active --quiet gpu-fault-node-agent.service || return 0\n"
        "    wait_for_node_action_ledger_idle\n"
        "}"
    ) in installer, "the pre-stop and pre-restart drains keep the plain bound"


def test_installer_job_passes_its_deadline_into_the_chroot() -> None:
    """The installer can only bound the drain by a budget it has been told."""
    job = INSTALLER_JOB.read_text()
    installer = INSTALLER.read_text()

    assert (
        "            - name: INSTALLER_ACTIVE_DEADLINE_SECONDS\n"
        '              value: "${INSTALLER_ACTIVE_DEADLINE_SECONDS}"\n'
    ) in job, (
        "the Job must pass the same deadline it sets on spec.activeDeadlineSeconds"
    )
    started = job.index('INSTALLER_STARTED_EPOCH="\\$(date +%s)"')
    chroot = job.index(
        "chroot /host /usr/bin/env \\\n                INSTALLER_ACTIVE_DEADLINE_SECONDS="
    )
    assert started < chroot, "the start time must be taken before the chroot"
    for name in ("INSTALLER_ACTIVE_DEADLINE_SECONDS", "INSTALLER_STARTED_EPOCH"):
        assert f'{name}="\\${{{name}}}" \\' in job, (
            f"{name} must be handed into the chroot environment"
        )
        assert f"${{{name}:-}}" in installer, (
            f"the installer must read {name} and tolerate its absence "
            "(a hand-run install has no Job budget)"
        )
    assert "NODE_AGENT_DRAIN_JOB_MARGIN_SECONDS" in installer, (
        "the margin before the Job deadline must be explicit"
    )

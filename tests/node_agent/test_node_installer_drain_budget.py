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

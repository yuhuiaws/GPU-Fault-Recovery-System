"""The node holder scope probe waits for a swept holder to exit, bounded."""

from __future__ import annotations

import signal
import subprocess
import sys
from pathlib import Path

from tests._script_loader import lazy_script_module

ROOT = Path(__file__).resolve().parents[2]
PROBE = lazy_script_module(ROOT / "scripts/e2e/regional/node_holder_scope_probe.py")


def _sleeper(ignore_sigterm: bool) -> subprocess.Popen[str]:
    prelude = (
        "import signal; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        if ignore_sigterm
        else ""
    )
    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            f"{prelude}import time; print('ready', flush=True); time.sleep(60)",
        ],
        stdout=subprocess.PIPE,
        text=True,
    )
    # Only signal a child that has finished installing its handler; a SIGTERM
    # sent while the interpreter is still starting kills it whatever it meant
    # to do about the signal.
    assert process.stdout is not None
    assert process.stdout.readline().strip() == "ready"
    return process


def test_settled_waits_for_the_exit_instead_of_sampling_after_a_fixed_sleep() -> None:
    process = _sleeper(ignore_sigterm=False)
    try:
        process.send_signal(signal.SIGTERM)
        assert PROBE.settled(process, timeout=5) is True
        assert PROBE.alive(process) is False
    finally:
        PROBE.stop(process)


def test_settled_reports_a_holder_that_survived_within_the_bound() -> None:
    process = _sleeper(ignore_sigterm=True)
    try:
        process.send_signal(signal.SIGTERM)
        assert PROBE.settled(process, timeout=0.5) is False
        assert PROBE.alive(process) is True
    finally:
        PROBE.stop(process)

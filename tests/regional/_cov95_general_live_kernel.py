"""A fake host for the GF-GENERAL-LIVE kernel-log source replay probe.

``FakeSystemd`` (``_cov95_general_live_systemd``) answers the probe's
``systemctl``/``systemd-run``/``journalctl`` calls with both collector units
installed, the log collector disabled and the kernel collector running.
``install_host`` points every host root the probe writes (``/run`` state, the
``/run/systemd/system`` drop-ins, ``/proc``) at a ``tmp_path`` tree and swaps the
probe's ``subprocess``/``shutil`` for the fake, so nothing reaches the real host.
"""

from __future__ import annotations

import base64
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts.e2e.regional.probes import general_live_kernel_source_probe as probe
from tests.regional._cov95_general_live_systemd import FakeSystemd, unit_properties

RUN_ID = "gl016-cov-window-a1-deadbeef"
LOG = probe.LOG_UNIT
KERNEL = probe.KERNEL_UNIT
#: Where ``which`` must look for the real journalctl the wrapper forwards to.
SYSTEM_PATH = "/usr/bin:/usr/local/bin:/bin:/sbin"
REAL_LINE = {"MESSAGE": "real kernel line", "__CURSOR": "s=real-1"}


def install_host(monkeypatch: pytest.MonkeyPatch, root: Path) -> SimpleNamespace:
    """Point the probe at ``root`` and ``FakeSystemd``; return the handles."""

    systemd = FakeSystemd(
        {
            LOG: unit_properties(),
            KERNEL: unit_properties(
                active="active", file_state="enabled", pid="4242", invocation="a" * 32
            ),
        }
    )
    journalctl = root / "usr" / "bin" / "journalctl"
    journalctl.parent.mkdir(parents=True)
    journalctl.write_text(
        f"#!/bin/sh\nprintf '%s\\n' '{json.dumps(REAL_LINE)}'\n", encoding="utf-8"
    )
    journalctl.chmod(0o755)
    found: dict[str, str] = {"journalctl": str(journalctl)}
    lookups: list[tuple[str, str | None]] = []

    def which(name: str, path: str | None = None) -> str | None:
        lookups.append((name, path))
        return found.get(name)

    monkeypatch.setattr(
        probe, "subprocess", SimpleNamespace(run=systemd.run, PIPE=subprocess.PIPE)
    )
    monkeypatch.setattr(
        probe,
        "shutil",
        SimpleNamespace(which=which, copyfile=shutil.copyfile, rmtree=shutil.rmtree),
    )
    monkeypatch.setattr(probe, "STATE_ROOT", root / "run" / "gpu-fault-kernel-replay")
    monkeypatch.setattr(probe, "DROPIN_ROOT", root / "run" / "systemd" / "system")
    monkeypatch.setattr(probe, "PROC_ROOT", root / "proc")
    return SimpleNamespace(
        systemd=systemd, found=found, lookups=lookups, journalctl=journalctl, root=root
    )


def records_flags(records: list[dict[str, Any]]) -> tuple[str, str]:
    """The ``--sha256``/``--b64`` pair the runner sends for ``records``."""

    content = json.dumps(records).encode("utf-8")
    return hashlib.sha256(content).hexdigest(), base64.b64encode(content).decode()

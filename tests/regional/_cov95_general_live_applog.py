"""A fake node for the GF-GENERAL-LIVE CRI Pod-log replay probe.

``install_host`` points the probe's host roots -- its ``/run`` state, the
``/run/systemd/system`` drop-ins, kubelet's ``/var/log/pods`` and the log
collector's state file -- at a ``tmp_path`` tree and answers its
``systemctl``/``systemd-run`` calls with ``FakeSystemd``
(``_cov95_general_live_systemd``): the node log collector installed and
disabled, as on the live sites. ``kubelet_pod`` lays out the directory kubelet
keeps for a running Pod.
"""

from __future__ import annotations

import base64
import hashlib
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts.e2e.regional.probes import general_live_applog_probe as probe
from tests.regional._cov95_general_live_systemd import FakeSystemd, unit_properties

RUN_ID = "gl017-cov-applog-a1-deadbeef"
LOG = probe.LOG_UNIT
UID = "0f3b7c2e-4d5a-4b6c-8d9e-0a1b2c3d4e5f"
OLD_UID = "1a2b3c4d-5e6f-4a7b-8c9d-0e1f2a3b4c5d"
CONTAINER = "pytorch"
NAMESPACE = "gpu-fault-system"
POD = "job-master-0"


def install_host(monkeypatch: pytest.MonkeyPatch, root: Path) -> SimpleNamespace:
    systemd = FakeSystemd({LOG: unit_properties()})
    pods = root / "var" / "log" / "pods"
    pods.mkdir(parents=True)
    collector_state = root / "var" / "lib" / "gpu-fault" / "log-collector-state.json"
    monkeypatch.setattr(probe, "STATE_ROOT", root / "run" / "gpu-fault-applog-replay")
    monkeypatch.setattr(probe, "DROPIN_ROOT", root / "run" / "systemd" / "system")
    monkeypatch.setattr(probe, "POD_LOG_ROOT", pods)
    monkeypatch.setattr(probe, "COLLECTOR_STATE_PATH", collector_state)
    monkeypatch.setattr(
        probe, "subprocess", SimpleNamespace(run=systemd.run, PIPE=subprocess.PIPE)
    )
    return SimpleNamespace(systemd=systemd, pods=pods, root=root)


def kubelet_pod(host: SimpleNamespace, pod_uid: str = UID) -> Path:
    """kubelet's ``<namespace>_<pod>_<uid>/<container>/0.log`` of a running Pod."""

    container = host.pods / f"{NAMESPACE}_{POD}_{pod_uid}" / CONTAINER
    container.mkdir(parents=True)
    (container / "0.log").write_text("", encoding="utf-8")
    return container.parent


def record_flags(records: Any) -> tuple[str, str]:
    """The ``--sha256``/``--b64`` pair for ``records`` (any JSON value or bytes)."""

    content = records if isinstance(records, bytes) else json.dumps(records).encode()
    return hashlib.sha256(content).hexdigest(), base64.b64encode(content).decode()


def fixture_name() -> str:
    return probe.fixture_file_name(RUN_ID)

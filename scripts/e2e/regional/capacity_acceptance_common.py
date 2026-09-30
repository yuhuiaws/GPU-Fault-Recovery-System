"""Names shared by the capacity runner: errors, the probe handle, evidence writes."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scripts.e2e.regional.acceptance_runner_common import write_json_atomic


class CapError(RuntimeError):
    pass


class ProbeTransportError(CapError):
    """The probe Pod restarted or vanished while a case was still measuring it.

    Distinct from a dead local ``kubectl port-forward`` (which the runner
    re-establishes): the metrics and queue state of the previous process are
    gone, so the case cannot continue.
    """


@dataclass
class Probe:
    case: str
    deployment: str
    service: str
    database: str
    pod: str
    local_port: int
    url: str
    port_forward: subprocess.Popen[str] | None
    # Identity of the process the measurements belong to: a different uid or a
    # higher restart count means the metrics and queue state were reset.
    pod_uid: str = ""
    restart_count: int = 0
    forward_log: Path | None = None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def write_json(path: Path, value: Any) -> None:
    """Write one evidence document all-or-nothing (0600, fsynced, renamed).

    The case verdict file is what the next case's predecessor gate reads; a
    half-written one is worse than none. Lists (metric samples, timelines)
    go through the same writer.
    """

    write_json_atomic(path, value)

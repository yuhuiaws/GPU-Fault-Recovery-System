"""The Aurora credential refresher's last verdict as a ``/metrics`` family.

Control-plane review 2026-09-08, H1-2 / CP-3. A ``/metrics`` contributor like
the ones in ``builtin_metric_contributors`` (that module sits at its size
ratchet); registered by ``app.metrics`` and read from the mounted Secret file,
never from the store or the Kubernetes API.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from gpu_fault.app.runtime import AppRuntime

# Where the credential refresher's outcome is mounted (H1-2 / CP-3). The
# CronJob writes ``last-refresh-status.json`` into the ``gpu-fault-aurora``
# Secret next to the DSN key, and the Deployment projects that Secret as files
# at the directory ``GPU_FAULT_STORE_URL_FILE`` points into, so every consumer
# can read the refresher's last verdict without the Kubernetes API.
AURORA_REFRESH_STATUS_FILENAME = "last-refresh-status.json"
AURORA_REFRESH_STATUS_FILE_DEFAULT = "/etc/gpu-fault/aurora/last-refresh-status.json"
_AURORA_REFRESH_AGE = "gpu_fault_aurora_credential_refresh_last_success_age_seconds"


def aurora_refresh_status_path() -> Path:
    url_file = os.getenv("GPU_FAULT_STORE_URL_FILE", "")
    if url_file:
        return Path(url_file).with_name(AURORA_REFRESH_STATUS_FILENAME)
    return Path(AURORA_REFRESH_STATUS_FILE_DEFAULT)


def _parse_status_timestamp(value: object) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def aurora_credential_refresh_metric_lines(_runtime: AppRuntime) -> list[str]:
    """The Aurora credential refresher's last verdict, from the mounted file
    (H1-2).

    The refresher is an hourly CronJob whose failures lived only in Job logs;
    a Job label mismatch made even the admin check's "last Job failed" branch
    dead code. Every run now writes ``{status, finished_at, error, rotated,
    restarted, reason}`` into the Secret. Three facts come out: the age of the
    last run, whether it succeeded, and -- only while the last run succeeded --
    the age of the last success, so ``absent()`` tells "never ran" from "ran
    and failed". No file means no refresher is configured for this role, so
    nothing is published rather than a zero that reads as "just refreshed"; an
    unparsable file is its own series.
    """

    try:
        text = aurora_refresh_status_path().read_text(encoding="utf-8")
    except OSError:
        return []
    lines = [
        "# HELP gpu_fault_aurora_credential_refresh_status_unreadable 1 when the "
        "mounted last-refresh-status.json exists but cannot be parsed (H1-2).",
        "# TYPE gpu_fault_aurora_credential_refresh_status_unreadable gauge",
    ]
    try:
        status = json.loads(text)
    except ValueError:
        status = None
    finished = (
        _parse_status_timestamp(status.get("finished_at"))
        if isinstance(status, dict)
        else None
    )
    if finished is None:
        lines.append("gpu_fault_aurora_credential_refresh_status_unreadable 1")
        return lines
    lines.append("gpu_fault_aurora_credential_refresh_status_unreadable 0")
    age = max(0.0, (datetime.now(timezone.utc) - finished).total_seconds())
    ok = str(status.get("status") or "").lower() == "ok"
    lines.extend(
        [
            "# HELP gpu_fault_aurora_credential_refresh_last_run_age_seconds Seconds since the Aurora credential refresher CronJob last finished a run, successful or not (H1-2).",
            "# TYPE gpu_fault_aurora_credential_refresh_last_run_age_seconds gauge",
            f"gpu_fault_aurora_credential_refresh_last_run_age_seconds {age:.3f}",
            "# HELP gpu_fault_aurora_credential_refresh_last_run_ok 1 when the refresher's most recent run succeeded; the reason for a failure is in the status file and the Job log (H1-2).",
            "# TYPE gpu_fault_aurora_credential_refresh_last_run_ok gauge",
            f"gpu_fault_aurora_credential_refresh_last_run_ok {int(ok)}",
        ]
    )
    if ok:
        lines.extend(
            [
                f"# HELP {_AURORA_REFRESH_AGE} Seconds since the Aurora credential "
                "refresher last verified and published the master password; the "
                "CronJob is hourly and the series is absent while the last run "
                "failed (H1-2).",
                f"# TYPE {_AURORA_REFRESH_AGE} gauge",
                f"{_AURORA_REFRESH_AGE} {age:.3f}",
            ]
        )
    return lines

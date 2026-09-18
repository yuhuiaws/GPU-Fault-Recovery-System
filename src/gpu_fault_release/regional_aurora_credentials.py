"""Run the Aurora credential refresh synchronously before a release mutates.

RDS rotates the managed master password every 7 days. The only thing that
copies AWSCURRENT into the ``gpu-fault-aurora`` Secret is the
``gpu-fault-aurora-credential-refresh`` CronJob
(``deploy/control-plane/regional/aurora-credential-refresh.yaml``,
``src/gpu_fault/aurora_credential_refresh.py``). Running Pods mount that
Secret as files and their pool re-reads the DSN on every connect (CP-3), so
once the Secret is current they recover without a restart -- but until it is,
every reconnect the pool makes after ``max_idle`` fails, and every NEW Pod
scheduled between the rotation and the CronJob's next tick dies on
``FATAL: password authentication failed``.

A release transaction is exactly a burst of new Pods. On 2026-09-07 12:13Z a
production upgrade landed in that window: the re-apply failed, the automatic
rollback restarted control-worker, its Pods could not authenticate, and the
release ended in ``rollback-failed``. Nothing in the release output told a
stale password from a broken release; the cure was a five-second
``kubectl create job --from=cronjob/...`` by hand.

An unchanged refresher runs before the upgrade's store gates. A planned repair
first captures its complete previous objects and applies the approved candidate,
then proves AWSCURRENT before DDL or consumer restarts. Rollback restores the
previous program before running it. It fails closed: a Job
that does not complete stops the transaction with the Job name in the message
and the Job left in place for ``kubectl logs``. When the CronJob is absent
(legacy sites that never installed it) there is nothing to run and the
preflight is skipped. A denied, timed-out or malformed read is not absence and
stops the transaction.
"""

from __future__ import annotations

import base64
import json
import os
import sys
import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from gpu_fault_release.rollout import RegionalRelease

from gpu_fault_release import repository_root
from gpu_fault_release.regional_release_config import ReleaseError
from gpu_fault_release.regional_resource_probe import (
    ProbeState,
    ResourceRef,
    probe_resource,
)

CRONJOB_NAME = "gpu-fault-aurora-credential-refresh"
REFRESH_WAIT_SECONDS_ENV = "GPU_FAULT_RELEASE_AURORA_REFRESH_WAIT_SECONDS"
# The CronJob's own activeDeadlineSeconds is 900 and a converged run finishes in
# seconds; a rotation that has to fetch AWSCURRENT and verify the new password
# against the database is the slow path this has to accommodate (it no longer
# rolls any Deployment). The operator-visible manual cure used the same 300s.
DEFAULT_REFRESH_WAIT_SECONDS = 300
# How much longer the client may block than the server-side wait, so a hung
# kubectl cannot hold the release forever while the Job decides the verdict.
CLIENT_TIMEOUT_MARGIN_SECONDS = 60
CRONJOB_READ_TIMEOUT_SECONDS = 20
REFRESH_STATUS_KEY = "last-refresh-status.json"
# Three missed hourly ticks cannot establish a current refresher.
REFRESH_STATUS_MAX_AGE_SECONDS = 3 * 3600


def read_aurora_refresh_cronjob(release: RegionalRelease) -> dict[str, Any] | None:
    """Read one scoped object; only a successful empty read means absent."""
    observation = probe_resource(
        release.runner,
        release._cpu(),
        ResourceRef("cronjob", "CronJob", CRONJOB_NAME, release.config.namespace),
        timeout_seconds=CRONJOB_READ_TIMEOUT_SECONDS,
    )
    if observation.state is ProbeState.ERROR:
        raise ReleaseError(
            "cannot read Aurora credential refresh CronJob; state is unknown: "
            + observation.error
        )
    return observation.require_readable()


def refresh_wait_seconds() -> int:
    raw = os.getenv(REFRESH_WAIT_SECONDS_ENV, str(DEFAULT_REFRESH_WAIT_SECONDS))
    try:
        value = int(raw)
    except ValueError as exc:
        raise ReleaseError(
            f"{REFRESH_WAIT_SECONDS_ENV} must be a positive integer number of "
            f"seconds, got {raw!r}"
        ) from exc
    if value <= 0:
        raise ReleaseError(
            f"{REFRESH_WAIT_SECONDS_ENV} must be a positive integer number of "
            f"seconds, got {raw!r}"
        )
    return value


def aurora_refresh_status(release: RegionalRelease) -> dict[str, Any] | None:
    """Decode the last run's status, retaining no database credentials."""
    data = (
        release._get_json(
            release._cpu(
                "-n", release.config.namespace, "get", "secret", "gpu-fault-aurora"
            )
        ).get("data")
        or {}
    )
    encoded = data.get(REFRESH_STATUS_KEY)
    if not encoded:
        return None
    try:
        payload = json.loads(base64.b64decode(encoded, validate=True).decode())
    except (ValueError, UnicodeDecodeError) as exc:
        raise ReleaseError(
            f"gpu-fault-aurora {REFRESH_STATUS_KEY} is not JSON"
        ) from exc
    if not isinstance(payload, dict):
        raise ReleaseError(f"gpu-fault-aurora {REFRESH_STATUS_KEY} is not an object")
    finished_at = payload.get("finished_at")
    age_seconds: float | None = None
    if isinstance(finished_at, str) and finished_at:
        try:
            finished = datetime.fromisoformat(finished_at.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ReleaseError(
                f"gpu-fault-aurora {REFRESH_STATUS_KEY} finished_at is not a timestamp"
            ) from exc
        if finished.tzinfo is None:
            finished = finished.replace(tzinfo=UTC)
        age_seconds = (datetime.now(UTC) - finished).total_seconds()
    return {
        "status": payload.get("status"),
        "finished_at": finished_at,
        "age_seconds": age_seconds,
        "stale": age_seconds is None
        or not 0 <= age_seconds <= REFRESH_STATUS_MAX_AGE_SECONDS,
        "error": payload.get("error"),
        "rotated": payload.get("rotated"),
        "restarted": payload.get("restarted"),
        "reason": payload.get("reason"),
    }


def require_current_refresh_status(
    release: RegionalRelease, started_at: datetime
) -> None:
    """Prove this successful Job published AWSCURRENT, reading only its status."""
    encoded = release.runner.run(
        release._cpu(
            "-n",
            release.config.namespace,
            "get",
            "secret",
            "gpu-fault-aurora",
            "-o",
            r"jsonpath={.data.last-refresh-status\.json}",
            "--request-timeout=15s",
        ),
        capture=True,
        timeout_seconds=20,
    ).strip()
    try:
        status = json.loads(base64.b64decode(encoded, validate=True))
        finished = datetime.fromisoformat(status["finished_at"].replace("Z", "+00:00"))
        valid = (
            status["status"] == "ok"
            and finished.tzinfo is not None
            and started_at <= finished <= datetime.now(UTC)
        )
    except (ValueError, TypeError, KeyError, AttributeError):
        valid = False
    if not valid:
        raise ReleaseError(
            "Aurora credential refresh did not publish a fresh successful "
            "AWSCURRENT proof; consumer restarts and DDL are blocked"
        )


def refresh_status_supersedes_failure(
    status: dict[str, Any] | None, conditions: list[dict[str, Any]]
) -> bool:
    """A retained failed Job is superseded only by a newer fresh success."""
    if not status or status["status"] != "ok" or status["stale"]:
        return False
    failed_at = next(
        (
            item.get("lastTransitionTime")
            for item in conditions
            if item.get("type") == "Failed" and item.get("status") == "True"
        ),
        None,
    )
    if not isinstance(failed_at, str):
        return False
    try:
        failure_time = datetime.fromisoformat(failed_at.replace("Z", "+00:00"))
        proof_time = datetime.fromisoformat(
            status["finished_at"].replace("Z", "+00:00")
        )
        return (
            failure_time.tzinfo is not None
            and proof_time.tzinfo is not None
            and proof_time > failure_time
        )
    except (ValueError, TypeError, AttributeError):
        return False


def refresh_aurora_credentials(
    release: RegionalRelease, *, required: bool = False
) -> dict[str, Any]:
    """Run the credential-refresh CronJob once and wait for it; fail closed.

    Returns a note describing what happened (``skipped`` or ``refreshed`` with
    the Job name). Raises :class:`ReleaseError` when the Job does not complete
    within the wait; the Job is then deliberately left in place so its logs can
    be read.
    """

    namespace = release.config.namespace
    wait_seconds = refresh_wait_seconds()
    if read_aurora_refresh_cronjob(release) is None:
        if required:
            raise ReleaseError(
                "Aurora credential refresh CronJob is absent after planned repair; "
                "AWSCURRENT has not been proved"
            )
        note = {
            "status": "skipped",
            "reason": f"cronjob/{CRONJOB_NAME} is not installed in {namespace}",
        }
        print(
            f"aurora-credential-refresh skipped: {note['reason']}",
            file=sys.stderr,
            flush=True,
        )
        return note

    # Unique per run: a fixed name would collide with a Job an earlier failed
    # run left behind for diagnosis, and the CronJob's concurrencyPolicy does
    # not govern Jobs created by hand.
    job = f"{CRONJOB_NAME}-{uuid.uuid4().hex[:8]}"
    started_at = datetime.now(UTC).replace(microsecond=0)
    release.runner.run(
        release._cpu(
            "-n",
            namespace,
            "create",
            "job",
            f"--from=cronjob/{CRONJOB_NAME}",
            job,
        )
    )
    try:
        release.runner.run(
            [
                "bash",
                str(
                    repository_root()
                    / "deploy/control-plane/tools/wait-for-kubernetes-job.sh"
                ),
                str(wait_seconds),
                job,
                *release._cpu("-n", namespace),
            ],
            timeout_seconds=wait_seconds + CLIENT_TIMEOUT_MARGIN_SECONDS,
        )
    except ReleaseError as exc:
        raise ReleaseError(
            f"Aurora credential refresh Job {namespace}/{job} did not complete "
            f"within {wait_seconds}s ({exc}). The release is stopped here on "
            "purpose: new control-plane Pods would fail with 'password "
            "authentication failed' if the gpu-fault-aurora Secret still holds "
            "a rotated-out password. The Job is left in place -- read "
            f"`kubectl -n {namespace} logs job/{job}` (and `describe job`), fix "
            "the cause, delete the Job, then rerun the same deploy command. "
            f"{REFRESH_WAIT_SECONDS_ENV} overrides the wait."
        ) from exc
    if required:
        require_current_refresh_status(release, started_at)
    release.runner.run(
        release._cpu("-n", namespace, "delete", "job", job, "--ignore-not-found")
    )
    return {"status": "refreshed", "job": job}

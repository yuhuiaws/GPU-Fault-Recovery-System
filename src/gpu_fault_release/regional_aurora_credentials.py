"""Run the Aurora credential refresh synchronously before a release mutates.

RDS rotates the managed master password every 7 days. The only thing that
copies AWSCURRENT into the ``gpu-fault-aurora`` Secret and reconciles the
control-plane Deployments is the ``gpu-fault-aurora-credential-refresh``
CronJob (``deploy/control-plane/regional/aurora-credential-refresh.yaml``,
``src/gpu_fault/aurora_credential_refresh.py``). Running replicas are not
disturbed by a rotation -- the pool authenticated before it -- but every NEW
Pod scheduled between the rotation and the CronJob's next tick dies on
``FATAL: password authentication failed``.

A release transaction is exactly a burst of new Pods. On 2026-09-07 12:13Z a
production upgrade landed in that window: the re-apply failed, the automatic
rollback restarted control-worker, its Pods could not authenticate, and the
release ended in ``rollback-failed``. Nothing in the release output told a
stale password from a broken release; the cure was a five-second
``kubectl create job --from=cronjob/...`` by hand.

So the orchestrator runs that same Job itself, before it captures the previous
release and before a rollback plans any restore, and it fails closed: a Job
that does not complete stops the transaction with the Job name in the message
and the Job left in place for ``kubectl logs``. When the CronJob is absent
(legacy sites that never installed it) there is nothing to run and the
preflight is skipped, which is what those sites did before this existed.
"""

from __future__ import annotations

import os
import sys
import uuid
from typing import Any

from gpu_fault_release.regional_release_config import ReleaseError

CRONJOB_NAME = "gpu-fault-aurora-credential-refresh"
REFRESH_WAIT_SECONDS_ENV = "GPU_FAULT_RELEASE_AURORA_REFRESH_WAIT_SECONDS"
# The CronJob's own activeDeadlineSeconds is 900 and a converged run finishes in
# seconds; a rotation that has to verify the new password against the database
# and roll three Deployments is the slow path this has to accommodate. The
# operator-visible manual cure used the same 300s.
DEFAULT_REFRESH_WAIT_SECONDS = 300
# How much longer the client may block than the server-side wait, so a hung
# kubectl cannot hold the release forever while the Job decides the verdict.
CLIENT_TIMEOUT_MARGIN_SECONDS = 60


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


def refresh_aurora_credentials(release: Any) -> dict[str, Any]:
    """Run the credential-refresh CronJob once and wait for it; fail closed.

    Returns a note describing what happened (``skipped`` or ``refreshed`` with
    the Job name). Raises :class:`ReleaseError` when the Job does not complete
    within the wait; the Job is then deliberately left in place so its logs can
    be read.
    """

    namespace = release.config.namespace
    if not release.runner.probe(
        release._cpu("-n", namespace, "get", "cronjob", CRONJOB_NAME)
    ):
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

    wait_seconds = refresh_wait_seconds()
    # Unique per run: a fixed name would collide with a Job an earlier failed
    # run left behind for diagnosis, and the CronJob's concurrencyPolicy does
    # not govern Jobs created by hand.
    job = f"{CRONJOB_NAME}-{uuid.uuid4().hex[:8]}"
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
            release._cpu(
                "-n",
                namespace,
                "wait",
                "--for=condition=complete",
                f"--timeout={wait_seconds}s",
                f"job/{job}",
            ),
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
    release.runner.run(
        release._cpu("-n", namespace, "delete", "job", job, "--ignore-not-found")
    )
    return {"status": "refreshed", "job": job}

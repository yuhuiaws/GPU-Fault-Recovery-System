"""Accepting a PostgreSQL schema change inside one release transaction.

A release whose ``database_schema_version`` differs from the live one cannot be
rolled back: the new wheel refuses to start on any other schema version, so the
old wheel would refuse to start on the new database. Until now the only way to
run such a release was to edit ``site.yaml`` (``spec.autoRollback: false``),
deploy, and remember to edit it back -- and a forgotten edit silently took
automatic rollback away from every later release.

``gpu-fault-admin deploy --accept-schema-change`` replaces that. The operator
states one thing -- "I know this release crosses a schema version and cannot be
rolled back" -- and the release engine does the rest for this transaction only:

* the transaction gate that refuses schema changes under ``autoRollback: true``
  lets this one through and records the acceptance in the release state;
* before the schema Jobs run, a manual Aurora cluster snapshot is created (or an
  earlier one with the same deterministic name reused) and waited for, so the
  one backward path -- restore the database, then roll back -- exists before
  anything moves;
* a verify or stability failure after the schema step is treated as
  fail-forward (``SKIPPED_POLICY``) instead of an automatic rollback that would
  fail at the schema anyway;
* ``site.yaml`` is never touched, so the next release reads ``autoRollback``
  from the site as before. There is nothing to switch back.

The acceptance travels from the operator's command to the release engine as an
environment variable, like the expected-state digest and the quick-validation
evidence path do: the deploy runs as a chain of processes (admin CLI, source
preparer, admin CLI again, release driver, rollout) and every hop inherits its
environment. The value is the mode, ``snapshot`` or ``no-snapshot``; the latter
is for databases nobody would restore (staging) and is spelled out separately on
the command line so it cannot be reached by habit.
"""

from __future__ import annotations

import base64
import os
import re
import time
from datetime import UTC, datetime
from typing import Any, Callable

from gpu_fault_release.regional_release_config import ReleaseError
from gpu_fault_release.regional_release_state import aws_json

ACCEPT_SCHEMA_CHANGE_ENV = "GPU_FAULT_RELEASE_ACCEPT_SCHEMA_CHANGE"
SNAPSHOT_MODE = "snapshot"
NO_SNAPSHOT_MODE = "no-snapshot"
ACCEPTANCE_MODES = frozenset({SNAPSHOT_MODE, NO_SNAPSHOT_MODE})
# The release-state key the acceptance is recorded under. Present only in a
# transaction that crossed a schema version with the operator's consent.
SCHEMA_CHANGE_ACCEPTANCE_KEY = "schema_change_acceptance"
# How long to wait for the snapshot to become ``available``. Aurora snapshots
# are incremental and a control-plane database is small, so minutes; the cap
# exists so a stuck snapshot stops the release before anything moves rather
# than after.
SNAPSHOT_WAIT_SECONDS_ENV = "GPU_FAULT_RELEASE_SCHEMA_SNAPSHOT_WAIT_SECONDS"
DEFAULT_SNAPSHOT_WAIT_SECONDS = 900.0
SNAPSHOT_POLL_SECONDS = 15.0
ACCEPT_FLAG = "--accept-schema-change"
NO_SNAPSHOT_FLAG = "--accept-schema-change-without-snapshot"

_SNAPSHOT_IDENTIFIER = re.compile(r"[^a-z0-9-]+")


def requested_acceptance_mode(
    environment: dict[str, str] | None = None,
) -> str | None:
    """The mode the operator asked for on this command, or None.

    ``1`` is accepted as ``snapshot`` so a hand-set variable behaves like the
    flag; anything else is refused loudly rather than read as consent.
    """

    raw = (environment if environment is not None else os.environ).get(
        ACCEPT_SCHEMA_CHANGE_ENV, ""
    )
    value = raw.strip().lower()
    if not value:
        return None
    if value == "1":
        return SNAPSHOT_MODE
    if value in ACCEPTANCE_MODES:
        return value
    raise ReleaseError(
        f"{ACCEPT_SCHEMA_CHANGE_ENV} must be {SNAPSHOT_MODE} or {NO_SNAPSHOT_MODE}, "
        f"got {raw!r}"
    )


def schema_change_needs_acceptance(
    config: Any, changed: set[str] | frozenset[str]
) -> bool:
    """Whether this diff is the case the transaction gate used to refuse.

    A schema change is never rollback-compatible: the store requires an exact
    schema version and migration history at start-up, so the old wheel cannot
    run on the new schema. There is no flag to say otherwise any more.
    """

    return "database_schema" in changed and bool(config.auto_rollback)


def recorded_acceptance(state: dict[str, Any] | None) -> dict[str, Any] | None:
    value = (state or {}).get(SCHEMA_CHANGE_ACCEPTANCE_KEY)
    return dict(value) if isinstance(value, dict) and value.get("mode") else None


def refusal_message() -> str:
    return (
        "automatic rollback across this PostgreSQL schema change is not declared "
        "backward-compatible; a release that changes the schema cannot be rolled "
        f"back. Rerun the same deploy with {ACCEPT_FLAG} to take an Aurora "
        "snapshot and run this one transaction fail-forward "
        f"(or {NO_SNAPSHOT_FLAG} for a database nobody would restore)"
    )


def resolve_acceptance(
    self: Any,
    *,
    changed: set[str] | frozenset[str],
    resume: bool,
    inherited: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """The acceptance this transaction runs under, or None when none is needed.

    A new transaction needs the operator's mode from the environment; a resumed
    one carries the acceptance it started with, and a mode passed again on the
    resume command is ignored in favour of the recorded one (the snapshot named
    there is the one that exists). Refuses -- before anything moves -- when the
    schema changes, automatic rollback is on, and nobody accepted.

    ``inherited`` is the acceptance of a failed transaction this one supersedes
    (``--supersede-failed-transaction``). It is carried only when it names the
    schema version this candidate requires: that schema already moved under the
    earlier consent and the snapshot named there is the one that exists. A
    candidate that requires yet another version is a new schema change, and the
    ordinary rule applies -- refuse unless this command accepted it.
    """

    if not schema_change_needs_acceptance(self.config, changed):
        return None
    if resume:
        recorded = recorded_acceptance(self._load_state())
        if recorded is not None:
            return recorded
    if inherited is not None:
        accepted_version = int(inherited.get("database_schema_version") or 0)
        if accepted_version == int(self.config.database_schema_version):
            return dict(inherited)
    mode = requested_acceptance_mode()
    if mode is None:
        raise ReleaseError(refusal_message())
    return {
        "mode": mode,
        "accepted_at": datetime.now(UTC).isoformat(),
        "database_schema_version": int(self.config.database_schema_version),
        "snapshot_id": None,
        "snapshot_status": None,
        "cluster_id": None,
    }


def fail_forward_reason(acceptance: dict[str, Any]) -> str:
    snapshot = acceptance.get("snapshot_id")
    tail = (
        f"; restore the database from snapshot {snapshot} before any rollback"
        if snapshot
        else "; no database snapshot was taken (accepted without one)"
    )
    return (
        "schema change accepted for this transaction "
        f"({ACCEPT_FLAG}); rollback cannot cross a schema version{tail}"
    )


def aurora_cluster_identifier(self: Any) -> str | None:
    """The Aurora cluster to snapshot: the configured id, else the writer
    endpoint's first label from the ``gpu-fault-aurora`` Secret."""

    configured = getattr(self.config.health, "aurora_cluster_id", None)
    if configured:
        return str(configured)
    try:
        secret = self._get_json(
            self._cpu("-n", self.config.namespace, "get", "secret", "gpu-fault-aurora")
        )
    except ReleaseError:
        return None
    encoded = (secret.get("data") or {}).get("postgres-url")
    if not isinstance(encoded, str) or not encoded:
        return None
    try:
        url = base64.b64decode(encoded).decode("utf-8")
    except (ValueError, UnicodeDecodeError):
        return None
    host = url.split("@", 1)[-1].split("/", 1)[0].split(":", 1)[0]
    # <cluster-id>.cluster-<hash>.<region>.rds.amazonaws.com
    label = host.split(".", 1)[0]
    return label or None


def snapshot_identifier(self: Any) -> str:
    """Deterministic per release, so a resume finds the snapshot it already took
    instead of taking another. RDS identifiers are lower-case letters, digits
    and hyphens, at most 63 characters, no leading digit or double hyphen."""

    version = int(self.config.database_schema_version)
    release = _SNAPSHOT_IDENTIFIER.sub("-", str(self.release_id).lower())[:12]
    name = f"gpu-fault-pre-v{version}-{release}".strip("-")
    return re.sub(r"-{2,}", "-", name)[:63].rstrip("-")


def _snapshot_status(self: Any, cluster_id: str, snapshot_id: str) -> str | None:
    document = aws_json(
        self,
        [
            "rds",
            "describe-db-cluster-snapshots",
            "--db-cluster-identifier",
            cluster_id,
            "--snapshot-type",
            "manual",
        ],
        cached=False,
    )
    for item in document.get("DBClusterSnapshots") or []:
        if item.get("DBClusterSnapshotIdentifier") == snapshot_id:
            return str(item.get("Status") or "")
    return None


def ensure_schema_change_snapshot(
    self: Any,
    acceptance: dict[str, Any],
    *,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    """Create (or reuse) and wait for the pre-schema snapshot; returns the
    acceptance with the snapshot recorded.

    Runs before the schema Jobs, so a failure here stops the release before the
    database or any Pod has changed. Idempotent through the deterministic name:
    a resume after a crash between snapshot and schema finds the snapshot
    ``available`` and continues.
    """

    if acceptance.get("mode") != SNAPSHOT_MODE:
        return dict(acceptance)
    if (
        acceptance.get("snapshot_id")
        and acceptance.get("snapshot_status") == "available"
    ):
        return dict(acceptance)
    cluster_id = aurora_cluster_identifier(self)
    if not cluster_id:
        raise ReleaseError(
            "cannot take the pre-schema Aurora snapshot: the cluster identifier is "
            "neither configured (health.aurora_cluster_id) nor derivable from the "
            f"gpu-fault-aurora Secret; fix the site or use {NO_SNAPSHOT_FLAG}"
        )
    snapshot_id = snapshot_identifier(self)
    try:
        status = _snapshot_status(self, cluster_id, snapshot_id)
        if status is None:
            self.runner.run(
                [
                    "aws",
                    "rds",
                    "create-db-cluster-snapshot",
                    "--db-cluster-identifier",
                    cluster_id,
                    "--db-cluster-snapshot-identifier",
                    snapshot_id,
                    "--tags",
                    "Key=Application,Value=gpu-fault-control-plane",
                    f"Key=ReleaseId,Value={self.release_id}",
                    "--region",
                    self.config.aws_region,
                    "--output",
                    "json",
                ],
                capture=True,
            )
            status = "creating"
    except ReleaseError as exc:
        raise ReleaseError(
            f"cannot take the pre-schema Aurora snapshot of {cluster_id}: {exc}; "
            "the deploy-host role needs rds:DescribeDBClusterSnapshots, "
            "rds:CreateDBClusterSnapshot and rds:AddTagsToResource on the cluster, "
            f"or use {NO_SNAPSHOT_FLAG} to give up the restore path"
        ) from exc
    wait_seconds = float(
        os.getenv(SNAPSHOT_WAIT_SECONDS_ENV, str(DEFAULT_SNAPSHOT_WAIT_SECONDS))
    )
    deadline = clock() + wait_seconds
    while status != "available":
        if status in {"failed", "deleted", "deleting"}:
            raise ReleaseError(
                f"pre-schema Aurora snapshot {snapshot_id} is {status}; delete it "
                "and rerun, or investigate the cluster before releasing"
            )
        if clock() >= deadline:
            raise ReleaseError(
                f"pre-schema Aurora snapshot {snapshot_id} did not become available "
                f"within {int(wait_seconds)}s (last status {status}); the schema "
                "Jobs did not run, rerun the same deploy once it is available"
            )
        sleep(SNAPSHOT_POLL_SECONDS)
        status = _snapshot_status(self, cluster_id, snapshot_id) or "missing"
    return {
        **acceptance,
        "cluster_id": cluster_id,
        "snapshot_id": snapshot_id,
        "snapshot_status": status,
    }

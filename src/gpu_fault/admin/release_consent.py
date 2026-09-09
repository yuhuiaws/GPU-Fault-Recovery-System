"""The operator's consents that ``gpu-fault-admin deploy`` hands the release engine.

Each is one flag on the public command and one environment variable the engine
reads. The deploy is a chain of processes that each inherit their environment,
so the CLI sets the variable once and never threads a new argument through the
source preparer, the inner CLI or the release driver. An explicit environment
value always wins over the flag, as with the other release-engine variables.

The variable spellings mirror the engine's constants
(``regional_schema_change``, ``regional_admin_commands``,
``regional_release_store_preflight``) rather than import them: the admin CLI does
not depend on the release engine package. Tests pin each pair equal.
"""

from __future__ import annotations

import argparse
import os

ACCEPT_SCHEMA_CHANGE_ENV = "GPU_FAULT_RELEASE_ACCEPT_SCHEMA_CHANGE"
SCHEMA_CHANGE_SNAPSHOT_MODE = "snapshot"
SCHEMA_CHANGE_NO_SNAPSHOT_MODE = "no-snapshot"
SUPERSEDE_FAILED_TRANSACTION_ENV = "GPU_FAULT_RELEASE_SUPERSEDE_FAILED_TRANSACTION"
ALLOW_INFLIGHT_INSTALLS_ENV = "GPU_FAULT_RELEASE_ALLOW_INFLIGHT_INSTALLS"


def schema_change_environment(arguments: argparse.Namespace) -> dict[str, str]:
    """Consent to a schema-version release.

    A release that changes the PostgreSQL schema cannot be rolled back (the new
    wheel requires the exact version), so the engine refuses it under
    ``autoRollback: true``. ``--accept-schema-change`` says once, on the command,
    that the operator knows; the engine then takes an Aurora snapshot before the
    schema Jobs and runs this one transaction fail-forward without touching
    ``site.yaml``. ``--accept-schema-change-without-snapshot`` is the same
    consent for a database nobody would restore. The engine ignores the
    variable when the release does not change the schema, so passing the flag
    on an ordinary release is harmless.
    """

    if os.environ.get(ACCEPT_SCHEMA_CHANGE_ENV, "").strip():
        return {}
    if getattr(arguments, "accept_schema_change_without_snapshot", False):
        return {ACCEPT_SCHEMA_CHANGE_ENV: SCHEMA_CHANGE_NO_SNAPSHOT_MODE}
    if getattr(arguments, "accept_schema_change", False):
        return {ACCEPT_SCHEMA_CHANGE_ENV: SCHEMA_CHANGE_SNAPSHOT_MODE}
    return {}


def supersede_environment(arguments: argparse.Namespace) -> dict[str, str]:
    """Consent to replace a failed fail-forward transaction.

    A transaction that stopped in ``failed``/``partial-convergence`` only
    resumes the release that failed; a deploy of a *different* candidate (the
    fix) is refused with this flag named. ``--supersede-failed-transaction``
    tells the engine to open a new transaction for the candidate whose rollback
    baseline is the failed transaction's last committed release and whose diff
    re-rolls everything the failed release moved. The engine refuses it when
    the recorded transaction is not such a failure, so it cannot be left on by
    habit.
    """

    if os.environ.get(SUPERSEDE_FAILED_TRANSACTION_ENV, "").strip():
        return {}
    if getattr(arguments, "supersede_failed_transaction", False):
        return {SUPERSEDE_FAILED_TRANSACTION_ENV: "1"}
    return {}


def inflight_installs_environment(arguments: argparse.Namespace) -> dict[str, str]:
    """Consent to release over an in-flight node install.

    Before it opens an upgrade or rollback transaction the release engine reads
    the control-plane store once and refuses while a REMEDIATE_DRIVER /
    UPDATE_SOFTWARE_FIRMWARE / REMEDIATE_EFA_DRIVER step is PENDING or WAITING:
    the control plane the transaction puts in place derives a different command
    id for that step and would submit the install a second time on a node still
    running it. ``--allow-inflight-installs`` says once, on the command, that
    the operator accepts that; the engine then logs what it skipped instead of
    refusing.
    """

    if os.environ.get(ALLOW_INFLIGHT_INSTALLS_ENV, "").strip():
        return {}
    if getattr(arguments, "allow_inflight_installs", False):
        return {ALLOW_INFLIGHT_INSTALLS_ENV: "1"}
    return {}


def release_consent_environment(arguments: argparse.Namespace) -> dict[str, str]:
    """Every consent this command carries, as the engine's variables."""

    return {
        **schema_change_environment(arguments),
        **supersede_environment(arguments),
        **inflight_installs_environment(arguments),
    }


def add_release_consent_arguments(deploy: argparse.ArgumentParser) -> None:
    deploy.add_argument(
        "--accept-schema-change",
        action="store_true",
        help=(
            "this release changes the PostgreSQL schema and cannot be rolled "
            "back: take an Aurora snapshot first and run this one transaction "
            "fail-forward without editing spec.autoRollback"
        ),
    )
    deploy.add_argument(
        "--accept-schema-change-without-snapshot",
        action="store_true",
        help=(
            "like --accept-schema-change but without the Aurora snapshot; only "
            "for a database nobody would restore"
        ),
    )
    deploy.add_argument(
        "--supersede-failed-transaction",
        action="store_true",
        help=(
            "the recorded transaction is a fail-forward release that stopped in "
            "failed/partial-convergence and this candidate is a different "
            "release: open a new transaction for it on the last committed "
            "baseline instead of resuming the failed one; refused in any other "
            "state"
        ),
    )
    deploy.add_argument(
        "--allow-inflight-installs",
        action="store_true",
        help=(
            "proceed although a REMEDIATE_DRIVER / UPDATE_SOFTWARE_FIRMWARE / "
            "REMEDIATE_EFA_DRIVER step is still PENDING or WAITING on a node; "
            "without it the release engine refuses to open an upgrade or "
            "rollback transaction (or to roll back automatically) and names "
            "the workflows, because the control plane it puts in place would "
            "submit each such install a second time"
        ),
    )

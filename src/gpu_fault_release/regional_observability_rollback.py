"""Capture and restore the live ADOT collector so rollback can compensate it.

Automatic rollback used to refuse any release whose diff touched the ADOT
manifest or image. The reason was real: the collector is applied straight from
the candidate checkout, so a rollback that restored every other component would
leave the collector running the candidate's configuration while the transaction
reported ``rollback PASSED``. Refusing beat lying -- but the refusal's
granularity was the whole release, so a single metrics-filter edit forced the
operator to give up automatic rollback for the wheels, Agents and schema too.

Capturing the live objects before the observability component mutates them makes
the collector compensable, exactly like the AMP rule and Alertmanager blobs that
the same snapshot already carries. The captured objects are the previous
release's own applied state, so restoring them restores the previous manifest
*and* the previous image without needing the previous checkout.

The object capture itself is shared with the control-plane endpoint, which is
applied from the tree the same way; see ``regional_manifest_snapshot``.

The AMP definitions (the static rule namespace, the Alertmanager definition,
and -- through ``regional_dataplane_observability`` -- the per-cluster
expected-collector rule namespace) are applied by AMP asynchronously: a put
returns at once, the definition sits CREATING/UPDATING while the PREVIOUS one
keeps serving, and a write issued meanwhile is a ConflictException. Every
restore here therefore waits for the definition to settle before it writes and
for AMP to report ACTIVE after (``wait_for_amp_definition``), or for
ResourceNotFoundException after a delete (``wait_for_amp_definition_gone``) --
the same pair the installer script has -- and a definition AMP rejects fails
the restore naming AMP's ``statusReason`` instead of being recorded as put back.
"""

from __future__ import annotations

import base64
import json
import sys
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from gpu_fault_release import repository_root
from gpu_fault_release.regional_manifest_snapshot import (
    apply_snapshot_objects,
    capture_declared_objects,
    declared_manifest_objects,
    delete_absent_objects,
    restart_snapshot_deployments,
    snapshot_parts,
)
from gpu_fault_release.regional_release_config import ReleaseError

ROOT = repository_root()
ADOT_MANIFEST = ROOT / "deploy/observability/adot-control-plane.yaml"
ADOT_ROLLOUT_TIMEOUT = "300s"
#: How often and how many times an AMP definition is described while waiting
#: for it to settle, become ACTIVE or disappear (the installer's own budget is
#: 300 s; 60 x 5 s matches it). Each wait has the whole budget.
AMP_DEFINITION_WAIT_ATTEMPTS = 60
AMP_DEFINITION_WAIT_SECONDS = 5.0
#: A write on a definition in one of these is a ConflictException.
AMP_SETTLING_STATUSES = frozenset({"CREATING", "UPDATING", "DELETING"})
#: AMP rejected the definition; the previous one (if any) is still serving.
AMP_FAILED_STATUSES = frozenset({"CREATION_FAILED", "UPDATE_FAILED"})
_sleep: Callable[[float], None] = time.sleep


def _say(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


# --- AMP definitions: describe and wait ----------------------------------------------


@dataclass(frozen=True)
class AmpDefinition:
    """One AMP definition as the ``aws amp`` CLI describes it.

    ``describe`` is the complete describe command (without ``--output``);
    ``root`` the key its JSON answer is wrapped in (``ruleGroupsNamespace`` or
    ``alertManagerDefinition``); ``label`` how messages name it.
    """

    describe: tuple[str, ...]
    root: str
    label: str


def describe_amp_definition(
    release: Any, definition: AmpDefinition
) -> dict[str, Any] | None:
    """The definition's describe document, ``None`` when it does not exist.

    Only ``ResourceNotFoundException`` is absence. A throttle, a credentials
    error or an unreadable answer raises: a restore built on a guess would
    create over a definition that exists (ConflictException) or skip deleting
    one that does.
    """

    code, stdout, stderr = release.runner.probe_output(
        [*definition.describe, "--output", "json"]
    )
    if code:
        if "ResourceNotFoundException" in stderr:
            return None
        raise ReleaseError(
            f"cannot read the {definition.label}: {stderr.strip() or code}"
        )
    try:
        document = json.loads(stdout)[definition.root]
    except (ValueError, KeyError, TypeError) as exc:
        raise ReleaseError(
            f"cannot read the {definition.label}: unexpected describe output"
        ) from exc
    if not isinstance(document, dict):
        raise ReleaseError(
            f"cannot read the {definition.label}: unexpected describe output"
        )
    return document


def amp_definition_status(document: dict[str, Any] | None) -> str | None:
    """``status.statusCode`` of a describe document; ``None`` for an absent one."""

    if document is None:
        return None
    status = document.get("status")
    return str(status.get("statusCode") or "") if isinstance(status, dict) else ""


def _amp_definition_status_reason(document: dict[str, Any] | None) -> str:
    status = document.get("status") if isinstance(document, dict) else None
    reason = status.get("statusReason") if isinstance(status, dict) else None
    return str(reason or "").strip() or "AMP gave no statusReason"


def poll_amp_definition(
    release: Any,
    definition: AmpDefinition,
    *,
    settled: Callable[[dict[str, Any] | None], bool],
    waiting_for: str,
    attempts: int | None = None,
    interval: float | None = None,
    sleep: Callable[[float], None] | None = None,
) -> dict[str, Any] | None:
    """Describe the definition until ``settled`` accepts the answer; the answer.

    One describe up front, then up to ``attempts`` sleeps of ``interval``
    seconds each followed by another describe (the module's budget unless the
    caller brings its own). A definition that never settles is a
    ``ReleaseError`` naming its last status, never a guess: every caller
    writes to AMP or records an outcome on the strength of this answer.
    """

    attempts = AMP_DEFINITION_WAIT_ATTEMPTS if attempts is None else attempts
    interval = AMP_DEFINITION_WAIT_SECONDS if interval is None else interval
    sleep = _sleep if sleep is None else sleep
    document = describe_amp_definition(release, definition)
    for _attempt in range(attempts):
        if settled(document):
            return document
        _say(
            f"{definition.label} is {amp_definition_status(document) or 'absent'}; "
            f"waiting for it to {waiting_for}"
        )
        sleep(interval)
        document = describe_amp_definition(release, definition)
    raise ReleaseError(
        f"{definition.label} is still {amp_definition_status(document) or 'absent'} "
        f"after {attempts * interval:.0f}s (waited for it to {waiting_for})"
    )


def wait_for_amp_definition_settled(
    release: Any, definition: AmpDefinition, **budget: Any
) -> bool:
    """Whether the definition exists once AMP has stopped changing it.

    A definition the candidate's installer just put or deleted may still be
    CREATING/UPDATING/DELETING; writing to it now is a ConflictException, so
    a restore waits for AMP to settle first and fails closed if it never
    does. A failed definition (``CREATION_FAILED``, ``UPDATE_FAILED``) exists
    and can be put over.
    """

    document = poll_amp_definition(
        release,
        definition,
        settled=lambda d: amp_definition_status(d) not in AMP_SETTLING_STATUSES,
        waiting_for="settle before restoring it",
        **budget,
    )
    return document is not None


def wait_for_amp_definition_active(
    release: Any, definition: AmpDefinition, verb: str, **budget: Any
) -> None:
    """After a put/create: wait until AMP reports the definition ACTIVE.

    The installer's ``wait_for_amp_definition``: a put returns at once and the
    previous definition keeps serving until AMP has validated the new one, so
    an outcome recorded at the put is a record of a write AMP may yet reject
    -- and the next writer (a resume, the next deploy's installer) lands on an
    UPDATING definition, a ConflictException. A definition AMP rejected
    (``CREATION_FAILED``/``UPDATE_FAILED``) fails the restore naming AMP's
    ``statusReason``; one that disappears meanwhile, reports a status this
    code does not know, or never leaves CREATING/UPDATING within the budget,
    fails it too.
    """

    def settled(document: dict[str, Any] | None) -> bool:
        status = amp_definition_status(document)
        if status is None:
            raise ReleaseError(
                f"{definition.label} disappeared after {verb} "
                "(ResourceNotFoundException while waiting for it to become ACTIVE)"
            )
        if status in AMP_FAILED_STATUSES:
            raise ReleaseError(
                f"AMP rejected the {definition.label} after {verb}: status "
                f"{status}: {_amp_definition_status_reason(document)}"
            )
        if status == "ACTIVE":
            return True
        if status in AMP_SETTLING_STATUSES:
            return False
        raise ReleaseError(
            f"{definition.label} reports an unexpected status {status!r} after {verb}"
        )

    poll_amp_definition(
        release, definition, settled=settled, waiting_for="become ACTIVE", **budget
    )


def wait_for_amp_definition_gone(
    release: Any, definition: AmpDefinition, **budget: Any
) -> None:
    """After a delete: wait until describe answers ResourceNotFoundException.

    The installer's ``wait_for_amp_definition_gone``: the delete is
    asynchronous, and a create issued on a DELETING definition -- the next
    deploy's installer -- is a ConflictException. Only absence is deleted;
    any other describe failure raises (``describe_amp_definition``).
    """

    poll_amp_definition(
        release,
        definition,
        settled=lambda d: d is None,
        waiting_for="be deleted",
        **budget,
    )


def _amp_common(release: Any) -> list[str]:
    return [
        "--region",
        release.config.aws_region,
        "--workspace-id",
        str(release.config.health.amp_workspace_id),
    ]


def _rule_namespace_definition(release: Any, name: str) -> AmpDefinition:
    return AmpDefinition(
        describe=(
            "aws",
            "amp",
            "describe-rule-groups-namespace",
            *_amp_common(release),
            "--name",
            name,
        ),
        root="ruleGroupsNamespace",
        label=f"AMP rule namespace {name}",
    )


def _alertmanager_definition(release: Any) -> AmpDefinition:
    return AmpDefinition(
        describe=(
            "aws",
            "amp",
            "describe-alert-manager-definition",
            *_amp_common(release),
        ),
        root="alertManagerDefinition",
        label="AMP Alertmanager definition",
    )


def _put_amp_definition(
    release: Any,
    definition: AmpDefinition,
    *,
    put: str,
    create: str,
    arguments: list[str],
) -> str:
    """Write one definition the way the installer does; the verb that ran.

    Waits for the definition to settle, puts it when it exists and creates it
    when only ResourceNotFoundException says it does not, then waits for AMP to
    report the result ACTIVE, so the caller moves on only once the previous
    definition is really the one serving.
    """

    verb = put if wait_for_amp_definition_settled(release, definition) else create
    release.runner.run(["aws", "amp", verb, *_amp_common(release), *arguments])
    wait_for_amp_definition_active(release, definition, verb)
    return verb


def declared_adot_objects(manifest_text: str) -> tuple[dict[str, str], ...]:
    """Return the objects the ADOT manifest applies, in manifest order.

    The installer applies this file as-is, so the file is the authority on which
    objects an upgrade may mutate -- and therefore on which objects a rollback
    has to put back.
    """

    return declared_manifest_objects(manifest_text, label="ADOT manifest")


def capture_adot_objects(release: Any) -> dict[str, Any]:
    """Read the live ADOT objects the observability component is about to mutate.

    An object the manifest declares but the cluster does not have yet is one the
    candidate adds; rollback removes it rather than pretending it was there.
    """

    captured = capture_declared_objects(
        release,
        declared_adot_objects(ADOT_MANIFEST.read_text(encoding="utf-8")),
        label="live ADOT object",
    )
    if not any(str(item.get("kind")) == "Deployment" for item in captured["objects"]):
        # The collector Deployment exists on every converged site -- the same
        # capture already fails closed when it cannot read its image. Missing
        # here means this snapshot is reading the wrong namespace, and a
        # rollback built on it would silently compensate nothing.
        raise ReleaseError(
            "live ADOT collector Deployment is missing from namespace "
            f"{captured['namespace']}"
        )
    return captured


def _snapshot_parts(snapshot: object) -> tuple[str, list[Any], list[Any]]:
    if not isinstance(snapshot, dict) or "adot" not in snapshot:
        raise ReleaseError(
            "previous observability snapshot has no ADOT collector objects"
        )
    namespace, objects, absent = snapshot_parts(
        snapshot["adot"],
        label="previous ADOT collector snapshot",
    )
    if not objects:
        raise ReleaseError("previous ADOT collector snapshot is empty")
    return namespace, objects, absent


def restore_adot_objects(release: Any, snapshot: object) -> bool:
    """Put the captured collector back and, if anything changed, prove it runs.

    The restart is not cosmetic: the collector reads its pipeline from the
    ConfigMap it mounts, so restoring the previous ConfigMap without replacing
    the running Pod would leave the candidate's configuration serving from
    memory, which is the exact state this compensation exists to end. When
    every object applied back as ``unchanged`` the candidate never touched
    them and the running Pod already serves the restored configuration; the
    installer skips its restart on the same evidence, and so does this.
    Returns whether the collector was restarted.
    """

    namespace, objects, absent = _snapshot_parts(snapshot)
    changed = apply_snapshot_objects(release, objects)
    delete_absent_objects(release, namespace, absent)
    if changed:
        restart_snapshot_deployments(
            release,
            namespace,
            objects,
            timeout=ADOT_ROLLOUT_TIMEOUT,
        )
    return changed


def capture_observability_snapshot(release: Any) -> dict[str, Any]:
    common = [
        "--region",
        release.config.aws_region,
        "--workspace-id",
        release.config.health.amp_workspace_id,
    ]
    rules = json.loads(
        release.runner.run(
            [
                "aws",
                "amp",
                "describe-rule-groups-namespace",
                *common,
                "--name",
                release.config.health.amp_rule_namespace,
                "--output",
                "json",
            ],
            capture=True,
        )
    )["ruleGroupsNamespace"]
    alertmanager = json.loads(
        release.runner.run(
            [
                "aws",
                "amp",
                "describe-alert-manager-definition",
                *common,
                "--output",
                "json",
            ],
            capture=True,
        )
    )["alertManagerDefinition"]
    return {
        "rule_namespace": release.config.health.amp_rule_namespace,
        "rules_data_base64": str(rules["data"]),
        "alertmanager_data_base64": str(alertmanager["data"]),
        "adot": capture_adot_objects(release),
    }


def restore_observability_snapshot(
    release: Any,
    snapshot: object,
) -> bool:
    """Put the AMP blobs and the collector back; returns whether the collector
    was restarted (``False`` when every object applied back ``unchanged``).

    The static rule namespace and the Alertmanager definition are each waited
    for -- to settle before the put, to be ACTIVE after -- and a definition
    AMP rejects fails the restore before the collector is touched, so the
    rollback never records the previous alerting as back while the candidate's
    is still the one evaluating.
    """

    if not isinstance(snapshot, dict):
        raise ReleaseError("previous observability snapshot is unavailable")
    try:
        rules = base64.b64decode(
            str(snapshot["rules_data_base64"]),
            validate=True,
        )
        alertmanager = base64.b64decode(
            str(snapshot["alertmanager_data_base64"]),
            validate=True,
        )
        rule_namespace = str(snapshot["rule_namespace"])
    except (KeyError, ValueError) as exc:
        raise ReleaseError("previous observability snapshot is invalid") from exc
    # The collector half is validated here too, before the AMP blobs are
    # rewritten: a snapshot that cannot be put back whole must not be put back
    # in part.
    _snapshot_parts(snapshot)
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        rules_path = root / "rules.yaml"
        alertmanager_path = root / "alertmanager.yaml"
        rules_path.write_bytes(rules)
        alertmanager_path.write_bytes(alertmanager)
        _put_amp_definition(
            release,
            _rule_namespace_definition(release, rule_namespace),
            put="put-rule-groups-namespace",
            create="create-rule-groups-namespace",
            arguments=["--name", rule_namespace, "--data", f"fileb://{rules_path}"],
        )
        _put_amp_definition(
            release,
            _alertmanager_definition(release),
            put="put-alert-manager-definition",
            create="create-alert-manager-definition",
            arguments=["--data", f"fileb://{alertmanager_path}"],
        )
    return restore_adot_objects(release, snapshot)

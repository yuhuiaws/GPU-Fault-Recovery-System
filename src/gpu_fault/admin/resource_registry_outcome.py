"""The outcome of the last installation registry sync, kept on the deploy host.

``scripts/release_deploy.py`` ends every release -- a ``deploy`` or a ``config``
re-roll -- by rebuilding the Aurora installation resource registry from the
bootstrap inventory and live discovery
(``resource_registry.sync_installation_resource_registry``). The transaction is
committed by then, so a failure stays a completion warning
(``release_deploy_evidence.installation_resource_registry_record``). A warning
alone was not enough: the rebuild failed on every deploy and ``config`` for nine
days (live 2026-10-04..13) and nobody saw it, while ``uninstall`` deletes exactly
what the registry records, and ``remove-cluster`` deletes the removed cluster's
registry rows and then drops that cluster's roles from the bootstrap inventory.
A resource the registry lacks is left behind by either, and after a removal
nothing on the deploy host knows about it any more.

So the outcome of the last sync is one record,
``<state-dir>/release-deploy/installation-registry-sync.json``, beside the
per-attempt release records that carry the same outcome under
``installation_registry``. Only a sync that ran writes it. A NOOP deploy skips
the sync only while the record is a success for the release it starts from
(:func:`noop_kept_record`; a NOOP that staged a new release id adds that id to
``noop_release_ids``), and otherwise retries it -- which is what makes "the next
deploy resynchronizes" true for a release that changes nothing.

``status`` reports the record (:func:`status_view`) as a finding, never as a
failing check: ``healthy`` is the release's runtime verdict, which the deploy
driver reads itself before it proves a NOOP or supersedes a failed transaction,
so a stale description of what was installed must not turn it -- or the deploy
that repairs the record -- red. The danger is confined to the two deletion
verbs, and they refuse on their own (:func:`deletion_gate`).
"""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

from gpu_fault.admin.atomic_json import write_json_atomic
from gpu_fault.admin.bootstrap_common import BootstrapError

RECORD_FILE = Path("release-deploy") / "installation-registry-sync.json"
SCHEMA_VERSION = 1
SYNCED = "SYNCED"
#: The per-attempt record's word for a failed sync, kept so both say the same.
UNAVAILABLE = "UNAVAILABLE"
MISSING = "MISSING"
UNREADABLE = "UNREADABLE"
ACCEPT_FLAG = "--accept-unsynchronized-registry"
#: The key of this record in the `status` document and the deletion journals.
FIELD = "installation_registry_sync"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class RegistrySyncRecordError(BootstrapError):
    """The record exists but is not one :func:`write_outcome` wrote."""


def record_path(state_dir: Path) -> Path:
    return state_dir / RECORD_FILE


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _text(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _problem(value: object) -> str | None:
    """Why ``value`` is not a record, or ``None`` when it is one."""

    if not isinstance(value, dict):
        return "is not a JSON object"
    if type(value.get("schema_version")) is not int or (
        value["schema_version"] != SCHEMA_VERSION
    ):
        return f"has schema_version {value.get('schema_version')!r}, not 1"
    status = value.get("status")
    if status not in (SYNCED, UNAVAILABLE):
        return f"has an unknown status {status!r}"
    if not _text(value.get("release_id")) or not _text(value.get("synced_at")):
        return "lacks its release_id or synced_at"
    if status == SYNCED and not (
        _text(value.get("path")) and _SHA256.fullmatch(str(value.get("sha256") or ""))
    ):
        return "records a sync without its snapshot path and sha256"
    if status == UNAVAILABLE and not _text(value.get("error")):
        return "records a failed sync without its error"
    noop = value.get("noop_release_ids")
    if not isinstance(noop, list) or not all(_text(item) for item in noop):
        return "has an invalid noop_release_ids list"
    return None


def write_outcome(
    state_dir: Path, *, release_id: str, outcome: Mapping[str, Any]
) -> Path:
    """Replace the record with the outcome of a sync that just ran.

    ``outcome`` is the per-attempt ``installation_registry`` record of the
    release ``release_id``. A record :func:`read_record` would refuse is never
    written; the caller treats that, like any write failure, as "not recorded".
    """

    value = {
        **outcome,
        "schema_version": SCHEMA_VERSION,
        "release_id": release_id,
        "noop_release_ids": [],
    }
    problem = _problem(value)
    if problem is not None:
        raise RegistrySyncRecordError(f"installation registry sync outcome {problem}")
    path = record_path(state_dir)
    write_json_atomic(path, value)
    return path


def read_record(state_dir: Path) -> dict[str, Any] | None:
    """The record; ``None`` when no sync outcome was ever recorded here.

    Anything that is not exactly a record raises :class:`RegistrySyncRecordError`,
    which the deletion verbs treat as a refusal (fail closed).
    """

    path = record_path(state_dir)
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except (OSError, UnicodeDecodeError) as exc:
        raise RegistrySyncRecordError(f"{path} cannot be read: {exc}") from exc
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RegistrySyncRecordError(f"{path} is not valid JSON: {exc}") from exc
    problem = _problem(value)
    if problem is not None:
        raise RegistrySyncRecordError(f"{path} {problem}")
    return dict(value)


def _covers(record: Mapping[str, Any], release_id: str | None) -> bool:
    return bool(release_id) and (
        record["release_id"] == release_id or release_id in record["noop_release_ids"]
    )


def noop_kept_record(
    state_dir: Path, *, release_id: str, live_release_id: str | None
) -> dict[str, Any] | None:
    """The success a NOOP release keeps; ``None`` when it must sync after all.

    The fast path proved nothing was installed, so a successful sync of the
    release it started from (``live_release_id``) or of ``release_id`` itself
    still describes the site; a staged new ``release_id`` is added to
    ``noop_release_ids`` and nothing else changes. No record, an unreadable one,
    a failure or a success of another release is no such proof: the caller then
    runs the sync, which is the remedy every refusal and finding names.
    """

    try:
        record = read_record(state_dir)
    except RegistrySyncRecordError:
        return None
    if record is None or record["status"] != SYNCED:
        return None
    if _covers(record, release_id):
        return record
    if not _covers(record, live_release_id):
        return None
    record["noop_release_ids"] = [*record["noop_release_ids"], release_id]
    try:
        write_json_atomic(record_path(state_dir), record)
    except OSError:
        return None
    return record


def _remedy(state_dir: Path) -> str:
    return (
        f"`gpu-fault-admin deploy --state-dir {state_dir}` (a NOOP deploy too: it "
        "retries the sync while this record is not a success) or the next "
        "`gpu-fault-admin config` re-roll synchronizes the registry again"
    )


def _assess(state_dir: Path) -> tuple[str, dict[str, Any] | None, str | None]:
    """``(status, record, problem)``: SYNCED, UNAVAILABLE, MISSING or UNREADABLE."""

    try:
        record = read_record(state_dir)
    except RegistrySyncRecordError as exc:
        return UNREADABLE, None, str(exc)
    if record is None:
        return MISSING, None, None
    return str(record["status"]), record, None


def _unsynchronized(
    state_dir: Path, status: str, record: Mapping[str, Any] | None, problem: str | None
) -> str:
    """One sentence on why the registry is not known to be synchronized."""

    if status == UNREADABLE:
        return f"the installation registry sync record is unreadable: {problem}"
    if status == MISSING:
        return (
            "no installation registry sync outcome is recorded in "
            f"{record_path(state_dir)} (the last release completed before outcomes "
            "were recorded, or no release has completed)"
        )
    assert record is not None
    return (
        f"the last installation registry sync failed at {record['synced_at']} "
        f"(release {record['release_id']}): {record['error']}"
    )


def status_view(state_dir: Path, *, release_id: str | None) -> dict[str, Any]:
    """The ``installation_registry_sync`` field of ``status``, with its finding.

    ``release_id`` is the release the site runs (the report's configured
    release, else its live one). A success that does not describe it -- a
    release completed without recording an outcome, e.g. an interrupted
    completion step -- is reported too, without the refusal note:
    :func:`deletion_gate` refuses only a failed, missing or unreadable record.
    """

    status, record, problem = _assess(state_dir)
    current = status == SYNCED and record is not None and _covers(record, release_id)
    view: dict[str, Any] = {
        "record": str(record_path(state_dir)),
        "status": status,
        "release_id": (record or {}).get("release_id"),
        "synced_at": (record or {}).get("synced_at"),
        "error": (record or {}).get("error") or problem,
        "live_release_id": release_id,
        "deletion_verbs_refused": status != SYNCED,
        "finding": None,
    }
    if status != SYNCED:
        view["finding"] = (
            _unsynchronized(state_dir, status, record, problem)
            + "; uninstall and remove-cluster refuse to start while it stands -- "
            + _remedy(state_dir)
        )
    elif release_id and not current:
        assert record is not None
        view["finding"] = (
            f"no installation registry sync outcome since the live release "
            f"{release_id} (the last sync succeeded at {record['synced_at']} for "
            f"release {record['release_id']}) -- " + _remedy(state_dir)
        )
    return view


def deletion_gate(state_dir: Path, *, verb: str, accept: bool) -> dict[str, Any]:
    """Refuse to start a deletion attempt over a registry not known to be synchronized.

    ``verb`` is ``uninstall`` or ``remove-cluster``; both fix their deletion
    list from the registry export that follows, so this runs once per attempt,
    before the attempt's journal exists: a refusal leaves no uninstall or
    removal in progress that would refuse (or be invalidated by) the deploy that
    is the remedy. A missing record refuses too -- no proof is no proof. The
    returned evidence, journaled by the caller, keeps the recorded outcome
    (with its failure) and whether ``--accept-unsynchronized-registry`` let the
    attempt start anyway.
    """

    status, record, problem = _assess(state_dir)
    overridden = status != SYNCED
    if overridden and not accept:
        scope = (
            "uninstall deletes exactly what the registry records"
            if verb == "uninstall"
            else "remove-cluster deletes the cluster's registry rows and then drops "
            "its roles from the bootstrap inventory"
        )
        raise BootstrapError(
            f"{verb} refuses to start: "
            + _unsynchronized(state_dir, status, record, problem)
            + f". {scope}, so a resource the registry lacks would be left behind. "
            + f"Resynchronize first -- {_remedy(state_dir)} -- and rerun, or rerun "
            + f"with {ACCEPT_FLAG} to delete only what the registry records (the "
            + f"flag and this outcome are recorded in the {verb} journal)"
        )
    return {
        "record": str(record_path(state_dir)),
        "status": status,
        "outcome": record,
        "error": problem,
        "accept_flag": accept,
        "accepted_unsynchronized": overridden,
        "evaluated_at": _utc_now(),
    }


def add_accept_flag(parser: Any, *, verb: str) -> None:
    """``--accept-unsynchronized-registry`` on ``uninstall`` and ``remove-cluster``."""

    parser.add_argument(
        ACCEPT_FLAG,
        action="store_true",
        help=(
            f"start the {verb} even though the last installation registry sync "
            "failed, is not recorded or is unreadable (release-deploy/"
            "installation-registry-sync.json): delete only what the registry "
            f"records; the flag and the recorded outcome go into the {verb} "
            f"journal. Without it the {verb} refuses until a deploy or config "
            "re-roll synchronizes the registry"
        ),
    )

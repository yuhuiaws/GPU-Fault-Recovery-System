from __future__ import annotations

import hashlib
import json
import re
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterator

from gpu_fault.admin import operator_identity
from gpu_fault.admin.atomic_json import write_json_atomic
from gpu_fault.admin.operation_lock import SiteOperationBusy, site_operation_lock
from gpu_fault.digests import SHA256_PATTERN

PROFILE_PLAN = Path("release-deploy/profile-plan.json")
PROFILE_APPROVAL = Path("release-deploy/profile-approval.json")
PROFILE_APPROVAL_ARCHIVE = Path("release-deploy/profile-approvals")
APPROVAL_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{2,127}$")
RELEASE_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
PLAN_IDENTITY_FIELDS = (
    "schema_version",
    "site_identity",
    "site_identity_sha256",
    "registration_cluster_id",
    "current_version",
    "desired_version",
    "current_policy_digest",
    "policy_digest",
    "live_profile_sha256",
    "active_source_sha256",
    "source_sha256",
    "snapshot_sha256",
    "change_kind",
    "changes",
    "approval_required",
)
PLAN_TARGET_FIELDS = (
    "site_identity",
    "site_identity_sha256",
    "registration_cluster_id",
    "desired_version",
    "policy_digest",
    "source_sha256",
    "snapshot_sha256",
)


class ProfileApprovalError(ValueError):
    pass


class StaleProfileApprovalError(ProfileApprovalError):
    pass


@dataclass(frozen=True)
class ResolvedProfileApproval:
    reference: str
    plan_sha256: str
    relation: str
    # ``None`` for a record written before approvals carried an approver (I2).
    approver_identity: str | None = None


def _utc_timestamp(value: datetime | None = None) -> str:
    return (value or datetime.now(UTC)).isoformat()


def _read_json(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProfileApprovalError(f"{description} is invalid") from exc
    if not isinstance(value, dict):
        raise ProfileApprovalError(f"{description} must be a JSON object")
    return value


def _unlink(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def profile_site_identity_sha256(value: object) -> str:
    if not isinstance(value, dict):
        raise ProfileApprovalError("Profile plan site_identity must be an object")
    expected = {"site_name", "aws_region", "cpu_eks_arn"}
    if set(value) != expected:
        raise ProfileApprovalError(
            "Profile plan site_identity must contain exactly "
            "site_name, aws_region, and cpu_eks_arn"
        )
    identity: dict[str, str] = {}
    for field in sorted(expected):
        item = value.get(field)
        if not isinstance(item, str) or not item.strip():
            raise ProfileApprovalError(
                f"Profile plan site_identity.{field} must be non-empty"
            )
        if item != item.strip():
            raise ProfileApprovalError(
                f"Profile plan site_identity.{field} must not have whitespace padding"
            )
        identity[field] = item
    return hashlib.sha256(
        json.dumps(
            identity,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def profile_plan_sha256(document: dict[str, Any]) -> str:
    try:
        identity = {field: document[field] for field in PLAN_IDENTITY_FIELDS}
    except KeyError as exc:
        raise ProfileApprovalError(
            f"Profile plan is missing identity field: {exc.args[0]}"
        ) from exc
    if identity["schema_version"] != 1:
        raise ProfileApprovalError("Profile plan schema is invalid")
    if not isinstance(identity["changes"], list):
        raise ProfileApprovalError("Profile plan changes must be a list")
    if not isinstance(identity["approval_required"], bool):
        raise ProfileApprovalError("Profile plan approval_required must be a boolean")
    return hashlib.sha256(
        json.dumps(
            identity,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def _validated_plan(document: dict[str, Any]) -> tuple[dict[str, Any], str]:
    site_identity_digest = profile_site_identity_sha256(document.get("site_identity"))
    if document.get("site_identity_sha256") != site_identity_digest:
        raise ProfileApprovalError(
            "Profile plan site identity digest does not match its content"
        )
    digest = profile_plan_sha256(document)
    if document.get("plan_sha256") != digest:
        raise ProfileApprovalError("Profile plan digest does not match its content")
    return document, digest


def _validated_approval_record(
    document: dict[str, Any],
) -> dict[str, Any]:
    if document.get("schema_version") != 1:
        raise ProfileApprovalError("Profile approval record schema is invalid")
    reference = str(document.get("reference") or "").strip()
    if not APPROVAL_PATTERN.fullmatch(reference):
        raise ProfileApprovalError("Profile approval reference has an invalid format")
    digest = str(document.get("plan_sha256") or "")
    if not SHA256_PATTERN.fullmatch(digest):
        raise ProfileApprovalError("Profile approval plan digest is invalid")
    site_identity_digest = profile_site_identity_sha256(document.get("site_identity"))
    if document.get("site_identity_sha256") != site_identity_digest:
        raise ProfileApprovalError(
            "Profile approval site identity digest does not match its content"
        )
    approved_at = str(document.get("approved_at") or "")
    try:
        timestamp = datetime.fromisoformat(approved_at.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ProfileApprovalError("Profile approval timestamp is invalid") from exc
    if timestamp.tzinfo is None:
        raise ProfileApprovalError("Profile approval timestamp must include a timezone")
    if "approver_identity" in document:
        approver = document.get("approver_identity")
        if not isinstance(approver, str) or not approver.strip():
            raise ProfileApprovalError("Profile approval approver identity is invalid")
    return document


def _approver_identity(document: dict[str, Any]) -> str | None:
    approver = document.get("approver_identity")
    return approver if isinstance(approver, str) and approver.strip() else None


def profile_plan_path(state_dir: Path) -> Path:
    return state_dir.expanduser().resolve() / PROFILE_PLAN


def profile_approval_path(state_dir: Path) -> Path:
    return state_dir.expanduser().resolve() / PROFILE_APPROVAL


def profile_approval_archive_path(state_dir: Path, plan_sha256: str) -> Path:
    if not SHA256_PATTERN.fullmatch(plan_sha256):
        raise ProfileApprovalError("Profile approval plan digest is invalid")
    return state_dir.expanduser().resolve() / PROFILE_APPROVAL_ARCHIVE / plan_sha256


@contextmanager
def profile_approval_lock(state_dir: Path) -> Iterator[None]:
    try:
        with site_operation_lock(state_dir, wait=False):
            yield
    except SiteOperationBusy as exc:
        raise ProfileApprovalError(
            "another administrator mutation is in progress"
        ) from exc


def write_profile_plan(state_dir: Path, document: dict[str, Any]) -> Path:
    plan, _digest = _validated_plan(document)
    path = profile_plan_path(state_dir)
    write_json_atomic(path, plan)
    return path


def clear_profile_plan(state_dir: Path) -> None:
    if profile_approval_path(state_dir).exists():
        raise ProfileApprovalError(
            "cannot clear a Profile plan while an approval is active"
        )
    _unlink(profile_plan_path(state_dir))


def _archive_initial_approval(
    state_dir: Path,
    *,
    plan: dict[str, Any],
    record: dict[str, Any],
) -> Path:
    digest = str(record["plan_sha256"])
    archive = profile_approval_archive_path(state_dir, digest)
    archived_plan = archive / "plan.json"
    archived_approval = archive / "approval.json"
    if archived_plan.exists():
        existing_plan, existing_digest = _validated_plan(
            _read_json(archived_plan, "archived Profile plan")
        )
        if existing_digest != digest or existing_plan != plan:
            raise ProfileApprovalError(
                "archived Profile plan differs from the active approval"
            )
    else:
        write_json_atomic(archived_plan, plan)
    if archived_approval.exists():
        existing_record = _validated_approval_record(
            _read_json(archived_approval, "archived Profile approval")
        )
        if existing_record != record:
            raise ProfileApprovalError(
                "archived Profile approval differs from the active approval"
            )
    else:
        write_json_atomic(archived_approval, record)
    return archive


def _write_superseded_result(
    state_dir: Path,
    *,
    record: dict[str, Any],
    replacement_plan_sha256: str | None,
    reason: str,
    superseded_at: datetime | None = None,
) -> Path:
    archive = profile_approval_archive_path(
        state_dir,
        str(record["plan_sha256"]),
    )
    result = {
        "schema_version": 1,
        "status": "SUPERSEDED",
        "plan_sha256": record["plan_sha256"],
        "reference": record["reference"],
        "approver_identity": _approver_identity(record),
        "replacement_plan_sha256": replacement_plan_sha256,
        "reason": reason,
        "superseded_at": _utc_timestamp(superseded_at),
    }
    path = archive / "superseded.json"
    if path.exists():
        existing = _read_json(path, "Profile approval supersession")
        if (
            existing.get("plan_sha256") != result["plan_sha256"]
            or existing.get("replacement_plan_sha256")
            != result["replacement_plan_sha256"]
        ):
            raise ProfileApprovalError(
                "Profile approval supersession audit is inconsistent"
            )
        return archive
    write_json_atomic(path, result)
    return archive


def approve_profile(
    state_dir: Path,
    *,
    reference: str,
    expected_plan_sha256: str,
    approved_at: datetime | None = None,
    approver_identity: str | None = None,
) -> dict[str, Any]:
    """Bind the operator's approval to the pending plan.

    ``approver_identity`` names who approved (I2): the STS caller ARN when not
    given, falling back to ``user@host`` so an approval is never anonymous. It
    lives on the record and in the archive but *outside* ``plan_sha256``: that
    digest binds the reviewed plan, is computed before anyone approves, and is
    what the operator types back on the rerun; folding the approver in would
    change it under them and make an approved plan unwinnable.

    Refusals name the pending plan's digest so the operator can see whether
    the plan they reviewed is still the one on disk.
    """

    normalized = reference.strip()
    if not APPROVAL_PATTERN.fullmatch(normalized):
        raise ProfileApprovalError("Profile approval reference has an invalid format")
    normalized_plan_sha256 = expected_plan_sha256.strip()
    if not SHA256_PATTERN.fullmatch(normalized_plan_sha256):
        raise ProfileApprovalError("reviewed Profile plan SHA-256 is invalid")
    approver = (approver_identity or "").strip() or (
        operator_identity.resolve_operator_identity(
            fallback=operator_identity.local_operator_identity()
        )
    )
    with profile_approval_lock(state_dir):
        plan_path = profile_plan_path(state_dir)
        if not plan_path.is_file():
            raise ProfileApprovalError(
                "no pending Profile plan; run gpu-fault-admin deploy first and "
                "review the release-deploy/profile-plan.json it writes"
            )
        plan, digest = _validated_plan(_read_json(plan_path, "Profile plan"))
        if digest != normalized_plan_sha256:
            raise ProfileApprovalError(
                f"pending Profile plan {digest} does not match the reviewed "
                f"plan SHA-256 {normalized_plan_sha256}; review "
                "release-deploy/profile-plan.json again and pass its plan_sha256"
            )
        if plan.get("approval_required") is not True:
            raise ProfileApprovalError("Profile plan does not require approval")
        approval_path = profile_approval_path(state_dir)
        if approval_path.is_file():
            existing = _validated_approval_record(
                _read_json(approval_path, "Profile approval record")
            )
            if existing["plan_sha256"] == digest:
                if existing["reference"] != normalized:
                    raise ProfileApprovalError(
                        "Profile plan already has a different approval reference"
                    )
                _archive_initial_approval(
                    state_dir,
                    plan=plan,
                    record=existing,
                )
                return existing
            _write_superseded_result(
                state_dir,
                record=existing,
                replacement_plan_sha256=digest,
                reason="a different pending Profile plan was approved",
                superseded_at=approved_at,
            )
        record = {
            "schema_version": 1,
            "reference": normalized,
            "plan_sha256": digest,
            "site_identity": dict(plan["site_identity"]),
            "site_identity_sha256": str(plan["site_identity_sha256"]),
            "policy_digest": str(plan.get("policy_digest") or ""),
            "desired_version": str(plan.get("desired_version") or ""),
            "change_kind": str(plan.get("change_kind") or ""),
            "approved_at": _utc_timestamp(approved_at),
            "approver_identity": approver,
        }
        _archive_initial_approval(
            state_dir,
            plan=plan,
            record=record,
        )
        write_json_atomic(approval_path, record)
        return record


def approve_profile_plan_inline(
    state_dir: Path,
    *,
    plan_sha256: str,
    reference: str,
) -> dict[str, Any]:
    """Approve the pending plan from the deploy rerun's own flags.

    Contract for ``gpu-fault-admin deploy --approve-profile-plan SHA
    --reference REF``: the CLI calls this *before* ``run_source_deploy``, with
    nothing but the ``--state-dir`` it was given -- no site load, no release
    preparation, no AWS lookups beyond the STS caller identity that names the
    approver. It performs exactly what the former ``approve-profile`` verb did
    (same operation lock, same approver identity, same
    ``profile-approvals/<plan_sha256>/`` archive, same SUPERSEDED handling of
    an earlier approval for a different plan), then returns the approval
    record. The deploy that follows resolves that record as ``EXACT`` and
    marks it ``CONSUMED`` on success; a plan that drifted in between is
    archived as ``SUPERSEDED`` and the deploy stops again with a new digest.

    The flag cannot be passed blind: with no ``release-deploy/profile-plan.json``
    under ``state_dir`` there is nothing to approve, and a ``plan_sha256`` that
    differs from the pending plan's digest is refused with a message naming
    the pending digest. Both checks happen under the lock inside
    :func:`approve_profile`, so the digest compared is the one on disk at
    approval time.

    Raises :class:`ProfileApprovalError` on every refusal.
    """

    normalized_plan_sha256 = plan_sha256.strip()
    if not SHA256_PATTERN.fullmatch(normalized_plan_sha256):
        raise ProfileApprovalError(
            "--approve-profile-plan must be the 64-hex plan_sha256 printed when "
            "the deploy stopped for Runtime Profile review"
        )
    return approve_profile(
        state_dir,
        reference=reference,
        expected_plan_sha256=normalized_plan_sha256,
    )


def _target_matches(
    current_plan: dict[str, Any],
    approved_plan: dict[str, Any],
) -> bool:
    return all(
        current_plan.get(field) == approved_plan.get(field)
        for field in PLAN_TARGET_FIELDS
    )


def resolve_profile_approval(
    state_dir: Path,
    *,
    current_plan: dict[str, Any],
) -> ResolvedProfileApproval | None:
    current, current_digest = _validated_plan(current_plan)
    approval_path = profile_approval_path(state_dir)
    if not approval_path.is_file():
        return None
    record = _validated_approval_record(
        _read_json(approval_path, "Profile approval record")
    )
    pending_path = profile_plan_path(state_dir)
    if not pending_path.is_file():
        raise ProfileApprovalError(
            "Profile approval record exists without its pending plan"
        )
    approved, approved_digest = _validated_plan(
        _read_json(pending_path, "Profile plan")
    )
    if approved_digest != record["plan_sha256"]:
        raise ProfileApprovalError(
            "Profile approval record does not match the pending plan"
        )
    _archive_initial_approval(state_dir, plan=approved, record=record)
    if current_digest == approved_digest:
        relation = "EXACT"
    elif (
        current.get("approval_required") is True
        and current.get("change_kind") == "UNKNOWN_BASELINE"
        and current.get("current_version") == approved.get("desired_version")
        and current.get("live_profile_sha256") == approved.get("live_profile_sha256")
        and current.get("active_source_sha256") == approved.get("snapshot_sha256")
        and _target_matches(current, approved)
    ):
        relation = "PREPARED_RESUME"
    elif (
        current.get("approval_required") is False
        and current.get("change_kind") == "UNCHANGED"
        and current.get("current_version") == approved.get("desired_version")
        and current.get("current_policy_digest") == approved.get("policy_digest")
        and current.get("live_profile_sha256") == approved.get("snapshot_sha256")
        and current.get("active_source_sha256") == approved.get("snapshot_sha256")
        and _target_matches(current, approved)
    ):
        relation = "ALREADY_APPLIED"
    else:
        raise StaleProfileApprovalError(
            "Profile approval is stale because the pending plan or live baseline changed"
        )
    return ResolvedProfileApproval(
        reference=str(record["reference"]),
        plan_sha256=approved_digest,
        relation=relation,
        approver_identity=_approver_identity(record),
    )


def supersede_profile_approval(
    state_dir: Path,
    *,
    replacement_plan: dict[str, Any] | None,
    reason: str,
    superseded_at: datetime | None = None,
) -> Path | None:
    replacement_digest: str | None = None
    if replacement_plan is not None:
        _replacement, replacement_digest = _validated_plan(replacement_plan)
    approval_path = profile_approval_path(state_dir)
    archive: Path | None = None
    if approval_path.is_file():
        record = _validated_approval_record(
            _read_json(approval_path, "Profile approval record")
        )
        pending, pending_digest = _validated_plan(
            _read_json(profile_plan_path(state_dir), "Profile plan")
        )
        if pending_digest != record["plan_sha256"]:
            raise ProfileApprovalError(
                "Profile approval record does not match the pending plan"
            )
        _archive_initial_approval(state_dir, plan=pending, record=record)
        archive = _write_superseded_result(
            state_dir,
            record=record,
            replacement_plan_sha256=replacement_digest,
            reason=reason,
            superseded_at=superseded_at,
        )
        _unlink(approval_path)
    if replacement_plan is None:
        _unlink(profile_plan_path(state_dir))
    else:
        write_profile_plan(state_dir, replacement_plan)
    return archive


def consume_profile_approval(
    state_dir: Path,
    *,
    expected_plan_sha256: str,
    release_id: str,
    relation: str,
    consumed_at: datetime | None = None,
) -> Path:
    if not RELEASE_ID_PATTERN.fullmatch(release_id):
        raise ProfileApprovalError("release ID is invalid")
    if relation not in {"EXACT", "PREPARED_RESUME", "ALREADY_APPLIED"}:
        raise ProfileApprovalError("Profile approval relation is invalid")
    root = state_dir.expanduser().resolve()
    approval_path = profile_approval_path(root)
    plan_path = profile_plan_path(root)
    record = _validated_approval_record(
        _read_json(approval_path, "Profile approval record")
    )
    if record.get("plan_sha256") != expected_plan_sha256:
        raise ProfileApprovalError(
            "Profile approval cannot be consumed for a different plan"
        )
    plan, digest = _validated_plan(_read_json(plan_path, "Profile plan"))
    if digest != expected_plan_sha256:
        raise ProfileApprovalError("Profile plan changed before approval consumption")
    archive = _archive_initial_approval(root, plan=plan, record=record)
    result_path = archive / "consumed.json"
    if result_path.exists():
        existing = _read_json(result_path, "Profile approval consumption")
        if (
            existing.get("plan_sha256") != expected_plan_sha256
            or existing.get("release_id") != release_id
        ):
            raise ProfileApprovalError(
                "Profile approval consumption audit is inconsistent"
            )
    else:
        write_json_atomic(
            result_path,
            {
                "schema_version": 1,
                "status": "CONSUMED",
                "plan_sha256": expected_plan_sha256,
                "reference": record["reference"],
                "approver_identity": _approver_identity(record),
                "release_id": release_id,
                "relation": relation,
                "consumed_at": _utc_timestamp(consumed_at),
            },
        )
    _unlink(approval_path)
    _unlink(plan_path)
    return archive

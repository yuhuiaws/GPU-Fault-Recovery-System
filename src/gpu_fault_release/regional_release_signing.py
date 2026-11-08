"""The admin CLI's half of ``--allow-unlogged-signature`` and the status view.

``scripts/signing_transparency.py`` signs and verifies (standard library only,
so the deploy-host setup can run it before any venv exists). The admin CLI
needs three things from it that it cannot import from ``scripts/``: the
variable the deploy hop hands its own children, the reason's rules, and how a
release record and a bundle say whether the release runs without a
transparency-log entry. They are spelled here, outside ``src/gpu_fault``: a
``GPU_FAULT_*`` literal there would enter ``env-inventory.json``, which every
component wheel ships, and roll the control plane and every node Agent for a
deploy-host change. ``tests/test_signing_transparency.py`` pins each pair equal.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

UNLOGGED_SIGNATURE_REASON_ENV = "GPU_FAULT_RELEASE_UNLOGGED_SIGNATURE_REASON"
ALLOW_UNLOGGED_FLAG = "--allow-unlogged-signature"
REASON_FLAG = "--reason"
REASON_MAX_CHARACTERS = 300
UNLOGGED = "unlogged"
#: After this long without a backfill, the release signed without a log entry
#: turns ``gpu-fault-admin status`` from a finding into a failing check.
UNLOGGED_ADVISORY_HOURS = 72
#: The pinned Sigstore trust root's age at which status warns (never fails).
PIN_AGE_WARNING_DAYS = 30
SIGNING_CHECK = "release_signature"
PIN_RECORD = "sigstore-trust-pin.json"
PIN_USE_RECORD = "sigstore-trust-pin-use.json"
RELEASE_DEPLOY_DIR = "release-deploy"


#: The signing wrapper's note beside a bundle it could not write because the
#: Sigstore services refused (make reports only its exit status).
REFUSAL_MARKER_SUFFIX = ".transparency-log-refused.json"


class UnloggedSignatureError(ValueError):
    pass


def refusal_marker(bundle: Path) -> Path:
    return bundle.with_name(bundle.name + REFUSAL_MARKER_SUFFIX)


def read_refusal_marker(bundle: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(refusal_marker(bundle).read_bytes())
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def validate_reason(value: object) -> str:
    """Same rules as ``signing_transparency.validate_reason``."""

    if not isinstance(value, str) or not value.strip():
        raise UnloggedSignatureError(
            f"{ALLOW_UNLOGGED_FLAG} requires {REASON_FLAG} saying why the release "
            "is signed without a transparency-log entry"
        )
    text = value.strip()
    if len(text) > REASON_MAX_CHARACTERS:
        raise UnloggedSignatureError(
            f"{REASON_FLAG} is longer than {REASON_MAX_CHARACTERS} characters"
        )
    if not text.isprintable():
        raise UnloggedSignatureError(
            f"{REASON_FLAG} must be one line of printable text"
        )
    return text


def backfill_exit(state_dir: object, *, what: str = "it") -> str:
    """Same words as ``signing_transparency.backfill_exit``: the next deploy
    without the switch backfills the signature first, the verb out of band."""

    return (
        f"the next deploy without {ALLOW_UNLOGGED_FLAG} backfills {what} before "
        "deploying once the transparency log answers (and refuses while it does "
        "not), or run gpu-fault-admin backfill-transparency-log --state-dir "
        f"{state_dir}"
    )


def _timestamp(value: object) -> datetime | None:
    try:
        return datetime.strptime(str(value), "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)
    except ValueError:
        return None


def bundle_has_log_entry(path: Path) -> bool | None:
    """Whether a Sigstore bundle carries a transparency-log entry; None if unreadable."""

    try:
        value = json.loads(path.read_bytes())
    except (OSError, ValueError):
        return None
    material = value.get("verificationMaterial") if isinstance(value, dict) else None
    if not isinstance(material, dict):
        return None
    entries = material.get("tlogEntries")
    return isinstance(entries, list) and bool(entries)


def _json_object(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def release_signing_view(
    state_dir: Path, *, release_id: str | None, now: datetime | None = None
) -> tuple[dict[str, Any] | None, list[str], dict[str, Any] | None]:
    """``(field, findings, failing check)`` for the release the site runs.

    The release record ``release-deploy/<id>/state.json`` names how its
    attestation was accepted (``scripts/release_deploy.py``); the bundle copy
    beside it says whether it is still without a log entry. A finding until
    ``UNLOGGED_ADVISORY_HOURS`` after the declaration, then a failing check,
    until the next deploy without the switch, or ``backfill-transparency-log``
    out of band, puts the signature in the log.
    """

    if not release_id or "/" in release_id:
        return None, [], None
    record = _json_object(state_dir / RELEASE_DEPLOY_DIR / release_id / "state.json")
    signing = (record or {}).get("signing")
    if not isinstance(signing, Mapping):
        return None, [], None
    field: dict[str, Any] = {
        "release_id": release_id,
        "transparency_log": signing.get("transparency_log"),
        "trusted_root": signing.get("trusted_root"),
    }
    if isinstance(signing.get("trust_pin"), Mapping):
        field["trust_pin"] = dict(signing["trust_pin"])
    if signing.get("transparency_log") != UNLOGGED:
        field["status"] = "LOGGED"
        return field, [], None
    field.update(
        reason=signing.get("reason"),
        declared_at=signing.get("declared_at"),
        approval=signing.get("approval"),
    )
    bundle = signing.get("bundle")
    logged = bundle_has_log_entry(Path(str(bundle))) if bundle else None
    if logged:
        field["status"] = "BACKFILLED"
        return field, [], None
    declared = _timestamp(signing.get("declared_at"))
    hours = (
        ((now or datetime.now(UTC)) - declared).total_seconds() / 3600
        if declared is not None
        else None
    )
    field["age_hours"] = round(hours, 1) if hours is not None else None
    line = (
        f"release {release_id} is signed without a transparency-log entry "
        f"(declared {signing.get('declared_at')}: {signing.get('reason')}); "
        + backfill_exit(state_dir)
    )
    overdue = hours is None or hours >= UNLOGGED_ADVISORY_HOURS
    field["status"] = "FAIL" if overdue else "ADVISORY"
    if not overdue:
        return field, [line], None
    check = {
        "name": SIGNING_CHECK,
        "status": "FAIL",
        "summary": f"{line} (more than {UNLOGGED_ADVISORY_HOURS} h)",
        "details": dict(field),
    }
    return field, [], check


SUCCESS_RECORD_SIGNATURE = "source-deploy-success.sigstore.json"


def unlogged_signatures_view(
    state_dir: Path, *, release_id: str | None
) -> tuple[list[str], list[str]]:
    """``(bundles, findings)``: the site key's other signatures still without a
    transparency-log entry.

    The running release's attestation is ``release_signing_view``'s; this names
    the rest a deploy with ``--allow-unlogged-signature`` signed -- the success
    record, the deploy-host archives, the CI receipts, other releases' copies --
    so a deploy-host-only deploy that changed no release still shows in
    ``status`` until the next deploy without the switch backfills them (it
    refuses while the log does not answer), or ``backfill-transparency-log``
    logs them out of band.
    """

    candidates = [
        state_dir / SUCCESS_RECORD_SIGNATURE,
        *sorted((state_dir / "deploy-host" / "by-content").glob("*/*.sigstore.json")),
        *sorted((state_dir / "ci-candidates").glob("*/verification.sigstore.json")),
        *(
            path
            for path in sorted(
                (state_dir / RELEASE_DEPLOY_DIR).glob("*/attestation.sigstore.json")
            )
            if path.parent.name != release_id
        ),
    ]
    unlogged = [path for path in candidates if bundle_has_log_entry(path) is False]
    if not unlogged:
        return [], []
    names = [str(path.relative_to(state_dir)) for path in unlogged]
    return names, [
        f"{len(names)} signature(s) of the site key have no transparency-log entry "
        f"({', '.join(names)}); " + backfill_exit(state_dir, what="them")
    ]


def trust_pin_view(
    state_dir: Path, *, now: datetime | None = None
) -> tuple[dict[str, Any] | None, list[str]]:
    """``(field, findings)`` of the pinned Sigstore trust root: its age, a WARN
    past ``PIN_AGE_WARNING_DAYS``. Never a failing check: the pin matters only
    while TUF does not answer, and every online deploy refreshes it."""

    directory = state_dir / "release-signing"
    record = _json_object(directory / PIN_RECORD)
    if record is None:
        return None, []
    pinned = _timestamp(record.get("pinned_at"))
    days = (
        ((now or datetime.now(UTC)) - pinned).total_seconds() / 86400
        if pinned is not None
        else None
    )
    field = {
        "pinned_at": record.get("pinned_at"),
        "tuf_last_timestamp": record.get("tuf_last_timestamp"),
        "age_days": round(days, 1) if days is not None else None,
    }
    use = _json_object(directory / PIN_USE_RECORD)
    if use is not None:
        field["last_use"] = use
    if days is not None and days <= PIN_AGE_WARNING_DAYS:
        return field, []
    return field, [
        f"WARN sigstore trust pin: the site-pinned Sigstore trust root is "
        f"{field['age_days']} days old (pinned {record.get('pinned_at')}); the "
        "next online deploy refreshes it"
    ]


def with_failing_check(
    report: dict[str, Any], check: Mapping[str, Any]
) -> dict[str, Any]:
    """``report`` with one more failing health check, ``healthy`` false."""

    health = report.get("health")
    health = dict(health) if isinstance(health, dict) else {}
    checks = [
        item
        for item in (health.get("checks") or [])
        if not (isinstance(item, dict) and item.get("name") == check.get("name"))
    ]
    health["checks"] = [*checks, dict(check)]
    summary = health.get("summary")
    summary = dict(summary) if isinstance(summary, dict) else {}
    summary["FAIL"] = int(summary.get("FAIL") or 0) + 1
    health["summary"] = summary
    health["healthy"] = False
    return {**report, "health": health, "healthy": False}

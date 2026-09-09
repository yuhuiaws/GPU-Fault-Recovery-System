"""Pure verdict functions and constants of GF-REGIONAL-BOOT-023.

The case proves ARCH-H2/H3/H5 on a real regional site through the sanctioned
release path: one NOOP release (the same ``RegionalRelease.noop`` that
``gpu-fault-admin deploy`` reaches when the diff classifies as NOOP) must
append exactly well-formed entries to the append-only
``gpu-fault-release-history`` ConfigMap and its admin-side mirror -- release
id, phase, digests, operator identity, redacted command -- without rewriting
what was there; the previous-snapshot ConfigMaps stay within the retention
bound and are untouched by a NOOP; every CPU Pod's durable registry head
digest equals the digest of the Secret it started from (``secret_drift`` is
false on ``/healthz``); and a manifest that still claims
``database.rollback_compatible: true`` is refused at plan time with the
explanatory error rather than being deployed.

Everything here judges plain dictionaries the runner recorded, so the unit
suite can drive every branch without a cluster.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Any

CASE_ID = "GF-REGIONAL-BOOT-023"
PREDECESSOR_CASE_ID = "GF-REGIONAL-BOOT-020"
CONFIRMATION = "BOOT023_RELEASE_HISTORY_REGISTRY_PUBLISH"
EXPECTED_CLASSIFICATION = "NOOP"
NOOP_PHASE = "complete"
HISTORY_CONFIG_MAP = "gpu-fault-release-history"
HISTORY_KEY = "history.ndjson"
HISTORY_MAX_ENTRIES = 200
HISTORY_MIRROR_FILE = "history.ndjson"
PREVIOUS_SNAPSHOT_LABEL = "gpu-fault.io/release-previous-snapshot"
PREVIOUS_SNAPSHOT_DIGEST_ANNOTATION = "gpu-fault.io/snapshot-sha256"
PREVIOUS_SNAPSHOTS_RETAINED = 3
HISTORY_ENTRY_FIELDS = (
    "timestamp",
    "release_id",
    "phase",
    "release_lifecycle",
    "state_sha256",
    "plan_sha256",
    "operator",
    "command",
)
HEX_DIGEST = re.compile(r"^[0-9a-f]{64}$")
# A credential-named option whose value survived redaction, or a token query
# string, is a leak; ``<redacted>`` is what the history writer substitutes.
UNREDACTED_CREDENTIAL = re.compile(
    r"(?i)(?:^|[\s'\"])--?[a-z0-9-]*(?:token|secret|password|credential|key)"
    r"(?:=|\s+)(?!<redacted>)[^\s'\"]+"
)
TOKEN_QUERY = re.compile(r"(?i)[?&]token=")
ROLLBACK_FLAG_MESSAGE_TERMS = ("rollback_compatible", "exact", "accept-schema-change")


def parse_time(value: Any) -> datetime | None:
    if not value:
        return None
    if isinstance(value, datetime):
        parsed = value
    else:
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except ValueError:
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def parse_history(raw: str) -> list[dict[str, Any]]:
    """The ndjson body of the history ConfigMap as a list of entries.

    A line that is not a JSON object is kept as ``{"_unparsed": line}`` so the
    verdict can name it instead of silently shortening the history.
    """

    entries: list[dict[str, Any]] = []
    for line in str(raw or "").splitlines():
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            entries.append({"_unparsed": line})
            continue
        entries.append(value if isinstance(value, dict) else {"_unparsed": line})
    return entries


def snapshot_groups(items: list[dict[str, Any]]) -> list[list[str]]:
    """Previous-snapshot ConfigMap names grouped per snapshot, newest first.

    Chunks of one snapshot share the digest annotation; the name prefix before
    the chunk index is the fallback for chunks written before the annotation.
    """

    groups: dict[str, tuple[str, list[str]]] = {}
    for item in items:
        metadata = item.get("metadata") or {}
        name = str(metadata.get("name") or "")
        if not name:
            continue
        annotations = metadata.get("annotations") or {}
        digest = str(annotations.get(PREVIOUS_SNAPSHOT_DIGEST_ANNOTATION) or "")
        key = digest or name.rsplit("-", 1)[0]
        created = str(metadata.get("creationTimestamp") or "")
        existing = groups.get(key)
        if existing is None:
            groups[key] = (created, [name])
        else:
            groups[key] = (max(existing[0], created), [*existing[1], name])
    return [
        sorted(names)
        for _created, names in sorted(
            groups.values(), key=lambda value: value[0], reverse=True
        )
    ]


# --------------------------------------------------------------------------- #
# Preflight
# --------------------------------------------------------------------------- #
def preflight_errors(
    *,
    classification: dict[str, Any],
    release_id: str,
    state_phase: Any,
    probes: list[dict[str, Any]],
    rollback_flag_message: str,
    predecessor_valid: bool,
    tests_passed: bool,
    history_before: list[dict[str, Any]],
    snapshot_groups_before: list[list[str]],
) -> list[str]:
    errors = classification_errors(classification)
    if not release_id:
        errors.append("the live release state carries no release_id")
    if state_phase != NOOP_PHASE:
        errors.append(
            f"the live release state is in phase {state_phase!r}, not "
            f"{NOOP_PHASE!r}; a NOOP release needs a completed transaction"
        )
    errors.extend(registry_probe_errors(probes))
    errors.extend(rollback_flag_errors(rollback_flag_message))
    if not predecessor_valid:
        errors.append(f"{PREDECESSOR_CASE_ID} predecessor evidence is not PASS")
    if not tests_passed:
        errors.append("focused regression tests failed")
    unparsed = [entry for entry in history_before if "_unparsed" in entry]
    if unparsed:
        errors.append(
            f"the existing release history carries {len(unparsed)} unparsable lines"
        )
    if len(history_before) > HISTORY_MAX_ENTRIES:
        errors.append(
            f"the existing release history holds {len(history_before)} entries, "
            f"above the {HISTORY_MAX_ENTRIES} bound"
        )
    if len(snapshot_groups_before) > PREVIOUS_SNAPSHOTS_RETAINED:
        errors.append(
            f"{len(snapshot_groups_before)} previous snapshots are retained, "
            f"above the {PREVIOUS_SNAPSHOTS_RETAINED} bound"
        )
    return errors


def classification_errors(classification: dict[str, Any]) -> list[str]:
    kind = classification.get("kind")
    if kind != EXPECTED_CLASSIFICATION:
        return [
            f"the release config classifies as {kind!r} against the live site, "
            f"not {EXPECTED_CLASSIFICATION}; this case only runs a NOOP release "
            f"(changed: {sorted(classification.get('changed') or [])})"
        ]
    return []


# --------------------------------------------------------------------------- #
# Release history (ARCH-H5)
# --------------------------------------------------------------------------- #
def redacted_command_errors(command: Any) -> list[str]:
    if not isinstance(command, str) or not command.strip():
        return ["history entry carries no command line"]
    errors: list[str] = []
    if UNREDACTED_CREDENTIAL.search(command):
        errors.append("history entry command line carries an unredacted credential")
    if TOKEN_QUERY.search(command):
        errors.append("history entry command line carries a token query string")
    return errors


def history_entry_errors(
    entry: dict[str, Any],
    *,
    release_id: str,
    label: str,
) -> list[str]:
    errors: list[str] = []
    if "_unparsed" in entry:
        return [f"{label}: history line is not a JSON object"]
    missing = [name for name in HISTORY_ENTRY_FIELDS if name not in entry]
    if missing:
        errors.append(f"{label}: history entry lacks fields {missing}")
    if parse_time(entry.get("timestamp")) is None:
        errors.append(f"{label}: history entry timestamp is not ISO-8601")
    if entry.get("release_id") != release_id:
        errors.append(
            f"{label}: history entry names release {entry.get('release_id')!r}, "
            f"not the live release {release_id!r}"
        )
    if not isinstance(entry.get("phase"), str) or not entry.get("phase"):
        errors.append(f"{label}: history entry has no phase")
    state_digest = entry.get("state_sha256")
    if not isinstance(state_digest, str) or HEX_DIGEST.fullmatch(state_digest) is None:
        errors.append(f"{label}: state_sha256 is not a sha256 hex digest")
    plan_digest = entry.get("plan_sha256")
    if plan_digest is not None and (
        not isinstance(plan_digest, str) or HEX_DIGEST.fullmatch(plan_digest) is None
    ):
        errors.append(f"{label}: plan_sha256 is neither null nor a sha256 digest")
    operator = entry.get("operator")
    if not isinstance(operator, str) or not operator.strip():
        errors.append(f"{label}: history entry names no operator")
    errors.extend(
        f"{label}: {item}" for item in redacted_command_errors(entry.get("command"))
    )
    return errors


def appended_history_entries(
    before: list[dict[str, Any]],
    after: list[dict[str, Any]],
    *,
    bound: int = HISTORY_MAX_ENTRIES,
) -> list[dict[str, Any]] | None:
    """The entries ``after`` gained over ``before``; ``None`` if it was rewritten.

    The history is a ring of ``bound`` entries: once full, every append drops
    the oldest, so the count stops growing and the retained part of ``after``
    is a *suffix* of ``before``, not the whole of it. Counting entries read a
    full ring as "did not grow" on the first live run (2026-09-09, 200 -> 200).
    Align the longest suffix of ``before`` with the prefix of ``after``; entries
    may only disappear from the front, and only when the ring is full.
    """

    for kept in range(min(len(before), len(after)), 0, -1):
        if after[:kept] != before[len(before) - kept :]:
            continue
        dropped = len(before) - kept
        if dropped and len(after) < bound:
            return None
        return after[kept:]
    return list(after) if not before else None


def history_append_errors(
    before: list[dict[str, Any]],
    after: list[dict[str, Any]],
    *,
    release_id: str,
    expected_phase: str = NOOP_PHASE,
) -> list[str]:
    """The history grew, and only grew: retained entries unchanged, new ones valid."""

    errors: list[str] = []
    appended = appended_history_entries(before, after)
    if appended is None:
        return ["the release history prefix was rewritten; it must be append-only"]
    if not appended:
        errors.append(
            f"the release history did not grow: {len(before)} -> {len(after)} entries"
        )
        return errors
    if len(after) > HISTORY_MAX_ENTRIES:
        errors.append(
            f"the release history holds {len(after)} entries, above the "
            f"{HISTORY_MAX_ENTRIES} bound"
        )
    for index, entry in enumerate(appended):
        errors.extend(
            history_entry_errors(
                entry, release_id=release_id, label=f"appended entry {index}"
            )
        )
    if appended[-1].get("phase") != expected_phase:
        errors.append(
            f"the newest history entry records phase {appended[-1].get('phase')!r}, "
            f"not {expected_phase!r}"
        )
    return errors


def mirror_errors(
    mirror_lines: list[str],
    appended: list[dict[str, Any]],
) -> list[str]:
    """The admin-side ndjson mirror carries every appended entry, in order."""

    if not appended:
        return ["no appended entries to mirror"]
    parsed = parse_history("\n".join(mirror_lines))
    if len(parsed) < len(appended):
        return [
            f"the history mirror holds {len(parsed)} entries, fewer than the "
            f"{len(appended)} appended to the ConfigMap"
        ]
    tail = parsed[-len(appended) :]
    if tail != appended:
        return ["the history mirror tail differs from the ConfigMap's appended entries"]
    return []


def snapshot_retention_errors(
    groups_before: list[list[str]],
    groups_after: list[list[str]],
    *,
    retained: int = PREVIOUS_SNAPSHOTS_RETAINED,
) -> list[str]:
    errors: list[str] = []
    if len(groups_after) > retained:
        errors.append(
            f"{len(groups_after)} previous snapshots remain after the release, "
            f"above the {retained} retained"
        )
    if groups_after != groups_before:
        errors.append(
            "a NOOP release changed the previous-snapshot ConfigMaps: "
            f"{groups_before} -> {groups_after}"
        )
    return errors


# --------------------------------------------------------------------------- #
# Registry durable head vs Secret (ARCH-H3)
# --------------------------------------------------------------------------- #
def registry_probe_errors(probes: list[dict[str, Any]]) -> list[str]:
    if not probes:
        return ["no CPU Pod answered the registry probe"]
    errors: list[str] = []
    for probe in probes:
        label = str(probe.get("pod") or probe.get("hostname") or "pod")
        secret = probe.get("secret_config_sha256")
        durable = probe.get("durable_config_sha256")
        for name, value in (("secret", secret), ("durable", durable)):
            if not isinstance(value, str) or HEX_DIGEST.fullmatch(value) is None:
                errors.append(f"{label}: {name} registry digest is not a sha256")
        if secret != durable:
            errors.append(
                f"{label}: durable registry head digest {durable} differs from the "
                f"Secret digest {secret}; the release did not publish durably"
            )
        healthz = probe.get("healthz") or {}
        if healthz.get("status") != 200:
            errors.append(f"{label}: /healthz returned {healthz.get('status')}")
        payload = healthz.get("payload") or {}
        registry = payload.get("regional_registry") or {}
        if registry.get("secret_drift") is not False:
            errors.append(
                f"{label}: /healthz reports secret_drift={registry.get('secret_drift')!r}"
            )
        if registry.get("ready") is not True:
            errors.append(f"{label}: /healthz reports the registry not ready")
        if registry.get("secret_config_sha256") not in (None, secret):
            errors.append(
                f"{label}: /healthz secret_config_sha256 differs from the probe's"
            )
        livez = probe.get("livez") or {}
        if livez.get("status") != 200:
            errors.append(f"{label}: /livez returned {livez.get('status')}")
    return errors


def registry_stability_errors(
    before: list[dict[str, Any]],
    after: list[dict[str, Any]],
) -> list[str]:
    """A NOOP release publishes nothing new: the head generation is unchanged."""

    generations_before = {
        str(item.get("pod")): item.get("head_generation") for item in before
    }
    generations_after = {
        str(item.get("pod")): item.get("head_generation") for item in after
    }
    if generations_before != generations_after:
        return [
            "the durable registry head generation moved across a NOOP release: "
            f"{generations_before} -> {generations_after}"
        ]
    return []


# --------------------------------------------------------------------------- #
# Manifest rollback flag (ARCH-H2)
# --------------------------------------------------------------------------- #
def rollback_flag_errors(message: str) -> list[str]:
    if not message:
        return [
            "a manifest declaring database.rollback_compatible: true was accepted; "
            "the release config must refuse it at plan time"
        ]
    missing = [term for term in ROLLBACK_FLAG_MESSAGE_TERMS if term not in message]
    if missing:
        return [
            "the rollback_compatible refusal does not explain itself; missing "
            f"{missing} in: {message[:200]}"
        ]
    return []

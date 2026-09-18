"""Digest-only inspection of literal and JSON-contained credentials."""

from __future__ import annotations

import hashlib
import json
from typing import Any

MAX_VALUE_BYTES = 2 * 1024 * 1024
MAX_VALUES = 100_000
MAX_DEPTH = 64


class CredentialScanError(ValueError):
    """The inventory cannot establish absence; never include the input value."""


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CredentialScanError("credential JSON contains duplicate fields")
        result[key] = value
    return result


def credential_value_digests(raw: bytes, *, require_json: bool = False) -> set[str]:
    """Inspect actual bytes and every JSON string, without returning their contents.

    JSON keys are inspected too: both keys and values can carry copied material.
    Nested JSON strings are decoded with the same structural budget. This is not
    an arbitrary encoding detector or proof about unobserved host files.
    """

    if len(raw) > MAX_VALUE_BYTES:
        raise CredentialScanError("credential value exceeds the inspection limit")
    pending: list[tuple[Any, int, bool]] = [(raw, 0, require_json)]
    result: set[str] = set()
    visited = 0
    while pending:
        value, depth, required = pending.pop()
        visited += 1
        if depth > MAX_DEPTH or visited > MAX_VALUES:
            raise CredentialScanError("credential JSON exceeds the structural limit")
        if isinstance(value, dict):
            for key, item in value.items():
                pending.extend(((key, depth + 1, False), (item, depth + 1, False)))
            continue
        if isinstance(value, list):
            pending.extend((item, depth + 1, False) for item in value)
            continue
        if isinstance(value, str):
            value = value.encode("utf-8")
        if not isinstance(value, bytes):
            continue
        result.update(
            hashlib.sha256(candidate).hexdigest()
            for candidate in (value, value.strip())
        )
        stripped = value.strip()
        if not required and stripped[:1] not in {b"{", b"[", b'"'}:
            continue
        try:
            decoded = json.loads(value, object_pairs_hook=_unique_object)
        except (ValueError, UnicodeError, RecursionError):
            raise CredentialScanError(
                "credential container is not valid JSON"
            ) from None
        pending.append((decoded, depth + 1, False))
    return result

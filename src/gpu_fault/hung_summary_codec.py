"""Lossless bounded envelopes keep full-rank evidence within RPC budgets."""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import zlib
from typing import Any

SUMMARY_VERSION = "hung-summary/v1"
MAX_NODE_BYTES = 1024 * 1024
MAX_SURVEY_RAW_BYTES = 32 * 1024 * 1024
ENCODING = "hung-summary-zlib-json/v1"


def encode_summary(value: dict[str, Any]) -> dict[str, Any]:
    raw = json.dumps(value, separators=(",", ":"), allow_nan=False).encode()
    if len(raw) > MAX_NODE_BYTES:
        raise ValueError("node summary exceeds decoded size limit")
    return {
        "encoding": ENCODING,
        "raw_bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "data": base64.b64encode(zlib.compress(raw, level=1)).decode("ascii"),
    }


def decode_summary(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("summary must be an object")
    if "encoding" not in value:
        if len(json.dumps(value, separators=(",", ":")).encode()) > MAX_NODE_BYTES:
            raise ValueError("node summary exceeds decoded size limit")
        return value
    try:
        if (
            value["encoding"] != ENCODING
            or type(value["raw_bytes"]) is not int
            or not 0 < value["raw_bytes"] <= MAX_NODE_BYTES
        ):
            raise ValueError("invalid summary envelope")
        encoded = value["data"]
        if not isinstance(encoded, str) or len(encoded) > 2 * MAX_NODE_BYTES:
            raise ValueError("encoded summary exceeds limit")
        packed = base64.b64decode(encoded, validate=True)
        decoder = zlib.decompressobj()
        raw = decoder.decompress(packed, MAX_NODE_BYTES + 1)
        if (
            not decoder.eof
            or decoder.unconsumed_tail
            or decoder.unused_data
            or len(raw) != value["raw_bytes"]
            or len(raw) > MAX_NODE_BYTES
            or hashlib.sha256(raw).hexdigest() != value["sha256"]
        ):
            raise ValueError("invalid or oversized summary payload")
        document = json.loads(raw)
        if not isinstance(document, dict):
            raise ValueError("decoded summary is not an object")
        return document
    except (
        KeyError,
        TypeError,
        binascii.Error,
        zlib.error,
        UnicodeError,
        RecursionError,
    ) as exc:
        raise ValueError("invalid summary encoding") from exc

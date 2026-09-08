from __future__ import annotations

import base64
import gzip
import hashlib
import json
from collections.abc import Callable, Mapping
from typing import Any

from gpu_fault.digests import SHA256_PATTERN

PREVIOUS_SNAPSHOT_SCHEMA_VERSION = 1
PREVIOUS_SNAPSHOT_ENCODING = "canonical-json+gzip"
PREVIOUS_SNAPSHOT_CHUNK_KEY = "snapshot.part"
PREVIOUS_SNAPSHOT_CONFIG_MAP_PREFIX = "gpu-fault-release-previous"
PREVIOUS_SNAPSHOT_CHUNK_BYTES = 512 * 1024


class ReleaseStateSnapshotError(ValueError):
    pass


def canonical_previous_bytes(previous: Mapping[str, Any]) -> bytes:
    return json.dumps(
        previous,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()


def encode_previous_snapshot(
    previous: Mapping[str, Any],
) -> tuple[dict[str, Any], list[tuple[str, bytes]]]:
    raw = canonical_previous_bytes(previous)
    digest = hashlib.sha256(raw).hexdigest()
    compressed = gzip.compress(raw, mtime=0)
    compressed_digest = hashlib.sha256(compressed).hexdigest()
    chunks: list[tuple[str, bytes]] = []
    chunk_refs = []
    for index, offset in enumerate(
        range(0, len(compressed), PREVIOUS_SNAPSHOT_CHUNK_BYTES)
    ):
        chunk = compressed[offset : offset + PREVIOUS_SNAPSHOT_CHUNK_BYTES]
        name = f"{PREVIOUS_SNAPSHOT_CONFIG_MAP_PREFIX}-{digest[:20]}-{index:03d}"
        chunks.append((name, chunk))
        chunk_refs.append(
            {
                "index": index,
                "config_map": name,
                "key": PREVIOUS_SNAPSHOT_CHUNK_KEY,
                "size_bytes": len(chunk),
                "sha256": hashlib.sha256(chunk).hexdigest(),
            }
        )
    if not chunks:
        raise ReleaseStateSnapshotError("previous snapshot cannot be empty")
    reference = {
        "schema_version": PREVIOUS_SNAPSHOT_SCHEMA_VERSION,
        "encoding": PREVIOUS_SNAPSHOT_ENCODING,
        "sha256": digest,
        "size_bytes": len(raw),
        "compressed_sha256": compressed_digest,
        "compressed_size_bytes": len(compressed),
        "chunks": chunk_refs,
    }
    return reference, chunks


def _chunk_bytes(document: Mapping[str, Any], key: str) -> bytes:
    binary = document.get("binaryData")
    encoded = binary.get(key) if isinstance(binary, Mapping) else None
    if not isinstance(encoded, str) or not encoded:
        raise ReleaseStateSnapshotError("previous snapshot ConfigMap chunk is missing")
    try:
        return base64.b64decode(encoded, validate=True)
    except ValueError as exc:
        raise ReleaseStateSnapshotError(
            "previous snapshot ConfigMap chunk is invalid"
        ) from exc


def validate_snapshot_config_map(
    document: Mapping[str, Any],
    *,
    key: str,
    expected_sha256: str,
    expected_size: int,
) -> bytes:
    chunk = _chunk_bytes(document, key)
    if len(chunk) != expected_size:
        raise ReleaseStateSnapshotError("previous snapshot chunk size changed")
    if hashlib.sha256(chunk).hexdigest() != expected_sha256:
        raise ReleaseStateSnapshotError("previous snapshot chunk digest changed")
    return chunk


def _validated_reference(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ReleaseStateSnapshotError("previous snapshot reference is invalid")
    if value.get("schema_version") != PREVIOUS_SNAPSHOT_SCHEMA_VERSION:
        raise ReleaseStateSnapshotError("previous snapshot schema is unsupported")
    if value.get("encoding") != PREVIOUS_SNAPSHOT_ENCODING:
        raise ReleaseStateSnapshotError("previous snapshot encoding is unsupported")
    for field in ("sha256", "compressed_sha256"):
        if not SHA256_PATTERN.fullmatch(str(value.get(field) or "")):
            raise ReleaseStateSnapshotError(f"previous snapshot {field} is invalid")
    for field in ("size_bytes", "compressed_size_bytes"):
        if not isinstance(value.get(field), int) or int(value[field]) < 1:
            raise ReleaseStateSnapshotError(f"previous snapshot {field} is invalid")
    chunks = value.get("chunks")
    if not isinstance(chunks, list) or not chunks:
        raise ReleaseStateSnapshotError("previous snapshot chunk list is invalid")
    for index, chunk in enumerate(chunks):
        if (
            not isinstance(chunk, dict)
            or chunk.get("index") != index
            or not str(chunk.get("config_map") or "")
            or chunk.get("key") != PREVIOUS_SNAPSHOT_CHUNK_KEY
            or not isinstance(chunk.get("size_bytes"), int)
            or int(chunk["size_bytes"]) < 1
            or not SHA256_PATTERN.fullmatch(str(chunk.get("sha256") or ""))
        ):
            raise ReleaseStateSnapshotError(
                "previous snapshot chunk reference is invalid"
            )
    return value


def hydrate_previous_snapshot(
    state: Mapping[str, Any],
    read_config_map: Callable[[str], Mapping[str, Any]],
) -> dict[str, Any]:
    hydrated = dict(state)
    reference_value = hydrated.get("previous_snapshot")
    inline = hydrated.get("previous")
    if reference_value is None:
        if inline is None:
            return hydrated
        if not isinstance(inline, dict):
            raise ReleaseStateSnapshotError("inline previous snapshot is invalid")
        # An inline snapshot with no reference is otherwise trusted verbatim. If
        # the release state recorded a digest for it, bind the record to that
        # digest so a tampered inline ``previous`` is rejected rather than
        # carried forward as the rollback baseline.
        recorded_digest = str(hydrated.get("previous_snapshot_sha256") or "")
        if recorded_digest:
            if not SHA256_PATTERN.fullmatch(recorded_digest):
                raise ReleaseStateSnapshotError("previous snapshot sha256 is invalid")
            if hashlib.sha256(canonical_previous_bytes(inline)).hexdigest() != (
                recorded_digest
            ):
                raise ReleaseStateSnapshotError(
                    "inline previous snapshot digest changed"
                )
        return hydrated
    reference = _validated_reference(reference_value)
    expected_digest = str(reference["sha256"])
    state_digest = str(hydrated.get("previous_snapshot_sha256") or "")
    if state_digest and state_digest != expected_digest:
        raise ReleaseStateSnapshotError(
            "previous snapshot reference disagrees with release state"
        )
    if isinstance(inline, dict):
        if hashlib.sha256(canonical_previous_bytes(inline)).hexdigest() != (
            expected_digest
        ):
            raise ReleaseStateSnapshotError("inline previous snapshot digest changed")
        return hydrated
    if inline is not None:
        raise ReleaseStateSnapshotError("inline previous snapshot is invalid")
    compressed_parts = []
    for chunk_ref in reference["chunks"]:
        document = read_config_map(str(chunk_ref["config_map"]))
        compressed_parts.append(
            validate_snapshot_config_map(
                document,
                key=str(chunk_ref["key"]),
                expected_sha256=str(chunk_ref["sha256"]),
                expected_size=int(chunk_ref["size_bytes"]),
            )
        )
    compressed = b"".join(compressed_parts)
    if len(compressed) != int(reference["compressed_size_bytes"]):
        raise ReleaseStateSnapshotError("previous snapshot compressed size changed")
    if hashlib.sha256(compressed).hexdigest() != reference["compressed_sha256"]:
        raise ReleaseStateSnapshotError("previous snapshot compressed digest changed")
    try:
        raw = gzip.decompress(compressed)
    except (OSError, EOFError) as exc:
        raise ReleaseStateSnapshotError("previous snapshot gzip is invalid") from exc
    if len(raw) != int(reference["size_bytes"]):
        raise ReleaseStateSnapshotError("previous snapshot size changed")
    if hashlib.sha256(raw).hexdigest() != expected_digest:
        raise ReleaseStateSnapshotError("previous snapshot digest changed")
    try:
        previous = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ReleaseStateSnapshotError("previous snapshot JSON is invalid") from exc
    if not isinstance(previous, dict):
        raise ReleaseStateSnapshotError("previous snapshot must be an object")
    if canonical_previous_bytes(previous) != raw:
        raise ReleaseStateSnapshotError("previous snapshot is not canonical JSON")
    hydrated["previous"] = previous
    return hydrated

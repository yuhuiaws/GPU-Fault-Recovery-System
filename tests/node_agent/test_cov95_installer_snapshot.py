from __future__ import annotations

import base64
import copy
import gzip
import hashlib

import pytest

from gpu_fault.release_state_snapshot import (
    PREVIOUS_SNAPSHOT_CHUNK_KEY,
    ReleaseStateSnapshotError,
    canonical_previous_bytes,
    encode_previous_snapshot,
    hydrate_previous_snapshot,
    validate_snapshot_config_map,
)


def snapshot():
    previous = {"release_id": "previous", "clusters": {"local": {"generation": 1}}}
    reference, chunks = encode_previous_snapshot(previous)
    documents = {
        name: {
            "binaryData": {PREVIOUS_SNAPSHOT_CHUNK_KEY: base64.b64encode(data).decode()}
        }
        for name, data in chunks
    }
    return previous, reference, documents


@pytest.mark.parametrize(
    "value,message",
    [
        ([], "reference is invalid"),
        ({"schema_version": 0}, "schema is unsupported"),
        ({"encoding": "raw-json"}, "encoding is unsupported"),
        ({"sha256": "invalid"}, "sha256 is invalid"),
        ({"compressed_sha256": "invalid"}, "compressed_sha256 is invalid"),
        ({"size_bytes": None}, "size_bytes is invalid"),
        ({"size_bytes": 0}, "size_bytes is invalid"),
        ({"compressed_size_bytes": -1}, "compressed_size_bytes is invalid"),
        ({"chunks": None}, "chunk list is invalid"),
        ({"chunks": []}, "chunk list is invalid"),
    ],
)
def test_snapshot_reference_is_validated_before_any_read(value, message) -> None:
    _, reference, _ = snapshot()
    value = {**reference, **value} if isinstance(value, dict) else value
    calls = []
    with pytest.raises(ReleaseStateSnapshotError, match=message):
        hydrate_previous_snapshot(
            {"previous_snapshot": value}, lambda name: calls.append(name)
        )
    assert calls == []


@pytest.mark.parametrize(
    "changes",
    [
        None,
        {"index": 1},
        {"config_map": ""},
        {"key": "different"},
        {"size_bytes": "1"},
        {"size_bytes": 0},
        {"sha256": "invalid"},
    ],
)
def test_invalid_chunk_reference_is_rejected_before_read(changes) -> None:
    _, reference, _ = snapshot()
    reference["chunks"][0] = (
        None if changes is None else {**reference["chunks"][0], **changes}
    )
    calls = []
    with pytest.raises(ReleaseStateSnapshotError, match="chunk reference is invalid"):
        hydrate_previous_snapshot(
            {"previous_snapshot": reference}, lambda name: calls.append(name)
        )
    assert calls == []


@pytest.mark.parametrize(
    "document,message",
    [
        ({}, "chunk is missing"),
        ({"binaryData": []}, "chunk is missing"),
        ({"binaryData": {"snapshot.part": ""}}, "chunk is missing"),
        ({"binaryData": {"snapshot.part": "!!!"}}, "chunk is invalid"),
        (
            {"binaryData": {"snapshot.part": base64.b64encode(b"bad").decode()}},
            "size changed",
        ),
        (
            {"binaryData": {"snapshot.part": base64.b64encode(b"abcd").decode()}},
            "digest changed",
        ),
    ],
)
def test_chunk_body_must_match_both_size_and_digest(document, message) -> None:
    with pytest.raises(ReleaseStateSnapshotError, match=message):
        validate_snapshot_config_map(
            document,
            key=PREVIOUS_SNAPSHOT_CHUNK_KEY,
            expected_sha256=hashlib.sha256(b"data").hexdigest(),
            expected_size=4,
        )


@pytest.mark.parametrize("with_reference", [True, False])
def test_invalid_inline_snapshot_never_falls_through_to_external_reads(with_reference):
    _, reference, _ = snapshot()
    state = {"previous": []}
    if with_reference:
        state["previous_snapshot"] = reference
    calls = []
    with pytest.raises(
        ReleaseStateSnapshotError, match="inline previous snapshot is invalid"
    ):
        hydrate_previous_snapshot(state, lambda name: calls.append(name))
    assert calls == []


@pytest.mark.parametrize("digest", ["invalid", "f" * 64])
def test_inline_recorded_digest_must_be_valid_and_match(digest) -> None:
    previous, _, _ = snapshot()
    with pytest.raises(
        ReleaseStateSnapshotError, match="sha256 is invalid|digest changed"
    ):
        hydrate_previous_snapshot(
            {"previous": previous, "previous_snapshot_sha256": digest},
            lambda name: pytest.fail("inline validation attempted a ConfigMap read"),
        )


def test_no_snapshot_and_already_hydrated_snapshot_need_no_reads() -> None:
    previous, reference, _ = snapshot()
    calls = []
    state = {"phase": "failed"}
    assert hydrate_previous_snapshot(state, lambda name: calls.append(name)) == state
    state.update(previous=previous, previous_snapshot=reference)
    hydrated = hydrate_previous_snapshot(state, lambda name: calls.append(name))
    assert hydrated == state
    assert hydrated is not state
    assert calls == []
    state["previous"] = {"release_id": "changed"}
    with pytest.raises(
        ReleaseStateSnapshotError, match="inline previous snapshot digest changed"
    ):
        hydrate_previous_snapshot(state, lambda name: calls.append(name))


def test_reference_and_state_digest_cannot_disagree() -> None:
    _, reference, documents = snapshot()
    with pytest.raises(ReleaseStateSnapshotError, match="disagrees with release state"):
        hydrate_previous_snapshot(
            {"previous_snapshot": reference, "previous_snapshot_sha256": "f" * 64},
            documents.__getitem__,
        )


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("compressed_size_bytes", 1, "compressed size changed"),
        ("compressed_sha256", "f" * 64, "compressed digest changed"),
        ("size_bytes", 1, "snapshot size changed"),
        ("sha256", "f" * 64, "snapshot digest changed"),
    ],
)
def test_whole_snapshot_integrity_is_checked_after_chunk_integrity(
    field, value, message
):
    _, reference, documents = snapshot()
    reference[field] = value
    with pytest.raises(ReleaseStateSnapshotError, match=message):
        hydrate_previous_snapshot(
            {"previous_snapshot": reference}, documents.__getitem__
        )


@pytest.mark.parametrize(
    "raw,compress,message",
    [
        (b"invalid gzip bytes", False, "gzip is invalid"),
        (b"{invalid", True, "JSON is invalid"),
        (b"[]", True, "must be an object"),
        (b'{"release_id": "previous"}', True, "not canonical JSON"),
    ],
)
def test_integrity_pins_do_not_substitute_for_content_validation(
    raw, compress, message
):
    _, reference, _ = snapshot()
    compressed = gzip.compress(raw, mtime=0) if compress else raw
    reference.update(
        sha256=hashlib.sha256(raw).hexdigest(),
        size_bytes=len(raw),
        compressed_sha256=hashlib.sha256(compressed).hexdigest(),
        compressed_size_bytes=len(compressed),
    )
    chunk = reference["chunks"][0]
    chunk.update(
        size_bytes=len(compressed), sha256=hashlib.sha256(compressed).hexdigest()
    )
    document = {
        "binaryData": {
            PREVIOUS_SNAPSHOT_CHUNK_KEY: base64.b64encode(compressed).decode()
        }
    }
    with pytest.raises(ReleaseStateSnapshotError, match=message):
        hydrate_previous_snapshot(
            {"previous_snapshot": reference}, lambda name: document
        )


def test_snapshot_round_trip_is_deterministic_and_does_not_mutate_input() -> None:
    previous, reference, documents = snapshot()
    before = copy.deepcopy(previous)
    assert encode_previous_snapshot(previous)[0] == reference
    state = {"phase": "failed", "previous_snapshot": reference}
    hydrated = hydrate_previous_snapshot(state, documents.__getitem__)
    assert hydrated["previous"] == previous
    assert previous == before
    assert "previous" not in state
    assert canonical_previous_bytes(previous) == canonical_previous_bytes(
        dict(reversed(list(previous.items())))
    )

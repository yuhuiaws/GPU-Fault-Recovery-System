from __future__ import annotations

import base64
import hashlib
import random

import pytest

from gpu_fault.release_state_snapshot import (
    ReleaseStateSnapshotError,
    canonical_previous_bytes,
    encode_previous_snapshot,
    hydrate_previous_snapshot,
)


def _documents(chunks: list[tuple[str, bytes]]) -> dict[str, dict]:
    return {
        name: {
            "immutable": True,
            "binaryData": {"snapshot.part": base64.b64encode(chunk).decode()},
        }
        for name, chunk in chunks
    }


def test_previous_snapshot_round_trip_uses_multiple_content_addressed_chunks() -> None:
    payload = base64.b64encode(random.Random(7).randbytes(800_000)).decode()
    previous = {
        "release_id": "release-a",
        "clusters": {"gpu-a": {"opaque_test_payload": payload}},
    }
    reference, chunks = encode_previous_snapshot(previous)
    documents = _documents(chunks)

    hydrated = hydrate_previous_snapshot(
        {
            "phase": "failed",
            "previous_snapshot": reference,
            "previous_snapshot_sha256": reference["sha256"],
        },
        documents.__getitem__,
    )

    assert len(chunks) > 1
    assert hydrated["previous"] == previous
    assert all(
        name.startswith("gpu-fault-release-previous-") for name, _chunk in chunks
    ), "snapshot chunk names are not content-addressed"


def test_previous_snapshot_rejects_tampered_chunk() -> None:
    previous = {"release_id": "release-a", "metadata": {"key": "value"}}
    reference, chunks = encode_previous_snapshot(previous)
    documents = _documents(chunks)
    name = chunks[0][0]
    documents[name]["binaryData"]["snapshot.part"] = base64.b64encode(
        chunks[0][1] + b"changed"
    ).decode()

    with pytest.raises(ReleaseStateSnapshotError, match="size changed"):
        hydrate_previous_snapshot(
            {
                "previous_snapshot": reference,
                "previous_snapshot_sha256": reference["sha256"],
            },
            documents.__getitem__,
        )


def test_legacy_inline_previous_snapshot_remains_supported() -> None:
    previous = {"release_id": "legacy"}

    hydrated = hydrate_previous_snapshot(
        {"phase": "failed", "previous": previous},
        lambda _name: pytest.fail("legacy state read an external snapshot"),
    )

    assert hydrated["previous"] == previous


def test_inline_previous_snapshot_is_bound_to_its_recorded_digest() -> None:
    """M-12: an inline snapshot with a recorded digest must match it.

    The digest is the provenance binding, so an inline ``previous`` that carries
    one is accepted only when it hashes to that value.
    """

    previous = {"release_id": "legacy", "clusters": {"gpu-a": {"x": "y"}}}
    digest = hashlib.sha256(canonical_previous_bytes(previous)).hexdigest()

    hydrated = hydrate_previous_snapshot(
        {"phase": "failed", "previous": previous, "previous_snapshot_sha256": digest},
        lambda _name: pytest.fail("inline state read an external snapshot"),
    )

    assert hydrated["previous"] == previous


def test_tampered_inline_previous_snapshot_is_rejected() -> None:
    """M-12: a tampered inline snapshot no longer passes as the rollback baseline.

    Before provenance binding an inline ``previous`` was trusted verbatim; now a
    recorded digest that the content does not hash to is a hard rejection.
    """

    previous = {"release_id": "legacy", "clusters": {"gpu-a": {"x": "y"}}}
    honest_digest = hashlib.sha256(canonical_previous_bytes(previous)).hexdigest()
    tampered = {**previous, "clusters": {"gpu-a": {"x": "z"}}}

    with pytest.raises(ReleaseStateSnapshotError, match="inline previous snapshot"):
        hydrate_previous_snapshot(
            {
                "phase": "failed",
                "previous": tampered,
                "previous_snapshot_sha256": honest_digest,
            },
            lambda _name: pytest.fail("tampered state read an external snapshot"),
        )

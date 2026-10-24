import base64
import hashlib
import zlib

import pytest

from gpu_fault.hung_summary_codec import (
    ENCODING,
    MAX_NODE_BYTES,
    decode_summary,
    encode_summary,
)


def test_summary_envelope_is_lossless_and_compact():
    original = {
        "ranks": [
            {
                "rank": rank,
                "cpu_ticks_delta": 0,
                "application": {"status": "UNAVAILABLE"},
            }
            for rank in range(256)
        ]
    }
    envelope = encode_summary(original)
    assert decode_summary(envelope) == original
    assert len(envelope["data"]) < envelope["raw_bytes"] / 4, envelope["raw_bytes"]


@pytest.mark.parametrize(
    "damage", ["checksum", "size", "base64", "trailer", "encoding"]
)
def test_corrupt_or_unsupported_envelope_is_rejected(damage):
    value = encode_summary({"ranks": []})
    if damage == "checksum":
        value["sha256"] = "0" * 64
    elif damage == "size":
        value["raw_bytes"] = MAX_NODE_BYTES + 1
    elif damage == "base64":
        value["data"] = "!"
    elif damage == "trailer":
        value["data"] = base64.b64encode(
            base64.b64decode(value["data"]) + b"extra"
        ).decode()
    else:
        value["encoding"] = "unknown"
    with pytest.raises(ValueError):
        decode_summary(value)


def test_small_compressed_input_cannot_expand_past_the_node_limit():
    raw = b"x" * (MAX_NODE_BYTES + 100)
    value = {
        "encoding": ENCODING,
        "raw_bytes": 1,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "data": base64.b64encode(zlib.compress(raw)).decode(),
    }
    with pytest.raises(ValueError):
        decode_summary(value)

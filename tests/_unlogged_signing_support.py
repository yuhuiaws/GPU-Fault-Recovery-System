"""Fixtures the unlogged-signature tests share (``--allow-unlogged-signature``)."""

from __future__ import annotations

import json
from pathlib import Path

from scripts import cosign_sign_blob
from scripts import signing_transparency as transparency

ROOT = Path(__file__).resolve().parents[1]
ENV = transparency.UNLOGGED_SIGNATURE_REASON_ENV
REASON = "public Rekor answers 500 since 14:21Z"
UNLOGGED_BUNDLE = {
    "mediaType": cosign_sign_blob.BUNDLE_MEDIA_TYPE,
    "verificationMaterial": {"publicKey": {"hint": "aGludA=="}},
    "messageSignature": {
        "messageDigest": {"algorithm": "SHA2_256", "digest": "ZGlnZXN0"},
        "signature": "c2lnbmF0dXJl",
    },
}
LOGGED_BUNDLE = {
    **UNLOGGED_BUNDLE,
    "verificationMaterial": {
        "publicKey": {"hint": "aGludA=="},
        "tlogEntries": [{"logIndex": "1"}],
    },
}


def write_json(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path

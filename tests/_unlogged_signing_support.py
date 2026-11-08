"""Fixtures the unlogged-signature tests share (``--allow-unlogged-signature``)."""

from __future__ import annotations

import base64
import hashlib
import json
import sys
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

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
#: A real P-256 public key: the fake log echoes it back and the rebuilt bundle
#: must name the same key.
PUBLIC_PEM = """-----BEGIN PUBLIC KEY-----
MFkwEwYHKoZIzj0CAQYIKoZIzj0DAQcDQgAEESFXE3kS1dflKaBQx+IWJbgs3xXM
jME6AXcluBD9Ej5GBq7VS0tjJDXOoMNq3W71kVapUjOkWTbxblbx0+ZHtQ==
-----END PUBLIC KEY-----
"""


def write_json(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def fake_cosign(directory: Path, *, offline_ok: bool = True) -> Path:
    """``directory/cosign``: every verification passes, except an offline one
    (``--insecure-ignore-tlog=true``) when ``offline_ok`` is false."""

    path = directory / "cosign"
    path.write_text(
        f"#!{sys.executable}\n"
        "import sys\n"
        "argv = sys.argv[1:]\n"
        f"offline_ok = {offline_ok!r}\n"
        "if '--insecure-ignore-tlog=true' in argv and not offline_ok:\n"
        "    sys.stderr.write('invalid signature'); raise SystemExit(1)\n"
        "raise SystemExit(0)\n",
        encoding="utf-8",
    )
    path.chmod(0o755)
    return path


def _log_entry(canonical: bytes) -> dict[str, Any]:
    return {
        "body": base64.b64encode(canonical).decode(),
        "integratedTime": 1793189892,
        "logID": "c0" * 32,
        "logIndex": 7,
        "verification": {
            "inclusionProof": {
                "checkpoint": "rekor - 1\n2\nroot\n",
                "hashes": ["ab" * 32],
                "logIndex": 6,
                "rootHash": "ef" * 32,
                "treeSize": 8,
            },
            "signedEntryTimestamp": "MEUCIQ==",
        },
    }


@contextmanager
def fake_rekor() -> Iterator[dict[str, Any]]:
    """A transparency log on localhost, as a mutable state mapping.

    ``url``; ``posts`` (every upload it received), ``entries`` (by UUID) and
    ``gets`` (every path read); ``log_info`` counts the reads of
    ``/api/v1/log``, the probe, answered ``log_info_body``. ``conflict``
    answers an upload 409 naming its entry; ``down`` answers every request
    500, ``fail_next`` the next that many probes; ``refuse`` holds artifact
    SHA-256 digests whose upload is answered 400. Anything else read is 404.
    """

    state: dict[str, Any] = {
        "url": "",
        "posts": [],
        "entries": {},
        "gets": [],
        "conflict": False,
        "down": False,
        "fail_next": 0,
        "refuse": set(),
        "log_info": 0,
        "log_info_body": {"treeSize": 8, "rootHash": "ef" * 32},
    }

    class Handler(BaseHTTPRequestHandler):
        def _answer(
            self, status: int, payload: object, location: str | None = None
        ) -> None:
            data = json.dumps(payload).encode()
            self.send_response(status)
            if location is not None:
                self.send_header("Location", location)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_POST(self) -> None:
            body = self.rfile.read(int(self.headers["Content-Length"]))
            if state["down"]:
                self._answer(500, {"code": 500, "message": "error"})
                return
            proposed = json.loads(body)
            state["posts"].append(proposed)
            if proposed["spec"]["data"]["hash"]["value"] in state["refuse"]:
                self._answer(400, {"code": 400, "message": "refused"})
                return
            canonical = json.dumps(proposed, separators=(",", ":")).encode()
            uuid = "108e9186e8c5677a" + hashlib.sha256(b"\x00" + canonical).hexdigest()
            state["entries"][uuid] = _log_entry(canonical)
            if state["conflict"]:
                self._answer(
                    409,
                    {"code": 409, "message": f"exists with UUID {uuid}"},
                    location=f"/api/v1/log/entries/{uuid}",
                )
            else:
                self._answer(201, {uuid: state["entries"][uuid]})

        def do_GET(self) -> None:
            state["gets"].append(self.path)
            uuid = self.path.rsplit("/", 1)[-1]
            if self.path == cosign_sign_blob.LOG_INFO_PATH:
                state["log_info"] += 1
                failing = state["down"] or state["fail_next"] > 0
                state["fail_next"] = max(state["fail_next"] - 1, 0)
                if failing:
                    self._answer(500, {"code": 500, "message": "error"})
                else:
                    self._answer(200, state["log_info_body"])
            elif state["down"]:
                self._answer(500, {"code": 500, "message": "error"})
            elif (
                self.path.startswith("/api/v1/log/entries/")
                and uuid in state["entries"]
            ):
                self._answer(200, {uuid: state["entries"][uuid]})
            else:
                self._answer(404, {"code": 404, "message": "not found"})

        def log_message(self, *_arguments: object) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    state["url"] = f"http://127.0.0.1:{server.server_address[1]}"
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield state
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

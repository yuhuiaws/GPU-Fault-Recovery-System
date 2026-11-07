"""``cosign sign-blob`` that survives Rekor's 409 for this signing's own entry.

Every ``cosign sign-blob`` of the release tooling goes through here: the
release attestation (``make release-build*``), the deploy-host archive
(``make deploy-host-bundle`` / ``deploy-host-sign``), the CI candidate receipt
and the successful source deploy record.

cosign v3 writes the new bundle format and uploads the signature to Rekor v1
through sigstore-go. Its HTTP client sends the identical request again when the
first one ends in a 5xx or a dropped connection. If Rekor had already
integrated the first request, the repeat is answered with ``[409]
createLogEntryConflict ... an equivalent entry already exists in the
transparency log with UUID <id>``. sigstore-go before v1.3.0 (cosign up to
v3.1.3) treats that as fatal and writes no bundle, although the entry it names
is the complete log entry of this very upload (fixed upstream by
https://github.com/sigstore/sigstore-go/pull/647, released in sigstore-go
v1.3.0 and cosign v3.1.4). The content is not the cause: an ECDSA signature is
randomized, so signing the same bytes again yields a different entry; only a
deterministic signer (ed25519, RSA PKCS#1 v1.5) meets an earlier signature of
the same bytes by the same key, which is equally valid.

The wrapper runs cosign unchanged and passes every outcome through except that
409 while signing with a key. Then it reads the named entry (a GET, nothing is
uploaded), requires a ``hashedrekord`` 0.0.1 whose canonical body hashes to the
UUID, whose artifact digest is this file's SHA-256 and whose public key is the
signing key's (``cosign public-key``), writes the Sigstore bundle v0.3 from the
entry with the field mapping sigstore-go uses for a fresh upload, and keeps it
only after ``cosign verify-blob --key`` accepted it. The RFC 3161 timestamp
cosign obtained before the upload ended with its process; a key-verified
bundle does not need it. No verifier of the release tooling reads a signing
time either: a ``--key`` verification needs no trusted timestamp, and the
entry's integration time is only reported here.
``COSIGN_PASSWORD`` reaches ``cosign sign-blob`` and ``cosign public-key`` only;
``cosign verify-blob`` runs without it.

A signing that a Sigstore service fails transiently -- Rekor answering 5xx
(live 2026-11-07: ``status 500 ... unexpected result from transparency log``),
a timeout, a refused connection, the TUF repository or the timestamp
authority not answering -- is run again, ``SIGN_ATTEMPTS`` times in all with
``SIGN_RETRY_SECONDS`` between, each a fresh cosign signing. When the last one
fails too the wrapper refuses: it writes no bundle, notes the refusal beside
the bundle path (``signing_transparency.refusal_marker``: make reports only
``Error 75``) and names the exit, ``gpu-fault-admin deploy ...
--allow-unlogged-signature --reason "..."``. Nothing else is retried; any
other failure passes through as before. When it is only the live TUF fetch
that fails, the next attempt signs with the site's pinned TUF signing config
and trust root (``signing_transparency``), which list the same Rekor and
timestamp authority; each online signing re-pins them.

``--unlogged`` is that exit, and nothing reaches it implicitly: the payload
must already carry the signed ``"signing": {"transparency_log": "unlogged",
...}`` declaration (a JSON payload's own field, the deploy-host archive's
manifest) and a key must be given -- a keyless signature is never written
without the log. cosign then signs with the key alone and contacts no
Sigstore service (``signing_transparency.UNLOGGED_SIGN_FLAGS``).
"""

from __future__ import annotations

import argparse
import base64
import codecs
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

if __package__:
    from scripts import signing_transparency as transparency
else:
    import signing_transparency as transparency

DEFAULT_REKOR_URL = "https://rekor.sigstore.dev"
BUNDLE_MEDIA_TYPE = "application/vnd.dev.sigstore.bundle.v0.3+json"
PASSWORD_ENV = "COSIGN_PASSWORD"
FETCH_ATTEMPTS = 3
FETCH_TIMEOUT_SECONDS = 30.0
FETCH_RETRY_SECONDS = 2.0
#: Bounded retry of a transient Sigstore failure: three signings in all, 30 s
#: and then 90 s apart -- enough for a blip, short of a deploy waiting on an
#: outage (cosign itself already repeats each upload once).
SIGN_ATTEMPTS = 3
SIGN_RETRY_SECONDS = (30.0, 90.0)
_CONFLICT = re.compile(
    r"\[POST /api/v1/log/entries\]\[409\] createLogEntryConflict\b.*?"
    r"with UUID ([0-9a-f]{80}|[0-9a-f]{64})\b",
    re.DOTALL,
)
_PEM_PUBLIC_KEY = re.compile(
    r"-----BEGIN PUBLIC KEY-----\s*([A-Za-z0-9+/=\s]+?)\s*-----END PUBLIC KEY-----"
)


class SignBlobError(RuntimeError):
    """The 409 could not be turned into a verified bundle."""


@dataclass(frozen=True)
class SignResult:
    returncode: int
    stderr: str
    recovered_entry: str | None = None
    unlogged: bool = False
    #: The pinned trust root's record when the live TUF fetch failed and the
    #: signing used the pin (``signing_transparency.TrustPin.usage``).
    trust_pin: dict[str, Any] | None = None


def conflict_entry_uuid(stderr: str) -> str | None:
    """The UUID of Rekor's 409 ``createLogEntryConflict``, or None."""

    match = _CONFLICT.search(stderr)
    return match.group(1) if match else None


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def public_key_der(pem: str) -> bytes:
    """DER bytes of the one ``PUBLIC KEY`` block in ``pem``."""

    blocks = _PEM_PUBLIC_KEY.findall(pem)
    if len(blocks) != 1:
        raise SignBlobError("expected exactly one PEM public key")
    try:
        return base64.b64decode("".join(blocks[0].split()), validate=True)
    except ValueError as exc:
        raise SignBlobError("PEM public key is not valid base64") from exc


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise SignBlobError(reason)


def _text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise SignBlobError(f"Rekor entry has no {field}")
    return value


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _hex_bytes(value: object, field: str) -> bytes:
    try:
        return bytes.fromhex(_text(value, field))
    except ValueError as exc:
        raise SignBlobError(f"Rekor entry {field} is not hex") from exc


def _integer(value: object, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise SignBlobError(f"Rekor entry has no {field}")
    return value


def bundle_from_log_entry(
    entry: Mapping[str, Any],
    *,
    uuid: str,
    artifact_sha256: str,
    public_key_pem: str,
) -> dict[str, Any]:
    """The bundle sigstore-go would have written had the 409 not hidden ``entry``.

    Raises :class:`SignBlobError` unless ``entry`` is the hashedrekord 0.0.1
    log entry named by ``uuid`` for exactly this artifact digest and key.
    """

    body_text = _text(entry.get("body"), "body")
    try:
        body_bytes = base64.b64decode(body_text, validate=True)
        body = json.loads(body_bytes)
    except ValueError as exc:
        raise SignBlobError("Rekor entry body is not base64 JSON") from exc
    # RFC 6962 leaf hash: the UUID names this body and no other.
    leaf = hashlib.sha256(b"\x00" + body_bytes).hexdigest()
    _require(leaf == uuid[-64:], "Rekor entry body does not hash to the UUID")
    body = _mapping(body)
    _require(
        body.get("kind") == "hashedrekord" and body.get("apiVersion") == "0.0.1",
        "Rekor entry is not a hashedrekord 0.0.1",
    )
    spec = _mapping(body.get("spec"))
    data_hash = _mapping(_mapping(spec.get("data")).get("hash"))
    _require(
        data_hash.get("algorithm") == "sha256"
        and data_hash.get("value") == artifact_sha256,
        "Rekor entry is not for this artifact",
    )
    signature = _mapping(spec.get("signature"))
    signature_content = _text(signature.get("content"), "signature")
    key_content = _text(_mapping(signature.get("publicKey")).get("content"), "key")
    try:
        entry_key_pem = base64.b64decode(key_content, validate=True).decode()
    except ValueError as exc:
        raise SignBlobError("Rekor entry public key is not base64 PEM") from exc
    expected_der = public_key_der(public_key_pem)
    _require(
        public_key_der(entry_key_pem) == expected_der,
        "Rekor entry was signed by another key",
    )
    verification = _mapping(entry.get("verification"))
    proof = _mapping(verification.get("inclusionProof"))
    hashes = proof.get("hashes")
    if not isinstance(hashes, list):
        raise SignBlobError("Rekor entry has no inclusion proof hashes")
    return {
        "mediaType": BUNDLE_MEDIA_TYPE,
        "verificationMaterial": {
            "publicKey": {"hint": _b64(hashlib.sha256(expected_der).digest())},
            "tlogEntries": [
                {
                    # The global (virtual) index, as sigstore-go records it.
                    "logIndex": str(_integer(entry.get("logIndex"), "logIndex")),
                    "logId": {"keyId": _b64(_hex_bytes(entry.get("logID"), "logID"))},
                    "kindVersion": {"kind": "hashedrekord", "version": "0.0.1"},
                    "integratedTime": str(
                        _integer(entry.get("integratedTime"), "integratedTime")
                    ),
                    "inclusionPromise": {
                        "signedEntryTimestamp": _text(
                            verification.get("signedEntryTimestamp"),
                            "signed entry timestamp",
                        )
                    },
                    "inclusionProof": {
                        # Relative to the shard the entry is in.
                        "logIndex": str(
                            _integer(proof.get("logIndex"), "proof logIndex")
                        ),
                        "rootHash": _b64(
                            _hex_bytes(proof.get("rootHash"), "proof rootHash")
                        ),
                        "treeSize": str(
                            _integer(proof.get("treeSize"), "proof treeSize")
                        ),
                        "hashes": [
                            _b64(_hex_bytes(item, "proof hash")) for item in hashes
                        ],
                        "checkpoint": {
                            "envelope": _text(proof.get("checkpoint"), "checkpoint")
                        },
                    },
                    "canonicalizedBody": body_text,
                }
            ],
        },
        "messageSignature": {
            "messageDigest": {
                "algorithm": "SHA2_256",
                "digest": _b64(bytes.fromhex(artifact_sha256)),
            },
            "signature": signature_content,
        },
    }


def fetch_log_entry(rekor_url: str, uuid: str) -> dict[str, Any]:
    """GET one log entry by UUID; retries only what may be transient."""

    url = f"{rekor_url.rstrip('/')}/api/v1/log/entries/{urllib.parse.quote(uuid)}"
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    last_error = "no attempt"
    for attempt in range(FETCH_ATTEMPTS):
        if attempt:
            time.sleep(FETCH_RETRY_SECONDS * attempt)
        try:
            with urllib.request.urlopen(
                request, timeout=FETCH_TIMEOUT_SECONDS
            ) as response:
                payload = json.loads(response.read())
        except urllib.error.HTTPError as exc:
            last_error = f"HTTP {exc.code}"
            # 404 too: the read path may trail the write that answered 409.
            if exc.code < 500 and exc.code not in (404, 429):
                break
            continue
        except (OSError, ValueError) as exc:
            last_error = type(exc).__name__
            continue
        values = list(payload.values()) if isinstance(payload, dict) else []
        if len(values) != 1 or not isinstance(values[0], dict):
            raise SignBlobError("Rekor did not return exactly one entry")
        return values[0]
    raise SignBlobError(f"cannot read the Rekor entry ({last_error})")


def _cosign_failure(stderr: str) -> str:
    lines = [line.strip() for line in stderr.splitlines() if line.strip()]
    return lines[-1] if lines else "no output"


def _signer_public_key(
    cosign: str, key: str, environment: Mapping[str, str], cwd: Path | None
) -> str:
    completed = subprocess.run(
        [cosign, "public-key", "--key", key],
        cwd=cwd,
        env=dict(environment),
        stdin=subprocess.DEVNULL,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode:
        raise SignBlobError(
            "cosign public-key failed: " + _cosign_failure(completed.stderr)
        )
    return completed.stdout


def recover_conflict(
    artifact: Path,
    *,
    bundle: Path,
    key: str,
    uuid: str,
    cosign: str,
    rekor_url: str,
    environment: Mapping[str, str],
    cwd: Path | None,
) -> int:
    """Write and verify ``bundle`` from the entry; return its integrated time."""

    public_key_pem = _signer_public_key(cosign, key, environment, cwd)
    entry = fetch_log_entry(rekor_url, uuid)
    value = bundle_from_log_entry(
        entry,
        uuid=uuid,
        artifact_sha256=_sha256_file(artifact),
        public_key_pem=public_key_pem,
    )
    verify_environment = {
        name: item for name, item in environment.items() if name != PASSWORD_ENV
    }
    # Verify the very bytes that replace the target, next to it (same file
    # system, so the final rename is atomic).
    staged = bundle.with_name(f".{bundle.name}.{os.getpid()}.tmp")
    try:
        descriptor = os.open(staged, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as target:
            target.write(json.dumps(value, separators=(",", ":")))
        with tempfile.TemporaryDirectory(prefix="cosign-sign-blob-") as scratch:
            public_key = Path(scratch) / "signer.pub"
            public_key.write_text(public_key_pem, encoding="utf-8")
            verified = subprocess.run(
                [
                    cosign,
                    "verify-blob",
                    "--key",
                    str(public_key),
                    "--bundle",
                    str(staged),
                    str(artifact),
                ],
                cwd=cwd,
                env=verify_environment,
                stdin=subprocess.DEVNULL,
                check=False,
                capture_output=True,
                text=True,
            )
        if verified.returncode:
            raise SignBlobError(
                "cosign verify-blob rejected the rebuilt bundle: "
                + _cosign_failure(verified.stderr)
            )
        os.replace(staged, bundle)
    finally:
        staged.unlink(missing_ok=True)
    return int(value["verificationMaterial"]["tlogEntries"][0]["integratedTime"])


def _run_cosign(
    command: Sequence[str],
    *,
    environment: Mapping[str, str],
    cwd: Path | None,
    echo: bool,
) -> tuple[int, str]:
    if not echo:
        completed = subprocess.run(
            list(command),
            cwd=cwd,
            env=dict(environment),
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        return completed.returncode, completed.stderr or ""
    # Pass cosign's prompts and progress through as they come (a password
    # prompt has no newline) and keep a copy to look for the 409.
    process = subprocess.Popen(
        list(command),
        cwd=cwd,
        env=dict(environment),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    stream = process.stderr
    if stream is None:  # stderr=PIPE always sets it
        raise RuntimeError("cosign stderr pipe is missing")
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    captured: list[str] = []
    while chunk := os.read(stream.fileno(), 65536):
        captured.append(decoder.decode(chunk))
        sys.stderr.write(captured[-1])
        sys.stderr.flush()
    stream.close()
    captured.append(decoder.decode(b"", final=True))
    sys.stderr.write(captured[-1])
    sys.stderr.flush()
    return process.wait(), "".join(captured)


def _notice(text: str) -> str:
    sys.stderr.write(text)
    sys.stderr.flush()
    return text


def _key_path(key: str | None, cwd: Path | None) -> Path | None:
    if not key or "://" in key:
        return None
    path = Path(key).expanduser()
    return path if path.is_absolute() else (cwd or Path.cwd()) / path


def _sign_unlogged(
    artifact: Path,
    *,
    bundle: Path,
    key: str | None,
    cosign: str,
    environment: Mapping[str, str],
    cwd: Path | None,
    echo: bool,
) -> SignResult:
    base = cwd or Path.cwd()
    if not key:
        return SignResult(
            2,
            _notice(
                "cosign_sign_blob: refusing to sign without a transparency-log "
                "entry: keyless signing always needs the log\n"
            ),
        )
    try:
        declaration = transparency.artifact_declaration(base / artifact)
    except transparency.SigningTransparencyError as exc:
        return SignResult(2, _notice(f"cosign_sign_blob: {exc}\n"))
    if declaration is None:
        return SignResult(
            2,
            _notice(
                f"cosign_sign_blob: refusing to sign {artifact} without a "
                "transparency-log entry: its signed payload declares no unlogged "
                "signing\n"
            ),
        )
    command = [
        cosign,
        "sign-blob",
        "--yes",
        "--key",
        key,
        *transparency.UNLOGGED_SIGN_FLAGS,
        "--bundle",
        str(bundle),
        str(artifact),
    ]
    returncode, stderr = _run_cosign(
        command, environment=environment, cwd=cwd, echo=echo
    )
    if returncode:
        return SignResult(returncode, stderr)
    if transparency.bundle_transparency(base / bundle) != transparency.UNLOGGED:
        return SignResult(
            2,
            stderr
            + _notice(
                f"cosign_sign_blob: {bundle} is not a bundle without a "
                "transparency-log entry; refusing it\n"
            ),
        )
    notice = _notice(
        f"cosign_sign_blob: signed {artifact} with the site key only, WITHOUT a "
        f"transparency-log entry (declared {declaration['declared_at']}: "
        f"{declaration['reason']})\n"
    )
    return SignResult(0, stderr + notice, unlogged=True)


def _refuse_unavailable(
    artifact: Path,
    *,
    bundle: Path,
    base: Path,
    key: str | None,
    cause: str,
    stderr: str,
) -> SignResult:
    transparency.write_refusal_marker(
        base / bundle, artifact=artifact, attempts=SIGN_ATTEMPTS, cause=cause
    )
    exit_line = (
        "retry once the public Sigstore services answer, or sign with the site "
        "key only, without a transparency-log entry: rerun the deploy with "
        f'{transparency.ALLOW_UNLOGGED_FLAG} {transparency.REASON_FLAG} "<why>"'
        if key
        else "retry once the public Sigstore services answer (a keyless signature "
        "always needs the transparency log)"
    )
    notice = _notice(
        f"cosign_sign_blob: the Sigstore services did not accept the signature of "
        f"{artifact} after {SIGN_ATTEMPTS} attempts (last: {cause}); no bundle "
        f"was written and nothing was signed; {exit_line}\n"
    )
    return SignResult(transparency.TRANSPARENCY_LOG_UNAVAILABLE_EXIT, stderr + notice)


def sign_blob(
    artifact: Path,
    *,
    bundle: Path,
    key: str | None,
    cosign: str = "cosign",
    rekor_url: str = DEFAULT_REKOR_URL,
    environment: Mapping[str, str] | None = None,
    cwd: Path | None = None,
    echo: bool = True,
    unlogged: bool = False,
) -> SignResult:
    env = dict(os.environ if environment is None else environment)
    base = cwd or Path.cwd()
    transparency.refusal_marker(base / bundle).unlink(missing_ok=True)
    if unlogged:
        return _sign_unlogged(
            artifact,
            bundle=bundle,
            key=key,
            cosign=cosign,
            environment=env,
            cwd=cwd,
            echo=echo,
        )
    pin_directory = transparency.pin_directory_for_key(_key_path(key, cwd))
    pin: transparency.TrustPin | None = None
    collected = ""
    cause = ""
    attempt = 0
    while attempt < SIGN_ATTEMPTS:
        command = [cosign, "sign-blob", "--yes"]
        if key:
            command.extend(("--key", key))
        if pin is not None:
            command.extend(
                (
                    "--signing-config",
                    str(pin.signing_config),
                    "--trusted-root",
                    str(pin.trusted_root),
                )
            )
        command.extend(("--bundle", str(bundle), str(artifact)))
        returncode, stderr = _run_cosign(command, environment=env, cwd=cwd, echo=echo)
        collected += stderr
        attempt += 1
        if returncode == 0:
            if pin is None:
                transparency.refresh_trust_pin(pin_directory, environment=env)
                return SignResult(0, collected)
            usage = pin.usage()
            transparency.record_pin_use(pin.directory, usage)
            return SignResult(
                0,
                collected
                + _notice(
                    f"cosign_sign_blob: signed {artifact} with the transparency "
                    f"log, using {pin.describe()} for the service list\n"
                ),
                trust_pin=usage,
            )
        uuid = conflict_entry_uuid(stderr)
        if uuid is not None and key:
            break
        tuf_failure = transparency.tuf_fetch_failure(stderr)
        if tuf_failure is not None and pin is None:
            pin = transparency.load_trust_pin(pin_directory)
            if pin is not None:
                collected += _notice(
                    f"cosign_sign_blob: Sigstore TUF did not answer ({tuf_failure}); "
                    f"signing again with {pin.describe()}\n"
                )
                attempt -= 1
                continue
        transient = transparency.transient_service_failure(stderr)
        if transient is None:
            return SignResult(returncode, collected)
        cause = transient
        if attempt < SIGN_ATTEMPTS:
            delay = SIGN_RETRY_SECONDS[min(attempt - 1, len(SIGN_RETRY_SECONDS) - 1)]
            collected += _notice(
                f"cosign_sign_blob: transient Sigstore failure ({transient}); "
                f"attempt {attempt + 1} of {SIGN_ATTEMPTS} in {delay:.0f} s\n"
            )
            time.sleep(delay)
    else:
        return _refuse_unavailable(
            artifact, bundle=bundle, base=base, key=key, cause=cause, stderr=collected
        )
    try:
        integrated = recover_conflict(
            base / artifact,
            bundle=base / bundle,
            key=key,
            uuid=uuid,
            cosign=cosign,
            rekor_url=rekor_url,
            environment=env,
            cwd=cwd,
        )
    except SignBlobError as exc:
        notice = f"cosign_sign_blob: cannot use Rekor entry {uuid}: {exc}\n"
        sys.stderr.write(notice)
        return SignResult(returncode, collected + notice)
    when = datetime.fromtimestamp(integrated, UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    notice = (
        f"cosign_sign_blob: Rekor answered 409, entry {uuid} (integrated {when}) "
        f"already logs this artifact signed by this key; wrote {bundle} from it "
        "and cosign verify-blob accepted it\n"
    )
    sys.stderr.write(notice)
    return SignResult(0, collected + notice, recovered_entry=uuid)


def main(arguments: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="cosign sign-blob that survives Rekor's 409 for its own entry"
    )
    parser.add_argument("--cosign", default="cosign")
    parser.add_argument("--key", help="private key file or KMS URI; keyless if empty")
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--rekor-url", default=DEFAULT_REKOR_URL)
    parser.add_argument(
        "--unlogged",
        action="store_true",
        help=(
            "sign with the key only, without a transparency-log entry; the "
            "payload must declare it (gpu-fault-admin deploy "
            "--allow-unlogged-signature --reason ...)"
        ),
    )
    parser.add_argument("artifact", type=Path)
    options = parser.parse_args(arguments)
    return sign_blob(
        options.artifact,
        bundle=options.bundle,
        key=options.key or None,
        cosign=options.cosign,
        rekor_url=options.rekor_url,
        unlogged=options.unlogged,
    ).returncode


if __name__ == "__main__":
    raise SystemExit(main())

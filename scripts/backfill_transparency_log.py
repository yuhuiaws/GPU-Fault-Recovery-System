"""Put the site's signatures made without a transparency-log entry into the log.

``gpu-fault-admin backfill-transparency-log`` runs this from the site's
repository once the public Sigstore transparency log (Rekor) answers again
after a deploy signed with ``--allow-unlogged-signature``. Nothing is signed
again: every bundle of the site key that has no log entry keeps its signature,
which is uploaded as a ``hashedrekord`` 0.0.1 entry (artifact SHA-256,
signature, site public key). The bundle is then rebuilt from the entry Rekor
returns -- the mapping ``cosign_sign_blob.bundle_from_log_entry`` already
uses for its 409 recovery -- and replaces the old one only after a
``cosign verify-blob --key`` that requires the log entry accepted it. A 409
means the same signature is in the log already (a second run): that entry is
read and used. Only the public key is needed; the private key and its password
are never read.

The bundles it looks at: the release attestations (``dist/`` and the copies the
release records keep), the successful source deploy record, the deploy-host
archives and the CI candidate receipts. A bundle that already has a log entry
is left alone, so the command can run any number of times. Afterwards the
release's ``gpu-fault-admin status`` finding (or failing check) is gone: status
reads the record's bundle copy.

A deploy without ``--allow-unlogged-signature`` runs this same code path
first, in its own process under the site lock it holds
(``staging_signing.backfill_before_deploy``); the verb is the same step out
of band.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

if __package__:
    from scripts import cosign_sign_blob
    from scripts import signing_transparency as transparency
else:
    import cosign_sign_blob
    import signing_transparency as transparency

UPLOAD_ATTEMPTS = 3
UPLOAD_TIMEOUT_SECONDS = 30.0
UPLOAD_RETRY_SECONDS = 5.0
_UUID = re.compile(r"([0-9a-f]{80}|[0-9a-f]{64})")


class BackfillError(RuntimeError):
    pass


#: What a bundle's backfill can fail with; it is reported, the next one runs.
FAILURES = (
    BackfillError,
    cosign_sign_blob.SignBlobError,
    OSError,
    KeyError,
    TypeError,
    ValueError,
)


@dataclass(frozen=True)
class Outcome:
    """One bundle's result: ``backfilled`` (with its log entry and when),
    ``already_logged``, ``skipped`` or ``failed`` (with the error)."""

    bundle: Path
    outcome: str
    log_entry: str | None = None
    at: str | None = None
    error: str | None = None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def candidates(
    state_dir: Path, repository_root: Path | None
) -> list[tuple[Path, Path]]:
    """``(artifact, bundle)`` pairs the site's own key signed, where present."""

    pairs: list[tuple[Path, Path]] = []
    if repository_root is not None:
        pairs.append(
            (
                repository_root / "dist/current-attestation.json",
                repository_root / "dist/current-attestation.bundle.json",
            )
        )
    for record in sorted((state_dir / "release-deploy").glob("*/attestation.json")):
        pairs.append((record, record.with_name("attestation.sigstore.json")))
    pairs.append(
        (
            state_dir / "source-deploy-success.json",
            state_dir / "source-deploy-success.sigstore.json",
        )
    )
    for bundle in sorted(
        (state_dir / "deploy-host" / "by-content").glob("*/*.sigstore.json")
    ):
        archive = bundle.with_name(
            bundle.name.removesuffix(".sigstore.json")
            + transparency.DEPLOY_HOST_ARCHIVE_SUFFIX
        )
        pairs.append((archive, bundle))
    for receipt in sorted((state_dir / "ci-candidates").glob("*/verification.json")):
        pairs.append((receipt, receipt.with_name("verification.sigstore.json")))
    return [(artifact, bundle) for artifact, bundle in pairs if bundle.is_file()]


def proposed_entry(
    artifact_sha256: str, signature: str, public_key_pem: str
) -> dict[str, Any]:
    return {
        "apiVersion": "0.0.1",
        "kind": "hashedrekord",
        "spec": {
            "data": {"hash": {"algorithm": "sha256", "value": artifact_sha256}},
            "signature": {
                "content": signature,
                "publicKey": {
                    "content": base64.b64encode(public_key_pem.encode()).decode()
                },
            },
        },
    }


def upload(rekor_url: str, entry: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """POST the entry; ``(uuid, log entry)``. A 409 reads the existing entry."""

    url = f"{rekor_url.rstrip('/')}/api/v1/log/entries"
    data = json.dumps(entry, separators=(",", ":")).encode()
    last = "no attempt"
    for attempt in range(UPLOAD_ATTEMPTS):
        if attempt:
            time.sleep(UPLOAD_RETRY_SECONDS * attempt)
        request = urllib.request.Request(
            url,
            data=data,
            method="POST",
            headers={"Content-Type": "application/json", "Accept": "application/json"},
        )
        try:
            with urllib.request.urlopen(
                request, timeout=UPLOAD_TIMEOUT_SECONDS
            ) as response:
                payload = json.loads(response.read())
        except urllib.error.HTTPError as exc:
            if exc.code == 409:
                text = " ".join(
                    (
                        str(exc.headers.get("Location") or ""),
                        exc.read().decode("utf-8", "replace"),
                    )
                )
                found = _UUID.search(text)
                if found is None:
                    raise BackfillError("Rekor answered 409 without the entry UUID")
                return found.group(1), cosign_sign_blob.fetch_log_entry(
                    rekor_url, found.group(1)
                )
            last = f"HTTP {exc.code}"
            if exc.code < 500 and exc.code != 429:
                break
            continue
        except (OSError, ValueError) as exc:
            last = type(exc).__name__
            continue
        if not isinstance(payload, dict) or len(payload) != 1:
            raise BackfillError("Rekor did not return exactly one entry")
        ((uuid, value),) = payload.items()
        if not isinstance(value, dict):
            raise BackfillError("Rekor returned an invalid entry")
        return str(uuid), value
    raise BackfillError(f"the transparency log did not accept the upload ({last})")


def backfill(
    artifact: Path,
    bundle: Path,
    *,
    public_key: Path,
    cosign: str = "cosign",
    rekor_url: str = cosign_sign_blob.DEFAULT_REKOR_URL,
) -> str:
    """``backfilled``, ``already_logged`` or ``skipped`` for one bundle."""

    return backfill_with_entry(
        artifact, bundle, public_key=public_key, cosign=cosign, rekor_url=rekor_url
    )[0]


def backfill_with_entry(
    artifact: Path,
    bundle: Path,
    *,
    public_key: Path,
    cosign: str = "cosign",
    rekor_url: str = cosign_sign_blob.DEFAULT_REKOR_URL,
) -> tuple[str, str | None]:
    """``backfill`` and the log entry's UUID when it wrote one."""

    state = transparency.bundle_transparency(bundle)
    if state == transparency.LOGGED:
        return "already_logged", None
    if state != transparency.UNLOGGED:
        return "skipped", None
    if not artifact.is_file():
        raise BackfillError(f"{artifact.name} is missing")
    value = json.loads(bundle.read_bytes())
    signature = value["messageSignature"]["signature"]
    if not isinstance(signature, str) or not signature:
        raise BackfillError(f"{bundle.name} has no signature")
    # The signature is the site key's over exactly this artifact, offline.
    offline = subprocess.run(
        [
            cosign,
            "verify-blob",
            "--key",
            str(public_key),
            "--bundle",
            str(bundle),
            transparency.IGNORE_TLOG_FLAG,
            str(artifact),
        ],
        stdin=subprocess.DEVNULL,
        check=False,
        capture_output=True,
        text=True,
    )
    if offline.returncode:
        raise BackfillError(
            f"{bundle.name} does not verify with the site key; nothing uploaded"
        )
    public_key_pem = public_key.read_text(encoding="utf-8")
    artifact_sha256 = _sha256_file(artifact)
    uuid, entry = upload(
        rekor_url, proposed_entry(artifact_sha256, signature, public_key_pem)
    )
    rebuilt = cosign_sign_blob.bundle_from_log_entry(
        entry,
        uuid=uuid,
        artifact_sha256=artifact_sha256,
        public_key_pem=public_key_pem,
    )
    if rebuilt["messageSignature"]["signature"] != signature:
        raise BackfillError(f"the log entry {uuid} holds another signature")
    staged = bundle.with_name(f".{bundle.name}.{os.getpid()}.tmp")
    try:
        descriptor = os.open(staged, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as target:
            target.write(json.dumps(rebuilt, separators=(",", ":")))
        verified = transparency.run_verify_blob(
            [
                cosign,
                "verify-blob",
                "--key",
                str(public_key),
                "--bundle",
                str(staged),
                str(artifact),
            ],
            pin_directory=transparency.pin_directory_for_key(public_key),
        )
        if verified.returncode:
            raise BackfillError(
                f"cosign verify-blob rejected the logged bundle for {artifact.name}"
            )
        os.replace(staged, bundle)
    finally:
        staged.unlink(missing_ok=True)
    print(
        f"backfill-transparency-log: {bundle} now carries log entry {uuid}",
        file=sys.stderr,
        flush=True,
    )
    return "backfilled", uuid


def backfill_bundles(
    pairs: Iterable[tuple[Path, Path]],
    *,
    public_key: Path,
    cosign: str = "cosign",
    rekor_url: str = cosign_sign_blob.DEFAULT_REKOR_URL,
) -> list[Outcome]:
    """Each ``(artifact, bundle)`` in turn; a failure is reported and the next
    one runs. Every replacement is its own atomic rename, so a bundle that
    failed is untouched and one that succeeded keeps its entry."""

    outcomes: list[Outcome] = []
    for artifact, bundle in pairs:
        try:
            outcome, uuid = backfill_with_entry(
                artifact,
                bundle,
                public_key=public_key,
                cosign=cosign,
                rekor_url=rekor_url,
            )
        except FAILURES as exc:
            outcomes.append(Outcome(bundle, "failed", error=str(exc)))
            continue
        outcomes.append(
            Outcome(
                bundle,
                outcome,
                log_entry=uuid,
                at=transparency.utc_timestamp() if uuid is not None else None,
            )
        )
    return outcomes


def main(arguments: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path)
    parser.add_argument("--public-key", type=Path, required=True)
    parser.add_argument("--cosign", default="cosign")
    parser.add_argument("--rekor-url", default=cosign_sign_blob.DEFAULT_REKOR_URL)
    options = parser.parse_args(arguments)
    state_dir = options.state_dir.expanduser().resolve()
    repository_root = (
        options.repository_root.expanduser().resolve()
        if options.repository_root is not None
        else None
    )
    result: dict[str, list[Any]] = {
        "backfilled": [],
        "already_logged": [],
        "skipped": [],
        "failed": [],
    }
    for item in backfill_bundles(
        candidates(state_dir, repository_root),
        public_key=options.public_key.expanduser().resolve(),
        cosign=options.cosign,
        rekor_url=options.rekor_url,
    ):
        result[item.outcome].append(
            {"bundle": str(item.bundle), "error": item.error}
            if item.outcome == "failed"
            else str(item.bundle)
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 1 if result["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())

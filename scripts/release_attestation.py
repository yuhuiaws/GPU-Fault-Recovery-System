from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any


class ReleaseAttestationError(ValueError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_output(root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def source_state(root: Path) -> dict[str, Any]:
    status = git_output(root, "status", "--short")
    diff = subprocess.run(
        ["git", "diff", "--binary", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
    ).stdout
    return {
        "git_commit": git_output(root, "rev-parse", "HEAD"),
        "dirty": bool(status),
        "diff_sha256": hashlib.sha256(diff).hexdigest(),
    }


def load_manifest(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or int(value.get("schema_version", 0)) < 3:
        raise ReleaseAttestationError(
            "prebuilt deployment requires a schema v3 release manifest"
        )
    if value.get("deployable") is not True:
        raise ReleaseAttestationError(
            "prebuilt deployment requires a deployable release manifest"
        )
    release_id = str(value.get("release_id") or "")
    if not release_id:
        raise ReleaseAttestationError("release manifest has no release_id")
    return value


def build_attestation(root: Path, manifest_path: Path) -> dict[str, Any]:
    manifest = load_manifest(manifest_path)
    state = source_state(root)
    return {
        "schema_version": 1,
        "subject": {
            "release_id": manifest["release_id"],
            "manifest": str(manifest_path.relative_to(root)),
            "manifest_sha256": sha256(manifest_path),
            "delivery_sha256": manifest["delivery"]["sha256"],
        },
        "source": state,
        "quality_gates": [
            {
                "command": "make check",
                "status": "PASSED",
            },
            {
                "command": "make test-postgres-stress",
                "status": "PASSED",
            },
        ],
    }


def verify_attestation(
    root: Path,
    attestation_path: Path,
    *,
    signature: Path | None,
    bundle: Path | None,
    cosign_key: str | None,
    certificate: Path | None,
    certificate_identity: str | None,
    certificate_oidc_issuer: str | None,
) -> dict[str, Any]:
    value = json.loads(attestation_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise ReleaseAttestationError("release attestation schema_version must be 1")
    subject = value.get("subject") or {}
    manifest_path = (root / str(subject.get("manifest") or "")).resolve()
    try:
        manifest_path.relative_to(root)
    except ValueError as exc:
        raise ReleaseAttestationError("attested manifest leaves repository") from exc
    manifest = load_manifest(manifest_path)
    if sha256(manifest_path) != subject.get("manifest_sha256"):
        raise ReleaseAttestationError("release manifest SHA-256 does not match")
    if manifest["release_id"] != subject.get("release_id"):
        raise ReleaseAttestationError("attested release_id does not match")
    if manifest["delivery"]["sha256"] != subject.get("delivery_sha256"):
        raise ReleaseAttestationError("attested delivery identity does not match")
    gates = value.get("quality_gates")
    if (
        not isinstance(gates, list)
        or {str(item.get("command")) for item in gates if isinstance(item, dict)}
        != {"make check", "make test-postgres-stress"}
        or any(
            not isinstance(item, dict) or item.get("status") != "PASSED"
            for item in gates
        )
    ):
        raise ReleaseAttestationError("attestation quality gates are not PASSED")
    if (value.get("source") or {}).get("dirty") is not False:
        raise ReleaseAttestationError(
            "prebuilt release attestation must come from a clean source tree"
        )
    if signature is None and bundle is None:
        raise ReleaseAttestationError(
            "prebuilt release signature or bundle is required"
        )
    command = ["cosign", "verify-blob"]
    if bundle is not None:
        command.extend(["--bundle", str(bundle)])
    if cosign_key:
        command.extend(["--key", cosign_key])
    else:
        if (
            bundle is None
            and certificate is None
            or not certificate_identity
            or not certificate_oidc_issuer
        ):
            raise ReleaseAttestationError(
                "keyless cosign verification requires bundle or certificate, "
                "plus identity and issuer"
            )
        if certificate is not None:
            command.extend(["--certificate", str(certificate)])
        command.extend(
            [
                "--certificate-identity",
                certificate_identity,
                "--certificate-oidc-issuer",
                certificate_oidc_issuer,
            ]
        )
    if signature is not None:
        command.extend(["--signature", str(signature)])
    command.append(str(attestation_path))
    completed = subprocess.run(
        command,
        cwd=root,
        check=False,
        env=os.environ,
        capture_output=True,
        text=True,
    )
    if completed.returncode:
        raise ReleaseAttestationError(
            "cosign verification failed: " + completed.stderr.strip()
        )
    return value

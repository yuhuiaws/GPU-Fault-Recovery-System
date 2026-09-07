from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any


class ReleaseAttestationError(ValueError):
    pass


PRODUCTION_QUALITY_GATES = (
    "make check",
    "make test-postgres-stress",
)
PROMOTED_QUALITY_GATES = (
    "python scripts/ci_gate.py verify",
    "pytest tests/test_artifact_consistency.py",
)
STAGING_QUALITY_GATES = (
    "make public-release-check",
    "make test-impact",
    "pytest tests/test_artifact_consistency.py",
)


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


def load_manifest(
    path: Path,
    *,
    allow_staging: bool = False,
) -> dict[str, Any]:
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
    tier = value.get("staging_only", False)
    if not isinstance(tier, bool):
        raise ReleaseAttestationError("release manifest staging_only must be a boolean")
    if tier and not allow_staging:
        raise ReleaseAttestationError(
            "staging-only release requires explicit staging authorization"
        )
    return value


def build_attestation(
    root: Path,
    manifest_path: Path,
    *,
    staging_only: bool = False,
    impact_base: str | None = None,
    impact_plan_path: Path | None = None,
    ci_gate_path: Path | None = None,
    sbom_dir: Path | None = None,
) -> dict[str, Any]:
    manifest = load_manifest(manifest_path, allow_staging=staging_only)
    if manifest.get("staging_only", False) is not staging_only:
        raise ReleaseAttestationError(
            "release manifest tier does not match attestation mode"
        )
    if staging_only and not impact_base:
        raise ReleaseAttestationError(
            "staging attestation requires an impact test base"
        )
    if not staging_only and impact_base is not None:
        raise ReleaseAttestationError(
            "production attestation must not declare an impact test base"
        )
    if not staging_only and impact_plan_path is not None:
        raise ReleaseAttestationError(
            "production attestation must not declare an impact plan"
        )
    if staging_only and ci_gate_path is not None:
        raise ReleaseAttestationError(
            "staging attestation must not declare a production CI gate"
        )
    state = source_state(root)
    result = {
        "schema_version": 1,
        "release_tier": "staging" if staging_only else "production",
        "subject": {
            "release_id": manifest["release_id"],
            "manifest": str(manifest_path.relative_to(root)),
            "manifest_sha256": sha256(manifest_path),
            "delivery_sha256": manifest["delivery"]["sha256"],
        },
        "source": state,
        "quality_gates": [
            {"command": command, "status": "PASSED"}
            for command in (
                STAGING_QUALITY_GATES
                if staging_only
                else PROMOTED_QUALITY_GATES
                if ci_gate_path is not None
                else PRODUCTION_QUALITY_GATES
            )
        ],
    }
    if impact_base is not None:
        result["impact_base"] = impact_base
    if impact_plan_path is not None:
        resolved_plan = impact_plan_path.resolve()
        try:
            relative_plan = resolved_plan.relative_to(root.resolve())
        except ValueError as exc:
            raise ReleaseAttestationError(
                "staging impact plan leaves repository"
            ) from exc
        if not resolved_plan.is_file():
            raise ReleaseAttestationError("staging impact plan is missing")
        result["impact_plan"] = {
            "path": str(relative_plan),
            "sha256": sha256(resolved_plan),
        }
    if ci_gate_path is not None:
        resolved_gate = ci_gate_path.resolve()
        try:
            relative_gate = resolved_gate.relative_to(root.resolve())
        except ValueError as exc:
            raise ReleaseAttestationError(
                "production CI gate leaves repository"
            ) from exc
        if not resolved_gate.is_file():
            raise ReleaseAttestationError("production CI gate is missing")
        result["ci_gate"] = {
            "path": str(relative_gate),
            "sha256": sha256(resolved_gate),
        }
    if sbom_dir is not None:
        sbom_entries = _sbom_entries(root, sbom_dir)
        if sbom_entries:
            result["sbom"] = sbom_entries
    return result


def _sbom_entries(root: Path, sbom_dir: Path) -> list[dict[str, str]]:
    """Path and digest of every CycloneDX document under ``sbom_dir``.

    The SBOM is generated from the shipped lock files by ``make sbom`` and is
    not itself signed; listing each file's SHA-256 here puts it under the
    cosign signature of the attestation, the same way the CI gate and the
    staging impact plan are bound. An empty or missing directory yields no
    entries: outside CI the generator may be absent, and ``make sbom`` is the
    place that decides whether that is acceptable.
    """
    resolved_dir = sbom_dir.resolve()
    try:
        resolved_dir.relative_to(root.resolve())
    except ValueError as exc:
        raise ReleaseAttestationError("SBOM directory leaves repository") from exc
    if not resolved_dir.is_dir():
        return []
    return [
        {
            "path": str(path.relative_to(root.resolve())),
            "sha256": sha256(path),
        }
        for path in sorted(resolved_dir.glob("*.cdx.json"))
        if path.is_file()
    ]


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
    allow_staging: bool = False,
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
    manifest = load_manifest(manifest_path, allow_staging=allow_staging)
    if sha256(manifest_path) != subject.get("manifest_sha256"):
        raise ReleaseAttestationError("release manifest SHA-256 does not match")
    if manifest["release_id"] != subject.get("release_id"):
        raise ReleaseAttestationError("attested release_id does not match")
    if manifest["delivery"]["sha256"] != subject.get("delivery_sha256"):
        raise ReleaseAttestationError("attested delivery identity does not match")
    staging_only = manifest.get("staging_only") is True
    expected_tier = "staging" if staging_only else "production"
    if value.get("release_tier", "production") != expected_tier:
        raise ReleaseAttestationError("attestation release tier does not match")
    impact_base = value.get("impact_base")
    if staging_only and (not isinstance(impact_base, str) or not impact_base.strip()):
        raise ReleaseAttestationError("staging attestation has no impact test base")
    if not staging_only and impact_base is not None:
        raise ReleaseAttestationError(
            "production attestation declares a staging impact test base"
        )
    impact_plan = value.get("impact_plan")
    if impact_plan is not None:
        if not staging_only or not isinstance(impact_plan, dict):
            raise ReleaseAttestationError("attestation impact plan is invalid")
        plan_path = (root / str(impact_plan.get("path") or "")).resolve()
        try:
            plan_path.relative_to(root.resolve())
        except ValueError as exc:
            raise ReleaseAttestationError(
                "attested impact plan leaves repository"
            ) from exc
        if not plan_path.is_file() or sha256(plan_path) != impact_plan.get("sha256"):
            raise ReleaseAttestationError("attested impact plan SHA-256 does not match")
    ci_gate = value.get("ci_gate")
    if ci_gate is not None:
        if staging_only or not isinstance(ci_gate, dict):
            raise ReleaseAttestationError("attestation CI gate is invalid")
        gate_path = (root / str(ci_gate.get("path") or "")).resolve()
        try:
            gate_path.relative_to(root.resolve())
        except ValueError as exc:
            raise ReleaseAttestationError("attested CI gate leaves repository") from exc
        if not gate_path.is_file() or sha256(gate_path) != ci_gate.get("sha256"):
            raise ReleaseAttestationError("attested CI gate SHA-256 does not match")
    sbom = value.get("sbom")
    if sbom is not None:
        if not isinstance(sbom, list) or not sbom:
            raise ReleaseAttestationError("attestation SBOM list is invalid")
        for entry in sbom:
            if not isinstance(entry, dict):
                raise ReleaseAttestationError("attestation SBOM entry is invalid")
            sbom_path = (root / str(entry.get("path") or "")).resolve()
            try:
                sbom_path.relative_to(root.resolve())
            except ValueError as exc:
                raise ReleaseAttestationError(
                    "attested SBOM leaves repository"
                ) from exc
            if not sbom_path.is_file() or sha256(sbom_path) != entry.get("sha256"):
                raise ReleaseAttestationError("attested SBOM SHA-256 does not match")
    expected_commands = set(
        STAGING_QUALITY_GATES
        if staging_only
        else PROMOTED_QUALITY_GATES
        if ci_gate is not None
        else PRODUCTION_QUALITY_GATES
    )
    gates = value.get("quality_gates")
    if (
        not isinstance(gates, list)
        or {str(item.get("command")) for item in gates if isinstance(item, dict)}
        != expected_commands
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

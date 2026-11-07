"""The signed release attestation: what it binds and how it is verified.

Verification is ``cosign verify-blob`` with the site key (or a keyless
identity) over the attestation, after every digest it binds was checked
against the files on disk. A bundle with a transparency-log entry takes the
same command as always; when cosign's live TUF fetch fails, it is repeated once
against the site's pinned trust root (``signing_transparency``). A bundle
without a log entry is refused unless the signed attestation declares unlogged
signing (``"signing": {"transparency_log": "unlogged", ...}``) and one of two
approvals holds: ``allow_unlogged`` -- the operator's
``--allow-unlogged-signature`` on the command that is running -- or the
cluster's own release record, read through ``live_release``, naming this
release and its delivery digest as the committed live release. That record is
written by the release engine with the operator's cluster credentials, so a
holder of the signing key alone can satisfy neither; a forged attestation
that passes the second one attests exactly the delivery already deployed.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

if __package__:
    from scripts import signing_transparency as transparency
else:
    import signing_transparency as transparency


class ReleaseAttestationError(ValueError):
    pass


@dataclass(frozen=True)
class SignatureVerification:
    """How the attestation's signature was accepted, for the release record."""

    #: ``logged``, ``unlogged`` or ``unknown`` (no bundle, another format).
    transparency_log: str
    declaration: Mapping[str, str] | None = None
    #: ``consent`` or ``live_release`` for an unlogged bundle, else None.
    approval: str | None = None
    #: ``signing_transparency.TrustPin.usage`` when the pinned root was used.
    trust_pin: Mapping[str, Any] | None = None

    def record(self, attestation: Path) -> dict[str, Any]:
        """The ``signing`` block of a release or source-deploy record."""

        value: dict[str, Any] = {
            "transparency_log": self.transparency_log,
            "attestation_sha256": sha256(attestation),
        }
        if self.declaration is not None:
            value["reason"] = self.declaration["reason"]
            value["declared_at"] = self.declaration["declared_at"]
        if self.approval is not None:
            value["approval"] = self.approval
        # Which Sigstore trust root the verification read: the live TUF one,
        # the site's pinned copy, or none (a key-only check without a log entry).
        value["trusted_root"] = (
            "pinned" if self.trust_pin else "none" if self.approval else "tuf"
        )
        if self.trust_pin:
            value["trust_pin"] = dict(self.trust_pin)
        return value


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
            "prebuilt deployment requires a schema v3 or v4 release manifest"
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
    unlogged_signature_reason: str | None = None,
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
    if unlogged_signature_reason is not None:
        # Inside the signed bytes: the only place a verifier trusts it.
        result[transparency.DECLARATION_FIELD] = transparency.unlogged_declaration(
            unlogged_signature_reason
        )
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
    allow_unlogged: bool = False,
    live_release: Callable[[], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    return verify_attestation_signature(
        root,
        attestation_path,
        signature=signature,
        bundle=bundle,
        cosign_key=cosign_key,
        certificate=certificate,
        certificate_identity=certificate_identity,
        certificate_oidc_issuer=certificate_oidc_issuer,
        allow_staging=allow_staging,
        allow_unlogged=allow_unlogged,
        live_release=live_release,
    )[0]


def _live_release_approves(
    live_release: Callable[[], Mapping[str, Any]] | None,
    subject: Mapping[str, Any],
) -> bool:
    """Whether the cluster's committed live release is this attestation's."""

    if live_release is None:
        return False
    try:
        state = live_release()
    except Exception as exc:  # noqa: BLE001 - any unreadable record approves nothing
        raise ReleaseAttestationError(
            "the release is signed without a transparency-log entry and the live "
            f"release record cannot be read to approve it: {exc}"
        ) from None
    return (
        state.get("transaction_committed") is True
        and str(state.get("phase") or "").lower() in {"complete", "completed"}
        and bool(subject.get("release_id"))
        and state.get("release_id") == subject.get("release_id")
        and bool(subject.get("delivery_sha256"))
        and state.get("release_delivery_sha256") == subject.get("delivery_sha256")
    )


def verify_attestation_signature(
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
    allow_unlogged: bool = False,
    live_release: Callable[[], Mapping[str, Any]] | None = None,
) -> tuple[dict[str, Any], SignatureVerification]:
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
    try:
        declaration = transparency.payload_declaration(value)
    except transparency.SigningTransparencyError as exc:
        raise ReleaseAttestationError(f"release attestation {exc}") from None
    transparency_log = (
        transparency.bundle_transparency(bundle)
        if bundle is not None
        else transparency.UNKNOWN
    )
    approval: str | None = None
    if transparency_log == transparency.UNLOGGED:
        if allow_unlogged:
            approval = "consent"
        elif (
            cosign_key
            and declaration is not None
            and _live_release_approves(live_release, subject)
        ):
            approval = "live_release"
        try:
            transparency.require_unlogged_acceptance(
                artifact=attestation_path,
                declaration=declaration,
                keyed=bool(cosign_key),
                approved=approval is not None,
                approval_hint=transparency.UNLOGGED_RELEASE_REFUSAL_HINT,
            )
        except transparency.SigningTransparencyError as exc:
            raise ReleaseAttestationError(str(exc)) from None
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
    if approval is not None:
        command.append(transparency.IGNORE_TLOG_FLAG)
    command.append(str(attestation_path))
    completed = transparency.run_verify_blob(
        command,
        pin_directory=transparency.pin_directory_for_key(cosign_key),
        cwd=root,
        environment=os.environ,
    )
    if completed.returncode:
        raise ReleaseAttestationError(
            "cosign verification failed: " + completed.stderr.strip()
        )
    if approval is not None and declaration is not None:
        print(
            f"release-attestation: {value['subject'].get('release_id')} is signed "
            "WITHOUT a transparency-log entry "
            f"(declared {declaration['declared_at']}: {declaration['reason']}); "
            f"accepted by {approval.replace('_', ' ')}",
            file=sys.stderr,
            flush=True,
        )
    return value, SignatureVerification(
        transparency_log=transparency_log,
        declaration=declaration,
        approval=approval,
        trust_pin=completed.pinned.usage() if completed.pinned else None,
    )

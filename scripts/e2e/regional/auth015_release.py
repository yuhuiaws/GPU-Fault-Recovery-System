"""Explicit, plan-bindable signed-release inputs for the AUTH015 subproof."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scripts.component_wheels import component_source_digest
from scripts.e2e.regional.auth015_protocol import Auth015ProofError
from scripts.release_attestation import verify_attestation
from scripts.release_identity import canonical_sha256

SHA256 = re.compile(r"[0-9a-f]{64}")
MAX_INPUT_BYTES = 8 * 1024 * 1024


@dataclass(frozen=True)
class Auth015ReleaseInputs:
    descriptor_path: Path
    repository_root: Path
    attestation_path: Path
    bundle_path: Path
    public_key_path: Path
    public_key_sha256: str
    expected_release_id: str
    allow_staging: bool = False
    expected_identity: dict[str, str] | None = None


@dataclass(frozen=True)
class VerifiedAuth015Release:
    release_id: str
    input_identity: dict[str, str]
    delivery_sha256: str
    control_plane_digest: str
    node_digest: str
    node_wheel_sha256: str
    bundle_sha256: str
    runtime_image: str
    agent_protocol_version: int


def read_input(path: Path) -> bytes:
    try:
        if not path.is_file():
            raise Auth015ProofError("AUTH015 release input is not a regular file")
        with path.open("rb") as handle:
            value = handle.read(MAX_INPUT_BYTES + 1)
        if not value or len(value) > MAX_INPUT_BYTES:
            raise Auth015ProofError("AUTH015 release input is empty or exceeds limit")
        return value
    except OSError:
        raise Auth015ProofError("AUTH015 release input cannot be read") from None


def input_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(read_input(path))
    except ValueError:
        raise Auth015ProofError("AUTH015 release input is not valid JSON") from None
    if not isinstance(value, dict):
        raise Auth015ProofError("AUTH015 release input must be an object")
    return value


def load_release_inputs(
    path: Path, *, expected_release_id: str
) -> Auth015ReleaseInputs:
    document = input_object(path)
    required = {
        "schema_version",
        "repository_root",
        "attestation",
        "bundle",
        "cosign_public_key",
        "cosign_public_key_sha256",
    }
    if (
        set(document) - {"allow_staging"} != required
        or type(document["schema_version"]) is not int
        or document["schema_version"] != 1
        or type(document.get("allow_staging", False)) is not bool
        or not isinstance(expected_release_id, str)
        or not expected_release_id
        or any(ord(character) < 32 for character in expected_release_id)
        or any(
            not isinstance(document[field], str) or not document[field].strip()
            for field in required - {"schema_version"}
        )
        or SHA256.fullmatch(document["cosign_public_key_sha256"]) is None
    ):
        raise Auth015ProofError("AUTH015 release descriptor fields are invalid")
    root = (path.parent / document["repository_root"]).resolve()
    if not root.is_dir():
        raise Auth015ProofError("AUTH015 release repository is unavailable")
    return Auth015ReleaseInputs(
        descriptor_path=path.resolve(),
        repository_root=root,
        attestation_path=(root / document["attestation"]).resolve(),
        bundle_path=(root / document["bundle"]).resolve(),
        public_key_path=(root / document["cosign_public_key"]).resolve(),
        public_key_sha256=document["cosign_public_key_sha256"],
        expected_release_id=expected_release_id,
        allow_staging=document.get("allow_staging", False),
    )


def manifest_path(inputs: Auth015ReleaseInputs) -> Path:
    attestation = input_object(inputs.attestation_path)
    try:
        relative = attestation["subject"]["manifest"]
        if (
            not isinstance(relative, str)
            or not relative
            or Path(relative).is_absolute()
        ):
            raise ValueError
        path = (inputs.repository_root / relative).resolve()
        path.relative_to(inputs.repository_root)
        return path
    except (KeyError, TypeError, ValueError):
        raise Auth015ProofError("AUTH015 attested manifest path is invalid") from None


def release_input_identity(inputs: Auth015ReleaseInputs) -> dict[str, str]:
    paths = {
        "descriptor": inputs.descriptor_path,
        "attestation": inputs.attestation_path,
        "signature_bundle": inputs.bundle_path,
        "public_key": inputs.public_key_path,
        "manifest": manifest_path(inputs),
    }
    values = {
        name: hashlib.sha256(read_input(path)).hexdigest()
        for name, path in paths.items()
    }
    if values["public_key"] != inputs.public_key_sha256:
        raise Auth015ProofError("AUTH015 release public-key pin differs")
    return {**values, "release_id": inputs.expected_release_id}


def verify_release_inputs(inputs: Auth015ReleaseInputs) -> VerifiedAuth015Release:
    before = release_input_identity(inputs)
    if inputs.expected_identity is not None and before != inputs.expected_identity:
        raise Auth015ProofError(
            "AUTH015 signed-release inputs differ from the approved plan"
        )
    public_key = read_input(inputs.public_key_path).strip()
    if not (
        public_key.startswith(b"-----BEGIN PUBLIC KEY-----")
        and public_key.endswith(b"-----END PUBLIC KEY-----")
    ):
        raise Auth015ProofError("AUTH015 requires an explicitly pinned public key")
    try:
        verify_attestation(
            inputs.repository_root,
            inputs.attestation_path,
            signature=None,
            bundle=inputs.bundle_path,
            cosign_key=str(inputs.public_key_path),
            certificate=None,
            certificate_identity=None,
            certificate_oidc_issuer=None,
            allow_staging=inputs.allow_staging,
        )
    except Exception:
        raise Auth015ProofError("AUTH015 signed-release verification failed") from None
    manifest = input_object(manifest_path(inputs))
    if before != release_input_identity(inputs):
        raise Auth015ProofError("AUTH015 release inputs changed during verification")
    try:
        delivery = manifest["delivery"]
        control = manifest["components"]["control_plane"]
        node = manifest["components"]["node_runtime"]
        image = delivery["images"]["runtime"]
        protocol = manifest["protocol_versions"]["agent"]
        digests = [
            delivery["sha256"],
            control["module_digest"],
            node["module_digest"],
            node["wheel_sha256"],
            manifest["bundle_sha256"],
        ]
        if (
            type(manifest["schema_version"]) is not int
            or manifest["schema_version"] != 4
            or manifest["release_id"] != inputs.expected_release_id
            or any(
                not isinstance(value, str) or SHA256.fullmatch(value) is None
                for value in digests
            )
            or canonical_sha256(
                {key: value for key, value in delivery.items() if key != "sha256"}
            )
            != delivery["sha256"]
            or not isinstance(image["reference"], str)
            or re.fullmatch(r"\S+@sha256:[0-9a-f]{64}", image["reference"]) is None
            or image["components"]["control_plane"]["module_digest"]
            != control["module_digest"]
            or type(protocol) is not int
            or protocol < 1
            or node["module_digest"] != component_source_digest("node_runtime")
        ):
            raise ValueError
    except (KeyError, TypeError, ValueError):
        raise Auth015ProofError(
            "AUTH015 signed release cannot bind the current protocol"
        ) from None
    return VerifiedAuth015Release(
        release_id=inputs.expected_release_id,
        input_identity=before,
        delivery_sha256=delivery["sha256"],
        control_plane_digest=control["module_digest"],
        node_digest=node["module_digest"],
        node_wheel_sha256=node["wheel_sha256"],
        bundle_sha256=manifest["bundle_sha256"],
        runtime_image=image["reference"],
        agent_protocol_version=protocol,
    )

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec

from scripts import release_attestation
from scripts.e2e.regional import auth015_release as release
from scripts.release_identity import canonical_sha256


class ReleaseFiles:
    def __init__(self, root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self.root = root
        self.descriptor_path = root / "release-proof.json"
        self.manifest_path = root / "manifest.json"
        self.attestation_path = root / "attestation.json"
        self.bundle_path = root / "signature.bundle.json"
        self.public_key_path = root / "cosign.pub"
        self.public_key_path.write_bytes(
            ec.generate_private_key(ec.SECP256R1())
            .public_key()
            .public_bytes(
                serialization.Encoding.PEM,
                serialization.PublicFormat.SubjectPublicKeyInfo,
            )
        )
        self.bundle_path.write_text("{}", encoding="ascii")
        self.delivery = {
            "images": {
                "runtime": {
                    "reference": "registry.invalid/runtime@sha256:" + "f" * 64,
                    "components": {"control_plane": {"module_digest": "f" * 64}},
                }
            }
        }
        self.delivery["sha256"] = canonical_sha256(self.delivery)
        self.manifest: dict[str, Any] = {
            "schema_version": 4,
            "deployable": True,
            "release_id": "release-a",
            "delivery": self.delivery,
            "components": {
                "control_plane": {"module_digest": "f" * 64},
                "node_runtime": {"module_digest": "b" * 64, "wheel_sha256": "a" * 64},
            },
            "bundle_sha256": "c" * 64,
            "protocol_versions": {"agent": 3},
        }
        self.descriptor = {
            "schema_version": 1,
            "repository_root": str(root),
            "attestation": self.attestation_path.name,
            "bundle": self.bundle_path.name,
            "cosign_public_key": self.public_key_path.name,
            "cosign_public_key_sha256": hashlib.sha256(
                self.public_key_path.read_bytes()
            ).hexdigest(),
        }
        self.write_descriptor()
        self.write_manifest()
        self.commands: list[list[str]] = []
        monkeypatch.setattr(release_attestation.subprocess, "run", self.cosign)
        monkeypatch.setattr(release, "component_source_digest", lambda _: "b" * 64)

    def cosign(self, command, **kwargs):
        assert command[:2] == ["cosign", "verify-blob"], (
            "only the external signature verifier may be simulated"
        )
        self.commands.append(command)
        return subprocess.CompletedProcess(command, 0, "", "")

    def write_descriptor(self) -> None:
        self.descriptor_path.write_text(json.dumps(self.descriptor), encoding="utf-8")

    def write_manifest(self) -> None:
        self.manifest_path.write_text(json.dumps(self.manifest), encoding="utf-8")
        self.attestation_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "subject": {
                        "release_id": self.manifest["release_id"],
                        "manifest": self.manifest_path.name,
                        "manifest_sha256": hashlib.sha256(
                            self.manifest_path.read_bytes()
                        ).hexdigest(),
                        "delivery_sha256": self.manifest["delivery"]["sha256"],
                    },
                    "source": {"dirty": False},
                    "quality_gates": [
                        {"command": command, "status": "PASSED"}
                        for command in release_attestation.PRODUCTION_QUALITY_GATES
                    ],
                }
            ),
            encoding="utf-8",
        )

    def inputs(self, expected_release_id: str = "release-a"):
        return release.load_release_inputs(
            self.descriptor_path, expected_release_id=expected_release_id
        )

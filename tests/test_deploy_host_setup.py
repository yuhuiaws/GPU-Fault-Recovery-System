from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from gpu_fault.admin_bootstrap_dependencies import (
    deploy_host_dependency_report,
    load_deploy_host_tool_manifest,
)
from scripts import deploy_host_bundle, setup_deploy_host

ROOT = Path(__file__).resolve().parents[1]


def test_deploy_host_lock_and_tool_manifest_are_complete() -> None:
    lock = (ROOT / "requirements/deploy-host.lock").read_text(encoding="utf-8")
    manifest = load_deploy_host_tool_manifest()
    names = {item["name"] for item in manifest["tools"]}

    assert "--hash=sha256:" in lock
    for package in ("build==", "boto3==", "pytest==", "ruff==", "yamllint=="):
        assert package in lock, f"deploy-host lock is missing {package}"
    assert {
        "aws",
        "cosign",
        "docker-daemon",
        "docker-buildx",
        "kubectl",
        "helm",
    } <= names


def _bundle_tree(tmp_path: Path) -> Path:
    root = tmp_path / deploy_host_bundle.BUNDLE_ROOT
    (root / "requirements").mkdir(parents=True)
    (root / "wheelhouse").mkdir()
    (root / "requirements/deploy-host.lock").write_text(
        "example==1\n", encoding="utf-8"
    )
    wheel = root / "wheelhouse/example-1-py3-none-any.whl"
    wheel.write_bytes(b"wheel")
    wheel.chmod(0o644)
    compatibility = deploy_host_bundle.host_compatibility()
    deploy_host_bundle.write_bundle_manifest(
        root,
        metadata={
            "compatibility": compatibility,
            "platform_id": deploy_host_bundle.bundle_platform_id(compatibility),
            "project_version": "1.0",
            "project_wheel": wheel.relative_to(root).as_posix(),
            "python": "3.12",
            "requirements": {"deploy_host": "requirements/deploy-host.lock"},
            "source": {"git_commit": "a" * 40, "dirty": False},
            "tool_manifest": "deploy-host-tools.json",
        },
    )
    return root


def test_deploy_host_bundle_is_deterministic_and_verified(tmp_path: Path) -> None:
    root = _bundle_tree(tmp_path)
    first = tmp_path / "first.tar.gz"
    second = tmp_path / "second.tar.gz"

    deploy_host_bundle.write_deterministic_archive(root, first)
    deploy_host_bundle.write_deterministic_archive(root, second)

    assert deploy_host_bundle.sha256_file(first) == deploy_host_bundle.sha256_file(
        second
    ), "identical deploy-host inputs produced different archives"
    extracted = deploy_host_bundle.extract_verified_bundle(
        first, tmp_path / "extracted"
    )
    assert deploy_host_bundle.verify_bundle_tree(extracted)["project_version"] == "1.0"


def test_deploy_host_bundle_rejects_tampered_payload(tmp_path: Path) -> None:
    root = _bundle_tree(tmp_path)
    archive = tmp_path / "bundle.tar.gz"
    deploy_host_bundle.write_deterministic_archive(root, archive)
    extracted = deploy_host_bundle.extract_verified_bundle(
        archive, tmp_path / "extracted"
    )
    (extracted / "requirements/deploy-host.lock").write_text(
        "tampered\n", encoding="utf-8"
    )

    with pytest.raises(
        deploy_host_bundle.DeployHostBundleError, match="digest mismatch"
    ):
        deploy_host_bundle.verify_bundle_tree(extracted)


def test_deploy_host_bundle_rejects_incompatible_platform(tmp_path: Path) -> None:
    root = _bundle_tree(tmp_path)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["compatibility"]["architecture"] = "different-architecture"

    with pytest.raises(
        deploy_host_bundle.DeployHostBundleError, match="incompatible with this host"
    ):
        deploy_host_bundle.validate_host_compatibility(manifest)


def test_setup_requires_signed_bundle_or_explicit_network(tmp_path: Path) -> None:
    with pytest.raises(setup_deploy_host.DeployHostSetupError, match="signed --bundle"):
        setup_deploy_host.setup_deploy_host(
            repo_root=tmp_path,
            venv=tmp_path / ".venv",
            python="python3.12",
            archive=None,
            signature_bundle=None,
            cosign_key=None,
            certificate_identity=None,
            certificate_oidc_issuer=None,
            allow_unsigned=False,
            allow_network=False,
            allow_source_mismatch=False,
        )


def test_bundle_signature_verification_uses_cosign(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive = tmp_path / "bundle.tar.gz"
    signature = tmp_path / "bundle.sigstore.json"
    archive.write_bytes(b"bundle")
    signature.write_text("{}", encoding="utf-8")
    commands = []
    monkeypatch.setattr(
        setup_deploy_host,
        "_run",
        lambda arguments, **_kwargs: commands.append(list(arguments)) or "",
    )

    setup_deploy_host.verify_bundle_signature(
        archive,
        signature_bundle=signature,
        cosign_key="/secure/cosign.pub",
        certificate_identity=None,
        certificate_oidc_issuer=None,
        allow_unsigned=False,
    )

    assert commands == [
        [
            "cosign",
            "verify-blob",
            "--bundle",
            str(signature),
            "--key",
            "/secure/cosign.pub",
            str(archive),
        ]
    ]


def test_bundle_source_binding_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    outputs = iter(("current-commit", " M changed.py"))
    monkeypatch.setattr(
        setup_deploy_host, "_run", lambda *_args, **_kwargs: next(outputs)
    )

    with pytest.raises(
        setup_deploy_host.DeployHostSetupError, match="source does not match"
    ):
        setup_deploy_host.validate_bundle_source(
            {"source": {"git_commit": "expected-commit", "dirty": False}},
            repo_root=tmp_path,
            allow_source_mismatch=False,
        )


def test_dependency_report_records_versions_without_credentials() -> None:
    manifest = {
        "schema_version": 1,
        "python": {"major": 3, "minor": 12},
        "tools": [
            {
                "name": "example",
                "executable": "example",
                "command": ["example", "--version"],
            }
        ],
    }

    def run(arguments, **_kwargs):
        return subprocess.CompletedProcess(arguments, 0, "example 1.2.3\n", "")

    report = deploy_host_dependency_report(
        manifest=manifest, which=lambda _name: "/opt/tools/example", runner=run
    )

    assert report["healthy"] is True
    assert report["tools"] == [
        {
            "name": "example",
            "executable": "/opt/tools/example",
            "status": "PASS",
            "version": "example 1.2.3",
        }
    ]

"""Manifest-root location and canonical artifact paths, without release builds."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import IO, Any

import pytest

import gpu_fault_release
from gpu_fault.admin.release_engine import build_release
from gpu_fault.admin.site import RenderedSite
from gpu_fault_release import regional_release_config as config
from gpu_fault_release.regional_release_images import node_dependency_environment
from scripts import release_attestation
from tests.regional._release_orchestrator_support import ROOT, config_file
from tests.regional.test_cov95_release_config import delivery_manifest, reseal

RELEASE_ID = "abc123def456"
ENTRYPOINT = "deploy/control-plane/regional/rollout-regional-release.sh"
ARTIFACT_NAMES = {
    "wheel": "control-plane.whl",
    "executor_wheel": "executor.whl",
    "node_wheel": "node-runtime.whl",
    "bundle": "node-installer.tar.gz",
}


def artifact_paths(root: Path) -> dict[str, Path]:
    return {
        field: root / "dist" / RELEASE_ID / filename
        for field, filename in ARTIFACT_NAMES.items()
    }


def write_snapshot(root: Path, *, schema: int = 4) -> Path:
    entrypoint = root / ENTRYPOINT
    entrypoint.parent.mkdir(parents=True)
    entrypoint.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    (root / "scripts").mkdir()
    paths = artifact_paths(root)
    paths["wheel"].parent.mkdir(parents=True)
    for field, path in paths.items():
        path.write_bytes(f"synthetic {field}".encode("ascii"))
    manifest: dict[str, Any] = {
        "schema_version": schema,
        "release_id": RELEASE_ID,
        "wheel": paths["wheel"].relative_to(root).as_posix(),
        "wheel_sha256": hashlib.sha256(paths["wheel"].read_bytes()).hexdigest(),
        "bundle": paths["bundle"].relative_to(root).as_posix(),
        "bundle_sha256": hashlib.sha256(paths["bundle"].read_bytes()).hexdigest(),
    }
    components = {
        component: {
            "wheel": paths[field].relative_to(root).as_posix(),
            "wheel_sha256": hashlib.sha256(paths[field].read_bytes()).hexdigest(),
            "module_digest": character * 64,
        }
        for field, component, character in (
            ("wheel", "control_plane", "1"),
            ("executor_wheel", "executor", "2"),
            ("node_wheel", "node_runtime", "3"),
        )
    }
    if schema >= 2:
        manifest["components"] = components
    if schema >= 3:
        delivery, template_components = delivery_manifest(schema)
        manifest.update(delivery)
        components["node_bundle"] = template_components["node_bundle"]
        manifest["delivery"]["components"]["aurora_refresh"] = {"sha256": "a" * 64}
        if schema == 4:
            images = manifest["delivery"]["images"]
            for image, component, character in (
                ("runtime", "control_plane", "b"),
                ("executor", "executor", "c"),
            ):
                images[image] = {
                    "reference": f"registry.example/{component}@sha256:"
                    + character * 64,
                    "components": {
                        component: {
                            field: components[component][field]
                            for field in ("wheel_sha256", "module_digest")
                        }
                    },
                }
        reseal(manifest)
    content = json.dumps(manifest)
    manifest_path = root / "dist/current-release.json"
    manifest_path.write_text(content, encoding="utf-8")
    (paths["wheel"].parent / "release.json").write_text(content, encoding="utf-8")
    return manifest_path


@pytest.mark.parametrize("schema", [1, 2, 3, 4])
@pytest.mark.parametrize("location", ["current", "immutable", "relative", "symlink"])
def test_manifest_paths_use_their_source_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, schema: int, location: str
) -> None:
    snapshot = tmp_path / "source snapshot"
    manifest_path = write_snapshot(snapshot, schema=schema)
    if location == "immutable":
        manifest_path = snapshot / "dist" / RELEASE_ID / "release.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if location == "symlink":
        operator = tmp_path / "operator"
        write_snapshot(operator, schema=schema)
        alias = operator / "dist/snapshot-release.json"
        alias.symlink_to(manifest_path)
        manifest_path = alias
    reference = (
        manifest_path.relative_to(tmp_path) if location == "relative" else manifest_path
    )
    monkeypatch.chdir(ROOT)

    artifacts = config.load_release_artifacts(
        {"manifest": str(reference)}, config_path=tmp_path / "materialized.json"
    )

    expected = artifact_paths(snapshot)
    if schema == 1:
        expected.update(executor_wheel=expected["wheel"], node_wheel=expected["wheel"])
    for field, path in expected.items():
        assert getattr(artifacts, field) == path, (
            f"{field} must come from the manifest's actual snapshot"
        )
    assert artifacts.manifest == manifest, "do not rewrite signed manifest contents"
    assert artifacts.manifest_schema_version == schema, "preserve the manifest schema"
    if schema >= 3:
        assert artifacts.delivery_identity == manifest["delivery"], (
            "local path resolution must preserve the complete delivery identity"
        )


@pytest.mark.parametrize("location", ["root", "file", "relative", "directory-link"])
def test_containing_root_canonicalizes_before_walking(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, location: str
) -> None:
    snapshot = tmp_path / "snapshot"
    manifest = write_snapshot(snapshot)
    path = snapshot if location == "root" else manifest
    if location == "relative":
        monkeypatch.chdir(tmp_path)
        path = path.relative_to(tmp_path)
    elif location == "directory-link":
        alias = tmp_path / "alias"
        alias.symlink_to(snapshot, target_is_directory=True)
        path = alias / "dist/current-release.json"

    assert gpu_fault_release.containing_repository_root(path) == snapshot, (
        "canonical file ancestry must determine the containing source root"
    )


@pytest.mark.parametrize(
    "invalid", ["directories-only", "entrypoint-directory", "scripts"]
)
def test_directory_names_alone_do_not_identify_a_repository(
    tmp_path: Path, invalid: str
) -> None:
    snapshot = tmp_path / "snapshot"
    manifest = write_snapshot(snapshot)
    if invalid == "scripts":
        (snapshot / "scripts").rmdir()
    else:
        (snapshot / ENTRYPOINT).unlink()
        if invalid == "entrypoint-directory":
            (snapshot / ENTRYPOINT).mkdir()

    assert gpu_fault_release.containing_repository_root(manifest) is None, (
        "root discovery requires directory anchors and the rollout entrypoint file"
    )


def test_nested_snapshot_wins_over_outer_repository_and_directory_decoy(
    tmp_path: Path,
) -> None:
    outer = tmp_path / "outer"
    write_snapshot(outer)
    snapshot = outer / "snapshots/current"
    manifest = write_snapshot(snapshot)
    for directory in ("deploy/control-plane/regional", "scripts"):
        (manifest.parent / directory).mkdir(parents=True)

    artifacts = config.load_release_artifacts(
        {"manifest": str(manifest)}, config_path=outer / "materialized.json"
    )

    assert artifacts.wheel == artifact_paths(snapshot)["wheel"], (
        "the nearest real source root wins over outer roots and directory-only decoys"
    )


@pytest.mark.parametrize("directory_decoy", [False, True])
def test_external_legacy_manifest_retains_engine_root_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, directory_decoy: bool
) -> None:
    engine = tmp_path / "engine"
    original = write_snapshot(engine, schema=1)
    external = tmp_path / "external"
    external.mkdir()
    if directory_decoy:
        for directory in ("deploy/control-plane/regional", "scripts"):
            (external / directory).mkdir(parents=True)
    manifest = external / "release.json"
    manifest.write_bytes(original.read_bytes())
    monkeypatch.setattr(config, "ROOT", engine)

    artifacts = config.load_release_artifacts(
        {"manifest": str(manifest)}, config_path=tmp_path / "materialized.json"
    )

    assert artifacts.wheel == artifact_paths(engine)["wheel"], (
        "a manifest outside any source tree keeps the legacy engine-root base"
    )
    assert artifacts.bundle == artifact_paths(engine)["bundle"], (
        "legacy fallback applies to the bundle as well as the shared wheel"
    )


@pytest.mark.parametrize("absolute", ["all", "mixed"])
def test_absolute_artifacts_are_not_rebased(tmp_path: Path, absolute: str) -> None:
    snapshot = tmp_path / "snapshot"
    manifest_path = write_snapshot(snapshot)
    manifest = json.loads(manifest_path.read_text())
    expected = artifact_paths(snapshot)
    external = tmp_path / "external"
    external.mkdir()
    for field, component in (
        ("wheel", "control_plane"),
        ("executor_wheel", "executor"),
        ("node_wheel", "node_runtime"),
        ("bundle", "node_bundle"),
    ):
        if absolute == "mixed" and field in {"wheel", "node_wheel"}:
            continue
        source = expected[field]
        target = external / source.name
        source.rename(target)
        expected[field] = target
        if field in {"wheel", "bundle"}:
            manifest[field] = str(target)
        if field != "bundle":
            manifest["components"][component]["wheel"] = str(target)
    manifest_path.write_text(json.dumps(manifest))

    artifacts = config.load_release_artifacts(
        {"manifest": str(manifest_path)}, config_path=tmp_path / "materialized.json"
    )

    for field, path in expected.items():
        assert getattr(artifacts, field) == path, (
            "explicit absolute paths and remaining snapshot-relative paths must coexist"
        )


@pytest.mark.parametrize("references", ["relative", "absolute", "mixed"])
def test_manifest_free_inputs_keep_the_config_directory_as_their_base(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, references: str
) -> None:
    snapshot = tmp_path / "snapshot"
    write_snapshot(snapshot)
    expected = artifact_paths(snapshot)
    release = {
        field: str(
            path
            if references == "absolute"
            or references == "mixed"
            and field == "executor_wheel"
            else path.relative_to(snapshot)
        )
        for field, path in expected.items()
    }
    monkeypatch.setattr(config, "ROOT", tmp_path / "different-engine")

    artifacts = config.load_release_artifacts(
        release, config_path=snapshot / "materialized.json"
    )

    for field, path in expected.items():
        assert getattr(artifacts, field) == path, (
            "manifest-free compatibility inputs remain config-relative"
        )
    assert artifacts.manifest is None, "do not invent a manifest for legacy inputs"


@pytest.mark.parametrize("fault", ["missing", "directory", "dangling-link"])
def test_missing_manifest_is_never_replaced_by_another_checkout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    engine = tmp_path / "engine"
    write_snapshot(engine)
    monkeypatch.setattr(config, "ROOT", engine)
    manifest_path = tmp_path / "current-release.json"
    if fault == "directory":
        manifest_path.mkdir()
    elif fault == "dangling-link":
        manifest_path.symlink_to(tmp_path / "missing.json")

    with pytest.raises(
        IsADirectoryError if fault == "directory" else FileNotFoundError
    ):
        config.load_release_artifacts(
            {"manifest": str(manifest_path)}, config_path=tmp_path / "materialized.json"
        )


@pytest.mark.parametrize("field", ARTIFACT_NAMES)
@pytest.mark.parametrize("fault", ["missing", "directory", "dangling-link", "hash"])
def test_invalid_snapshot_artifacts_never_fall_back_to_another_checkout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str, fault: str
) -> None:
    snapshot = tmp_path / "snapshot"
    manifest = write_snapshot(snapshot)
    engine = tmp_path / "engine"
    write_snapshot(engine)
    monkeypatch.setattr(config, "ROOT", engine)
    path = artifact_paths(snapshot)[field]
    if fault == "hash":
        path.write_bytes(b"different artifact")
    else:
        path.unlink()
        if fault == "directory":
            path.mkdir()
        elif fault == "dangling-link":
            path.symlink_to(snapshot / "missing-artifact")
    message = "hashes do not match" if fault == "hash" else "must exist"

    with pytest.raises(config.ReleaseError, match=message):
        config.load_release_artifacts(
            {"manifest": str(manifest)}, config_path=tmp_path / "materialized.json"
        )


def test_manifest_alias_is_pinned_before_reading_and_discovering_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshot = tmp_path / "snapshot"
    manifest = write_snapshot(snapshot)
    decoy = write_snapshot(tmp_path / "other")
    alias = tmp_path / "release.json"
    alias.symlink_to(manifest)
    read_text = Path.read_text

    def read_and_retarget(
        path: Path, encoding: str | None = None, errors: str | None = None
    ) -> str:
        content = read_text(path, encoding=encoding, errors=errors)
        if path.resolve() == manifest:
            alias.unlink()
            alias.symlink_to(decoy)
        return content

    monkeypatch.setattr(Path, "read_text", read_and_retarget)
    artifacts = config.load_release_artifacts(
        {"manifest": str(alias)}, config_path=tmp_path / "materialized.json"
    )

    assert alias.resolve() == decoy, "the fixture must retarget the alias during read"
    assert artifacts.wheel == artifact_paths(snapshot)["wheel"], (
        "root discovery must use the same manifest target whose bytes were read"
    )


def test_artifact_alias_returns_the_canonical_path_that_was_hashed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshot = tmp_path / "snapshot"
    manifest = write_snapshot(snapshot)
    monkeypatch.setattr(config, "ROOT", snapshot)
    wheel_alias = artifact_paths(snapshot)["wheel"]
    verified = wheel_alias.with_name("verified.whl")
    wheel_alias.rename(verified)
    wheel_alias.symlink_to(verified)
    unverified = wheel_alias.with_name("unverified.whl")
    unverified.write_bytes(b"not the verified wheel")
    open_file = Path.open
    retargeted = False

    def open_and_retarget(
        path: Path,
        mode: str = "r",
        buffering: int = -1,
        encoding: str | None = None,
        errors: str | None = None,
        newline: str | None = None,
    ) -> IO[Any]:
        nonlocal retargeted
        handle = open_file(path, mode, buffering, encoding, errors, newline)
        if not retargeted and path.resolve() == verified:
            wheel_alias.unlink()
            wheel_alias.symlink_to(unverified)
            retargeted = True
        return handle

    monkeypatch.setattr(Path, "open", open_and_retarget)
    artifacts = config.load_release_artifacts(
        {"manifest": str(manifest)}, config_path=tmp_path / "materialized.json"
    )

    assert retargeted, "the fixture must change the alias after the wheel is opened"
    assert artifacts.wheel == verified, "return the path that was actually hashed"
    expected_sha = json.loads(manifest.read_text())["components"]["control_plane"][
        "wheel_sha256"
    ]
    assert hashlib.sha256(artifacts.wheel.read_bytes()).hexdigest() == expected_sha, (
        "the returned wheel must still identify the validated bytes"
    )


def test_in_process_engine_uses_snapshot_artifacts_and_preserves_split_images(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshot = tmp_path / "snapshot"
    manifest_path = write_snapshot(snapshot)
    manifest = json.loads(manifest_path.read_text())
    config_path = config_file(tmp_path)
    value = json.loads(config_path.read_text())
    value["release"] = {"manifest": str(manifest_path), "agent_config_digest": "a" * 64}
    site = RenderedSite(
        source=tmp_path / "site.yaml",
        repository_root=snapshot,
        release_config=value,
        environment={},
        source_sha256="d" * 64,
    )
    monkeypatch.chdir(ROOT)

    release = build_release(site)

    assert config.ROOT == ROOT and config.ROOT != snapshot, (
        "the in-process engine must still be imported from the other checkout"
    )
    for field, path in artifact_paths(snapshot).items():
        assert getattr(release.config, field) == path, (
            f"materialized in-process config must bind {field} to the snapshot"
        )
    images = manifest["delivery"]["images"]
    assert release.runtime_image == images["runtime"]["reference"], (
        "the CPU keeps its independently pinned OCI image"
    )
    assert release.executor_image == images["executor"]["reference"], (
        "the Executor must not inherit the CPU image"
    )
    assert release.runtime_image != release.executor_image, (
        "the fixture must exercise independent CPU and Executor images"
    )
    assert release.config.release_delivery_identity == manifest["delivery"], (
        "constructing an engine cannot rewrite the signed delivery identity"
    )
    node_dependencies = release.config.release_delivery_identity["images"][
        "node_dependencies"
    ]
    assert node_dependency_environment(identity=node_dependencies, required=True) == {
        "GPU_FAULT_NODE_DEPENDENCY_IMAGE": images["node_dependencies"]["reference"],
        "GPU_FAULT_NODE_WHEELHOUSE_SHA256": images["node_dependencies"][
            "wheelhouse_sha256"
        ],
    }, "the offline wheelhouse remains an OCI input, never a local-path fallback"


@pytest.mark.parametrize(
    "fault,problem",
    [
        ("missing-signature", "signature or bundle is required"),
        ("invalid-signature", "cosign verification failed"),
        ("manifest-digest", "manifest SHA-256 does not match"),
    ],
)
def test_located_snapshot_does_not_bypass_attestation_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str, problem: str
) -> None:
    snapshot = tmp_path / "snapshot"
    manifest_path = write_snapshot(snapshot)
    artifacts = config.load_release_artifacts(
        {"manifest": str(manifest_path)}, config_path=tmp_path / "materialized.json"
    )
    attestation = snapshot / "attestation.json"
    attestation.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "subject": {
                    "release_id": artifacts.release_id,
                    "manifest": manifest_path.relative_to(snapshot).as_posix(),
                    "manifest_sha256": (
                        "0" * 64
                        if fault == "manifest-digest"
                        else hashlib.sha256(manifest_path.read_bytes()).hexdigest()
                    ),
                    "delivery_sha256": artifacts.delivery_sha256,
                },
                "source": {"dirty": False},
                "quality_gates": [
                    {"command": command, "status": "PASSED"}
                    for command in release_attestation.PRODUCTION_QUALITY_GATES
                ],
            }
        )
    )
    calls: list[list[str]] = []

    def reject_signature(
        command: list[str], **_kwargs: Any
    ) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        return subprocess.CompletedProcess(
            command, 1, "", "synthetic invalid signature"
        )

    monkeypatch.setattr(subprocess, "run", reject_signature)
    with pytest.raises(release_attestation.ReleaseAttestationError, match=problem):
        release_attestation.verify_attestation(
            snapshot,
            attestation,
            signature=None,
            bundle=None
            if fault == "missing-signature"
            else snapshot / "signature.json",
            cosign_key="synthetic-public-key",
            certificate=None,
            certificate_identity=None,
            certificate_oidc_issuer=None,
        )
    assert [command[:2] for command in calls] == (
        [["cosign", "verify-blob"]] if fault == "invalid-signature" else []
    ), "root discovery cannot substitute for the independent signature gate"

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping

import yaml  # type: ignore[import-untyped,unused-ignore]

from gpu_fault.admin.atomic_json import write_json_atomic
from gpu_fault.admin.config import (
    AdminConfig,
    load_desired_admin_config,
    persist_desired_admin_config,
)
from gpu_fault.admin.site import SiteConfigError

# The recorded previous release_id becomes a directory name below the state
# directory. It arrives from live release state the CLI did not author, so it is
# bounded to the same anchored shape a release_id is validated against elsewhere
# -- letters, digits and ``._-`` only -- so it can never carry a path separator
# or ``..`` segment that would escape the state directory.
RELEASE_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def _consistent_previous_value(
    previous: dict[str, Any],
    key: str,
) -> str:
    values = {
        str(item.get(key))
        for item in (previous.get("clusters") or {}).values()
        if isinstance(item, dict) and item.get(key)
    }
    return next(iter(values)) if len(values) == 1 else ""


def _release_manifest_identity(
    path: Path,
    previous: dict[str, Any],
) -> tuple[str, str, bool, bool, bool] | None:
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(value, dict):
        return None
    candidate_id = str(value.get("release_id") or "")
    delivery = value.get("delivery")
    candidate_delivery = (
        str(delivery.get("sha256") or "") if isinstance(delivery, dict) else ""
    )
    if not candidate_id:
        return None
    components = value.get("components")
    components = components if isinstance(components, dict) else {}
    control_plane = components.get("control_plane")
    control_plane = control_plane if isinstance(control_plane, dict) else {}
    executor = components.get("executor")
    executor = executor if isinstance(executor, dict) else {}
    node = components.get("node_runtime")
    node = node if isinstance(node, dict) else {}
    protocols = value.get("protocol_versions")
    protocols = protocols if isinstance(protocols, dict) else {}
    delivery_components = (
        delivery.get("components") if isinstance(delivery, dict) else {}
    )
    delivery_components = (
        delivery_components if isinstance(delivery_components, dict) else {}
    )
    node_bundle = delivery_components.get("node_bundle")
    node_bundle = node_bundle if isinstance(node_bundle, dict) else {}
    metadata = previous.get("metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    expected = {
        "control_plane_artifact": str(previous.get("cpu_wheel_sha256") or ""),
        "executor_artifact": str(
            metadata.get("required-regional-executor-artifact-sha256") or ""
        ),
        "executor_compatibility": str(
            metadata.get("required-regional-executor-compatibility-digest") or ""
        ),
        "executor_protocol": str(
            metadata.get("required-regional-executor-protocol-version") or ""
        ),
        "agent_artifact": str(metadata.get("required-agent-artifact-sha256") or ""),
        "agent_compatibility": str(
            metadata.get("required-agent-compatibility-digest") or ""
        ),
        "agent_protocol": str(metadata.get("required-agent-protocol-version") or ""),
        "bundle_sha256": _consistent_previous_value(previous, "bundle_sha256"),
        "template_sha256": _consistent_previous_value(previous, "template_sha256"),
    }
    actual = {
        "control_plane_artifact": str(control_plane.get("wheel_sha256") or ""),
        "executor_artifact": str(executor.get("wheel_sha256") or ""),
        "executor_compatibility": str(executor.get("module_digest") or ""),
        "executor_protocol": str(protocols.get("executor") or ""),
        "agent_artifact": str(node.get("wheel_sha256") or ""),
        "agent_compatibility": str(node.get("module_digest") or ""),
        "agent_protocol": str(protocols.get("agent") or ""),
        "bundle_sha256": str(value.get("bundle_sha256") or ""),
        "template_sha256": str(
            node_bundle.get("template_sha256")
            or (delivery.get("node_template_inputs") or {}).get("sha256")
            if isinstance(delivery, dict)
            else ""
        ),
    }
    evidence = {
        key: expected_value
        for key, expected_value in expected.items()
        if expected_value
    }
    physical_identity = bool(evidence) and all(
        actual.get(key) == expected_value for key, expected_value in evidence.items()
    )
    previous_id = str(previous.get("release_id") or "").strip()
    release_id_match = bool(previous_id) and candidate_id == previous_id
    delivery_match = candidate_delivery == str(
        previous.get("release_delivery_sha256") or ""
    )
    semantic = dict(value)
    semantic.pop("created_at", None)
    return (
        candidate_id,
        hashlib.sha256(
            json.dumps(semantic, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        physical_identity,
        release_id_match,
        delivery_match,
    )


def find_previous_release_manifest(
    state_dir: Path,
    previous: dict[str, Any],
) -> Path:
    delivery_sha256 = str(previous.get("release_delivery_sha256") or "")
    if len(delivery_sha256) != 64:
        raise SiteConfigError("rolled-back release has no previous delivery identity")
    physical_matches: list[tuple[Path, str, str]] = []
    release_id_matches: list[tuple[Path, str, str]] = []
    delivery_matches: list[tuple[Path, str, str]] = []
    snapshots = state_dir.expanduser().resolve() / "source-snapshots"
    for path in snapshots.glob("*/repository-*/dist/*/release.json"):
        identity = _release_manifest_identity(path, previous)
        if identity is None:
            continue
        (
            candidate_id,
            digest,
            physical_identity,
            release_id_match,
            delivery_match,
        ) = identity
        if physical_identity:
            physical_matches.append((path, candidate_id, digest))
        if release_id_match and delivery_match:
            release_id_matches.append((path, candidate_id, digest))
        if delivery_match:
            delivery_matches.append((path, candidate_id, digest))
    matches = physical_matches or release_id_matches or delivery_matches
    if not matches:
        raise SiteConfigError(
            "cannot locate the immutable previous release after rollback"
        )
    identities = {(candidate_id, digest) for _path, candidate_id, digest in matches}
    if len(identities) != 1:
        # Physical identity proves a manifest builds the artifacts that are
        # live, which is why it outranks a release_id the state may have
        # recorded stale. It cannot single one out though: releases that
        # differ only in configuration build byte-identical artifacts, so
        # every one of them matches. When that happens, narrow the tier with
        # the evidence that does name a release instead of refusing -- the
        # refusal would strand the rollback with the answer already on disk.
        narrowed = [item for item in matches if item in release_id_matches] or [
            item for item in matches if item in delivery_matches
        ]
        if narrowed:
            matches = narrowed
            identities = {
                (candidate_id, digest) for _path, candidate_id, digest in matches
            }
    if len(identities) != 1:
        raise SiteConfigError(
            "previous release evidence matches multiple manifest identities"
        )
    return max(matches, key=lambda item: item[0].stat().st_mtime)[0]


def rollback_management_document(
    site_file: Path,
    live_state: dict[str, Any],
    *,
    site_before: Path | None = None,
    management_manifest: Path | None = None,
    management_repository_root: Path | None = None,
) -> tuple[dict[str, Any], AdminConfig, Path]:
    if str(live_state.get("phase") or "") != "rolled-back":
        raise SiteConfigError("live release is not in rolled-back state")
    rollback_result = live_state.get("rollback_result")
    if (
        not isinstance(rollback_result, dict)
        or rollback_result.get("status") != "PASSED"
    ):
        raise SiteConfigError("live rollback has no PASSED verification result")
    previous = live_state.get("previous")
    if not isinstance(previous, dict):
        raise SiteConfigError("rolled-back release has no previous baseline")
    source = (
        site_before if site_before is not None and site_before.is_file() else site_file
    )
    try:
        document = yaml.safe_load(source.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise SiteConfigError("cannot read the rollback management site") from exc
    if not isinstance(document, dict) or not isinstance(document.get("spec"), dict):
        raise SiteConfigError("rollback management site is invalid")
    source_manifest = find_previous_release_manifest(site_file.parent, previous)
    repository_root = management_repository_root or source_manifest.parents[2]
    manifest = (
        _materialize_management_manifest(source_manifest, management_manifest)
        if management_manifest is not None
        else source_manifest
    )
    manifest_value = json.loads(source_manifest.read_text(encoding="utf-8"))
    spec = document["spec"]
    release = spec.setdefault("release", {})
    profile = spec.setdefault("runtimeProfile", {})
    images = spec.setdefault("images", {})
    metadata = previous.get("metadata") or {}
    previous_profile = str(previous.get("runtime_profile_version") or "")
    profile_snapshot = site_file.parent / "profiles" / f"{previous_profile}.yaml"
    if not previous_profile or not profile_snapshot.is_file():
        raise SiteConfigError("rolled-back Runtime Profile snapshot is unavailable")
    spec["repositoryRoot"] = str(repository_root)
    release["manifest"] = (
        str(manifest.relative_to(repository_root))
        if manifest.is_relative_to(repository_root)
        else str(manifest)
    )
    release["agentConfigDigest"] = str(
        metadata.get("required-agent-config-digest") or ""
    )
    profile["source"] = str(profile_snapshot)
    profile["version"] = previous_profile
    delivery = manifest_value.get("delivery")
    raw_images = delivery.get("images") if isinstance(delivery, dict) else None
    locked_images: dict[str, Any] = (
        dict(raw_images) if isinstance(raw_images, dict) else {}
    )

    def image_reference(name: str) -> object:
        image = locked_images.get(name)
        return image.get("reference") if isinstance(image, dict) else None

    for key, value in (
        ("runtime", image_reference("runtime")),
        ("nodeInstaller", image_reference("node_installer")),
        ("dcgmExporter", image_reference("dcgm_exporter")),
        ("adot", image_reference("adot")),
    ):
        if value:
            images[key] = str(value)
    admin_config = _restored_admin_config(
        site_file.parent,
        previous.get("admin_config"),
    )
    return document, admin_config, repository_root


def _overlay(base: Mapping[str, Any], patch: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in patch.items():
        current = merged.get(key)
        merged[key] = (
            _overlay(current, value)
            if isinstance(value, Mapping) and isinstance(current, Mapping)
            else value
        )
    return merged


def _restored_admin_config(state_dir: Path, snapshot: object) -> AdminConfig:
    """Restore the recorded admin config without inventing the rest of it.

    A snapshot only carries the fields the release that wrote it knew about, so
    reading it straight through ``AdminConfig.from_mapping`` turns every later
    addition into a default -- and then persists that default as though an
    administrator had chosen it. ``aurora`` is how this surfaced: a snapshot
    written before the block existed restored 0.5/8 ACU over a live 8/32 ACU
    cluster, the next deployment dutifully reconciled the cluster downwards, and
    nothing in the record distinguished the fabricated value from an authored
    one. Overlay the snapshot onto what the site currently desires instead, so a
    rollback can only move the fields it actually recorded.
    """

    recorded = dict(snapshot) if isinstance(snapshot, Mapping) else {}
    # A release never moves Aurora capacity, so a rollback has nothing to restore
    # there: the ``config`` command settles the window before its release starts
    # and restores its own ``before`` when that release fails. A snapshot's
    # ``aurora`` is at best a copy of what the site desired at the time and at
    # worst a parser default (live 2026-09-13: 0.5/8 restored over 82/128, and
    # the next deploy scaled the production database down to it).
    recorded.pop("aurora", None)
    current = load_desired_admin_config(state_dir).as_dict()
    return AdminConfig.from_mapping(_overlay(current, recorded))


def _write_yaml_atomic(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    descriptor, raw = tempfile.mkstemp(
        prefix=path.name + ".",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(raw)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            yaml.safe_dump(document, handle, sort_keys=False)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.chmod(0o600)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _materialize_management_manifest(
    source: Path,
    destination: Path,
) -> Path:
    raw = source.read_bytes()
    value = json.loads(raw)
    source_root = source.parents[2]

    def absolute(raw_path: object) -> str:
        path = Path(str(raw_path))
        return str(path if path.is_absolute() else (source_root / path).resolve())

    if not value.get("wheel") or not value.get("bundle"):
        raise SiteConfigError("previous release manifest has no wheel or bundle path")
    value["wheel"] = absolute(value["wheel"])
    value["bundle"] = absolute(value["bundle"])
    components = value.get("components")
    if isinstance(components, dict):
        for component in components.values():
            if isinstance(component, dict) and component.get("wheel"):
                component["wheel"] = absolute(component["wheel"])
    value["management_baseline"] = {
        "source_manifest": str(source),
        "source_manifest_sha256": hashlib.sha256(raw).hexdigest(),
        "purpose": "rollback-status-only",
    }
    write_json_atomic(destination, value)
    return destination


def reconcile_rollback_management(
    site_file: Path,
    live_state: dict[str, Any],
    *,
    site_before: Path | None = None,
    source: str,
) -> Path:
    current = yaml.safe_load(site_file.read_text(encoding="utf-8"))
    current_root = Path(str(current["spec"]["repositoryRoot"])).resolve()
    release_id = str(
        ((live_state.get("previous") or {}).get("release_id") or "previous")
    )
    if not RELEASE_ID_PATTERN.fullmatch(release_id):
        raise SiteConfigError(
            f"rolled-back previous release_id is malformed: {release_id!r}"
        )
    management_manifest = (
        site_file.parent / "rollback-management" / release_id / "release.status.json"
    )
    document, admin_config, repository_root = rollback_management_document(
        site_file,
        live_state,
        site_before=site_before,
        management_manifest=management_manifest,
        management_repository_root=current_root,
    )
    _write_yaml_atomic(site_file, document)
    persist_desired_admin_config(
        site_file.parent,
        config=admin_config,
        source=source,
    )
    return repository_root


@contextmanager
def materialized_rollback_status_site(
    site_file: Path,
    live_state: dict[str, Any],
    *,
    management_repository_root: Path | None = None,
) -> Iterator[Path]:
    with tempfile.TemporaryDirectory(prefix="gpu-fault-rollback-status-") as raw:
        directory = Path(raw)
        current_document = yaml.safe_load(site_file.read_text(encoding="utf-8"))
        repository_root = (
            management_repository_root.expanduser().resolve()
            if management_repository_root is not None
            else Path(str(current_document["spec"]["repositoryRoot"])).resolve()
        )
        document, admin_config, _repository_root = rollback_management_document(
            site_file,
            live_state,
            management_manifest=directory / "release.status.json",
            management_repository_root=repository_root,
        )
        temporary_site = directory / "site.yaml"
        _write_yaml_atomic(temporary_site, document)
        persist_desired_admin_config(
            directory,
            config=admin_config,
            source="rolled-back-status-baseline",
        )
        yield temporary_site

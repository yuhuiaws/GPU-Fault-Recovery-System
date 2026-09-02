from __future__ import annotations

import os
from pathlib import Path
import shutil
import tempfile

if __package__:
    from scripts.component_artifacts import (
        ComponentArtifactError,
        component_build_identity,
        load_component_artifacts,
    )
else:
    from component_artifacts import (
        ComponentArtifactError,
        component_build_identity,
        load_component_artifacts,
    )


def _cache_entry(root: Path, cache_root: Path) -> Path:
    return cache_root.resolve() / component_build_identity(root)


def find_cached_manifest(root: Path, cache_root: Path) -> Path | None:
    entry = _cache_entry(root, cache_root)
    manifest = entry / "dist/current-release.json"
    if not manifest.is_file():
        return None
    try:
        load_component_artifacts(
            root,
            manifest,
            artifact_root=entry,
            require_delivery_identity=False,
        )
    except ComponentArtifactError:
        return None
    return manifest


def store_component_artifacts(
    root: Path,
    cache_root: Path,
    manifest_path: Path,
) -> Path:
    root = root.resolve()
    artifacts = load_component_artifacts(root, manifest_path)
    cache_root = cache_root.resolve()
    cache_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    cache_root.chmod(0o700)
    target = _cache_entry(root, cache_root)
    if find_cached_manifest(root, cache_root) is not None:
        return target / "dist/current-release.json"

    temporary = Path(tempfile.mkdtemp(prefix=".component-", dir=cache_root))
    try:
        dist = temporary / "dist"
        dist.mkdir()
        release_id = str(artifacts.manifest["release_id"])
        shutil.copytree(root / "dist" / release_id, dist / release_id)
        shutil.copy2(manifest_path, dist / "current-release.json")
        load_component_artifacts(
            root,
            dist / "current-release.json",
            artifact_root=temporary,
            require_delivery_identity=False,
        )
        if target.exists():
            shutil.rmtree(target)
        os.replace(temporary, target)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return target / "dist/current-release.json"

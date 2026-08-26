"""Resolve and verify one artifact from current-release.json."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "src/gpu_fault"
ARTIFACT_KEYS = {
    "wheel": ("control_plane", "wheel"),
    "control-plane-wheel": ("control_plane", "wheel"),
    "executor-wheel": ("executor", "wheel"),
    "node-runtime-wheel": ("node_runtime", "wheel"),
    "bundle": ("node_bundle", "bundle"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_artifact(manifest_path: Path, key: str) -> Path:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    component_name, component_key = ARTIFACT_KEYS[key]
    component = (manifest.get("components") or {}).get(component_name) or {}
    value = Path(str(component.get(component_key) or manifest[component_key]))
    path = value if value.is_absolute() else ROOT / value
    if not path.is_file():
        raise RuntimeError(f"release artifact is missing: {path}")
    expected = str(
        component.get(f"{component_key}_sha256") or manifest[f"{component_key}_sha256"]
    )
    if sha256(path) != expected:
        raise RuntimeError(f"release artifact hash does not match manifest: {path}")
    return path.resolve()


def checkout_module_digest(component: str = "control_plane") -> str:
    """Digest src/gpu_fault with the package's own function, not a copy of it.

    gpu_fault imports with a bare interpreter and no third-party package, so
    this works for the documented ``python3 scripts/release-artifact-path.py``
    invocations too.
    """
    if str(ROOT / "src") not in sys.path:
        sys.path.insert(0, str(ROOT / "src"))
    scripts = ROOT / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    from component_wheels import component_source_digest

    return component_source_digest(component)


def verify_module_digest(
    manifest_path: Path,
    key: str = "wheel",
) -> str:
    """Refuse to hand out an artifact built from different code.

    The path/sha256 pair only proves the file named in the manifest is intact.
    It says nothing about whether that file was built from the checkout the
    operator is reading: the manifest carries ``module_digest`` for exactly
    that comparison (see scripts/build-release-artifacts.py) and nothing was
    checking it, so every documented ``WHEEL="$(release-artifact-path.py
    wheel)"`` step would happily install a wheel from an older tree.
    """
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    component_name, _component_key = ARTIFACT_KEYS[key]
    component = (manifest.get("components") or {}).get(component_name) or {}
    expected = component.get("module_digest") or manifest.get("module_digest")
    if component_name == "node_bundle":
        component_name = "node_runtime"
        component = (manifest.get("components") or {}).get(component_name) or {}
        expected = component.get("module_digest") or expected
    if not expected:
        raise RuntimeError(
            f"release manifest has no module_digest: {manifest_path}; "
            "rebuild with python3 scripts/build-release-artifacts.py"
        )
    if not PACKAGE.is_dir():
        raise RuntimeError(
            f"cannot verify module_digest without a checkout at {PACKAGE}; "
            "pass --skip-module-digest to resolve a path anyway"
        )
    actual = checkout_module_digest(component_name)
    if actual != str(expected):
        raise RuntimeError(
            "release was built from different code: manifest module_digest "
            f"{expected} != checkout {actual}; rebuild with "
            "python3 scripts/build-release-artifacts.py"
        )
    return actual


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("key", choices=tuple(ARTIFACT_KEYS))
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "dist/current-release.json",
    )
    parser.add_argument(
        "--skip-module-digest",
        action="store_true",
        help="resolve without proving the release matches this checkout",
    )
    args = parser.parse_args()
    # 手册里这条命令都写在 WHEEL="$(...)" 里，操作员看到的是 stderr，
    # 不是 traceback；stdout 必须干净，否则会被当成路径代入下一步。
    try:
        if not args.skip_module_digest:
            verify_module_digest(args.manifest, args.key)
        resolved = resolve_artifact(args.manifest, args.key)
    except (RuntimeError, KeyError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(resolved)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

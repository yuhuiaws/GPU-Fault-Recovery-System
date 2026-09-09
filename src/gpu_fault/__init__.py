"""Portable GPU fault control-plane core."""

from __future__ import annotations

import hashlib
from pathlib import Path

__version__ = "0.10.0"

# Files that decide behaviour. Anything else in the package directory
# (compiled caches, editor droppings, a stray .whl someone unzipped) is
# excluded so the digest is reproducible between a source checkout, an
# installed site-packages copy and the contents of a wheel.
_DIGESTED_SUFFIXES = frozenset({".py", ".yaml", ".yml", ".json"})
# Local packages a distribution may install next to ``gpu_fault``; see module_digest.
_SIBLING_PACKAGES = ("gpu_fault_release",)

_MODULE_DIGEST: str | None = None


def module_digest(package_dir: str | Path | None = None) -> str:
    """SHA-256 over the package's own behaviour-bearing files.

    A wheel SHA-256 only proves which *file* a deploy referenced, and
    manifest annotations record what a deploy *claimed* -- both control
    plane Deployments carried
    ``gpu-fault.io/artifact-sha256=8baa9d7c...`` while the ConfigMap they
    actually mounted held wheel ``60c7deb6b7bc...``. This digest is
    computed from the code the process really imported, so /v1/version
    can be compared against a checkout, a wheel or a node bundle without
    trusting any label.

    ``package_dir`` exists for callers that want to digest an unpacked
    wheel or a different checkout; the default is the running package.
    """

    global _MODULE_DIGEST
    if package_dir is None and _MODULE_DIGEST is not None:
        return _MODULE_DIGEST

    root = (
        Path(package_dir)
        if package_dir is not None
        else Path(__file__).resolve().parent
    )
    # A distribution that also ships a sibling local package (the deploy-host
    # wheel carries ``gpu_fault_release`` so the admin CLI imports the release
    # engine at its own version) is identified by both: the sibling's files are
    # keyed under its package name and everything is hashed in one global key
    # order -- exactly what ``scripts/component_wheels.package_digest`` does
    # for the staged wheel, so the installed copy and the bundle manifest agree.
    # A copy without the sibling (control plane, executor, node runtime) is
    # unchanged.
    entries: dict[str, Path] = {}
    staged = [(root, "")]
    staged.extend(
        (root.parent / name, f"{name}/")
        for name in _SIBLING_PACKAGES
        if (root.parent / name).is_dir()
    )
    for base, prefix in staged:
        for path in base.rglob("*"):
            if path.is_dir() or "__pycache__" in path.parts:
                continue
            if path.suffix not in _DIGESTED_SUFFIXES:
                continue
            entries[prefix + path.relative_to(base).as_posix()] = path
    digest = hashlib.sha256()
    for relative in sorted(entries):
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(hashlib.sha256(entries[relative].read_bytes()).digest())
        digest.update(b"\n")
    value = digest.hexdigest()
    if package_dir is None:
        _MODULE_DIGEST = value
    return value

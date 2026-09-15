"""Regional release orchestrator: the deploy host's rollout, rollback and checks.

The modules here used to live under ``deploy/control-plane/regional/`` as
scripts importing each other by bare name, which made them reachable only with
that directory on ``sys.path``. They are a package now, imported as
``gpu_fault_release.<module>``; the shell launcher
``deploy/control-plane/regional/rollout-regional-release.sh`` runs
``python3 -m gpu_fault_release.rollout`` with ``src`` on ``PYTHONPATH`` so the
administrator CLI's path-based contract is unchanged.

Non-Python inputs stay under ``deploy/control-plane/regional/``: the rendered
manifests in ``generated/``, the patch and prerequisite YAML, the cleanup
inventory, and the probe programs in ``probes/`` that the engine ships to Pods
as source text. Every module here locates them from the repository root
(``repository_root()``): the checkout that contains this package when it runs
from source, else -- for the copy the deploy-host wheel carries so the admin
CLI's in-process imports resolve at the same version -- the bound site's
``spec.repositoryRoot`` or ``GPU_FAULT_REPOSITORY_ROOT``.
"""

from __future__ import annotations

import json
import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Mapping

REPOSITORY_ROOT_ENV = "GPU_FAULT_REPOSITORY_ROOT"
STATE_DIR_BINDING = "gpu-fault-managed-state-dir.json"
_ANCHORS = ("deploy/control-plane/regional", "scripts")


def _is_repository(path: Path) -> bool:
    return all((path / anchor).is_dir() for anchor in _ANCHORS)


def containing_repository_root(path: Path) -> Path | None:
    """The checkout or source snapshot ``path`` lies in, or None.

    A release manifest names its wheels and bundle relative to the root that
    built them (``dist/<release-id>/...`` under the snapshot the site was
    deployed from). A reader running from another checkout -- the admin CLI
    joining a cluster in-process from the operator's tree -- must resolve them
    against the manifest's root, not its own.
    """

    for candidate in (path, *path.parents):
        if _is_repository(candidate):
            return candidate.resolve()
    return None


def _bound_repository_root(prefix: Path) -> Path | None:
    """The bound site's ``spec.repositoryRoot`` for an installed copy.

    The deploy-host install binds its venv to one managed state directory
    (``<prefix>/gpu-fault-managed-state-dir.json``); that directory's
    ``site.yaml`` names the release snapshot the site runs. Read with the
    standard library only: this runs at import time, before anything else.
    """

    binding = prefix / STATE_DIR_BINDING
    if not binding.is_file():
        return None
    try:
        state_dir = Path(
            str(json.loads(binding.read_text(encoding="utf-8"))["state_dir"])
        )
    except (OSError, ValueError, KeyError, TypeError):
        return None
    site = state_dir / "site.yaml"
    if not site.is_file():
        return None
    for line in site.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("repositoryRoot:"):
            value = stripped.split(":", 1)[1].strip().strip("'\"")
            return Path(value) if value else None
    return None


def resolve_repository_root(
    package_file: Path,
    *,
    environ: Mapping[str, str] | None = None,
    prefix: Path | None = None,
) -> Path:
    """Where ``deploy/`` and ``scripts/`` live for this copy of the engine.

    1. the checkout containing the package (running from source);
    2. ``GPU_FAULT_REPOSITORY_ROOT`` (the installer's smoke test, a driver);
    3. the venv's state-dir binding -> ``site.yaml`` ``spec.repositoryRoot``.
    """

    environment = os.environ if environ is None else environ
    checkout = package_file.resolve().parents[2]
    if _is_repository(checkout):
        return checkout
    declared = environment.get(REPOSITORY_ROOT_ENV, "").strip()
    if declared:
        candidate = Path(declared).expanduser()
        if _is_repository(candidate):
            return candidate.resolve()
    bound = _bound_repository_root(Path(sys.prefix) if prefix is None else prefix)
    if bound is not None and _is_repository(bound):
        return bound.resolve()
    raise RuntimeError(
        "gpu_fault_release cannot locate its repository root: "
        f"{checkout} is not a checkout, {REPOSITORY_ROOT_ENV} is "
        f"{declared!r}, and the venv binding {STATE_DIR_BINDING} resolved to "
        f"{bound}"
    )


@lru_cache(maxsize=1)
def repository_root() -> Path:
    return resolve_repository_root(Path(__file__))

"""Read-only Deployment observations for the BOOT-020 runner.

Split out of ``run_boot020_release_rolling.py`` (architecture file-length
ratchet, 2026-09-20). Everything here reads; nothing mutates the site.
"""

from __future__ import annotations

import hashlib
import json
from contextlib import AbstractContextManager, nullcontext
from typing import Any, cast


class AcceptanceCheckError(RuntimeError):
    """A live observation contradicted the BOOT-020 contract.

    Raised explicitly rather than through ``assert``: ``python -O`` strips
    ``assert`` statements, and a runner whose checks vanish under an
    optimisation flag would report a constant PASS.
    """


def release_read_snapshot(release: Any) -> AbstractContextManager[Any]:
    """The engine's read snapshot when the release offers one, else a no-op.

    Inside it every read-only ``kubectl get`` is served once from one
    observation, and a nested ``read_snapshot`` -- ``_capture_previous`` opens
    its own -- joins the outer one instead of discarding its warm cache.
    """

    factory = getattr(release, "_read_snapshot", None)
    return (
        cast(AbstractContextManager[Any], factory())
        if callable(factory)
        else nullcontext()
    )


def _deployment_listing(release: Any, args: list[str]) -> dict[str, Any]:
    """ONE ``get deployment`` list read for a kube context (see the callers)."""
    listing = release._get_json(
        args + ["-n", release.config.namespace, "get", "deployment"]
    )
    if not isinstance(listing, dict) or not isinstance(listing.get("items"), list):
        raise AcceptanceCheckError("Deployment inventory is missing")
    return listing


def deployment_identities_from(
    listing: dict[str, Any], names: tuple[str, ...]
) -> dict[str, dict[str, Any]]:
    """What a rollout of each named Deployment actually changes.

    ``replicas`` and the Pod template digest, read from the same listing as
    ``deployment_generations_from`` so a snapshot costs one list read per
    context. ``metadata.generation`` is deliberately absent: Kubernetes bumps it
    on annotation changes as well, and every release re-stamps the admin-config
    digest annotation on all CPU Deployments (live 2026-09-20).
    """
    found: dict[str, dict[str, Any]] = {}
    for item in listing["items"]:
        if not isinstance(item, dict):
            continue
        metadata = item.get("metadata", {}) or {}
        name = metadata.get("name")
        if name in names:
            spec = item.get("spec") or {}
            replicas, template = spec.get("replicas"), spec.get("template")
            if (
                name in found
                or type(replicas) is not int
                or replicas < 0
                or not isinstance(template, dict)
            ):
                raise AcceptanceCheckError("Deployment identity is missing or invalid")
            found[str(name)] = {
                "replicas": replicas,
                "template_sha256": hashlib.sha256(
                    json.dumps(template, sort_keys=True, separators=(",", ":")).encode()
                ).hexdigest(),
            }
    if set(found) != set(names):
        raise AcceptanceCheckError("Deployment inventory is incomplete")
    return {name: found[name] for name in names}


def deployment_generations(
    release: Any,
    args: list[str],
    names: tuple[str, ...],
) -> dict[str, int]:
    """``metadata.generation`` of each named Deployment from ONE list read.

    ``args`` is the kube-context prefix (``release._cpu()`` or
    ``release._gpu(target)``); the argv is built exactly as the engine's
    ``prime_deployment_snapshot`` builds it, so inside a read snapshot the list
    ``_capture_previous`` already primed answers this without another kubectl
    call. Missing, repeated or malformed generations cannot prove stability.
    """
    return deployment_generations_from(_deployment_listing(release, args), names)


def deployment_generations_from(
    listing: dict[str, Any], names: tuple[str, ...]
) -> dict[str, int]:
    found: dict[str, int] = {}
    for item in listing["items"]:
        if not isinstance(item, dict):
            continue
        metadata = item.get("metadata", {}) or {}
        name = metadata.get("name")
        if name in names:
            generation = metadata.get("generation")
            if name in found or type(generation) is not int or generation < 1:
                raise AcceptanceCheckError(
                    "Deployment generation is missing or invalid"
                )
            found[str(name)] = generation
    if set(found) != set(names):
        raise AcceptanceCheckError("Deployment inventory is incomplete")
    return {name: found[name] for name in names}

"""How BOOT-020 compares two observations of the live release.

Every stage proves what it did by comparing ``live`` -- a fresh
``RegionalRelease._capture_previous()`` -- before and after it. The capture
also carries ``cpu_restart_baseline``, what a rollback proves an untouched CPU
tier against: the digest of each role Deployment's spec and ``captured_at``,
the time the capture began. ``captured_at`` moves with every capture, so two
captures of an unchanged release are never equal (live: the NOOP stage failed
on that timestamp alone). The comparison leaves out exactly that field;
everything else, every ``spec_sha256`` included, still counts. The product
keeps the field: the rollback measures Pod start times against it.

A rollback that restarts a CPU tier gives its Pod template a new
``kubectl.kubernetes.io/restartedAt``, which moves that Deployment's spec
digest although the rollback restored it. The rollback names those tiers
itself (``cpu_restart.restarted`` in the details of each of its CPU stages in
``rollback_timing``, fallback restarts included); only their digests leave the
rollback comparison, so a tier the rollback kept must keep its digest.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from gpu_fault_release.regional_release_rollback_restart import (
    BASELINE_KEY,
    ROLE_DEPLOYMENTS,
)
from scripts.e2e.regional.boot020_deployment_reads import AcceptanceCheckError

#: When the capture began: different in every capture by construction.
CAPTURE_TIME_FIELD = "captured_at"
SPEC_DIGESTS_FIELD = "spec_sha256"


def comparable_capture(capture: object, *, restarted: Iterable[str] = ()) -> object:
    """``capture`` without its capture time and the ``restarted`` spec digests.

    Returns new mappings wherever something is left out and never changes
    ``capture``. A value that is no capture (``None`` from a ``live=False``
    snapshot) compares as it is.
    """

    if not isinstance(capture, Mapping):
        return capture
    baseline = capture.get(BASELINE_KEY)
    if not isinstance(baseline, Mapping):
        return dict(capture)
    kept = {key: value for key, value in baseline.items() if key != CAPTURE_TIME_FIELD}
    skipped = frozenset(restarted)
    specs = kept.get(SPEC_DIGESTS_FIELD)
    if skipped and isinstance(specs, Mapping):
        kept[SPEC_DIGESTS_FIELD] = {
            name: digest for name, digest in specs.items() if name not in skipped
        }
    return {**capture, BASELINE_KEY: kept}


def same_capture(left: object, right: object, *, restarted: Iterable[str] = ()) -> bool:
    """Whether two captures describe the same live release."""

    skipped = frozenset(restarted)
    return comparable_capture(left, restarted=skipped) == comparable_capture(
        right, restarted=skipped
    )


def same_live(before: Mapping[str, Any], after: Mapping[str, Any]) -> bool:
    """Whether two BOOT-020 snapshots observed the same live release."""

    return same_capture(before["live"], after["live"])


def rollback_restarted_deployments(rollback: Mapping[str, Any]) -> frozenset[str]:
    """The CPU role Deployments a rollback deploy result says it restarted.

    The union over every rollback phase that reports ``cpu_restart`` (the
    controller stage and the CPU restore); a result without that evidence
    names none, so every digest is compared.
    """

    timing = rollback.get("rollback_timing")
    phases = timing.get("phases") if isinstance(timing, Mapping) else None
    names: set[str] = set()
    for entry in phases.values() if isinstance(phases, Mapping) else ():
        details = entry.get("details") if isinstance(entry, Mapping) else None
        restart = details.get("cpu_restart") if isinstance(details, Mapping) else None
        roles = restart.get("restarted") if isinstance(restart, Mapping) else None
        if isinstance(roles, Mapping):
            names.update(
                ROLE_DEPLOYMENTS[role] for role in roles if role in ROLE_DEPLOYMENTS
            )
    return frozenset(names)


def assert_restored(
    before: Mapping[str, Any],
    rolled_back: Mapping[str, Any],
    rollback: Mapping[str, Any],
) -> None:
    """A rollback put the live release back to what ``before`` observed."""

    restarted = rollback_restarted_deployments(rollback)
    if not same_capture(before["live"], rolled_back["live"], restarted=restarted):
        raise AcceptanceCheckError(
            f"rollback did not restore live state: {(before, rolled_back)!r}"
        )

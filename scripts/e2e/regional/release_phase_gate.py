"""Evidence binds only to a committed release.

``gpu-fault-regional-release-state`` records the transaction's *candidate*,
not the release the site runs: every checkpoint stamps the candidate's
``release_id``. Read at any phase other than a committed ``complete`` it names
a release the site does not run, or does not run everywhere, or may still
leave:

* ``rolled-back``: the rollback finished; the state still names the candidate
  that failed while every Pod runs the previous release (live after
  BOOT-020: the next case bound its evidence to the failed candidate);
* a ``rollback-*`` phase: a rollback is in progress or stopped;
* a forward phase (``preflight`` .. ``verified``, ``data-plane-paused``) or
  ``failed``/``partial-convergence``: a transaction is in flight or stopped
  part way, and the site runs a mix of the candidate and its previous release;
* a ``bootstrap-*`` phase: the site's first install has not finished;
* ``complete`` with ``transaction_committed`` not true: the candidate rolled
  out, but the deploy's verify/stability gate has not passed (or the deploy
  is still running), so the release can still roll back. This is refused
  too, as every case that already demanded a completed transaction did.

A caller with a legitimate non-committed phase names it in
``allowed_phases``; the only one is the BOOT store-lifecycle witness's
``verify --mode fail-forward``, whose subject is a transaction that stopped
in ``failed``/``partial-convergence``. ``complete`` itself can never be
allowed uncommitted.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping
from typing import Any

from scripts.e2e.regional.regional_commands import RegionalFixtureError

COMMITTED_PHASE = "complete"
FAIL_FORWARD_STOPPED_PHASES = frozenset({"failed", "partial-convergence"})
REMEDY = "run `gpu-fault-admin deploy` of the site's own release to reach one"
NOT_AN_OBJECT = "regional release state is not an object"


class ReleasePhaseError(RegionalFixtureError):
    """The release state names no committed release; says which and why."""

    def __init__(self, message: str, *, phase: str, release_id: str) -> None:
        super().__init__(message)
        self.phase = phase
        self.release_id = release_id


def _text(state: Mapping[str, Any], key: str) -> str:
    value = state.get(key)
    return value if isinstance(value, str) else ""


def _refusal(phase: str, state: Mapping[str, Any]) -> tuple[str, str]:
    """Why ``phase`` binds no evidence, and what the remedy adds for it."""

    if not phase:
        return "no phase is recorded", ""
    if phase == COMMITTED_PHASE:
        return (
            "the release rolled out but its transaction is not committed "
            f"(transaction_committed={state.get('transaction_committed')!r}): "
            "the deploy's verify/stability gate has not passed or the deploy "
            "is still running, so the release can still roll back",
            "; a deploy still running commits it when that gate passes",
        )
    if phase == "rolled-back":
        extra = ""
        if state.get("rollback_cleanup_completed") is False:
            extra = (
                "; the rollback cleanup is still pending: the first run "
                "finishes it, a second one deploys the release"
            )
        return (
            "a rollback finished: the recorded release_id is the candidate "
            "that failed and the site runs its previous release",
            extra,
        )
    if phase.startswith("rollback-") or phase == "rolling-back":
        return (
            "a rollback is active (in progress or stopped) and the site runs "
            "a mix of the candidate and its previous release",
            "; the first run finishes the rollback, a second one deploys the release",
        )
    if phase.startswith("bootstrap-"):
        return "the site's bootstrap has not finished", ""
    if phase in FAIL_FORWARD_STOPPED_PHASES:
        return (
            "a fail-forward transaction stopped before completing and the "
            "site runs a mix of the candidate and its previous release",
            "; a stopped transaction of another release needs "
            "--supersede-failed-transaction",
        )
    if phase == "data-plane-paused":
        return "the data-plane rollout paused on a failed cluster", ""
    return (
        "a release transaction is in flight or stopped before completing",
        "; a deploy still running finishes it",
    )


def release_phase_error(
    state: object, *, allowed_phases: Collection[str] = ()
) -> str | None:
    """Why ``state`` cannot bind evidence, or ``None`` when it can.

    The message names the phase, the recorded release id and the remedy, so a
    caller with its own error type raises it with this text unchanged.
    """

    if isinstance(allowed_phases, str):
        raise TypeError("allowed_phases is a collection of phase names")
    if COMMITTED_PHASE in allowed_phases:
        raise ValueError(
            f"{COMMITTED_PHASE!r} binds evidence only once committed; it "
            "cannot be allowed uncommitted"
        )
    if not isinstance(state, Mapping):
        return NOT_AN_OBJECT
    phase = _text(state, "phase")
    release_id = _text(state, "release_id")
    committed = phase == COMMITTED_PHASE and state.get("transaction_committed") is True
    if committed or (phase and phase in allowed_phases):
        if release_id:
            return None
        reason, extra = "the state records no release id", ""
    else:
        reason, extra = _refusal(phase, state)
    phase_label = repr(phase) if phase else "none"
    release_label = repr(release_id) if release_id else "none"
    return (
        f"regional release state (phase {phase_label}, recorded release_id "
        f"{release_label}) binds no evidence: {reason}. Evidence binds only to "
        f"a committed {COMMITTED_PHASE!r} release: {REMEDY}{extra}"
    )


def committed_release_id(state: object, *, allowed_phases: Collection[str] = ()) -> str:
    """The release id evidence binds to, refusing any other phase."""

    if not isinstance(state, Mapping):
        raise RegionalFixtureError(NOT_AN_OBJECT)
    message = release_phase_error(state, allowed_phases=allowed_phases)
    if message is not None:
        raise ReleasePhaseError(
            message,
            phase=_text(state, "phase"),
            release_id=_text(state, "release_id"),
        )
    return _text(state, "release_id")

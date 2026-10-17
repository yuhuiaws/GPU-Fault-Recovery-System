"""One bounded start of a host DCGM engine that is down, before a quick
diagnostic is graded.

The DCGM 4 package unit ``nvidia-dcgm`` (nv-hostengine, ``Type=simple``) is
``Restart=on-abort`` with at most 5 starts in 10 s on both the plain EKS
AL2023 NVIDIA AMI and the HyperPod AMI: systemd restarts it only after an
unclean signal, never after a non-zero exit, and gives up after a crash loop.
A host-mode node (an ``eks-managed`` node group declaring
``gpu-fault.io/dcgm-hostengine=host``, every HyperPod node) whose engine went
down that way stays down, and its next ``dcgmi diag`` -- which then reaches no
engine -- is graded FAIL with ``DCGM_EXECUTION_REVIEW`` and drains a node a
start would have brought back.

So before such a result is graded the Agent tries one recovery:

* only when the unit is down (``inactive`` or ``failed``: ``systemctl
  reset-failed`` first, so a unit systemd gave up on can start, then
  ``systemctl start``) or already coming up (``activating``: waited for only).
  An ``active`` engine that does not answer is never restarted, and a stop in
  progress (``deactivating``) is never fought;
* only when the unit is enabled: an administrator who disabled it (leaving
  host mode, §5.0.1) decided, and the deploy that follows refuses an active
  engine next to the embedded one;
* never while a quiesce window is open on the node, whichever incident's: the
  quiesce stopped ``nvidia-dcgm`` on purpose and its restore starts it again;
* then a bounded wait (:data:`RECOVERY_BUDGET_SECONDS`) for ``dcgmi discovery
  -l`` to answer, after which the caller runs the diagnostic once more.

Packages are never installed, and the caller never calls this on an
embedded-mode node. Every spawn goes through the executor's runner, so a quiesce
window that opens meanwhile stops the activity at its next spawn
(``gpu_activity``); only the spawn failures ``OSError`` and
``TimeoutExpired`` are absorbed here.
"""

from __future__ import annotations

import logging
import subprocess
import time
from collections.abc import Callable
from typing import Any

__all__ = [
    "RECOVERY_BUDGET_SECONDS",
    "recover_host_engine",
]

LOGGER = logging.getLogger(__name__)

#: The host DCGM engine's unit (the DCGM 4 package's ``nvidia-dcgm``).
DCGM_HOSTENGINE_UNIT = "nvidia-dcgm"
#: How long the engine may take to answer ``dcgmi discovery -l`` after the start.
RECOVERY_BUDGET_SECONDS = 30.0
DISCOVERY_INTERVAL_SECONDS = 2.0
DISCOVERY_TIMEOUT_SECONDS = 10
SYSTEMCTL_TIMEOUT_SECONDS = 30
START_TIMEOUT_SECONDS = 60
#: ``systemctl is-active`` answers the Agent acts on: the unit is down, so it
#: is started, or systemd is starting it already, so it is only waited for.
DOWN_STATES = frozenset({"inactive", "failed"})
COMING_UP_STATES = frozenset({"activating"})
#: ``systemctl is-enabled`` answers that mean the node's image or administrator
#: wants the engine running; ``disabled``, ``masked`` and a unit systemd does
#: not know are left alone.
ENABLED_STATES = frozenset(
    {
        "enabled",
        "enabled-runtime",
        "static",
        "indirect",
        "generated",
        "alias",
        "linked",
        "linked-runtime",
    }
)
OUTPUT_TAIL = 400

Runner = Callable[..., Any]


def _call(run: Runner, argv: list[str], timeout: int) -> Any:
    try:
        return run(argv, check=False, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired):
        return None


def _answer(run: Runner, verb: str) -> str:
    """``systemctl <verb> nvidia-dcgm``'s one-word answer; ``unknown`` when
    systemctl could not be asked or said nothing."""

    completed = _call(
        run, ["systemctl", verb, DCGM_HOSTENGINE_UNIT], SYSTEMCTL_TIMEOUT_SECONDS
    )
    words = str(getattr(completed, "stdout", "") or "").split()
    return words[0] if words else "unknown"


def _open_windows(windows: Callable[[], list[dict[str, Any]]]) -> str | None:
    """Why a quiesce window keeps the engine stopped, ``None`` when none is open."""

    try:
        opened = windows()
    except Exception as exc:  # noqa: BLE001 - an unknown window state is a window
        return f"the quiesce window state could not be read ({type(exc).__name__})"
    if not opened:
        return None
    owners = ", ".join(
        f"{item.get('incident_id') or 'unknown'} {item.get('phase') or 'OPEN'}"
        for item in opened
    )
    return (
        f"a quiesce window is open on this node ({owners}): the quiesce stopped "
        f"{DCGM_HOSTENGINE_UNIT} and its restore starts it"
    )


def _refusal(before: str, enabled: str, windows: str | None) -> str | None:
    if windows is not None:
        return windows
    if before not in DOWN_STATES | COMING_UP_STATES:
        if before == "unknown":
            return f"the state of {DCGM_HOSTENGINE_UNIT} could not be read"
        return (
            f"{DCGM_HOSTENGINE_UNIT} is {before}: an engine systemd runs or is "
            "stopping is never restarted"
        )
    if before in DOWN_STATES and enabled not in ENABLED_STATES:
        return (
            f"{DCGM_HOSTENGINE_UNIT} is {enabled}: an engine that is not enabled is "
            "left alone (packages are never installed, and a disabled engine is "
            "an administrator's decision)"
        )
    return None


def _await_discovery(
    run: Runner,
    *,
    sleep: Callable[[float], None],
    monotonic: Callable[[], float],
    deadline: float,
) -> tuple[bool, int]:
    """Poll ``dcgmi discovery -l`` until it answers or the deadline passes."""

    polls = 0
    limit = int(RECOVERY_BUDGET_SECONDS / DISCOVERY_INTERVAL_SECONDS) + 1
    while True:
        polls += 1
        remaining = max(1, min(DISCOVERY_TIMEOUT_SECONDS, int(deadline - monotonic())))
        completed = _call(run, ["dcgmi", "discovery", "-l"], remaining)
        if completed is not None and getattr(completed, "returncode", 1) == 0:
            return True, polls
        if polls >= limit or monotonic() + DISCOVERY_INTERVAL_SECONDS > deadline:
            return False, polls
        sleep(DISCOVERY_INTERVAL_SECONDS)


def recover_host_engine(
    run: Runner,
    *,
    windows: Callable[[], list[dict[str, Any]]],
    sleep: Callable[[float], None],
    monotonic: Callable[[], float] = time.monotonic,
    only_when_down: bool = False,
) -> dict[str, Any] | None:
    """Try one start of a down host engine; the ``hostengine_recovery`` evidence.

    ``windows`` lists the node's open quiesce windows (``GpuActivityGate.
    open_windows``). With ``only_when_down`` (the diagnostic failed for a
    reason other than an unreachable engine) an engine systemd runs is no
    recovery case at all: ``None``. Otherwise the evidence names what was
    tried: ``attempted``, ``before_state``/``after_state`` (``systemctl
    is-active``), ``unit_enabled``, ``seconds`` and, when attempted,
    ``actions``, ``engine_answers`` and ``discovery_polls``; a recovery that
    was not attempted carries the ``reason``.
    """

    started = monotonic()
    before = _answer(run, "is-active")
    if only_when_down and before not in DOWN_STATES | COMING_UP_STATES:
        return None
    enabled = _answer(run, "is-enabled")
    evidence: dict[str, Any] = {
        "attempted": False,
        "unit": DCGM_HOSTENGINE_UNIT,
        "before_state": before,
        "unit_enabled": enabled,
    }
    refusal = _refusal(before, enabled, _open_windows(windows))
    if refusal is not None:
        evidence.update(
            after_state=before,
            seconds=round(monotonic() - started, 3),
            reason=refusal,
        )
        return evidence
    actions: list[str] = []
    started_ok = True
    if before in DOWN_STATES:
        if before == "failed":
            _call(
                run,
                ["systemctl", "reset-failed", DCGM_HOSTENGINE_UNIT],
                SYSTEMCTL_TIMEOUT_SECONDS,
            )
            actions.append("reset-failed")
        start = _call(
            run, ["systemctl", "start", DCGM_HOSTENGINE_UNIT], START_TIMEOUT_SECONDS
        )
        actions.append("start")
        started_ok = start is not None and getattr(start, "returncode", 1) == 0
        if not started_ok:
            evidence["start_returncode"] = getattr(start, "returncode", None)
            evidence["start_error"] = str(getattr(start, "stderr", "") or "")[
                -OUTPUT_TAIL:
            ]
    else:
        actions.append("wait")
    answered, polls = (
        _await_discovery(
            run,
            sleep=sleep,
            monotonic=monotonic,
            deadline=started + RECOVERY_BUDGET_SECONDS,
        )
        if started_ok
        else (False, 0)
    )
    evidence.update(
        attempted=True,
        actions=actions,
        engine_answers=answered,
        discovery_polls=polls,
        after_state=_answer(run, "is-active"),
        seconds=round(monotonic() - started, 3),
    )
    if not answered:
        evidence["reason"] = (
            f"{DCGM_HOSTENGINE_UNIT} did not answer dcgmi discovery -l within "
            f"{RECOVERY_BUDGET_SECONDS:g} s"
            if started_ok
            else f"systemctl start {DCGM_HOSTENGINE_UNIT} failed"
        )
    LOGGER.warning(
        "DCGM host engine recovery: %s was %s (%s); %s; engine %s after %.1f s",
        DCGM_HOSTENGINE_UNIT,
        before,
        enabled,
        "+".join(actions),
        "answers" if answered else "still unreachable",
        evidence["seconds"],
    )
    return evidence

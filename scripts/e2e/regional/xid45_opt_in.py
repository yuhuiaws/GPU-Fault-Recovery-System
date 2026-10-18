"""The control-worker window that opts a graded node's solo XID 45 into RESTART_FM.

From batch 131 a solo XID 45 on a graded NVSwitch node (Hopper SXM, HGX
B200/B300) is observe-only by default (``gpu_fault.orchestration.xid45_policy``).
``GPU_FAULT_XID45_SOLO_NVSWITCH_RESTART_FM=true`` -- the site switch, read where
the control plane finalizes the XID: the control-worker's XID correlation loop
and fault processor -- brings back batch 128's full GPU + NVSwitch reset chain.
DESTR-010 proves the default first and then the chain under this window;
DESTR-019 audits the chain's RESET_ALL_GPUS_NVSWITCHES under it.

The window is ``control_plane_env_window``'s, for this one switch: the
Deployment's pre-window env is recorded in a baseline file before the patch,
the open converges only once every ready replica reads ``true``, the close
restores exactly the baseline (an absent variable goes back to absent), a
window another writer opened is refused, and one a later release re-rendered is
retired. Opening and closing each roll the control-worker once, so the case
opens it before it injects and closes it once the workflow is terminal; no other
case may run while it is open, because every solo XID 45 of every graded node
of the site would run the chain.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from scripts.e2e.regional import control_plane_env_window as control_window
from scripts.e2e.regional import fabric_restart_shape as shapes
from scripts.e2e.regional.acceptance_runner_common import write_json_atomic

SWITCH = control_window.XID45_SOLO_RESTART_FM_VARIABLE
OPTED_IN = control_window.POLICY_SWITCHES[SWITCH]
BASELINE_NAME = "xid45-opt-in-window.json"
ROLLOUT_TIMEOUT_SECONDS = 600
#: The window rolls the control-worker twice: the open and the close.
WINDOW_WRITES = 2
TRUE_TOKENS = frozenset({"1", "on", "true", "yes"})


def settings(case_dir: Path) -> control_window.Settings:
    return control_window.Settings(
        baseline=case_dir / BASELINE_NAME,
        rollout_timeout_seconds=ROLLOUT_TIMEOUT_SECONDS,
    )


def opted_in(survey: dict[str, Any]) -> bool | None:
    """Whether the control-worker already opts in: every ready replica reads a
    true value (True), none does (False), or they disagree or none is ready
    (None, a half rollout nothing may be judged on)."""

    replicas = survey.get("replicas") or []
    values = {
        str((item.get("values") or {}).get(SWITCH) or "").strip().lower() in TRUE_TOKENS
        for item in replicas
    }
    if len(values) != 1:
        return None
    return values.pop()


def switch_survey(regional: Any) -> dict[str, Any]:
    """What every ready control-worker replica reads for the switch (read-only)."""

    survey = control_window.survey(regional)
    return {"opted_in": opted_in(survey), "replicas": survey.get("replicas") or []}


def needs_window(grading: Any, switch: dict[str, Any] | None) -> bool:
    """Whether the case must open the opt-in to drive the chain: a graded node
    whose control plane defaults to observe-only and has not opted in."""

    return (
        shapes.shape_of(grading, opted_in=(switch or {}).get("opted_in") is True)
        == shapes.OBSERVE_ONLY
    )


def switch_errors(grading: Any, switch: dict[str, Any] | None) -> list[str]:
    """A half-rolled control-worker says nothing the case may plan on."""

    if (
        switch is not None
        and switch.get("opted_in") is None
        and shapes.observe_only_default(grading)
    ):
        return [
            f"ready control-worker replicas disagree on {SWITCH} (or none is ready)"
        ]
    return []


def open_window(regional: Any, case_dir: Path) -> dict[str, Any]:
    """Open the opt-in; the record without its surveys is the case's evidence."""

    record = control_window.open_window(
        settings(case_dir),
        regional,
        control_window.survey(regional),
        {SWITCH: OPTED_IN},
    )
    report = control_window.without_survey(record)
    write_json_atomic(case_dir / "xid45-opt-in-open.json", report)
    return report


def close_window(regional: Any, case_dir: Path) -> dict[str, Any] | None:
    """Close the opt-in this case opened; None when it never opened one."""

    if not settings(case_dir).baseline.is_file():
        return None
    record = control_window.close_window(
        settings(case_dir), regional, control_window.survey(regional)
    )
    report = control_window.without_survey(record)
    write_json_atomic(case_dir / "xid45-opt-in-close.json", report)
    return report


def window_errors(
    opened: dict[str, Any] | None, closed: dict[str, Any] | None
) -> list[str]:
    """The window the case opened is closed again and owned what it said."""

    if opened is None:
        return []
    errors: list[str] = []
    if opened.get("assignments") != {SWITCH: OPTED_IN}:
        errors.append(f"the opt-in window carried {opened.get('assignments')}")
    if opened.get("state") != "OPEN":
        errors.append(f"the opt-in window never opened ({opened.get('state')})")
    if closed is None or closed.get("state") != "CLOSED":
        errors.append("the opt-in window was not closed again")
    return errors


def identity_errors(before: dict[str, Any], after: dict[str, Any]) -> list[str]:
    """The runtime identity across the window: identical but for the
    control-worker's generation, which moves by exactly the two rolls, and
    whose pod template is back to its pre-window digest (DESTR-018's rule)."""

    from scripts.e2e.regional.run_destr018_lifetime_deadline import (
        identity_errors as window_identity_errors,
    )

    return window_identity_errors(before, after, worker_generation_delta=WINDOW_WRITES)

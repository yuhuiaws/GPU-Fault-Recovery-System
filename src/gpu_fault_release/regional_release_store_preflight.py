"""The control-plane store reads a transaction must pass before it starts.

Two questions, each one ``kubectl exec`` of a probe into the CPU ingress Pod:

* Is the remote command queue idle? A PENDING / LEASED / WAITING remote command
  is a node action mid-flight; the transaction that rolls the executor or the
  control plane under it would lose or duplicate it.

* Is a node install in flight? R4 (2026-09-09) changed the node-action command
  id of REMEDIATE_DRIVER / UPDATE_SOFTWARE_FIRMWARE / REMEDIATE_EFA_DRIVER from
  ``<key>/<node>/agent-N`` to ``<key>/<node>``. The new control plane reads old
  rows back through a one-release shim; a control plane of the other shape --
  the previous release a rollback restores, or the old one an upgrade replaces
  mid-step -- derives an id the agent ledger has never seen and submits a
  SECOND install on a node still running the first (install timeout 1800 s).
  So before an upgrade or rollback transaction opens, and before the automatic
  rollback, the engine asks once whether any of those steps is PENDING or
  WAITING and refuses with the workflow ids and nodes named. A store that
  cannot answer refuses too -- fail closed, naming the error -- because the
  install the probe could not see is the one that gets doubled.
  ``--allow-inflight-installs`` on ``gpu-fault-admin deploy`` is the operator's
  consent to proceed anyway; it travels down the deploy's process chain as
  ``GPU_FAULT_RELEASE_ALLOW_INFLIGHT_INSTALLS`` like the other release-engine
  consents, and the gate then logs what it is skipping instead of refusing.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

from gpu_fault_release.regional_release_config import ReleaseError
from gpu_fault_release.regional_release_narration import narrate_step
from gpu_fault_release.regional_release_probes import probe_source
from gpu_fault_release.regional_release_runtime_identity import (
    cpu_ingress_pod_if_running,
    exec_cpu_ingress_probe,
)

# Mirrored in ``gpu_fault.admin.release_consent``; a test pins the spellings.
ALLOW_INFLIGHT_INSTALLS_ENV = "GPU_FAULT_RELEASE_ALLOW_INFLIGHT_INSTALLS"
ALLOW_INFLIGHT_INSTALLS_FLAG = "--allow-inflight-installs"
# The operations whose command id R4 made generation-stable. Pinned equal to
# ``operation_registry.GENERATION_STABLE_COMMAND_OPERATIONS`` by a test rather
# than imported: the probe spells them out for the deployed ``gpu_fault`` that
# predates the registry field, and this module names them for the operator.
INSTALL_OPERATIONS = (
    "REMEDIATE_DRIVER",
    "UPDATE_SOFTWARE_FIRMWARE",
    "REMEDIATE_EFA_DRIVER",
)
_CHECK = "the in-flight install check"


def remote_commands_are_idle(release: Any) -> bool:
    script = (
        "from gpu_fault.app import ApplicationContext;"
        "stats=ApplicationContext.from_environment()"
        ".store.remote_command_stats();"
        "bad={name:int(stats['by_status'].get(name,0)) "
        "for name in ('PENDING','LEASED','WAITING') "
        "if int(stats['by_status'].get(name,0))};"
        "print(bad if bad else '')"
    )
    # Asked once, outside the retry loop: a control plane with no Running
    # ingress Pod has nothing dispatching, so the absence answers the
    # question rather than failing it, which is why this cannot use the
    # probe helper's resolver -- for every other caller an absent Pod is a
    # check that could not be made.
    if not cpu_ingress_pod_if_running(release):
        return True
    for attempt in range(3):
        try:
            output = exec_cpu_ingress_probe(
                release,
                script=script,
                failure="the remote command idle check",
                interactive=False,
                retries=0,
            )
        except ReleaseError:
            if attempt == 2:
                raise
            time.sleep(2)
            continue
        return not output.strip()
    raise ReleaseError("remote command idle check exhausted retries")


class InflightInstallsRefused(ReleaseError):
    """The gate refused: an install is in flight, or the store could not say.

    A distinct type because the automatic rollback has to tell this apart from
    a rollback that started and failed: nothing was restored, so the
    transaction must stay in ``failed`` rather than go to ``rollback-failed``.
    """


def inflight_installs_allowed(environment: dict[str, str] | None = None) -> bool:
    """Whether this command carried ``--allow-inflight-installs``."""

    return bool(
        (environment if environment is not None else os.environ)
        .get(ALLOW_INFLIGHT_INSTALLS_ENV, "")
        .strip()
    )


def inflight_install_snapshot(release: Any) -> dict[str, Any]:
    """One store read: the in-flight install steps, as the probe reports them."""

    raw = exec_cpu_ingress_probe(
        release,
        script=probe_source("inflight_installs"),
        failure=_CHECK,
        sensitive=True,
        interactive=False,
    )
    try:
        result = json.loads(raw)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ReleaseError(f"{_CHECK} returned invalid evidence") from exc
    if not isinstance(result, dict) or not isinstance(result.get("inflight"), list):
        raise ReleaseError(f"{_CHECK} returned non-object evidence")
    return result


def describe_inflight_step(step: dict[str, Any]) -> str:
    nodes = ",".join(str(node) for node in step.get("node_ids") or []) or "?"
    return (
        f"{step.get('workflow_id')} {step.get('operation')} "
        f"step {step.get('step_index')} on {nodes} ({step.get('step_status')})"
    )


def _operations() -> str:
    return " / ".join(INSTALL_OPERATIONS)


def require_no_inflight_installs(release: Any, *, action: str) -> dict[str, Any]:
    """Refuse ``action`` while an install step is in flight; return the snapshot.

    ``action`` names the transaction for the message (``upgrade`` /
    ``rollback``). With the override set the gate still reads the store when
    it can, so the log says what was skipped, and proceeds either way.
    """

    allowed = inflight_installs_allowed()
    try:
        snapshot = inflight_install_snapshot(release)
    except ReleaseError as exc:
        if allowed:
            narrate_step(
                "inflight-installs-unchecked",
                action=action,
                override=ALLOW_INFLIGHT_INSTALLS_FLAG,
                error=f"{type(exc).__name__}: {exc}",
            )
            return {"inflight": [], "inflight_count": 0, "error": str(exc)}
        raise InflightInstallsRefused(
            f"{action} refused: {_CHECK} could not read the control-plane store "
            f"({type(exc).__name__}: {exc}). A {_operations()} step already "
            "handed to a node agent would be submitted a second time by the "
            "control plane this transaction puts in place; retry when the "
            f"control plane answers, or rerun with {ALLOW_INFLIGHT_INSTALLS_FLAG} "
            "to proceed without the check"
        ) from exc
    count = int(snapshot.get("inflight_count") or 0)
    if not count:
        return snapshot
    steps = "; ".join(describe_inflight_step(step) for step in snapshot["inflight"])
    if allowed:
        narrate_step(
            "inflight-installs-overridden",
            action=action,
            override=ALLOW_INFLIGHT_INSTALLS_FLAG,
            count=count,
            steps=steps,
        )
        return snapshot
    raise InflightInstallsRefused(
        f"{action} refused: {count} {_operations()} step(s) in flight, and the "
        "control plane this transaction puts in place would submit each install "
        f"a second time on a node still running it: {steps}. Wait for them to "
        "finish (a driver or firmware install takes up to 30 minutes), or rerun "
        f"with {ALLOW_INFLIGHT_INSTALLS_FLAG} to proceed anyway"
    )

"""The control-plane store reads a transaction must pass before it starts.

Two questions, each one ``kubectl exec`` of a probe into a Running
control-plane Pod:

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
  WAITING and refuses with the workflow ids and nodes named.

  A store that cannot answer refuses too -- fail closed, naming the error --
  with one exception, decided by the caller: the AUTOMATIC rollback proceeds
  and logs ``inflight-installs-unchecked`` loudly, because its primary scenario
  is a control plane that is down (CrashLoop, Aurora window, broken wheel), a
  dead control plane dispatches nothing, and wedging production in ``failed``
  is the worse failure. The read tries every Running control-plane role, not
  only the ingress Pod the failed upgrade just rolled.

  ``--allow-inflight-installs`` on ``gpu-fault-admin deploy`` is the operator's
  consent to proceed anyway; it travels down the deploy's process chain as
  ``GPU_FAULT_RELEASE_ALLOW_INFLIGHT_INSTALLS`` like the other release-engine
  consents. The engine's own ``rollback`` mode has no flag, so a refusal there
  names the variable instead.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Literal

from gpu_fault_release import regional_deployment_inventory as inventory
from gpu_fault_release.regional_release_config import ReleaseError
from gpu_fault_release.regional_release_narration import narrate_step
from gpu_fault_release.regional_release_probes import probe_source
from gpu_fault_release.regional_release_runtime_identity import (
    CONTROL_PLANE_PYTHON,
    cpu_ingress_pod_if_running,
    exec_cpu_ingress_probe,
)

from gpu_fault.execution.config import NODE_INSTALL_OPERATIONS

# Mirrored in ``gpu_fault.admin.release_consent``; a test pins the spellings.
ALLOW_INFLIGHT_INSTALLS_ENV = "GPU_FAULT_RELEASE_ALLOW_INFLIGHT_INSTALLS"
ALLOW_INFLIGHT_INSTALLS_FLAG = "--allow-inflight-installs"
# The executor's install set is the one whose command id R4 made
# generation-stable; a test pins it equal to
# ``operation_registry.GENERATION_STABLE_COMMAND_OPERATIONS``. The probe spells
# the same three out inline for the deployed ``gpu_fault`` it runs against.
INSTALL_OPERATIONS = tuple(operation.value for operation in NODE_INSTALL_OPERATIONS)
# The probe's row window; more executable workflows than this and the scan is
# unbounded (the probe says so), which the gate treats as "could not check".
SCAN_WINDOW = 1000
# What ``rollout-regional-release.sh`` exits with when a transaction was
# refused for an in-flight install. The release driver
# (``scripts/release_failure_recovery.py``, which mirrors the value) sees only
# the exit code, and must not record the refusal as a failed rollback: nothing
# was rolled back. Distinct from the generic 2 every other engine error uses.
INFLIGHT_INSTALLS_REFUSED_EXIT_CODE = 3
_CHECK = "the in-flight install check"

Unreadable = Literal["refuse", "proceed"]


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


def _exec_in_role(release: Any, deployment: str, script: str) -> str:
    """One read through any Running Pod of ``deployment``; raises ReleaseError."""

    namespace = release.config.namespace
    pod = str(
        release.runner.run(
            release._cpu(
                "-n",
                namespace,
                "get",
                "pod",
                "-l",
                f"app={deployment}",
                "--field-selector=status.phase=Running",
                "-o",
                "jsonpath={.items[0].metadata.name}",
            ),
            capture=True,
        )
        or ""
    )
    if not pod:
        raise ReleaseError("no Running Pod")
    return str(
        release.runner.run(
            release._cpu(
                "-n",
                namespace,
                "exec",
                pod,
                "--",
                CONTROL_PLANE_PYTHON,
                "-c",
                script,
            ),
            capture=True,
            sensitive=True,
        )
        or ""
    )


def _exec_store_probe(release: Any, *, script: str, failure: str) -> str:
    """Read through the ingress Pod, then through every other control-plane
    role that is Running; raise naming every role tried when none answers.

    The ingress Pod is the one an upgrade rolls; right after a failed upgrade it
    may be the only role that is down. The consumer roles run the same wheel
    against the same store and answer the same probe.
    """

    errors: list[str] = []
    try:
        return exec_cpu_ingress_probe(
            release,
            script=script,
            failure=failure,
            sensitive=True,
            interactive=False,
        )
    except ReleaseError as exc:
        errors.append(f"{inventory.CPU_INGRESS_DEPLOYMENT}: {exc}")
    for deployment in inventory.CPU_RUNTIME_DEPLOYMENTS:
        if deployment == inventory.CPU_INGRESS_DEPLOYMENT:
            continue
        try:
            return _exec_in_role(release, deployment, script)
        except ReleaseError as exc:
            errors.append(f"{deployment}: {exc}")
    raise ReleaseError(
        f"{failure} could not reach any Running control-plane Pod: " + "; ".join(errors)
    )


def inflight_install_snapshot(release: Any) -> dict[str, Any]:
    """One store read: the in-flight install steps, as the probe reports them."""

    raw = _exec_store_probe(
        release, script=probe_source("inflight_installs"), failure=_CHECK
    )
    try:
        result = json.loads(raw)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ReleaseError(f"{_CHECK} returned invalid evidence") from exc
    if not isinstance(result, dict) or not isinstance(result.get("inflight"), list):
        raise ReleaseError(f"{_CHECK} returned non-object evidence")
    if result.get("bounded") is False:
        raise ReleaseError(
            f"{_CHECK} could not bound the scan: more than {SCAN_WINDOW} "
            f"executable workflows ({result.get('scanned')} rows read), so an "
            "install past the window would read as clear"
        )
    return result


def describe_inflight_step(step: dict[str, Any]) -> str:
    nodes = ",".join(str(node) for node in step.get("node_ids") or []) or "?"
    return (
        f"{step.get('workflow_id')} {step.get('operation')} "
        f"step {step.get('step_index')} on {nodes} ({step.get('step_status')})"
    )


def _operations() -> str:
    return " / ".join(INSTALL_OPERATIONS)


def _lever(action: str) -> str:
    """The override as the operator can actually reach it from this entrypoint."""

    if action == "rollback":
        return (
            f"set {ALLOW_INFLIGHT_INSTALLS_ENV}=1 on the rollback to proceed anyway "
            f"(gpu-fault-admin deploy passes it as {ALLOW_INFLIGHT_INSTALLS_FLAG})"
        )
    return (
        f"rerun gpu-fault-admin deploy with {ALLOW_INFLIGHT_INSTALLS_FLAG} "
        "to proceed anyway"
    )


def require_no_inflight_installs(
    release: Any,
    *,
    action: str,
    unreadable: Unreadable = "refuse",
) -> dict[str, Any]:
    """Refuse ``action`` while an install step is in flight; return the snapshot.

    ``action`` names the transaction for the message (``upgrade`` /
    ``rollback``). ``unreadable`` is what a store that cannot answer means:
    ``refuse`` (manual upgrade and rollback: fail closed, the env lever opens
    it) or ``proceed`` (the automatic rollback: log and go on, an actual
    in-flight install still refuses). With the operator's consent set the gate
    still reads when it can, so the log says what was skipped.
    """

    allowed = inflight_installs_allowed()
    try:
        snapshot = inflight_install_snapshot(release)
    except ReleaseError as exc:
        if allowed or unreadable == "proceed":
            narrate_step(
                "inflight-installs-unchecked",
                action=action,
                reason=(
                    ALLOW_INFLIGHT_INSTALLS_FLAG
                    if allowed
                    else "automatic rollback proceeds on an unreadable store"
                ),
                error=f"{type(exc).__name__}: {exc}",
            )
            return {"inflight": [], "inflight_count": 0, "error": str(exc)}
        raise InflightInstallsRefused(
            f"{action} refused: {_CHECK} could not read the control-plane store "
            f"({type(exc).__name__}: {exc}). A {_operations()} step already "
            "handed to a node agent would be submitted a second time by the "
            "control plane this transaction puts in place; retry when the "
            f"control plane answers, or {_lever(action)} without the check"
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
        f"finish (a driver or firmware install takes up to 30 minutes), or "
        f"{_lever(action)}"
    )

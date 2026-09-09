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

  What the check could not establish is classified by WHERE it failed. The
  probe is exec'd through a shell wrapper that prints an exit marker after it,
  whatever it did, so its output proves whether the probe process ran at all;
  the Pod list carries each container's ``ready`` flag, so the engine knows
  whether the Pod it could not enter was a live dispatcher. Three classes:

  * ``StoreUnreachable`` -- the probe provably never ran AND nothing could have
    been dispatching: no Running control-plane Pod in any role, or ``kubectl
    exec`` failing before the wrapper answered on Pods with NO ready container
    (CrashLoopBackOff, ContainerCreating, terminating). The one failure the
    AUTOMATIC rollback proceeds over, logging ``inflight-installs-unchecked``
    loudly: its primary scenario is a control plane that is down (broken
    wheel), a dead control plane dispatches nothing, and wedging production in
    ``failed`` is the worse failure.
  * ``KubectlFailure`` -- a live control plane the engine could not ask:
    ``kubectl exec`` failed before the wrapper answered on a Pod with a READY
    container (kubelet down or the node partitioned while the API server still
    reports Running, a deploy-host identity without ``pods/exec``, an image
    without ``sh``), or ``get pod`` itself failed (API server unreachable,
    expired ``aws eks get-token``, no list RBAC). The ready Pod's warm pool may
    still be dispatching, so this refuses in every mode, automatic rollback
    included. The consent variable MAY override it -- a transport / RBAC fault
    is something the operator can verify by hand -- and the log then says
    exactly what was skipped and why.
  * Evidence defects -- everything after the probe started: it exited non-zero
    (its output is then not trusted even when it parses), it printed no JSON,
    it reported its own ``probe_error`` (a fresh connection failing in the
    Aurora window while the Pod's warm pool still dispatches), it hung past the
    timeout, its scan was unbounded. Refuse in every mode, consent included:
    the store did not say "clear", and the cause has to be fixed.

  ``--allow-inflight-installs`` on ``gpu-fault-admin deploy`` is the operator's
  consent to proceed over a LISTED in-flight set, over a store no Pod could
  answer for, or over a kubectl-level failure; it travels down the deploy's
  process chain as ``GPU_FAULT_RELEASE_ALLOW_INFLIGHT_INSTALLS`` like the other
  release-engine consents. The engine's own ``rollback`` mode has no flag, so a
  refusal there names the variable instead. The gate returns a verdict
  (``checked`` / ``verdict`` / ``reason`` / ``steps`` / ``checked_at``) that
  both the upgrade and the rollback persist with their first checkpoint and
  the release driver copies into its own record.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from datetime import datetime, timezone
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
# A store read that has not returned in two minutes is not going to say
# "clear"; refusing beats hanging the engine (and the driver behind it).
PROBE_TIMEOUT_SECONDS = 120.0
# The wrapper's last line, whatever the probe did. Its presence on stdout is
# the proof that the probe process ran; ``kubectl`` failing before it = the
# probe never started.
PROBE_EXIT_MARKER = "__GPU_FAULT_PROBE_EXIT"
# The probe travels on stdin (``python -``), so the ``+`` echo shows this line
# rather than the probe body. stderr is merged so the cause of a probe that died
# on import reaches the log; the marker echo makes the wrapper itself exit 0,
# which is what lets ``Runner.run`` hand the output back instead of raising.
PROBE_WRAPPER = f'{CONTROL_PLANE_PYTHON} - 2>&1; echo "{PROBE_EXIT_MARKER}=$?"'
# Every Running Pod of a role, one per line: its name, a tab, then the
# ``ready`` flag of each of its containers (space-separated; empty for a Pod
# whose containers have no status yet). Tests key on this selector to answer
# the gate's ``get pod`` as a healthy control plane would.
RUNNING_PODS_JSONPATH = (
    'jsonpath={range .items[*]}{.metadata.name}{"\\t"}'
    '{.status.containerStatuses[*].ready}{"\\n"}{end}'
)
# How much of the probe's non-JSON output a refusal quotes.
_OUTPUT_TAIL = 600
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


class StoreUnreachable(ReleaseError):
    """The probe provably never ran and nothing could have been dispatching: no
    Running control-plane Pod in any role, or ``kubectl exec`` failed before the
    wrapper could print its exit marker on Pods with NO ready container
    (CrashLoopBackOff, ContainerCreating, terminating).

    The one failure an automatic rollback (or the consent) may proceed over.
    A Pod with a ready container that kubectl could not enter is not this -- it
    is ``KubectlFailure``, a live dispatcher the engine could not ask. And
    everything that happens after the probe started -- a non-zero exit, no
    JSON, a ``probe_error``, a hang, an unbounded scan -- is evidence, raised as
    a plain ``ReleaseError``, and refuses in every mode: the store answered, and
    the answer was not "clear".
    """


class KubectlFailure(ReleaseError):
    """A live control plane the engine could not ask, because ``kubectl`` --
    not the store -- failed: ``exec`` into a Pod with a READY container failed
    before the wrapper answered (kubelet down or the node partitioned while the
    API server still reports Running, a deploy-host identity without
    ``pods/exec``, an image without ``sh``), or ``get pod`` itself failed (API
    server unreachable, expired ``aws eks get-token``, no list RBAC).

    Refuses in every mode, the automatic rollback included: the ready Pod's warm
    pool may be dispatching right now. The consent variable may override it (a
    transport / RBAC fault is checkable by hand, unlike a probe defect); the
    narration then carries ``short`` so the log says what was skipped.
    """

    def __init__(self, message: str, *, short: str) -> None:
        super().__init__(message)
        self.short = short


def inflight_installs_allowed(environment: dict[str, str] | None = None) -> bool:
    """Whether this command carried ``--allow-inflight-installs``."""

    return bool(
        (environment if environment is not None else os.environ)
        .get(ALLOW_INFLIGHT_INSTALLS_ENV, "")
        .strip()
    )


def _running_pods(release: Any, deployment: str) -> list[tuple[str, bool]]:
    """Every Running Pod of ``deployment`` with whether any container of it is
    ready -- not only ``items[0]``: a stuck rollout leaves the old ReplicaSet's
    Pod Running beside the new CrashLooping one, and the old Pod is the one
    still dispatching. ``ready`` decides what a ``kubectl exec`` failure on the
    Pod means (``_exec_store_probe``). Raises ``KubectlFailure`` when the list
    itself fails: that is the API server or the kubeconfig, not the store.
    """

    try:
        listed = release.runner.run(
            release._cpu(
                "-n",
                release.config.namespace,
                "get",
                "pod",
                "-l",
                f"app={deployment}",
                "--field-selector=status.phase=Running",
                "-o",
                RUNNING_PODS_JSONPATH,
            ),
            capture=True,
        )
    except ReleaseError as exc:
        raise KubectlFailure(
            f"kubectl could not list control-plane Pods for role {deployment}: "
            f"{exc} -- the API server or kubeconfig, not the store",
            short="kubectl could not list the control-plane Pods",
        ) from exc
    pods: list[tuple[str, bool]] = []
    for line in str(listed or "").splitlines():
        name, _tab, readiness = line.strip().partition("\t")
        if name:
            pods.append((name, "true" in readiness.split()))
    return pods


def _exec_probe_in_pod(release: Any, pod: str, script: str) -> str:
    """One run of the wrapped probe in ``pod``; returns the wrapper's stdout.

    Raises ``KubectlFailure`` when ``kubectl`` failed before the wrapper
    answered (the wrapper itself always exits 0, so a non-zero exit is
    kubectl's: API server unreachable, container not running, no exec RBAC);
    the caller decides from the Pod's readiness whether that is an unreachable
    store or a live dispatcher it could not ask. Raises a plain ``ReleaseError``
    when the exec started and hung past the timeout -- the store did not answer
    in time, which is an answer of sorts.
    """

    try:
        return str(
            release.runner.run(
                release._cpu(
                    "-n",
                    release.config.namespace,
                    "exec",
                    "-i",
                    pod,
                    "--",
                    "sh",
                    "-c",
                    PROBE_WRAPPER,
                ),
                input_text=script,
                capture=True,
                timeout_seconds=PROBE_TIMEOUT_SECONDS,
            )
            or ""
        )
    except ReleaseError as exc:
        if isinstance(exc.__cause__, subprocess.TimeoutExpired):
            raise ReleaseError(
                f"{_CHECK} did not answer within {PROBE_TIMEOUT_SECONDS:.0f} s in "
                f"{pod}: the control-plane store is not answering"
            ) from exc
        raise KubectlFailure(str(exc), short="exec failed on a ready Pod") from exc


def _exec_store_probe(release: Any, *, script: str, failure: str) -> str:
    """Run the wrapped probe in the first Running control-plane Pod that lets it
    start: the ingress role first (the one an upgrade rolls), then every other
    role, every Running Pod of each. The consumer roles run the same wheel
    against the same store and answer the same probe. First evidence wins.

    When no Pod produced the marker: ``KubectlFailure`` if any role's list
    failed or any Pod with a ready container refused the exec (a live control
    plane the engine could not ask), else ``StoreUnreachable`` naming every Pod
    and role tried (nothing was dispatching). A timeout propagates at once (the
    roles read the same store, and each would cost the whole window again).
    """

    unreachable: list[str] = []
    kubectl: list[str] = []
    exec_failed_on_ready = False
    roles = [
        inventory.CPU_INGRESS_DEPLOYMENT,
        *(
            deployment
            for deployment in inventory.CPU_RUNTIME_DEPLOYMENTS
            if deployment != inventory.CPU_INGRESS_DEPLOYMENT
        ),
    ]
    for deployment in roles:
        try:
            pods = _running_pods(release, deployment)
        except KubectlFailure as exc:
            kubectl.append(str(exc))
            continue
        if not pods:
            unreachable.append(f"{deployment}: no Running Pod")
            continue
        for pod, ready in pods:
            try:
                return _exec_probe_in_pod(release, pod, script)
            except KubectlFailure as exc:
                if ready:
                    exec_failed_on_ready = True
                    kubectl.append(
                        f"{deployment}/{pod} is Running with a ready container, "
                        f"but kubectl exec failed before the probe started: {exc}"
                    )
                else:
                    unreachable.append(
                        f"{deployment}/{pod} has no ready container: {exc}"
                    )
    if kubectl:
        raise KubectlFailure(
            f"{failure} could not ask a live control plane -- kubectl failed, not "
            "the store: " + "; ".join([*kubectl, *unreachable]),
            short=(
                "exec failed on a ready Pod"
                if exec_failed_on_ready
                else "kubectl could not list the control-plane Pods"
            ),
        )
    raise StoreUnreachable(
        f"{failure} could not reach any Running control-plane Pod: "
        + "; ".join(unreachable)
    )


def _parse_probe_output(raw: str) -> dict[str, Any]:
    """The probe's JSON out of the wrapper's stdout, or the evidence defect.

    The marker line says the probe process ran and how it exited; the JSON is
    the last line that parses as an object (stderr is merged in, so import-time
    chatter may precede it). Anything else is quoted so the cause is readable.
    """

    lines = raw.splitlines()
    marker_at = next(
        (
            index
            for index in range(len(lines) - 1, -1, -1)
            if lines[index].strip().startswith(f"{PROBE_EXIT_MARKER}=")
        ),
        None,
    )
    if marker_at is None:
        raise ReleaseError(
            f"{_CHECK} returned invalid evidence: no exit marker, so the probe "
            f"wrapper did not run to completion: {_tail(lines)}"
        )
    try:
        exit_code = int(lines[marker_at].strip().split("=", 1)[1])
    except ValueError:
        exit_code = -1
    body = lines[:marker_at]
    for line in reversed(body):
        candidate = line.strip()
        if not candidate.startswith("{"):
            continue
        try:
            document = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(document, dict):
            if document.get("probe_error") is not None:
                raise ReleaseError(
                    f"{_CHECK} probe reported an error instead of evidence: "
                    f"{document['probe_error']}"
                )
            if exit_code:
                # It printed a snapshot and then died (an exception after the
                # print, a killed interpreter): not the run it meant to finish.
                break
            if isinstance(document.get("inflight"), list):
                return document
            break
    if exit_code:
        raise ReleaseError(
            f"{_CHECK} probe exited {exit_code}; its evidence is not trusted: "
            f"{_tail(body)}"
        )
    raise ReleaseError(
        f"{_CHECK} returned invalid evidence (no JSON object with an 'inflight' "
        f"list): {_tail(body)}"
    )


def _tail(lines: list[str]) -> str:
    text = " | ".join(line.strip() for line in lines if line.strip())
    if not text:
        return "<no output>"
    return text if len(text) <= _OUTPUT_TAIL else "..." + text[-_OUTPUT_TAIL:]


def inflight_install_snapshot(release: Any) -> dict[str, Any]:
    """One store read: the in-flight install steps, as the probe reports them.

    Raises ``StoreUnreachable`` when the probe provably never ran and nothing
    was dispatching, ``KubectlFailure`` when a live control plane could not be
    asked (kubectl, not the store, failed), and a plain ``ReleaseError`` for a
    defect in evidence the probe did return (non-zero exit, no JSON,
    ``probe_error``, unbounded scan) or a hang. The caller treats only the first
    as "unreadable", the second as refusable-but-consentable, the third as an
    answer that is binding.
    """

    raw = _exec_store_probe(
        release, script=probe_source("inflight_installs"), failure=_CHECK
    )
    result = _parse_probe_output(raw)
    if result.get("bounded") is False:
        raise ReleaseError(
            f"{_CHECK} could not bound the scan: more than {SCAN_WINDOW} "
            f"executable workflows ({result.get('scanned')} rows read), so an "
            "install past the window would read as clear; drain the executable "
            f"backlog below {SCAN_WINDOW} rows (or raise the probe's window)"
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


def _verdict(
    verdict: str,
    *,
    checked: bool,
    reason: str | None,
    steps: list[str],
    snapshot: dict[str, Any] | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    """The durable record of what the gate decided and on what evidence."""

    record: dict[str, Any] = {
        "checked": checked,
        "verdict": verdict,
        "reason": reason,
        "steps": steps,
        # A rollback re-entered after ``rollback-cpu-restored`` skips the gate
        # and leaves the FIRST attempt's verdict in the state; the reader needs
        # to know how old it is. UTC, ISO-8601, like the driver's record.
        "checked_at": datetime.now(timezone.utc).isoformat(),
    }
    if snapshot is not None:
        record["inflight_count"] = int(snapshot.get("inflight_count") or 0)
        record["scanned"] = snapshot.get("scanned")
    if error is not None:
        record["error"] = error
    return record


def require_no_inflight_installs(
    release: Any,
    *,
    action: str,
    unreadable: Unreadable = "refuse",
) -> dict[str, Any]:
    """Refuse ``action`` while an install step is in flight; return the verdict.

    ``action`` names the transaction for the message (``upgrade`` /
    ``rollback``). ``unreadable`` is what a store that could not be reached
    means: ``refuse`` (manual upgrade and rollback: fail closed, the env lever
    opens it) or ``proceed`` (the automatic rollback: log and go on). Only a
    probe that provably never ran on a control plane with no ready Pod is
    "unreadable" (``StoreUnreachable``). A live control plane kubectl could not
    ask (``KubectlFailure``: exec refused on a ready Pod, or the Pod list
    failed) refuses in every mode, automatic included, and the consent may
    override it. Evidence the probe did return -- an in-flight install, an
    unbounded scan, a non-zero exit, no JSON, its own ``probe_error``, a hang --
    is binding in every mode. The operator's explicit consent thus covers the
    listed in-flight set, the unreachable store and the kubectl-level failure,
    never an evidence defect: a check that ran and could not vouch for anything
    has a cause to fix, not a flag to pass.

    The verdict is ``{"checked", "verdict": clear|unchecked|overridden,
    "reason", "steps", "checked_at", ...}``; the caller persists it (the upgrade
    and the rollback with their first checkpoint, the release driver in its
    rollback record).
    """

    allowed = inflight_installs_allowed()
    try:
        snapshot = inflight_install_snapshot(release)
    except StoreUnreachable as exc:
        if allowed or unreadable == "proceed":
            reason = (
                ALLOW_INFLIGHT_INSTALLS_FLAG
                if allowed
                else "automatic rollback: no Running control-plane Pod could run "
                "the probe"
            )
            narrate_step(
                "inflight-installs-unchecked",
                action=action,
                reason=reason,
                error=f"{type(exc).__name__}: {exc}",
            )
            return _verdict(
                "unchecked", checked=False, reason=reason, steps=[], error=str(exc)
            )
        raise InflightInstallsRefused(
            f"{action} refused: {_CHECK} could not read the control-plane store "
            f"({type(exc).__name__}: {exc}). A {_operations()} step already "
            "handed to a node agent would be submitted a second time by the "
            "control plane this transaction puts in place; retry when the "
            f"control plane answers, or {_lever(action)} without the check"
        ) from exc
    except KubectlFailure as exc:
        # A live control plane the engine could not ask. Not the store's fault
        # and not the probe's; the consent may open it because the operator can
        # verify a transport / RBAC fault by hand -- and the log then says what
        # was skipped in those words, not as "unreachable".
        if allowed:
            reason = f"{exc.short}, consent given"
            narrate_step(
                "inflight-installs-unchecked",
                action=action,
                reason=reason,
                error=f"{type(exc).__name__}: {exc}",
            )
            return _verdict(
                "unchecked", checked=False, reason=reason, steps=[], error=str(exc)
            )
        raise InflightInstallsRefused(
            f"{action} refused: {exc}. The control plane is live and could not be "
            f"asked, so a {_operations()} step already handed to a node agent "
            "cannot be ruled out; fix kubectl's access to the control-plane Pods "
            "(the API server, the kubeconfig / aws eks get-token, the deploy "
            "host's pods/exec RBAC, an image with sh) and retry, or "
            f"{_lever(action)} without the check"
        ) from exc
    except ReleaseError as exc:
        # The probe ran and the answer cannot be read as "clear": refused in
        # every mode, automatic rollback and consent included. The message
        # points at the cause; the flag is named only to say it does not apply.
        raise InflightInstallsRefused(
            f"{action} refused: {exc}. An install past what the check could see "
            f"would be submitted a second time, and {ALLOW_INFLIGHT_INSTALLS_FLAG} "
            "does not cover a check that could not be made: fix the cause named "
            "above (drain the backlog, restore the control plane's store access, "
            "or fix the probe), then retry"
        ) from exc
    count = int(snapshot.get("inflight_count") or 0)
    if not count:
        return _verdict("clear", checked=True, reason=None, steps=[], snapshot=snapshot)
    described = [describe_inflight_step(step) for step in snapshot["inflight"]]
    steps = "; ".join(described)
    if allowed:
        narrate_step(
            "inflight-installs-overridden",
            action=action,
            override=ALLOW_INFLIGHT_INSTALLS_FLAG,
            count=count,
            steps=steps,
        )
        return _verdict(
            "overridden",
            checked=True,
            reason=ALLOW_INFLIGHT_INSTALLS_FLAG,
            steps=described,
            snapshot=snapshot,
        )
    raise InflightInstallsRefused(
        f"{action} refused: {count} {_operations()} step(s) in flight, and the "
        "control plane this transaction puts in place would submit each install "
        f"a second time on a node still running it: {steps}. Wait for them to "
        f"finish (a driver or firmware install takes up to 30 minutes), or "
        f"{_lever(action)}"
    )

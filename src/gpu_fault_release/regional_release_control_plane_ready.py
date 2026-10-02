"""A registry publish that follows a CPU control-plane roll, made safe.

``rotate-token`` ends by restarting the three CPU Deployments that load the
registry Secret and then publishing the final registry revision through an
``exec`` into an api-ha Pod. Twice in a row that publish failed within seconds
of the roll completing and succeeded unchanged on the rerun minutes later. The
rollout barrier (:func:`wait_deployment_rollout_with`) is satisfied by the
Deployment's own counters, which stop counting a Pod the moment its ReplicaSet
scales it down; the Pod itself keeps ``status.phase=Running`` with a
``deletionTimestamp`` until its grace period ends, and a Pod that *is* new has
a Ready condition up to one probe period old. The publish path
(:func:`exec_cpu_ingress_command`) resolves "any Running ingress Pod" -- or
reuses a name memoised before the roll -- and runs the converge probe exactly
once, with its output marked sensitive, so every failure arrives as one
redacted ``ReleaseError`` whether the exec target vanished, the container was
reaped mid-exec, or the API inside the Pod answered 503 to the first request
after its restart.

This module makes that publish fail closed on real refusals and shrug off the
roll it just caused:

* :func:`select_current_ingress_pod` picks the exec target from the ingress
  Deployment's *current* ReplicaSet only (the ``pod-template-hash`` of the
  revision the Deployment points at), Ready and not Terminating, and memoises
  it so the publish uses exactly that Pod.
* :func:`wait_control_plane_ready` waits, bounded, until every Pod of every
  consumer's current ReplicaSet is Ready and -- where the Deployment's
  readiness probe is the ``/healthz`` document -- each one's registry runtime
  reports ``ready`` on a single generation.
* :func:`publish_after_control_plane_roll` wraps the publish in a bounded retry
  for the transient classes only (exec target gone, container killed in
  flight, control plane unavailable). A generation conflict, an identity
  refusal or a convergence window run out is raised at once: those say the
  registry disagrees with the caller, and repeating the publish would only
  repeat the disagreement.

Only an idempotent publish may go through the retry. ``publish_current_revision``
is one: its content is the bootstrap Secret and it reads the generation it
expects immediately before the POST, so a repeat after a publish that did land
mints an identical revision instead of a conflicting one.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from gpu_fault.admin.command_log import FAILURE_EXIT_CODE_ATTRIBUTE
from gpu_fault.admin.diagnostics import diagnostic_text
from gpu_fault_release import regional_deployment_inventory as inventory
from gpu_fault_release.regional_release_config import ReleaseError
from gpu_fault_release.regional_release_probes import probe_source
from gpu_fault_release.regional_release_runtime_identity import (
    CPU_INGRESS_POD_ATTRIBUTE,
    forget_cpu_ingress_pod,
)
from gpu_fault_release.regional_resource_probe import ResourceRef, probe_resource

CURRENT_REVISION_ANNOTATION = "deployment.kubernetes.io/revision"
POD_TEMPLATE_HASH_LABEL = "pod-template-hash"
HEALTHZ_PATH = "/healthz"
CONTROL_PLANE_READY_TIMEOUT_SECONDS = 120.0
CONTROL_PLANE_READY_POLL_SECONDS = 2.0
REGISTRY_HEALTHZ_TIMEOUT_SECONDS = 30.0
PUBLISH_RETRY_ATTEMPTS = 3
PUBLISH_RETRY_DELAY_SECONDS = 5.0
# ``kubectl exec`` reports a container reaped under it with the signal's exit
# code: 137 (SIGKILL) and 143 (SIGTERM). Read from the child's exit status,
# never from its output, which a sensitive publish does not expose.
KILLED_IN_FLIGHT_EXIT_CODES = frozenset({137, 143})
# The codes ``diagnostic_text`` keeps when it redacts a sensitive failure, as
# far as they name a control plane that is not answering rather than one that
# refused. ``NotFound`` is the exec target itself.
EXEC_TARGET_GONE_CODES = frozenset({"NotFound"})
CONTROL_PLANE_UNAVAILABLE_CODES = frozenset(
    {
        "ServiceUnavailable",
        "Throttling",
        "ThrottlingException",
        "TooManyRequestsException",
        "Unable to connect",
        "RequestTimeout",
    }
)
FAILURE_EXEC_KILLED = "exec-killed"
FAILURE_EXEC_TARGET_GONE = "exec-target-gone"
FAILURE_CONTROL_PLANE_UNAVAILABLE = "control-plane-unavailable"
FAILURE_REFUSED = "refused"
TRANSIENT_FAILURE_CLASSES = frozenset(
    {
        FAILURE_EXEC_KILLED,
        FAILURE_EXEC_TARGET_GONE,
        FAILURE_CONTROL_PLANE_UNAVAILABLE,
    }
)


@dataclass(frozen=True)
class ControlPlanePod:
    name: str
    ready: bool
    terminating: bool

    @property
    def serving(self) -> bool:
        return self.ready and not self.terminating


@dataclass(frozen=True)
class CurrentReplicaSet:
    """The Pods a Deployment currently points at, and how many it wants."""

    deployment: str
    revision: str
    pod_template_hash: str
    replicas: int
    healthz_port: int | None
    pods: tuple[ControlPlanePod, ...]

    @property
    def serving_pods(self) -> tuple[ControlPlanePod, ...]:
        return tuple(pod for pod in self.pods if pod.serving)

    @property
    def complete(self) -> bool:
        return (
            self.replicas > 0
            and len(self.serving_pods) == self.replicas
            and not any(pod.terminating for pod in self.pods)
        )

    def evidence(self) -> dict[str, Any]:
        return {
            "revision": self.revision,
            "pod_template_hash": self.pod_template_hash,
            "replicas": self.replicas,
            "ready_pods": [pod.name for pod in self.serving_pods],
            "pending_pods": [pod.name for pod in self.pods if not pod.serving],
        }


def _selector(deployment: dict[str, Any]) -> str:
    labels = (deployment.get("spec") or {}).get("selector", {}).get("matchLabels", {})
    if not isinstance(labels, dict) or not labels:
        raise ReleaseError("Deployment selector has no matchLabels")
    return ",".join(f"{key}={value}" for key, value in sorted(labels.items()))


def _pod_state(item: dict[str, Any]) -> ControlPlanePod:
    metadata = item.get("metadata") or {}
    ready = False
    for condition in (item.get("status") or {}).get("conditions") or []:
        if isinstance(condition, dict) and condition.get("type") == "Ready":
            ready = str(condition.get("status")) == "True"
    return ControlPlanePod(
        name=str(metadata.get("name") or ""),
        ready=ready,
        terminating=bool(metadata.get("deletionTimestamp")),
    )


def _healthz_port(deployment: dict[str, Any]) -> int | None:
    """The port of an HTTP ``/healthz`` readiness probe, or ``None``.

    Only a Deployment whose kubelet readiness already *is* the health document
    gets the in-Pod registry check; anything else is judged on Ready alone.
    """

    template = ((deployment.get("spec") or {}).get("template") or {}).get("spec") or {}
    for container in template.get("containers") or []:
        probe = (container.get("readinessProbe") or {}).get("httpGet") or {}
        path = str(probe.get("path") or "")
        port = probe.get("port")
        if path.startswith(HEALTHZ_PATH) and isinstance(port, int) and port > 0:
            return port
    return None


def current_replicaset(release: Any, deployment_name: str) -> CurrentReplicaSet:
    """Resolve the ReplicaSet the Deployment's current revision points at.

    ``kubectl rollout restart`` bumps ``deployment.kubernetes.io/revision`` on
    the Deployment and stamps the same value on the new ReplicaSet; the old
    ReplicaSet keeps its own. Listing Pods by the Deployment selector alone
    returns both generations for as long as the old Pods take to terminate, so
    the Pods are listed by the current ReplicaSet's ``pod-template-hash`` and
    nothing from the outgoing generation can be selected. Exactly one
    ReplicaSet must carry the current revision; anything else is a Deployment
    in a state this code has not reasoned about, and it fails closed.
    """

    namespace = release.config.namespace
    deployment = release._get_json(
        release._cpu("-n", namespace, "get", "deployment", deployment_name)
    )
    annotations = (deployment.get("metadata") or {}).get("annotations") or {}
    revision = str(annotations.get(CURRENT_REVISION_ANNOTATION) or "")
    if not revision:
        raise ReleaseError(
            f"control plane Deployment {deployment_name} has no current revision"
        )
    selector = _selector(deployment)
    replicasets = release._get_json(
        release._cpu("-n", namespace, "get", "replicaset", "-l", selector)
    )
    current = [
        item
        for item in replicasets.get("items") or []
        if isinstance(item, dict)
        and str(
            ((item.get("metadata") or {}).get("annotations") or {}).get(
                CURRENT_REVISION_ANNOTATION
            )
            or ""
        )
        == revision
    ]
    if len(current) != 1:
        raise ReleaseError(
            f"control plane Deployment {deployment_name} has {len(current)} "
            f"ReplicaSets at revision {revision}; expected exactly one"
        )
    labels = (current[0].get("metadata") or {}).get("labels") or {}
    pod_template_hash = str(labels.get(POD_TEMPLATE_HASH_LABEL) or "")
    if not pod_template_hash:
        raise ReleaseError(
            f"control plane Deployment {deployment_name} ReplicaSet at revision "
            f"{revision} has no {POD_TEMPLATE_HASH_LABEL} label"
        )
    pods = release._get_json(
        release._cpu(
            "-n",
            namespace,
            "get",
            "pods",
            "-l",
            f"{selector},{POD_TEMPLATE_HASH_LABEL}={pod_template_hash}",
        )
    )
    return CurrentReplicaSet(
        deployment=deployment_name,
        revision=revision,
        pod_template_hash=pod_template_hash,
        replicas=int((deployment.get("spec") or {}).get("replicas") or 0),
        healthz_port=_healthz_port(deployment),
        pods=tuple(
            sorted(
                (
                    _pod_state(item)
                    for item in pods.get("items") or []
                    if isinstance(item, dict)
                ),
                key=lambda pod: pod.name,
            )
        ),
    )


def select_current_ingress_pod(release: Any) -> str:
    """Memoise a Ready, non-Terminating Pod of the ingress's current ReplicaSet.

    Fails closed when none qualifies: the fallback to "any Running Pod" that
    :func:`resolve_cpu_ingress_pod` keeps for bootstrap windows is exactly the
    choice that hands a publish to a Pod the rollout is reaping.
    """

    current = current_replicaset(release, inventory.CPU_INGRESS_DEPLOYMENT)
    serving = current.serving_pods
    if not serving:
        raise ReleaseError(
            f"no Ready CPU ingress Pod in the current ReplicaSet of "
            f"{inventory.CPU_INGRESS_DEPLOYMENT} (revision {current.revision}, "
            f"{POD_TEMPLATE_HASH_LABEL}={current.pod_template_hash})"
        )
    name = serving[0].name
    setattr(release, CPU_INGRESS_POD_ATTRIBUTE, name)
    return name


def registry_runtime_health(release: Any, pod: str, port: int) -> dict[str, Any]:
    """Read ``pod``'s own ``/healthz?verbose=1`` registry fields through an exec."""

    output = release.runner.run(
        release._cpu(
            "-n",
            release.config.namespace,
            "exec",
            "-i",
            pod,
            "--",
            "python3",
            "-c",
            probe_source("control_plane_registry_healthz"),
        ),
        input_text=json.dumps({"port": port}, separators=(",", ":")),
        capture=True,
        timeout_seconds=REGISTRY_HEALTHZ_TIMEOUT_SECONDS,
    )
    try:
        value = json.loads(output)
    except (TypeError, json.JSONDecodeError):
        raise ReleaseError(f"{pod} health probe returned invalid JSON") from None
    if not isinstance(value, dict):
        raise ReleaseError(f"{pod} health probe returned a non-object")
    return value


def _registry_pending(
    release: Any,
    current: CurrentReplicaSet,
    generations: dict[str, int],
) -> list[str]:
    """Why ``current``'s Pods are not yet serving the registry, if they are not."""

    pending: list[str] = []
    if current.healthz_port is None:
        return pending
    for pod in current.serving_pods:
        label = f"{current.deployment}/{pod.name}"
        try:
            health = registry_runtime_health(release, pod.name, current.healthz_port)
        except ReleaseError as exc:
            # A Pod that cannot be exec'd into right now is "not yet", not a
            # verdict: the roll may still be settling. The deadline bounds it.
            pending.append(f"{label}: health probe failed: {exc}")
            continue
        registry = health.get("regional_registry")
        if not isinstance(registry, dict):
            if health.get("http_status") != 200:
                pending.append(f"{label}: not ready (HTTP {health.get('http_status')})")
            continue
        generation = registry.get("generation")
        if registry.get("ready") is not True:
            pending.append(
                f"{label}: registry runtime not ready (generation={generation}, "
                f"target={registry.get('target_generation')}, "
                f"error={registry.get('error')})"
            )
        if isinstance(generation, int) and not isinstance(generation, bool):
            generations[label] = generation
    return pending


def wait_control_plane_ready(
    release: Any,
    deployments: Sequence[str] = inventory.CPU_RUNTIME_DEPLOYMENTS,
    *,
    timeout_seconds: float = CONTROL_PLANE_READY_TIMEOUT_SECONDS,
    poll_seconds: float = CONTROL_PLANE_READY_POLL_SECONDS,
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    """Wait until every consumer's current ReplicaSet serves the registry.

    Complete means, for each Deployment: exactly ``spec.replicas`` Pods of the
    current ReplicaSet are Ready, none of them is Terminating, and -- for the
    Deployments whose readiness probe is the health document -- every one of
    those Pods reports ``regional_registry.ready`` on the same generation as
    every other probed Pod. Runs out of time with a ``ReleaseError`` naming
    what was still pending; never returns a partial answer.
    """

    if release.runner.dry_run:
        return {"dry_run": True, "deployments": list(deployments)}
    started = monotonic()
    deadline = started + timeout_seconds
    while True:
        pending: list[str] = []
        generations: dict[str, int] = {}
        evidence: dict[str, Any] = {}
        for name in deployments:
            current = current_replicaset(release, name)
            evidence[name] = current.evidence()
            if not current.complete:
                pending.append(
                    f"{name}: {len(current.serving_pods)}/{current.replicas} Pods "
                    f"of revision {current.revision} Ready and not Terminating"
                )
                continue
            pending.extend(_registry_pending(release, current, generations))
        if len(set(generations.values())) > 1:
            pending.append(
                "registry generations differ across the control plane: "
                + ", ".join(f"{k}={v}" for k, v in sorted(generations.items()))
            )
        if not pending:
            return {
                "waited_seconds": round(max(0.0, monotonic() - started), 3),
                "registry_generation": next(iter(generations.values()), None),
                "deployments": evidence,
            }
        now = monotonic()
        if now >= deadline:
            raise ReleaseError(
                f"control plane did not become ready within {int(timeout_seconds)} "
                "seconds after its roll: " + "; ".join(pending)
            )
        sleep(max(0.0, min(poll_seconds, deadline - now)))


def classify_publish_failure(
    release: Any, error: ReleaseError, *, pod: str | None
) -> str:
    """Name the class of a failed publish from what the failure reliably carries.

    The publish's output is sensitive, so its text is redacted down to error
    codes before it reaches here; the child's exit status and the exec
    target's existence are the facts that survive. Anything not provably
    transient is ``refused`` and never retried: a generation conflict, an
    identity refusal, a convergence window that ran out, a probe that crashed.
    """

    if getattr(error, FAILURE_EXIT_CODE_ATTRIBUTE, None) in KILLED_IN_FLIGHT_EXIT_CODES:
        return FAILURE_EXEC_KILLED
    message = str(error)

    def carries(codes: frozenset[str]) -> bool:
        return any(
            re.search(rf"(?<![A-Za-z]){re.escape(code)}(?![A-Za-z])", message)
            for code in codes
        )

    if carries(EXEC_TARGET_GONE_CODES):
        return FAILURE_EXEC_TARGET_GONE
    if carries(CONTROL_PLANE_UNAVAILABLE_CODES):
        return FAILURE_CONTROL_PLANE_UNAVAILABLE
    if pod:
        try:
            present = probe_resource(
                release.runner,
                release._cpu(),
                ResourceRef("pod", "Pod", pod, release.config.namespace),
            ).exists()
        except ReleaseError:
            # Cannot tell whether the target is gone; do not guess a retry.
            return FAILURE_REFUSED
        if not present:
            return FAILURE_EXEC_TARGET_GONE
    return FAILURE_REFUSED


def publish_after_control_plane_roll(
    release: Any,
    *,
    publish: Callable[[], dict[str, Any]],
    record: Callable[[list[dict[str, Any]]], None] | None = None,
    deployments: Sequence[str] = inventory.CPU_RUNTIME_DEPLOYMENTS,
    timeout_seconds: float | None = None,
    attempts: int | None = None,
    delay_seconds: float | None = None,
    sleep: Callable[[float], None] = time.sleep,
    now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
) -> dict[str, Any]:
    """Wait for the rolled control plane, then run ``publish`` with bounded retry.

    ``publish`` must be idempotent (see the module docstring). Each attempt
    first re-selects the exec target from the ingress's current ReplicaSet,
    so a Pod that went away between attempts is never tried twice. ``record``
    receives the attempt history after every attempt, so a journal can keep it
    even when the last attempt raises. Returns ``{"result", "control_plane_ready",
    "publish_attempts"}``; raises the publish's own error, annotated with the
    attempt it failed on and its class. The bounds default to the module
    constants, read at call time so one place sets the product's numbers.
    """

    if timeout_seconds is None:
        timeout_seconds = CONTROL_PLANE_READY_TIMEOUT_SECONDS
    if attempts is None:
        attempts = PUBLISH_RETRY_ATTEMPTS
    if delay_seconds is None:
        delay_seconds = PUBLISH_RETRY_DELAY_SECONDS
    if attempts < 1:
        raise ReleaseError("a publish needs at least one attempt")
    readiness = wait_control_plane_ready(
        release, deployments, timeout_seconds=timeout_seconds, sleep=sleep
    )
    history: list[dict[str, Any]] = []
    attempt = 0
    while True:
        attempt += 1
        pod = select_current_ingress_pod(release)
        entry: dict[str, Any] = {
            "attempt": attempt,
            "pod": pod,
            "started_at": now().isoformat(),
        }
        try:
            result = publish()
        except ReleaseError as exc:
            forget_cpu_ingress_pod(release)
            failure_class = classify_publish_failure(release, exc, pod=pod)
            entry.update(
                {
                    "outcome": "failed",
                    "failure_class": failure_class,
                    "error": diagnostic_text(str(exc)),
                }
            )
            history.append(entry)
            if record is not None:
                record(history)
            if failure_class not in TRANSIENT_FAILURE_CLASSES or attempt >= attempts:
                exc.add_note(
                    f"registry publish failed on attempt {attempt}/{attempts} "
                    f"({failure_class}); exec target {pod}"
                )
                raise
            sleep(delay_seconds)
            continue
        entry["outcome"] = "published"
        history.append(entry)
        if record is not None:
            record(history)
        return {
            "result": result,
            "control_plane_ready": readiness,
            "publish_attempts": history,
        }

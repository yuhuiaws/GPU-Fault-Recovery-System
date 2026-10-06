"""Validate complete Pod readiness without treating missing data as health.

Two selections share one validation of the inventory: ``ready_pod_records``
(Running, every container Ready) is what a healthy component looks like and
is the default everywhere; ``running_pod_records`` (Running, every container
started, readiness ignored) exists for the one shape a runner must probe while
the component is *deliberately* unhealthy. The executor's readiness probe is an
authenticated POST to the control plane (``cluster_executor.readiness_probe``),
so an executor Pod whose cluster a case cuts off from the control plane flips
NotReady by design -- Running, zero restarts, Ready again after the unblock --
and a probe that insists on Ready finds nothing to exec into for the whole
block (live, ISO-006: ``no Ready gpu Pod for app=gpu-fault-cluster-executor``).
Readiness stays control-plane-bound: it is the product's own signal that the
executor can claim, and the runner reads it as such from the healthy cluster.
"""

from __future__ import annotations

import json
from typing import Any

from scripts.e2e.regional.regional_commands import RegionalFixtureError


def ready_pod_records(value: Any) -> list[dict[str, Any]]:
    """Running Pods whose every container reports Ready (the default selection)."""

    return _pod_records(value, require_ready=True)


def running_pod_records(value: Any) -> list[dict[str, Any]]:
    """Running Pods whose every container is started; readiness is not consulted.

    Only for probing a component the case itself has made NotReady; a Pod that
    is Pending, terminating, or has a container not (yet) running is still
    excluded, because an exec into it answers nothing about the cluster.
    """

    return _pod_records(value, require_ready=False)


def pod_inventory(fixture: Any, plane: str, app: str) -> Any:
    """``kubectl get pod -l app=<app> -o json`` through ``fixture``'s kubectl."""

    return json.loads(
        fixture.kubectl(plane, "get", "pod", "-l", f"app={app}", "-o", "json")
    )


def running_pods(fixture: Any, plane: str, app: str) -> list[dict[str, Any]]:
    """Running Pods of ``app`` on ``plane`` whether or not they are Ready.

    For a component the case itself has cut off from the control plane: the
    executor's readiness is an authenticated POST to the control plane, so its
    Pods are NotReady for exactly as long as the cut holds. Every other probe
    keeps ``RegionalLiveFixture.ready_pods``.
    """

    return running_pod_records(pod_inventory(fixture, plane, app))


def running_pod(fixture: Any, plane: str, app: str) -> str:
    pods = running_pods(fixture, plane, app)
    if not pods:
        raise RegionalFixtureError(f"no Running {plane} Pod for app={app}")
    return str(pods[0]["name"])


def _pod_records(value: Any, *, require_ready: bool) -> list[dict[str, Any]]:
    if not isinstance(value, dict) or not isinstance(value.get("items"), list):
        raise RegionalFixtureError("Pod inventory is missing or malformed")
    result: list[dict[str, Any]] = []
    identities: set[tuple[str, str]] = set()
    for item in value["items"]:
        if not isinstance(item, dict):
            raise RegionalFixtureError("Pod inventory entry is malformed")
        metadata = item.get("metadata")
        spec = item.get("spec")
        status = item.get("status")
        if not all(isinstance(part, dict) for part in (metadata, spec, status)):
            raise RegionalFixtureError("Pod identity or state is missing")
        name, uid = metadata.get("name"), metadata.get("uid")
        if not isinstance(name, str) or not name or not isinstance(uid, str) or not uid:
            raise RegionalFixtureError("Pod identity is missing")
        if (name, uid) in identities:
            raise RegionalFixtureError("Pod inventory repeats an identity")
        identities.add((name, uid))
        if metadata.get("deletionTimestamp") or status.get("phase") != "Running":
            continue
        conditions = status.get("conditions")
        containers = spec.get("containers")
        statuses = status.get("containerStatuses")
        if not all(
            isinstance(part, list) for part in (conditions, containers, statuses)
        ):
            continue
        ready = [
            condition.get("status")
            for condition in conditions
            if isinstance(condition, dict) and condition.get("type") == "Ready"
        ]
        if require_ready and ready != ["True"]:
            continue
        names = [
            container.get("name")
            for container in containers
            if isinstance(container, dict)
        ]
        reported = [
            container.get("name")
            for container in statuses
            if isinstance(container, dict)
            and (
                container.get("ready") is True
                if require_ready
                else _container_running(container)
            )
        ]
        if (
            not names
            or len(names) != len(containers)
            or any(not isinstance(name, str) or not name for name in names)
            or len(set(names)) != len(names)
            or len(reported) != len(statuses)
            or len(reported) != len(names)
            or set(reported) != set(names)
        ):
            continue
        result.append({"name": name, "uid": uid, "node": spec.get("nodeName")})
    return sorted(result, key=lambda item: str(item["name"]))


def _container_running(status: dict[str, Any]) -> bool:
    """A container status whose current state is ``running``.

    kubelet reports exactly one of ``running``/``waiting``/``terminated`` under
    ``state``; ``started`` alone is not consulted because it is absent on a
    container without a startup probe on older kubelets.
    """

    state = status.get("state")
    return isinstance(state, dict) and isinstance(state.get("running"), dict)

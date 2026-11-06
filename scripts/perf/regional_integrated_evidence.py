"""Evidence the integrated verdict judges beyond the round's own counts.

Perf plan §8.4 and §9: a formal integrated round passes only when

* the processor queue and the telemetry spool are back at their depth from
  before the round once the load has stopped (``summary.json.queue_drain``,
  ``regional_capacity_timeseries.queue_drain``: the pre-run depth plus 4, or 1
  when that depth was at most 1, within the drain probe's 900 s);
* the round ran on one site configuration, and the record names it: the global
  AdminConfig digest every role Deployment carries and the configuration digest
  of each of the three roles (``config_identity``, in ``run.json`` and
  ``summary.json``). A Running Pod whose role digest is not its Deployment's
  comes from an older ReplicaSet: a configuration roll was still in flight;
* every request's final response is 2xx (§13, item 8): an ingress 202 only
  means the request was enqueued, and the processor's own answer is what the
  priority latency probe reads per tier (``response_status_counts``), with
  every fault-tier request of the matrix accounted for.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from gpu_fault.processor.models import is_reserved_tier

#: The role Deployments of the control plane and their Pods' ``app`` label.
ROLE_DEPLOYMENTS = {
    "ingress": "gpu-fault-api-ha",
    "worker": "gpu-fault-control-worker",
    "spool": "gpu-fault-telemetry-spool-worker",
}
#: Stamped by ``apply-control-plane-role-split.sh`` at every apply and config
#: roll: the AdminConfig digest on each Deployment's metadata, the role's own
#: configuration digest on its Pod template (so on every Pod it created).
ADMIN_CONFIG_ANNOTATION = "gpu-fault.io/admin-config-sha256"
ROLE_CONFIG_ANNOTATION = "gpu-fault.io/role-config-sha256"


def drain_errors(drain: dict[str, Any] | None) -> list[str]:
    """Why the queue or the spool is not back at its pre-run depth, if so."""

    samples = (drain or {}).get("samples") or []
    if not samples:
        return ["processor queue drain was not observed after the load"]
    last = samples[-1]
    waited = float(last.get("elapsed_seconds") or 0.0)
    errors = []
    for name, depth_key, target_key in (
        ("processor queue", "queue_depth", "target_queue_depth"),
        ("telemetry spool", "telemetry_spool_depth", "target_spool_depth"),
    ):
        depth = last.get(depth_key)
        target = (drain or {}).get(target_key)
        if depth is None:
            # The probe stops on an unreadable spool depth too; only the
            # queue depth is required (a release without a spool family).
            if depth_key == "queue_depth":
                errors.append("processor queue depth was unreadable after the load")
            continue
        if target is not None and float(depth) > float(target):
            errors.append(
                f"{name} did not drain to its pre-run depth: {float(depth):g} > "
                f"{float(target):g} after {waited:.0f} s"
            )
    return errors


def config_identity(control: Callable[..., str]) -> dict[str, Any]:
    """The configuration the round runs on, read from the live role
    Deployments and their Running Pods (read-only)."""

    deployments = json.loads(
        control("get", "deployment", *ROLE_DEPLOYMENTS.values(), "-o", "json")
    ).get("items", [])
    apps = ",".join(ROLE_DEPLOYMENTS.values())
    pods = json.loads(
        control(
            "get",
            "pod",
            "-l",
            f"app in ({apps})",
            "--field-selector=status.phase=Running",
            "-o",
            "json",
        )
    ).get("items", [])
    by_name = {
        str((item.get("metadata") or {}).get("name")): item for item in deployments
    }
    roles: dict[str, Any] = {}
    for role, name in ROLE_DEPLOYMENTS.items():
        item = by_name.get(name) or {}
        template = ((item.get("spec") or {}).get("template") or {}).get("metadata")
        roles[role] = {
            "deployment": name,
            "admin_config_sha256": (
                ((item.get("metadata") or {}).get("annotations") or {}).get(
                    ADMIN_CONFIG_ANNOTATION
                )
            ),
            "role_config_sha256": ((template or {}).get("annotations") or {}).get(
                ROLE_CONFIG_ANNOTATION
            ),
            "running_pods": 0,
            "pods_on_another_role_config": {},
        }
    role_of = {name: role for role, name in ROLE_DEPLOYMENTS.items()}
    for pod in pods:
        metadata = pod.get("metadata") or {}
        role = role_of.get(str((metadata.get("labels") or {}).get("app")))
        if role is None:
            continue
        record = roles[role]
        record["running_pods"] += 1
        digest = (metadata.get("annotations") or {}).get(ROLE_CONFIG_ANNOTATION)
        if digest != record["role_config_sha256"]:
            record["pods_on_another_role_config"][str(metadata.get("name"))] = digest
    admin = {record["admin_config_sha256"] for record in roles.values()}
    return {
        "admin_config_sha256": admin.pop() if len(admin) == 1 else None,
        "role_config_sha256": {
            role: record["role_config_sha256"] for role, record in roles.items()
        },
        "roles": roles,
    }


def config_errors(identity: dict[str, Any] | None) -> list[str]:
    """Why the recorded configuration does not name one configuration."""

    roles = (identity or {}).get("roles")
    if not isinstance(roles, dict) or set(roles) != set(ROLE_DEPLOYMENTS):
        return ["control-plane configuration digests were not recorded"]
    errors = []
    missing = sorted(
        f"{role} {field}"
        for role, record in roles.items()
        for field in ("admin_config_sha256", "role_config_sha256")
        if not record.get(field)
    )
    if missing:
        errors.append(
            "control-plane configuration digests are missing: " + ", ".join(missing)
        )
    elif not (identity or {}).get("admin_config_sha256"):
        errors.append("the role Deployments carry different AdminConfig digests")
    stale = sorted(
        pod
        for record in roles.values()
        for pod in record.get("pods_on_another_role_config") or {}
    )
    if stale:
        errors.append(
            "control-plane Pods run another role configuration than their "
            "Deployment (a roll in flight): " + ", ".join(stale)
        )
    return errors


def final_response_errors(
    latency: dict[str, Any] | None, *, fault_requests: int
) -> list[str]:
    """Why the processor's final responses do not show every request answered
    2xx: a non-2xx (or missing) status on any tier, or fewer fault-tier final
    responses than the matrix's ``fault_requests`` (P0: XID plus SXID) -- a
    probe that read too little proves nothing."""

    if not isinstance(latency, dict) or latency.get("error"):
        detail = (latency or {}).get("error") if isinstance(latency, dict) else None
        return [f"processor final responses were not read: {detail}"]
    counts = latency.get("response_status_counts")
    if not isinstance(counts, dict):
        return ["processor final responses were not recorded"]
    errors = [
        f"processor final response {status} on {count} tier-{tier} requests"
        for tier, statuses in sorted(counts.items(), key=lambda item: int(item[0]))
        for status, count in sorted(statuses.items())
        if not (status.isdigit() and 200 <= int(status) < 300)
    ]
    answered = sum(
        int(count)
        for tier, statuses in counts.items()
        if is_reserved_tier(int(tier))
        for count in statuses.values()
    )
    if answered < fault_requests:
        errors.append(
            f"only {answered} of the {fault_requests} fault-tier requests have a "
            "processor final response"
        )
    return errors

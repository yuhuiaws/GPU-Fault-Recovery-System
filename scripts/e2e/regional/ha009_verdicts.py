"""Pure HA-009 rotation verdicts over the gathered continuity observations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING or __package__:
    from . import run_ha005_rollout_continuity as BASE
    from .ha009_observation import observation_errors, steady_deployments
else:
    import run_ha005_rollout_continuity as BASE
    from ha009_observation import observation_errors, steady_deployments


DEPLOYMENTS = (
    "gpu-fault-api-ha",
    "gpu-fault-control-worker",
    "gpu-fault-telemetry-spool-worker",
)


def role_status(deployments: dict[str, Any]) -> dict[str, str]:
    """Record the explicitly disabled spool role without claiming it was tested."""

    return {
        name: (
            "SKIPPED_NOT_ENABLED"
            if int(deployments[name].get("replicas") or 0) == 0
            else "STEADY"
        )
        for name in DEPLOYMENTS
    }


def enabled_roles(deployments: dict[str, Any]) -> list[str]:
    return [
        name for name, status in role_status(deployments).items() if status == "STEADY"
    ]


def enabled_pods(deployments: dict[str, Any]) -> list[str]:
    return [
        pod
        for name in enabled_roles(deployments)
        for pod, _value in deployments[name]["pods"]
    ]


def deployments_steady(before: dict[str, Any], current: dict[str, Any]) -> list[str]:
    return steady_deployments(before, current, DEPLOYMENTS)


def deployments_rolled(before: dict[str, Any], current: dict[str, Any]) -> bool:
    """Legacy rollout predicate; HA-009 itself requires steady Deployments."""

    for name in enabled_roles(before):
        item = current[name]
        if int(item["generation"]) <= int(before[name]["generation"]):
            return False
        old_uids = {value["uid"] for _pod, value in before[name]["pods"]}
        if not BASE.rollout_complete(item, old_uids):
            return False
    return True


def rotation_errors(
    *,
    versions_after: dict[str, Any],
    current_before: str,
    digest_before: str,
    digest_after: str,
    first_job: dict[str, Any],
    second_job: dict[str, Any],
    deployments_before: dict[str, Any],
    deployments_after: dict[str, Any],
    propagation: dict[str, Any],
    idle_observation: dict[str, Any],
    final_probe: dict[str, Any],
    receipts: dict[str, Any],
    runtime: dict[str, Any],
    digest_before_noop: str,
    digest_after_noop: str,
    before_noop: dict[str, Any],
    after_noop: dict[str, Any],
) -> list[str]:
    errors = []
    if versions_after["stages"].get("AWSCURRENT") == current_before:
        errors.append("AWSCURRENT did not change")
    if versions_after["stages"].get("AWSPREVIOUS") != current_before:
        errors.append("old AWSCURRENT did not become AWSPREVIOUS")
    if digest_after == digest_before:
        errors.append("Kubernetes Aurora Secret digest did not change")
    first_logs = "\n".join(first_job["logs"])
    if "rotated=True" not in first_logs:
        errors.append("first refresh Job did not report a rotation")
    if "restarted=False" not in first_logs:
        errors.append(
            "first refresh Job did not report restarted=False: path A must not roll"
        )
    errors.extend(deployments_steady(deployments_before, deployments_after))
    errors.extend(
        observation_errors(
            set(enabled_pods(deployments_before)), propagation, idle_observation
        )
    )
    accepted_ids = sorted(set(final_probe.get("accepted_request_ids", [])))
    errors.extend(
        BASE.continuity_errors(final_probe, receipts, accepted_ids=accepted_ids)
    )
    if runtime["command"].get("status") != "SUCCEEDED":
        errors.append("synthetic remote command did not succeed")
    notification = runtime["notification"]
    for kind in ("notification", "notification_delivery", "notification_result"):
        if notification.get(kind, {}).get("count") != 1:
            errors.append(f"{kind} count is not one")
    if notification.get("notification_result", {}).get("status") != "SKIPPED":
        errors.append("drill notification was not safely suppressed")
    if "rotated=False restarted=False" not in "\n".join(second_job["logs"]):
        errors.append("second refresh Job was not a NOOP")
    if digest_after_noop != digest_before_noop:
        errors.append("NOOP refresh changed the Kubernetes Secret")
    for name in DEPLOYMENTS:
        if after_noop[name]["generation"] != before_noop[name]["generation"]:
            errors.append(f"NOOP refresh changed {name} generation")
        if after_noop[name]["pods"] != before_noop[name]["pods"]:
            errors.append(f"NOOP refresh rolled {name} Pods")
    return errors

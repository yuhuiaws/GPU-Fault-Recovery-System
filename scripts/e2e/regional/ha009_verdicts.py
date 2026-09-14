#!/usr/bin/env python3
"""HA-009 path-A verdicts: role status and the rotation-error contract.

Split out of ``run_ha009_aurora_credential_rotation.py`` so neither file needs a
new architecture size baseline. These are pure functions over gathered snapshots
plus HA-005's continuity helpers; the runner imports them back under both its
package-relative and script-style import branches.
"""

from __future__ import annotations

if __package__:
    from . import run_ha005_rollout_continuity as BASE
else:  # pragma: no cover - script-style import
    import run_ha005_rollout_continuity as BASE


DEPLOYMENTS = (
    "gpu-fault-api-ha",
    "gpu-fault-control-worker",
    "gpu-fault-telemetry-spool-worker",
)


def role_status(deployments: dict) -> dict[str, str]:
    """``STEADY`` for an enabled role, ``SKIPPED_NOT_ENABLED`` for replicas=0.

    The catalog wants an unconfigured role recorded as skipped, not silently
    passed or failed; the spool-worker ships with replicas=0 on sites without
    telemetry spooling. ``STEADY`` is the path-A expectation: the role's Pods
    are the same before and after the rotation.
    """

    return {
        name: (
            "SKIPPED_NOT_ENABLED"
            if int(deployments[name].get("replicas") or 0) == 0
            else "STEADY"
        )
        for name in DEPLOYMENTS
    }


def enabled_roles(deployments: dict) -> list[str]:
    return [
        name for name, status in role_status(deployments).items() if status == "STEADY"
    ]


def enabled_pods(deployments: dict) -> list[str]:
    return [
        pod
        for name in enabled_roles(deployments)
        for pod, _value in deployments[name]["pods"]
    ]


def deployments_steady(before: dict, current: dict) -> list[str]:
    """Path A: nothing about an enabled role may have moved.

    Same generation (no template patch), the same Pod UIDs (no replacement),
    the same restartCount (no crash into the rotated password), and every
    declared replica Ready. Returns the violations, empty when steady.
    """

    errors = []
    for name in enabled_roles(before):
        item = current[name]
        if item["generation"] != before[name]["generation"]:
            errors.append(f"{name} generation changed: a rotation must not roll it")
        before_pods = dict(before[name]["pods"])
        after_pods = dict(item["pods"])
        if {v["uid"] for v in before_pods.values()} != {
            v["uid"] for v in after_pods.values()
        }:
            errors.append(f"{name} Pod set changed: a rotation must not replace Pods")
        for pod, value in after_pods.items():
            restarts_before = int((before_pods.get(pod) or {}).get("restarts", 0))
            if int(value.get("restarts", 0)) != restarts_before:
                errors.append(f"{name} Pod {pod} restarted during the rotation")
            if not value.get("ready"):
                errors.append(f"{name} Pod {pod} is not Ready after the rotation")
        if int(item.get("ready") or 0) != int(item.get("replicas") or 0):
            errors.append(f"{name} is not fully Ready after the rotation")
    return errors


def deployments_rolled(before: dict, current: dict) -> bool:
    """Every enabled role rolled to a new generation with all old Pods replaced.

    Only meaningful for the refresher's ``--restart-deployments`` compatibility
    mode (Pods that do not mount the Secret); path A asserts the opposite via
    ``deployments_steady``. Uses HA-005's ``rollout_complete`` -- Ready, updated
    and available all equal to replicas, generation observed, old UIDs gone --
    rather than a bare ``ready == replicas``. A replicas=0 role is skipped.
    """

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
    versions_after: dict,
    current_before: str,
    digest_before: str,
    digest_after: str,
    first_job: dict,
    second_job: dict,
    deployments_before: dict,
    deployments_after: dict,
    propagation: dict,
    idle_observation: dict,
    final_probe: dict,
    receipts: dict,
    runtime: dict,
    digest_before_noop: str,
    digest_after_noop: str,
    before_noop: dict,
    after_noop: dict,
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
    # Path A: the Pods that served before the rotation still serve after it.
    errors.extend(deployments_steady(deployments_before, deployments_after))
    for pod, digest in sorted(propagation.get("pods", {}).items()):
        if digest != propagation.get("digest"):
            errors.append(
                f"{pod} projected postgres-url did not catch up with the Secret"
            )
    # H1-5: after max_idle every reconnect used the new password.
    for pod, samples in sorted(idle_observation.get("samples", {}).items()):
        if not samples:
            errors.append(f"{pod} was not observed after the idle window")
            continue
        for index, sample in enumerate(samples):
            if int(sample.get("healthz_status") or 0) != 200:
                errors.append(
                    f"{pod} /healthz returned {sample.get('healthz_status')} after the "
                    f"idle window (sample {index})"
                )
            metrics = sample.get("metrics") or {}
            if "gpu_fault_postgres_pool_connections_errors_total" not in metrics:
                errors.append(
                    f"{pod} /metrics does not export "
                    "gpu_fault_postgres_pool_connections_errors_total"
                )
                break
    for pod, count in sorted(idle_observation.get("auth_failures_in_logs", {}).items()):
        if int(count or 0) != 0:
            errors.append(
                f"{pod} logged {count} password authentication failure(s) after the idle "
                "window: the pool did not pick up the rotated password"
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

"""Control-plane Pod lifecycle across a capacity run.

``runtime_snapshot`` reads every control-plane Pod's phase, readiness and
container restart counters (nothing else: no UID, IP, node or container ID);
``lifecycle`` compares two snapshots -- a Pod replaced, added, restarted, not
Ready at the end, or one whose Ready condition moved during the run (an api-ha
Pod that left Ready had its NLB target deregistered). The integrated runner's
verdict rejects every one of them.
"""

from __future__ import annotations

import json
from collections.abc import Callable

if __package__:
    from .regional_capacity_placement import CONTROL_POD_PREFIXES
else:
    from regional_capacity_placement import CONTROL_POD_PREFIXES


def runtime_snapshot(control: Callable[..., str]) -> dict[str, dict]:
    """Every control-plane Pod's phase, readiness and container restarts,
    redacted to what the lifecycle verdict reads."""

    document = json.loads(control("get", "pod", "-o", "json"))
    snapshot: dict[str, dict] = {}
    for pod in document.get("items") or []:
        metadata = pod.get("metadata") or {}
        name = str(metadata.get("name") or "")
        if not name.startswith(CONTROL_POD_PREFIXES):
            continue
        status = pod.get("status") or {}
        conditions = {item.get("type"): item for item in status.get("conditions") or []}
        ready = (conditions.get("Ready") or {}).get("status") == "True"
        containers = {}
        restart_count = 0
        for item in status.get("containerStatuses") or []:
            container_name = str(item.get("name") or "")
            restarts = int(item.get("restartCount") or 0)
            restart_count += restarts
            running = (item.get("state") or {}).get("running") or {}
            terminated = (item.get("lastState") or {}).get("terminated") or {}
            containers[container_name] = {
                "ready": bool(item.get("ready")),
                "restart_count": restarts,
                "running_started_at": running.get("startedAt"),
                "last_termination": (
                    {
                        "reason": terminated.get("reason"),
                        "exit_code": terminated.get("exitCode"),
                        "started_at": terminated.get("startedAt"),
                        "finished_at": terminated.get("finishedAt"),
                    }
                    if terminated
                    else None
                ),
            }
        snapshot[name] = {
            "role": (metadata.get("labels") or {}).get("app"),
            "phase": status.get("phase"),
            "ready": ready,
            "ready_since": (conditions.get("Ready") or {}).get("lastTransitionTime"),
            "restart_count": restart_count,
            "containers": dict(sorted(containers.items())),
        }
    return dict(sorted(snapshot.items()))


def lifecycle(
    before: dict[str, dict],
    after: dict[str, dict],
) -> dict[str, object]:
    before_names = set(before)
    after_names = set(after)
    restarted_pods: dict[str, int] = {}
    counter_regressions: dict[str, dict[str, int]] = {}
    readiness_flaps: list[str] = []
    for name in sorted(before_names & after_names):
        if before[name].get("ready_since") != after[name].get("ready_since"):
            readiness_flaps.append(name)  # left Ready (api-ha: NLB target out)
        before_count = int(before[name].get("restart_count", 0))
        after_count = int(after[name].get("restart_count", 0))
        if after_count > before_count:
            restarted_pods[name] = after_count - before_count
        elif after_count < before_count:
            counter_regressions[name] = {
                "before": before_count,
                "after": after_count,
            }
    return {
        "restart_count_delta": sum(restarted_pods.values()),
        "restarted_pods": restarted_pods,
        "missing_pods": sorted(before_names - after_names),
        "added_pods": sorted(after_names - before_names),
        "counter_regressions": counter_regressions,
        "not_ready_after": sorted(
            name for name, item in after.items() if not item.get("ready")
        ),
        "readiness_flaps": readiness_flaps,
    }

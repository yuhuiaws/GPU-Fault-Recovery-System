"""Undo COLLECT-004's collector env override through the host probe.

Split from ``run_collector_destructive.py`` (file-size ratchet). The restore
has to survive the reboot the case provokes: the executor restarts the node
seconds after planning RESTART_NODE, so the first exec may lose its Pod and no
Pod can become Ready while the node is down.
"""

from __future__ import annotations

from typing import Any, cast

from scripts.e2e.regional.host_probe_fixture import HostProbeError


def restore_collector_env(collector: Any, run_id: str) -> dict[str, Any]:
    """Undo the collector env override, through a fresh probe Pod if needed.

    The first attempt goes through the Pod the case has been using; after a
    real RESTART_NODE that Pod is Failed and `kubectl exec` refuses it, which
    used to leave the override in place until the on-node deadman timer
    expired. One recreate-and-retry is what the reboot costs. When even the
    recreate cannot reach the node -- the restore raced the reboot and the node
    is down (attempts 3 and 4 failed the whole case here) -- the answer is a
    deferred restore, not an exception: the on-node boot-time oneshot restores
    the env before the collector starts, the caller retries once the node is
    back, and the env digest is checked against the baseline either way.
    """

    def attempt() -> dict[str, Any]:
        return cast(
            dict[str, Any],
            collector.execute("restore-collector-env", "--run-id", run_id, timeout=300),
        )

    try:
        return attempt()
    except HostProbeError as first:
        try:
            collector.recreate()
            return attempt()
        except HostProbeError as exc:
            return {
                "restored": False,
                "deferred": True,
                "reason": f"probe unavailable while the node reboots: {exc}",
                "first_error": str(first),
            }

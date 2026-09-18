"""Undo COLLECT-004's collector env override through the host probe.

Split from ``run_collector_destructive.py`` (file-size ratchet). The restore
has to survive the reboot the case provokes: the executor restarts the node
seconds after planning RESTART_NODE, so the first exec may lose its Pod and no
Pod can become Ready while the node is down.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

from scripts.e2e.regional.host_probe_fixture import (
    HostProbeMissingResponseError,
    HostProbeTransportError,
)


def restore_collector_env(
    collector: Any,
    run_id: str,
    *,
    owner_nonce: str,
    reboot_transition: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """Attempt restoration once, deferring only an authorized reboot wait.

    The callback must freshly confirm a bounded, identity-bound reboot wait, not
    node health. The caller waits for the original node's new boot before
    recreating the probe and restoring with deferral disabled. Explicit host
    rejection is always a hard failure; deferral never proves cleanup.
    """

    try:
        return cast(
            dict[str, Any],
            collector.execute(
                "restore-collector-env",
                "--run-id",
                run_id,
                "--owner-nonce",
                owner_nonce,
                timeout=300,
            ),
        )
    except (HostProbeTransportError, HostProbeMissingResponseError) as exc:
        if reboot_transition is None or reboot_transition() is not True:
            raise
        return {
            "run_id": run_id,
            "restored": False,
            "cleanup_verified": False,
            "deferred": True,
            "reason": "probe response unavailable during authorized reboot wait",
            "error_type": (
                "HostProbeMissingResponseError"
                if isinstance(exc, HostProbeMissingResponseError)
                else "HostProbeTransportError"
            ),
        }

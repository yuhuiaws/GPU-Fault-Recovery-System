"""Read-back of just-suspended workloads for the one-tick STOP_WORKLOADS.

Kept outside ``KubernetesWorkloadOperationsMixin`` on purpose: the mixin is
at the architecture size ceiling, and this is a pure function over the
workload list and a reader callable, which is also how it is tested.
"""

from __future__ import annotations

from typing import Any, Callable

WorkloadActiveReader = Callable[[str, str, str], bool | None]


def stop_state_after_mutation(
    workloads: list[tuple[str, str, str, str, Any]],
    workload_active: WorkloadActiveReader,
) -> tuple[list[str], list[str]]:
    """Split the just-suspended workloads into still-active and unknown.

    The objects in ``workloads`` were read before the patch, so each one is
    read back fresh through ``workload_active(namespace, kind, name)``. A
    workload that vanished since (404/410) is stopped. Any other read error
    keeps the workload in ``unknown`` so the caller falls back to WAITING,
    which is what every first call did before this fast path existed,
    rather than failing a step whose patch has already landed.
    """

    active: list[str] = []
    unknown: list[str] = []
    for namespace, kind, name, workload_id, _ in workloads:
        try:
            active_state = workload_active(namespace, kind, name)
        except Exception as exc:  # noqa: BLE001 - any read error is "unknown"
            if getattr(exc, "status", None) in {404, 410}:
                continue
            unknown.append(workload_id)
            continue
        if active_state is True:
            active.append(workload_id)
        elif active_state is None:
            unknown.append(workload_id)
    return active, unknown

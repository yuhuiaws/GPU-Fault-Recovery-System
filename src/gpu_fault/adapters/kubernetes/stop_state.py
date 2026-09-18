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


TERMINAL_WORKLOAD_CONDITIONS = frozenset({"Failed", "Succeeded", "Completed"})


def custom_workload_active(status: Any) -> bool | None:
    """Whether a PyTorchJob/JobSet still runs, judged from its ``status``.

    A true terminal condition wins over the replica counts. The training
    operator stops reconciling a job once it is Failed, so ``replicaStatuses``
    keeps the counts of the moment it failed however many Pods are deleted
    afterwards and whatever ``suspend`` says (live, training-operator
    v1-855e096: a Failed PyTorchJob kept ``Worker.active=1`` after its worker
    Pod was force-deleted and ``suspend`` was patched). Those counts carry no
    information; the passive STOP of a dead attempt waited on them forever.
    Suspended and Running jobs are still reconciled, so their counts are read
    first and a Suspended job mid-teardown stays active until they reach zero.
    ``None`` means the status cannot answer at all.
    """

    if not isinstance(status, dict):
        return None
    conditions = [
        condition
        for condition in status.get("conditions", [])
        if isinstance(condition, dict)
        and str(condition.get("status", "")).lower() == "true"
    ]
    if any(item.get("type") in TERMINAL_WORKLOAD_CONDITIONS for item in conditions):
        return False
    counts = []
    for value in status.get("replicaStatuses", {}).values():
        if isinstance(value, dict):
            counts.append(value.get("active"))
    for value in status.get("replicatedJobs", []):
        if isinstance(value, dict):
            counts.append(value.get("active"))
    known = [value for value in counts if value is not None]
    if known and any(int(value) > 0 for value in known):
        return True
    for condition in conditions:
        if condition.get("type") == "Suspended":
            return False
        if condition.get("type") == "Running":
            return True
    return False if known else None

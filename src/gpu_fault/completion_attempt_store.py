"""What the Completion Watcher persists about a running attempt, and no more.

The watcher keeps one record per active attempt in the ``<name>-active``
ConfigMap so that a restart still knows which attempts it was watching. The
record used to be the whole published observation: every container with its GPU
UUIDs, container id, cgroup path and log snapshot, ~1 KB per rank. That cost the
watcher twice (F6):

* **Capacity.** A ConfigMap is capped at 1 MiB per object, so a few hundred
  managed Pods filled it. Every further write then raised, including the
  write-ahead copy of a terminal event -- routine state starving the one thing
  the outbox exists to make durable.
* **Write amplification.** The fields that change most often (``restart_count``,
  a re-read GPU UUID list, a fresh log snapshot) are exactly the ones a restart
  does not need, and every change rewrote the whole document.

So only the identity of the attempt (its spec) and the identity of each Pod are
persisted. On restore the observation is rebuilt from these records and the very
next reconcile pass replaces each container with the live Pod's full
observation, keyed by ``pod_uid/container_name``; a Pod that is gone takes the
existing missing-attempt grace path (F7) instead of being invented here.
"""

from __future__ import annotations

from typing import Any

#: Spec fields of an ``AttemptObservation``: everything except ``containers``.
#: Restated as data rather than derived from the model so that a field added to
#: the model without a thought about persistence does not silently start (or
#: stop) being written to a bounded ConfigMap.
SPEC_FIELDS: tuple[str, ...] = (
    "cluster_id",
    "environment",
    "job_id",
    "attempt_id",
    "workload_phase",
    "observed_at",
    "started_at",
    "expected_critical_ranks",
    "workload_ids",
    "cleanup_timeout_seconds",
    "checkpoint_manifest_ref",
    "termination_initiator_incident_id",
    "runtime_profile_version",
    "restart_budget",
)

#: Per-Pod fields kept, as (record key, container field). ``uid`` and
#: ``container`` are the observation key the next pass replaces; ``rank`` and
#: ``critical`` are what a terminal decision counts (an attempt is complete when
#: it has ``expected_critical_ranks`` critical ranks); ``node``/``instance`` are
#: what a fault has to be attributed to before any Pod is listed again; the exit
#: fields keep a rank that already finished from reading as still running.
POD_FIELDS: tuple[tuple[str, str], ...] = (
    ("uid", "pod_uid"),
    ("name", "pod_name"),
    ("container", "container_name"),
    ("role", "role"),
    ("rank", "rank"),
    ("critical", "critical"),
    ("node", "node_id"),
    ("instance", "instance_id"),
    ("gpus", "gpu_count"),
    ("terminated", "terminated"),
    ("exit_code", "exit_code"),
    ("finished_at", "finished_at"),
)

#: Key under which the Pod list is stored. Named so that a legacy record (which
#: has ``containers``) is told apart from a compact one without a version field.
PODS_KEY = "pods"


def attempt_state_record(payload: dict[str, Any]) -> dict[str, Any]:
    """The compact record to persist for a published observation payload.

    ``payload`` is an ``AttemptObservation.model_dump(mode="json")``, so every
    value here is already JSON. Absent and default-ish values are dropped: this
    document is bounded in bytes, and ``attempt_observation_payload`` puts the
    model defaults back.
    """

    record: dict[str, Any] = {
        field: payload[field] for field in SPEC_FIELDS if payload.get(field) is not None
    }
    pods = []
    for container in payload.get("containers") or []:
        pod = {
            key: container[field]
            for key, field in POD_FIELDS
            if container.get(field) is not None
        }
        pods.append(pod)
    record[PODS_KEY] = pods
    return record


def attempt_observation_payload(record: dict[str, Any]) -> dict[str, Any]:
    """Rebuild an ``AttemptObservation`` payload from a persisted record.

    A record written by an earlier release carries ``containers`` verbatim and is
    returned as it is, so an upgrade restores the attempts that were running
    when the old process stopped.
    """

    if PODS_KEY not in record:
        return record
    payload = {key: value for key, value in record.items() if key != PODS_KEY}
    containers = []
    for pod in record.get(PODS_KEY) or []:
        if not isinstance(pod, dict):
            continue
        container = {field: pod[key] for key, field in POD_FIELDS if key in pod}
        containers.append(container)
    payload["containers"] = containers
    return payload

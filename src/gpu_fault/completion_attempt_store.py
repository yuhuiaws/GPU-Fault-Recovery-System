"""What the Completion Watcher persists about a running attempt, and no more.

The watcher keeps one record per active attempt in the ``<name>-active``
ConfigMap so that a restart still knows which attempts it was watching. The
record used to be the whole published observation: every container with its log
snapshot and its restart count, ~1 KB per rank. That cost the watcher twice
(F6):

* **Capacity.** A ConfigMap is capped at 1 MiB per object, so a few hundred
  managed Pods filled it. Every further write then raised, including the
  write-ahead copy of a terminal event -- routine state starving the one thing
  the outbox exists to make durable.
* **Write amplification.** The fields that change most often (``restart_count``,
  a re-read GPU UUID list, a fresh log snapshot) are exactly the ones a restart
  does not need, and every change rewrote the whole document.

So what is persisted is the identity of the attempt (its spec) and the identity
of each Pod -- including the fields a fault is attributed by, see
``POD_FIELDS``. What is dropped is what the next pass re-reads from the live Pod
and no restart needs: the log snapshot, the restart count, the exit signal, the
fabric partition. On restore the observation is rebuilt from these records and
the very next reconcile pass replaces each container with the live Pod's full
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
#: it has ``expected_critical_ranks`` critical ranks); the exit fields keep a
#: rank that already finished from reading as still running.
#:
#: The rest is the *attribution identity*, and it is why this list is not
#: shorter. A fault arrives naming a GPU UUID -- or a cgroup path, a container
#: id, a host pid -- and ``WorkloadTopologyService.resolve`` matches it against
#: the container observations of the cluster; ``node``/``instance`` only narrow
#: the search. A restored attempt whose Pods are already gone republishes
#: exactly this record for a whole missing-attempt grace, and it replaces the
#: stored observation, so a record without those fields matches nothing while
#: looking perfectly fresh: the node reads IDLE and a RESET or REBOOT plan
#: proceeds without stopping the workload that is still holding the GPUs. The
#: same fields are what the tombstone's terminal reports as ``allocation``.
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
    ("uuids", "gpu_uuids"),
    ("cid", "container_id"),
    ("cgroup", "cgroup_path"),
    ("pid", "host_pid"),
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
    value here is already JSON. A value that is ``None`` is dropped and nothing
    else is: this document is bounded in bytes, and the model's own default
    (``None``, or the empty list of ``gpu_uuids``) comes back on restore.
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

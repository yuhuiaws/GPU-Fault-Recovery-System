"""COLLECT-021: the product's durable record that the killed Pod died.

The Store keeps one attempt observation per attempt, the latest, and the
product supersedes the reading that shows the killed container within seconds.
The failure-detected event opens the passive containment (FREEZE_EVIDENCE,
STOP_WORKLOADS); the stop annotates the attempt's Pods, suspends the workload
and deletes the Pods with zero grace, one after another. A Completion Watcher
pass that lands between two of those deletes lists only the surviving Pod, and
the TIMED_OUT terminal observation is later built from that reading. Live on
plain EKS the reading with the killed container existed for about two seconds,
inside one poll, and every later one named no container of the killed Pod: the
terminal observation carried ``containers: []`` while the terminal event of the
same attempt still listed rank 0's exit 137 on that node.

So the death is proven from what the product keeps:

* the passive failure-containment incident, ``inc-attempt-failure-<sha>`` of
  ``<cluster>/<attempt>/TrainingAttemptFailureDetected``, opened by the
  failure-detected event about a second after the container exited and never
  superseded. Its reasons name ``first_failed_rank=<r>`` and the non-zero
  ``exit_code=<c>``; the product's own pre-kill observation binds the killed
  Pod's UID to rank ``r`` on the node;
* or an attempt observation that shows the killed container terminated with a
  non-zero exit code, when a poll catches one (the reading the earlier proof
  waited for alone).

The terminal event's ``rank_exit_status`` is checked as well once the passive
restart is read (``terminal_death_errors``). Every reading here is pure; the
only Store access is the CPU-side ``DEATH_PROBE``.
"""

from __future__ import annotations

import re
import time
from datetime import datetime, timedelta
from typing import Any

from gpu_fault.watcher import failure_containment_ids
from scripts.e2e.regional.regional_commands import RegionalFixtureError
from scripts.e2e.regional.regional_live_fixture import RegionalLiveFixture

FAILURE_EVENT_TYPE = "TRAINING_ATTEMPT_FAILURE_DETECTED"
FAILED_CONTAINER_REASON = "critical container exited non-zero"
#: The runner host and the control plane both keep NTP time; the incident may
#: not predate the kill by more than this.
CLOCK_SKEW_SECONDS = 5
FIRST_FAILED_RANK = re.compile(r"first_failed_rank=(\d+)")
EXIT_CODE = re.compile(r"exit_code=(-?\d+)")
#: The terminal path's containment ("terminal ... reported before failure
#: detection") names each failed rank instead.
RANK_EXIT_CODE = re.compile(r"rank (\d+) exit_code=(-?\d+)")

DEATH_PROBE = r"""
import json
import sys
from datetime import datetime, timezone
from gpu_fault.app import ApplicationContext
from gpu_fault.store import NotFoundError
from gpu_fault.watcher import failure_containment_ids

cluster_id, job_id, attempt_id = sys.argv[1:]
store = ApplicationContext.from_environment().store
incident_id, _workflow_id = failure_containment_ids(
    f"{cluster_id}/{attempt_id}/TrainingAttemptFailureDetected"
)
try:
    incident = store.get_incident(incident_id)
except NotFoundError:
    incident = None
print(json.dumps({
    "failure_incident": (
        incident.model_dump(mode="json") if incident is not None else None
    ),
    "observations": [
        item.model_dump(mode="json")
        for item in store.list_attempt_observations(cluster_id)
        if item.job_id == job_id and item.attempt_id == attempt_id
    ],
    "captured_at": datetime.now(timezone.utc).isoformat(),
}, sort_keys=True))
"""


def _instant(value: Any) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(str(value or "").replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.utcoffset() is not None else None


def killed_pod_rank(
    observation: dict[str, Any] | None, *, pod_uid: str, node: str
) -> int | None:
    """The rank the product's own reading gives the killed Pod on its node."""

    ranks = {
        item.get("rank")
        for item in (observation or {}).get("containers") or []
        if item.get("pod_uid") == pod_uid and item.get("node_id") == node
    }
    rank = next(iter(ranks)) if len(ranks) == 1 else None
    return rank if type(rank) is int and rank >= 0 else None


def death_observation_errors(
    observation: dict[str, Any],
    *,
    node: str | None,
    job_id: str | None,
    attempt_id: str,
    pod_uid: str,
) -> list[str]:
    errors = []
    if job_id is not None and observation.get("job_id") != job_id:
        errors.append("death observation belongs to another job")
    if observation.get("attempt_id") != attempt_id:
        errors.append("death observation belongs to another attempt")
    containers = [
        item
        for item in observation.get("containers") or []
        if item.get("pod_uid") == pod_uid
        and (node is None or item.get("node_id") == node)
    ]
    killed_terminated = bool(containers) and all(
        item.get("terminated") is True
        and type(item.get("exit_code")) is int
        and item["exit_code"] != 0
        for item in containers
    )
    # The product's failure-detected reading is posted the moment the killed
    # container's exit is seen; the PyTorchJob flips to Failed a moment later
    # and no re-post follows for that flip alone, so the next observation with
    # phase FAILED may be the terminal one -- which is also when the passive
    # restart re-owns the node (live 2026-09-28: 57 s vs the usual 22 s, and
    # the late-XID window was gone). Pod death is the terminated killed
    # container, in a RUNNING/PENDING phase as much as in FAILED.
    phase = observation.get("workload_phase")
    if phase != "FAILED" and not (
        phase in {"RUNNING", "PENDING"} and killed_terminated
    ):
        errors.append("death observation phase is not FAILED")
    if not containers:
        errors.append("death observation does not contain the killed Pod on its node")
    elif any(item.get("terminated") is not True for item in containers):
        errors.append("killed container is not terminated in the observation")
    if containers and not any(
        type(item.get("exit_code")) is int and item["exit_code"] != 0
        for item in containers
    ):
        errors.append("killed container has no non-zero exit code")
    return errors


def failure_incident_errors(
    incident: dict[str, Any] | None,
    *,
    cluster_id: str,
    node: str | None,
    job_id: str,
    attempt_id: str,
    rank: int | None,
    killed_after: datetime | None,
) -> list[str]:
    """Whether the passive failure containment records the killed rank's death."""

    if not isinstance(incident, dict):
        return ["the product recorded no failure containment for the attempt"]
    event_key = f"{cluster_id}/{attempt_id}/TrainingAttemptFailureDetected"
    errors = []
    if (
        incident.get("incident_id") != failure_containment_ids(event_key)[0]
        or incident.get("event_id") != event_key
        or incident.get("event_type") != FAILURE_EVENT_TYPE
        or incident.get("cluster_id") != cluster_id
        or incident.get("job_id") != job_id
        or incident.get("attempt_id") != attempt_id
    ):
        errors.append("failure containment identity differs from the attempt")
    if node is not None and node not in (incident.get("node_ids") or []):
        errors.append("failure containment does not name the killed Pod's node")
    reasons = [str(item) for item in incident.get("reasons") or []]
    exits: dict[int, int] = {}
    if FAILED_CONTAINER_REASON in reasons:
        ranks = [int(m[1]) for r in reasons if (m := FIRST_FAILED_RANK.fullmatch(r))]
        codes = [int(m[1]) for r in reasons if (m := EXIT_CODE.fullmatch(r))]
        if len(ranks) == 1 and len(codes) == 1:
            exits[ranks[0]] = codes[0]
    for reason in reasons:
        if match := RANK_EXIT_CODE.fullmatch(reason):
            exits.setdefault(int(match[1]), int(match[2]))
    if rank is None:
        errors.append("the product's pre-kill reading did not rank the killed Pod")
    elif rank not in exits:
        errors.append(f"failure containment does not name rank {rank} as failed")
    elif exits[rank] == 0:
        errors.append("failure containment records no non-zero exit code")
    created = _instant(incident.get("created_at"))
    if (
        created is None
        or killed_after is None
        or created < killed_after - timedelta(seconds=CLOCK_SKEW_SECONDS)
    ):
        errors.append("failure containment predates the kill")
    return errors


def death_record(
    state: dict[str, Any],
    *,
    cluster_id: str,
    node: str | None,
    job_id: str,
    attempt_id: str,
    pod_uid: str,
    rank: int | None,
    killed_after: datetime | None,
) -> tuple[dict[str, Any], list[str]]:
    """One probe reading as a death record, and why it does not prove death.

    The record carries what judged it (cluster, job, killed rank, kill
    instant), so ``death_record_errors`` can re-judge it on its own later.
    """

    observations = [
        item for item in state.get("observations") or [] if isinstance(item, dict)
    ]
    incident = state.get("failure_incident")
    record: dict[str, Any] = {
        "proven_by": None,
        "recorded_at": None,
        "cluster_id": cluster_id,
        "job_id": job_id,
        "killed_rank": rank,
        "killed_after": killed_after.isoformat() if killed_after else None,
        "observation": observations[-1] if observations else None,
        "failure_incident": incident if isinstance(incident, dict) else None,
        "captured_at": state.get("captured_at"),
    }
    observation_errors: list[str] = ["the Store holds no observation of the attempt"]
    for item in observations:
        observation_errors = death_observation_errors(
            item, node=node, job_id=job_id, attempt_id=attempt_id, pod_uid=pod_uid
        )
        if not observation_errors:
            record.update(
                proven_by="observation",
                recorded_at=item.get("observed_at"),
                observation=item,
            )
            return record, []
    incident_errors = failure_incident_errors(
        record["failure_incident"],
        cluster_id=cluster_id,
        node=node,
        job_id=job_id,
        attempt_id=attempt_id,
        rank=rank,
        killed_after=killed_after,
    )
    if not incident_errors:
        record.update(
            proven_by="failure_incident",
            recorded_at=record["failure_incident"].get("created_at"),
        )
        return record, []
    return record, [*incident_errors, *observation_errors]


def death_record_errors(
    record: dict[str, Any] | None,
    *,
    node: str | None,
    job_id: str | None,
    attempt_id: str,
    pod_uid: str,
    xid_injected_at: datetime | None = None,
) -> list[str]:
    """Re-judge a death record and, once injected, its order before the XID."""

    record = record if isinstance(record, dict) else {}
    observation = record.get("observation")
    _record, errors = death_record(
        {
            "observations": [observation] if isinstance(observation, dict) else [],
            "failure_incident": record.get("failure_incident"),
        },
        cluster_id=str(record.get("cluster_id") or ""),
        node=node,
        job_id=job_id if job_id is not None else str(record.get("job_id") or ""),
        attempt_id=attempt_id,
        pod_uid=pod_uid,
        rank=record.get("killed_rank"),
        killed_after=_instant(record.get("killed_after")),
    )
    recorded = _instant(record.get("recorded_at"))
    if (
        not errors
        and xid_injected_at is not None
        and (recorded is None or recorded >= xid_injected_at)
    ):
        errors.append("the Pod death was not recorded before the late XID")
    return errors


def terminal_death_errors(
    event: dict[str, Any] | None, *, rank: int | None, node: str | None
) -> list[str]:
    """Whether the attempt's terminal event names the killed rank's exit."""

    exits = [
        item
        for item in (event or {}).get("rank_exit_status") or []
        if (rank is None or item.get("rank") == rank)
        and (node is None or item.get("node_id") == node)
    ]
    if not any(
        type(item.get("exit_code")) is int and item["exit_code"] != 0 for item in exits
    ):
        return ["terminal event does not record the killed rank's non-zero exit"]
    return []


def wait_death_record(
    regional: RegionalLiveFixture,
    *,
    node: str,
    job_id: str,
    attempt_id: str,
    pod_uid: str,
    rank: int | None,
    killed_after: datetime,
    timeout_seconds: int = 300,
    poll_seconds: int = 5,
) -> dict[str, Any]:
    """Poll until the product has recorded the killed Pod's death."""

    cluster_id = regional.settings.cluster_id
    deadline = time.monotonic() + timeout_seconds
    errors: list[str] = []
    record: dict[str, Any] = {}
    while True:
        state = regional.cpu_python(DEATH_PROBE, cluster_id, job_id, attempt_id)
        record, errors = death_record(
            state if isinstance(state, dict) else {},
            cluster_id=cluster_id,
            node=node,
            job_id=job_id,
            attempt_id=attempt_id,
            pod_uid=pod_uid,
            rank=rank,
            killed_after=killed_after,
        )
        if not errors:
            return record
        if time.monotonic() >= deadline:
            break
        time.sleep(poll_seconds)
    raise RegionalFixtureError(
        "the product never recorded the killed Pod's death: "
        + "; ".join(errors)
        + f" (last reading: {record})"
    )

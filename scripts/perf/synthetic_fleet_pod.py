"""A perf run's synthetic healthy fleet, inside a control-plane Pod (§8.5).

The runner side is ``regional_synthetic_fleet.py``; this program runs in an
api-ha Pod through ``kubectl exec -i`` with the product installed and nothing
else (it imports no harness module). The seed script registers the fleet
(``fleet-agents`` mode); this program keeps it and counts it:

* ``fleet-heartbeat-watch`` renews every fleet Agent of the run for the whole
  run -- ``last_seen_at`` and the lease -- by compare-and-set
  (``replace_agent_if_matches``), so an Agent the teardown purged, or one the
  product moved (DRAINING, REVOKED, another incarnation), is never written
  back. One pass goes cluster by cluster, paced so that it spreads over
  ``FLEET_INTERVAL_SECONDS`` instead of writing the whole fleet in one burst.
  The program arrives length-prefixed on stdin and the rest of stdin is its
  stop channel (the arrangement of ``regional_validation_telemetry``): closing
  it ends the session, and so does a runner that dies. It also ends at
  ``FLEET_DEADLINE_SECONDS`` and once no fleet Agent of the run is left.
* ``fleet-census`` reads, per perf cluster of the run, the Agents the
  remediation storm breaker counts (ACTIVE or DRAINING), with the product's own
  ``cluster_gpu_nodes`` and ``storm_threshold`` when the deployed image has
  them, how many of them are the fleet's and current, and the cluster's
  breaker record. After the teardown the same census is the fleet's proof of
  cleanup: no Agent and no breaker record of the run's clusters is left.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from collections import Counter
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from typing import Any

HEARTBEAT_MODE = "fleet-heartbeat-watch"
CENSUS_MODE = "fleet-census"
#: ``seed_regional_action_workflows.FLEET_NODE_PREFIX`` (this program cannot
#: import the harness; the runner's tests pin the two equal).
FLEET_NODE_PREFIX = "fleet-node-"
#: The lifecycle states ``storm_admission.cluster_gpu_nodes`` counts.
COUNTED_STATES = frozenset({"ACTIVE", "DRAINING"})
PROGRESS_SECONDS = 60.0


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def cluster_ids(clusters: int) -> list[str]:
    return [f"perf-cap-{index:03d}" for index in range(clusters)]


def fleet_prefix(run_id: str) -> str:
    return f"{FLEET_NODE_PREFIX}{run_id}-c"


def lifecycle(agent: Any) -> str:
    return str(getattr(agent.lifecycle_state, "value", agent.lifecycle_state))


def current(agent: Any, now: datetime) -> bool:
    lease = agent.lease_expires_at
    return lifecycle(agent) == "ACTIVE" and lease is not None and lease >= now


def renew(store: Any, agent: Any, *, now: datetime, lease_seconds: float) -> bool:
    """One fleet heartbeat; ``False`` when the Agent is not ACTIVE or moved."""

    if lifecycle(agent) != "ACTIVE":
        return False
    renewed = agent.model_copy(
        update={
            "last_seen_at": now,
            "lease_expires_at": now + timedelta(seconds=lease_seconds),
        }
    )
    return bool(store.replace_agent_if_matches(renewed, agent))


def watch_heartbeats(
    store: Any,
    *,
    run_id: str,
    clusters: int,
    lease_seconds: float,
    interval_seconds: float,
    deadline_seconds: float,
    stop: threading.Event,
    clock: Callable[[], datetime] = utc_now,
    monotonic: Callable[[], float] = time.monotonic,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Renew the run's fleet, pass after pass, until stopped (module docstring).

    Each cluster of a pass gets a slot of ``interval_seconds / clusters``:
    its renewals, then ``stop.wait`` for what is left of the slot, so a pass
    takes about ``interval_seconds`` whatever the store's write latency.
    ``stop`` is the stop signal: stdin EOF in the Pod, an event in tests.
    """

    prefix = fleet_prefix(run_id)
    ids = cluster_ids(clusters)
    slot = interval_seconds / max(1, len(ids))
    totals: Counter[str] = Counter()
    passes = 0
    last_pass: dict[str, int] = {}
    longest_gap = 0.0
    last_renewed: dict[str, datetime] = {}
    ended_by: str | None = None
    end = monotonic() + deadline_seconds
    next_progress = monotonic() + PROGRESS_SECONDS
    while ended_by is None:
        tally: Counter[str] = Counter()
        for cluster_id in ids:
            if stop.is_set():
                ended_by = "stopped"
                break
            if monotonic() >= end:
                ended_by = "deadline"
                break
            slot_started = monotonic()
            for agent in store.list_agents(cluster_id):
                if not agent.node_id.startswith(prefix):
                    continue
                tally["fleet"] += 1
                written_at = clock()
                if renew(store, agent, now=written_at, lease_seconds=lease_seconds):
                    tally["renewed"] += 1
                    previous = last_renewed.get(agent.node_id)
                    if previous is not None:
                        gap = (written_at - previous).total_seconds()
                        longest_gap = max(longest_gap, gap)
                    last_renewed[agent.node_id] = written_at
                else:
                    tally["not_renewed"] += 1
            if stop.wait(max(0.0, slot - (monotonic() - slot_started))):
                ended_by = "stopped"
                break
        else:
            passes += 1
            last_pass = dict(tally)
            if not tally["fleet"]:
                ended_by = "fleet-gone"
        totals.update(tally)
        if progress is not None and monotonic() >= next_progress:
            progress({"passes": passes, "renewed": totals["renewed"]})
            next_progress = monotonic() + PROGRESS_SECONDS
    return {
        "mode": HEARTBEAT_MODE,
        "run_id": run_id,
        "ended_by": ended_by,
        "passes": passes,
        "renewed": totals["renewed"],
        "not_renewed": totals["not_renewed"],
        "last_pass": last_pass,
        "lease_seconds": lease_seconds,
        "interval_seconds": interval_seconds,
        "longest_renewal_gap_seconds": round(longest_gap, 3),
    }


def product_count(store: Any, cluster_id: str) -> tuple[int | None, int | None]:
    """The breaker's own node count and threshold; ``(None, None)`` on an image
    without the breaker."""

    try:
        from gpu_fault.execution.storm_admission import cluster_gpu_nodes
        from gpu_fault.remediation_storm import storm_threshold
    except ImportError:
        return None, None
    nodes = cluster_gpu_nodes(store, cluster_id)
    return nodes, storm_threshold(nodes)


def product_breaker_available() -> bool:
    try:
        import gpu_fault.remediation_storm  # noqa: F401
    except ImportError:
        return False
    return True


def breaker_summary(store: Any, cluster_id: str) -> dict[str, Any] | None:
    reader = getattr(store, "get_remediation_storm_breaker", None)
    record = reader(cluster_id) if reader is not None else None
    if record is None:
        return None

    def stamp(value: datetime | None) -> str | None:
        return value.isoformat() if value is not None else None

    return {
        "state": str(getattr(record.state, "value", record.state)),
        "open": bool(record.open),
        "threshold": record.threshold,
        "window_nodes": len({entry.node_id for entry in record.window}),
        "trip_id": record.trip_id,
        "tripped_at": stamp(record.tripped_at),
        "trip_nodes": len(record.trip_node_ids),
        "trip_domains": record.trip_domains,
        "trip_threshold": record.trip_threshold,
        "trip_cluster_nodes": record.trip_cluster_nodes,
        "trip_workflow_id": record.trip_workflow_id,
        "notified_trip_id": record.notified_trip_id,
        "released_at": stamp(record.released_at),
        "released_by": record.released_by,
        "aborted_at": stamp(record.aborted_at),
    }


def census(store: Any, *, run_id: str, clusters: int, now: datetime) -> dict[str, Any]:
    prefix = fleet_prefix(run_id)
    report: dict[str, Any] = {}
    for cluster_id in cluster_ids(clusters):
        agents = store.list_agents(cluster_id)
        fleet = [agent for agent in agents if agent.node_id.startswith(prefix)]
        nodes, threshold = product_count(store, cluster_id)
        report[cluster_id] = {
            "agents": len(agents),
            "counted_nodes": sum(
                lifecycle(agent) in COUNTED_STATES for agent in agents
            ),
            "lifecycle": dict(sorted(Counter(map(lifecycle, agents)).items())),
            "breaker_cluster_nodes": nodes,
            "threshold": threshold,
            "fleet_agents": len(fleet),
            "fleet_current": sum(current(agent, now) for agent in fleet),
            "model_agents": len(agents) - len(fleet),
            "breaker": breaker_summary(store, cluster_id),
        }
    return {
        "mode": CENSUS_MODE,
        "run_id": run_id,
        "observed_at": now.isoformat(),
        "product_breaker": product_breaker_available(),
        "clusters": report,
    }


def stdin_closed() -> threading.Event:
    """Set at stdin EOF: the runner's stop signal (or a runner that died).

    The thread reads the raw descriptor, not ``sys.stdin.buffer``: a session
    that ends on its own (deadline, no fleet left) while stdin is still open
    would otherwise shut down with this daemon thread holding the buffered
    reader's lock, which aborts the interpreter.
    """

    closed = threading.Event()

    def wait_for_eof() -> None:
        try:
            while os.read(sys.stdin.fileno(), 65536):
                pass
        except OSError:
            pass
        finally:
            closed.set()

    threading.Thread(target=wait_for_eof, name="stdin-eof", daemon=True).start()
    return closed


def main() -> None:
    from gpu_fault.app import ApplicationContext

    run_id = os.environ["FLEET_RUN_ID"]
    clusters = int(os.environ["FLEET_CLUSTERS"])
    mode = os.environ["FLEET_MODE"]
    context = ApplicationContext.from_environment()
    try:
        if mode == CENSUS_MODE:
            document = census(
                context.store, run_id=run_id, clusters=clusters, now=utc_now()
            )
        elif mode == HEARTBEAT_MODE:
            document = watch_heartbeats(
                context.store,
                run_id=run_id,
                clusters=clusters,
                lease_seconds=float(os.environ["FLEET_LEASE_SECONDS"]),
                interval_seconds=float(os.environ["FLEET_INTERVAL_SECONDS"]),
                deadline_seconds=float(os.environ["FLEET_DEADLINE_SECONDS"]),
                stop=stdin_closed(),
                progress=lambda status: print(
                    json.dumps({"progress": status}, sort_keys=True), flush=True
                ),
            )
        else:
            raise RuntimeError(f"unsupported fleet mode: {mode}")
    finally:
        close = getattr(context.store, "close", None)
        if close is not None:
            close()
    print(json.dumps(document, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

"""The remediation storm breaker's decision, without a store (design §9.5).

It counts the distinct nodes that newly entered destructive remediation in a
cluster over 600 s and trips at max(3, ceil(20% of the cluster's GPU nodes))
nodes across at least two workload domains; one job (or one incident) is one
domain. Open, it holds every first admission; only an operator closes it.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from gpu_fault.remediation_storm import (
    STORM_WINDOW_SECONDS,
    StormBreakerState,
    StormCandidate,
    aborted_breaker,
    released_breaker,
    storm_admission,
    storm_domain,
    storm_threshold,
)
from tests._builders import fault_incident

NOW = datetime(2026, 10, 29, 12, 0, tzinfo=timezone.utc)


def _candidate(name: str, *nodes: str, domain: str | None = None, **values):
    return StormCandidate(
        cluster_id="cluster-a",
        workflow_id=f"workflow-{name}",
        node_ids=frozenset(nodes),
        domain=domain or f"incident:{name}",
        **values,
    )


def _admit_all(candidates, *, cluster_nodes, at=NOW, record=None):
    verdicts = []
    for index, candidate in enumerate(candidates):
        record, refused = storm_admission(
            record,
            candidate,
            now=at + timedelta(seconds=index),
            cluster_nodes=cluster_nodes,
        )
        verdicts.append(refused is None)
    return record, verdicts


@pytest.mark.parametrize(
    ("cluster_nodes", "threshold"),
    [(None, 3), (0, 3), (4, 3), (15, 3), (16, 4), (100, 20), (500, 100), (501, 101)],
)
def test_the_threshold_is_twenty_percent_of_the_cluster_with_a_floor_of_three(
    cluster_nodes, threshold
) -> None:
    assert storm_threshold(cluster_nodes) == threshold


def test_many_idle_nodes_trip_it_and_the_tripping_admission_is_held() -> None:
    record, admitted = _admit_all(
        [_candidate(f"idle-{index}", f"node-{index}") for index in range(4)],
        cluster_nodes=10,
    )

    assert admitted == [True, True, False, False]
    assert record.state is StormBreakerState.TRIPPED
    assert record.trip_node_ids == ["node-0", "node-1", "node-2"]
    assert record.trip_threshold == 3 and record.trip_domains == 3
    assert record.trip_workflow_id == "workflow-idle-2"


def test_one_job_on_many_nodes_never_trips_it() -> None:
    job = storm_domain(fault_incident("incident-j", "event-j", job_id="job-7"))
    nodes = [f"node-{index}" for index in range(40)]

    record, admitted = _admit_all(
        [
            _candidate("job-first", *nodes[:20], domain=job),
            _candidate("job-second", *nodes[20:], domain=job),
        ],
        cluster_nodes=50,
    )

    assert admitted == [True, True]
    assert record.state is StormBreakerState.CLOSED
    assert len({entry.node_id for entry in record.window}) == 40


def test_two_nodes_never_trip_it() -> None:
    record, admitted = _admit_all(
        [_candidate("a", "node-a"), _candidate("b", "node-b")], cluster_nodes=None
    )

    assert admitted == [True, True]
    assert record.state is StormBreakerState.CLOSED


def test_a_large_cluster_trips_at_its_percentage_not_at_the_floor() -> None:
    nodes = [_candidate(f"idle-{index}", f"node-{index}") for index in range(20)]

    record, admitted = _admit_all(nodes[:19], cluster_nodes=100)
    assert all(admitted) and record.state is StormBreakerState.CLOSED

    record, admitted = _admit_all(nodes[19:], cluster_nodes=100, record=record)
    assert admitted == [False]
    assert record.trip_threshold == 20


def test_nodes_age_out_of_the_window() -> None:
    record, _ = _admit_all(
        [_candidate("a", "node-a"), _candidate("b", "node-b")], cluster_nodes=None
    )

    later = NOW + timedelta(seconds=STORM_WINDOW_SECONDS + 5)
    record, refused = storm_admission(
        record, _candidate("c", "node-c"), now=later, cluster_nodes=None
    )

    assert refused is None
    assert [entry.node_id for entry in record.window] == ["node-c"]


def test_a_candidate_bringing_no_new_node_is_admitted_and_restamped() -> None:
    record, _ = _admit_all(
        [_candidate("a", "node-a"), _candidate("b", "node-b")], cluster_nodes=None
    )
    # Window: two nodes, two domains. node-a again, from a third incident.
    record, refused = storm_admission(
        record,
        _candidate("again", "node-a"),
        now=NOW + timedelta(seconds=30),
        cluster_nodes=None,
    )

    assert refused is None
    assert record.state is StormBreakerState.CLOSED
    restamped = next(entry for entry in record.window if entry.node_id == "node-a")
    assert restamped.workflow_id == "workflow-again"


def test_open_it_holds_everything_until_an_operator_releases_it() -> None:
    record, _ = _admit_all(
        [_candidate(f"idle-{index}", f"node-{index}") for index in range(3)],
        cluster_nodes=None,
    )
    tripped_at = record.tripped_at
    assert tripped_at is not None

    # Hours later it is still open: no auto-release.
    _, refused = storm_admission(
        record,
        _candidate("late", "node-z"),
        now=NOW + timedelta(hours=6),
        cluster_nodes=None,
    )
    assert refused is not None and "storm:cluster-a" in refused

    released = released_breaker(
        record,
        operator="alice",
        reason="driver rollout reverted",
        reference="CHG-1",
        now=NOW + timedelta(hours=7),
    )
    assert released.state is StormBreakerState.CLOSED and released.window == []
    assert released.released_by == "alice"
    assert (
        released_breaker(
            released, operator="bob", reason="again", reference="CHG-2", now=NOW
        )
        == released
    ), "release is idempotent"

    # The row held through the trip is admitted without counting again; a
    # new fault after the release counts.
    held = _candidate("idle-2", "node-2", held_since=tripped_at + timedelta(seconds=1))
    after, refused = storm_admission(
        released, held, now=NOW + timedelta(hours=7, minutes=1), cluster_nodes=None
    )
    assert refused is None and after.window == []
    after, refused = storm_admission(
        after,
        _candidate("new", "node-n"),
        now=NOW + timedelta(hours=7, minutes=2),
        cluster_nodes=None,
    )
    assert refused is None and [entry.node_id for entry in after.window] == ["node-n"]


def test_abort_keeps_it_open_and_needs_a_tripped_breaker() -> None:
    record, _ = _admit_all(
        [_candidate(f"idle-{index}", f"node-{index}") for index in range(3)],
        cluster_nodes=None,
    )

    aborted = aborted_breaker(
        record, operator="alice", reason="bad image", reference="INC-9", now=NOW
    )
    assert aborted.state is StormBreakerState.ABORTED and aborted.open
    assert (
        aborted_breaker(aborted, operator="bob", reason="x", reference="y", now=NOW)
        == aborted
    ), "abort is idempotent"
    _, refused = storm_admission(
        aborted, _candidate("late", "node-z"), now=NOW, cluster_nodes=None
    )
    assert refused is not None and "aborted by an operator" in refused

    released = released_breaker(
        aborted, operator="alice", reason="fixed", reference="INC-9", now=NOW
    )
    with pytest.raises(ValueError, match="not tripped"):
        aborted_breaker(released, operator="a", reason="b", reference="c", now=NOW)


def test_the_domain_is_the_job_else_the_incident() -> None:
    job = fault_incident("incident-1", "event-1", job_id="job-7", attempt_id="a-2")
    idle = fault_incident("incident-2", "event-2")

    assert storm_domain(job) == "job:job-7"
    assert storm_domain(idle) == "incident:incident-2"

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from gpu_fault.processor import (
    ProcessorCoordinator,
    ProcessorPoolSettings,
    ProcessorRequest,
    ProcessorRequestStatus,
)
from gpu_fault.processor.batching import telemetry_batch_size
from tests._builders import copy_model, processor_request
from tests.processor._leadership_support import (
    LEADER_LEASE,
    NOW,
    REQUEST_LEASE,
    _telemetry,
    correlated_request,
)


def test_only_one_processor_leader_holds_live_lease(stores) -> None:
    first, second = stores

    leader = first.acquire_processor_leadership(
        "pod-a", now=NOW, lease_duration=LEADER_LEASE
    )
    contender = second.acquire_processor_leadership(
        "pod-b", now=NOW + timedelta(seconds=1), lease_duration=LEADER_LEASE
    )

    assert leader.owner_id == "pod-a"
    assert leader.epoch == 1
    assert contender == leader


def test_expired_processor_leader_is_replaced_with_new_epoch(stores) -> None:
    first, second = stores
    first.acquire_processor_leadership("pod-a", now=NOW, lease_duration=LEADER_LEASE)

    replacement = second.acquire_processor_leadership(
        "pod-b",
        now=NOW + LEADER_LEASE + timedelta(milliseconds=1),
        lease_duration=LEADER_LEASE,
    )

    assert replacement.owner_id == "pod-b"
    assert replacement.epoch == 2


def test_periodic_tasks_have_independent_leases(stores) -> None:
    first, second = stores
    lease = timedelta(seconds=15)

    cleanup = first.acquire_periodic_task_lease(
        "processor-cleanup", "pod-a", now=NOW, lease_duration=lease
    )
    cleanup_contender = second.acquire_periodic_task_lease(
        "processor-cleanup",
        "pod-b",
        now=NOW + timedelta(seconds=1),
        lease_duration=lease,
    )
    identity = second.acquire_periodic_task_lease(
        "identity-refresh",
        "pod-b",
        now=NOW + timedelta(seconds=1),
        lease_duration=lease,
    )
    cleanup_replacement = second.acquire_periodic_task_lease(
        "processor-cleanup",
        "pod-b",
        now=NOW + lease + timedelta(milliseconds=1),
        lease_duration=lease,
    )

    assert cleanup.owner_id == "pod-a"
    assert cleanup_contender == cleanup
    assert identity.owner_id == "pod-b"
    assert identity.epoch == 1
    assert cleanup_replacement.owner_id == "pod-b"
    assert cleanup_replacement.epoch == 2


def test_active_consumers_serialize_lane_and_parallelize_clusters(stores) -> None:
    first, second = stores
    requests = [
        processor_request("/v1/gpu-events/xid", cluster_id=cluster_id)
        for cluster_id in ("cluster-a", "cluster-a", "cluster-b")
    ]
    for item in requests:
        first.enqueue_processor_request(item)

    claimed_a = first.claim_active_processor_requests(
        "pod-a", now=NOW + timedelta(seconds=2), lease_duration=REQUEST_LEASE, limit=1
    )
    claimed_b = second.claim_active_processor_requests(
        "pod-b",
        now=NOW + timedelta(seconds=2, milliseconds=1),
        lease_duration=REQUEST_LEASE,
        limit=4,
    )

    assert [item.cluster_id for item in claimed_a] == ["cluster-a"]
    assert [item.cluster_id for item in claimed_b] == ["cluster-b"]


def test_processor_request_derives_attempt_correlation_key() -> None:
    request = correlated_request("/v1/workload-observations")

    assert request.correlation_key == (
        '["cluster-a","training-job","training-job-a001"]'
    )
    assert (
        processor_request(
            "/v1/workload-observations", body=b'{"job_id":"training-job"}'
        ).correlation_key
        is None
    )
    assert (
        processor_request(
            "/v1/incidents/example",
            body=b'{"job_id":"training-job","attempt_id":"training-job-a001"}',
        ).correlation_key
        is None
    )


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        (
            "/v1/incidents/incident-a/advisory-notifications",
            "cluster-a:incident:incident-a",
        ),
        ("/v1/workflows/workflow-a/simulate", "cluster-a:workflow:workflow-a"),
        ("/v1/recovery-plans/plan-a/simulate", "cluster-a:recovery-plan:plan-a"),
    ],
)
def test_control_resources_use_resource_scoped_lanes(path: str, expected: str) -> None:
    request = processor_request(path)

    assert request.ordering_key() == expected


def test_unscoped_xid_uses_node_lane() -> None:
    request = processor_request("/v1/gpu-events/xid", body=b'{"node_id":"node-a"}')

    assert request.ordering_key() == "cluster-a:node:node-a"


@pytest.mark.parametrize(
    "path",
    ["/v1/collector-events/nvidia-kernel", "/v1/collector-events/fabric-manager"],
)
def test_real_fault_collectors_use_attempt_priority_and_node_lane(path: str) -> None:
    request = processor_request(
        path,
        body=b'{"node_id":"node-a","job_id":"training-job","attempt_id":"training-job-a001"}',
    )
    peer = processor_request(
        path,
        body=b'{"node_id":"node-b","job_id":"training-job","attempt_id":"training-job-a001"}',
    )

    assert request.correlation_key == (
        '["cluster-a","training-job","training-job-a001"]'
    )
    assert request.is_correlated_fault()
    assert request.correlation_scope_keys == [
        '["cluster-a","training-job","training-job-a001"]',
        '["cluster-a","node","node-a"]',
    ]
    assert request.queue_priority() == 10
    assert request.is_reserved_tier()
    assert request.ordering_key() == "cluster-a:node:node-a"
    assert peer.ordering_key() != request.ordering_key()


@pytest.mark.parametrize(
    "path",
    [
        "/v1/incidents/inc-1/acknowledge",
        "/v1/workflows/wf-1/steps/s1/complete",
        "/v1/attempts/attempt-1/hang-check",
        "/v1/recovery-plans/plan-1/approve",
    ],
)
def test_control_plane_actions_are_the_first_tier(path) -> None:
    """Tier 0: the actions that recover from a decided fault.

    They are claimed before the device events of the storm they end; the
    reserved queue depth and the dedicated worker pool cover both tiers via
    ``is_reserved_tier``.
    """

    request = processor_request(path, body=b'{"node_id":"node-a"}')

    assert request.queue_priority() == 0, path
    assert request.is_reserved_tier(), path
    assert request.spoolable() is False, path


@pytest.mark.parametrize(
    "path",
    [
        "/v1/gpu-events/nvidia-kernel",
        "/v1/provider-events/hyperpod-hma/health",
        "/v1/collector-events/nvidia-kernel",
        "/v1/collector-events/fabric-manager",
    ],
)
def test_device_events_are_the_second_fault_tier(path) -> None:
    """Tier 10: a decided fault reported by a node or a provider.

    Still a fault -- reserved depth, fault pool, never spooled -- but ordered
    after the control-plane actions so a storm cannot starve its own cure.
    """

    request = processor_request(path, body=b'{"node_id":"node-a"}')

    assert request.queue_priority() == 10, path
    assert request.is_reserved_tier(), path
    assert request.spoolable() is False, path


@pytest.mark.parametrize(
    ("routine_age_seconds", "expected_priority"), [(5, 50), (31, 100)]
)
def test_routine_aging_bounds_priority_fifty_starvation(
    stores, routine_age_seconds: int, expected_priority: int
) -> None:
    store, _peer = stores
    observed = datetime.now(timezone.utc)
    evidence = processor_request(
        "/v1/collector-events/host-telemetry",
        body=b'{"node_id":"node-evidence","edge_filter_reasons":["threshold:test"]}',
    )
    routine = copy_model(
        processor_request(
            "/v1/collector-events/host-telemetry",
            body=b'{"node_id":"node-routine","edge_filter_reasons":["health-summary"]}',
        ),
        created_at=observed - timedelta(seconds=routine_age_seconds),
        updated_at=observed - timedelta(seconds=routine_age_seconds),
    )
    store.enqueue_processor_request(routine)
    store.enqueue_processor_request(evidence)

    claimed = store.claim_active_processor_requests(
        "pod-aging",
        now=observed,
        lease_duration=REQUEST_LEASE,
        limit=1,
        include_paths={"/v1/collector-events/host-telemetry"},
        routine_starvation_seconds=30,
    )

    assert claimed[0].queue_priority() == expected_priority


def test_fault_pressure_caps_but_does_not_stop_evidence_pools(stores) -> None:
    store, _peer = stores
    fault = processor_request(
        "/v1/collector-events/nvidia-kernel", body=b'{"node_id":"node-fault"}'
    )
    store.enqueue_processor_request(fault)
    for index in range(20):
        store.enqueue_processor_request(
            _telemetry(f"gpu-{index}", index, reasons=["threshold:synthetic"])
        )
        store.enqueue_processor_request(
            _telemetry(
                f"host-{index}",
                index,
                path="/v1/collector-events/host-telemetry",
                reasons=["threshold:synthetic"],
            )
        )
    processor = ProcessorCoordinator(
        store,
        owner_id="pod-pressure",
        internal_token="pressure-token",
        active_consumers=True,
        pools=ProcessorPoolSettings(
            fault_worker_count=1,
            observation_worker_count=0,
            gpu_telemetry_worker_count=4,
            host_telemetry_worker_count=4,
            fault_pressure_evidence_workers=1,
        ),
    )

    claimed = processor._claim_active_by_pool(
        {"fault": 1, "gpu": 4, "host": 4}, lease_duration=REQUEST_LEASE
    )

    by_path: dict[str, list[ProcessorRequest]] = {}
    for item in claimed:
        by_path.setdefault(item.path, []).append(item)
    assert len(by_path["/v1/collector-events/nvidia-kernel"]) == 1
    assert (
        1
        <= len(by_path["/v1/collector-events/gpu-metrics"])
        <= telemetry_batch_size("/v1/collector-events/gpu-metrics")
    )
    assert (
        1
        <= len(by_path["/v1/collector-events/host-telemetry"])
        <= telemetry_batch_size("/v1/collector-events/host-telemetry")
    )
    pressure = processor.metrics_snapshot()["fault_pressure"]
    assert pressure == {"active": 1, "evidence_workers": 1, "activations_total": 1}


@pytest.mark.parametrize(
    "reasons",
    [
        ["threshold:gpu_ecc_uncorrectable"],
        ["sustained:memory_available_percent"],
        ["collection-error:dcgm_unreachable"],
        ["efa-traffic-drop"],
        ["efa-traffic-spike"],
        ["health-summary", "threshold:sm_clock_throttle"],
        ["counter-increased"],
        ["xid-changed"],
        ["candidate-confirmed"],
        # A breach clearing is a state change, and the token is the only
        # place that records it: not droppable either.
        ["recovered"],
        ["candidate-recovered"],
        ["recovered:gpu"],
        ["threshold:gpu_kubernetes_allocatable_mismatch"],
    ],
)
@pytest.mark.parametrize(
    "path", ["/v1/collector-events/gpu-metrics", "/v1/collector-events/host-telemetry"]
)
def test_a_batch_reporting_something_wrong_is_the_middle_tier(path, reasons) -> None:
    """Tier 50: suspicious, and therefore neither droppable nor mergeable.

    A single routine reason alongside evidence is still evidence: the
    batch is only routine when every reason in it says all is well.
    """

    request = _telemetry("node-a", 1, path=path, reasons=reasons)

    assert request.queue_priority() == 50
    assert request.is_routine_telemetry() is False
    assert request.spoolable() is False


@pytest.mark.parametrize(
    "path", ["/v1/collector-events/gpu-metrics", "/v1/collector-events/host-telemetry"]
)
@pytest.mark.parametrize(
    "reasons",
    [
        ["health-summary"],
        # Three collectors, three spellings of "first sample".
        ["initial-baseline"],
        ["baseline"],
        ["baseline:gpu", "baseline:efa"],
        ["filter-disabled"],
        ["health-summary", "initial-baseline"],
        # No reasons at all is the nvidia-smi fallback batch: the edge
        # filter is not running, so the batch is a full periodic sample.
        [],
        None,
    ],
)
def test_a_periodic_check_up_batch_is_the_routine_tier(path, reasons) -> None:
    request = _telemetry("node-a", 1, path=path, reasons=reasons)

    assert request.queue_priority() == 100
    assert request.is_routine_telemetry() is True
    assert request.spoolable() is True


def test_a_collection_error_outweighs_a_routine_reason() -> None:
    """The reasons can say all is well while the batch is incomplete.

    ``collection_errors`` is a separate field on the batch, and a probe
    that did not answer is exactly the shape of a GPU falling over.
    """

    request = processor_request(
        "/v1/collector-events/host-telemetry",
        body=json.dumps(
            {
                "node_id": "node-a",
                "edge_filter_reasons": ["health-summary"],
                "collection_errors": [{"probe": "nvidia-smi", "error": "timeout"}],
            },
            separators=(",", ":"),
        ).encode(),
    )

    assert request.is_routine_telemetry() is False
    assert request.queue_priority() == 50
    assert request.spoolable() is False


@pytest.mark.parametrize(
    "path, body",
    [
        (
            "/v1/collector-events/gpu-inventory",
            b'{"node_id":"node-a","snapshot_id":"snap-1"}',
        ),
        (
            "/v1/collector-events/node-logs",
            b'{"node_id":"node-a","entries":[],'
            b'"edge_filter_reasons":["health-summary"]}',
        ),
        ("/v1/training-progress", b'{"attempt_id":"attempt-1","rank":0}'),
    ],
)
def test_inventory_logs_and_heartbeats_are_the_routine_tier(path, body) -> None:
    """Deferrable behind anything that is reporting a problem.

    ``_complete_if_stale`` is what decides whether one of these may be
    dropped, and it asks the path, not the tier -- so the demotion here
    changes when they are served, not whether they are kept.
    """

    request = processor_request(path, body=body)

    assert request.queue_priority() == 100


def test_an_unrecognised_path_lands_in_the_middle_tier() -> None:
    """A new endpoint is neither reserved capacity nor droppable.

    Defaulting to 100 would let an unreviewed path be coalesced and
    starved; defaulting to 0 would let it eat the fault reservation.
    """

    request = processor_request("/v1/something-nobody-has-tiered-yet")

    assert request.queue_priority() == 50
    assert request.spoolable() is False


def test_a_body_that_is_not_a_json_object_is_not_routine() -> None:
    """Undecodable means the tier cannot be lowered on the body's word."""

    for body in (b"[1,2,3]", b"not json", b"\xff\xfe"):
        request = processor_request("/v1/collector-events/gpu-metrics", body=body)
        assert request.is_routine_telemetry() is False, body
        assert request.queue_priority() == 50, body


@pytest.mark.parametrize(
    "path",
    ["/v1/collector-events/nvidia-kernel", "/v1/collector-events/fabric-manager"],
)
def test_real_collector_fault_waits_for_same_node_observation(stores, path) -> None:
    first, _ = stores
    fault = processor_request(path, body=b'{"node_id":"node-a"}')
    observation = processor_request(
        "/v1/workload-observations",
        body=b'{"job_id":"training-job","attempt_id":"training-job-a001","containers":[{"node_id":"node-a"}]}',
    )
    assert fault.ordering_key() != observation.ordering_key()
    assert set(fault.correlation_scope_keys).intersection(
        observation.correlation_scope_keys
    )
    first.enqueue_processor_request(fault)
    first.enqueue_processor_request(observation)
    claimed_at = datetime.now(timezone.utc)

    claimed = first.claim_active_processor_requests(
        "pod-a", now=claimed_at, lease_duration=REQUEST_LEASE, limit=4
    )

    assert [item.request_id for item in claimed] == [observation.request_id]
    first.complete_active_processor_request(
        observation.request_id,
        "pod-a",
        claimed[0].leader_epoch,
        claimed[0].lease_token,
        response_status=200,
        response_content_type="application/json",
        response_body_base64="e30=",
    )
    fault_claimed = first.claim_active_processor_requests(
        "pod-b",
        now=claimed_at + timedelta(seconds=1),
        lease_duration=REQUEST_LEASE,
        limit=4,
    )
    assert [item.request_id for item in fault_claimed] == [fault.request_id]


def test_real_collector_fault_does_not_wait_for_other_node(stores) -> None:
    first, _ = stores
    fault = processor_request(
        "/v1/collector-events/nvidia-kernel", body=b'{"node_id":"node-a"}'
    )
    observation = processor_request(
        "/v1/workload-observations",
        body=b'{"job_id":"training-job","attempt_id":"training-job-a001","containers":[{"node_id":"node-b"}]}',
    )
    first.enqueue_processor_request(fault)
    first.enqueue_processor_request(observation)

    claimed = first.claim_active_processor_requests(
        "pod-a", now=NOW, lease_duration=REQUEST_LEASE, limit=4
    )

    assert {item.request_id for item in claimed} == {
        fault.request_id,
        observation.request_id,
    }


def test_matching_observation_is_claimed_before_same_attempt_faults(stores) -> None:
    first, _ = stores
    xid = correlated_request("/v1/gpu-events/xid")
    sxid = correlated_request("/v1/gpu-events/sxid")
    observation = correlated_request("/v1/workload-observations")
    for item in (xid, sxid, observation):
        first.enqueue_processor_request(item)

    claimed = first.claim_active_processor_requests(
        "pod-a", now=datetime.now(timezone.utc), lease_duration=REQUEST_LEASE, limit=1
    )

    assert [item.request_id for item in claimed] == [observation.request_id]


def test_historical_observations_do_not_starve_unrelated_fault(stores) -> None:
    first, _ = stores
    observations = [
        correlated_request(
            "/v1/workload-observations", attempt_id=f"old-attempt-{index}"
        )
        for index in range(5)
    ]
    fault = correlated_request("/v1/gpu-events/xid", attempt_id="current-attempt")
    for item in (*observations, fault):
        first.enqueue_processor_request(item)

    claimed = first.claim_active_processor_requests(
        "pod-a", now=datetime.now(timezone.utc), lease_duration=REQUEST_LEASE, limit=1
    )

    assert [item.request_id for item in claimed] == [fault.request_id]


def test_same_cluster_different_attempts_claim_in_parallel(stores) -> None:
    first, _ = stores
    requests = [
        correlated_request(
            "/v1/workload-observations",
            job_id=f"job-{index}",
            attempt_id=f"attempt-{index}",
        )
        for index in range(4)
    ]
    for item in requests:
        first.enqueue_processor_request(item)

    claimed = first.claim_active_processor_requests(
        "pod-a", now=datetime.now(timezone.utc), lease_duration=REQUEST_LEASE, limit=4
    )

    assert {item.request_id for item in claimed} == {
        item.request_id for item in requests
    }
    assert len({item.ordering_key() for item in claimed}) == 4


def test_same_attempt_paths_share_one_lane(stores) -> None:
    first, _ = stores
    observation = correlated_request("/v1/workload-observations")
    terminal = correlated_request("/v1/attempts/terminal")
    first.enqueue_processor_request(observation)
    first.enqueue_processor_request(terminal)

    claimed = first.claim_active_processor_requests(
        "pod-a", now=datetime.now(timezone.utc), lease_duration=REQUEST_LEASE, limit=4
    )

    assert observation.ordering_key() == terminal.ordering_key()
    assert [item.request_id for item in claimed] == [observation.request_id]


def test_same_node_collector_paths_share_one_lane(stores) -> None:
    first, _ = stores
    kernel = processor_request(
        "/v1/collector-events/nvidia-kernel", body=b'{"node_id":"node-a"}'
    )
    telemetry = processor_request(
        "/v1/collector-events/host-telemetry", body=b'{"node_id":"node-a"}'
    )
    first.enqueue_processor_request(kernel)
    first.enqueue_processor_request(telemetry)

    claimed = first.claim_active_processor_requests(
        "pod-a", now=datetime.now(timezone.utc), lease_duration=REQUEST_LEASE, limit=4
    )

    assert kernel.ordering_key() == telemetry.ordering_key()
    assert len(claimed) == 1


def test_expired_lane_is_reclaimed_and_fences_stale_consumer(stores) -> None:
    first, second = stores
    lease = timedelta(seconds=5)
    request = first.enqueue_processor_request(processor_request("/v1/gpu-events/xid"))
    claimed = first.claim_active_processor_requests(
        "pod-a", now=NOW, lease_duration=lease, limit=1
    )[0]
    reclaimed = second.claim_active_processor_requests(
        "pod-b",
        now=NOW + timedelta(seconds=5, milliseconds=2),
        lease_duration=REQUEST_LEASE,
        limit=1,
    )[0]

    assert reclaimed.request_id == request.request_id
    assert reclaimed.leader_epoch == claimed.leader_epoch + 1
    assert reclaimed.lease_token != claimed.lease_token
    with pytest.raises(ValueError, match="stale processor lane"):
        first.complete_active_processor_request(
            request.request_id,
            "pod-a",
            claimed.leader_epoch,
            claimed.lease_token,
            response_status=200,
            response_content_type="application/json",
            response_body_base64="e30=",
        )


def test_active_lane_renewal_delays_reclaim_and_preserves_fencing(stores) -> None:
    first, second = stores
    claimed_at = datetime.now(timezone.utc)
    initial_lease = timedelta(seconds=5)
    renewed_lease = timedelta(seconds=30)
    request = first.enqueue_processor_request(
        processor_request("/v1/workload-observations")
    )
    claimed = first.claim_active_processor_requests(
        "pod-a", now=claimed_at, lease_duration=initial_lease, limit=1
    )[0]

    assert not second.renew_active_processor_request(
        request.request_id,
        "pod-b",
        claimed.leader_epoch,
        claimed.lease_token,
        lease_duration=renewed_lease,
    )
    assert first.renew_active_processor_request(
        request.request_id,
        "pod-a",
        claimed.leader_epoch,
        claimed.lease_token,
        lease_duration=renewed_lease,
    )
    assert not second.claim_active_processor_requests(
        "pod-b",
        now=claimed_at + initial_lease + timedelta(seconds=1),
        lease_duration=REQUEST_LEASE,
        limit=1,
    )


def test_expired_active_lane_cannot_be_renewed(stores) -> None:
    first, _ = stores
    claimed_at = datetime.now(timezone.utc) - timedelta(seconds=10)
    request = first.enqueue_processor_request(
        processor_request("/v1/workload-observations")
    )
    claimed = first.claim_active_processor_requests(
        "pod-a", now=claimed_at, lease_duration=timedelta(seconds=1), limit=1
    )[0]

    assert not first.renew_active_processor_request(
        request.request_id,
        "pod-a",
        claimed.leader_epoch,
        claimed.lease_token,
        lease_duration=REQUEST_LEASE,
    )


def test_stale_processor_cannot_complete_request_after_failover(stores) -> None:
    first, second = stores
    first.acquire_processor_leadership("pod-a", now=NOW, lease_duration=LEADER_LEASE)
    request = first.enqueue_processor_request(processor_request("/v1/gpu-events/xid"))
    claimed = first.claim_processor_requests(
        "pod-a", 1, now=NOW, lease_duration=REQUEST_LEASE, limit=1
    )[0]

    second.acquire_processor_leadership(
        "pod-b",
        now=NOW + LEADER_LEASE + timedelta(milliseconds=1),
        lease_duration=LEADER_LEASE,
    )

    with pytest.raises(ValueError, match="stale processor"):
        first.complete_processor_request(
            request.request_id,
            "pod-a",
            1,
            claimed.lease_token,
            response_status=200,
            response_content_type="application/json",
            response_body_base64="e30=",
        )

    stored = second.get_processor_request(request.request_id)
    assert stored.status is ProcessorRequestStatus.LEASED
    assert stored.lease_owner == "pod-a"


def test_processor_cannot_complete_after_request_lease_deadline(stores) -> None:
    first, _ = stores
    claimed_at = datetime.now(timezone.utc) - timedelta(seconds=2)
    leadership = first.acquire_processor_leadership(
        "pod-a", now=claimed_at, lease_duration=timedelta(minutes=1)
    )
    request = first.enqueue_processor_request(
        processor_request("/v1/workload-observations")
    )
    claimed = first.claim_processor_requests(
        "pod-a",
        leadership.epoch,
        now=claimed_at,
        lease_duration=timedelta(seconds=1),
        limit=1,
    )[0]

    with pytest.raises(ValueError, match="stale processor"):
        first.complete_processor_request(
            request.request_id,
            "pod-a",
            leadership.epoch,
            claimed.lease_token,
            response_status=200,
            response_content_type="application/json",
            response_body_base64="e30=",
        )


def test_incomplete_processor_request_query_is_cluster_scoped(stores) -> None:
    first, second = stores
    pending = processor_request("/v1/gpu-events/xid")
    completed = copy_model(
        processor_request("/v1/gpu-events/xid", cluster_id="cluster-b"),
        status=ProcessorRequestStatus.COMPLETED,
    )
    first.enqueue_processor_request(pending)
    first.enqueue_processor_request(completed)

    assert second.has_incomplete_processor_requests("cluster-a")
    assert not second.has_incomplete_processor_requests("cluster-b")
    assert not second.has_incomplete_processor_requests("cluster-c")


def test_claims_run_clusters_in_parallel_but_serialize_each_cluster(stores) -> None:
    first, second = stores
    now = datetime.now(timezone.utc)
    leadership = first.acquire_processor_leadership(
        "pod-a", now=now, lease_duration=timedelta(minutes=1)
    )
    requests = [
        processor_request("/v1/gpu-events/xid", cluster_id=cluster_id)
        for cluster_id in ("cluster-a", "cluster-a", "cluster-b")
    ]
    for item in requests:
        first.enqueue_processor_request(item)

    claimed = second.claim_processor_requests(
        "pod-a", leadership.epoch, now=now, lease_duration=REQUEST_LEASE, limit=16
    )

    assert [item.cluster_id for item in claimed] == ["cluster-a", "cluster-b"]
    assert (
        second.claim_processor_requests(
            "pod-a",
            leadership.epoch,
            now=now + timedelta(milliseconds=1),
            lease_duration=REQUEST_LEASE,
            limit=16,
        )
        == []
    )
    first.complete_processor_request(
        claimed[0].request_id,
        "pod-a",
        leadership.epoch,
        claimed[0].lease_token,
        response_status=200,
        response_content_type="application/json",
        response_body_base64="e30=",
    )
    next_claim = second.claim_processor_requests(
        "pod-a",
        leadership.epoch,
        now=now + timedelta(milliseconds=2),
        lease_duration=REQUEST_LEASE,
        limit=16,
    )
    assert [item.cluster_id for item in next_claim] == ["cluster-a"]

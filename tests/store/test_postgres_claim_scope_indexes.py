"""Native scope-index plans and nullable counterpart membership semantics."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from gpu_fault.channel_registry import NVIDIA_KERNEL_PATH, WORKLOAD_OBSERVATIONS_PATH
from tests.store import test_postgres_claim_progress as progress
from tests.store._postgres_processor_claim_support import POSTGRES_URL, REQUEST_LEASE
from tests.store.test_postgres_claim_progress import claim
from tests.store.test_postgres_claim_progress import request as queue_request

SCOPE_INDEX = "gpu_fault_processor_queue_correlation_scopes"
store = progress.store
pytestmark = pytest.mark.skipif(
    not POSTGRES_URL, reason="requires the parent-allocated PostgreSQL"
)


@pytest.fixture
def connection():
    import psycopg

    with psycopg.connect(POSTGRES_URL, autocommit=True) as connected:
        yield connected


@pytest.fixture
def plan_properties(request):
    return request.node.user_properties


def index_names(plan):
    from tests.store.test_postgres_state_tables import plan_nodes

    return {node["Index Name"] for node in plan_nodes(plan) if "Index Name" in node}


def explain(connection, query, parameters):
    return connection.execute("EXPLAIN (FORMAT JSON) " + query, parameters).fetchone()[
        0
    ]


def seed_backlog(connection, path, at, count=6000):
    with connection.cursor() as cursor:
        with cursor.copy(
            "COPY gpu_fault_processor_queue "
            "(request_id, status, cluster_id, correlation_key, ordering_key, "
            "priority, created_at, updated_at, payload) FROM STDIN"
        ) as copy:
            for index in range(count):
                item = queue_request(
                    path,
                    "scope-backlog",
                    f"scope-backlog-{index:05d}",
                    at,
                    job_id=f"job-{index}",
                    attempt_id=f"attempt-{index}",
                    node_id=f"node-{index}",
                )
                copy.write_row(
                    (
                        item.request_id,
                        item.status.value,
                        item.cluster_id,
                        item.correlation_key,
                        item.ordering_key(),
                        item.queue_priority(),
                        item.created_at,
                        item.updated_at,
                        item.model_dump_json(),
                    )
                )
    connection.execute("ANALYZE gpu_fault_processor_queue")


@pytest.mark.parametrize(
    ("claimed_path", "backlog_path"),
    [
        (NVIDIA_KERNEL_PATH, WORKLOAD_OBSERVATIONS_PATH),
        (WORKLOAD_OBSERVATIONS_PATH, NVIDIA_KERNEL_PATH),
    ],
)
def test_actual_claim_uses_scope_gin_in_both_interlock_directions(
    store, connection, claimed_path, backlog_path, plan_properties
):
    at = datetime.now(timezone.utc)
    seed_backlog(connection, backlog_path, at)
    target = queue_request(
        claimed_path,
        "target-cluster",
        "target",
        at,
        job_id="target-job",
        attempt_id="target-attempt",
        node_id="target-node",
    )
    store.enqueue_processor_request(target)
    connection.execute("ANALYZE gpu_fault_processor_queue")
    query, parameters = store.claim_active_processor_query(
        "scope-owner",
        now=at,
        lease_duration=REQUEST_LEASE,
        limit=4,
        include_paths={claimed_path},
    )

    fresh_plan = explain(connection, query, parameters)
    # COPY leaves a large pending list that changes GIN startup cost. Preserve
    # both plans; test index selection after normal index maintenance as well.
    flushed = connection.execute(
        "SELECT gin_clean_pending_list(%s::regclass)", (SCOPE_INDEX,)
    ).fetchone()[0]
    plan = explain(connection, query, parameters)
    plan_properties.extend(
        [
            ("fresh_scope_gin_selected", SCOPE_INDEX in index_names(fresh_plan)),
            ("gin_pending_pages_flushed", flushed),
            ("scope_plan", json.dumps(plan)),
        ]
    )

    assert SCOPE_INDEX in index_names(plan), json.dumps(
        {"pending_pages_flushed": flushed, "plan": plan}, indent=2
    )


def test_wrapped_membership_cannot_use_the_existing_scope_expression(store, connection):
    at = datetime.now(timezone.utc)
    seed_backlog(connection, WORKLOAD_OBSERVATIONS_PATH, at)
    scope = json.dumps(["scope-backlog", "node", "node-3000"], separators=(",", ":"))
    query = (
        "SELECT request_id FROM gpu_fault_processor_queue "
        "WHERE status IN ('PENDING', 'LEASED') AND {} ? %s"
    )
    wrapped = explain(
        connection,
        query.format("coalesce(payload->'correlation_scope_keys', '[]'::jsonb)"),
        (scope,),
    )
    aligned = explain(
        connection, query.format("payload->'correlation_scope_keys'"), (scope,)
    )

    assert SCOPE_INDEX not in index_names(wrapped), (
        "the counterfactual wrapped expression must demonstrate the mismatch"
    )
    assert SCOPE_INDEX in index_names(aligned), (
        "query alignment must make the existing index usable without DDL"
    )


def replace_scopes(connection, request_id, value):
    if value == "missing":
        connection.execute(
            "UPDATE gpu_fault_processor_queue "
            "SET payload=payload-'correlation_scope_keys' WHERE request_id=%s",
            (request_id,),
        )
    else:
        connection.execute(
            "UPDATE gpu_fault_processor_queue SET payload="
            "jsonb_set(payload, '{correlation_scope_keys}', %s::jsonb) "
            "WHERE request_id=%s",
            (json.dumps(value), request_id),
        )


@pytest.mark.parametrize("scope_value", ["missing", None, [], ["foreign"], "matching"])
@pytest.mark.parametrize("observation_status", ["PENDING", "LEASED"])
def test_missing_null_or_empty_observation_scopes_keep_membership_semantics(
    store, connection, scope_value, observation_status
):
    at = datetime.now(timezone.utc)
    fault = queue_request(
        NVIDIA_KERNEL_PATH, "scope-cluster", "fault", at, node_id="scope-node"
    )
    observation = queue_request(
        WORKLOAD_OBSERVATIONS_PATH,
        "scope-cluster",
        "observation",
        at,
        attempt_id="scope-attempt",
        node_id="scope-node",
    )
    store.enqueue_processor_request(fault)
    store.enqueue_processor_request(observation)
    replace_scopes(
        connection,
        observation.request_id,
        fault.correlation_scope_keys if scope_value == "matching" else scope_value,
    )
    if observation_status == "LEASED":
        connection.execute(
            "UPDATE gpu_fault_processor_queue SET status='LEASED', "
            "lease_expires_at=%s WHERE request_id=%s",
            (at + REQUEST_LEASE, observation.request_id),
        )

    held = store.count_fault_rows_blocked_by_observation(now=at)
    rows = claim(store, at, include_paths={NVIDIA_KERNEL_PATH})

    assert held == int(scope_value == "matching"), (
        "NULL, absent, empty and unrelated counterpart scopes must not match"
    )
    assert [row.request_id for row in rows] == (
        [] if scope_value == "matching" else [fault.request_id]
    ), "query alignment must preserve the observation-before-fault decision"


@pytest.mark.parametrize("scope_value", ["missing", None, [], ["foreign"], "matching"])
def test_missing_null_or_empty_fault_scopes_preserve_observation_priority(
    store, connection, scope_value
):
    at = datetime.now(timezone.utc)
    target = queue_request(
        WORKLOAD_OBSERVATIONS_PATH,
        "scope-cluster",
        "target",
        at,
        attempt_id="target-attempt",
        node_id="target-node",
    )
    older = queue_request(
        WORKLOAD_OBSERVATIONS_PATH,
        "scope-cluster",
        "older",
        at - timedelta(seconds=1),
        attempt_id="older-attempt",
        node_id="older-node",
    )
    fault = queue_request(
        NVIDIA_KERNEL_PATH, "scope-cluster", "fault", at, node_id="target-node"
    )
    for item in (target, older, fault):
        store.enqueue_processor_request(item)
    replace_scopes(
        connection,
        fault.request_id,
        target.correlation_scope_keys if scope_value == "matching" else scope_value,
    )

    rows = claim(store, at, limit=1, include_paths={WORKLOAD_OBSERVATIONS_PATH})

    assert [row.request_id for row in rows] == [
        target.request_id if scope_value == "matching" else older.request_id
    ], "only a real scope match may promote the newer observation"

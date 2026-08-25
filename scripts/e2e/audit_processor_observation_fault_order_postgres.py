from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from uuid import uuid4

import psycopg
from psycopg import sql

from gpu_fault.processor import ProcessorRequest
from gpu_fault.store import PostgresStore


def schema_url(url: str, schema: str) -> str:
    parsed = urlsplit(url)
    query = parse_qsl(parsed.query, keep_blank_values=True)
    query.append(("options", f"-csearch_path={schema}"))
    return urlunsplit(
        (
            parsed.scheme,
            parsed.netloc,
            parsed.path,
            urlencode(query),
            parsed.fragment,
        )
    )


def request(path: str, body: bytes) -> ProcessorRequest:
    return ProcessorRequest.from_http(
        method="POST",
        path=path,
        query="",
        body=body,
        content_type="application/json",
        cluster_id="audit-cluster",
    )


def main() -> None:
    url = os.environ["GPU_FAULT_STORE_URL"]
    schema = f"gpu_fault_audit_order_{uuid4().hex[:12]}"
    with psycopg.connect(url, autocommit=True) as connection:
        connection.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
    store = PostgresStore(
        schema_url(url, schema),
        pool_min_size=1,
        pool_max_size=8,
        pool_timeout_seconds=10,
    )
    try:
        fault = request(
            "/v1/collector-events/nvidia-kernel",
            b'{"node_id":"node-a"}',
        )
        observation = request(
            "/v1/workload-observations",
            (
                b'{"job_id":"job-a","attempt_id":"attempt-a",'
                b'"containers":[{"node_id":"node-a"}]}'
            ),
        )
        store.enqueue_processor_request(fault)
        store.enqueue_processor_request(observation)
        now = datetime.now(timezone.utc)
        first = store.claim_active_processor_requests(
            "audit-owner",
            now=now,
            lease_duration=timedelta(seconds=30),
            limit=8,
        )
        assert [item.request_id for item in first] == [observation.request_id]
        store.complete_active_processor_request(
            observation.request_id,
            "audit-owner",
            first[0].leader_epoch,
            first[0].lease_token,
            response_status=200,
            response_content_type="application/json",
            response_body_base64="e30=",
        )
        second = store.claim_active_processor_requests(
            "audit-owner-2",
            now=now + timedelta(seconds=1),
            lease_duration=timedelta(seconds=30),
            limit=8,
        )
        assert [item.request_id for item in second] == [fault.request_id]

        unrelated_fault = request(
            "/v1/collector-events/fabric-manager",
            b'{"node_id":"node-c"}',
        )
        unrelated_observation = request(
            "/v1/workload-observations",
            (
                b'{"job_id":"job-b","attempt_id":"attempt-b",'
                b'"containers":[{"node_id":"node-d"}]}'
            ),
        )
        store.enqueue_processor_request(unrelated_fault)
        store.enqueue_processor_request(unrelated_observation)
        third = store.claim_active_processor_requests(
            "audit-owner-3",
            now=now + timedelta(seconds=2),
            lease_duration=timedelta(seconds=30),
            limit=8,
        )
        assert {item.request_id for item in third} == {
            unrelated_fault.request_id,
            unrelated_observation.request_id,
        }
        with store._db.cursor() as cursor:
            cursor.execute(
                """
                SELECT count(*)
                FROM pg_indexes
                WHERE schemaname=current_schema()
                  AND indexname=
                      'gpu_fault_processor_queue_correlation_scopes'
                """
            )
            index_count = cursor.fetchone()[0]
        assert index_count == 1
        print(
            json.dumps(
                {
                    "same_node_first_claim": [item.path for item in first],
                    "same_node_second_claim": [item.path for item in second],
                    "different_node_parallel_claim": sorted(
                        item.path for item in third
                    ),
                    "correlation_index_count": index_count,
                },
                indent=2,
                sort_keys=True,
            )
        )
    finally:
        store.close()
        with psycopg.connect(url, autocommit=True) as connection:
            connection.execute(
                sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema))
            )


if __name__ == "__main__":
    main()

"""The three read-only stuck-workflow criteria against a real Postgres.

``scripts/e2e/regional/audit_stuck_workflow_baseline.py`` is the acceptance
baseline for the workflow-review fixes (FINAL-建议汇总 F-B3 / F-D5 / F-A2). Its
queries encode judgement calls -- a same-generation twin *is* an orphan, a queued
successor's pending predecessor is *not*, a lease that merely looks old is not a
zombie until it has expired -- and those calls are only testable on Postgres,
where the jsonb predicates actually run. Each case seeds one shape that must be
counted next to one that must not.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from gpu_fault.models import IncidentState, WorkflowStatus
from tests._builders import fault_incident, processor_request, workflow_request
from tests._script_loader import lazy_script_module
from tests.store._postgres_processor_claim_support import postgres_store_instance

POSTGRES_URL = os.getenv("GPU_FAULT_TEST_POSTGRES_URL")
pytestmark = pytest.mark.skipif(
    not POSTGRES_URL, reason="GPU_FAULT_TEST_POSTGRES_URL is not configured"
)

ROOT = Path(__file__).resolve().parents[2]
AUDIT = lazy_script_module(
    ROOT / "scripts/e2e/regional/audit_stuck_workflow_baseline.py"
)

NOW = datetime(2026, 9, 5, 15, 0, tzinfo=timezone.utc)
HOUR_AGO = NOW - timedelta(hours=1)


@pytest.fixture
def store():
    yield from postgres_store_instance()


def _incident(incident_id: str, *, points_at: str, token: int):
    return fault_incident(
        incident_id,
        f"event-{incident_id}",
        state=IncidentState.ACTION_PENDING,
        workflow_request_id=points_at,
        fencing_token=token,
        created_at=HOUR_AGO,
        updated_at=HOUR_AGO,
    )


def _pending(
    request_id: str,
    incident_id: str,
    *,
    token: int,
    created_at: datetime = HOUR_AGO,
    updated_at: datetime = HOUR_AGO,
    **values,
):
    return workflow_request(
        request_id,
        incident_id,
        status=WorkflowStatus.PENDING,
        fencing_token=token,
        created_at=created_at,
        updated_at=updated_at,
        **values,
    )


def _collect(store):
    with store._db.cursor() as cursor:
        return AUDIT.collect(cursor, now=NOW)


def test_orphan_counts_the_same_generation_twin_but_not_a_queued_successor(
    store,
) -> None:
    # Twin: the incident names B; A shares incident and generation and nobody
    # links to it. Nothing will ever dispatch, fence or sweep A.
    store.save_incident_and_workflow(
        _incident("inc-twin", points_at="wf-b", token=2),
        _pending("wf-b", "inc-twin", token=2),
    )
    store.save_workflow(_pending("wf-a", "inc-twin", token=2))
    # Queued successor: the incident names S, and S names P as predecessor, so
    # P is somebody's business and must not be reported.
    store.save_incident_and_workflow(
        _incident("inc-queue", points_at="wf-s", token=1),
        _pending("wf-s", "inc-queue", token=1, predecessor_workflow_id="wf-p"),
    )
    store.save_workflow(_pending("wf-p", "inc-queue", token=1))
    # A lower-generation leftover is an orphan too, but not a same-generation one.
    store.save_incident_and_workflow(
        _incident("inc-old", points_at="wf-new", token=4),
        _pending("wf-new", "inc-old", token=4),
    )
    store.save_workflow(_pending("wf-old", "inc-old", token=1))
    # Inside the aggregation window the twin shape is still normal churn.
    store.save_incident_and_workflow(
        _incident("inc-fresh", points_at="wf-fresh-b", token=2),
        _pending("wf-fresh-b", "inc-fresh", token=2),
    )
    store.save_workflow(
        _pending("wf-fresh-a", "inc-fresh", token=2, created_at=NOW, updated_at=NOW)
    )

    result = _collect(store)

    assert result["orphan"] == {"total": 2, "same_generation": 1}


def test_zombie_counts_only_expired_leases(store) -> None:
    for request_id, expires in (
        ("req-expired", NOW - timedelta(minutes=5)),
        ("req-live", datetime.now(timezone.utc) + timedelta(hours=1)),
    ):
        request = processor_request(
            "/v1/gpu-events/xid", body=b'{"node_id":"node-a"}', cluster_id="cluster-a"
        ).model_copy(update={"request_id": request_id})
        with store._db.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO gpu_fault_processor_queue (
                    request_id, status, cluster_id, ordering_key, priority,
                    lease_owner, leader_epoch, lease_token, lease_expires_at,
                    created_at, updated_at, payload
                )
                VALUES (%s, 'LEASED', %s, %s, %s, 'worker-1', 1, 'tok', %s,
                        %s, %s, %s::jsonb)
                """,
                (
                    request_id,
                    request.cluster_id,
                    f"lane-{request_id}",
                    request.queue_priority(),
                    expires,
                    HOUR_AGO,
                    HOUR_AGO,
                    request.model_dump_json(),
                ),
            )

    result = _collect(store)

    assert result["zombie"] == 1


def test_stuck_pending_needs_a_passed_not_before_a_terminal_predecessor_and_idleness(
    store,
) -> None:
    store.save_incident_and_workflow(
        _incident("inc-stuck", points_at="wf-stuck", token=1),
        _pending("wf-stuck", "inc-stuck", token=1),
    )
    store.save_incident_and_workflow(
        _incident("inc-wait", points_at="wf-wait", token=1),
        _pending("wf-wait", "inc-wait", token=1, not_before=NOW + timedelta(minutes=5)),
    )
    store.save_incident_and_workflow(
        _incident("inc-busy", points_at="wf-busy", token=1),
        _pending("wf-busy", "inc-busy", token=1, updated_at=NOW),
    )
    store.save_incident_and_workflow(
        _incident("inc-behind", points_at="wf-behind", token=1),
        _pending(
            "wf-behind", "inc-behind", token=1, predecessor_workflow_id="wf-running"
        ),
    )
    store.save_workflow(
        workflow_request(
            "wf-running",
            "inc-behind",
            status=WorkflowStatus.RUNNING,
            fencing_token=1,
            created_at=HOUR_AGO,
            updated_at=HOUR_AGO,
        )
    )
    store.save_incident_and_workflow(
        _incident("inc-released", points_at="wf-released", token=1),
        _pending(
            "wf-released", "inc-released", token=1, predecessor_workflow_id="wf-done"
        ),
    )
    store.save_workflow(
        workflow_request(
            "wf-done",
            "inc-released",
            status=WorkflowStatus.SUCCEEDED,
            fencing_token=1,
            created_at=HOUR_AGO,
            updated_at=HOUR_AGO,
        )
    )

    result = _collect(store)

    # wf-stuck (no predecessor) and wf-released (terminal predecessor) count;
    # a future not_before, a fresh touch and a RUNNING predecessor do not.
    assert result["stuck_pending"] == 2


def test_run_is_read_only_and_the_cli_reports_and_fails_closed(
    store, tmp_path, capsys
) -> None:
    store.save_incident_and_workflow(
        _incident("inc-twin", points_at="wf-b", token=2),
        _pending("wf-b", "inc-twin", token=2),
    )
    store.save_workflow(_pending("wf-a", "inc-twin", token=2))
    table = tmp_path / "baseline.md"

    assert POSTGRES_URL is not None
    code = AUDIT.main(
        [
            "--store-url",
            POSTGRES_URL,
            "--orphan-cutoff-seconds",
            "0",
            "--json",
            "--append-markdown",
            str(table),
            "--note",
            "seeded",
            "--fail-on-nonzero",
        ]
    )

    assert code == 1
    output = capsys.readouterr().out
    assert POSTGRES_URL not in output
    assert '"total": 1' in output
    # Both seeded PENDING rows are also idle and dispatchable, so the stuck
    # count is 2 while the orphan count stays 1: the columns are independent.
    row = table.read_text(encoding="utf-8").strip()
    assert row.endswith("| 1 / 1 | 0 | 2 | seeded |"), (
        'expected row.endswith("| 1 / 1 | 0 | 2 | seeded |") to be true'
    )
    # Nothing was written to the database by the audit itself.
    assert store.get_workflow("wf-a").status is WorkflowStatus.PENDING
    with store._db.cursor() as cursor:
        cursor.execute("SELECT count(*) FROM gpu_fault_objects WHERE kind='workflow'")
        assert cursor.fetchone()[0] == 2

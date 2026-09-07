from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest

from gpu_fault.fleet import (
    AgentHeartbeat,
    FleetRegistry,
    SignedAgentHeartbeat,
    sign_agent_heartbeat,
)
from gpu_fault.gpu_metrics import (
    GpuMetricBatch,
    GpuMetricSample,
    GpuMetricSource,
    GpuMetricsService,
)
from gpu_fault.managed_recovery import HyperPodNodeIdentity
from gpu_fault.models import (
    AdvisoryNotification,
    IncidentState,
    MarkerScope,
    NodeMarker,
    PlanStatus,
    RecoveryAction,
    RecoveryPlan,
    Severity,
    WorkflowOperation,
    WorkflowStatus,
)
from gpu_fault.policy import GpuFaultPolicyEngine, XidEvent
from gpu_fault.regional import RegionalClusterRegistration
from gpu_fault.schema_migrations import (
    POSTGRES_SCHEMA_MIGRATIONS,
    postgres_ddl_source_checksum,
)
from gpu_fault.store import (
    POSTGRES_SCHEMA_VERSION,
    PostgresStore,
    RemediationBudgetError,
    WorkflowLeaseError,
)
from gpu_fault.telemetry import EvidenceKind, EvidenceService
from gpu_fault.training_health import TrainingProgressHeartbeat
from gpu_fault.xid_correlation import XidCorrelationCoordinator
from tests._builders import (
    asgi_client,
    attempt_observation,
    container_observation,
    fault_incident,
    gpu_metric_batch,
    workflow_request,
    workflow_step,
)
from tests.store._blocked_backlog_support import (
    FLAWS,
    blocked,
    break_one_clause,
    restore,
)

POSTGRES_URL = os.getenv("GPU_FAULT_TEST_POSTGRES_URL")
pytestmark = pytest.mark.skipif(
    not POSTGRES_URL, reason="GPU_FAULT_TEST_POSTGRES_URL is not configured"
)


@pytest.fixture(scope="module", autouse=True)
def initialized_postgres_schema():
    if POSTGRES_URL is None:
        yield
        return
    store = PostgresStore(POSTGRES_URL)
    store.close()
    yield


def _store(*, initialize_schema: bool = False) -> PostgresStore:
    assert POSTGRES_URL is not None
    return PostgresStore(POSTGRES_URL, initialize_schema=initialize_schema)


def test_latest_migration_is_derived_from_current_ddl() -> None:
    latest = POSTGRES_SCHEMA_MIGRATIONS[-1]
    assert latest.ddl_checksum == postgres_ddl_source_checksum()
    assert latest.apply is not None


def test_postgres_registry_sync_migrates_legacy_record_before_model_decode() -> None:
    import psycopg

    assert POSTGRES_URL is not None
    cluster_id = f"legacy-registry-{uuid4().hex}"
    payload = {
        "cluster_id": cluster_id,
        "region": "us-west-2",
        "hyperpod_cluster_name": f"hyperpod-{cluster_id}",
        "eks_cluster_arn": (f"arn:aws:eks:us-west-2:123456789012:cluster/{cluster_id}"),
        "token_sha256": "a" * 64,
        "enabled": True,
        "allowed_namespaces": ["training"],
    }
    with psycopg.connect(POSTGRES_URL) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO gpu_fault_objects(kind, key, payload)
                VALUES ('regional_cluster', %s, %s::jsonb)
                """,
                (cluster_id, json.dumps(payload)),
            )

    store = _store()
    try:
        configured = store.save_regional_cluster(
            RegionalClusterRegistration(
                cluster_id=cluster_id,
                region="us-west-2",
                hyperpod_cluster_name=f"hyperpod-{cluster_id}",
                eks_cluster_arn=(
                    f"arn:aws:eks:us-west-2:123456789012:cluster/{cluster_id}"
                ),
                token_sha256="b" * 64,
                allowed_namespaces=["training"],
                agent_endpoint_allowed_cidrs=["10.0.0.0/16"],
            )
        )

        assert configured.agent_endpoint_allowed_cidrs == ["10.0.0.0/16"]
        assert store.get_regional_cluster(cluster_id).agent_endpoint_allowed_cidrs == [
            "10.0.0.0/16"
        ]
    finally:
        store.delete_regional_cluster(cluster_id)
        store.close()


def test_postgres_processor_queue_has_no_virtual_partition_state() -> None:
    store = _store()
    try:
        with store._db.cursor() as cursor:
            cursor.execute(
                """
                SELECT EXISTS (
                    SELECT 1
                    FROM information_schema.columns
                    WHERE table_schema=current_schema()
                      AND table_name='gpu_fault_processor_queue'
                      AND column_name='partition_id'
                )
                """
            )
            column_exists = cursor.fetchone()[0]
            cursor.execute(
                """
                SELECT indexname
                FROM pg_indexes
                WHERE schemaname=current_schema()
                  AND indexname IN (
                      'gpu_fault_processor_queue_claim',
                      'gpu_fault_processor_queue_partition_claim'
                  )
                """
            )
            obsolete_indexes = cursor.fetchall()
            cursor.execute(
                """
                SELECT pg_get_functiondef(
                    'gpu_fault_processor_queue_notify_pending()'
                    ::regprocedure
                )
                """
            )
            notify_function = cursor.fetchone()[0]
    finally:
        store.close()

    assert column_exists is False
    assert obsolete_indexes == []
    assert "'partition'" not in notify_function


def test_postgres_processor_retry_schedule_schema_is_present() -> None:
    store = _store()
    store.close()
    assert POSTGRES_URL is not None
    import psycopg

    with psycopg.connect(POSTGRES_URL) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT column_name
                FROM information_schema.columns
                WHERE table_schema=current_schema()
                  AND table_name='gpu_fault_processor_queue'
                  AND column_name IN (
                      'not_before',
                      'retry_count',
                      'lane_policy'
                  )
                ORDER BY column_name
                """
            )
            columns = [row[0] for row in cursor.fetchall()]
            cursor.execute(
                """
                SELECT to_regclass(
                    'gpu_fault_processor_queue_available'
                )
                """
            )
            available_index = cursor.fetchone()[0]

    assert columns == ["lane_policy", "not_before", "retry_count"]
    assert available_index == "gpu_fault_processor_queue_available"


def test_postgres_schema_migration_apply_callback_runs() -> None:
    import psycopg

    store = _store()
    store.close()
    assert POSTGRES_URL is not None
    with psycopg.connect(POSTGRES_URL) as connection:
        comment = connection.execute(
            """
            SELECT obj_description(
                'gpu_fault_schema_migrations'::regclass
            )
            """
        ).fetchone()[0]
    assert comment == "Monotonic gpu-fault schema migration history"


def test_postgres_marker_lookup_is_incident_scoped() -> None:
    store = _store()
    suffix = uuid4().hex
    observed_at = datetime.now(timezone.utc)
    try:
        for index, incident_id in enumerate(
            (f"incident-a-{suffix}", f"incident-b-{suffix}")
        ):
            store.add_marker(
                NodeMarker(
                    marker_id=f"marker-{index}-{suffix}",
                    source="postgres-test",
                    trusted=True,
                    incident_id=incident_id,
                    observed_at=observed_at,
                    expires_at=observed_at + timedelta(hours=1),
                    scope=MarkerScope(node_ids=["node-a"]),
                    severity=Severity.CRITICAL,
                    recommended_action=(RecoveryAction.REBOOT_NODE),
                    action_owner="test",
                    mapping_version="test-v1",
                )
            )

        markers = store.list_markers_for_incident(f"incident-a-{suffix}")
    finally:
        store.close()

    assert [marker.marker_id for marker in markers] == [f"marker-0-{suffix}"]


def test_postgres_schema_version_fails_closed() -> None:
    import psycopg

    store = _store()
    store.close()
    assert POSTGRES_URL is not None
    with psycopg.connect(POSTGRES_URL, autocommit=True) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                UPDATE gpu_fault_schema_version
                SET version=%s
                WHERE singleton=TRUE
                """,
                (POSTGRES_SCHEMA_VERSION + 1,),
            )
    try:
        with pytest.raises(RuntimeError, match="schema version mismatch"):
            PostgresStore(POSTGRES_URL, initialize_schema=False)
    finally:
        repaired = _store(initialize_schema=True)
        repaired.close()


def test_postgres_schema_migration_checksum_fails_closed() -> None:
    import psycopg

    store = _store()
    store.close()
    migration = POSTGRES_SCHEMA_MIGRATIONS[-1]
    assert POSTGRES_URL is not None
    with psycopg.connect(POSTGRES_URL, autocommit=True) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                UPDATE gpu_fault_schema_migrations
                SET checksum=%s
                WHERE version=%s
                """,
                ("0" * 64, migration.version),
            )
    try:
        with pytest.raises(RuntimeError, match="migration history mismatch"):
            PostgresStore(POSTGRES_URL, initialize_schema=False)
    finally:
        with psycopg.connect(POSTGRES_URL, autocommit=True) as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    UPDATE gpu_fault_schema_migrations
                    SET name=%s, checksum=%s
                    WHERE version=%s
                    """,
                    (migration.name, migration.checksum, migration.version),
                )


def test_postgres_missing_spool_path_index_fails_closed() -> None:
    import psycopg

    store = _store()
    store.close()
    assert POSTGRES_URL is not None
    with psycopg.connect(POSTGRES_URL, autocommit=True) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                DROP INDEX IF EXISTS
                gpu_fault_telemetry_spool_path_available
                """
            )
    try:
        with pytest.raises(RuntimeError, match="schema is not initialized"):
            PostgresStore(POSTGRES_URL, initialize_schema=False)
    finally:
        repaired = _store(initialize_schema=True)
        repaired.close()


def test_postgres_missing_processor_notify_trigger_fails_closed() -> None:
    import psycopg

    store = _store()
    store.close()
    assert POSTGRES_URL is not None
    with psycopg.connect(POSTGRES_URL, autocommit=True) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                DROP TRIGGER IF EXISTS
                gpu_fault_processor_queue_notify_pending_trigger
                ON gpu_fault_processor_queue
                """
            )
    try:
        with pytest.raises(RuntimeError, match="notification trigger is missing"):
            PostgresStore(POSTGRES_URL, initialize_schema=False)
    finally:
        repaired = _store(initialize_schema=True)
        repaired.close()


def test_postgres_missing_spool_notify_trigger_fails_closed() -> None:
    import psycopg

    store = _store()
    store.close()
    assert POSTGRES_URL is not None
    with psycopg.connect(POSTGRES_URL, autocommit=True) as connection:
        connection.execute(
            """
            DROP TRIGGER IF EXISTS
            gpu_fault_telemetry_spool_notify_available_trigger
            ON gpu_fault_telemetry_spool
            """
        )
    try:
        with pytest.raises(
            RuntimeError, match="telemetry spool notification trigger is missing"
        ):
            PostgresStore(POSTGRES_URL, initialize_schema=False)
    finally:
        repaired = _store(initialize_schema=True)
        repaired.close()


def test_postgres_missing_fault_counter_trigger_fails_closed() -> None:
    import psycopg

    store = _store()
    store.close()
    assert POSTGRES_URL is not None
    with psycopg.connect(POSTGRES_URL, autocommit=True) as connection:
        connection.execute(
            """
            DROP TRIGGER IF EXISTS
            gpu_fault_processor_priority_count_insert
            ON gpu_fault_processor_queue
            """
        )
    try:
        with pytest.raises(
            RuntimeError, match="fault counter shard triggers are missing"
        ):
            PostgresStore(POSTGRES_URL, initialize_schema=False)
    finally:
        repaired = _store(initialize_schema=True)
        repaired.close()


def test_postgres_persists_hyperpod_identity() -> None:
    store = _store()
    logical_id = f"worker-{uuid4()}"
    identity = HyperPodNodeIdentity(
        cluster_name="hp-integration",
        node_logical_id=logical_id,
        instance_id="i-integration",
        kubernetes_node_name="k8s-integration",
        status="Running",
        aliases=[logical_id, "i-integration"],
        retired_aliases=["i-retired"],
        observed_at=datetime.now(timezone.utc),
    )

    try:
        store.save_hyperpod_node_identity(identity)
        loaded = store.get_hyperpod_node_identity("hp-integration", logical_id)
    finally:
        store.close()

    assert loaded == identity


def test_postgres_lease_takeover_fences_stale_writer() -> None:
    first = _store()
    second = _store()
    request_id = f"workflow-{uuid4()}"
    incident_id = f"incident-{uuid4()}"
    now = datetime.now(timezone.utc)
    lease = timedelta(seconds=30)
    incident = fault_incident(
        incident_id,
        f"event-{uuid4()}",
        state=IncidentState.ACTION_PENDING,
        fencing_token=3,
        created_at=now,
        updated_at=now,
    )
    workflow = workflow_request(request_id, incident_id, created_at=now, updated_at=now)
    first.save_incident(incident)
    first.save_workflow(workflow)

    try:
        claimed_a = first.claim_workflow(
            request_id, "executor-a", 3, now=now, lease_duration=lease
        )
        with pytest.raises(WorkflowLeaseError, match="another executor"):
            second.claim_workflow(
                request_id,
                "executor-b",
                3,
                now=now + timedelta(seconds=10),
                lease_duration=lease,
            )

        claimed_b = second.claim_workflow(
            request_id,
            "executor-b",
            3,
            now=now + timedelta(seconds=31),
            lease_duration=lease,
        )

        assert claimed_a.execution_epoch == 1
        assert claimed_b.execution_epoch == 2
        assert claimed_b.execution_owner_id == "executor-b"
        with pytest.raises(WorkflowLeaseError, match="stale"):
            first.save_workflow_if_leased(
                claimed_a.model_copy(update={"status": WorkflowStatus.SUCCEEDED}),
                "executor-a",
                claimed_a.execution_epoch,
                now=now + timedelta(seconds=32),
            )
        assert second.get_workflow(request_id).execution_owner_id == "executor-b"
        terminal_workflow = claimed_b.model_copy(
            update={
                "status": WorkflowStatus.SUCCEEDED,
                "execution_owner_id": None,
                "execution_lease_expires_at": None,
            }
        )
        terminal_incident = incident.model_copy(
            update={"state": IncidentState.RECOVERED}
        )
        second.save_workflow_and_incident_if_leased(
            terminal_workflow,
            terminal_incident,
            "executor-b",
            claimed_b.execution_epoch,
            now=now + timedelta(seconds=32),
        )
        assert first.get_workflow(request_id).status is WorkflowStatus.SUCCEEDED
        assert first.get_incident(incident_id).state is IncidentState.RECOVERED
    finally:
        first.close()
        second.close()


def test_postgres_remediation_budget_claim_is_atomic() -> None:
    first = _store()
    second = _store()
    suffix = uuid4().hex
    now = datetime.now(timezone.utc)

    def save(store, name):
        incident = fault_incident(
            f"incident-budget-{name}-{suffix}",
            f"event-budget-{name}-{suffix}",
            state=IncidentState.ACTION_PENDING,
            fencing_token=3,
            created_at=now,
            updated_at=now,
        )
        workflow = workflow_request(
            f"workflow-budget-{name}-{suffix}",
            incident.incident_id,
            WorkflowStatus.PENDING,
            fencing_token=3,
            official_steps=[workflow_step(WorkflowOperation.RESTART_NODE)],
            created_at=now,
            updated_at=now,
        )
        incident = incident.model_copy(
            update={"workflow_request_id": workflow.request_id}
        )
        store.save_incident_and_workflow(incident, workflow)
        return workflow

    try:
        left = save(first, "left")
        right = save(second, "right")
        claims = {"region": 1, "cluster:cluster-a": 1}
        first.claim_workflow(
            left.request_id,
            "executor-left",
            left.fencing_token,
            remediation_budget_claims=claims,
        )
        with pytest.raises(RemediationBudgetError, match="cluster:cluster-a") as raised:
            second.claim_workflow(
                right.request_id,
                "executor-right",
                right.fencing_token,
                remediation_budget_claims=claims,
            )
        assert raised.value.scope == "cluster:cluster-a"
        blocked = second.get_workflow(right.request_id)
        assert blocked.remediation_budget_wait_count == 1
        assert blocked.remediation_budget_last_blocked_scope == "cluster:cluster-a"
    finally:
        first.close()
        second.close()


def test_postgres_restored_workflow_reconcile_is_atomic() -> None:
    store = _store()
    suffix = uuid4().hex
    now = datetime.now(timezone.utc)
    incident_id = f"incident-reconcile-{suffix}"
    blocked_id = f"workflow-reconcile-blocked-{suffix}"
    successor_id = f"workflow-reconcile-restored-{suffix}"
    plan_id = f"plan-reconcile-{suffix}"
    plan = RecoveryPlan(
        plan_id=plan_id,
        incident_id=incident_id,
        attempt_id=f"attempt-{suffix}",
        trigger="postgres reconciliation contract",
        runtime_profile_version="profile-v1",
        steps=[],
        workflow_request_id=blocked_id,
        status=PlanStatus.FAILED,
        created_at=now - timedelta(hours=2),
    )
    blocked = workflow_request(
        blocked_id,
        incident_id,
        WorkflowStatus.BLOCKED,
        fencing_token=7,
        source_plan_id=plan_id,
        official_steps=[workflow_step(WorkflowOperation.QUARANTINE)],
        created_at=now - timedelta(hours=2),
        updated_at=now - timedelta(hours=1),
    )
    successor = workflow_request(
        successor_id,
        incident_id,
        WorkflowStatus.SUCCEEDED,
        fencing_token=7,
        predecessor_workflow_id=blocked_id,
        completed_operations=[WorkflowOperation.RESTORE_SCHEDULING],
        created_at=now - timedelta(hours=1),
        updated_at=now - timedelta(minutes=30),
    )
    incident = fault_incident(
        incident_id,
        f"event-reconcile-{suffix}",
        state=IncidentState.RECOVERED,
        workflow_request_id=successor_id,
        fencing_token=7,
        created_at=now - timedelta(hours=2),
        updated_at=now - timedelta(minutes=30),
    )
    try:
        store.save_plan(plan)
        store.save_workflow(blocked)
        store.save_workflow(successor)
        store.save_incident(incident)

        updated, current_incident, updated_plan = store.reconcile_restored_workflow(
            blocked_id,
            successor_id,
            expected_fencing_token=blocked.fencing_token,
            expected_execution_epoch=blocked.execution_epoch,
            reference="CHG-POSTGRES-RECONCILE",
            reconciled_at=now,
        )

        assert updated.status is WorkflowStatus.SUPERSEDED
        assert current_incident.workflow_request_id == successor_id
        assert updated_plan.resolved_by_restore_workflow_id == successor_id
        assert store.get_workflow(blocked_id).status is WorkflowStatus.SUPERSEDED
        assert store.get_incident(incident_id).workflow_request_id == successor_id
        assert (
            store.get_plan(plan_id).reconciliation_reference == "CHG-POSTGRES-RECONCILE"
        )
    finally:
        store.close()


def test_postgres_replacement_group_merge_is_concurrent() -> None:
    first = _store()
    second = _store()
    group_key = f"group-{uuid4()}"
    incident_id = f"incident-{uuid4()}"
    workflow_id = f"workflow-{uuid4()}"

    def merge(store: PostgresStore, node_id: str):
        event_id = f"event-{node_id}-{uuid4()}"

        def build(existing_incident, existing_workflow):
            now = datetime.now(timezone.utc)
            nodes = sorted(
                {
                    *(
                        existing_incident.node_ids
                        if existing_incident is not None
                        else []
                    ),
                    node_id,
                }
            )
            incident = fault_incident(
                existing_incident.incident_id
                if existing_incident is not None
                else incident_id,
                existing_incident.event_id
                if existing_incident is not None
                else event_id,
                "NODE_HEALTH_GROUP",
                node_ids=nodes,
                policy_version="site/v1",
                policy_source="SITE",
                effective_action=RecoveryAction.REPLACE_NODE,
                state=IncidentState.ACTION_PENDING,
                workflow_request_id=workflow_id,
                created_at=existing_incident.created_at
                if existing_incident is not None
                else now,
                updated_at=now,
            )
            workflow = workflow_request(
                existing_workflow.request_id
                if existing_workflow is not None
                else workflow_id,
                incident.incident_id,
                fencing_token=1,
                created_at=existing_workflow.created_at
                if existing_workflow is not None
                else now,
                updated_at=now,
            )
            return incident, workflow

        return store.merge_replacement_workflow(group_key, event_id, build)

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(
                pool.map(
                    lambda args: merge(*args), [(first, "node-a"), (second, "node-b")]
                )
            )

        assert len({incident.incident_id for incident, _ in results}) == 1
        assert first.get_incident(incident_id).node_ids == ["node-a", "node-b"]
    finally:
        first.close()
        second.close()


def test_postgres_sxid_group_merge_is_concurrent() -> None:
    first = _store()
    second = _store()
    group_key = f"sxid-group-{uuid4()}"
    incident_id = f"sxid-incident-{uuid4()}"
    workflow_id = f"sxid-workflow-{uuid4()}"

    def merge(store: PostgresStore, node_id: str):
        event_id = f"sxid-event-{node_id}-{uuid4()}"

        def build(existing_incident, existing_workflow):
            now = datetime.now(timezone.utc)
            nodes = sorted(
                {
                    *(
                        existing_incident.node_ids
                        if existing_incident is not None
                        else []
                    ),
                    node_id,
                }
            )
            incident = fault_incident(
                existing_incident.incident_id
                if existing_incident is not None
                else incident_id,
                existing_incident.event_id
                if existing_incident is not None
                else event_id,
                "SXID",
                node_ids=nodes,
                policy_version="sxid-test/v1",
                effective_action=RecoveryAction.RESET_GPU,
                state=IncidentState.ACTION_PENDING,
                workflow_request_id=workflow_id,
                created_at=existing_incident.created_at
                if existing_incident is not None
                else now,
                updated_at=now,
            )
            workflow = workflow_request(
                existing_workflow.request_id
                if existing_workflow is not None
                else workflow_id,
                incident.incident_id,
                fencing_token=1,
                created_at=existing_workflow.created_at
                if existing_workflow is not None
                else now,
                updated_at=now,
            )
            return incident, workflow

        return store.merge_sxid_workflow(group_key, event_id, build)

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(
                pool.map(
                    lambda args: merge(*args), [(first, "node-a"), (second, "node-b")]
                )
            )

        assert len({incident.incident_id for incident, _ in results}) == 1
        assert first.get_incident(incident_id).node_ids == ["node-a", "node-b"]
    finally:
        first.close()
        second.close()


def test_postgres_incident_workflow_create_is_concurrent() -> None:
    first = _store()
    second = _store()
    event_id = f"attempt-failure-{uuid4()}"

    def create(store: PostgresStore, suffix: str):
        def build():
            now = datetime.now(timezone.utc)
            incident_id = f"incident-{suffix}-{uuid4()}"
            workflow_id = f"workflow-{suffix}-{uuid4()}"
            return (
                fault_incident(
                    incident_id,
                    event_id,
                    "TRAINING_ATTEMPT_FAILURE_DETECTED",
                    policy_version="passive-containment-v1",
                    policy_source="completion-watcher",
                    effective_action=RecoveryAction.STOP_WORKLOAD,
                    state=IncidentState.ACTION_PENDING,
                    workflow_request_id=workflow_id,
                    created_at=now,
                    updated_at=now,
                ),
                workflow_request(
                    workflow_id,
                    incident_id,
                    fencing_token=1,
                    created_at=now,
                    updated_at=now,
                ),
            )

        return store.create_incident_workflow_if_absent(event_id, build)

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(
                pool.map(lambda args: create(*args), [(first, "a"), (second, "b")])
            )

        assert len({item[0].incident_id for item in results}) == 1
        assert len({item[1].request_id for item in results}) == 1
        assert sorted(item[2] for item in results) == [False, True]
    finally:
        first.close()
        second.close()


def test_postgres_notification_deduplication_is_concurrent() -> None:
    first = _store()
    second = _store()
    deduplication_key = f"dedup-{uuid4()}"

    def save(store: PostgresStore, suffix: str):
        return store.save_notification_if_absent(
            AdvisoryNotification(
                notification_id=f"notification-{suffix}-{uuid4()}",
                deduplication_key=deduplication_key,
                cluster_name="cluster-a",
                incident_id="incident-a",
                subject="GPU fault",
                body_text="GPU fault detected",
                support_case_draft="Support case",
            )
        )

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(
                pool.map(lambda args: save(*args), [(first, "a"), (second, "b")])
            )

        assert results[0].notification_id == results[1].notification_id
        matching = [
            item
            for item in first.list_notifications()
            if item.deduplication_key == deduplication_key
        ]
        assert len(matching) == 1
    finally:
        first.close()
        second.close()


def test_postgres_xid_metric_transition_has_one_winner() -> None:
    first = _store()
    second = _store()
    cluster_id = f"cluster-{uuid4()}"
    node_id = f"node-{uuid4()}"
    gpu_key = f"GPU-{uuid4()}"
    baseline_at = datetime.now(timezone.utc)
    changed_at = baseline_at + timedelta(seconds=15)

    try:
        assert not first.observe_xid_metric(
            cluster_id, node_id, gpu_key, 0, baseline_at
        )
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(
                pool.map(
                    lambda store: store.observe_xid_metric(
                        cluster_id, node_id, gpu_key, 94, changed_at
                    ),
                    [first, second],
                )
            )

        assert sorted(results) == [False, True]
        assert not first.observe_xid_metric(
            cluster_id, node_id, gpu_key, 94, changed_at + timedelta(seconds=15)
        )
    finally:
        first.close()
        second.close()


def test_postgres_gpu_counter_delta_is_shared_across_replicas() -> None:
    first_store = _store()
    second_store = _store()
    first = GpuMetricsService(store=first_store)
    second = GpuMetricsService(store=second_store)
    cluster_id = f"cluster-{uuid4()}"
    node_id = f"node-{uuid4()}"
    gpu_uuid = f"GPU-{uuid4()}"
    baseline_at = datetime.now(timezone.utc)
    changed_at = baseline_at + timedelta(seconds=15)

    def metric_batch(
        batch_id: str, value: float, observed_at: datetime
    ) -> GpuMetricBatch:
        return gpu_metric_batch(
            batch_id,
            observed_at,
            GpuMetricSource.DCGM_EXPORTER,
            [
                GpuMetricSample(
                    metric_name="DCGM_FI_DEV_PCIE_REPLAY_COUNTER",
                    canonical_name="pcie_replay_total",
                    value=value,
                    gpu_index="0",
                    gpu_uuid=gpu_uuid,
                    pci_bdf="0000:b9:00.0",
                )
            ],
            cluster_id=cluster_id,
            node_id=node_id,
        )

    try:
        baseline = first.ingest(metric_batch("baseline", 1000, baseline_at))
        assert baseline.accepted_samples == 1
        assert not baseline.findings

        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(
                pool.map(
                    lambda args: args[0].ingest(
                        metric_batch(args[1], 1200, changed_at)
                    ),
                    [(first, f"replica-a-{uuid4()}"), (second, f"replica-b-{uuid4()}")],
                )
            )

        assert sorted(result.accepted_samples for result in results) == [0, 1]
        winner = next(result for result in results if result.accepted_samples == 1)
        assert winner.findings[0].delta == 200
        assert winner.new_findings[0].canonical_name == ("pcie_replay_total")

        for service in (first, second):
            latest = service.latest(cluster_id, node_id)
            assert len(latest) == 1
            assert latest[0].sample.value == 1200
            active = service.findings(cluster_id, node_id)
            assert len(active) == 1
            assert active[0].delta == 200
    finally:
        first_store.close()
        second_store.close()


def test_postgres_training_topology_progress_and_evidence_are_shared() -> None:
    first = _store()
    second = _store()
    cluster_id = f"cluster-{uuid4()}"
    attempt_id = f"attempt-{uuid4()}"
    node_id = f"node-{uuid4()}"
    observed_at = datetime.now(timezone.utc)
    topology = attempt_observation(
        f"job-{uuid4()}",
        attempt_id,
        observed_at,
        cluster_id=cluster_id,
        workload_ids=["training/job/shared-state"],
        runtime_profile_version="profile-a",
        containers=[
            container_observation(
                f"pod-{uuid4()}", "worker-0", 0, node_id, gpu_uuids=["GPU-a"]
            )
        ],
    )
    initial = TrainingProgressHeartbeat(
        cluster_id=cluster_id,
        attempt_id=attempt_id,
        rank=0,
        observed_at=observed_at,
        node_id=node_id,
        step=10,
    )
    repeated = initial.model_copy(
        update={
            "heartbeat_id": f"heartbeat-{uuid4()}",
            "observed_at": observed_at + timedelta(seconds=30),
        }
    )

    try:
        assert first.save_attempt_observation(topology)
        observed = second.list_attempt_observations(cluster_id)
        assert observed[0].containers[0].gpu_uuids == ["GPU-a"]

        first.observe_training_progress(initial)
        second.observe_training_progress(repeated)
        state = first.list_training_progress_states(cluster_id, attempt_id)[0]
        assert state.heartbeat.observed_at == repeated.observed_at
        assert state.last_progress_at == observed_at

        EvidenceService(first).capture(
            record_id=f"evidence-{uuid4()}",
            cluster_id=cluster_id,
            node_id=node_id,
            kind=EvidenceKind.TRAINING_PROGRESS,
            observed_at=observed_at,
            attempt_ids=[attempt_id],
            payload={"step": 10},
        )
        evidence = second.list_raw_evidence(cluster_id, attempt_id=attempt_id)
        assert evidence[0].payload == {"step": 10}
    finally:
        first.close()
        second.close()


def test_postgres_expired_raw_evidence_cleanup() -> None:
    store = _store()
    cluster_id = f"cluster-{uuid4()}"
    try:
        EvidenceService(store, retention=timedelta(hours=1)).capture(
            record_id=f"evidence-{uuid4()}",
            cluster_id=cluster_id,
            node_id="node-a",
            kind=EvidenceKind.TRAINING_PROGRESS,
            observed_at=datetime.now(timezone.utc),
            attempt_ids=["attempt-a"],
            payload={"step": 1},
        )

        deleted = store.cleanup_expired_raw_evidence(
            now=datetime.now(timezone.utc) + timedelta(hours=2)
        )

        assert deleted == 1
        assert store.list_raw_evidence(cluster_id) == []
    finally:
        store.close()


def test_postgres_active_deployment_lookup_uses_the_scope_index() -> None:
    """The heartbeat path must not scan every deployment ever recorded.

    The pre-existing index leads with created_at across all clusters and
    all statuses, so it cannot serve the cluster predicate. This asserts
    both the retention drain and that the planner picks the partial
    index the drain leaves small.
    """
    from gpu_fault.fleet import (
        DeploymentNode,
        DeploymentNodeStatus,
        DeploymentStatus,
        FleetDeployment,
    )

    store = _store()
    cluster_id = f"cluster-{uuid4()}"
    now = datetime.now(timezone.utc)

    def deployment(status: DeploymentStatus) -> FleetDeployment:
        return FleetDeployment(
            cluster_id=cluster_id,
            desired_agent_version="0.9.0",
            desired_artifact_sha256="a" * 64,
            desired_policy_version="catalog-a",
            desired_runtime_profile_version="profile-a",
            desired_config_digest="c" * 64,
            max_unavailable=1,
            waves=[["node-a"]],
            nodes=[
                DeploymentNode(
                    node_id="node-a", status=DeploymentNodeStatus.READY, updated_at=now
                )
            ],
            status=status,
            created_at=now,
            updated_at=now,
        )

    try:
        open_roll = deployment(DeploymentStatus.IN_PROGRESS)
        done = deployment(DeploymentStatus.SUCCEEDED)
        failed = deployment(DeploymentStatus.FAILED)
        for record in (open_roll, done, failed):
            store.save_fleet_deployment(record)
        # The index earns its keep on a fleet with many clusters' deployments;
        # with only this cluster's three rows the primary key's ``kind=``
        # probe estimates one row too and the planner's pick depends on
        # whichever autoanalyze ran last. Give it the shape it exists for.
        for other in range(40):
            # Open rolls, so the retention assertions below still see exactly
            # this cluster's two terminal rows.
            store.save_fleet_deployment(
                deployment(DeploymentStatus.IN_PROGRESS).model_copy(
                    update={"cluster_id": f"cluster-other-{other}"}
                )
            )
        with store._db.cursor() as cursor:
            cursor.execute("ANALYZE gpu_fault_objects")

        assert [
            item.deployment_id
            for item in store.list_active_fleet_deployments(cluster_id)
        ] == [open_roll.deployment_id]
        with store._db.cursor() as cursor:
            cursor.execute("SET enable_seqscan=off")
            try:
                cursor.execute(
                    """
                    EXPLAIN
                    SELECT payload
                    FROM gpu_fault_objects
                    WHERE kind='fleet_deployment'
                      AND payload->>'cluster_id'=%s
                      AND payload->>'status'
                          NOT IN ('SUCCEEDED', 'FAILED')
                    ORDER BY payload->>'created_at', key
                    """,
                    (cluster_id,),
                )
                plan = "\n".join(row[0] for row in cursor.fetchall())
            finally:
                cursor.execute("SET enable_seqscan=on")

        assert "gpu_fault_fleet_deployment_active_scope" in plan, plan
        # Retention only takes the terminal rows, and honours its limit.
        assert (
            store.cleanup_terminal_fleet_deployments(
                older_than=now + timedelta(seconds=1), limit=1
            )
            == 1
        )
        assert (
            store.cleanup_terminal_fleet_deployments(
                older_than=now + timedelta(seconds=1), limit=10
            )
            == 1
        )
        assert (
            store.cleanup_terminal_fleet_deployments(
                older_than=now + timedelta(seconds=1), limit=10
            )
            == 0
        )
        assert [
            item.deployment_id
            for item in store.list_active_fleet_deployments(cluster_id)
        ] == [open_roll.deployment_id]
    finally:
        store.close()


def test_postgres_blocked_backlog_gauge_answers_the_same_question() -> None:
    """The ``jsonb`` rewrite of the BLOCKED backlog aggregate.

    ``blocked_workflows_without_verified_restore`` is hand-written per backend,
    and this is the one that runs in production. The records come from the same
    builders as ``tests/store/test_blocked_backlog_gauge.py``, so a clause the
    SQLite query checks and this one does not shows up as a disagreement rather
    than as a quietly cleared alert.

    Asserted as deltas against a baseline, and with record names scoped to a
    uuid: the gauge is a whole-table aggregate and this database is shared with
    every other test in the serial Postgres invocation.
    """
    store = _store()
    try:
        name = f"backlog-{uuid4().hex}"
        base = store.blocked_workflows_without_verified_restore()

        blocked(store, name)
        assert store.blocked_workflows_without_verified_restore() == base + 1

        restore(store, name)
        assert store.blocked_workflows_without_verified_restore() == base

        for flaw in FLAWS:
            restore(store, name)
            break_one_clause(store, flaw, name)
            assert store.blocked_workflows_without_verified_restore() == base + 1, flaw
    finally:
        store.close()


def test_postgres_agent_registration_serializes_node_lease() -> None:
    first = _store()
    second = _store()
    secret = "agent-registration-" + "x" * 32
    now = datetime.now(timezone.utc)
    node_id = f"node-{uuid4()}"
    cluster_id = f"cluster-{uuid4()}"
    registries = [
        FleetRegistry(first, secret, now=lambda: now),
        FleetRegistry(second, secret, now=lambda: now),
    ]

    def register(index: int):
        heartbeat = AgentHeartbeat(
            cluster_id=cluster_id,
            node_id=node_id,
            endpoint=f"http://10.0.0.{index + 1}:9099",
            agent_protocol_version=3,
            agent_version="0.10.0",
            artifact_sha256="a" * 64,
            policy_version="610",
            runtime_profile_version="profile-a",
            config_digest="c" * 64,
            allowed_operations=[
                WorkflowOperation.VERIFY_NO_GPU_CLIENTS,
                WorkflowOperation.RESET_GPU,
            ],
            boot_id=f"boot-{index}",
            node_instance_id=f"instance-{index}",
            agent_incarnation_id=f"incarnation-{index}",
            observed_at=now,
        )
        envelope = SignedAgentHeartbeat(
            heartbeat=heartbeat, signature=sign_agent_heartbeat(heartbeat, secret)
        )
        try:
            return registries[index].register(envelope)
        except ValueError as exc:
            return exc

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(register, [0, 1]))

        records = [result for result in results if not isinstance(result, Exception)]
        errors = [result for result in results if isinstance(result, Exception)]
        assert len(records) == 1
        assert len(errors) == 1
        assert "holds the node lease" in str(errors[0])
        stored = first.get_agent(cluster_id, node_id)
        assert stored.generation == 1
        assert stored.agent_incarnation_id in {"incarnation-0", "incarnation-1"}
    finally:
        first.close()
        second.close()


def test_postgres_xid45_correlation_claim_is_single_writer() -> None:
    first = _store()
    second = _store()
    event_id = f"xid45-{uuid4()}"
    now = datetime.now(timezone.utc)
    current = [now]
    calls = []
    coordinators = [
        XidCorrelationCoordinator(
            first,
            GpuFaultPolicyEngine(),
            lambda _event, decision: calls.append("first") or decision,
            owner=f"replica-{uuid4()}",
            now=lambda: current[0],
        ),
        XidCorrelationCoordinator(
            second,
            GpuFaultPolicyEngine(),
            lambda _event, decision: calls.append("second") or decision,
            owner=f"replica-{uuid4()}",
            now=lambda: current[0],
        ),
    ]
    event = XidEvent(
        event_id=event_id,
        cluster_id=f"cluster-{uuid4()}",
        node_id="node-a",
        observed_at=now,
        xid=45,
        gpu_uuid="GPU-a",
        product="H200",
        driver_branch=575,
        cuda_version="12.9",
    )

    try:
        coordinators[0].ingest(event)
        current[0] += timedelta(seconds=31)
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(lambda item: item.run_once(), coordinators))
        assert sum(results) == 1
        assert len(calls) == 1
    finally:
        first.close()
        second.close()


def test_concurrent_first_start_bootstraps_schema_once() -> None:
    """Greenfield deployments start every replica against an empty database.

    ``CREATE INDEX IF NOT EXISTS`` is not atomic: the existence check and
    the ``pg_class`` insert are separate steps, so without serialization
    two replicas both pass the check and the loser dies with
    ``UniqueViolation`` on ``pg_class_relname_nsp_index``. Reproduced on a
    real deployment as ``GF-REGIONAL-BOOT-016``.
    """
    import psycopg

    assert POSTGRES_URL is not None
    schema = "gpu_fault_bootstrap_" + uuid4().hex[:16]
    with psycopg.connect(POSTGRES_URL, autocommit=True) as conn:
        conn.execute(f'CREATE SCHEMA "{schema}"')
    separator = "&" if "?" in POSTGRES_URL else "?"
    # search_path confines the DDL to the throwaway schema, so the tables
    # really are absent and the race window is open.
    url = f"{POSTGRES_URL}{separator}options=-csearch_path%3D{schema}"

    stores: list[PostgresStore] = []
    try:
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = [pool.submit(PostgresStore, url) for _ in range(3)]
            for future in futures:
                stores.append(future.result())
        assert len(stores) == 3
    finally:
        for store in stores:
            store.close()
        with psycopg.connect(POSTGRES_URL, autocommit=True) as conn:
            conn.execute(f'DROP SCHEMA "{schema}" CASCADE')


def test_telemetry_group_commit_rolls_back_one_item_only(monkeypatch) -> None:
    """A failed item must lose its own writes, not the group's.

    The telemetry batch endpoint commits a whole claim batch in one
    transaction because commits per second, not CPU, is what limits the
    32/50-cluster fleets. That is only safe if the per-item
    ``collector_ingestion_transaction`` nests as a SAVEPOINT: one bad
    payload rolls back to its savepoint and its siblings still commit.
    ``InMemoryStore`` transactions are no-ops, so this invariant can only
    be observed against a real database.
    """
    import asyncio
    import json

    import httpx

    from gpu_fault.app import ApplicationContext, create_app

    token = "processor-group-commit-token-" + "x" * 32
    monkeypatch.setenv("GPU_FAULT_PROCESSOR_MODE", "active-active")
    monkeypatch.setenv("POD_UID", "pod-group-commit")
    cluster_id = f"group-commit-{uuid4().hex[:12]}"
    observed_at = datetime.now(timezone.utc).isoformat()

    def gpu_item(node_id: str) -> dict:
        return {
            "request_id": f"req-{node_id}",
            "path": "/v1/collector-events/gpu-metrics",
            "payload": {
                "batch_id": f"batch-{node_id}",
                "cluster_id": cluster_id,
                "node_id": node_id,
                "observed_at": observed_at,
                "source": "DCGM_EXPORTER",
                "edge_filter_reasons": ["health-summary"],
                "samples": [
                    {
                        "metric_name": "DCGM_FI_DEV_SM_CLOCK",
                        "canonical_name": "sm_clock_mhz",
                        "value": 345.0,
                        "unit": "MHz",
                        "gpu_index": "0",
                        "gpu_uuid": f"GPU-{node_id}-0",
                    }
                ],
            },
        }

    store = _store()
    try:
        context = ApplicationContext(store=store, execution_token=token)
        app = create_app(context)
        original = context.gpu_metrics.ingest

        def ingest(batch):
            if batch.node_id == "node-2":
                raise RuntimeError("simulated ingestion failure")
            return original(batch)

        context.gpu_metrics.ingest = ingest
        item_transaction = store.collector_ingestion_transaction
        opened: list[str] = []

        @contextmanager
        def counting_item_transaction(cluster_key: str, node_key: str, batch_key: str):
            opened.append(node_key)
            with item_transaction(cluster_key, node_key, batch_key):
                yield

        store.collector_ingestion_transaction = counting_item_transaction

        async def replay() -> httpx.Response:
            async with asgi_client(app) as client:
                return await client.post(
                    "/v1/internal/processor/telemetry-batch",
                    headers={
                        "X-GPU-Fault-Processor-Replay": (
                            "test-processor-replay-secret-" + "r" * 32
                        ),
                        "Content-Type": "application/json",
                    },
                    content=json.dumps(
                        {
                            "items": [
                                gpu_item("node-1"),
                                gpu_item("node-2"),
                                gpu_item("node-3"),
                            ]
                        }
                    ).encode(),
                )

        response = asyncio.run(replay())
        assert response.status_code == 200
        results = {
            item["request_id"]: item["status"] for item in response.json()["results"]
        }
        assert results == {"req-node-1": 200, "req-node-2": 500, "req-node-3": 200}
        # Three savepoints, not six: had the failed item aborted the
        # outer transaction, the endpoint would have replayed every item
        # in its own transaction and reached the same statuses by a much
        # more expensive route.
        assert opened == ["node-1", "node-2", "node-3"]
    finally:
        store.close()

    # A second connection pool reads only committed rows: the two good
    # items are durable and the failed one left nothing behind.
    verifier = _store()
    try:
        assert verifier.list_collector_statuses(cluster_id, "node-1")
        assert verifier.list_collector_statuses(cluster_id, "node-3")
        assert not verifier.list_collector_statuses(cluster_id, "node-2")
        assert verifier.list_gpu_metrics_latest(cluster_id, "node-1")
        assert not verifier.list_gpu_metrics_latest(cluster_id, "node-2")
    finally:
        verifier.close()

from __future__ import annotations

import importlib
import inspect
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from typing import get_type_hints
from uuid import uuid4

import pytest

from gpu_fault.models import (
    AdvisoryNotification,
    Environment,
    NotificationResult,
    NotificationStatus,
    TerminalEvent,
    TerminalStatus,
)
from gpu_fault.policy import GpuFaultPolicyEngine, SxidClassification, SxidEvent
from gpu_fault.schema_migrations import POSTGRES_SCHEMA_MIGRATIONS
from gpu_fault.store import InMemoryStore, PostgresStore, SqliteStore
from gpu_fault.store.contracts import (
    ControlPlaneStore,
    FleetStore,
    ProcessorStore,
    WorkflowStore,
)
from gpu_fault.store.memory.fleet import MemoryFleetMixin
from gpu_fault.store.memory.processor_leases import MemoryProcessorLeaseMixin
from gpu_fault.store.memory.processor_queue import MemoryProcessorQueueMixin
from gpu_fault.store.postgres.processor_admin import PostgresProcessorAdminMixin
from gpu_fault.store.postgres.processor_admission import PostgresProcessorAdmissionMixin
from gpu_fault.store.postgres.processor_claims import PostgresProcessorClaimsMixin
from gpu_fault.store.postgres.processor_completion import (
    PostgresProcessorCompletionMixin,
)
from gpu_fault.store.postgres.processor_completion_runtime import (
    complete_cluster_groups,
)
from gpu_fault.store.postgres.processor_leases import PostgresProcessorLeaseMixin
from gpu_fault.store.postgres.processor_storage import PostgresProcessorStorageMixin
from gpu_fault.store.shared.transactional_workflows import TransactionalWorkflowMixin
from gpu_fault.store.sqlite.processor_leases import SqliteProcessorLeaseMixin
from gpu_fault.store.sqlite.processor_queue import SqliteProcessorQueueMixin
from gpu_fault.watcher import WorkloadPhase
from tests._builders import (
    attempt_observation,
    build_store,
    container_observation,
    processor_request,
)

# The three stores are peers; each composes the shared mixins it needs.
# These sets pin, per store, which public methods resolve to a class under
# ``gpu_fault.store.shared`` (or, for SQLite's non-durable telemetry spool,
# under ``gpu_fault.store.memory``), so a method silently falling through
# to a shared fallback -- or a shared one silently overridden -- is a
# reviewed change here.
MEMORY_SHARED_PUBLIC = frozenset(
    {
        "complete_active_processor_requests_batch",
        "efa_traffic_state_key",
        "get_decision_by_attempt",
        "get_xid_events",
        "list_xid_events_for_scopes",
        "processor_batch_transaction",
        "processor_queue_count_status",
        "remote_command_cluster_health",
        "save_attempt_observations_batch",
        "try_enqueue_processor_requests_batch",
    }
)

SQLITE_SHARED_PUBLIC = frozenset(
    {
        "acquire_periodic_task_lease",
        "acquire_processor_leadership",
        "add_marker",
        "amend_workflow",
        "apply_efa_traffic_admin_action",
        "complete_active_processor_requests_batch",
        "complete_notification_delivery",
        "complete_remote_command",
        "complete_xid_correlation",
        "create_incident_workflow_if_absent",
        "delete_regional_cluster",
        "efa_traffic_state_key",
        "enqueue_notification_delivery",
        "ensure_remote_command",
        "establish_notification_watermark",
        "get_agent",
        "get_barrier",
        "get_collector_metrics_snapshot",
        "get_decision_by_attempt",
        "get_decision_by_event",
        "get_diagnostic",
        "get_efa_traffic_state",
        "get_event_by_attempt",
        "get_fleet_deployment",
        "get_gpu_inventory_snapshot",
        "get_gpu_metrics_batch",
        "get_hyperpod_node_identity",
        "get_hyperpod_submission",
        "get_incident",
        "get_incident_by_event",
        "get_notification",
        "get_notification_result",
        "get_notification_watermark",
        "get_plan",
        "get_processor_leadership",
        "get_profile",
        "get_regional_cluster",
        "get_regional_registry_head",
        "get_regional_registry_revision",
        "get_restart_budget",
        "get_workflow",
        "get_xid_correlation",
        "get_xid_event",
        "get_xid_events",
        "get_xid_policy_decision",
        "link_event_to_incident",
        "list_attempt_observation_states",
        "list_gpu_metrics_latest",
        "list_regional_registry_members",
        "list_training_progress_states",
        "list_xid_events_for_scopes",
        "merge_attempt_fault_workflow",
        "merge_replacement_workflow",
        "merge_sxid_workflow",
        "observe_efa_traffic",
        "observe_gpu_metrics",
        "observe_telemetry_metric",
        "observe_training_progress",
        "processor_batch_transaction",
        "processor_queue_count_status",
        "publish_regional_registry_revision",
        "reconcile_restored_workflow",
        "reconcile_retired_generation_workflow",
        "record_xid74_occurrences",
        "release_job_restart",
        "release_notification_delivery",
        "remote_command_cluster_health",
        "renew_remote_command_lease",
        "reserve_hyperpod_submission",
        "reserve_job_restart",
        "save_agent",
        "save_attempt_observation",
        "save_attempt_observations_batch",
        "save_barrier",
        "save_collector_metrics_snapshot",
        "save_collector_status",
        "save_decision",
        "save_diagnostic",
        "save_fleet_deployment",
        "save_gpu_inventory_snapshot",
        "save_gpu_metrics_batch",
        "save_hyperpod_node_identity",
        "save_hyperpod_submission",
        "save_incident_and_workflow",
        "save_notification_result",
        "save_plan",
        "save_profile",
        "save_regional_cluster",
        "save_regional_registry_member",
        "save_triage_report",
        "save_xid_policy_decision",
        "try_enqueue_processor_requests_batch",
    }
)

SQLITE_MEMORY_PUBLIC = frozenset(
    {
        "abandon_telemetry_spool_claims",
        "claim_telemetry_spool",
        "complete_telemetry_spool",
        "release_telemetry_spool",
        "telemetry_spool_stats",
        "try_spool_telemetry_requests",
    }
)

POSTGRES_SHARED_PUBLIC = frozenset(
    {
        "acquire_periodic_task_lease",
        "acquire_processor_leadership",
        "add_marker",
        "amend_workflow",
        "apply_efa_traffic_admin_action",
        "complete_notification_delivery",
        "complete_remote_command",
        "complete_xid_correlation",
        "create_incident_workflow_if_absent",
        "delete_regional_cluster",
        "efa_traffic_state_key",
        "enqueue_notification_delivery",
        "ensure_remote_command",
        "establish_notification_watermark",
        "get_agent",
        "get_barrier",
        "get_collector_metrics_snapshot",
        "get_decision_by_attempt",
        "get_decision_by_event",
        "get_diagnostic",
        "get_efa_traffic_state",
        "get_event_by_attempt",
        "get_fleet_deployment",
        "get_gpu_inventory_snapshot",
        "get_hyperpod_node_identity",
        "get_hyperpod_submission",
        "get_incident",
        "get_incident_by_event",
        "get_notification",
        "get_notification_result",
        "get_notification_watermark",
        "get_plan",
        "get_processor_leadership",
        "get_profile",
        "get_regional_cluster",
        "get_regional_registry_head",
        "get_regional_registry_revision",
        "get_restart_budget",
        "get_workflow",
        "get_xid_correlation",
        "get_xid_event",
        "get_xid_policy_decision",
        "link_event_to_incident",
        "list_regional_registry_members",
        "merge_attempt_fault_workflow",
        "merge_replacement_workflow",
        "merge_sxid_workflow",
        "observe_efa_traffic",
        "observe_telemetry_metric",
        "publish_regional_registry_revision",
        "reconcile_restored_workflow",
        "reconcile_retired_generation_workflow",
        "record_xid74_occurrences",
        "release_job_restart",
        "release_notification_delivery",
        "remote_command_cluster_health",
        "renew_remote_command_lease",
        "reserve_hyperpod_submission",
        "reserve_job_restart",
        "save_agent",
        "save_barrier",
        "save_collector_metrics_snapshot",
        "save_collector_status",
        "save_decision",
        "save_diagnostic",
        "save_fleet_deployment",
        "save_gpu_inventory_snapshot",
        "save_hyperpod_node_identity",
        "save_hyperpod_submission",
        "save_incident_and_workflow",
        "save_notification_result",
        "save_plan",
        "save_profile",
        "save_regional_cluster",
        "save_regional_registry_member",
        "save_triage_report",
        "save_xid_policy_decision",
    }
)

SQLITE_PROCESSOR_QUEUE_PUBLIC = frozenset(
    {
        "active_backlog_is_lane_blocked",
        "claim_active_processor_requests",
        "claim_processor_requests",
        "cleanup_completed_processor_requests",
        "count_fault_rows_blocked_by_observation",
        "enqueue_processor_request",
        "get_processor_request",
        "has_incomplete_processor_requests",
        "has_incomplete_processor_requests_for_scopes",
        "processor_fault_backlog_depth",
        "processor_queue_stats",
        "try_enqueue_processor_request",
    }
)

SQLITE_PROCESSOR_LEASE_PUBLIC = frozenset(
    {
        "cleanup_processor_lanes",
        "complete_active_processor_request",
        "complete_processor_request",
        "release_active_processor_request",
        "release_processor_request",
        "renew_active_processor_request",
        "validate_processor_lane",
        "reclaim_expired_processor_leases",
    }
)

POSTGRES_PROCESSOR_PUBLIC = {
    PostgresProcessorAdminMixin: frozenset(
        {
            "backfill_processor_queue_state_columns",
            "cleanup_completed_processor_requests",
            "finalize_processor_counter_shards",
            "listen_processor_queue_notifications",
            "processor_batch_transaction",
            "processor_counter_mode",
            "processor_queue_count_status",
            "processor_queue_state_status",
            "restore_legacy_processor_counters",
        }
    ),
    PostgresProcessorAdmissionMixin: frozenset(
        {"try_enqueue_processor_request", "try_enqueue_processor_requests_batch"}
    ),
    PostgresProcessorClaimsMixin: frozenset(
        {
            "active_backlog_is_lane_blocked",
            "claim_active_processor_query",
            "claim_active_processor_requests",
            "claim_processor_requests",
            "count_fault_rows_blocked_by_observation",
        }
    ),
    PostgresProcessorCompletionMixin: frozenset(
        {
            "complete_active_processor_request",
            "complete_active_processor_requests_batch",
            "complete_processor_request",
        }
    ),
    PostgresProcessorLeaseMixin: frozenset(
        {
            "cleanup_processor_lanes",
            "release_active_processor_request",
            "release_processor_request",
            "renew_active_processor_request",
            "validate_processor_lane",
            "reclaim_expired_processor_leases",
        }
    ),
    PostgresProcessorStorageMixin: frozenset(
        {
            "enqueue_processor_request",
            "get_processor_request",
            "has_incomplete_processor_requests",
            "has_incomplete_processor_requests_for_scopes",
            "processor_fault_backlog_depth",
            "processor_queue_stats",
        }
    ),
}

TRANSACTIONAL_WORKFLOW_PUBLIC = frozenset(
    {
        "create_incident_workflow_if_absent",
        "merge_attempt_fault_workflow",
        "merge_replacement_workflow",
        "merge_sxid_workflow",
        "reconcile_restored_workflow",
        "reconcile_retired_generation_workflow",
        "amend_workflow",
        "save_incident_and_workflow",
    }
)

MEMORY_FLEET_PUBLIC = frozenset(
    {
        "cleanup_terminal_fleet_deployments",
        "delete_regional_cluster",
        "get_agent",
        "get_barrier",
        "get_fleet_deployment",
        "get_regional_cluster",
        "get_regional_registry_head",
        "get_regional_registry_revision",
        "list_active_fleet_deployments",
        "list_agents",
        "list_barriers",
        "list_fleet_deployments",
        "list_regional_cluster_ids",
        "list_regional_clusters",
        "list_regional_registry_members",
        "publish_regional_registry_revision",
        "replace_agent_if_matches",
        "replace_fleet_deployment_if_matches",
        "save_agent",
        "save_barrier",
        "save_fleet_deployment",
        "save_regional_cluster",
        "save_regional_registry_member",
    }
)

MEMORY_PROCESSOR_QUEUE_PUBLIC = frozenset(
    {
        "active_backlog_is_lane_blocked",
        "claim_active_processor_requests",
        "claim_processor_requests",
        "cleanup_completed_processor_requests",
        "count_fault_rows_blocked_by_observation",
        "enqueue_processor_request",
        "get_processor_request",
        "has_incomplete_processor_requests",
        "has_incomplete_processor_requests_for_scopes",
        "processor_fault_backlog_depth",
        "processor_queue_stats",
        "try_enqueue_processor_request",
    }
)

MEMORY_PROCESSOR_LEASE_PUBLIC = frozenset(
    {
        "acquire_periodic_task_lease",
        "acquire_processor_leadership",
        "cleanup_processor_lanes",
        "complete_active_processor_request",
        "complete_processor_request",
        "get_processor_leadership",
        "release_active_processor_request",
        "release_processor_request",
        "renew_active_processor_request",
        "validate_processor_lane",
        "reclaim_expired_processor_leases",
    }
)


def _public_methods(cls) -> frozenset[str]:
    return frozenset(
        name
        for name, value in cls.__dict__.items()
        if not name.startswith("_") and callable(value)
    )


def _protocol_methods(protocol) -> set[str]:
    methods: set[str] = set()
    for item in inspect.getmro(protocol):
        methods.update(_public_methods(item))
    return methods


def _protocol_members(protocol) -> dict[str, object]:
    members: dict[str, object] = {}
    for item in reversed(inspect.getmro(protocol)):
        for name in _public_methods(item):
            members[name] = getattr(item, name)
    return members


def _declared_attribute(cls, name: str):
    for item in inspect.getmro(cls):
        if name in vars(item):
            return vars(item)[name]
    raise AttributeError(name)


def _resolved_type_hints(member) -> dict[str, object]:
    namespaces: dict[str, object] = {}
    for module_name in (
        "gpu_fault.store.contracts",
        "gpu_fault.fleet",
        "gpu_fault.models",
        "gpu_fault.processor.models",
        "gpu_fault.regional",
        "gpu_fault.telemetry",
        "gpu_fault.telemetry_models",
        member.__module__,
    ):
        namespaces.update(vars(importlib.import_module(module_name)))
    return get_type_hints(member, globalns=namespaces)


def _parameter_shape(member) -> list[tuple[str, object, bool]]:
    return [
        (
            parameter.name,
            parameter.kind,
            parameter.default is not inspect.Signature.empty,
        )
        for parameter in inspect.signature(member).parameters.values()
    ]


def _mro_public_methods(cls) -> set[str]:
    methods: set[str] = set()
    for item in inspect.getmro(cls):
        methods.update(_public_methods(item))
    return methods


def _public_methods_resolved_from(cls, package: str) -> set[str]:
    """Public methods of ``cls`` whose first definer in the MRO lives under
    ``package``."""

    resolved: set[str] = set()
    seen: set[str] = set()
    for item in inspect.getmro(cls):
        for name in _public_methods(item) - seen:
            seen.add(name)
            if item.__module__.startswith(package):
                resolved.add(name)
    return resolved


def test_store_classes_cover_application_protocol() -> None:
    required: dict[str, object] = {}
    for protocol in (ControlPlaneStore, ProcessorStore, FleetStore, WorkflowStore):
        required.update(_protocol_members(protocol))

    for implementation in (InMemoryStore, SqliteStore, PostgresStore):
        missing = sorted(name for name in required if not hasattr(implementation, name))
        assert missing == [], f"{implementation.__name__} misses {missing}"
        for name, protocol_member in required.items():
            implementation_member = getattr(implementation, name)
            implementation_parameters = inspect.signature(
                implementation_member
            ).parameters.values()
            if not any(
                parameter.kind is inspect.Parameter.VAR_KEYWORD
                for parameter in implementation_parameters
            ):
                assert _parameter_shape(implementation_member) == _parameter_shape(
                    protocol_member
                ), (
                    f"{implementation.__name__}.{name} parameter shape "
                    "does not match the store Protocol"
                )

            protocol_attribute = _declared_attribute(ControlPlaneStore, name)
            if isinstance(protocol_attribute, staticmethod):
                assert isinstance(
                    _declared_attribute(implementation, name), staticmethod
                ), f"{implementation.__name__}.{name} must be static"

            protocol_return = _resolved_type_hints(protocol_member).get(
                "return", inspect.Signature.empty
            )
            implementation_return = _resolved_type_hints(implementation_member).get(
                "return", inspect.Signature.empty
            )
            if implementation_return is not inspect.Signature.empty:
                assert implementation_return == protocol_return, (
                    f"{implementation.__name__}.{name} returns "
                    f"{implementation_return!r}; expected {protocol_return!r}"
                )


def test_applied_postgres_migration_checksums_are_immutable() -> None:
    historical = {
        migration.version: migration.checksum
        for migration in POSTGRES_SCHEMA_MIGRATIONS
        if migration.version <= 5
    }

    assert historical == {
        1: "47c2772094d164b8d018d3c8de5d8bf023c9d2b4e9a4c08c1ded51bb112c69f5",
        2: "39f7a4329cec008640850dcf408a88391e97a055d696156b6a8000fa4ce0f1c4",
        3: "2a82490502995bac570f3a533c53b899645630f786f260972a72c82641ac0680",
        4: "9ce8369e79e881c566bc429c72a157a0e7fe6a2557bcf2b4d050f2565db2b947",
        5: "884820da9fdf5521dad40dffd8a40b1a3acf415de1871f563202eafd083ebb97",
    }
    assert POSTGRES_SCHEMA_MIGRATIONS[-1].version == 12


def test_completion_cluster_groups_use_bounded_parallelism() -> None:
    active = 0
    max_active = 0
    lock = threading.Lock()
    started = threading.Barrier(2)

    def complete(batch):
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
        started.wait(timeout=2)
        time.sleep(0.01)
        with lock:
            active -= 1
        return {batch[0]: batch[0]}

    with ThreadPoolExecutor(max_workers=2) as executor:
        completed = complete_cluster_groups(
            {"cluster-a": ["a"], "cluster-b": ["b"]},
            concurrency=2,
            executor=executor,
            complete=complete,
        )

    assert completed == {"a": "a", "b": "b"}
    assert max_active == 2


def test_store_composition_is_an_explicit_reviewed_contract() -> None:
    shared = "gpu_fault.store.shared"
    memory = "gpu_fault.store.memory"
    assert _public_methods_resolved_from(InMemoryStore, shared) == MEMORY_SHARED_PUBLIC
    assert _public_methods_resolved_from(SqliteStore, shared) == SQLITE_SHARED_PUBLIC
    assert _public_methods_resolved_from(SqliteStore, memory) == SQLITE_MEMORY_PUBLIC
    assert (
        _public_methods_resolved_from(PostgresStore, shared) == POSTGRES_SHARED_PUBLIC
    )
    assert _public_methods_resolved_from(PostgresStore, memory) == set()
    assert _public_methods_resolved_from(PostgresStore, "gpu_fault.store.sqlite") == (
        set()
    )
    assert _mro_public_methods(PostgresStore) >= _mro_public_methods(SqliteStore) - (
        SQLITE_MEMORY_PUBLIC
    )
    assert _public_methods(MemoryFleetMixin) == MEMORY_FLEET_PUBLIC
    assert _public_methods(MemoryProcessorQueueMixin) == MEMORY_PROCESSOR_QUEUE_PUBLIC
    assert _public_methods(MemoryProcessorLeaseMixin) == MEMORY_PROCESSOR_LEASE_PUBLIC
    assert _public_methods(SqliteProcessorQueueMixin) == SQLITE_PROCESSOR_QUEUE_PUBLIC
    assert _public_methods(SqliteProcessorLeaseMixin) == SQLITE_PROCESSOR_LEASE_PUBLIC
    for mixin, expected in POSTGRES_PROCESSOR_PUBLIC.items():
        assert _public_methods(mixin) == expected
    assert _public_methods(TransactionalWorkflowMixin) == TRANSACTIONAL_WORKFLOW_PUBLIC


def test_postgres_sxid_workflow_uses_shared_transaction_template() -> None:
    assert (
        PostgresStore.merge_sxid_workflow
        is TransactionalWorkflowMixin.merge_sxid_workflow
    )
    assert (
        PostgresStore.merge_attempt_fault_workflow
        is TransactionalWorkflowMixin.merge_attempt_fault_workflow
    )


def test_postgres_schema_state_and_ddl_are_distinct_modules() -> None:
    from gpu_fault.store.postgres.ddl import create_postgres_schema
    from gpu_fault.store.postgres.schema_state import PostgresSchemaMixin

    assert create_postgres_schema.__module__.endswith(".ddl"), (
        'expected create_postgres_schema.__module__.endswith(".ddl") to be true'
    )
    assert PostgresSchemaMixin.__module__.endswith(".schema_state"), (
        'expected PostgresSchemaMixin.__module__.endswith(".schema_state") to be true'
    )


@pytest.fixture(params=["memory", "sqlite", "postgres"])
def processor_store(request, tmp_path):
    if request.param == "memory":
        yield build_store()
        return
    store = SqliteStore(str(tmp_path / "contract.db"))
    if request.param == "sqlite":
        try:
            yield store
        finally:
            store.close()
        return
    store.close()
    postgres_url = os.getenv("GPU_FAULT_TEST_POSTGRES_URL")
    if not postgres_url:
        pytest.skip("GPU_FAULT_TEST_POSTGRES_URL is required")
    postgres = PostgresStore(postgres_url)
    try:
        yield postgres
    finally:
        postgres.close()


def test_processor_queue_contract(processor_store) -> None:
    request = processor_request(
        "/v1/collector-events/nvidia-kernel", body=b'{"node_id":"node-a"}'
    )
    accepted, reason = processor_store.try_enqueue_processor_request(
        request, max_depth=10, max_cluster_depth=10
    )
    assert accepted is not None
    assert reason is None
    claimed = processor_store.claim_active_processor_requests(
        "contract-owner",
        now=datetime.now(timezone.utc),
        lease_duration=timedelta(seconds=30),
        limit=1,
    )
    assert [item.request_id for item in claimed] == [request.request_id]
    processor_store.complete_active_processor_request(
        request.request_id,
        "contract-owner",
        claimed[0].leader_epoch,
        claimed[0].lease_token,
        response_status=200,
        response_content_type="application/json",
        response_body_base64="e30=",
    )
    assert processor_store.processor_queue_stats()["depth"] == 0


def test_processor_queue_stats_report_oldest_age_per_cluster(processor_store) -> None:
    """S6: the claim window is FIFO across the region, so one cluster's storm
    delays every other cluster; the per-cluster oldest age is what says which
    cluster is actually waiting. The region-wide age stays the overall max."""

    now = datetime.now(timezone.utc)
    ages = {"node-a": ("cluster-a", 100.0), "node-b": ("cluster-a", 40.0)}
    ages["node-c"] = ("cluster-b", 10.0)
    for node_id, (cluster_id, age) in ages.items():
        request = processor_request(
            "/v1/collector-events/nvidia-kernel",
            body=f'{{"node_id":"{node_id}"}}'.encode(),
            cluster_id=cluster_id,
        ).model_copy(update={"created_at": now - timedelta(seconds=age)})
        accepted, reason = processor_store.try_enqueue_processor_request(
            request, max_depth=10, max_cluster_depth=10
        )
        assert accepted is not None, reason

    stats = processor_store.processor_queue_stats(now=now)

    assert stats["by_cluster"] == {"cluster-a": 2, "cluster-b": 1}
    assert stats["oldest_age_seconds"] == pytest.approx(100.0)
    assert stats["oldest_age_by_cluster"] == {
        "cluster-a": pytest.approx(100.0),
        "cluster-b": pytest.approx(10.0),
    }

    claimed = processor_store.claim_active_processor_requests(
        "contract-owner", now=now, lease_duration=timedelta(seconds=30), limit=3
    )
    for item in claimed:
        processor_store.complete_active_processor_request(
            item.request_id,
            "contract-owner",
            item.leader_epoch,
            item.lease_token,
            response_status=200,
            response_content_type="application/json",
            response_body_base64="e30=",
        )
    drained = processor_store.processor_queue_stats(now=now)
    assert drained["oldest_age_by_cluster"] == {}
    assert drained["oldest_age_seconds"] == 0.0


def test_notification_status_counts_contract(processor_store) -> None:
    before = processor_store.notification_status_counts()
    assert set(before) == set(NotificationStatus)
    suffix = uuid4().hex
    notifications = []
    for index in range(3):
        notifications.append(
            processor_store.save_notification_if_absent(
                AdvisoryNotification(
                    notification_id=f"notification-status-{suffix}-{index}",
                    deduplication_key=f"notification-status-{suffix}-{index}",
                    cluster_name="cluster-a",
                    incident_id=f"incident-status-{suffix}",
                    subject="subject",
                    body_text="body",
                    support_case_draft="body",
                )
            )
        )
    processor_store.save_notification_result(
        NotificationResult(
            notification_id=notifications[0].notification_id,
            status=NotificationStatus.SENT,
        )
    )
    processor_store.save_notification_result(
        NotificationResult(
            notification_id=notifications[1].notification_id,
            status=NotificationStatus.FAILED,
        )
    )

    after = processor_store.notification_status_counts()

    assert set(after) == set(NotificationStatus)
    assert after[NotificationStatus.SENT] == before[NotificationStatus.SENT] + 1
    assert after[NotificationStatus.FAILED] == before[NotificationStatus.FAILED] + 1
    assert after[NotificationStatus.QUEUED] == before[NotificationStatus.QUEUED] + 1


def test_policy_decision_upsert_contract(processor_store) -> None:
    event_id = f"sxid-store-contract-{uuid4()}"
    decision = GpuFaultPolicyEngine().evaluate_sxid(
        SxidEvent(
            event_id=event_id,
            cluster_id="cluster-a",
            node_id="node-a",
            observed_at=datetime.now(timezone.utc),
            sxid=11001,
            classification=SxidClassification.FATAL,
            classification_source=("NVIDIA_FABRIC_MANAGER_CATALOG"),
            product="H200",
            runtime_profile_version="simulated-v1",
        )
    )
    first = decision.model_copy(
        update={
            "incident_id": f"inc-{event_id}",
            "workflow_request_id": f"workflow-{event_id}",
        }
    )
    processor_store.save_xid_policy_decision(first)
    processor_store.save_xid_policy_decision(first)
    assert processor_store.get_xid_policy_decision(event_id) == first

    updated = first.model_copy(
        update={"advisory_notification_id": f"notice-{event_id}"}
    )
    processor_store.save_xid_policy_decision(updated)
    assert processor_store.get_xid_policy_decision(event_id) == updated


def _postgres_store() -> PostgresStore:
    postgres_url = os.getenv("GPU_FAULT_TEST_POSTGRES_URL")
    if not postgres_url:
        pytest.skip("GPU_FAULT_TEST_POSTGRES_URL is required")
    return LegacyPostgresStore(postgres_url)


class LegacyPostgresStore(PostgresStore):
    def seed_terminal_event(self, event: TerminalEvent) -> None:
        storage_key = self._state_key((event.cluster_id, event.attempt_id))
        with self._state_transaction(f"test-terminal/{storage_key}"):
            self._put("event", event.event_key, event)
            self._link("attempt_event", storage_key, event.event_key)


def test_postgres_terminal_event_rejects_late_active_observation() -> None:
    first = _postgres_store()
    second = _postgres_store()
    cluster_id = f"cluster-{uuid4()}"
    attempt_id = f"attempt-{uuid4()}"
    observed_at = datetime.now(timezone.utc)
    active = attempt_observation(
        f"job-{uuid4()}",
        attempt_id,
        observed_at,
        cluster_id=cluster_id,
        runtime_profile_version="profile-a",
        containers=[
            container_observation(
                f"pod-{uuid4()}", "worker-0", 0, f"node-{uuid4()}", gpu_uuids=["GPU-a"]
            )
        ],
    )
    terminal = TerminalEvent(
        cluster_id=cluster_id,
        environment=Environment.HYPERPOD_EKS,
        job_id=active.job_id,
        attempt_id=attempt_id,
        terminal_status=TerminalStatus.TIMED_OUT,
        ended_at=observed_at + timedelta(seconds=30),
        runtime_profile_version="profile-a",
    )

    try:
        assert first.save_attempt_observation(active), (
            "PostgreSQL did not store the active observation"
        )
        assert first.save_event_if_absent(terminal), (
            "PostgreSQL did not insert the terminal event"
        )
        late = active.model_copy(
            update={"observed_at": observed_at + timedelta(minutes=1)}
        )

        assert second.save_attempt_observation(late) is False
        assert second.save_attempt_observations_batch([late]) == [False]
        state = second.list_attempt_observation_states(cluster_id)[0]
        assert state.observation.workload_phase is WorkloadPhase.FAILED
    finally:
        first.close()
        second.close()


def test_postgres_batch_terminalizes_historical_active_observation() -> None:
    first = _postgres_store()
    second = _postgres_store()
    cluster_id = f"cluster-{uuid4()}"
    attempt_id = f"attempt-{uuid4()}"
    observed_at = datetime.now(timezone.utc)
    active = attempt_observation(
        f"job-{uuid4()}",
        attempt_id,
        observed_at,
        cluster_id=cluster_id,
        runtime_profile_version="profile-a",
        containers=[
            container_observation(
                f"pod-{uuid4()}", "worker-0", 0, f"node-{uuid4()}", gpu_uuids=["GPU-a"]
            )
        ],
    )
    terminal = TerminalEvent(
        cluster_id=cluster_id,
        environment=Environment.HYPERPOD_EKS,
        job_id=active.job_id,
        attempt_id=attempt_id,
        terminal_status=TerminalStatus.TIMED_OUT,
        ended_at=observed_at + timedelta(seconds=30),
        runtime_profile_version="profile-a",
    )

    try:
        assert first.save_attempt_observation(active), (
            "PostgreSQL did not store the historical active observation"
        )
        first.seed_terminal_event(terminal)
        late = active.model_copy(
            update={"observed_at": observed_at + timedelta(minutes=1)}
        )

        assert second.save_attempt_observations_batch([late]) == [False]
        state = second.list_attempt_observation_states(cluster_id)[0]
        assert state.observation.workload_phase is WorkloadPhase.FAILED
    finally:
        first.close()
        second.close()


def test_postgres_cleanup_terminal_reconcile_is_not_starved_by_old_events() -> None:
    store = _postgres_store()
    suffix = uuid4().hex
    cluster_id = f"cluster-{suffix}"
    observed_at = datetime.now(timezone.utc)
    old = attempt_observation(
        f"job-old-{suffix}",
        f"attempt-old-{suffix}",
        observed_at - timedelta(minutes=3),
        cluster_id=cluster_id,
        runtime_profile_version="profile-a",
    )
    current = attempt_observation(
        f"job-current-{suffix}",
        f"attempt-current-{suffix}",
        observed_at,
        cluster_id=cluster_id,
        runtime_profile_version="profile-a",
    )
    old_terminal = TerminalEvent(
        cluster_id=cluster_id,
        environment=Environment.HYPERPOD_EKS,
        job_id=old.job_id,
        attempt_id=old.attempt_id,
        terminal_status=TerminalStatus.SUCCEEDED,
        ended_at=observed_at - timedelta(minutes=2),
        runtime_profile_version="profile-a",
    )
    current_terminal = TerminalEvent(
        cluster_id=cluster_id,
        environment=Environment.HYPERPOD_EKS,
        job_id=current.job_id,
        attempt_id=current.attempt_id,
        terminal_status=TerminalStatus.TIMED_OUT,
        ended_at=observed_at + timedelta(seconds=30),
        runtime_profile_version="profile-a",
    )

    try:
        store.save_attempt_observation(old)
        store.save_event_if_absent(old_terminal)
        store.save_attempt_observation(current)
        store.seed_terminal_event(current_terminal)

        result = store.cleanup_hot_state(
            now=observed_at + timedelta(minutes=1), limit=1
        )

        assert result["attempt_observation_terminalized"] == 1
        states = {
            item.observation.attempt_id: item.observation.workload_phase
            for item in store.list_attempt_observation_states(cluster_id)
        }
        assert states[current.attempt_id] is WorkloadPhase.FAILED
    finally:
        store.close()

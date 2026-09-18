from __future__ import annotations

from contextlib import ExitStack, closing
from datetime import timedelta
from threading import Barrier, Thread

import pytest

from gpu_fault.hyperpod import HyperPodAction, HyperPodSubmissionRecord
from gpu_fault.installation_resources import (
    InstallationResource,
    InstallationResourceDeletePolicy,
    InstallationResourceOwnership,
)
from gpu_fault.managed_recovery import HyperPodNodeIdentity
from gpu_fault.models import (
    CompletionDecision,
    DecisionStatus,
    EffectiveRuntimeProfile,
    Environment,
    RecoveryPlan,
    TerminalEvent,
    TerminalStatus,
    WorkflowStatus,
)
from gpu_fault.store import InMemoryStore, NotFoundError, SqliteStore
from tests._builders import fault_incident, workflow_request
from tests.store._cov95_compat_support import NOW
from tests.store._cov95_compat_support import (
    compat_store_fixture as compat_store_fixture,
)


def terminal(attempt, *, age_days):
    return TerminalEvent(
        cluster_id="compat-cluster",
        environment=Environment.HYPERPOD_EKS,
        job_id="compat-job",
        attempt_id=attempt,
        terminal_status=TerminalStatus.FAILED,
        ended_at=NOW - timedelta(days=age_days),
        runtime_profile_version="compat-profile",
    )


def plan(event, *, incident_id="archived-incident"):
    return RecoveryPlan(
        plan_id=f"plan-{event.attempt_id}",
        incident_id=incident_id,
        attempt_id=event.attempt_id,
        trigger="terminal-event",
        runtime_profile_version="compat-profile",
        steps=[],
    )


def decision(event, recovery_plan_id=None):
    return CompletionDecision(
        cluster_id=event.cluster_id,
        attempt_id=event.attempt_id,
        event_key=event.event_key,
        status=DecisionStatus.PLAN_CREATED
        if recovery_plan_id
        else DecisionStatus.NO_ACTION,
        reason="compatibility record",
        recovery_plan_id=recovery_plan_id,
    )


@pytest.mark.parametrize(
    "method,args",
    [
        ("get_restart_budget", ("missing", "job")),
        ("release_job_restart", ("missing", "job", "reservation")),
        ("get_event_by_attempt", ("missing", "attempt")),
        ("get_plan", ("missing",)),
        ("get_profile", ("missing",)),
        ("get_installation_resource", ("missing", "key")),
        ("get_hyperpod_node_identity", ("missing", "node")),
        ("get_hyperpod_submission", ("missing", "request")),
    ],
)
def test_missing_control_records_fail_without_allocating_defaults(
    compat_store, method, args
):
    store = compat_store
    with pytest.raises(NotFoundError):
        getattr(store, method)(*args)
    assert store.list_installation_resources() == []
    assert store.list_hyperpod_node_identities() == []
    assert store.count_completion_events_without_decision() == 0


def test_restart_budget_is_scoped_immutable_and_idempotently_releasable(compat_store):
    store = compat_store
    first, reserved = store.reserve_job_restart("cluster-a", "job", 1, "reservation-a")
    assert reserved is True
    assert first.restart_count == 1
    duplicate, reserved = store.reserve_job_restart(
        "cluster-a", "job", 1, "reservation-a"
    )
    assert reserved is True
    assert duplicate.reservation_ids == ["reservation-a"]
    exhausted, reserved = store.reserve_job_restart(
        "cluster-a", "job", 1, "reservation-b"
    )
    assert reserved is False
    assert exhausted.restart_count == 1
    with pytest.raises(ValueError, match="budget is immutable"):
        store.reserve_job_restart("cluster-a", "job", 2, "reservation-b")
    assert (
        store.release_job_restart("cluster-a", "job", "not-reserved").restart_count == 1
    )
    assert (
        store.release_job_restart("cluster-a", "job", "reservation-a").restart_count
        == 0
    )
    assert (
        store.release_job_restart("cluster-a", "job", "reservation-a").restart_count
        == 0
    )
    assert store.reserve_job_restart("cluster-a", "job", 1, "reservation-b")[1] is True
    assert store.reserve_job_restart("cluster-b", "job", 1, "reservation-a")[1] is True


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_concurrent_budget_reservations_cannot_oversubscribe_local_state(
    tmp_path, backend
):
    with ExitStack() as stack:
        if backend == "memory":
            first = second = InMemoryStore()
        else:
            path = str(tmp_path / "budget.db")
            first = stack.enter_context(closing(SqliteStore(path)))
            second = stack.enter_context(closing(SqliteStore(path)))
        barrier = Barrier(2)
        outcomes, errors = [], []

        def reserve(store, reservation):
            try:
                barrier.wait(timeout=5)
                outcomes.append(
                    store.reserve_job_restart("cluster", "job", 1, reservation)
                )
            except BaseException as error:
                errors.append(error)

        workers = [
            Thread(target=reserve, args=(first, "first")),
            Thread(target=reserve, args=(second, "second")),
        ]
        try:
            for worker in workers:
                worker.start()
            for worker in workers:
                worker.join(timeout=10)
            assert all(not worker.is_alive() for worker in workers), (
                "budget writers did not finish"
            )
            assert errors == []
            assert sorted(reserved for _, reserved in outcomes) == [False, True]
            for store in (first, second):
                state = store.get_restart_budget("cluster", "job")
                assert state.restart_count == 1
                assert len(state.reservation_ids) == 1
        finally:
            barrier.abort()
            for worker in workers:
                worker.join(timeout=10)


def test_completion_retention_skips_pins_and_never_spends_limit_on_missing_events(
    compat_store,
):
    store = compat_store
    pinned, retired, unplanned, recent = [
        terminal(name, age_days=age)
        for name, age in (
            ("pinned", 4),
            ("retired", 3),
            ("unplanned", 2),
            ("recent", 0),
        )
    ]
    store.save_incident(
        fault_incident("pin", "pin-source", cluster_id="compat-cluster")
    )
    pinned_plan, retired_plan = plan(pinned, incident_id="pin"), plan(retired)
    store.save_plan(pinned_plan)
    store.save_plan(retired_plan)
    for event, plan_id in (
        (pinned, pinned_plan.plan_id),
        (retired, retired_plan.plan_id),
        (unplanned, None),
        (recent, None),
    ):
        store.save_event_if_absent(event)
        store.save_decision(decision(event, plan_id))
    dangling = decision(terminal("missing-event", age_days=10))
    store.save_decision(dangling)
    cutoff = NOW - timedelta(days=1)
    assert store.cleanup_completion_records(older_than=cutoff, limit=1) == 1
    assert store.get_decision_by_event(retired.event_key) is None
    with pytest.raises(NotFoundError):
        store.get_plan(retired_plan.plan_id)
    with pytest.raises(NotFoundError):
        store.get_event_by_attempt(retired.cluster_id, retired.attempt_id)
    assert store.get_decision_by_event(pinned.event_key) == decision(
        pinned, pinned_plan.plan_id
    )
    assert store.get_decision_by_event(dangling.event_key) == dangling
    assert store.cleanup_completion_records(older_than=cutoff, limit=1) == 1
    assert store.get_decision_by_event(unplanned.event_key) is None
    assert store.get_decision_by_event(recent.event_key) == decision(recent)
    assert store.cleanup_completion_records(older_than=cutoff, limit=1) == 0
    assert store.get_plan(pinned_plan.plan_id) == pinned_plan


def test_profile_versions_and_installation_sites_are_independent_keys(compat_store):
    store = compat_store
    profiles = [
        EffectiveRuntimeProfile(
            cluster_id="compat-cluster",
            environment=Environment.HYPERPOD_EKS,
            profile_version=version,
            capabilities=[],
        )
        for version in ("profile-a", "profile-b")
    ]
    for profile in profiles:
        store.save_profile(profile)
    assert [
        store.get_profile(profile.profile_version) for profile in profiles
    ] == profiles
    resources = [
        InstallationResource(
            site_id=site,
            resource_key="local/resource",
            resource_type="security_group",
            resource_id=f"sg-{site}",
            region="us-west-2",
            account_id="123456789012",
            ownership=InstallationResourceOwnership.CREATED,
            delete_policy=InstallationResourceDeletePolicy.DELETE,
            created_at=NOW,
            updated_at=NOW,
        )
        for site in ("site-b", "site-a")
    ]
    for resource in resources:
        store.save_installation_resource(resource)
    assert [resource.site_id for resource in store.list_installation_resources()] == [
        "site-a",
        "site-b",
    ]
    assert store.list_installation_resources("site-a") == [resources[1]]
    assert store.list_installation_resources("missing") == []


def test_preempting_successor_read_ignores_nonpending_and_unrelated_children(
    compat_store,
):
    store = compat_store
    store.save_incident(fault_incident("incident", "event"))
    root = workflow_request("root", "incident", created_at=NOW)
    store.save_workflow(root)
    assert store.get_preempting_successor("root") is None
    children = [
        workflow_request(
            name,
            "incident",
            status=status,
            predecessor_workflow_id=parent,
            preempt_predecessor=preempt,
            created_at=NOW + timedelta(seconds=index),
        )
        for index, (name, status, parent, preempt) in enumerate(
            (
                ("running", WorkflowStatus.RUNNING, "root", True),
                ("plain", WorkflowStatus.PENDING, "root", False),
                ("foreign", WorkflowStatus.PENDING, "other", True),
                ("first", WorkflowStatus.PENDING, "root", True),
                ("second", WorkflowStatus.SAFETY_PENDING, "root", True),
            )
        )
    ]
    for child in children:
        store.save_workflow(child)
    first = store.get_preempting_successor("root")
    assert first.request_id == "first"
    store.save_workflow(
        first.model_copy(update={"status": WorkflowStatus.RUNNING}), expected=first
    )
    assert store.get_preempting_successor("root").request_id == "second"
    assert store.get_preempting_successor("missing") is None


def test_hyperpod_identity_and_intent_records_do_not_invoke_provider_actions(
    compat_store,
):
    store = compat_store
    current = HyperPodNodeIdentity(
        cluster_name="compat-cluster",
        node_logical_id="logical-node",
        status="Running",
        instance_id="i-compat",
        generation=2,
        observed_at=NOW,
    )
    older = current.model_copy(
        update={"generation": 1, "observed_at": NOW - timedelta(seconds=1)}
    )
    assert store.save_hyperpod_node_identity(current) == current
    assert store.save_hyperpod_node_identity(older) == current
    assert store.get_hyperpod_node_identity("compat-cluster", "logical-node") == current
    assert store.list_hyperpod_node_identities("compat-cluster") == [current]
    assert store.list_hyperpod_node_identities("other") == []
    intent = HyperPodSubmissionRecord(
        cluster_name="compat-cluster",
        idempotency_key="compat-intent",
        action=HyperPodAction.REBOOT,
        requested_node_identifiers=["logical-node"],
        created_at=NOW,
        updated_at=NOW,
    )
    assert store.reserve_hyperpod_submission(intent) == (intent, True)
    assert store.reserve_hyperpod_submission(intent) == (intent, False)
    assert (
        store.get_hyperpod_submission("compat-cluster", "compat-intent").state
        == "INTENDED"
    )
    uncertain = intent.model_copy(
        update={"state": "FAILED", "error": "unit unknown provider outcome"}
    )
    store.save_hyperpod_submission(uncertain)
    assert store.reserve_hyperpod_submission(intent) == (uncertain, False)

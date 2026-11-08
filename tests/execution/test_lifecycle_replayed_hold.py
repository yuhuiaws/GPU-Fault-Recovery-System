"""A node lifecycle step re-claimed after an executor hold or a retryable error.

The regional executor replays every re-claimed command that has recorded
details as a WAITING record named ``remote/<command id>``
(``cluster_executor/dispatch.py``), also when the first claim stopped before the
provider call: the fleet-fence, barrier and lease-guard holds post WAITING
without the adapter running, and a retryable error (a Kubernetes 503, a
control-plane 503 at the submission ledger) posts WAITING with the adapter's
run cut short. The node lifecycle adapter (HyperPod, and plain EKS through the
same class) read any such record as a submitted reboot: the replay skipped the
ownership guard, the isolation read and the submission, and waited every cycle
with no provider call -- one wasted cycle when the replayed record was empty,
until the workflow deadline when a hold's keys kept it non-empty; a
REPLACE_NODE ran its warm-spare failover without the isolation read or the
preflight.

Now a remote record resumes only on evidence of a started operation
(``lifecycle.replayed_operation_started``): for a reboot, the step's own
submission key or a record in the write-ahead provider-submission ledger -- a
record in any state, a failed lookup or no durable ledger resumes, so nothing
is ever submitted twice; for a replacement, its own continuation state. Without
it the step runs the first claim's path again: the ownership guard,
``_observe_isolation`` (so an isolation lifted meanwhile gets the b265 hand-off)
and then the submission.

Every claim here goes through the executor's real dispatch layer, the real
lifecycle adapter and the real provider (HyperPod over a fake SageMaker client,
plain EKS over fake EC2/EKS clients); the ledger is a memory store, the store
the control plane keeps it in behind ``RegionalHyperPodSubmissionStore``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from gpu_fault.adapters import HyperPodLifecycleStepAdapter
from gpu_fault.adapters.node_action.lease_guard import active_lease_guard
from gpu_fault.cluster_executor import ClusterActionExecutor, ClusterExecutorError
from gpu_fault.eks_provider import EKS_ADAPTER_OWNER
from gpu_fault.hyperpod import (
    HyperPodAction,
    HyperPodAdapterConfig,
    HyperPodLifecycleAdapter,
    HyperPodSubmissionRecord,
    hyperpod_submission_idempotency_key,
)
from gpu_fault.hyperpod_spares import SpareAllocation
from gpu_fault.models import WorkflowOperation, WorkflowStatus
from gpu_fault.orchestration.failure_outcome import (
    ISOLATION_NOT_ISOLATED,
    LIFECYCLE_ISOLATION_DETAIL,
)
from gpu_fault.regional import (
    RemoteActionCommand,
    RemoteCommandResult,
    RemoteCommandStatus,
)
from gpu_fault.store import InMemoryStore
from tests._builders import fault_incident, workflow_request, workflow_step
from tests.eks_provider._support import CLUSTER_ID, INSTANCE_A, NODE_B
from tests.eks_provider._support import NODE_A as EKS_NODE
from tests.eks_provider._support import provider as eks_provider
from tests.eks_provider.test_eks_lifecycle_step import FakeRegistry
from tests.execution._support import (
    FakeSpareCoordinator,
    RecordingNodeActionAdapter,
    isolated_kubernetes_adapter,
)
from tests.execution.test_cluster_executor_lease_and_report import (
    FakeExecutorClient,
    api_exception,
)
from tests.hyperpod.test_hyperpod import FakeHyperPodClient
from tests.hyperpod.test_managed_recovery import agent

OP = WorkflowOperation
HP_CLUSTER = "hp-cluster"
HP_NODE = "node-a"
KUBERNETES_OWNER = "gpu-fault-kubernetes-adapter"
WORKFLOW = "workflow-a"
INCIDENT = "incident-a"
STEP_INDEX = 2
SHUTDOWN = "executor shutdown requested: stop"
PROVIDERS = ["hyperpod", "eks"]


class NamedHyperPodNode(FakeHyperPodClient):
    """``worker-group-1`` is the Kubernetes node ``node-a``."""

    def describe_cluster_node(self, **kwargs: Any) -> dict[str, Any]:
        result = super().describe_cluster_node(**kwargs)
        if kwargs["NodeLogicalId"] == "worker-group-1":
            labels = result["NodeDetails"]["KubernetesConfig"]["CurrentLabels"]
            labels["kubernetes.io/hostname"] = HP_NODE
        return result


class OwnershipCheck:
    """The STOP-ownership validator of a chain that stopped nothing: asked
    before every new mutation (``require_mutation_ownership``), never by a
    poll of a submitted one; it allows."""

    def __init__(self) -> None:
        self.checked: list[WorkflowOperation] = []

    def check_idle(self, context: Any) -> None:
        self.checked.append(context.step.operation)

    def check(self, context: Any) -> None:
        self.checked.append(context.step.operation)


class Fence:
    """A fleet registry whose rollout fence is closed."""

    def fleet_rollout_fence_deployments(self, _cluster_id: str) -> list[str]:
        return ["deployment-1"]


class FlakyLedger:
    """The control plane's ledger behind the regional proxy; its next
    ``failing_reads`` reads fail the way ``RegionalHyperPodSubmissionStore``
    raises a control-plane 503."""

    def __init__(self, store: InMemoryStore) -> None:
        self.store = store
        self.failing_reads = 0

    def get_hyperpod_submission(self, cluster_name: str, key: str) -> Any:
        if self.failing_reads > 0:
            self.failing_reads -= 1
            raise ClusterExecutorError(
                "regional control plane rejected request (503): unavailable",
                status_code=503,
            )
        return self.store.get_hyperpod_submission(cluster_name, key)

    def reserve_hyperpod_submission(self, record: Any) -> Any:
        return self.store.reserve_hyperpod_submission(record)

    def save_hyperpod_submission(self, record: Any) -> None:
        self.store.save_hyperpod_submission(record)


class LifecycleClaims:
    """One lifecycle step as the regional executor claims it, cycle after
    cycle: the cordon and quarantine before it done, the step at index 2."""

    def __init__(
        self,
        provider: str,
        operation: WorkflowOperation = OP.RESTART_NODE,
        *,
        spares: FakeSpareCoordinator | None = None,
    ) -> None:
        self.store = InMemoryStore()
        self.ledger = FlakyLedger(self.store)
        self.operation = operation
        self.scheduler = isolated_kubernetes_adapter(INCIDENT, 3)
        # The fleet registry every regional lifecycle adapter runs with: the
        # Agent baselines, the revocation and the reboot confirmation.
        self.registry = FakeRegistry(self.store)
        lifecycle: Any
        if provider == "hyperpod":
            self.node, self.cluster_name = HP_NODE, HP_CLUSTER
            client = NamedHyperPodNode(node_recovery="None")
            lifecycle = HyperPodLifecycleAdapter(
                HyperPodAdapterConfig(
                    cluster_name=HP_CLUSTER,
                    region_name="us-east-1",
                    execution_enabled=True,
                ),
                client=client,
                store=self.ledger,
            )
            owner = "gpu-fault-hyperpod-adapter"
            self.submissions: Callable[[], int] = lambda: len(client.reboot_requests)
        else:
            self.node, self.cluster_name = EKS_NODE, CLUSTER_ID
            lifecycle, _core, ec2, _eks = eks_provider(store=self.ledger)
            owner = EKS_ADAPTER_OWNER
            self.submissions = lambda: len(ec2.reboots)
        self.store.save_agent(
            agent(self.node, INSTANCE_A, "incarnation-before").model_copy(
                update={"cluster_id": FakeExecutorClient.cluster_id}
            )
        )
        self.adapter = HyperPodLifecycleStepAdapter(
            lifecycle,
            owner=owner,
            registry=self.registry,
            kubernetes_adapter=self.scheduler,
            spare_coordinator=spares,
            node_action_adapter=(
                RecordingNodeActionAdapter({NODE_B: "https://spare:9099"})
                if spares is not None
                else None
            ),
            post_reboot_stabilization_seconds=0,
        )
        self.ownership = OwnershipCheck()
        self.executor = ClusterActionExecutor(
            FakeExecutorClient(),
            [self.adapter],
            executor_id="executor-a",
            allowed_namespaces={"training"},
            confirm_cluster_name=self.cluster_name,
        )
        setattr(self.executor, "stop_ownership_validator", self.ownership)
        self.key = hyperpod_submission_idempotency_key(
            WORKFLOW, STEP_INDEX, self.operation
        )

    def command(
        self, result_details: dict[str, Any], *, cordoned: bool
    ) -> RemoteActionCommand:
        parameters = (
            {"replacement_strategy": "HEALTHY_WARM_SPARE_ONLY"}
            if self.operation is OP.REPLACE_NODE
            else {}
        )
        steps = [
            workflow_step(
                OP.MARK_UNSCHEDULABLE, KUBERNETES_OWNER, node_ids=[self.node]
            ),
            workflow_step(OP.QUARANTINE, KUBERNETES_OWNER, node_ids=[self.node]),
            workflow_step(
                self.operation,
                self.adapter.owner,
                node_ids=[self.node],
                parameters=parameters,
            ),
        ]
        # A chain whose cordon is on record never meets the fleet fence
        # (``fleet_preflight_reason`` lets a started workflow finish); only a
        # record without it can show the fence hold's shape.
        done = [0, 1] if cordoned else []
        return RemoteActionCommand(
            command_id="remote-lifecycle-1",
            cluster_id=FakeExecutorClient.cluster_id,
            workflow_request_id=WORKFLOW,
            incident_id=INCIDENT,
            step_index=STEP_INDEX,
            fencing_token=3,
            idempotency_key=f"{WORKFLOW}/{STEP_INDEX}/{self.operation.value}",
            step=steps[STEP_INDEX],
            workflow=workflow_request(
                WORKFLOW,
                INCIDENT,
                status=WorkflowStatus.RUNNING,
                official_steps=steps,
                completed_step_indexes=done,
                completed_operations=[steps[index].operation for index in done],
            ),
            incident=fault_incident(
                INCIDENT, "event-a", fencing_token=3, node_ids=[self.node]
            ),
            lease_token="lease-1",
            status=RemoteCommandStatus.LEASED,
            result_details=dict(result_details),
        )

    def claim(
        self, result_details: dict[str, Any] | None = None, *, cordoned: bool = True
    ) -> RemoteCommandResult:
        """One claim; ``result_details`` is what the previous claim posted
        (``complete_remote_command`` replaces the record with it)."""

        return self.executor.dispatch.execute(
            self.command(result_details or {}, cordoned=cordoned)
        )

    def ledger_state(self) -> str | None:
        try:
            record: HyperPodSubmissionRecord = self.store.get_hyperpod_submission(
                self.cluster_name, self.key
            )
        except LookupError:
            return None
        return record.state

    def lift_cordon(self) -> None:
        self.scheduler.core.node["spec"]["unschedulable"] = False


# --------------------------------------------------------------------------- #
# The first claim's ways of stopping before the provider call. Each returns the
# WAITING result the executor posts and whether the record's cordon is done.


def fence_hold(claims: LifecycleClaims) -> tuple[RemoteCommandResult, bool]:
    registry = claims.executor.fleet_registry
    claims.executor.fleet_registry = Fence()
    try:
        return claims.claim(cordoned=False), False
    finally:
        claims.executor.fleet_registry = registry


def lease_guard_hold(claims: LifecycleClaims) -> tuple[RemoteCommandResult, bool]:
    token = active_lease_guard.set(lambda: SHUTDOWN)
    try:
        return claims.claim(), True
    finally:
        active_lease_guard.reset(token)


def barrier_hold(claims: LifecycleClaims) -> tuple[RemoteCommandResult, bool]:
    # ``CommandDispatch._barrier_hold`` holds only the multi-node barrier
    # operations, never a lifecycle step: this is the record shape it posts.
    return (
        RemoteCommandResult(
            lease_token="lease-1",
            status=RemoteCommandStatus.WAITING,
            status_source="executor-barrier-unavailable",
            details={
                "multi_node_barrier_unavailable": True,
                "operation": claims.operation.value,
                "node_ids": [claims.node],
                "reason": "needs a multi-node barrier coordinator",
                "executor_id": "executor-a",
            },
        ),
        True,
    )


def _with_unavailable_api(
    claims: LifecycleClaims, result_details: dict[str, Any] | None = None
) -> RemoteCommandResult:
    core = claims.scheduler.core
    read_node = core.read_node

    def unavailable(_name: str) -> Any:
        raise api_exception(503)

    core.read_node = unavailable
    try:
        return claims.claim(result_details)
    finally:
        core.read_node = read_node


def kubernetes_503(claims: LifecycleClaims) -> tuple[RemoteCommandResult, bool]:
    """The Node read before the submission answers 503."""

    return _with_unavailable_api(claims), True


def control_plane_503(claims: LifecycleClaims) -> tuple[RemoteCommandResult, bool]:
    """The ledger read inside the submission and the one after it answer 503."""

    claims.ledger.failing_reads = 2
    return claims.claim(), True


def lease_guard_then_503(claims: LifecycleClaims) -> tuple[RemoteCommandResult, bool]:
    """The record that stayed stuck to the deadline: a retryable error on a
    command whose record already held a hold's keys (the error's own keys are
    stripped on replay, the hold's are not)."""

    held, cordoned = lease_guard_hold(claims)
    return _with_unavailable_api(claims, held.details), cordoned


FIRST_CLAIM_STOPS = [
    pytest.param(fence_hold, "fleet_preflight_blocked", id="fleet-fence-hold"),
    pytest.param(lease_guard_hold, "lease_guard_blocked", id="lease-guard-hold"),
    pytest.param(
        barrier_hold, "multi_node_barrier_unavailable", id="barrier-hold-shape"
    ),
    pytest.param(kubernetes_503, "retryable_adapter_error", id="kubernetes-503"),
    pytest.param(
        control_plane_503, "retryable_control_plane_error", id="control-plane-503"
    ),
    pytest.param(lease_guard_then_503, "retryable_adapter_error", id="hold-then-503"),
]


@pytest.mark.parametrize("provider", PROVIDERS)
@pytest.mark.parametrize(("stop", "marker"), FIRST_CLAIM_STOPS)
def test_a_reboot_held_before_submission_runs_the_first_claims_path_on_reclaim(
    provider: str,
    stop: Callable[[LifecycleClaims], tuple[RemoteCommandResult, bool]],
    marker: str,
) -> None:
    claims = LifecycleClaims(provider)

    held, cordoned = stop(claims)

    assert held.status is RemoteCommandStatus.WAITING, held
    assert held.details.get(marker) is True, held.details
    assert claims.submissions() == 0, "the first claim stopped before the provider"
    assert claims.ledger_state() is None, "nothing was reserved"
    checked = len(claims.ownership.checked)

    reclaimed = claims.claim(held.details, cordoned=cordoned)

    assert reclaimed.status is RemoteCommandStatus.WAITING, reclaimed
    assert reclaimed.details.get("submission_idempotency_key") == claims.key, (
        "the re-claim must submit the reboot, not resume a replay of the hold: "
        f"{sorted(reclaimed.details)}"
    )
    assert claims.submissions() == 1, "exactly one provider submission"
    assert claims.ledger_state() == "SUBMITTED", "the ledger holds the submission"
    assert claims.node in reclaimed.details.get("observed_isolation", {}), (
        "_observe_isolation ran before the submission"
    )
    assert len(claims.ownership.checked) > checked, (
        "the ownership guard ran before the submission"
    )
    assert reclaimed.details.get("revoked_agents") == [claims.node], (
        "the node's Agent is revoked before the reboot, as on a first claim"
    )
    assert marker not in reclaimed.details, (
        "the adapter's own outcome replaces the hold's record"
    )

    submitted = len(claims.ownership.checked)
    polled = claims.claim(reclaimed.details, cordoned=cordoned)
    polled_again = claims.claim(polled.details, cordoned=cordoned)

    assert polled_again.status is RemoteCommandStatus.WAITING, polled_again
    assert polled_again.details.get("submission_idempotency_key") == claims.key, (
        polled_again.details
    )
    assert claims.submissions() == 1, "a poll of the submitted reboot never resubmits"
    assert len(claims.ownership.checked) == submitted, (
        "a poll of a submitted reboot prepares no new mutation"
    )


@pytest.mark.parametrize("provider", PROVIDERS)
def test_a_submitted_reboot_held_on_a_later_poll_resumes_without_a_second_submission(
    provider: str,
) -> None:
    claims = LifecycleClaims(provider)
    submitted = claims.claim()
    assert submitted.details.get("submission_idempotency_key") == claims.key, (
        submitted.details
    )
    token = active_lease_guard.set(lambda: SHUTDOWN)
    try:
        held = claims.claim(submitted.details)
    finally:
        active_lease_guard.reset(token)
    assert held.details.get("lease_guard_blocked") is True, held.details
    assert held.details.get("submission_idempotency_key") == claims.key, (
        "the hold keeps the submission's continuation state"
    )
    checked = len(claims.ownership.checked)

    resumed = claims.claim(held.details)

    assert resumed.status is RemoteCommandStatus.WAITING, resumed
    assert resumed.details.get("submission_idempotency_key") == claims.key, resumed
    assert claims.submissions() == 1, "the reboot is submitted once"
    assert len(claims.ownership.checked) == checked, "resumed, not prepared again"


@pytest.mark.parametrize("provider", PROVIDERS)
def test_a_submission_whose_reply_was_lost_resumes_on_the_ledgers_word(
    provider: str,
) -> None:
    """The re-claim after a hold submitted the reboot but its WAITING never
    landed, so the next claim replays the hold's record again: the ledger, not
    the record, says the reboot is in, and it is not prepared or submitted
    again."""

    claims = LifecycleClaims(provider)
    held, cordoned = lease_guard_hold(claims)
    claims.claim(held.details, cordoned=cordoned)
    assert claims.submissions() == 1, "the re-claim submitted"
    checked = len(claims.ownership.checked)

    replayed = claims.claim(held.details, cordoned=cordoned)

    assert replayed.status is RemoteCommandStatus.WAITING, replayed
    assert claims.submissions() == 1, "a lost reply is never a second submission"
    assert len(claims.ownership.checked) == checked, (
        "a reboot on record is polled, not prepared again"
    )


@pytest.mark.parametrize("provider", PROVIDERS)
def test_a_reservation_of_unknown_outcome_is_never_submitted_again(
    provider: str,
) -> None:
    """The ledger holds the step's reservation, its outcome UNKNOWN (the
    provider call raised and may have taken effect), the record a hold's: the
    replay resumes and the provider is never asked again."""

    claims = LifecycleClaims(provider)
    claims.store.save_hyperpod_submission(
        HyperPodSubmissionRecord(
            cluster_name=claims.cluster_name,
            idempotency_key=claims.key,
            action=HyperPodAction.REBOOT,
            requested_node_identifiers=[claims.node],
            state="UNKNOWN",
            error="ReadTimeoutError: the call may have taken effect",
        )
    )
    held, cordoned = lease_guard_hold(claims)

    resumed = claims.claim(held.details, cordoned=cordoned)

    assert resumed.status is RemoteCommandStatus.WAITING, resumed
    assert claims.submissions() == 0, "a reservation on record is never resubmitted"
    assert claims.ledger_state() == "UNKNOWN", "the record is left as it was"
    assert claims.ownership.checked == [], "nothing new was prepared"


@pytest.mark.parametrize("provider", PROVIDERS)
def test_a_ledger_that_cannot_answer_never_submits_and_is_asked_again(
    provider: str,
) -> None:
    """Fail closed: a replay whose ledger lookup fails resumes and keeps its
    record; the next claim asks again and, told "none", submits once."""

    claims = LifecycleClaims(provider)
    held, cordoned = lease_guard_hold(claims)
    claims.ledger.failing_reads = 1

    unanswered = claims.claim(held.details, cordoned=cordoned)

    assert unanswered.status is RemoteCommandStatus.WAITING, unanswered
    assert unanswered.details.get("lease_guard_blocked") is True, (
        "the replayed record is kept for the next claim"
    )
    assert claims.submissions() == 0, "an unreadable ledger proves nothing"
    assert claims.ownership.checked == [], "no new mutation was prepared"

    answered = claims.claim(unanswered.details, cordoned=cordoned)

    assert answered.details.get("submission_idempotency_key") == claims.key, (
        answered.details
    )
    assert claims.submissions() == 1, "submitted once the ledger answered none"


@pytest.mark.parametrize("provider", PROVIDERS)
def test_an_isolation_lifted_while_held_is_refused_on_reclaim(provider: str) -> None:
    claims = LifecycleClaims(provider)
    held, cordoned = lease_guard_hold(claims)
    claims.lift_cordon()

    refused = claims.claim(held.details, cordoned=cordoned)

    assert refused.status is RemoteCommandStatus.FAILED, refused
    assert refused.details.get(LIFECYCLE_ISOLATION_DETAIL) == ISOLATION_NOT_ISOLATED, (
        "the b265 refusal, so the escalation hands the node over"
    )
    assert refused.details.get("provider_submission_record") == "none", (
        "the refusal says the ledger holds no reboot for the step"
    )
    assert refused.details["node_failures"] == {
        claims.node: [
            "RESTART_NODE not submitted, isolation not observed: "
            "spec.unschedulable is not true"
        ]
    }, refused.details
    assert claims.submissions() == 0, "nothing reached the provider"


WARM_SPARE_STOPS = [
    pytest.param(lease_guard_hold, id="lease-guard-hold"),
    pytest.param(kubernetes_503, id="kubernetes-503"),
]


def _replacement(
    stop: Callable[[LifecycleClaims], tuple[RemoteCommandResult, bool]],
) -> tuple[LifecycleClaims, FakeSpareCoordinator, RemoteCommandResult, bool]:
    spares = FakeSpareCoordinator(
        SpareAllocation(
            applicable=True, sufficient=True, required=1, selected_node_ids=(NODE_B,)
        )
    )
    claims = LifecycleClaims("eks", OP.REPLACE_NODE, spares=spares)
    held, cordoned = stop(claims)
    assert held.status is RemoteCommandStatus.WAITING, held
    assert spares.calls == [], "the first claim reached no spare"
    return claims, spares, held, cordoned


@pytest.mark.parametrize("stop", WARM_SPARE_STOPS)
def test_a_warm_spare_replacement_held_before_it_started_reads_the_isolation_first(
    stop: Callable[[LifecycleClaims], tuple[RemoteCommandResult, bool]],
) -> None:
    claims, spares, held, cordoned = _replacement(stop)
    claims.lift_cordon()

    refused = claims.claim(held.details, cordoned=cordoned)

    assert refused.status is RemoteCommandStatus.FAILED, refused
    assert refused.details.get(LIFECYCLE_ISOLATION_DETAIL) == ISOLATION_NOT_ISOLATED, (
        "the faulty node's isolation is read before any failover"
    )
    assert spares.calls == [], "no spare is allocated for an unisolated node"


@pytest.mark.parametrize("stop", WARM_SPARE_STOPS)
def test_a_warm_spare_replacement_held_before_it_started_fails_over_on_reclaim(
    stop: Callable[[LifecycleClaims], tuple[RemoteCommandResult, bool]],
) -> None:
    claims, spares, held, cordoned = _replacement(stop)
    checked = len(claims.ownership.checked)

    replaced = claims.claim(held.details, cordoned=cordoned)

    assert replaced.status is RemoteCommandStatus.SUCCEEDED, replaced
    assert replaced.details.get("action") == "SPARE_FAILOVER", replaced.details
    assert len(spares.calls) == 1, "one allocation, on the first claim's path"
    assert len(claims.ownership.checked) > checked, "the ownership guard ran"
    assert replaced.details.get("revoked_agents") == [claims.node], (
        "the faulty node's Agent is revoked as on a first claim"
    )

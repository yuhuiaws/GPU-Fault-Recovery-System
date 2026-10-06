"""A replacement plan compiles for the cluster's own lifecycle adapter.

Live (gpu-eks-cluster, DESTR-003): the synthetic replacement finding carried
the site's ``hyperpod-v1`` -- the version the workload was submitted under and
its observation reported -- so REPLACE_NODE compiled with
``gpu-fault-hyperpod-adapter`` as owner, no executor on the eks cluster
claimed it, and the step failed on the unclaimed deadline 932 s later with the
node quarantined by the hand-off. The cluster's registration names its
provider and its pinned Runtime Profile; planning now reads both.
"""

from __future__ import annotations

from gpu_fault.app import default_simulated_profile
from gpu_fault.capabilities import compile_runtime_profile
from gpu_fault.models import (
    CapabilityClaim,
    CapabilityMode,
    CapabilityName,
    Environment,
    ObservedCapability,
    RuntimeProfile,
    WorkflowOperation,
    WorkflowStatus,
)
from gpu_fault.orchestration.provider_lifecycle_owner import (
    PROVIDER_LIFECYCLE_OPERATIONS,
    pinned_profile_version,
    provider_owner_errors,
)
from gpu_fault.regional import RegionalClusterRegistration
from tests._builders import (
    attempt_observation,
    container_observation,
    node_health_finding,
)

from ._support import (
    NOW,
    ApplicationContext,
    NodeHealthCategory,
    RecoveryAction,
    WorkloadState,
)

WORKLOAD = "training/pytorchjob/job-a"


def _profile(version: str, lifecycle_owner: str) -> object:
    """The simulated profile with the node lifecycle owned by ``lifecycle_owner``
    through the regional executor, under another version name."""

    simulated = default_simulated_profile()

    def owner(capability: CapabilityName) -> str:
        if capability in (CapabilityName.NODE_REBOOT, CapabilityName.NODE_REPLACE):
            return lifecycle_owner
        return "simulated-runtime"

    def adapter(capability: CapabilityName) -> str:
        if capability in (CapabilityName.NODE_REBOOT, CapabilityName.NODE_REPLACE):
            return "regional-cluster-executor"
        return "simulated"

    capabilities = [item.capability for item in simulated.capabilities]
    profile = RuntimeProfile(
        cluster_id="gpu-eks-cluster",
        environment=Environment.KUBERNETES,
        profile_version=version,
        claims=[
            CapabilityClaim(
                capability=capability,
                mode=CapabilityMode.OWN,
                owner=owner(capability),
                adapter=adapter(capability),
            )
            for capability in capabilities
        ],
        observed=[
            ObservedCapability(
                capability=capability,
                owner=owner(capability),
                available=True,
                version="v1",
            )
            for capability in capabilities
        ],
    )
    return compile_runtime_profile(profile)


def _registration(
    *, provider: str, pinned: str, cluster_id: str = "cluster-a"
) -> RegionalClusterRegistration:
    return RegionalClusterRegistration(
        cluster_id=cluster_id,
        region="us-west-2",
        hyperpod_cluster_name="" if provider == "eks-managed" else "gpu-a",
        eks_cluster_arn="arn:aws:eks:us-west-2:123456789012:cluster/gpu-eks-cluster",
        provider=provider,
        runtime_profile_version=pinned,
        token_sha256="a" * 64,
        agent_endpoint_allowed_cidrs=["10.0.0.0/16"],
    )


def _observe(context: ApplicationContext) -> None:
    context.store.save_attempt_observation(
        attempt_observation(
            "job-a",
            "job-a-a001",
            NOW,
            containers=[
                container_observation(
                    "pod-a", "worker-a", 0, "node-a", gpu_uuids=["GPU-a"]
                )
            ],
            workload_ids=[WORKLOAD],
        )
    )


def _finding(named_version: str):
    return node_health_finding(
        "finding-eks-warm-spare",
        "eks-warm-spare",
        observed_at=NOW,
        category=NodeHealthCategory.GPU,
        severity="critical",
        reason="synthetic warm-spare replacement test",
        recommended_action=RecoveryAction.REPLACE_NODE,
        runtime_profile_version=named_version,
        workload_state=WorkloadState.ACTIVE,
        affected_workload_ids=[WORKLOAD],
        diagnostic_parameters={"replacement_strategy": "HEALTHY_WARM_SPARE_ONLY"},
    )


def test_a_finding_naming_the_site_profile_is_repinned_to_the_clusters_own(
    context: ApplicationContext,
) -> None:
    """The live shape: the finding names ``hyperpod-v1``-like owners while the
    cluster is registered eks-managed and pinned to its own profile. The plan
    compiles REPLACE_NODE for ``gpu-fault-eks-adapter`` and says why."""

    context.store.save_profile(_profile("eks-v1", "gpu-fault-eks-adapter"))
    context.store.save_profile(_profile("site-hyperpod", "gpu-fault-hyperpod-adapter"))
    context.store.save_regional_cluster(
        _registration(provider="eks-managed", pinned="eks-v1")
    )
    _observe(context)

    incident, workflow = context.orchestrator.ingest_node_health(
        _finding("site-hyperpod")
    )

    replace = next(
        step
        for step in workflow.official_steps
        if step.operation is WorkflowOperation.REPLACE_NODE
    )
    assert replace.execution_owner == "gpu-fault-eks-adapter"
    assert workflow.runtime_profile_version == "eks-v1"
    assert workflow.status is WorkflowStatus.PENDING, workflow.status
    assert any(
        "re-pinned from site-hyperpod" in reason and "eks-v1" in reason
        for reason in incident.reasons
    ), incident.reasons


def test_a_provider_step_nobody_can_claim_is_refused_at_plan_time(
    context: ApplicationContext,
) -> None:
    """A misconfigured pin: the eks cluster's own profile still names the
    HyperPod adapter. The plan is BLOCKED with the owner in words instead of
    a REPLACE_NODE that times out unclaimed and drains the node."""

    context.store.save_profile(
        _profile("eks-misconfigured", "gpu-fault-hyperpod-adapter")
    )
    context.store.save_regional_cluster(
        _registration(provider="eks-managed", pinned="eks-misconfigured")
    )
    _observe(context)

    incident, workflow = context.orchestrator.ingest_node_health(
        _finding("eks-misconfigured")
    )

    assert workflow.status is WorkflowStatus.BLOCKED, workflow.status
    assert any(
        "REPLACE_NODE is owned by gpu-fault-hyperpod-adapter" in reason
        and "eks-managed cluster's executor never advertises" in reason
        and "gpu-fault-eks-adapter" in reason
        for reason in workflow.blocked_reasons
    ), workflow.blocked_reasons
    assert incident.reasons == ["synthetic warm-spare replacement test"], (
        "the finding's own version was the pin: nothing was re-pinned"
    )


def test_an_unregistered_cluster_plans_exactly_as_before(
    context: ApplicationContext,
) -> None:
    _observe(context)
    _, workflow = context.orchestrator.ingest_node_health(_finding("simulated-v1"))
    replace = next(
        step
        for step in workflow.official_steps
        if step.operation is WorkflowOperation.REPLACE_NODE
    )
    assert replace.execution_owner == "simulated-runtime"
    assert workflow.runtime_profile_version == "simulated-v1"
    assert workflow.status is WorkflowStatus.PENDING


def test_the_guard_reads_only_provider_lifecycle_steps() -> None:
    assert PROVIDER_LIFECYCLE_OPERATIONS == frozenset(
        {WorkflowOperation.RESTART_NODE, WorkflowOperation.REPLACE_NODE}
    )
    hyperpod = _registration(provider="hyperpod", pinned="")
    assert pinned_profile_version(hyperpod, "named") == "named", (
        "an empty pin inherits the site profile, which is what the finding named"
    )
    assert pinned_profile_version(None, "named") == "named"
    eks = _registration(provider="eks-managed", pinned="eks-v1")
    from tests._builders import workflow_step

    steps = [
        workflow_step(
            WorkflowOperation.MARK_UNSCHEDULABLE, "gpu-fault-kubernetes-adapter"
        ),
        workflow_step(WorkflowOperation.REPLACE_NODE, "gpu-fault-hyperpod-adapter"),
        workflow_step(WorkflowOperation.RESTART_NODE, "gpu-fault-eks-adapter"),
    ]
    errors = provider_owner_errors(steps, eks)
    assert len(errors) == 1 and errors[0].startswith(
        "REPLACE_NODE is owned by gpu-fault-hyperpod-adapter, which a eks-managed "
    ), errors
    assert provider_owner_errors(steps, None) == []
    assert provider_owner_errors(steps[2:], eks) == []

"""``IncidentOrchestrator.compile_branch_steps`` -- the compiler the branch
escalator borrows from the orchestrator (F-N1).

``BranchEscalator`` is handed this callable by ``app/context.py`` and treats an
empty list as "this branch cannot be escalated in place", which falls back to
the whole-workflow failure path. So each way of returning ``[]`` is a
behaviour, not an implementation detail: a workflow with no runtime profile
version, a profile that is no longer in the store, and a rung whose capability
nobody owns must all compile nothing rather than emit an unownable step for
the executor to dispatch. The success path has to carry the branch's own node
and GPU scope, not the workflow's.
"""

from __future__ import annotations

import logging

from gpu_fault.app import default_simulated_profile
from gpu_fault.capabilities import compile_runtime_profile
from gpu_fault.execution.branch_escalation import ESCALATION_TAIL
from gpu_fault.models import (
    CapabilityClaim,
    CapabilityMode,
    CapabilityName,
    Environment,
    ObservedCapability,
    RuntimeProfile,
    WorkflowOperation,
    WorkflowRequest,
)
from gpu_fault.orchestration import IncidentOrchestrator
from tests._builders import build_store, workflow_request, workflow_step

REBOOT_RUNG = [WorkflowOperation.RESTART_NODE, *ESCALATION_TAIL]


def _orchestrator() -> IncidentOrchestrator:
    """Wired exactly like ``app/context.py``: one store holding the profile."""

    store = build_store()
    store.save_profile(default_simulated_profile())
    return IncidentOrchestrator(store)


def _job_workflow(profile_version: str | None) -> WorkflowRequest:
    return workflow_request(
        "wf-job",
        "inc-job",
        fencing_token=1,
        runtime_profile_version=profile_version,
        official_action=WorkflowOperation.RESET_GPU.value,
        official_steps=[
            workflow_step(
                WorkflowOperation.STOP_WORKLOADS,
                node_ids=["node-a", "node-b"],
                workload_ids=["training/job/job-a"],
            ),
            workflow_step(
                WorkflowOperation.RESET_GPU, node_ids=["node-b"], gpu_uuids=["GPU-b"]
            ),
        ],
    )


def _profile_without(capability: CapabilityName, version: str):
    """The simulated profile minus one capability's claim and observation."""

    return compile_runtime_profile(
        RuntimeProfile(
            cluster_id="cluster-a",
            environment=Environment.KUBERNETES,
            profile_version=version,
            claims=[
                CapabilityClaim(
                    capability=item.capability,
                    mode=CapabilityMode.OWN,
                    owner="simulated-runtime",
                    adapter="simulated",
                )
                for item in default_simulated_profile().capabilities
                if item.capability is not capability
            ],
            observed=[
                ObservedCapability(
                    capability=item.capability,
                    owner="simulated-runtime",
                    available=True,
                    version="v1",
                )
                for item in default_simulated_profile().capabilities
                if item.capability is not capability
            ],
        )
    )


def test_a_branch_rung_compiles_against_the_workflows_own_runtime_profile():
    orchestrator = _orchestrator()

    steps = orchestrator.compile_branch_steps(
        _job_workflow("simulated-v1"), REBOOT_RUNG, "node-b", ["GPU-b"]
    )

    assert [step.operation for step in steps] == REBOOT_RUNG, (
        "the rung and its validation/release tail compile in order"
    )
    assert [step.execution_owner for step in steps] == ["simulated-runtime"] * len(
        REBOOT_RUNG
    ), "each step names the owner the profile claims for its capability"
    assert {tuple(step.node_ids) for step in steps} == {("node-b",)}, (
        "the branch's node, not every node the workflow touches"
    )
    assert {tuple(step.gpu_uuids) for step in steps} == {("GPU-b",)}, (
        "the failed step's GPUs carry into the escalated branch"
    )


def test_a_workflow_without_a_runtime_profile_version_compiles_no_branch_steps():
    """The guard is the orchestrator's, not the store's: a workflow that
    recorded no profile version must compile nothing even against a store
    willing to answer for a missing one."""

    permissive = build_store(get_profile=lambda _version: default_simulated_profile())
    orchestrator = IncidentOrchestrator(permissive)

    steps = orchestrator.compile_branch_steps(
        _job_workflow(None), REBOOT_RUNG, "node-b", ["GPU-b"]
    )

    assert steps == [], (
        "with no profile recorded there is nothing to resolve owners against"
    )


def test_a_branch_rung_compiles_nothing_when_the_profile_left_the_store():
    orchestrator = _orchestrator()

    steps = orchestrator.compile_branch_steps(
        _job_workflow("retired-v0"), REBOOT_RUNG, "node-b", ["GPU-b"]
    )

    assert steps == [], "a missing profile is a compile failure, not a fallback"


def test_a_branch_rung_nobody_owns_compiles_nothing_and_warns(caplog):
    orchestrator = _orchestrator()
    orchestrator.store.save_profile(
        _profile_without(CapabilityName.NODE_REBOOT, "no-reboot-v1")
    )

    with caplog.at_level(logging.WARNING, logger="gpu_fault.orchestration.coordinator"):
        steps = orchestrator.compile_branch_steps(
            _job_workflow("no-reboot-v1"), REBOOT_RUNG, "node-b", ["GPU-b"]
        )

    assert steps == [], (
        "a partially compiled branch must not be planned: the tail would run "
        "without the rung that was supposed to repair the node"
    )
    warnings = [
        record.getMessage()
        for record in caplog.records
        if record.levelno == logging.WARNING
    ]
    assert len(warnings) == 1, f"expected one warning, got {warnings}"
    assert "wf-job" in warnings[0], "the operator needs the workflow that failed"
    assert "node-b" in warnings[0], "and the node whose branch could not escalate"
    assert "nodeReboot" in warnings[0], (
        "and the capability with no owner, which is the actionable part"
    )

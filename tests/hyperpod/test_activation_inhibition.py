from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import pytest

from gpu_fault.execution import WorkflowStepOutcome
from gpu_fault.hyperpod_spares import (
    ACTIVATION_FORBIDDEN_REASON,
    SPARE_POOL_STATE_ANNOTATION,
    SPARE_RESERVATION_ANNOTATION,
    SpareActivationForbidden,
    SpareHealthPending,
)
from gpu_fault.models import WorkflowOperation, WorkflowStepStatus
from tests.hyperpod._cov95_runtime_failover import FailoverHarness
from tests.hyperpod.test_hyperpod_spares import (
    coordinator,
    hyperpod_node,
    kubernetes_node,
)


def healthy_coordinator():
    nodes = [
        hyperpod_node("fault", "i-fault"),
        hyperpod_node("spare", "i-spare", spare=True),
    ]
    core_nodes = {
        "hyperpod-i-fault": kubernetes_node(),
        "hyperpod-i-spare": kubernetes_node(),
    }
    for node in nodes:
        core_nodes[node.kubernetes_labels["kubernetes.io/hostname"]]["metadata"][
            "labels"
        ].update(
            {
                **node.kubernetes_labels,
                "sagemaker.amazonaws.com/instance-group-name": node.instance_group_name,
                "node.kubernetes.io/instance-type": node.instance_type,
            }
        )
    return coordinator(nodes, core_nodes)


@pytest.mark.parametrize("local_only", [False, True])
@pytest.mark.parametrize("marker", [True, False, None, 0, 1, "true", "", {}, []])
def test_real_healthy_allocation_is_inhibited_for_every_present_marker(
    local_only: bool, marker: Any
) -> None:
    service, store = healthy_coordinator()
    result = service.allocate(
        cluster_id="hp-cluster",
        incident_id="inhibited",
        fault_node_ids=["hyperpod-i-fault" if local_only else "fault"],
        local_only=local_only,
        activation_forbidden=marker,
    )
    assert not result.sufficient and result.selected_node_ids == ()
    assert result.activation_inhibited is True
    assert result.reason == (
        ACTIVATION_FORBIDDEN_REASON
        if marker is True
        else "ACTIVATION_FORBIDDEN: invalid activation_forbidden marker"
    )
    assert service.core.patches == []
    assert store.list_notifications() == []


def test_absence_preserves_healthy_allocation_and_same_incident_noop() -> None:
    service, _ = healthy_coordinator()
    first = service.allocate(
        cluster_id="hp-cluster", incident_id="ordinary", fault_node_ids=["fault"]
    )
    assert first.sufficient and first.selected_node_ids == ("hyperpod-i-spare",)
    assert first.activation_inhibited is False
    assert len(service.core.patches) == 1
    service.reserve(service.lifecycle.nodes[1], "hyperpod-i-spare", "ordinary")
    assert len(service.core.patches) == 1
    assert service.core.nodes["hyperpod-i-spare"]["spec"]["unschedulable"] is False


def test_missing_spare_pool_remains_an_ordinary_shortage_without_guard_firing() -> None:
    service, _ = healthy_coordinator()
    node = service.lifecycle.nodes[1]
    labels = dict(node.kubernetes_labels)
    labels.pop("gpu-fault.io/spare")
    service.lifecycle.nodes[1] = node.model_copy(update={"kubernetes_labels": labels})
    service.core.nodes["hyperpod-i-spare"]["metadata"]["labels"].pop(
        "gpu-fault.io/spare"
    )
    arguments = {
        "cluster_id": "hp-cluster",
        "incident_id": "empty",
        "fault_node_ids": ["fault"],
    }
    ordinary = service.allocate(**arguments)
    inhibited = service.allocate(**arguments, activation_forbidden=True)
    assert ordinary == inhibited
    assert not inhibited.applicable and not inhibited.activation_inhibited
    assert service.core.patches == []


@pytest.mark.parametrize(
    "defect",
    ["topology", "not-ready", "fleet", "provider", "occupied", "reserved", "client"],
)
def test_existing_shortage_reason_is_identical_with_and_without_inhibition(
    defect: str,
) -> None:
    service, _ = healthy_coordinator()
    spare = service.core.nodes["hyperpod-i-spare"]
    checker = None
    if defect == "topology":
        service.lifecycle.nodes[1] = service.lifecycle.nodes[1].model_copy(
            update={"instance_group_name": "incompatible"}
        )
    elif defect == "not-ready":
        spare["status"]["conditions"][0]["status"] = "False"
    elif defect == "fleet":
        service.registry.readiness = lambda *args: SimpleNamespace(ready=False)
    elif defect == "provider":
        service.lifecycle.nodes[1] = service.lifecycle.nodes[1].model_copy(
            update={"status": "Stopped"}
        )
    elif defect == "occupied":
        service.core.pod_batches = [
            [
                {
                    "metadata": {"name": "holder", "namespace": "test"},
                    "spec": {
                        "containers": [{"resources": {"limits": {"nvidia.com/gpu": 1}}}]
                    },
                    "status": {"phase": "Running"},
                }
            ]
        ]
    elif defect == "reserved":
        spare["metadata"]["annotations"][SPARE_RESERVATION_ANNOTATION] = "other"
    else:

        def checker(*args):
            return ["GPU clients remain"]

    arguments = {
        "cluster_id": "hp-cluster",
        "incident_id": "shortage",
        "fault_node_ids": ["fault"],
        "gpu_client_checker": checker,
    }
    ordinary = service.allocate(**arguments)
    guarded = service.allocate(**arguments, activation_forbidden=True)
    assert not ordinary.sufficient and not guarded.sufficient
    assert guarded.reason == ordinary.reason
    assert guarded.rejected_candidates == ordinary.rejected_candidates
    assert not guarded.activation_inhibited and not ordinary.activation_inhibited
    assert "ACTIVATION_FORBIDDEN" not in str(guarded.reason)
    assert service.core.patches == []


def test_activation_phase_occupancy_rejection_remains_primary() -> None:
    service, _ = healthy_coordinator()
    phases = []

    def clients(node, name, phase):
        phases.append(phase)
        return ["late GPU client"] if phase == "activation" else []

    arguments = {
        "cluster_id": "hp-cluster",
        "incident_id": "late-occupancy",
        "fault_node_ids": ["fault"],
        "gpu_client_checker": clients,
    }
    ordinary = service.allocate(**arguments)
    inhibited = service.allocate(**arguments, activation_forbidden=True)
    assert ordinary.reason == inhibited.reason
    assert "late GPU client" in inhibited.reason
    assert phases == ["candidate", "activation", "candidate", "activation"]
    assert service.core.patches == []


def test_newly_declared_healthy_candidate_cannot_bypass_inhibition() -> None:
    service, store = healthy_coordinator()
    service.core.nodes["hyperpod-i-spare"]["status"]["conditions"][0]["status"] = (
        "False"
    )
    newcomer = hyperpod_node("new-spare", "i-new", spare=True)
    service.lifecycle.nodes.append(newcomer)
    service.core.nodes["hyperpod-i-new"] = kubernetes_node()
    store.save_agent(
        SimpleNamespace(
            cluster_id="hp-cluster",
            node_id="hyperpod-i-new",
            runtime_profile_version="simulated-v1",
        )
    )
    result = service.allocate(
        cluster_id="hp-cluster",
        incident_id="new-candidate",
        fault_node_ids=["fault"],
        activation_forbidden=True,
    )
    assert result.reason == ACTIVATION_FORBIDDEN_REASON
    assert service.core.patches == []


@pytest.mark.parametrize("reserved", [False, True])
def test_direct_reserve_cannot_activate_or_accept_previous_reservation(
    reserved: bool,
) -> None:
    service, _ = healthy_coordinator()
    if reserved:
        service.core.nodes["hyperpod-i-spare"]["metadata"]["annotations"].update(
            {
                SPARE_RESERVATION_ANNOTATION: "same",
                SPARE_POOL_STATE_ANNOTATION: "ALLOCATED",
            }
        )
    with pytest.raises(SpareActivationForbidden, match="ACTIVATION_FORBIDDEN"):
        service.reserve(
            service.lifecycle.nodes[1],
            "hyperpod-i-spare",
            "same",
            activation_forbidden=True,
        )
    assert service.core.patches == []


def real_adapter(*, inhibited: bool = True) -> FailoverHarness:
    harness = FailoverHarness()
    fault = hyperpod_node("node-a", "i-old")
    spare = hyperpod_node("spare-logical", "i-spare", spare=True)
    spare.kubernetes_labels["kubernetes.io/hostname"] = "spare-a"
    harness.core.nodes["spare-a"]["spec"]["unschedulable"] = True
    for provider, name in ((fault, "node-a"), (spare, "spare-a")):
        harness.core.nodes[name]["metadata"]["labels"].update(
            {
                **provider.kubernetes_labels,
                "sagemaker.amazonaws.com/instance-group-name": provider.instance_group_name,
                "node.kubernetes.io/instance-type": provider.instance_type,
            }
        )
    service, _ = coordinator([fault, spare], harness.core.nodes, core=harness.core)
    service.store.save_agent(
        SimpleNamespace(
            cluster_id=harness.context.incident.cluster_id,
            node_id="spare-a",
            runtime_profile_version="simulated-v1",
        )
    )
    harness.adapter.spare_coordinator = service
    parameters = dict(harness.context.step.parameters)
    if inhibited:
        parameters["activation_forbidden"] = True
    step = harness.context.step.model_copy(update={"parameters": parameters})
    harness.context = replace(
        harness.context,
        step=step,
        workflow=harness.context.workflow.model_copy(update={"official_steps": [step]}),
    )
    return harness


@pytest.mark.parametrize("cached", [False, True])
def test_actual_adapter_and_coordinator_never_patch_inhibited_healthy_candidates(
    cached: bool,
) -> None:
    harness = real_adapter()
    if cached:
        harness.core.nodes["spare-a"]["metadata"]["annotations"].update(
            {
                SPARE_RESERVATION_ANNOTATION: harness.context.incident.incident_id,
                SPARE_POOL_STATE_ANNOTATION: "ALLOCATED",
            }
        )
        harness.follow(
            WorkflowStepOutcome.waiting(
                operation_id="old-operation",
                details={
                    "spare_failover_pending": True,
                    "activated_spare_nodes": ["spare-a"],
                },
            )
        )
    result = harness.execute()
    assert result.status is WorkflowStepStatus.FAILED
    assert result.error == ACTIVATION_FORBIDDEN_REASON
    assert result.details["activation_inhibited"] is True
    assert harness.core.patches == []
    assert harness.provider.submissions == []
    assert all(
        c.step.operation is WorkflowOperation.VERIFY_NO_GPU_CLIENTS
        for c in harness.actions.contexts
    ), "inhibited allocation may only verify GPU client readiness"


def test_explicit_confirmation_cannot_override_inhibition() -> None:
    harness = real_adapter()
    harness.follow(
        WorkflowStepOutcome.waiting(operation_id="confirmed-operation", details={})
    )
    harness.context = replace(
        harness.context,
        request=harness.context.request.model_copy(
            update={"confirmed_adapter_operation_ids": ["confirmed-operation"]}
        ),
    )
    result = harness.execute()
    assert (
        result.status is WorkflowStepStatus.FAILED
        and result.error == ACTIVATION_FORBIDDEN_REASON
    )
    assert harness.core.patches == [] and harness.provider.submissions == []


def test_default_adapter_keeps_real_coordinator_activation_behavior() -> None:
    harness = real_adapter(inhibited=False)
    result = harness.execute()
    assert result.status is WorkflowStepStatus.SUCCEEDED, result
    assert result.details["activated_spare_nodes"] == ["spare-a"]
    assert harness.core.patches, "unmarked healthy allocation must patch the real spare"
    assert harness.provider.submissions == []


@pytest.mark.parametrize(
    "method",
    ["_waiting_spare_failover", "_replacement_confirmation", "_spare_failover_details"],
)
@pytest.mark.parametrize("healthy", [False, True])
@pytest.mark.parametrize("cached", [False, True])
def test_direct_confirmation_boundaries_preserve_health_and_inhibit_activation(
    method: str, healthy: bool, cached: bool
) -> None:
    harness = real_adapter()
    if not healthy:
        harness.core.nodes["spare-a"]["status"]["conditions"][0]["status"] = "False"

    def call():
        if method == "_spare_failover_details":
            return getattr(harness.adapter, method)(
                harness.context,
                ["spare-a"] if cached else [],
                revoked_agents=[],
                provider_baselines={},
            )
        return getattr(harness.adapter, method)(
            harness.context, {"activated_spare_nodes": ["spare-a"]} if cached else {}
        )

    message = "ACTIVATION_FORBIDDEN" if healthy else "Kubernetes node is not Ready"
    expected = SpareActivationForbidden if healthy or cached else ValueError
    with pytest.raises(expected, match=message):
        call()
    assert harness.core.patches == []
    assert harness.provider.submissions == []
    assert all(
        c.step.operation is WorkflowOperation.VERIFY_NO_GPU_CLIENTS
        for c in harness.actions.contexts
    ), "inhibited confirmation may only verify GPU client readiness"


def test_cached_activation_is_not_accepted_as_ordinary_shortage_evidence() -> None:
    harness = real_adapter()
    harness.core.nodes["spare-a"]["status"]["conditions"][0]["status"] = "False"
    harness.follow(
        WorkflowStepOutcome.waiting(
            details={
                "spare_failover_pending": True,
                "activated_spare_nodes": ["spare-a"],
            }
        )
    )
    result = harness.execute()
    assert result.status is WorkflowStepStatus.FAILED
    assert "Kubernetes node is not Ready" in result.error
    assert result.details == {
        "activation_inhibited": True,
        "cached_activation_rejected": True,
    }
    assert harness.core.patches == []


def test_missing_coordinator_cannot_confirm_an_inhibited_replacement() -> None:
    harness = real_adapter()
    harness.adapter.spare_coordinator = None
    harness.follow(WorkflowStepOutcome.waiting(operation_id="prior", details={}))
    result = harness.execute()
    assert result.status is WorkflowStepStatus.FAILED
    assert result.error == ACTIVATION_FORBIDDEN_REASON
    assert result.details == {"activation_inhibited": True}
    with pytest.raises(SpareActivationForbidden, match="ACTIVATION_FORBIDDEN"):
        getattr(harness.adapter, "_replacement_confirmation")(harness.context, {})
    assert harness.core.patches == [] and harness.provider.submissions == []


def test_pending_real_health_check_stays_pending_without_activation() -> None:
    harness = real_adapter()
    harness.actions.outcomes[WorkflowOperation.VERIFY_NO_GPU_CLIENTS] = (
        WorkflowStepOutcome.waiting(operation_id="health-pending", details={})
    )
    result = harness.execute()
    assert result.status is WorkflowStepStatus.WAITING
    with pytest.raises(SpareHealthPending):
        getattr(harness.adapter, "_waiting_spare_failover")(harness.context, {})
    assert harness.core.patches == [] and harness.provider.submissions == []


def test_ordinary_reboot_is_not_intercepted_by_spare_inhibition() -> None:
    harness = real_adapter(inhibited=False)
    step = harness.context.step.model_copy(
        update={"operation": WorkflowOperation.RESTART_NODE}
    )
    harness.context = replace(
        harness.context,
        step=step,
        workflow=harness.context.workflow.model_copy(update={"official_steps": [step]}),
    )
    with pytest.raises(AssertionError, match="provider mutation"):
        harness.execute()
    assert len(harness.provider.submissions) == 1
    assert harness.core.patches == []

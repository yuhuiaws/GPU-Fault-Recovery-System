from __future__ import annotations

from copy import deepcopy

from gpu_fault.hyperpod import (
    HyperPodAction,
    HyperPodAdapterConfig,
    HyperPodLifecycleAdapter,
)
from gpu_fault.models import WorkflowOperation
from gpu_fault.store import InMemoryStore
from tests._builders import fault_incident, workflow_request, workflow_step
from tests.hyperpod.test_hyperpod import FakeHyperPodClient

OWNER = "gpu-fault-hyperpod-adapter"
TARGET = "worker-group-1"


class RecordingProvider(FakeHyperPodClient):
    def __init__(self):
        super().__init__(node_recovery="None")
        self.cluster_reads = []
        self.node_reads = []
        self.summary_override = None
        self.detail_override = None
        self.submit_error = None
        self.submit_response = None

    def describe_cluster(self, **kwargs):
        self.cluster_reads.append(kwargs)
        return super().describe_cluster(**kwargs)

    def list_cluster_nodes(self, **kwargs):
        if self.summary_override is not None:
            self.list_requests.append(kwargs)
            return deepcopy(self.summary_override)
        return super().list_cluster_nodes(**kwargs)

    def describe_cluster_node(self, **kwargs):
        self.node_reads.append(kwargs)
        if self.detail_override is not None:
            return deepcopy(self.detail_override)
        return super().describe_cluster_node(**kwargs)

    def batch_reboot_cluster_nodes(self, **kwargs):
        result = super().batch_reboot_cluster_nodes(**kwargs)
        if self.submit_error is not None:
            raise self.submit_error
        return (
            deepcopy(self.submit_response)
            if self.submit_response is not None
            else result
        )

    def batch_replace_cluster_nodes(self, **kwargs):
        raise AssertionError("provider replacement must remain unreachable")


class RootHarness:
    def __init__(self, *, durable=True, store=None, **config):
        self.store = InMemoryStore() if store is None else store
        self.client = RecordingProvider()
        self.config = HyperPodAdapterConfig(
            **{
                "cluster_name": "hp-cluster",
                "region_name": "us-east-1",
                "execution_enabled": True,
                **config,
            }
        )
        self.adapter = HyperPodLifecycleAdapter(
            self.config, client=self.client, store=self.store if durable else None
        )
        self.arguments = {
            "isolation_verified_nodes": [TARGET],
            "confirm_cluster_name": self.config.cluster_name,
            "workflow_fencing_token": 7,
            "expected_fencing_token": 7,
            "idempotency_key": "unit-provider-extra/reboot/0",
        }

    def submit(self, identifiers=None, **overrides):
        return self.adapter.submit(
            HyperPodAction.REBOOT,
            [TARGET] if identifiers is None else identifiers,
            **{**self.arguments, **overrides},
        )

    def workflow(self, **updates):
        incident = fault_incident(
            "inc-provider-extra", "event-provider-extra", fencing_token=7
        )
        workflow = workflow_request(
            "wf-provider-extra",
            incident.incident_id,
            **{
                "fencing_token": 7,
                "official_steps": [
                    workflow_step(
                        WorkflowOperation.RESTART_NODE, OWNER, node_ids=[TARGET]
                    )
                ],
                **updates,
            },
        )
        self.store.save_incident_and_workflow(
            incident.model_copy(update={"workflow_request_id": workflow.request_id}),
            workflow,
        )
        return self.store.get_workflow(workflow.request_id)

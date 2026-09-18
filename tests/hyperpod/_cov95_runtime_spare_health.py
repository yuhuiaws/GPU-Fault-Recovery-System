from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from gpu_fault.app import default_simulated_profile
from gpu_fault.models import WorkflowOperation, WorkflowStatus
from gpu_fault.orchestration import IncidentOrchestrator
from gpu_fault.spare_health import INCIDENT_ANNOTATION, HyperPodSpareHealthController
from tests._builders import fault_incident, workflow_request, workflow_step
from tests.hyperpod._cov95_runtime_spares import TypedCore
from tests.hyperpod.test_hyperpod_spares import (
    FakeCore,
    MutableClock,
    coordinator,
    hyperpod_node,
    kubernetes_node,
)

NODE = "hyperpod-i-spare1"


class HealthHarness:
    def __init__(
        self, *, ready: bool = True, typed: bool = False, **options: Any
    ) -> None:
        self.node = kubernetes_node(ready=ready)
        self.annotations = self.node["metadata"]["annotations"]
        self.clock = MutableClock(datetime.now(timezone.utc))
        self.sent: list[str] = []
        self.core = (TypedCore if typed else FakeCore)({NODE: self.node})
        self.coordinator, self.store = coordinator(
            [hyperpod_node("worker-spare", "i-spare1", spare=True)],
            self.core.nodes,
            core=self.core,
        )
        self.store.save_profile(default_simulated_profile())
        self.controller = HyperPodSpareHealthController(
            self.coordinator,
            IncidentOrchestrator(self.store),
            self.store,
            now=self.clock,
            alert_sender=self.sent.append,
            **options,
        )

    def incident(self, status: WorkflowStatus | None = None) -> None:
        incident = fault_incident(
            "spare-repair",
            "spare-health-event",
            cluster_id="hp-cluster",
            node_ids=[NODE],
            workflow_request_id="spare-workflow" if status is not None else None,
        )
        self.store.save_incident(incident)
        if status is not None:
            self.store.save_workflow(
                workflow_request(
                    "spare-workflow",
                    incident.incident_id,
                    status,
                    official_steps=[
                        workflow_step(WorkflowOperation.RESTART_NODE, node_ids=[NODE])
                    ],
                )
            )
        self.annotations[INCIDENT_ANNOTATION] = incident.incident_id

    def scan(self) -> dict[str, Any]:
        [result] = self.controller.scan()
        return result

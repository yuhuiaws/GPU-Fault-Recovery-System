"""The node-health incident labels are the ones the product's routes give (b213).

``incident_labels.NODE_HEALTH_INCIDENT_EVENT_TYPES`` is what every reader of
"the incident holding a node-health finding" takes (the shared-storage failure
domain, the EFA traffic admin action). The pin drives each route through the
product's own ingestion (``app/ingest/node_health.py`` under the rule table
``NodeHealthPolicy.METRIC_RULES``, over an in-memory store) and holds the labels
the routes give to the set exactly, and the grouped subset to every label but
the one a finding's own record is minted under.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from gpu_fault.app import ApplicationContext
from gpu_fault.app.ingest.node_health import NodeHealthIngestionService
from gpu_fault.host_health import NodeHealthFinding, NodeHealthPolicy
from gpu_fault.models import FaultIncident, WorkloadState
from gpu_fault.orchestration.collector_outbox_maintenance import OPERATOR_EVENT_TYPE
from gpu_fault.orchestration.families.incident_labels import (
    GROUPED_INCIDENT_EVENT_TYPES,
    NODE_HEALTH_EVENT_TYPE,
    NODE_HEALTH_INCIDENT_EVENT_TYPES,
)
from gpu_fault.orchestration.placement_hold import EVENT_TYPE as PLACEMENT_HOLD
from gpu_fault.policy.models import FaultEventType
from gpu_fault.shared_storage_failure_domain import (
    SHARED_STORAGE_FAILURE_DOMAIN_EVENT_TYPE,
)
from scripts.e2e.regional.general_live_verdicts import NODE_HEALTH_EVENT_TYPES
from tests._builders import attempt_observation, container_observation

WORKLOAD = "training/pytorchjob/job"


class _Product:
    """The product's node-health ingestion over an in-memory store."""

    def __init__(self) -> None:
        self.context = ApplicationContext()
        self.batches = 0

    def host(self, metric: str, node: str, **values: Any) -> FaultIncident:
        """One host finding under the product's rule and event-id shape: the
        incident ingestion routed it to."""

        self.batches += 1
        batch_id = f"b213-batch-{self.batches}"
        _threshold, category, severity, action, reason = NodeHealthPolicy.METRIC_RULES[
            metric
        ]
        event_id = f"{batch_id}-{metric}-node"
        finding = NodeHealthFinding(
            finding_id=f"finding-{event_id}",
            event_id=event_id,
            cluster_id="cluster-a",
            node_id=node,
            observed_at=datetime.now(timezone.utc),
            category=category,
            severity=severity,
            reason=reason,
            recommended_action=action,
            metric_name=metric,
            value=1.0,
            runtime_profile_version="simulated-v1",
            **{"workload_state": WorkloadState.IDLE, **values},
        )
        NodeHealthIngestionService(self.context).ingest(batch_id, [finding])
        incident = self.context.store.get_incident_by_event(event_id)
        assert incident is not None, f"{event_id} reached no incident"
        return incident

    def attempt(self, node: str) -> dict[str, Any]:
        """A running managed attempt on ``node``: what its findings name."""

        self.context.store.save_attempt_observation(
            attempt_observation(
                f"job-{node}",
                f"job-{node}-a001",
                datetime.now(timezone.utc),
                containers=[
                    container_observation(
                        f"pod-{node}", "worker-0", 0, node, gpu_uuids=[f"GPU-{node}"]
                    )
                ],
                workload_ids=[WORKLOAD],
            )
        )
        return {
            "workload_state": WorkloadState.ACTIVE,
            "affected_workload_ids": [WORKLOAD],
        }


def test_the_set_is_every_label_the_products_routes_give_a_node_health_record() -> None:
    product = _Product()
    labels: dict[str, str] = {}
    # families/health.py: a critical quarantine on an idle node.
    minted = product.host("smart_health_failed", "node-1")
    labels["minted"] = minted.event_type
    # families/drain.py ingest_terminal_node_quarantine: another one joins
    # the isolation in flight.
    joined = product.host("network_link_down", "node-1")
    assert joined.incident_id == minted.incident_id, joined
    labels["critical_quarantine_joins"] = joined.event_type
    # families/drain.py ingest_grouped_node_resource_finding: the allocatable
    # count of a node whose GPU went missing joins its reboot chain.
    missing = product.host("gpu_inventory_mismatch", "node-2")
    resource = product.host("gpu_kubernetes_allocatable_mismatch", "node-2")
    assert resource.incident_id == missing.incident_id, resource
    labels["device_resource_joins"] = resource.event_type
    # families/grouped_health.py: a critical quarantine of a managed attempt.
    labels["managed_attempt"] = product.host(
        "smart_health_failed", "node-3", **product.attempt("node-3")
    ).event_type
    # families/node_lifecycle.py: a reboot finding of a running attempt opens
    # a group record of its own.
    labels["attempt_reboot"] = product.host(
        "gpu_inventory_mismatch", "node-4", **product.attempt("node-4")
    ).event_type

    assert labels == {
        "minted": "NODE_HEALTH",
        "critical_quarantine_joins": "NODE_HEALTH_GROUP",
        "device_resource_joins": "NODE_RESOURCE_GROUP",
        "managed_attempt": "GPU_FAULT_GROUP",
        "attempt_reboot": "NODE_HEALTH_GROUP",
    }, "a route changed its label: update incident_labels and its readers"
    assert set(labels.values()) == NODE_HEALTH_INCIDENT_EVENT_TYPES
    assert GROUPED_INCIDENT_EVENT_TYPES == (
        NODE_HEALTH_INCIDENT_EVENT_TYPES - {NODE_HEALTH_EVENT_TYPE}
    ), "only a finding's own record is minted NODE_HEALTH"
    assert NODE_HEALTH_EVENT_TYPES == NODE_HEALTH_INCIDENT_EVENT_TYPES, (
        "the GF-GENERAL-LIVE matchers take the product's set"
    )


def test_the_products_other_incident_labels_stay_out_of_the_set() -> None:
    # The shared-storage failure domain is a cluster-scoped correlation record
    # with no workflow; a placement hold, an operator maintenance record and an
    # XID/SXID decision are not a record holding a node-health finding.
    others = {
        SHARED_STORAGE_FAILURE_DOMAIN_EVENT_TYPE,
        PLACEMENT_HOLD,
        OPERATOR_EVENT_TYPE,
        *(item.value for item in FaultEventType),
    }
    assert not others & NODE_HEALTH_INCIDENT_EVENT_TYPES, (
        others & NODE_HEALTH_INCIDENT_EVENT_TYPES
    )

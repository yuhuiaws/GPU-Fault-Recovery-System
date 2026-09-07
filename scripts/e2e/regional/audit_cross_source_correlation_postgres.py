from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import psycopg

from gpu_fault.host_health import (
    NodeHealthCategory,
    NodeHealthFinding,
)
from gpu_fault.models import (
    FaultIncident,
    IncidentState,
    RecoveryAction,
    Severity,
)
from gpu_fault.orchestration import IncidentOrchestrator
from gpu_fault.policy import (
    GpuFaultPolicyEngine,
    SxidClassification,
    SxidEvent,
    XidEvent,
)
from gpu_fault.store import PostgresStore


def _save_finding(
    store: PostgresStore,
    finding: NodeHealthFinding,
) -> tuple[str, str]:
    marker = finding.marker()
    store.add_marker(marker)
    incident = FaultIncident(
        incident_id=marker.incident_id,
        event_id=finding.event_id,
        event_type="NODE_HEALTH",
        cluster_id=finding.cluster_id,
        node_ids=[finding.node_id],
        gpu_uuids=finding.gpu_uuids,
        policy_version=finding.policy_version,
        policy_source=finding.policy_source,
        effective_action=finding.recommended_action,
        state=IncidentState.DETECTED,
        reasons=[finding.reason],
        created_at=finding.observed_at,
        updated_at=finding.observed_at,
    )
    store.save_incident(incident)
    return marker.marker_id, incident.incident_id


def main() -> None:
    dsn = os.environ["GPU_FAULT_STORE_URL"]
    suffix = uuid4().hex
    now = datetime.now(timezone.utc)
    gpu_node = f"audit-p17-gpu-{suffix}"
    efa_node = f"audit-p17-efa-{suffix}"
    other_node = f"audit-p17-other-{suffix}"
    store = PostgresStore(dsn)
    marker_ids: list[str] = []
    incident_ids: list[str] = []
    event_ids: list[str] = []
    try:
        memory = NodeHealthFinding(
            finding_id=f"audit-p17-memory-{suffix}",
            event_id=f"audit-p17-memory-{suffix}",
            cluster_id="audit-p17-cluster",
            node_id=gpu_node,
            observed_at=now,
            category=NodeHealthCategory.GPU,
            severity=Severity.CRITICAL,
            reason="DCGM field 319 increased by one",
            metric_name="dcgm_ecc_dbe_delta",
            gpu_uuids=["GPU-audit-a"],
            pci_bdf="0000:59:00.0",
            recommended_action=RecoveryAction.RESET_GPU,
        )
        marker_id, incident_id = _save_finding(store, memory)
        marker_ids.append(marker_id)
        incident_ids.append(incident_id)
        event_ids.append(memory.event_id)

        old = memory.model_copy(
            update={
                "finding_id": f"audit-p17-old-{suffix}",
                "event_id": f"audit-p17-old-{suffix}",
                "observed_at": now - timedelta(hours=1),
            }
        )
        marker_id, incident_id = _save_finding(store, old)
        marker_ids.append(marker_id)
        incident_ids.append(incident_id)
        event_ids.append(old.event_id)

        other = memory.model_copy(
            update={
                "finding_id": f"audit-p17-other-{suffix}",
                "event_id": f"audit-p17-other-{suffix}",
                "node_id": other_node,
            }
        )
        marker_id, incident_id = _save_finding(store, other)
        marker_ids.append(marker_id)
        incident_ids.append(incident_id)
        event_ids.append(other.event_id)

        policy = GpuFaultPolicyEngine()
        orchestrator = IncidentOrchestrator(store)
        xid = XidEvent(
            event_id=f"audit-p17-xid48-{suffix}",
            cluster_id="audit-p17-cluster",
            node_id=gpu_node,
            observed_at=now + timedelta(seconds=10),
            event_source="NVIDIA_KERNEL_LOG",
            xid=48,
            gpu_uuid="GPU-audit-a",
            pci_bdf="0000:59:00.0",
            product="H100",
            driver_branch=575,
            cuda_version="12.9",
        )
        correlated = orchestrator.correlate_provider_event(policy.evaluate_xid(xid))
        assert correlated.duplicate
        assert correlated.marker.incident_id == memory.marker().incident_id

        recent = store.list_recent_markers_for_nodes(
            {gpu_node},
            now - timedelta(minutes=5),
        )
        assert [item.marker_id for item in recent] == [memory.marker().marker_id]

        efa = NodeHealthFinding(
            finding_id=f"audit-p17-efa-fatal-{suffix}",
            event_id=f"audit-p17-efa-fatal-{suffix}",
            cluster_id="audit-p17-cluster",
            node_id=efa_node,
            observed_at=now,
            category=NodeHealthCategory.RDMA,
            severity=Severity.CRITICAL,
            reason="EFA RDMA link fatal timeout",
            metric_name="rdma_link_down",
            pci_bdf="0000:ab:00.0",
            recommended_action=RecoveryAction.QUARANTINE,
        )
        marker_id, incident_id = _save_finding(store, efa)
        marker_ids.append(marker_id)
        incident_ids.append(incident_id)
        event_ids.append(efa.event_id)
        sxid = SxidEvent(
            event_id=f"audit-p17-sxid-{suffix}",
            cluster_id="audit-p17-cluster",
            node_id=efa_node,
            observed_at=now + timedelta(seconds=10),
            event_source="FABRIC_MANAGER_LOG",
            sxid=23001,
            classification=SxidClassification.ALWAYS_FATAL,
            classification_source="NVIDIA_FABRIC_MANAGER",
            product="H200",
            pci_bdf="0000:ab:00.0",
            participating_gpu_uuids=["GPU-audit-a"],
        )
        not_correlated = orchestrator.correlate_provider_event(
            policy.evaluate_sxid(sxid)
        )
        assert not not_correlated.duplicate
        assert not_correlated.marker.incident_id != efa.marker().incident_id

        print(
            "PASS",
            {
                "positive_incident": correlated.marker.incident_id,
                "recent_marker_count": len(recent),
                "efa_fault_class": efa.marker().fault_class,
                "sxid_fault_class": (not_correlated.marker.fault_class),
            },
        )
    finally:
        store.close()
        with psycopg.connect(dsn) as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    DELETE FROM gpu_fault_links
                    WHERE key = ANY(%s)
                       OR value = ANY(%s)
                    """,
                    (event_ids + marker_ids + incident_ids, incident_ids),
                )
                cursor.execute(
                    """
                    DELETE FROM gpu_fault_objects
                    WHERE (kind='marker' AND key=ANY(%s))
                       OR (kind='incident' AND key=ANY(%s))
                    """,
                    (marker_ids, incident_ids),
                )


if __name__ == "__main__":
    main()

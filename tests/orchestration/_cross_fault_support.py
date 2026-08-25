"""Shared fixtures and builders for split test shards."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import httpx

from gpu_fault.app import ApplicationContext, create_app
from gpu_fault.models import WorkflowOperation, WorkflowRequest, WorkflowStatus
from gpu_fault.watcher import AttemptObservation
from tests._builders import (
    attempt_observation,
    container_observation,
    workflow_request,
    workflow_step,
)

NOW = datetime.now(timezone.utc)

WORKLOAD_ID = "training/pytorchjob/train"


def observation(
    *,
    job_id: str = "train",
    attempt_id: str = "train-a001",
    observed_at: datetime = NOW,
    started_at: datetime | None = None,
) -> AttemptObservation:
    return attempt_observation(
        job_id,
        attempt_id,
        observed_at,
        started_at=started_at,
        containers=[
            container_observation(
                f"pod-{attempt_id}",
                f"trainer-{attempt_id}",
                0,
                "node-a",
                gpu_uuids=["GPU-a"],
                cgroup_path=f"/kubepods/pod-{attempt_id}/trainer",
            )
        ],
        workload_ids=[WORKLOAD_ID],
        restart_budget=3,
    )


def xid_payload(xid: int, event_id: str) -> dict:
    return {
        "event_id": event_id,
        "cluster_id": "cluster-a",
        "node_id": "node-a",
        "observed_at": NOW.isoformat(),
        "xid": xid,
        "gpu_uuid": "GPU-a",
        "product": "H100",
        "driver_branch": 575,
        "cuda_version": "12.9",
        "runtime_profile_version": "simulated-v1",
    }


def sxid_payload(event_id: str) -> dict:
    return {
        "event_id": event_id,
        "cluster_id": "cluster-a",
        "node_id": "node-a",
        "observed_at": NOW.isoformat(),
        "sxid": 11001,
        "classification": "FATAL",
        "classification_source": ("NVIDIA_FABRIC_MANAGER_CATALOG"),
        "link_scope": "TRUNK",
        "link_scope_source": "TRUSTED_NVSWITCH_TOPOLOGY",
        "product": "H200",
        "fabric_partition": "cluster-a/node-a/local-nvswitch",
        "participating_gpu_uuids": ["GPU-a"],
        "runtime_profile_version": "simulated-v1",
    }


def dcgm_pcie_batch(batch_id: str, value: float, observed_at: datetime) -> dict:
    return {
        "batch_id": batch_id,
        "cluster_id": "cluster-a",
        "node_id": "node-a",
        "observed_at": observed_at.isoformat(),
        "collected_at": observed_at.isoformat(),
        "source": "DCGM_EXPORTER",
        "samples": [
            {
                "metric_name": ("DCGM_FI_DEV_PCIE_REPLAY_COUNTER"),
                "canonical_name": "pcie_replay_total",
                "value": value,
                "gpu_index": "0",
                "gpu_uuid": "GPU-a",
                "pci_bdf": "0000:b9:00.0",
            }
        ],
        "runtime_profile_version": "simulated-v1",
        "product": "H200",
    }


def kernel_xid11_collector_event() -> dict:
    observed_at = NOW + timedelta(seconds=15)
    return {
        "cluster_id": "cluster-a",
        "node_id": "node-a",
        "record_id": "kmsg-xid-11",
        "observed_at": observed_at.isoformat(),
        "collected_at": observed_at.isoformat(),
        "message": ("NVRM: Xid (PCI:0000:b9:00): 11, Ch 00000001"),
        "product": "H200",
        "runtime_profile_version": "simulated-v1",
    }


def fabric_sxid12020_collector_event() -> dict:
    observed_at = NOW + timedelta(seconds=16)
    return {
        "cluster_id": "cluster-a",
        "node_id": "node-a",
        "record_id": "fm-sxid-12020",
        "observed_at": observed_at.isoformat(),
        "collected_at": observed_at.isoformat(),
        "message": (
            "nvidia-nvswitch3: "
            "SXid (PCI:0000:c1:00.0): 12020, "
            "Fatal, Link 46 egress sequence ID error"
        ),
        "source": "journal",
        "unit": "nvidia-fabricmanager.service",
        "runtime_profile_version": "simulated-v1",
        "product": "H200",
    }


def arbitration_workflow(
    operation: WorkflowOperation,
    *,
    request_id: str,
    gpu_uuid: str = "GPU-a",
    status: WorkflowStatus = WorkflowStatus.PENDING,
) -> WorkflowRequest:
    return workflow_request(
        request_id,
        "incident-arbitration",
        status,
        1,
        official_steps=[workflow_step(operation, "owner", gpu_uuids=[gpu_uuid])],
        execution_owner_id="executor-a" if status is WorkflowStatus.RUNNING else None,
    )


async def post_faults(
    context: ApplicationContext, payloads: list[tuple[str, dict]]
) -> list[dict]:
    transport = httpx.ASGITransport(app=create_app(context))
    responses = []
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        for path, payload in payloads:
            response = await client.post(path, json=payload)
            assert response.status_code == 200, response.text
            responses.append(response.json())
    return responses

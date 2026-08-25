from __future__ import annotations


from pydantic import Field

from gpu_fault.models import StrictModel, WorkloadState


class CollectorContext(StrictModel):
    cluster_id: str
    runtime_profile_version: str | None = None
    product: str | None = None
    driver_branch: int | None = Field(default=None, ge=0)
    cuda_version: str | None = None
    workload_state: WorkloadState = WorkloadState.UNKNOWN
    affected_workload_ids: list[str] = Field(default_factory=list)
    checkpoint_manifest_ref: str | None = None


class CollectorStats(StrictModel):
    observed: int = 0
    delivered: int = 0
    skipped: int = 0
    duplicates: int = 0

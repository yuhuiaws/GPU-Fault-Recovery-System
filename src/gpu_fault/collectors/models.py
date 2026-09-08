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
    """What one collection round did with the records it saw.

    ``delivered`` counts only records the control plane *accepted*
    (``DeliveryStatus.DELIVERED``). A record the durable outbox took instead
    (``BUFFERED``) is on its way but not there yet, so it is counted in
    ``buffered`` -- never in ``delivered``. The two used to be conflated per
    collector: the kernel, Fabric Manager and Kubernetes collectors left a
    buffered record out of ``delivered``, the CloudWatch subscription and the
    SQS consumer counted it in, so the same word meant two things depending on
    which log line an operator was reading. A record that went nowhere is in
    neither; ``observed - skipped - duplicates - delivered - buffered`` is what
    a round lost or left for its caller to report.
    """

    #: Records read from the source this round, before any filter.
    observed: int = 0
    #: Records the control plane accepted (DELIVERED only).
    delivered: int = 0
    #: Records the durable outbox took for later replay (BUFFERED).
    buffered: int = 0
    #: Records not for this collector (no NVIDIA event, a control message, a
    #: poison queue message dropped on purpose).
    skipped: int = 0
    #: Records already seen and not re-sent.
    duplicates: int = 0

from __future__ import annotations

from datetime import datetime
from uuid import uuid4

from pydantic import Field

from gpu_fault.models import StrictModel


class TrainingProgressHeartbeat(StrictModel):
    heartbeat_id: str = Field(default_factory=lambda: f"training-progress-{uuid4()}")
    cluster_id: str
    attempt_id: str
    rank: int = Field(ge=0)
    observed_at: datetime
    node_id: str | None = None
    pod_uid: str | None = None
    container_name: str | None = None
    gpu_uuids: list[str] = Field(default_factory=list)
    step: int | None = Field(default=None, ge=0)
    samples_per_second: float | None = Field(default=None, ge=0)
    loss: float | None = None
    numerical_error: bool = False
    checkpoint_ref: str | None = None
    labels: dict[str, str] = Field(default_factory=dict)


class TrainingProgressState(StrictModel):
    heartbeat: TrainingProgressHeartbeat
    last_progress_at: datetime

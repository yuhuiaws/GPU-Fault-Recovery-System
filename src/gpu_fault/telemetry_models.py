from __future__ import annotations

from datetime import datetime

from pydantic import Field

from gpu_fault.models import StrictModel
from gpu_fault.watcher import AttemptObservation


class TelemetryMetricLatest(StrictModel):
    cluster_id: str
    node_id: str
    observed_at: datetime
    name: str
    value: float
    unit: str | None = None
    device: str | None = None
    labels: dict[str, str] = Field(default_factory=dict)


class WorkloadObservationState(StrictModel):
    first_observed_at: datetime
    observation: AttemptObservation

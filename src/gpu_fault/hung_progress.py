"""Host activity context only; never a gate for NCCL detection or recovery."""

from __future__ import annotations

import math
import os
from datetime import timedelta
from typing import Any
from gpu_fault.env import env_bool


def validate_legacy_progress_configuration() -> None:
    """Accept rollout-era settings without making them detection inputs."""
    env_bool("GPU_FAULT_EFA_TRAFFIC_PROGRESS_SUPPRESSION_ENABLED", True)
    maximum = float(
        os.getenv("GPU_FAULT_EFA_TRAFFIC_PROGRESS_SUPPRESSION_MAX_SECONDS", "1800")
    )
    if maximum <= 0:
        raise ValueError(
            "GPU_FAULT_EFA_TRAFFIC_PROGRESS_SUPPRESSION_MAX_SECONDS must be positive"
        )


def rank_liveness_evidence(batch: Any, enabled: bool) -> dict[str, Any]:
    samples = [
        sample
        for sample in batch.samples
        if sample.device is None and sample.name.startswith("training_rank_")
    ]
    values = {sample.name: sample.value for sample in samples}
    count = values.get("training_rank_process_count")
    seconds = values.get("training_rank_seconds_since_progress")
    advancing = values.get("training_rank_advancing_count")
    evidence = {
        "process_count": count,
        "advancing_count": advancing,
        "seconds_since_progress": seconds,
        "write_bytes_delta": values.get("training_rank_write_bytes_delta"),
        "write_chars_delta": values.get("training_rank_write_chars_delta"),
        "cpu_ticks_delta": values.get("training_rank_cpu_ticks_delta"),
    }
    if not enabled:
        return {**evidence, "status": "disabled"}
    if (
        len(samples) != len(values)
        or any(
            not math.isfinite(sample.value) or sample.value < 0 for sample in samples
        )
        or count is None
        or not math.isfinite(count)
        or count <= 0
        or seconds is None
        or not 0 <= seconds <= 86400
        or advancing is not None
        and (not math.isfinite(advancing) or not 0 <= advancing <= count)
    ):
        return {**evidence, "status": "unavailable"}
    stamp = batch.observed_at - timedelta(seconds=seconds)
    return {
        **evidence,
        "status": "advancing" if advancing else "stalled",
        "progress_observed_at": stamp.isoformat(),
    }

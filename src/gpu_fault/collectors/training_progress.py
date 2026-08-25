from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from gpu_fault.channel_registry import TRAINING_PROGRESS_PATH
from gpu_fault.training_health import TrainingProgressHeartbeat

from gpu_fault.collectors.sinks import CollectorError, EventSink

LOGGER = logging.getLogger(__name__)


class TrainingProgressCollector:
    """Reports rank liveness and application-written progress JSON."""

    def __init__(
        self,
        sink: EventSink,
        *,
        cluster_id: str,
        attempt_id: str,
        rank: int,
        progress_path: str,
        node_id: str | None = None,
        pod_uid: str | None = None,
        container_name: str | None = None,
        gpu_uuids: list[str] | None = None,
        interval_seconds: float = 15,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        self.sink = sink
        self.cluster_id = cluster_id
        self.attempt_id = attempt_id
        self.rank = rank
        self.progress_path = Path(progress_path)
        self.node_id = node_id
        self.pod_uid = pod_uid
        self.container_name = container_name
        self.gpu_uuids = gpu_uuids or []
        self.interval_seconds = interval_seconds
        self.now = now or (lambda: datetime.now(timezone.utc))

    def collect_once(self) -> TrainingProgressHeartbeat:
        value = {}
        if self.progress_path.exists():
            parsed = json.loads(self.progress_path.read_text(encoding="utf-8"))
            if not isinstance(parsed, dict):
                raise CollectorError(
                    "training progress file must contain a JSON object"
                )
            value = parsed
        heartbeat = TrainingProgressHeartbeat(
            cluster_id=self.cluster_id,
            attempt_id=self.attempt_id,
            rank=self.rank,
            observed_at=self.now(),
            node_id=self.node_id,
            pod_uid=self.pod_uid,
            container_name=self.container_name,
            gpu_uuids=self.gpu_uuids,
            step=value.get("step"),
            samples_per_second=value.get("samples_per_second"),
            loss=value.get("loss"),
            numerical_error=bool(value.get("numerical_error", False)),
            checkpoint_ref=value.get("checkpoint_ref"),
            labels={
                str(key): str(item) for key, item in (value.get("labels") or {}).items()
            },
        )
        self.sink.post(
            TRAINING_PROGRESS_PATH,
            heartbeat.model_dump(mode="json"),
        )
        return heartbeat

    def run(self) -> None:
        while True:
            try:
                self.collect_once()
            except Exception:
                LOGGER.exception("training progress collection failed")
            time.sleep(self.interval_seconds)

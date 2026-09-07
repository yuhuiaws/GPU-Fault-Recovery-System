from __future__ import annotations

import logging
import os
import secrets
import time
from threading import Thread
from urllib.parse import urlsplit

from gpu_fault.app.identity import pod_process_owner
from gpu_fault.env import env_bool
from gpu_fault.processor import (
    ProcessorCoordinator,
    ProcessorLeaseSettings,
    ProcessorPoolSettings,
    ProcessorSpoolSettings,
    ProcessorStaleSettings,
)


LOGGER = logging.getLogger(__name__)


class ProcessorFactory:
    def __init__(
        self,
        context,
        *,
        mode: str,
        exit_grace_seconds: float,
    ) -> None:
        self.context = context
        self.mode = mode
        self.exit_grace_seconds = exit_grace_seconds
        self.processor: ProcessorCoordinator | None = None

    def build(self) -> ProcessorCoordinator | None:
        if self.mode not in {
            "direct",
            "active-active",
        }:
            raise RuntimeError(
                "GPU_FAULT_PROCESSOR_MODE must be direct or "
                "active-active; active-standby has been removed"
            )
        if self.exit_grace_seconds < 0:
            raise ValueError("processor exit grace seconds cannot be negative")
        if self.mode == "direct":
            return None
        owner_id, replay_secret, local_url = self._identity()
        pools = ProcessorPoolSettings.from_environment()
        self.processor = ProcessorCoordinator(
            self.context.store,
            owner_id=owner_id,
            internal_token=replay_secret,
            execution_token=self.context.execution_token,
            local_url=local_url,
            active_consumers=self.mode == "active-active",
            lease=ProcessorLeaseSettings.from_environment(),
            pools=pools,
            stale=ProcessorStaleSettings.from_environment(),
            spool=ProcessorSpoolSettings.from_environment(
                default_pool=pools.default_pool
            ),
            on_unhealthy=self._recycle,
        )
        return self.processor

    def _identity(self) -> tuple[str, str, str]:
        owner_id = pod_process_owner()
        if not owner_id:
            raise RuntimeError("queued processor mode requires POD_UID")
        if not self.context.execution_token:
            raise RuntimeError("queued processor mode requires the execution token")
        replay_secret = self.context.processor_replay_secret or os.getenv(
            "GPU_FAULT_PROCESSOR_REPLAY_SECRET", ""
        )
        if len(replay_secret) < 32:
            raise RuntimeError(
                "queued processor mode requires an independent "
                "GPU_FAULT_PROCESSOR_REPLAY_SECRET of at least "
                "32 characters"
            )
        if secrets.compare_digest(
            replay_secret,
            self.context.execution_token,
        ):
            raise RuntimeError(
                "GPU_FAULT_PROCESSOR_REPLAY_SECRET must differ from "
                "GPU_FAULT_EXECUTION_TOKEN"
            )
        local_url = os.getenv(
            "GPU_FAULT_PROCESSOR_LOCAL_URL",
            "http://127.0.0.1:8080",
        )
        host = urlsplit(local_url).hostname or ""
        if host not in {"127.0.0.1", "::1", "localhost"}:
            raise RuntimeError(
                "GPU_FAULT_PROCESSOR_LOCAL_URL must use a loopback "
                "host so replay traffic cannot traverse the NLB"
            )
        return owner_id, replay_secret, local_url

    def _recycle(self, reason: str) -> None:
        if not env_bool("GPU_FAULT_PROCESSOR_EXIT_ON_DEADLINE"):
            return
        processor = self.processor
        if processor is None:
            return

        def exit_process() -> None:
            LOGGER.critical(
                "processor requires process restart reason=%s exit_grace_seconds=%s",
                reason,
                self.exit_grace_seconds,
            )
            if self.exit_grace_seconds > 0:
                time.sleep(self.exit_grace_seconds)
            abandon = processor.abandon_in_flight()
            LOGGER.critical(
                "terminating unhealthy processor process "
                "in_flight=%s released=%s failed=%s",
                abandon["in_flight"],
                abandon["released"],
                abandon["failed"],
            )
            os._exit(70)

        Thread(
            target=exit_process,
            name="gpu-fault-processor-fatal-exit",
            daemon=True,
        ).start()

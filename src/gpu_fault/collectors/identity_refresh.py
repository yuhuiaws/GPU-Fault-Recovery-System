"""Late GPU identity discovery for collectors that started before the driver.

``context_from_environment`` discovers the GPU product once, at process start.
A collector that comes up before the driver answers -- every boot, when the
unit is ordered after the network only -- keeps ``product=None`` for its whole
life and stamps that on every record it posts. The control plane's product
gate then reads an XID from that node as "not applicable to product UNKNOWN"
and quarantines the node instead of resetting the GPU (live, an ``eks-managed``
node after a reboot: one of the two injected XID 46 landed as a NOT_APPLICABLE
quarantine record while its sibling started the reset workflow).

:class:`GpuIdentityRefresher` retries the discovery for as long as the product
is unknown -- at most once per :data:`gpu_fault.collectors.context.GPU_IDENTITY_REFRESH_INTERVAL_SECONDS`,
and only when a caller is about to stamp the identity -- and hands back the
context to stamp. It never raises: late identity is a repair, not a
precondition, and a record must not be lost to it.
"""

from __future__ import annotations

import logging
import subprocess
from datetime import datetime, timedelta
from typing import Any, Callable

from gpu_fault.collectors.context import (
    GPU_IDENTITY_REFRESH_INTERVAL_SECONDS,
    gpu_product_discovery_mode,
)
from gpu_fault.collectors.gpu.discovery import (
    discover_gpu_product,
    discover_gpu_software_versions,
)
from gpu_fault.collectors.models import CollectorContext
from gpu_fault.collectors.sinks import CollectorError

LOGGER = logging.getLogger(__name__)

Runner = Callable[..., subprocess.CompletedProcess[str]]


class GpuIdentityRefresher:
    """Fills a product-less collector context once ``nvidia-smi`` answers."""

    def __init__(
        self,
        context: CollectorContext,
        *,
        runner: Runner | None = None,
        interval_seconds: float = GPU_IDENTITY_REFRESH_INTERVAL_SECONDS,
        discovery_mode: str | None = None,
    ) -> None:
        if interval_seconds <= 0:
            raise ValueError("GPU identity refresh interval must be positive")
        self.context = context
        self.runner = runner
        self.interval = timedelta(seconds=interval_seconds)
        self.discovery_mode = (
            discovery_mode
            if discovery_mode is not None
            else gpu_product_discovery_mode()
        )
        #: Late discovery attempts made so far (failed or not).
        self.attempts = 0
        #: When the product was found, or ``None`` while it is still unknown.
        self.discovered_at: datetime | None = None
        self._next_attempt_at: datetime | None = None

    @property
    def pending(self) -> bool:
        """True while the product is unknown and discovery is allowed."""

        return self.context.product is None and self.discovery_mode != "disabled"

    def refresh(self, now: datetime) -> CollectorContext:
        """Return the context to stamp, re-discovering the identity when due."""

        if not self.pending:
            return self.context
        if self._next_attempt_at is not None and now < self._next_attempt_at:
            return self.context
        self._next_attempt_at = now + self.interval
        self.attempts += 1
        try:
            product = discover_gpu_product(self.runner)
        except CollectorError as exc:
            LOGGER.warning(
                "nvidia-smi product discovery still unavailable (attempt %d); "
                "records keep leaving without a GPU product: %s",
                self.attempts,
                exc,
            )
            return self.context
        except Exception:
            LOGGER.exception(
                "late GPU product discovery failed unexpectedly (attempt %d); "
                "records keep leaving without a GPU product",
                self.attempts,
            )
            return self.context
        updates: dict[str, Any] = {"product": product}
        updates.update(self._software_versions())
        self.context = self.context.model_copy(update=updates)
        self.discovered_at = now
        LOGGER.warning(
            "GPU identity discovered after start (attempt %d): product=%s "
            "driver_branch=%s cuda_version=%s",
            self.attempts,
            self.context.product,
            self.context.driver_branch,
            self.context.cuda_version,
        )
        return self.context

    def _software_versions(self) -> dict[str, Any]:
        """Driver/CUDA versions for the fields the configuration left empty."""

        if self.context.driver_branch is not None and self.context.cuda_version:
            return {}
        try:
            driver_branch, cuda_version = discover_gpu_software_versions(self.runner)
        except CollectorError as exc:
            LOGGER.warning(
                "nvidia-smi software version discovery unavailable after the "
                "product was found; keeping the configured driver/CUDA versions: %s",
                exc,
            )
            return {}
        except Exception:
            LOGGER.exception(
                "late NVIDIA software version discovery failed unexpectedly; "
                "keeping the configured driver/CUDA versions"
            )
            return {}
        updates: dict[str, Any] = {}
        if self.context.driver_branch is None:
            updates["driver_branch"] = driver_branch
        if not self.context.cuda_version and cuda_version:
            updates["cuda_version"] = cuda_version
        return updates

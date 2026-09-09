from __future__ import annotations

import logging
import os
import subprocess
from typing import Callable

from gpu_fault.collectors.gpu.discovery import (
    GpuProductDiscoveryUnavailable,
    GpuSoftwareDiscoveryUnavailable,
    discover_gpu_product,
    discover_gpu_software_versions,
    normalize_gpu_product,
)
from gpu_fault.collectors.models import CollectorContext
from gpu_fault.collectors.process import BoundedProcessRunner
from gpu_fault.collectors.sinks import (
    CollectorError,
    HttpEventSink,
)
from gpu_fault.models import WorkloadState

LOGGER = logging.getLogger(__name__)


def context_from_environment(
    *,
    discover_product: bool = False,
    runner: Callable[..., subprocess.CompletedProcess[str]] | None = None,
) -> CollectorContext:
    """Build the collector context, discovering the GPU product if asked.

    Discovery shells out to ``nvidia-smi``, and it happens at startup --
    before the host collector's ``READY=1``. On a node whose driver is already
    wedged, ``subprocess.run``'s unbounded kill path meant the process never
    reached its first tick: systemd killed the activating unit on
    ``TimeoutStartSec`` and retried forever, and the node installer's
    ``systemctl restart`` failed, rolling back the whole node install over a
    GPU fault we could have reported. The default runner bounds its own kill
    path instead.
    """

    if runner is None:
        runner = BoundedProcessRunner()
    cluster_id = os.getenv("GPU_FAULT_CLUSTER_ID")
    if not cluster_id:
        raise CollectorError("GPU_FAULT_CLUSTER_ID is required")
    workloads = [
        item
        for item in os.getenv("GPU_FAULT_AFFECTED_WORKLOAD_IDS", "").split(",")
        if item
    ]
    driver = os.getenv("GPU_FAULT_DRIVER_BRANCH")
    driver_branch = int(driver) if driver else None
    cuda_version = os.getenv("GPU_FAULT_CUDA_VERSION") or None
    configured_product = os.getenv("GPU_FAULT_GPU_PRODUCT")
    product = normalize_gpu_product(configured_product) if configured_product else None
    discovery_mode = os.getenv("GPU_FAULT_GPU_PRODUCT_DISCOVERY", "auto").lower()
    if discovery_mode not in {"auto", "required", "disabled"}:
        raise CollectorError(
            "GPU_FAULT_GPU_PRODUCT_DISCOVERY must be auto, required, or disabled"
        )
    if discover_product and discovery_mode != "disabled":
        try:
            discovered_product = discover_gpu_product(runner)
        except GpuProductDiscoveryUnavailable:
            if discovery_mode == "required" or product is None:
                raise
            LOGGER.warning(
                "nvidia-smi product discovery unavailable; using "
                "configured GPU product %s",
                product,
            )
        else:
            if product is not None and product != discovered_product:
                raise CollectorError(
                    "configured GPU product does not match "
                    f"nvidia-smi: {product} != {discovered_product}"
                )
            product = discovered_product
        try:
            discovered_driver, discovered_cuda = discover_gpu_software_versions(runner)
        except GpuSoftwareDiscoveryUnavailable:
            if discovery_mode == "required" and driver_branch is None:
                raise
            LOGGER.warning(
                "nvidia-smi software version discovery unavailable; "
                "using configured driver/CUDA versions"
            )
        else:
            if driver_branch is not None and driver_branch != discovered_driver:
                raise CollectorError(
                    "configured NVIDIA driver branch does not match "
                    f"nvidia-smi: {driver_branch} != "
                    f"{discovered_driver}"
                )
            driver_branch = discovered_driver
            if cuda_version is None:
                cuda_version = discovered_cuda
    return CollectorContext(
        cluster_id=cluster_id,
        runtime_profile_version=os.getenv("GPU_FAULT_RUNTIME_PROFILE_VERSION"),
        product=product,
        driver_branch=driver_branch,
        cuda_version=cuda_version,
        workload_state=os.getenv("GPU_FAULT_WORKLOAD_STATE", WorkloadState.UNKNOWN),
        affected_workload_ids=workloads,
        checkpoint_manifest_ref=os.getenv("GPU_FAULT_CHECKPOINT_MANIFEST_REF"),
    )


def sink_from_environment() -> HttpEventSink:
    endpoint = os.getenv("GPU_FAULT_CONTROL_PLANE_URL")
    if not endpoint:
        raise CollectorError("GPU_FAULT_CONTROL_PLANE_URL is required")
    return HttpEventSink(
        endpoint,
        bearer_token=os.getenv("GPU_FAULT_CONTROL_PLANE_TOKEN"),
        timeout_seconds=float(
            os.getenv(
                "GPU_FAULT_COLLECTOR_HTTP_TIMEOUT_SECONDS",
                "10",
            )
        ),
        outbox_path=(os.getenv("GPU_FAULT_COLLECTOR_OUTBOX_PATH") or None),
        outbox_max_records=int(
            os.getenv("GPU_FAULT_COLLECTOR_OUTBOX_MAX_RECORDS", "1000")
        ),
    )

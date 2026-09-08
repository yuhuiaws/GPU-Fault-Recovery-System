from __future__ import annotations

import csv
import io
import logging
import math
import os
import re
import subprocess
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from pydantic import ValidationError

from gpu_fault.channel_registry import GPU_INVENTORY_PATH
from gpu_fault.gpu_metrics import (
    GpuInventoryDevice,
    GpuInventorySnapshot,
    GpuMetricSample,
    GpuMetricSource,
)


from gpu_fault.collectors.models import CollectorContext
from gpu_fault.collectors.sinks import (
    CollectorError,
    EventSink,
    deliver_event,
    deliver_or_raise,
)

LOGGER = logging.getLogger(__name__)

INSTANCE_ACCELERATOR_COUNTS = {
    "p5.4xlarge": {"gpu": 1, "efa": 1},
    "p5.48xlarge": {"gpu": 8, "efa": 32},
    "p5e.48xlarge": {"gpu": 8, "efa": 32},
    "p5en.48xlarge": {"gpu": 8, "efa": 16},
    "p6-b200.48xlarge": {"gpu": 8, "efa": 8},
    "p6-b300.48xlarge": {"gpu": 8, "efa": 16},
}


def expected_accelerator_counts(instance_type: str | None) -> dict[str, int] | None:
    """The GPU/EFA counts an instance type is built with, or ``None``.

    HyperPod names the type with an ``ml.`` prefix that the table does not
    carry; an unknown type answers ``None`` so a caller never invents a
    count for hardware it cannot vouch for.
    """

    if not instance_type:
        return None
    normalized = instance_type.strip().lower().removeprefix("ml.")
    return INSTANCE_ACCELERATOR_COUNTS.get(normalized)


def expected_gpu_count_from_environment() -> int | None:
    """``GPU_FAULT_EXPECTED_GPU_COUNT`` when set, else the instance type's count."""

    explicit = os.getenv("GPU_FAULT_EXPECTED_GPU_COUNT")
    if explicit:
        return int(explicit)
    counts = expected_accelerator_counts(os.getenv("GPU_FAULT_NODE_INSTANCE_TYPE"))
    return counts["gpu"] if counts is not None else None


NVIDIA_TEMPERATURE_LIMIT_TAGS = {
    "gpu_slowdown_temperature_c": ("gpu_temp_slow_threshold",),
    "gpu_shutdown_temperature_c": ("gpu_temp_max_threshold",),
    "gpu_max_operating_temperature_c": (
        "gpu_temp_max_gpu_threshold",
        "gpu_max_operating_temp",
    ),
    "memory_max_operating_temperature_c": (
        "gpu_temp_max_mem_threshold",
        "memory_max_operating_temp",
    ),
}

GPU_PRODUCT_PATTERN = re.compile(
    r"\b(?:(GB|GH|A|H|B)\s*[-_]?\s*(\d{2,4})|"
    r"(L40S|L4|T4|V100))\b",
    re.IGNORECASE,
)

CUDA_VERSION_PATTERN = re.compile(r"\bCUDA Version:\s*(\d+(?:\.\d+)*)", re.IGNORECASE)


def query_nvidia_temperature_limits(
    runner: Callable[..., subprocess.CompletedProcess[str]] = (subprocess.run),
) -> list[GpuMetricSample]:
    """Read firmware/driver temperature limits from nvidia-smi XML.

    Every failure mode answers ``CollectorError``, the way
    ``query_gpu_inventory`` does: a wedged driver hangs this call into
    ``TimeoutExpired`` and a missing binary raises ``FileNotFoundError``, and
    both used to escape the caller's ``except CollectorError`` guard *after*
    the DCGM scrape had already succeeded, discarding the whole tick.
    """

    try:
        result = runner(
            ["nvidia-smi", "-q", "-x"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise CollectorError(
            f"nvidia-smi temperature limit query unavailable: {exc}"
        ) from exc
    if result.returncode != 0:
        raise CollectorError(
            "nvidia-smi temperature limit query failed: " + result.stderr.strip()
        )
    try:
        root = ET.fromstring(result.stdout)
    except ET.ParseError as exc:
        raise CollectorError("nvidia-smi returned invalid XML") from exc

    samples = []
    for index, gpu in enumerate(root.findall(".//gpu")):
        gpu_index = (gpu.findtext("minor_number") or str(index)).strip()
        gpu_uuid = (gpu.findtext("uuid") or "").strip() or None
        product = (gpu.findtext("product_name") or "").strip()
        pci_bdf = (gpu.findtext("pci/pci_bus_id") or "").strip() or None
        temperature = gpu.find("temperature")
        if temperature is None:
            continue
        labels = {
            "gpu": gpu_index,
            "threshold_source": "nvidia-smi-xml",
        }
        if gpu_uuid:
            labels["UUID"] = gpu_uuid
        if product:
            labels["modelName"] = product
        if pci_bdf:
            labels["pci_bus_id"] = pci_bdf
        for canonical_name, tags in NVIDIA_TEMPERATURE_LIMIT_TAGS.items():
            raw = next(
                (
                    temperature.findtext(tag)
                    for tag in tags
                    if temperature.findtext(tag)
                ),
                None,
            )
            if raw is None:
                continue
            match = re.search(r"-?\d+(?:\.\d+)?", raw)
            if match is None:
                continue
            value = float(match.group())
            if not math.isfinite(value) or value <= 0:
                continue
            samples.append(
                GpuMetricSample(
                    metric_name=("nvidia_smi_" + canonical_name),
                    canonical_name=canonical_name,
                    value=value,
                    unit="celsius",
                    gpu_index=gpu_index,
                    gpu_uuid=gpu_uuid,
                    pci_bdf=pci_bdf,
                    labels=labels,
                )
            )
    return samples


def normalize_gpu_product(name: str) -> str:
    match = GPU_PRODUCT_PATTERN.search(name.strip().upper())
    if match is None:
        raise CollectorError(f"unsupported NVIDIA GPU product name: {name!r}")
    if match.group(3):
        return match.group(3).upper()
    return f"{match.group(1).upper()}{match.group(2)}"


def discover_gpu_product(
    runner: Callable[..., subprocess.CompletedProcess[str]] = (subprocess.run),
) -> str:
    command = [
        "nvidia-smi",
        "--query-gpu=index,uuid,name",
        "--format=csv,noheader",
    ]
    try:
        result = runner(
            command,
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise GpuProductDiscoveryUnavailable(
            f"nvidia-smi GPU identity query unavailable: {exc}"
        ) from exc
    if result.returncode != 0:
        detail = result.stderr.strip() or "no error detail"
        raise GpuProductDiscoveryUnavailable(
            f"nvidia-smi GPU identity query failed: {detail}"
        )

    identities: list[tuple[str, str, str]] = []
    for row in csv.reader(io.StringIO(result.stdout)):
        if not row or all(not item.strip() for item in row):
            continue
        if len(row) != 3:
            raise CollectorError("unexpected nvidia-smi GPU identity column count")
        index, gpu_uuid, raw_name = (item.strip() for item in row)
        if not index or not gpu_uuid or not raw_name:
            raise CollectorError("nvidia-smi returned incomplete GPU identity")
        identities.append((index, gpu_uuid, normalize_gpu_product(raw_name)))
    if not identities:
        raise GpuProductDiscoveryUnavailable("nvidia-smi returned no visible GPUs")
    uuids = [item[1] for item in identities]
    if len(uuids) != len(set(uuids)):
        raise CollectorError("nvidia-smi returned duplicate GPU UUIDs")
    products = {item[2] for item in identities}
    if len(products) != 1:
        details = ", ".join(
            f"index={index} uuid={gpu_uuid} product={product}"
            for index, gpu_uuid, product in identities
        )
        raise CollectorError(f"mixed GPU products detected on one node: {details}")
    product = products.pop()
    LOGGER.info(
        "discovered GPU product %s from %d nvidia-smi identities",
        product,
        len(identities),
    )
    return product


def discover_gpu_software_versions(
    runner: Callable[..., subprocess.CompletedProcess[str]] = (subprocess.run),
) -> tuple[int, str | None]:
    command = [
        "nvidia-smi",
        "--query-gpu=driver_version",
        "--format=csv,noheader",
    ]
    try:
        result = runner(
            command,
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise GpuSoftwareDiscoveryUnavailable(
            f"nvidia-smi driver version query unavailable: {exc}"
        ) from exc
    if result.returncode != 0:
        detail = result.stderr.strip() or "no error detail"
        raise GpuSoftwareDiscoveryUnavailable(
            f"nvidia-smi driver version query failed: {detail}"
        )
    versions = {line.strip() for line in result.stdout.splitlines() if line.strip()}
    if not versions:
        raise GpuSoftwareDiscoveryUnavailable("nvidia-smi returned no driver versions")
    branches = set()
    for version in versions:
        match = re.match(r"^(\d+)(?:\.|$)", version)
        if match is None:
            raise CollectorError(f"invalid NVIDIA driver version: {version!r}")
        branches.add(int(match.group(1)))
    if len(branches) != 1:
        raise CollectorError(
            "mixed NVIDIA driver branches detected: "
            + ", ".join(str(item) for item in sorted(branches))
        )

    cuda_version = None
    try:
        summary = runner(
            ["nvidia-smi"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        summary = None
    if summary is not None and summary.returncode == 0:
        match = CUDA_VERSION_PATTERN.search(summary.stdout)
        if match is not None:
            cuda_version = match.group(1)
    branch = branches.pop()
    LOGGER.info(
        "discovered NVIDIA driver branch %d and CUDA version %s",
        branch,
        cuda_version or "unknown",
    )
    return branch, cuda_version


def read_host_boot_id() -> str:
    path = Path(
        os.getenv(
            "GPU_FAULT_BOOT_ID_PATH",
            "/proc/sys/kernel/random/boot_id",
        )
    )
    try:
        value = path.read_text(encoding="ascii").strip()
    except OSError as exc:
        raise CollectorError(f"cannot read host boot ID from {path}: {exc}") from exc
    if not value:
        raise CollectorError("host boot ID is empty")
    return value


def query_gpu_inventory(
    runner: Callable[..., subprocess.CompletedProcess[str]] = (subprocess.run),
) -> list[GpuInventoryDevice]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,uuid,pci.bus_id,name",
        "--format=csv,noheader",
    ]
    try:
        result = runner(
            command,
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise CollectorError(
            f"nvidia-smi GPU inventory query unavailable: {exc}"
        ) from exc
    if result.returncode != 0:
        raise CollectorError(
            "nvidia-smi GPU inventory query failed: "
            + (result.stderr.strip() or "no error detail")
        )
    devices = []
    for row in csv.reader(io.StringIO(result.stdout)):
        if not row or all(not item.strip() for item in row):
            continue
        if len(row) != 4:
            raise CollectorError("unexpected nvidia-smi GPU inventory column count")
        index, gpu_uuid, pci_bdf, raw_name = (item.strip() for item in row)
        if not index.isdigit() or not gpu_uuid or not pci_bdf or not raw_name:
            raise CollectorError("nvidia-smi returned incomplete GPU inventory")
        devices.append(
            GpuInventoryDevice(
                gpu_index=int(index),
                gpu_uuid=gpu_uuid,
                pci_bdf=pci_bdf.lower(),
                product=normalize_gpu_product(raw_name),
            )
        )
    if not devices:
        raise CollectorError("nvidia-smi returned no GPU inventory")
    # Reuse the model's duplicate/invariant validation. Its ``ValidationError``
    # is a ``ValueError``, so it has to be converted here or a duplicate UUID
    # escapes every caller's ``except CollectorError``.
    try:
        GpuInventorySnapshot(
            cluster_id="validation",
            node_id="validation",
            observed_at=datetime.now(timezone.utc),
            source=GpuMetricSource.NVIDIA_SMI,
            source_boot_id="validation",
            devices=devices,
        )
    except ValidationError as exc:
        raise CollectorError(
            f"nvidia-smi GPU inventory failed validation: {exc}"
        ) from exc
    return sorted(devices, key=lambda item: item.gpu_index)


def deliver_gpu_inventory(
    sink: EventSink,
    context: CollectorContext,
    *,
    node_id: str,
    observed_at: datetime,
    runner: Callable[..., subprocess.CompletedProcess[str]],
) -> GpuInventorySnapshot:
    devices = query_gpu_inventory(runner)
    boot_id = read_host_boot_id()
    # ``GpuInventorySnapshot`` enforces its own invariants with pydantic, whose
    # ``ValidationError`` is a ``ValueError`` and not a ``CollectorError``: an
    # operator ``--expected-gpu-count`` below the real device count (or a
    # malformed ``GPU_FAULT_EXPECTED_GPU_COUNT``) used to escape every caller's
    # guard and take the whole collection tick down with it.
    try:
        expected = expected_gpu_count_from_environment()
        snapshot = GpuInventorySnapshot(
            snapshot_id=(
                f"gpu-inventory-{node_id}-{int(observed_at.timestamp() * 1_000_000)}"
            ),
            cluster_id=context.cluster_id,
            node_id=node_id,
            observed_at=observed_at,
            collected_at=observed_at,
            source=GpuMetricSource.NVIDIA_SMI,
            source_boot_id=boot_id,
            node_instance_id=(os.getenv("GPU_FAULT_NODE_INSTANCE_ID") or None),
            devices=devices,
            expected_gpu_count=expected,
            runtime_profile_version=context.runtime_profile_version,
            evidence_ref=f"nvidia-smi://{node_id}/inventory",
        )
    except ValidationError as exc:
        raise CollectorError(f"GPU inventory snapshot is not valid: {exc}") from exc
    except ValueError as exc:
        raise CollectorError(f"GPU inventory snapshot is not usable: {exc}") from exc
    result = deliver_or_raise(
        sink,
        GPU_INVENTORY_PATH,
        snapshot.model_dump(mode="json"),
        logger=LOGGER,
        what=f"GPU inventory {snapshot.snapshot_id}",
    )
    # A snapshot the outbox took is replayed from there, so the caller may
    # advance its inventory schedule; only a snapshot that went nowhere raises
    # (ARCH-G3).
    return snapshot


class GpuProductDiscoveryUnavailable(CollectorError):
    pass


class GpuSoftwareDiscoveryUnavailable(CollectorError):
    pass

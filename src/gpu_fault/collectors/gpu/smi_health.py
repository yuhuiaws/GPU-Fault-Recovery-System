"""Memory-repair flags and fabric health from ``nvidia-smi -q -x`` (batch 235).

Read once per inventory interval (60 s by default), like the InfoROM answer
(``collectors/gpu/inforom``), and published beside every scrape until the next
read. Tags are found by name under each ``<gpu>`` element, never by position
(live H200 R580, driver 580.178.04: the repair tags are XML lines 194-196 of
the per-GPU memory section, the fabric health sub-tags lines 298-312 under
``<fabric><health>``):

* ``<channel_repair_pending>``, ``<tpc_repair_pending>``,
  ``<unrepairable_memory>`` -> ``memory_channel_repair_pending``,
  ``memory_tpc_repair_pending``, ``memory_unrepairable``: ``Yes`` 1, ``No`` 0,
  anything else (``N/A``, a missing tag) nothing. The control plane grades them
  (``gpu_memory_repair_policy``; ``gpu_memory_policy`` for unrepairable).
* ``<fabric><health>`` ``summary``, ``bandwidth``,
  ``route_recovery_in_progress``, ``route_unhealthy``,
  ``access_timeout_recovery``, ``incorrect_configuration``,
  ``partition_assigned`` -> one ``fabric_health_xml`` sample per GPU: the
  summary's code as value (0 when the summary itself is ``N/A`` or a string
  nvidia-smi does not document) and every tag that is not ``N/A`` as a
  ``fabric_health_<tag>`` label; nothing at all while every tag is ``N/A``
  (live H200). Evidence for the DCGM 174/175 grade and VALIDATE_GPU's text
  (``gpu_fabric_health``), never a grade of its own.

DCGM field 1507 is a second source of the unrepairable flag (from NVML v590);
:func:`merge_unrepairable` keeps one sample per GPU, at the higher reading. A
failed or unparsable query keeps nothing: a wedged driver must not keep a stale
``No`` alive; the retry waits for the next interval, and from the third failure
in a row for a doubling wait (:class:`SmiHealthProbe`).
"""

from __future__ import annotations

import logging
import subprocess
import xml.etree.ElementTree as ET
from collections.abc import Sequence
from datetime import datetime, timedelta
from typing import Callable

from gpu_fault.gpu_fabric_health import (
    FABRIC_HEALTH_XML,
    XML_HEALTH_TAGS,
    XML_LABEL_PREFIX,
    XML_SUMMARY_CODES,
)
from gpu_fault.gpu_memory_repair_policy import MEMORY_UNREPAIRABLE, XML_REPAIR_TAGS
from gpu_fault.gpu_metrics import GpuMetricSample

Runner = Callable[..., subprocess.CompletedProcess[str]]
SMI_XML_QUERY = ["nvidia-smi", "-q", "-x"]
YES_NO = {"yes": 1.0, "no": 0.0}
NOT_AVAILABLE = "n/a"
#: Consecutive failures after which the probe backs off, and the longest wait.
FAILURE_BACKOFF_THRESHOLD = 3
RETRY_CAP_SECONDS = 3600


def _identity(gpu: ET.Element, position: int) -> tuple[str, str | None, str | None]:
    index = (gpu.findtext("minor_number") or str(position)).strip()
    uuid = (gpu.findtext("uuid") or "").strip() or None
    pci_bdf = (gpu.findtext("pci/pci_bus_id") or gpu.get("id") or "").strip() or None
    return index, uuid, pci_bdf


def _sample(
    canonical: str,
    value: float,
    identity: tuple[str, str | None, str | None],
    labels: dict[str, str],
) -> GpuMetricSample:
    index, uuid, pci_bdf = identity
    return GpuMetricSample(
        metric_name=f"nvidia_smi_{canonical}",
        canonical_name=canonical,
        value=value,
        unit=None,
        gpu_index=index,
        gpu_uuid=uuid,
        pci_bdf=pci_bdf,
        labels={
            "gpu": index,
            **({"UUID": uuid} if uuid else {}),
            **({"pci_bus_id": pci_bdf} if pci_bdf else {}),
            **labels,
        },
    )


def parse_smi_health(text: str) -> list[GpuMetricSample]:
    """The repair-flag and fabric-health samples of one ``nvidia-smi -q -x``."""

    root = ET.fromstring(text)
    samples = []
    for position, gpu in enumerate(root.findall(".//gpu")):
        identity = _identity(gpu, position)
        for tag, canonical in XML_REPAIR_TAGS.items():
            raw = (gpu.findtext(f".//{tag}") or "").strip()
            value = YES_NO.get(raw.lower())
            if value is not None:
                samples.append(_sample(canonical, value, identity, {tag: raw}))
        health = gpu.find("fabric/health")
        if health is None:
            continue
        readings = {
            tag: raw
            for tag in XML_HEALTH_TAGS
            if (raw := (health.findtext(tag) or "").strip())
            and raw.lower() != NOT_AVAILABLE
        }
        if not readings:
            continue
        code = XML_SUMMARY_CODES.get(readings.get("summary", "").lower(), 0.0)
        samples.append(
            _sample(
                FABRIC_HEALTH_XML,
                code,
                identity,
                {XML_LABEL_PREFIX + tag: raw for tag, raw in readings.items()},
            )
        )
    return samples


def _device(sample: GpuMetricSample) -> str:
    return sample.gpu_uuid or sample.pci_bdf or sample.gpu_index or "node"


def merge_unrepairable(
    samples: list[GpuMetricSample], extra: Sequence[GpuMetricSample]
) -> list[GpuMetricSample]:
    """``extra`` without the unrepairable flags a DCGM sample (field 1507) of
    ``samples`` already reports for the GPU; that sample keeps the higher of
    the two readings, so either source's 1 counts (``samples``, the
    collector's own list, is updated in place)."""

    dcgm = {
        _device(item): position
        for position, item in enumerate(samples)
        if item.canonical_name == MEMORY_UNREPAIRABLE
    }
    if not dcgm:
        return list(extra)
    kept = []
    for item in extra:
        position = dcgm.get(_device(item))
        if item.canonical_name != MEMORY_UNREPAIRABLE or position is None:
            kept.append(item)
            continue
        current = samples[position]
        if item.value > current.value:
            samples[position] = current.model_copy(
                update={
                    "value": item.value,
                    "labels": {**current.labels, **item.labels},
                }
            )
    return kept


class SmiHealthProbe:
    """The cached XML answer, refreshed once per ``interval_seconds``.

    A wedged driver hangs ``nvidia-smi -q -x`` for its full 15 s timeout on
    the collector's tick, so after :data:`FAILURE_BACKOFF_THRESHOLD`
    consecutive failures each further failure doubles the wait, up to
    :data:`RETRY_CAP_SECONDS` -- the temperature-limit probe's backoff
    (``inventory_cadence.TemperatureLimitProbe``); a success restores the
    interval.
    """

    def __init__(self, *, interval_seconds: int, logger: logging.Logger) -> None:
        if interval_seconds <= 0:
            raise ValueError("nvidia-smi health probe interval must be positive")
        self.interval = timedelta(seconds=interval_seconds)
        self.logger = logger
        self.samples: list[GpuMetricSample] = []
        self.next_at: datetime | None = None
        self.failures = 0
        self._attempt_at = datetime.min
        self._warned: str | None = None

    def refresh(self, observed_at: datetime, runner: Runner) -> list[GpuMetricSample]:
        """The cached samples, re-read when the interval has passed."""

        if self.next_at is not None and observed_at < self.next_at:
            return self.samples
        self._attempt_at = observed_at
        self.next_at = observed_at + self._wait()
        try:
            result = runner(
                SMI_XML_QUERY,
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return self._failed(f"nvidia-smi health query unavailable: {exc}")
        if result.returncode != 0:
            return self._failed(
                "nvidia-smi health query failed: "
                + (result.stderr.strip() or "no error detail")
            )
        try:
            self.samples = parse_smi_health(result.stdout)
        except ET.ParseError as exc:
            return self._failed(f"nvidia-smi health query returned invalid XML: {exc}")
        self.failures = 0
        self.next_at = observed_at + self.interval
        self._warned = None
        return self.samples

    def _wait(self) -> timedelta:
        """The interval, doubled per failure past the threshold, capped."""

        excess = self.failures - FAILURE_BACKOFF_THRESHOLD + 1
        if excess <= 0:
            return self.interval
        seconds = self.interval.total_seconds() * (1 << min(excess, 16))
        return timedelta(
            seconds=min(seconds, max(RETRY_CAP_SECONDS, self.interval.total_seconds()))
        )

    def _failed(self, detail: str) -> list[GpuMetricSample]:
        self.samples = []
        self.failures += 1
        self.next_at = self._attempt_at + self._wait()
        log = self.logger.debug if detail == self._warned else self.logger.warning
        self._warned = detail
        log("%s; no repair flags or fabric health until the next probe", detail)
        return self.samples

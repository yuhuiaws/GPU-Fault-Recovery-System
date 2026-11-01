"""The absolute shutdown temperature of a T.Limit GPU, from DCGM field 159.

Driver 580 prints only T.Limit thresholds in ``nvidia-smi -q -x`` on H200 (the
DTD makes each threshold one tag or the other, ``discovery.py``), so
``gpu_shutdown_temperature_c`` was never published there and the one-sample
shutdown protection (``gpu_temperature_policy``, ``gpu_environment_policy``)
never fired on any H200 of either cluster. Every other source still reports
the absolute value: on 32 live H200 (driver 580.178.04, NVML 13.580.178.04,
DCGM 4.7.0 host engine) ``nvmlDeviceGetTemperatureThreshold`` answered 95 for
NVML_TEMPERATURE_THRESHOLD_SHUTDOWN, DCGM field 159 95, and ``gpu_temp +
gpu_temp_tlimit`` was 87 on every GPU, so 87 - (-8) = 95.

**Source: DCGM field 159 through the product's exporter, not NVML.** No node
collector binds NVML (they read the exporter and nvidia-smi), and a ctypes
NVML binding in the metrics unit would be a new native dependency and a new
place for a call to hang in the driver (``collectors/process.py``). DCGM is the
product's GPU telemetry on both clusters, scraped every tick with the UUID and
bus labels every other DCGM sample carries; ``deploy/dataplane/dcgm-counters.csv``
exports the field under its pre-4.6 name, which the pinned exporter's go-dcgm
(v1.4611.2) resolves to 159 through its legacy map, as it does for field 153.
DCGM reads the field with the legacy NVML call itself (DCGM 4.7.0
``dcgmlib/src/DcgmCacheManager.cpp``: ``DCGM_FI_DEV_GPU_TEMP_SHUTDOWN_CELSIUS``
-> ``NvmlDeviceGetTemperatureThreshold(..., NVML_TEMPERATURE_THRESHOLD_SHUTDOWN,
...)``), so NVML's note binds this source too (:data:`NVML_THRESHOLD_NOTE`).
It is acceptable here because the preferred interface it names,
``NVML_FI_DEV_TEMPERATURE_SHUTDOWN_TLIMIT``, reports the T.Limit offset only
(the product already grades it, and NVIDIA documents no conversion); the driver
the fleet runs still answers with the absolute value; the value feeds one rule,
the shutdown emergency, never the T.Limit grading; it is published only after
the device's own T.Limit arithmetic confirms it; and a driver that stops
answering makes DCGM serve a blank sentinel, which the parser drops: nothing is
published and the GPU is graded on the T.Limit scale alone, as before.

**Validation, per tick and GPU, of the value the exporter served:**

* An absolute shutdown limit the XML reports always wins; field 159 is never
  published beside it (``xml-authoritative``).
* Finite and 0 < value <= 150 C (``plausible_absolute_limit``), else
  ``withheld:implausible``.
* Within :data:`CROSS_CHECK_TOLERANCE_C` of the T.Limit arithmetic: current
  temperature + current T.Limit margin = absolute max operating temperature,
  and max operating - shutdown T.Limit threshold = shutdown temperature
  (:data:`NVIDIA_SMI_CURRENT_TLIMIT_QUOTE`, :data:`NVIDIA_SMI_SHUTDOWN_TLIMIT_QUOTE`).
  The readings are the XML snapshot's own ``gpu_temp`` and ``gpu_temp_tlimit``
  (one nvidia-smi read, ``discovery.SNAPSHOT_*_LABEL``; 87 on all 32 live
  GPUs), not the tick's DCGM fields 150 and 153, which are sampled one by one
  (their sum was 88 on 2 of the 32): under a fast thermal ramp, or a replayed
  temperature, those would withhold the limit exactly when it matters, and the
  limit itself does not change between ticks.
* A missing input (no shutdown T.Limit threshold, a snapshot reading N/A) or a
  mismatch withholds (``withheld:tlimit-input-missing`` /
  ``withheld:tlimit-mismatch``): a wrong limit would DRAIN a healthy node on
  one sample, a withheld one leaves the T.Limit rules grading as before. Two
  disagreeing names of the field on one GPU are ``withheld:ambiguous``.

Withheld is visible: the raw sample (``dcgm_shutdown_temperature_c``) stays in
the batch with ``shutdown_limit_state`` naming why, and the collector warns
once per GPU and state. Published is ``gpu_shutdown_temperature_c`` with the
DCGM sample's identity and labels plus ``threshold_source``
``NVIDIA_DEVICE_LIMIT``, ``limit_source`` (the field name the exporter served),
``dcgm_field_id`` and ``tlimit_implied_shutdown_c``; only the emergency reads
it (``gpu_temperature_policy.shutdown_from_device_field``). nvidia-smi metrics
mode has no such source: a T.Limit GPU there keeps no absolute shutdown limit.

Evidence, each re-read from the live pages: :data:`NVIDIA_DCGM_SHUTDOWN_FIELD_QUOTE`
(DCGM field identifiers, docs "Version: 4.6 (Latest)",
``gpu_temperature_policy.NVIDIA_DCGM_SHUTDOWN_FIELD_REFERENCE``), the DCGM 4.6.0
rename (:data:`NVIDIA_DCGM_ALIAS_QUOTE`, :data:`NVIDIA_DCGM_RENAME_QUOTE`), and
the NVML API Reference, v615 docs (:data:`NVML_THRESHOLD_QUOTE`,
:data:`NVML_THRESHOLD_NOTE`, :data:`NVML_SHUTDOWN_ENUM_QUOTE`). The DCGM source
line and the go-dcgm legacy map are read from NVIDIA's own repositories (tags
v4.7.0 and v1.4611.2), not from documentation.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from collections.abc import Iterable

from gpu_fault.gpu_metrics import GpuMetricSample
from gpu_fault.gpu_temperature_policy import (
    DEVICE_SHUTDOWN_FIELD_ID,
    plausible_absolute_limit,
    plausible_tlimit,
)

from gpu_fault.collectors.gpu.discovery import (
    SNAPSHOT_GPU_TEMP_LABEL,
    SNAPSHOT_TLIMIT_LABEL,
)

#: DCGM field identifiers, section "Field Identifiers": the entries
#: ``DCGM_FI_DEV_GPU_TEMP_SHUTDOWN_CELSIUS 159`` and
#: ``DCGM_FI_DEV_GPU_TEMP_SLOWDOWN_CELSIUS 158``.
NVIDIA_DCGM_SHUTDOWN_FIELD_QUOTE = "Shutdown temperature for the device."
NVIDIA_DCGM_SLOWDOWN_FIELD_QUOTE = "Slowdown temperature for the device."
#: DCGM release notes, 4.6.0 > Improvements: the old names stay as aliases.
NVIDIA_DCGM_RELEASE_NOTES_REFERENCE = (
    "https://docs.nvidia.com/datacenter/dcgm/latest/release-notes/changelog.html"
)
NVIDIA_DCGM_ALIAS_QUOTE = (
    "Existing names remain available as deprecated aliases, so current "
    "integrations continue to work while new integrations can move to the "
    "clearer names."
)
NVIDIA_DCGM_RENAME_QUOTE = (
    "DCGM_FI_DEV_SHUTDOWN_TEMP to DCGM_FI_DEV_GPU_TEMP_SHUTDOWN_CELSIUS."
)
#: NVML API Reference, Device Queries, nvmlDeviceGetTemperatureThreshold.
NVML_THRESHOLD_REFERENCE = (
    "https://docs.nvidia.com/deploy/nvml-api/latest/api/group__nvmlDeviceQueries.html"
    "#_CPPv433nvmlDeviceGetTemperatureThreshold12nvmlDevice_t"
    "27nvmlTemperatureThresholds_tPj"
)
NVML_THRESHOLD_QUOTE = (
    "Retrieves the temperature threshold for the GPU with the specified "
    "threshold type in degrees C."
)
NVML_THRESHOLD_NOTE = (
    "This API is no longer the preferred interface for retrieving the following "
    "temperature thresholds on Ada and later architectures: "
    "NVML_TEMPERATURE_THRESHOLD_SHUTDOWN, NVML_TEMPERATURE_THRESHOLD_SLOWDOWN, "
    "NVML_TEMPERATURE_THRESHOLD_MEM_MAX and NVML_TEMPERATURE_THRESHOLD_GPU_MAX. "
    "Support for reading these temperature thresholds for Ada and later "
    "architectures would be removed from this API in future releases. Please use "
    "nvmlDeviceGetFieldValues with NVML_FI_DEV_TEMPERATURE_* fields to retrieve "
    "temperature thresholds on these architectures."
)
#: NVML API Reference, Device Enums, nvmlTemperatureThresholds_t.
NVML_SHUTDOWN_ENUM_REFERENCE = (
    "https://docs.nvidia.com/deploy/nvml-api/latest/api/group__nvmlDeviceEnums.html"
    "#_CPPv4N27nvmlTemperatureThresholds_t35NVML_TEMPERATURE_THRESHOLD_SHUTDOWNE"
)
NVML_SHUTDOWN_ENUM_QUOTE = (
    "Temperature at which the GPU will shut down for HW protection."
)
#: nvidia-smi documentation, GPU ATTRIBUTES > Temperature (v610 docs): the
#: definitions the cross-check's arithmetic rests on.
NVIDIA_SMI_CURRENT_TLIMIT_QUOTE = (
    "Current margin in degrees Celsius from the maximum GPU operating temperature."
)
NVIDIA_SMI_SHUTDOWN_TLIMIT_QUOTE = (
    "The T.Limit temperature below which a GPU may shutdown."
)

#: The raw field-159 sample: never graded, kept in the batch as evidence.
DCGM_SHUTDOWN_CANDIDATE = "dcgm_shutdown_temperature_c"
#: Input validation, not an NVIDIA value: the readings are integers, so one
#: degree absorbs rounding between the snapshot's reading and its margin.
CROSS_CHECK_TOLERANCE_C = 1.0
STATE_LABEL = "shutdown_limit_state"
IMPLIED_LABEL = "tlimit_implied_shutdown_c"
PUBLISHED = "published"
XML_AUTHORITATIVE = "xml-authoritative"
IMPLAUSIBLE = "withheld:implausible"
INPUT_MISSING = "withheld:tlimit-input-missing"
MISMATCH = "withheld:tlimit-mismatch"
AMBIGUOUS = "withheld:ambiguous"
#: GPUs whose last state is remembered to rate-limit the log, oldest first.
MAX_REMEMBERED_DEVICES = 256


def _device(sample: GpuMetricSample) -> str:
    return sample.gpu_uuid or sample.pci_bdf or sample.gpu_index or "node"


def _number(text: str | None) -> float | None:
    try:
        return float(text) if text else None
    except ValueError:
        return None


def tlimit_implied_shutdown(reference: GpuMetricSample | None) -> float | None:
    """The XML snapshot's absolute shutdown temperature, by T.Limit arithmetic.

    ``reference`` is the device's ``gpu_shutdown_tlimit_c`` sample: its value
    is the shutdown T.Limit threshold, its snapshot labels the reading and
    margin of the same nvidia-smi read. ``None`` when any input is missing or
    implausible. A cross-check only: the published limit is the device's own.
    """

    if reference is None:
        return None
    reading = plausible_absolute_limit(
        _number(reference.labels.get(SNAPSHOT_GPU_TEMP_LABEL))
    )
    margin = plausible_tlimit(_number(reference.labels.get(SNAPSHOT_TLIMIT_LABEL)))
    shutdown_tlimit = plausible_tlimit(reference.value)
    if reading is None or margin is None or shutdown_tlimit is None:
        return None
    maximum = plausible_absolute_limit(reading + margin)
    if maximum is None:
        return None
    return plausible_absolute_limit(maximum - shutdown_tlimit)


def shutdown_limit_state(
    candidate: GpuMetricSample, device_samples: Iterable[GpuMetricSample]
) -> tuple[str, float | None]:
    """Whether one GPU's field-159 value is published, and the arithmetic's value."""

    samples = list(device_samples)
    names = [item.canonical_name for item in samples]
    if names.count(DCGM_SHUTDOWN_CANDIDATE) > 1:
        return AMBIGUOUS, None
    if "gpu_shutdown_temperature_c" in names:
        return XML_AUTHORITATIVE, None
    if plausible_absolute_limit(candidate.value) is None:
        return IMPLAUSIBLE, None
    implied = tlimit_implied_shutdown(
        next(
            (
                item
                for item in samples
                if item.canonical_name == "gpu_shutdown_tlimit_c"
            ),
            None,
        )
    )
    if implied is None:
        return INPUT_MISSING, None
    if abs(candidate.value - implied) > CROSS_CHECK_TOLERANCE_C:
        return MISMATCH, implied
    return PUBLISHED, implied


class ShutdownLimitResolver:
    """Publish each GPU's field-159 limit or say why not, once per change."""

    def __init__(self, logger: logging.Logger) -> None:
        self.logger = logger
        self.states: OrderedDict[str, str] = OrderedDict()

    def resolve(self, samples: list[GpuMetricSample]) -> list[GpuMetricSample]:
        """``samples`` with every field-159 value labelled, published if valid."""

        by_device: dict[str, list[GpuMetricSample]] = {}
        for sample in samples:
            by_device.setdefault(_device(sample), []).append(sample)
        result: list[GpuMetricSample] = []
        for sample in samples:
            if sample.canonical_name != DCGM_SHUTDOWN_CANDIDATE:
                result.append(sample)
                continue
            device = _device(sample)
            state, implied = shutdown_limit_state(sample, by_device[device])
            extra = {STATE_LABEL: state}
            if implied is not None:
                extra[IMPLIED_LABEL] = f"{implied:g}"
            result.append(
                sample.model_copy(update={"labels": {**sample.labels, **extra}})
            )
            if state == PUBLISHED:
                result.append(self._published(sample, extra))
            self._report(device, state, sample.value, implied)
        return result

    @staticmethod
    def _published(sample: GpuMetricSample, extra: dict[str, str]) -> GpuMetricSample:
        labels = {
            **sample.labels,
            "threshold_source": "NVIDIA_DEVICE_LIMIT",
            "limit_source": sample.metric_name,
            "dcgm_field_id": str(DEVICE_SHUTDOWN_FIELD_ID),
            IMPLIED_LABEL: extra[IMPLIED_LABEL],
        }
        return sample.model_copy(
            update={
                "canonical_name": "gpu_shutdown_temperature_c",
                "unit": "celsius",
                "labels": labels,
            }
        )

    def _report(
        self, device: str, state: str, value: float, implied: float | None
    ) -> None:
        if self.states.get(device) == state:
            return
        self.states[device] = state
        self.states.move_to_end(device)
        while len(self.states) > MAX_REMEMBERED_DEVICES:
            self.states.popitem(last=False)
        if state == XML_AUTHORITATIVE:
            return
        log = self.logger.info if state == PUBLISHED else self.logger.warning
        log(
            "absolute shutdown temperature of %s from DCGM field %d: %s "
            "(reported %g C, T.Limit arithmetic %s)%s",
            device,
            DEVICE_SHUTDOWN_FIELD_ID,
            state,
            value,
            "n/a" if implied is None else f"{implied:g} C",
            ""
            if state == PUBLISHED
            else "; the one-sample shutdown protection stays off for this GPU, "
            "its T.Limit thresholds still grade it",
        )

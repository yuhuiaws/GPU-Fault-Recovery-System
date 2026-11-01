"""Opt-in loopback DCGM *field* replay for one isolated GPU acceptance node.

The GF-GENERAL-LIVE counterpart of ``tools/gpu_environment_replay.py`` for the
readings that tool deliberately refuses: ECC / row remap / page retirement,
the NVLink error counters, the NVLink fabric state (DCGM field 170), the PCIe
replay counter and the *structure* of the scrape (a missing field or GPU, a
conflicting alias, a NaN/negative or duplicate slot). The environment tool's
``ALLOWED_FIELDS`` is not widened: every category has its own reviewed
profile here -- the canonical fields it may touch, the operations it may
apply and the labels it may rewrite -- and one plan names exactly one profile.

Only the targeted raw reading changes. The tool sits between the product's
own DCGM exporter (``127.0.0.1:9400``) and the node's single metrics
collector; the collector's parser, the transport, the control plane's
grading, database, scheduling and node actions all stay real. It starts
pass-through and stays byte-identical until an explicit ``POST /arm``; after
the plan, the deadline, ``POST /disarm`` or an abort it passes fresh upstream
data through unchanged. It never invents a GPU or a field: every operation
needs the real sample it derives from in the upstream scrape, and a real
upstream change of a targeted reading during the run aborts the replay (a
real fault is never masked by a replayed value).

The one exception is declared, per field, in the plan and allow-listed here
(``ABSENT_WHEN_HEALTHY``): the DCGM last-XID gauge, which the exporter does
not publish until the GPU has had an XID -- its absence *is* the healthy
reading, so a plan that must replay it can never find it upstream on a healthy
node (live 2026-10-26, every XID-join variant of GF-GENERAL-LIVE-022 aborted
at its baseline read, "targeted reading absent upstream"). With ``fields:
{"xid_last_error": {"absent_when_healthy": true}}`` the baseline accepts the
missing reading, a ``set`` synthesises the line from the same GPU's own
published label set (plus the exporter's XID labels) under the exporter's
HELP/TYPE header, a real appearance of the gauge upstream still aborts, and
the line vanishes again with the step that set it, the disarm, the deadline or
the end of the plan -- where the real exporter keeps the last XID published for
as long as DCGM's cache holds the sample (its watch asks for the legacy
10-minute keep-age: dcgm-exporter 4.8.4 ``internal/pkg/devicewatcher/const.go``
lines 23-24, ``maxKeepAge = 600.0`` / ``maxKeepSamples = 0``; whether DCGM
drops the lone event sample at that age was not verified; a hostengine restart
clears it). Every other field keeps the refusal.

The gauge's values are allow-listed per profile. ``pcie_xid`` (022) may set
only 0 and XID 32; ``xid_twin`` (GF-GENERAL-LIVE-024 ``xid_twin_*``, batch 237)
replays the gauge as the field-230 *twin* of a reviewed kernel line that names
the same XID on the same GPU -- 0 (the product's "no XID" baseline), 92, 48 or
154, the XIDs those variants pair -- and nothing else: no counter, no label,
no other field. A twin value alone would be a second fault to a product
without batch 237, which is why its variants refuse to deploy before they have
asked the control plane which generation it runs.

Every journal record carries ``at``, this host's wall clock (``time.time()``)
when the read was served or the event taken: the runner places a kernel line
and the gauge's change on the collector's own scrape schedule from it.

No GPU, Kubernetes or collector configuration is changed by this tool. Do not
point a production collector at this test endpoint. A replay proves the
processing chain, never the physical fault: every receipt says
``proof_scope: live-replay`` and ``physical_fault_injected: false``.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import re
import time
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, build_opener

from prometheus_client.exposition import generate_latest
from prometheus_client.metrics_core import Metric
from prometheus_client.parser import text_string_to_metric_families

from gpu_fault.collectors.gpu.dcgm import DCGM_METRICS
from gpu_fault.dcgm_fields import (
    DCGM_CORE_CANONICAL_FIELDS,
    DCGM_FABRIC_CANONICAL_GROUPS,
)

MAX_BYTES = 2 * 1024 * 1024
ACKNOWLEDGEMENT = "ISOLATED_GPU_TEST_NODE"
PROOF_SCOPE = "live-replay"
MAX_STEPS = 100
MAX_READS_PER_STEP = 200
MAX_TOTAL_READS = 400
MAX_SECONDS = 3600
MAX_ADD = 1000
MAX_ABSOLUTE = 2**52
#: How many earlier outputs a ``stale`` step may serve again.
STALE_DEPTH = 8
GPU_UUID = re.compile(r"^GPU-[0-9a-fA-F-]{8,64}$")

MEMORY_FIELDS = frozenset(
    {
        "ecc_sbe_volatile_total",
        "ecc_dbe_volatile_total",
        "ecc_sbe_aggregate_total",
        "ecc_dbe_aggregate_total",
        "retired_pages_sbe_total",
        "retired_pages_dbe_total",
        "retired_pages_pending",
        "row_remap_failure",
        "row_remap_pending",
        "row_remap_correctable_total",
        "row_remap_uncorrectable_total",
    }
)
#: Every member of the product's NVLink error families: the legacy per-class
#: counters (``DCGM_FI_DEV_NVLINK_*_ERROR_COUNT_TOTAL``, NVLink3 and earlier)
#: and the DL aggregates (``DCGM_FI_DEV_NVLINK_ERROR_DL_*``, the only NVLink
#: error counters a Hopper/NVLink4 exporter publishes). Both reach the same
#: canonical names through ``canonical_name`` and are served and altered alike.
NVLINK_FIELDS = frozenset(
    name for group in DCGM_FABRIC_CANONICAL_GROUPS.values() for name in group
)
FABRIC_FIELDS = frozenset({"fabric_manager_status", "fabric_manager_error_code"})
PCIE_FIELDS = frozenset({"pcie_replay_total"})
#: The DCGM last-XID gauge. The product turns its every change into an XID
#: event it trusts like a kernel one, so only XID 32 (catalog RESTART_APP,
#: monitor-only on an idle node) may be replayed; 79 would plan a reboot. ``0``
#: is the product's own "no XID" reading (``gpu_metrics._xid_event`` mints an
#: event only for ``xid > 0`` and only against a stored baseline; the collector
#: breaches only ``> 0``): a plan serves it first so the later 32 is a *dated
#: change*, not a first-seen value the product baselines silently.
XID_FIELDS = frozenset({"xid_last_error"})
#: The gauge values of the ``xid_twin`` profile: 0 (the product's "no XID"
#: baseline) and the three XIDs GF-GENERAL-LIVE-024's ``xid_twin_*`` variants
#: pair with a reviewed kernel line of the same code on the same GPU -- 92
#: (EXCESSIVE_SBE_INTERRUPTS, catalog IGNORE: MONITOR_ONLY), 48
#: (ROBUST_CHANNEL_GPU_ECC_DBE, WORKFLOW_XID_48: solo RESET_GPU) and 154
#: (GPU_RECOVERY_ACTION_CHANGED, XID_154: the kernel line's recovery-action
#: label decides; field 230 carries none). 79 and 32 stay out of it: no twin
#: variant pairs them.
XID_TWIN_VALUES = frozenset({0, 48, 92, 154})


@dataclass(frozen=True)
class AbsentGauge:
    """How to synthesise one gauge the exporter omits while the GPU is healthy."""

    #: The exporter sample name and the HELP text this product's exporter
    #: configuration gives it (``deploy/dataplane/dcgm-counters.csv``).
    name: str
    documentation: str
    #: Labels the exporter adds to this sample only, by value.
    attributes: dict[int, dict[str, str]]

    def labels(self, value: float) -> dict[str, str]:
        return dict(self.attributes.get(int(value), {}))


#: Fields whose *absence* is the healthy reading: a plan may declare one of
#: these ``absent_when_healthy`` (``fields`` in the plan; nothing else, never a
#: blanket switch) and the baseline then accepts the missing sample instead of
#: refusing to invent it. The only member is the DCGM last-XID gauge, which
#: dcgm-exporter publishes only once DCGM holds an XID for the GPU:
#:
#: * dcgm-exporter 4.8.4 (the image this product deploys is
#:   ``nvcr.io/nvidia/k8s/dcgm-exporter:4.6.1-4.8.4-distroless``; tag 4.8.4 of
#:   https://github.com/NVIDIA/dcgm-exporter, 2026-09-18),
#:   ``internal/pkg/collector/gpu_collector.go``: ``toString`` (line 1022)
#:   ``if value.Status != dcgm.DCGM_ST_OK { return skipDCGMValue }`` and
#:   ``if isInt64Blank(v) { return skipDCGMValue }``; ``toMetric`` (line 902,
#:   loop at 931-940) ``// Filter out counters with no value and ignored fields
#:   for this entity`` -> ``if v == skipDCGMValue { ... continue }``;
#:   ``internal/pkg/collector/utils.go`` ``isInt64Blank`` ("These values
#:   indicate no valid data is available.", log text "Skipping blank DCGM field
#:   value"). The same ``toMetric`` (lines 950-958) adds ``err_code`` /
#:   ``err_msg`` labels to the XID sample from ``xid_errors.go`` (line 21
#:   ``0: "No Error"``, line 53 ``32: "Invalid or corrupted push buffer
#:   stream"``; for the ``xid_twin`` values line 69 ``48:  "Double Bit ECC
#:   Error"``, line 113 ``92:  "High single-bit ECC error rate"`` and line 175
#:   ``154: "GPU Recovery Action Changed"`` -- the file at tag 4.8.4,
#:   https://github.com/NVIDIA/dcgm-exporter/blob/4.8.4/internal/pkg/collector/xid_errors.go,
#:   194 lines, sha256 fa5b8d85325e3eab6318b4a5e5b1cdf6f54595c28c584a0fc7215be9f7ba9d4c,
#:   fetched 2026-11-01), rendered as labels by ``rendermetrics/render_metrics.go``
#:   line 486 (``builder.addMap(metric.Attributes)``).
#: * DCGM v4.6.1 (https://github.com/NVIDIA/DCGM, tag v4.6.1)
#:   ``dcgmlib/src/DcgmCacheManager.cpp``: the polled-field switch, lines
#:   13759-13761, ``case DCGM_FI_DEV_XID_ERROR: case DCGM_FI_DEV_NVLINK_ERROR:
#:   break; /* These are handled by the NVML event thread (m_eventThread) */``;
#:   ``RecordXidForGpu`` (line 7535) appends the value only when an XID event
#:   arrives (``AppendEntityInt64(threadCtx, eventData, 0, now, expireTime)``,
#:   from ``nvmlEventTypeXidCriticalError`` at lines 7883-7891 and the kmsg
#:   parser at line 7804). Before the first XID the cache holds no sample of
#:   field 230, so the exporter's read is not ``DCGM_ST_OK`` / is blank and the
#:   line is skipped. ``dcgmlib/dcgm_fields.h`` lines 803-806: ``/** XID
#:   errors. The value is the specific XID error */ #define
#:   DCGM_FI_DEV_XID_ERROR 230`` (line 3659: ``DCGM_FI_DEV_XID_ERRORS`` is the
#:   compatibility alias); ``dcgmlib/dcgm_structs.h`` lines 56-59: ``Base value
#:   for 64 bits integer blank. can be used as an unspecified blank`` ->
#:   ``#define DCGM_INT64_BLANK 0x7ffffffffffffff0``.
#: * DCGM API docs "Field Identifiers",
#:   https://docs.nvidia.com/datacenter/dcgm/latest/dcgm-api/dcgm-api-field-ids.html
#:   ("Last updated on 2026-06-17"): ``DCGM_FI_DEV_XID_ERROR 230 -- XID errors.
#:   The value is the specific XID error``. The vendor text does not say the
#:   field is blank until an XID; that follows from the cache-manager source
#:   above and from the live scrapes (2026-10-26 14:17-14:21Z, both HyperPod
#:   nodes and the plain-EKS fault node: 28 ``DCGM_FI_*`` names per GPU, no
#:   ``DCGM_FI_DEV_XID_ERRORS`` on any of the 16 GPUs, although
#:   ``deploy/dataplane/dcgm-counters.csv`` line 14 requests it).
#: * The product: ``adapters/gpu_validation.py`` ("DCGM omits XID_ERRORS on
#:   healthy GPUs on some driver/DCGM combinations"),
#:   ``docs/en/deployment-and-operations-manual.md`` A.2 ("an optional
#:   persistent-state supplement and is therefore not listed as a default hard
#:   requirement of the exporter") and ``docs/部署和运维手册.md``
#:   ("可选的持久状态补充，因此未列为 exporter 的默认硬性条件").
#:
#: Whether any exporter generation publishes ``0`` while healthy is not stated
#: by the vendor text read above; the replay's ``0`` is the product's own "no
#: XID" value (``SET_VALUES``), served to give the product a baseline.
ABSENT_WHEN_HEALTHY: dict[str, AbsentGauge] = {
    "xid_last_error": AbsentGauge(
        "DCGM_FI_DEV_XID_ERRORS",
        # deploy/dataplane/dcgm-counters.csv line 14 (NVIDIA's own default CSV
        # says "Value of the last XID error encountered."; the live exporter
        # prints this product's text).
        "Last XID error code",
        {
            0: {"err_code": "0", "err_msg": "No Error"},
            32: {
                "err_code": "32",
                "err_msg": "Invalid or corrupted push buffer stream",
            },
            48: {"err_code": "48", "err_msg": "Double Bit ECC Error"},
            92: {"err_code": "92", "err_msg": "High single-bit ECC error rate"},
            154: {"err_code": "154", "err_msg": "GPU Recovery Action Changed"},
        },
    ),
}
#: The sample of the same GPU whose label set a synthesised line clones first
#: (every GPU-scoped line of dcgm-exporter carries the same labels: gpu, UUID,
#: pci_bus_id, device, modelName, Hostname, DCGM_FI_DRIVER_VERSION ...); any
#: other GPU-scoped sample of that GPU serves when the counter is not there.
LABEL_TEMPLATE_FIELD = "pcie_replay_total"
#: The live thermal-safety readings. Structural operations never touch them:
#: blinding the product's temperature protection is not a completeness test.
SAFETY_FIELDS = frozenset(
    {
        "gpu_temperature_c",
        "memory_temperature_c",
        "gpu_tlimit_c",
    }
)
#: The required fields of the product's completeness check
#: (``gpu_fault.dcgm_fields``), minus the thermal-safety readings. The clock
#: event reasons stay: their two exporter names are the real alias pair.
COMPLETENESS_FIELDS = (
    frozenset(DCGM_CORE_CANONICAL_FIELDS) | NVLINK_FIELDS
) - SAFETY_FIELDS
#: Fault readings whose real value must not change while a plan targets them:
#: a real error during the run aborts the replay instead of being masked. The
#: clock event reasons and the power limit (alias / duplicate templates) may
#: move on their own and are not fault evidence.
MONITORED_FIELDS = (
    MEMORY_FIELDS | NVLINK_FIELDS | FABRIC_FIELDS | PCIE_FIELDS | XID_FIELDS
)
#: Gauges a ``set`` may write, with the values it may write.
SET_VALUES: dict[str, frozenset[int]] = {
    "retired_pages_pending": frozenset({0, 1}),
    "row_remap_failure": frozenset({0, 1}),
    "row_remap_pending": frozenset({0, 1}),
    # dcgmFabricManagerStatus_t (gpu_fault.gpu_fabric_manager).
    "fabric_manager_status": frozenset(range(0, 7)),
    "fabric_manager_error_code": frozenset(range(0, 256)),
    "xid_last_error": frozenset({0, 32}),
}
LABEL_VALUES = {
    "modelName": re.compile(r"^NVIDIA [A-Za-z0-9][A-Za-z0-9 .-]{0,47}$"),
    "boot_id": re.compile(r"^[0-9a-f][0-9a-f-]{7,63}$"),
}
OPERATION_KEYS: dict[str, frozenset[str]] = {
    "add": frozenset({"op", "field", "gpus", "value"}),
    "clear": frozenset({"op", "field", "gpus"}),
    "set": frozenset({"op", "field", "gpus", "value"}),
    "label": frozenset({"op", "field", "gpus", "label", "value"}),
    "omit": frozenset({"op", "field", "gpus"}),
    "omit_gpu": frozenset({"op", "gpus"}),
    "alias": frozenset({"op", "field", "gpus", "name", "value"}),
    "nan": frozenset({"op", "field", "gpus"}),
    "negative": frozenset({"op", "field", "gpus"}),
    "duplicate": frozenset({"op", "field", "gpus", "value"}),
    "stale": frozenset({"op", "age"}),
}


@dataclass(frozen=True)
class Profile:
    """What one category's plans may do: fields, operations, labels, and the
    values a ``set`` may write where they are narrower or other than the
    field's ``SET_VALUES``."""

    fields: frozenset[str]
    operations: frozenset[str]
    labels: frozenset[str] = frozenset()
    values: Mapping[str, frozenset[int]] = field(default_factory=dict)

    def set_values(self, name: str) -> frozenset[int] | None:
        """The values a ``set`` of ``name`` may write in this profile."""

        return self.values.get(name, SET_VALUES.get(name))


PROFILES: dict[str, Profile] = {
    # GF-GENERAL-LIVE-019 ECC / row remap / page retirement.
    "memory": Profile(MEMORY_FIELDS, frozenset({"add", "set"})),
    # GF-GENERAL-LIVE-020 NVLink error counters (+ stale scrape, counter reset).
    "nvlink": Profile(NVLINK_FIELDS, frozenset({"add", "clear", "stale"})),
    # GF-GENERAL-LIVE-021 NVLink fabric state; ``modelName`` only for the
    # ungraded-product variant.
    "fabric": Profile(
        FABRIC_FIELDS, frozenset({"set", "label"}), frozenset({"modelName"})
    ),
    # GF-GENERAL-LIVE-022 PCIe replay counter; ``boot_id`` only for the
    # exporter boot-generation variant.
    "pcie": Profile(
        PCIE_FIELDS, frozenset({"add", "clear", "label"}), frozenset({"boot_id"})
    ),
    # GF-GENERAL-LIVE-022 expired XID join: the PCIe counter and the gauge.
    "pcie_xid": Profile(PCIE_FIELDS | XID_FIELDS, frozenset({"add", "set"})),
    # GF-GENERAL-LIVE-024 xid_twin_*: the gauge alone, as the field-230 twin of
    # a reviewed kernel line of the same XID on the same GPU (batch 237).
    "xid_twin": Profile(
        XID_FIELDS,
        frozenset({"set"}),
        values={"xid_last_error": XID_TWIN_VALUES},
    ),
    # GF-GENERAL-LIVE-024 non-environment DCGM components of the composites.
    "composite": Profile(MEMORY_FIELDS | NVLINK_FIELDS, frozenset({"add", "set"})),
    # GF-GENERAL-LIVE-025 field / alias / NaN / duplicate structure.
    "completeness": Profile(
        COMPLETENESS_FIELDS,
        frozenset({"omit", "omit_gpu", "alias", "nan", "negative", "duplicate"}),
    ),
}


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError("upstream redirects are forbidden")


def open_upstream(url: str, *, timeout: int):
    return build_opener(ProxyHandler({}), NoRedirect()).open(url, timeout=timeout)


def canonical_name(sample_name: str) -> str | None:
    """The product's canonical field of an exporter sample name.

    Mirrors ``parse_dcgm_samples``: an exporter counter is parsed as
    ``<name>_total``, so the ``_total`` suffix maps to the bare name.
    """

    mapping = DCGM_METRICS.get(sample_name)
    if mapping is None and sample_name.endswith("_total"):
        mapping = DCGM_METRICS.get(sample_name[:-6])
    return mapping[0] if mapping is not None else None


def _finite(value: Any, label: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{label} must be a finite number")
    if abs(value) >= MAX_ABSOLUTE:
        raise ValueError(f"{label} is out of range")
    return float(value)


def _gpus(value: Any) -> tuple[str, ...]:
    if (
        not isinstance(value, list)
        or not 1 <= len(value) <= 16
        or len(set(value)) != len(value)
        or not all(isinstance(item, str) and GPU_UUID.fullmatch(item) for item in value)
    ):
        raise ValueError("gpus must be 1..16 distinct physical GPU UUIDs")
    return tuple(value)


def validate_operation(operation: Any, profile: Profile) -> dict[str, Any]:
    if not isinstance(operation, dict):
        raise ValueError("each operation must be an object")
    kind = operation.get("op")
    if kind not in OPERATION_KEYS or kind not in profile.operations:
        raise ValueError(f"operation {kind!r} is not allowed by this profile")
    if set(operation) != OPERATION_KEYS[kind]:
        raise ValueError(
            f"operation {kind} needs exactly {sorted(OPERATION_KEYS[kind])}"
        )
    if kind == "stale":
        age = operation["age"]
        if type(age) is not int or not 1 <= age <= STALE_DEPTH:
            raise ValueError(f"stale age must be an integer in 1..{STALE_DEPTH}")
        return dict(operation)
    _gpus(operation["gpus"])
    if kind == "omit_gpu":
        return dict(operation)
    name = operation["field"]
    if name not in profile.fields:
        raise ValueError(f"field {name!r} is not in this profile's allow-list")
    if kind == "add":
        value = operation["value"]
        if type(value) is not int or not 1 <= value <= MAX_ADD:
            raise ValueError(f"add needs an integer increment in 1..{MAX_ADD}")
    elif kind == "set":
        allowed = profile.set_values(name)
        if allowed is None:
            raise ValueError(f"{name} is a counter; use add, not set")
        if type(operation["value"]) is not int or operation["value"] not in allowed:
            raise ValueError(f"{name} may only be set to {sorted(allowed)}")
    elif kind == "label":
        label = operation["label"]
        if label not in profile.labels:
            raise ValueError(f"label {label!r} is not in this profile's allow-list")
        if (
            not isinstance(operation["value"], str)
            or LABEL_VALUES[label].fullmatch(operation["value"]) is None
        ):
            raise ValueError(f"label {label} value is malformed")
    elif kind == "alias":
        if canonical_name(str(operation["name"])) != name:
            raise ValueError(
                f"{operation['name']!r} is not an exporter alias of {name}"
            )
        _finite(operation["value"], "alias value")
        if operation["value"] < 0:
            raise ValueError("alias value must be non-negative")
    elif kind == "duplicate":
        _finite(operation["value"], "duplicate value")
        if operation["value"] < 0:
            raise ValueError("duplicate value must be non-negative")
    return dict(operation)


def validate_fields(value: Any, profile: Profile) -> dict[str, dict[str, bool]]:
    """The plan's per-field declarations: ``absent_when_healthy`` and only on
    the allow-list, inside the profile. Everything else keeps the refusal."""

    if not isinstance(value, dict):
        raise ValueError("fields must map field names to their declarations")
    for name, declaration in value.items():
        if name not in ABSENT_WHEN_HEALTHY:
            raise ValueError(
                f"{name} may not be declared absent_when_healthy; only "
                f"{sorted(ABSENT_WHEN_HEALTHY)} is absent while healthy"
            )
        if name not in profile.fields:
            raise ValueError(f"field {name!r} is not in this profile's allow-list")
        if (
            not isinstance(declaration, dict)
            or set(declaration) != {"absent_when_healthy"}
            or declaration["absent_when_healthy"] is not True
        ):
            raise ValueError(
                f'{name} may only be declared {{"absent_when_healthy": true}}'
            )
    return {name: dict(declaration) for name, declaration in value.items()}


def validate_plan(value: Any) -> dict[str, Any]:
    """The reviewed plan, or ValueError naming what a reviewer would refuse."""

    if (
        not isinstance(value, dict)
        or not {"profile", "steps"} <= set(value)
        or not set(value) <= {"profile", "steps", "fields"}
    ):
        raise ValueError(
            "plan must contain exactly profile and steps (and may declare fields)"
        )
    profile = PROFILES.get(value["profile"])
    if profile is None:
        raise ValueError(f"unknown profile {value['profile']!r}")
    fields = validate_fields(value.get("fields", {}), profile)
    steps = value["steps"]
    if not isinstance(steps, list) or not 1 <= len(steps) <= MAX_STEPS:
        raise ValueError(f"plan requires 1..{MAX_STEPS} steps")
    total = 0
    validated = []
    for step in steps:
        if (
            not isinstance(step, dict)
            or not {"reads", "ops"} <= set(step)
            or not (set(step) <= {"reads", "ops", "until_advance"})
        ):
            raise ValueError("each step requires reads and ops (and may hold)")
        reads, operations = step["reads"], step["ops"]
        hold = step.get("until_advance", False)
        if type(hold) is not bool:
            raise ValueError("until_advance must be a boolean")
        if type(reads) is not int or not 1 <= reads <= MAX_READS_PER_STEP:
            raise ValueError(f"reads must be an integer in 1..{MAX_READS_PER_STEP}")
        if not isinstance(operations, list) or len(operations) > 32:
            raise ValueError("ops must be a list of at most 32 operations")
        checked = [validate_operation(item, profile) for item in operations]
        if any(item["op"] == "stale" for item in checked) and len(checked) != 1:
            raise ValueError("a stale step serves an earlier scrape and nothing else")
        total += reads
        validated.append({"reads": reads, "ops": checked, "until_advance": hold})
    if total > MAX_TOTAL_READS:
        raise ValueError(f"plan exceeds {MAX_TOTAL_READS} collector reads")
    return {"profile": value["profile"], "steps": validated, "fields": fields}


def absent_when_healthy(plan: dict[str, Any]) -> set[str]:
    """The fields a validated plan declares absent while the GPU is healthy."""

    return {
        name
        for name, declaration in (plan.get("fields") or {}).items()
        if declaration.get("absent_when_healthy") is True
    }


def _reading(value: float | None) -> str:
    return "absent" if value is None else str(value)


def validate_upstream(url: str, listen_port: int) -> str:
    parsed = urlsplit(url)
    if (
        parsed.scheme != "http"
        or parsed.hostname != "127.0.0.1"
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path != "/metrics"
        or parsed.query
        or parsed.fragment
        or parsed.port is None
        or parsed.port == listen_port
    ):
        raise ValueError("upstream must be http://127.0.0.1:<different-port>/metrics")
    return url


def targets(plan: dict[str, Any]) -> set[tuple[str, str]]:
    """Every ``(gpu, field)`` a value operation of the plan reads or writes."""

    result: set[tuple[str, str]] = set()
    for step in plan["steps"]:
        for operation in step["ops"]:
            if "field" in operation:
                for gpu in operation["gpus"]:
                    result.add((gpu, operation["field"]))
    return result


def target_gpus(plan: dict[str, Any]) -> set[str]:
    return {
        gpu
        for step in plan["steps"]
        for operation in step["ops"]
        for gpu in operation.get("gpus", ())
    }


def sample_gpu(sample: Any) -> str | None:
    return sample.labels.get("UUID") or sample.labels.get("uuid")


def real_readings(text: str) -> tuple[list[Any], dict[tuple[str, str], float]]:
    """The parsed families and ``{(gpu, field): value}`` of the upstream scrape."""

    families = list(text_string_to_metric_families(text))
    readings: dict[tuple[str, str], float] = {}
    for family in families:
        for sample in family.samples:
            name = canonical_name(sample.name)
            gpu = sample_gpu(sample)
            if name is not None and gpu:
                readings.setdefault((gpu, name), float(sample.value))
    return families, readings


@dataclass
class ReadPlan:
    """What one armed read applies (the active step's operations)."""

    step: int
    operations: list[dict[str, Any]]
    first_read_of_step: bool


@dataclass
class Replay:
    plan: dict[str, Any]
    seconds: int
    reads: int = 0
    start_read: int | None = None
    deadline: float = 0.0
    disarmed: bool = False
    aborted: str | None = None
    #: The real reading of every monitored target at the baseline; ``None``
    #: for a declared absent-when-healthy gauge the upstream did not publish.
    baseline: dict[tuple[str, str], float | None] = field(default_factory=dict)
    baseline_taken: bool = False
    offsets: dict[tuple[str, str], int] = field(default_factory=dict)
    outputs: deque = field(default_factory=lambda: deque(maxlen=STALE_DEPTH))
    #: Index of the step a held (``until_advance``) step was left for, and the
    #: read it was left at: ``advance`` ends a held step early.
    advanced_to: int | None = None
    advanced_at_read: int = 0

    @property
    def armed(self) -> bool:
        return self.start_read is not None

    def arm(self) -> None:
        if self.armed or self.aborted or self.disarmed:
            raise ValueError("replay already armed or ended; start a new audited run")
        self.start_read = self.reads
        self.deadline = time.monotonic() + self.seconds

    def disarm(self) -> None:
        """End the replay for good: fresh upstream data passes through."""

        self.disarmed = True

    def advance(self) -> int:
        """Leave the current held step for the next one; return its index."""

        current = self.active()
        if current is None:
            raise ValueError("the replay is not on a step")
        if not self.plan["steps"][current.step].get("until_advance"):
            raise ValueError("the current step is not held for an advance")
        self.advanced_to = current.step + 1
        self.advanced_at_read = self.reads
        return self.advanced_to

    def active(self) -> ReadPlan | None:
        if (
            self.start_read is None
            or self.disarmed
            or self.aborted
            or time.monotonic() >= self.deadline
        ):
            return None
        steps = self.plan["steps"]
        first, remaining = 0, self.reads - self.start_read
        if self.advanced_to is not None:
            first, remaining = self.advanced_to, self.reads - self.advanced_at_read
        for index in range(first, len(steps)):
            step = steps[index]
            if remaining < step["reads"]:
                return ReadPlan(index, step["ops"], remaining == 0)
            if step.get("until_advance"):
                # A held step outlasting its read bound passes through: the
                # bound is a safety limit, never a reason to replay on.
                return None
            remaining -= step["reads"]
        return None

    def _check_baseline(self, readings: dict[tuple[str, str], float]) -> None:
        wanted = targets(self.plan)
        if not self.baseline_taken:
            # A declared absent-when-healthy gauge may be missing; its GPU may
            # not (the synthesised line clones that GPU's real label set).
            tolerated = absent_when_healthy(self.plan)
            missing = sorted(
                f"{gpu}/{name}"
                for gpu, name in wanted - set(readings)
                if name not in tolerated
            )
            absent_gpus = sorted(
                gpu
                for gpu in target_gpus(self.plan)
                if not any(key[0] == gpu for key in readings)
            )
            if missing or absent_gpus:
                self.aborted = (
                    "targeted reading absent upstream; refusing invented samples: "
                    + ",".join(missing + absent_gpus)
                )
                return
            self.baseline = {
                key: readings.get(key) for key in wanted if key[1] in MONITORED_FIELDS
            }
            self.baseline_taken = True
            return
        for key, value in self.baseline.items():
            current = readings.get(key)
            # An absent gauge that appears is a real XID: never masked either.
            if current != value:
                self.aborted = (
                    f"real upstream change of {key[0]}/{key[1]}: "
                    f"{_reading(value)} -> {_reading(current)}; "
                    "a real reading is never masked"
                )
                return

    def render(self, body: bytes) -> tuple[bytes, dict[str, Any]]:
        if len(body) > MAX_BYTES:
            raise ValueError("upstream response exceeds byte bound")
        text = body.decode("utf-8")
        current = self.active()
        receipt: dict[str, Any] = {
            "read": self.reads + 1,
            "at": time.time(),
            "profile": self.plan["profile"],
            "proof_scope": PROOF_SCOPE,
            "physical_fault_injected": False,
            "armed": self.armed,
            "disarmed": self.disarmed,
            "upstream_sha256": hashlib.sha256(body).hexdigest(),
        }
        output = body
        if current is not None:
            families, readings = real_readings(text)
            self._check_baseline(readings)
            receipt["real"] = {
                f"{gpu}/{name}": readings.get((gpu, name))
                for gpu, name in sorted(targets(self.plan))
            }
            if not self.aborted:
                output, served = self._apply(families, current, body)
                receipt.update(step=current.step, ops=current.operations, served=served)
        receipt["aborted"] = self.aborted
        receipt["output_sha256"] = hashlib.sha256(output).hexdigest()
        if current is not None and not self.aborted:
            self.outputs.append(output)
        return output, receipt

    def _apply(
        self, families: list[Any], current: ReadPlan, body: bytes
    ) -> tuple[bytes, dict[str, Any]]:
        operations = current.operations
        stale = [item for item in operations if item["op"] == "stale"]
        if stale:
            age = stale[0]["age"]
            if len(self.outputs) < age:
                self.aborted = f"no scrape {age} reads old to serve as stale"
                return body, {}
            return self.outputs[-age], {"stale_age_reads": age}
        for item in operations:
            if item["op"] == "add":
                for gpu in item["gpus"]:
                    key = (gpu, item["field"])
                    self.offsets[key] = self.offsets.get(key, 0) + item["value"]
            elif item["op"] == "clear" and current.first_read_of_step:
                for gpu in item["gpus"]:
                    self.offsets.pop((gpu, item["field"]), None)
        values: dict[tuple[str, str], float] = {}
        puts: set[tuple[str, str]] = set()
        omitted: set[tuple[str, str]] = set()
        omitted_gpus: set[str] = set()
        labels: dict[tuple[str, str], dict[str, str]] = {}
        extras: list[tuple[str, dict[str, Any]]] = []
        for item in operations:
            kind = item["op"]
            if kind == "omit_gpu":
                omitted_gpus.update(item["gpus"])
                continue
            for gpu in item["gpus"]:
                key = (gpu, item["field"])
                if kind == "set":
                    values[key] = float(item["value"])
                    puts.add(key)
                elif kind == "nan":
                    values[key] = math.nan
                elif kind == "negative":
                    values[key] = -1.0
                elif kind == "omit":
                    omitted.add(key)
                elif kind == "label":
                    labels.setdefault(key, {})[item["label"]] = item["value"]
                elif kind in {"alias", "duplicate"}:
                    extras.append((gpu, item))
        served: dict[str, Any] = {}
        templates: dict[tuple[str, str], tuple[Any, Any]] = {}
        for family in families:
            replaced = []
            for sample in family.samples:
                gpu = sample_gpu(sample)
                name = canonical_name(sample.name)
                if gpu in omitted_gpus:
                    continue
                key = (gpu or "", name or "")
                if name is None or not gpu:
                    replaced.append(sample)
                    continue
                templates.setdefault(key, (family, sample))
                if key in omitted:
                    served[f"{gpu}/{name}"] = "omitted"
                    continue
                value = float(sample.value)
                if key in values:
                    value = values[key]
                elif key in self.offsets:
                    value = value + self.offsets[key]
                updated = sample._replace(value=value)
                if key in labels:
                    updated = updated._replace(labels={**sample.labels, **labels[key]})
                if updated != sample:
                    served[f"{gpu}/{name}"] = "NaN" if math.isnan(value) else value
                replaced.append(updated)
            family.samples = replaced
        for gpu in sorted(omitted_gpus):
            served[gpu] = "omitted"
        by_name = {family.name: family for family in families}
        if not self._synthesise(puts, values, templates, families, by_name, served):
            return body, {}
        for gpu, item in extras:
            template = templates.get((gpu, item["field"]))
            if template is None:
                self.aborted = f"no real {gpu}/{item['field']} sample to derive from"
                return body, {}
            family, sample = template
            if item["op"] == "duplicate":
                family.samples.append(sample._replace(value=float(item["value"])))
                served[f"{gpu}/{item['field']}#duplicate"] = item["value"]
                continue
            name = str(item["name"])
            if name == sample.name or canonical_name(name) != item["field"]:
                self.aborted = f"{name} is not a distinct alias of {sample.name}"
                return body, {}
            target = by_name.get(name)
            if target is None:
                target = Metric(name, family.documentation, family.type)
                families.append(target)
                by_name[name] = target
            target.samples.append(
                sample._replace(name=name, value=float(item["value"]))
            )
            served[f"{gpu}/{item['field']}#alias:{name}"] = item["value"]

        class Registry:
            def collect(self):
                return families

        return generate_latest(Registry()), served

    def _synthesise(
        self,
        puts: set[tuple[str, str]],
        values: dict[tuple[str, str], float],
        templates: dict[tuple[str, str], tuple[Any, Any]],
        families: list[Any],
        by_name: dict[str, Any],
        served: dict[str, Any],
    ) -> bool:
        """Serve a ``set`` of a declared absent-when-healthy gauge the upstream
        does not publish: the line is built from the same GPU's own published
        label set (``LABEL_TEMPLATE_FIELD`` first) plus the exporter's labels
        for that value, under the exporter's HELP/TYPE header -- an existing
        upstream family of the gauge (another GPU's real XID) is joined, not
        duplicated. Only a ``set`` synthesises; a real sample of the gauge is
        served the ordinary way. False after an abort.
        """

        tolerated = absent_when_healthy(self.plan)
        for key in sorted(puts):
            gpu, name = key
            if key in templates or name not in tolerated:
                continue
            template = templates.get((gpu, LABEL_TEMPLATE_FIELD)) or next(
                (item for (owner, _field), item in templates.items() if owner == gpu),
                None,
            )
            if template is None:
                self.aborted = f"no real sample of {gpu} to clone a label set from"
                return False
            _family, sample = template
            gauge = ABSENT_WHEN_HEALTHY[name]
            target = by_name.get(gauge.name)
            if target is None:
                target = Metric(gauge.name, gauge.documentation, "gauge")
                families.append(target)
                by_name[gauge.name] = target
            value = values[key]
            target.samples.append(
                sample._replace(
                    name=gauge.name,
                    labels={**sample.labels, **gauge.labels(value)},
                    value=value,
                )
            )
            served[f"{gpu}/{name}"] = value
            served[f"{gpu}/{name}#synthesised"] = gauge.name
        return True


def make_server(upstream: str, port: int, replay: Replay, audit: Any) -> HTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup()
            self.connection.settimeout(5)

        def log_message(self, format, *args):  # noqa: A002 - BaseHTTPRequestHandler API
            return

        def do_GET(self):
            if self.path == "/status":
                current = replay.active()
                body = json.dumps(
                    {
                        "reads": replay.reads,
                        "profile": replay.plan["profile"],
                        "proof_scope": PROOF_SCOPE,
                        "armed": replay.armed,
                        "disarmed": replay.disarmed,
                        "aborted": replay.aborted,
                        "step": current.step if current is not None else None,
                        "offsets": {
                            f"{gpu}/{name}": value
                            for (gpu, name), value in sorted(replay.offsets.items())
                        },
                    }
                ).encode()
                self.respond(200, body, "application/json")
                return
            if self.path != "/metrics":
                self.respond(404, b"not found\n")
                return
            try:
                with open_upstream(upstream, timeout=5) as response:
                    original = response.read(MAX_BYTES + 1)
                output, receipt = replay.render(original)
            except (OSError, ValueError, UnicodeError) as error:
                audit.write(
                    json.dumps(
                        {
                            "error": str(error),
                            "read": replay.reads + 1,
                            "at": time.time(),
                        }
                    )
                    + "\n"
                )
                audit.flush()
                self.respond(
                    503, b"upstream/replay unavailable; no healthy data fabricated\n"
                )
                return
            # Journal before publishing. The collector's and the control plane's
            # own records, not this prepared receipt, prove delivery.
            audit.write(json.dumps({**receipt, "delivery": "prepared"}) + "\n")
            audit.flush()
            self.respond(200, output)
            replay.reads += 1

        def do_POST(self):
            if (
                self.path not in {"/arm", "/disarm", "/advance"}
                or self.headers.get("X-GFRS-Replay-Ack") != ACKNOWLEDGEMENT
            ):
                self.respond(403, b"explicit local replay acknowledgement required\n")
                return
            if self.path == "/advance":
                try:
                    step = replay.advance()
                except ValueError as error:
                    self.respond(409, str(error).encode())
                    return
                audit.write(
                    json.dumps(
                        {
                            "event": "ADVANCE",
                            "read": replay.reads,
                            "step": step,
                            "at": time.time(),
                        }
                    )
                    + "\n"
                )
                audit.flush()
                self.respond(200, b"advanced\n")
                return
            if self.path == "/disarm":
                replay.disarm()
                audit.write(
                    json.dumps(
                        {"event": "DISARM", "read": replay.reads, "at": time.time()}
                    )
                    + "\n"
                )
                audit.flush()
                self.respond(200, b"disarmed\n")
                return
            try:
                replay.arm()
            except ValueError as error:
                self.respond(409, str(error).encode())
                return
            audit.write(
                json.dumps(
                    {
                        "event": "ARM",
                        "start_read": replay.start_read,
                        "at": time.time(),
                    }
                )
                + "\n"
            )
            audit.flush()
            self.respond(200, b"armed\n")

        def respond(
            self, status: int, body: bytes, content_type="text/plain; version=0.0.4"
        ):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return HTTPServer(("127.0.0.1", port), Handler)


def serve(upstream: str, port: int, replay: Replay, audit: Any) -> None:
    with make_server(upstream, port, replay, audit) as server:
        print(
            f"DCGM field replay http://127.0.0.1:{port}/metrics "
            f"(profile {replay.plan['profile']}); /status does not consume steps",
            flush=True,
        )
        print("Pass-through until POST /arm with X-GFRS-Replay-Ack.", flush=True)
        try:
            server.serve_forever(poll_interval=0.5)
        except KeyboardInterrupt:
            pass


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--profile", required=True, choices=sorted(PROFILES))
    parser.add_argument("--upstream", default="http://127.0.0.1:9400/metrics")
    parser.add_argument("--port", type=int, default=9412)
    parser.add_argument("--max-seconds", type=int, default=1800)
    parser.add_argument(
        "--serve", action="store_true", help="default validates only; no network"
    )
    parser.add_argument("--ack", default="")
    parser.add_argument("--audit", type=Path)
    args = parser.parse_args(argv)
    if not 1024 <= args.port <= 65535 or not 1 <= args.max_seconds <= MAX_SECONDS:
        parser.error(f"invalid port or max-seconds (1..{MAX_SECONDS})")
    try:
        plan = validate_plan(json.loads(args.plan.read_text()))
        upstream = validate_upstream(args.upstream, args.port)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    if plan["profile"] != args.profile:
        parser.error("the plan's profile differs from --profile")
    if not args.serve:
        print(json.dumps({"validated": True, "network_started": False, "plan": plan}))
        return 0
    if args.ack != ACKNOWLEDGEMENT or args.audit is None:
        parser.error(
            "--serve requires --ack ISOLATED_GPU_TEST_NODE and a new --audit path"
        )
    # Exclusive creation keeps two acceptance runs from sharing one journal.
    with args.audit.open("x", encoding="utf-8") as audit:
        serve(upstream, args.port, Replay(copy.deepcopy(plan), args.max_seconds), audit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

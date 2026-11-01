"""The NVIDIA evidence of the batch-235 (HGX B200/B300) DCGM rules, pinned.

Every rule the 2026-11-01 Blackwell field audit led to cites NVIDIA -- the exact
page (URL and section), the docs version and a verbatim quote -- in the code
constants that carry it, in both language editions of the fault table, the
detailed design and the NVIDIA policy audit. The quotes were re-fetched for
batch 235 (NVML API Reference and GPU Memory Error Management docs version
615; DCGM docs "Version: 4.6 (Latest)"; DCGM source tag v4.7.0; the Fabric
Manager User Guide "Last updated on Oct 07, 2026"; the nvidia-smi
documentation of the v610 era; NVIDIA's ``nvidia-ml-py`` 13.615.71). A quote
that changes here must change on NVIDIA's page first. Everything Blackwell is
stated as unit-tested from NVIDIA documentation, not live-verified.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from gpu_fault import (
    dcgm_fields,
    gpu_fabric_health,
    gpu_memory_policy,
    gpu_memory_repair_policy,
    gpu_nvlink5_policy,
    gpu_product_windows,
)
from gpu_fault.policy import xid_cross_source

ROOT = Path(__file__).resolve().parents[1]
DESIGNS = ("docs/详细设计.md", "docs/en/detailed-design.md")
FAULT_TABLES = (
    "docs/故障类别与处置动作总表.md",
    "docs/en/fault-categories-and-actions.md",
)
POLICY_AUDITS = (
    "docs/components/nvidia-policy.md",
    "docs/en/components/nvidia-policy.md",
)
EN_DOCUMENTS = (
    "docs/en/detailed-design.md",
    "docs/en/fault-categories-and-actions.md",
    "docs/en/components/nvidia-policy.md",
    "docs/en/deployment-and-operations-manual.md",
    "docs/validation-limitations.md",
)
CN_DOCUMENTS = (
    "docs/详细设计.md",
    "docs/故障类别与处置动作总表.md",
    "docs/components/nvidia-policy.md",
    "docs/部署和运维手册.md",
)
BLACKWELL_EN = (
    "unit-tested from NVIDIA documentation, not live-verified (no B-series hardware)"
)
BLACKWELL_CN = "依据 NVIDIA 文档做了单元测试，未经现网验证（没有 B 系列硬件）"
#: The ops note of the deployment manuals (both editions name the version).
HOST_ENGINE_NOTE = "4.7.0"

#: (module, constant) -> the verbatim NVIDIA text.
QUOTES = {
    (gpu_nvlink5_policy, "NVML_DL_UNSUPPORTED_QUOTE"): (
        "This is unsupported for Blackwell+ Please use "
        "NVML_FI_DEV_NVLINK_COUNT_LINK_RECOVERY_*"
    ),
    (gpu_nvlink5_policy, "DCGM_NVLINK5_SCOPE_QUOTE"): (
        "Returns aggregate value across all links. Not supported on NVLink4 and "
        "earlier."
    ),
    (gpu_nvlink5_policy, "DCGM_AGGREGATE_QUOTE"): (
        "Aggregate (GPU entity) — returns the sum across all active NVLink ports on "
        "a GPU."
    ),
    (gpu_nvlink5_policy, "DCGM_NVLINK_CRITICAL_QUOTE"): (
        "An NVLink reported 1 or more errors. This includes datalink CRC, recovery, "
        "and replay errors (pre-Blackwell GPUs) and link recovery events "
        "(Blackwell+ GPUs)."
    ),
    (gpu_nvlink5_policy, "DCGM_NVLINK_CRITICAL_MESSAGE"): (
        "Detected %ld %s NvLink errors on GPU %u's NVLink (should be 0)"
    ),
    (gpu_nvlink5_policy, "DCGM_FIELD_DIAGNOSTIC_NEXT"): (
        "Run a field diagnostic on the GPU."
    ),
    (gpu_nvlink5_policy, "DCGM_BER_REMOVED_QUOTE"): (
        "Removed NVLink bit-error-rate thresholds that caused false health and "
        "diagnostic failures on healthy GB200 systems."
    ),
    (gpu_fabric_health, "NVML_ROUTE_UNHEALTHY_QUOTE"): (
        "Fabric Health Mask: Route is unhealthy."
    ),
    (gpu_fabric_health, "NVML_DEGRADED_BW_QUOTE"): (
        "Fabric Health Mask: Bandwidth degraded."
    ),
    (gpu_fabric_health, "NVML_SUMMARY_UNHEALTHY_QUOTE"): (
        "Fabric Health Summary: Unhealthy."
    ),
    (gpu_fabric_health, "NVML_SUMMARY_LIMITED_QUOTE"): (
        "Fabric Health Summary: Limited Capacity."
    ),
    (gpu_fabric_health, "DCGM_FABRIC_HEALTH_QUOTE"): (
        "Use DCGM_GPU_FABRIC_HEALTH_TEST macro to check the different health statuses."
    ),
    (gpu_fabric_health, "DCGM_SYSTEM_TRIAGE_NEXT"): (
        "Check DCGM and system logs for errors. Reset GPU. Restart DCGM. Rerun "
        "diagnostics."
    ),
    (gpu_fabric_health, "NVIDIA_SMI_SUMMARY_QUOTE"): (
        "Summary - Summary of Fabric Health <Healthy, Unhealthy, Limited Capacity>"
    ),
    (gpu_fabric_health, "NVIDIA_SMI_ROUTE_UNHEALTHY_QUOTE"): (
        "Route Unhealthy - is NVLink route recovery failed or aborted <True/False>"
    ),
    (gpu_fabric_health, "FM_DEGRADED_MODE_QUOTE"): (
        "These high availability modes and their corresponding dynamic "
        "reconfiguration of the NVSwitch-based system are applied in response to "
        "errors that are detected during FM initialization."
    ),
    (gpu_memory_repair_policy, "NVIDIA_MEMORY_REPAIR_QUOTE"): (
        "GPU memory repair is another resiliency feature available on select "
        "Blackwell products. GPU memory repair consists of swapping spare DRAM "
        "channels or L2 cache slices."
    ),
    (gpu_memory_repair_policy, "NVIDIA_REPAIR_REBOOT_QUOTE"): (
        "XID 160 will be printed if a DRAM channel or L2 slice is successfully "
        "marked for repair. The process of channel repair requires a reboot to "
        "take effect."
    ),
    (gpu_memory_repair_policy, "NVIDIA_SMI_REPAIR_QUOTE"): (
        "If one of the repair flags is pending, check the GPU Recovery action and "
        "take the appropriate steps."
    ),
    (gpu_memory_repair_policy, "NVIDIA_SMI_REPAIR_CHANGE_LOG_QUOTE"): (
        "Added Channel/TPC repair pending flags to ECC field: 'nvidia-smi -q -d ECC'"
    ),
    (gpu_memory_repair_policy, "NVML_UNREPAIRABLE_CHANGE_LOG_QUOTE"): (
        "Added nvmlDeviceGetUnrepairableMemoryFlag_v1() to report whether or not "
        "there is unrepairable memory."
    ),
    (gpu_memory_repair_policy, "DCGM_UNREPAIRABLE_QUOTE"): (
        "Unrepairable memory flag indicating if memory has unrepairable errors "
        "1=yes, 0=no"
    ),
}
#: NVML v615 ``NVML_FI_DEV_NVLINK_COUNT_*``, field id -> the verbatim line.
NVML_FIELD_QUOTES = {
    1204: "Number of packets Rx on a link where packets are malformed.",
    1205: "Number of packets that were discarded on Rx due to buffer overrun.",
    1206: "Total number of packets with errors Rx on a link.",
    1207: "Total number of packets Rx - stomp/EBP marker.",
    1208: "Total number of packets Rx with header mismatch.",
    1209: "Total number of times that the count of local errors exceeded a threshold.",
    1211: (
        "Number of times link went from Up to recovery, succeeded and link came "
        "back up."
    ),
    1212: (
        "Number of times link went from Up to recovery, failed and link was "
        "declared down."
    ),
    1213: "Number of times link went from Up to recovery, irrespective of the result.",
    1214: "Number of errors in rx symbols.",
    1219: "Sum of the number of errors in each Nvlink packet.",
    1220: "NVLink data ECC Error Counter total for all Links.",
}
NVIDIA_SMI_FLAG_QUOTES = {
    "memory_channel_repair_pending": "Indicates if a Channel repair is pending",
    "memory_tpc_repair_pending": "Indicates if a TPC repair is pending",
    "memory_unrepairable": "Indicates if there is unrepairable memory",
}
#: Every page a batch-235 rule cites; :data:`DESIGN_ONLY` the ones the fault
#: tables and the policy audits leave to the detailed design.
REFERENCES = {
    "NVML field values": gpu_nvlink5_policy.NVML_FIELD_VALUE_REFERENCE,
    "DCGM field identifiers": gpu_nvlink5_policy.DCGM_FIELD_REFERENCE,
    "DCGM NVLink COUNT fields": gpu_nvlink5_policy.DCGM_AGGREGATE_REFERENCE,
    "DCGM health checks": gpu_nvlink5_policy.DCGM_HEALTH_REFERENCE,
    "DcgmHealthWatch.cpp": gpu_nvlink5_policy.DCGM_HEALTH_WATCH_REFERENCE,
    "DCGM release notes": gpu_nvlink5_policy.DCGM_RELEASE_NOTES_REFERENCE,
    "NVML fabric definitions": gpu_fabric_health.NVML_FABRIC_REFERENCE,
    "DCGM fabric health mask": gpu_fabric_health.DCGM_FABRIC_HEALTH_REFERENCE,
    "nvidia-smi": gpu_fabric_health.NVIDIA_SMI_FABRIC_HEALTH_REFERENCE,
    "Fabric Manager HA modes": gpu_fabric_health.FM_DEGRADED_MODE_REFERENCE,
    "GPU memory repair": gpu_memory_repair_policy.NVIDIA_MEMORY_REPAIR_REFERENCE,
    "NVML change log": gpu_memory_repair_policy.NVML_CHANGE_LOG_REFERENCE,
    "DCGM field 1507": gpu_memory_policy.DCGM_UNREPAIRABLE_REFERENCE,
}
DESIGN_ONLY = {"DCGM NVLink COUNT fields", "Fabric Manager HA modes"}


def text_of(document: str) -> str:
    """The document with every whitespace run collapsed (quotes wrap lines)."""

    return re.sub(r"\s+", " ", (ROOT / document).read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    ("owner", "quote"), list(QUOTES.items()), ids=[name for _module, name in QUOTES]
)
def test_the_code_carries_nvidias_text_verbatim(owner, quote):
    module, name = owner
    assert getattr(module, name) == quote, f"{module.__name__}.{name}"


def test_the_nvml_counter_lines_and_the_flag_lines_are_verbatim():
    assert gpu_nvlink5_policy.NVML_FIELD_QUOTES == NVML_FIELD_QUOTES
    assert set(NVML_FIELD_QUOTES) == {
        field for field, _legacy, _current in gpu_nvlink5_policy.NVLINK5_FIELDS.values()
    }
    assert gpu_memory_repair_policy.NVIDIA_SMI_FLAG_QUOTES == NVIDIA_SMI_FLAG_QUOTES


@pytest.mark.parametrize("document", DESIGNS)
def test_both_detailed_designs_quote_every_rule(document):
    text = text_of(document)
    missing = [name for (_m, name), quote in QUOTES.items() if quote not in text]
    missing += [
        str(field) for field, line in NVML_FIELD_QUOTES.items() if line not in text
    ]
    missing += [
        name for name, line in NVIDIA_SMI_FLAG_QUOTES.items() if line not in text
    ]

    assert missing == [], f"{document} lacks NVIDIA's text for {missing}"
    assert "Version: 4.6 (Latest)" in text, document
    assert "docs version 615" in text, document
    assert "13.615.71" in text, document


@pytest.mark.parametrize("document", DESIGNS + FAULT_TABLES + POLICY_AUDITS)
def test_every_document_cites_the_pages(document):
    text = text_of(document)
    skipped = set() if document in DESIGNS else DESIGN_ONLY
    missing = [
        name
        for name, url in REFERENCES.items()
        if name not in skipped and url not in text
    ]

    assert missing == [], f"{document} lacks the NVIDIA page for {missing}"


@pytest.mark.parametrize("document", EN_DOCUMENTS)
def test_the_english_documents_state_the_blackwell_status(document):
    assert BLACKWELL_EN in text_of(document), document


@pytest.mark.parametrize("document", CN_DOCUMENTS)
def test_the_chinese_documents_state_the_blackwell_status(document):
    assert BLACKWELL_CN in text_of(document), document


@pytest.mark.parametrize(
    "module",
    [gpu_nvlink5_policy, gpu_fabric_health, gpu_memory_repair_policy],
    ids=lambda module: module.__name__,
)
def test_the_policy_modules_state_the_blackwell_status(module):
    text = re.sub(r"\s+", " ", module.__doc__ or "")
    assert "not live-verified" in text and "NVIDIA documentation" in text, (
        module.__name__
    )


@pytest.mark.parametrize(
    "document",
    ["docs/en/deployment-and-operations-manual.md", "docs/部署和运维手册.md"],
)
def test_the_manuals_carry_the_b_series_host_engine_note(document):
    text = text_of(document)
    assert "B200" in text and HOST_ENGINE_NOTE in text, document
    assert "DCGM_FI_DEV_NVLINK_COUNT_LINK_RECOVERY_FAILED_EVENTS" in text, document


def test_the_modules_name_their_windows_and_twins():
    """The window and the twin rules carry their reasons in the code."""

    assert "Fabric Manager" in (gpu_product_windows.__doc__ or "")
    assert "kernel twin" in (xid_cross_source.__doc__ or "")
    assert dcgm_fields.NVLINK5_LINK_RECOVERY_CANONICAL == (
        "nvlink_link_recovery_successful_total",
        "nvlink_link_recovery_failed_total",
        "nvlink_link_recovery_total",
    )

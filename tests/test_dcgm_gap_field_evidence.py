"""The NVIDIA evidence of the batch-233 DCGM field rules, pinned.

Every rule the 2026-11-01 field audit added cites NVIDIA: the exact page
(URL and section), the docs version and a verbatim quote, in the code
constants that carry it, in both language editions of the fault table, the
detailed design and the NVIDIA policy audit. The quotes were re-fetched on
2026-11-01 (NVML and GPU Memory Error Management docs version 615, pages "Last
updated on Sep 09, 2026"; DCGM docs "Version: 4.6 (Latest)"; DCGM source tag
v4.7.0; nvidia-smi documentation of the v610 era). A quote that changes here
must change on NVIDIA's page first. Blackwell (B200/B300) behaviour is stated
as unit-tested from NVIDIA documentation, not live-verified.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from gpu_fault import (
    gpu_environment_policy,
    gpu_field_rules,
    gpu_memory_policy,
    gpu_pcie_link_policy,
    gpu_recovery_action_policy,
)
from gpu_fault.gpu_power_policy import NVIDIA_DCGM_HEALTH_REFERENCE

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

#: (module, constant) -> the verbatim NVIDIA text.
QUOTES = {
    (gpu_recovery_action_policy, "NVIDIA_RECOVERY_ACTION_QUOTE"): (
        "Recovery action flag will be triggered for uncorrectable errors."
    ),
    (gpu_recovery_action_policy, "DCGM_RECOVERY_ACTION_FIELD_QUOTE"): (
        "GPU Recovery Action (see nvmlDeviceGpuRecoveryAction_t for return values)"
    ),
    (gpu_recovery_action_policy, "NVML_RECOVERY_ACTION_CHANGE_LOG_QUOTE"): (
        "Added NVML_GPU_RECOVERY_ACTION_BUS_RESET and "
        "NVML_GPU_RECOVERY_ACTION_SYSTEM_REBOOT to nvmlDeviceGpuRecoveryAction_t."
    ),
    (gpu_recovery_action_policy, "NVML_IMEX_CHANGE_LOG_QUOTE"): (
        "Added new NVML_GPU_RECOVERY_ACTION_RECOVER_IMEX_DOMAIN enum to "
        "nvmlDeviceGpuRecoveryAction_s."
    ),
    (gpu_recovery_action_policy, "DCGM_IMEX_RECOVERY_QUOTE"): (
        "Added DCGM_FR_GPU_RECOVERY_IMEX_DOMAIN for the separate case in which "
        "driver recovery requires restoring an IMEX domain."
    ),
    (gpu_recovery_action_policy, "DCGM_IMEX_RECOVERY_NEXT"): (
        "Recover or re-establish the IMEX domain."
    ),
    (gpu_recovery_action_policy, "NVIDIA_SMI_RESET_TYPE_QUOTE"): (
        "Reset triggered without extra arguments, will be a default Function Level "
        "Reset (FLR). To issue a Bus Reset, use -r bus. For certain platforms only "
        "Function Level Reset is possible."
    ),
    (gpu_recovery_action_policy, "DCGM_HEALTH_RECOVERY_RANGE_QUOTE"): (
        "Invalid GPU recovery action value"
    ),
    (gpu_pcie_link_policy, "DCGM_PCIE_REPLAY_DOC_QUOTE"): (
        "There were more than 8 PCIe replays in the last minute."
    ),
    (gpu_pcie_link_policy, "DCGM_DIAG_REPLAY_DOC_QUOTE"): (
        "This is based on an expected replay rate less than per minute for PCIe Gen "
        "3.0, assuming this plugin will run for less than a minute and allowing 10x "
        "as many replays before failure."
    ),
    (gpu_pcie_link_policy, "DCGM_REPLAY_RATE_CODE_QUOTE"): (
        "Expected replay rate = Replay rate per lane * Num of pcie Lanes. Note that "
        "value is ceiled."
    ),
    (gpu_pcie_link_policy, "DCGM_REPLAY_RATE_CAP_QUOTE"): (
        "Cap the version at the max we know about."
    ),
    (gpu_pcie_link_policy, "NVIDIA_SMI_CURRENT_LINK_QUOTE"): (
        "The current link generation and width. These may be reduced when the GPU "
        "is not in use."
    ),
    (gpu_pcie_link_policy, "NVIDIA_SMI_MAX_LINK_QUOTE"): (
        "The maximum link generation and width possible with this GPU and system "
        "configuration."
    ),
    (gpu_pcie_link_policy, "NVML_MAX_LINK_QUOTE"): (
        "Retrieves the maximum PCIe link generation possible with this device and "
        "system."
    ),
    (gpu_pcie_link_policy, "NVIDIA_DGX_B300_PCIE_QUOTE"): (
        "ConnectX-8 networking connects directly to GPUs via PCIe Gen6."
    ),
    (gpu_pcie_link_policy, "NVML_PCIE_CORRECTABLE_QUOTE"): (
        "PCIe Correctable Errors Counter."
    ),
    (gpu_pcie_link_policy, "DCGM_PCIE_AER_RATE_QUOTE"): (
        "/* Default is hopper, but fall back for other devices */"
    ),
    (gpu_memory_policy, "NVIDIA_SRAM_RMA_QUOTE"): (
        "The NVIDIA Field Diagnostic tool will determine whether a GPU qualifies for "
        "RMA. Regarding SRAM uncorrectable errors, the RMA criteria is met for events "
        "outlined below. Any of the following events will trigger the SRAM Threshold "
        "Exceeded flag:"
    ),
    (gpu_memory_policy, "NVIDIA_SMI_BANK_REMAP_QUOTE"): (
        "Maximum availability means that all reserved rows are available for "
        "remapping while None means that no reserved rows are available."
    ),
    (gpu_memory_policy, "DCGM_ROW_REMAP_LIMIT_QUOTE"): (
        "Health monitoring now warns when total uncorrectable row remaps reach 512 on "
        "Ampere and newer GPUs, allowing administrators to run field diagnostics "
        "before GPU memory health degrades further."
    ),
    (gpu_memory_policy, "NVIDIA_ROW_REMAP_FAILURE_BANK_QUOTE"): (
        "A remapping attempt for an uncorrectable memory error on a bank that already "
        "has eight uncorrectable error rows remapped."
    ),
    (gpu_memory_policy, "NVIDIA_BLACKWELL_CHANNEL_REPAIR_QUOTE"): (
        "For Blackwell GPUs, a third remapping attempt for an uncorrectable memory "
        "error on a bank invokes HBM channel repair if there is a spare channel "
        "available. If the channel repair is successful, the hardware will be "
        "recovered and ready to use. Otherwise, the GPU will continue remapping until "
        "Row-Remap Failure flag is set."
    ),
    (gpu_field_rules, "NVML_CLOCK_EVENT_COUNTERS_QUOTE"): (
        "Field values for Clock Throttle Reason Counters All counters are in "
        "nanoseconds."
    ),
    (gpu_field_rules, "NVML_ECC_REQUIRED_QUOTE"): "Requires ECC Mode to be enabled.",
    (gpu_field_rules, "NVML_ECC_PENDING_QUOTE"): (
        "Changing ECC modes requires a reboot. The “pending” ECC mode refers to the "
        "target mode following the next reboot."
    ),
    (gpu_field_rules, "NVIDIA_INFOROM_RETIREMENT_QUOTE"): (
        "These addresses are stored in the InfoROM."
    ),
    (gpu_field_rules, "NVIDIA_SMI_INFOROM_QUOTE"): (
        'Inforom checksum validation ("valid", "invalid", "N/A") Only available via '
        "--query-gpu inforom.checksum_validation"
    ),
    (gpu_environment_policy, "NVIDIA_SMI_HW_SLOWDOWN_QUOTE"): (
        "This option will be removed a future CUDA release. HW Slowdown is engaged, "
        "reducing the core clocks by a factor of 2 or more. It is active if either "
        "HW Thermal Slowdown or HW Power Brake are active."
    ),
    (gpu_environment_policy, "NVML_HW_SLOWDOWN_QUOTE"): (
        "HW Slowdown (reducing the core clocks by a factor of 2 or more) is engaged. "
        "This is an indicator of: temperature being too high External Power Brake "
        "Assertion is triggered (e.g. by the system power supply) Power draw is too "
        "high and Fast Trigger protection is reducing the clocks May be also "
        "reported during PState or clock change This behavior may be removed in a "
        "later release."
    ),
    (gpu_environment_policy, "NVML_BOARD_LIMIT_QUOTE"): (
        "The board limit (operating) policy is currently limiting the GPU clocks."
    ),
    (gpu_environment_policy, "NVML_RELIABILITY_QUOTE"): (
        "The reliability policy is currently limiting the GPU clocks."
    ),
}
#: Every page a batch-233 rule cites; :data:`DESIGN_ONLY` and
#: :data:`NOT_IN_POLICY_AUDITS` the ones the fault tables and the policy audits
#: leave to the detailed design.
REFERENCES = {
    "recovery flags": gpu_recovery_action_policy.NVIDIA_RECOVERY_ACTION_REFERENCE,
    "recovery enum": gpu_recovery_action_policy.NVML_RECOVERY_ACTION_ENUM_REFERENCE,
    "DCGM release notes": gpu_recovery_action_policy.DCGM_IMEX_RECOVERY_REFERENCE,
    "NVML change log": (
        gpu_recovery_action_policy.NVML_RECOVERY_ACTION_CHANGE_LOG_REFERENCE
    ),
    "DCGM field 1523": gpu_recovery_action_policy.DCGM_RECOVERY_ACTION_FIELD_REFERENCE,
    "nvidia-smi": gpu_pcie_link_policy.NVIDIA_SMI_REFERENCE,
    "DGX B300": gpu_pcie_link_policy.NVIDIA_DGX_B300_PCIE_REFERENCE,
    "DCGM health checks": NVIDIA_DCGM_HEALTH_REFERENCE,
    "DCGM diagnostics": gpu_pcie_link_policy.DCGM_DIAG_REPLAY_DOC_REFERENCE,
    "DcgmHealthWatch.cpp": gpu_pcie_link_policy.DCGM_REPLAY_RATE_CODE_REFERENCE,
    "Pcie.cpp": gpu_pcie_link_policy.DCGM_PCIE_AER_RATE_REFERENCE,
    "NVML correctable": gpu_pcie_link_policy.NVML_PCIE_CORRECTABLE_REFERENCE,
    "NVML device queries": gpu_field_rules.NVML_DEVICE_QUERIES_REFERENCE,
    "SRAM RMA": gpu_memory_policy.NVIDIA_SRAM_RMA_REFERENCE,
    "DRAM RMA": gpu_memory_policy.NVIDIA_DRAM_RMA_REFERENCE,
    "NVML field values": gpu_field_rules.NVML_FIELD_VALUE_REFERENCE,
    "NVML clock events": gpu_environment_policy.NVML_CLOCK_REFERENCE,
    "InfoROM": gpu_field_rules.NVIDIA_INFOROM_RETIREMENT_REFERENCE,
}
DESIGN_ONLY = {"DCGM diagnostics"}
NOT_IN_POLICY_AUDITS = DESIGN_ONLY | {"DCGM field 1523"}


def text_of(document: str) -> str:
    """The document with every whitespace run collapsed (quotes wrap lines)."""

    return re.sub(r"\s+", " ", (ROOT / document).read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    ("owner", "quote"),
    [(key, value) for key, value in QUOTES.items()],
    ids=[name for _module, name in QUOTES],
)
def test_the_code_carries_nvidias_text_verbatim(owner, quote):
    module, name = owner
    assert getattr(module, name) == quote, f"{module.__name__}.{name}"


def test_the_recovery_action_enum_meanings_are_nvml_v615s():
    assert gpu_recovery_action_policy.NVML_RECOVERY_ACTION_ENUM_MEANINGS == {
        5: "Recover IMEX Domain.",
        6: "Reset the GPU’s PCIe bus.",
        7: "Reboot the system.",
    }
    assert gpu_recovery_action_policy.NVIDIA_RECOVERY_ACTION_DOCS_VERSION == "615"


@pytest.mark.parametrize("document", DESIGNS)
def test_both_detailed_designs_quote_every_rule(document):
    text = text_of(document)
    missing = [name for (_m, name), quote in QUOTES.items() if quote not in text]
    meanings = gpu_recovery_action_policy.NVML_RECOVERY_ACTION_ENUM_MEANINGS
    missing += [str(value) for value, line in meanings.items() if line not in text]

    assert missing == [], f"{document} lacks NVIDIA's text for {missing}"
    assert "Version: 4.6 (Latest)" in text, document
    assert "docs version 615" in text, document


@pytest.mark.parametrize("document", DESIGNS + FAULT_TABLES + POLICY_AUDITS)
def test_every_document_cites_the_pages(document):
    text = text_of(document)
    skipped = (
        set()
        if document in DESIGNS
        else (DESIGN_ONLY if document in FAULT_TABLES else NOT_IN_POLICY_AUDITS)
    )
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
    [gpu_recovery_action_policy, gpu_pcie_link_policy],
    ids=lambda module: module.__name__,
)
def test_the_policy_modules_state_the_blackwell_status(module):
    assert BLACKWELL_EN in re.sub(r"\s+", " ", module.__doc__ or ""), module.__name__

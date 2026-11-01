from __future__ import annotations

from collections.abc import Mapping, Set

#: The NVLink5 link-recovery counters, DCGM fields 1211-1213 (batch 235). NVML
#: does not report the data-link counters 497-499 on Blackwell -- NVML API
#: Reference (docs version 615), "Field Value Enums",
#: ``NVML_FI_DEV_NVLINK_ERROR_DL_CRC``: "This is unsupported for Blackwell+
#: Please use NVML_FI_DEV_NVLINK_COUNT_LINK_RECOVERY_*" (``_DL_RECOVERY`` and
#: ``_DL_REPLAY`` say the same) -- so on a B200/B300 these are the counters
#: that answer for the GPU's NVLink. Each is an alternative of every NVLink
#: group below; Hopper and earlier parts never report them ("Note: NVLink5+
#: only. ... Not supported on NVLink4 and earlier.", DCGM Field Identifiers,
#: docs 4.6), so what a Hopper scrape needs is unchanged.
NVLINK5_LINK_RECOVERY_CANONICAL = (
    "nvlink_link_recovery_successful_total",
    "nvlink_link_recovery_failed_total",
    "nvlink_link_recovery_total",
)
#: Their exporter names as the bundled counters CSV exports them: the DCGM
#: 4.5.3 names the pinned exporter's go-dcgm v1.4611.2 resolves through its
#: legacy map, then the DCGM 4.6 names an exporter of the customer's may use.
NVLINK5_LINK_RECOVERY_EXPORTER_NAMES = (
    "DCGM_FI_DEV_NVLINK_COUNT_LINK_RECOVERY_SUCCESSFUL_EVENTS",
    "DCGM_FI_DEV_NVLINK_COUNT_LINK_RECOVERY_FAILED_EVENTS",
    "DCGM_FI_DEV_NVLINK_COUNT_LINK_RECOVERY_EVENTS",
    "DCGM_FI_DEV_NVLINK_RECOVERY_SUCCESSFUL_TOTAL",
    "DCGM_FI_DEV_NVLINK_RECOVERY_FAILED_TOTAL",
    "DCGM_FI_DEV_NVLINK_RECOVERY_EVENT_TOTAL",
)

DCGM_FABRIC_CANONICAL_GROUPS = {
    "nvlink_crc": frozenset(
        {
            "nvlink_crc_aggregate_error_total",
            "nvlink_crc_flit_error_total",
            "nvlink_crc_data_error_total",
            *NVLINK5_LINK_RECOVERY_CANONICAL,
        }
    ),
    "nvlink_recovery": frozenset(
        {
            "nvlink_recovery_aggregate_error_total",
            "nvlink_recovery_error_total",
            *NVLINK5_LINK_RECOVERY_CANONICAL,
        }
    ),
    "nvlink_replay": frozenset(
        {
            "nvlink_replay_aggregate_error_total",
            "nvlink_replay_error_total",
            *NVLINK5_LINK_RECOVERY_CANONICAL,
        }
    ),
}

DCGM_CORE_CANONICAL_FIELDS = frozenset(
    {
        "gpu_temperature_c",
        "ecc_dbe_volatile_total",
        "row_remap_failure",
        "row_remap_pending",
        "pcie_replay_total",
        "clock_throttle_reasons",
        "power_limit_w",
    }
)

DCGM_EXPORTER_FIELD_GROUPS = (
    ("DCGM_FI_DEV_ROW_REMAP_FAILURE",),
    ("DCGM_FI_DEV_ECC_DBE_VOL_TOTAL",),
    ("DCGM_FI_DEV_PCIE_REPLAY_COUNTER",),
    (
        "DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_TOTAL",
        "DCGM_FI_DEV_NVLINK_ERROR_DL_CRC",
        *NVLINK5_LINK_RECOVERY_EXPORTER_NAMES,
    ),
    (
        "DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_TOTAL",
        "DCGM_FI_DEV_NVLINK_ERROR_DL_RECOVERY",
        *NVLINK5_LINK_RECOVERY_EXPORTER_NAMES,
    ),
    (
        "DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_TOTAL",
        "DCGM_FI_DEV_NVLINK_ERROR_DL_REPLAY",
        *NVLINK5_LINK_RECOVERY_EXPORTER_NAMES,
    ),
    ("DCGM_FI_DEV_CLOCK_THROTTLE_REASONS", "DCGM_FI_DEV_CLOCKS_EVENT_REASONS"),
    ("DCGM_FI_DEV_POWER_MGMT_LIMIT",),
)


def missing_fabric_metric_groups(
    observed: set[str],
) -> list[str]:
    return [
        name
        for name, alternatives in DCGM_FABRIC_CANONICAL_GROUPS.items()
        if observed.isdisjoint(alternatives)
    ]


def missing_dcgm_metric_groups(
    observed: set[str],
) -> list[str]:
    missing = sorted(DCGM_CORE_CANONICAL_FIELDS - observed)
    missing.extend(missing_fabric_metric_groups(observed))
    return missing


def missing_dcgm_metric_groups_by_gpu(
    observed_by_gpu: Mapping[str, Set[str]],
) -> tuple[list[str], dict[str, list[str]]]:
    """The required groups no GPU reports, and per GPU those only it lacks.

    Completeness is judged GPU by GPU: a field a sibling still reports never
    stands in for one this GPU stopped reporting (GF-GENERAL-LIVE-025
    ``one_gpu_missing``). The list holds the groups missing on every GPU -- a
    scrape-wide gap such as an exporter or counters change; the mapping, in GPU
    order, holds what each GPU lacks beyond that list. Both keep the order of
    :func:`missing_dcgm_metric_groups`.
    """

    by_gpu = {
        gpu: missing_dcgm_metric_groups(set(names))
        for gpu, names in sorted(observed_by_gpu.items())
    }
    if not by_gpu:
        return [], {}
    first, *others = by_gpu.values()
    everywhere = [
        group for group in first if all(group in missing for missing in others)
    ]
    own = {
        gpu: [group for group in missing if group not in everywhere]
        for gpu, missing in by_gpu.items()
    }
    return everywhere, {gpu: groups for gpu, groups in own.items() if groups}


def dcgm_exporter_required_expression() -> str:
    return ",".join("|".join(group) for group in DCGM_EXPORTER_FIELD_GROUPS)

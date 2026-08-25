from __future__ import annotations


DCGM_FABRIC_CANONICAL_GROUPS = {
    "nvlink_crc": frozenset(
        {
            "nvlink_crc_aggregate_error_total",
            "nvlink_crc_flit_error_total",
            "nvlink_crc_data_error_total",
        }
    ),
    "nvlink_recovery": frozenset(
        {
            "nvlink_recovery_aggregate_error_total",
            "nvlink_recovery_error_total",
        }
    ),
    "nvlink_replay": frozenset(
        {
            "nvlink_replay_aggregate_error_total",
            "nvlink_replay_error_total",
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
    ),
    (
        "DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_TOTAL",
        "DCGM_FI_DEV_NVLINK_ERROR_DL_RECOVERY",
    ),
    (
        "DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_TOTAL",
        "DCGM_FI_DEV_NVLINK_ERROR_DL_REPLAY",
    ),
    ("DCGM_FI_DEV_CLOCK_THROTTLE_REASONS",),
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


def dcgm_exporter_required_expression() -> str:
    return ",".join("|".join(group) for group in DCGM_EXPORTER_FIELD_GROUPS)

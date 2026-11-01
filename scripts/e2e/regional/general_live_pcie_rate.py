"""The PCIe replay rule rate the GF-GENERAL-LIVE-022 variants drive (batch 233).

The product grades DCGM field 202 against DCGM's expected error rate of the
GPU's current link (fields 237/238, ``gpu_pcie_link_policy``: 31/min at Gen5
x16, 62/min at Gen6 x16) and against the documented 8/min when the exporter
serves no link. The variants' at-, over- and under-limit increments derive
from the same rate, read from the probe's survey of the GPU.
"""

from __future__ import annotations

from typing import Any

from gpu_fault.gpu_pcie_link_policy import (
    DCGM_DOCUMENTED_REPLAY_RATE_PER_MINUTE,
    PCIE_LINK_GEN_CURRENT,
    PCIE_LINK_WIDTH_CURRENT,
    expected_pcie_error_rate,
)


def pcie_replay_rate(survey: Any, gpu: str) -> float:
    """The product's PCIe replay WARNING rate for ``gpu`` in ``survey``."""

    expected = expected_pcie_error_rate(
        survey.value(gpu, PCIE_LINK_GEN_CURRENT),
        survey.value(gpu, PCIE_LINK_WIDTH_CURRENT),
    )
    return DCGM_DOCUMENTED_REPLAY_RATE_PER_MINUTE if expected is None else expected

"""The PCIe link downtrain verdict, counted where every scrape is seen (batch 233).

The DCGM edge filter delivers only edges and health summaries, so the control
plane cannot count consecutive scrapes; the collector sees every one. Per GPU
it reads DCGM fields 235-238 (maximum and current link generation and width)
and ``gpu_utilization_percent`` (field 203) of the same scrape:

* a **valid** sample has all four link values positive, current <= maximum,
  and the GPU under load -- utilization >= 50 % (nvidia-smi: the current link
  "may be reduced when the GPU is not in use"; ``gpu_pcie_link_policy``);
* a valid sample below the maximum adds one to the GPU's streak; a valid sample
  at the maximum resets it and proves the link;
* any other sample (idle, a field missing) carries the last verdict: idle says
  nothing about the link.

It publishes ``pcie_link_downtrain`` beside the scrape once the GPU has a
verdict: 1 from the third consecutive valid sample below the maximum, 0 after a
valid sample at the maximum, and the last verdict while idle or while a new
streak is still short of three. Before its first verdict -- this process has
not yet seen a valid full link nor three valid samples below it -- a GPU gets
no sample, so the control plane keeps what it knew (a downtrain finding raised
before a reset window stays as it was until the load returns). A quiesce or
reset window stops this collector, so its streak never spans one; "the same
boot" holds because the process does not outlive one. The labels carry the
link of the sample the verdict rests on.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass, field

from gpu_fault.gpu_metrics import GpuMetricSample
from gpu_fault.gpu_pcie_link_policy import (
    PCIE_DOWNTRAIN_CONFIRMATION_SAMPLES,
    PCIE_DOWNTRAIN_LOAD_UTILIZATION_PERCENT,
    PCIE_LINK_DOWNTRAIN,
    PCIE_LINK_FIELDS,
)

UTILIZATION = "gpu_utilization_percent"


@dataclass
class _LinkState:
    streak: int = 0
    verdict: int | None = None
    labels: dict[str, str] = field(default_factory=dict)


def _device(sample: GpuMetricSample) -> str:
    return sample.gpu_uuid or sample.pci_bdf or sample.gpu_index or "node"


class PcieLinkDowntrainTracker:
    """Per-GPU downtrain streaks of one collector process (module docstring)."""

    def __init__(self, max_devices: int = 4096) -> None:
        self.max_devices = max_devices
        self._states: OrderedDict[str, _LinkState] = OrderedDict()

    def observe(self, samples: Sequence[GpuMetricSample]) -> list[GpuMetricSample]:
        """The verdict samples this scrape adds, one per GPU that has one."""

        by_device: dict[str, dict[str, GpuMetricSample]] = {}
        for sample in samples:
            if sample.canonical_name in (*PCIE_LINK_FIELDS, UTILIZATION):
                by_device.setdefault(_device(sample), {})[sample.canonical_name] = (
                    sample
                )
        derived = []
        for device, fields in sorted(by_device.items()):
            if not set(PCIE_LINK_FIELDS) & set(fields):
                continue
            state = self._states.setdefault(device, _LinkState())
            self._states.move_to_end(device)
            self._update(state, fields)
            if state.verdict is not None:
                derived.append(self._verdict_sample(fields, state))
        while len(self._states) > self.max_devices:
            self._states.popitem(last=False)
        return derived

    @staticmethod
    def _update(state: _LinkState, fields: dict[str, GpuMetricSample]) -> None:
        values = {
            name: float(sample.value)
            for name, sample in fields.items()
            if math.isfinite(sample.value)
        }
        link = [values.get(name, 0.0) for name in PCIE_LINK_FIELDS]
        gen_max, width_max, gen, width = link
        valid = (
            all(value > 0 for value in link)
            and gen <= gen_max
            and width <= width_max
            and values.get(UTILIZATION, 0.0) >= PCIE_DOWNTRAIN_LOAD_UTILIZATION_PERCENT
        )
        if not valid:
            return
        if gen < gen_max or width < width_max:
            state.streak += 1
            if state.streak >= PCIE_DOWNTRAIN_CONFIRMATION_SAMPLES:
                state.verdict = 1
        else:
            state.streak = 0
            state.verdict = 0
        if state.verdict is not None and (
            state.verdict == 0 or state.streak >= PCIE_DOWNTRAIN_CONFIRMATION_SAMPLES
        ):
            state.labels = {
                name: f"{value:g}" for name, value in zip(PCIE_LINK_FIELDS, link)
            }
            state.labels[UTILIZATION] = f"{values.get(UTILIZATION, 0.0):g}"

    @staticmethod
    def _verdict_sample(
        fields: dict[str, GpuMetricSample], state: _LinkState
    ) -> GpuMetricSample:
        identity = next(iter(fields.values()))
        return GpuMetricSample(
            metric_name="gpu_fault_pcie_link_downtrain",
            canonical_name=PCIE_LINK_DOWNTRAIN,
            value=float(state.verdict or 0),
            unit=None,
            gpu_index=identity.gpu_index,
            gpu_uuid=identity.gpu_uuid,
            pci_bdf=identity.pci_bdf,
            labels={
                **identity.labels,
                **state.labels,
                "consecutive_valid_samples": str(state.streak),
            },
        )

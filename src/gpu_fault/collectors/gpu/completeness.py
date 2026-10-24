"""The DCGM scrape's field completeness: judged per GPU, delivered on its edges.

The control plane books a batch's ``collection_errors`` as the node's
``dcgm_field_completeness`` finding, clears it on the next clean batch and asks
for diagnostics once the erroring batches have lasted its sustain window
(``gpu_collection_errors``). This module says what a scrape is missing and when
the DCGM collector's edge filter must deliver a batch for that alone.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import TYPE_CHECKING

from gpu_fault.dcgm_fields import (
    missing_dcgm_metric_groups,
    missing_dcgm_metric_groups_by_gpu,
)

if TYPE_CHECKING:
    from gpu_fault.gpu_metrics import GpuMetricSample

#: The start of the collection error a scrape with missing required fields
#: carries; a gap that only some GPUs have names them.
MISSING_FIELDS_ERROR = "DCGM exporter is missing required fields"


def completeness_errors(samples: Sequence[GpuMetricSample]) -> list[str]:
    """The scrape's missing required fields, judged GPU by GPU.

    ``samples`` are what the exporter served, parsed: a blank or non-finite
    slot is already gone, so it reads as missing, never as a healthy value.
    The check used to run over the union of every GPU's fields, which let the
    GPUs that still reported a required field hide the one that stopped
    (GF-GENERAL-LIVE-025 ``one_gpu_missing``). A group no GPU reports keeps the
    batch-wide message; a group only some GPUs lack is reported once per GPU,
    under its device key. Samples that name no GPU cannot stand in for one;
    only when no sample names a GPU is the union all there is to judge.
    """

    observed: dict[str, set[str]] = {}
    for sample in samples:
        device = sample.gpu_uuid or sample.pci_bdf or sample.gpu_index or "node"
        observed.setdefault(device, set()).add(sample.canonical_name)
    gpus = {device: names for device, names in observed.items() if device != "node"}
    by_gpu: dict[str, list[str]] = {}
    if gpus:
        everywhere, by_gpu = missing_dcgm_metric_groups_by_gpu(gpus)
    else:
        everywhere = missing_dcgm_metric_groups(set().union(*observed.values()))
    errors = [f"{MISSING_FIELDS_ERROR}: " + ",".join(everywhere)] if everywhere else []
    errors.extend(
        f"{MISSING_FIELDS_ERROR} on {gpu}: " + ",".join(groups)
        for gpu, groups in by_gpu.items()
    )
    return errors


class CollectionErrorEdges:
    """When a scrape must be delivered for its collection errors alone.

    The control plane can neither book, clear nor escalate what the edge
    filter suppresses: an incomplete scrape used to reach it only on the next
    health summary, and the first clean scrape after an incomplete or failed
    one not at all -- the finding opened late, stayed active after the
    exporter recovered, and escalated only if two health summaries both
    happened to be incomplete.

    Errors that differ from the last delivered batch's are an edge
    (``fields-incomplete``), and so is a clean scrape after erroring
    deliveries (``collection-recovered``). An unchanged gap is sent once more
    when it has lasted ``sustain_seconds`` -- the threshold the control plane
    grades the streak on, read from the collecting node's environment -- since
    its first delivery, unless a delivery since already lasted that long
    (``fields-incomplete-sustained``): one extra delivery per gap, not one per
    tick, and the diagnostic opens at the window rather than at whichever
    health summary comes after it.
    """

    def __init__(self, sustain_seconds: float) -> None:
        self.sustain_seconds = sustain_seconds
        self._delivered: tuple[str, ...] = ()
        # When the current run of erroring deliveries began, and whether one
        # of them has lasted the window yet.
        self._since: datetime | None = None
        self._sustained = False

    def changed(self, errors: Sequence[str]) -> list[str]:
        """The edge of errors that differ from the last delivered batch's."""

        current = tuple(errors)
        if current == self._delivered:
            return []
        return ["fields-incomplete" if current else "collection-recovered"]

    def sustained(self, errors: Sequence[str], observed_at: datetime) -> bool:
        """Whether an unchanged gap is due its one delivery at the window."""

        return (
            bool(errors)
            and not self._sustained
            and self._since is not None
            and (observed_at - self._since).total_seconds() >= self.sustain_seconds
        )

    def delivered(self, errors: Sequence[str], observed_at: datetime) -> None:
        """Record the errors of a batch that left (or was buffered)."""

        self._delivered = tuple(errors)
        if not self._delivered:
            self._since = None
            self._sustained = False
            return
        if self._since is None:
            self._since = observed_at
        if (observed_at - self._since).total_seconds() >= self.sustain_seconds:
            self._sustained = True

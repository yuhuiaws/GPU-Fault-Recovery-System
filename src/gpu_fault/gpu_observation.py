"""Reject ambiguous metric slots without discarding other valid GPU evidence."""

from __future__ import annotations

import math
from collections import Counter
from typing import Any

from gpu_fault.gpu_environment_policy import valid_clock_mask


def metric_device_key(sample: Any) -> str:
    return str(sample.gpu_uuid or sample.pci_bdf or sample.gpu_index or "node")


def normalize_gpu_observation(batch: Any) -> Any:
    errors = list(batch.collection_errors)
    counts = Counter(
        (metric_device_key(sample), sample.canonical_name) for sample in batch.samples
    )
    samples = []
    for sample in batch.samples:
        key = (metric_device_key(sample), sample.canonical_name)
        invalid = (
            not math.isfinite(sample.value)
            or sample.value < 0
            and "tlimit" not in sample.canonical_name
            or counts[key] != 1
            or sample.canonical_name == "clock_throttle_reasons"
            and not valid_clock_mask(float(sample.value))
        )
        if invalid:
            errors.append(f"invalid or ambiguous GPU observation: {key[0]}/{key[1]}")
        else:
            samples.append(sample)
    if not samples and not errors:
        errors.append("empty GPU observation cannot establish health")
    return batch.model_copy(
        update={"samples": samples, "collection_errors": list(dict.fromkeys(errors))}
    )

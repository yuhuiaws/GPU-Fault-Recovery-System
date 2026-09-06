from __future__ import annotations

from importlib.resources import files
from pathlib import Path

import yaml  # type: ignore[import-untyped,unused-ignore]


from gpu_fault.policy.catalog_integrity import (
    validate_xid_catalog_document,
)
from gpu_fault.policy.models import (
    NVIDIA_ALWAYS_FATAL_SXIDS,
    SxidCatalogRule,
    SxidClassification,
    SxidPolicy,
    XidPolicy,
)


def load_xid_policy(path: str | Path | None = None) -> XidPolicy:
    if path is None:
        resource = files("gpu_fault").joinpath(
            "data/nvidia-xid-catalog-610.generated.yaml"
        )
        raw = yaml.safe_load(resource.read_text(encoding="utf-8"))
    else:
        with Path(path).open(encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)

    generated_sha256 = validate_xid_catalog_document(raw)
    spec = raw["spec"]
    metadata = raw["metadata"]
    return XidPolicy(
        name=metadata["name"],
        catalog_version=spec["catalog"]["version"],
        coverage=spec["catalog"]["coverage"],
        source_url=metadata["generatedFrom"],
        source_sha256=metadata["sourceSha256"],
        generated_sha256=generated_sha256,
        product_families=spec["catalog"]["productFamilies"],
        marker_ttl_seconds=spec["defaults"]["markerTtlSeconds"],
        companion_window_seconds=spec["correlation"]["companionWindowSeconds"],
        catalog_rules=spec["catalogRules"],
        nvlink5=spec["nvlink5"],
        resolution_buckets=spec["resolutionBuckets"],
    )


def load_sxid_policy(path: str | Path | None = None) -> SxidPolicy:
    if path is None:
        resource = files("gpu_fault").joinpath(
            "data/nvidia-fabric-manager-sxid-2025-11-14.yaml"
        )
        raw = yaml.safe_load(resource.read_text(encoding="utf-8"))
    else:
        with Path(path).open(encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)

    metadata = raw["metadata"]
    spec = raw["spec"]
    rules = []
    seen = set()
    for group in spec["rules"]:
        for code in group["codes"]:
            if code in seen:
                raise ValueError(f"official SXID catalog contains duplicate {code}")
            seen.add(code)
            rules.append(
                SxidCatalogRule(
                    sxid=code,
                    classification=group["classification"],
                    officialAction=group["officialAction"],
                    investigatoryAction=(
                        None
                        if group.get("investigatoryAction") == "NONE"
                        else group.get("investigatoryAction")
                    ),
                    applicability=group["applicability"],
                )
            )
    policy = SxidPolicy(
        name=metadata["name"],
        source_url=metadata["generatedFrom"],
        source_last_updated=metadata["sourceLastUpdated"],
        source_sha256=metadata["sourceSha256"],
        coverage=spec["coverage"],
        rules=rules,
    )
    catalog_always_fatal = {
        rule.sxid
        for rule in policy.rules
        if rule.classification is SxidClassification.ALWAYS_FATAL
    }
    if catalog_always_fatal != NVIDIA_ALWAYS_FATAL_SXIDS:
        raise ValueError("pinned Always-Fatal SXID set does not match Table 23")
    return policy

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from typing import Any


PINNED_XID_CATALOG_VERSION = "610"
PINNED_XID_SOURCE_SHA256 = (
    "7f70ce9684be0c9d98a367770341ccc1d8bca4465cebe42dab7e23d57df5d5a5"
)
EXPECTED_XID_RULES = 172
EXPECTED_NVLINK5_DECODE_RULES = 95
EXPECTED_RESOLUTION_BUCKETS = 32
GENERATED_SHA_FIELD = "generatedSha256"


def xid_catalog_artifact_sha256(document: dict[str, Any]) -> str:
    canonical = deepcopy(document)
    metadata = canonical.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("XID catalog metadata must be a mapping")
    metadata.pop(GENERATED_SHA_FIELD, None)
    payload = json.dumps(
        canonical,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def validate_xid_catalog_document(document: dict[str, Any]) -> str:
    if document.get("apiVersion") != "gpu-fault.io/v1alpha1":
        raise ValueError("unexpected XID catalog apiVersion")
    if document.get("kind") != "NvidiaXidCatalogPolicy":
        raise ValueError("unexpected XID catalog kind")
    metadata = document.get("metadata")
    spec = document.get("spec")
    if not isinstance(metadata, dict) or not isinstance(spec, dict):
        raise ValueError("XID catalog metadata/spec must be mappings")
    catalog = spec.get("catalog")
    nvlink5 = spec.get("nvlink5")
    rules = spec.get("catalogRules")
    buckets = spec.get("resolutionBuckets")
    if not isinstance(catalog, dict) or not isinstance(nvlink5, dict):
        raise ValueError("XID catalog policy sections must be mappings")
    decode_rules = nvlink5.get("decodeRules")
    if (
        catalog.get("version") != PINNED_XID_CATALOG_VERSION
        or metadata.get("sourceSha256") != PINNED_XID_SOURCE_SHA256
    ):
        raise ValueError("XID catalog source version/digest is not pinned")
    if not isinstance(rules, list) or len(rules) != EXPECTED_XID_RULES:
        raise ValueError(f"XID catalog must contain {EXPECTED_XID_RULES} rules")
    if (
        not isinstance(decode_rules, list)
        or len(decode_rules) != EXPECTED_NVLINK5_DECODE_RULES
    ):
        raise ValueError(
            "XID catalog must contain "
            f"{EXPECTED_NVLINK5_DECODE_RULES} NVLink5 decode rules"
        )
    if not isinstance(buckets, dict) or len(buckets) != EXPECTED_RESOLUTION_BUCKETS:
        raise ValueError(
            f"XID catalog must contain {EXPECTED_RESOLUTION_BUCKETS} resolution buckets"
        )
    expected = metadata.get(GENERATED_SHA_FIELD)
    if not isinstance(expected, str) or len(expected) != 64:
        raise ValueError("XID catalog generatedSha256 is missing or invalid")
    actual = xid_catalog_artifact_sha256(document)
    if actual != expected:
        raise ValueError(
            "XID catalog generated artifact digest mismatch: "
            f"expected {expected}, got {actual}"
        )
    return actual

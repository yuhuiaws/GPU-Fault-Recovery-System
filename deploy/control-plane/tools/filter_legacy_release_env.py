#!/usr/bin/env python3
from __future__ import annotations

import sys
from typing import Any

import yaml  # type: ignore[import-untyped,unused-ignore]

COMPONENT_PIN_ENV = frozenset(
    {
        "GPU_FAULT_REQUIRED_AGENT_COMPATIBILITY_DIGEST",
        "GPU_FAULT_COMPATIBLE_AGENT_COMPATIBILITY_DIGESTS",
        "GPU_FAULT_REQUIRED_REGIONAL_EXECUTOR_ARTIFACT_SHA256",
        "GPU_FAULT_COMPATIBLE_REGIONAL_EXECUTOR_ARTIFACT_SHA256S",
        "GPU_FAULT_REQUIRED_REGIONAL_EXECUTOR_COMPATIBILITY_DIGEST",
        "GPU_FAULT_COMPATIBLE_REGIONAL_EXECUTOR_COMPATIBILITY_DIGESTS",
    }
)


def filter_document(document: Any) -> Any:
    if not isinstance(document, dict) or document.get("kind") != "Deployment":
        return document
    pod_spec = document.get("spec", {}).get("template", {}).get("spec", {})
    for container in [
        *pod_spec.get("initContainers", []),
        *pod_spec.get("containers", []),
    ]:
        container["env"] = [
            item
            for item in container.get("env", [])
            if item.get("name") not in COMPONENT_PIN_ENV
        ]
    return document


def main() -> int:
    documents = [
        filter_document(document) for document in yaml.safe_load_all(sys.stdin)
    ]
    yaml.safe_dump_all(documents, sys.stdout, sort_keys=False, width=88)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

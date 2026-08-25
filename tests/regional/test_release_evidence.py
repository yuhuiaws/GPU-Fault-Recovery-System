from __future__ import annotations

import base64
from pathlib import Path

from tests._script_loader import lazy_script_module

ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "scripts/capture_release_evidence.py"
MODULE = lazy_script_module("capture_release_evidence", MODULE_PATH)


def encoded(value: str) -> str:
    return base64.b64encode(value.encode()).decode()


def test_secret_inventory_contains_only_hashes() -> None:
    secret = {
        "metadata": {"name": "example", "creationTimestamp": "2026-08-23T00:00:00Z"},
        "type": "Opaque",
        "data": {"token": encoded("do-not-print-me")},
    }

    inventory = MODULE.secret_inventory([secret])

    assert "do-not-print-me" not in inventory
    assert encoded("do-not-print-me") not in inventory
    assert "example\tOpaque" in inventory
    assert len(inventory.splitlines()[-1].split("\t")[-1]) == 64


def test_binary_configmap_is_replaced_by_digest() -> None:
    config_map = {
        "metadata": {
            "name": "wheel",
            "annotations": {
                "kubectl.kubernetes.io/last-applied-configuration": ("leaked")
            },
        },
        "binaryData": {"release.whl": encoded("wheel-bytes")},
    }

    sanitized = MODULE.sanitize_config_map(config_map)

    assert "binaryData" not in sanitized
    assert "release.whl" in sanitized["evidenceBinaryData"]
    assert "kubectl.kubernetes.io/last-applied-configuration" not in sanitized[
        "metadata"
    ].get("annotations", {})

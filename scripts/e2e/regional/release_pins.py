"""Release-bound pins a CPU control-plane Pod exports, read from the release metadata.

The CPU Deployments take ``GPU_FAULT_REQUIRED_*`` through ``configMapKeyRef``
from ``gpu-fault-release-metadata`` at Pod start, so inside a Pod they name the
release that Pod was rolled for. ``GPU_FAULT_RELEASE_ID`` is *not* rendered into
the CPU role environment (``gpu_fault.app.factory`` defaults it to "local");
Pod-side probes therefore bind their receipts to these pins and carry the
release ID only as evidence.
"""

from __future__ import annotations

import re
from typing import Any

from gpu_fault.fleet_pins import CONFIG_MAP_KEY_ENVIRONMENT

from scripts.e2e.regional.regional_commands import RegionalFixtureError

RELEASE_METADATA_CONFIGMAP = "gpu-fault-release-metadata"
RELEASE_PIN_CONFIG_MAP_KEYS = (
    "required-regional-executor-artifact-sha256",
    "required-agent-artifact-sha256",
    "required-agent-compatibility-digest",
)
RELEASE_PIN_ENVIRONMENT = tuple(
    CONFIG_MAP_KEY_ENVIRONMENT[key] for key in RELEASE_PIN_CONFIG_MAP_KEYS
)
SHA256_HEX = re.compile(r"[0-9a-f]{64}")


def release_pins_from_metadata(document: Any) -> dict[str, str]:
    """``RELEASE_PIN_CONFIG_MAP_KEYS`` of a release-metadata ConfigMap, by env name.

    Fail-closed: a ConfigMap without ``data``, a missing key, a blank or
    malformed value is a refusal, never "unpinned".
    """

    data = document.get("data") if isinstance(document, dict) else None
    if not isinstance(data, dict):
        raise RegionalFixtureError(
            f"{RELEASE_METADATA_CONFIGMAP} ConfigMap has no data"
        )
    pins: dict[str, str] = {}
    for key in RELEASE_PIN_CONFIG_MAP_KEYS:
        raw = data.get(key)
        value = raw.strip() if isinstance(raw, str) else ""
        if SHA256_HEX.fullmatch(value) is None:
            raise RegionalFixtureError(
                f"{RELEASE_METADATA_CONFIGMAP} pin {key} is missing or malformed"
            )
        pins[CONFIG_MAP_KEY_ENVIRONMENT[key]] = value
    return pins
